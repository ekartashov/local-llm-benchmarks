#!/usr/bin/env bash
# T12_prismaquant_thinker_27b.sh — Eval PrismaQuant 5.5bit as thinker candidate.
#
# Context (BENCH_12):
#   Model: rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm
#   Format: PrismaQuant 5.5-bit (dense GEMM, SM120 bugs N/A)
#   Engine: vLLM (must use VLLM_USE_V1=0 for stability)
#   Placement: GPU1 TP=1 (production thinker slot)
#
# Baseline (Arclight Thinker):
#   Qwen3.6-27B-AWQ @ 77.4 t/s (seq=1), ~105 t/s (seq=4)
#   Quality: th02 correct (5/5)
#
# Success Criteria:
#   1. Quality: th02 scores 5/5 (quantization did not corrupt DeltaNet).
#   2. Performance: Throughput > AWQ baseline.
#
# Usage: ./benchmarks/queue/T12_prismaquant_thinker_27b.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

ITEM_ID="T12_prismaquant_thinker_27b"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm"
PORT="${PORT_VLLM_GPU1}"
ENDPOINT="http://localhost:${PORT}/v1"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[T12 $(date -u +%H:%M:%S)] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

# ── Baseline numbers for comparison ───────────────────────────────────────────
BASELINE_TPS_SEQ1=77.4
BASELINE_TPS_SEQ4=105.0

# ── Step 0: Environment and Image Setup ───────────────────────────────────────
# Use the default production image (vLLM 0.19.x).
# v0.18.0 was wrong here: VLLM_USE_V1=0 is not respected in 0.18.0 on Blackwell —
# the V1 engine was hardwired for this architecture regardless of the flag.
# VLLM_NVFP4_GEMM_BACKEND defaults to FLASHINFER_CUTLASS on SM120 automatically; no export needed.

log "Ensuring GPU1 is free (stopping production thinker)..."
if podman container exists bench-vllm-gpu1 2>/dev/null; then
    log "Stopping bench-vllm-gpu1..."
    podman stop bench-vllm-gpu1 2>/dev/null || true
    podman rm   bench-vllm-gpu1 2>/dev/null || true
fi

# ── Step 1: Baseline VRAM (idle) ──────────────────────────────────────────────
log "Recording base VRAM (idle) on GPU1..."
VRAM_IDLE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_1_ID}" | tr -d ' ')
log "GPU1 Idle VRAM: ${VRAM_IDLE} MiB"

# ── Step 2: Deploy benchmark model ───────────────────────────────────────────
log "Deploying ${MODEL} on GPU1 (TP=1)..."
log "Flags: VLLM_USE_V1=0, max-num-seqs=4, fp8 KV, chunked-prefill=ON"

export VLLM_USE_V1=0
export VLLM_USE_V1_ENGINE=0
export VLLM_V1=0
export VLLM_ENGINE_ITERATOR_SOURCE=LEGACY

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
    --gpu-mem-util 0.90 \
    --ctx 32768 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 4 \
    --enable-chunked-prefill \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
    2>&1 | tee -a "${LOG}" || die "Deployment failed."

log "Deployment successful. Endpoint: ${ENDPOINT}"

# ── Step 3: VRAM (post-load) ──────────────────────────────────────────────────
VRAM_LOADED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_1_ID}" | tr -d ' ')
VRAM_DELTA=$(( VRAM_LOADED - VRAM_IDLE ))
log "GPU1 Loaded VRAM: ${VRAM_LOADED} MiB (Delta: ${VRAM_DELTA} MiB)"

# ── Step 4: Throughput Sweep ──────────────────────────────────────────────────
# Logic from T_PAR1_parallel_throughput_sweep.sh
MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
import asyncio, httpx, json, sys, time

async def _measure_one(client, endpoint, model, prompt, max_tokens):
    t0 = time.monotonic()
    fttt = None
    count = 0
    async with client.stream("POST", f"{endpoint}/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }) as resp:
        resp.raise_for_status()
        async for raw in resp.aiter_lines():
            if not raw.startswith("data: ") or "[DONE]" in raw:
                continue
            delta = json.loads(raw[6:])["choices"][0]["delta"]
            tok = delta.get("content") or delta.get("reasoning") or ""
            if tok:
                if fttt is None:
                    fttt = time.monotonic() - t0
                count += 1
    total = time.monotonic() - t0
    decode_s = total - (fttt or 0)
    return {
        "ttft_ms":    round((fttt or 0) * 1000, 1),
        "decode_tps": round(count / decode_s, 1) if decode_s > 0 and count > 0 else 0.0,
        "tokens":     count,
    }

async def sweep(endpoint, model, seq_list, reps, out_dir):
    prompt = "Write a detailed technical explanation of how consistent hashing works."
    results = {}
    async with httpx.AsyncClient(timeout=300) as client:
        for n in seq_list:
            print(f"  N={n} concurrent requests...")
            round_agg_tps = []
            for r in range(reps):
                t_start = time.monotonic()
                batch = await asyncio.gather(*[_measure_one(client, endpoint, model, prompt, 512) for _ in range(n)])
                wall_s = time.monotonic() - t_start
                total_tokens = sum(b["tokens"] for b in batch)
                round_agg_tps.append(total_tokens / wall_s)
            
            results[n] = {
                "n": n,
                "avg_agg_tps": round(sum(round_agg_tps) / len(round_agg_tps), 1),
                "max_agg_tps": round(max(round_agg_tps), 1),
                "med_ttft_ms": sorted([b["ttft_ms"] for b in batch])[n // 2]
            }
            print(f"    Agg TPS: {results[n]['avg_agg_tps']}")

    with open(f"{out_dir}/throughput_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

endpoint, model, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
asyncio.run(sweep(endpoint, model, [1, 2, 4], 3, out_dir))
PYEOF

log "Starting throughput sweep (N=1, 2, 4)..."
python3 "${MEASURE_PY}" "${ENDPOINT}" "${MODEL}" "${RESULTS_DIR}/raw" | tee -a "${LOG}"

TPS_SEQ1=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/raw/throughput_sweep.json')); print(d['1']['avg_agg_tps'])")
TPS_SEQ4=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/raw/throughput_sweep.json')); print(d['4']['avg_agg_tps'])")

# ── Step 5: Quality Suite ─────────────────────────────────────────────────────
log "Running Thinker Quality Suite (8 tasks)..."
python3 -m benchmarks.phase2_model_selection.bench \
    --endpoint "${ENDPOINT}" \
    --results-dir "${RESULTS_DIR}/phase2_quality" \
    --mode quality \
    --tasks "${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker/" \
    --model "${MODEL}" \
    --label "Qwen3.6-27B-PrismaQuant-5.5bit" \
    --max-tokens 16384 \
    2>&1 | tee -a "${LOG}" || log "WARNING: Quality bench exited non-zero."

# ── Step 6: Cleanup ───────────────────────────────────────────────────────────
log "Cleaning up benchmark container..."
podman stop bench-vllm-gpu1 2>/dev/null || true
podman rm   bench-vllm-gpu1 2>/dev/null || true

# ── Step 7: Final Metrics and Summary ─────────────────────────────────────────
python3 - <<PYEOF
import json, pathlib

out = pathlib.Path("${RESULTS_DIR}")
metrics = {
    "item_id": "${ITEM_ID}",
    "timestamp": "${TIMESTAMP}",
    "config": {
        "model": "${MODEL}",
        "engine": "vllm (V0)",
        "quantization": "PrismaQuant 5.5bit",
        "kv_cache": "fp8",
        "ctx": 32768,
        "max_num_seqs": 4,
    },
    "metrics": {
        "vram_idle_mib": ${VRAM_IDLE},
        "vram_loaded_mib": ${VRAM_LOADED},
        "vram_delta_mib": ${VRAM_DELTA},
        "tps_seq1": ${TPS_SEQ1},
        "tps_seq4": ${TPS_SEQ4},
        "baseline_tps_seq1": ${BASELINE_TPS_SEQ1},
        "baseline_tps_seq4": ${BASELINE_TPS_SEQ4},
    },
    "verdict": "PENDING_HUMAN_REVIEW",
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

tps_gain_seq1 = round((${TPS_SEQ1} - ${BASELINE_TPS_SEQ1}) / ${BASELINE_TPS_SEQ1} * 100, 1)
tps_gain_seq4 = round((${TPS_SEQ4} - ${BASELINE_TPS_SEQ4}) / ${BASELINE_TPS_SEQ4} * 100, 1)

summary = f"""# T12 PrismaQuant Thinker 27B Eval

**Model:** {metrics['config']['model']}
**Date:** {metrics['timestamp']}

## Throughput vs Baseline

| Metric | PrismaQuant | AWQ Baseline | Gain |
|--------|-------------|--------------|------|
| TPS (seq=1) | {metrics['metrics']['tps_seq1']:.1f} | {metrics['metrics']['baseline_tps_seq1']:.1f} | {tps_gain_seq1}% |
| TPS (seq=4) | {metrics['metrics']['tps_seq4']:.1f} | {metrics['metrics']['baseline_tps_seq4']:.1f} | {tps_gain_seq4}% |

## Resource Footprint

- Idle VRAM: {metrics['metrics']['vram_idle_mib']} MiB
- Loaded VRAM: {metrics['metrics']['vram_loaded_mib']} MiB
- Model Delta: {metrics['metrics']['vram_delta_mib']} MiB

## Quality (Human Review Required)

Open **phase2_quality/human_review.md** to score the 8 tasks.
**CRITICAL:** Check **th02** (Algorithm Logic). It MUST be 5/5 (Correct).
If th02 is incorrect, PrismaQuant has corrupted DeltaNet recurrence.

**Next Steps:**
1. Score the quality suite.
2. Update metrics.json and this summary with the final verdict.
3. Restart production thinker manually:
   \`./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 --max-num-seqs 4 --enable-chunked-prefill --tool-call-parser qwen3_coder --reasoning-parser qwen3\`
"""
(out / "summary.md").write_text(summary)
PYEOF

log "Benchmark complete. Results in: ${RESULTS_DIR}"
log "Note: GPU1 is now idle. Restart production thinker manually when ready."
