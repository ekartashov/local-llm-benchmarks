#!/usr/bin/env bash
# T2.3b_gemma4_31b_thinker.sh — Gemma 4 31B Dense AWQ as Arclight thinker candidate.
#
# Research findings (R13, 2026-04-23):
#   Model:    QuantTrio/gemma-4-31B-it-AWQ  (~20 GiB — NOT the ~16 GiB estimate in T2.3b spec)
#   Image:    vllm/vllm-openai:gemma4  (NOT :latest — 0.19.0/0.19.1 has tool-call JSON bug #39468)
#   Parsers:  --tool-call-parser gemma4 ONLY.
#             DO NOT add --reasoning-parser gemma4: streaming path waits for <think> close tags;
#             if model skips reasoning and goes straight to tools, parser never activates and
#             raw tool tokens appear as text. Same root cause as Qwen3-Next-80B requiring hermes
#             alone. See DECISIONS.md.
#   Trust:    --trust-remote-code required.
#   kvcached: Phase B (single GPU) NOT viable — 22+20=42 GiB > 32 GiB. Cross-GPU KV sharing
#             needs separate research before re-testing T1.5 Phase B.
#
# Placement: GPU1 TP=1 (thinker slot). Falls back to TP=2 on tp2b if CUDA graph OOM.
# Suite:     Phase 2.2 thinker quality suite (8 tasks).
#            Infra-shaped tasks not yet authored (T6.1) — running base suite only.
# Baseline:  Qwen3.5-27B — quality 4.0/5, 76.5 t/s seq=1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

ITEM_ID="T2.3b_arclight_thinker_gemma4_31b_candidate"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="QuantTrio/gemma-4-31B-it-AWQ"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[T2.3b] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

BASELINE_TPS=76.5
BASELINE_QUALITY=4.0

# ── Decode measurement helper ──────────────────────────────────────────────────
MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
"""T2.3b decode measurement helper."""
import asyncio, httpx, json, sys, time

DECODE_PROMPT = (
    "Explain the Byzantine Generals Problem and describe two distinct consensus "
    "protocols that solve it. For each protocol, outline the key invariants, "
    "failure modes, and the trade-off between safety and liveness."
)
MAX_DECODE_TOKENS = 2048

async def _measure_one(client, endpoint, model, prompt, max_tokens):
    t0 = time.monotonic()
    fttt = None
    count = 0
    text_out = []
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
                text_out.append(tok)
    total = time.monotonic() - t0
    decode_t = total - (fttt or 0)
    return {
        "ttft_ms":    round((fttt or 0) * 1000, 1),
        "decode_tps": round(count / decode_t, 1) if decode_t > 0 and count > 0 else 0.0,
        "tokens":     count,
        "text_snippet": "".join(text_out)[:300],
    }

async def measure(endpoint, model, num_seqs, out_path):
    async with httpx.AsyncClient(timeout=600) as c:
        results = await asyncio.gather(*[
            _measure_one(c, endpoint, model, DECODE_PROMPT, MAX_DECODE_TOKENS)
            for _ in range(num_seqs)
        ])
    agg = {
        "avg_tps_per_seq": round(sum(r["decode_tps"] for r in results) / num_seqs, 1),
        "total_tps":       round(sum(r["decode_tps"] for r in results), 1),
        "avg_ttft_ms":     round(sum(r["ttft_ms"] for r in results) / num_seqs, 1),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(agg, f)
    print(json.dumps({k: agg[k] for k in ("avg_tps_per_seq", "total_tps", "avg_ttft_ms")}))

mode, endpoint, model, num_seqs, out_path = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
asyncio.run(measure(endpoint, model, num_seqs, out_path))
PYEOF

# ── Step 1: Deploy on GPU1 TP=1 ───────────────────────────────────────────────
log "Deploying ${MODEL} on GPU1 TP=1 ..."
log "Image: vllm/vllm-openai:gemma4  (required — :latest/0.19.x has tool-call JSON bug #39468)"
log "Parser: --tool-call-parser gemma4 only — no --reasoning-parser (streaming interception bug)"

PLACEMENT="tp=1 (gpu1)"
ENDPOINT="http://localhost:${PORT_VLLM_GPU1}/v1"

if ! BENCH_IMAGE=vllm/vllm-openai:gemma4 \
        VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
        "${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
        --gpu-mem-util 0.90 \
        --ctx 32768 \
        --tool-call-parser gemma4 \
        --trust-remote-code \
        2>&1 | tee -a "${LOG}"; then
    log "TP=1 on gpu1 FAILED (likely CUDA graph OOM at 20 GiB weights)."
    log "Falling back to TP=2 on tp2b (borrows both GPUs — stop coder/thinker first) ..."
    for c in bench-vllm-gpu0 bench-vllm-gpu1; do
        if podman container exists "${c}" 2>/dev/null; then
            log "Stopping conflicting container ${c} ..."
            podman stop "${c}" 2>/dev/null || true
            podman rm   "${c}" 2>/dev/null || true
        fi
    done
    BENCH_IMAGE=vllm/vllm-openai:gemma4 \
        VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
        "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2b "${MODEL}" \
        --gpu-mem-util 0.85 \
        --ctx 32768 \
        --tool-call-parser gemma4 \
        --trust-remote-code \
        2>&1 | tee -a "${LOG}"
    PLACEMENT="tp=2 (tp2b)"
    ENDPOINT="http://localhost:${PORT_VLLM_TP2_B}/v1"
    log "TP=2 fallback deployed."
fi

log "Placement: ${PLACEMENT}  Endpoint: ${ENDPOINT}"

# ── Step 2: Warmup ─────────────────────────────────────────────────────────────
log "Warmup request ..."
python3 "${MEASURE_PY}" measure "${ENDPOINT}" "${MODEL}" 1 /dev/null 2>/dev/null || true

# ── Step 3: Decode TPS ─────────────────────────────────────────────────────────
log "Measuring decode TPS seq=1 ..."
python3 "${MEASURE_PY}" measure "${ENDPOINT}" "${MODEL}" 1 \
    "${RESULTS_DIR}/raw/decode_seq1.json" | tee -a "${LOG}"
DEC_TPS_1=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/decode_seq1.json'))['avg_tps_per_seq'])")

log "Measuring decode TPS seq=4 ..."
python3 "${MEASURE_PY}" measure "${ENDPOINT}" "${MODEL}" 4 \
    "${RESULTS_DIR}/raw/decode_seq4.json" | tee -a "${LOG}"
DEC_TPS_4=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/decode_seq4.json'))['total_tps'])")

log "Decode: seq=1 ${DEC_TPS_1} t/s  seq=4 total ${DEC_TPS_4} t/s  (baseline: ${BASELINE_TPS} t/s)"

# ── Step 4: Phase 2.2 thinker quality suite (8 tasks) ─────────────────────────
log "Running Phase 2.2 thinker quality suite (8 tasks) ..."
log "Note: infra-shaped tasks not yet authored (T6.1) — base suite only."
log "Using max_tokens=4096 (thinking models exhaust 1024 budget — settled R10)."
python3 -m benchmarks.phase2_model_selection.bench \
    --endpoint "${ENDPOINT}" \
    --results-dir "${RESULTS_DIR}/phase2_quality" \
    --mode quality \
    --tasks "${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker/" \
    --model "${MODEL}" \
    --label "Gemma4-31B-Dense-AWQ" \
    --max-tokens 4096 \
    2>&1 | tee -a "${LOG}" || {
    log "WARNING: phase2 quality bench exited non-zero — check logs."
}

TASK_COMPLETION_RATE="0.0"
if [[ -f "${RESULTS_DIR}/phase2_quality/metrics.json" ]]; then
    TASK_COMPLETION_RATE=$(python3 -c "
import json; d=json.load(open('${RESULTS_DIR}/phase2_quality/metrics.json'))
print(d['metrics'].get('task_completion_rate', 0.0))
" 2>/dev/null || echo "0.0")
fi
log "Task completion rate: ${TASK_COMPLETION_RATE}"

# ── Step 5: Write metrics.json and summary.md ──────────────────────────────────
python3 - <<PYEOF
import json, pathlib

item_id          = "${ITEM_ID}"
timestamp        = "${TIMESTAMP}"
out              = pathlib.Path("${RESULTS_DIR}")
placement        = "${PLACEMENT}"
dec_tps_1        = float("${DEC_TPS_1}")
dec_tps_4        = float("${DEC_TPS_4}")
task_completion  = float("${TASK_COMPLETION_RATE}")
baseline_tps     = float("${BASELINE_TPS}")
baseline_quality = float("${BASELINE_QUALITY}")
tps_ratio        = round(dec_tps_1 / baseline_tps, 3) if baseline_tps > 0 else 0.0

metrics = {
    "item_id":   item_id,
    "timestamp": timestamp,
    "config": {
        "engine":         "vllm",
        "engine_version": "gemma4-tag",
        "model":          "${MODEL}",
        "quantization":   "AWQ-INT4",
        "placement":      placement,
        "context_length": 32768,
        "extra_args":     "--tool-call-parser gemma4 --trust-remote-code",
    },
    "metrics": {
        "decode_tps_seq1":          dec_tps_1,
        "decode_tps_seq4_total":    dec_tps_4,
        "tps_ratio_vs_baseline":    tps_ratio,
        "task_completion_rate":     task_completion,
        "quality_mean_8task":       None,  # fill in after human review
        "baseline_tps_thinker_27b": baseline_tps,
        "baseline_quality_27b":     baseline_quality,
    },
    "verdict": "PENDING_HUMAN_REVIEW",
    "notes":   "Quality scoring (1-5 per task) required. See phase2_quality/human_review.md.",
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

md = f"""# T2.3b Gemma 4 31B Dense AWQ — Arclight Thinker Candidate

**vLLM** gemma4 tag | **Model** {metrics['config']['model']} | **Date** {timestamp[:10]}
**Placement** {placement} | **ctx** 32768 | **Parser** gemma4 (no reasoning-parser)

## Auto-scored metrics

| Metric | Gemma4-31B | Baseline (Qwen3.5-27B) | Notes |
|--------|------------|------------------------|-------|
| Decode TPS (seq=1) | {dec_tps_1:.1f} t/s | {baseline_tps} t/s | ratio {tps_ratio:.2f}× |
| Decode TPS (seq=4 total) | {dec_tps_4:.1f} t/s | — | |
| Task completion rate | {task_completion:.0%} | — | non-empty answer |

## Human review required

Open **{out}/phase2_quality/human_review.md** and score all 8 tasks 1–5.

**Scoring criteria (thinker role):**
- 5: Correct analysis, concrete and actionable, anticipates edge cases
- 4: Mostly correct, minor gaps in reasoning or recommendations
- 3: Correct framing but shallow or misses key considerations
- 2: Partially correct, significant gaps or wrong conclusions
- 1: Wrong or incoherent

After scoring, update **{out}/metrics.json**:
- Set `quality_mean_8task` to the mean score
- Set `verdict` to PASS/FAIL/INCONCLUSIVE

**Decision rule (T2.3b):**
- PASS: mean ≥ {baseline_quality}/5 AND no thinking-budget pathology on th03 (task 03)
- FAIL: mean < {baseline_quality}/5
- MIXED/INCONCLUSIVE: wins on some task types, loses on others → hand back to research

**Special attention — th03 (architecture tradeoffs):**
Qwen3.5-27B consistently exhausts its thinking budget on this task (empty output, T1.4 FAIL).
Check whether Gemma4 produces a non-empty answer. If yes, this is a significant advantage.

**kvcached Phase B note:**
Single-GPU Phase B (both models on gpu0) is NOT viable:
  Gemma4-31B ~20 GiB + Qwen3.6-35B ~22 GiB = 42 GiB > 32 GiB physical VRAM.
  The T2.3b spec estimated ~16 GiB for Gemma4 — that was wrong.
  Cross-GPU KV sharing (one model per GPU, shared pool) needs separate research
  before T1.5 Phase B can be redesigned and re-run.

**Verdict: PENDING HUMAN REVIEW**
"""
(out / "summary.md").write_text(md)

print(f"[T2.3b] TPS={dec_tps_1:.1f} t/s (ratio {tps_ratio:.2f}x vs baseline {baseline_tps})")
print(f"[T2.3b] Task completion: {task_completion:.0%}")
print(f"[T2.3b] Results: {out}")
print(f"[T2.3b] Next: score human_review.md (1-5 per task), update metrics.json with quality_mean_8task and verdict.")
PYEOF

log "Done."
log "Human review: ${RESULTS_DIR}/phase2_quality/human_review.md"
log "After scoring: update ${RESULTS_DIR}/metrics.json  quality_mean_8task + verdict."
