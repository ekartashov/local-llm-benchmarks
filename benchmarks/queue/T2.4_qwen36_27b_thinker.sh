#!/usr/bin/env bash
# T2.4_qwen36_27b_thinker.sh — Qwen3.6-27B-AWQ as Arclight thinker candidate.
#
# Research findings (R14, 2026-04-24):
#   Model:    QuantTrio/Qwen3.6-27B-AWQ (~21 GiB)
#   Image:    vllm/vllm-openai:latest  (standard; vllm>=0.19.0 confirmed compatible)
#   Parsers:  --tool-call-parser qwen3_coder --reasoning-parser qwen3
#             Same stack as Qwen3.6-35B coder — proven at 96.7% reliability in T2.5.
#             DO NOT omit --reasoning-parser: unlike Gemma4/Core, Qwen3.6 family
#             requires both parsers together (coder T2.5 confirmed this).
#   Language-model-only: --language-model-only sheds the vision encoder (~1-3 GiB)
#             to free KV headroom. If vLLM 0.19.x does not support this flag the
#             deploy will fail; script retries without it (still fits at 21+~2 GiB).
#   transformers: >=5.5.4 required. Script logs installed version for confirmation.
#   kvcached: GDN (Gated DeltaNet) hybrid → DeltaNetSpec unsupported by kvcached
#             v0.1.5. T1.5 Phase B remains deferred. TP=1 isolated deploy unaffected.
#
# Placement: GPU1 TP=1 (thinker slot). Falls back to TP=2 on tp2b only if OOM.
#            Note: 21 GiB + vision encoder (~2 GiB) = ~23 GiB on 28.8 GiB usable
#            (gpu-mem-util 0.90) → 5.8 GiB headroom. If CUDA graphs OOM, add
#            --max-num-seqs 1 (same fix as Qwen3.5-27B).
# Suite:     Phase 2.2 thinker quality suite (8 tasks, max_tokens=4096).
#            Infra-shaped tasks not yet authored (T6.1) — running base suite only.
# Baseline:  Qwen3.5-27B — quality 4.0/5, 76.5 t/s seq=1.
# Special:   Watch th03 (architecture_tradeoffs). Qwen3.5-27B emits empty output
#            there (thinking budget exhaustion). Record whether Qwen3.6-27B produces
#            non-empty output — this is a meaningful defect fix.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

ITEM_ID="T2.4_arclight_thinker_qwen36_27b_candidate"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="QuantTrio/Qwen3.6-27B-AWQ"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[T2.4] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

BASELINE_TPS=76.5
BASELINE_QUALITY=4.0

# ── Preflight: log transformers version ───────────────────────────────────────
log "=== T2.4 Qwen3.6-27B-AWQ Arclight Thinker Candidate ==="
log "Timestamp: ${TIMESTAMP}"
TRANSFORMERS_VER=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "UNKNOWN")
log "transformers version: ${TRANSFORMERS_VER}  (required: >=5.5.4)"
if python3 -c "
import transformers, sys
parts = transformers.__version__.split('.')
major, minor = int(parts[0]), int(parts[1])
sys.exit(0 if (major > 5 or (major == 5 and minor >= 5)) else 1)
" 2>/dev/null; then
    log "transformers version OK."
else
    log "WARNING: transformers < 5.5.4 detected. Qwen3.6 architecture may not load correctly."
    log "         Consider updating: pip install 'transformers>=5.5.4'"
    log "         Proceeding anyway — deploy will fail if incompatible."
fi

# ── Decode measurement helper ──────────────────────────────────────────────────
MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
"""T2.4 decode measurement helper."""
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
        "extra_body": {"top_k": 20},
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
PLACEMENT="tp=1 (gpu1)"
ENDPOINT="http://localhost:${PORT_VLLM_GPU1}/v1"

_deploy_gpu1() {
    local extra_flags="${1:-}"
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
        "${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
        --gpu-mem-util 0.90 \
        --ctx 32768 \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        ${extra_flags} \
        2>&1 | tee -a "${LOG}"
}

log "Attempt 1: TP=1 on GPU1 with --language-model-only (sheds vision encoder) ..."
if _deploy_gpu1 "--language-model-only"; then
    log "Deployed with --language-model-only."
else
    log "Attempt 1 FAILED. Retrying without --language-model-only ..."
    log "(Either the flag is unsupported in this vLLM build, or OOM with vision encoder.)"
    if _deploy_gpu1 ""; then
        log "Deployed without --language-model-only (vision encoder included in VRAM)."
    else
        log "TP=1 on GPU1 FAILED entirely. Possible CUDA graph OOM."
        log "Fallback A: Try --max-num-seqs 1 (same fix as Qwen3.5-27B baseline)..."
        if _deploy_gpu1 "--max-num-seqs 1"; then
            log "Deployed with --max-num-seqs 1."
        else
            log "All GPU1 TP=1 attempts failed. Falling back to TP=2 (borrows both GPUs)..."
            log "Stopping conflicting containers if any..."
            for c in bench-vllm-gpu0 bench-vllm-gpu1; do
                if podman container exists "${c}" 2>/dev/null; then
                    log "Stopping ${c} ..."
                    podman stop "${c}" 2>/dev/null || true
                    podman rm   "${c}" 2>/dev/null || true
                fi
            done
            VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
                "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2b "${MODEL}" \
                --gpu-mem-util 0.85 \
                --ctx 32768 \
                --tool-call-parser qwen3_coder \
                --reasoning-parser qwen3 \
                --enable-auto-tool-choice \
                2>&1 | tee -a "${LOG}" || die "All deployment attempts failed. Check bench.log."
            PLACEMENT="tp=2 (tp2b)"
            ENDPOINT="http://localhost:${PORT_VLLM_TP2_B}/v1"
            log "TP=2 fallback deployed."
        fi
    fi
fi

log "Placement: ${PLACEMENT}  Endpoint: ${ENDPOINT}"

# ── Step 2: Warmup ─────────────────────────────────────────────────────────────
log "Warmup request (discarded) ..."
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
log "max_tokens=4096 (thinking models exhaust 1024 budget — settled in DECISIONS.md)."
log "Special watch: th03_architecture_tradeoffs — Qwen3.5-27B emits empty here."
python3 -m benchmarks.phase2_model_selection.bench \
    --endpoint "${ENDPOINT}" \
    --results-dir "${RESULTS_DIR}/phase2_quality" \
    --mode quality \
    --tasks "${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker/" \
    --model "${MODEL}" \
    --label "Qwen3.6-27B-AWQ" \
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
        "engine_version": "0.19.x",
        "model":          "QuantTrio/Qwen3.6-27B-AWQ",
        "quantization":   "AWQ-INT4",
        "placement":      placement,
        "context_length": 32768,
        "extra_args":     "--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice",
    },
    "metrics": {
        "decode_tps_seq1":          dec_tps_1,
        "decode_tps_seq4_total":    dec_tps_4,
        "tps_ratio_vs_baseline":    tps_ratio,
        "task_completion_rate":     task_completion,
        "quality_mean_8task":       None,   # fill after human review
        "th03_non_empty":           None,   # fill: True if th03 produced output, False if empty
        "baseline_tps_thinker_27b": baseline_tps,
        "baseline_quality_27b":     baseline_quality,
    },
    "verdict": "PENDING_HUMAN_REVIEW",
    "notes":   "Quality scoring (1-5 per task) required. See phase2_quality/human_review.md. Also record th03_non_empty.",
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

md = f"""# T2.4 Qwen3.6-27B-AWQ — Arclight Thinker Candidate

**vLLM** 0.19.x | **Model** {metrics['config']['model']} | **Date** {timestamp[:10]}
**Placement** {placement} | **ctx** 32768 | **Parsers** qwen3_coder + reasoning-parser qwen3

## Auto-scored metrics

| Metric | Qwen3.6-27B | Baseline (Qwen3.5-27B) | Notes |
|--------|-------------|------------------------|-------|
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

**Special attention — th03 (architecture_tradeoffs):**
Qwen3.5-27B ALWAYS emits empty output on this task (thinking budget exhaustion — known defect).
Record whether Qwen3.6-27B produces a non-empty answer — this is a key differentiator.
Set `th03_non_empty: true/false` in metrics.json.

**Pay attention to th02 and th05:**
These are the tasks Gemma4-31B (T2.3b) failed on (scores 2 and 3).
- th02: algorithm design with complex multi-constraint scheduling (deadline vs priority ordering)
- th05: distributed consistency edge cases (cache versioning across model hot-swap)
These reveal depth-of-reasoning under production constraints.

After scoring, update **{out}/metrics.json**:
- Set `quality_mean_8task` to the mean score
- Set `th03_non_empty` to true/false
- Set `verdict` to PASS/FAIL/INCONCLUSIVE

**Decision rule (T2.4):**
- PASS: mean >= {baseline_quality}/5 AND tool calling works
- FAIL: mean < {baseline_quality}/5 OR tool calling broken
- INCONCLUSIVE: mean matches baseline but specific task type regression → hand to research

**If PASS on th02 and th05:** Qwen3.6-27B is the new Arclight thinker. Update DECISIONS.md.
**If FAIL on th02/th05 specifically:** consider T2.4b (Qwopus SFT) — see TESTING_QUEUE.md.

**Verdict: PENDING HUMAN REVIEW**
"""
(out / "summary.md").write_text(md)

print(f"[T2.4] TPS={dec_tps_1:.1f} t/s (ratio {tps_ratio:.2f}x vs baseline {baseline_tps})")
print(f"[T2.4] Task completion: {task_completion:.0%}")
print(f"[T2.4] Results: {out}")
print(f"[T2.4] Next: score human_review.md, fill quality_mean_8task + th03_non_empty + verdict in metrics.json.")
PYEOF

log "Done."
log "Human review: ${RESULTS_DIR}/phase2_quality/human_review.md"
log "After scoring: update ${RESULTS_DIR}/metrics.json  (quality_mean_8task, th03_non_empty, verdict)."
