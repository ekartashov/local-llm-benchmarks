#!/usr/bin/env bash
# T2.5_qwen36_35b_coder_shootout.sh — solo benchmark for Qwen3.6-35B-A3B-AWQ-4bit
# as a coder candidate vs the Qwen3-Coder-30B-A3B-AWQ baseline.
#
# Deployment plan:
#   1. Try TP=1 on gpu0 (port 30000), gpu-mem-util=0.85 (22 GiB weights, ~7–8 GiB headroom).
#   2. If startup fails during CUDA graph capture, auto-fall back to TP=2 on tp2c (port 30002).
#
# Suite: Phase 0 tool-reliability (9 tasks) + Phase 2 quality coder suite (10 tasks).
# Comparison baseline: Qwen3-Coder-30B-A3B-AWQ (251 t/s, 100% tool pass).
#
# IMPORTANT: do NOT add --reasoning-parser. Qwen3-Next-80B (same family) showed 100%
# failures with all reasoning-parser combos. The qwen3_coder tool-call parser handles
# content that includes <think>…</think> tags. If all results are no_call, try hermes parser.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

ITEM_ID="T2.5_coder_shootout_qwen36_35b_a3b_vs_qwen3coder30b"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log()  { echo "[T2.5] $*" | tee -a "${LOG}"; }
die()  { log "FATAL: $*"; exit 1; }

# ── Baseline numbers for comparison ───────────────────────────────────────────
BASELINE_TPS=251
BASELINE_TOOL_RATE=1.00

# ── Write decode measurement helper ───────────────────────────────────────────
MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
"""T2.5 decode measurement helper."""
import asyncio, httpx, json, sys, time

DECODE_PROMPT = (
    "Write a complete Python implementation of a red-black tree with insert, "
    "delete, search, and in-order traversal. Include full type hints. "
    "Do not stop until the implementation is fully complete."
)
MAX_DECODE_TOKENS = 1500

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
        "text_snippet": "".join(text_out)[:200],
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
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
)
asyncio.run(measure(endpoint, model, num_seqs, out_path))
PYEOF

# ── Step 1: Try TP=1 on gpu0, fall back to TP=2 if CUDA graph capture fails ──
PLACEMENT=""
ENDPOINT=""

log "Attempting TP=1 deployment on gpu0 (22 GiB weights, tight headroom) ..."
# Added VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 to prevent OOM during CUDA graph capture on V1 engine.
if VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 "${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu0 "${MODEL}" \
        --gpu-mem-util 0.95 \
        --ctx 32768 \
        --tool-call-parser qwen3_coder \
        2>&1 | tee -a "${LOG}"; then
    PLACEMENT="tp=1 (gpu0)"
    ENDPOINT="http://localhost:${PORT_VLLM_GPU0}/v1"
    log "TP=1 deploy succeeded. Using gpu0."
else
    log "TP=1 deploy FAILED (likely CUDA graph OOM). Falling back to TP=2 on tp2c ..."
    log "NOTE: TP=2 borrows both GPUs. Any concurrent coder/thinker instances on gpu0/gpu1 must be stopped first."
    # Stop any conflicting containers
    for c in bench-vllm-gpu0 bench-vllm-gpu1; do
        if podman container exists "${c}" 2>/dev/null; then
            log "Stopping conflicting container ${c} ..."
            podman stop "${c}" 2>/dev/null || true
            podman rm   "${c}" 2>/dev/null || true
        fi
    done
    "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2c "${MODEL}" \
        --gpu-mem-util 0.85 \
        --ctx 32768 \
        --tool-call-parser qwen3_coder \
        2>&1 | tee -a "${LOG}"
    PLACEMENT="tp=2 (tp2c)"
    ENDPOINT="http://localhost:${PORT_VLLM_TP2_C}/v1"
    log "TP=2 fallback deploy succeeded."
fi

log "Deployment: ${PLACEMENT} | Endpoint: ${ENDPOINT}"

# ── Step 2: Warmup ─────────────────────────────────────────────────────────────
log "Warmup request ..."
python3 "${MEASURE_PY}" measure "${ENDPOINT}" "${MODEL}" 1 /dev/null 2>/dev/null || true

# ── Step 3: Decode TPS — seq=1 and seq=4 ─────────────────────────────────────
log "Measuring decode TPS at seq=1 ..."
python3 "${MEASURE_PY}" measure "${ENDPOINT}" "${MODEL}" 1 \
    "${RESULTS_DIR}/raw/decode_seq1.json" | tee -a "${LOG}"
DEC_TPS_1=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/decode_seq1.json'))['avg_tps_per_seq'])")

log "Measuring decode TPS at seq=4 ..."
python3 "${MEASURE_PY}" measure "${ENDPOINT}" "${MODEL}" 4 \
    "${RESULTS_DIR}/raw/decode_seq4.json" | tee -a "${LOG}"
DEC_TPS_4=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/decode_seq4.json'))['total_tps'])")

log "Decode: seq=1 ${DEC_TPS_1} t/s  seq=4 total ${DEC_TPS_4} t/s  (baseline: ${BASELINE_TPS} t/s)"

# ── Step 4: Phase 0 tool-reliability suite ────────────────────────────────────
log "Running Phase 0 tool-reliability suite (9 tasks) ..."
python3 -m benchmarks.phase0_tool_reliability.bench \
    --endpoint "${ENDPOINT}" \
    --results-dir "${RESULTS_DIR}/phase0" \
    --tasks "${REPO_ROOT}/benchmarks/phase0_tool_reliability/tasks/" \
    --max-tokens 2048 \
    2>&1 | tee -a "${LOG}" || {
    log "WARNING: phase0 bench exited non-zero — check for exceptions."
}

TOOL_PASS_RATE="0.0"
TOOL_PASS_COUNT="0"
TOOL_TOTAL="0"
if [[ -f "${RESULTS_DIR}/phase0/metrics.json" ]]; then
    TOOL_PASS_RATE=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/phase0/metrics.json')); print(d['metrics'].get('tool_call_success_rate', 0.0))")
    TOOL_PASS_COUNT=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/phase0/metrics.json')); print(d['metrics'].get('counts', {}).get('pass', 0))")
    TOOL_TOTAL=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/phase0/metrics.json')); print(d['metrics'].get('total', 0))")
fi
log "Phase 0 result: ${TOOL_PASS_COUNT}/${TOOL_TOTAL} PASS (${TOOL_PASS_RATE})"

# ── Sanity alert: if tool calls are all no_call, suggest hermes parser ─────────
NO_CALL_COUNT=$(python3 -c "
import json, pathlib
try:
    d = json.load(open('${RESULTS_DIR}/phase0/metrics.json'))
    print(d['metrics'].get('counts', {}).get('no_call', 0))
except Exception:
    print(0)
" 2>/dev/null || echo 0)

if (( NO_CALL_COUNT > 5 )); then
    log "ALERT: ${NO_CALL_COUNT} tasks scored no_call — qwen3_coder parser may not work for this model."
    log "SUGGESTION: stop this script, redeploy with '--tool-call-parser hermes' instead, and rerun."
    log "This pattern was observed with Qwen3-Next-80B-A3B (same family) — hermes was the fix."
fi

# ── Step 5: Phase 2 quality suite (coder, 10 tasks) — output for human review ─
log "Running Phase 2 quality suite (coder, 10 tasks) ..."
python3 -m benchmarks.phase2_model_selection.bench \
    --endpoint "${ENDPOINT}" \
    --results-dir "${RESULTS_DIR}/phase2_quality" \
    --mode quality \
    --tasks "${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/quality/" \
    --label "Qwen3.6-35B-A3B-AWQ" \
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
log "Phase 2 quality task completion rate: ${TASK_COMPLETION_RATE}"

# ── Step 6: Compute verdict and write metrics ──────────────────────────────────
python3 - <<PYEOF
import json, pathlib

item_id   = "${ITEM_ID}"
timestamp = "${TIMESTAMP}"
out       = pathlib.Path("${RESULTS_DIR}")
placement = "${PLACEMENT}"

dec_tps_1         = float("${DEC_TPS_1}")
dec_tps_4         = float("${DEC_TPS_4}")
tool_rate         = float("${TOOL_PASS_RATE}")
task_completion   = float("${TASK_COMPLETION_RATE}")
baseline_tps      = float("${BASELINE_TPS}")
baseline_tool     = float("${BASELINE_TOOL_RATE}")
no_call_count     = int("${NO_CALL_COUNT}")

# TPS regression vs baseline
tps_regression = (baseline_tps - dec_tps_1) / baseline_tps if baseline_tps > 0 else 0

# Thresholds from T2.5 in TESTING_QUEUE.md / thresholds.yaml T2.2 (shared criteria)
PASS_TOOL   = 0.95
INCON_TOOL  = 0.80
PASS_REGR   = 0.20   # ≤20% TPS regression
INCON_REGR  = 0.40

parser_warning = no_call_count > 5

# Verdict: auto-scoreable part only; quality_win_fraction requires human scores
if parser_warning:
    verdict = "FAIL"
    verdict_reason = f"Parser ineffective ({no_call_count} no_call). Retry with --tool-call-parser hermes."
elif tool_rate >= PASS_TOOL and tps_regression <= PASS_REGR:
    verdict = "PASS (auto)"
    verdict_reason = "Tool reliability PASS, TPS regression within threshold. Quality requires human scoring."
elif tool_rate >= INCON_TOOL and tps_regression <= INCON_REGR:
    verdict = "INCONCLUSIVE"
    verdict_reason = "Partial tool reliability or higher TPS regression. Check quality scores."
else:
    verdict = "FAIL"
    verdict_reason = f"Tool rate {tool_rate:.0%} (need ≥80%) or TPS regression {tps_regression:.0%} too large."

metrics = {
    "item_id":   item_id,
    "timestamp": timestamp,
    "config": {
        "engine":         "vllm",
        "engine_version": "0.19.0",
        "model":          "${MODEL}",
        "quantization":   "AWQ-INT4",
        "placement":      placement,
        "context_length": 32768,
        "extra_args":     "--tool-call-parser qwen3_coder",
    },
    "metrics": {
        "decode_tps_seq1":          dec_tps_1,
        "decode_tps_seq4_total":    dec_tps_4,
        "tps_regression_vs_baseline": round(tps_regression, 4),
        "tool_call_pass_rate":      tool_rate,
        "tool_no_call_count":       no_call_count,
        "task_completion_rate":     task_completion,
        "baseline_tps_coder_30b":   baseline_tps,
        "baseline_tool_rate":       baseline_tool,
    },
    "verdict": verdict,
    "notes": verdict_reason,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

def fmt(v, p, i): return "PASS" if v >= p else ("INCON" if v >= i else "FAIL")
def fmt_r(v, p, i): return "PASS" if v <= p else ("INCON" if v <= i else "FAIL")

md = f"""# T2.5 Qwen3.6-35B-A3B Coder Shootout — {verdict}

**vLLM** 0.19.0 | **Model** {metrics['config']['model']} | **Date** {timestamp[:10]}
**Placement** {placement} | **ctx** 32768 | **Parser** qwen3_coder (no reasoning-parser)

## Auto-scored metrics

| Metric | Qwen3.6-35B-A3B | Baseline (30B) | Pass / Incon threshold | Result |
|--------|-----------------|----------------|------------------------|--------|
| Decode TPS (seq=1) | {dec_tps_1:.1f} t/s | {baseline_tps} t/s | regression ≤20% / ≤40% | {fmt_r(tps_regression, 0.20, 0.40)} |
| TPS regression | {tps_regression:.1%} | — | ≤20% / ≤40% | {fmt_r(tps_regression, 0.20, 0.40)} |
| Tool-call pass rate | {tool_rate:.0%} | {baseline_tool:.0%} | ≥95% / ≥80% | {fmt(tool_rate, 0.95, 0.80)} |
| Task completion rate | {task_completion:.0%} | — | (orientation) | - |
| Decode capacity (seq=4) | {dec_tps_4:.1f} t/s total | — | — | - |

{"⚠️  **PARSER ALERT:** " + str(no_call_count) + " tasks returned no_call — qwen3_coder parser may be ineffective. Redeploy with --tool-call-parser hermes and rerun." if parser_warning else ""}

## Human review required
"""
md += f"""
Quality scoring (1–5 per task) is needed to determine whether Qwen3.6-35B-A3B wins the
coder role. Open `{RESULTS_DIR}/phase2_quality/human_review.md` to score all 10 tasks.

**Decision rule (T2.5):**
- WINS: ≥95% tool reliability AND quality sum ≥ baseline on ≥60% of tasks AND TPS regression ≤20%
- LOSES: fails on tool reliability OR loses quality on ≥60% of tasks
- MIXED: hand back to research with the task-level breakdown

**Verdict: {verdict}**
{verdict_reason}
"""
(out / "summary.md").write_text(md)

print(f"[T2.5] Verdict: {verdict}")
print(f"[T2.5] TPS={dec_tps_1:.1f} t/s  (baseline {baseline_tps} t/s, regression {tps_regression:.1%})")
print(f"[T2.5] Tool pass rate: {tool_rate:.0%}  no_call: {no_call_count}")
print(f"[T2.5] Results: {out}")
PYEOF
