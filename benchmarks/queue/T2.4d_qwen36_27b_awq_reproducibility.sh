#!/usr/bin/env bash
# T2.4d_qwen36_27b_awq_reproducibility.sh
#
# AWQ run 4 reproducibility test — run exact run 4 config three times.
# Determines whether run 4's correct th02 was stable or a lucky sample.
#
# Run 4 config (only clean-passing run from T2.4):
#   Model:       QuantTrio/Qwen3.6-27B-AWQ
#   Placement:   TP=1 GPU1
#   KV cache:    fp8
#   ctx:         32768
#   max_tokens:  16384
#   max-num-seqs 1
#
# Optional corrections from T2.4f (set via env before running):
#   ROPE_THETA_FLAG="--rope-theta 10000000"
#     (or whatever override flag vLLM 0.19.x supports — check findings.md)
#   DISABLE_CHUNKED_PREFILL=1
#     adds --enable-chunked-prefill=False (tests H2)
#
# Decision gate:
#   th02 correct >=2/3 → reproducible → proceed to T2.4e
#   th02 correct <=1/3 → capability ceiling → consider T2.4b (Qwopus SFT)
#
# th02 scoring:
#   CORRECT       = ALL jobs assigned, including misses (→ busiest GPU)
#   SEMANTIC ERROR = missed jobs assigned to -1 (not processed)
#                   model argues this is correct — it is not
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY_RUN=1; shift ;;
        *) shift ;;
    esac
done

# ── Corrections from T2.4f ─────────────────────────────────────────────────────
# ROPE_THETA_FLAG: set to override rope_theta if T2.4f found mismatch, e.g.:
#   export ROPE_THETA_FLAG="--rope-theta 10000000"
#   export ROPE_THETA_FLAG="--hf-overrides '{\"rope_theta\": 10000000}'"
# Leave empty if T2.4f showed H1 is OK.
ROPE_THETA_FLAG="${ROPE_THETA_FLAG:-}"

# DISABLE_CHUNKED_PREFILL: set to 1 to add --enable-chunked-prefill=False (test H2).
# When set, runs Variant A (standard) AND Variant B (no chunked prefill) in sequence.
DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-}"

ITEM_ID="T2.4d_qwen36_27b_awq_reproducibility"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="QuantTrio/Qwen3.6-27B-AWQ"

mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/bench.log"

log() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "[T2.4d] $*" | tee -a "${LOG}"
    fi
}
die() { log "FATAL: $*"; exit 1; }

log "=== T2.4d Qwen3.6-27B-AWQ Reproducibility Test (run 4 config ×3) ==="
log "Timestamp:             ${TIMESTAMP}"
log "ROPE_THETA_FLAG:       '${ROPE_THETA_FLAG}'"
log "DISABLE_CHUNKED_PREFILL: '${DISABLE_CHUNKED_PREFILL}'"
log ""
log "Config: AWQ TP=1 GPU1, fp8 KV, ctx=32768, max_tokens=16384, max-num-seqs 1"
log ""
log "th02 scoring:"
log "  CORRECT       = all jobs assigned, including misses → busiest GPU"
log "  SEMANTIC ERROR = missed jobs returned as -1 (not assigned)"

# ── Helper: deploy on GPU1 ────────────────────────────────────────────────────
_deploy() {
    local label="$1"; shift
    local extra_args=("$@")
    log ""
    log "--- Deploy: ${label} ---"
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
        "${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
        --gpu-mem-util 0.90 \
        --ctx 32768 \
        --kv-cache-dtype fp8 \
        --max-num-seqs 1 \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        "${extra_args[@]+"${extra_args[@]}"}" \
        2>&1 | tee -a "${LOG}" \
        || die "Deployment failed for ${label}."
    log "Deployed. Endpoint: http://localhost:${PORT_VLLM_GPU1}/v1"
}

# ── Helper: stop GPU1 container ───────────────────────────────────────────────
_stop_gpu1() {
    podman stop bench-vllm-gpu1 2>/dev/null || true
    podman rm   bench-vllm-gpu1 2>/dev/null || true
}

# ── Helper: run one bench pass ────────────────────────────────────────────────
_run_bench() {
    local run_dir="$1"
    local label="$2"
    mkdir -p "${run_dir}"
    python3 -m benchmarks.phase2_model_selection.bench \
        --endpoint "http://localhost:${PORT_VLLM_GPU1}/v1" \
        --results-dir "${run_dir}" \
        --mode quality \
        --tasks "${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker/" \
        --model "${MODEL}" \
        --label "${label}" \
        --max-tokens 16384 \
        2>&1 | tee -a "${LOG}" || {
        log "WARNING: bench exited non-zero for ${label} — check logs."
    }
    log "Bench complete: ${run_dir}/human_review.md"
}

# ── Build base extra args ──────────────────────────────────────────────────────
BASE_EXTRA_ARGS=()
# shellcheck disable=SC2206
[[ -n "${ROPE_THETA_FLAG}" ]] && BASE_EXTRA_ARGS+=(${ROPE_THETA_FLAG})

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "WOULD DEPLOY: vllm gpu1 ${MODEL} ctx=32768 fp8 max-num-seqs=1 ${ROPE_THETA_FLAG}"
    log "WOULD RUN bench ×3 → run1/, run2/, run3/"
    if [[ -n "${DISABLE_CHUNKED_PREFILL}" ]]; then
        log "WOULD ALSO RUN: Variant B (--disable-chunked-prefill) × 1 → variant_b/"
    fi
else
    # Warmup helper (small inline python — no temp file needed)
    _warmup() {
        python3 -c "
import httpx, asyncio
async def w():
    async with httpx.AsyncClient(timeout=120) as c:
        try:
            await c.post('http://localhost:${PORT_VLLM_GPU1}/v1/chat/completions',
                json={'model': '${MODEL}',
                      'messages': [{'role': 'user', 'content': 'hi'}],
                      'max_tokens': 10})
        except Exception:
            pass
asyncio.run(w())
" 2>/dev/null || true
    }

    # ── Variant A: standard run 4 config × 3 ──────────────────────────────────
    log ""
    log "=== Variant A: standard run 4 config (fp8 KV, no extra flags) ==="
    _deploy "Variant A" "${BASE_EXTRA_ARGS[@]+"${BASE_EXTRA_ARGS[@]}"}"

    log "Warmup request..."
    _warmup

    for RUN_NUM in 1 2 3; do
        log ""
        log "--- Variant A / Run ${RUN_NUM}/3 ---"
        _run_bench "${RESULTS_DIR}/run${RUN_NUM}" "Qwen3.6-27B-AWQ-run4-A${RUN_NUM}"
    done

    # ── Variant B: --disable-chunked-prefill (optional, tests H2) ─────────────
    if [[ -n "${DISABLE_CHUNKED_PREFILL}" ]]; then
        log ""
        log "=== Variant B: --disable-chunked-prefill (tests H2 — GDN recurrence) ==="
        _stop_gpu1
        _deploy "Variant B" \
            "${BASE_EXTRA_ARGS[@]+"${BASE_EXTRA_ARGS[@]}"}" \
            --no-enable-chunked-prefill

        log "Warmup request..."
        _warmup

        log ""
        log "--- Variant B / Run 1/1 ---"
        _run_bench "${RESULTS_DIR}/variant_b" "Qwen3.6-27B-AWQ-run4-B1-no-chunked-prefill"
    fi

    _stop_gpu1
    log "Container stopped."
fi

# ── Write summary.md ──────────────────────────────────────────────────────────
python3 - <<PYEOF
import json, pathlib

out = pathlib.Path("${RESULTS_DIR}")
ts = "${TIMESTAMP}"
rope_flag = "${ROPE_THETA_FLAG}".strip()
cp_flag = "1" if "${DISABLE_CHUNKED_PREFILL}".strip() else ""

config_str = "AWQ-INT4, TP=1 GPU1, fp8 KV, ctx=32768, max_tokens=16384, max-num-seqs 1"
if rope_flag:
    config_str += f", {rope_flag}"

variant_b_section = ""
if cp_flag:
    variant_b_section = """
## Variant B (--disable-chunked-prefill)

Score th02 in variant_b/human_review.md.
Compare to Variant A: if th02 correct in B but wrong in A, H2 (GDN recurrence) confirmed.

| Variant B | th02 result | th02 correct? | Mean quality | Notes |
|-----------|-------------|---------------|--------------|-------|
| Run 1 | | | | variant_b/human_review.md |
"""

md = f"""# T2.4d Qwen3.6-27B-AWQ — Reproducibility Test (run 4 config ×3)

**Model:** QuantTrio/Qwen3.6-27B-AWQ
**Config:** {config_str}
**Timestamp:** {ts}

## th02 scoring criteria

**CORRECT:** All jobs assigned to a GPU. Jobs that miss their deadline are assigned
to the **busiest GPU** (not dropped). The scheduler must handle misses explicitly.

**SEMANTIC ERROR:** Missed jobs returned as -1 or not assigned to any GPU.
The model argues "-1 means not processed, which is correct if it misses everywhere."
This is the confident incorrectness pattern from all prior T2.4 runs.
It is NOT correct — missed jobs should still be assigned to the busiest GPU.

## Variant A — standard run 4 config

Score th02 in each run's human_review.md.

| Run | th02 result | th02 correct? | Mean quality (1–5) | Notes |
|-----|-------------|---------------|--------------------|-------|
| Run 1 | | | | run1/human_review.md |
| Run 2 | | | | run2/human_review.md |
| Run 3 | | | | run3/human_review.md |

{variant_b_section}
## Decision gate

| th02 correct | Count | Verdict | Next action |
|--------------|-------|---------|-------------|
| ≥2/3 runs | | **REPRODUCIBLE** | Run 4 is stable. Proceed to T2.4e (bf16 KV + TP=2). |
| ≤1/3 runs | | **CAPABILITY CEILING** | Model cannot reliably solve this problem class at 27B. Proceed to T2.4b (Qwopus SFT) or accept Qwen3.5-27B as permanent thinker. |

If Variant B fixes th02 but Variant A does not: **H2 (GDN + chunked prefill) confirmed**.
Correct fix is to add --enable-chunked-prefill=False to all Qwen3.6-27B deployments.

## Files

- run1/human_review.md — first full 8-task run
- run2/human_review.md — second full 8-task run
- run3/human_review.md — third full 8-task run
- variant_b/human_review.md — H2 test run (if DISABLE_CHUNKED_PREFILL was set)
- bench.log — full deploy + run log
"""
(out / "summary.md").write_text(md)
print(f"[T2.4d] summary.md → {out}/summary.md")
PYEOF

log ""
log "=== T2.4d complete ==="
log "Results:  ${RESULTS_DIR}"
log "Summary:  ${RESULTS_DIR}/summary.md"
log ""
log "Next:"
log "  1. Score th02 in run1/, run2/, run3/human_review.md"
log "  2. Fill in the decision table in summary.md"
log "  3. th02 correct >=2/3 → run T2.4e"
log "     th02 correct <=1/3 → capability ceiling → T2.4b (Qwopus)"
