#!/usr/bin/env bash
# Sub-test 2.1 — Coder model quality: Qwen3.5-35B-A3B vs Qwen3-Coder-30B-A3B.
#
# Runs 10 coding quality tasks on both models using quality mode.
# Produces human_review.md for each — open both side-by-side and score.
# Automated check: task_completion_rate (did it produce a response at all).
#
# Decision:
#   If Qwen3.5-35B-A3B scores ≥1 point higher on average → use Qwen3.5-35B
#   If Qwen3-Coder-30B scores higher AND is faster → use Qwen3-Coder-30B
#   Tie → prefer whichever passes Phase 0 tool calling more reliably
#
# Usage:
#   ./benchmarks/phase2_model_selection/run_2.1_coder_quality.sh
#   SKIP_CODER30=1 ./benchmarks/phase2_model_selection/run_2.1_coder_quality.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ENGINE="${ENGINE:-vllm}"
GPU="gpu0"
CTX_LEN=32768
SKIP_CODER30="${SKIP_CODER30:-0}"

TASKS="${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/quality"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

_run_quality() {
    local model="$1" quant="$2" label="$3" extra_args="$4"
    local safe_label="${label//[^a-zA-Z0-9_-]/_}"
    local results_dir="${REPO_ROOT}/results/phase2_2.1_${safe_label}_${TIMESTAMP}"
    mkdir -p "${results_dir}/raw"

    local port_var="PORT_VLLM_GPU0"
    [[ "${ENGINE}" == "sglang" ]] && port_var="PORT_SGLANG_GPU0"

    echo ""
    echo "── ${label} ──────────────────────────────────────────────────────────"
    "${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${model}" \
        --ctx "${CTX_LEN}" ${extra_args}

    local port="${!port_var}"
    python -m benchmarks.phase2_model_selection.bench \
        --endpoint "http://localhost:${port}/v1" \
        --results-dir "${results_dir}" \
        --tasks "${TASKS}" \
        --mode quality \
        --label "${label}" \
        --engine "${ENGINE}" \
        --quantization "${quant}" \
        --gpu-label "RTX 5090" \
        --gpu-id "${GPU_0_ID}" \
        --ctx-len "${CTX_LEN}" \
        --extra-args "${extra_args}" \
        --thresholds "${REPO_ROOT}/config/thresholds.yaml" \
        --max-tokens 1024 \
        --notes "Sub-test 2.1: coder quality comparison" \
        2>&1 | tee "${results_dir}/bench.log" || true

    "${REPO_ROOT}/infra/scripts/teardown.sh"
    echo "${results_dir}"
}

echo "=========================================================="
echo " Sub-test 2.1: Coder model quality comparison"
echo " Engine: ${ENGINE} / ${GPU}"
echo "=========================================================="

QWEN35_RESULTS="$(_run_quality \
    "Qwen/Qwen3.5-35B-A3B-AWQ" "AWQ-INT4" "Qwen3.5-35B-A3B-AWQ" \
    "--tool-call-parser qwen3_coder --reasoning-parser qwen3")"

CODER30_RESULTS=""
if [[ "${SKIP_CODER30}" != "1" ]]; then
    CODER30_RESULTS="$(_run_quality \
        "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ" "AWQ-INT4" "Qwen3-Coder-30B-AWQ" \
        "--tool-call-parser qwen3_coder")"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "── Speed comparison ────────────────────────────────────────────────────"
if [[ -n "${CODER30_RESULTS}" ]]; then
    python -m lib.reporter compare "${QWEN35_RESULTS}" "${CODER30_RESULTS}" \
        --key decode_tps_mean || true
else
    echo "(Qwen3-Coder-30B skipped)"
fi

echo ""
echo "── Human review sheets ─────────────────────────────────────────────────"
echo "  Qwen3.5-35B : ${QWEN35_RESULTS}/human_review.md"
[[ -n "${CODER30_RESULTS}" ]] && echo "  Coder-30B   : ${CODER30_RESULTS}/human_review.md"
echo ""
echo "Open both sheets, score each response 1–5, then decide."
