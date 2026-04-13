#!/usr/bin/env bash
# Sub-test 2.2 — Thinker model comparison: Qwen3.5-27B vs DeepSeek-R1-32B.
#
# Runs 8 reasoning-heavy tasks on both dense thinker models.
# These tasks have multi-step answers where quality matters more than speed.
#
# Both models are bf16 dense — they fit in 32 GB GDDR7 with ≥2 GB to spare.
# Spec-decode is viable for these models (Phase 4.3) — NOT tested here.
#
# Decision:
#   Score human_review.md (1–5 per task). Pick the higher-quality model.
#   If tied: prefer the faster one (decode_tps_mean).
#   If neither reaches 80% of cloud quality (Phase 6 baseline), consider
#   running without a thinker and relying on the coder model only.
#
# Usage:
#   ./benchmarks/phase2_model_selection/run_2.2_thinker.sh
#   SKIP_R1=1 ./benchmarks/phase2_model_selection/run_2.2_thinker.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ENGINE="${ENGINE:-vllm}"
GPU="gpu0"
CTX_LEN=32768
SKIP_R1="${SKIP_R1:-0}"

TASKS="${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

_run_thinker() {
    local model="$1" quant="$2" label="$3" extra_args="$4"
    local safe_label="${label//[^a-zA-Z0-9_-]/_}"
    local results_dir="${REPO_ROOT}/results/phase2_2.2_${safe_label}_${TIMESTAMP}"
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
        --max-tokens 2048 \
        --notes "Sub-test 2.2: thinker quality comparison" \
        2>&1 | tee "${results_dir}/bench.log" || true

    "${REPO_ROOT}/infra/scripts/teardown.sh"
    echo "${results_dir}"
}

echo "=========================================================="
echo " Sub-test 2.2: Thinker model comparison"
echo " Engine: ${ENGINE} / ${GPU}"
echo "=========================================================="

QWEN_RESULTS="$(_run_thinker \
    "Qwen/Qwen3.5-27B" "bf16" "Qwen3.5-27B-bf16" \
    "--reasoning-parser qwen3")"

R1_RESULTS=""
if [[ "${SKIP_R1}" != "1" ]]; then
    R1_RESULTS="$(_run_thinker \
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "bf16" "DeepSeek-R1-32B-bf16" \
        "--reasoning-parser deepseek_r1")"
fi

echo ""
echo "── Speed comparison ────────────────────────────────────────────────────"
if [[ -n "${R1_RESULTS}" ]]; then
    python -m lib.reporter compare "${QWEN_RESULTS}" "${R1_RESULTS}" \
        --key decode_tps_mean || true
fi

echo ""
echo "── Human review sheets ─────────────────────────────────────────────────"
echo "  Qwen3.5-27B  : ${QWEN_RESULTS}/human_review.md"
[[ -n "${R1_RESULTS}" ]] && echo "  R1-32B       : ${R1_RESULTS}/human_review.md"
echo ""
echo "Score reasoning depth, correctness, and actionability (1–5 per task)."
