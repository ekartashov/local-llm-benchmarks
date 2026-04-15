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
# bf16 27B/32B DO NOT fit in 32 GB GDDR7 (confirmed OOM in Phase 2.2 attempt 2026-04-15).
# AWQ INT4 variants needed: ~13-16 GiB each. Override with env vars if repos differ.
QWEN_MODEL="${QWEN_MODEL:-QuantTrio/Qwen3.5-27B-AWQ}"
R1_MODEL="${R1_MODEL:-casperhansen/deepseek-r1-distill-qwen-32b-awq}"

TASKS="${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

_run_thinker() {
    local model="$1" quant="$2" label="$3" extra_args="$4"
    local safe_label="${label//[^a-zA-Z0-9_-]/_}"
    local results_dir="${REPO_ROOT}/results/phase2_2.2_${safe_label}_${TIMESTAMP}"
    mkdir -p "${results_dir}/raw"

    local port_var="PORT_VLLM_GPU0"
    [[ "${ENGINE}" == "sglang" ]] && port_var="PORT_SGLANG_GPU0"

    echo "" >&2
    echo "── ${label} ──────────────────────────────────────────────────────────" >&2
    "${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${model}" \
        --ctx "${CTX_LEN}" ${extra_args} >&2

    local port="${!port_var}"
    python3 -m benchmarks.phase2_model_selection.bench \
        --endpoint "http://localhost:${port}/v1" \
        --model "${model}" \
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
        --max-tokens 8192 \
        --notes "Sub-test 2.2: thinker quality comparison" \
        2>&1 | tee "${results_dir}/bench.log" >&2 || true

    "${REPO_ROOT}/infra/scripts/teardown.sh" >&2
    echo "${results_dir}"
}

echo "=========================================================="
echo " Sub-test 2.2: Thinker model comparison"
echo " Engine: ${ENGINE} / ${GPU}"
echo "=========================================================="

# Qwen3.5-27B AWQ: hybrid SSM architecture (GDN/FLA layers) has large profiling overhead.
# Default vLLM graph profiling needs 1.53 GiB for ~51 batch sizes; only 1.11 GiB free after
# 19.78 GiB model load → OOM. --max-num-seqs 1 limits to 1 batch size (0.04 GiB actual),
# CUDA graphs stay active. Confirmed 76 t/s (Phase 2.2, 2026-04-15).
# Env override: QWEN_EXTRA_ARGS to use --enforce-eager if needed (cuts to 24 t/s).
QWEN_EXTRA_ARGS="${QWEN_EXTRA_ARGS:---tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-num-seqs 1}"
QWEN_RESULTS="$(_run_thinker \
    "${QWEN_MODEL}" "AWQ-INT4" "Qwen3.5-27B-AWQ" \
    "${QWEN_EXTRA_ARGS}")"

R1_RESULTS=""
if [[ "${SKIP_R1}" != "1" ]]; then
    R1_RESULTS="$(_run_thinker \
        "${R1_MODEL}" "AWQ-INT4" "DeepSeek-R1-32B-AWQ" \
        "--reasoning-parser deepseek_r1")"
fi

echo ""
echo "── Speed comparison ────────────────────────────────────────────────────"
if [[ -n "${R1_RESULTS}" ]]; then
    python3 -m lib.reporter compare "${QWEN_RESULTS}" "${R1_RESULTS}" \
        --key decode_tps_mean || true
fi

echo ""
echo "── Human review sheets ─────────────────────────────────────────────────"
echo "  Qwen3.5-27B  : ${QWEN_RESULTS}/human_review.md"
[[ -n "${R1_RESULTS}" ]] && echo "  R1-32B       : ${R1_RESULTS}/human_review.md"
echo ""
echo "Score reasoning depth, correctness, and actionability (1–5 per task)."
