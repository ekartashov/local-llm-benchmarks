#!/usr/bin/env bash
# Sub-test 1.4 — llama.cpp throughput baseline comparison.
#
# Runs the same throughput tasks as 1.1 but against llama.cpp with the
# GGUF variant of the model. Establishes whether the speed gap between
# llama.cpp and vLLM/SGLang justifies ruling it out for the daily driver.
#
# Expected outcome: llama.cpp decode speed < vLLM/SGLang by 20-50%
# (fewer optimizations for MoE routing, no Flash Attention v3 support yet).
#
# Note: requires the GGUF to be present in MODEL_CACHE.
# Default: Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf (Unsloth UD variant)
#
# Usage:
#   ./benchmarks/phase1_engine_selection/run_1.4_llamacpp.sh
#   MODEL_FILE=Qwen3.5-35B-A3B-Q4_K_M.gguf \
#     ./benchmarks/phase1_engine_selection/run_1.4_llamacpp.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# Activate Python environment: try .venv symlink, then pyenv hf, then fail loud.
if [[ -f "${REPO_ROOT}/.venv/bin/python3" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
elif command -v pyenv &>/dev/null && pyenv activate hf 2>/dev/null; then
    : # activated via pyenv hf
elif ! python3 -c "import openai" &>/dev/null 2>&1; then
    echo "[ERROR] No Python env with openai found. Run: pyenv activate hf" >&2
    exit 1
fi

ENGINE="llamacpp"
GPU="gpu0"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf}"
CTX_LEN=32768
QUANT="GGUF-Q4_K_XL"

# All layers on GPU (avoids llama.cpp issue #19480 — 5× CPU slowdown)
# No spec decode (llama.cpp PR #20075 — crashes on MoE)
EXTRA_ENGINE_ARGS="--n-gpu-layers 99 --flash-attn --jinja"

TASKS_DIR="${REPO_ROOT}/benchmarks/phase1_engine_selection/tasks/throughput"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/phase1_1.4_llamacpp_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

echo "============================================="
echo " Sub-test 1.4: llama.cpp throughput baseline"
echo " Model file: ${MODEL_FILE}"
echo " Results   : ${RESULTS_DIR}"
echo "============================================="

GGUF_PATH="${MODEL_CACHE}/${MODEL_FILE}"
if [[ ! -f "${GGUF_PATH}" ]]; then
    echo "ERROR: GGUF not found at ${GGUF_PATH}" >&2
    echo "Run: hf download unsloth/Qwen3.5-35B-A3B-GGUF --include '${MODEL_FILE}' --local-dir ${MODEL_CACHE}" >&2
    exit 1
fi

export MODEL_FILE
"${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "" \
    --ctx "${CTX_LEN}" ${EXTRA_ENGINE_ARGS}

python3 -m benchmarks.phase1_engine_selection.bench \
    --endpoint "http://localhost:${PORT_LLAMACPP_GPU0}/v1" \
    --results-dir "${RESULTS_DIR}" \
    --tasks "${TASKS_DIR}" \
    --mode throughput \
    --concurrency 1 \
    --engine "${ENGINE}" \
    --quantization "${QUANT}" \
    --gpu-label "RTX 5090" \
    --gpu-id "${GPU_0_ID}" \
    --ctx-len "${CTX_LEN}" \
    --extra-args "${EXTRA_ENGINE_ARGS}" \
    --thresholds "${REPO_ROOT}/config/thresholds.yaml" \
    --notes "Sub-test 1.4: llama.cpp baseline. Compare to vLLM (1.1) decode_tps_mean." \
    2>&1 | tee "${RESULTS_DIR}/bench.log"

BENCH_EXIT=${PIPESTATUS[0]}
"${REPO_ROOT}/infra/scripts/teardown.sh"

# Compare to most recent vLLM 1.1 result
VLLM_RESULTS="$(ls -td "${REPO_ROOT}/results/phase1_1.1_vllm_"* 2>/dev/null | head -1 || echo "")"
if [[ -n "${VLLM_RESULTS}" ]]; then
    echo ""
    echo "── Comparison vs vLLM (1.1) ─────────────────────────────────────────────"
    python3 -m lib.reporter compare "${VLLM_RESULTS}" "${RESULTS_DIR}" \
        --key decode_tps_mean || true
fi

cat "${RESULTS_DIR}/summary.md"
exit "${BENCH_EXIT}"
