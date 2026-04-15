#!/usr/bin/env bash
# Phase 0 — Tool-call reliability full run.
# Deploys vLLM on GPU 0 with Qwen3.5-35B-A3B-AWQ, runs all 30 tasks,
# tears down the engine, and generates a summary.
#
# Usage:
#   ./benchmarks/phase0_tool_reliability/run.sh
#
# Override defaults via env vars:
#   ENGINE=sglang GPU=gpu1 MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ \
#     ./benchmarks/phase0_tool_reliability/run.sh

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

# ── Configurable defaults ──────────────────────────────────────────────────────
ENGINE="${ENGINE:-vllm}"
GPU="${GPU:-gpu0}"
MODEL="${MODEL:-QuantTrio/Qwen3.5-35B-A3B-AWQ}"
# 16384: after loading 22 GiB AWQ model, only ~0.57 GiB remains for KV cache.
# --enforce-eager skips CUDA graph profiling which also OOMs at this VRAM budget.
CTX_LEN="${CTX_LEN:-16384}"
QUANT="${QUANT:-AWQ-INT4}"
EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:---tool-call-parser qwen3_coder --reasoning-parser qwen3 --enforce-eager}"
CONCURRENCY="${CONCURRENCY:-1}"

# ── Select port ────────────────────────────────────────────────────────────────
case "${ENGINE}-${GPU}" in
    vllm-gpu0)   PORT="${PORT_VLLM_GPU0}" ;;
    vllm-gpu1)   PORT="${PORT_VLLM_GPU1}" ;;
    sglang-gpu0) PORT="${PORT_SGLANG_GPU0}" ;;
    sglang-gpu1) PORT="${PORT_SGLANG_GPU1}" ;;
    llamacpp-gpu0) PORT="${PORT_LLAMACPP_GPU0}" ;;
    *)
        echo "Unknown engine/gpu: ${ENGINE}/${GPU}" >&2
        exit 1
        ;;
esac

PHASE="phase0_tool_reliability"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/${PHASE}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

echo "=============================================="
echo " Phase 0: Tool-call Reliability"
echo " Engine  : ${ENGINE} / ${GPU}"
echo " Model   : ${MODEL}"
echo " CTX     : ${CTX_LEN}"
echo " Results : ${RESULTS_DIR}"
echo "=============================================="

# ── Deploy ─────────────────────────────────────────────────────────────────────
"${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${MODEL}" \
    --ctx "${CTX_LEN}" \
    ${EXTRA_ENGINE_ARGS}

ENDPOINT="http://localhost:${PORT}/v1"

# ── Bench ──────────────────────────────────────────────────────────────────────
python3 -m benchmarks.phase0_tool_reliability.bench \
    --endpoint "${ENDPOINT}" \
    --model "${MODEL}" \
    --results-dir "${RESULTS_DIR}" \
    --tasks "${REPO_ROOT}/benchmarks/phase0_tool_reliability/tasks/" \
    --engine "${ENGINE}" \
    --quantization "${QUANT}" \
    --gpu-label "RTX 5090" \
    --gpu-id 0 \
    --ctx-len "${CTX_LEN}" \
    --extra-args "${EXTRA_ENGINE_ARGS}" \
    --concurrency "${CONCURRENCY}" \
    --thresholds "${REPO_ROOT}/config/thresholds.yaml" \
    2>&1 | tee "${RESULTS_DIR}/bench.log"

BENCH_EXIT=${PIPESTATUS[0]}

# ── Teardown ───────────────────────────────────────────────────────────────────
"${REPO_ROOT}/infra/scripts/teardown.sh"

# ── Report ─────────────────────────────────────────────────────────────────────
echo ""
echo "--- Summary ---"
cat "${RESULTS_DIR}/summary.md"

exit "${BENCH_EXIT}"
