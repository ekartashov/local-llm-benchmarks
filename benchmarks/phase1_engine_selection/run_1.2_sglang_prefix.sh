#!/usr/bin/env bash
# Sub-test 1.2 — SGLang prefix reuse (RadixAttention KV-cache hit rate).
#
# SGLang's RadixAttention caches KV states for shared prefixes.
# This test measures how much TTFT drops on the 2nd+ request with the same prefix.
#
# Pass criterion: prefix_reuse_speedup ≤ 0.50
#   (warm TTFT is at most 50% of cold TTFT — cache cuts TTFT by ≥50%)
#
# Tasks share a ~2100-token system prompt (realistic coding-agent context).
# warmup-rounds=3 ensures the prefix is fully cached before measurement.
#
# Usage:
#   ./benchmarks/phase1_engine_selection/run_1.2_sglang_prefix.sh

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

ENGINE="sglang"
GPU="gpu0"
MODEL="${MODEL:-QuantTrio/Qwen3.5-35B-A3B-AWQ}"
CTX_LEN=16384
QUANT="AWQ-INT4"
# SGLang uses qwen3_coder (not qwen3) for Qwen3.5 MoE tool calling.
# --quantization awq_marlin: forces awq_marlin loading path which handles
# per-expert weight tensors (experts.N.w2.weight) vs the default qwen3_5 loader
# which expects fused experts.w2_weight tensors (KeyError if missing).
EXTRA_ENGINE_ARGS="--tool-call-parser qwen3_coder --quantization awq_marlin"

TASKS_DIR="${REPO_ROOT}/benchmarks/phase1_engine_selection/tasks/prefix_cache"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/phase1_1.2_sglang_prefix_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

echo "============================================="
echo " Sub-test 1.2: SGLang prefix reuse"
echo " Model   : ${MODEL}"
echo " Results : ${RESULTS_DIR}"
echo "============================================="

"${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${MODEL}" \
    --ctx "${CTX_LEN}" ${EXTRA_ENGINE_ARGS}

python3 -m benchmarks.phase1_engine_selection.bench \
    --endpoint "http://localhost:${PORT_SGLANG_GPU0}/v1" \
    --results-dir "${RESULTS_DIR}" \
    --tasks "${TASKS_DIR}" \
    --mode prefix-cache \
    --warmup-rounds 3 \
    --engine "${ENGINE}" \
    --quantization "${QUANT}" \
    --gpu-label "RTX 5090" \
    --gpu-id "${GPU_0_ID}" \
    --ctx-len "${CTX_LEN}" \
    --extra-args "${EXTRA_ENGINE_ARGS}" \
    --thresholds "${REPO_ROOT}/config/thresholds.yaml" \
    --notes "Sub-test 1.2: SGLang RadixAttention prefix reuse. Pass=ratio≤0.50." \
    2>&1 | tee "${RESULTS_DIR}/bench.log"

BENCH_EXIT=${PIPESTATUS[0]}
"${REPO_ROOT}/infra/scripts/teardown.sh"

VERDICT="$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/metrics.json')); print(d.get('verdict','?'))" 2>/dev/null || echo "?")"
echo ""
echo "Verdict: ${VERDICT}"
case "${VERDICT}" in
    PASS)        echo "✓ SGLang prefix reuse passes — RadixAttention is working." ;;
    INCONCLUSIVE) echo "⚠ Marginal speedup — check if --enable-torch-compile is active." ;;
    FAIL)         echo "✗ Prefix reuse not working. Check SGLang RadixAttention is enabled (default on)." ;;
esac

cat "${RESULTS_DIR}/summary.md"
exit "${BENCH_EXIT}"
