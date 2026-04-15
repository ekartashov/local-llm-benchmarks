#!/usr/bin/env bash
# Sub-test 1.3 — vLLM prefix caching (automatic KV cache reuse).
#
# vLLM caches KV states via automatic prefix caching (enabled by default in
# recent versions via --enable-chunked-prefill). This test measures TTFT
# reduction when the same long system prompt is reused across requests.
#
# Pass criterion: prefix_reuse_speedup ≤ 0.50
#
# Compare result to 1.2 (SGLang) to inform the Phase 1 engine decision.
#
# Usage:
#   ./benchmarks/phase1_engine_selection/run_1.3_vllm_prefix.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ENGINE="vllm"
GPU="gpu0"
MODEL="${MODEL:-QuantTrio/Qwen3.5-35B-A3B-AWQ}"
CTX_LEN=32768
QUANT="AWQ-INT4"

# --enable-chunked-prefill activates automatic prefix caching in vLLM
EXTRA_ENGINE_ARGS="--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-chunked-prefill"

TASKS_DIR="${REPO_ROOT}/benchmarks/phase1_engine_selection/tasks/prefix_cache"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/phase1_1.3_vllm_prefix_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

echo "============================================="
echo " Sub-test 1.3: vLLM prefix caching"
echo " Model   : ${MODEL}"
echo " Results : ${RESULTS_DIR}"
echo "============================================="

"${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${MODEL}" \
    --ctx "${CTX_LEN}" ${EXTRA_ENGINE_ARGS}

python -m benchmarks.phase1_engine_selection.bench \
    --endpoint "http://localhost:${PORT_VLLM_GPU0}/v1" \
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
    --notes "Sub-test 1.3: vLLM automatic prefix caching. Pass=ratio≤0.50." \
    2>&1 | tee "${RESULTS_DIR}/bench.log"

BENCH_EXIT=${PIPESTATUS[0]}
"${REPO_ROOT}/infra/scripts/teardown.sh"

VERDICT="$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/metrics.json')); print(d.get('verdict','?'))" 2>/dev/null || echo "?")"
echo ""
echo "Verdict: ${VERDICT}"
case "${VERDICT}" in
    PASS)         echo "✓ vLLM prefix caching passes." ;;
    INCONCLUSIVE) echo "⚠ Marginal speedup — vLLM may need explicit --enable-prefix-caching flag depending on version." ;;
    FAIL)         echo "✗ Prefix caching not effective. Add --enable-prefix-caching if not present, or compare with SGLang 1.2." ;;
esac

# Compare to SGLang 1.2 if available
SGLANG_PREFIX_RESULTS="$(ls -td "${REPO_ROOT}/results/phase1_1.2_sglang_prefix_"* 2>/dev/null | head -1 || echo "")"
if [[ -n "${SGLANG_PREFIX_RESULTS}" ]]; then
    echo ""
    echo "── Comparison vs SGLang (1.2) ──────────────────────────────────────────"
    python -m lib.reporter compare "${RESULTS_DIR}" "${SGLANG_PREFIX_RESULTS}" \
        --key prefix_reuse_speedup || true
fi

cat "${RESULTS_DIR}/summary.md"
exit "${BENCH_EXIT}"
