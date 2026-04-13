#!/usr/bin/env bash
# Sub-test 2.4 — Devstral-Small tool-call reliability.
#
# Runs the full Phase 0 tool-call task suite against Devstral-Small-2505.
# Pass criterion is identical to Phase 0: ≥95% tool_call_success_rate.
#
# Devstral is a Mistral-based model (not Qwen) — do NOT use qwen3_coder parser.
# vLLM tool-call-parser: "mistral" (auto-detected if omitted).
#
# Usage:
#   ./benchmarks/phase2_model_selection/run_2.4_devstral.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ENGINE="vllm"
GPU="gpu0"
MODEL="mistralai/Devstral-Small-2505"
CTX_LEN=32768
QUANT="bf16"
EXTRA_ENGINE_ARGS="--tool-call-parser mistral"

TASKS="${REPO_ROOT}/benchmarks/phase0_tool_reliability/tasks"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/phase2_2.4_devstral_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

echo "=========================================================="
echo " Sub-test 2.4: Devstral tool-call reliability"
echo " Model  : ${MODEL}"
echo " Tasks  : Phase 0 suite (30 tasks)"
echo "=========================================================="

"${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${MODEL}" \
    --ctx "${CTX_LEN}" ${EXTRA_ENGINE_ARGS}

python -m benchmarks.phase2_model_selection.bench \
    --endpoint "http://localhost:${PORT_VLLM_GPU0}/v1" \
    --results-dir "${RESULTS_DIR}" \
    --tasks "${TASKS}" \
    --mode tool-reliability \
    --label "Devstral-Small-2505" \
    --engine "${ENGINE}" \
    --quantization "${QUANT}" \
    --gpu-label "RTX 5090" \
    --gpu-id "${GPU_0_ID}" \
    --ctx-len "${CTX_LEN}" \
    --extra-args "${EXTRA_ENGINE_ARGS}" \
    --thresholds "${REPO_ROOT}/config/thresholds.yaml" \
    --notes "Sub-test 2.4: Devstral tool-call reliability. Pass=same as Phase 0 (≥95%)." \
    2>&1 | tee "${RESULTS_DIR}/bench.log"

BENCH_EXIT=${PIPESTATUS[0]}
"${REPO_ROOT}/infra/scripts/teardown.sh"

VERDICT="$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/metrics.json')); print(d.get('verdict','?'))" 2>/dev/null || echo "?")"
echo ""
echo "Verdict: ${VERDICT}"
case "${VERDICT}" in
    PASS)         echo "✓ Devstral passes tool-call reliability — viable for Phase 2 quality comparison." ;;
    INCONCLUSIVE) echo "⚠ Devstral marginal tool-call reliability — investigate format_error/dropped counts." ;;
    FAIL)         echo "✗ Devstral fails tool-call reliability — eliminate from consideration." ;;
esac

cat "${RESULTS_DIR}/summary.md"
exit "${BENCH_EXIT}"
