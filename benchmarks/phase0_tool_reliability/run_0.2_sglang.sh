#!/usr/bin/env bash
# Sub-test 0.2 — Qwen3.5-35B-A3B-AWQ on SGLang (GPU 0).
#
# Runs the same 30 tasks as 0.1 but on SGLang. Used to:
#   (a) validate SGLang as an alternative engine for Phase 1
#   (b) determine if any tool-call issues seen in 0.1 are engine-specific
#
# If this FAILS → eliminates SGLang for this model in Phase 1.
# SGLang uses --tool-call-parser qwen3 (not qwen3_coder) — different parser name.
# SGLang context is capped at 32768 for safety; raise if testing longer contexts.
#
# Usage:
#   ./benchmarks/phase0_tool_reliability/run_0.2_sglang.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# Activate project venv if present (installs openai, httpx, etc.)
VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/python3" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
fi

ENGINE="sglang"
GPU="gpu0"
MODEL="QuantTrio/Qwen3.5-35B-A3B-AWQ"
CTX_LEN=32768   # conservative; SGLang's MoE prefix cache benefits shorter contexts anyway
QUANT="AWQ-INT4"

# SGLang uses a different parser name than vLLM
EXTRA_ENGINE_ARGS="--tool-call-parser qwen3"

PHASE="phase0_tool_reliability"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/${PHASE}_0.2_sglang_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

echo "=================================================="
echo " Sub-test 0.2: Qwen3.5-35B-A3B-AWQ on SGLang"
echo " GPU     : ${GPU} (${GPU_0_ID})"
echo " CTX     : ${CTX_LEN}"
echo " Results : ${RESULTS_DIR}"
echo "=================================================="

# ── Deploy ─────────────────────────────────────────────────────────────────────
"${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${MODEL}" \
    --ctx "${CTX_LEN}" \
    ${EXTRA_ENGINE_ARGS}

ENDPOINT="http://localhost:${PORT_SGLANG_GPU0}/v1"

# ── Bench ──────────────────────────────────────────────────────────────────────
python3 -m benchmarks.phase0_tool_reliability.bench \
    --endpoint "${ENDPOINT}" \
    --results-dir "${RESULTS_DIR}" \
    --tasks "${REPO_ROOT}/benchmarks/phase0_tool_reliability/tasks/" \
    --engine "${ENGINE}" \
    --quantization "${QUANT}" \
    --gpu-label "RTX 5090" \
    --gpu-id "${GPU_0_ID}" \
    --ctx-len "${CTX_LEN}" \
    --extra-args "${EXTRA_ENGINE_ARGS}" \
    --concurrency 1 \
    --thresholds "${REPO_ROOT}/config/thresholds.yaml" \
    --notes "Sub-test 0.2: SGLang validation. If FAIL, eliminate SGLang for this model in Phase 1." \
    2>&1 | tee "${RESULTS_DIR}/bench.log"

BENCH_EXIT=${PIPESTATUS[0]}

# ── Teardown ───────────────────────────────────────────────────────────────────
"${REPO_ROOT}/infra/scripts/teardown.sh"

# ── Outcome guidance ───────────────────────────────────────────────────────────
echo ""
VERDICT="$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/metrics.json')); print(d.get('verdict','?'))" 2>/dev/null || echo "?")"
echo "Verdict: ${VERDICT}"
case "${VERDICT}" in
    PASS)
        echo "✓ Sub-test 0.2 PASSED — SGLang is viable for Phase 1 comparison."
        echo "  Run Phase 1.1 (vLLM vs SGLang throughput) to decide the winner."
        ;;
    INCONCLUSIVE | FAIL)
        echo "✗ Sub-test 0.2 ${VERDICT} — SGLang eliminated for Qwen3.5-35B-A3B in Phase 1."
        echo "  Phase 1 will run vLLM only."
        ;;
esac

cat "${RESULTS_DIR}/summary.md"
exit "${BENCH_EXIT}"
