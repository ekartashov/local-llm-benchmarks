#!/usr/bin/env bash
# Sub-test 0.4 — Chat template verification.
# Deploys vLLM with Qwen3.5-35B-A3B-AWQ, runs targeted probes to detect known
# engine/template bugs, then tears down.
#
# Run this FIRST within Phase 0 (~30 min total including model load time).
# If all probes pass, proceed with run_0.1_vllm.sh.
#
# Usage:
#   ./benchmarks/phase0_tool_reliability/run_0.4_chat_template.sh
#
# Override model / engine via env:
#   MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ \
#     ./benchmarks/phase0_tool_reliability/run_0.4_chat_template.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ENGINE="${ENGINE:-vllm}"
GPU="${GPU:-gpu0}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-AWQ}"
CTX_LEN="${CTX_LEN:-32768}"   # smaller context — just probing, not production run

# vLLM flags critical for this model (see CLAUDE.md known bugs table)
# --reasoning-parser qwen3  : separates <think> tokens from tool-call tokens (PR #39055 workaround)
# --tool-call-parser qwen3_coder : correct parser for this model family
EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:---tool-call-parser qwen3_coder --reasoning-parser qwen3}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/phase0_0.4_chat_template_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "=================================================="
echo " Sub-test 0.4: Chat Template Verification"
echo " Engine  : ${ENGINE} / ${GPU}"
echo " Model   : ${MODEL}"
echo " Results : ${RESULTS_DIR}"
echo "=================================================="

PORT="${PORT_VLLM_GPU0}"
case "${ENGINE}-${GPU}" in
    vllm-gpu0)   PORT="${PORT_VLLM_GPU0}" ;;
    vllm-gpu1)   PORT="${PORT_VLLM_GPU1}" ;;
    sglang-gpu0) PORT="${PORT_SGLANG_GPU0}" ;;
    sglang-gpu1) PORT="${PORT_SGLANG_GPU1}" ;;
esac

# ── Deploy ─────────────────────────────────────────────────────────────────────
"${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${MODEL}" \
    --ctx "${CTX_LEN}" \
    ${EXTRA_ENGINE_ARGS}

ENDPOINT="http://localhost:${PORT}/v1"

# ── Run probes ─────────────────────────────────────────────────────────────────
python -m benchmarks.phase0_tool_reliability.verify_chat_template \
    --endpoint "${ENDPOINT}" \
    --results-dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/probe.log"

PROBE_EXIT=${PIPESTATUS[0]}

# ── Teardown ───────────────────────────────────────────────────────────────────
"${REPO_ROOT}/infra/scripts/teardown.sh"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
if [[ "${PROBE_EXIT}" -eq 0 ]]; then
    echo "✓ Sub-test 0.4 PASSED — safe to run: just phase0 (or run_0.1_vllm.sh)"
elif [[ "${PROBE_EXIT}" -eq 2 ]]; then
    echo "✗ Sub-test 0.4 ABORTED — endpoint unreachable (container failed to start?)"
else
    echo "✗ Sub-test 0.4 FAILED — fix template issues before running Phase 0.1"
    echo "  See remediation advice above and in ${RESULTS_DIR}/probe_results.json"
fi

exit "${PROBE_EXIT}"
