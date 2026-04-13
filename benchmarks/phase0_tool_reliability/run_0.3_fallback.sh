#!/usr/bin/env bash
# Sub-test 0.3 — Qwen3-Coder-Next on vLLM (fallback if 0.1 fails).
#
# Only run this if sub-test 0.1 (Qwen3.5-35B-A3B-AWQ) FAILS.
# Qwen3-Coder-Next is 160 GB bf16 — too large for a single GPU.
# On this hardware we load the GGUF Q4 variant via llama.cpp instead.
#
# IMPORTANT: llama.cpp requires the GGUF file to be present at:
#   ${MODEL_CACHE}/Qwen3-Coder-Next-Q4_K_M.gguf
# Download first:
#   huggingface-cli download Qwen/Qwen3-Coder-Next \
#     --include "*.gguf" --cache-dir ${MODEL_CACHE}
# Then set MODEL_FILE to the filename.
#
# Do NOT enable speculative decoding here — llama.cpp PR #20075 crashes on MoE.
# Do NOT use CPU layers — llama.cpp issue #19480 (5× slowdown on i9-14900K).
#
# Usage:
#   ./benchmarks/phase0_tool_reliability/run_0.3_fallback.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ENGINE="llamacpp"
GPU="gpu0"
# llamacpp uses MODEL_FILE (filename inside MODEL_CACHE), not the HF repo
MODEL_FILE="${MODEL_FILE:-Qwen3-Coder-Next-Q4_K_M.gguf}"
CTX_LEN=32768
QUANT="GGUF-Q4_K_M"

# All layers on GPU — avoids the 5× CPU slowdown (llama.cpp issue #19480)
# No spec decode — crashes on MoE (llama.cpp PR #20075)
EXTRA_ENGINE_ARGS="--n-gpu-layers 99 --flash-attn --jinja"

PHASE="phase0_tool_reliability"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/${PHASE}_0.3_fallback_llamacpp_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

echo "=================================================="
echo " Sub-test 0.3: Qwen3-Coder-Next on llama.cpp"
echo " (FALLBACK — run only if sub-test 0.1 failed)"
echo " GPU       : ${GPU} (${GPU_0_ID})"
echo " Model file: ${MODEL_FILE}"
echo " CTX       : ${CTX_LEN}"
echo " Results   : ${RESULTS_DIR}"
echo "=================================================="

# Verify GGUF file exists before deploying
GGUF_PATH="${MODEL_CACHE}/${MODEL_FILE}"
if [[ ! -f "${GGUF_PATH}" ]]; then
    echo "ERROR: GGUF file not found at ${GGUF_PATH}" >&2
    echo "Download it first:" >&2
    echo "  huggingface-cli download Qwen/Qwen3-Coder-Next --include '*.gguf' --cache-dir ${MODEL_CACHE}" >&2
    exit 1
fi

# ── Deploy ─────────────────────────────────────────────────────────────────────
export MODEL_FILE
"${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "" \
    --ctx "${CTX_LEN}" \
    ${EXTRA_ENGINE_ARGS}

ENDPOINT="http://localhost:${PORT_LLAMACPP_GPU0}/v1"

# ── Bench ──────────────────────────────────────────────────────────────────────
python -m benchmarks.phase0_tool_reliability.bench \
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
    --notes "Sub-test 0.3: fallback model (Qwen3-Coder-Next GGUF Q4). Run only when 0.1 fails." \
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
        echo "✓ Sub-test 0.3 PASSED — Qwen3-Coder-Next (GGUF) is the Phase 0 winner."
        echo "  Update Phase 1 scripts to use this model + llamacpp."
        echo "  Note: decode speed will be lower than MoE AWQ — measure in Phase 1."
        ;;
    INCONCLUSIVE | FAIL)
        echo "✗ Sub-test 0.3 ${VERDICT} — both primary and fallback models failed."
        echo "  Investigate raw/ outputs. Possible causes:"
        echo "   - GGUF chat template not applying tool call format (use --jinja)"
        echo "   - Model responding in Chinese (wrong system prompt language)"
        echo "   - Context overflow (try --ctx-len 16384)"
        ;;
esac

cat "${RESULTS_DIR}/summary.md"
exit "${BENCH_EXIT}"
