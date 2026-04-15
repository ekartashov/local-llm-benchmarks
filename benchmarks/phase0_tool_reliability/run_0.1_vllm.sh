#!/usr/bin/env bash
# Sub-test 0.1 — Qwen3.5-35B-A3B-AWQ on vLLM (GPU 0).
#
# Primary candidate: this is the model + engine combo we want to use for all
# coding agent workflows. Phase 0 PASS here means we proceed to Phase 1.
#
# Engine flags (per CLAUDE.md known bugs table as of April 2026):
#   --tool-call-parser qwen3_coder   : correct parser for Qwen3.5 MoE family
#   --reasoning-parser qwen3         : separates <think> from tool tokens (PR #39055 workaround)
#   --enable-auto-tool-choice        : required for tool use
#
# If this run scores FAIL → run sub-test 0.3 (fallback model).
# If this run scores INCONCLUSIVE → investigate raw/ results, rerun with --concurrency 1.
#
# Usage:
#   ./benchmarks/phase0_tool_reliability/run_0.1_vllm.sh

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

ENGINE="vllm"
GPU="gpu0"
MODEL="QuantTrio/Qwen3.5-35B-A3B-AWQ"
CTX_LEN=16384   # Engine reports max viable at this GPU util is ~26400; 16384 ensures KV fits
QUANT="AWQ-INT4"

# Critical flags — do not remove without checking bug status in CLAUDE.md
# --enforce-eager: skip CUDA graph capture (graphs for batch 1–512 exhaust remaining
#   VRAM after 22 GiB AWQ model load on 32 GB GPU; trade-off is ~10% slower decode).
EXTRA_ENGINE_ARGS="--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enforce-eager"

PHASE="phase0_tool_reliability"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/results/${PHASE}_0.1_vllm_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

echo "=================================================="
echo " Sub-test 0.1: Qwen3.5-35B-A3B-AWQ on vLLM"
echo " GPU     : ${GPU} (${GPU_0_ID})"
echo " CTX     : ${CTX_LEN}"
echo " Results : ${RESULTS_DIR}"
echo "=================================================="

# ── Deploy ─────────────────────────────────────────────────────────────────────
"${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${MODEL}" \
    --ctx "${CTX_LEN}" \
    ${EXTRA_ENGINE_ARGS}

ENDPOINT="http://localhost:${PORT_VLLM_GPU0}/v1"

# ── Bench ──────────────────────────────────────────────────────────────────────
python3 -m benchmarks.phase0_tool_reliability.bench \
    --endpoint "${ENDPOINT}" \
    --model "${MODEL}" \
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
    --notes "Sub-test 0.1: primary candidate. If FAIL, run sub-test 0.3." \
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
        echo "✓ Sub-test 0.1 PASSED — proceed to Phase 1 (engine selection)."
        echo "  Optionally run 0.2 to validate SGLang as well."
        ;;
    INCONCLUSIVE)
        echo "⚠ Sub-test 0.1 INCONCLUSIVE — inspect ${RESULTS_DIR}/raw/ for engine errors."
        echo "  Check if --reasoning-parser qwen3 is needed (vLLM PR #39055)."
        ;;
    FAIL)
        echo "✗ Sub-test 0.1 FAILED — run sub-test 0.3 (fallback model):"
        echo "  ./benchmarks/phase0_tool_reliability/run_0.3_fallback.sh"
        ;;
    *)
        echo "Unknown verdict — check ${RESULTS_DIR}/metrics.json"
        ;;
esac

cat "${RESULTS_DIR}/summary.md"
exit "${BENCH_EXIT}"
