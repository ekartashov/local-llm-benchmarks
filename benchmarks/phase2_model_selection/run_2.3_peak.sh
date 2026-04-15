#!/usr/bin/env bash
# Sub-test 2.3 — Peak mode: Qwen3-Coder-Next (GGUF) vs best daily driver.
#
# Qwen3-Coder-Next is 160 GB bf16 — loaded as GGUF Q4 via llama.cpp.
# Compares it against the Phase 2.1 winner on the same quality tasks.
# This determines whether the complexity of maintaining a large GGUF model
# is worth the quality uplift for "peak mode" requests.
#
# Note: Qwen3-Coder-Next decode speed will be ~30-50% lower than MoE AWQ.
#       Only use it for requests where quality matters most (no latency SLA).
#
# Usage:
#   ./benchmarks/phase2_model_selection/run_2.3_peak.sh
#   DAILY_DRIVER_MODEL=QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ \
#     ./benchmarks/phase2_model_selection/run_2.3_peak.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

GPU="gpu0"
CTX_LEN=32768
MODEL_FILE="${MODEL_FILE:-Qwen3-Coder-Next-Q4_K_M.gguf}"
# Phase 2.1 winner: 30B-AWQ at 251 t/s. 35B-AWQ needs --enforce-eager (22 t/s, not viable).
DAILY_DRIVER_MODEL="${DAILY_DRIVER_MODEL:-QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ}"
DAILY_DRIVER_QUANT="${DAILY_DRIVER_QUANT:-AWQ-INT4}"
DAILY_DRIVER_ARGS="${DAILY_DRIVER_ARGS:---tool-call-parser qwen3_coder}"

TASKS="${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/quality"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "=========================================================="
echo " Sub-test 2.3: Peak mode — Coder-Next vs daily driver"
echo "=========================================================="

# ── Daily driver run ───────────────────────────────────────────────────────────
DD_RESULTS="${REPO_ROOT}/results/phase2_2.3_daily_driver_${TIMESTAMP}"
mkdir -p "${DD_RESULTS}/raw"

echo ""
echo "── Daily driver: ${DAILY_DRIVER_MODEL} ──────────────────────────────────"
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm "${GPU}" "${DAILY_DRIVER_MODEL}" \
    --ctx "${CTX_LEN}" ${DAILY_DRIVER_ARGS}

python3 -m benchmarks.phase2_model_selection.bench \
    --endpoint "http://localhost:${PORT_VLLM_GPU0}/v1" \
    --model "${DAILY_DRIVER_MODEL}" \
    --results-dir "${DD_RESULTS}" \
    --tasks "${TASKS}" \
    --mode quality \
    --label "DailyDriver-${DAILY_DRIVER_MODEL##*/}" \
    --engine vllm \
    --quantization "${DAILY_DRIVER_QUANT}" \
    --gpu-label "RTX 5090" \
    --gpu-id "${GPU_0_ID}" \
    --ctx-len "${CTX_LEN}" \
    --extra-args "${DAILY_DRIVER_ARGS}" \
    --thresholds "${REPO_ROOT}/config/thresholds.yaml" \
    --max-tokens 1024 \
    --notes "Sub-test 2.3: daily driver baseline" \
    2>&1 | tee "${DD_RESULTS}/bench.log" || true

"${REPO_ROOT}/infra/scripts/teardown.sh"

# ── Coder-Next run ─────────────────────────────────────────────────────────────
GGUF_PATH="${MODEL_CACHE}/${MODEL_FILE}"
if [[ ! -f "${GGUF_PATH}" ]]; then
    echo "WARNING: Coder-Next GGUF not found at ${GGUF_PATH}" >&2
    echo "         Skipping peak-mode run. Download with:" >&2
    echo "         huggingface-cli download Qwen/Qwen3-Coder-Next --include '${MODEL_FILE}'" >&2
    echo ""
    echo "Daily driver results: ${DD_RESULTS}/human_review.md"
    exit 0
fi

CN_RESULTS="${REPO_ROOT}/results/phase2_2.3_coder_next_${TIMESTAMP}"
mkdir -p "${CN_RESULTS}/raw"

echo ""
echo "── Peak mode: Qwen3-Coder-Next (${MODEL_FILE}) ──────────────────────────"
export MODEL_FILE
"${REPO_ROOT}/infra/scripts/deploy.sh" llamacpp "${GPU}" "" \
    --ctx "${CTX_LEN}" \
    --n-gpu-layers 99 --flash-attn --jinja

python3 -m benchmarks.phase2_model_selection.bench \
    --endpoint "http://localhost:${PORT_LLAMACPP_GPU0}/v1" \
    --model "${MODEL_FILE}" \
    --results-dir "${CN_RESULTS}" \
    --tasks "${TASKS}" \
    --mode quality \
    --label "CoderNext-Q4" \
    --engine llamacpp \
    --quantization "GGUF-Q4_K_M" \
    --gpu-label "RTX 5090" \
    --gpu-id "${GPU_0_ID}" \
    --ctx-len "${CTX_LEN}" \
    --extra-args "--n-gpu-layers 99 --flash-attn --jinja" \
    --thresholds "${REPO_ROOT}/config/thresholds.yaml" \
    --max-tokens 1024 \
    --notes "Sub-test 2.3: peak mode (Coder-Next GGUF)" \
    2>&1 | tee "${CN_RESULTS}/bench.log" || true

"${REPO_ROOT}/infra/scripts/teardown.sh"

echo ""
echo "── Speed comparison ────────────────────────────────────────────────────"
python3 -m lib.reporter compare "${DD_RESULTS}" "${CN_RESULTS}" \
    --key decode_tps_mean || true

echo ""
echo "── Human review sheets ─────────────────────────────────────────────────"
echo "  Daily driver : ${DD_RESULTS}/human_review.md"
echo "  Coder-Next   : ${CN_RESULTS}/human_review.md"
echo ""
echo "If Coder-Next quality is not meaningfully better, skip peak mode."
