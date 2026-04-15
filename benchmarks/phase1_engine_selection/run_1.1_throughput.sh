#!/usr/bin/env bash
# Sub-test 1.1 — vLLM vs SGLang throughput comparison.
#
# Runs the same 10 code-generation tasks against both engines sequentially,
# then emits a side-by-side comparison. Run after Phase 0 confirms tool calling works.
#
# Pass criteria (from config/thresholds.yaml):
#   decode_tps  ≥ 150 t/s
#   ttft_ms     ≤ 500 ms
#
# Usage:
#   ./benchmarks/phase1_engine_selection/run_1.1_throughput.sh
#   SKIP_SGLANG=1 ./benchmarks/phase1_engine_selection/run_1.1_throughput.sh

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

MODEL="${MODEL:-QuantTrio/Qwen3.5-35B-A3B-AWQ}"
# 16384: after loading 22 GiB AWQ model, only ~0.57 GiB remains for KV cache on 32 GB GPU.
# Engine reports max viable context ~26400; 16384 is conservative and safe.
# For 30B model (15.74 GiB), use CTX_LEN=32768 — plenty of VRAM headroom.
CTX_LEN="${CTX_LEN:-16384}"
QUANT="${QUANT:-AWQ-INT4}"
CONCURRENCY="${CONCURRENCY:-4}"   # concurrent requests — stress test
SKIP_SGLANG="${SKIP_SGLANG:-0}"
# EXTRA_ENGINE_ARGS: override the vLLM flags.
# Default includes --enforce-eager (required for 35B: only 510 MiB free after model load).
# For 30B (11+ GiB free), override: EXTRA_ENGINE_ARGS="--tool-call-parser qwen3_coder"
EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:---tool-call-parser qwen3_coder --reasoning-parser qwen3 --enforce-eager}"

TASKS_DIR="${REPO_ROOT}/benchmarks/phase1_engine_selection/tasks/throughput"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

_run_engine() {
    local engine="$1" gpu="$2" port_var="$3" extra_args="$4" label="$5"
    local results_dir="${REPO_ROOT}/results/phase1_1.1_${label}_${TIMESTAMP}"
    mkdir -p "${results_dir}/raw"

    # All diagnostic output → stderr so only the final results_dir path goes to stdout.
    echo "" >&2
    echo "── ${label} ────────────────────────────────────────────────────────" >&2
    "${REPO_ROOT}/infra/scripts/deploy.sh" "${engine}" "${gpu}" "${MODEL}" \
        --ctx "${CTX_LEN}" ${extra_args} >&2

    local port="${!port_var}"
    python3 -m benchmarks.phase1_engine_selection.bench \
        --endpoint "http://localhost:${port}/v1" \
        --results-dir "${results_dir}" \
        --tasks "${TASKS_DIR}" \
        --mode throughput \
        --concurrency "${CONCURRENCY}" \
        --engine "${engine}" \
        --quantization "${QUANT}" \
        --gpu-label "RTX 5090" \
        --gpu-id "${GPU_0_ID}" \
        --ctx-len "${CTX_LEN}" \
        --extra-args "${extra_args}" \
        --thresholds "${REPO_ROOT}/config/thresholds.yaml" \
        --notes "Sub-test 1.1: throughput (concurrency=${CONCURRENCY})" \
        2>&1 | tee "${results_dir}/bench.log" >&2 || true

    "${REPO_ROOT}/infra/scripts/teardown.sh" >&2
    echo "${results_dir}"
}

echo "=========================================="
echo " Sub-test 1.1: vLLM vs SGLang throughput"
echo " Model      : ${MODEL}"
echo " Concurrency: ${CONCURRENCY}"
echo "=========================================="

VLLM_RESULTS="$(_run_engine vllm gpu0 PORT_VLLM_GPU0 \
    "${EXTRA_ENGINE_ARGS}" \
    "vllm")"

SGLANG_RESULTS=""
if [[ "${SKIP_SGLANG}" != "1" ]]; then
    # Phase 1.1 is a pure throughput test (no tool calls) — no tool-call-parser needed.
    # --quantization awq_marlin: QuantTrio checkpoint stores per-expert tensors
    # (experts.N.w2.weight); SGLang default loader expects fused experts.w2_weight
    # and hits KeyError. awq_marlin path handles per-expert format correctly.
    SGLANG_RESULTS="$(_run_engine sglang gpu0 PORT_SGLANG_GPU0 \
        "--quantization awq_marlin" \
        "sglang")"
fi

# ── Side-by-side comparison ────────────────────────────────────────────────────
echo ""
echo "── Comparison ──────────────────────────────────────────────────────────"
if [[ -n "${SGLANG_RESULTS}" ]]; then
    python3 -m lib.reporter compare "${VLLM_RESULTS}" "${SGLANG_RESULTS}" \
        --key decode_tps_mean
else
    echo "(SGLang skipped — vLLM-only run)"
    cat "${VLLM_RESULTS}/summary.md"
fi

echo ""
echo "Results:"
echo "  vLLM   : ${VLLM_RESULTS}"
[[ -n "${SGLANG_RESULTS}" ]] && echo "  SGLang : ${SGLANG_RESULTS}"
