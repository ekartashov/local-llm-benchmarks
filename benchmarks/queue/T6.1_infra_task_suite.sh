#!/usr/bin/env bash
# T6.1_infra_task_suite.sh
# Quality evaluation of the Arclight Coder on infrastructure automation tasks.
# Uses the stable vLLM engine to restore production-level TPS (expected ~230+).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP=$(date +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/T6.1_infra_task_suite_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# --- Configuration ---
export BENCH_IMAGE="vllm/vllm-openai:latest"
export VLLM_V1_ENABLED=0
export VLLM_USE_V1=0
export VLLM_V1=0
export VLLM_USE_V1_ENGINE=0

MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
CTX="32768"
GPU_MEM_UTIL="0.85"

for TP_SIZE in 2 1; do
    if [ "$TP_SIZE" -eq 1 ]; then
        PLACEMENT="gpu0"
        GPU_UTIL="0.98"
    else
        PLACEMENT="tp2a"
        GPU_UTIL="0.90"
    fi

    RUN_DIR="${RESULTS_DIR}/tp${TP_SIZE}"
    mkdir -p "${RUN_DIR}"

    # --- Step 1: Deploy Coder ---
    echo "[T6.1] Deploying Arclight Coder (TP=${TP_SIZE}, Stable Engine)..."
    ./infra/scripts/deploy.sh vllm "${PLACEMENT}" "${MODEL}" \
        --gpu-mem-util "${GPU_UTIL}" \
        --ctx "${CTX}" \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --enforce-eager

    # --- Step 2: Run Infrastructure Tasks ---
    echo "[T6.1] Running Infra Task Suite (in01-in05) for TP=${TP_SIZE}..."
    python3 -m benchmarks.phase2_model_selection.bench \
        --mode quality \
        --tasks benchmarks/infra_tasks/tasks/ \
        --label "Qwen3.6-35B-A3B (T6.1 Infra Suite, TP=${TP_SIZE})" \
        --results-dir "${RUN_DIR}"

    # --- Step 3: Cleanup ---
    echo "[T6.1] Cleaning up TP=${TP_SIZE}..."
    podman stop "bench-vllm-${PLACEMENT}" || true
done

echo "[T6.1] Complete. Results in: ${RESULTS_DIR}"
