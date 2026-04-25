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
# Force stable vLLM image to avoid V1 engine bugs/slowdown
export BENCH_IMAGE="vllm/vllm-openai:latest"

MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
PLACEMENT="tp2a"
GPU_MEM_UTIL="0.85"
CTX="32768"

# --- Step 1: Deploy Coder ---
echo "[T6.1] Deploying Arclight Coder (Stable Engine)..."
./infra/scripts/deploy.sh vllm "${PLACEMENT}" "${MODEL}" \
    --gpu-mem-util "${GPU_MEM_UTIL}" \
    --ctx "${CTX}" \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice

# --- Step 2: Run Infrastructure Tasks ---
echo "[T6.1] Running Infra Task Suite (in01-in05)..."
python3 -m benchmarks.phase2_model_selection.bench \
    --mode quality \
    --tasks benchmarks/infra_tasks/tasks/ \
    --label "Qwen3.6-35B-A3B (T6.1 Infra Suite)" \
    --results-dir "${RESULTS_DIR}"

# --- Step 3: Cleanup ---
echo "[T6.1] Cleaning up..."
podman stop "bench-vllm-${PLACEMENT}" || true

echo "[T6.1] Complete. Results in: ${RESULTS_DIR}"
