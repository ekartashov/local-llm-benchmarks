#!/usr/bin/env bash
# T2.4h_qwen36_27b_awq_tp2_enforce_eager.sh
#
# Question: Does --enforce-eager fix the GDN state-split logic error at TP=2?
#
# Hypothesis: The semantic error at TP=2 is caused by CUDA graph capture 
# failing to correctly synchronize the recurrent state updates of DeltaNet. 
# Enforce-eager bypasses graphs and uses the standard Torch path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP=$(date +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/T2.4h_thinker_tp2_eager_fix_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# --- Configuration ---
MODEL="QuantTrio/Qwen3.6-27B-AWQ"
CTX="32768"
GPU_MEM_UTIL="0.90"

# --- Step 1: Deploy Thinker (TP=2, Enforce Eager) ---
echo "[T2.4h] Deploying Arclight Thinker (TP=2, --enforce-eager)..."
./infra/scripts/deploy.sh vllm tp2a "${MODEL}" \
    --gpu-mem-util "${GPU_MEM_UTIL}" \
    --ctx "${CTX}" \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --enforce-eager \
    --kv-cache-dtype fp8

# --- Step 2: Run th02 Correctness Repro (3x) ---
echo "[T2.4h] Running th02 (Algorithm Logic) reproducibility test..."
python3 -m benchmarks.phase2_model_selection.bench \
    --mode quality \
    --tasks benchmarks/infra_tasks/tasks/th02.json \
    --label "Qwen3.6-27B (T2.4h Eager Fix, TP=2)" \
    --results-dir "${RESULTS_DIR}" \
    --repetitions 3

# --- Step 3: Cleanup ---
echo "[T2.4h] Cleaning up..."
podman stop "bench-vllm-tp2a" || true

echo "[T2.4h] Complete. Results in: ${RESULTS_DIR}"
echo "[T2.4h] Check if th02 scores 5/5. If yes, TP=2 is restored."
