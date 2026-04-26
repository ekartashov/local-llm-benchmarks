#!/bin/bash
# T_KV2: Arclight Hot-Swap Benchmark (Host-Native with cuda-checkpoint)
# ---------------------------------------------------------------------
# Measures the transition from "Cold Start" to "Hot Swapped" state
# using host-native CRIU and NVIDIA cuda-checkpoint utility.
# ---------------------------------------------------------------------

set -e

# Configuration
MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
TP=2
PORT=30000
CHECKPOINT_DIR="/srv/ai/checkpoints/host-kv2-swap"
RESULTS_BASE="/srv/ai/projects/local-llm-benchmarks/results"
TIMESTAMP=$(date +%Y%m%dT%H%M%SZ)
RUN_ID="T_KV2_host_hot_restart_${TIMESTAMP}"
RUN_DIR="${RESULTS_BASE}/${RUN_ID}"

mkdir -p "${RUN_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# Environment
export HF_HOME="/srv/ai/models"
export UVLOOP_NO_IO_URING=1
export VLLM_USE_V1=0
export VLLM_V1_ENABLED=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

# Utility paths
CUDA_CHECKPOINT="/usr/local/bin/cuda-checkpoint"
CRIU="/usr/sbin/criu"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " T_KV2: HOST-NATIVE HOT RESTART BENCHMARK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[T_KV2] Model: ${MODEL}"
echo "[T_KV2] Run ID: ${RUN_ID}"
echo "[T_KV2] Result Dir: ${RUN_DIR}"

# 1. Start vLLM (Cold Start)
echo "[T_KV2] Starting vLLM (Cold Start)..."
START_COLD=$(date +%s%N)

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 65536 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 4 \
    --enable-sleep-mode \
    --port "${PORT}" 2>&1 | tee "${RUN_DIR}/vllm_cold.log" &

VLLM_PID=$!

# Wait for healthy
echo "[T_KV2] Waiting for engine health (PID: ${VLLM_PID})..."
while true; do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "[T_KV2] Engine is healthy."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[T_KV2] ERROR: vLLM process (PID: ${VLLM_PID}) died early."
        cat "${RUN_DIR}/vllm_cold.log"
        exit 1
    fi
    sleep 5
done

END_COLD=$(date +%s%N)
COLD_START_SEC=$(echo "scale=3; ($END_COLD - $START_COLD) / 1000000000" | bc)
echo "[T_KV2] Cold Start Complete: ${COLD_START_SEC}s"

# 2. Checkpoint Phase
echo "[T_KV2] Suspending GPU state..."
# Get all PIDs in the process tree (excluding threads)
PIDS=$(ps -ef | grep $VLLM_PID | grep -v grep | awk '{print $2}')

for p in $PIDS; do
    echo "[T_KV2] Toggling GPU for PID: $p"
    sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || echo "Warning: PID $p not a CUDA process"
done

echo "[T_KV2] Performing CRIU Dump..."
START_DUMP=$(date +%s%N)
if ! sudo "${CRIU}" dump \
    --tree "${VLLM_PID}" \
    --images-dir "${CHECKPOINT_DIR}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --leave-stopped; then
    echo "[T_KV2] ERROR: CRIU Dump failed. Running diagnostics..."
    for p in $PIDS; do
        echo "[T_KV2] Checking FDs for PID $p:"
        sudo ls -l "/proc/$p/fd" | grep -i "io_uring" || true
    done
    exit 1
fi

END_DUMP=$(date +%s%N)
DUMP_SEC=$(echo "scale=3; ($END_DUMP - $START_DUMP) / 1000000000" | bc)
echo "[T_KV2] Checkpoint created in ${DUMP_SEC}s. Process is now frozen."

# 3. Hot Restart (Restore) Phase
echo "[T_KV2] Measuring Hot Restart (Restore)..."
START_HOT=$(date +%s%N)

# Restore the process tree
sudo "${CRIU}" restore \
    --images-dir "${CHECKPOINT_DIR}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --restore-detached

# Resume GPU state
for p in $PIDS; do
    sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true
done

# Wait for health again
while true; do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "[T_KV2] Engine is healthy after restore."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[T_KV2] ERROR: vLLM process (PID: ${VLLM_PID}) died during hot restart."
        exit 1
    fi
    sleep 1
done

END_HOT=$(date +%s%N)
HOT_RESTART_SEC=$(echo "scale=3; ($END_HOT - $START_HOT) / 1000000000" | bc)
echo "[T_KV2] Hot Restart Complete: ${HOT_RESTART_SEC}s"

# 4. Save Metrics
cat <<EOF > "${RUN_DIR}/metrics.json"
{
  "test_id": "T_KV2",
  "model": "${MODEL}",
  "hardware": "Dual 5090 (Blackwell)",
  "metrics": {
    "cold_start_time_sec": ${COLD_START_SEC},
    "checkpoint_dump_time_sec": ${DUMP_SEC},
    "hot_restart_time_sec": ${HOT_RESTART_SEC}
  },
  "config": {
    "tp": ${TP},
    "io_uring": false,
    "v1_engine": false,
    "cuda_checkpoint": true
  }
}
EOF

cat <<EOF > "${RUN_DIR}/summary.md"
# T_KV2: Arclight Hot-Swap Benchmark Results

| Metric | Duration (s) |
| :--- | :--- |
| **Cold Start** | ${COLD_START_SEC}s |
| **Checkpoint Dump** | ${DUMP_SEC}s |
| **Hot Restart (Swap)** | **${HOT_RESTART_SEC}s** |

## Configuration
- **Model:** \`${MODEL}\`
- **Engine:** vLLM 0.19.1 (V1 disabled)
- **Method:** Host-native CRIU + NVIDIA cuda-checkpoint
- **Hardware:** Dual RTX 5090 (TP=2)

## Analysis
$( ( ( $(echo "$HOT_RESTART_SEC < 10.0" | bc -l) == 1 ) ) && echo "✅ SUCCESS: Sub-10s hot swap achieved!" || echo "❌ FAILURE: Still above 10s target.")
EOF

echo "[T_KV2] Benchmark complete. Results saved to ${RUN_DIR}"

# Cleanup (kill the restored process)
kill $VLLM_PID || true
