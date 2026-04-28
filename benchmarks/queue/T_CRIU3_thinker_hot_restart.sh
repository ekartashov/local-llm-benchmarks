#!/usr/bin/env bash
# benchmarks/queue/T_CRIU3_thinker_hot_restart.sh
#
# T_CRIU3 Phase 1 — Thinker TP=1 CRIU Hot Restart (Host-Native)
#
# Objective: Confirm that host-native CRIU + cuda-checkpoint works for the thinker.
# This avoids Podman CDI/namespace conflicts encountered in previous attempts.
#
# Procedure matches the successful T_CRIU2 (Convergence) and T_KV2 (Coder) paths.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# Tool Paths
CUDA_CHECKPOINT="/usr/local/bin/cuda-checkpoint"
CRIU="/usr/sbin/criu"

# Configuration
MODEL="QuantTrio/Qwen3.6-27B-AWQ"
PORT=30001
CHECKPOINT_DIR="/srv/ai/checkpoints/thinker-tp1-host"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/T_CRIU3_thinker_hot_restart_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}"
rm -rf "${CHECKPOINT_DIR}" && mkdir -p "${CHECKPOINT_DIR}"

PROMPT="List the three laws of thermodynamics in one sentence each."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# shellcheck disable=SC2027
echo " T_CRIU3: THINKER HOST-NATIVE HOT RESTART"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[T_CRIU3] Model: ${MODEL}"
echo "[T_CRIU3] Result Dir: ${RESULTS_DIR}"

# --- Step 0: Cleanup ---
echo "[T_CRIU3] Cleaning up existing thinker (port ${PORT})..."
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker\|gpu1" | head -1 || true)
if [ -n "${THINKER_CONTAINER}" ]; then
    podman stop "${THINKER_CONTAINER}" >/dev/null 2>&1 && podman rm "${THINKER_CONTAINER}" >/dev/null 2>&1 || true
fi
# Force kill any host process on this port
sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
sleep 2

# --- Step 1: Start vLLM on Host (Python Sanitized) ---
echo "[T_CRIU3] Starting vLLM on Host (Python Sanitized)..."
export UV_USE_IO_URING=0
export VLLM_V1_ENABLED=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

# Use Python to launch the process with a clean session and closed FDs
# This prevents the 'session ID' and 'dirty FD' errors in CRIU
python3 -c "
import subprocess, os
log = open('${RESULTS_DIR}/vllm_host.log', 'w')
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '${GPU_1_ID}'
subprocess.Popen(
    ['python3', '-m', 'vllm.entrypoints.openai.api_server',
     '--model', '${MODEL}',
     '--port', '${PORT}',
     '--gpu-memory-utilization', '0.90',
     '--max-model-len', '32768',
     '--kv-cache-dtype', 'fp8',
     '--max-num-seqs', '1',
     '--enable-chunked-prefill',
     '--tool-call-parser', 'qwen3_coder',
     '--reasoning-parser', 'qwen3',
     '--enable-auto-tool-choice'],
    stdout=log, stderr=log, stdin=subprocess.DEVNULL,
    env=env, close_fds=True, start_new_session=True
)
"
sleep 5
MAIN_PID=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" | head -1)
if [ -z "${MAIN_PID}" ]; then
    echo "[T_CRIU3] ERROR: vLLM failed to start. Check ${RESULTS_DIR}/vllm_host.log"
    exit 1
fi
echo "[T_CRIU3] vLLM started with Host PID: ${MAIN_PID}"

# Wait for healthy
echo "[T_CRIU3] Waiting for health check..."
READY=0
for i in $(seq 1 300); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[T_CRIU3] HEALTH OK"
        READY=1
        break
    fi
    if ! kill -0 "${MAIN_PID}" 2>/dev/null; then
        echo "[T_CRIU3] ERROR: vLLM process died."
        exit 1
    fi
    sleep 1
done

if [ "${READY}" -eq 0 ]; then
    echo "[T_CRIU3] ERROR: Health check timeout."
    exit 1
fi

# --- Step 2: Pre-checkpoint baseline inference ---
echo "[T_CRIU3] Recording baseline inference..."
START_MS=$(date +%s%3N)
PRE_RESPONSE=$(curl -sf "http://localhost:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
END_MS=$(date +%s%3N)
PRE_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "[T_CRIU3] Pre-checkpoint TTFT: ${PRE_TTFT_S}s"
echo "${PRE_RESPONSE}" > "${RESULTS_DIR}/pre_checkpoint_response.json"
echo "pre_ttft_s=${PRE_TTFT_S}" > "${RESULTS_DIR}/timings.txt"

# --- Step 3: Checkpoint ---
echo "[T_CRIU3] --- Checkpointing ---"
echo "[T_CRIU3] Suspending GPU state for all vLLM pids..."
PIDS=$(ps -ef | grep "${MAIN_PID}" | grep -v grep | awk '{print $2}')
for p in ${PIDS}; do
    echo "[T_CRIU3] Toggling GPU for PID: $p"
    sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || echo "Warning: PID $p not a CUDA process"
done

echo "[T_CRIU3] Performing CRIU Dump (logs -> results/criu_dump.log)..."
CKPT_START_MS=$(date +%s%3N)
if ! sudo "${CRIU}" dump \
    --tree "${MAIN_PID}" \
    --images-dir "${CHECKPOINT_DIR}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --log-file "${RESULTS_DIR}/criu_dump.log" \
    --verbosity=3; then
    echo "[T_CRIU3] ERROR: CRIU Dump failed. See ${RESULTS_DIR}/criu_dump.log"
    for p in ${PIDS}; do sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true; done
    exit 1
fi
CKPT_END_MS=$(date +%s%3N)
CKPT_ELAPSED_S=$(python3 -c "print(round(($CKPT_END_MS - $CKPT_START_MS) / 1000.0, 2))")
CKPT_SIZE_GB=$(sudo du -sh "${CHECKPOINT_DIR}" 2>/dev/null | awk '{print $1}')

echo "[T_CRIU3] Checkpoint time: ${CKPT_ELAPSED_S}s"
echo "[T_CRIU3] Checkpoint size: ${CKPT_SIZE_GB}"
echo "checkpoint_elapsed_s=${CKPT_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"
echo "checkpoint_size_gb=${CKPT_SIZE_GB}" >> "${RESULTS_DIR}/timings.txt"

# Wait for PIDs to clear
echo "[T_CRIU3] Waiting for PIDs to clear..."
sleep 2
for p in ${PIDS}; do
    if kill -0 "$p" 2>/dev/null; then
        sudo kill -9 "$p" || true
    fi
done

# --- Step 4: Restore ---
echo "[T_CRIU3] --- Restoring ---"
RESTORE_START_MS=$(date +%s%3N)
echo "[T_CRIU3] Performing CRIU Restore (logs -> results/criu_restore.log)..."
if ! sudo "${CRIU}" restore \
    --images-dir "${CHECKPOINT_DIR}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --restore-detached \
    --log-file "${RESULTS_DIR}/criu_restore.log" \
    --verbosity=3; then
    echo "[T_CRIU3] ERROR: CRIU Restore failed. See ${RESULTS_DIR}/criu_restore.log"
    exit 1
fi

echo "[T_CRIU3] Resuming GPU state..."
for p in ${PIDS}; do
    sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true
done

# Wait for healthy again
RESTORE_READY=0
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[T_CRIU3] HEALTH OK after restore"
        RESTORE_READY=1
        break
    fi
    sleep 1
done

if [ "${RESTORE_READY}" -eq 0 ]; then
    echo "[T_CRIU3] ERROR: Health check timeout after restore."
    exit 1
fi

RESTORE_END_MS=$(date +%s%3N)
RESTORE_ELAPSED_S=$(python3 -c "print(round(($RESTORE_END_MS - $RESTORE_START_MS) / 1000.0, 2))")
echo "[T_CRIU3] Restore time: ${RESTORE_ELAPSED_S}s"
echo "restore_elapsed_s=${RESTORE_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

# --- Step 5: Post-restore inference ---
echo "[T_CRIU3] Recording post-restore inference..."
START_MS=$(date +%s%3N)
POST_RESPONSE=$(curl -sf "http://localhost:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
END_MS=$(date +%s%3N)
POST_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "[T_CRIU3] Post-restore TTFT: ${POST_TTFT_S}s"
echo "${POST_RESPONSE}" > "${RESULTS_DIR}/post_restore_response.json"
echo "post_restore_ttft_s=${POST_TTFT_S}" >> "${RESULTS_DIR}/timings.txt"

# Check text match
PRE_TEXT=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/pre_checkpoint_response.json'))['choices'][0]['text'])")
POST_TEXT=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/post_restore_response.json'))['choices'][0]['text'])")
MATCH="differs"
if [[ "${PRE_TEXT}" == "${POST_TEXT}" ]]; then
    MATCH="identical"
fi
echo "[T_CRIU3] TEXT: ${MATCH}"
echo "text_match=${MATCH}" >> "${RESULTS_DIR}/timings.txt"

# --- Step 6: Summary ---
cat > "${RESULTS_DIR}/summary.md" <<EOMD
# T_CRIU3 Phase 1 — Thinker TP=1 CRIU Hot Restart (Host-Native) — ${TIMESTAMP}

## Result
RESTORE_OK

| Metric | Value |
|--------|-------|
| Checkpoint time | ${CKPT_ELAPSED_S}s |
| Checkpoint size | ${CKPT_SIZE_GB} |
| Restore time | ${RESTORE_ELAPSED_S}s |
| Pre-checkpoint TTFT | ${PRE_TTFT_S}s |
| Post-restore TTFT | ${POST_TTFT_S}s |
| Text match | ${MATCH} |

## Status
RESTORE_OK
EOMD

# Final Cleanup: Force kill all processes in the vLLM tree to free port 30001
echo "[T_CRIU3] Final cleanup of host processes..."
sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
sleep 2

# Redeploy production container
echo "[T_CRIU3] Redeploying production container..."
VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 1 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
