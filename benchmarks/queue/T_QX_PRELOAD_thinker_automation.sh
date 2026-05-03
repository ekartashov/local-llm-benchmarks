#!/usr/bin/env bash
# benchmarks/queue/T_QX_PRELOAD_thinker_automation.sh
#
# T_QX_PRELOAD — Thinker Checkpoint Automation and NVMe Pre-warm Benchmark
#
# Objective:
# 1. Automate the creation of a standardized Thinker checkpoint.
# 2. Measure CRIU restore time improvement from OS page-cache pre-warming (posix_fadvise).
#
# This script combines BENCH_17 (Thinker part) and BENCH_18.

set -euo pipefail

# ── Argument Parsing ─────────────────────────────────────────────────────────
SKIP_INIT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-init) SKIP_INIT=1; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# --- Trap for Cleanup on Exit ---
cleanup() {
    echo ""
    echo "[T_QX_PRELOAD] Interrupted. Cleaning up..."
    [ -n "${LOGS_PID:-}" ] && kill "${LOGS_PID}" 2>/dev/null || true
    # Kill any vLLM/VLLM processes on our port or related to our user
    sudo fuser -k 30001/tcp >/dev/null 2>&1 || true
    pkill -u "$USER" -9 -f "vllm|VLLM|resource_tracker|vllm_criu_wrapper" || true
    sudo rm -f /dev/shm/sem.* /dev/shm/link_remap.* >/dev/null 2>&1 || true
    # Kill background tail if it exists
    [ -n "${TAIL_PID:-}" ] && kill "${TAIL_PID}" 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# Tool Paths
CUDA_CHECKPOINT="/usr/local/bin/cuda-checkpoint"
CRIU="/usr/sbin/criu"
PRELOAD_SCRIPT="${REPO_ROOT}/infra/scripts/preload-checkpoint.sh"

# Configuration
MODEL="QuantTrio/Qwen3.6-27B-AWQ"
PORT=30001
CHECKPOINT_DIR="/srv/ai/checkpoints/thinker"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/qwen36-27b-awq-tp1-ctx32k.ckpt"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/T_QX_PRELOAD_thinker_${TIMESTAMP}"
STABLE_LOG="/tmp/vllm_thinker_checkpoint.log"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " T_QX_PRELOAD: THINKER CHECKPOINT & PRE-WARM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[T_QX_PRELOAD] Model: ${MODEL}"
echo "[T_QX_PRELOAD] Checkpoint: ${CHECKPOINT_FILE}"
echo "[T_QX_PRELOAD] Results: ${RESULTS_DIR}"

# --- Step 0: Cleanup ---
echo "[T_QX_PRELOAD] Cleaning up existing thinker (port ${PORT})..."
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker\|gpu1" | head -1 || true)
if [ -n "${THINKER_CONTAINER}" ]; then
    podman stop "${THINKER_CONTAINER}" >/dev/null 2>&1 && podman rm "${THINKER_CONTAINER}" >/dev/null 2>&1 || true
fi
sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
pkill -u "$USER" -9 -f "vllm|VLLM|resource_tracker|vllm_criu_wrapper" || true
sudo rm -f /dev/shm/sem.* /dev/shm/link_remap.* >/dev/null 2>&1 || true
sleep 5

if [ "${SKIP_INIT}" -eq 1 ]; then
    echo "[T_QX_PRELOAD] Skipping engine restart and checkpoint creation (--skip-init active)."
fi

# --- Step 1: Create Checkpoint (if missing and not skipped) ---
if [ "${SKIP_INIT}" -eq 0 ] && [ ! -d "${CHECKPOINT_FILE}" ]; then
    echo "[T_QX_PRELOAD] Checkpoint missing. Creating it now..."
    
    # Start vLLM via CRIU-stabilized Wrapper
    export PYTHONPATH="${REPO_ROOT}/benchmarks/queue/python_hijack:${PYTHONPATH:-}"
    export UV_USE_IO_URING=0
    export VLLM_USE_V1=1
    export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
    export VLLM_MQ_MAX_CHUNK_BYTES_MB=999999
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export LD_PRELOAD="${REPO_ROOT}/benchmarks/queue/io_uring_shim.so"
    
    touch "${STABLE_LOG}"
    ln -sf "${STABLE_LOG}" "${RESULTS_DIR}/vllm_host_init.log"
    
    python3 -c "
import subprocess, os
log = open('${STABLE_LOG}', 'w')
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '${GPU_1_ID}'
subprocess.Popen(
    ['python3', '${REPO_ROOT}/benchmarks/queue/vllm_criu_wrapper.py',
     '--model', '${MODEL}',
     '--port', '${PORT}',
     '--gpu-memory-utilization', '0.90',
     '--max-model-len', '32768',
     '--kv-cache-dtype', 'fp8',
     '--max-num-seqs', '4',
     '--enable-chunked-prefill',
     '--enforce-eager',
     '--tool-call-parser', 'qwen3_coder',
     '--reasoning-parser', 'qwen3',
     '--enable-auto-tool-choice'],
    stdout=log, stderr=log, stdin=subprocess.DEVNULL,
    env=env, close_fds=True, start_new_session=True
)
"
    echo ""
    echo "[T_QX_PRELOAD] ── Live log stream (Ctrl-C does not stop the engine) ──"
    echo "[T_QX_PRELOAD] Follow logs manually: tail -f ${STABLE_LOG}"
    echo ""
    tail -f "${STABLE_LOG}" 2>&1 | sed "s/^/  [vllm-host] /" &
    LOGS_PID=$!
    
    echo "[T_QX_PRELOAD] Waiting for model to load..."
    READY=0
    for i in $(seq 1 600); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "[T_QX_PRELOAD] HEALTH OK"
            READY=1
            break
        fi
        sleep 2
    done
    
    kill "${LOGS_PID}" 2>/dev/null || true
    
    if [ "${READY}" -eq 0 ]; then
        echo "[T_QX_PRELOAD] ERROR: Model failed to start. Check ${RESULTS_DIR}/vllm_host_init.log"
        exit 1
    fi
    
    MAIN_PID=$(pgrep -f "vllm_criu_wrapper.py.*--port ${PORT}" | head -1)
    
    # Checkpoint
    echo "[T_QX_PRELOAD] Toggling GPU for PIDs..."
    PIDS=$(ps -ef | grep "${MAIN_PID}" | grep -v grep | awk '{print $2}')
    for p in ${PIDS}; do
        if ps -p "$p" -o cmd= | grep -q "resource_tracker"; then
            continue
        fi
        echo "[T_QX_PRELOAD] Toggling PID: $p"
        sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true
    done
    
    echo "[T_QX_PRELOAD] Syncing logs for dump..."
    truncate -s 0 "${STABLE_LOG}"
    
    echo "[T_QX_PRELOAD] Dumping checkpoint..."
    TMP_CKPT_DIR=$(mktemp -d -p /srv/ai/checkpoints)
    if ! sudo "${CRIU}" dump \
        --tree "${MAIN_PID}" \
        --images-dir "${TMP_CKPT_DIR}" \
        --shell-job \
        --tcp-established \
        --file-locks \
        --link-remap \
        --log-file "${RESULTS_DIR}/criu_dump_init.log" \
        --verbosity=3; then
        echo "[T_QX_PRELOAD] ERROR: CRIU Dump failed."
        # Resume GPU for cleanup
        for p in ${PIDS}; do
            if ! ps -p "$p" -o cmd= | grep -q "resource_tracker"; then
                echo "[T_QX_PRELOAD] Resuming PID: $p (Cleanup)"
                sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true
            fi
        done
        sudo chown -R "$(id -u):$(id -g)" "${RESULTS_DIR}"
        rm -rf "${TMP_CKPT_DIR}"
        exit 1
    fi
    sudo chown -R "$(id -u):$(id -g)" "${RESULTS_DIR}" "${TMP_CKPT_DIR}"
    
    # Standardize checkpoint file (using Podman format or directory?)
    # Since we use host-native CRIU, we store the directory or a tarball.
    # BENCH_18 implies a single file path for THINKER_CKPT.
    # We will tar it for Podman compatibility or just keep the directory.
    # Actually, Podman restore --import takes a tarball.
    # But since we use host-native restore, we'll keep the directory.
    # Let's use the directory path as the "file" for now.
    
    sudo rm -rf "${CHECKPOINT_FILE}" >/dev/null 2>&1 || true
    sudo mv "${TMP_CKPT_DIR}" "${CHECKPOINT_FILE}"
    echo "[T_QX_PRELOAD] Checkpoint created at ${CHECKPOINT_FILE}"
    
    echo "[T_QX_PRELOAD] Backing up /dev/shm state..."
    sudo mkdir -p "${CHECKPOINT_FILE}/shm_backup"
    sudo cp -a /dev/shm/sem.* /dev/shm/link_remap.* "${CHECKPOINT_FILE}/shm_backup/" 2>/dev/null || true
    
    # Final cleanup of init process
    sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    sleep 2
else
    echo "[T_QX_PRELOAD] Using existing checkpoint: ${CHECKPOINT_FILE}"
fi

# --- Step 2: Cold Restore Measurement ---
echo "[T_QX_PRELOAD] --- Cold Restore (Cache Dropped) ---"
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
echo "[T_QX_PRELOAD] Page cache dropped."

# Truncate log to match dump state (0 bytes)
truncate -s 0 "${STABLE_LOG}"
ln -sf "${STABLE_LOG}" "${RESULTS_DIR}/vllm_host_init.log"

echo "[T_QX_PRELOAD] Restoring /dev/shm state..."
sudo cp -a "${CHECKPOINT_FILE}/shm_backup/"* /dev/shm/ 2>/dev/null || true

COLD_START_MS=$(date +%s%3N)
if ! sudo "${CRIU}" restore \
    --images-dir "${CHECKPOINT_FILE}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --link-remap \
    --restore-detached \
    --log-file "${RESULTS_DIR}/criu_restore_cold.log" \
    --verbosity=3; then
    echo "[T_QX_PRELOAD] ERROR: Cold restore failed."
    sudo chown -R "$(id -u):$(id -g)" "${RESULTS_DIR}"
    sudo rm -f /dev/shm/sem.* /dev/shm/link_remap.* >/dev/null 2>&1 || true
    exit 1
fi
sudo chown -R "$(id -u):$(id -g)" "${RESULTS_DIR}"

# Resume GPU
MAIN_PID=""
for i in $(seq 1 10); do
    MAIN_PID=$(pgrep -f "vllm_criu_wrapper.py.*--port ${PORT}" | head -1 || true)
    if [ -n "${MAIN_PID}" ]; then break; fi
    sleep 1
done

if [ -z "${MAIN_PID}" ]; then
    echo "[T_QX_PRELOAD] ERROR: Could not find MAIN_PID after cold restore."
    exit 1
fi

PIDS=$(ps -ef | grep "${MAIN_PID}" | grep -v grep | awk '{print $2}')
for p in ${PIDS}; do
    if ! ps -p "$p" -o cmd= | grep -q "resource_tracker"; then
        echo "[T_QX_PRELOAD] Resuming PID: $p"
        sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true
    fi
done

# Wait for health
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then break; fi
    sleep 1
done

COLD_END_MS=$(date +%s%3N)
COLD_MS=$(( COLD_END_MS - COLD_START_MS ))
echo "[T_QX_PRELOAD] Cold restore completed in ${COLD_MS}ms"

# Cleanup for warm test
sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
pkill -u "$USER" -9 -f "vllm|VLLM|resource_tracker|vllm_criu_wrapper" || true
sudo rm -f /dev/shm/sem.* /dev/shm/link_remap.* >/dev/null 2>&1 || true
sleep 5

# --- Step 3: Warm Restore Measurement ---
echo "[T_QX_PRELOAD] --- Warm Restore (Pre-loaded) ---"
# Trigger posix_fadvise
"${PRELOAD_SCRIPT}" "${CHECKPOINT_FILE}"
# Wait for kernel to load pages
sleep 2

# Truncate log to match dump state (0 bytes)
truncate -s 0 "${STABLE_LOG}"
ln -sf "${STABLE_LOG}" "${RESULTS_DIR}/vllm_host_init.log"

echo "[T_QX_PRELOAD] Restoring /dev/shm state..."
sudo cp -a "${CHECKPOINT_FILE}/shm_backup/"* /dev/shm/ 2>/dev/null || true

WARM_START_MS=$(date +%s%3N)
if ! sudo "${CRIU}" restore \
    --images-dir "${CHECKPOINT_FILE}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --link-remap \
    --restore-detached \
    --log-file "${RESULTS_DIR}/criu_restore_warm.log" \
    --verbosity=3; then
    echo "[T_QX_PRELOAD] ERROR: Warm restore failed."
    sudo chown -R "$(id -u):$(id -g)" "${RESULTS_DIR}"
    sudo rm -f /dev/shm/sem.* /dev/shm/link_remap.* >/dev/null 2>&1 || true
    exit 1
fi
sudo chown -R "$(id -u):$(id -g)" "${RESULTS_DIR}"

# Resume GPU
MAIN_PID=""
for i in $(seq 1 10); do
    MAIN_PID=$(pgrep -f "vllm_criu_wrapper.py.*--port ${PORT}" | head -1 || true)
    if [ -n "${MAIN_PID}" ]; then break; fi
    sleep 1
done

if [ -z "${MAIN_PID}" ]; then
    echo "[T_QX_PRELOAD] ERROR: Could not find MAIN_PID after warm restore."
    exit 1
fi

PIDS=$(ps -ef | grep "${MAIN_PID}" | grep -v grep | awk '{print $2}')
for p in ${PIDS}; do
    if ! ps -p "$p" -o cmd= | grep -q "resource_tracker"; then
        echo "[T_QX_PRELOAD] Resuming PID: $p"
        sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true
    fi
done

# Wait for health
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then break; fi
    sleep 1
done

WARM_END_MS=$(date +%s%3N)
WARM_MS=$(( WARM_END_MS - WARM_START_MS ))
echo "[T_QX_PRELOAD] Warm restore completed in ${WARM_MS}ms"

# --- Step 4: Analysis ---
SPEEDUP=$(python3 -c "print(round(${COLD_MS} / ${WARM_MS}, 2))")
echo "[T_QX_PRELOAD] SPEEDUP: ${SPEEDUP}x"

# --- Step 5: Summary & Production Restore ---
cat > "${RESULTS_DIR}/summary.md" <<EOMD
# T_QX_PRELOAD — Thinker Pre-warm Benchmark — ${TIMESTAMP}

## Restore Times
| Condition | Time (ms) |
|-----------|-----------|
| Cold (NVMe, Cache Dropped) | ${COLD_MS} |
| Warm (posix_fadvise) | ${WARM_MS} |
| **Speedup** | **${SPEEDUP}x** |

## Checkpoint
Path: ${CHECKPOINT_FILE}
Size: $(sudo du -sh "${CHECKPOINT_FILE}" | awk '{print $1}')

## Verdict
$([[ $(python3 -c "print(1 if ${SPEEDUP} >= 1.2 else 0)") == "1" ]] && echo "PASS" || echo "MARGINAL")
EOMD

echo "[T_QX_PRELOAD] Summary written to ${RESULTS_DIR}/summary.md"

# Final Cleanup
sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
pkill -u "$USER" -9 -f "vllm|VLLM|resource_tracker|vllm_criu_wrapper" || true
sudo rm -f /dev/shm/sem.* /dev/shm/link_remap.* >/dev/null 2>&1 || true
sleep 2

# Redeploy production container
echo "[T_QX_PRELOAD] Redeploying production container..."
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

echo "[T_QX_PRELOAD] DONE."
