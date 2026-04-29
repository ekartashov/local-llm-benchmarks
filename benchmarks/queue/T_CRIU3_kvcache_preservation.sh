#!/usr/bin/env bash
# benchmarks/queue/T_CRIU3_kvcache_preservation.sh
#
# T_CRIU3 Phase 2 — CRIU KV Cache Preservation (Coder TP=2, Host-Native)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# -----------------------------------------------------------------------------
# 0. SETUP & CLEANUP
# -----------------------------------------------------------------------------
pkill -u "$USER" -9 -f "vllm|VLLM::" || true

# ── Settled constants (do not guess) ─────────────────────────────────────────
CUDA_CHECKPOINT="/usr/local/bin/cuda-checkpoint"
CRIU="/usr/sbin/criu"

MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
PORT=30000
CHECKPOINT_DIR="/srv/ai/checkpoints/coder-tp2-kvcache"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/T_CRIU3_kvcache_preservation_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}"
rm -rf "${CHECKPOINT_DIR}" && mkdir -p "${CHECKPOINT_DIR}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " T_CRIU3 PHASE 2: KV CACHE PRESERVATION (TP=2)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[T_CRIU3] Model:      ${MODEL}"
echo "[T_CRIU3] Port:       ${PORT}"
echo "[T_CRIU3] Results:    ${RESULTS_DIR}"
echo "[T_CRIU3] Checkpoint: ${CHECKPOINT_DIR}"

# ── Step 0: Stop existing coder container and any host-native residue ─────────
echo "[T_CRIU3] Step 0: Clearing port ${PORT}..."

# Ensure nvidia-persistenced is healthy and the socket exists
if [ ! -S /run/nvidia-persistenced/socket ]; then
    sudo mkdir -p /run/nvidia-persistenced
    sudo nvidia-persistenced --user cassini >/dev/null 2>&1 || true
    sudo chmod 777 /run/nvidia-persistenced/socket || true
fi

CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -E "bench-vllm-tp2a|bench-vllm-tp2|coder" | head -1 || true)
if [ -n "${CODER_CONTAINER}" ]; then
    echo "[T_CRIU3]   Stopping podman container: ${CODER_CONTAINER}"
    podman stop "${CODER_CONTAINER}" >/dev/null 2>&1 || true
    podman rm   "${CODER_CONTAINER}" >/dev/null 2>&1 || true
fi
# Surgical cleanup
rm -rf /dev/shm/* 2>/dev/null || true
sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true

# Reset GPUs to clear ghost VRAM
echo "[T_CRIU3]   Resetting GPUs to clear ghost VRAM..."
sudo nvidia-smi --gpu-reset -i 0 2>/dev/null || true
sudo nvidia-smi --gpu-reset -i 1 2>/dev/null || true
sleep 2

# ── Step 1: Build benchmark payload ──────────────────────────────────────────
python3 -c "
import json
base = 'The quick brown fox jumps over the lazy dog. '
prompt = base * 300 + 'Summarize the above in one sentence.'
payload = {'model': 'coder', 'prompt': prompt, 'max_tokens': 10, 'temperature': 0.0}
with open('${RESULTS_DIR}/payload.json', 'w') as f:
    json.dump(payload, f)
"

# ── Step 2: Start vLLM host-native with prefix caching ───────────────────────
echo "[T_CRIU3] Step 2: Starting vLLM Coder TP=2 (host-native, prefix caching)..."
echo "[T_CRIU3]   Logs → ${RESULTS_DIR}/vllm_host.log"

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
WRAPPER="${REPO_ROOT}/benchmarks/queue/vllm_criu_wrapper.py"

# vLLM host-native launch

touch "${RESULTS_DIR}/vllm_host.log"
python3 -c "
import subprocess, os
log = open('${RESULTS_DIR}/vllm_host.log', 'w')
env = os.environ.copy()
env['UV_USE_IO_URING'] = '0'
env['VLLM_USE_V1'] = '1'
env['VLLM_V1'] = '1'
env['VLLM_LOGGING_LEVEL'] = 'DEBUG'
env['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '600'
env['VLLM_RPC_TIMEOUT'] = '600000'
env['CUDA_VISIBLE_DEVICES'] = '0,1'
env['LD_PRELOAD'] = '${REPO_ROOT}/benchmarks/queue/io_uring_shim.so'
subprocess.Popen(
    ['python3', '${WRAPPER}',
     '--model', '${MODEL}',
     '--port', '${PORT}',
     '--tensor-parallel-size', '2',
     '--gpu-memory-utilization', '0.90',
     '--kv-cache-dtype', 'fp8',
     '--enable-prefix-caching',
     '--enforce-eager',
     '--max-model-len', '32768',
     '--served-model-name', 'coder',
     '--max-num-batched-tokens', '4096',
     '--reasoning-parser', 'qwen3',
     '--tool-call-parser', 'qwen3_coder',
     '--enable-auto-tool-choice',
     '--no-async-scheduling'],
    stdout=log, stderr=log, stdin=subprocess.DEVNULL,
    env=env, close_fds=True, start_new_session=True
)
"

tail -f "${RESULTS_DIR}/vllm_host.log" | stdbuf -oL sed 's/\r//g; s/^/[vllm-tp2] /' &
LOGS_PID=$!

# Wait for the actual API server process
sleep 5
MAIN_PID=$(pgrep -f "python3.*vllm_criu_wrapper[.]py" | sort -n | tail -1 || true)
if [ -z "${MAIN_PID}" ]; then
    kill "${LOGS_PID}" 2>/dev/null || true
    echo "[T_CRIU3] ERROR: vLLM failed to start."
    exit 1
fi
echo "[T_CRIU3]   vLLM host PID: ${MAIN_PID}"

# Wait for health
echo "[T_CRIU3]   Waiting for health check (up to 600s)..."
READY=0
for i in $(seq 1 600); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "${MAIN_PID}" 2>/dev/null; then
        break
    fi
    sleep 1
done

kill "${LOGS_PID}" 2>/dev/null || true
if [ "${READY}" -eq 0 ]; then
    exit 1
fi
echo "[T_CRIU3]   HEALTH OK"

# ── Step 3: Cold prefill ──────────────────────────────────────────────────────
echo "[T_CRIU3] Step 3: Cold TTFT (cache miss)..."
START_MS=$(date +%s%3N)
curl -sf "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" -d @"${RESULTS_DIR}/payload.json" > /dev/null
END_MS=$(date +%s%3N)
COLD_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "[T_CRIU3]   Cold TTFT: ${COLD_TTFT_S}s"

# ── Step 4: Warm prefill ──────────────────────────────────────────────────────
echo "[T_CRIU3] Step 4: Warm TTFT (cache hit expected)..."
START_MS=$(date +%s%3N)
curl -sf "http://localhost:${PORT}/v1/completions" -H "Content-Type: application/json" -d @"${RESULTS_DIR}/payload.json" > /dev/null
END_MS=$(date +%s%3N)
WARM_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "[T_CRIU3]   Warm TTFT: ${WARM_TTFT_S}s"

# ── Step 5: Record VRAM ───────────────────────────────────────────────────────
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_pre_checkpoint.txt"
echo "[T_CRIU3] Step 5: VRAM pre-checkpoint: $(paste -s -d'  ' "${RESULTS_DIR}/vram_pre_checkpoint.txt")"

# ── Step 6: Checkpoint ────────────────────────────────────────────────────────
echo "[T_CRIU3] Step 6: Checkpointing..."

get_tree_pids() {
    local pid=$1
    echo "$pid"
    local children
    children=$(ps --ppid "$pid" -o pid= 2>/dev/null | tr -d ' \t' || true)
    for child in $children; do
        [[ -n "$child" ]] && get_tree_pids "$child"
    done
}
PIDS=$(get_tree_pids "${MAIN_PID}" | sort -u | tr '\n' ' ')

echo "[T_CRIU3]   Toggling GPU state OFF..."
# Use nvidia-smi to find the actual PIDs using the GPUs
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs echo || true)
ALL_TARGET_PIDS=$(echo "${PIDS} ${GPU_PIDS}" | tr ' ' '\n' | sort -u)

echo "[T_CRIU3]   Checking for real io_uring rings in process tree..."
for pid in ${ALL_TARGET_PIDS}; do
    if sudo grep -q "anon_inode:\[io_uring\]" "/proc/${pid}/maps" 2>/dev/null; then
        echo "[T_CRIU3]   WARNING: Real io_uring ring found in PID ${pid}"
        sudo grep "anon_inode:\[io_uring\]" "/proc/${pid}/maps"
    fi
done

for pid in ${ALL_TARGET_PIDS}; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[T_CRIU3]   Suspending CUDA for PID ${pid}..."
        # We ignore errors here because non-CUDA processes in the tree (API server, etc.)
        # will return "initialization error", which is expected.
        sudo "${CUDA_CHECKPOINT}" --toggle --pid "${pid}" || true
    fi
done

echo "[T_CRIU3]   Running CRIU dump → ${CHECKPOINT_DIR}..."
CKPT_START_MS=$(date +%s%3N)
if sudo "${CRIU}" dump \
    --tree "${MAIN_PID}" \
    --images-dir "${CHECKPOINT_DIR}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --log-file "${RESULTS_DIR}/criu_dump.log" \
    --link-remap \
    --verbosity=3; then

    CKPT_END_MS=$(date +%s%3N)
    CKPT_ELAPSED_S=$(python3 -c "print(round(($CKPT_END_MS - $CKPT_START_MS) / 1000.0, 2))")
    CKPT_SIZE_GB=$(sudo du -sh "${CHECKPOINT_DIR}" 2>/dev/null | awk '{print $1}')
    echo "[T_CRIU3]   Checkpoint OK: ${CKPT_ELAPSED_S}s, ${CKPT_SIZE_GB}"
    sudo chown -R $(id -u):$(id -g) "${RESULTS_DIR}" "${CHECKPOINT_DIR}"
else
    echo "[T_CRIU3] ERROR: CRIU dump failed. See ${RESULTS_DIR}/criu_dump.log"
    sudo chown -R $(id -u):$(id -g) "${RESULTS_DIR}" "${CHECKPOINT_DIR}" 2>/dev/null || true
    for p in ${ALL_TARGET_PIDS}; do sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" >/dev/null 2>&1 || true; done
    # Aggressive cleanup before recovery
    sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    pkill -u "$USER" -9 -f "vllm|VLLM::" || true
    bash "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" --gpu-mem-util 0.90 --ctx 32768 \
        --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
    exit 1
fi

# Wait for process tree to clear
sleep 2
for p in ${PIDS}; do kill -0 "$p" 2>/dev/null && { sudo kill -9 "$p" 2>/dev/null || true; }; done

# ── Step 7: Restore ───────────────────────────────────────────────────────────
echo "[T_CRIU3] Step 7: CRIU restore..."
RESTORE_START_MS=$(date +%s%3N)

if ! sudo "${CRIU}" restore \
    --images-dir "${CHECKPOINT_DIR}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --restore-detached \
    --log-file "${RESULTS_DIR}/criu_restore.log" \
    --verbosity=3; then
    echo "[T_CRIU3] ERROR: CRIU restore failed."
    sudo nvidia-smi --gpu-reset -i 0 || true
    sudo nvidia-smi --gpu-reset -i 1 || true
    bash "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" --gpu-mem-util 0.90 --ctx 32768 \
        --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
    exit 1
fi

echo "[T_CRIU3]   Resuming GPU state..."
for pid in ${ALL_TARGET_PIDS}; do
    sudo "${CUDA_CHECKPOINT}" --toggle --pid "${pid}" >/dev/null 2>&1 || true
done
echo "[T_CRIU3]   Waiting 10s for GPU VRAM migration to settle..."
sleep 10
sudo chown $(id -u):$(id -g) "${RESULTS_DIR}/criu_restore.log" 2>/dev/null || true

# Wait for health
RESTORE_READY=0
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[T_CRIU3]   HEALTH OK after restore"
        RESTORE_READY=1
        break
    fi
    sleep 1
done

RESTORE_END_MS=$(date +%s%3N)
RESTORE_ELAPSED_S=$(python3 -c "print(round(($RESTORE_END_MS - $RESTORE_START_MS) / 1000.0, 2))")
echo "[T_CRIU3]   Restore time: ${RESTORE_ELAPSED_S}s"

if [ "${RESTORE_READY}" -eq 0 ]; then
    bash "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" --gpu-mem-util 0.90 --ctx 32768 \
        --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
    exit 1
fi

# ── Step 8: Post-restore KV cache check ──────────────────────────────────────
echo "[T_CRIU3] Step 8: Post-restore KV cache check..."
START_MS=$(date +%s%3N)
# Use a temporary file to capture the response body on failure
RESPONSE_FILE="${RESULTS_DIR}/post_restore_response.json"
if ! curl -s -f "http://localhost:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d @"${RESULTS_DIR}/payload.json" \
    -o "${RESPONSE_FILE}" 2>"${RESULTS_DIR}/curl_error.log"; then
    echo "[T_CRIU3]   ERROR: Post-restore check failed (curl code $?)."
    echo "[T_CRIU3]   Server Response: $(cat "${RESPONSE_FILE}" 2>/dev/null || echo "No response body")"
    exit 22
fi
END_MS=$(date +%s%3N)
POST_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
POST_COLD_RATIO=$(python3 -c "print(round(${POST_TTFT_S} / ${COLD_TTFT_S}, 3))" 2>/dev/null || echo "0")
echo "[T_CRIU3]   Post-restore TTFT: ${POST_TTFT_S}s (ratio vs cold: ${POST_COLD_RATIO})"

KV_VERDICT=$(python3 -c "
r = ${POST_COLD_RATIO}
print('KV_CACHE_PRESERVED' if r < 0.30 else ('KV_CACHE_LOST' if r > 0.70 else 'KV_CACHE_PARTIAL'))
")
echo "[T_CRIU3]   KV verdict: ${KV_VERDICT}"

# ── Step 9: Write summary ─────────────────────────────────────────────────────
cat > "${RESULTS_DIR}/summary.md" <<EOMD
# T_CRIU3 Phase 2 — CRIU KV Cache Preservation — ${TIMESTAMP}
## Result
RESTORE_OK
| Metric | Value |
|--------|-------|
| Cold TTFT | ${COLD_TTFT_S}s |
| Warm TTFT | ${WARM_TTFT_S}s |
| Post-restore TTFT | ${POST_TTFT_S}s |
| KV verdict | ${KV_VERDICT} |
EOMD

# ── Step 10: Cleanup and redeploy production coder ────────────────────────────
echo "[T_CRIU3] Step 10: Cleanup and redeploy production coder..."
sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
sudo rm -rf /dev/shm/* || true
sleep 2

bash "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" --gpu-mem-util 0.90 --ctx 32768 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
