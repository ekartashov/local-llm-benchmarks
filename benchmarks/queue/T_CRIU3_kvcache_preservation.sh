#!/usr/bin/env bash
# benchmarks/queue/T_CRIU3_kvcache_preservation.sh
#
# T_CRIU3 Phase 2 — CRIU KV Cache Preservation (Coder TP=2, Host-Native)
#
# Objective: Confirm that host-native CRIU + cuda-checkpoint preserves the
# vLLM prefix cache (VRAM KV blocks + CPU metadata across checkpoint/restore).
#
# Key design decisions (DO NOT CHANGE without reading docs/procedures/criu-ops.md):
#   1. vLLM runs on the HOST, not in a container.
#      Podman CDI mount-point conflicts break CRIU inside containers (T_CRIU2).
#   2. UV_USE_IO_URING=0 is REQUIRED.  The vllm/entrypoints/openai/api_server.py
#      and vllm/v1/utils.py uvloop patch (asyncio.run → uvloop.run) must be applied.
#      Without these, CRIU fails: "Unknown shit 600 (anon_inode:[io_uring])".
#   3. vLLM stdout/stderr MUST go to a log file via subprocess.Popen — never
#      inherited from the shell.  Inherited output corrupts the terminal on
#      TP=2 because vLLM's worker threads write raw bytes with no carriage returns.
#   4. Tool paths are settled from T_KV2 / T_CRIU3 Phase 1:
#        CUDA_CHECKPOINT = /usr/local/bin/cuda-checkpoint
#        CRIU            = /usr/sbin/criu

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

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
CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -E "bench-vllm-tp2a|bench-vllm-tp2|coder" | head -1 || true)
if [ -n "${CODER_CONTAINER}" ]; then
    echo "[T_CRIU3]   Stopping podman container: ${CODER_CONTAINER}"
    podman stop "${CODER_CONTAINER}" >/dev/null 2>&1 || true
    podman rm   "${CODER_CONTAINER}" >/dev/null 2>&1 || true
fi
sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
sudo rm -rf /dev/shm/* 2>/dev/null || true
# Reset GPUs to clear ghost VRAM from any previous failed CRIU checkpoint/restore.
# Must run after container stop (gpu-reset fails if any process holds the GPU).
echo "[T_CRIU3]   Resetting GPUs to clear ghost VRAM..."
sudo nvidia-smi --gpu-reset -i 0 2>/dev/null || echo "[T_CRIU3]   GPU 0 reset skipped (may still be in use)"
sudo nvidia-smi --gpu-reset -i 1 2>/dev/null || echo "[T_CRIU3]   GPU 1 reset skipped (may still be in use)"
sleep 2

# ── Step 1: Build benchmark payload ──────────────────────────────────────────
python3 -c "
import json
base = 'The quick brown fox jumps over the lazy dog. '
prompt = base * 300 + 'Summarize the above in one sentence.'
payload = {'model': 'coder', 'prompt': prompt, 'max_tokens': 10, 'temperature': 0.0}
with open('${RESULTS_DIR}/payload.json', 'w') as f:
    json.dump(payload, f)
approx_tokens = len(prompt) // 4
print(f'[T_CRIU3] Payload written. Approx tokens: {approx_tokens}')
"

# ── Step 2: Start vLLM host-native with prefix caching ───────────────────────
echo "[T_CRIU3] Step 2: Starting vLLM Coder TP=2 (host-native, prefix caching)..."
echo "[T_CRIU3]   Logs → ${RESULTS_DIR}/vllm_host.log"

export UV_USE_IO_URING=0
export VLLM_V1_ENABLED=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

# subprocess.Popen writes stdout+stderr to a file and closes all inherited FDs.
# start_new_session=True makes the child a session leader, required for --shell-job.
touch "${RESULTS_DIR}/vllm_host.log"
python3 -c "
import subprocess, os
log = open('${RESULTS_DIR}/vllm_host.log', 'w')
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0,1'
subprocess.Popen(
    ['python3', '-m', 'vllm.entrypoints.openai.api_server',
     '--model', '${MODEL}',
     '--port', '${PORT}',
     '--tensor-parallel-size', '2',
     '--gpu-memory-utilization', '0.90',
     '--kv-cache-dtype', 'fp8',
     '--enable-prefix-caching',
     '--max-model-len', '32768',
     '--served-model-name', 'coder',
     '--max-num-batched-tokens', '4096',
     '--reasoning-parser', 'qwen3',
     '--tool-call-parser', 'qwen3_coder',
     '--enable-auto-tool-choice'],
    stdout=log, stderr=log, stdin=subprocess.DEVNULL,
    env=env, close_fds=True, start_new_session=True
)
"

# Stream logs to terminal.  stdbuf -oL forces line-buffered output so lines appear
# immediately.  s/\r//g strips tqdm carriage-returns before they reach the tty.
# $! is the sed PID (last in pipeline); killing it sends SIGPIPE to tail which exits.
tail -f "${RESULTS_DIR}/vllm_host.log" | stdbuf -oL sed 's/\r//g; s/^/[vllm-tp2] /' &
LOGS_PID=$!

# Give vLLM a moment to fork, then grab its PID for liveness checks.
sleep 3
MAIN_PID=$(pgrep -f "vllm[.]entrypoints[.]openai[.]api_server.*--port ${PORT}" | head -1 || true)
if [ -z "${MAIN_PID}" ]; then
    kill "${LOGS_PID}" 2>/dev/null || true
    echo "[T_CRIU3] ERROR: vLLM failed to start (no PID found). Check ${RESULTS_DIR}/vllm_host.log"
    echo "STARTUP_FAILED" > "${RESULTS_DIR}/status.txt"
    exit 1
fi
echo "[T_CRIU3]   vLLM host PID: ${MAIN_PID}"

# Wait for health (600s — TP=2 weight load + CUDA graph capture can take ~100s).
# Bail immediately if the process dies instead of waiting out the full timeout.
echo "[T_CRIU3]   Waiting for health check (up to 600s)..."
READY=0
for i in $(seq 1 600); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "${MAIN_PID}" 2>/dev/null; then
        echo "[T_CRIU3] ERROR: vLLM process ${MAIN_PID} died. Check ${RESULTS_DIR}/vllm_host.log"
        break
    fi
    sleep 1
done

kill "${LOGS_PID}" 2>/dev/null || true
if [ "${READY}" -eq 0 ]; then
    echo "STARTUP_FAILED" > "${RESULTS_DIR}/status.txt"
    exit 1
fi
echo "[T_CRIU3]   HEALTH OK"

# ── Step 3: Cold prefill (first call, no cache) ───────────────────────────────
echo "[T_CRIU3] Step 3: Cold TTFT (cache miss)..."
START_MS=$(date +%s%3N)
COLD_RESPONSE=$(curl -sf "http://localhost:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d @"${RESULTS_DIR}/payload.json")
END_MS=$(date +%s%3N)
COLD_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "[T_CRIU3]   Cold TTFT: ${COLD_TTFT_S}s"
echo "${COLD_RESPONSE}" > "${RESULTS_DIR}/cold_response.json"
echo "cold_ttft_s=${COLD_TTFT_S}" > "${RESULTS_DIR}/timings.txt"

# ── Step 4: Warm prefill (same prompt, cache hit expected) ────────────────────
echo "[T_CRIU3] Step 4: Warm TTFT (cache hit expected)..."
START_MS=$(date +%s%3N)
WARM_RESPONSE=$(curl -sf "http://localhost:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d @"${RESULTS_DIR}/payload.json")
END_MS=$(date +%s%3N)
WARM_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
CACHE_RATIO=$(python3 -c "print(round(${WARM_TTFT_S} / ${COLD_TTFT_S}, 3))" 2>/dev/null || echo "1.0")
echo "[T_CRIU3]   Warm TTFT: ${WARM_TTFT_S}s  (ratio vs cold: ${CACHE_RATIO})"
echo "${WARM_RESPONSE}" > "${RESULTS_DIR}/warm_response.json"
echo "warm_ttft_s=${WARM_TTFT_S}" >> "${RESULTS_DIR}/timings.txt"
echo "warm_cold_ratio=${CACHE_RATIO}" >> "${RESULTS_DIR}/timings.txt"

if python3 -c "import sys; sys.exit(0 if ${CACHE_RATIO} < 0.30 else 1)"; then
    echo "[T_CRIU3]   CACHE HIT CONFIRMED (ratio ${CACHE_RATIO} < 0.30 ✓)"
else
    echo "[T_CRIU3]   WARN: ratio ${CACHE_RATIO} >= 0.30 — prefix cache may not be active."
    echo "[T_CRIU3]   Proceeding to measure restoration behavior regardless."
fi

# ── Step 5: Record VRAM (with populated KV blocks) ────────────────────────────
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    > "${RESULTS_DIR}/vram_pre_checkpoint.txt"
echo "[T_CRIU3] Step 5: VRAM pre-checkpoint: $(paste -s -d'  ' "${RESULTS_DIR}/vram_pre_checkpoint.txt")"

# ── Step 6: Checkpoint (with populated prefix cache) ─────────────────────────
echo "[T_CRIU3] Step 6: Checkpointing..."

# Collect full process tree (recursive pstree via ps --ppid) for cuda-checkpoint
# toggle.  TP=2 workers may be children-of-children of MAIN_PID.
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
echo "[T_CRIU3]   Process tree: ${PIDS}"

echo "[T_CRIU3]   Toggling GPU state OFF..."
for p in ${PIDS}; do
    sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" \
        || echo "[T_CRIU3]   PID $p: not a CUDA process (expected for API server — see criu-ops.md)"
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
    --verbosity=3; then

    CKPT_END_MS=$(date +%s%3N)
    CKPT_ELAPSED_S=$(python3 -c "print(round(($CKPT_END_MS - $CKPT_START_MS) / 1000.0, 2))")
    CKPT_SIZE_GB=$(sudo du -sh "${CHECKPOINT_DIR}" 2>/dev/null | awk '{print $1}')
    echo "[T_CRIU3]   Checkpoint OK: ${CKPT_ELAPSED_S}s, ${CKPT_SIZE_GB}"
    echo "checkpoint_elapsed_s=${CKPT_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"
    echo "checkpoint_size_gb=${CKPT_SIZE_GB}"     >> "${RESULTS_DIR}/timings.txt"
else
    echo "[T_CRIU3] ERROR: CRIU dump failed. See ${RESULTS_DIR}/criu_dump.log"
    for p in ${PIDS}; do sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true; done
    echo "CHECKPOINT_FAILED" > "${RESULTS_DIR}/status.txt"
    echo "checkpoint_elapsed_s=N/A" >> "${RESULTS_DIR}/timings.txt"
    # Cleanup and restore production coder
    sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    sudo rm -rf /dev/shm/* || true
    VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" \
        --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
        --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
    exit 1
fi

# Wait for process tree to clear after dump (CRIU terminates dumped processes)
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
    echo "[T_CRIU3] ERROR: CRIU restore failed. See ${RESULTS_DIR}/criu_restore.log"
    echo "RESTORE_FAILED" > "${RESULTS_DIR}/status.txt"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        > "${RESULTS_DIR}/vram_at_failure.txt"
    sudo nvidia-smi --gpu-reset -i 0 || true
    sudo nvidia-smi --gpu-reset -i 1 || true
    VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" \
        --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
        --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
    exit 1
fi

echo "[T_CRIU3]   Resuming GPU state..."
for p in ${PIDS}; do
    sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" \
        || echo "[T_CRIU3]   PID $p: toggle on restore (may be normal — see criu-ops.md)"
done

# Wait for health after restore
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
echo "restore_elapsed_s=${RESTORE_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

if [ "${RESTORE_READY}" -eq 0 ]; then
    echo "[T_CRIU3] ERROR: Health timeout after restore."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        > "${RESULTS_DIR}/vram_at_failure.txt"
    echo "RESTORE_FAILED" > "${RESULTS_DIR}/status.txt"
    sudo nvidia-smi --gpu-reset -i 0 || true
    sudo nvidia-smi --gpu-reset -i 1 || true
    VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" \
        --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
        --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
    exit 1
fi

# ── Step 8: Post-restore KV cache check ──────────────────────────────────────
echo "[T_CRIU3] Step 8: Post-restore KV cache check..."
START_MS=$(date +%s%3N)
POST_RESPONSE=$(curl -sf "http://localhost:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d @"${RESULTS_DIR}/payload.json")
END_MS=$(date +%s%3N)
POST_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")

POST_COLD_RATIO=$(python3 -c "print(round(${POST_TTFT_S} / ${COLD_TTFT_S}, 3))" 2>/dev/null || echo "0")
POST_WARM_RATIO=$(python3 -c "print(round(${POST_TTFT_S} / ${WARM_TTFT_S}, 3))" 2>/dev/null || echo "0")

echo "[T_CRIU3]   Post-restore TTFT: ${POST_TTFT_S}s"
echo "[T_CRIU3]   ratio vs cold: ${POST_COLD_RATIO}  (< 0.30 → preserved; > 0.70 → lost)"
echo "[T_CRIU3]   ratio vs warm: ${POST_WARM_RATIO}  (≈ 1.0 → preserved)"

echo "${POST_RESPONSE}" > "${RESULTS_DIR}/post_restore_response.json"
echo "post_restore_ttft_s=${POST_TTFT_S}" >> "${RESULTS_DIR}/timings.txt"
echo "post_cold_ratio=${POST_COLD_RATIO}"  >> "${RESULTS_DIR}/timings.txt"
echo "post_warm_ratio=${POST_WARM_RATIO}"  >> "${RESULTS_DIR}/timings.txt"

KV_VERDICT=$(python3 -c "
r = ${POST_COLD_RATIO}
print('KV_CACHE_PRESERVED' if r < 0.30 else ('KV_CACHE_LOST' if r > 0.70 else 'KV_CACHE_PARTIAL'))
")
echo "kv_verdict=${KV_VERDICT}" >> "${RESULTS_DIR}/timings.txt"
echo "[T_CRIU3]   KV verdict: ${KV_VERDICT}"

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    > "${RESULTS_DIR}/vram_post_restore.txt"

echo "RESTORE_OK" > "${RESULTS_DIR}/status.txt"

# ── Step 9: Write summary ─────────────────────────────────────────────────────
cat > "${RESULTS_DIR}/summary.md" <<EOMD
# T_CRIU3 Phase 2 — CRIU KV Cache Preservation — ${TIMESTAMP}

## Result
RESTORE_OK

| Metric | Value |
|--------|-------|
| Cold TTFT | ${COLD_TTFT_S}s |
| Warm TTFT | ${WARM_TTFT_S}s |
| Warm/cold ratio | ${CACHE_RATIO} |
| Checkpoint exit | 0 |
| Checkpoint time | ${CKPT_ELAPSED_S}s |
| Checkpoint size | ${CKPT_SIZE_GB} |
| Restore time | ${RESTORE_ELAPSED_S}s |
| Post-restore TTFT | ${POST_TTFT_S}s |
| Post-restore/cold ratio | ${POST_COLD_RATIO} |
| Post-restore/warm ratio | ${POST_WARM_RATIO} |
| KV verdict | ${KV_VERDICT} |

## Status
RESTORE_OK
EOMD

echo "[T_CRIU3] Results: ${RESULTS_DIR}/"
echo "[T_CRIU3] KV verdict: ${KV_VERDICT}"

# ── Step 10: Cleanup and redeploy production coder ────────────────────────────
echo "[T_CRIU3] Step 10: Cleanup and redeploy production coder..."
sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
sudo rm -rf /dev/shm/* || true
sleep 2

VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" \
    --gpu-mem-util 0.90 \
    --ctx 32768 \
    --kv-cache-dtype fp8 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice
