#!/usr/bin/env bash
# benchmarks/queue/22_qx_preload.sh
#
# BENCH_22 — QX_PRELOAD: Convergence GGUF page-cache pre-warm for fast CRIU restore
#
# Objective: Measure CRIU restore-to-interactive TTFT with and without page-cache 
# pre-warming of the 123GB GGUF model files.
#
# Following project standards:
# - Host-native execution (prevents Podman CDI/namespace conflicts)
# - sudo elevation for CRIU, cuda-checkpoint, and drop_caches
# - Results/logs owned by user
# - io_uring neutralized via LD_PRELOAD shim and UV_USE_IO_URING=0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# Tool Paths
CUDA_CHECKPOINT="/usr/local/bin/cuda-checkpoint"
CRIU="/usr/sbin/criu"
IK_LLAMA_DIR="/srv/ai/projects/ik_llama.cpp"
IK_BUILD="${IK_LLAMA_BUILD_DIR:-${IK_LLAMA_DIR}/build}"
IK_BIN="${IK_BUILD}/bin/llama-server"
IO_URING_SHIM="${REPO_ROOT}/benchmarks/queue/io_uring_shim.so"

# Configuration
GGUF_DIR="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M"
GGUF_FILES="${GGUF_DIR}"/Qwen3.5-397B-A17B-UD-IQ2_M-*.gguf
PORT=8002
CHECKPOINT_DIR="/srv/ai/checkpoints/convergence-qx-preload"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/BENCH_22_qx_preload_convergence_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}"
sudo rm -rf "${CHECKPOINT_DIR}" && sudo mkdir -p "${CHECKPOINT_DIR}"
sudo chown "${USER}:${USER}" "${CHECKPOINT_DIR}"

TEST_PROMPT="List the three laws of thermodynamics in one sentence each."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " BENCH_22: QX_PRELOAD CONVERGENCE GGUF"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[BENCH_22] GGUF Dir: ${GGUF_DIR}"
echo "[BENCH_22] Result Dir: ${RESULTS_DIR}"

# --- Step 0: Cleanup ---
# Cleanup and chown on exit
cleanup() {
    echo "[BENCH_22] Cleaning up..."
    sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    # Ensure logs/results are owned by user even on failure
    if [ -d "${RESULTS_DIR}" ]; then
        sudo chown -R "${USER}:${USER}" "${RESULTS_DIR}" || true
    fi
}
trap cleanup EXIT

# --- Step 1: Start llama-server on Host ---
echo "[BENCH_22] Starting llama-server on Host (mmap mode)..."
# We use python to launch to ensure a clean session for CRIU
python3 -c "
import subprocess, os
log = open('${RESULTS_DIR}/convergence_host.log', 'w')
env = os.environ.copy()
env['UV_USE_IO_URING'] = '0'
env['LD_PRELOAD'] = '${IO_URING_SHIM}'
# Set LD_LIBRARY_PATH for ik-llama.cpp shared libs
env['LD_LIBRARY_PATH'] = f'{os.environ.get(\"LD_LIBRARY_PATH\", \"\")}:${IK_BUILD}/src:${IK_BUILD}/ggml/src:${IK_BUILD}/examples/mtmd:/usr/local/cuda/targets/x86_64-linux/lib'

subprocess.Popen(
    ['${IK_BIN}',
     '-m', '${GGUF_DIR}/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf',
     '--port', '${PORT}',
     '--host', '0.0.0.0',
     '-ngl', '15',
     '--cpu-moe',
     '-fa', 'off',
     '-b', '4096', '-ub', '2048',
     '-t', '32',
     '-np', '1',
     '-c', '4096',
     '--jinja'],
    stdout=log, stderr=log, stdin=subprocess.DEVNULL,
    env=env, close_fds=True, start_new_session=True
)
"
sleep 5
MAIN_PID=$(pgrep -f "llama-server.*--port ${PORT}" | head -1)
if [ -z "${MAIN_PID}" ]; then
    echo "[BENCH_22] ERROR: llama-server failed to start. Check ${RESULTS_DIR}/convergence_host.log"
    exit 1
fi
echo "[BENCH_22] llama-server started with Host PID: ${MAIN_PID}"

# Wait for healthy
echo "[BENCH_22] Waiting for health check..."
READY=0
for i in $(seq 1 300); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[BENCH_22] HEALTH OK"
        READY=1
        break
    fi
    if ! kill -0 "${MAIN_PID}" 2>/dev/null; then
        echo "[BENCH_22] ERROR: llama-server process died."
        exit 1
    fi
    sleep 1
done

if [ "${READY}" -eq 0 ]; then
    echo "[BENCH_22] ERROR: Health check timeout."
    exit 1
fi

# Warm-up to fully populate page cache
echo "[BENCH_22] Warm-up inferences (populating page cache)..."
for W in 1 2; do
  curl -sf "http://localhost:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"convergence\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}" \
    > "${RESULTS_DIR}/warmup_${W}.json"
done

# Pre-checkpoint baseline
echo "[BENCH_22] Recording baseline inference..."
PRECHECK_RESPONSE=$(curl -sf "http://localhost:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"convergence\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
echo "${PRECHECK_RESPONSE}" > "${RESULTS_DIR}/pre_checkpoint_response.json"
PRE_TEXT=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/pre_checkpoint_response.json'))['choices'][0]['text'])")
echo "[BENCH_22] Pre-checkpoint text: ${PRE_TEXT}"

# --- Step 2: Checkpoint ---
echo "[BENCH_22] --- Checkpointing ---"
echo "[BENCH_22] Suspending GPU state..."
PIDS=$(ps -ef | grep "${MAIN_PID}" | grep -v grep | awk '{print $2}')
for p in ${PIDS}; do
    sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || echo "Warning: PID $p not a CUDA process"
done

echo "[BENCH_22] Performing CRIU Dump..."
CKPT_START_MS=$(date +%s%3N)
if ! sudo "${CRIU}" dump \
    --tree "${MAIN_PID}" \
    --images-dir "${CHECKPOINT_DIR}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --log-file "${RESULTS_DIR}/criu_dump.log" \
    --verbosity=3; then
    echo "[BENCH_22] ERROR: CRIU Dump failed. See ${RESULTS_DIR}/criu_dump.log"
    for p in ${PIDS}; do sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true; done
    exit 1
fi
sudo chown "${USER}:${USER}" "${RESULTS_DIR}/criu_dump.log"
CKPT_END_MS=$(date +%s%3N)
CKPT_ELAPSED_S=$(python3 -c "print(round(($CKPT_END_MS - $CKPT_START_MS) / 1000.0, 1))")
CKPT_SIZE_GB=$(du -sh "${CHECKPOINT_DIR}" 2>/dev/null | awk '{print $1}')

echo "[BENCH_22] Checkpoint time: ${CKPT_ELAPSED_S}s"
echo "[BENCH_22] Checkpoint size: ${CKPT_SIZE_GB}"
echo "checkpoint_elapsed_s=${CKPT_ELAPSED_S}" > "${RESULTS_DIR}/timings.txt"
echo "checkpoint_size=${CKPT_SIZE_GB}" >> "${RESULTS_DIR}/timings.txt"

# Store log file size to satisfy CRIU's consistency check during restore
LOG_FILE="${RESULTS_DIR}/convergence_host.log"
LOG_SIZE_AT_CKPT=$(stat -c%s "${LOG_FILE}")
echo "[BENCH_22] Log size at checkpoint: ${LOG_SIZE_AT_CKPT} bytes"

# Wait for PIDs to clear
echo "[BENCH_22] Waiting for PIDs to clear..."
sleep 2
for p in ${PIDS}; do
    if kill -0 "$p" 2>/dev/null; then
        sudo kill -9 "$p" || true
    fi
done

# --- Step 3: TEST A — Cold restore ---
echo "[BENCH_22] === TEST A: Cold restore (drop page cache) ==="

# Truncate log to checkpoint size to avoid "bad size" error
sudo truncate -s "${LOG_SIZE_AT_CKPT}" "${LOG_FILE}"

sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
echo "[BENCH_22] Page cache dropped."

RESTORE_A_START_MS=$(date +%s%3N)
if ! sudo "${CRIU}" restore \
    --images-dir "${CHECKPOINT_DIR}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --restore-detached \
    --log-file "${RESULTS_DIR}/criu_restore_A.log" \
    --verbosity=3; then
    echo "[BENCH_22] ERROR: CRIU Restore A failed. See ${RESULTS_DIR}/criu_restore_A.log"
    exit 1
fi
sudo chown "${USER}:${USER}" "${RESULTS_DIR}/criu_restore_A.log"

for p in ${PIDS}; do
    sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true
done

# Wait for healthy
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[BENCH_22] [restore A] HEALTH OK"
        break
    fi
    sleep 1
done

RESTORE_A_END_MS=$(date +%s%3N)
RESTORE_A_ELAPSED_S=$(python3 -c "print(round(($RESTORE_A_END_MS - $RESTORE_A_START_MS) / 1000.0, 2))")
echo "restore_A_elapsed_s=${RESTORE_A_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

echo "rep,condition,ttft_s,text_match" > "${RESULTS_DIR}/restore_reps.csv"
for REP in 1 2 3; do
  START_MS=$(date +%s%3N)
  RESPONSE=$(curl -sf "http://localhost:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"convergence\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
  END_MS=$(date +%s%3N)
  TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  POST_TEXT=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null)
  MATCH=$( [ "${PRE_TEXT}" = "${POST_TEXT}" ] && echo "identical" || echo "differs" )
  echo "A_cold,${REP},${TTFT_S},${MATCH}" >> "${RESULTS_DIR}/restore_reps.csv"
  echo "[TEST A rep ${REP}] TTFT=${TTFT_S}s text=${MATCH}"
done

# Stop for next test
sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
sleep 3

# --- Step 4: TEST B — Warm restore ---
echo "[BENCH_22] === TEST B: Warm restore (pre-warm GGUF files) ==="

# Save Test A logs before truncating
cp "${LOG_FILE}" "${RESULTS_DIR}/convergence_host_testA_full.log" || true
# Truncate log to checkpoint size
sudo truncate -s "${LOG_SIZE_AT_CKPT}" "${LOG_FILE}"

sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
echo "[BENCH_22] Page cache dropped (pre-warm starting point)."

# Pre-warm: cat all shards into /dev/null
PREWARM_START_MS=$(date +%s%3N)
echo "[BENCH_22] Pre-warming ${GGUF_DIR}/*.gguf into page cache..."
# Use cat on all shards
cat ${GGUF_FILES} > /dev/null
PREWARM_END_MS=$(date +%s%3N)
PREWARM_ELAPSED_S=$(python3 -c "print(round(($PREWARM_END_MS - $PREWARM_START_MS) / 1000.0, 1))")
echo "[BENCH_22] Pre-warm complete: ${PREWARM_ELAPSED_S}s"
echo "prewarm_elapsed_s=${PREWARM_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

# Restore immediately
RESTORE_B_START_MS=$(date +%s%3N)
if ! sudo "${CRIU}" restore \
    --images-dir "${CHECKPOINT_DIR}" \
    --shell-job \
    --tcp-established \
    --file-locks \
    --restore-detached \
    --log-file "${RESULTS_DIR}/criu_restore_B.log" \
    --verbosity=3; then
    echo "[BENCH_22] ERROR: CRIU Restore B failed. See ${RESULTS_DIR}/criu_restore_B.log"
    exit 1
fi
sudo chown "${USER}:${USER}" "${RESULTS_DIR}/criu_restore_B.log"

for p in ${PIDS}; do
    sudo "${CUDA_CHECKPOINT}" --toggle --pid "$p" || true
done

# Wait for healthy
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[BENCH_22] [restore B] HEALTH OK"
        break
    fi
    sleep 1
done

RESTORE_B_END_MS=$(date +%s%3N)
RESTORE_B_ELAPSED_S=$(python3 -c "print(round(($RESTORE_B_END_MS - $RESTORE_B_START_MS) / 1000.0, 2))")
echo "restore_B_elapsed_s=${RESTORE_B_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

for REP in 1 2 3; do
  START_MS=$(date +%s%3N)
  RESPONSE=$(curl -sf "http://localhost:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"convergence\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
  END_MS=$(date +%s%3N)
  TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  POST_TEXT=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null)
  MATCH=$( [ "${PRE_TEXT}" = "${POST_TEXT}" ] && echo "identical" || echo "differs" )
  echo "B_warm,${REP},${TTFT_S},${MATCH}" >> "${RESULTS_DIR}/restore_reps.csv"
  echo "[TEST B rep ${REP}] TTFT=${TTFT_S}s text=${MATCH}"
done

# --- Step 5: Production Restore ---

echo "[BENCH_22] Restoring production Convergence (container mode)..."
"${REPO_ROOT}/infra/scripts/deploy.sh" ikllamacpp convergence

echo "COMPLETE" > "${RESULTS_DIR}/status.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " BENCH_22 COMPLETE"
echo " Results in: ${RESULTS_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
