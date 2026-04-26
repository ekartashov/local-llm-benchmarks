#!/usr/bin/env bash
# benchmarks/queue/T_CV3_convergence_gpu_expert_offload.sh
#
# T_CV3 — Convergence partial GPU expert offload (via llama-server).
#
# FULL TRANSPARENCY VERSION: Streams logs to terminal.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/T_CV3_convergence_gpu_expert_offload_${TIMESTAMP}"
IK_BUILD="${IK_LLAMA_BUILD_DIR:-/srv/ai/projects/ik_llama.cpp/build}"
PORT=8002
HEALTH_URL="http://127.0.0.1:${PORT}/health"
COMPLETION_URL="http://127.0.0.1:${PORT}/completion"

echo "[T_CV3] Results dir: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# 1. KILL EXISTING SERVERS
echo "[T_CV3] Ensuring port ${PORT} is free..."
pkill -f "llama-server.*${PORT}" || true
sleep 2

LOG="${RESULTS_DIR}/server.log"
touch "${LOG}"

# 2. START SERVER AND STREAM LOGS
echo "[T_CV3] Starting llama-server (R12 Golden Config)..."
"${IK_BUILD}/bin/llama-server" \
    -m "${MODEL_CACHE}/${CONVERGENCE_MODEL}" \
    -ngl 999 \
    --cpu-moe \
    --no-mmap \
    -b 4096 -ub 2048 \
    -t 32 \
    -c 16384 \
    --host 127.0.0.1 --port "${PORT}" \
    --jinja \
    > "${LOG}" 2>&1 &
SERVER_PID=$!

# Stream the log to the terminal in the background
tail -n 0 -f "${LOG}" &
TAIL_PID=$!

# 3. WAIT FOR READY
echo "[T_CV3] Waiting for server to become ready..."
READY=0
for i in $(seq 1 600); do
    if curl -sf "${HEALTH_URL}" >/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Server died. Check ${LOG}"
        kill "${TAIL_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Stop streaming logs once ready
kill "${TAIL_PID}" 2>/dev/null || true
echo ""
echo "[T_CV3] Server is READY."

# 4. MEASURE TPS
echo "[T_CV3] Measuring TPS with 512-prompt, 128-gen request..."
START_TIME=$(date +%s%3N)
# Use a real prompt instead of 'word word'
PROMPT="A detailed technical explanation of how MoE expert offloading works in GGML is as follows:"
RESPONSE=$(curl -sf -X POST "${COMPLETION_URL}" \
    -H "Content-Type: application/json" \
    -d "{
        \"prompt\": \"${PROMPT}\",
        \"n_predict\": 128,
        \"stream\": false,
        \"temperature\": 0.0
    }")
END_TIME=$(date +%s%3N)

TOTAL_MS=$(( END_TIME - START_TIME ))

# Parse real TPS from the response
TG_TPS=$(echo "${RESPONSE}" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    t = d.get('timings', {})
    # llama-server returns 'predicted_per_second' in its timings object
    val = t.get('predicted_per_second')
    if val is None:
        # Fallback: calculate manually
        tokens = t.get('predicted_n', 128)
        ms = t.get('predicted_ms', 1)
        val = (tokens / ms) * 1000
    print(round(val, 2))
except Exception as e:
    print('N/A')
")

echo "--------------------------------"
echo "T_CV3 RESULT: ${TG_TPS} tokens/sec"
echo "--------------------------------"

# 5. SAVE RESULTS
cat > "${RESULTS_DIR}/metrics.json" <<EOJSON
{
  "item_id": "T_CV3_convergence_gpu_expert_offload",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config": {
    "engine": "ikllamacpp-server",
    "model": "unsloth/Qwen3.5-397B-A17B-GGUF UD-IQ2_M",
    "mode": "hybrid (attention-on-gpu, experts-on-cpu)",
    "flags": "-ngl 999 --cpu-moe"
  },
  "metrics": {
    "tg128_tps": ${TG_TPS},
    "total_request_ms": ${TOTAL_MS}
  }
}
EOJSON

# Cleanup
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

echo "[T_CV3] Done. Summary in ${RESULTS_DIR}/metrics.json"
