#!/usr/bin/env bash
# benchmarks/queue/T_CV4_convergence_parallel_tps.sh
#
# T_CV4 — Convergence Parallel Throughput Audit.
#
# Question: Does MoE on CPU scale with batching, or does the expert-loading
# bottleneck cause throughput to collapse?

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/T_CV4_convergence_parallel_tps_${TIMESTAMP}"
IK_BUILD="${IK_LLAMA_BUILD_DIR:-/srv/ai/projects/ik_llama.cpp/build}"
PORT=8002
HEALTH_URL="http://127.0.0.1:${PORT}/health"
COMPLETION_URL="http://127.0.0.1:${PORT}/completion"

echo "[T_CV4] Results dir: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# 1. KILL EXISTING SERVERS
echo "[T_CV4] Ensuring port ${PORT} is free..."
pkill -f "llama-server.*${PORT}" || true
sleep 2

LOG="${RESULTS_DIR}/server.log"
touch "${LOG}"

# 2. START SERVER
echo "[T_CV4] Starting llama-server (R12 Golden Config)..."
"${IK_BUILD}/bin/llama-server" \
    -m "${MODEL_CACHE}/${CONVERGENCE_MODEL}" \
    -ngl 999 \
    --cpu-moe \
    --no-mmap \
    -b 4096 -ub 2048 \
    -t 32 \
    -c 32768 \
    -np 4 \
    --host 127.0.0.1 --port "${PORT}" \
    --jinja \
    > "${LOG}" 2>&1 &
SERVER_PID=$!

# 3. WAIT FOR READY
echo "[T_CV4] Waiting for server to become ready (streaming logs)..."
tail -n 0 -f "${LOG}" &
TAIL_PID=$!

for i in $(seq 1 600); do
    if curl -sf "${HEALTH_URL}" >/dev/null; then
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Server died. Check ${LOG}"
        kill "${TAIL_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

kill "${TAIL_PID}" 2>/dev/null || true
echo "[T_CV4] Server is READY."

# 4. MEASURE CONCURRENT TPS (C=4)
echo "[T_CV4] Starting 4 concurrent requests (128 tokens each)..."

# Python script to run parallel requests
cat > "${RESULTS_DIR}/parallel_bench.py" <<'EOF'
import json
import time
import threading
import requests
import sys

URL = "http://127.0.0.1:8002/completion"
CONCURRENCY = 4
PROMPT = "Explain the difference between dense and sparse MoE models."
GEN_LEN = 128

results = []

def send_request():
    payload = {
        "prompt": PROMPT,
        "n_predict": GEN_LEN,
        "stream": False,
        "temperature": 0.0
    }
    try:
        start = time.time()
        r = requests.post(URL, json=payload, timeout=300)
        end = time.time()
        d = r.json()
        timings = d.get('timings', {})
        results.append({
            "tokens": timings.get('predicted_n', GEN_LEN),
            "ms": (end - start) * 1000,
            "tps": timings.get('predicted_per_second', 0)
        })
    except Exception as e:
        print(f"Request failed: {e}")

threads = []
global_start = time.time()
for _ in range(CONCURRENCY):
    t = threading.Thread(target=send_request)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
global_end = time.time()

total_tokens = sum(r['tokens'] for r in results)
total_ms = (global_end - global_start) * 1000
agg_tps = (total_tokens / total_ms) * 1000 if total_ms > 0 else 0

print(json.dumps({
    "agg_tps": round(agg_tps, 2),
    "concurrency": CONCURRENCY,
    "total_tokens": total_tokens,
    "total_ms": round(total_ms, 2),
    "individual_results": results
}, indent=2))
EOF

# Run the parallel benchmark
python3 "${RESULTS_DIR}/parallel_bench.py" | tee "${RESULTS_DIR}/parallel_results.json"

AGG_TPS=$(grep '"agg_tps"' "${RESULTS_DIR}/parallel_results.json" | awk '{print $2}' | tr -d ',')

echo "--------------------------------"
echo "T_CV4 AGGREGATE TPS (C=4): ${AGG_TPS} tokens/sec"
echo "--------------------------------"

# 5. SAVE METRICS
cat > "${RESULTS_DIR}/metrics.json" <<EOJSON
{
  "item_id": "T_CV4_convergence_parallel_tps",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config": {
    "engine": "ikllamacpp-server",
    "concurrency": 4,
    "mode": "hybrid"
  },
  "metrics": {
    "aggregate_tps": ${AGG_TPS}
  }
}
EOJSON

# Cleanup
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

echo "[T_CV4] Done. Results in ${RESULTS_DIR}"
