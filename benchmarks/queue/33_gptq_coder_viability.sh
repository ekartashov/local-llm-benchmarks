#!/bin/bash
# BENCH_33 — T_PQ3: GPTQ-Int4 Coder Viability (Host-Native vLLM 0.20.0 + cu130)

set -e

# Cleanup function to kill background processes on exit
cleanup() {
    echo "[cleanup] Shutting down..."
    [ -n "${TAIL_PID}" ] && kill "${TAIL_PID}" 2>/dev/null || true
    if [ -f "${RESULTS_DIR}/server.pid" ]; then
        SERVER_PID=$(cat "${RESULTS_DIR}/server.pid")
        [ -n "${SERVER_PID}" ] && kill "${SERVER_PID}" 2>/dev/null || true
        echo "[cleanup] vLLM server (PID ${SERVER_PID}) stopped"
    fi
}
trap cleanup EXIT INT TERM

echo "=== BENCH_33 START ==="

# Step 0 — Stop APEX coder
echo "[step0] Freeing port 8080..."
APEX_PID=$(lsof -ti:8080 2>/dev/null || true)
if [ -n "${APEX_PID}" ]; then
  kill "${APEX_PID}" && echo "[step0] APEX coder stopped (PID ${APEX_PID})"
  sleep 5
else
  echo "[step0] port 8080 already free"
fi

# Step 1 — Install vLLM host-native
echo "[step1] Setting up vLLM environment..."
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv virtualenv 3.12.7 vllm-host 2>/dev/null || echo "[step1] vllm-host already exists"
pyenv activate vllm-host
pip install --upgrade pip --quiet
pip install vllm>=0.20.0 --quiet
echo "[step1] vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"

# Step 2 — Download model
echo "[step2] Downloading model..."
export HF_HOME=/srv/ai/models
MODEL_REPO="groxaxo/Qwen3.6-35B-A3B-GPTQ-Pro-FOEM-4bit-g128"
echo "[step2] Attempting ${MODEL_REPO} download..."
pyenv exec hf download "${MODEL_REPO}" || { echo "FATAL: model download failed"; exit 1; }

# Resolve model path
MODEL_PATH=$(python -c "
import os; hf_home = os.environ.get('HF_HOME', '/srv/ai/models')
repo = '${MODEL_REPO}'.replace('/', '--').replace('--', '--models--', 1)
snapshots = os.path.join(hf_home, 'hub', 'models--' + repo.replace('models--', ''), 'snapshots')
import glob; snaps = sorted(glob.glob(snapshots + '/*/'))
print(snaps[-1].rstrip('/') if snaps else '')
" 2>/dev/null)
[ -n "${MODEL_PATH}" ] || { echo "FATAL: could not resolve model path"; exit 1; }
echo "[step2] Model path: ${MODEL_PATH}"

# Step 3 — Create results directory
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="/srv/ai/projects/local-llm-benchmarks/results/BENCH_33_gptq_coder_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
echo "[step3] Results dir: ${RESULTS_DIR}"

# Step 4 — Deploy vLLM coder
echo "[step4] Deploying vLLM..."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Lowering utilization to 0.80 and seqs to 8 for OOM safety
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name coder \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 8 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 8080 \
  >> "${RESULTS_DIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${RESULTS_DIR}/server.pid"

# Stream log to terminal
tail -f "${RESULTS_DIR}/server.log" | stdbuf -oL sed 's/\r//g; s/^/[vllm] /' &
TAIL_PID=$!

echo "[step4] Waiting for server health (check [vllm] logs above)..."
for i in $(seq 1 300); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "FATAL: vLLM server died. Check ${RESULTS_DIR}/server.log for errors."
    exit 1
  fi
  curl -sf http://localhost:8080/health 2>/dev/null && break
  sleep 1
  [ $((i % 30)) -eq 0 ] && echo "[step4] still waiting... ${i}s"
done

# Kill log streaming
kill "${TAIL_PID}" 2>/dev/null || true
TAIL_PID=""

curl -sf http://localhost:8080/health || { echo "FATAL: server did not start"; exit 1; }
echo "[step4] Server up"

# Step 5 — TPS Measurement N=1
echo "[step5] Measuring TPS N=1..."
PROMPT="Write a Python function that implements binary search on a sorted list. Include type hints, handle edge cases, and add a brief docstring."
for REP in 1 2 3; do
  START=$(date +%s%N)
  RESP=$(curl -sf http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"coder\",
      \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
      \"max_tokens\": 1024,
      \"temperature\": 0
    }")
  END=$(date +%s%N)
  ELAPSED_MS=$(( (END - START) / 1000000 ))
  TOKENS=$(echo "${RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])")
  TPS=$(python3 -c "print(round(${TOKENS} / (${ELAPSED_MS} / 1000.0), 1))")
  echo "[step5] rep${REP}: ${TPS} t/s"
  echo "${RESP}" > "${RESULTS_DIR}/tps_n1_rep${REP}.json"
done

# Step 6 — TPS Measurement N=4
echo "[step6] Measuring TPS N=4..."
for ROUND in 1 2 3; do
  START=$(date +%s%N)
  PIDS=()
  for i in 1 2 3 4; do
    curl -sf http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"coder\",
        \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
        \"max_tokens\": 512,
        \"temperature\": 0
      }" > "${RESULTS_DIR}/tps_n4_round${ROUND}_req${i}.json" &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  END=$(date +%s%N)
  ELAPSED_MS=$(( (END - START) / 1000000 ))
  TOTAL_TOKENS=0
  for i in 1 2 3 4; do
    T=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/tps_n4_round${ROUND}_req${i}.json'))['usage']['completion_tokens'])")
    TOTAL_TOKENS=$((TOTAL_TOKENS + T))
  done
  AGG_TPS=$(python3 -c "print(round(${TOTAL_TOKENS} / (${ELAPSED_MS} / 1000.0), 1))")
  echo "[step6] round${ROUND}: ${AGG_TPS} t/s aggregate"
done

# Step 7 — Tool-call probes
echo "[step7] Running tool-call probes..."
TOOLS='[{"type": "function", "function": {"name": "execute_code", "description": "Execute code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]'
PROMPTS=("Compute Fibonacci 10" "Check if 17 is prime" "Sort [3,1,4,1,5,9,2,6]" "Factorial 10" "Sieve up to 50")
PASS=0
for i in 1 2 3 4 5; do
  RESP=$(curl -sf http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"coder\",
      \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPTS[$((i-1))]}\"}],
      \"tools\": ${TOOLS},
      \"tool_choice\": \"auto\",
      \"max_tokens\": 2048,
      \"temperature\": 0
    }")
  echo "${RESP}" > "${RESULTS_DIR}/tool_call_${i}.json"
  HAS_TOOL=$(echo "${RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); tc=d['choices'][0]['message'].get('tool_calls',[]); print('YES' if tc else 'NO')")
  echo "[step7] probe${i}: ${HAS_TOOL}"
  [ "${HAS_TOOL}" = "YES" ] && PASS=$((PASS + 1))
done
echo "[step7] Pass rate: ${PASS}/5"

# Step 8 — th02 Quality probe
echo "[step8] Running th02 quality probe..."
TH02_PROMPT="Implement an Earliest Deadline First (EDF) scheduler in Python. Requirements: Task class with task_id, arrival_time, execution_time, deadline attributes. EDF scheduler that processes tasks selecting the one with earliest deadline. Calculate and return schedule order, average waiting time, missed deadline count, CPU utilization. Test with tasks: [(0,3,5),(1,2,4),(2,1,3),(3,4,8)] format: (arrival, exec, deadline). Verify: task (2,1,3) executes first when available due to deadline=3."
RESP=$(curl -sf http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"coder\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${TH02_PROMPT}\"}],
    \"max_tokens\": 16000,
    \"temperature\": 0
  }")
echo "${RESP}" > "${RESULTS_DIR}/th02.json"

# Step 9 — Cleanup and Restore
echo "[step9] Restoring APEX..."
# Cleanup already handled by trap but we do it explicitly here to restore APEX
[ -f "${RESULTS_DIR}/server.pid" ] && kill $(cat "${RESULTS_DIR}/server.pid") 2>/dev/null || true
sleep 5
APEX_MODEL="/srv/ai/models/hub/models--mudler--Qwen3.6-35B-A3B-APEX-GGUF/snapshots/$(ls /srv/ai/models/hub/models--mudler--Qwen3.6-35B-A3B-APEX-GGUF/snapshots/ | tail -1)/Qwen3.6-35B-A3B-APEX-I-Compact.gguf"
IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
CUDA_VISIBLE_DEVICES=0 GGML_CUDA_NO_PINNED=1 \
  "${IK_BIN}" \
  -m "${APEX_MODEL}" \
  -ngl 999 -t 32 -np 4 -c 32768 \
  --kv-type q8_0 --no-mmap --jinja \
  --port 8080 \
  >> /tmp/apex_restore.log 2>&1 &
echo "[step9] APEX restore launched."

echo "=== BENCH_33 COMPLETE ==="
echo "Results in: ${RESULTS_DIR}"
