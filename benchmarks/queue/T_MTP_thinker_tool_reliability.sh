#!/usr/bin/env bash
# T_MTP_thinker_tool_reliability.sh
# Standalone benchmark to verify tool-calling reliability on the Thinker model
# under MTP n=3 speculation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ITEM_ID="T_MTP_thinker_tool_check"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

LOG="${RESULTS_DIR}/bench.log"
log() { echo "[T_MTP_TOOL $(date -u +%H:%M:%S)] $*" | tee -a "${LOG}"; }

MODEL="rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm"
ENDPOINT="http://localhost:30001/v1"

log "Starting Thinker Tool Reliability Benchmark..."

# --- Step 1: Deploy ---
log "Deploying PrismaQuant + MTP n=3..."
EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1 || true)
if [ -n "${EXISTING}" ]; then
    log "Stopping existing container: ${EXISTING}"
    podman stop "${EXISTING}" && podman rm "${EXISTING}"
    sleep 3
fi

VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  2>&1 | tee "${RESULTS_DIR}/deploy.log"

if [ $? -ne 0 ]; then
    log "FATAL: Deployment failed."
    exit 1
fi
log "MTP n=3 READY."

# --- Step 2: Tool Probes ---
log "Running 5 tool-call reliability probes..."

PROBE_RESULTS="${RESULTS_DIR}/toolcall_check.txt"
PASS_COUNT=0
TOTAL_PROBES=5

for i in $(seq 1 ${TOTAL_PROBES}); do
    log "  Sending probe $i..."
    
    RESPONSE=$(curl -s "${ENDPOINT}/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "'"${MODEL}"'",
        "messages": [
          {"role": "user", "content": "What is the weather in Tokyo?"}
        ],
        "tools": [
          {
            "type": "function",
            "function": {
              "name": "get_weather",
              "description": "Get the current weather",
              "parameters": {
                "type": "object",
                "properties": {
                  "location": {"type": "string"}
                },
                "required": ["location"]
              }
            }
          }
        ],
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0
      }')
    
    VALID=$(echo "${RESPONSE}" | python3 -c "
import sys, json
ok = False
try:
    r = json.load(sys.stdin)
    msg = r['choices'][0]['message']
    if msg.get('tool_calls'):
        tc = msg['tool_calls'][0]
        if tc['function']['name'] == 'get_weather':
            ok = True
except Exception:
    pass

if ok:
    print('OK')
    sys.exit(0)
else:
    print('FAIL')
    sys.exit(1)
" || echo "FAIL")

    echo "Probe $i: ${VALID}" >> "${PROBE_RESULTS}"
    if [ "${VALID}" == "OK" ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
    fi
    sleep 1
done

log "Tool-call results: ${PASS_COUNT}/${TOTAL_PROBES} PASS"

# --- Step 3: Cleanup ---
log "Restoring production thinker (No MTP)..."
EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1 || true)
[ -n "${EXISTING}" ] && podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 3
VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' >/dev/null 2>&1 &

log "Done. Results in ${RESULTS_DIR}"
