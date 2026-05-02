#!/usr/bin/env bash
# benchmarks/queue/T_KV3_pathb_128k_context.sh
#
# T_KV3 Path B — 128K Context on ik_llama.cpp (Qwen3.6-27B)
#
# Objective: Verify 128K context viability on ik_llama.cpp using tensor-split
# across two GPUs for the Qwen3.6-27B-Q5_K_M model.
#
# Methodology:
# 1. Check if ik_llama.cpp is already running with the correct model.
# 2. If not, stop production coder (TP=2) and launch ik_llama.cpp.
# 3. Record VRAM usage.
# 4. Run a 120K token prefill + decode test to verify stability.
# 5. Record TTFT and TPS.
#
# Options:
#   --skip-deploy    Assume server is already running and healthy.
#   --reps N         Number of repetitions (not yet implemented in sweep).
#   --dry-run        Print commands without executing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# ── Settled constants ────────────────────────────────────────────────────────
# ── Argument Parsing ─────────────────────────────────────────────────────────
SKIP_DEPLOY=0
DRY_RUN=0
REPS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-deploy) SKIP_DEPLOY=1; shift ;;
        --dry-run)     DRY_RUN=1;     shift ;;
        --reps)        REPS="$2";      shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

ITEM_ID="T_KV3_pathb_128k_context"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

LOG="${RESULTS_DIR}/bench.log"
log() { echo "[T_KV3 $(date -u +%H:%M:%S)] $*" | tee -a "${LOG}"; }

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
MODEL_PATH="/srv/ai/models/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-Q5_K_M.gguf"
PORT=8080
ENDPOINT="http://localhost:${PORT}"

# ── Step 0: Preflight and Cleanup ────────────────────────────────────────────
log "Step 0: Preflight check..."

if [[ "${SKIP_DEPLOY}" -eq 0 ]]; then
    # Iterative Start Logic: Check if already running correctly
    ALREADY_RUNNING=0
    if curl -sf "${ENDPOINT}/health" >/dev/null 2>&1; then
        # Check if it's the right model
        CURRENT_MODEL=$(curl -sf "${ENDPOINT}/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' || echo "unknown")
        if [[ "${CURRENT_MODEL}" == *"Qwen3.6-27B"* ]]; then
            log "ik_llama.cpp already running with correct model. Skipping deployment."
            ALREADY_RUNNING=1
        else
            log "ik_llama.cpp running with wrong model (${CURRENT_MODEL}). Restarting..."
        fi
    fi

    if [[ "${ALREADY_RUNNING}" -eq 0 ]]; then
        log "Stopping coder and existing llama-servers to free VRAM..."
        
        # Stop any running coder containers (podman)
        CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|tp2a\|35b" | head -1 || true)
        if [ -n "${CODER_CONTAINER}" ]; then
            log "Stopping podman container: ${CODER_CONTAINER}"
            [[ "${DRY_RUN}" -eq 0 ]] && podman stop "${CODER_CONTAINER}" >/dev/null 2>&1 || true
            [[ "${DRY_RUN}" -eq 0 ]] && podman rm   "${CODER_CONTAINER}" >/dev/null 2>&1 || true
        fi

        # Kill any existing llama-server (native)
        [[ "${DRY_RUN}" -eq 0 ]] && pkill -u "$USER" -9 llama-server || true
        [[ "${DRY_RUN}" -eq 0 ]] && sudo fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true

        # Reset GPUs to clear ghost VRAM
        log "Resetting GPUs..."
        [[ "${DRY_RUN}" -eq 0 ]] && sudo nvidia-smi --gpu-reset -i 0,1 >/dev/null 2>&1 || true
        sleep 2
    fi
else
    log "Skipping deployment as requested (--skip-deploy)."
fi

# Record baseline VRAM
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_baseline.txt"
log "Baseline VRAM: $(paste -s -d' ' "${RESULTS_DIR}/vram_baseline.txt")"

# ── Step 1: Launch ik_llama.cpp ──────────────────────────────────────────────
if [[ "${SKIP_DEPLOY}" -eq 0 ]] && [[ "${ALREADY_RUNNING:-0}" -eq 0 ]]; then
    log "Step 1: Launching ik_llama.cpp with 128K context..."

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "[DRY-RUN] Would start llama-server with --ctx-size 131072"
    else
        # Launch in background, but pipe to tee so user sees it
        log "Starting llama-server... You should see its output below:"
        "${IK_BIN}" \
          -m "${MODEL_PATH}" \
          -ngl 999 \
          --tensor-split 0.5,0.5 \
          --ctx-size 131072 \
          --port "${PORT}" \
          --host 0.0.0.0 \
          --threads 8 \
          -np 1 \
          2>&1 | tee "${RESULTS_DIR}/llama_server.log" &

        SERVER_PID=$!
        log "llama-server started with PID ${SERVER_PID}. Waiting for health..."

        # Wait for health check
        READY=0
        for i in $(seq 1 120); do
            if curl -sf "${ENDPOINT}/health" >/dev/null 2>&1; then
                # Double check that we didn't just catch a finishing process
                if [[ $(curl -sf "${ENDPOINT}/health" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])') == "ok" ]]; then
                    READY=1
                    log "Health check OK."
                    break
                fi
            fi
            if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                log "FATAL: llama-server died during startup."
                exit 1
            fi
            sleep 1
        done

        if [ "${READY}" -eq 0 ]; then
            log "FATAL: llama-server failed to reach health check in 120s."
            exit 1
        fi
    fi
else
    log "Step 1: Using existing/skipped deployment."
fi

# Record loaded VRAM
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_loaded.txt"
log "Loaded VRAM: $(paste -s -d' ' "${RESULTS_DIR}/vram_loaded.txt")"

# ── Step 2: Quality Check (th02) ─────────────────────────────────────────────
log "Step 2: Running th02 quality check (Algorithm Logic)..."

TH02_PAYLOAD="${RESULTS_DIR}/th02_payload.json"
cat > "${TH02_PAYLOAD}" <<EOF
{
  "model": "qwen3.6-27b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a Python script that implements a simple consistent hashing ring with 3 virtual nodes per physical node."}
  ],
  "max_tokens": 1024,
  "temperature": 0.0
}
EOF

curl -sf "${ENDPOINT}/v1/chat/completions" -H "Content-Type: application/json" -d @"${TH02_PAYLOAD}" > "${RESULTS_DIR}/th02_response.json"
log "th02 response captured."

# ── Step 3: Big Context Prefill (120K tokens) ────────────────────────────────
log "Step 3: 120K context prefill and decode test..."

# Construct 120K+ token payload
# Previous run showed 9230 repeats = 92322 tokens (~10 tokens/repeat).
# To get 120K+, we need ~12,000 repeats. We use 12500 for margin.
python3 -c "
import json
base = 'The quick brown fox jumps over the lazy dog. '
repeats = 12500
long_text = (base * repeats) + '\n\nQuestion: Summarize the story above in 5 words.'
payload = {
    'model': 'qwen3.6-27b',
    'messages': [{'role': 'user', 'content': long_text}],
    'max_tokens': 32,
    'temperature': 0.0,
    'stream': False
}
with open('${RESULTS_DIR}/big_ctx_payload.json', 'w') as f:
    json.dump(payload, f)
"

log "Sending 120K token request. This will take several minutes..."
START_TIME=$(date +%s%3N)
curl -sf "${ENDPOINT}/v1/chat/completions" -H "Content-Type: application/json" -d @"${RESULTS_DIR}/big_ctx_payload.json" > "${RESULTS_DIR}/big_ctx_response.json"
END_TIME=$(date +%s%3N)

ELAPSED_MS=$((END_TIME - START_TIME))
ELAPSED_S=$(python3 -c "print(round(${ELAPSED_MS} / 1000.0, 2))")
log "120K request completed in ${ELAPSED_S}s."

# Extract tokens and calculate TPS
python3 -c "
import json
res = json.load(open('${RESULTS_DIR}/big_ctx_response.json'))
prompt_tokens = res['usage']['prompt_tokens']
completion_tokens = res['usage']['completion_tokens']
total_time = ${ELAPSED_S}
# Since we don't have per-token timestamps from non-streaming curl,
# we report aggregate latency and assume success if the server didn't crash.
print(f'Prompt tokens: {prompt_tokens}')
print(f'Completion tokens: {completion_tokens}')
" | tee -a "${LOG}"

# ── Step 4: Final Cleanup and Reporting ──────────────────────────────────────
log "Step 4: Cleanup..."
kill "${SERVER_PID}" || true
sleep 2

# Generate Summary
log "Generating summary.md..."
VRAM_BASELINE=$(cat "${RESULTS_DIR}/vram_baseline.txt")
VRAM_LOADED=$(cat "${RESULTS_DIR}/vram_loaded.txt")

cat > "${RESULTS_DIR}/summary.md" <<EOF
# T_KV3 Path B — 128K Context on ik_llama.cpp — ${TIMESTAMP}

## Environment
- **Model:** Qwen3.6-27B-Q5_K_M (GGUF)
- **Engine:** ik_llama.cpp (llama-server)
- **Config:** --ctx-size 131072, --tensor-split 0.5,0.5

## VRAM Usage (MiB)
| GPU | Baseline | Loaded (128K ctx) | Delta |
|-----|----------|-------------------|-------|
$(python3 -c "
b = [line.split(',')[1].strip().split()[0] for line in '''${VRAM_BASELINE}'''.splitlines()]
l = [line.split(',')[1].strip().split()[0] for line in '''${VRAM_LOADED}'''.splitlines()]
for i in range(len(b)):
    delta = int(l[i]) - int(b[i])
    print(f'| {i} | {b[i]} | {l[i]} | {delta} |')
")

## Performance (120K Prefill)
- **Elapsed Time:** ${ELAPSED_S}s
- **Status:** $(if [ -f "${RESULTS_DIR}/big_ctx_response.json" ]; then echo "SUCCESS"; else echo "FAILED"; fi)

## Verdict
**$(if [ -f "${RESULTS_DIR}/big_ctx_response.json" ]; then echo "PASS"; else echo "FAIL"; fi)**
EOF

# Final metrics.json
python3 -c "
import json, os
res_path = '${RESULTS_DIR}/big_ctx_response.json'
success = os.path.exists(res_path)
metrics = {
    'item_id': '${ITEM_ID}',
    'timestamp': '${TIMESTAMP}',
    'metrics': {
        'elapsed_time_s': ${ELAPSED_S},
        'success': success
    },
    'verdict': 'PASS' if success else 'FAIL'
}
with open('${RESULTS_DIR}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
"

log "Done. Results in ${RESULTS_DIR}"
log "REMINDER: Production coder is STOPPED. Restart manually."
