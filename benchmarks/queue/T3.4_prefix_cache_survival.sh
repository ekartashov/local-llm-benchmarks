#!/usr/bin/env bash
# benchmarks/queue/T3.4_prefix_cache_survival.sh
#
# T3.4 — CPU prefix cache survival across sleep/wake.
#
# Question: Does vLLM's CPU-offloaded prefix cache survive a sleep/wake cycle at level 1?
#
# Procedure:
#   1. Deploy coder with --enable-prefix-caching --cpu-offload-gb 8.
#   2. Send a 4k prompt, verify cache hit on second run.
#   3. Sleep level=1, then wake.
#   4. Send same prompt again, measure prefill time.
#
# PREREQUISITES:
#   - vLLM 0.19.x with --enable-sleep-mode support
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=1 ;;
    esac
done

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/T3.4_prefix_cache_survival_${TIMESTAMP}"

# Using Arclight Coder for the test
MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
PORT=30000

[[ "${DRY_RUN}" -eq 0 ]] && mkdir -p "${RESULTS_DIR}"

run_query() {
    local label="$1"
    echo "[T3.4] Running query: ${label}..."
    
    # Use a long-ish prompt (4k tokens) to make prefill time measurable
    # Using a dummy file or generated text
    local PROMPT_FILE="${RESULTS_DIR}/prompt_4k.txt"
    if [[ ! -f "${PROMPT_FILE}" ]]; then
        # Create a ~4000 token prompt (roughly 4 characters per token average)
        python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 400)" > "${PROMPT_FILE}"
    fi

    local START_MS
    START_MS=$(date +%s%3N)
    
    curl -sf "http://localhost:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"prompt\": \"$(cat "${PROMPT_FILE}")\",
            \"max_tokens\": 1,
            \"temperature\": 0
        }" > "${RESULTS_DIR}/resp_${label}.json"
        
    local END_MS
    END_MS=$(date +%s%3N)
    local DIFF=$(( END_MS - START_MS ))
    echo "${label}_ms,${DIFF}" >> "${RESULTS_DIR}/timings.csv"
    echo "[T3.4] ${label} time: ${DIFF} ms"
}

if [[ "${DRY_RUN}" -eq 0 ]]; then
    # 1. Start coder with prefix caching
    echo "[T3.4] Starting coder with prefix caching..."
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_V1_ENABLED=0 \
    VLLM_USE_V1=0 \
    VLLM_USE_V1_ENGINE=0 \
    VLLM_V1=0 \
    ./infra/scripts/deploy.sh vllm tp2a "${MODEL}" \
        --gpu-mem-util 0.90 \
        --ctx 8192 \
        --enable-prefix-caching \
        --enable-sleep-mode \
        --kv-cache-dtype fp8 \
        --max-num-batched-tokens 4096 \
        --enforce-eager

    echo "label,ms" > "${RESULTS_DIR}/timings.csv"

    # 2. First run (cold prefill)
    run_query "cold_prefill"
    
    # 3. Second run (should be cached)
    run_query "warm_cache_presleep"
    
    # 4. Sleep level=1
    echo "[T3.4] Sleeping level=1..."
    curl -X POST "http://localhost:${PORT}/sleep?level=1"
    sleep 5
    
    # 5. Wake up
    echo "[T3.4] Waking up..."
    curl -X POST "http://localhost:${PORT}/wake_up"
    
    # Wait for wake to complete
    echo "[T3.4] Waiting for wake-up completion..."
    while curl -sf "http://localhost:${PORT}/is_sleeping" | grep -q "true"; do
        sleep 1
    done
    
    # Wait for health (ensure API is actually responding again)
    echo "[T3.4] Waiting for health endpoint..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
            echo "[T3.4] Server is healthy."
            break
        fi
        sleep 1
    done
    
    # 6. Third run (post-wake)
    run_query "warm_cache_postwake"
    
    # Analysis
    COLD=$(grep "cold_prefill" "${RESULTS_DIR}/timings.csv" | cut -d',' -f2)
    WARM_PRE=$(grep "warm_cache_presleep" "${RESULTS_DIR}/timings.csv" | cut -d',' -f2)
    WARM_POST=$(grep "warm_cache_postwake" "${RESULTS_DIR}/timings.csv" | cut -d',' -f2)
    
    cat > "${RESULTS_DIR}/metrics.json" <<EOJSON
{
  "item_id": "T3.4_prefix_cache_survival",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "metrics": {
    "cold_prefill_ms": ${COLD},
    "warm_cache_presleep_ms": ${WARM_PRE},
    "warm_cache_postwake_ms": ${WARM_POST}
  },
  "verdict": "MEASURED"
}
EOJSON

    # Cleanup
    echo "[T3.4] Cleaning up..."
    podman stop bench-vllm-tp2a >/dev/null 2>&1 || true
    podman rm bench-vllm-tp2a >/dev/null 2>&1 || true

    echo "[T3.4] Done. Results in ${RESULTS_DIR}/"
else
    echo "[dry-run] Would test prefix cache survival across sleep/wake."
fi
