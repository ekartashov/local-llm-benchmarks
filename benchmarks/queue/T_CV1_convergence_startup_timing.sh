#!/usr/bin/env bash
# benchmarks/queue/T_CV1_convergence_startup_timing.sh
#
# T_CV1 — Convergence startup timing measurement.
#
# Question: How long does Convergence take to become ready from cold start?
#   - NVMe -> RAM -> GPU (attention layers)
#   - Measures both cold-cache (dropped caches) and warm-cache scenarios.
#
# Procedure:
#   1. Measure cold-cache startup (requires sudo to drop caches).
#   2. Measure warm-cache startup.
#
# PREREQUISITES:
#   - ik_llama.cpp pr-1288 built
#   - Model at ${MODEL_CACHE}/${CONVERGENCE_MODEL}
#   - Sudo access for drop_caches (optional, will skip if not available)
#
# OPTIONS:
#   --dry-run       Print commands, do not execute
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

run() { [[ "${DRY_RUN}" -eq 1 ]] && { echo "[dry-run] $*"; return; }; "$@"; }

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/T_CV1_convergence_startup_timing_${TIMESTAMP}"
IK_SERVER="${IK_LLAMA_BUILD_DIR:-/srv/ai/projects/ik_llama.cpp/build}/bin/llama-server"
MODEL_PATH="${MODEL_CACHE}/${CONVERGENCE_MODEL}"

# ── Validation ────────────────────────────────────────────────────────────────
[[ ! -x "${IK_SERVER}" ]] && {
    echo "ERROR: llama-server not found at ${IK_SERVER}" >&2; exit 1
}
[[ ! -f "${MODEL_PATH}" ]] && {
    echo "ERROR: Model not found: ${MODEL_PATH}" >&2; exit 1
}

echo "[T_CV1] Results dir: ${RESULTS_DIR}"
[[ "${DRY_RUN}" -eq 0 ]] && mkdir -p "${RESULTS_DIR}"

measure_startup() {
    local label="$1"
    local drop_caches="$2"
    
    echo "[T_CV1] Measuring ${label} startup..."
    
    if [[ "${drop_caches}" -eq 1 ]]; then
        if sudo -n true 2>/dev/null; then
            echo "[T_CV1] Dropping page caches..."
            sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
        else
            echo "[T_CV1] Sudo not available without password. Skipping drop_caches."
        fi
    fi

    local START_MS
    START_MS=$(date +%s%3N)
    
    # Start server in background
    # Note: Using production flags (ngl 0 for CPU-only)
    "${IK_SERVER}" \
        -m "${MODEL_PATH}" \
        -ngl 0 \
        --no-mmap \
        -b 4096 -ub 2048 \
        -t "$(nproc)" \
        -c 16384 \
        --host 127.0.0.1 --port 8002 \
        --jinja \
        > "${RESULTS_DIR}/server_${label}.log" 2>&1 &
    
    local SERVER_PID=$!
    
    # Poll health endpoint
    local READY_MS=0
    local TIMEOUT=300 # 5 minutes
    local ELAPSED=0
    
    while [[ ${ELAPSED} -lt ${TIMEOUT} ]]; do
        if curl -sf "http://127.0.0.1:8002/health" >/dev/null 2>&1; then
            READY_MS=$(date +%s%3N)
            break
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[T_CV1] ERROR: Server died during ${label} startup."
            cat "${RESULTS_DIR}/server_${label}.log"
            return 1
        fi
        sleep 1
        ELAPSED=$(( ELAPSED + 1 ))
    done
    
    # Kill server
    kill "${SERVER_PID}"
    wait "${SERVER_PID}" 2>/dev/null || true
    
    if [[ ${READY_MS} -eq 0 ]]; then
        echo "[T_CV1] ERROR: Timeout waiting for ${label} startup."
        return 1
    fi
    
    local DIFF=$(( READY_MS - START_MS ))
    echo "${label}_ms,${DIFF}" >> "${RESULTS_DIR}/timings.csv"
    echo "[T_CV1] ${label} startup: ${DIFF} ms"
}

if [[ "${DRY_RUN}" -eq 0 ]]; then
    echo "label,ms" > "${RESULTS_DIR}/timings.csv"
    
    # 1. Cold Cache
    measure_startup "cold_cache" 1
    
    # 2. Warm Cache
    measure_startup "warm_cache" 0
    
    # Generate metrics.json
    COLD_MS=$(grep "cold_cache" "${RESULTS_DIR}/timings.csv" | cut -d',' -f2)
    WARM_MS=$(grep "warm_cache" "${RESULTS_DIR}/timings.csv" | cut -d',' -f2)
    
    cat > "${RESULTS_DIR}/metrics.json" <<EOJSON
{
  "item_id": "T_CV1_convergence_startup_timing",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config": {
    "engine": "ikllamacpp",
    "model": "unsloth/Qwen3.5-397B-A17B-GGUF UD-IQ2_M",
    "ngl": 0,
    "no_mmap": true
  },
  "metrics": {
    "cold_cache_ms": ${COLD_MS},
    "warm_cache_ms": ${WARM_MS}
  },
  "verdict": "MEASURED",
  "notes": "Convergence startup time measured with production ngl=0 setting."
}
EOJSON

    cat > "${RESULTS_DIR}/summary.md" <<EOMD
# T_CV1 — Convergence Startup Timing

**Cold cache startup:** $(( COLD_MS / 1000 ))s
**Warm cache startup:** $(( WARM_MS / 1000 ))s

## Analysis

Convergence is intended to be RAM-resident but not necessarily always-on if startup is fast enough. 
With a warm cache startup of $(( WARM_MS / 1000 ))s, it is viable for manual escalation but may be slow for transparent routing.
EOMD

    echo "[T_CV1] Done. Results in ${RESULTS_DIR}/"
else
    echo "[dry-run] Would measure cold and warm startup for Convergence."
fi
