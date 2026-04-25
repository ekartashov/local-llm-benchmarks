#!/usr/bin/env bash
# benchmarks/queue/T_CV1_convergence_startup_timing.sh
#
# T_CV1 — Convergence startup timing.
#
# Measures how long Qwen3.5-397B UD-IQ2_M takes to become ready from:
#   (a) cold start: page cache explicitly dropped (NVMe → RAM → serve)
#   (b) warm start: page cache already populated from prior run (RAM → serve)
#
# QUESTION: Is cold-start-on-demand acceptable (<90s) or is always-resident required?
#   < 30s  → cold-start on demand is fine
#   30–90s → acceptable for explicit escalation, not transparent routing
#   > 90s  → always-resident strongly preferred
#
# Also records baseline generation speed (tg128 via single llama-bench rep) so
# T_CV2 thread sweep has a known -ngl 0 TPS baseline.
#
# PREREQUISITES:
#   - local-ik-llama:runtime image built (podman build -f infra/Containerfile.ik-llama-runtime -t local-ik-llama:runtime .)
#   - infra/scripts/build-ik-llama.sh completed successfully
#   - Model at ${MODEL_CACHE}/${CONVERGENCE_MODEL}
#   - Container bench-ikllamacpp-convergence not already running
#
# OPTIONS:
#   --runs N        Number of cold-start reps (default: 3)
#   --skip-drop     Skip page cache drop (only warm starts)
#   --dry-run       Print commands, do not execute

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

RUNS=3
SKIP_DROP=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --runs)        shift; RUNS="${1:-3}" ;;
        --skip-drop)   SKIP_DROP=1 ;;
        --dry-run)     DRY_RUN=1 ;;
    esac
done

run() { [[ "${DRY_RUN}" -eq 1 ]] && { echo "[dry-run] $*"; return; }; "$@"; }

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/T_CV1_convergence_startup_timing_${TIMESTAMP}"
CONTAINER_NAME="bench-ikllamacpp-convergence"
IK_BUILD="${IK_LLAMA_BUILD_DIR:-/srv/ai/projects/ik_llama.cpp/build}"
HEALTH_URL="http://localhost:${PORT_CONVERGENCE}/health"
BENCH_URL="http://localhost:${PORT_CONVERGENCE}/v1"

# ── Validation ────────────────────────────────────────────────────────────────
[[ ! -x "${IK_BUILD}/bin/llama-server" ]] && {
    echo "ERROR: llama-server not found at ${IK_BUILD}/bin/llama-server" >&2
    echo "  Run: ./infra/scripts/build-ik-llama.sh" >&2; exit 1
}
[[ ! -f "${MODEL_CACHE}/${CONVERGENCE_MODEL}" ]] && {
    echo "ERROR: Model not found: ${MODEL_CACHE}/${CONVERGENCE_MODEL}" >&2; exit 1
}

echo "[T_CV1] Results dir: ${RESULTS_DIR}"
[[ "${DRY_RUN}" -eq 0 ]] && mkdir -p "${RESULTS_DIR}"

# ── Helper: start container and time health ───────────────────────────────────
startup_once() {
    local rep="$1"
    local mode="$2"   # "cold" or "warm"

    echo "[T_CV1] Rep ${rep} (${mode}): starting container..."
    local T_START_MS
    T_START_MS=$(date +%s%3N)

    podman run -d \
        --name "${CONTAINER_NAME}" \
        -v "${IK_BUILD}:/app/build:ro,z" \
        -v "${MODEL_CACHE}:/models:ro,z" \
        -p "${PORT_CONVERGENCE}:8000" \
        --shm-size=4g \
        --restart=no \
        --entrypoint "/app/build/bin/llama-server" \
        local-ik-llama:runtime \
        --model "/models/${CONVERGENCE_MODEL}" \
        --port 8000 \
        --host 0.0.0.0 \
        -ngl 0 \
        --no-mmap \
        -b 4096 -ub 2048 \
        -t "$(nproc)" \
        -c 16384 \
        --jinja

    echo "[T_CV1] Rep ${rep}: polling ${HEALTH_URL} every 500ms..."
    local T_READY_MS
    while true; do
        if curl -sf --max-time 2 "${HEALTH_URL}" &>/dev/null; then
            T_READY_MS=$(date +%s%3N)
            break
        fi
        sleep 0.5
        # Safety: abort after 600s
        local _elapsed=$(( $(date +%s%3N) - T_START_MS ))
        if [[ "${_elapsed}" -gt 600000 ]]; then
            echo "[T_CV1] ERROR: health never came up in 600s. Container logs:" >&2
            podman logs --tail 30 "${CONTAINER_NAME}" >&2 || true
            podman stop "${CONTAINER_NAME}" 2>/dev/null || true
            podman rm   "${CONTAINER_NAME}" 2>/dev/null || true
            exit 1
        fi
    done

    local STARTUP_MS=$(( T_READY_MS - T_START_MS ))
    echo "[T_CV1] Rep ${rep} (${mode}): ready in ${STARTUP_MS} ms ($(( STARTUP_MS / 1000 ))s)"
    echo "${STARTUP_MS}"
}

stop_container() {
    podman stop "${CONTAINER_NAME}" 2>/dev/null || true
    podman rm   "${CONTAINER_NAME}" 2>/dev/null || true
}

# ── Ensure container is not already running ──────────────────────────────────
if podman container exists "${CONTAINER_NAME}" 2>/dev/null; then
    echo "[T_CV1] Stopping existing ${CONTAINER_NAME}..."
    run stop_container
fi

# ── Cold-start reps ───────────────────────────────────────────────────────────
declare -a COLD_MS=()
for i in $(seq 1 "${RUNS}"); do
    if [[ "${SKIP_DROP}" -eq 0 ]]; then
        echo "[T_CV1] Dropping page cache (requires sudo)..."
        run sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
    fi
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        ms=$(startup_once "${i}" "cold")
        COLD_MS+=("${ms}")
    else
        echo "[dry-run] startup_once ${i} cold"
    fi
    run stop_container
    sleep 2
done

# ── Warm-start rep (no cache drop; cache populated by last cold run) ─────────
echo "[T_CV1] Warm-start rep (no cache drop)..."
WARM_MS=0
if [[ "${DRY_RUN}" -eq 0 ]]; then
    ms=$(startup_once "warm" "warm")
    WARM_MS="${ms}"
fi
run stop_container

# ── Quick TPS baseline (ngl=0, 1 rep, 128 tokens) ────────────────────────────
echo "[T_CV1] Measuring TPS baseline (ngl=0, 1 rep, t=128)..."
TPS_JSON="${RESULTS_DIR}/tps_baseline_ngl0.json"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    podman run --rm \
        --name ik-llama-bench-cv1 \
        --userns=keep-id \
        -v "${IK_BUILD}:/app/build:ro,z" \
        -v "${MODEL_CACHE}:/models:ro,z" \
        --entrypoint "/app/build/bin/llama-bench" \
        local-ik-llama:runtime \
        -m "/models/${CONVERGENCE_MODEL}" \
        -ngl 0 \
        --no-mmap \
        -b 4096 -ub 2048 \
        -t "$(nproc)" \
        -p 512 -n 128 \
        -r 1 \
        --output json \
        > "${TPS_JSON}" 2>&1
    echo "[T_CV1] llama-bench output → ${TPS_JSON}"
else
    echo "[dry-run] podman run llama-bench → ${TPS_JSON}"
fi

# ── Compute median of cold starts ─────────────────────────────────────────────
median_of() {
    local -n _arr=$1
    local sorted
    IFS=$'\n' sorted=($(sort -n <<<"${_arr[*]}")); unset IFS
    local mid=$(( ${#sorted[@]} / 2 ))
    echo "${sorted[$mid]}"
}

if [[ "${DRY_RUN}" -eq 0 ]] && [[ ${#COLD_MS[@]} -gt 0 ]]; then
    COLD_MEDIAN_MS=$(median_of COLD_MS)
    COLD_MEDIAN_S=$(echo "scale=1; ${COLD_MEDIAN_MS}/1000" | bc)
    WARM_S=$(echo "scale=1; ${WARM_MS}/1000" | bc)

    # Verdict
    if   [[ "${COLD_MEDIAN_MS}" -lt 30000 ]]; then VERDICT="cold-start-on-demand-ok"
    elif [[ "${COLD_MEDIAN_MS}" -lt 90000 ]]; then VERDICT="escalation-acceptable"
    else                                            VERDICT="always-resident-required"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " T_CV1 Results"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Cold start (median of ${RUNS}): ${COLD_MEDIAN_S}s"
    printf " Cold start reps: %s ms\n" "${COLD_MS[@]}"
    echo " Warm start:                  ${WARM_S}s"
    echo " Verdict: ${VERDICT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Write metrics.json
    cat > "${RESULTS_DIR}/metrics.json" <<EOJSON
{
  "item_id": "T_CV1_convergence_startup_timing",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config": {
    "engine": "ikllamacpp",
    "model": "unsloth/Qwen3.5-397B-A17B-GGUF UD-IQ2_M",
    "quantization": "IQ2_M",
    "placement": "convergence (cpu-only, ngl=0)",
    "context_length": 16384,
    "no_mmap": true,
    "threads": "$(nproc)"
  },
  "metrics": {
    "cold_start_ms_reps": [$(IFS=,; echo "${COLD_MS[*]}")],
    "cold_start_ms_median": ${COLD_MEDIAN_MS},
    "cold_start_s_median": ${COLD_MEDIAN_S},
    "warm_start_ms": ${WARM_MS},
    "warm_start_s": ${WARM_S}
  },
  "verdict": "${VERDICT}",
  "notes": "ngl=0 (CPU-only, no GPU). Cold start = NVMe → RAM via --no-mmap. Warm start = page-cache already hot."
}
EOJSON

    # Write summary.md
    cat > "${RESULTS_DIR}/summary.md" <<EOMD
# T_CV1 — Convergence Startup Timing

**Timestamp:** ${TIMESTAMP}
**Config:** ik_llama.cpp pr-1288 · Qwen3.5-397B UD-IQ2_M · ngl=0 (CPU-only) · --no-mmap · t=$(nproc)

## Results

| Mode | Startup time |
|------|-------------|
| Cold start median (${RUNS} reps) | **${COLD_MEDIAN_S}s** |
| Warm start (page cache hot) | **${WARM_S}s** |

Cold start reps (ms): ${COLD_MS[*]}

## Verdict: ${VERDICT}

| Range | Interpretation |
|-------|---------------|
| < 30s | Cold-start on demand OK |
| 30–90s | Acceptable for explicit escalation |
| > 90s | Always-resident strongly preferred |

## TPS baseline (ngl=0)

See \`tps_baseline_ngl0.json\`. Compare this to the original measured 13.15 t/s at
ngl=999 --cpu-moe (attention on GPU) from R12. The ngl=0 baseline is expected to be
lower — the tradeoff for running in parallel with Arclight without VRAM conflict.

## Next steps

T_CV2: Thread count sweep at ngl=0 to find the optimal -t value.
EOMD

    echo "[T_CV1] Results written to ${RESULTS_DIR}/"
fi
