#!/usr/bin/env bash
# benchmarks/queue/T_CV1_convergence_startup_timing.sh
#
# T_CV1 — Convergence startup timing + TPS baseline + context ceiling sweep.
#
# Part A: How long does Convergence take to become ready?
#   - cold start: page cache dropped (NVMe → RAM → serve)
#   - warm start: page cache already populated
#
# Part B: TPS baseline at ngl=0 (feeds T_CV2 thread sweep).
#
# Part C: Context ceiling sweep — vary -c to find practical max before OOM.
#
# QUESTION: Is cold-start-on-demand acceptable (<90s) or is always-resident required?
#   < 30s  → cold-start on demand is fine
#   30–90s → acceptable for explicit escalation, not transparent routing
#   > 90s  → always-resident strongly preferred
#
# PREREQUISITES:
#   - ik_llama.cpp pr-1288 built (infra/scripts/build-ik-llama.sh)
#   - Model at ${MODEL_CACHE}/${CONVERGENCE_MODEL}
#   - Container bench-ikllamacpp-convergence not already running
#
# OPTIONS:
#   --runs N        Number of cold-start reps (default: 3)
#   --skip-drop     Skip page cache drop (only warm starts)
#   --skip-ctx      Skip Part C context sweep
#   --dry-run       Print commands, do not execute

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

RUNS=3
SKIP_DROP=0
SKIP_CTX=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --runs)        shift; RUNS="${1:-3}" ;;
        --skip-drop)   SKIP_DROP=1 ;;
        --skip-ctx)    SKIP_CTX=1 ;;
        --dry-run)     DRY_RUN=1 ;;
    esac
done

run() { [[ "${DRY_RUN}" -eq 1 ]] && { echo "[dry-run] $*"; return; }; "$@"; }

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/T_CV1_convergence_startup_timing_${TIMESTAMP}"
CONTAINER_NAME="bench-ikllamacpp-convergence"
IK_BUILD="${IK_LLAMA_BUILD_DIR:-/srv/ai/projects/ik_llama.cpp/build}"
HEALTH_URL="http://localhost:${PORT_CONVERGENCE:-8002}/health"

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

# ── Helper: start container and time until health ─────────────────────────────
startup_once() {
    local rep="$1"
    local mode="$2"   # "cold" or "warm"

    echo "[T_CV1] Rep ${rep} (${mode}): starting server..."
    local T_START_MS
    T_START_MS=$(date +%s%3N)

    "${IK_BUILD}/bin/llama-server" \
        -m "${MODEL_CACHE}/${CONVERGENCE_MODEL}" \
        --port "${PORT_CONVERGENCE:-8002}" \
        --host 127.0.0.1 \
        -ngl 0 \
        --no-mmap \
        -b 4096 -ub 2048 \
        -t "$(nproc)" \
        -c 16384 \
        --jinja \
        > "${RESULTS_DIR}/server_${mode}_${rep}.log" 2>&1 &
    local SERVER_PID=$!

    echo "[T_CV1] Rep ${rep}: polling ${HEALTH_URL} every 500ms..."
    local T_READY_MS=0
    while true; do
        if curl -sf --max-time 2 "${HEALTH_URL}" &>/dev/null; then
            T_READY_MS=$(date +%s%3N)
            break
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[T_CV1] ERROR: server died during ${mode} rep ${rep}" >&2
            cat "${RESULTS_DIR}/server_${mode}_${rep}.log" >&2
            return 1
        fi
        sleep 0.5
    done

    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true

    local STARTUP_MS=$(( T_READY_MS - T_START_MS ))
    echo "[T_CV1] Rep ${rep} (${mode}): ready in ${STARTUP_MS} ms ($(( STARTUP_MS / 1000 ))s)"
    echo "${STARTUP_MS}"
}

stop_server() {
    pkill -f "llama-server.*${PORT_CONVERGENCE:-8002}" 2>/dev/null || true
    sleep 1
}

# ── Ensure no existing server ─────────────────────────────────────────────────
stop_server

# ── Part A: Cold-start reps ───────────────────────────────────────────────────
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
    sleep 2
done

# ── Part A: Warm-start rep ────────────────────────────────────────────────────
echo "[T_CV1] Warm-start rep (page cache hot from last cold run)..."
WARM_MS=0
if [[ "${DRY_RUN}" -eq 0 ]]; then
    WARM_MS=$(startup_once "warm" "warm")
fi

# ── Part B: TPS baseline (ngl=0) ─────────────────────────────────────────────
echo "[T_CV1] Measuring TPS baseline (ngl=0, t=$(nproc), 3 reps)..."
TPS_JSON="${RESULTS_DIR}/tps_baseline_ngl0.json"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${IK_BUILD}/bin/llama-bench" \
        -m "${MODEL_CACHE}/${CONVERGENCE_MODEL}" \
        -ngl 0 \
        --no-mmap \
        -b 4096 -ub 2048 \
        -t "$(nproc)" \
        -p 512 -n 128 \
        -r 3 \
        --output json \
        > "${TPS_JSON}" 2>&1
    echo "[T_CV1] llama-bench output → ${TPS_JSON}"
    TPS_TG=$(python3 -c "
import json
d = json.load(open('${TPS_JSON}'))
rows = [r for r in d if r.get('n_prompt')==512 and r.get('n_gen')==128]
print(rows[0]['avg_ts'] if rows else 'N/A')
" 2>/dev/null || echo "N/A")
    echo "[T_CV1] Baseline tg128 at ngl=0: ${TPS_TG} t/s"
else
    echo "[dry-run] llama-bench ngl=0 → ${TPS_JSON}"
    TPS_TG="N/A"
fi

# ── Part C: Context ceiling sweep ────────────────────────────────────────────
declare -A CTX_RESULTS=()
if [[ "${SKIP_CTX}" -eq 0 ]] && [[ "${DRY_RUN}" -eq 0 ]]; then
    echo "[T_CV1] Starting context ceiling sweep..."
    for CTX in 16384 32768 65536 131072; do
        FREE_RAM=$(free -g | awk '/^Mem:/{print $7}')
        if [[ "${FREE_RAM}" -lt 8 ]]; then
            echo "[T_CV1] Only ${FREE_RAM}GB RAM free — stopping context sweep."
            break
        fi

        echo "[T_CV1] Testing -c ${CTX}..."
        "${IK_BUILD}/bin/llama-server" \
            -m "${MODEL_CACHE}/${CONVERGENCE_MODEL}" \
            --port "${PORT_CONVERGENCE:-8002}" \
            --host 127.0.0.1 \
            -ngl 0 --no-mmap \
            -b 4096 -ub 2048 \
            -t "$(nproc)" \
            -c "${CTX}" \
            --jinja \
            > "${RESULTS_DIR}/server_ctx${CTX}.log" 2>&1 &
        CTX_PID=$!

        # Wait for ready
        CTX_READY=0
        for _ in $(seq 1 600); do
            if curl -sf --max-time 2 "${HEALTH_URL}" &>/dev/null; then
                CTX_READY=1; break
            fi
            if ! kill -0 "${CTX_PID}" 2>/dev/null; then break; fi
            sleep 0.5
        done

        if [[ ${CTX_READY} -eq 0 ]]; then
            echo "[T_CV1] -c ${CTX}: server failed to start (OOM or crash)"
            CTX_RESULTS[${CTX}]="FAIL"
            kill "${CTX_PID}" 2>/dev/null || true
            break
        fi

        # Quick benchmark: send a prompt at 50% of context
        PROMPT_TOKENS=$(( CTX / 2 ))
        CTX_BENCH_JSON="${RESULTS_DIR}/bench_ctx${CTX}.json"
        "${IK_BUILD}/bin/llama-bench" \
            -m "${MODEL_CACHE}/${CONVERGENCE_MODEL}" \
            -ngl 0 --no-mmap \
            -b 4096 -ub 2048 \
            -t "$(nproc)" \
            -p "${PROMPT_TOKENS}" -n 64 \
            -r 1 \
            --output json \
            > "${CTX_BENCH_JSON}" 2>/dev/null || true

        CTX_TPS=$(python3 -c "
import json, os
f = '${CTX_BENCH_JSON}'
if not os.path.exists(f): print('N/A'); exit()
d = json.load(open(f))
rows = [r for r in d if r.get('n_gen', 0) > 0]
print(rows[0]['avg_ts'] if rows else 'N/A')
" 2>/dev/null || echo "N/A")

        echo "[T_CV1] -c ${CTX}: tg64 = ${CTX_TPS} t/s"
        CTX_RESULTS[${CTX}]="${CTX_TPS}"

        kill "${CTX_PID}" 2>/dev/null || true
        wait "${CTX_PID}" 2>/dev/null || true
        sleep 2
    done
elif [[ "${SKIP_CTX}" -eq 0 ]]; then
    echo "[dry-run] Would sweep context: 16384 32768 65536 131072"
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
    echo " TPS baseline (ngl=0):        ${TPS_TG} t/s"
    echo " Verdict: ${VERDICT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CTX_SWEEP_JSON="null"
    if [[ ${#CTX_RESULTS[@]} -gt 0 ]]; then
        CTX_SWEEP_JSON="{"
        for k in "${!CTX_RESULTS[@]}"; do
            CTX_SWEEP_JSON+="\"${k}\": \"${CTX_RESULTS[$k]}\","
        done
        CTX_SWEEP_JSON="${CTX_SWEEP_JSON%,}}"
    fi

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
    "warm_start_s": ${WARM_S},
    "tps_baseline_ngl0": "${TPS_TG}",
    "context_ceiling_sweep": ${CTX_SWEEP_JSON}
  },
  "verdict": "${VERDICT}",
  "notes": "ngl=0 (CPU-only). Cold start = NVMe → RAM. Warm = page-cache hot. TPS baseline feeds T_CV2. Context sweep finds practical RAM ceiling."
}
EOJSON

    cat > "${RESULTS_DIR}/summary.md" <<EOMD
# T_CV1 — Convergence Startup Timing + Context Ceiling

**Timestamp:** ${TIMESTAMP}
**Config:** ik_llama.cpp pr-1288 · Qwen3.5-397B UD-IQ2_M · ngl=0 (CPU-only) · --no-mmap · t=$(nproc)

## Part A — Startup timing

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

## Part B — TPS baseline (ngl=0, t=$(nproc))

**tg128:** ${TPS_TG} t/s (baseline for T_CV2 thread sweep comparison)

Compare to the original 13.15 t/s measured at ngl=999 --cpu-moe (attention on GPU, R12).
The ngl=0 baseline is expected to be slightly lower — the tradeoff for running always-on alongside Arclight.

## Part C — Context ceiling sweep

| -c value | tg64 TPS |
|----------|---------|
$(for k in "${!CTX_RESULTS[@]}"; do echo "| ${k} | ${CTX_RESULTS[$k]} |"; done | sort -t'|' -k2 -n)

Sweep stopped at first OOM/crash or when free RAM dropped below 8GB.
EOMD

    echo "[T_CV1] Results written to ${RESULTS_DIR}/"
else
    echo "[dry-run] Would run ${RUNS} cold-start reps, 1 warm rep, TPS baseline, and context sweep."
fi
