#!/usr/bin/env bash
# benchmarks/queue/T_CV2_convergence_thread_sweep.sh
#
# T_CV2 — Convergence thread count sweep (CPU-only).
#
# Question: What thread count maximizes token generation speed for Convergence
# at -ngl 0 (CPU-only baseline)?
#
# Procedure:
#   1. Runs llama-bench inside the local-ik-llama:runtime container.
#   2. Sweeps thread counts [8, 12, 16, 20, 24, 28, 32].
#   3. Records tg128 and pp512 for each count.
#
# PREREQUISITES:
#   - local-ik-llama:runtime image built
#   - infra/scripts/build-ik-llama.sh completed successfully
#   - Model at ${MODEL_CACHE}/${CONVERGENCE_MODEL}
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
RESULTS_DIR="${REPO_ROOT}/results/T_CV2_convergence_thread_sweep_${TIMESTAMP}"
IK_BUILD="${IK_LLAMA_BUILD_DIR:-/srv/ai/projects/ik_llama.cpp/build}"

# ── Validation ────────────────────────────────────────────────────────────────
[[ ! -x "${IK_BUILD}/bin/llama-bench" ]] && {
    echo "ERROR: llama-bench not found at ${IK_BUILD}/bin/llama-bench" >&2
    echo "  Run: ./infra/scripts/build-ik-llama.sh" >&2; exit 1
}
[[ ! -f "${MODEL_CACHE}/${CONVERGENCE_MODEL}" ]] && {
    echo "ERROR: Model not found: ${MODEL_CACHE}/${CONVERGENCE_MODEL}" >&2; exit 1
}

echo "[T_CV2] Results dir: ${RESULTS_DIR}"
[[ "${DRY_RUN}" -eq 0 ]] && mkdir -p "${RESULTS_DIR}"

CSV_FILE="${RESULTS_DIR}/thread_sweep_ngl0.csv"
echo "threads,pp512_tps,tg128_tps" | tee "${CSV_FILE}"

# ── Thread Sweep ──────────────────────────────────────────────────────────────
for T in 8 12 16 20 24 28 32; do
    echo "[T_CV2] Testing -t ${T}..."
    
    JSON_OUT="${RESULTS_DIR}/bench_t${T}.json"
    
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        podman run --rm \
            --name "ik-llama-bench-t${T}" \
            --userns=keep-id \
            -v "${IK_BUILD}:/app/build:ro,z" \
            -v "${MODEL_CACHE}:/models:ro,z" \
            --entrypoint "/app/build/bin/llama-bench" \
            local-ik-llama:runtime \
            -m "/models/${CONVERGENCE_MODEL}" \
            -ngl 0 \
            --no-mmap \
            -b 4096 -ub 2048 \
            -t "${T}" \
            -p 512 -n 128 \
            -r 3 \
            --output json \
            > "${JSON_OUT}" 2>/dev/null
            
        # Extract TPS values from JSON
        # jq is usually available in this environment. If not, use python.
        PP=$(python3 -c "import json, sys; d=json.load(open('${JSON_OUT}')); print([r for r in d if r['n_prompt']==512 and r['n_gen']==0][0]['tps'])" 2>/dev/null || echo "N/A")
        TG=$(python3 -c "import json, sys; d=json.load(open('${JSON_OUT}')); print([r for r in d if r['n_prompt']==512 and r['n_gen']==128][0]['tps'])" 2>/dev/null || echo "N/A")
        
        echo "${T},${PP},${TG}" | tee -a "${CSV_FILE}"
    else
        echo "[dry-run] podman run llama-bench -t ${T} -ngl 0"
    fi
done

# ── Summary and metrics.json ──────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 0 ]]; then
    BEST_TG_ROW=$(tail -n +2 "${CSV_FILE}" | sort -t',' -k3 -nr | head -1)
    BEST_T=$(echo "${BEST_TG_ROW}" | cut -d',' -f1)
    BEST_TG=$(echo "${BEST_TG_ROW}" | cut -d',' -f3)

    cat > "${RESULTS_DIR}/metrics.json" <<EOJSON
{
  "item_id": "T_CV2_convergence_thread_sweep",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config": {
    "engine": "ikllamacpp",
    "model": "unsloth/Qwen3.5-397B-A17B-GGUF UD-IQ2_M",
    "quantization": "IQ2_M",
    "placement": "convergence (cpu-only, ngl=0)",
    "context_length": 16384,
    "no_mmap": true
  },
  "metrics": {
    "thread_sweep_csv": "thread_sweep_ngl0.csv",
    "best_threads": ${BEST_T},
    "best_tg128_tps": ${BEST_TG}
  },
  "verdict": "MEASURED",
  "notes": "Determined optimal thread count for CPU-only production mode. Baseline was 32 threads."
}
EOJSON

    cat > "${RESULTS_DIR}/summary.md" <<EOMD
# T_CV2 — Convergence Thread Sweep (ngl=0)

**Timestamp:** ${TIMESTAMP}
**Config:** ik_llama.cpp pr-1288 · Qwen3.5-397B UD-IQ2_M · ngl=0 (CPU-only)

## Results

| Threads | PP512 TPS | TG128 TPS |
|---------|-----------|-----------|
$(awk -F, 'NR>1 {printf "| %d | %s | **%s** |\n", $1, $2, $3}' "${CSV_FILE}")

**Optimal threads for generation:** ${BEST_T} (${BEST_TG} t/s)

## Analysis

Compare these results to the 13.15 t/s measured at -ngl 999. The ngl=0 baseline is the
true production performance for always-on Convergence running in parallel with Arclight.
EOMD

    echo "[T_CV2] Done. Results in ${RESULTS_DIR}/"
fi
