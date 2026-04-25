#!/usr/bin/env bash
# benchmarks/queue/T_CV3_convergence_gpu_expert_offload.sh
#
# T_CV3 — Convergence partial GPU expert offload.
#
# Question: Can we improve Convergence generation speed by offloading some
# MoE expert layers to GPU, given that GPU VRAM is idle?
#
# Procedure:
#   1. Requires Arclight/Core to be sleeping (frees VRAM).
#   2. Runs llama-bench with varying --n-cpu-moe N values.
#   3. Uses CDI GPU device args (nvidia.com/gpu=0, nvidia.com/gpu=1).
#   4. Compares tg128 at various offload depths.
#
# PREREQUISITES:
#   - Arclight/Core SLEEPING (at least level=1)
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
RESULTS_DIR="${REPO_ROOT}/results/T_CV3_convergence_gpu_expert_offload_${TIMESTAMP}"
IK_BUILD="${IK_LLAMA_BUILD_DIR:-/srv/ai/projects/ik_llama.cpp/build}"

# ── Validation ────────────────────────────────────────────────────────────────
[[ ! -x "${IK_BUILD}/bin/llama-bench" ]] && {
    echo "ERROR: llama-bench not found at ${IK_BUILD}/bin/llama-bench" >&2
    echo "  Run: ./infra/scripts/build-ik-llama.sh" >&2; exit 1
}
[[ ! -f "${MODEL_CACHE}/${CONVERGENCE_MODEL}" ]] && {
    echo "ERROR: Model not found: ${MODEL_CACHE}/${CONVERGENCE_MODEL}" >&2; exit 1
}

# Check if Arclight is sleeping (warning only)
if curl -sf "http://localhost:30000/is_sleeping" | grep -q "false"; then
    echo "WARNING: Arclight Coder (port 30000) is NOT sleeping. VRAM conflict possible."
fi
if curl -sf "http://localhost:30001/is_sleeping" | grep -q "false"; then
    echo "WARNING: Arclight Thinker (port 30001) is NOT sleeping. VRAM conflict possible."
fi

echo "[T_CV3] Results dir: ${RESULTS_DIR}"
[[ "${DRY_RUN}" -eq 0 ]] && mkdir -p "${RESULTS_DIR}"

CSV_FILE="${RESULTS_DIR}/gpu_offload_sweep.csv"
echo "n_cpu_moe,gpu_layers,tg128_tps,vram_used_mb" | tee "${CSV_FILE}"

# ── Offload Sweep ─────────────────────────────────────────────────────────────
# Model has 60 layers.
# --n-cpu-moe 60: 0 layers on GPU
# --n-cpu-moe 50: 10 layers on GPU
# --n-cpu-moe 40: 20 layers on GPU
# --n-cpu-moe 30: 30 layers on GPU
for N_CPU in 60 55 50 45 40 35 30; do
    GPU_LAYERS=$(( 60 - N_CPU ))
    echo "[T_CV3] Testing --n-cpu-moe ${N_CPU} (${GPU_LAYERS} layers on GPU)..."
    
    JSON_OUT="${RESULTS_DIR}/bench_ncpu${N_CPU}.json"
    
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        # Capture VRAM before
        VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
        
        podman run --rm \
            --name "ik-llama-bench-ncpu${N_CPU}" \
            --userns=keep-id \
            --device nvidia.com/gpu=0 --device nvidia.com/gpu=1 \
            -e NVIDIA_VISIBLE_DEVICES=0,1 \
            -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
            -v "${IK_BUILD}:/app/build:ro,z" \
            -v "${MODEL_CACHE}:/models:ro,z" \
            --entrypoint "/app/build/bin/llama-bench" \
            local-ik-llama:runtime \
            -m "/models/${CONVERGENCE_MODEL}" \
            -ngl 999 \
            --n-cpu-moe "${N_CPU}" \
            --no-mmap \
            -b 4096 -ub 2048 \
            -t "$(nproc)" \
            -p 512 -n 128 \
            -r 2 \
            --output json \
            > "${JSON_OUT}" 2>/dev/null || {
                echo "[T_CV3] OOM or CRASH at --n-cpu-moe ${N_CPU}. Stopping sweep."
                break
            }
            
        VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
        VRAM_DIFF=$(( VRAM_AFTER - VRAM_BEFORE ))
        
        TG=$(python3 -c "import json, sys; d=json.load(open('${JSON_OUT}')); print([r for r in d if r['n_prompt']==512 and r['n_gen']==128][0]['tps'])" 2>/dev/null || echo "N/A")
        
        echo "${N_CPU},${GPU_LAYERS},${TG},${VRAM_DIFF}" | tee -a "${CSV_FILE}"
    else
        echo "[dry-run] podman run llama-bench --n-cpu-moe ${N_CPU} -ngl 999"
    fi
done

# ── Summary and metrics.json ──────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 0 ]]; then
    BEST_TG_ROW=$(tail -n +2 "${CSV_FILE}" | sort -t',' -k3 -nr | head -1)
    BEST_N_CPU=$(echo "${BEST_TG_ROW}" | cut -d',' -f1)
    BEST_LAYERS=$(echo "${BEST_TG_ROW}" | cut -d',' -f2)
    BEST_TG=$(echo "${BEST_TG_ROW}" | cut -d',' -f3)

    cat > "${RESULTS_DIR}/metrics.json" <<EOJSON
{
  "item_id": "T_CV3_convergence_gpu_expert_offload",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config": {
    "engine": "ikllamacpp",
    "model": "unsloth/Qwen3.5-397B-A17B-GGUF UD-IQ2_M",
    "quantization": "IQ2_M",
    "placement": "convergence (partial GPU offload)",
    "context_length": 16384,
    "no_mmap": true
  },
  "metrics": {
    "offload_sweep_csv": "gpu_offload_sweep.csv",
    "best_n_cpu_moe": ${BEST_N_CPU},
    "best_gpu_layers": ${BEST_LAYERS},
    "best_tg128_tps": ${BEST_TG}
  },
  "verdict": "MEASURED",
  "notes": "Experiment to see if generation speed improves when Arclight/Core are sleeping. Not for production always-on use."
}
EOJSON

    cat > "${RESULTS_DIR}/summary.md" <<EOMD
# T_CV3 — Convergence GPU Expert Offload Experiment

**Timestamp:** ${TIMESTAMP}
**Config:** ik_llama.cpp pr-1288 · Qwen3.5-397B UD-IQ2_M · -ngl 999 · --n-cpu-moe N

## Results

| N_CPU_MOE | GPU Layers | TG128 TPS | VRAM Delta (MB) |
|-----------|------------|-----------|-----------------|
$(awk -F, 'NR>1 {printf "| %d | %d | **%s** | %s |\n", $1, $2, $3, $4}' "${CSV_FILE}")

**Optimal offload depth:** ${BEST_LAYERS} layers on GPU (${BEST_TG} t/s)

## Analysis

This experiment establishes the "maximum possible" speed for Convergence when Arclight/Core
are not using the GPUs. Since each expert layer offloaded to GPU avoids DDR5 bandwidth bottlenecks,
we expect a linear speedup until VRAM is saturated or diminishing returns hit.
EOMD

    echo "[T_CV3] Done. Results in ${RESULTS_DIR}/"
fi
