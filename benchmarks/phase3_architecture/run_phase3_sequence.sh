#!/usr/bin/env bash
# run_phase3_sequence.sh — Full Phase 3 orchestrator
#
# Execution order:
#   3.3 (swap timing)  — standalone, fastest, no dual-GPU required
#   3.4 (swarm)        — standalone, single GPU
#   3.1 (dual-model)   — deploys BOTH GPUs (longest)
#   3.2 (routing)      — passive; runs while both models are up from 3.1
#
# Skip flags:
#   --skip-3.3   skip swap timing
#   --skip-3.4   skip swarm test
#   --skip-3.1   skip dual-model (implies skip 3.2 unless --litellm-endpoint set)
#   --skip-3.2   skip routing test
#
# Usage:
#   ./benchmarks/phase3_architecture/run_phase3_sequence.sh
#   ./benchmarks/phase3_architecture/run_phase3_sequence.sh --skip-3.3

set -euo pipefail
source config/hardware.env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_33=false
SKIP_34=false
SKIP_31=false
SKIP_32=false

for arg in "$@"; do
    case "${arg}" in
        --skip-3.3) SKIP_33=true ;;
        --skip-3.4) SKIP_34=true ;;
        --skip-3.1) SKIP_31=true ;;
        --skip-3.2) SKIP_32=true ;;
        *) echo "[sequence] Unknown flag: ${arg}" >&2; exit 1 ;;
    esac
done

PHASE_LOG="results/phase3_sequence_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results

echo "======================================================================"
echo "Phase 3 — Architecture sequence"
echo "Order: 3.3 → 3.4 → 3.1 → 3.2"
echo "Log: ${PHASE_LOG}"
echo "======================================================================"

run_step() {
    local label="$1"
    local script="$2"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ${label}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    bash "${script}" 2>&1 | tee -a "${PHASE_LOG}"
    echo ""
}

# ── 3.3: Swap timing ──────────────────────────────────────────────────────────
if [[ "${SKIP_33}" == "false" ]]; then
    run_step "3.3 — Swap timing" "${SCRIPT_DIR}/run_3.3_swap_timing.sh"
else
    echo "[sequence] Skipping 3.3 (swap timing)"
fi

# ── 3.4: Swarm ────────────────────────────────────────────────────────────────
if [[ "${SKIP_34}" == "false" ]]; then
    run_step "3.4 — Swarm concurrency" "${SCRIPT_DIR}/run_3.4_swarm.sh"
else
    echo "[sequence] Skipping 3.4 (swarm)"
fi

# ── 3.1: Dual-model ───────────────────────────────────────────────────────────
if [[ "${SKIP_31}" == "false" ]]; then
    run_step "3.1 — Dual-model co-resident" "${SCRIPT_DIR}/run_3.1_dual_model.sh"

    # ── 3.2: Routing (passive, piggybacks on 3.1's deployments) ────────────────
    # Note: run_3.1 tears down at the end, so we redeploy both models here
    # if we also want to run 3.2. To run 3.2 after 3.1 while models are up,
    # call them individually or use --skip-3.1 and start models manually.
    if [[ "${SKIP_32}" == "false" ]]; then
        echo ""
        echo "[sequence] Note: re-deploying both models for 3.2 routing test..."
        export CODER_MODEL="${CODER_MODEL:-Qwen/Qwen3.5-35B-A3B-AWQ}"
        export THINKER_MODEL="${THINKER_MODEL:-Qwen/Qwen3.5-27B}"
        ./infra/scripts/deploy.sh vllm gpu0 "${CODER_MODEL}" \
            --ctx "${CTX_LEN:-32768}" --tool-call-parser qwen3_coder --reasoning-parser qwen3
        ./infra/scripts/deploy.sh vllm gpu1 "${THINKER_MODEL}" \
            --ctx "${CTX_LEN:-32768}" --reasoning-parser qwen3 --dtype bfloat16
        run_step "3.2 — Routing accuracy" "${SCRIPT_DIR}/run_3.2_routing.sh"
        ./infra/scripts/teardown.sh
    else
        echo "[sequence] Skipping 3.2 (routing)"
    fi
else
    echo "[sequence] Skipping 3.1 (dual-model)"
    if [[ "${SKIP_32}" == "false" ]]; then
        echo "[sequence] Skipping 3.2 (routing) — requires 3.1 or manual model deploy"
    fi
fi

echo ""
echo "======================================================================"
echo "Phase 3 complete. Results in results/phase3_*"
echo "======================================================================"
