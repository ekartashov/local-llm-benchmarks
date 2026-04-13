#!/usr/bin/env bash
# Sub-test 3.1 — Co-resident dual-model vs single-model throughput
#
# Hypothesis: Can we run coder (35B MoE on GPU 0) + thinker (27B dense on
# GPU 1) simultaneously without severe throughput degradation vs single-model?
#
# Protocol:
#   Baseline A — single Qwen3.5-35B-A3B-AWQ on GPU 0 (coder quality tasks)
#   Baseline B — single Qwen3.5-27B on GPU 1 (thinker tasks)
#   Test       — both models co-resident; run same tasks concurrently
#
# Pass criterion: INCONCLUSIVE or PASS from bench (informational; compare
# coder/thinker decode_tps against Phase 2 single-model baselines manually).
#
# "4 days" note: this script runs one ~30-min session. Run it repeatedly
# over several days to collect stable numbers under realistic load.

set -euo pipefail
source config/hardware.env

PHASE="phase3_3.1_dual"
RESULTS_DIR="results/${PHASE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}/raw"

# ── Models (edit to Phase 2 winners) ─────────────────────────────────────────
CODER_MODEL="${CODER_MODEL:-Qwen/Qwen3.5-35B-A3B-AWQ}"
THINKER_MODEL="${THINKER_MODEL:-Qwen/Qwen3.5-27B}"
CTX_LEN="${CTX_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
CODER_QUANT="AWQ-INT4"
THINKER_QUANT="BF16"

CODER_ENGINE_ARGS="--tool-call-parser qwen3_coder --reasoning-parser qwen3"
THINKER_ENGINE_ARGS="--reasoning-parser qwen3 --dtype bfloat16"

echo "======================================================================"
echo "Phase 3.1 — Dual-model co-resident benchmark"
echo "Coder : ${CODER_MODEL} → GPU 0 (port ${PORT_VLLM_GPU0})"
echo "Thinker: ${THINKER_MODEL} → GPU 1 (port ${PORT_VLLM_GPU1})"
echo "Results: ${RESULTS_DIR}"
echo "======================================================================"

# ── Deploy BOTH models ────────────────────────────────────────────────────────
echo ""
echo "=== Deploying coder on GPU 0 ==="
./infra/scripts/deploy.sh vllm gpu0 "${CODER_MODEL}" \
    --ctx "${CTX_LEN}" ${CODER_ENGINE_ARGS}

echo ""
echo "=== Deploying thinker on GPU 1 ==="
./infra/scripts/deploy.sh vllm gpu1 "${THINKER_MODEL}" \
    --ctx "${CTX_LEN}" ${THINKER_ENGINE_ARGS}

echo ""
echo "=== Both models running. Starting dual-model bench ==="

python -m benchmarks.phase3_architecture.bench \
    --mode dual-model \
    --endpoint "${VLLM_GPU0_ENDPOINT}" \
    --thinker-endpoint "${VLLM_GPU1_ENDPOINT}" \
    --tasks benchmarks/phase2_model_selection/tasks/quality/ \
    --thinker-tasks benchmarks/phase2_model_selection/tasks/thinker/ \
    --model "${CODER_MODEL}" \
    --thinker-model "${THINKER_MODEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --results-dir "${RESULTS_DIR}" \
    --label "Dual ${CODER_MODEL}+${THINKER_MODEL}" \
    --engine vllm \
    --quantization "${CODER_QUANT}" \
    --ctx-len "${CTX_LEN}" \
    --extra-args "${CODER_ENGINE_ARGS}" \
    --notes "Sub-test 3.1: co-resident dual-model" \
    2>&1 | tee "${RESULTS_DIR}/bench.log"

echo ""
echo "=== Tearing down ==="
./infra/scripts/teardown.sh

python -m lib.reporter "${RESULTS_DIR}" --thresholds config/thresholds.yaml
cat "${RESULTS_DIR}/summary.md"

echo ""
echo "======================================================================"
echo "Results: ${RESULTS_DIR}"
echo "Compare coder_decode_tps and thinker_decode_tps against Phase 2"
echo "single-model baselines to determine if co-residency is viable."
echo "======================================================================"
