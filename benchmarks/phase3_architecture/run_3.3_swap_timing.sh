#!/usr/bin/env bash
# Sub-test 3.3 — Model hot-swap latency
#
# Hypothesis: With weights pre-cached in page cache, swapping from coder
# (35B MoE) to thinker (27B dense) completes in ≤30 seconds end-to-end.
#
# Protocol:
#   1. Deploy model A on GPU 0.
#   2. Call swap-model.sh to swap to model B; record wall time.
#   3. Swap back to model A; record wall time.
#   4. Repeat SWAP_ROUNDS times.
#   5. Report p50 and p95 swap latency.
#
# Pass:   p95 swap_latency_s ≤ 30 s
# Incon.: p95 swap_latency_s ≤ 60 s
#
# Pre-requisite: run `just precache` first so weights are in page cache.

set -euo pipefail
source config/hardware.env

PHASE="phase3_3.3_swap"
RESULTS_DIR="results/${PHASE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}/raw"

MODEL_A="${MODEL_A:-QuantTrio/Qwen3.5-35B-A3B-AWQ}"
MODEL_B="${MODEL_B:-Qwen/Qwen3.5-27B}"
SWAP_ROUNDS="${SWAP_ROUNDS:-5}"
CTX_LEN="${CTX_LEN:-32768}"

MODEL_A_EXTRA="--tool-call-parser qwen3_coder --reasoning-parser qwen3"
MODEL_B_EXTRA="--reasoning-parser qwen3 --dtype bfloat16"

echo "======================================================================"
echo "Phase 3.3 — Model hot-swap latency"
echo "Model A : ${MODEL_A}"
echo "Model B : ${MODEL_B}"
echo "Rounds  : ${SWAP_ROUNDS} (each direction → total ${SWAP_ROUNDS}×2 swaps)"
echo "Results : ${RESULTS_DIR}"
echo "======================================================================"

# ── Deploy initial model ──────────────────────────────────────────────────────
echo ""
echo "=== Deploying initial model A ==="
./infra/scripts/deploy.sh vllm gpu0 "${MODEL_A}" \
    --ctx "${CTX_LEN}" ${MODEL_A_EXTRA}

echo ""
echo "=== Running swap-timing bench ==="

python -m benchmarks.phase3_architecture.bench \
    --mode swap-timing \
    --engine vllm \
    --gpu gpu0 \
    --model "${MODEL_A}" \
    --swap-target "${MODEL_B}" \
    --swap-rounds "${SWAP_ROUNDS}" \
    --extra-args "--ctx ${CTX_LEN} ${MODEL_A_EXTRA}" \
    --swap-target-extra-args "--ctx ${CTX_LEN} ${MODEL_B_EXTRA}" \
    --results-dir "${RESULTS_DIR}" \
    --label "swap ${MODEL_A} ↔ ${MODEL_B}" \
    --notes "Sub-test 3.3: hot-swap latency" \
    2>&1 | tee "${RESULTS_DIR}/bench.log"

echo ""
echo "=== Tearing down ==="
./infra/scripts/teardown.sh

python -m lib.reporter "${RESULTS_DIR}" --thresholds config/thresholds.yaml
cat "${RESULTS_DIR}/summary.md"

echo ""
echo "======================================================================"
echo "Results: ${RESULTS_DIR}"
echo "swap_latency_s p95 ≤ 30 s → PASS   ≤ 60 s → INCONCLUSIVE"
echo "If slow: run 'just precache' to warm page cache before this test."
echo "======================================================================"
