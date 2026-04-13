#!/usr/bin/env bash
# Sub-test 3.4 — Parallel swarm throughput
#
# Hypothesis: Running 8 concurrent coding requests delivers ≥40% wall-time
# reduction vs sequential (swarm_speedup ratio ≤ 0.60).
#
# Protocol:
#   1. Deploy coder model on GPU 0.
#   2. Run 8 lightweight tasks sequentially; record wall time.
#   3. Run same 8 tasks at concurrency=8; record wall time.
#   4. Report swarm_speedup = parallel_wall / sequential_wall.
#
# Pass:   swarm_speedup ≤ 0.60  (parallel ≥40% faster)
# Incon.: swarm_speedup ≤ 0.85

set -euo pipefail
source config/hardware.env

PHASE="phase3_3.4_swarm"
RESULTS_DIR="results/${PHASE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}/raw"

CODER_MODEL="${CODER_MODEL:-Qwen/Qwen3.5-35B-A3B-AWQ}"
CTX_LEN="${CTX_LEN:-32768}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"

CODER_ENGINE_ARGS="--tool-call-parser qwen3_coder --reasoning-parser qwen3"

echo "======================================================================"
echo "Phase 3.4 — Swarm concurrency test"
echo "Model       : ${CODER_MODEL}"
echo "Concurrency : ${CONCURRENCY}"
echo "Results     : ${RESULTS_DIR}"
echo "======================================================================"

echo ""
echo "=== Deploying coder model ==="
./infra/scripts/deploy.sh vllm gpu0 "${CODER_MODEL}" \
    --ctx "${CTX_LEN}" ${CODER_ENGINE_ARGS}

echo ""
echo "=== Running swarm bench (sequential → parallel) ==="

python -m benchmarks.phase3_architecture.bench \
    --mode swarm \
    --endpoint "${VLLM_GPU0_ENDPOINT}" \
    --tasks benchmarks/phase3_architecture/tasks/swarm/ \
    --model "${CODER_MODEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --concurrency "${CONCURRENCY}" \
    --results-dir "${RESULTS_DIR}" \
    --label "Swarm ${CODER_MODEL} c=${CONCURRENCY}" \
    --engine vllm \
    --quantization AWQ-INT4 \
    --ctx-len "${CTX_LEN}" \
    --extra-args "${CODER_ENGINE_ARGS}" \
    --notes "Sub-test 3.4: parallel swarm at concurrency ${CONCURRENCY}" \
    2>&1 | tee "${RESULTS_DIR}/bench.log"

echo ""
echo "=== Tearing down ==="
./infra/scripts/teardown.sh

python -m lib.reporter "${RESULTS_DIR}" --thresholds config/thresholds.yaml
cat "${RESULTS_DIR}/summary.md"

echo ""
echo "======================================================================"
echo "Results: ${RESULTS_DIR}"
echo "swarm_speedup ≤ 0.60 → PASS (parallel ≥40% faster than sequential)"
echo "If speedup is poor, check vLLM --max-num-seqs and --max-num-batched-tokens"
echo "======================================================================"
