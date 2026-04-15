#!/usr/bin/env bash
# Sub-test 3.2 — LiteLLM routing accuracy
#
# Hypothesis: A heuristic keyword classifier can route coder vs thinker
# requests correctly ≥95% of the time.
#
# Protocol:
#   1. Deploy coder on GPU 0, thinker on GPU 1 (both must be running).
#   2. Deploy LiteLLM proxy (routes "coder" alias → GPU 0, "thinker" → GPU 1).
#   3. Send 10 routing tasks through the bench's classify_tier() heuristic.
#   4. Report routing_accuracy and proxy overhead.
#
# Pass: routing_accuracy ≥ 0.95
# Note: This test runs passively during 3.1 if both models are already up.

set -euo pipefail
source config/hardware.env

PHASE="phase3_3.2_routing"
RESULTS_DIR="results/${PHASE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}/raw"

CODER_MODEL="${CODER_MODEL:-QuantTrio/Qwen3.5-35B-A3B-AWQ}"
THINKER_MODEL="${THINKER_MODEL:-Qwen/Qwen3.5-27B}"

echo "======================================================================"
echo "Phase 3.2 — LiteLLM routing accuracy"
echo "Proxy   : ${LITELLM_ENDPOINT}"
echo "Coder   : ${VLLM_GPU0_ENDPOINT} (${CODER_MODEL})"
echo "Thinker : ${VLLM_GPU1_ENDPOINT} (${THINKER_MODEL})"
echo "Results : ${RESULTS_DIR}"
echo "======================================================================"

# ── Pre-flight: check both inference endpoints are up ─────────────────────────
for url in "${VLLM_GPU0_ENDPOINT%/v1}/health" "${VLLM_GPU1_ENDPOINT%/v1}/health"; do
    if ! curl -sf "${url}" >/dev/null; then
        echo "[ERROR] Inference endpoint not healthy: ${url}"
        echo "        Deploy coder + thinker first (e.g., just phase3-dual)."
        exit 1
    fi
done

# ── Deploy LiteLLM proxy ──────────────────────────────────────────────────────
echo ""
echo "=== Starting LiteLLM proxy ==="
export PORT_LITELLM
export CODER_MODEL THINKER_MODEL
export PORT_VLLM_GPU0 PORT_VLLM_GPU1

podman compose \
    -f infra/compose/litellm.yaml \
    -p bench-litellm \
    up -d

./infra/scripts/wait-healthy.sh "http://localhost:${PORT_LITELLM}/health" 120

echo ""
echo "=== Running routing accuracy bench ==="

python -m benchmarks.phase3_architecture.bench \
    --mode routing \
    --endpoint "${LITELLM_ENDPOINT}" \
    --tasks benchmarks/phase3_architecture/tasks/routing/ \
    --coder-model-id "${CODER_MODEL}" \
    --thinker-model-id "${THINKER_MODEL}" \
    --results-dir "${RESULTS_DIR}" \
    --label "LiteLLM routing" \
    --engine litellm \
    --notes "Sub-test 3.2: routing accuracy" \
    2>&1 | tee "${RESULTS_DIR}/bench.log"

echo ""
echo "=== Stopping LiteLLM proxy ==="
podman compose -f infra/compose/litellm.yaml -p bench-litellm down 2>/dev/null || true

python -m lib.reporter "${RESULTS_DIR}" --thresholds config/thresholds.yaml
cat "${RESULTS_DIR}/summary.md"

echo ""
echo "======================================================================"
echo "Results: ${RESULTS_DIR}"
echo "routing_accuracy ≥ 0.95 → PASS"
echo "If accuracy is low, review classify_tier() heuristics in bench.py"
echo "and update THINKER_PATTERNS / CODER_PATTERNS accordingly."
echo "======================================================================"
