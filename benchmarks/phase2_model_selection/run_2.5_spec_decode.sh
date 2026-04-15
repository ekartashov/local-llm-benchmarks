#!/usr/bin/env bash
# Sub-test 2.5 — Dense model + speculative decoding vs MoE baseline.
#
# Compares:
#   A) Qwen3-Coder-30B-A3B-AWQ (MoE, Phase 2.1 winner) — baseline, 251 t/s
#   B) Qwen3.5-27B-AWQ (dense, Phase 2.2 winner) WITHOUT spec decode — 76 t/s
#   C) Qwen3.5-27B-AWQ (dense) WITH spec decode (draft: Qwen3.5-1.5B-Instruct)
#
# The interesting question: can spec-decode push the dense thinker above 100 t/s?
# The MoE coder at 251 t/s is already settled — this test is about whether
# spec-decode makes the dense path viable for the coder role too.
#
# Notes:
#   - bf16 27B (50 GiB) and bf16 35B (22 GiB but OOM on graphs) are NOT used.
#     AWQ INT4 variants confirmed working (Phase 2.2, 2026-04-15).
#   - Do NOT test spec decode on MoE (not accelerated in vLLM; crashes on llamacpp).
#   - Dense AWQ + spec decode: vLLM supports speculative decoding with AWQ main model;
#     draft model runs bf16 (1.5B fits in remaining VRAM after 14 GiB AWQ).
#
# Pass criterion (from thresholds.yaml): decode_tps ≥ 100 t/s
# Meaningful result: spec-decode speedup ≥ 1.30 (Phase 4 threshold)
#
# Usage:
#   ./benchmarks/phase2_model_selection/run_2.5_spec_decode.sh
#   SKIP_SPEC=1 ./benchmarks/phase2_model_selection/run_2.5_spec_decode.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ENGINE="vllm"
GPU="gpu0"
# Phase 2.1 winner (MoE coder, no spec decode — not supported on MoE)
MOE_MODEL="${MOE_MODEL:-QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ}"
# Phase 2.2 winner (dense thinker, candidate for spec decode)
DENSE_MODEL="${DENSE_MODEL:-QuantTrio/Qwen3.5-27B-AWQ}"
# Draft model: same family, 1.5B fits in the ~12 GiB headroom after 27B-AWQ loads
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3.5-1.5B-Instruct}"
CTX_LEN=32768
SKIP_SPEC="${SKIP_SPEC:-0}"

# Tasks: reuse Phase 1 throughput tasks (code generation, substantial output)
TASKS="${REPO_ROOT}/benchmarks/phase1_engine_selection/tasks/throughput"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

_run_throughput() {
    local model="$1" quant="$2" label="$3" extra_args="$4"
    local safe_label="${label//[^a-zA-Z0-9_-]/_}"
    local results_dir="${REPO_ROOT}/results/phase2_2.5_${safe_label}_${TIMESTAMP}"
    mkdir -p "${results_dir}/raw"

    echo "" >&2
    echo "── ${label} ──────────────────────────────────────────────────────────" >&2
    "${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${model}" \
        --ctx "${CTX_LEN}" ${extra_args} >&2

    python3 -m benchmarks.phase2_model_selection.bench \
        --endpoint "http://localhost:${PORT_VLLM_GPU0}/v1" \
        --model "${model}" \
        --results-dir "${results_dir}" \
        --tasks "${TASKS}" \
        --mode throughput \
        --label "${label}" \
        --engine "${ENGINE}" \
        --quantization "${quant}" \
        --gpu-label "RTX 5090" \
        --gpu-id "${GPU_0_ID}" \
        --ctx-len "${CTX_LEN}" \
        --extra-args "${extra_args}" \
        --thresholds "${REPO_ROOT}/config/thresholds.yaml" \
        --max-tokens 512 \
        --notes "Sub-test 2.5: spec decode comparison" \
        2>&1 | tee "${results_dir}/bench.log" >&2 || true

    "${REPO_ROOT}/infra/scripts/teardown.sh" >&2
    echo "${results_dir}"
}

echo "=========================================================="
echo " Sub-test 2.5: Dense + spec-decode vs MoE"
echo " MoE (baseline) : ${MOE_MODEL}"
echo " Dense model    : ${DENSE_MODEL}"
echo " Draft model    : ${DRAFT_MODEL}"
echo "=========================================================="

# A: MoE baseline (no spec decode — not supported on MoE in vLLM)
MOE_RESULTS="$(_run_throughput \
    "${MOE_MODEL}" "AWQ-INT4" "MoE-Coder-30B-AWQ" \
    "--tool-call-parser qwen3_coder")"

# B: Dense thinker without spec decode (verified 76 t/s in Phase 2.2)
DENSE_RESULTS="$(_run_throughput \
    "${DENSE_MODEL}" "AWQ-INT4" "Dense-Thinker-27B-AWQ-no-spec" \
    "--tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-num-seqs 1")"

# C: Dense thinker with spec decode
# 27B-AWQ loads 19.78 GiB; 1.5B-Instruct bf16 ≈ 3 GiB; total ≈ 23 GiB — fits in 32 GB.
SPEC_RESULTS=""
if [[ "${SKIP_SPEC}" != "1" ]]; then
    SPEC_RESULTS="$(_run_throughput \
        "${DENSE_MODEL}" "AWQ-INT4" "Dense-Thinker-27B-AWQ-spec" \
        "--tool-call-parser qwen3_coder --reasoning-parser qwen3 --speculative-model ${DRAFT_MODEL} --num-speculative-tokens 5")"
fi

# ── Comparison ─────────────────────────────────────────────────────────────────
echo ""
echo "── Speed: MoE (30B coder) vs Dense (27B thinker) ───────────────────────"
python3 -m lib.reporter compare "${MOE_RESULTS}" "${DENSE_RESULTS}" \
    --key decode_tps_mean || true

if [[ -n "${SPEC_RESULTS}" ]]; then
    echo ""
    echo "── Spec-decode speedup on dense 27B ────────────────────────────────────"
    python3 -m lib.reporter compare "${DENSE_RESULTS}" "${SPEC_RESULTS}" \
        --key decode_tps_mean || true

    python3 - <<PYEOF
import json
dense = json.load(open("${DENSE_RESULTS}/metrics.json"))["metrics"].get("decode_tps_mean", 0)
spec  = json.load(open("${SPEC_RESULTS}/metrics.json"))["metrics"].get("decode_tps_mean", 0)
if dense > 0:
    ratio = spec / dense
    print(f"\nSpec-decode speedup: {ratio:.2f}x  (target ≥1.30 per thresholds.yaml)")
    if ratio >= 1.30:
        print(f"✓ Spec decode useful — 27B+spec at {spec:.0f} t/s")
    else:
        print(f"✗ Spec decode below target — stick with --max-num-seqs 1 baseline ({dense:.0f} t/s)")
PYEOF
fi

echo ""
echo "Results:"
echo "  MoE baseline     : ${MOE_RESULTS}"
echo "  Dense no-spec    : ${DENSE_RESULTS}"
[[ -n "${SPEC_RESULTS}" ]] && echo "  Dense + spec     : ${SPEC_RESULTS}"
