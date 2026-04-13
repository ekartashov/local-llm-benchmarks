#!/usr/bin/env bash
# Sub-test 2.5 — Dense model + speculative decoding vs MoE baseline.
#
# Compares:
#   A) Qwen3.5-35B-A3B-AWQ (MoE) — baseline, no spec decode
#   B) Qwen3.5-27B bf16 (dense) WITHOUT spec decode
#   C) Qwen3.5-27B bf16 (dense) WITH spec decode
#
# This answers: can a dense model + spec-decode match the MoE in speed
# while potentially offering better quality (dense models often > MoE at same params)?
#
# Spec-decode config: draft model = Qwen3.5-1.5B-Instruct (fast, same family)
# Per CLAUDE.md: do NOT test spec decode on MoE (crashes on llama.cpp PR #20075,
# and vLLM doesn't accelerate MoE with spec decode anyway).
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
DENSE_MODEL="Qwen/Qwen3.5-27B"
MOE_MODEL="Qwen/Qwen3.5-35B-A3B-AWQ"
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

    echo ""
    echo "── ${label} ──────────────────────────────────────────────────────────"
    "${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${model}" \
        --ctx "${CTX_LEN}" ${extra_args}

    python -m benchmarks.phase2_model_selection.bench \
        --endpoint "http://localhost:${PORT_VLLM_GPU0}/v1" \
        --results-dir "${results_dir}" \
        --tasks "${TASKS}" \
        --mode spec-decode \
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
        2>&1 | tee "${results_dir}/bench.log" || true

    "${REPO_ROOT}/infra/scripts/teardown.sh"
    echo "${results_dir}"
}

echo "=========================================================="
echo " Sub-test 2.5: Dense + spec-decode vs MoE"
echo " Dense model : ${DENSE_MODEL}"
echo " MoE model   : ${MOE_MODEL}"
echo " Draft model : ${DRAFT_MODEL}"
echo "=========================================================="

# A: MoE baseline (no spec decode — not supported on MoE)
MOE_RESULTS="$(_run_throughput \
    "${MOE_MODEL}" "AWQ-INT4" "MoE-Qwen35-35B-AWQ" \
    "--tool-call-parser qwen3_coder")"

# B: Dense without spec decode
DENSE_RESULTS="$(_run_throughput \
    "${DENSE_MODEL}" "bf16" "Dense-Qwen35-27B-no-spec" \
    "--reasoning-parser qwen3")"

# C: Dense with spec decode
SPEC_RESULTS=""
if [[ "${SKIP_SPEC}" != "1" ]]; then
    SPEC_RESULTS="$(_run_throughput \
        "${DENSE_MODEL}" "bf16" "Dense-Qwen35-27B-spec" \
        "--reasoning-parser qwen3 --speculative-model ${DRAFT_MODEL} --num-speculative-tokens 5")"
fi

# ── Comparison ─────────────────────────────────────────────────────────────────
echo ""
echo "── Speed comparison (MoE vs Dense vs Dense+spec) ────────────────────────"
python -m lib.reporter compare "${MOE_RESULTS}" "${DENSE_RESULTS}" \
    --key decode_tps_mean || true

if [[ -n "${SPEC_RESULTS}" ]]; then
    echo ""
    python -m lib.reporter compare "${DENSE_RESULTS}" "${SPEC_RESULTS}" \
        --key decode_tps_mean || true

    # Compute speedup ratio
    python3 - <<PYEOF
import json
dense = json.load(open("${DENSE_RESULTS}/metrics.json"))["metrics"].get("decode_tps_mean", 0)
spec  = json.load(open("${SPEC_RESULTS}/metrics.json"))["metrics"].get("decode_tps_mean", 0)
if dense > 0:
    ratio = spec / dense
    print(f"\nSpec-decode speedup: {ratio:.2f}x  (target ≥1.30)")
    print("✓ Spec decode accelerates dense model." if ratio >= 1.30 else "✗ Spec decode below target speedup.")
PYEOF
fi

echo ""
echo "Results:"
echo "  MoE baseline : ${MOE_RESULTS}"
echo "  Dense no-spec: ${DENSE_RESULTS}"
[[ -n "${SPEC_RESULTS}" ]] && echo "  Dense + spec : ${SPEC_RESULTS}"
