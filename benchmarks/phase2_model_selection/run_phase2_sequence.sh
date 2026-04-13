#!/usr/bin/env bash
# Phase 2 complete sequence — model selection.
#
# Requires Phase 1 to have identified the winning engine.
# Writes phase2_sequence_summary.md with model recommendations.
#
# Execution order:
#   2.1 → Coder quality: Qwen3.5-35B vs Qwen3-Coder-30B
#   2.2 → Thinker: Qwen3.5-27B vs DeepSeek-R1-32B
#   2.3 → Peak mode: Coder-Next vs best daily driver
#   2.4 → Devstral tool-call reliability check
#   2.5 → Dense + spec-decode vs MoE speed comparison
#
# Flags:
#   --skip-thinker     Skip 2.2 (run only coder tests)
#   --skip-peak        Skip 2.3 (no Coder-Next GGUF available)
#   --skip-devstral    Skip 2.4
#   --skip-spec        Skip 2.5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

SKIP_THINKER=false; SKIP_PEAK=false; SKIP_DEVSTRAL=false; SKIP_SPEC=false
for arg in "$@"; do
    case "$arg" in
        --skip-thinker)  SKIP_THINKER=true ;;
        --skip-peak)     SKIP_PEAK=true ;;
        --skip-devstral) SKIP_DEVSTRAL=true ;;
        --skip-spec)     SKIP_SPEC=true ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SEQ_DIR="${REPO_ROOT}/results/phase2_sequence_${TIMESTAMP}"
mkdir -p "${SEQ_DIR}"

declare -A VERDICTS

_run() {
    local label="$1" script="$2"; shift 2
    echo ""; echo "══════════════════════════════════════════════════"
    echo " Running sub-test ${label}"; echo "══════════════════════════════════════════════════"
    if bash "${script}" "$@"; then VERDICTS["${label}"]="DONE"
    else VERDICTS["${label}"]="ERR"; fi
}

_run "2.1" "${REPO_ROOT}/benchmarks/phase2_model_selection/run_2.1_coder_quality.sh"

if [[ "${SKIP_THINKER}" == "false" ]]; then
    _run "2.2" "${REPO_ROOT}/benchmarks/phase2_model_selection/run_2.2_thinker.sh"
else
    VERDICTS["2.2"]="SKIPPED"
fi

if [[ "${SKIP_PEAK}" == "false" ]]; then
    _run "2.3" "${REPO_ROOT}/benchmarks/phase2_model_selection/run_2.3_peak.sh"
else
    VERDICTS["2.3"]="SKIPPED"
fi

if [[ "${SKIP_DEVSTRAL}" == "false" ]]; then
    _run "2.4" "${REPO_ROOT}/benchmarks/phase2_model_selection/run_2.4_devstral.sh" || true
    VERDICTS["2.4"]="${VERDICTS[2.4]:-DONE}"
else
    VERDICTS["2.4"]="SKIPPED"
fi

if [[ "${SKIP_SPEC}" == "false" ]]; then
    _run "2.5" "${REPO_ROOT}/benchmarks/phase2_model_selection/run_2.5_spec_decode.sh"
else
    VERDICTS["2.5"]="SKIPPED"
fi

SUMMARY="${SEQ_DIR}/phase2_sequence_summary.md"
{
    echo "# Phase 2 Sequence Summary"
    echo ""; echo "**Timestamp:** ${TIMESTAMP}"; echo ""
    echo "| Sub-test | Description | Status |"
    echo "|----------|-------------|--------|"
    echo "| 2.1 | Coder quality: Qwen3.5-35B vs Qwen3-Coder-30B | ${VERDICTS[2.1]:-N/A} |"
    echo "| 2.2 | Thinker: Qwen3.5-27B vs DeepSeek-R1-32B | ${VERDICTS[2.2]:-N/A} |"
    echo "| 2.3 | Peak mode: Coder-Next vs daily driver | ${VERDICTS[2.3]:-N/A} |"
    echo "| 2.4 | Devstral tool-call reliability | ${VERDICTS[2.4]:-N/A} |"
    echo "| 2.5 | Dense + spec-decode vs MoE speed | ${VERDICTS[2.5]:-N/A} |"
    echo ""
    echo "## Decisions required (human review)"
    echo ""
    echo "Review the human_review.md files from 2.1 and 2.2, then fill in:"
    echo ""
    echo "| Decision | Question | Your answer |"
    echo "|----------|----------|-------------|"
    echo "| Daily driver coder | Qwen3.5-35B or Qwen3-Coder-30B? | _fill in_ |"
    echo "| Thinker model | Qwen3.5-27B, R1-32B, or none? | _fill in_ |"
    echo "| Peak mode | Worth the complexity (Coder-Next)? | _fill in_ |"
    echo "| Devstral | Add to Phase 3 architecture? | _fill in_ |"
    echo "| Spec decode | Dense > MoE in speed+quality combined? | _fill in_ |"
    echo ""
    echo "## Next step: Phase 3 (architecture)"
    echo ""
    echo "Once decisions are recorded above, run:"
    echo "\`\`\`bash"
    echo "just phase3"
    echo "\`\`\`"
} > "${SUMMARY}"

echo ""; echo "--- Phase 2 Sequence Summary ---"; cat "${SUMMARY}"
