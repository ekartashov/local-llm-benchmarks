#!/usr/bin/env bash
# Phase 1 complete sequence — engine selection.
#
# Requires Phase 0 to have passed first.
# Writes a phase1_sequence_summary.md with the winning engine recommendation.
#
# Execution order:
#   1.1 → vLLM vs SGLang throughput (both at concurrency=4)
#   1.2 → SGLang prefix reuse
#   1.3 → vLLM prefix caching
#   1.4 → llama.cpp baseline (optional — skip with --skip-llamacpp)
#
# Flags:
#   --skip-sglang     Skip SGLang tests (1.1 SGLang + 1.2)
#   --skip-llamacpp   Skip llama.cpp baseline (1.4)
#
# Usage:
#   ./benchmarks/phase1_engine_selection/run_phase1_sequence.sh
#   ./benchmarks/phase1_engine_selection/run_phase1_sequence.sh --skip-llamacpp

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# Activate Python environment: try .venv symlink, then pyenv hf, then fail loud.
if [[ -f "${REPO_ROOT}/.venv/bin/python3" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
elif command -v pyenv &>/dev/null && pyenv activate hf 2>/dev/null; then
    : # activated via pyenv hf
elif ! python3 -c "import openai" &>/dev/null 2>&1; then
    echo "[ERROR] No Python env with openai found. Run: pyenv activate hf" >&2
    exit 1
fi

SKIP_SGLANG=false
SKIP_LLAMACPP=false
for arg in "$@"; do
    case "$arg" in
        --skip-sglang)   SKIP_SGLANG=true ;;
        --skip-llamacpp) SKIP_LLAMACPP=true ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SEQ_DIR="${REPO_ROOT}/results/phase1_sequence_${TIMESTAMP}"
mkdir -p "${SEQ_DIR}"

declare -A VERDICTS

_run() {
    local label="$1" script="$2"
    shift 2
    echo ""
    echo "══════════════════════════════════════════════════"
    echo " Running sub-test ${label}"
    echo "══════════════════════════════════════════════════"
    if bash "${script}" "$@"; then
        VERDICTS["${label}"]="PASS"
    else
        VERDICTS["${label}"]="FAIL"
    fi
}

# ── 1.1: Throughput comparison ─────────────────────────────────────────────────
if [[ "${SKIP_SGLANG}" == "true" ]]; then
    SKIP_SGLANG=1 _run "1.1" "${REPO_ROOT}/benchmarks/phase1_engine_selection/run_1.1_throughput.sh"
else
    _run "1.1" "${REPO_ROOT}/benchmarks/phase1_engine_selection/run_1.1_throughput.sh"
fi

# ── 1.2: SGLang prefix reuse ───────────────────────────────────────────────────
if [[ "${SKIP_SGLANG}" == "true" ]]; then
    VERDICTS["1.2"]="SKIPPED"
else
    _run "1.2" "${REPO_ROOT}/benchmarks/phase1_engine_selection/run_1.2_sglang_prefix.sh"
fi

# ── 1.3: vLLM prefix caching ──────────────────────────────────────────────────
_run "1.3" "${REPO_ROOT}/benchmarks/phase1_engine_selection/run_1.3_vllm_prefix.sh"

# ── 1.4: llama.cpp baseline ───────────────────────────────────────────────────
if [[ "${SKIP_LLAMACPP}" == "true" ]]; then
    VERDICTS["1.4"]="SKIPPED"
else
    _run "1.4" "${REPO_ROOT}/benchmarks/phase1_engine_selection/run_1.4_llamacpp.sh" || true
    # 1.4 failure doesn't block — it's informational
    VERDICTS["1.4"]="${VERDICTS[1.4]:-DONE}"
fi

# ── Write sequence summary ─────────────────────────────────────────────────────
SUMMARY="${SEQ_DIR}/phase1_sequence_summary.md"
{
    echo "# Phase 1 Sequence Summary"
    echo ""
    echo "**Timestamp:** ${TIMESTAMP}"
    echo ""
    echo "| Sub-test | Description | Verdict |"
    echo "|----------|-------------|---------|"
    echo "| 1.1 | vLLM vs SGLang throughput | ${VERDICTS[1.1]:-N/A} |"
    echo "| 1.2 | SGLang prefix reuse | ${VERDICTS[1.2]:-N/A} |"
    echo "| 1.3 | vLLM prefix caching | ${VERDICTS[1.3]:-N/A} |"
    echo "| 1.4 | llama.cpp baseline | ${VERDICTS[1.4]:-N/A} |"
    echo ""

    # Decision logic
    VLLM_OK="${VERDICTS[1.1]:-FAIL}"
    VLLM_PREFIX_OK="${VERDICTS[1.3]:-FAIL}"
    SGLANG_OK="${VERDICTS[1.2]:-SKIPPED}"

    echo "## Engine decision"
    echo ""
    if [[ "${VLLM_OK}" == "PASS" && "${VLLM_PREFIX_OK}" == "PASS" ]]; then
        echo "**Recommendation: vLLM** — passes throughput and prefix caching."
        echo ""
        echo "Proceed to Phase 2:"
        echo "\`\`\`bash"
        echo "just phase2"
        echo "\`\`\`"
    elif [[ "${SGLANG_OK}" == "PASS" ]]; then
        echo "**Recommendation: SGLang** — vLLM failed but SGLang passes."
        echo ""
        echo "Update Phase 2 scripts to use SGLang endpoint, then:"
        echo "\`\`\`bash"
        echo "just phase2"
        echo "\`\`\`"
    else
        echo "**Recommendation: inconclusive** — review per-sub-test results above."
        echo "Check raw/ output directories for error details."
    fi
} > "${SUMMARY}"

echo ""
echo "--- Phase 1 Sequence Summary ---"
cat "${SUMMARY}"
