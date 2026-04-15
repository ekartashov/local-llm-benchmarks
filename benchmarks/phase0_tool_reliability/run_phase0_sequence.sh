#!/usr/bin/env bash
# Phase 0 complete sequence — orchestrates all sub-tests in the correct order.
#
# Execution order (per CLAUDE.md):
#   0.4 → chat template verification (fast gate — abort if broken)
#   0.1 → Qwen3.5-35B-A3B-AWQ on vLLM (primary)
#   0.2 → Qwen3.5-35B-A3B-AWQ on SGLang (optional; validates engine for Phase 1)
#   0.3 → Qwen3-Coder-Next on llamacpp  (fallback; only if 0.1 fails)
#
# This script writes a phase0_sequence_summary.md to results/ at the end.
#
# Flags:
#   --skip-sglang    Skip sub-test 0.2 (saves ~30 min if you already know SGLang works)
#   --skip-template  Skip sub-test 0.4 (dangerous — only if you've already verified)
#
# Usage:
#   ./benchmarks/phase0_tool_reliability/run_phase0_sequence.sh
#   ./benchmarks/phase0_tool_reliability/run_phase0_sequence.sh --skip-sglang

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
SKIP_TEMPLATE=false
for arg in "$@"; do
    case "$arg" in
        --skip-sglang)   SKIP_SGLANG=true ;;
        --skip-template) SKIP_TEMPLATE=true ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SEQ_DIR="${REPO_ROOT}/results/phase0_sequence_${TIMESTAMP}"
mkdir -p "${SEQ_DIR}"

# Track outcomes
declare -A VERDICTS
declare -A RESULT_DIRS

_write_summary() {
    local summary="${SEQ_DIR}/phase0_sequence_summary.md"
    {
        echo "# Phase 0 Sequence Summary"
        echo ""
        echo "**Timestamp:** ${TIMESTAMP}"
        echo ""
        echo "| Sub-test | Description | Verdict |"
        echo "|----------|-------------|---------|"
        echo "| 0.4 | Chat template verification | ${VERDICTS[0.4]:-N/A} |"
        echo "| 0.1 | Qwen3.5-35B-A3B on vLLM (primary) | ${VERDICTS[0.1]:-N/A} |"
        echo "| 0.2 | Qwen3.5-35B-A3B on SGLang | ${VERDICTS[0.2]:-N/A} |"
        echo "| 0.3 | Qwen3-Coder-Next on llamacpp (fallback) | ${VERDICTS[0.3]:-N/A} |"
        echo ""
        # Determine overall outcome and next step
        if [[ "${VERDICTS[0.1]:-}" == "PASS" ]]; then
            echo "## Overall: PASS"
            echo ""
            echo "**Winner:** Qwen3.5-35B-A3B-AWQ on vLLM"
            echo ""
            echo "**Next step:** Run Phase 1 (engine selection)"
            echo "\`\`\`bash"
            echo "just phase1"
            echo "\`\`\`"
            if [[ "${VERDICTS[0.2]:-}" == "PASS" ]]; then
                echo ""
                echo "SGLang also passed — Phase 1.1 (vLLM vs SGLang throughput) is a valid comparison."
            elif [[ "${VERDICTS[0.2]:-}" == "FAIL" ]]; then
                echo ""
                echo "SGLang failed Phase 0 — **eliminate SGLang from Phase 1**. Run vLLM-only Phase 1."
            fi
        elif [[ "${VERDICTS[0.3]:-}" == "PASS" ]]; then
            echo "## Overall: PASS (via fallback)"
            echo ""
            echo "**Winner:** Qwen3-Coder-Next (GGUF Q4) on llama.cpp"
            echo ""
            echo "**Next step:** Update Phase 1 scripts to use llamacpp, then run Phase 1."
        else
            echo "## Overall: FAIL"
            echo ""
            echo "Both primary (0.1) and fallback (0.3) failed."
            echo "Investigate raw/ outputs in the results directories above."
        fi
    } > "${summary}"
    echo ""
    echo "--- Phase 0 Sequence Summary ---"
    cat "${summary}"
}

_run_sub_test() {
    local label="$1"
    local script="$2"
    shift 2
    echo ""
    echo "══════════════════════════════════════════════════"
    echo " Running sub-test ${label}"
    echo "══════════════════════════════════════════════════"
    if bash "${script}" "$@"; then
        VERDICTS["${label}"]="PASS"
    else
        local exit_code=$?
        if [[ "${exit_code}" -eq 2 ]]; then
            VERDICTS["${label}"]="UNREACHABLE"
        else
            VERDICTS["${label}"]="FAIL"
        fi
    fi
    # Find most recent results dir for this sub-test and record it
    local latest
    latest="$(ls -td "${REPO_ROOT}/results/phase0_tool_reliability_"* 2>/dev/null | head -1 || echo "")"
    RESULT_DIRS["${label}"]="${latest}"
}

# ── 0.4: Chat template verification ───────────────────────────────────────────
if [[ "${SKIP_TEMPLATE}" == "false" ]]; then
    _run_sub_test "0.4" \
        "${REPO_ROOT}/benchmarks/phase0_tool_reliability/run_0.4_chat_template.sh"

    if [[ "${VERDICTS[0.4]}" != "PASS" ]]; then
        echo ""
        echo "✗ Sub-test 0.4 FAILED or was unreachable."
        echo "  Aborting sequence — fix the chat template issues before continuing."
        echo "  See remediation advice in the output above."
        _write_summary
        exit 1
    fi
    echo "✓ 0.4 passed — continuing to 0.1"
else
    echo "⚠ Skipping sub-test 0.4 (--skip-template was set)"
    VERDICTS["0.4"]="SKIPPED"
fi

# ── 0.1: Primary (vLLM) ───────────────────────────────────────────────────────
_run_sub_test "0.1" \
    "${REPO_ROOT}/benchmarks/phase0_tool_reliability/run_0.1_vllm.sh"

# ── 0.2: SGLang validation (optional) ─────────────────────────────────────────
if [[ "${SKIP_SGLANG}" == "false" ]]; then
    _run_sub_test "0.2" \
        "${REPO_ROOT}/benchmarks/phase0_tool_reliability/run_0.2_sglang.sh"
else
    echo ""
    echo "⚠ Skipping sub-test 0.2 (--skip-sglang was set)"
    VERDICTS["0.2"]="SKIPPED"
fi

# ── 0.3: Fallback (only if 0.1 failed) ────────────────────────────────────────
if [[ "${VERDICTS[0.1]}" == "FAIL" ]]; then
    echo ""
    echo "Sub-test 0.1 failed — running fallback (sub-test 0.3)..."
    _run_sub_test "0.3" \
        "${REPO_ROOT}/benchmarks/phase0_tool_reliability/run_0.3_fallback.sh"
else
    VERDICTS["0.3"]="SKIPPED (0.1 passed)"
fi

# ── Write sequence summary ─────────────────────────────────────────────────────
_write_summary
