# local-llm-benchmarks task runner
# Usage: just <recipe>

set shell := ["bash", "-euo", "pipefail", "-c"]

# Default: list recipes
default:
    @just --list

# ── Python env ────────────────────────────────────────────────────────────────

# Install all dependencies (including dev)
install:
    uv sync --extra dev

# Lint
lint:
    ruff check .

# Format
fmt:
    ruff format .

# Type check
typecheck:
    pyright

# Run all tests
test:
    pytest -v

# Run a single test file
test-file file:
    pytest -v {{ file }}

# ── Infra ─────────────────────────────────────────────────────────────────────

# Deploy engine+model. Usage: just deploy vllm gpu0 Qwen/Qwen3.5-35B-A3B-AWQ --ctx 32768
deploy engine gpu model *args:
    ./infra/scripts/deploy.sh {{ engine }} {{ gpu }} {{ model }} {{ args }}

# Tear down all running inference containers
teardown:
    ./infra/scripts/teardown.sh

# Pre-download and page-cache all benchmark models
precache:
    ./infra/scripts/precache-models.sh

# ── Phase runners ─────────────────────────────────────────────────────────────

# Run full Phase 0 sequence: 0.4 → 0.1 → 0.2 → 0.3 (if needed) — BLOCKER
phase0:
    ./benchmarks/phase0_tool_reliability/run_phase0_sequence.sh

# Sub-test 0.4: chat template verification (run FIRST — fast gate)
phase0-verify:
    ./benchmarks/phase0_tool_reliability/run_0.4_chat_template.sh

# Sub-test 0.1: Qwen3.5-35B-A3B on vLLM (primary)
phase0-vllm:
    ./benchmarks/phase0_tool_reliability/run_0.1_vllm.sh

# Sub-test 0.2: Qwen3.5-35B-A3B on SGLang
phase0-sglang:
    ./benchmarks/phase0_tool_reliability/run_0.2_sglang.sh

# Sub-test 0.3: Qwen3-Coder-Next on llamacpp (fallback — only if 0.1 fails)
phase0-fallback:
    ./benchmarks/phase0_tool_reliability/run_0.3_fallback.sh

# Run full Phase 1 sequence (1.1→1.2→1.3→1.4)
phase1:
    ./benchmarks/phase1_engine_selection/run_phase1_sequence.sh

# Sub-test 1.1: vLLM vs SGLang throughput at concurrency=4
phase1-throughput:
    ./benchmarks/phase1_engine_selection/run_1.1_throughput.sh

# Sub-test 1.2: SGLang prefix reuse (RadixAttention)
phase1-sglang-prefix:
    ./benchmarks/phase1_engine_selection/run_1.2_sglang_prefix.sh

# Sub-test 1.3: vLLM prefix caching
phase1-vllm-prefix:
    ./benchmarks/phase1_engine_selection/run_1.3_vllm_prefix.sh

# Sub-test 1.4: llama.cpp throughput baseline
phase1-llamacpp:
    ./benchmarks/phase1_engine_selection/run_1.4_llamacpp.sh

# Run full Phase 2 sequence (2.1→2.2→2.3→2.4→2.5)
phase2:
    ./benchmarks/phase2_model_selection/run_phase2_sequence.sh

# Sub-test 2.1: coder quality — Qwen3.5-35B vs Qwen3-Coder-30B
phase2-coder:
    ./benchmarks/phase2_model_selection/run_2.1_coder_quality.sh

# Sub-test 2.2: thinker — Qwen3.5-27B vs DeepSeek-R1-32B
phase2-thinker:
    ./benchmarks/phase2_model_selection/run_2.2_thinker.sh

# Sub-test 2.3: peak mode — Coder-Next vs daily driver
phase2-peak:
    ./benchmarks/phase2_model_selection/run_2.3_peak.sh

# Sub-test 2.4: Devstral tool-call reliability
phase2-devstral:
    ./benchmarks/phase2_model_selection/run_2.4_devstral.sh

# Sub-test 2.5: dense + spec-decode vs MoE speed
phase2-spec:
    ./benchmarks/phase2_model_selection/run_2.5_spec_decode.sh

# Run full Phase 3 (architecture)
phase3:
    ./benchmarks/phase3_architecture/run.sh

# Run full Phase 4 (optimizations)
phase4:
    ./benchmarks/phase4_optimizations/run.sh

# Run full Phase 5 (integration)
phase5:
    ./benchmarks/phase5_integration/run.sh

# Run full Phase 6 (baselines)
phase6:
    ./benchmarks/phase6_baselines/run.sh

# ── Debug helpers ─────────────────────────────────────────────────────────────

# Run a single Phase 0 task against a live endpoint.
# Usage: just debug-task 01_read_file http://localhost:30000/v1
debug-task task endpoint="http://localhost:30000/v1":
    python -m benchmarks.phase0_tool_reliability.bench \
        --endpoint {{ endpoint }} \
        --results-dir results/debug_$(date +%s) \
        --tasks benchmarks/phase0_tool_reliability/tasks/ \
        --task-filter {{ task }}

# Compare two result directories on a given metric key
compare dir-a dir-b key="tool_call_success_rate":
    python -m lib.reporter compare {{ dir-a }} {{ dir-b }} --key {{ key }}

# Show all summaries for a phase (e.g. phase0)
summaries phase:
    @ls results/{{ phase }}*/summary.md 2>/dev/null || echo "No results for {{ phase }} yet"
