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

# Run full Phase 0 (tool-call reliability) — BLOCKER
phase0:
    ./benchmarks/phase0_tool_reliability/run.sh

# Run full Phase 1 (engine selection)
phase1:
    ./benchmarks/phase1_engine_selection/run.sh

# Run full Phase 2 (model selection)
phase2:
    ./benchmarks/phase2_model_selection/run.sh

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
