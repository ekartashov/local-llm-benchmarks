# local-llm-benchmarks task runner
# Usage: just <recipe>
#
# Two generations of targets:
# 1. Queue items (T1.1, T1.2, ...) — current plan, see TESTING_QUEUE.md
# 2. Legacy phase runners (phase0, phase1, ...) — historical, kept for rerunning prior tests
#
# When in doubt, look at TESTING_QUEUE.md and run the queue-item target.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# ── Python env ────────────────────────────────────────────────────────────────

install:
    uv sync --extra dev

lint:
    ruff check .

fmt:
    ruff format .

typecheck:
    pyright

test:
    pytest -v

test-file file:
    pytest -v {{ file }}

# ── Infra ─────────────────────────────────────────────────────────────────────

# Deploy engine+model. Usage: just deploy vllm tp2 cpatonn/GLM-4.5-Air-AWQ-4bit --ctx 32768
deploy engine placement model *args:
    ./infra/scripts/deploy.sh {{ engine }} {{ placement }} {{ model }} {{ args }}

teardown:
    ./infra/scripts/teardown.sh

precache:
    ./infra/scripts/precache-models.sh

# Diagnose GPU and container stack state (NVIDIA_VISIBLE_DEVICES, nvidia-smi, podman)
diagnose:
    ./infra/scripts/diagnose-gpu.sh

# ── Queue items (current plan — see TESTING_QUEUE.md) ─────────────────────────

# Tier 1 — architecture-defining

t1.1:
    ./benchmarks/queue/T1.1_sleep_mode.sh

t1.2:
    ./benchmarks/queue/T1.2_concurrent_dual_process.sh

t1.2a:
    ./benchmarks/queue/T1.2a_tp1_per_gpu.sh

t1.3:
    ./benchmarks/queue/T1.3_coder_next_tp2.sh

t1.4:
    ./benchmarks/queue/T1.4_th03_budget_fix.sh

# Tier 2 — model shortlist

t2.1:
    ./benchmarks/queue/T2.1_glm47_mla_verify.sh

t2.2:
    ./benchmarks/queue/T2.2_coder_shootout.sh

t2.3:
    ./benchmarks/queue/T2.3_thinker_shootout.sh

t2.4:
    ./benchmarks/queue/T2.4_behemoth_quality.sh

# Tier 3 — optimization axes

t3.1:
    ./benchmarks/queue/T3.1_kv_cache_q8_q4.sh

t3.2:
    ./benchmarks/queue/T3.2_mtp_sm120_glm47.sh

t3.3:
    ./benchmarks/queue/T3.3_mtp_sm120_qwen3next.sh

t3.4:
    ./benchmarks/queue/T3.4_prefix_cache_across_sleep.sh

# Tier 5 — integration

t5.1:
    ./benchmarks/queue/T5.1_opencode_routing.sh

# Show what queue items have results
queue-status:
    @echo "Results per queue item:"
    @for d in results/T*_*/; do \
        if [ -d "$$d" ]; then \
            echo "  $$d"; \
        fi \
    done

# ── Debug helpers ─────────────────────────────────────────────────────────────

# Run a single tool-call task against a live endpoint.
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

# Show summaries matching a prefix (queue items or legacy phases)
summaries prefix:
    @ls results/{{ prefix }}*/summary.md 2>/dev/null || echo "No results for {{ prefix }}"

# ── Legacy phase runners (historical — see PHASE2_RESULTS.md) ─────────────────
# These reuse the existing run_*.sh scripts. Still work; their results inform
# but do not drive current decisions. Prefer queue items above for new runs.

phase0:
    ./benchmarks/phase0_tool_reliability/run_phase0_sequence.sh

phase0-verify:
    ./benchmarks/phase0_tool_reliability/run_0.4_chat_template.sh

phase0-vllm:
    ./benchmarks/phase0_tool_reliability/run_0.1_vllm.sh

phase0-sglang:
    ./benchmarks/phase0_tool_reliability/run_0.2_sglang.sh

phase0-fallback:
    ./benchmarks/phase0_tool_reliability/run_0.3_fallback.sh

phase1:
    ./benchmarks/phase1_engine_selection/run_phase1_sequence.sh

phase1-throughput:
    ./benchmarks/phase1_engine_selection/run_1.1_throughput.sh

phase1-sglang-prefix:
    ./benchmarks/phase1_engine_selection/run_1.2_sglang_prefix.sh

phase1-vllm-prefix:
    ./benchmarks/phase1_engine_selection/run_1.3_vllm_prefix.sh

phase1-llamacpp:
    ./benchmarks/phase1_engine_selection/run_1.4_llamacpp.sh

phase2:
    ./benchmarks/phase2_model_selection/run_phase2_sequence.sh

phase2-coder:
    ./benchmarks/phase2_model_selection/run_2.1_coder_quality.sh

phase2-thinker:
    ./benchmarks/phase2_model_selection/run_2.2_thinker.sh

phase2-peak:
    ./benchmarks/phase2_model_selection/run_2.3_peak.sh

phase2-devstral:
    ./benchmarks/phase2_model_selection/run_2.4_devstral.sh

phase2-spec:
    ./benchmarks/phase2_model_selection/run_2.5_spec_decode.sh

phase3:
    ./benchmarks/phase3_architecture/run.sh
