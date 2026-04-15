# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A systematic benchmark suite for validating local LLM inference configurations on a dual RTX 5090 workstation. The goal is to convert architectural hypotheses into measured, reproducible results that determine the optimal model + engine + quantization + agent topology for autonomous coding workflows.

**This is NOT a general benchmarking framework.** It is a specific, opinionated test harness for ONE hardware configuration and ONE use case (coding agents like OpenCode/aider). Every test has a concrete hypothesis, a pass/fail criterion, and an "if it fails" action.

**Bootstrap status:** As of April 2026, the repository contains only this CLAUDE.md and configuration docs. All files listed in the project structure below must be created. When implementing, follow the conventions here exactly.

## Execution environment

Claude runs inside a rootless Podman container. The target repository is bind-mounted read-write at `/workspace` (the project root). Persistent Claude state lives under `/home/node/.claude`. Host paths outside the mount are unavailable. Use repo-relative or `/workspace/...` paths everywhere.

## Hardware under test

```
CPU:    Intel Core i9-14900K (24 cores, NO AMX, Raptor Lake)
RAM:    192 GB DDR5 (4×48 GB, ~83 GB/s actual due to 4-DIMM downclock)
GPU 0:  NVIDIA RTX 5090 32 GB GDDR7 (1790 GB/s, PCIe 5.0 x8)
GPU 1:  NVIDIA RTX 5090 32 GB GDDR7 (1790 GB/s, PCIe 5.0 x8)
Inter:  NO NVLink — PCIe x8/x8 bifurcation
OS:     Linux (kernel 6.x, rootless podman, NVIDIA container toolkit)
```

## Python environment

**There are two separate Python contexts. Do not confuse them.**

### 1. Host — running benchmark scripts

Benchmark scripts (`run_*.sh`) call `python3 -m benchmarks...` and must be run
on the **host** with the `hf` pyenv virtualenv active:

```bash
pyenv activate hf
# Now python3, pytest, ruff, pyright, and the hf CLI are all on PATH.
```

The `.venv` symlink in the repo root points to this same env:
```
.venv -> /home/cassini/.pyenv/versions/3.12.7/envs/hf
```
So `source .venv/bin/activate` works too, but `pyenv activate hf` is canonical.

First-time setup (installs repo deps into the hf venv):
```bash
pyenv activate hf
pip install -e ".[dev]"
```

### 2. Claude container — editing/searching code

Claude runs inside a rootless Podman container (see CLAUDE.local.md). The host
`hf` venv and `/home/cassini` are **not mounted** — Claude cannot activate it
or run scripts against the live GPU. All Claude does is read/write repo files.
Scripts that deploy containers, run bench.py, etc. must be run by the **human**
on the host terminal.

### HuggingFace model downloads (host only)

```bash
pyenv activate hf
HF_HOME=/srv/ai/models hf download QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ
# For gated repos:
HF_TOKEN=hf_... HF_HOME=/srv/ai/models hf download mistralai/Devstral-Small-2505
```

Do NOT use a Python container for downloads — `pip install huggingface_hub` in a
throwaway container has repeated friction (PATH, HOME, token forwarding). The
host `hf` venv already has everything needed.

## Development commands

```bash
# (All commands run on host with pyenv activate hf)

# Install / sync deps
pip install -e ".[dev]"

# Lint and format
ruff check .
ruff format .

# Type check
pyright

# Run tests
pytest

# Run a single test file
pytest benchmarks/phase0_tool_reliability/tests/test_scorer.py -v
```

Shell scripts are run directly from the repo root on the host:
```bash
./infra/scripts/deploy.sh vllm gpu0 Qwen/Qwen3.5-35B-A3B-AWQ --ctx 114688
./infra/scripts/teardown.sh
./infra/scripts/wait-healthy.sh http://localhost:30000/health
```

## Project structure

```
local-llm-benchmarks/
├── config/
│   ├── hardware.env       # GPU IDs, RAM, MODEL_CACHE path — source this everywhere
│   ├── models.yaml        # Model registry: HF repo, quant, format, VRAM estimate
│   └── thresholds.yaml    # Pass/fail criteria for every test
├── infra/
│   ├── compose/           # One yaml per engine+GPU combo (vllm-gpu0, sglang-gpu0, etc.)
│   ├── scripts/           # deploy.sh, teardown.sh, wait-healthy.sh, swap-model.sh, precache-models.sh
│   └── Containerfile.bench
├── benchmarks/
│   ├── phase0_tool_reliability/   # BLOCKER — run first
│   ├── phase1_engine_selection/
│   ├── phase2_model_selection/
│   ├── phase3_architecture/
│   ├── phase4_optimizations/
│   ├── phase5_integration/
│   └── phase6_baselines/
├── lib/
│   ├── client.py     # BenchClient — OpenAI-compatible client with TTFT/decode instrumentation
│   ├── metrics.py    # TTFT, decode speed, throughput calculators
│   ├── scorer.py     # ToolCallScore enum and scoring logic
│   ├── reporter.py   # Generates summary.md from metrics.json; handles compare subcommand
│   ├── tool_tasks.py # Task loader and validator
│   └── deploy.py     # Python wrapper around deploy.sh
├── results/          # Auto-created per run: {phase}_{timestamp}/raw/, metrics.json, summary.md
├── pyproject.toml    # deps: openai, httpx, pyyaml, pydantic, rich, pytest, ruff, pyright
└── justfile
```

## Critical conventions

### Container orchestration

All inference engines run as **rootless podman containers** via `podman compose`. Never install vLLM/SGLang/llama.cpp on the host. Always use the deploy script — never raw podman commands. Scripts use `podman compose`, not `docker compose`.

**GPU isolation**: Use `NVIDIA_VISIBLE_DEVICES` in compose files, not `CUDA_VISIBLE_DEVICES`.

```yaml
# Canonical compose snippet (vLLM on GPU 0)
services:
  vllm-gpu0:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    command: >
      --model ${MODEL_ID}
      --port 8000
      --gpu-memory-utilization 0.9
      --max-model-len ${CTX_LEN}
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
    ports:
      - "30000:8000"
    volumes:
      - ${MODEL_CACHE}:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### Benchmark execution pattern

Every `benchmarks/phaseN/run.sh` follows:

```bash
set -euo pipefail
source config/hardware.env
RESULTS_DIR="results/${PHASE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR/raw"

./infra/scripts/deploy.sh "$ENGINE" "$GPU" "$MODEL" $ENGINE_ARGS
python -m benchmarks.${PHASE}.bench \
  --endpoint "http://localhost:${PORT}/v1" \
  --results-dir "$RESULTS_DIR" \
  --tasks benchmarks/${PHASE}/tasks/ \
  2>&1 | tee "$RESULTS_DIR/bench.log"
./infra/scripts/teardown.sh
python -m lib.reporter "$RESULTS_DIR" --thresholds config/thresholds.yaml
cat "$RESULTS_DIR/summary.md"
```

### Metric collection rules

All benchmark scripts MUST:
1. Save raw API responses as JSON in `$RESULTS_DIR/raw/`
2. Write computed metrics to `$RESULTS_DIR/metrics.json`
3. Use `lib.client.BenchClient` (handles TTFT and decode-speed instrumentation automatically)
4. Never use `print()` for metrics — only for progress. Structured data goes to files.

### `BenchClient` interface

```python
class BenchClient:
    def __init__(self, base_url: str, results_dir: Path): ...

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        task_id: str = "",
        temperature: float = 0.0,
    ) -> BenchResult:
        # Returns: .ttft_ms, .decode_tps, .total_tokens,
        #          .tool_calls, .raw_text, .error, .latency_ms
```

### Tool-call task JSON schema

```json
{
  "id": "01_read_file",
  "description": "...",
  "system_prompt": "...",
  "user_message": "...",
  "tools": [ /* standard OpenAI function-tool objects */ ],
  "expected_tool_calls": [
    { "name": "read_file", "must_include_args": ["path"] }
  ],
  "mock_tool_response": "...",
  "difficulty": "simple | medium | hard",
  "category": "file_read | file_write | search | multi_step | ..."
}
```

`bench.py` auto-discovers all `.json` files in `tasks/`. Use `--task-filter <id>` for single-task debugging.

### Scoring (`lib/scorer.py`)

```python
class ToolCallScore:
    PASS         = "pass"          # Correct tool call, parsed successfully
    FORMAT_ERROR = "format_error"  # Engine couldn't parse the model's output
    DROPPED      = "dropped"       # Engine silently returned text-only
    WRONG_TOOL   = "wrong_tool"    # Parsed but wrong tool name
    WRONG_ARGS   = "wrong_args"    # Right tool, missing/wrong arguments
    NO_CALL      = "no_call"       # Model made no tool call attempt
    EXCEPTION    = "exception"     # Timeout, 400, 500
```

`DROPPED` or `FORMAT_ERROR` means the **engine parser is broken**, not the model.

**Phase 0 pass criterion:** ≥95% of 30 tasks score PASS.

### `metrics.json` schema

```json
{
  "phase": "phase0_tool_reliability",
  "timestamp": "2026-04-14T10:30:00Z",
  "config": {
    "engine": "vllm",
    "engine_version": "0.19.2",
    "model": "Qwen/Qwen3.5-35B-A3B-AWQ",
    "quantization": "AWQ-INT4",
    "gpu": "RTX 5090",
    "gpu_id": 0,
    "context_length": 114688,
    "extra_args": "--tool-call-parser qwen3_coder"
  },
  "metrics": {},
  "verdict": "PASS | FAIL | INCONCLUSIVE",
  "notes": ""
}
```

## Phase execution order and dependencies

```
Phase 0: Tool-call reliability          ← BLOCKER, run first
  ├── 0.4: Chat template verification   ← run first within phase (~30 min)
  ├── 0.1: Qwen3.5-35B-A3B on vLLM     ← if fails → test 0.3
  ├── 0.2: Qwen3.5-35B-A3B on SGLang   ← if fails → eliminates SGLang for this model
  └── 0.3: Qwen3-Coder-Next on vLLM    ← fallback if 0.1 fails

Phase 1: Engine selection               ← Needs: Phase 0 winner
  ├── 1.1: vLLM vs SGLang throughput
  ├── 1.2: SGLang prefix reuse
  ├── 1.3: vLLM prefix caching
  └── 1.4: llama.cpp comparison

Phase 2: Model selection                ← Needs: Phase 1 winner
  ├── 2.1: Qwen3.5-35B vs Qwen3-Coder-30B (quality)
  ├── 2.2: Thinker: Qwen3.5-27B vs R1-32B
  ├── 2.3: Peak mode: Coder-Next vs best daily driver
  ├── 2.4: Devstral tool-call reliability
  └── 2.5: Dense + spec-decode vs MoE (speed)

Phase 3: Architecture                   ← Needs: Phase 2 winners
  ├── 3.1: Co-resident dual-model vs single Coder-Next (4 days)
  ├── 3.2: LiteLLM routing accuracy (passive, runs during 3.1)
  ├── 3.3: Mode swap timing
  └── 3.4: Parallel swarm test

Phase 4: Optimizations                  ← Can run in parallel with Phase 3
  ├── 4.1: Unsloth UD vs standard quant quality
  ├── 4.2: TurboQuant KV cache
  ├── 4.3: Speculative decode on dense thinker
  ├── 4.4: AWQ vs GGUF speed
  └── 4.5: Context length OOM boundary

Phase 5: Integration                    ← Needs: Phase 3 decision
  ├── 5.1: Ripgrep pre-filtering
  ├── 5.2: Small model file classifier
  └── 5.3: MCP end-to-end

Phase 6: Baselines                      ← No dependencies, run anytime
  ├── 6.1: Cloud quality comparison
  └── 6.2: Cost analysis
```

## Code style

- Python 3.12+, type hints everywhere, `async` for all API calls.
- `httpx.AsyncClient` for raw HTTP; `openai.AsyncOpenAI` for chat completions.
- `pydantic` for config/result schemas; `rich` for CLI progress and tables.
- Shell scripts: `set -euo pipefail`, shellcheck-clean, POSIX-compatible.
- Compose files: `docker compose` v2 CLI with podman-compose as backend.

## Decisions already settled (do not re-evaluate)

1. **No dense 70B with tensor parallelism.** PCIe x8/x8 without NVLink = 20–35 t/s. MoE at 200+ t/s wins.
2. **System RAM is not VRAM.** KV cache in DDR5 = 50–80% speed loss. Keep everything in 32 GB GDDR7 per GPU.
3. **KTransformers not viable.** i9-14900K lacks AMX.
4. **Speculative decoding doesn't help MoE.** Only test on dense models (Phase 2.5, 4.3).
5. **No Ollama.** Adds 10–30% overhead vs raw engine containers.

## Decisions to be made by benchmarks

1. Does Qwen3.5-35B-A3B tool calling actually work? (Phase 0 — BLOCKER)
2. vLLM or SGLang for MoE on 5090? (Phase 1)
3. Which coder model? 35B-A3B vs 30B-A3B vs Coder-Next (Phase 2)
4. Which thinker model? Qwen3.5-27B vs R1-32B vs none (Phase 2)
5. Is multi-tier worth the complexity? (Phase 3)
6. Does the LiteLLM router classify correctly? (Phase 3)

## Known engine bugs (as of April 2026 — verify fix status before testing)

| Bug | Engine | Impact | Workaround |
|-----|--------|--------|------------|
| Tool calls inside `<think>` silently dropped | vLLM 0.19 (PR #39055 open) | CRITICAL for thinking+tool models | Use `--reasoning-parser qwen3`; check PR status |
| Qwen3.5 tool call JSON malformat | vLLM (PR #35347 merged?) | Format errors | Verify your vLLM version includes fix |
| Wrong tool parser for qwen3.5 | Ollama (Issue #14493) | Tool calling completely broken | Don't use Ollama |
| Qwen3.5 chat template broken | All HF-template engines | Malformed tool-call rendering | Use barubary's fixed template or Unsloth March 2026 GGUFs |
| Spec decode crashes on hybrid SSM/MoE | llama.cpp (PR #20075 open) | Crash on Qwen3.5 MoE | Don't test spec decode on MoE |
| Coder-Next CPU inference 5× slower | llama.cpp (Issue #19480) | MoE routing overhead | GPU-only: `--n-gpu-layers 99` |
