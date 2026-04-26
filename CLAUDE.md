# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

A personal test harness to determine the optimal local LLM configuration (model + engine + quantization + KV cache settings + GPU placement + swap policy) for **one specific hardware setup** and **one operator's workload mix**.

The operator is an infrastructure engineer. The workload is not generic "coding" — it is:

- Infrastructure projects like this benchmark repo (Python + shell + containers)
- Container orchestration & isolation (rootless podman, custom wrappers like `claude-box`)
- Matrix microservice fleet (12-container deployment)
- Compilation and build work for custom tooling
- VM-based development of a Debian-derived custom OS
- Idempotent deployment harness for Armbian on Orange Pi boards (cloud-init-adjacent)
- Ceph, OpenStack, network switch tuning for low-latency quant-analysis workloads
- Eventually: OpenCode + MCP servers (firecrawl, searxng, RAG backends) for doc-aware agentic work

The current agent frontend is **OpenCode**. Subagents route to different model endpoints natively (v1.3+), so models are evaluated as **role endpoints** (coder / thinker / behemoth), not as "the one model." This changes what we test and how we score it.

**Scope discipline:** we only test what is not already settled by published research. When web research answers a question definitively, we record the decision in `DECISIONS.md` and do not re-test. When research leaves ambiguity or the claim depends on details of our specific hardware/engine/quant combination, we test. Saving cycles on non-tests is as valuable as running tests — both narrow the search.

## Hardware under test

```
CPU:    Intel Core i9-14900K (24 cores, NO AMX, Raptor Lake)
RAM:    192 GB DDR5 (4×48 GB, ~83 GB/s actual due to 4-DIMM downclock)
GPU 0:  NVIDIA RTX 5090 32 GB GDDR7 (1790 GB/s, PCIe 5.0 x8)
GPU 1:  NVIDIA RTX 5090 32 GB GDDR7 (1790 GB/s, PCIe 5.0 x8)
Inter:  NO NVLink — PCIe x8/x8 bifurcation
OS:     Linux (kernel 6.x, rootless podman, NVIDIA container toolkit)
```

The two GPUs are a **shared pool via tensor parallelism**, not two independent slots. Single-GPU-only constraints (e.g. "this model is 22 GB bf16, doesn't fit in 32 GB") are usually false for our rig because TP=2 gives us effectively 64 GB minus TP overhead. See `ARCHITECTURE.md`.

## Operating model: research ↔ testing loop

- **Arclight Hot-Swap (T_KV2)**: Uses host-native CRIU + `cuda-checkpoint`.
    - **CRITICAL**: vLLM is patched to disable `uvloop` (io_uring). Do not revert `api_server.py` or `v1/utils.py` changes.
    - **Environment**: Always set `UV_USE_IO_URING=0`.

```
┌──────────────┐  open question    ┌──────────────┐
│   RESEARCH   │  ───────────────▶ │   TESTING    │
│  (web/docs,  │                   │  (on host,   │
│  Claude +    │  ◀─────────────── │   GPU live)  │
│  operator)   │  wall hit /       └──────────────┘
└──────────────┘  new insight
```

- **Research mode** (this container): the assistant reads web sources, HF model cards, vLLM/SGLang docs, issue trackers, and prior conversation. Output is updates to `RESEARCH_STATE.md`, `TESTING_QUEUE.md`, `DECISIONS.md`, and `config/models.yaml`.
- **Testing mode** (host, operator + Claude Code): runs real benchmarks against live GPUs. Output is `results/…/metrics.json` + `summary.md`, plus short entries in `RESEARCH_STATE.md` under "findings from last test cycle."

**Testing mode MUST hand control back to research mode when any of the following happens:**

1. The current item in `TESTING_QUEUE.md` is complete (one or more items).
2. An engine bug, kernel mismatch, OOM, or unexpected parser behavior blocks progress and the cause is not already documented in `DECISIONS.md`.
3. A result contradicts a prior assumption recorded as a "provisional" decision.
4. A result opens a new direction (e.g. unexpectedly high TPS on a model we thought was dead) that could change subsequent priorities.

The hand-off is explicit: Claude Code writes a short block in `RESEARCH_STATE.md` under `## Open from testing` describing the wall or insight, then stops and asks the operator to switch to research mode. The research pass updates the queue and decisions, then testing resumes.

**This is not a phase plan.** There are no phase gates. There is a queue, and items in the queue have dependencies. See `TESTING_QUEUE.md`.

## Document map

| File | Purpose | Edited by |
|------|---------|-----------|
| `CLAUDE.md` | This file — durable project brief | Research mode |
| `CLAUDE.local.md` | Claude launcher (rootless podman wrapper) | Operator |
| `ARCHITECTURE.md` | Current working architecture hypothesis | Research mode |
| `RESEARCH_STATE.md` | What we know, what we don't, cycle log | Both |
| `TESTING_QUEUE.md` | Ordered question queue with dependencies | Both |
| `DECISIONS.md` | Settled decisions + kill list with rationale | Research mode |
| `PHASE2_RESULTS.md` | Historical artifact from the phased plan | Frozen |
| `config/models.yaml` | Model registry (annotated) | Both |
| `config/thresholds.yaml` | Pass/fail criteria — annotate per item, not per phase | Both |
| `config/hardware.env` | Ports, cache paths, GPU IDs | Operator |

## Context discipline

Do NOT read into these paths unless the current task explicitly requires it:
- `results/*/raw/` — raw API response files, large and rarely needed
- `*.log`, `bench.log` — use `results/*/summary.md` instead
- `*.jsonl` in `~/.claude/` — session history, not project data

When analyzing a benchmark run, read `results/<item_id>-<timestamp>/summary.md` first.
Only open `metrics.json` if the summary is insufficient.

## Execution environment

Claude runs inside a rootless Podman container via the claude-box wrapper (see CLAUDE.local.md).
The repo is bind-mounted at the exact same absolute path as on the host (e.g. `/srv/ai/projects/local-llm-benchmarks`). Host paths outside this mount are unavailable.

**You cannot run benchmarks. You read and write repo files only.**

**Python 3.11.x is available with:** `pydantic`, `httpx`, `pyyaml`, `rich`, `openai`, `ruff`, `pytest` — plus full stdlib.

Claude Code MAY use Python for:
- Parsing `results/*/metrics.json` with `json` + `pathlib` or `pydantic` schemas
- Parsing `config/models.yaml` / `config/thresholds.yaml` with `pyyaml`
- Generating `summary.md` tables with `rich`
- Running `pytest` on `lib/` unit tests after making changes to lib
- Running `ruff check .` to lint before handing back to operator

**The container CANNOT:**
- Talk to the GPU
- Activate the host `hf` pyenv or any host venv
- Run benchmarks (`infra/scripts/deploy.sh` invokes `podman compose` — host only)
- Execute `just` targets that call deploy scripts
- Reach vLLM endpoints — ports 30000–30002 are host-side only, connection will be refused

Running benchmarks is the operator's job.

## Python on host (operator runs these)

```bash
pyenv activate hf               # canonical — .venv symlink points to the same env
pip install -e ".[dev]"         # first-time setup

ruff check . && ruff format .
pyright
pytest
```

Benchmark scripts live under `benchmarks/` and the `justfile` has shortcuts. All of them call `./infra/scripts/deploy.sh` which runs engines in rootless podman containers — engines are never installed on the host.

## Model downloads (host only)

```bash
pyenv activate hf
HF_HOME=/srv/ai/models hf download QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ
# gated:
HF_TOKEN=hf_... HF_HOME=/srv/ai/models hf download mistralai/Devstral-Small-2505
```

Do not use a throwaway Python container for downloads — the host venv has everything.

## Container orchestration

All engines run as rootless podman containers via `podman compose`. GPU isolation uses `NVIDIA_VISIBLE_DEVICES` (not `CUDA_VISIBLE_DEVICES`). Scripts use `podman compose`, not `docker compose`. See `infra/compose/` for templates; `infra/scripts/deploy.sh` is the one canonical entry point.

## Benchmark run convention

Every runnable test writes to `results/{item_id}_{timestamp}/`:
- `raw/` — raw API responses per task
- `metrics.json` — structured metrics
- `bench.log` — full log
- `summary.md` — human-readable summary via `lib.reporter`
- `human_review.md` — when quality scoring needed

`metrics.json` schema:
```json
{
  "item_id": "coder_glm47_flash_awq_tp2_kv_q8",
  "timestamp": "2026-04-17T10:30:00Z",
  "config": {
    "engine": "vllm",
    "engine_version": "0.19.x",
    "model": "cyankiwi/GLM-4.7-Flash-AWQ-4bit",
    "quantization": "AWQ-INT4",
    "kv_cache_dtype": "fp8",
    "placement": "tp=2",
    "context_length": 65536,
    "extra_args": "--tool-call-parser glm47 --reasoning-parser glm45"
  },
  "metrics": {},
  "verdict": "PASS | FAIL | INCONCLUSIVE",
  "notes": ""
}
```

The `item_id` should match the entry in `TESTING_QUEUE.md`.

## Scoring (`lib/scorer.py`)

```python
class ToolCallScore:
    PASS         = "pass"
    FORMAT_ERROR = "format_error"  # engine parser broken
    DROPPED      = "dropped"       # engine silently dropped tool call
    WRONG_TOOL   = "wrong_tool"
    WRONG_ARGS   = "wrong_args"
    NO_CALL      = "no_call"
    EXCEPTION    = "exception"
```

`DROPPED` / `FORMAT_ERROR` indicate engine bugs, not model problems.

## `BenchClient` interface (`lib/client.py`)

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

## Code style

- Python 3.12+, type hints everywhere, `async` for API calls.
- `httpx.AsyncClient` raw HTTP, `openai.AsyncOpenAI` for chat.
- `pydantic` for config/result schemas, `rich` for CLI tables.
- Shell: `set -euo pipefail`, shellcheck-clean.
- Compose: `docker compose` v2 CLI with podman-compose backend.

## Task suite scope — important caveat

The current task suites under `benchmarks/phase0_*/tasks/` and `benchmarks/phase2_*/tasks/` are coding-oriented (Python-heavy fix-the-race-condition, refactor-this-class, etc). They are a partial proxy for the operator's real workload. Over time, `TESTING_QUEUE.md` will add infra-specific task categories:

- Shell + containerfile authoring
- systemd unit / compose / podman-specific errors
- Network and kernel tuning (sysctl, ethtool, tc)
- Ceph/OpenStack/openstack-ansible troubleshooting
- Cross-arch work (arm64 for Orange Pi, x86 for desktop)
- RAG-assisted doc lookup flows (once MCP servers are wired)

Do not assume a model that wins on the Python code suite also wins on these. When adding infra-shaped tasks, note this in the queue entry so reruns capture the expanded workload.