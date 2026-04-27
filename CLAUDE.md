# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

A personal test harness to find the optimal local LLM configuration (model + engine + quantization + KV cache + GPU placement) for **one specific hardware setup** and one infrastructure engineer's workload: Python/shell projects, rootless podman, Ceph/OpenStack, Orange Pi ARM64, and eventually doc-aware agentic work via OpenCode + MCP.

Models are evaluated as **role endpoints** (coder / thinker / convergence), not as "the one model." OpenCode v1.3+ routes subagents to different endpoints natively.

**Scope discipline:** only test what published research cannot settle. When web research answers definitively, record in `docs/decisions/settled.md` and skip the test. Saving cycles on non-tests is as valuable as running tests.

## Hardware under test

```
CPU:    Intel Core i9-14900K (24 cores, NO AMX, Raptor Lake)
RAM:    192 GB DDR5 (4×48 GB, ~83 GB/s actual due to 4-DIMM downclock)
GPU 0:  NVIDIA RTX 5090 32 GB GDDR7 (1790 GB/s, PCIe 5.0 x8)
GPU 1:  NVIDIA RTX 5090 32 GB GDDR7 (1790 GB/s, PCIe 5.0 x8)
Inter:  NO NVLink — PCIe x8/x8 bifurcation
SSD:    Lexar NM790 4 TB NVMe (7,400 MB/s read, 6,500 MB/s write, 3,000 TBW, PCIe 4.0 x4)
OS:     Linux (kernel 6.x, rootless podman, NVIDIA container toolkit)
```

TP=2 gives effectively 64 GB VRAM. Do not reject models as "too big" for one GPU without checking TP=2 fit.

## Operating model: research ↔ testing loop

```
┌──────────────┐  open question    ┌──────────────┐
│   RESEARCH   │  ───────────────▶ │   TESTING    │
│  (web/docs,  │                   │  (on host,   │
│  Claude +    │  ◀─────────────── │   GPU live)  │
│  operator)   │  wall hit /       └──────────────┘
└──────────────┘  new insight
```

- **Research mode** (this container): reads web sources, HF model cards, vLLM/SGLang docs, issue trackers. Writes to `docs/`, `RESEARCH_STATE.md`, `config/models.yaml`.
- **Testing mode** (host, operator + Claude Code): runs real benchmarks against live GPUs. Writes to `results/…/metrics.json` + `summary.md`, then `RESEARCH_STATE.md`.

**Testing mode MUST hand control back to research mode when:**

1. The current queue item is complete.
2. An engine bug, OOM, or unexpected behavior blocks progress and is not already documented.
3. A result contradicts a prior assumption.
4. A result opens a new direction that could change subsequent priorities.

Hand-off: write a short block in `RESEARCH_STATE.md` under `## Open from testing`, then ask operator to switch.

**This is not a phase plan.** There is a queue with dependencies. See `docs/queue/open.md`.

## Behavioral rules

- **Rerun cleanly if unsure.** If a result looks suspicious, fabricated, or the raw `metrics.json` has null fields — rerun from scratch rather than patching inconsistencies into docs. This applies especially to results from agent-assisted runs (Gemini, Claude) that may not have had live endpoints.
- **Verify raw data before documenting.** Always check `results/*/metrics.json` and the `raw/` directory before writing results to docs. A summary.md can be written from broken data. Trust the raw API response files.
- **Use `rg` (ripgrep) for searching.** Not grep. `rg "<pattern>" docs/` or `rg -l "<pattern>"` for file list.
- **No fabrication.** If you don't have data, say so. `null`, `UNKNOWN`, or `NOT MEASURED` is always better than a plausible-sounding number.

## Document map

Load `docs/INDEX.md` for the full current-state summary (~150 lines — covers active decisions, procedures, open queue, and key gotchas). Always load this on init for any non-trivial task.

| File / Dir | Purpose |
|------------|---------|
| `CLAUDE.md` | This file — always loaded |
| `docs/INDEX.md` | Master index — load on init |
| `docs/arch/` | Architecture (current.md, extended-arclight.md, convergence.md) |
| `docs/decisions/` | Settled decisions (settled.md, models.md, scoring.md) |
| `docs/procedures/` | Ops how-to (vllm-deploy.md, criu-ops.md) |
| `docs/queue/open.md` | Open items with full specs |
| `docs/queue/status.md` | One-line status for every item |
| `docs/history/` | Cycle log + done-items (grep target, rarely needed) |
| `docs/handoffs/` | Gemini testing session handoff documents (naming: HANDOFF_GEMINI_YYYYMMDD.md) |
| `RESEARCH_STATE.md` | Current research summary + open-from-testing blocks |
| `config/models.yaml` | Model registry (annotated) |
| `results/*/summary.md` | Benchmark run results (read first) |

## Context discipline

Do NOT read unless the task explicitly requires it:
- `results/*/raw/` — large raw API responses
- `*.log`, `bench.log` — use `results/*/summary.md` first; only open `metrics.json` if summary is insufficient
- `*.jsonl` in `~/.claude/` — session history, not project data

## Execution environment

Claude runs inside a rootless Podman container (claude-box). The repo is bind-mounted at the same absolute path as on the host. **You cannot run benchmarks, talk to GPUs, or reach vLLM endpoints (ports 30000–30002 refused).** Read and write repo files only.

Python 3.11.x available with: `pydantic`, `httpx`, `pyyaml`, `rich`, `openai`, `ruff`, `pytest`. Use for: parsing `results/*/metrics.json`, running `ruff check .` before handback, running `pytest` on `lib/` unit tests.
