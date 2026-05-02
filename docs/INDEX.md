# Project Index

**Load this file on every session init.** Use `rg` (ripgrep) to search across docs/.

---

## Current production configuration (R27, 2026-04-26)

| Role | Model | Config | Port | TPS | Status |
|------|-------|--------|------|-----|--------|
| Arclight Coder | Qwen3.6-35B-A3B-AWQ | vLLM TP=2 GPU0+1, fp8 KV, ctx=32768 | 30000 | 232 t/s | SETTLED |
| Arclight Thinker | Qwen3.6-27B-AWQ | vLLM TP=1 GPU1, fp8 KV, cp-ON, max-num-seqs 4 | 30001 | 77 t/s seq=1 / 269 t/s N=4 | SETTLED |
| Extended Arclight | Coder as TP=2 (thinker sleeping) | ctx=65536, CRIU hot-restart 0.28s | 30000 | 238 t/s | SETTLED |
| Convergence | unsloth/Qwen3.5-397B-A17B UD-IQ2_M | ik_llama.cpp main (merged DeltaNet support), -ngl 999 --cpu-moe, -np 4, -t 32 | 8002 | 14 t/s | SETTLED |

**Open questions:** T_MTP1/T_MTP2 HIGH — MTP n=1 on AWQ thinker/coder (+27% TPS potential, verify vLLM #40756 fix first). T_PQ1 MEDIUM — PrismaQuant thinker (CUDA 13.0 rebuild). QX_PRELOAD HIGH — CRIU on Convergence (100s first-inference without pre-warm). T_KV1 swap blocked (vLLM 0.19 flag issue). T_KV3 SETTLED — 128K context verified (1,892 t/s prefill, 49 t/s decode, Path B ik_llama.cpp).

---

## Chapter map

### Architecture
- **[docs/arch/current.md](arch/current.md)** — Deployment topology, tier diagram, process sequences, model registry
  - *Summary:* Arclight hot-pair (TP=1-per-GPU), Extended Arclight (CRIU 0.28s restart), Convergence always-resident. Core RETIRED.
  - *Grep for:* tier names, port assignments, VRAM budgets, deployment commands
- **[docs/arch/extended-arclight.md](arch/extended-arclight.md)** — Extended Arclight (CRIU) deep-dive
  - *Summary:* Thinker sleeps → coder spans TP=2 at 65K ctx. Hot-restart via CRIU+cuda-checkpoint.
  - *Grep for:* CRIU procedure, uvloop patch, UV_USE_IO_URING, ghost VRAM, 65K context numbers
- **[docs/arch/convergence.md](arch/convergence.md)** — Convergence + Singularity operational guide
  - *Summary:* 397B always-resident at 13.99 t/s hybrid. 83s cold start → always-on policy. Concurrent-request crash N≥2 (investigating `main` branch fix).
  - *Grep for:* launch command, -np caveat, DeltaNet crash, DDR5 bandwidth ceiling, model path

### Decisions
- **[docs/decisions/settled.md](decisions/settled.md)** — All SETTLED decisions, most recent first
  - *Summary:* All architectural, infra, and engine decisions with rationale and measured values.
  - *Grep for:* any specific technology/flag/model to check if it's already decided — check here first before testing
- **[docs/decisions/models.md](decisions/models.md)** — Model role assignments, candidates, eliminated models
  - *Summary:* Who won each role and why; what was eliminated and the specific failure mode.
  - *Grep for:* model names, parser flags, quantization choices, benchmark scores
- **[docs/decisions/scoring.md](docs/decisions/scoring.md)** — Per-role evaluation weights and engine selection criteria
  - *Summary:* What matters for each role (TPS vs quality vs context vs TTFT) and decision rules.
  - *Load when:* evaluating a new model or engine candidate to ensure consistent scoring

### Queue
- **[docs/queue/open.md](docs/queue/open.md)** — OPEN and BLOCKED items only (full item spec)
  - *Summary:* T_PAR1 (Coder/Thinker rerun), T_KV3 (research blocked), T3.x optimization, T5.x integration, T6.x task suite
  - *Grep for:* item_id, procedure steps, pass criteria, hand-back triggers
- **[docs/queue/status.md](docs/queue/status.md)** — One-line status for every item (DONE + OPEN + BLOCKED)
  - *Summary:* Quick lookup for whether any specific item has been run and what it found.

### Handoffs
- **[docs/handoffs/](docs/handoffs/)** — Gemini testing session handoff files (naming: HANDOFF_GEMINI_YYYYMMDD.md)
  - *Write new handoffs here*, not repo root.
  - *Load when:* handing off to Gemini — contains step-by-step procedure, deploy commands, termination conditions.

### History (grep target, rarely loaded inline)
- **[docs/history/cycles.md](docs/history/cycles.md)** — Chronological cycle log R1–R27
  - *Load when:* debugging why a decision was made, understanding context behind a specific test result
  - *Grep for:* cycle dates, specific findings, "triggered by", model names in historical context
- **[docs/history/done-items.md](docs/history/done-items.md)** — Full procedures for all DONE/CANCELLED/SKIPPED queue items
  - *Load when:* reproducing a past test, understanding exactly what was measured and how
  - *Grep for:* item_id, pass criteria, "what failure means", result numbers

### Procedures (load when executing an operation)
- **[docs/procedures/vllm-deploy.md](docs/procedures/vllm-deploy.md)** — vLLM deployment patterns, env vars, container flags
  - *Load when:* deploying or scripting any vLLM endpoint change
  - *Grep for:* env var names, flag names, VLLM_USE_V1, CUDA graph flags, port assignments
- **[docs/procedures/criu-ops.md](docs/procedures/criu-ops.md)** — CRIU + cuda-checkpoint operational procedure
  - *Load when:* running Extended Arclight mode switches, checkpointing, debugging restore failures
  - *Grep for:* uvloop patch, checkpoint command, ghost VRAM cleanup, restore time expectations
- **[docs/procedures/handoff-standard.md](docs/procedures/handoff-standard.md)** — Normative standard for writing BENCH_XX handoffs
  - *Load when:* writing a new handoff document for a Gemini testing session
  - *Grep for:* skip logic pattern, log streaming pattern, incidental findings rule, anti-patterns

---

## Key gotchas (things easy to get wrong — scan this list before acting)

1. **T_PAR1 Coder/Thinker numbers are fabricated** — do not cite "1,196 t/s at N=8" or "698 t/s at N=4". No measurement exists. Rerun required.
2. **`--swap-space N` is unrecognized in vLLM 0.19.0** (R-V0 engine path). Context beyond 65K requires a vLLM version bump.
3. **TP=2 is broken for GDN (Qwen3.6-27B) regardless of chunked-prefill setting** — H-TP2 confirmed T2.4g. Do not attempt thinker TP=2 without a non-GDN replacement.
4. **vLLM uvloop patch is mandatory for CRIU** — do not revert `api_server.py` / `v1/utils.py` asyncio changes; always export `UV_USE_IO_URING=0`.
5. **`--reasoning-parser` must NOT be combined with tool-call parser for models that skip thinking** — causes all-`no_call` on Gemma4, Core/80B, and any model that emits tool calls directly. Use `--tool-call-parser X` alone.
6. **T_CV4 15.6 t/s is sequential pipelining, not true concurrency** — concurrent HTTP to Convergence previously crashed at N≥2 (T_PAR1 on older branches); investigating if `main` branch fix (merged DeltaNet update) resolves this.
7. **Convergence always-resident** — 83s cold start; never on-demand. Always start before Arclight if doing a full system restart.
8. **VLLM_USE_V1=0 is required** — V1 engine causes 10× TPS regression in eager mode and CUDA graph instability on Blackwell.
9. **VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 required for coder TP=2** — prevents OOM during CUDA graph capture.
10. **After a failed CRIU restore: `sudo nvidia-smi --gpu-reset -i 1`** — clears ghost VRAM leaks.
11. **Convergence model path uses split GGUF** — reference only `00001-of-00004.gguf`; loader finds the rest. All 4 files must be in the same directory.
12. **`--max-num-seqs 4` for thinker** — upgraded from 1 (T_PAR1 R30). 3.5× parallel throughput (269 t/s at N=4), 4 MiB VRAM delta. The seqs=1 constraint was empirically unnecessary.
13. **CRIU is TP=1 only** — TP=2 CRIU restore succeeds but post-restore inference fails (SHM IPC broken; Blackwell forces V1 engine). 26s restore is also only 4× vs cold start. Do not attempt for coder. See `docs/decisions/settled.md` and `docs/procedures/criu-ops.md`.

---

## Hardware quick-ref

```
CPU:    i9-14900K  (24 cores, NO AMX, Raptor Lake)
RAM:    192 GB DDR5  (~83 GB/s actual)
GPU 0:  RTX 5090 32GB  (sm_120, PCIe 5.0 x8)
GPU 1:  RTX 5090 32GB  (sm_120, PCIe 5.0 x8)
Link:   NO NVLink — PCIe x8/x8 bifurcation only
SSD:    Lexar NM790 4 TB NVMe  (7,400 MB/s read, 6,500 MB/s write, 3,000 TBW, PCIe 4.0 x4)
OS:     Linux 6.x, rootless podman, NVIDIA container toolkit
```

TP=2 allreduce over PCIe x8: fine for MoE A3B (~60 MB/s at 150 t/s). Catastrophic for dense 70B (two-context collapse to ~2 t/s each, T1.2).

---

## Search guide

```bash
# Find where a specific model/flag/decision is documented:
rg "VLLM_USE_V1" docs/
rg "qwen3_coder" docs/decisions/models.md
rg "T_PAR1" docs/queue/
rg "GDN" docs/decisions/

# Find a cycle log entry:
rg "R22" docs/history/cycles.md

# Check if something is already SETTLED before testing:
rg "SETTLED" docs/decisions/settled.md | head -40
```
