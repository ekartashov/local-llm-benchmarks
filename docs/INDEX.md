# Project Index

**Load this file on every session init.** Use `rg` (ripgrep) to search across docs/.

---

## Current production configuration (R27, 2026-04-26)

| Role | Model | Config | Port | TPS | Status |
|------|-------|--------|------|-----|--------|
| Arclight Coder | Qwen3.6-35B-A3B-AWQ | vLLM TP=2 GPU0+1, fp8 KV, ctx=32768 | 30000 | 232 t/s | SETTLED |
| Arclight Thinker | Qwen3.6-27B-AWQ | vLLM TP=1 GPU1, fp8 KV, cp-ON, max-num-seqs 1 | 30001 | 77 t/s | SETTLED |
| Extended Arclight | Coder as TP=2 (thinker sleeping) | ctx=65536, CRIU hot-restart 0.28s | 30000 | 238 t/s | SETTLED |
| Convergence | Qwen3.5-397B UD-IQ2_M | ik_llama.cpp -ngl 999 --cpu-moe -t 32 -np 4 | 8002 | 13.99 t/s | SETTLED |

**Open questions:** T_PAR1 Coder/Thinker max-num-seqs UNKNOWN (rerun needed). T_KV3 blocked on research (TP=2-capable thinker candidate). T_KV1 swap blocked (vLLM 0.19 flag issue).

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
  - *Summary:* 397B always-resident at 13.99 t/s hybrid. 83s cold start → always-on policy. Concurrent-request crash at N≥2.
  - *Grep for:* launch command, -np caveat, pr-1288 crash, DDR5 bandwidth ceiling, model path

### Decisions
- **[docs/decisions/settled.md](decisions/settled.md)** — All SETTLED decisions, most recent first
  - *Summary:* All architectural, infra, and engine decisions with rationale and measured values.
  - *Grep for:* any specific technology/flag/model to check if it's already decided — check here first before testing
- **[docs/decisions/models.md](decisions/models.md)** — Model role assignments, candidates, eliminated models
  - *Summary:* Who won each role and why; what was eliminated and the specific failure mode.
  - *Grep for:* model names, parser flags, quantization choices, benchmark scores

### Queue
- **[docs/queue/open.md](queue/open.md)** — OPEN and BLOCKED items only (full item spec)
  - *Summary:* T_PAR1 (Coder/Thinker rerun), T_KV3 (research blocked), T3.x optimization, T5.x integration, T6.x task suite
  - *Grep for:* item_id, procedure steps, pass criteria, hand-back triggers
- **[docs/queue/status.md](queue/status.md)** — One-line status for every item (DONE + OPEN + BLOCKED)
  - *Summary:* Quick lookup for whether any specific item has been run and what it found.

### History (grep target, rarely loaded inline)
- **[docs/history/cycles.md](history/cycles.md)** — Chronological cycle log R1–R27
  - *Load when:* debugging why a decision was made, understanding context behind a specific test result
  - *Grep for:* cycle dates, specific findings, "triggered by", model names in historical context
- **[docs/history/done-items.md](history/done-items.md)** — Full procedures for all DONE/CANCELLED/SKIPPED queue items
  - *Load when:* reproducing a past test, understanding exactly what was measured and how
  - *Grep for:* item_id, pass criteria, "what failure means", result numbers

### Procedures (load when executing an operation)
- **[docs/procedures/vllm-deploy.md](procedures/vllm-deploy.md)** — vLLM deployment patterns, env vars, container flags
  - *Load when:* deploying or scripting any vLLM endpoint change
  - *Grep for:* env var names, flag names, VLLM_V1_ENABLED, CUDA graph flags, port assignments
- **[docs/procedures/criu-ops.md](procedures/criu-ops.md)** — CRIU + cuda-checkpoint operational procedure
  - *Load when:* running Extended Arclight mode switches, checkpointing, debugging restore failures
  - *Grep for:* uvloop patch, checkpoint command, ghost VRAM cleanup, restore time expectations

---

## Key gotchas (things easy to get wrong — scan this list before acting)

1. **T_PAR1 Coder/Thinker numbers are fabricated** — do not cite "1,196 t/s at N=8" or "698 t/s at N=4". No measurement exists. Rerun required.
2. **`--swap-space N` is unrecognized in vLLM 0.19.0** (R-V0 engine path). Context beyond 65K requires a vLLM version bump.
3. **TP=2 is broken for GDN (Qwen3.6-27B) regardless of chunked-prefill setting** — H-TP2 confirmed T2.4g. Do not attempt thinker TP=2 without a non-GDN replacement.
4. **vLLM uvloop patch is mandatory for CRIU** — do not revert `api_server.py` / `v1/utils.py` asyncio changes; always export `UV_USE_IO_URING=0`.
5. **`--reasoning-parser` must NOT be combined with tool-call parser for models that skip thinking** — causes all-`no_call` on Gemma4, Core/80B, and any model that emits tool calls directly. Use `--tool-call-parser X` alone.
6. **T_CV4 15.6 t/s is sequential pipelining, not true concurrency** — concurrent HTTP to Convergence crashes at N≥2 (T_PAR1). Do not quote T_CV4 as "parallel" capacity.
7. **Convergence always-resident** — 83s cold start; never on-demand. Always start before Arclight if doing a full system restart.
8. **VLLM_V1_ENABLED=0 is required** — V1 engine causes 10× TPS regression in eager mode and CUDA graph instability on Blackwell.
9. **VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 required for coder TP=2** — prevents OOM during CUDA graph capture.
10. **After a failed CRIU restore: `sudo nvidia-smi --gpu-reset -i 1`** — clears ghost VRAM leaks.
11. **Convergence model path uses split GGUF** — reference only `00001-of-00004.gguf`; loader finds the rest. All 4 files must be in the same directory.
12. **`--max-num-seqs 1` mandatory for thinker** — CUDA graph stability constraint on single GPU. Do not raise without testing.

---

## Hardware quick-ref

```
CPU:    i9-14900K  (24 cores, NO AMX, Raptor Lake)
RAM:    192 GB DDR5  (~83 GB/s actual)
GPU 0:  RTX 5090 32GB  (sm_120, PCIe 5.0 x8)
GPU 1:  RTX 5090 32GB  (sm_120, PCIe 5.0 x8)
Link:   NO NVLink — PCIe x8/x8 bifurcation only
OS:     Linux 6.x, rootless podman, NVIDIA container toolkit
```

TP=2 allreduce over PCIe x8: fine for MoE A3B (~60 MB/s at 150 t/s). Catastrophic for dense 70B (two-context collapse to ~2 t/s each, T1.2).

---

## Search guide

```bash
# Find where a specific model/flag/decision is documented:
rg "VLLM_V1_ENABLED" docs/
rg "qwen3_coder" docs/decisions/models.md
rg "T_PAR1" docs/queue/
rg "GDN" docs/decisions/

# Find a cycle log entry:
rg "R22" docs/history/cycles.md

# Check if something is already SETTLED before testing:
rg "SETTLED" docs/decisions/settled.md | head -40
```
