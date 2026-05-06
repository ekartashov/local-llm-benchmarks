# Project Index

**Load this file on every session init.** Use `rg` (ripgrep) to search across docs/.

---

## Current production configuration (R35, 2026-05-06)

| Role | Model | Config | Port | TPS | Status |
|------|-------|--------|------|-----|--------|
| Arclight Coder | Qwen3.6-35B-A3B APEX GGUF (mudler) | ik_llama.cpp, -ngl 999, fp8 KV, -np 4, ctx=32768 | 8080 | 185 t/s | SETTLED (BENCH_24) |
| Arclight Thinker | Qwen3.6-27B PrismaQuant-5.5bit (rdtand) | vLLM TP=1 GPU1, V1 engine, fp8 KV, cp-ON, --max-num-seqs 4, MTP n=3 | 30001 | 92 t/s seq=1 / 315 t/s N=4 | SETTLED (BENCH_19) |
| Extended Arclight | Coder as TP=2 (thinker sleeping) | ctx=65536, CRIU hot-restart 0.28s | 30000 | 238 t/s | SETTLED |
| Convergence | unsloth/Qwen3.5-397B-A17B UD-IQ2_M | ik_llama.cpp main, GGML_CUDA_NO_PINNED=1, `-ngl 999` (auto-allocate) | 8002 | 13.8 t/s (co-load) | SETTLED (BENCH_25) |

**Open questions:** **T_APEX1/2 SETTLED SUCCESS (BENCH_24/25)** — APEX GGUF coder on ik_llama.cpp delivers 3.2x TPS over vLLM PrismaQuant baseline by bypassing the SM120 FlashInfer bottleneck. Convergence co-load performance restored to 98% of isolated speed (13.8 t/s vs 14.0 t/s) via VRAM reclamation. T_MTP2 CLOSED FAIL — MTP breaks tool-call generation on A3B MoE coder. T_KV1 swap blocked. T_KV3 SETTLED — 128K context verified. T_HARD1 CLOSED — PQ/AWQ statistical tie on hard suite.

---

## Chapter map

### Architecture
- **[docs/arch/current.md](arch/current.md)** — Deployment topology, tier diagram, process sequences, model registry
  - *Summary:* Arclight hot-pair (TP=1-per-GPU), Extended Arclight (CRIU 0.28s restart), Convergence dual-mode (always-resident or 12s CRIU restore). Core RETIRED.
  - *Grep for:* tier names, port assignments, VRAM budgets, deployment commands
- **[docs/arch/extended-arclight.md](arch/extended-arclight.md)** — Extended Arclight (CRIU) deep-dive
  - *Summary:* Thinker sleeps → coder spans TP=2 at 65K ctx. Hot-restart via CRIU+cuda-checkpoint.
  - *Grep for:* CRIU procedure, uvloop patch, UV_USE_IO_URING, ghost VRAM, 65K context numbers
- **[docs/arch/convergence.md](arch/convergence.md)** — Convergence + Singularity operational guide
  - *Summary:* 397B always-resident or CRIU-restored (12s). ik_llama.cpp main branch.
  - *Grep for:* launch command, -np caveat, DeltaNet crash, DDR5 bandwidth ceiling, model path, GGML_CUDA_NO_PINNED

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

1. **T_PAR1 real data exists (BENCH_01–03, 2026-04-28):** Coder TP=2: N=1 241 t/s → N=8 1,205 t/s (no saturation). Thinker max-num-seqs=4: 269.4 t/s at N=4. Old "fabricated" warning is retired — do NOT repeat it.
2. **`--swap-space N` is unrecognized in vLLM 0.19.0** (R-V0 engine path). Context beyond 65K requires a vLLM version bump.
3. **TP=2 is broken for GDN (Qwen3.6-27B) regardless of chunked-prefill setting** — H-TP2 confirmed T2.4g. Do not attempt thinker TP=2 without a non-GDN replacement.
4. **vLLM uvloop patch is mandatory for CRIU** — do not revert `api_server.py` / `v1/utils.py` asyncio changes; always export `UV_USE_IO_URING=0`.
5. **`--reasoning-parser` must NOT be combined with tool-call parser for models that skip thinking** — causes all-`no_call` on Gemma4, Core/80B, and any model that emits tool calls directly. Use `--tool-call-parser X` alone.
6. **T_CV4 15.6 t/s is sequential pipelining, not true concurrency** — concurrent HTTP to Convergence is permanently limited to N=1 (SETTLED FAIL, PR #1288: Qwen3.5-MoE architectural constraint, not a bug).
7. **Convergence On-Demand viable (12s)** — Success via CRIU + `GGML_CUDA_NO_PINNED=1` + QX_PRELOAD (BENCH_22). Preferred over always-resident mode for co-load RAM efficiency.
8. **`VLLM_USE_V1=0` is version-scoped** — required on vLLM 0.19.x; vLLM 0.20+ logs it as unknown/ignored (BENCH_21/BENCH_23 logs).
9. **VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 required for coder TP=2** — prevents OOM during CUDA graph capture.
10. **After a failed CRIU restore: `sudo nvidia-smi --gpu-reset -i 1`** — clears ghost VRAM leaks.
11. **Convergence model path uses split GGUF** — reference only `00001-of-00004.gguf`; loader finds the rest. All 4 files must be in the same directory.
12. **`--max-num-seqs 4` for thinker** — upgraded from 1 (T_PAR1 R30). 3.5× parallel throughput (269 t/s at N=4), 4 MiB VRAM delta. The seqs=1 constraint was empirically unnecessary.
13. **CRIU is TP=1 only** — TP=2 CRIU restore succeeds but post-restore inference fails (SHM IPC broken; Blackwell forces V1 engine). 26s restore is also only 4× vs cold start. Do not attempt for coder. See `docs/decisions/settled.md` and `docs/procedures/criu-ops.md`.
14. **PrismaQuant 4.75bit 35B MoE startup OOM without tuned `max_num_seqs`** — model weights (~29.4 GiB) exceed any `gpu-mem-util` budget; OOM is in profiling forward pass sized by `max_num_seqs`. Default 1024 seqs → ~1.02 GiB needed / ~1.01 GiB free. Fix: `--max-num-seqs 16` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `--gpu-mem-util 0.90`. **Logical stability confirmed on V1 engine at TP=1** (BENCH_23/23b). Do not use V0/Eager for this model at TP=1. TPS is capped at ~57 t/s (agg N=1, 120.9 t/s decode) due to SM120 grouped GEMM software maturity. **MTP at TP=1** is logically stable (5/5 tools, BENCH_23b) but incurs -38.6% at N=1 (34.7 t/s) and -50.6% at N=4 (227 t/s tuned vs 459 no-MTP). Do not set `VLLM_USE_FLASHINFER_NVFP4` — unknown env var in vLLM 0.20.0, no-op.
15. **AWQ TP=1 V1 engine fails tool calls (BENCH_23a)** — AWQ (INT4 Marlin) at TP=1 with V1 engine = **2/5 tool calls**. TP=1 stability is specific to **PrismaQuant+V1**, not V1 engine alone. Do not substitute AWQ at TP=1 thinking it will be stable.
16. **Coder vs Thinker per-request TPS crossover** — At N=1: thinker 91.9 t/s vs coder 56.5 t/s (thinker 63% faster). At N=4: coder 115.4 t/s/req vs thinker 78.7 t/s/req (coder 47% faster in batch). Coder benefits far more from batching; thinker favors single-request latency.
17. **Thinker KV cache is nearly unused (GDN architecture)** — GDN layers maintain O(d) recurrent state, not O(n) KV tokens. Measured: 0 MiB KV delta at 50K context (T3.1). fp16 KV for thinker provides negligible quality benefit. Do not tune thinker util expecting KV pool size to affect reasoning quality.

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
