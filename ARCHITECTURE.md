<!-- last_updated: 2026-04-18 — Architecture CONFIRMED via T1.2a passing metrics -->
# ARCHITECTURE.md

Working architectural source of truth.

Everything here is contingent on the tests listed in `TESTING_QUEUE.md`. This is the plan; reality will adjust it.

> **Gating status (R6+T1.3, 2026-04-18):** all four Tier-1 architecture questions answered.
> The architecture is officially TP=1-per-GPU for the always-hot models to guarantee physical concurrent isolation. Behemoth remains TP=2-asleep and borrows both GPUs on demand.
>
> **Behemoth (Qwen3-Coder-Next-80B-A3B-AWQ) — T1.3 PASS:**
> - **Verified performance**: 189 t/s seq=1 decode, 13 kt/s prefill at 32k context.
> - **Deployment**: TP=2 (spanning both GPUs) via vLLM with `--tool-call-parser hermes`.
> - **Headroom**: Requires `--gpu-memory-utilization 0.95` on 32GB cards to accommodate weights + CUDA graphs + 32k KV cache.

---

## Deployment target

```
┌────────────────────────────────────────────────────────────────┐
│  Host (rootless podman, 2×RTX 5090, 192GB DDR5)                │
│                                                                │
│  ┌────────────────────────┐    ┌─────────────────────────┐     │
│  │  GPU 0                 │    │  GPU 1                  │     │
│  │                        │    │                         │     │
│  │  ┌────────────────┐    │    │  ┌────────────────┐     │     │
│  │  │ Process A      │    │    │  │ Process B      │     │     │
│  │  │ CODER (TP=1)   │    │    │  │ THINKER (TP=1) │     │     │
│  │  │ gpu-mem 0.85   │    │    │  │ gpu-mem 0.85   │     │     │
│  │  │ AWAKE          │    │    │  │ AWAKE          │     │     │
│  │  │ ~18 GiB total  │    │    │  │ ~18 GiB total  │     │     │
│  │  └───────┬────────┘    │    │  └───────┬────────┘     │     │
│  │          │             │    │          │              │     │
│  │          │             │    │          │              │     │
│  │  ┌───────┴────────┐    │    │  ┌───────┴────────┐     │     │
│  │  │ Process C      │    │    │  │ Process C      │     │     │
│  │  │ BEHEMOTH       │┈┈┈┈┼┈┈┈┈┼┈┈┤ (TP=2 across   │     │     │
│  │  │ gpu-mem 0.95   │    │    │  │ GPU 0+1)       │     │     │
│  │  │ ASLEEP L1      │    │    │  │ ASLEEP L1      │     │     │
│  │  └───────┬────────┘    │    │  └───────┬────────┘     │     │
│  └──────────┼─────────────┘    └──────────┼──────────────┘     │
│             │                             │                    │
│     http://localhost:30000        :30001  │         :30002     │
│             │                             │           │        │
│             └─────────────────────────────┴───────────┘        │
│                               │                                │
│                      ┌────────┴─────────┐                      │
│                      │    OpenCode      │                      │
│                      │  subagent route  │                      │
│                      │  (native v1.3+)  │                      │
│                      └──────────────────┘                      │
└────────────────────────────────────────────────────────────────┘
```

**Three roles, two steady-state processes, one on standby**:

| Role | When | Model candidates |
|------|------|------------------|
| Coder | Always hot | Qwen3-Coder-30B-A3B-AWQ *(baseline)*, GLM-4.7-Flash-AWQ *(MLA verified, V1 tool-broken)* |
| Thinker | Always hot | Qwen3.5-27B-AWQ *(baseline)*, GLM-4.5-Air-AWQ |
| Behemoth | On-demand wake | Qwen3-Coder-Next-80B-A3B-AWQ |

Steady state: coder + thinker both awake, parallel subagent calls are real (not serialized). Behemoth wakes when an escalated subagent is spawned.

## Why this shape

Three design moves, in order of importance.

### 1. TP=1-per-GPU is our placement for concurrent isolation

The prior assumption ("run both concurrent models sharing both GPUs at TP=2 to pool bandwidth") was dismantled by testing. Without root privileges to run NVIDIA MPS, two vLLM processes on the same GPU time-slice. With TP=2 amplifying kernel counts via NCCL allreduces, the context-switch penalty destroys throughput (~98% drop).

By deploying Coder to GPU0 exclusively and Thinker to GPU1 exclusively, we achieve absolute hardware-level concurrent isolation. 

Tradeoffs accepted:
- Coder (30B A3B) gets single-GPU bandwidth, but testing showed it actually decodes *faster* at TP=1 (251 t/s) than TP=2 (212 t/s) due to removed allreduce overhead.
- Thinker (27B dense-ish hybrid) drops from 106 t/s at TP=2 to ~76 t/s on a single GPU. This regression is acceptable since the thinker is not the latency-critical path.

### 2. Two small models coexist, one big model stands by

OpenCode's strength is parallel subagents (planner spawns coder, reviewer spawns critic). Coder and Thinker are both awake on separate GPUs.

The behemoth (Qwen3-Coder-Next 80B-A3B-AWQ) wouldn't fit alongside the other two — ~40 GiB weights across TP=2 = ~20 GiB per GPU. It lives as a pre-warmed **sleeping** vLLM process. Because Behemoth spans both GPUs (TP=2), waking it requires sleeping BOTH hot models first:

```
sleep(A) + sleep(B)   →   wake(C)   →   (use)   →   sleep(C) + wake(A) + wake(B)
  ~5–10s each                ~3–6s                       ~5–10s + ~0.3–1s each
```

Round trip into and out of behemoth: ~10–20s. Not free, but acceptable for "stuck, need the big gun" path.

### 3. vLLM Sleep Mode, not process restart

Cold-starting a vLLM process is 30–100s. Sleep-Mode wake is 0.1–6s because CUDA graphs, allocator, and JIT kernels are preserved. This is the only reason the three-tier plan is viable. Without it, "swap to the big model" would be a 2-minute operation and we'd be stuck at two-tier.

Consequence: all three processes are started **once** at system boot and sleep/wake thereafter. Total DRAM used by sleeping weights: ~18 + ~18 + ~40 = ~76 GiB. Fits easily in 192 GB.

## What this does NOT require

- LiteLLM for routing (OpenCode handles it).
- A classifier ML model for agent selection (OpenCode's agent descriptions drive `spawn_subagent`).
- NVLink (TP=2 for A3B is fine on PCIe).
- Model reloading / container restarts on switch.
- An external orchestrator — the three vLLM processes + OpenCode are the whole stack.

## Known unknowns (feed into `TESTING_QUEUE.md`)

These are the items that can kill or modify this architecture. Listed in rough blast-radius order:

1. ~~**Sleep Mode works under rootless podman with our socket setup.**~~ **SETTLED — T1.1 PASS.** Sleep Mode frees 92.8% VRAM in ~4s, wakes in 0.9s, post-wake TPS within 0.1% of baseline.
2. ~~**Qwen3-Coder-Next-80B-A3B-AWQ TP=2 decode speed.**~~ **SETTLED — T1.3 PASS.** 189 t/s seq=1, 13 kt/s prefill@32k. Behemoth tier is viable.
3. ~~**GLM-4.7-Flash MLA auto-detection**~~ **SETTLED — T2.1 INCONCLUSIVE (MLA active / tool-broken).** TRITON_MLA backend confirmed active in all tested vLLM builds via bench.log. Measured footprint ~129 KB/token (47 layers × kv_lora_rank=512 ≈ 94 KB base + CUDA-graph overhead). The original reference values (~54 KB MLA vs ~98 KB GQA) were incorrect for this model; actual GQA would be ~376 KB. GLM-4.7-Flash is in cold storage due to vLLM V1 tool-call crashes (EngineDeadError on Tasks 02/03, V0 engine cannot be forced). See `DECISIONS.md` and `RESEARCH_STATE.md` Open from testing.
4. **CPU prefix cache survival across sleep/wake.** Would let a re-woken model keep most of its prompt cache in DRAM. Not verified; if false, each wake pays full prefill on first request.

## How to change this document

This is the **plan**, not the record. When testing invalidates part of it:
1. Update `RESEARCH_STATE.md` with the finding.
2. In research mode, update this file (move the broken assumption to `DECISIONS.md` → SUPERSEDED, rewrite the affected section).
3. Update `TESTING_QUEUE.md` if new questions arise.