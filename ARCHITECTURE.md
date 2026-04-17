<!-- last_updated: R5, 2026-04-17 — sleep-mode gating added after T1.1 fail-analysis -->
<!-- status: HYPOTHESIS — three-tier gated on T1.1 rerun; fallback path (T1.1c) defined but not yet selected -->
# ARCHITECTURE.md

Working architectural hypothesis. Single source of truth for "how we intend to deploy" — updated as research and testing converge.

Everything here is contingent on the tests listed in `TESTING_QUEUE.md`. This is the plan; reality will adjust it.

> **Gating note (R5, 2026-04-17):** the three-tier design hinges on vLLM Sleep Mode freeing weight VRAM on `/sleep?level=1`. The first T1.1 attempt ran the engine without `--enable-sleep-mode` and produced an invalid (negative) result. Rerun pending. If the corrected rerun passes, this document stands as-is. If the rerun fails for known-bug reasons, see T1.1a (vLLM version pin) or T1.1c (fall back to `podman stop`/`start` for the behemoth tier, swap ~30s instead of ~1s). The coder+thinker concurrent pair is not affected — they are always-hot, no sleep needed for steady state.

---

## Deployment target

```
┌────────────────────────────────────────────────────────────────┐
│  Host (rootless podman, 2×RTX 5090, 192GB DDR5)                │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GPU 0 + GPU 1, shared pool via TP=2 (PCIe 5.0 x8/x8)    │  │
│  │                                                          │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌───────────┐   │  │
│  │  │ Process A      │  │ Process B      │  │ Process C │   │  │
│  │  │ CODER (TP=2)   │  │ THINKER (TP=2) │  │ BEHEMOTH  │   │  │
│  │  │ gpu-mem 0.40   │  │ gpu-mem 0.40   │  │ (TP=2)    │   │  │
│  │  │ AWAKE          │  │ AWAKE          │  │ gpu-mem   │   │  │
│  │  │                │  │                │  │ 0.85      │   │  │
│  │  │ ~18 GiB each   │  │ ~18 GiB each   │  │ ASLEEP L1 │   │  │
│  │  │ across TP      │  │ across TP      │  │ (in DRAM) │   │  │
│  │  └───────┬────────┘  └───────┬────────┘  └─────┬─────┘   │  │
│  └──────────┼────────────────────┼──────────────────┼───────┘  │
│             │                    │                  │          │
│     http://localhost:30000  :30001            :30002           │
│             │                    │                  │          │
│             └────────────────────┴──────────────────┘          │
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
| Coder | Always hot | Qwen3-Coder-30B-A3B-AWQ *(baseline)*, GLM-4.7-Flash-AWQ |
| Thinker | Always hot | Qwen3.5-27B-AWQ *(baseline)*, GLM-4.5-Air-AWQ |
| Behemoth | On-demand wake | Qwen3-Coder-Next-80B-A3B-AWQ |

Steady state: coder + thinker both awake, parallel subagent calls are real (not serialized). Behemoth wakes when an escalated subagent is spawned.

## Why this shape

Three design moves, in order of importance.

### 1. TP=2 is our default placement, not a workaround

The old assumption was "one model per GPU." That was wrong for our workload:
- Most A3B MoE models have ~18 GiB AWQ weights. Two of them on one GPU is tight; two on separate GPUs wastes the other card's VRAM when one model is idle.
- Real agentic sessions have lopsided KV cache demand: a long architecture-planning thinker session can want 15+ GiB of KV while the coder is idle. Single-GPU placement can't reallocate.
- For A3B models, TP=2 over PCIe 5.0 x8/x8 has negligible decode cost (~0.2% bandwidth, see `DECISIONS.md`). The folk wisdom "no NVLink = no TP" applies to dense models at large hidden dims, not to A3B MoE.

So: every model serves via TP=2 by default. Single-GPU is a fallback for models whose quant kernel can't split (e.g. Marlin MoE group_size=64 caps at TP=2 but wouldn't scale to 4; group_size choice can rule out TP entirely for some quants).

### 2. Two small models coexist, one big model stands by

OpenCode's strength is parallel subagents (planner spawns coder, reviewer spawns critic, etc). Forcing these serial because only one model can be hot defeats the frontend design. So: coder and thinker both awake, at `--gpu-memory-utilization 0.40` each.

The behemoth (Qwen3-Coder-Next 80B-A3B-AWQ) wouldn't fit alongside the other two — ~40 GiB weights across TP=2 = ~20 GiB per GPU, colliding with the active pair. It lives as a pre-warmed **sleeping** vLLM process with `--gpu-memory-utilization 0.85`. When needed:

```
sleep(A) + sleep(B)   →   wake(C)   →   (use)   →   sleep(C) + wake(A) + wake(B)
  ~5–10s each                ~3–6s                       ~5–10s + ~0.3–1s each
```

Round trip into and out of behemoth: ~10–20s. Not free, but acceptable for "stuck, need the big gun" path. Not the default path.

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

1. **Two concurrent vLLM processes on the same GPU pair at `--gpu-memory-utilization 0.40` each coexist cleanly.** Core assumption. If they interfere (CUDA graph corruption, allocator fights, context contention beyond what math predicts), we fall back to one-hot-at-a-time via Sleep Mode for the small pair too.
2. **Sleep Mode works under rootless podman with our socket setup.** Endpoints require `VLLM_SERVER_DEV_MODE=1` on trusted networks — the operational path through podman + OpenCode has not been verified end-to-end.
3. **Qwen3-Coder-Next-80B-A3B-AWQ TP=2 decode speed.** If it lands below ~40 t/s, the behemoth tier is not viable and we become two-tier.
4. **Prefill concurrency under parallel subagents.** Decode is fine for two concurrent A3B models (math shows ~34% bandwidth utilization). Prefill is compute-bound and heavier; if two subagents prefill simultaneously we may need a semaphore in the router or accept serialized prefill with concurrent decode.
5. **GLM-4.7-Flash MLA auto-detection** in our current vLLM. A known bug had MLA not triggering → 10× KV cache bloat. One-line fix landed; we need to verify per-token KV size matches MLA (~54 KB) not GQA (~98 KB).
6. **CPU prefix cache survival across sleep/wake.** Would let a re-woken model keep most of its prompt cache in DRAM. Not verified; if false, each wake pays full prefill on first request.

## How to change this document

This is the **plan**, not the record. When testing invalidates part of it:
1. Update `RESEARCH_STATE.md` with the finding.
2. In research mode, update this file (move the broken assumption to `DECISIONS.md` → SUPERSEDED, rewrite the affected section).
3. Update `TESTING_QUEUE.md` if new questions arise.