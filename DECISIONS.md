# DECISIONS.md

Settled and provisional decisions, tiered by confidence. An item here is **not retested** — if you're about to run a benchmark, check this file first.

Legend:
- **SETTLED** — research + reasoning + (where applicable) measured evidence converge. Do not retest.
- **PROVISIONAL** — strong prior from research but contingent on engine/hardware/quant details that can shift. Re-evaluate if the condition listed changes.
- **SUPERSEDED** — was decided, is now invalidated (explains why).

Items removed entirely: the previous "Decisions already settled (do not re-evaluate)" list in the old `CLAUDE.md` is fully migrated here with updated rationale.

---

## SETTLED — infrastructure / tooling

### No Ollama
Adds 10–30% overhead vs raw engine containers; known broken tool parser for Qwen3.5 family (Ollama issue #14493). Nothing to measure.

### No KTransformers
CPU offload path depends on AMX instructions. i9-14900K (Raptor Lake) does not have AMX. Project is architecturally unsuited to our CPU.

### Engines are containerized
Rootless podman only. No host installs of vLLM/SGLang/llama.cpp. Not a performance decision — an operational one (reproducibility, isolation, cleanup).

### System RAM is not VRAM for KV cache
Offloading KV cache to DDR5 costs 50–80% decode speed (memory bandwidth gap: 83 GB/s vs 1790 GB/s). We keep working set in GDDR7. System RAM is fine for **weight storage during sleep** (see vLLM Sleep Mode) but not for active KV.

### vLLM Sleep Mode is our swap primitive (gated on T1.1 rerun)
Level 1 offloads weights to CPU RAM, discards KV cache, preserves CUDA graphs / allocator / JIT kernels. Wake times reported at 0.1–6s depending on model size — 18–200× faster than cold reload. Official docs confirm this works with TP/PP/EP. SGLang and llama.cpp have no equivalent first-class feature; this alone keeps vLLM as our primary engine for multi-model setups.

**Two flags are required — both are mandatory, not alternatives:**
- `VLLM_SERVER_DEV_MODE=1` (environment variable): exposes the `/sleep`, `/wake_up`, `/is_sleeping`, `/collective_rpc`, `/pause`, `/resume` HTTP routes. Without this, the routes return 404.
- `--enable-sleep-mode` (vllm serve flag): makes the engine initialize with `CuMemAllocator` and reserve the "weights" memory pool. Without this, the `/sleep` route is a control-plane no-op — `is_sleeping` toggles but no memory is released, because there is no pool context that the allocator can release from.

The R5 cycle (2026-04-17) diagnosed T1.1's initial FAIL as the `--enable-sleep-mode` flag being missing from the vllm serve command. The rerun has not been executed yet — hence "gated."

**Level 1 only.** Do not use level=2. Bug #29341 (Nov 2025, H100): level=2 wake produces gibberish output. Level=2 also requires manual `collective_rpc reload_weights` + `reset_prefix_cache` after wake (vLLM blog docs) — easy to get wrong. Level=1 keeps weights in CPU RAM, which is fine for us (192 GB DDR5, plenty of headroom for all three tiers' worth of weights).

Caveats, not blockers:
- Trusted-network only — `VLLM_SERVER_DEV_MODE=1` exposes dev endpoints. Our rootless podman local socket qualifies.
- KV cache is flushed on sleep (cannot free GPU memory while preserving blocks).
- CPU prefix cache survival across sleep is **not verified** — flagged as test item T3.4.

**Known risks to watch for in T1.1 rerun:**
1. **Regression bug #32714** ("Sleep is broken since 0.14.0"): on v0.14+, sleep frees ~30% of expected memory rather than ~90%+. Issue marked Closed but RFC #34303 (Feb 2026) still cites it as "broken since v0.14.0." If our 0.19.0 rerun shows ~30% VRAM freed, this regression applies and we need to pin an earlier version or wait for upstream fix.
2. **Blackwell crash bug #21336** (RTX PRO 6000 sm_120 + vLLM 0.9.2 + TP=2 + GPTQ-Marlin + `--enable-sleep-mode` → crash at startup). Our hardware is the same architecture class (consumer sm_120), same TP placement, but AWQ-Marlin not GPTQ-Marlin. Status on 0.19.0 unknown. Watch for crash at startup after adding the flag.

last_verified_vllm: "0.19.0" (API routes present and functional; weight-offload mechanism not yet verified end-to-end)

---

## SETTLED — models

### Qwen3-Coder-30B-A3B-AWQ is a viable coder baseline
Measured: 251 t/s single-request, ~730 t/s aggregate at concurrency=4, single-GPU. Parser `--tool-call-parser qwen3_coder --reasoning-parser qwen3` works. Default contender against any new coder candidate.

### Qwen3.5-27B-AWQ is viable as a thinker on quality
Measured 4.0/5 vs DeepSeek-R1-32B 2.6/5 on 8 reasoning tasks. Hybrid SSM needs `--max-num-seqs 1` on vLLM to avoid CUDA graph profiling OOM; 76 t/s with graphs active. See `PHASE2_RESULTS.md`.

Outstanding defect: `th03_architecture_tradeoffs` always exceeds 8192-token `<think>` budget → empty output. Fix: raise `--max-tokens` to ≥16384 on the thinker endpoint, or route architecture-heavy tasks around the thinker. Must resolve before "thinker: Qwen3.5-27B" is truly settled.

### DeepSeek-R1-32B-AWQ is not our thinker
Quality 2.6/5 dominated by Qwen3.5-27B 4.0/5 across task categories. Not marginally worse — structurally worse (surface-level fixes, wrong diagnoses, misses the point on consistency scenarios). Do not pursue further.

### Devstral is eliminated
bf16 OOMs at 30.4 GiB, and at any quant its quality is below Qwen3-Coder-30B-AWQ on our tasks. No path to viability.

### Qwen3.5-35B-A3B-AWQ on single GPU is dead, but **not** in general
Measured: 22 t/s with `--enforce-eager` (forced because the 22 GiB bf16 weight load leaves <1 GiB headroom, below the ~1.03 GiB CUDA graph profiling needs). 10× slowdown. **Status on TP=2 is unknown** — more VRAM headroom could allow graphs. Not currently a priority; if retested, queue item should be explicit. SGLang is separately incompatible (see config note on `qwen3_5.py:1662` weight-map bug — a code bug in SGLang, not a config problem).

### GLM-4.6-Air does not exist
Z.ai released GLM-4.6V (vision, Air-sized) but skipped text-only Air. They went to GLM-4.7 flagship + GLM-4.7-Flash. No research gap — it was simply never released.

### GLM-4.6 and GLM-4.7 full (357B / 358B) are out of reach
Need ~8× datacenter-class GPUs at any serving quant that preserves capability. Not feasible on 2×5090.

---

## SETTLED — architecture rules of thumb

### Dense 70B via TP=2 on PCIe x8/x8 no-NVLink → slow
Measured across community reports: 20–35 t/s. Our first-principles math says this is NOT a bandwidth issue (decode allreduce = ~390 MB/s at 40 t/s, < 2% of x8 PCIe). The cause is NCCL sync-point overhead and kernel launch latency accumulating over many layers at large hidden dims. Does not generalize to MoE with small active-parameter counts — see next.

### MoE with ≤20B active params via TP=2 → fine
First-principles: A3B decode allreduce is ~60 MB/s at 150 t/s, ~0.2% of x8 PCIe bandwidth. Prefill at 32k ctx adds ~500ms to TTFT (~8% overhead). Expected to deliver 85–95% of TP=1-equivalent throughput. Measured numbers for our specific stack are a test item, but the architectural viability is settled — we don't need to ask "does TP=2 work for A3B at all."

### Speculative decoding does not help MoE
MoE active-param savings already address the memory-bandwidth bottleneck that spec decode targets. Only test spec decode on dense models.

**Re-evaluate if:** a new spec decode variant explicitly designed for MoE (e.g. MTP in GLM-4.7, Qwen3-Next MTP) ships with kernel-level validation on Blackwell sm_120. Currently:
- GLM-4.7-Flash MTP: reported 10× throughput regression on B200 (Blackwell pro). Hopper OK. Consumer Blackwell sm_120 unknown → **do not enable by default, measure first** if pursuing GLM-4.7-Flash.
- Qwen3-Next MTP: documented in model card, status on sm_120 untested.

---

## PROVISIONAL — re-check if condition changes

### NVFP4 as a standalone optimization phase — deprioritized
Blackwell FP4/FP8 kernels in vLLM/CUTLASS are still maturing. NVFP4-W4A16 only beats AWQ-Marlin in narrow regimes; AWQ-Marlin is already excellent for A3B MoE on 5090.

**Re-evaluate if:**
- We add a dense model where VRAM headroom becomes tight.
- TensorRT-LLM's FP4 path becomes accessible through our container stack.
- vLLM's CUTLASS FP4 kernels land a major perf update.
- vLLM major version bump (currently verified on 0.19.x).

### LiteLLM as a classifier / router — not needed
OpenCode v1.3+ (Feb 2026) supports native multi-endpoint subagent routing. Agent ID → Model ID → Provider baseURL is deterministic, and autonomous subagent spawning routes across ports natively. LiteLLM is redundant at the routing layer for our setup.

**Re-evaluate if:** we later want a frontend other than OpenCode (e.g. direct IDE integrations) that lacks native multi-endpoint routing.
- vLLM major version bump (currently verified on 0.19.x).

Retained LiteLLM use case: observability / failover / request logging across endpoints. Not routing.

### Alternative AWQ publishers — not pursued
Inter-publisher AWQ quality variance is within noise for our tasks. QuantTrio and cyankiwi/cpatonn are known-good.

**Re-evaluate if:** a specific quality discrepancy is observed on a task where the quantizer is a plausible suspect.

### Abliterated variants — not pursued
Orthogonal axis (refusal behavior), not a coding/infra capability axis. Irrelevant to our goals.

### Single-GPU-per-model placement — deprioritized as default
Was the default in the original phase plan. Supplanted by TP=2 as working default. Single-GPU placement is still useful when:
- We want two concurrent models hot on separate cards (but this usually loses to TP=2 both + sleep mode due to memory imbalance).
- A specific model's AWQ kernel can't tensor-parallelize (e.g. group_size=64 Marlin restriction — model-specific).

---

## SUPERSEDED — invalidated by later findings

### OLD: "No dense 70B with tensor parallelism"
**Still true** but don't generalize it. The old wording implied "no TP period." TP=2 is fine for A3B MoE; it's dense-at-large-hidden-dim that pays the PCIe-sync penalty.

### OLD: bf16 fit test determines viability
Models were eliminated based on "doesn't fit in 32 GB bf16." This test is nearly meaningless for our deployment: we serve AWQ-INT4 (or similar), not bf16. Any model where the AWQ variant fits TP=2 at our chosen KV settings is a candidate regardless of bf16 size.

Affected models to re-evaluate if desired:
- Qwen3.5-35B-A3B on TP=2 (never retested there)
- Anything that OOM'd under single-GPU bf16 assumptions

### OLD: "Qwen3-Coder-Next needs GGUF, 160B bf16 doesn't fit single GPU"
**Superseded** — `cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` now exists. Fits TP=2 on 2×5090 with room for KV cache. Promoted to behemoth-tier test item.

### OLD: LiteLLM as router
**Superseded** — OpenCode native routing replaces it. See PROVISIONAL entry above.

### OLD: "Phase 2 winner: Qwen3.5-27B-AWQ" (standalone)
**Still provisionally valid** as a single-model thinker. But the architectural context changed: we now plan concurrent coder + thinker + behemoth-on-standby via sleep mode. Thinker selection must be re-scored against GLM-4.5-Air as a TP=2 candidate before final commitment.

---

## How to edit this file

When a new decision is made in research mode, add it under the appropriate tier with:
1. One-line claim.
2. The evidence (measured / reasoned-from-math / cited).
3. A "re-evaluate if" clause for PROVISIONAL items, or what superseded it for SUPERSEDED items.

When testing discovers something that contradicts an item here, do NOT edit it directly from testing mode. Record the contradiction in `RESEARCH_STATE.md` under "Open from testing" and hand back to research.