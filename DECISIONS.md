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

### Two vLLM processes on shared GPUs collapse to ~2% throughput
Two parallel vLLM processes sharing the same GPU(s) without NVIDIA MPS experience catastrophic performance degradation. Measured throughput drops to 4.2 t/s (vs 212 t/s isolated). Root cause: GPU-wide CUDA context time-slicing (the GPU time-slices contexts coarsely, and TP=2 amplify this via multiple kernel dispatches per layer), not just direct NCCL serialization. Do not retest without MPS.

### NVIDIA MPS (Multi-Process Service) is skipped
While MPS solves the context time-slicing issue above, it requires a privileged root daemon on the host which breaks our rootless podman invariant. sm_120 (consumer Blackwell) support is also unverified outside datacenter environments. A TP=1-per-GPU architecture avoids the problem entirely without root.

### Multi-model in a single vLLM process is not supported
vLLM does not support hosting more than one model weight set in one server process. This is an open upstream feature request with no ETA. Community workarounds are separate vLLM instances behind an nginx router, or third-party `llmux` for zero-reload switching. Do not plan around "merge coder + thinker into one vLLM" — it is not a path.

### Sleep Mode level=1 ~4 GiB residual is a design floor, not a tunable
When a vLLM instance sleeps at level=1, it retains ~4 GiB GPU VRAM (measured in T1.1). This is the caching allocator instance, captured CUDA graphs, JIT-compiled kernels, and process state — deliberately preserved to enable <1s wake. It cannot be shrunk without breaking the wake-time guarantee. Level=2 offloads more but is unusable (gibberish-on-wake, see separate entry). `wake_up(tags=[...])` affects wake granularity, not sleep footprint. The way to reclaim GPU memory while "sleeping" is to fully stop the container (T1.1c fallback path), not to tune level=1 lower.

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

### Qwen3.5-35B-A3B-AWQ is superseded by Qwen3.6-35B-A3B-AWQ
Old REVIEW status retained for the single-GPU failure record (22 t/s with `--enforce-eager`, <1 GiB headroom, 10× slowdown). **Qwen3.6-35B-A3B-AWQ** (Apache 2.0, released 2026-04-14, publisher `cyankiwi`) is the fresh-generation 35B-A3B candidate of record — same active-param class (3B), 262k native context, tool parser `qwen3_coder`, thinking-by-default. Queued as T2.5. The old Qwen3.5-35B-A3B entry should not be revived as a coder candidate; any 35B-A3B re-evaluation targets Qwen3.6. SGLang is separately incompatible with the QuantTrio Qwen3.5 weights (see config note on `qwen3_5.py:1662` weight-map bug — a code bug in SGLang, not a config problem).

### GLM-4.7-Flash (30B-A3B) MLA is active but tool-broken in vLLM V1
- **MLA Status**: **PASS**. Verified by measurement at **129.2 KB/token** (TP=2).
- **Infrastructure**: Requires `cu130-nightly` image and `transformers` from git to natively support `Glm4MoeLiteForCausalLM`.
- **Engine Status**: **BLOCKED**. vLLM forces the V1 engine for this architecture and ignores all V0 legacy-disable flags.
- **Quality Status**: **UNSTABLE**. V1 crashes (EngineDeadError) during tool generation for complex schemas (Tasks 02, 03).
- **Decision**: Avoid for tool-intensive roles until vLLM V1 stabilizes.

### GLM-4.6-Air does not exist
Z.ai released GLM-4.6V (vision, Air-sized) but skipped text-only Air. They went to GLM-4.7 flagship + GLM-4.7-Flash. No research gap — it was simply never released.

### GLM-4.6 and GLM-4.7 full (357B / 358B) are out of reach
Need ~8× datacenter-class GPUs at any serving quant that preserves capability. Not feasible on 2×5090.

---

## SETTLED — hardware/infra truth

### Behemoth (80B A3B MoE) on TP=2 is extremely viable
Verified in T1.3 (2026-04-18). 189.5 t/s seq=1 decode, 610 t/s at seq=4, 13007 t/s prefill at 32k context.
- **Requirement**: `--gpu-memory-utilization 0.95` on 32GB cards. 0.85 OOM'd during CUDA graph capture.
- **Requirement**: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` env var to allow vLLM to estimate graph memory without pre-allocating it; without this, graph capture itself triggers OOM.
- **Requirement**: `--tool-call-parser hermes --enable-auto-tool-choice`. Do **NOT** add `--reasoning-parser qwen3` — this causes hard failure regardless of the tool-call parser used. With `hermes`, the reasoning parser intercepts XML-tagged content (including tool calls) into the `reasoning` field before the tool parser sees it, resulting in 100% `no_call`. With `qwen3_xml`, it causes 100% `exception`. This is not a "benchmark-only" restriction — avoid entirely for this model.

### Dense 70B via TP=2 on PCIe x8/x8 no-NVLink → slow
Community-measured: 20–35 t/s. Our first-principles math shows this is NOT a bandwidth issue (decode allreduce = ~390 MB/s at 40 t/s, <2% of x8 PCIe). The cause is NCCL sync-point overhead and kernel launch latency accumulating over many layers at large hidden dims. Does not generalize to MoE with small active-parameter counts — see "MoE with ≤20B active params via TP=2 → fine" below.

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
Inter-publisher AWQ quality variance is within noise for our tasks. QuantTrio and cyankiwi are known-good. Note: `cpatonn/` returned HTTP 401 for Qwen3-Next-80B during T1.3 — do not use `cpatonn/` for that model; `cyankiwi/` is the working publisher.

**Re-evaluate if:** a specific quality discrepancy is observed on a task where the quantizer is a plausible suspect.

### Abliterated variants — not pursued
Orthogonal axis (refusal behavior), not a coding/infra capability axis. Irrelevant to our goals.

### Single-GPU-per-model placement — reinstated as default for the hot pair
Was deprioritized in R4, but T1.2's collapse under multi-process GPU sharing makes TP=1-per-GPU the mandatory topology for concurrent isolated execution (coder on GPU0, thinker on GPU1). TP=2 is now reserved exclusively for the behemoth model (which borrows both GPUs while the hot pair sleeps).

**Re-evaluate if:** T1.5 (`kvcached` spike) Phase B or Phase C passes — that would validate a memory-layer sharing primitive that sidesteps the context time-slicing problem without requiring MPS or root, and could revive TP=2-for-both-hot.

### kvcached as a third GPU-sharing path — provisional, test in T1.5
`ovg-project/kvcached` provides virtualized elastic KV cache (decouples GPU virtual from physical addressing), allowing multiple vLLM instances to share a KV pool dynamically on the same GPU. Tested with vLLM 0.19.0 (our version), supports MHA/GQA/MLA, actively maintained (v0.1.5, Red Hat endorsement). Operates at the memory layer, not the CUDA context layer, so it may sidestep the problem that killed naive concurrent TP=2 in T1.2 — without needing MPS or root.

**Status:** promising on paper. Not validated on our stack. Queued as T1.5 spike.

**Re-evaluate if:** T1.5 Phase B/C settles (either confirming it works — possibly reviving TP=2-for-both-hot — or confirming that context-slicing is orthogonal to KV sharing).

---

## SUPERSEDED — invalidated by later findings

### OLD: "One-line source patch for glm4_moe_lite"
**Superseded** — This was a legacy workaround for older vLLM images where `glm4_moe_lite` mapping was missing in `model_arch_config_convertor.py`. The current `cu130-nightly` image based stack with git-transformers resolves this natively. Do not apply source patches to modern images.

### OLD: "No dense 70B with tensor parallelism"
**Still true** but don't generalize it. The old wording implied "no TP period." TP=2 is fine for A3B MoE; it's dense-at-large-hidden-dim that pays the PCIe-sync penalty.

### OLD: bf16 fit test determines viability
Models were eliminated based on "doesn't fit in 32 GB bf16." This test is nearly meaningless for our deployment: we serve AWQ-INT4 (or similar), not bf16. Any model where the AWQ variant fits TP=2 at our chosen KV settings is a candidate regardless of bf16 size.

Affected models to re-evaluate if desired:
- Qwen3.5-35B-A3B on TP=2 (never retested there)
- Anything that OOM'd under single-GPU bf16 assumptions

### OLD: "Qwen3-Coder-Next needs GGUF, 160B bf16 doesn't fit single GPU"
**Superseded** — `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` exists and fits TP=2 on 2×5090 with room for KV cache. T1.3 PASS confirmed. Note: `cpatonn/` repo for this model returned HTTP 401 during T1.3 testing — use `cyankiwi/` exclusively.

### OLD: "Two concurrent TP=2 processes sharing GPUs"
**Superseded** — This assumption from R4 completely collapses due to CUDA context time-slicing reducing concurrent throughput to ~2% (see SETTLED). Architecture is pivoting to TP=1-per-GPU (coder GPU0, thinker GPU1) + behemoth TP=2 asleep.

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