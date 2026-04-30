# Open Queue

Only OPEN and BLOCKED items. Full status table: docs/queue/status.md. Full procedures for DONE items: docs/history/done-items.md.

Legend: **OPEN** = ready to run (deps met). **BLOCKED** = deps or research needed. **DEFERRED** = not now.

---

## HIGH priority

### T_MTP1 — mtp_speculative_thinker — OPEN

**Question:** Does MTP n=1 give measurable TPS improvement on the production thinker (AWQ, no model swap)?

**Why HIGH priority:** No container changes, no new model download. One flag addition to an existing endpoint. Community benchmark (RTX 3090, vLLM 0.19.1) shows −21.6% TPOT ≡ **+27.5% decode TPS** at n=1. For max-num-seqs=4, thinker at single-request load is the best-case scenario for MTP acceptance rates.

**No prerequisite blockers.** vLLM #40756 (MTP crash) does not apply to our config: bug conditions are FP8+TP4+n=5+25K tokens; we use AWQ+TP1+n=1. vLLM 0.19.1 was a Gemma4-only patch — no difference from 0.19.0 for this test. Ready to run.

**Deploy command:**
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

**Metrics to collect:**
- TPS at N=1 with MTP vs baseline (N=1 without MTP, same max-num-seqs=4 config)
- TTFT at N=1 with MTP vs baseline
- TPS at N=4 with MTP (does MTP hurt throughput under load?)
- Quality smoke check: th02 correct? (MTP acceptance rate can affect output quality)

**Pass:** TPS at N=1 improves ≥10% vs baseline with th02 still correct.
**Fail / rollback:** MTP causes CUDA error, crash, or th02 regression → revert to no `--speculative-config`.

**Hand-back trigger:** vLLM crash (not rejection/fallback), or th02 semantic regression → research.

---

### T_MTP2 — mtp_speculative_coder — OPEN (run after T_MTP1)

**Question:** Does vLLM native MTP give TPS gain on the A3B MoE coder, where llama.cpp spec-decode gives no gain?

**Why this differs from llama.cpp:** llama.cpp speculative decoding uses a separate draft model → different expert routing per speculative token → overhead kills the gain. vLLM MTP uses the model's own MTP head in the same forward pass → expert routing happens once → no extra expert loading.

**Prerequisite:** T_MTP1 complete (confirm approach is stable before touching coder).

**Deploy command:**
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu0gpu1 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --tensor-parallel-size 2 --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

**Pass:** Aggregate TPS improves ≥5% vs current 232–237 t/s baseline.

---

### T_KV3 — thinker_extended_context — UNBLOCKED (Path B ready)

**Target: 128K context** — Qwen3.6-27B native limit. T3.1 Phase 1 (50K) showed 0 MiB VRAM delta; the same should hold all the way to 128K (DeltaNet fixed-size recurrent state, zero KV cache growth). This is now a functional ceiling test, not a VRAM feasibility test.

**Why this is CRITICAL:** Real thinker workloads hit 32K ceiling. 128K context with zero VRAM cost changes the use case entirely.

**Why this is CRITICAL (not just high-priority):** The thinker is the model most in need of large context — reasoning over long chains, large codebases, multi-document synthesis. Real workloads already show the 27B thinker hitting context ceiling and failing to conclude. Coder extended context (65K, SETTLED) is useful but less urgent. Extended thinker is operationally necessary.

**Sub-Q1 SETTLED (T2.4g):** GDN TP=2 broken regardless of chunked-prefill. Tensor-parallel sharding of DeltaNet state is mathematically incorrect.

**Sub-Q2 BLOCKED — two unblocking paths (either one suffices):**

**Path A: Non-GDN replacement (vLLM + AWQ)**
Find a thinker candidate that:
- Is NOT GDN-hybrid (pure Transformer or MLA) — TP=2 shard is mathematically safe
- Quality ≥ Qwen3.6-27B (4.875/5) on the 8-task thinker suite
- Fits ~21GB AWQ at TP=1 for normal hot-pair mode
- First candidates to research: DeepSeek-R1-Distill-Qwen-32B, QwQ-32B (pure Transformer reasoning models with strong AIME/GPQA scores). Verify architecture in model card before testing.

**Path B: ik_llama.cpp + tensor-split on existing Qwen3.6-27B (no model swap needed)**
ik_llama.cpp pr-1288 already has DeltaNet support (used by 397B MoE). llama.cpp's `--tensor-split` is layer-split (pipeline parallelism), NOT tensor-parallel sharding. DeltaNet recurrent state lives entirely within a single layer and is never split across GPUs — the vLLM TP=2 failure mode does not apply.

Test: run Qwen3.6-27B GGUF with `--tensor-split 0.5,0.5` on ik_llama.cpp, verify inference correctness on the thinker task suite (specifically th02 which catches GDN recurrent state errors). If correct: VRAM splits ~16GB per GPU, enabling larger KV cache. This path is lower-risk and doesn't require finding a new model.

**No script yet.** Research mode required first. Return here after research provides a model slug + deploy config.

---

### T_CRIU3 — criu_universal_checkpoint_library — OPEN (design + implement)

**Question:** Standardize CRIU checkpointing for ALL vLLM processes, not just coder-tp2. Enable the model checkpoint library concept.

**What this enables:**
- Any model config can be restored in 0.28s instead of ~100s cold start
- T_KV3 model swap iteration speed: checkpoint settled thinker → swap candidate → evaluate → restore settled thinker (0.28s instead of 100s per iteration)
- Path to Sequential TP=2 architecture: all models checkpointed, any one restored on demand
- Frees 44GB RAM currently held by sleep-mode Arclight weights → enables UD-IQ3_XXS for Convergence

**Scope:**
1. Document checkpoint management: naming convention, storage location, expiry policy
2. Add checkpoint targets: coder-tp2-32k, thinker-tp1-32k, (future) extended variants
3. Automate: deploy.sh should optionally checkpoint before stopping any vLLM process
4. Pre-load hook: `posix_fadvise(POSIX_FADV_WILLNEED)` or background cat on checkpoint file when a model switch is anticipated. Relevant for large checkpoints (ik_llama.cpp --no-mmap) where SSD restore is 18s without pre-warm vs 0.28s with page-cache warm.

**KV cache preservation sub-question (measure as part of this item):**
`cuda-checkpoint` captures all CUDA memory state, including vLLM's paged-attention KV blocks. A checkpoint taken after a long-context session should restore with the KV cache populated — meaning no re-prefill on restore. This is qualitatively better than sleep mode (which drops KV state on process restart). Questions to answer:
- Confirm: does the restored model respond to continuations without re-prefilling the prior context?
- Checkpoint size delta: how much does a populated KV cache (e.g., 16K tokens at fp8) add to the checkpoint file size?
- SSD wear policy: frequent CRIU dumps with full KV cache vs. checkpoint only at idle (minimal cache). The NM790 has 3,000 TBW; at 37GB checkpoint + 20 dumps/day = 740 GB/day = ~11 years. Generous, but establish the write-cost model before designing an aggressive dump policy.

**Deps:** T_KV2 ✓ (CRIU mechanism settled for vLLM). T_CRIU2 ✓ DONE.

---

## MEDIUM priority

### T_PQ1 — prismaquant_thinker — OPEN (requires CUDA 13.0 container rebuild)

**Question:** Does rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm give better TPS and/or quality than the current AWQ thinker?

**Why MEDIUM (not HIGH):** Requires a non-trivial container rebuild (CUDA 13.0 + FlashInfer 0.6.5 SM120 patches). Without the rebuild, the model may still load via compressed-tensors with a cutlass fallback — worth attempting without rebuild first.

**Two-phase approach:**
1. **Phase 1 (no rebuild):** Deploy as-is, `VLLM_USE_V1=0`, measure TPS vs AWQ. If compressed-tensors loads and falls back to non-FP4 kernels, quality improvement from GPTQ calibration (0.33× RTN MSE) is still real.
2. **Phase 2 (after CUDA 13.0 rebuild):** Confirm FlashInfer NVFP4 kernels activate. Compare TPS Phase 1 vs Phase 2.

**Deploy command (Phase 1):**
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
  --trust-remote-code --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
```

**Metrics:** TPS at N=1, TPS at N=4, quality on th02 (semantic correctness test), VRAM. Compare all vs AWQ baseline.

**Pass:** Quality ≥ 4.875/5 on thinker suite AND TPS within 10% of AWQ (quality win alone is sufficient — PrismaQuant at 5.5bit > AWQ 4bit per-bit quality).

**Deps:** T_MTP1 complete (confirm MTP is stable before stacking with a new model).

---

### T3.4 — prefix_cache_survival_across_sleep — OPEN (scope updated)

**Original question:** Does vLLM's CPU-offloaded prefix cache survive a sleep/wake cycle at level=1?

**Updated scope with CRIU context:** CRIU checkpoints all process memory (including CPU-side prefix cache, since it lives in anonymous process RAM). Prefix cache survival across CRIU restore is expected by construction — but needs empirical verification. Run this test against both sleep/wake AND CRIU restore to understand the full picture.

**Procedure:**
1. Deploy any model with `--enable-prefix-caching --cpu-offload-gb 8`, TP=2.
2. Send 4k-token prompt, record prefill time.
3. Send same prompt again — verify cache hit (prefill time drops).
4a. `POST /sleep?level=1`, then `POST /wake_up`. Send same prompt, measure prefill time.
4b. CRIU checkpoint the process. Restore. Send same prompt, measure prefill time.

**Pass:** CRIU restore preserves prefix cache (prefill ≈ cached time). Sleep/wake result may differ.

**What failure means (sleep):** Prefix cache flushed on sleep → user-visible latency on first post-wake request. Note in `docs/arch/current.md`. Does not affect CRIU path.

**Deps:** T1.1 ✓. Script at `benchmarks/queue/T3.4_prefix_cache_survival.sh` (verify exists or create).

---

### T2.6 — behemoth_archetype_scouting — DESIGN ITEM (no benchmark)

**What it produces:**
1. Candidate archetype taxonomy (2–3 archetypes, short rationale).
2. Per-archetype 1–2 model shortlist with VRAM, context, tool-parser status.
3. Concrete sub-items (T2.6.1, T2.6.2, …) queued after triage.

**Out of scope:** running any benchmark. This is a research-mode item.

**Deps:** T1.3 ✓, T2.4 ✓.

---

### T_ENGINE_EVAL — engine_comparison_for_arclight_roles — OPEN (research + test)

**Context:** vLLM was originally chosen partly for its sleep mode. With CRIU providing engine-agnostic fast-swap, engine selection should now be based purely on capability: TPS, architecture support, tool-call reliability.

**Models that failed on vLLM and should be re-evaluated on ik_llama.cpp:**
- GLM-4.7-Flash (30B-A3B): `TRITON_MLA PIECEWISE CUDA graph instability on Blackwell`. This is a vLLM Triton kernel issue. ik_llama.cpp uses its own CUDA kernels (not Triton) for MLA — the same crash may not occur.
- Any future model with CUDA graph / EngineCore conflicts on vLLM 0.19.0 + sm_120.

**Scope of this item:**
1. Test GLM-4.7-Flash on ik_llama.cpp: load GGUF, run tool-call suite, measure TPS. Compare to coder baseline (232 t/s).
2. If GLM passes: re-open T2.2 with ik_llama.cpp as the engine instead of vLLM.
3. Document cross-engine compatibility decisions in a new `docs/decisions/engines.md`.

**Deps:** T_CRIU2 (confirms ik_llama.cpp is a viable engine for CRIU deployment). Can be researched in parallel.

---

### QX_PRELOAD — nvme_checkpoint_preload_mechanism — OPEN (design + implement)

**Context:** CRIU restore from page cache (RAM-warm checkpoint) = 0.28s. Restore from cold NVMe = proportional to checkpoint size (e.g., 135GB at 7,400 MB/s = ~18s for ik_llama.cpp --no-mmap). Pre-loading the checkpoint into OS page cache before the switch eliminates the disk latency.

**Hardware:** Lexar NM790 4TB (7,400 MB/s read). Loading 135GB → ~18s. Router needs ~10s advance notice to trigger pre-warm before the switch is visible to the user.

**Pre-warm mechanism:**
```bash
posix_fadvise(fd, 0, checkpoint_size, POSIX_FADV_WILLNEED)
# or equivalently:
cat /srv/ai/checkpoints/convergence/checkpoint.tar.gz > /dev/null &
```

**Router integration points:**
- OpenCode routing: "next task invokes Convergence" → trigger pre-warm of Convergence checkpoint
- Context growth triggers: token count approaching 30K → pre-warm extended-coder checkpoint
- Subagent dispatch: explicit @thinker routing → pre-warm thinker checkpoint

**Pass:** warm restore ≤ 1s for checkpoints whose pre-warm completed. Cold restore time documented per checkpoint type.

**Deps:** T_CRIU3 (checkpoint library standardization). NVMe hardware already in place.

---

## LOW priority (optimization tier — pending settled roles)

### T3.1 — kv_cache_q8_q4_on_thinker

**Blocked on:** T2.3 settled thinker (T2.3 depends on T2.2 which depends on T2.1 in cold storage).
But the current thinker (Qwen3.6-27B) is settled — this can run against it now.

**Question:** At what KV cache quantization can the thinker hold 30k–50k context without quality loss?

**Configs to test:**
- `--kv-cache-dtype auto` (bf16 equiv), context 32k — baseline
- `--kv-cache-dtype fp8`, context 30k
- `--kv-cache-dtype fp8`, context 50k
- `--kv-cache-dtype int4` (if supported), context 50k

**Pass:** fp8 at 30k shows no quality regression vs baseline. Decide int4 trade-off for 50k.

---

### T3.3 — qwen3_next_mtp_on_behemoth

**Question:** Does Qwen3-Coder-Next's `qwen3_next_mtp` spec decode work on sm_120?

**Deps:** T1.3 ✓, T2.4 ✓. Only if Core/behemoth tier is re-activated (currently suspended).

---

### T_NVFP4 — nvfp4_thinker_mass_survey — DEFERRED

**Deferred indefinitely.** T2.4c used untrusted publisher (sakamakismile). Cannot distinguish format benefit from publisher quality. TP=1 only (TP=2 broken for GDN). When reconsidered: use `nvidia/` or `bartowski/` publishers only. Re-evaluate if: AWQ + bf16 KV + TP=2 passes (KV precision was the issue) OR if a non-GDN thinker with TP=2 viability is found.

---

### T_TRT_LLM — tensorrt_llm_peak_throughput — LOW (post-settlement optimization)

**What it is:** TensorRT-LLM (TRT-LLM) is NVIDIA's native inference compiler. It compiles a model into a GPU-specific TRT engine and can extract 20–50% higher TPS than vLLM on the same hardware for well-supported architectures. FP8 is natively accelerated on Blackwell sm_120, not emulated.

**Why it's LOW priority now:** Compilation step takes hours per model and is GPU-specific. This makes it incompatible with the model exploration phase — every candidate swap requires a full recompile. It only makes sense once:
1. All role assignments are SETTLED and stable (no more model changes expected for 6+ months)
2. T_CRIU3 and T_ENGINE_EVAL are done (CRIU library and engine-agnostic deployment settled)
3. The architecture is in a maintenance phase, not an exploration phase

**What it would prove:** TPS ceiling for production Arclight. If coder goes from 232 → 300+ t/s and thinker from 77 → 100+ t/s, that's a meaningful latency improvement for the agent loop.

**Risks:** Architecture support for MoE A3B + GDN models in TRT-LLM may be incomplete. TRITON_MLA issues that affect vLLM on sm_120 could have TRT equivalents. Start with the coder (well-understood MoE, no recurrent state) before attempting thinker.

**Prerequisites:** All Arclight roles SETTLED. TRT-LLM version with confirmed sm_120 / Blackwell support. Per-model FP8 calibration dataset available.

---

## Integration tier (T5.x — BLOCKED on T2.x settling)

### T5.1 — opencode_endpoint_binding_with_subagents
**Deps:** T2.2, T2.3 (both blocked due to GLM cold storage). Can partially test with current Arclight + Convergence endpoints.

**Procedure:**
1. Configure `opencode.json` with `vllm-coder` / `vllm-thinker` / `vllm-convergence` providers.
2. Define three agents with matching model overrides.
3. Run task invoking all three. Verify correct endpoint routing from request logs.
4. Kill one endpoint mid-task; verify graceful failure surfacing.

### T5.2 — behemoth_wake_trigger_integration
DESIGN — write up in `docs/arch/current.md` after T2.4 settles. Likely: explicit `@behemoth` subagent in OpenCode.

### T5.3 — mcp_servers_firecrawl_searxng
**Deps:** T5.1. Functional smoke test: 5 doc-lookup queries, verify answers cite sources.

---

## Task suite extension (T6.x — ongoing, not blocking)

### T6.2 — cross_arch_tasks
Tasks: Orange Pi / armbian cross-compilation, kernel module questions, cloud-init idempotency.

### T6.3 — ops_tasks_ceph_openstack
Ceph PG troubleshooting, OpenStack service diagnosis. At least 3 tasks each.

### T6.4 — rag_aware_tasks
Explicit doc-lookup tasks to test MCP pipeline end-to-end. **Deps:** T5.3.

**Scheduling:** run T6.x back through T2.x to re-score settled roles. A coder winner on coding suite may not win on infra suite.

---

## Parking lot — explicit non-items

- NVFP4 as a standalone phase — deferred (see T_NVFP4 above)
- Alternative AWQ publishers — QuantTrio and cyankiwi are known-good; not worth testing others
- Abliterated variants — off-topic
- Dense 70B TP=2 — settled FAIL (20–35 t/s, see `docs/decisions/settled.md`)
- Cloud quality baselines — not relevant (local-only goal)
- Cost analysis — irrelevant (electricity already paid)
