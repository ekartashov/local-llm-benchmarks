# Open Queue

Only OPEN and BLOCKED items. Full status table: docs/queue/status.md. Full procedures for DONE items: docs/history/done-items.md.

Legend: **OPEN** = ready to run (deps met). **BLOCKED** = deps or research needed. **DEFERRED** = not now.

---

## HIGH priority

### T_MTP1 — mtp_speculative_thinker — **DONE ✓ (BENCH_19, 2026-05-03)**

**Result:** MTP n=3 is optimal. 91.9 t/s N=1 (+79.1%), 314.8 t/s N=4 (+58.3%) vs PrismaQuant baseline. th02 reasoning intact. Tool calls: **5/5 PASS** under MTP n=3 (unlike the Coder which is 0/3 FAIL). MTP n=3 promoted to production. See `results/BENCH_19_mtp1_prismaquant_*/summary.md`.

---

### T_HARD1 — thinker_hard_suite — **DONE ✓ (BENCH_20, 2026-05-03)**

**Result:** PQ 41/50, AWQ 42/50 — statistical tie on a 10-task hard systems engineering suite. No quality gap found even at maximum task difficulty. Production decision confirmed on TPS grounds: PQ+MTP n=3 at 92 t/s vs AWQ 77 t/s.

**Score detail:** PQ task 03 (Raft asymmetric partition) truncated at 28K reasoning tokens in the final run (finish_reason=length, empty response content). A complete response from the prior run (095341Z, finish_reason=stop) covers all 4 sub-questions correctly and scores 5/5, giving PQ 43 vs AWQ 42 with that substitution. The truncation was an infrastructure artifact (max_tokens ceiling), not a quality failure.

**Both evaluations contained scoring errors:** research-mode Claude hallucinated specific term evidence for PQ task 04 (claimed XSNP_HITM present — absent in raw response) and AWQ task 10 (claimed --disable-eviction present — absent). Corrected scores are 41/50 and 42/50 respectively.

**Operational finding — context for re-runs:** The thinker's extended `<think>` budget on hard multi-step tasks can exceed 20–28K reasoning tokens. Any re-run of this suite must use:
- `--max-model-len 131072` on the vLLM deploy (costs ~0 extra VRAM on this hybrid DeltaNet model)
- `max_tokens: 32000` (or higher) in the task request payload

Without these, Raft-class and cache-coherence tasks will truncate mid-reasoning. The 32K default context window leaves only ~31K tokens after the system prompt — enough for most tasks but not for the deepest reasoning chains.

**Raw results:** `results/T_HARD1_thinker_hard_suite_20260503T105322Z/` (final scored run). Prior partial run: `results/T_HARD1_thinker_hard_suite_20260503T095341Z/` (PQ only, tasks 01/04/08 truncated at max_tokens=20K).

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

**Deps:** T_MTP1 ✓ DONE (BENCH_19 — MTP n=3 stable, no tool-call breakage).

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


### QX_PRELOAD — nvme_checkpoint_preload_mechanism — OPEN (design + implement)

**Context:** BENCH_18 (2026-05-03) proved posix_fadvise pre-warming works for fat checkpoints: Thinker 31GB fat dump (full GPU state) reduced from 19.8s cold to 12.0s warm (1.65×). **Convergence is a different case:** the Convergence CRIU dump is only 8.7GB (mmap, dirty pages only), but restore hits 100s first-inference due to demand-paging of the 123GB GGUF model files from NVMe. QX_PRELOAD for Convergence must pre-warm the GGUF model files, not the checkpoint images. This has not been tested.

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
