# Open Queue

Only OPEN and BLOCKED items. Full status table: docs/queue/status.md. Full procedures for DONE items: docs/history/done-items.md.

Legend: **OPEN** = ready to run (deps met). **BLOCKED** = deps or research needed. **DEFERRED** = not now.

---

## HIGH priority

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

### T_PQ2 — prismaquant_coder_phase1 — OPEN (stability rerun required)

**Question:** Is `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` stable enough on tool calls to be a viable TP=1 coder fallback?

**Current evidence (BENCH_23, completed runs):**
1. `20260505T130957Z` (eager): 12.0 t/s agg (N=1), tool calls 4/5.
2. `20260505T170250Z` (graphs): 56.5 t/s agg / 120.9 t/s decode (N=1), 459.3 t/s agg (N=4), tool calls 5/5.
3. `20260505T172451Z` (graphs): 56.4 t/s agg / 197.2 t/s decode (N=1), 477.3 t/s agg (N=4), tool calls 3/5.
4. `20260505T180500Z` (graphs): 55.3 t/s agg / 117.9 t/s decode (N=1), 458.1 t/s agg (N=4), tool calls 4/5.

**Conclusion:** throughput is good in graph mode, but tool-call reliability is not stable across reruns. Keep AWQ TP=2 production.

**Critical constraint — NO MTP:** `--speculative-config` must NOT be added to the coder (BENCH_14: 0/3 tool-call failure).

**Deploy command (Ph.1 rerun):**
```bash
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./infra/scripts/deploy.sh vllm gpu0 rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm \
  --trust-remote-code --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 --max-num-seqs 16 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
```

**Pass (closure criteria):**
1. Three consecutive runs with tool-call pass rate 5/5.
2. th02 quality PASS in each run.
3. N=1 agg TPS ≥ 50 and N=4 agg TPS ≥ 430 in graph mode.

**Deps:** T_MTP2 ✓ CLOSED FAIL (MTP excluded).

---

### T_CV6 — convergence_extended_architecture_ngl999 — OPEN

**Question:** Can we free one Arclight GPU path and run Convergence at/near `-ngl 999` while still keeping thinker availability?

**Scope:**
1. Test topology variants: thinker + Convergence (`-ngl 999`), thinker + Convergence (`-ngl` sweep), and thinker swap to Arclight Full when needed.
2. Keep `-np 1` for Convergence (architectural limit).
3. Record Convergence TPS + VRAM + impact on thinker latency.

**Pass:** find a topology with Convergence TPS ≥ 10 t/s and no thinker correctness regression.

---

### T_CV7 — convergence_speculative_expert_offload_engine_eval — OPEN

**Question:** For speculative expert offload, should we use upstream `llama.cpp` or `ik_llama.cpp`?

**Scope:**
1. Feature audit (flags + architecture support) on both engines.
2. Run identical Convergence/Singularity prompts on both.
3. Compare correctness, TPS, and operational stability.

**Pass:** one engine selected with explicit rationale and benchmark deltas.

---

### T_CV8 — convergence_speculative_decoding_engine_eval — OPEN

**Question:** For speculative decoding (MTP / DFlash), should we use `llama.cpp` or `ik_llama.cpp`?

**Scope:** capability check, then A/B benchmark on same model + prompts.

**Pass:** one engine selected with evidence for both throughput and output correctness.

---

### T_KV4 — arclight_kv_batch_utilization_matrix — OPEN

**Question:** What are max KV cache size and max batch limits for Arclight coder/thinker versus GPU utilization, including actual MiB usage (max/avg/steady)?

**Scope:** sweep `gpu-mem-util` and batch/seq settings on both roles; record KV pool and real process memory.

**Pass:** publish operating envelopes with safe default + aggressive profile.

---

### T_KV5 — convergence_kv_fp8_bf16_256k — OPEN

**Question:** What is the practical Convergence KV ceiling for fp8 vs bf16 up to 256K context?

**Scope:** context ramp with fixed prompt and deterministic sampling; track OOM point, TPS decay, and memory growth.

**Pass:** choose default KV precision for long-context Convergence.

---

### T_ARCH3 — arclight_full_vs_extended_gdn_rationale — OPEN

**Question:** Is Extended Arclight still rational for GDN thinker workloads, or should Arclight Full with fp16/bf16 KV be the primary long-context strategy?

**Scope:** compare quality/latency/VRAM for Extended vs Full under the same long-context task set.

**Pass:** document a settled architectural recommendation and demote the losing mode.

---

### T2.6 — behemoth_archetype_scouting — DESIGN ITEM (no benchmark)

**What it produces:**
1. Candidate archetype taxonomy (2–3 archetypes, short rationale).
2. Per-archetype 1–2 model shortlist with VRAM, context, tool-parser status.
3. Concrete sub-items (T2.6.1, T2.6.2, …) queued after triage.

**Out of scope:** running any benchmark. This is a research-mode item.

**Deps:** T1.3 ✓, T2.4 ✓.

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
