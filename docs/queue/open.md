# Open Queue

Only OPEN and BLOCKED items. Full status table: docs/queue/status.md. Full procedures for DONE items: docs/history/done-items.md.

Legend: **OPEN** = ready to run (deps met). **BLOCKED** = deps or research needed. **DEFERRED** = not now.

---

## HIGH priority

### T_PQ3 — gptq_int4_coder_viability — OPEN

**Question:** Does GPTQ-Int4 on host-native vLLM 0.20.0 (cu130, Marlin kernels) give 185+ t/s N=1 AND pass 5/5 tool calls at TP=1? If yes, it replaces APEX ik_llama.cpp with a single engine that wins at both N=1 latency and N=4 throughput.

**Background:**
APEX ik_llama.cpp (current production coder, BENCH_24): 185 t/s N=1, 217 t/s N=4 aggregate (1.17× scaling — poor batching). vLLM PQ (previous coder, BENCH_23): 56.5 t/s N=1, 459 t/s N=4 (8.1× scaling). The N=4 gap exists because ik_llama.cpp doesn't do continuous batching across sequences; vLLM does.

GPTQ-Int4 on a single RTX 5090 (TP=1): community-measured 194–197 t/s N=1 (source: HuggingFace `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` discussion). INT4 ≈ 17.5 GB weights (fits GPU0 alongside thinker). Uses Marlin kernels — same path that avoids the SM120 FlashInfer CUTLASS grouped GEMM bottleneck. With vLLM continuous batching, N=4 aggregate should scale toward 400–500 t/s.

**Critical risk:** AWQ (also INT4 Marlin) failed 2/5 tool calls at TP=1 V1 (BENCH_23a). GPTQ uses different weight packing. Whether the failure was precision-related (INT4 affecting tool-call structure generation) or AWQ-format-specific is unknown. This is the primary gate.

**vLLM + CUDA requirement:** vLLM 0.20.0 with cu130 (CUDA 13.0 default). Host-native pyenv install — no container. NVIDIA driver ≥ 575.xx required for CUDA 13.0 runtime.

**Model priority:**
1. `Qwen/Qwen3.6-35B-A3B-GPTQ-Int4` — try first (architecture parity with current coder)
2. `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` — fallback (confirmed to exist, benchmarked on RTX 5090)

**Pass:** TPS N=1 ≥ 150 t/s AND tool-call rate ≥ 4/5.
**Fail — tool calls < 3/5:** CLOSE (INT4 precision insufficient for tool-call generation, same as AWQ). Pivot to T_PQ4 (FP8 TP=2, Extended Arclight only).

**Deps:** Host-native vLLM 0.20.0 + CUDA 13.0 environment. No other deps.

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


---

### T_CV6 — convergence_extended_architecture_ngl999 — OPEN

**Question:** Can we remove the coder from GPU0 and run Convergence at full `-ngl 94` (all attention layers) on GPU0 alone while keeping thinker on GPU1?

**Pre-bench research (BENCH_21 data):**
Convergence VRAM model: fixed overhead ≈ 6,247 MiB + per-layer ≈ 113 MiB. At ngl=94 (all 94 transformer layers): **6,247 + 94×113 = ~16,869 MiB ≈ 16.9 GB**. GPU0 has 32 GB GDDR7 → this fits comfortably with 15 GB headroom. Expected TPS ≈ **13.99 t/s** (same as isolated `-ngl 999` measured in T_CV3, no GPU contention with thinker).

**Proposed production topology:**
```
GPU0: ik_llama.cpp Convergence  -ngl 94 --cpu-moe  (~17 GB VRAM)
GPU1: vLLM Thinker TP=1                             (~22-29 GB VRAM)
(Coder removed from hot-pair; available via CRIU swap from Extended Arclight)
```

**Scope:**
1. Launch Convergence on GPU0 only at ngl=94 with thinker on GPU1. Measure Convergence TPS + VRAM.
2. Confirm thinker correctness is unaffected (no PCIe contention impact on GDN recurrent state).
3. Measure thinker VRAM headroom: at util=0.95, ~2.8 GB KV free. With coder gone, can thinker run at lower util and get fp16 KV with equivalent or better quality?
4. Measure coder boot time from CRIU checkpoint when needed (Extended Arclight mode).

**Coder/thinker per-request TPS context:**
- N=1: thinker 91.9 t/s vs coder 56.5 t/s (thinker 63% faster single-request)
- N=4: thinker 78.7 t/s/req vs coder 115.4 t/s/req (coder 47% faster in batch)
- Removing coder from hot-pair degrades agentic batch workloads but coder can be restored via CRIU.

**Pass:** Convergence TPS ≥ 12 t/s at ngl=94 with thinker unaffected; confirm coder CRIU restore time < 15s.

---

### T_CV7 — convergence_speculative_expert_offload_engine_eval — OPEN

**Question:** For speculative expert offload on Convergence, should we use upstream `llama.cpp` or `ik_llama.cpp`? Does the APEX GGUF format change the answer?

**Background on speculative expert offloading:**
Standard --cpu-moe keeps ALL expert weights in CPU RAM; expert computation happens on CPU. Speculative expert offloading (SEO) predicts which experts will be activated on the NEXT token and preloads them to VRAM during the current token's computation, overlapping PCIe transfer with GPU compute. This is different from MTP (speculative decoding). ik_llama.cpp has a tuned offload batch threshold: `32 × (total_experts / active_experts)` — for Convergence (512 experts, 10 active) this is 32 × 51.2 ≈ 1638 tokens before switching to GPU offload path. Standard llama.cpp uses a fixed 32-token threshold.

**APEX + SEO interaction:** APEX's per-expert mixed precision (Q6_K edge, Q4-Q6 middle routed) means the preloaded expert weights are smaller on average vs standard K-quants → faster PCIe transfer during preload → potentially higher SEO throughput gains. This interaction is untested.

**Scope:**
1. Feature audit: what SEO flags exist in ik_llama.cpp vs upstream llama.cpp (`--expert-offload`, `--lookup-cache-static`, etc.)?
2. Baseline Convergence TPS on both engines at current config (UD-IQ2_M, ngl=15, thinker co-load).
3. Enable SEO on each engine. Measure TPS delta and output correctness (same prompt set).
4. If T_APEX4 has APEX Convergence available: repeat with APEX variant.
5. Flag: do NOT use `-rtr` (row-tile-repack) flag with MoE experts on CPU — forces all mul_mat to CPU path (ik_llama.cpp bug).

**Pass:** one engine selected with explicit TPS delta from SEO and verified output correctness.

---

### T_CV8 — convergence_speculative_decoding_engine_eval — OPEN

**Question:** For speculative decoding (MTP/DFlash) on Convergence/Singularity, which engine? Does Convergence GGUF (UD-IQ2_M or APEX) include MTP draft heads?

**Background:**
MTP in llama.cpp/ik_llama.cpp requires trained MTP heads in the GGUF checkpoint. Qwen3.5-397B-A17B was trained with MTP heads. Whether UD-IQ2_M (unsloth) or APEX (mudler, if available) exports these heads is unknown — must check the GGUF file metadata.
- ik_llama.cpp MTP flag: `--draft-max N` or `--speculative-max-tokens N`
- llama.cpp MTP flag: speculative via separate draft model or `--draft-model` if heads are embedded

**Scope:**
1. Check UD-IQ2_M GGUF metadata for MTP head tensors (`gguf-dump` or Python `gguf` library).
2. If heads present: enable MTP on ik_llama.cpp with N=1, N=3. Measure TPS delta on Convergence prompts.
3. If not present: document as SKIPPED (no heads in this quant). Flag for re-test if APEX Convergence ships with heads.
4. DFlash (speculative draft model): research whether a small Qwen3.5-MoE draft model is available or practical.

**Pass:** one path selected (MTP if heads found, DFlash if viable small draft exists) with measured TPS delta ≥ +20% vs baseline.

---

### T_APEX3 — apex_coder_mtp_ik_llama — OPEN (depends on T_APEX1)

**Question:** Does the APEX GGUF export of Qwen3.6-35B-A3B include MTP draft heads? If so, what TPS gain does ik_llama.cpp MTP deliver — and does it exceed the vLLM MTP penalty (-38.6% at N=1)?

**Background:**
Qwen3.6-35B-A3B was trained with MTP heads (confirmed by vLLM MTP working at TP=1 in BENCH_23b). Whether mudler's APEX GGUF export includes these heads is unknown. If present, ik_llama.cpp MTP would differ from vLLM in two ways:
1. No FlashInfer SM120 grouped GEMM bottleneck → expert verification overhead is lower
2. GGUF MTP implementation may have different acceptance rate characteristics than vLLM's

In vLLM MTP at TP=1: -38.6% N=1, -50.6% N=4. The overhead was expert-routing verification (kernel-bound). On ik_llama.cpp with native CUDA kernels, verification overhead should be lower → MTP may be net-positive.

**Scope:**
1. `gguf-dump` or Python `gguf` library: check for tensor names matching `model.layers.*.mlp.draft*` or similar MTP head naming.
2. If heads present: `ik_llama.cpp` with `--draft-max 1` and `--draft-max 3`. Measure N=1 and N=4 TPS vs no-MTP baseline from T_APEX1.
3. Tool-call probe (5× probes) with MTP enabled.
4. If MTP is net-positive on ik_llama.cpp (unlike vLLM), document as preferred config.

**Pass:** MTP TPS ≥ no-MTP TPS from T_APEX1 AND tool calls ≥ 4/5. If heads absent: close as SKIPPED.

**Deps:** T_APEX1 PASS. Same binary/config — no additional download needed.

---

### T_APEX4 — convergence_apex_gguf_viability — DEFERRED (quality evaluation, post-T_APEX1)

**Question:** Does APEX Convergence deliver meaningfully better output quality than UD-IQ2_M, and is the TPS trade-off acceptable?

**Context:** APEX 397B variants are larger than UD-IQ2_M and will be slower due to DDR5 bandwidth. This is a quality trade-off evaluation, not a TPS optimisation. Only worth pursuing once Convergence quality becomes the binding constraint.

**Actual file sizes** (`mudler/Qwen3.5-397B-A17B-APEX-GGUF`, confirmed 2026-05-05):
- APEX-Compact: **187 GB** (+52% vs UD-IQ2_M 123 GB)
- APEX-Quality: **243 GB** (+98%)
- APEX-Balanced: **289 GB** (+135%)

**TPS cost** (DDR5 ceiling ~83 GB/s, with --cpu-moe):
- APEX Compact: 13.99 × (123/187) ≈ **9.2 t/s** (−34%)
- APEX Quality: 13.99 × (123/243) ≈ **7.1 t/s** (−49%)
- APEX Balanced: 13.99 × (123/289) ≈ **6.0 t/s** (−57%)

**Scope (when activated):**
1. Start with Compact (smallest download, best TPS of the three). Run standard Convergence task set vs UD-IQ2_M baseline.
2. Score quality on reasoning depth, factual accuracy, instruction following. If quality gap is clear and meaningful, accept the TPS cost.
3. If Compact quality is not better enough: skip Quality and Balanced (larger, slower, marginal gain).

**Activation criterion:** Convergence output quality identified as a blocker in production use. Not before.

**Deps:** T_APEX1 settled (engine path confirmed). Storage: 187 GB free on NVMe (3,000 TBW budget: 187 GB once = negligible).

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

**Question:** Is Extended Arclight (coder TP=2, 65K ctx) still rational? And is "Arclight Full" (thinker with fp16 KV) even meaningful given GDN architecture?

**Pre-research analysis:**
- **Extended Arclight** frees GPU1 (thinker sleeping) and gives coder TP=2 at 65K context. This makes sense for deep single-context **coding** tasks that exceed 32K tokens.
- **Arclight Full** concept = run thinker with fp16 KV instead of fp8, using the freed GPU0 space (if coder removed). But **GDN (Gated DeltaNet) barely uses KV cache**: O(d) recurrent state per layer, not O(n) KV tokens. Measured in T3.1: 0 MiB KV delta at 50K context. So fp16 KV gives **negligible quality benefit** for the thinker — the precision change affects only the few traditional attention heads in the hybrid, not the GDN layers.
- **Hypothesis:** Arclight Full is architecturally questionable for the thinker. It should be documented as SETTLED ILLOGICAL for GDN models.

**Scope:**
1. Confirm: at thinker util=0.73 (model floor, 23,218 MiB), how much KV pool remains? (~585 MiB per BENCH_21 projections.) How many tokens does 585 MiB fp16 vs fp8 KV support?
2. Run th02 + 1 hard thinker task at fp16 KV with small pool vs fp8 KV at standard pool. Measure quality delta.
3. For Extended Arclight: run coder at TP=2 with a 64K context task. Confirm 238 t/s decode and correctness.
4. Document final recommendation: Extended Arclight = LOGICAL for long-context coding. Arclight Full (thinker fp16 KV) = SETTLED ILLOGICAL for GDN (KV precision immaterial; GDN recurrent state is fp32 by default regardless).

**Pass:** explicit SETTLED recommendation for both modes with evidence from the GDN KV measurement.

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
