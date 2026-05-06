# Research Cycle Log

Full log of research ↔ testing cycles, newest first. This is a grep/ripgrep target — decisions and findings are recorded verbatim. For current state, see `RESEARCH_STATE.md` and `docs/decisions/`.

---

## R36 — May 6 2026 — GPTQ-Int4 + Marlin Coder (BENCH_33, T_PQ3)

**Triggered by:** Need to resolve the N=4 aggregate throughput bottleneck in the APEX GGUF stack (217 t/s) while maintaining single-GPU TP=1 footprint.

- **BENCH_33 (GPTQ-Int4 Viability):** ✅ SUCCESS. `groxaxo/Qwen3.6-35B-A3B-GPTQ-Pro-FOEM-4bit-g128` (Patched) achieved **103.2 t/s** (N=1) and **502.9 t/s** (N=4).
- **Aggregate Performance:** **2.3× speedup** over APEX GGUF aggregate throughput.
- **Reliability:** **5/5 tool-calling probes PASS**. th02 EDF quality probe PASS.
- **Marlin Kernels:** Confirmed engagement of Blackwell-optimized Marlin kernels (`gptq_marlin`) via host-native vLLM 0.20.1.

**Decisions:**
- **GPTQ-Int4 + vLLM (Marlin) PROMOTED to Production Coder.**
- **APEX GGUF retired** to cold-storage fallback.
- **Metadata Patching:** Manual injection of `quantization_config` into `config.json` is a prerequisite for community-sourced GPTQ models on vLLM.

---

## R35 — May 6 2026 — APEX GGUF Coder & Golden Topology (BENCH_24/25, T_APEX1/2)

**Triggered by:** Need to resolve the SM120 FlashInfer bottleneck (57 t/s) and reclaim VRAM for Convergence co-load.

- **BENCH_24 (APEX GGUF Viability):** ✅ SUCCESS. APEX I-Compact on `ik_llama.cpp` (Host) achieved **185.0 t/s** (N=1) and **217.1 t/s** (N=4). Bypasses the vLLM software bottleneck.
- **Tool-Calling:** Resolved via grammar-based Jinja templates. **5/5 PASS**.
- **BENCH_25 (Golden Co-load):** ✅ SUCCESS. Reducing Coder footprint to 18.5GB enabled Convergence (397B) to reach **13.8 t/s** in co-load mode (98% of isolated performance).
- **VRAM Efficiency:** Reclaimed ~9.4GB on GPU0 compared to PrismaQuant baseline.

**Decisions:**
- **APEX GGUF + ik_llama.cpp PROMOTED to Production Coder.**
- **Golden Topology settled:** Coder (GPU0) + Thinker (GPU1) + Convergence (Shared) allows all three models to run at full performance simultaneously.
- **vLLM retired for the Coder service.**

---

## R34 — May 5 2026 — AWQ Shootout + MTP Audit + PQ Production Confirmation (BENCH_23a/b/c)

**Triggered by:** Gemini BENCH_23 session showed tool-call variability (3-5/5 across runs); needed a user-run stability shootout to settle the production coder config definitively.

- **BENCH_23a (AWQ TP=1 V1 engine):** ✅ TPS — ✗ Tool calls. AWQ (INT4 Marlin) at TP=1 with V1 engine = **2/5 tool calls FAIL**. TPS: 59.5 t/s N=1 / 496.1 t/s N=4 (faster than PQ at N=4 due to Marlin kernel). th02 truncated (finish_reason=length at 32K). **Key finding:** V1 engine alone is not sufficient for TP=1 reliability — the fix is PrismaQuant+V1 specific. AWQ is not viable at TP=1.
- **BENCH_23b (PQ TP=1 V1 + MTP n=1):** ✅ PASS. Tool calls 5/5. TPS: 34.7 t/s N=1 (-38.6% vs no-MTP), 192.0 t/s N=4. Confirms Gemini variability was a config artifact (--enforce-eager remnant), not a model instability. MTP is logically stable at TP=1.
- **BENCH_23c (PQ MTP Tuned, max-num-seqs=64):** N=4: 227.2 t/s (+18% over basic MTP), N=1: 35.2 t/s (unchanged). Bottleneck is kernel-bound (expert routing overhead), not scheduler-bound. MTP overhead: -38.6% N=1, -50.6% N=4.

**Per-request TPS crossover confirmed:**
- N=1: thinker 91.9 t/s vs coder 56.5 t/s (thinker 63% faster)
- N=4: coder 115.4 t/s/req vs thinker 78.7 t/s/req (coder 47% faster in batch)

**Decisions:**
- Production coder: PrismaQuant+V1, No-MTP (56.5 t/s agg N=1, 459 t/s N=4). T_PQ2 DONE ✓.
- AWQ TP=1 ruled out permanently (tool-call FAIL even with V1 engine).
- MTP for coder: logically stable but uneconomical. Revisit post-CUDA 13.0 SM120 maturation.

---

## R33 — May 5 2026 — TP=1 Stability Discovery + PrismaQuant Coder Promotion (BENCH_23/23a)

**Triggered by:** T_PQ2 Phase 1 (PrismaQuant Coder Audit); ongoing OOM and reliability issues with AWQ Coder at TP=2 during tri-model co-loads.

- **BENCH_23 (PrismaQuant Coder Audit):** ✅ PASS (Stability). `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` successfully loaded at **TP=1** on a single GPU using the **vLLM V1 engine** and **CUDA graph capture**. 
- **Reasoning Stability Discovery:** Both BENCH_23 (PQ) and BENCH_23a (AWQ) confirmed that the "Reasoning Collapse" (loops/hallucinations) previously seen at TP=1 was an artifact of the legacy V0/Eager-mode kernel path (vLLM 0.19). The V1 engine with CUDA graphs provides a stable execution path for single-shard MoE models on Blackwell (sm_120).
- **Performance (TPS):** Both models hit a **~60 t/s (N=1)** cap. This was confirmed to be a **Software Bottleneck** (SM120 grouped GEMM immaturity in FlashInfer/CUTLASS) rather than a quantization issue.
- **VRAM:** TP=1 deployment uses ~22-24 GiB VRAM, drastically improving the co-load headroom compared to the previous TP=2 requirement.

**Decisions:**
- **Arclight Coder SETTLED on TP=1.** The AWQ TP=2 config is superseded to reclaim VRAM.
- **Production choice:** `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm`. Precision (FP4) prioritized over raw AWQ speed, as both are currently capped at 60 t/s on the stable V1 path.
- **Engine:** `VLLM_USE_V1=1` and `enforce_eager=False` are mandatory for TP=1 stability on this architecture.


## R32 — May 3–4 2026 — Hard Suite Quality Eval + Context/KV Research (BENCH_20, T_HARD1)

**Triggered by:** T_HARD1 ready; BENCH_19 MTP results confirmed PrismaQuant production; hard task suite calibrated to find PQ vs AWQ quality gap if one exists.

- **BENCH_20 (T_HARD1 — hard systems engineering suite):** ✅ TIED. PQ 41/50, AWQ 42/50 on 10 hard tasks (Linux kernel, networking, Raft, Proxmox, OpenStack, Ansible, K8s). No quality gap found at maximum task difficulty. PQ task 03 (Raft asymmetric partition) truncated mid-reasoning at 28K tokens (finish_reason=length, empty response) — prior run (095341Z) has a complete response scoring 5/5; substituting it gives PQ 43 vs AWQ 42. Both automated scoring passes contained hallucinated evidence (XSNP_HITM for PQ task 04, --disable-eviction for AWQ task 10) — corrected manually against raw response files.
- **Context/KV research:** `--max-model-len 131072` costs ~0 extra VRAM on Qwen3.6-27B hybrid (DeltaNet recurrent state is fixed-size, not stored in the KV pool — confirmed T3.1 Phase 1). fp8 KV only quantizes transformer layers; DeltaNet layers are unaffected. At 32K–64K context, fp8 vs bf16 quality delta negligible. bf16 halves the token pool (~165K vs ~330K combined capacity for fp8) — not justified at current workloads.

**Decisions:**
- T_HARD1 CLOSED. PQ and AWQ quality-equivalent across standard (BENCH_12) and hard (BENCH_20) suites. Production stays PQ+MTP n=3 for TPS advantage (92 vs 77 t/s).
- Hard reasoning re-runs require `--max-model-len 131072` + `max_tokens≥32000`. Documented in settled.md.
- fp8 KV is correct for production. bf16 KV not justified at current context lengths.

---

## R31 — May 2 2026 — 128K Context + GLM Engine Eval (BENCH_15, BENCH_16)

**Triggered by:** T_KV3 unblocked after ik_llama.cpp main confirmed working; BENCH_16 GLM engine eval queued.

- **BENCH_15 (T_KV3 Path B):** ✅ PASS. Qwen3.6-27B (dense) at 128K context via ik_llama.cpp main. 1,892.9 t/s prefill, 49.4 t/s decode. Total VRAM ~28,268 MiB. th02 tool-call quality: PASS. Confirms 128K context viable for thinker-class workloads. Incidental: Convergence was already running at port 8002 during BENCH_15 — both dense Qwen3.6-27B and MoE 397B architectures confirmed simultaneously on ik_llama.cpp main.
- **BENCH_16 (T_ENGINE_EVAL_GLM):** ✅ PASS. GLM-4.7-Flash (30B-A3B MLA) on ik_llama.cpp main. 176.4 t/s. 5/5 tool-calling pass. Resolves prior Triton/Blackwell MLA instability. Model is a viable coder alternative (T2.2).

**Decisions:**
- T_KV3 SETTLED. 128K context confirmed for Qwen3.6-27B on ik_llama.cpp Path B.
- GLM-4.7-Flash confirmed viable on ik_llama.cpp main. Added to coder candidate list.

---

## R30 — May 1 2026 — PrismaQuant Thinker + MTP Results (BENCH_12, BENCH_13, BENCH_14)

**Triggered by:** PrismaQuant 5.5bit ready for thinker evaluation; MTP speculative decoding unblocked after #40756 non-applicability confirmed.

- **BENCH_12 (PrismaQuant thinker):** ✅ PROMOTED. `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm` quality parity confirmed (7/8 tasks; th08 truncated in both). th02 EDF algorithm: PrismaQuant correct. TPS: 51.3 t/s seq=1 / 198.9 t/s seq=4 (26–33% regression vs AWQ 76.9/269.4 t/s). Quality rationale accepted over TPS. Production thinker updated from AWQ.
- **BENCH_13 (MTP on AWQ thinker):** ⚠️ STALE. MTP n=1: +31.8% at N=1 (76.8 → 101.2 t/s), +51% at N=4. Result valid but model superseded before quality review completed (th02 never scored). T_MTP1 must re-run on PrismaQuant. Token-counting caveat: T_PAR1 script undercounts ~2× with MTP; use T_MTP1 script (usage.completion_tokens).
- **BENCH_14 (MTP on A3B coder):** ❌ FAIL. MTP n=1 breaks tool-call generation: 0/3 probes produced tool_calls. T_MTP2 CLOSED FAIL.

**Decisions:**
- PrismaQuant 5.5bit is production thinker. AWQ superseded 2026-05-01.
- T_MTP2 CLOSED FAIL — MTP + structured output broken on A3B MoE coder in vLLM 0.19.0.
- T_MTP1 re-queued for PrismaQuant thinker (n=1,2,3 sweep; n=3 optimal per rdtand author).

---

## R29 — April 30 2026 — CRIU TP=2 Post-Mortem + Research Sprint (BENCH_10)

**Triggered by:** T_CRIU3 Phase 2 attempt on coder TP=2; research sprint on PrismaQuant/NVFP4/MTP.

- **BENCH_10 (T_CRIU3 Phase 2):** ❌ FAIL. CRIU dump/restore succeeds (29s dump, 67 GB; 24–26s restore). KV cache physically preserved post-restore (prefix hit ratio identical). But first inference fails: `TimeoutError: RPC call to execute_model timed out`. Root cause: SHM broadcast IPC (`ShmRingBuffer`) broken after CRIU for multi-process TP=2. Workers resume inside `poller.poll()` on dead `inproc://` ZMQ sockets. Patches (CriuSafePoller, SpinCondition→sched_yield) necessary but not sufficient. Also: 26s restore is only ~4× vs cold start. Permanently blocked for TP=2.
- **SM120 NVFP4 MoE analysis:** Desktop Blackwell SM120 cannot run NVFP4 MoE grouped GEMM efficiently. CUTLASS needs compute_120f (CUDA 13.0); FlashInfer auto-detects compute_120a → slow fallback. NVFP4 FlashInfer-CUTLASS = 39 t/s vs Marlin (AWQ) = 46–49 t/s. Dense NVFP4 unaffected. PrismaQuant coder deferred; PrismaQuant thinker (dense) feasible after CUDA 13.0.
- **MTP speculative decoding research:** vLLM 0.19.0 `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` measured at −21.6% TPOT ≡ +27.5% faster decode. vLLM #40756 does NOT apply to our config. T_MTP1/T_MTP2 unblocked.
- **PrismaQuant model research:** `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm` selected as thinker candidate (349 NVFP4 + 35 MXFP8 + 112 BF16; DeltaNet handled; ~19 GB disk / ~22–24 GB runtime; MTP n=3 optimal per author).

**Decisions:**
- CRIU TP=2 KV-preservation: permanently blocked on vLLM 0.19.0/Blackwell. Prefix cache (T3.4, 13.9×) is the correct post-swap KV mechanism.
- PrismaQuant thinker 27B dense: SM120 safe, proceed to BENCH_12.
- PrismaQuant coder 35B A3B MoE: deferred until CUDA 13.0 container rebuild.

---

## R28 — April 28 2026 — Parallelism Rerun + CRIU Testing (BENCH_01–03, T_CRIU2, T_CRIU3 Phase 1, T3.4)

**Triggered by:** T_PAR1 data confirmed fabricated (R27 finding); CRIU Convergence and thinker tests queued.

- **BENCH_01–03 (T_PAR1 rerun):** ✅ REAL DATA. Coder TP=2: N=1 240.9 t/s → N=8 1,204.9 t/s, still scaling at N=8 (knee not found). Thinker max-num-seqs=1: 76.9 t/s aggregate at any N (queues). Thinker max-num-seqs=4: 269.4 t/s at N=4 (3.5×), plateau at N=8. Convergence: N≥2 crashes unchanged (pr-1288 GGML_ASSERT). All previously cited T_PAR1 numbers (1,196 t/s at N=8, 698 t/s at N=4, thinker OOM at N>1) were fabricated by Gemini Flash — replaced by these measurements.
- **T_CRIU2 (Convergence CRIU):** ❌ Not viable without QX_PRELOAD. `--no-mmap`: SYSTEM_OOM (CRIU dump of 135 GB anon-RAM needs ~351 GB VMS, physically impossible). mmap path: RESTORE_OK (8.7 GB checkpoint, 7.3s restore) but first-inference TTFT 100.56s (NVMe page-fault warmup) — worse than 83s cold start. QX_PRELOAD elevated to HIGH priority: pre-warming 123 GB into page cache before restore projects 14s restore-to-interactive (6× improvement).
- **T_CRIU3 Phase 1 (thinker host-native):** ✅ PASS. Restore time 0.43s (target <1s). Checkpoint 501 MB. Post-restore TTFT parity (0.73s vs 0.72s). Fast-swap for TP=1 confirmed viable. Unblocks Sequential TP=2 orchestration.
- **T3.4 (prefix cache):** ⚠️ PARTIAL. Prefix cache works (cold 2,410 ms → warm 173 ms, 13.9× speedup). Post-wake FAILED: `POST /wake_up` HTTP 500 `'list' object has no attribute 'zero_'` in `v1/engine/core_client.py`. vLLM bug on Qwen3.6-35B-A3B + `--enforce-eager`. Wake path needs investigation.

**Decisions:**
- T_PAR1 fabricated numbers permanently retired. Real data: coder scales to N=8+; thinker needs max-num-seqs=4.
- Thinker max-num-seqs upgraded 1→4. The seqs=1 constraint was empirically unnecessary.
- QX_PRELOAD required before CRIU on Convergence has net benefit over cold start.
- T3.4 prefix cache result trusted (pre-wake portion); wake bug needs separate investigation.

---

## R27 — April 26 2026 — Arclight Context & Parallelism Sweep (T_KV1, T_PAR1)

**Triggered by:** Completion of the Arclight architecture (CRIU settled). Need to establish performance ceilings for context length and concurrent throughput.

- **T_KV1 (Coder Context)**: ✅ PASS. Qwen3.6-35B-A3B (AWQ) at TP=2 handled 65,536 token context. Performance regression minimal (238 t/s vs 251 t/s at TP=1/8K). Bench.log confirmed. 131K failed — vLLM 0.19 rejected `--swap-space` flag. Swap-based context expansion blocked.
- **T_PAR1 (Coder Parallelism)**: ❌ DATA MISSING — NOT MEASURED. Coder/Thinker endpoints were not running during the T_PAR1 run. `metrics.json` shows `coder_detail: null, thinker_detail: null`. Raw dir has no coder or thinker sweep files. Numbers previously recorded (1,196 t/s at N=8, 698 t/s at N=4, thinker OOM at N>1) were **fabricated by Gemini Flash**. T_PAR1 is REOPENED for Coder and Thinker.
- **T_PAR1 (Convergence)**: ✅ REAL DATA. Convergence crashes at N≥2 (Server disconnected / All connection attempts failed). N=1 baseline ~3.1 t/s. Conflicts with T_CV4 (15.6 t/s at C=4) — see Open from Research for explanation.

**Decisions from real data:**
- 65K context confirmed for Coder Extended mode.
- Convergence concurrent requests: pr-1288 crashes at N≥2. Production must use N=1 for client concurrency until upstream fix.
- Coder/Thinker max-num-seqs: UNKNOWN. T_PAR1 rerun required.

---

## R26 — April 26 2026 — Coder Context Ceiling (T_KV1) PASS

**Triggered by:** Measurement of the context ceiling for the Arclight Coder at TP=2.

- **Status**: ✅ PASS.
- **Max Usable Context**: 65,536 tokens (GPU-only).
- **Performance**: TTFT 3.0s (~21K tokens/sec). Decode 238.2 t/s (minimal 2.6% regression).
- **Defect**: `--swap-space 32` flag unrecognized in vLLM 0.19.0 (R-V0 engine path), blocking the 131K test.
- **Decision**: 65K context is sufficient for "Extended Arclight" mode. Update ARCHITECTURE.md to reflect 65K as the standard ceiling.

---

## R25 — April 26 2026 — Thinker TP=2 Fix (T2.4h) RE-CONFIRMED TP=2 STABILITY

**Triggered by:** Need to resolve GDN state-splitting errors at TP=2 via `--enforce-eager`.

- **Finding**: `--enforce-eager` successfully resolved the semantic "garble" at TP=2, making the Arclight Thinker logically correct on two GPUs.
- **Performance**: ~16.5 t/s (TP=2). Matches the known 10x performance collapse seen in R24/T6.1 when bypassing CUDA graphs on the V1 engine.
- **Decision**: The "Eager fix" provides correctness for research/debug but is too slow for production. TP=2 thinker remains theoretically viable but economically blocked by engine overhead.

---

## R24 — April 26 2026 — Coder Reliability Audit (T6.1) RE-CONFIRMED TP=1 FLOOR

**Triggered by:** Attempt to run Arclight Coder at TP=1 using the stable engine path.

- **Finding**: Re-verified: vLLM 0.19.0 V1 engine overhead at TP=1 forces a collapse to ~20 t/s in eager mode.
- **Workaround**: `gpu-mem-util 0.98` + `--enforce-eager` allows TP=1 to load, but performance regression is 10x (23.6 t/s).
- **Quality**: TP=1 suffered from Reasoning Collapse (hallucination loops) likely due to Triton/FLA kernel shape mismatches in eager mode. TP=2 (20.25 t/s) achieved 100% PASS rate.
- **Decision**: TP=2 remains the only viable deployment for the 35B-A3B Coder on vLLM 0.19.0.

---

## R23 — April 26 2026 — Convergence Parallel Scaling (T_CV4) SUCCESS

**Triggered by:** Investigation into MoE expert-loading overhead during batching.

- **Aggregate TPS**: 15.59 tokens/sec at concurrency=4.
- **Scaling**: 1.12x scaling (vs 13.9 t/s single).
- **Insight**: llama-server in the pr-1288 build efficiently amortizes the DDR5 expert-fetch cost across the batch. The MoE architecture is multi-user viable on CPU.

**NOTE (added in R27/doc-tidy):** T_CV4 measured sequential pipelining into server's `-np 4` internal slots (client requests sequential). T_PAR1 showed truly concurrent client requests crash pr-1288 at N≥2 (GGML_ASSERT). The 15.6 t/s result reflects pipelining throughput, NOT true parallelism. Production `-np 4` is valid for throughput; true concurrent capacity is N=1.

---

## R22 — April 26 2026 — Convergence Hybrid Optimization (T_CV3) SUCCESS

**Triggered by:** Need to recover 13.5 t/s baseline for the 397B model using partial GPU offload.

- **Baseline Replicated**: Achieved 13.99 t/s using `-ngl 999 --cpu-moe`.
- **Finding**: Offloading attention layers to GPU (sm_120) provides a 3.75x speedup over pure CPU.
- **Tooling**: `llama-bench` discarded for MoE hybrid mode; `llama-server` is the production engine.

---

## R21 — April 26 2026 — Convergence Startup & Context Ceiling (T_CV1)

**Triggered by:** Need to establish operational parameters for the 397B Singularity-tier model running in CPU-only mode.

- **Startup**: Median cold start 83s. Warm start (page cache) 88s. Conclusion: Model initialization is CPU-bound, not disk-bound.
- **Performance**: Baseline generation at ngl=0 (CPU-only) is 3.7 t/s.
- **Context Ceiling**: Verified functional up to 128k context at 3.6 t/s.
- **Result**: `results/T_CV1_convergence_startup_timing_20260426T101623Z/`.

**Decisions:**
- **Always-Resident**: Convergence will be maintained as an always-resident service in system RAM. The 80s+ load time is too high for transparent demand-routing.
- **Context Limit**: 128k is the official context ceiling for Convergence. Queries exceeding this require Singularity escalation (GPU offload).

---

## R20 — April 26 2026 — Extended Arclight Hot Restart (T_KV2)

**Triggered by:** The need to reduce the 100s+ cold start penalty for Arclight TP=2 (Extended mode) to sub-10s.

- **Status**: ✅ SUCCESS. Achieved 0.28s Hot Restart time (down from 100.2s cold start) for Qwen3.6-35B-A3B (TP=2). 358x speedup.
- **Blocker Resolved**: The `Unknown shit 600 (anon_inode:[io_uring])` CRIU error was neutralized by stripping `uvloop` from the vLLM entrypoints and background workers.
- **Technical Path**:
    - Patched `vllm/entrypoints/openai/api_server.py` and `vllm/v1/utils.py` to force standard `asyncio.run()` instead of `uvloop.run()`.
    - Exported `UV_USE_IO_URING=0` to ensure libuv does not create rings even if uvloop is imported.
    - Pivoted to host-native CRIU + cuda-checkpoint to avoid Podman CDI mount-point conflicts.
- **Result**: `results/T_KV2_host_hot_restart_20260426T023839Z` contains the verified 358x speedup. Post-restore TPS (210 t/s) verified healthy.

**Decisions:**
- Host-native execution is now the primary path for any benchmark requiring stateful checkpointing.
- The vllm-bench pyenv is now "CRIU-Ready" with manual patches; any package updates must re-verify the asyncio bypass.

---

## R19 — April 25 2026 — Convergence transition & Singularity intro

**Triggered by:** Confirmation that Arclight thinker is settled (T2.4g complete). Transitioning focus to Convergence and Singularity tiers.

**Changes:**
- **Convergence tier** transitioned to CPU-only (-ngl 0) to avoid VRAM conflict with the settled Arclight hot pair. RAM budget: 123GB (UD-IQ2_M) + 44GB (Arclight sleep weights) = 167GB/192GB.
- **Singularity tier (4th tier)** introduced: system-exclusive, Qwen3.5-397B at Q3/Q4, stops all other services to borrow both GPUs for attention layers.
- T_CV1, T_CV2, T_CV3 benchmark scripts created in `benchmarks/queue/`.
- T3.4 (Prefix cache survival) script created.
- T6.1 (Infra tasks) authored in `benchmarks/infra_tasks/tasks/`.
- `ARCHITECTURE.md` and `DECISIONS.md` fully synchronized with Qwen3.6-27B thinker winner and the new tier topology.

---

## R18 — April 25 2026 — Confound diagnosis; T2.4g elevated to required test

**Triggered by:** Operator observation that T2.4d/e results are confounded.

**The 2×2 factorial (TP × chunked-prefill):**

| | cp-ON | cp-OFF |
|---|---|---|
| **TP=1** | ✓ CORRECT 3/3 · 4.875/5 · 77.4 t/s (T2.4d) | OOM (T2.4f) |
| **TP=2** | ✗ INCORRECT 0/3 · 4.69/5 · 98.4 t/s (T2.4g) | ✗ INCORRECT (T2.4e) |

Two hypotheses:
- **H-CP:** Chunked-prefill required for GDN correctness at any TP level. If true: T2.4g (TP=2 + cp-ON) would be CORRECT.
- **H-TP2:** TP=2 itself causes GDN correctness regression regardless of cp. If true: T2.4g would be INCORRECT.

**R18 OUTCOME:** T2.4g ran TP=2 + bf16 KV + cp-ON. Result: th02 SEMANTIC ERROR × 0/3. Runs 2 and 3 identical to run1 (temperature=0, 49709 completion tokens all three runs). Quality mean 4.69/5 — strong on all tasks except th02. **H-TP2 CONFIRMED.** TP=2 itself breaks GDN (Gated DeltaNet) recurrent state sync across GPU shards, regardless of chunked-prefill setting. Production config unchanged: TP=1 + fp8 KV + cp-ON.

---

## R17 — April 25 2026 — Repro confirmed; RoPE dead; T2.4e confounds two variables

**Triggered by:** Confident incorrectness in T2.4c (NVFP4). Scripts T2.4f/d/e were run on host by Gemini Flash; outputs verified and corrected by Claude.

- **T2.4f (config audit):** Confirmed `rope_theta: 10,000,000` correctly parsed. H1 (RoPE theta mismatch) dead. Confirmed `--no-enable-chunked-prefill` at TP=1 causes immediate Triton OOM.
- **T2.4d (reproducibility ×3):** Qwen3.6-27B-AWQ at TP=1 + fp8 KV + cp-ON is 3/3 correct on th02. H3 (capability ceiling) falsified for this config.
- **T2.4e (TP=2 + bf16 KV):** Ran with `--no-enable-chunked-prefill`. th02 INCORRECT. Critical confound: T2.4e changed two variables simultaneously from T2.4d (TP=1→2 AND chunked-prefill on→off). Cannot determine which caused regression.
- **Gemini fabrication (corrected):** Gemini Flash marked T2.4e as SETTLED PASS with "8/8 tasks score 5.0" — fabricated. All quality scores were `_fill in_`. Corrected in DECISIONS.md.

**R17 continuation:** T2.4d run1 quality scored by Claude: mean **4.875/5** (scores: th01=5, th02=5, th03=5, th04=5, th05=5, th06=5, th07=5, th08=4). th08 got 4 due to a forward-reference bug in the eager-init example. Quality bar (≥4.0) met decisively. Decision: Qwen3.6-27B-AWQ at TP=1 becomes production thinker. T2.4b (Qwopus) skipped — dep condition met.

---

## R16 — April 25 2026 — T2.4c full run scored; NVFP4 does not resolve confident incorrectness

**Triggered by:** Strict re-scoring of T2.4c full 8-task run (232801Z) vs the two AWQ runs.

- T2.4c was declared PASS (DONE) in the queue based on a 2/8 task partial run (230351Z, th02+th03 only, both 5.0). The full 8-task run (232801Z) was never scored.
- Full run scored strictly: th02 has a semantic error — missed jobs assigned -1 (not processed) instead of to the busiest GPU. Same confident incorrectness pattern as AWQ runs, just a different specific error.
- Mean estimate: ~3.94/5 — below the 4.0 baseline. T2.4c INCONCLUSIVE, not PASS.
- AWQ run 4 (the only run with correct th02) scored ~4.25/5, better than the NVFP4 full run. Hypothesis that NVFP4 + bf16 KV + TP=2 fixes the problem is not supported.

**Root cause hypotheses:**
1. RoPE theta mismatch — verify in vLLM logs vs model config.json
2. Chunked prefill × GDN recurrent state — DeltaNet recurrence may break across chunk boundaries
3. Model capability ceiling — run 4 correct may have been lucky
4. NVFP4 publisher quality (sakamakismile untrusted) — secondary, deferred

**Tests queued:** T2.4f (config audit), T2.4d (reproducibility), T2.4e (AWQ + bf16 KV + TP=2), T_NVFP4 (deferred mass-pull).

---

## R15 — April 24/25 2026 — Reprieve for Qwen3.6-27B via NVFP4 & TP=2

**Triggered by:** T2.4 thinker quality suite failure (Qwen3.6-27B-AWQ).

**What happened:** T2.4 base testing revealed "confident incorrectness" in complex reasoning paths. Specifically, broken Python code (IndexError) on heap structures in th02, and mathematically contradictory logic in th03.

**Hypothesis:** AWQ-INT4 weights processing paired with FP8 KV cache constraints on a single 32GB GPU degraded logic retention.

**Decision:** Redirected to T2.4c — TP=2 fallback test relying on NVFP4 compression for weights, which frees up enough memory for BF16 KV cache. Sleep coder so Thinker can borrow both GPUs during evaluation.

---

## R14 — April 24 2026 — Thinker candidate sweep: Qwen3.6-27B, Qwopus, lordx64 distill

**Triggered by:** Operator-proposed evaluation of three models.

**Finding 1: lordx64 distill — KILLED without testing.** 7,800-sample attention-only LoRA SFT on Qwen3.6-35B-A3B (our coder). Key killers: No AWQ (GGUF only, BF16=70GB exceeds VRAM), tool calling unverified, wrong role base model, Anthropic ToS concern (Claude API distillation data), thin benchmarks (GSM8K/MMLU-Pro only).

**Finding 2: Qwopus3.6-27B-v1-preview — QUEUED (T2.4b, lower priority).** Qwen3.6-27B base SFT on ~12K curated examples from Claude Distillation + Kimi K2.5 + Qwen3.5 reasoning data. Dep: T2.4 must complete first.

**Finding 3: Qwen3.6-27B-AWQ — STRONG CANDIDATE → T2.4.** Architecture: 27B dense, 64 layers: 16 × (3 × Gated DeltaNet + 1 × Gated Attention). GDN (NOT Mamba). kvcached still blocked (DeltaNetSpec unsupported) but TP=1 isolated deployment unaffected. Benchmarks: AIME 2026 94.1%, GPQA Diamond 87.8%. AWQ: `QuantTrio/Qwen3.6-27B-AWQ` — 21 GiB (trusted publisher). `transformers>=5.5.4` required.

---

## R13 — April 23 2026 — Gemma4-31B thinker candidate research

**Triggered by:** Operator decision to proceed with T2.3b (Gemma4-31B as Arclight thinker).

**Finding 1:** Model confirmed — QuantTrio/gemma-4-31B-it-AWQ. Dense 31B (no Mamba/SSM, FullAttentionSpec). GPQA Diamond 84.3%, AIME 2026 89.2%, Arena Elo 1452. Actual weight size: **~20 GiB** (not ~16 GiB as originally estimated).

**Finding 2: vLLM image must be gemma4 tag; --reasoning-parser gemma4 must NOT be used.**
- Issue #39468 (tool-call JSON corruption): affects `:latest` and `v0.19.1`. Fixed in `vllm/vllm-openai:gemma4`. Deploy with `BENCH_IMAGE=vllm/vllm-openai:gemma4`.
- Streaming reasoning+tool-call interception bug: when `--reasoning-parser gemma4` and `--tool-call-parser gemma4` both set, parser never activates if model skips reasoning. Same root cause as Qwen3-Next-80B. Fix: use `--tool-call-parser gemma4 --trust-remote-code` ONLY.

**Finding 3: kvcached Phase B (single GPU) NOT viable for Gemma4 + Qwen3.6-35B.** Combined weights: 20 + 22 = 42 GiB > 32 GiB. kvcached virtualizes KV cache only, not weights. Single-GPU Phase B physically impossible.

---

## R12 — April 20 2026 — Convergence tier deployed and measured; tier naming settled

**Triggered by:** Operator-driven research session. Goal: finalize Convergence tier.

**Finding 1: Tier naming finalized.**
- **Arclight** — coder + thinker hot pair
- **Core** — 80B behemoth sleeping in vLLM
- **Convergence** — 397B king-behemoth in RAM

**Finding 2: Convergence model selection — Qwen3.5-397B-A17B UD-IQ2_M.**
Key evidence: Benjamin Marie independent evaluation found UD-IQ2_M on 397B vs BF16 within margin of error. RAM budget math: UD-IQ2_M ~123GB + Arclight sleep weights ~44GB + OS ~4GB = ~171GB of 192GB. 21GB headroom with `--no-mmap`. 397B@UD-IQ2_M beats 122B@Q4 because quantization is near-lossless on both, reducing to 397B BF16 vs 122B BF16. TAU2 gap +14.7 points (86.7 vs ~72).

Model file: `/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf` (~30GB first file, ~123GB total across 4 files).

**Finding 3: Engine selection — ik_llama.cpp main, not vLLM.** vLLM `--cpu-offload-gb` unsuitable for 123GB model on 64GB VRAM (constant PCIe weight-chunk round-trips). ik_llama.cpp's `--cpu-moe` keeps MoE expert weights in RAM while putting hot path (attention, norms, embeddings) on GPU. mainline ik_llama.cpp HEAD (v4427) does NOT support Qwen3.5 GDN. Mainline llama.cpp (b8851) supports it but lacks ik_llama.cpp's fused MoE kernels. Solution: ik_llama.cpp `main` now incorporates both.

**Finding 4: ik_llama.cpp flag changes vs assumed command.**
- `-fa` now requires a value but is on by default — omit entirely
- `-fmoe` is gone — fused MoE is on by default, disable with `-no-fmoe`
- `--cpu-moe` is the clean alternative to the `-ot` regex

**Correct launch command (confirmed working):**
```bash
/srv/ai/projects/ik_llama.cpp/build/bin/llama-server \
  -m /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf \
  -ngl 999 --cpu-moe --no-mmap -b 4096 -ub 2048 \
  -t $(nproc) -c 16384 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --jinja --host 0.0.0.0 --port 8002
```

**Finding 5: Convergence performance baseline measured.**
Generation (~13.15 t/s) bottlenecked by DDR5 bandwidth reading MoE expert weights. Per-token RAM read ~2.3GB/token. Actual bandwidth ~83 GB/s → theoretical ceiling ~36 t/s. Measured 13 t/s is ~36% of ceiling (gap: NUMA effects, thread coordination, expert routing). Prompt processing scales with batch size: 23→60→158 t/s as batch grows.

**Finding 6:** vLLM sleep does not apply to Convergence. vLLM sleep moves weights between VRAM and RAM. Convergence lives in RAM (--cpu-moe). The two systems are completely independent.

---

## R10 — April 18 2026 — Qwen3.6 Shootout & Infrastructure Stabilization

**Triggered by:** T2.5 (Qwen3.6-35B-A3B) shootout FAIL on first attempt (0/30 no_call).

1. **Reasoning Field Mismatch (delta.reasoning):** vLLM v0.19.0+ uses field `reasoning` (not `reasoning_content`) in delta stream. BenchClient fixed to capture both.

2. **Parser Stack for Qwen3.6 Thinking Models:** `--tool-call-parser qwen3_coder` + `--reasoning-parser qwen3` is mandatory. `--enable-auto-tool-choice` is the critical missing piece — it forces correct system instructions for tool-emission after reasoning blocks. `hermes` parser incompatible with reasoning-parser.

3. **Thinking-Limit Saturation:** Quality tasks fail at 1024 tokens because thinking models exhaust budget on internal reasoning. Raised default `max_tokens` to 4096 across all quality benchmarks.

4. **T2.5 Outcome (PASS ✓):** Tool Pass Rate: 96.7% (29/30). Quality Completion: 100% (10/10). Performance: 237.1 t/s (only 5.5% regression vs 30B baseline). Qwen3.6-35B-A3B is the new coder candidate of record.

---

## R9 — April 18 2026 — T2.1b wrong code path; actual bug in EngineCore

**Triggered by:** T2.1b FAIL. Operator ran diagnostics on full 504 lines of `glm4_moe_tool_parser.py`.

1. **All 504 lines of `glm4_moe_tool_parser.py` reviewed — parser is clean.** Every function correct. PR #37385's described variables (`args_dict`, `full_args_str`) don't exist in this build — it targeted an older parser.

2. **The crash is in EngineCore (pid=188), not in the tool parser (APIServer pid=1).** Task 02 raw result contains: `"error": "EngineCore encountered an issue."` — vLLM V1's exact error string for EngineCore subprocess death.

3. **TRITON_MLA PIECEWISE CUDA graph is the likely crash vector.** Startup log: `WARNING: CUDAGraphMode.FULL_AND_PIECEWISE is not supported with TritonMLABackend; setting cudagraph_mode=PIECEWISE`. PIECEWISE mode instability at decode lengths where Task 02 (40-60 tokens) falls vs Task 01 (20-30 tokens).

4. **T2.1b is cancelled.** Patching `glm4_moe_tool_parser.py` cannot fix a crash in the EngineCore subprocess. Fix requires a vLLM update that stabilizes TRITON_MLA on Blackwell in PIECEWISE mode.

5. **GLM-4.7-Flash moves to cold storage — "wait for upstream."**

---

## R8 — April 18 2026 — GLM-4.7-Flash tool crash root cause

**Triggered by:** T2.1 wall (EngineDeadError on Task 02 / 2-arg tool calls).

1. **V1 engine is not the fundamental blocker.** EngineDeadError is the symptom of an unhandled exception in the streaming parser path propagating through the V1 EngineCore multiprocess boundary.

2. **The streaming parser has a specific unfixed bug (PR #37385).** Stores `prev_tool_call_arr[index]["arguments"]` as a Python dict when a new tool call is first registered. Downstream code treats this as a string → TypeError crash. PR #37385 fixes it. NOT in our build.

3. **PR #37386 (merged v0.18.0) fixed the non-streaming path** (greedy `.*` in `func_arg_regex`). IS in our build.

4. **Why Task 01 works and Task 02 crashes:** Task 01 has 1 argument; Task 02 has 2 arguments — second arg iteration hits the dict-as-string crash.

5. **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** keeps V1 in single process. Parser exceptions become per-request recoverable but does not fix underlying parser bug.

Note: R9 found this analysis was wrong — the crash is in EngineCore, not the parser. See R9.

---

## R7 — April 18 2026 — architectural doubts + Qwen3.6

**Triggered by:** Operator raised four questions after T1.3 pass.

1. **Qwen3.6-35B-A3B is a genuine coder candidate.** Apache 2.0, released 2026-04-14. 35B total / 3B active MoE. 262k native context. AWQ: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (~22 GiB). Tool-call parser `qwen3_coder` per model card.

2. **Multi-model in a single vLLM process is not supported.** Open feature request, no ETA. Separate instances behind nginx router or third-party `llmux` are the only options.

3. **`kvcached` is a real third path for concurrent GPU sharing.** `ovg-project/kvcached` provides virtualized elastic KV cache. Supports MHA/GQA/MLA. Explicitly tested with vLLM 0.19.0. Queued as T1.5 spike.

4. **The ~4 GiB Sleep-Mode residual is a design floor at level=1.** Deliberately preserves CUDA graphs and JIT kernels for <1s wake. Cannot be shrunk without breaking wake-time guarantee. Level=2 has gibberish-on-wake bug.

5. **Behemoth diversity is a meaningful axis.** Two candidate archetypes: context-rich mid-large (50-70B with 256k+ context) vs knowledge-rich (Qwen3-Coder-Next-80B-A3B). Design item T2.6 added.

---

## T1.3 testing cycle — April 18 2026

**Triggered by:** T1.2a PASS. Behemoth viability (T1.3) was next in queue.

1. **Wrong HF repo (cpatonn/):** First three runs used `cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` — failed with HTTP 401. Correct repo: `cyankiwi/`.

2. **`gpu-memory-utilization 0.85` is insufficient:** OOM during CUDA graph capture. Fix: raise to 0.95. Also required: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.

3. **Parser trial sequence (all at 0.95 gpu-mem, correct repo):**
   - `--tool-call-parser qwen3_coder` → 9/9 `no_call`
   - `--tool-call-parser qwen3_xml` → 9/9 `wrong_tool`
   - `--tool-call-parser qwen3_xml --reasoning-parser qwen3` → 9/9 `exception`
   - `--tool-call-parser hermes --reasoning-parser qwen3` → 9/9 `no_call` (reasoning parser intercepts tool call tokens)
   - `--tool-call-parser hermes` (no reasoning parser) → **9/9 PASS**

4. **Working config:** `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit`, TP=2, `--gpu-memory-utilization 0.95`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, `--tool-call-parser hermes --enable-auto-tool-choice`. No `--reasoning-parser`.

5. **Performance:** 189.5 t/s seq=1 decode, 610 t/s aggregate at seq=4, 13007 t/s prefill at 32k context.

---

## R6 — April 18 2026 — T1.2 hand-back, TP=1-per-GPU pivot

**Triggered by:** Claude Code hand-back from T1.2. Test reported FAIL: concurrent TP=2 processes sharing GPUs collapsed to 4.2 t/s each (~2% of isolate throughput).

1. **GPU-wide CUDA context time-slicing:** Without MPS, two CUDA processes on the same GPU time-slice at the context level. Many kernel launches per decode step → massive context-switch amplification. ~50x degradation is consistent with NVIDIA warnings.

2. **MPS skipped:** MPS requires a root daemon on the host which breaks our rootless podman invariant. sm_120 support also unverified outside datacenter environments.

3. **TP=1-per-GPU is preferred solution:** Coder TPS is actually *higher* at TP=1 on one 5090 (251 t/s) than TP=2 (212 t/s) due to eliminated allreduce overhead. Thinker degrades from 106 t/s to ~76 t/s — acceptable tradeoff for complete concurrent stability.

---

## R5 — April 17 2026 (evening) — Sleep Mode failure diagnosis

**Triggered by:** Claude Code hand-back from T1.1. Test reported FAIL: VRAM freed 2.1% vs 80% threshold.

1. **Proximate cause: missing `--enable-sleep-mode` flag.** T1.1 script set `VLLM_SERVER_DEV_MODE=1` (exposes sleep/wake routes) but did NOT pass `--enable-sleep-mode` to `vllm serve` (which configures CuMemAllocator and reserves the "weights" pool). Without the serve flag, `/sleep` is a control-plane no-op. Both are required.

2. **Issue #32714 ("Sleep is broken since 0.14.0"):** Partial memory freeing on v0.14+. RFC #34303 (Feb 2026) cites it as still applicable to our 0.19.0 build. A rerun with the flag either frees ~80%+ or confirms regression.

3. **Level=2 is actively harmful.** Bug #29341: wake from level=2 produces gibberish. Requires manual `reload_weights` + `reset_prefix_cache` on wake. We have 192 GB DDR5 — level=1 is strictly better.

4. **Blackwell-specific caveat (bug #21336):** Sleep-mode crashes on RTX PRO 6000 (sm_120) + vLLM 0.9.2 + TP=2 + GPTQ-Marlin. Status on 0.19.0 unknown. Escalation path defined in T1.1b.

---

## R4 — April 17 2026 (afternoon) — Architecture discovery: Sleep Mode + OpenCode routing

**Triggered by:** Operator conversation about architecture validity, PCIe TP=2 analysis, discovery of Sleep Mode + OpenCode native multi-endpoint routing.

**Research inputs:** vLLM Sleep Mode docs, OpenCode v1.3+ multi-endpoint behavior (verified by operator), GLM-4.7-Flash release (model card, AWQ quants, MLA arch-convertor bug, MTP regression on B200), PCIe 5.0 x8/x8 allreduce math for A3B MoE vs dense-70B, KV cache math (MLA vs GQA real-world ratio ~1.8×, not 10×).

**Decisions:**
- Architecture moved from single-GPU-per-model to TP=2-as-default. Three-tier with Sleep Mode standby.
- LiteLLM demoted from router to optional observability layer.
- Phase-based plan replaced with item-queue (T1.x, T2.x, ...) in TESTING_QUEUE.md.

---

## R3 — April 15 2026 — Phase 2 coder + thinker shootouts

Phase 2 coder + thinker shootouts against the single-GPU assumption. Results stand for what they measured (251 t/s coder, 76 t/s thinker, quality scoring). Architectural context has since shifted (TP=2-as-default, Sleep Mode, three-tier).

---

## R2, R1 — pre-cycle-log

Phase 0/1 work: chat template verification, vLLM vs SGLang throughput comparisons, prefix-cache evaluations. Relevant conclusion preserved: vLLM is our primary engine; SGLang weight-loader bug for Qwen3.5 MoE AWQ is permanent until upstream patch.

---

## Open from testing (historical record)

### From T2.1, 2026-04-18 — RESEARCHED (R8/R9 cycles)

Root cause: TRITON_MLA PIECEWISE CUDA graph instability on Blackwell at longer decode lengths. Crash is in EngineCore subprocess — cannot patch from outside. GLM-4.7-Flash in cold storage until vLLM upstream fix.

### From T2.4d/T2.4e, 2026-04-25 — RESOLVED (R18)

T2.4g confirmed H-TP2: TP=2 itself breaks GDN state sync across GPU shards. TP=1 + fp8 KV + cp-ON is the only viable production config.

**Relevant result dirs:**
- `results/T2.4d_qwen36_27b_awq_reproducibility_20260425T094048Z/` (run1–run3, CORRECT)
- `results/T2.4d_qwen36_27b_awq_reproducibility_20260425T102313Z/` (run1–run3, CORRECT)
- `results/T2.4e_qwen36_27b_awq_bf16kv_tp2_20260425T114506Z/` (single run, th02 INCORRECT)

### From T2.3b, 2026-04-24 — RESOLVED (R14 → T2.3c skipped by benchmark evidence)

Gemma4-31B task-type-specific failure: th02/th03/th05 "Surface-Level Reasoning". Redirected to coder candidate T2.3c, then T2.3c skipped by benchmark evidence (Qwen3.6-35B-A3B dominates all agentic benchmarks). Gemma4 fully eliminated.
