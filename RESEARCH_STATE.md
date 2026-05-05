# RESEARCH_STATE.md

Living document. Current-state summary only. Full cycle log: `docs/history/cycles.md`.

**Last complete cycle:** R32 (2026-05-04) — BENCH_20: T_HARD1 hard suite closed. PQ 41/50 vs AWQ 42/50 (tie). Context/KV research: --max-model-len 131072 is free on DeltaNet hybrid; fp8 KV correct for production.

**Current mode:** Research

## Open from testing

*(none — BENCH_10 closed)*

## BENCH_21 COMPLETE (2026-05-05) — run 20260505T092106Z

Three-model co-load VERIFIED. Thinker TP=1 (GPU1, util=0.95, ctx=131K, MTP n=3) + Coder TP=1 (GPU0, util=0.80, enforce-eager, ctx=32K) + Convergence (ngl=15, --cpu-moe). All three models co-resident on 2×32 GiB RTX 5090. See `results/BENCH_21_coload_vram_20260505T092106Z/summary.md`.

**Key numbers:**

| Phase | GPU0 used | GPU0 free | GPU1 used | GPU1 free |
|-------|-----------|-----------|-----------|-----------|
| After thinker | — | 32,045 | 29,280 | 2,832 |
| After thinker+coder | 25,800 | 6,278 | 29,280 | 2,832 |
| After all three (ngl=15) | 31,377 | **701** | 31,645 | **467** |

Per-process at ngl=15: Coder 25,758 MiB (GPU0), Thinker 29,238 MiB (GPU1), Convergence 5,336 MiB (GPU0) + 2,124 MiB (GPU1). ik_llama.cpp auto-skewed 2.5:1 to GPU0 (more room available).

Convergence TPS under co-load: **4.05 t/s warm** (reps 2–3) vs **13.99 t/s isolated** (71% reduction).

**Critical findings:**
1. **vLLM is 0.20.0**: `VLLM_USE_V1`, `VLLM_ENGINE_ITERATOR_SOURCE` are unknown/ignored env vars. V1 is the only engine. All gotcha #8 / deploy.sh references to "VLLM_USE_V1=0 mandatory" are obsolete.
2. **Co-load feasible but Convergence TPS degraded at ngl=15**: 4.05 t/s warm vs 3.7 t/s CPU-only — the 15 layers barely help. However: thinker util is NOT constrained by KV pool size (see GDN correction below). Higher ngl is achievable by reducing thinker util.
3. **Coder TP=1 enforce-eager severe TPS degradation**: 25,800 MiB GPU0 with minimal KV cache. enforce-eager observed to cause up to 20× TPS degradation on the coder model (reason unknown — possibly CUDA graph warm-path vs fallback kernel path). Production coder baseline is TP=2 CUDA-graph at 237 t/s (N=1) / 1205 t/s (N=8); enforce-eager co-load TPS NOT measured in BENCH_21 but expected ~12–60 t/s range. This is a second major degradation in the co-load config alongside Convergence TPS loss.
4. **Residual margins are razor-thin at util=0.95**: 701/467 MiB free. With thinker util reduced, this relaxes substantially.

**GDN architecture correction (2026-05-05):** The thinker model (Qwen3.6-27B PrismaQuant) uses Gated DeltaNet (GDN) hybrid architecture. GDN layers maintain O(d) recurrent state — they do NOT populate the KV cache regardless of context length. "DeltaNet KV pool ~0 growth with context" (BENCH_20). The 5.92 GiB KV pool at util=0.95 is only for the few traditional attention heads in the hybrid; it is NOT a constraint on reasoning quality at 128K context. This invalidates any analysis that constrained thinker util to preserve KV token capacity.

**Two-point Convergence VRAM model** (ngl=15 measured: 7,942 MiB total; ngl=94 all-layers isolated: ~16,866 MiB):
- Fixed overhead ≈ 6,247 MiB (KV auto-allocated + shared weights, regardless of ngl)
- Per additional layer ≈ 113 MiB

| Thinker util | GPU1 free | Convergence budget | Est. max ngl | Est. Convergence TPS |
|---|---|---|---|---|
| 0.95 (current) | ~2,832 MiB | ~9,110 MiB total | ~25 | ~4.6 t/s |
| 0.80 | ~6,026 MiB | ~12,304 MiB total | ~53 | ~6.3 t/s |
| 0.73 (model floor) | ~8,309 MiB | ~14,587 MiB total | ~73 | ~8.6 t/s |

Model floor: non-KV overhead = 23,218 MiB → min_util = 23,218/32,112 ≈ 0.723. At util=0.73, KV pool ≈ 585 MiB — trivially sufficient for GDN hybrid. TPS from formula TPS(ngl) ≈ 1/(0.07148 + (94−ngl)×0.00211).

**Architecture decision needed (research mode):** Whether to:
- (A) Accept co-load at ~4 t/s Convergence with current util=0.95 (established, verified)
- (B) Run BENCH_21b: thinker util=0.73 + CONVERGENCE_NGL=70, targeting ~8.6 t/s (potentially 61% of isolated vs 29% now) — requires verifying thinker loads at util=0.73 and measuring actual ngl ceiling
- (C) Keep Convergence at ngl=999 (14 t/s) but require CRIU-based swap (option B from before)

---

## Open from research / known issues

- **T_PAR1 COMPLETE (2026-04-28):** Valid reruns completed (BENCH_01–03). Coder (TP=2): N=1 240.9 t/s → N=8 1204.9 t/s, still scaling at N=8 (knee not found within tested range). Thinker (TP=1): max-num-seqs=1 queues at N>1 (76.9 t/s aggregate regardless of N); max-num-seqs=4 gives 269.4 t/s at N=4 (3.5×), plateau at N=8. Convergence: N≥2 crashes unchanged (T_CV4 result stands). **Implication for Sequential TP=2:** coder has extreme batching headroom (no saturation seen); thinker benefits from max-num-seqs=4 when multiple subagents run simultaneously. The agent framework parallelism question remains open but data supports both Arclight (concurrent) and Sequential TP=2 models.

- **T_CV4 `-np 4` vs T_PAR1 Convergence conflict:** T_CV4 measured 15.6 t/s aggregate at C=4 (server's `-np 4` internal slots, sequential client requests). T_PAR1 found concurrent client requests crash the pr-1288 server at N≥2 (GGML_ASSERT). **SETTLED FAIL (PR #1288):** Qwen3.5-MoE has an architectural constraint preventing more than one concurrent sequence — not a bug to investigate. T_CV4 measured sequential-pipelining throughput, not true parallelism. Production `-np 4` remains valid for throughput pipelining only. True concurrent-request capacity: permanently N=1.

- **T_KV1 swap-space blocked:** `--swap-space 32` flag unrecognized in vLLM 0.19.0. 131K context test skipped. 65K is the current ceiling.

- **T_KV3 (128K Context Viability):** [SETTLED] Qwen3.6-27B (dense) confirmed viable at 128K context using ik_llama.cpp `main` branch. 1,892 t/s prefill, 49.4 t/s decode. VRAM delta ~28GB total across TP=2 (Path B confirmed).

- **Thinker max-num-seqs upgrade: 1→4 (R30, 2026-04-30):** T_PAR1 data (BENCH_02/03) proves max-num-seqs=4 is safe: 269.4 t/s at N=4 (3.5× gain), VRAM delta 4 MiB (27,736→27,732 MiB). The max-num-seqs=1 constraint was set conservatively for CUDA graph stability but is empirically unnecessary. **Action:** production thinker config should be updated to max-num-seqs=4. Updated in docs/decisions/models.md and config/models.yaml.

- **SM120 NVFP4 MoE — Marlin faster (R30, 2026-04-30):** Desktop Blackwell SM120 (RTX 5090) cannot run NVFP4 MoE grouped GEMM efficiently. Root cause: CUTLASS needs compute_120f (CUDA 13.0) for correct TMA WS grouped GEMM tactics; FlashInfer auto-detection produces compute_120a which forces slower fallback. Result: NVFP4 FlashInfer-CUTLASS = 39 t/s vs Marlin (AWQ) = 46–49 t/s on MoE. **Dense NVFP4 GEMM is NOT affected** — only the MoE grouped path has this bug. **CUDA 13.0 already released (August 2025) — not the blocker.** Actual blocker: FlashInfer #38718 (compute_120a vs compute_120f for TMA WS grouped GEMM); community Docker fix exists; upstream ETA late Q2 2026. **T_PQ2 Phase 1 now OPEN** — Marlin fallback path, no CUDA rebuild needed (same path as BENCH_12 for thinker). PrismaQuant thinker (27B dense): dense GEMM unaffected by #38718; NVFP4 unlockable via CUDA 13.0 container rebuild (not yet scheduled).

- **BENCH_12 COMPLETE (2026-05-01):** PrismaQuant 5.5bit promoted to production thinker. Quality parity confirmed on 7/8 tasks vs AWQ baseline (th08 truncated in both files). th02 EDF algorithm: PrismaQuant correct, DeltaNet not corrupted. TPS: 51.3 t/s seq=1 / 198.9 t/s seq=4 — 26–33% regression vs AWQ (76.9 / 269.4 t/s from T_PAR1). Quality rationale accepted over TPS. Currently running V0 engine + marlin/cutlass fallback (no CUDA 13.0 yet); full NVFP4 path expected to recover TPS gap.

- **BENCH_13 STALE (2026-05-01):** MTP n=1 tested on AWQ thinker: +31.8% at N=1 (76.8 → 101.2 t/s), +51% at N=4. Result valid for AWQ but model superseded by PrismaQuant before quality review was completed (th02 never scored). T_MTP1 must re-run on PrismaQuant. Sweep n=1,2,3 (author says n=3 optimal for PrismaQuant). **Warning:** T_PAR1 script undercounts tokens ~2× when MTP delivers multiple tokens per SSE chunk — use T_MTP1 script (usage.completion_tokens) for accurate TPS.

- **BENCH_14 FAIL (2026-05-01):** MTP n=1 on A3B MoE coder breaks tool-call generation. 0/3 probes produced tool_calls in response. TPS delta (+4–7%) irrelevant. MTP on coder is not viable until vLLM resolves the speculative-decoding / structured-output interaction. T_MTP2 is CLOSED FAIL — no re-test planned.

- **BENCH_19 COMPLETE (2026-05-03):** MTP n=1,2,3 sweep performed on PrismaQuant 5.5bit thinker. Result: **n=3 is optimal**. TPS reached 91.9 t/s at N=1 (+79.1% vs baseline) and 314.8 t/s at N=4 (+58.3%), utterly erasing the previous 26-33% TPS regression against the AWQ baseline without waiting for CUDA 13.0 NVFP4 kernels. Reasoning quality (th02) passed. Critically, unlike the Coder model, the Thinker model (via `qwen3_coder` tool parser) passed 5/5 tool-calling probes under MTP n=3. **Action:** MTP n=3 promoted to production Thinker configuration.

- **BENCH_20 COMPLETE (2026-05-03):** T_HARD1 hard 10-task systems engineering suite. PQ 41/50, AWQ 42/50 — statistical tie. No quality differentiation found even at maximum task difficulty. PQ task 03 (Raft) truncated at 28K reasoning tokens (finish_reason=length) — prior run has complete 5/5 response; corrected total PQ 43 vs AWQ 42. Production decision unchanged: PQ+MTP n=3 at 92 t/s vs AWQ 77 t/s. **Key operational finding:** hard thinker tasks need `--max-model-len 131072` + `max_tokens≥32000` to avoid mid-reasoning truncation. Both free on this hardware (DeltaNet KV pool ~0 growth with context).

- **MTP speculative decoding (R30, 2026-04-30):** Qwen3.6 models have native MTP heads. vLLM supports `--speculative-config '{"method":"mtp","num_speculative_tokens":X}'`. For the Thinker (PrismaQuant dense), MTP n=3 is highly successful (+79% N=1 TPS, tool calls intact). For the Coder (A3B MoE), MTP breaks structured tool calls. **#40756 does NOT apply (R30 verified):** Bug conditions are FP8+TP4+n=5+25K tokens — none match our PrismaQuant+TP1+n=1 config.
  - `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm` — thinker CANDIDATE. 349 NVFP4 + 35 MXFP8 + 112 BF16. DeltaNet layers explicitly handled. ~19 GB disk / ~22–24 GB runtime. MTP n=3 optimal per author. Dense → SM120 safe.
  - `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` — coder CANDIDATE, DEFERRED. 192 NVFP4 + 45 MXFP8 + 274 BF16. MoE → SM120 MoE kernel not ready. Preferred over cyburn 4.9bit when SM120 kernel matures.
  - `cyburn/35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-PrismaQuant-4.75bit-vllm` — wrong slot. Coder base (35B A3B MoE) with Claude reasoning distillation. Not a thinker replacement. Quality on thinker task suite unknown. Keep as quality research candidate only.

- **T_CRIU2 COMPLETE (2026-04-28):** Two findings. (1) --no-mmap: CHECKPOINT_FAILED (SYSTEM_OOM). CRIU dump of a 135 GB anon-RAM process requires VMS to spike to ~351 GB — physically impossible on 188 GB RAM. (2) mmap (--no-mmap removed): RESTORE_OK. Checkpoint 8.7 GB in 7.6 s. Restore 7.3 s. First-inference TTFT 100.56 s (page-fault warmup from NVMe, 123 GB → ~17 s theoretical, but access is demand-paged across the full generation), rep-2 36.1 s, rep-3 7.7 s. **Critical implication:** without QX_PRELOAD, CRIU mmap restore-to-interactive (100 s) is WORSE than cold start (83 s). QX_PRELOAD is now a prerequisite for CRIU to benefit Convergence. With QX_PRELOAD (pre-warm 123 GB into page cache 17 s before restore): projected 14 s restore-to-interactive — 6× improvement. QX_PRELOAD elevated to HIGH priority.

- **BENCH_18 COMPLETE (2026-05-03):** QX_PRELOAD NVMe Checkpoint Pre-warm mechanism verified — on the **Thinker** (vLLM, 31 GB GPU fat checkpoint). **Key architectural clarification:** a fat checkpoint (full GPU memory dump) is the CORRECT approach for actually freeing a GPU slot and restoring it later — the thin 501 MB checkpoint from T_CRIU3 Ph.1 retains GPU state on the card (GPU stays occupied, usable only for pause/resume of the same model). For real swaps, you need the full dump. `posix_fadvise(WILLNEED)` pre-warming of the checkpoint images reduced restore from 19.8s cold to **12.0s warm** (1.65× speedup). Convergence was NOT tested — the Convergence QX_PRELOAD target is different: pre-warming the **GGUF model files** (123 GB) into page cache, since the Convergence CRIU dump is only 8.7 GB (mmap-backed, dirty pages only) but restore hits 100s demand-paging the GGUF files from NVMe. That test is still OPEN.

- **T_CRIU3 Phase 1 COMPLETE (2026-04-28):** Thinker host-native CRIU success. Restore time **0.43s** (target < 1s). Checkpoint size **501 MB** (CPU state only; GPU state retained via `cuda-checkpoint --toggle`). Post-restore TTFT parity (0.73s vs 0.72s). **Important caveat:** this is a **pause/resume** mechanism — `cuda-checkpoint --toggle` keeps GPU weights resident in VRAM (GPU slot still occupied). This does NOT free the GPU for another model. For real GPU slot swapping (free thinker GPU, load coder, restore thinker later), a fat checkpoint with full GPU dump is required (BENCH_18 path). The 0.43s is the correct number for pause/resume-same-model only.

- **T_CRIU3 Phase 2 FAILED (2026-04-30, BENCH_10):** CRIU dump/restore succeeds for Coder TP=2 (29s dump, 67 GB; 24–26s restore). KV cache IS physically preserved (SchedulerOutput hits=2096/queries=3009 post-restore). But first inference fails: `TimeoutError: RPC call to execute_model timed out`. Root cause: SHM broadcast IPC (`ShmRingBuffer`) broken after CRIU restore for multi-process TP=2. Workers resume inside `poller.poll()` on dead `inproc://` ZMQ sockets and can never see EngineCore's `written_flag=1` writes to the SHM slot. Patches (CriuSafePoller 1000ms cap, SpinCondition.wait→sched_yield) are necessary but not sufficient for TP=2 — the underlying SHM cache coherency failure persists. VLLM_USE_V1=0 cannot help because Blackwell sm_120 forces V1 regardless. **Conclusion: CRIU KV-cache preservation for TP=2 is not feasible with vLLM 0.19.0 on Blackwell. Mechanism also undesirable: 26s restore time provides only ~4× speedup over cold start (vs 358× for TP=1). Prefix cache prefill (13.9× speedup, T3.4) is the correct KV continuity mechanism after a swap.** See BENCH_10 handoff doc for full root cause analysis.

- **T3.4 PARTIAL (2026-04-28):** Prefix cache works cleanly (cold 2410 ms → warm presleep 173 ms, 0.071 ratio, 13.9× speedup). Post-wake FAILED: `POST /wake_up` HTTP 500 `'list' object has no attribute 'zero_'` in `v1/engine/core_client.py`. New vLLM bug on Qwen3.6-35B-A3B + `--enforce-eager`. The prefix cache result is valid and trustworthy. The wake bug needs investigation: is `--enforce-eager` required for sleep on Blackwell sm_120, and if so, is the wake path broken at the engine level for this model? May be related to AWQ quantization or DeltaNet architecture interacting with eager mode state restoration.

- **Dual-architecture requirement clarified (2026-04-27):** Arclight (concurrent hot-pair) and Sequential TP=2 / Extended Arclight are complementary operating modes, not alternatives. Arclight serves agent frameworks that fan out parallel subagents; Sequential TP=2 serves deep single-context work. Both must be supported. T_PAR1 data informs research priority, not which mode to build. The arch/current.md "alternative" framing has been corrected.

- **KVcached (T1.5) partially superseded by CRIU:** KVcached solved (a) dynamic context size changes via fast swap and (b) aimed to improve concurrent TP=2 access. CRIU addresses (a) more effectively (0.28s swap between any pre-checkpointed config). (b) is now addressed via engine-agnostic deployment + layer-split parallelism. T1.5 Phase B remains DEFERRED (GDN unsupported in kvcached v0.1.5). KV cache persistence research moves to T_CRIU3.

- **Engine selection rationale obsolete:** vLLM was chosen partly for sleep mode. With CRIU providing equivalent fast-pause on any engine, the engine decision should be based on TPS, architecture support, and tool-call reliability. ik_llama.cpp already has DeltaNet support (pr-1288) and may run models that fail on vLLM. Comprehensive re-evaluation queued as T_ENGINE_EVAL. TRT-LLM (T_TRT_LLM) queued as post-settlement peak-TPS optimization (compilation cost prohibitive during exploration phase).

- **Scoring framework created (2026-04-27):** `docs/decisions/scoring.md` defines per-role evaluation weights. Model/engine selection is a usability balance (TPS × quality × context × TTFT), not a single-variable maximization. Key point: Convergence prefill throughput matters more than decode TPS; thinker quality floor is higher than coder's.

- **Handoff files relocated (2026-04-27):** HANDOFF_GEMINI_* files moved from repo root to `docs/handoffs/`. All future handoff files should be written there.

- **NVMe hardware added (2026-04-27):** Lexar NM790 4TB, 7,400 MB/s read, 6,500 MB/s write, 3,000 TBW, PCIe 4.0 x4. Pre-loading mechanism (QX_PRELOAD) can warm CRIU checkpoint images into page cache before a model switch, reducing cold-SSD restore from ~18s (for large --no-mmap checkpoints) to 0.28s. Hardware tables updated in CLAUDE.md, docs/INDEX.md, docs/arch/current.md.

- **docs/arch/current.md diagram fixed:** Coder was incorrectly shown as TP=1. Corrected to TP=2 (production config per T6.1 and T2.5). RAM table annotated with CRIU obsolescence note.

---

## Current production configuration

| Tier | Model | Config | TPS | Status |
|------|-------|--------|-----|--------|
| Arclight Coder | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | TP=2 GPU0+1, fp8 KV, ctx 32K | 237 t/s | SETTLED |
| Arclight Thinker | rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm | TP=1 GPU1, V0 engine, fp8 KV, cp-ON, --max-num-seqs 4, MTP n=3 | 92 t/s | SETTLED (BENCH_19) |
| Convergence | unsloth/Qwen3.5-397B-A17B UD-IQ2_M | ik_llama.cpp main (merged pr-1288), -ngl 999 --cpu-moe, -np 4, -t 32 | 14 t/s | SETTLED |
| Extended Arclight | same as Coder, 65K ctx | CRIU hot restore from checkpoint, 0.28s | 238 t/s | SETTLED |

---

## Known good / settled

- **Sleep Mode:** `VLLM_SERVER_DEV_MODE=1` + `--enable-sleep-mode` frees 92.8% VRAM (~4s sleep, 0.9s wake). Level=1 only — level=2 produces gibberish on wake. ~4 GiB residual is a design floor (CUDA graphs).
- **TP=1-per-GPU isolation:** Perfect 1.0× concurrent isolation. Coder=237 t/s (T2.5), Thinker=77.4 t/s (T2.4d).
- **GDN TP=2 broken (H-TP2 confirmed T2.4g):** Gated DeltaNet recurrent state does NOT commute across TP shards. TP=2 is semantically incorrect for Qwen3.6-27B regardless of chunked-prefill. TP=1 + fp8 KV + cp-ON is the only viable thinker config.
- **CRIU hot restart:** 0.28s vs 100.2s cold. Host-native only (Podman CDI conflicts). vLLM uvloop must be patched to asyncio. `UV_USE_IO_URING=0` required.
- **VLLM_USE_V1=0 mandatory:** V1 engine unstable on Blackwell sm_120 for our models. Always set in deploy.sh.
- **VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 mandatory for TP=2:** Prevents OOM during CUDA graph capture.
- **--swap-space blocked:** Flag unrecognized in vLLM 0.19.0 (R-V0 engine path). 65K is the hard context ceiling for Extended Arclight.
- **Convergence always-resident:** 83s cold start is too high for on-demand routing. Keep as always-resident service. Context ceiling 128k tokens. True concurrency = **permanently N=1** (SETTLED FAIL, PR #1288 — Qwen3.5-MoE architectural constraint, not a bug).

## Known bad / excluded

- **BENCH_16 COMPLETE (2026-05-02):** GLM-4.7-Flash (30B-A3B) verified on `ik_llama.cpp`. 176.4 t/s achieved, 5/5 tool-calling pass. Resolved previous Triton/Blackwell MLA instability. Model is now a viable coder alternative (T2.2).

- vLLM Sleep Mode level=2: known to produce gibberish outputs on wake (bug #29341). Use level=1 exclusively.
- Gemma4-31B: REJECTED from both thinker and coder roles. See `docs/decisions/models.md`.
- Dense 70B TP=2: ~20–35 t/s, settled FAIL. Not worth testing again.
- kvcached T1.5 Phase B: GDN/DeltaNet unsupported in v0.1.5. Deferred indefinitely.
- SGLang for AWQ MoE: KeyError in qwen3_5.py:1662 (per-expert vs fused tensor format). Permanent incompatibility. Punted.
- **CRIU KV-cache preservation for vLLM TP=2 (T_CRIU3 Phase 2):** SHM broadcast IPC (`ShmRingBuffer`) is broken after CRIU restore for multi-process configurations. Workers cannot see EngineCore writes to SHM slots. Blackwell sm_120 forces V1 engine regardless of `VLLM_USE_V1=0`, making the SHM path unavoidable. Additionally, TP=2 checkpoint is 67 GB (26s restore) — only ~4× faster than cold start. Not worth fixing. TP=1 CRIU (Phase 1) is unaffected.

---

## Open queue summary

See `docs/queue/open.md` for full specs. Key items:

| Item | Priority | Status |
|------|----------|--------|
| QX_PRELOAD | HIGH | OPEN — BENCH_18 proven posix_fadvise on checkpoint images (31 GB Thinker fat dump: 19.8s→12.0s). Convergence is different: CRIU dump is only 8.7 GB (mmap, dirty pages) but restore hits 100s demand-paging 123 GB GGUF from NVMe. QX_PRELOAD for Convergence must pre-warm the GGUF model files, not the checkpoint images. Never tested. |
| T_CRIU3 Ph.1 | DONE ✓ | Thinker TP=1: 0.43s restore, 501 MB, TTFT parity. Sequential TP=2 swaps unblocked. |
| T_CRIU3 Ph.2 | DONE ✗ | Coder TP=2: dump/restore OK (29s/67GB), KV preserved, inference FAIL (SHM IPC broken post-restore). |
| T_ENGINE_EVAL | DONE ✓ | GLM-4.7-Flash verified on ik_llama.cpp (BENCH_16). |
| T2.6 | MEDIUM | OPEN — Behemoth archetype scouting (design item) |
| T_PAR1 | DONE ✓ | Coder: 240–1205 t/s (N=1–8, no saturation). Thinker: 269 t/s at max-num-seqs=4. Convergence: N≥2 crashes. |
| T_CRIU2 | DONE ✓ | --no-mmap OOM. mmap: 8.7 GB / 7s restore / 100s first-inference. QX_PRELOAD required. |
| T_CV5 | DONE ✓ | NGL sweep + MoE offload done. |
| T3.4 | DONE ✗ | Prefix cache works. Wake broken for Qwen3.6-35B-A3B + --enforce-eager. |

---

## New candidate models (pre-queue)

| Model | Role candidate | AWQ available | Main risks |
|-------|----------------|---------------|-----------|
| GLM-4.7-Flash (30B-A3B) | Coder alternative | Yes: cyankiwi/cpatonn | COLD STORAGE — EngineCore crash at complex tools |
| Qwen3.6-35B-A3B (settled) | Coder | Yes: cyankiwi | SETTLED T2.5 PASS |
| Qwen3.6-27B (settled) | Thinker | Yes: QuantTrio | SETTLED T2.4d PASS |
