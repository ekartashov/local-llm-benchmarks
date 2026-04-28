# RESEARCH_STATE.md

Living document. Current-state summary only. Full cycle log: `docs/history/cycles.md`.

**Last complete cycle:** R28 (2026-04-28) — T_PAR1 COMPLETE (Coder/Thinker reruns valid), T3.4 PARTIAL (prefix cache works, wake broken), T_CRIU2 COMPLETE (Convergence mmap RESTORE_OK), T_CRIU3 Phase 1 COMPLETE (Thinker host-native CRIU success).

**Current mode:** Research — results recorded, next priority is QX_PRELOAD design + T_KV3 Path B.

---

## Open from research / known issues

- **T_PAR1 COMPLETE (2026-04-28):** Valid reruns completed (BENCH_01–03). Coder (TP=2): N=1 240.9 t/s → N=8 1204.9 t/s, still scaling at N=8 (knee not found within tested range). Thinker (TP=1): max-num-seqs=1 queues at N>1 (76.9 t/s aggregate regardless of N); max-num-seqs=4 gives 269.4 t/s at N=4 (3.5×), plateau at N=8. Convergence: N≥2 crashes unchanged (T_CV4 result stands). **Implication for Sequential TP=2:** coder has extreme batching headroom (no saturation seen); thinker benefits from max-num-seqs=4 when multiple subagents run simultaneously. The agent framework parallelism question remains open but data supports both Arclight (concurrent) and Sequential TP=2 models.

- **T_CV4 `-np 4` vs T_PAR1 Convergence conflict:** T_CV4 measured 15.6 t/s aggregate at C=4 (server's `-np 4` internal slots, sequential client requests). T_PAR1 found concurrent client requests crash the pr-1288 server at N≥2 (GGML_ASSERT). T_CV4 measured sequential-pipelining throughput, not true parallelism. Production `-np 4` remains valid for throughput pipelining; true concurrent-request capacity is N=1 until upstream fix.

- **T_KV1 swap-space blocked:** `--swap-space 32` flag unrecognized in vLLM 0.19.0. 131K context test skipped. 65K is the current ceiling.

- **T_KV3 CRITICAL (elevated 2026-04-27):** Extended thinker is operationally necessary, not just optimization — real workloads show 27B model hitting context ceiling and failing to conclude. Two unblocking paths now documented: Path A (non-GDN replacement: DeepSeek-R1-Distill-Qwen-32B, QwQ-32B) and Path B (ik_llama.cpp tensor-split on existing Qwen3.6-27B — layer-split avoids DeltaNet sharding problem). Path B should be investigated first as it requires no model swap.

- **T_CRIU2 COMPLETE (2026-04-28):** Two findings. (1) --no-mmap: CHECKPOINT_FAILED (SYSTEM_OOM). CRIU dump of a 135 GB anon-RAM process requires VMS to spike to ~351 GB — physically impossible on 188 GB RAM. (2) mmap (--no-mmap removed): RESTORE_OK. Checkpoint 8.7 GB in 7.6 s. Restore 7.3 s. First-inference TTFT 100.56 s (page-fault warmup from NVMe, 123 GB → ~17 s theoretical, but access is demand-paged across the full generation), rep-2 36.1 s, rep-3 7.7 s. **Critical implication:** without QX_PRELOAD, CRIU mmap restore-to-interactive (100 s) is WORSE than cold start (83 s). QX_PRELOAD is now a prerequisite for CRIU to benefit Convergence. With QX_PRELOAD (pre-warm 123 GB into page cache 17 s before restore): projected 14 s restore-to-interactive — 6× improvement. QX_PRELOAD elevated to HIGH priority.

- **T_CRIU3 Phase 1 COMPLETE (2026-04-28):** Thinker host-native CRIU success. Restore time **0.43s** (target < 1s). Checkpoint size **501 MB** (CPU state only; GPU state retained via `cuda-checkpoint --toggle`). Post-restore TTFT parity (0.73s vs 0.72s). **Conclusion:** Fast-swapping is viable for TP=1 configurations using the host-native path. This unblocks Sequential TP=2 orchestration (swapping thinker/coder on GPU1).

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
| Arclight Thinker | QuantTrio/Qwen3.6-27B-AWQ | TP=1 GPU1, fp8 KV, cp-ON, --max-num-seqs 1 | 77 t/s | SETTLED |
| Convergence | unsloth/Qwen3.5-397B-A17B UD-IQ2_M | ik_llama.cpp pr-1288, -ngl 999 --cpu-moe, -np 4, -t 32 | 14 t/s | SETTLED |
| Extended Arclight | same as Coder, 65K ctx | CRIU hot restore from checkpoint, 0.28s | 238 t/s | SETTLED |

---

## Known good / settled

- **Sleep Mode:** `VLLM_SERVER_DEV_MODE=1` + `--enable-sleep-mode` frees 92.8% VRAM (~4s sleep, 0.9s wake). Level=1 only — level=2 produces gibberish on wake. ~4 GiB residual is a design floor (CUDA graphs).
- **TP=1-per-GPU isolation:** Perfect 1.0× concurrent isolation. Coder=237 t/s (T2.5), Thinker=77.4 t/s (T2.4d).
- **GDN TP=2 broken (H-TP2 confirmed T2.4g):** Gated DeltaNet recurrent state does NOT commute across TP shards. TP=2 is semantically incorrect for Qwen3.6-27B regardless of chunked-prefill. TP=1 + fp8 KV + cp-ON is the only viable thinker config.
- **CRIU hot restart:** 0.28s vs 100.2s cold. Host-native only (Podman CDI conflicts). vLLM uvloop must be patched to asyncio. `UV_USE_IO_URING=0` required.
- **VLLM_V1_ENABLED=0 mandatory:** V1 engine unstable on Blackwell sm_120 for our models. Always set in deploy.sh.
- **VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 mandatory for TP=2:** Prevents OOM during CUDA graph capture.
- **--swap-space blocked:** Flag unrecognized in vLLM 0.19.0 (R-V0 engine path). 65K is the hard context ceiling for Extended Arclight.
- **Convergence always-resident:** 83s cold start is too high for on-demand routing. Keep as always-resident service. Context ceiling 128k tokens. True concurrency = N=1 per client (pr-1288 crashes at N≥2).

## Known bad / excluded

- GLM-4.7-Flash (30B-A3B): TRITON_MLA PIECEWISE CUDA graph instability on Blackwell at longer decode lengths. Cold storage until vLLM upstream fix.
- vLLM Sleep Mode level=2: known to produce gibberish outputs on wake (bug #29341). Use level=1 exclusively.
- Gemma4-31B: REJECTED from both thinker and coder roles. See `docs/decisions/models.md`.
- Dense 70B TP=2: ~20–35 t/s, settled FAIL. Not worth testing again.
- kvcached T1.5 Phase B: GDN/DeltaNet unsupported in v0.1.5. Deferred indefinitely.
- SGLang for AWQ MoE: KeyError in qwen3_5.py:1662 (per-expert vs fused tensor format). Permanent incompatibility. Punted.

---

## Open queue summary

See `docs/queue/open.md` for full specs. Key items:

| Item | Priority | Status |
|------|----------|--------|
| T_KV3 | CRITICAL | BLOCKED — Path A: non-GDN thinker replacement; Path B: ik_llama.cpp tensor-split on existing 27B |
| QX_PRELOAD | HIGH | OPEN — Required for CRIU on Convergence. Without pre-warm: 100s first-inference (worse than cold). With pre-warm: ~14s projected. |
| T_CRIU3 | HIGH | OPEN — Checkpoint library for all vLLM processes; enables Sequential TP=2 + frees 44GB RAM |
| T_ENGINE_EVAL | MEDIUM | OPEN — GLM-4.7-Flash + others on ik_llama.cpp; vLLM sleep no longer required |
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
