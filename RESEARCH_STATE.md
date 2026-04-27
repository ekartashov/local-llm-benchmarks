# RESEARCH_STATE.md

Living document. Current-state summary only. Full cycle log: `docs/history/cycles.md`.

**Last complete cycle:** R27 (2026-04-26) — T_KV1 PASS (65K context), T_PAR1 PARTIAL (Convergence only — Coder/Thinker data UNRELIABLE, fabricated by Gemini Flash), T_CV1–4 PASS, T_KV2 PASS (0.28s hot restart).

**Current mode:** Research — repo tidy + doc restructure after Gemini Flash session.

---

## Open from research / known issues

- **T_PAR1 Coder/Thinker data UNRELIABLE (2026-04-26):** `metrics.json` shows `coder_detail: null, thinker_detail: null`. Raw dir contains only `convergence_sweep.json` — coder/thinker endpoints were not running. Numbers reported by Gemini ("1,196 t/s at N=8", "698 t/s at N=4", "Thinker OOM at N>1") are invented. T_PAR1 REOPENED for Coder and Thinker. Convergence portion (crashes at N≥2) is backed by real data. NOTE: T_PAR1 now also serves as workload characterization for the Sequential TP=2 architecture decision — the key question is whether OpenCode genuinely requires simultaneous coder+thinker access.

- **T_CV4 `-np 4` vs T_PAR1 Convergence conflict:** T_CV4 measured 15.6 t/s aggregate at C=4 (server's `-np 4` internal slots, sequential client requests). T_PAR1 found concurrent client requests crash the pr-1288 server at N≥2 (GGML_ASSERT). T_CV4 measured sequential-pipelining throughput, not true parallelism. Production `-np 4` remains valid for throughput pipelining; true concurrent-request capacity is N=1 until upstream fix.

- **T_KV1 swap-space blocked:** `--swap-space 32` flag unrecognized in vLLM 0.19.0. 131K context test skipped. 65K is the current ceiling.

- **T_KV3 CRITICAL (elevated 2026-04-27):** Extended thinker is operationally necessary, not just optimization — real workloads show 27B model hitting context ceiling and failing to conclude. Two unblocking paths now documented: Path A (non-GDN replacement: DeepSeek-R1-Distill-Qwen-32B, QwQ-32B) and Path B (ik_llama.cpp tensor-split on existing Qwen3.6-27B — layer-split avoids DeltaNet sharding problem). Path B should be investigated first as it requires no model swap.

- **Architecture direction shift (2026-04-27):** CRIU established as universal fast-swap mechanism, not vLLM-specific. Three new research lines open: (1) T_CRIU2: CRIU for ik_llama.cpp/Convergence — if confirmed, always-resident policy is optional and UD-IQ3_XXS quant becomes viable with freed RAM; (2) T_CRIU3: checkpoint library standardization (KV cache preservation sub-question included — CRIU may preserve populated KV blocks across restores, qualitatively better than sleep mode); (3) T_ENGINE_EVAL: vLLM sleep is no longer the differentiator — cold-storage models (GLM-4.7-Flash) and other engines (ik_llama.cpp) should be re-evaluated on capability merits alone.

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
| T_PAR1 | HIGH | OPEN — Coder/Thinker rerun needed (fabricated data); also workload characterization for Sequential TP=2 decision |
| T_KV3 | CRITICAL | BLOCKED — Path A: non-GDN thinker replacement; Path B: ik_llama.cpp tensor-split on existing 27B |
| T_CRIU2 | HIGH | OPEN — Test CRIU on ik_llama.cpp/Convergence; confirms engine-agnostic fast-swap |
| T_CRIU3 | HIGH | OPEN — Checkpoint library for all models; enables Sequential TP=2 + frees 44GB RAM |
| T_CV5 | MEDIUM | OPEN — Convergence -ngl sweep for optimal GPU offload fraction |
| T_ENGINE_EVAL | MEDIUM | OPEN — GLM-4.7-Flash + others on ik_llama.cpp; vLLM sleep no longer required |
| QX_PRELOAD | MEDIUM | OPEN — NVMe pre-load for warm CRIU restores |
| T3.4 | MEDIUM | OPEN — Prefix cache survival across sleep/wake |
| T2.6 | MEDIUM | OPEN — Behemoth archetype scouting (design item) |

---

## New candidate models (pre-queue)

| Model | Role candidate | AWQ available | Main risks |
|-------|----------------|---------------|-----------|
| GLM-4.7-Flash (30B-A3B) | Coder alternative | Yes: cyankiwi/cpatonn | COLD STORAGE — EngineCore crash at complex tools |
| Qwen3.6-35B-A3B (settled) | Coder | Yes: cyankiwi | SETTLED T2.5 PASS |
| Qwen3.6-27B (settled) | Thinker | Yes: QuantTrio | SETTLED T2.4d PASS |
