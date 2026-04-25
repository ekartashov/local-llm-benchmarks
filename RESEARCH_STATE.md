# RESEARCH_STATE.md

Living document. What we currently believe, what is still open, and the log of research ↔ testing cycles.

**Current cycle:** R16 OPEN — T2.4c full 8-task run (232801Z) scored; NVFP4 did NOT resolve confident incorrectness on th02 (semantic error, mean ~3.94/5). Root cause under investigation. Three hypotheses: RoPE theta mismatch, chunked-prefill × GDN recurrence, model capability ceiling.
**Current mode:** TESTING — priority queue: T2.4f (RoPE/chunked-prefill config audit, zero-cost, run first) → T2.4d (AWQ run 4 reproducibility ×3) → T2.4e (AWQ + bf16 KV + TP=2). T2.3c and T_CV1 remain OPEN and can run in parallel on GPU0.

---

## What we believe right now

### Known good, ready to deploy

- Qwen3-Coder-30B-A3B-AWQ on vLLM, single GPU: 251 t/s seq=1, 730 t/s aggregate at c=4. Tool calls reliable with `--tool-call-parser qwen3_coder --reasoning-parser qwen3`. Measured.
- Qwen3.6-35B-A3B-AWQ on vLLM, single GPU / TP=2 fallback: 237.1 t/s seq=1, 715.6 t/s aggregate at c=4. 100% Quality completion, 96.7% Tool Reliability. **New Coder Winner**. Requires `--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice`. Measured T2.5 (2026-04-18).
- Qwen3.5-27B-AWQ on vLLM, single GPU: 76 t/s, quality 4.0/5 on 8-task thinker suite. Needs `--max-num-seqs 1`. **Defect th03 remains**: Task T1.4 (2026-04-18) confirmed that even at `max_tokens=16384`, the model exhausts its budget in a reasoning loop. Architecture-heavy tasks must be routed to coder or behemoth.
- vLLM is our primary engine. vLLM launches cleanly with our rootless podman setup on Blackwell sm_120 at TP=2. AWQ-Marlin kernel path confirmed functional (T1.1 run loaded 18 GiB weights across TP=2 cleanly).
- Sleep Mode confirmed working end-to-end (T1.1 PASS 2026-04-17): `VLLM_SERVER_DEV_MODE=1` + `--enable-sleep-mode` frees 92.8% VRAM (59 → 4 GiB) in ~4s, wake in 0.9s, post-wake TPS 212.3 t/s (ratio 1.000). vLLM 0.19 reasoning-parser streaming field is `delta.reasoning` (o1 style), not `delta.reasoning_content`.
- Qwen3-Coder-Next-80B-A3B-AWQ (behemoth) on vLLM TP=2: 189.5 t/s seq=1, 610 t/s aggregate at seq=4, 13007 t/s prefill@32k. Tool calls 100% reliable with `--tool-call-parser hermes` and **no** `--reasoning-parser`. Requires `--gpu-memory-utilization 0.95` and env `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`. HF repo: `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit`. Measured T1.3 (2026-04-18).

### Known bad / excluded

- Ollama, KTransformers, Devstral, DeepSeek-R1-32B, GLM-4.6/4.7 full, GLM-4.6-Air-doesn't-exist — see `DECISIONS.md` SETTLED.
- GLM-4.7-Flash (30B-A3B) — MLA confirmed active (TRITON_MLA backend), but V1 tool-call crashes block tool-intensive use. In cold storage until vLLM V1 stabilizes for this architecture. See `DECISIONS.md` and Open from testing.
- bf16 deployment — the quant we serve is AWQ-INT4, bf16 "does it fit" tests were misleading and have been discarded.
- vLLM Sleep Mode **level=2** — do not use. See `DECISIONS.md`; known to produce gibberish outputs on wake (bug #29341) and requires manual `reload_weights` + `reset_prefix_cache` after wake which is easy to get wrong. Use level=1 exclusively. We have 192 GB DDR5, there is no reason to prefer level=2 for us.
- **Gemma4-31B-it-AWQ** (Arclight Thinker candidate) — REJECTED as primary thinker. Mean quality 4.0/5 matches Qwen3.5-27B but fails depth-of-reasoning bar on th02/th03/th05. Redirected: strong 5/8 task profile (th01, th04, th06, th07, th08 all scored 5), 100% task completion, dense/no-MambaSpec → queued as **coder** candidate T2.3c.
- **lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled** — KILLED without testing. 7,800-sample attention-only LoRA on our current coder base. No AWQ, no verified tool calling, Anthropic ToS concern for distillation data.
- **Qwen3.6-27B — INCONCLUSIVE thinker (T2.4/T2.4c, 2026-04-24/25).** Quality is config-sensitive: AWQ run 4 (fp8 KV, max_tokens=16384, ctx=32768) produced correct th02 code (~4.25/5). All other runs — including NVFP4+bf16+TP=2 — have semantic errors on th02 (confident incorrectness pattern). Root cause under investigation (RoPE theta, chunked-prefill × GDN, capability ceiling). Qwen3.5-27B remains thinker baseline. See DECISIONS.md for run history and hypotheses.

### Working architectural hypothesis

Two-GPU-two-role (coder TP=1 on GPU0, thinker TP=1 on GPU1, concurrent isolation) + behemoth TP=2 asleep. See `ARCHITECTURE.md`.

Critical unknowns remaining:
1. ~~Does Sleep Mode work?~~ **SETTLED — yes (T1.1 PASS)**
2. ~~Do two vLLM processes coexist at gpu-mem 0.40 each on shared GPUs?~~ **SETTLED — FAIL. Both fit in memory, but CUDA context time-slicing reduces concurrent decode to ~2%. Not viable.**
3. ~~Does TP=1-per-GPU provide sufficient TPS for coder and thinker?~~ **SETTLED — PASS (T1.2a). Perfect 1.0x concurrent isolation. Coder=251t/s, Thinker=76.5t/s.**
4. ~~Does Qwen3-Coder-Next-80B-A3B-AWQ TP=2 hit ≥40 t/s?~~ **SETTLED — PASS (T1.3). 189 t/s seq=1 decode, 100% tool reliability with `hermes` parser.**

---

## New candidate models (added since last cycle)

| Model | Role candidate | AWQ available | Main risks |
|-------|----------------|---------------|-----------|
| GLM-4.7-Flash (30B-A3B) | Coder alternative | Yes: cyankiwi/cpatonn | MLA detection in vLLM (T2.1); MTP regression on Blackwell (T3.2) |
| GLM-4.5-Air (106B/12B) | Thinker alternative | Yes: cpatonn | Marlin MoE group_size=64 caps TP=2 (fine for us) |
| Qwen3-Coder-Next (80B-A3B) | Behemoth | Yes: cyankiwi | ~~TP=2 viability~~ SETTLED T1.3 PASS; MTP on sm_120 (T3.3) |
| Qwen3.6-35B-A3B (Apache 2.0, 2026-04-14) | Coder alternative | Yes: cyankiwi | SETTLED T2.5 PASS (97%/100%/237tps) |

---

## Cycle log

### R16 — April 25 2026 — T2.4c full run scored; NVFP4 does not resolve confident incorrectness

**Triggered by:** Strict re-scoring of T2.4c full 8-task run (232801Z) vs the two AWQ runs (run 4 at 152735Z, run 5 at 163624Z) that T2.4c was supposed to supersede.

**What happened:**
- T2.4c was declared PASS (DONE) in the queue based on a 2/8 task partial run (230351Z, th02+th03 only, both 5.0). The full 8-task run (232801Z) was never scored.
- Full run scored strictly (1–5, skeptical): th02 has a **semantic error** — missed jobs are assigned `-1` (not processed) instead of being assigned to the busiest GPU. The model explicitly argues this is correct ("if it misses on the best GPU, it misses on all"). This is the same confident incorrectness pattern as the AWQ runs, just a different specific error.
- Mean estimate: ~3.94/5 — below the 4.0 baseline. T2.4c is INCONCLUSIVE, not PASS.
- Critically: AWQ **run 4** (the only run with correct th02) scored ~4.25/5, which is better than the NVFP4 full run. The hypothesis that NVFP4 + bf16 KV + TP=2 fixes the problem is not supported by the full run data.

**Root cause hypotheses identified (see DECISIONS.md for detail):**
1. RoPE theta mismatch — verify `rope_theta` in vLLM logs vs model config.json (zero cost, do first)
2. Chunked prefill × GDN recurrent state — DeltaNet recurrence may break across chunk boundaries
3. Model capability ceiling — run 4 correct may have been lucky; needs reproducibility test
4. NVFP4 publisher quality (sakamakismile untrusted) — secondary, deferred

**Decisions updated:** DECISIONS.md Qwen3.6-27B entry expanded with run history and hypotheses.
**Tests queued:** T2.4f (config audit), T2.4d (reproducibility), T2.4e (AWQ + bf16 KV + TP=2), T_NVFP4 (deferred mass-pull).
**T2.4b status:** restored to OPEN (Qwopus SFT still potentially relevant if capability ceiling is confirmed).

---

### R15 — April 24/25 2026 — Reprieve for Qwen3.6-27B via NVFP4 & TP=2

**Triggered by:** T2.4 thinker quality suite failure (Qwen3.6-27B-AWQ).

**What happened:** T2.4 base testing of Qwen3.6-27B-AWQ at TP=1 completed with a 100% answer rate but revealed "confident incorrectness" in complex reasoning paths. Specifically, it produced broken Python code (IndexError) on heap structures in th02, and mathematically contradictory logic in th03. 

**Decisions updated:**
- Qwen3.6-27B is **not** totally rejected. We hypothesized that intense AWQ-INT4 weights processing paired with FP8 KV cache constraints on a single 32GB GPU degraded its logic retention.
- It is retained as a CANDIDATE and redirected to **T2.4c**: a TP=2 fallback test relying on NVFP4 compression for weights, which frees up enough memory for BF16 KV cache.
- To execute this, the coder will be slept (`vLLM sleep-mode`) so the Thinker can borrow both GPUs during intensive evaluation chunks.
- `DECISIONS.md`, `TESTING_QUEUE.md`, and `config/models.yaml` were updated to open this new pathway.

**Tests queued:** T2.4c (NVFP4 + TP=2 test).

### R14 — April 24 2026 — Thinker candidate sweep: Qwen3.6-27B, Qwopus, lordx64 distill

**Triggered by:** Operator-proposed evaluation of three models: `lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled`, `Jackrong/Qwopus3.6-27B-v1-preview`, and `Qwen/Qwen3.6-27B` (newly released).

---

**Finding 1: lordx64 distill — KILLED without testing.**

This model is a 7,800-sample attention-only LoRA SFT (0.01% of params, only `q/k/v/o_proj`) on top of Qwen3.6-35B-A3B — our current coder base. Key killers:

1. **No AWQ available.** Only GGUF (IQ4_XS 18.9GB, Q5_K_M 25GB, Q8_0 35GB). BF16 = ~70GB, exceeds 64GB total VRAM for in-VRAM AWQ calibration. CPU-offloaded AWQ conversion is possible but takes 4-6h for uncertain gain.
2. **Tool calling unverified.** Model card has no mention of tool calling or `--tool-call-parser`. Fine-tuning preserved only attention matrices — tool call format depends on the template and post-training, which was not validated.
3. **Thin fine-tune on wrong role.** Qwen3.6-35B-A3B is our coder, not a thinker base. The SFT teaches "Claude reasoning style" (chain-of-thought format), not factual uplift — the limitations section explicitly says reasoning ≠ knowledge transfer.
4. **Anthropic ToS concern.** Training data was generated via Claude API. Model card acknowledges users "must verify compliance" with Anthropic usage policies. Commercial distillation from the Claude API may violate ToS.
5. **Benchmarks thin.** Only GSM8K and MMLU-Pro reported; no GPQA Diamond or hard reasoning evals.

Decision: do not download, do not convert. If a Claude-distilled reasoning model becomes interesting, wait for a properly released version with verified tool calling and clean provenance.

---

**Finding 2: Qwopus3.6-27B-v1-preview — QUEUED (T2.4b, lower priority).**

Architecture confirmed from model card: Qwen3.6-27B base (27B dense, Gated DeltaNet hybrid, same 64-layer layout as base). SFT on ~12K curated examples from Claude Distillation + Kimi K2.5 reasoning + Qwen3.5 reasoning data. v1-preview status, 16-prompt evaluation only.

**Why it stays in the queue despite preview status:** The SFT data direction (reasoning distillation) directly targets the failure modes found in Gemma4 (th02/th05 — multi-step constraint reasoning). If Qwen3.6-27B base (T2.4) misses on those same tasks, a reasoning-distilled variant is the logical next step.

**Deployment for T2.4b:** No AWQ available. Two options:
1. Serve BF16 at TP=2 — 27B × 2 bytes ≈ 54GB fits 64GB with `--gpu-mem-util 0.85` (requires sleeping coder)
2. Community GGUF if QuantTrio or others release one before T2.4b runs

**Dep:** T2.4 must complete first. If Qwen3.6-27B base clears the quality bar, T2.4b is skipped. Only run if base fails on th02/th05.

---

**Finding 3: Qwen3.6-27B-AWQ — STRONG CANDIDATE → T2.4.**

This is the top thinker candidate since Qwen3.5-27B. Key facts:

**Architecture:** 27B dense, 64 layers: 16 × (3 × Gated DeltaNet + 1 × Gated Attention). GDN (Gated Delta Network) hybrid — NOT Mamba. User's observation "no Mamba anymore" is accurate; GDN is a different linear attention mechanism. kvcached status: DeltaNetSpec is not in kvcached v0.1.5's supported list (FullAttentionSpec, SlidingWindowSpec, MLAAttentionSpec only) → kvcached T1.5 Phase B remains blocked, same as with Qwen3.5-27B. But this does NOT block isolated TP=1 deployment.

**Benchmarks (published by Qwen, April 2026):**

| Benchmark | Qwen3.6-27B | Gemma4-31B | Qwen3.5-27B (est.) |
|-----------|-------------|------------|---------------------|
| AIME 2026 | **94.1%** | 89.2% | — |
| GPQA Diamond | **87.8%** | 84.3% | — |
| MMLU-Pro | **86.2%** | — | — |
| SWE-bench Verified | **77.2%** | — | — |

Across every published benchmark, Qwen3.6-27B dominates Gemma4-31B, which was already our quality bar. The 4.9pp AIME gap and 3.5pp GPQA gap are meaningful, not noise.

**Deployment:**
- AWQ: `QuantTrio/Qwen3.6-27B-AWQ` — 21 GiB (our trusted publisher) ✓
- Weight fit: 21 GiB + `--gpu-mem-util 0.90` → 28.8 GiB usable → 7.8 GiB for KV + CUDA graphs. Sufficient for thinker workload at ctx=32768.
- Parser stack: `--tool-call-parser qwen3_coder --reasoning-parser qwen3` — same as our coder (Qwen3.6-35B-A3B-AWQ). This combination is proven at 96.7% tool reliability in T2.5. No new parser risk.
- Vision encoder: model card includes vision capabilities. Use `--language-model-only` to shed the vision encoder and reclaim 1-3 GiB VRAM for KV cache. Verify this flag is supported in vLLM 0.19.x; if not, model still loads within budget without it.
- `transformers>=5.5.4` required per QuantTrio card — verify during deployment; may need a newer vllm image if the standard one ships an older transformers.
- `vllm>=0.19.0` confirmed compatible (QuantTrio card).

**Open vLLM issues (from GitHub search):**
- #40621 batch inference: affects Qwen3.5/3.6 series with multiple concurrent requests. For thinker role (typically single sequential requests), unlikely to manifest. Monitor but don't pre-block.
- #40756 MTP speculative decoding crash: only if `--speculative-config` used. We do NOT use MTP. Not applicable.
- #40725 TP=4 non-English corruption: only at TP=4. We use TP=1. Not applicable.

**kvcached T1.5 Phase B status:** GDN hybrid means this remains blocked (DeltaNetSpec unsupported in kvcached v0.1.5). Same as Qwen3.5-27B. This is already accounted for in TESTING_QUEUE.md (T1.5 Phase B deferred). No regression relative to current thinker.

---

**Decisions updated this cycle:**
- `DECISIONS.md`: lordx64 distill killed; Qwopus deferred; Qwen3.6-27B queued as T2.4; GDN/kvcached note added
- `TESTING_QUEUE.md`: T2.4 + T2.4b added; status table updated
- `config/models.yaml`: qwen36_27b_awq CANDIDATE added; qwopus entry added; lordx64 entry added as ELIMINATED
- `RESEARCH_STATE.md`: "What we know" updated with Qwen3.6-27B architecture facts

**Tests queued for next cycle:**
- **T2.4**: Qwen3.6-27B-AWQ as Arclight thinker — run next, no deps
- **T2.4b**: Qwopus3.6-27B SFT — lower priority, run only if T2.4 misses on th02/th05

---

### R13 — April 23 2026 — Gemma4-31B thinker candidate research

**Triggered by:** Operator decision to proceed with T2.3b (Gemma4-31B as Arclight thinker).

**Research findings:**

**Finding 1: Model confirmed — QuantTrio/gemma-4-31B-it-AWQ.**

Gemma 4 was released April 2, 2026 under Apache 2.0. Dense 31B (no Mamba/SSM layers — pure Transformer, FullAttentionSpec). Published benchmarks: GPQA Diamond 84.3%, AIME 2026 89.2%, Arena Elo 1452 (ranks #3 open model globally). QuantTrio/gemma-4-31B-it-AWQ is the AWQ-4bit quant from our trusted publisher.

**Weight size correction:** The T2.3b spec estimated ~16 GiB. Actual model card shows **~20 GiB**. This changes the kvcached Phase B analysis — see Finding 3.

---

**Finding 2: vLLM image must be gemma4 tag; --reasoning-parser gemma4 must NOT be used.**

Two bugs in vLLM 0.19.0/0.19.1 affect Gemma4:

1. **Issue #39468 (tool-call JSON corruption):** String values in tool call arguments are wrapped with `<|"|>` chars, producing malformed JSON. Affects `vllm/vllm-openai:latest` and `v0.19.1`. Fixed in the `vllm/vllm-openai:gemma4` docker tag. Deploy with `BENCH_IMAGE=vllm/vllm-openai:gemma4`.

2. **Streaming reasoning+tool-call interception bug:** When `--reasoning-parser gemma4` and `--tool-call-parser gemma4` are both set, the streaming code path waits for `</think>` before activating the tool call parser. If the model skips reasoning and goes directly to tool calls, the parser never activates — raw tool call tokens appear as text content. This is the same root cause as the Qwen3-Next-80B problem (which required `--tool-call-parser hermes` alone, no reasoning-parser). **Fix: use `--tool-call-parser gemma4 --trust-remote-code` ONLY. Do not add `--reasoning-parser gemma4`.**

The correct deploy flags: `--tool-call-parser gemma4 --trust-remote-code` (deploy.sh auto-adds `--enable-auto-tool-choice` when `--tool-call-parser` is present).

---

**Finding 3: kvcached Phase B (single GPU) is NOT viable for Gemma4 + Qwen3.6-35B.**

Combined weight footprint: Gemma4-31B ~20 GiB + Qwen3.6-35B ~22 GiB = **42 GiB** — exceeds the 32 GiB of a single RTX 5090. kvcached virtualizes KV cache pages only, not model weights. Single-GPU Phase B is physically impossible regardless of kvcached version.

The kvcached opportunity with Gemma4 is different: **cross-GPU elastic KV sharing** (coder on GPU0 TP=1, thinker on GPU1 TP=1, shared virtual KV pool across both GPUs). This is a different topology from the original Phase B design and needs separate research before T1.5 Phase B can be redesigned.

Implication for current architecture: TP=1-per-GPU (coder on GPU0, thinker on GPU1) remains the baseline regardless of T2.3b outcome. kvcached cross-GPU KV sharing is a potential optimization, not a blocker.

---

**Finding 4: TP=1 on single 32 GiB GPU is viable.**

20 GiB weights on 32 GiB card: with `--gpu-memory-utilization 0.90` → 28.8 GiB usable, ~8.8 GiB remaining for KV and CUDA graphs. Sufficient for ctx=32768 on a thinker workload (lower concurrency than coder). The `vllm/vllm-openai:gemma4` tag explicitly supports TP=1 single-GPU for the AWQ variant.

---

**Decisions updated this cycle:**
- `config/models.yaml`: added `gemma4_31b_it_awq` CANDIDATE entry with corrected weight size and known bugs
- `DECISIONS.md`: added `--reasoning-parser gemma4` prohibition (same pattern as Core/80B)
- `TESTING_QUEUE.md`: T2.3b procedure updated with correct weight size, image tag, and kvcached Phase B note
- `benchmarks/queue/T2.3b_gemma4_31b_thinker.sh`: created with all findings embedded

**Tests queued for next cycle:**
- **T2.3b**: run immediately — script ready, no dependencies.

**Open from research:** none — queue is ready for testing.

---

### R12 — April 20 2026 — Convergence tier deployed and measured; tier naming settled

**Triggered by:** Operator-driven research session. Goal: finalize Convergence tier model selection, engine selection, deployment configuration, and record all findings before any data is lost.

**Context:** This research cycle was conducted conversationally alongside live deployment on the host. Many decisions were validated interactively by running commands on ZRH01-AIRIG.

---

**Finding 1: Tier naming finalized.**

Permanent names agreed:
- **Arclight** — coder + thinker hot pair (Steins;Gate operation theme — fast, electric)
- **Core** — 80B behemoth sleeping in vLLM (Undertale Core — slower but powerful)
- **Convergence** — 397B king-behemoth in RAM (ephemeral, anomalous, omnipotent — deeper than the Core)

All docs, configs, and scripts should use these names going forward.

---

**Finding 2: Convergence model selection — Qwen3.5-397B-A17B UD-IQ2_M.**

Key evidence chain:
1. Benjamin Marie (independent, H200s) evaluated Unsloth UD-IQ2_M on 397B vs BF16 across MMLU-Pro/GPQA Diamond/LiveCodeBench/Math-500 — found performance difference within margin of error. He reran twice because the result was surprising. This is the most credible external evaluation available.
2. 512-expert MoE architecture tolerates 2-bit compression on expert weights better than dense models — each expert handles a narrow specialization, so per-expert compression loss doesn't accumulate the way it does in dense forward passes.
3. RAM budget math: UD-IQ2_M at ~123GB + Arclight sleep weights (~44GB) + OS (~4GB) = ~171GB of 192GB. 21GB headroom with `--no-mmap` (fully pinned). UD-IQ3_XXS at ~140GB leaves only ~8GB — dangerously tight.
4. 397B@UD-IQ2_M beats 122B@Q4 as Convergence because: quantization is near-lossless on both, so this reduces to 397B BF16 vs 122B BF16. TAU2 gap is +14.7 points (86.7 vs ~72) — genuine behavioral difference in multi-step agentic orchestration, which is exactly what Convergence is invoked for.
5. MiniMax M2.5 eliminated: community-verified catastrophic quality degradation at IQ2-Q4. Do not use as Convergence despite having similar MoE architecture.

**Model file location:**
```
/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/
  da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/
  Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf  (~30GB)
  Qwen3.5-397B-A17B-UD-IQ2_M-00002-of-00004.gguf  (~50GB)
  Qwen3.5-397B-A17B-UD-IQ2_M-00003-of-00004.gguf  (~50GB)
  Qwen3.5-397B-A17B-UD-IQ2_M-00004-of-00004.gguf  (~24GB)
Total: ~123GB across 4 split files. Reference 00001-of-00004; loader finds the rest.
```

---

**Finding 3: Engine selection — ik_llama.cpp pr-1288, not vLLM.**

vLLM `--cpu-offload-gb` is unsuitable for a 123GB model on 64GB VRAM — it involves constant PCIe weight-chunk round-trips per forward pass. ik_llama.cpp's `--cpu-moe` keeps the MoE expert weights in RAM (sequential read during inference) while putting the hot path (attention, norms, embeddings) on GPU — the correct split for sparse MoE.

**The engine problem:** mainline ik_llama.cpp HEAD (version 4427, commit 07516cec) does not support Qwen3.5 GDN architecture. Grep confirms: no `ssm_alpha`, `Qwen3_5`, or `GatedDelta` in `src/llama.cpp`. Mainline llama.cpp (b8851) does support it — has `src/models/qwen35moe.cpp`, `src/models/delta-net-base.cpp`, `LLM_ARCH_QWEN35MOE`. But mainline llama.cpp lacks ik_llama.cpp's `-fmoe` (fused MoE kernel) optimization.

**Solution:** PR #1288 on ik_llama.cpp adds Qwen3.5 MoE support (`LLM_ARCH_QWEN35MOE`, `build_qwen35moe()`, `llama-delta-net.cpp` with `ssm_alpha`, `ssm_beta`). Checking out this branch and rebuilding gives both Qwen3.5 support and the ik_llama.cpp optimizations.

```bash
cd /srv/ai/projects/ik_llama.cpp
git fetch origin pull/1288/head:pr-1288
git checkout pr-1288
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
# Verify: find src/ -name "*.cpp" | xargs grep -l "qwen35\|ssm_alpha" 2>/dev/null
# Expected output includes: src/llama-delta-net.cpp, src/llama-arch.cpp, src/llama-build-context.cpp
```

---

**Finding 4: ik_llama.cpp pr-1288 flag changes vs assumed command.**

Running the originally-assumed command (`-fa -fmoe`) produces `error: invalid parameter for argument: -fa`. Inspecting `--help` reveals:

- `-fa` now requires a value: `-fa on|off|auto` — but it's on by default, so omit entirely
- `-fmoe` is gone — fused MoE is **on by default**, disable with `-no-fmoe`
- `--cpu-moe` exists as a clean alternative to the `-ot "blk\..*\.ffn_(gate|up|down)_exps\.weight=CPU"` regex

**Correct launch command (confirmed working):**
```bash
/srv/ai/projects/ik_llama.cpp/build/bin/llama-server \
  -m /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf \
  -ngl 999 \
  --cpu-moe \
  --no-mmap \
  -b 4096 -ub 2048 \
  -t $(nproc) \
  -c 16384 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --jinja \
  --host 0.0.0.0 --port 8002
```

Server starts successfully. Both 5090s recognized (32077 MiB + 32110 MiB GDDR7).

---

**Finding 5: Convergence performance baseline measured.**

Two measurements from first live run (32 threads, ctx=16384, --no-mmap, --cpu-moe):

```
Run 1 (469 token prompt, 308 token generation):
  prompt eval:  925.75ms / 22 tokens = 23.76 t/s   [warmup tokens]
  eval:       22692.08ms / 308 tokens = 13.57 t/s   [generation]

Run 2 (469 token prompt, 4631 token generation — full thinking + answer):
  prompt eval:  7731.05ms / 469 tokens = 60.66 t/s
  eval:       352267.07ms / 4631 tokens = 13.15 t/s

Run 3 (2348 token prompt, 3685 token generation):
  prompt eval: 14772.99ms / 2348 tokens = 158.94 t/s
  eval:       280332.58ms / 3685 tokens = 13.15 t/s
```

**Analysis:**
- Generation (~13.15 t/s) is bottlenecked by DDR5 bandwidth reading MoE expert weights. Per-token RAM read: ~10 experts × 3 matrices × 60 layers × ~1.28MB per matrix at IQ2_M ≈ 2.3GB/token. Actual bandwidth ~83 GB/s → theoretical ceiling ~36 t/s. Measured 13 t/s is ~36% of ceiling, gap explained by NUMA effects, thread coordination, expert routing compute.
- Prompt processing scales with batch size: 23.76 → 60.66 → 158.94 t/s as batch grows. This is the expected behavior for RAM-bandwidth-bound MoE inference — larger batches amortize the bandwidth cost across more tokens.
- GPU VRAM barely consumed — only attention/norm/embedding layers on GPU (~8-12GB of 64GB available). This leaves significant headroom for partial expert offload experiments (T_CV3).

**Operator note:** "i think we can do better than the parameters we used" — thread count (32) and GPU layer distribution are not yet optimized. T_CV2 (thread sweep) and T_CV3 (partial GPU offload) queued.

---

**Finding 6: vLLM sleep does not apply to Convergence.**

This was clarified definitively. vLLM sleep moves weights between VRAM and RAM — it's a GPU memory management primitive. Convergence doesn't live in VRAM at all (its weights are in RAM via `--cpu-moe`). The two systems are completely independent:

- vLLM sleep frees VRAM so Convergence's attention layers can use it
- Convergence is started/stopped independently as an ik_llama.cpp process
- No sleep/wake coordination needed between vLLM and ik_llama.cpp

The original "cold-start Convergence via vLLM sleep" framing was wrong. The correct framing: sleep Arclight (frees VRAM for Convergence attention layers), start Convergence as a separate process, run it, kill it, wake Arclight.

---

**Finding 7: Arclight thinker alternatives.**

Qwen3.5-27B confirmed viable only for TP=1 (MambaSpec blocks kvcached Phase B per R11). For a thinker that could work with kvcached shared-pool, need a non-Mamba model. Gemma4-31B identified as strong candidate:
- Dense (no Mamba), ~16GB AWQ
- kvcached-compatible (pure Transformer = FullAttentionSpec)
- Could potentially enable both models on one GPU with elastic KV pool
- Strong benchmarks: GPQA 84.3%, AIME 2026 89.2%, Arena Elo 1452

Both Qwen3.5-27B and Gemma4-31B remain viable paths. T2.3b added to test Gemma4-31B.

---

**Decisions updated this cycle:**
- `DECISIONS.md`: Convergence tier settled (model, engine, command, performance baseline, tier naming)
- `DECISIONS.md`: Gemma4-31B added as PROVISIONAL Arclight thinker candidate
- `ARCHITECTURE.md`: Full rewrite with Convergence tier, tier naming, deployment topology
- `TESTING_QUEUE.md`: T_CV1, T_CV2, T_CV3 added (Convergence benchmarks); T2.3b added (Gemma4)
- Benchmark script: `benchmarks/bench_convergence.sh` created

**Tests queued for next cycle:**
- T_CV1: Convergence cold-start timing (measure NVMe load time)
- T_CV2: Convergence thread count optimization sweep (8, 16, 24, 32)
- T_CV3: Convergence partial GPU expert offload (first N layers on GPU)
- T2.3b: Gemma4-31B as Arclight thinker candidate

**Open from this research:** Convergence startup time not measured — this was done mid-session and timing was not captured. Run T_CV1 as first priority in the next testing cycle.

---

### R10 — April 18 2026 — Qwen3.6 Shootout & Infrastructure Stabilization
+
+**Triggered by:** T2.5 (Qwen3.6-35B-A3B) shootout FAIL on first attempt (0/30 no_call).
+
+**Research findings:**
+
+1. **Reasoning Field Mismatch (delta.reasoning):** vLLM v0.19.0+ uses the field name `reasoning` in the delta stream for models using the `reasoning-parser`. Our BenchClient was looking for `reasoning_content`. Fixed in `lib/client.py` to capture both.
+
+2. **Parser Stack for Qwen3.6 Thinking Models:**
+   - `--tool-call-parser qwen3_coder` + `--reasoning-parser qwen3` is mandatory.
+   - `--enable-auto-tool-choice` is the critical missing piece from earlier failed runs; it forces the correct system instructions for tool-emission after reasoning blocks.
+   - `hermes` parser is incompatible with the reasoning-parser as it expects raw XML which the reasoning-parser peels off into the reasoning field.
+
+3. **Thinking-Limit Saturation:** Quality tasks fail at 1024 tokens because thinking models exhaust their budget on internal reasoning. Raised default `max_tokens` to 4096 across all quality benchmarks.
+
+4. **T2.5 Outcome (PASS ✓):**
+   - Tool Pass Rate: 96.7% (29/30).
+   - Quality Completion: 100% (10/10).
+   - Performance: 237.1 t/s (only 5.5% regression vs 30B baseline).
+   - Qwen3.6-35B-A3B is the new coder candidate of record.
+
+**Decisions updated:** Qwen3.6-35B DECISIONS entries updated. T2.5 marked PASS in TESTING_QUEUE. `lib/client.py` and `bench.py` infrastructure fixes verified.
+
+**Tests queued for next cycle:** T1.5 (kvcached spike) or T3.X (MTP/sm_120 overhead).
+
+---
+
+### R9 — April 18 2026 — T2.1b wrong code path; actual bug in unseen helpers (lines 1–394)
+
**Triggered by:** T2.1b FAIL. Operator ran two diagnostics (`sed -n "395,445p"` and `sed -n "443,600p"`), giving us lines 395–504 (the complete second half of the 504-line file).

**Research findings:**

1. **All 504 lines of `glm4_moe_tool_parser.py` reviewed — parser is clean.** Three diagnostic dumps (lines 1–200, 200–395, 395–504) cover the complete file. Every function is correct:
   - `extract_tool_calls_streaming` — clean. Calls `_build_args_json_so_far` (returns str), `_compute_args_diff` (diffs str correctly).
   - `_build_args_json_so_far` — handles both complete and partial arg states. Returns string in all code paths. Correctly handles 1-arg and 2-arg via `func_arg_regex.findall`.
   - `_extract_tool_call_regions`, `_extract_tool_name_from_region`, `_extract_content` — clean.
   - `extract_tool_calls` (non-streaming) — has broad `except Exception` handler, cannot crash the server.
   - `__init__`, `_is_string_type`, `_deserialize`, `_json_escape_string_content`, `_tools_enabled`, `adjust_request` — all clean.
   - No dict/string type confusion anywhere. PR #37385 described variables (`args_dict`, `full_args_str`) that don't exist in this build — it targeted an older, simpler parser.

2. **The crash is in EngineCore (pid=188), not in the tool parser (APIServer pid=1).** The Task 02 raw result contains: `"error": "EngineCore encountered an issue. See stack trace (above) for the root cause."` — vLLM V1's exact error string for EngineCore subprocess death. The APIServer (where the tool parser runs) and EngineCore are separate processes. An exception in the parser cannot produce this error.

3. **TRITON_MLA PIECEWISE CUDA graph is the likely crash vector.** Startup log shows:
   ```
   WARNING: CUDAGraphMode.FULL_AND_PIECEWISE is not supported with TritonMLABackend
   (support: AttentionCGSupport.NEVER); setting cudagraph_mode=PIECEWISE
   ```
   PIECEWISE mode handles different batch sizes with/without graphs at a boundary. Task 01 generates ~20–30 tokens (1-arg tool call), Task 02 generates ~40–60 tokens (2-arg tool call). If the EngineCore's PIECEWISE path has an instability at certain decode lengths, the longer Task 02 generation hits it while Task 01 doesn't. The actual traceback is in container stderr (not captured by the bench script).

4. **T2.1b is cancelled.** The "patch the streaming parser" approach was based on an incorrect root cause. Patching `glm4_moe_tool_parser.py` cannot fix a crash in the EngineCore subprocess. The fix requires either: (a) a vLLM update that stabilizes TRITON_MLA on Blackwell in PIECEWISE mode, or (b) a workaround that avoids the crash condition (e.g., forcing eager mode, different graph capture settings).

5. **GLM-4.7-Flash moves to cold storage — "wait for upstream."** The crash is in vLLM's engine/model execution layer, not in anything we can patch in a Containerfile RUN layer. The correct action is to monitor vLLM releases for `Glm4MoeLite` + TRITON_MLA + Blackwell fixes.

**Decisions updated:** All GLM-4.7-Flash DECISIONS entries updated. T2.1b CANCELLED in TESTING_QUEUE. `TESTING_QUEUE.md` status updated.

**Tests queued for next cycle:** T2.5 (Qwen3.6-35B-A3B coder shootout) — no dependencies, ready to run.

---

### R8 — April 18 2026 — GLM-4.7-Flash tool crash root cause

**Triggered by:** T2.1 wall (EngineDeadError on Task 02 / 2-arg tool calls). Research question: is the V1 engine the blocker, or is the parser the issue?

**Research findings:**

1. **V1 engine is not the fundamental blocker.** The EngineDeadError is the symptom of an unhandled exception in `glm4_moe_tool_parser.py`'s streaming path propagating through the V1 EngineCore multiprocess boundary. When the parser crashes in the EngineCore subprocess, V1 raises EngineDeadError for all subsequent requests.

2. **The streaming parser has a specific unfixed bug (PR #37385).** The streaming `extract_tool_calls_streaming` path stores `prev_tool_call_arr[index]["arguments"]` as a Python dict (`args_dict`) when a new tool call is first registered. Downstream finalization code that treats this value as a string crashes with TypeError. This bug is in our build (`v0.19.1rc1.dev391`). PR #37385 fixes it by storing `full_args_str` instead.

3. **PR #37386 (merged v0.18.0) fixed a different bug** in the non-streaming path: non-greedy regex `.*?` in `func_arg_regex` failed to capture multiple argument pairs. This fix IS in our build. The non-streaming path likely works correctly for Task 02 — confirmed by the fact that the crash takes 14s (the model generates full output before the streaming parser crashes on finalization).

4. **Why Task 01 works and Task 02 crashes:** Task 01 has 1 argument; the streaming finalization code path that misuses `prev_tool_call_arr["arguments"]` as a string may be bypassed for single-arg calls. Task 02 has 2 arguments; the second arg iteration hits the dict-as-string crash.

5. **Quick diagnostic available:** `curl` Task 02 with `"stream": false` against the live endpoint. If it succeeds, the non-streaming path is confirmed clean and the streaming path is the sole crash vector — which directly points to PR #37385.

6. **`VLLM_ENABLE_V1_MULTIPROCESSING=0`** keeps V1 in a single process. Parser exceptions become per-request recoverable instead of server-fatal. This is a useful safety net for the transition period but does not fix the underlying parser bug.

**Decisions updated:**
- `DECISIONS.md` SETTLED: "GLM-4.7-Flash EngineDeadError root cause is `glm4_moe_tool_parser.py` streaming bug (PR #37385), not V1 architecture. Fix is a one-line patch in Containerfile."
- `DECISIONS.md` updated: GLM-4.7-Flash DECISIONS entry revised to reflect the known fix path.
- `TESTING_QUEUE.md`: T2.1 status updated to BLOCKED (parser bug). Added T2.1b (re-test with streaming parser patch).

**Tests queued for the next testing cycle:**
- T2.1b: rebuild vllm-glm47 with PR #37385 one-line patch, rerun tool sanity, expect ≥2/3 pass.
- T2.5: Qwen3.6-35B-A3B shootout — independent, no deps on GLM fix.

**Open from research:** none — fix path is clear.

### R7 — April 18 2026 — architectural doubts + Qwen3.6

**Triggered by:** operator raised four questions after the T1.3 pass:
1. Qwen3.6 lineup (just released) — candidacy as higher-quality endpoint.
2. Is TP=1-per-GPU really the only way? Can vLLM engines be coordinated, or two models merged into one process?
3. KV cache: per-role distribution + dynamic resizing.
4. Reducing the ~4 GiB Sleep-Mode residual to revive TP=2.
5. (Implicit) Behemoth archetype diversity — context-rich vs knowledge-rich.

**Research findings:**

1. **Qwen3.6-35B-A3B is a genuine coder candidate.** Apache 2.0, released 2026-04-14. 35B total / 3B active MoE. 262k native context (extensible to 1M via YaRN). Thinking-by-default (can be disabled via `enable_thinking: False` in tokenizer or request). Tool-call parser `qwen3_coder` per model card. AWQ published as `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` — the same publisher that worked for Qwen3-Next-80B. Weight size (~22 GiB) is the same class as the old `qwen35_35b_a3b_awq` REVIEW entry, but this is a fresh generation that supersedes it as the 35B-A3B candidate of record. Qwen3.6-Plus is proprietary API-only (1M context), not relevant for local hosting.

2. **Multi-model in a single vLLM process is not supported.** Verified upstream: open feature request, no ETA. Community workarounds are separate instances behind an nginx router, or third-party `llmux` for zero-reload switching. This closes path "merge two models into one vLLM process."

3. **`kvcached` is a real third path for concurrent GPU sharing.** `ovg-project/kvcached` provides a virtualized elastic KV cache that decouples GPU virtual addressing from physical memory, allowing multiple vLLM instances to share a GPU KV pool dynamically. Supports MHA/GQA/MLA. Explicitly tested with vLLM 0.19.0 (our exact version). Red Hat endorsement, v0.1.5 active maintenance. This operates at the memory layer, not the CUDA context layer, so it may sidestep the context time-slicing that killed naive concurrent-TP=2 in T1.2 — without requiring MPS or root. Queued as T1.5 spike.

4. **The ~4 GiB Sleep-Mode residual is a design floor at level=1.** Level 1 deliberately preserves the caching allocator instance, captured CUDA graphs, JIT-compiled kernels, and process state. This is what buys <1s wake. Cannot be shrunk without breaking the wake-time guarantee. Level 2 offloads more but has the gibberish-on-wake bug we already documented. `wake_up(tags=["weights"])` exists for selective wake granularity but not sleep footprint. Conclusion: to "make TP=2 viable again," change the sharing model (kvcached), not the residual.

5. **Behemoth diversity is a meaningful axis.** The behemoth slot is on-demand and tolerates archetype diversity (different models for different escalation types). Two candidate archetypes: context-rich mid-large (50–70B with 256k+ context) for long-document/repo-wide work, vs knowledge-rich (Qwen3-Coder-Next-80B-A3B, currently settled). Reserved as a standing design item T2.6 to prompt candidate scouting without committing to a test yet.

**Decisions updated:**
- `DECISIONS.md` SETTLED: "Multi-model in a single vLLM process is not supported upstream — do not plan around it."
- `DECISIONS.md` SETTLED: "Sleep Mode level=1 ~4 GiB residual is a design floor, not a tunable."
- `DECISIONS.md` PROVISIONAL: "`kvcached` as a third concurrent-GPU sharing path — test in T1.5."
- `DECISIONS.md` SUPERSEDED: `qwen35_35b_a3b_awq` as the 35B-A3B candidate-of-record is replaced by `qwen36_35b_a3b_awq` (Qwen3.6, Apache 2.0, fresh generation).
- `TESTING_QUEUE.md`: added **T1.5** (kvcached spike), **T2.5** (Qwen3.6-35B-A3B shootout), **T2.6** (behemoth archetype scouting — design item).
- `config/models.yaml`: added `qwen36_35b_a3b_awq` as CANDIDATE coder; old `qwen35_35b_a3b_awq` REVIEW entry annotated as superseded by Qwen3.6.

**Tests queued for the next testing cycle:**
- T2.1 (re-test with `vllm-glm47` image) — verify MLA and tool-calling fix.
- T2.5 (Qwen3.6-35B-A3B shootout) — solo-TP=1 ready.

**Findings from testing (R7 cycle):**

1. **T1.4 (Thinker th03) — [FAIL]**: Increasing `max_tokens` to 16384 did not fix the empty-output issue in reasoning tasks. The thinker exhausts its budget in a reasoning loop. Architecture-heavy tasks must move to the coder or behemoth tiers.
- [x] **R7: GLM-4.7-Flash Stabilization (2026-04-18) — [INCONCLUSIVE (Tool-Broken)]**
    - **Hypothesis (revised)**: MLA path is actually active in all runs including the standard image — the original 60 KB pass threshold was wrong for this model. Confirmed via bench.log: all runs show `Using TRITON_MLA attention backend`.
    - **Results**:
        - **MLA Active (all runs)**: TRITON_MLA backend confirmed in bench.log from the first run onward. The `cu130-nightly` + git-transformers custom image resolved `Glm4MoeLiteForCausalLM` architecture name but did not change the attention backend used. Measured KV footprint ~129 KB/token reflects MLA (47 layers × kv_lora_rank=512 → ~94 KB base) plus CUDA graph overhead, not GQA (~380 KB for this model's num_heads=16 × head_dim=128 × 47 layers × TP=2 → ~376 KB). The original "MLA ≈54 KB, GQA ≈98 KB" reference values in TESTING_QUEUE.md were wrong for this specific model.
        - **V1 Constraint**: vLLM Nightly forces V1 engine for Glm4MoeLiteForCausalLM and ignores all V0 legacy flags (`VLLM_V1=0`, `VLLM_USE_V1=0`, `VLLM_V1_ENABLED=0`, `VLLM_USE_V1_ENGINE=0`, `VLLM_ENGINE_ITERATOR_SOURCE=LEGACY` — all reported as "Unknown" env vars).
        - **Tool Blocker**: V1 crashes (EngineDeadError) on complex tool schemas (Tasks 02, 03). Task 01 (simple) passes at 44.7 t/s. Tool sanity locked at 33% across all run variants.
    - **Conclusion**: GLM-4.7-Flash MLA is active and functional. Model is unusable for tool-calling roles on current vLLM nightly due to V1 instability. T2.1 verdict: INCONCLUSIVE (MLA confirmed / tool-broken).

**Open from research:** none — fixes identified: ensure `config.json` contains MLA dimensions (192/64/512) and re-test on clean `vllm-glm47` image stack. (Note: the previous "one-line source patch" advice is now deprecated as a legacy workaround for older images).

### R6 — April 18 2026 — T1.2 hand-back, TP=1-per-GPU pivot

**Triggered by:** Claude Code hand-back from T1.2. Test reported FAIL: concurrent TP=2 processes sharing GPUs collapsed to 4.2 t/s each (~2% of isolate throughput).

**Research findings:**
1. **GPU-wide CUDA context time-slicing:** Without MPS, two CUDA processes on the same GPU time-slice at the context level. For TP=2, each decode step launches many kernels, generating massive context-switch amplification. The ~50x degradation is consistent with NVIDIA warnings about naive multi-process sharing.
2. **MPS skipped:** MPS requires a root daemon on the host which breaks our rootless podman invariant. sm_120 support is also unverified outside datacenter environments.
3. **TP=1-per-GPU is preferred solution:** Tying one engine process exactly to one GPU completely eliminates contention. Coder TPS is actually *higher* at TP=1 on one 5090 (251 t/s) than at TP=2 (212 t/s) due to missing allreduce overhead. Thinker degrades from 106 t/s to ~76 t/s, but this is an acceptable tradeoff for complete concurrent stability.

**Decisions updated:**
- `DECISIONS.md` SETTLED: "Two vLLM processes on shared GPUs without MPS collapse to ~2% throughput. Root cause: GPU-wide CUDA context time-slicing, not just NCCL serialization. Do not retest without MPS."
- `DECISIONS.md` SETTLED: "MPS skipped — requires root daemon (breaks rootless invariant), sm_120 support unverified, and TP=1-per-GPU avoids the problem entirely."
- `DECISIONS.md` SUPERSEDED: the "two concurrent TP=2 processes sharing GPUs" assumption from R4.
- `ARCHITECTURE.md` updated: Pivot to Coder TP=1 on GPU0, Thinker TP=1 on GPU1, Behemoth TP=2 asleep. Swapping to Behemoth requires sleeping both hot models.
- `TESTING_QUEUE.md`: T1.2 DONE (FAIL), superseded by T1.2a (TP=1-per-GPU). Added conditional T1.2b (sleep-mode sequential) and T1.2c (MPS). T1.3 updated to reflect Behemoth borrowing both GPUs.

**Tests queued for the next testing cycle:** T1.2a (TP=1-per-GPU concurrent eval).

**Open from research:** none — queue is ready for testing.

### T1.3 testing cycle — April 18 2026

**Triggered by:** T1.2a PASS. Behemoth viability (T1.3) was next in queue.

**Testing findings:**

1. **Wrong HF repo (`cpatonn/`):** First three runs (`105618Z`–`112638Z`) used `cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` — failed with HTTP 401 Unauthorized. Correct repo is `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit`. `config/models.yaml` hf_repo corrected.

2. **`gpu-memory-utilization 0.85` is insufficient:** With the correct repo, the initial planned gpu-mem (0.85) caused OOM during CUDA graph capture. Fix: raise to 0.95. Also required: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` env var to allow estimating graph memory without allocating it; without this, vLLM OOMs before graphs are captured.

3. **Parser trial sequence (all at 0.95 gpu-mem, correct repo):**
   - `--tool-call-parser qwen3_coder` → 9/9 `no_call`. Parser does not emit tool calls for this model's output format.
   - `--tool-call-parser qwen3_xml` → 9/9 `wrong_tool`. Format parsed but tool name/schema mismatch.
   - `--tool-call-parser qwen3_xml --reasoning-parser qwen3` → 9/9 `exception`. Reasoning parser causes hard exceptions.
   - `--tool-call-parser hermes --reasoning-parser qwen3` → 9/9 `no_call`. Reasoning parser intercepts XML-tagged content (including tool calls) into the `reasoning` field before the tool-call parser sees it, starving it entirely.
   - `--tool-call-parser hermes` (no reasoning parser) → 9/9 PASS. 100% tool-call reliability.

4. **Working config:** `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit`, TP=2, `--gpu-memory-utilization 0.95`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, `--tool-call-parser hermes --enable-auto-tool-choice`. No `--reasoning-parser`.

5. **Performance:** 189.5 t/s seq=1 decode, 610 t/s aggregate at seq=4, 13007 t/s prefill at 32k context. Verdict: PASS.

**Decisions updated:**
- `DECISIONS.md` SETTLED: "Behemoth (80B A3B MoE) on TP=2 is extremely viable — T1.3 PASS."
- `DECISIONS.md` SETTLED: `--tool-call-parser hermes` is required; `--reasoning-parser qwen3` must NOT be added (hard failures at all parser combinations).
- `config/models.yaml`: hf_repo corrected to `cyankiwi/`, flags updated to `hermes`, gpu-mem-util 0.95, added `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` to env.
- `ARCHITECTURE.md`: Behemoth deployment details updated; Known unknowns items 1 and 2 settled.
- `TESTING_QUEUE.md`: T1.3 marked DONE (PASS ✓), result block added, procedure corrected.
- `infra/scripts/deploy.sh`: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` passthrough added.

**Tests queued for next testing cycle:** T2.1 (GLM-4.7-Flash MLA verification) or T1.4 (thinker token budget fix) — both are independent and cheap.

**Open from research:** none — queue is ready for testing.

### R5 — April 17 2026 (evening)

**Triggered by:** Claude Code hand-back from T1.1. Test reported FAIL: VRAM freed 2.1% vs 80% threshold, RAM flat throughout, wake in 85 ms. Narrative: "does vLLM 0.19 sleep mode actually implement weight offload for AWQ/MoE on Blackwell, or is this a rootless podman CUDA VMM restriction?"

**Research findings:**

1. **Proximate cause: missing `--enable-sleep-mode` flag.** The T1.1 script sets `VLLM_SERVER_DEV_MODE=1` (which exposes the sleep/wake routes) but does not pass `--enable-sleep-mode` to `vllm serve` (which is what configures the engine to use `CuMemAllocator` and reserves the "weights" pool). Per official vLLM docs, **both** are required. Without the serve flag, `/sleep` is a control-plane no-op: `is_sleeping` toggles but there is no allocator context that can release the weight memory. This explains every observed symptom (API works, `is_sleeping` transitions, wake in 85 ms, VRAM/RAM unchanged). No need to invoke rootless podman or sm_120 issues to explain the result.

2. **Secondary concern, relevant for what happens after the rerun:** vLLM issue #32714 ("Sleep is broken since 0.14.0") reports partial memory freeing on v0.14+ compared to v0.13. RFC #34303 (Feb 2026) cites the bug as still applicable: "vLLM's existing sleep mode (--enable-sleep-mode) is broken since v0.14.0". Issue is marked Closed but fix-version unclear from search alone — we're on 0.19.0 which is post-0.14.0. A rerun with the flag is the cheapest test: it either frees ~80%+ (past both bugs) or frees ~30% (regression applies to 0.19 too).

3. **Level=2 is actively harmful for our use case.** Bug #29341 (Nov 2025, H100): wake from level=2 produces gibberish. Docs note level=2 requires manual `reload_weights` + `reset_prefix_cache` on wake. We have 192 GB DDR5, so level=1 (weights → CPU RAM) is strictly better for us. Updated `DECISIONS.md`.

4. **Blackwell-specific caveat carried forward:** Bug #21336 (July 2025) reported sleep-mode crashes on RTX PRO 6000 (sm_120 workstation Blackwell) + vLLM 0.9.2 + TP=2 + GPTQ-Marlin. Our hardware is the same arch family, same TP placement, AWQ-Marlin (not GPTQ-Marlin — related but different kernel). Status on 0.19.0 is unknown. If the rerun crashes at startup with the flag added, this is the suspect. Escalation path defined in `TESTING_QUEUE.md` T1.1b.

**Decisions updated:**
- `DECISIONS.md` SETTLED: "Sleep mode requires BOTH `VLLM_SERVER_DEV_MODE=1` env **and** `--enable-sleep-mode` serve flag."
- `DECISIONS.md` SETTLED: "Use Sleep Mode level=1 only; level=2 has known output-corruption bugs."
- `TESTING_QUEUE.md` T1.1 reshaped into T1.1 (rerun with fix) + T1.1a (fallback: pin earlier vLLM if regression bites) + T1.1b (Blackwell crash fallback).
- `ARCHITECTURE.md`: three-tier design marked "gated on T1.1 rerun"; no structural changes yet.
- `benchmarks/queue/T1.1_sleep_mode.sh`: one-line fix queued (add `--enable-sleep-mode` to the deploy extra args).

**Tests queued for the next testing cycle:** T1.1 rerun (highest priority). T1.1a / T1.1b are conditional.

**Open from research:** none — queue is ready for testing.

### R4 — April 17 2026 (afternoon)

**Triggered by:** operator conversation about architecture validity, PCIe TP=2 analysis, and discovery of Sleep Mode + OpenCode native multi-endpoint routing.

**Research inputs:**
- vLLM Sleep Mode docs + blog post (the canonical Qwen3-235B ↔ Qwen3-Coder-30B swap example directly applies).
- OpenCode v1.3+ multi-endpoint behavior (verified by operator-provided technical summary).
- GLM-4.7-Flash release: model card, AWQ quants by cyankiwi/cpatonn, MLA arch-convertor bug, MTP regression reports on B200.
- GLM-4.5-Air availability + AWQ TP=2 constraints.
- Qwen3-Coder-Next AWQ publication (superseded the old "GGUF only" note).
- PCIe 5.0 x8/x8 allreduce math for A3B MoE vs dense-70B sync-overhead mechanism.
- KV cache math: MLA vs GQA real-world ratio (~1.8×, not 10×).

**Decisions updated:**
- `DECISIONS.md` migrated and tiered (SETTLED / PROVISIONAL / SUPERSEDED).
- Architecture moved from single-GPU-per-model to TP=2-as-default. Three-tier with Sleep Mode standby.
- LiteLLM demoted from router to optional observability layer.
- Phase-based plan replaced with item-queue (T1.x, T2.x, ...) in `TESTING_QUEUE.md`.

**Tests queued for R5:** T1.1, T1.2, T1.3, T1.4 (architecture-defining). T2.1 (MLA verification) is cheap and can be interleaved.

### R3 — April 15 2026 (captured from `PHASE2_RESULTS.md`)

Phase 2 coder + thinker shootouts against the single-GPU assumption. Results stand for what they measured (251 t/s coder, 76 t/s thinker, quality scoring), but architectural context has shifted.

### R2, R1 — pre-cycle-log

Phase 0/1 work: chat template verification, vLLM vs SGLang throughput comparisons, prefix-cache evaluations. Relevant conclusion preserved: vLLM is our primary engine; SGLang weight-loader bug for Qwen3.5 MoE AWQ is permanent until upstream patch.

---

## Open from testing

### From T2.1, 2026-04-18 — RESEARCHED (R8 cycle)

**What happened:** 8 T2.1 runs. MLA confirmed active. Tool sanity locked at 33%: Task 01 (`read_file`, 1 arg) passes at ~44 t/s. Task 02 (`write_file`, 2 args) crashes after 14s with EngineDeadError. Task 03 gets "Connection error" (engine dead from Task 02 crash).

**Root cause identified (R8):** The crash is in `glm4_moe_tool_parser.py`'s **streaming** path, not a fundamental V1 incompatibility. The bench client uses `stream=True` always. Two separate bugs are relevant:

1. **PR #37386** (merged v0.18.0, March 2026) — fixed non-greedy `.*?` in `func_arg_regex` that failed to capture multiple argument pairs in the non-streaming path. This fix IS in our build (`v0.19.1rc1.dev391` is post-v0.18.0). So the non-streaming path should handle 2-arg tools correctly.

2. **PR #37385** (open, awaiting review, NOT in our build) — the streaming path stores `prev_tool_call_arr[index]["arguments"]` as a Python dict (`args_dict`) instead of a JSON string (`full_args_str`). When the finalization code uses this as a string (e.g., for length comparison or concatenation), it crashes with TypeError. For 1-arg tools, this specific code path may be reached safely; for 2-arg tools, the additional streaming iteration hits the crash. This PR was not merged as of the T2.1 test date.

**V1 engine is not the fundamental blocker** — it cannot be disabled for this architecture, but that is a red herring. The EngineDeadError is the symptom of the parser crashing in the V1 EngineCore subprocess, not a V1 architectural incompatibility.

**Fix path (see DECISIONS.md and T2.1 updated procedure):**
- **Step 1** (diagnostic, host only): Confirm PR #37385 bug is in the image — `podman run --rm vllm-glm47 grep -rn "arguments.*args_dict\|args_dict.*arguments" /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/`
- **Step 2a** (quick non-streaming test, host only): `curl` Task 02 with `"stream": false` to confirm the non-streaming path works — if so, the streaming parser is definitively the crash point.
- **Step 2b** (fix): Patch `glm4_moe_tool_parser.py` in `infra/Containerfile.vllm_glm47` to change `"arguments": args_dict` → `"arguments": full_args_str` at the point where a new tool call entry is first added to `prev_tool_call_arr`. Then rebuild (`--rebuild` flag on the bench script) and rerun T2.1.
- **Step 3** (safety net): Add `VLLM_ENABLE_V1_MULTIPROCESSING=0` to the container env — keeps V1 in single-process mode so parser exceptions don't kill the entire engine; failures become per-request recoverable instead of fatal.

### From T2.3b, 2026-04-24 — thinker question still open; Gemma4 redirected to coder

**What happened:** T2.3b ran the Phase 2.2 thinker quality suite (8 tasks) against Gemma4-31B-it-AWQ at 70.3 t/s seq=1. Mean 4.0/5 — matches Qwen3.5-27B baseline. Task completion 100% (including non-empty output on th03 where Qwen3.5-27B emits empty). Three tasks scored ≤3 due to "Surface-Level Reasoning" pathologies: th02 (algorithm design), th03 (LLM-specific architecture), th05 (distributed consistency edge cases).

**Why this needs research, not another test:** Gemma4's failure profile is task-type-specific, not uniformly inferior. Its dense architecture, no-MambaSpec, and strong 5/8 task profile make it worth evaluating in the coder role rather than discarding. This is a redirect decision, not a simple FAIL verdict.

**Open questions:**
1. Should Gemma4 be queued as a coder candidate (T2.3c)? It's dense, kvcached-compatible, strong benchmarks, 100% completion.
2. Is there a better thinker candidate to test? Qwen3.5-27B remains the provisional baseline.
3. Does the thinker question block any current work? (kvcached T1.5 Phase B remains deferred until a kvcached-compatible thinker is settled.)

**Suggested direction:** Queue T2.3c (Gemma4 as coder) and proceed with Convergence benchmarks (T_CV1–T_CV3) in parallel. Thinker selection is not blocking any current test. Revisit thinker after coder evaluation.

**Result dirs:** `results/T2.3b_arclight_thinker_gemma4_31b_candidate_20260423T163106Z/`

---

### From T2.1b, 2026-04-18 — RESOLVED (R9 cycle)

**What happened:** T2.1b sed patch was a no-op (wrong variable name). After three diagnostic dumps covering all 504 lines of `glm4_moe_tool_parser.py`, the parser was confirmed clean. The crash is in EngineCore (not the parser). T2.1b CANCELLED. See R9 cycle log above.



<!-- Template for testing mode to fill in:

### From T1.X, date YYYY-MM-DD

**What happened:** short narrative.
**Relevant logs / result dirs:** paths under `results/`.
**Why this needs research, not another test:** the specific reason a parameter tweak is not enough.
**Suggested direction:** what the operator thinks the research should investigate.

-->

---

## How to update this file

**In research mode:** add a new dated entry under `## Cycle log`. Summarize inputs, decisions, and what got queued. Move anything that testing reported under `## Open from testing` into the new cycle entry once addressed, then clear that section.

**In testing mode:** only append to `## Open from testing`. Do not edit cycle log entries. Do not edit `## What we believe right now` — those changes require research validation.

**Handoff protocol:**
- When testing mode hits a wall → write to `## Open from testing` → stop → tell operator "research mode needed."
- When research mode finishes a cycle → write to `## Cycle log` → update downstream docs (`DECISIONS.md`, `TESTING_QUEUE.md`, `ARCHITECTURE.md`, `config/models.yaml`) → tell operator "testing mode ready, pick from T-items with all deps DONE."