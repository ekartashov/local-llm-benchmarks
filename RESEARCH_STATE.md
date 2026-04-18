# RESEARCH_STATE.md

Living document. What we currently believe, what is still open, and the log of research ↔ testing cycles.

**Current cycle:** R7 updated — T1.4 FAIL recorded. T2.1 REBUILD/PATCH pending.
**Current mode:** TESTING READY — pull next OPEN item from the queue (T2.1 re-test, T2.5).

---

## What we believe right now

### Known good, ready to deploy

- Qwen3-Coder-30B-A3B-AWQ on vLLM, single GPU: 251 t/s seq=1, 730 t/s aggregate at c=4. Tool calls reliable with `--tool-call-parser qwen3_coder --reasoning-parser qwen3`. Measured.
- Qwen3.5-27B-AWQ on vLLM, single GPU: 76 t/s, quality 4.0/5 on 8-task thinker suite. Needs `--max-num-seqs 1`. **Defect th03 remains**: Task T1.4 (2026-04-18) confirmed that even at `max_tokens=16384`, the model exhausts its budget in a reasoning loop. Architecture-heavy tasks must be routed to coder or behemoth.
- vLLM is our primary engine. vLLM launches cleanly with our rootless podman setup on Blackwell sm_120 at TP=2. AWQ-Marlin kernel path confirmed functional (T1.1 run loaded 18 GiB weights across TP=2 cleanly).
- Sleep Mode confirmed working end-to-end (T1.1 PASS 2026-04-17): `VLLM_SERVER_DEV_MODE=1` + `--enable-sleep-mode` frees 92.8% VRAM (59 → 4 GiB) in ~4s, wake in 0.9s, post-wake TPS 212.3 t/s (ratio 1.000). vLLM 0.19 reasoning-parser streaming field is `delta.reasoning`, not `delta.reasoning_content`.
- Qwen3-Coder-Next-80B-A3B-AWQ (behemoth) on vLLM TP=2: 189.5 t/s seq=1, 610 t/s aggregate at seq=4, 13007 t/s prefill@32k. Tool calls 100% reliable with `--tool-call-parser hermes` and **no** `--reasoning-parser`. Requires `--gpu-memory-utilization 0.95` and env `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`. HF repo: `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit`. Measured T1.3 (2026-04-18).

### Known bad / excluded

- Ollama, KTransformers, Devstral, DeepSeek-R1-32B, GLM-4.6/4.7 full, GLM-4.6-Air-doesn't-exist — see `DECISIONS.md` SETTLED.
- bf16 deployment — the quant we serve is AWQ-INT4, bf16 "does it fit" tests were misleading and have been discarded.
- vLLM Sleep Mode **level=2** — do not use. See `DECISIONS.md`; known to produce gibberish outputs on wake (bug #29341) and requires manual `reload_weights` + `reset_prefix_cache` after wake which is easy to get wrong. Use level=1 exclusively. We have 192 GB DDR5, there is no reason to prefer level=2 for us.

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
| Qwen3.6-35B-A3B (Apache 2.0, 2026-04-14) | Coder alternative | Yes: cyankiwi | 22 GiB weights tight at TP=1 (may need TP=2); thinking-by-default (disable flag needed); tool parser `qwen3_coder` per model card — test T2.5 |

---

## Cycle log

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
2. **T2.1 (GLM-4.7-Flash) — [FAIL]**: 
    - **Load failure**: vLLM 0.19/latest didn't recognize `glm4_moe_lite` architecture.
    - **MLA Bloat**: Initial run with `cu130-nightly` image reported **127.4 KB/token** KV cache. This confirms the MLA regression on Blackwell still persists in the nightly build.
    - **Tool Sanity**: Filter `'01 02 03'` failed due to literal string matching.
    - **Script Bug**: Shell expansion in the metrics heredoc caused syntax errors.

**Open from research:** none — fixes identified; re-test T2.1 queued with custom image patch.

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

### From T1.X, date YYYY-MM-DD

**What happened:** short narrative.
**Relevant logs / result dirs:** paths under `results/`.
**Why this needs research, not another test:** the specific reason a parameter tweak is not enough.
**Suggested direction:** what the operator thinks the research should investigate.

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