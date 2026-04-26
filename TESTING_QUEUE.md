# TESTING_QUEUE.md

The prioritized queue of open questions. Items have IDs, dependencies, and an explicit "what changes if this fails" field so we know whether a failure blocks or just narrows.

This replaces the old `phase0 → phase1 → ...` linear plan. Items can run in any order subject to their `deps`. Pick the highest-value item whose deps are satisfied and whose preconditions are met on the host.

## Legend

- **item_id** — matches `metrics.json::item_id` and the results directory name.
- **status** — `OPEN` (to do), `RUNNING` (test in progress), `DONE` (result recorded, see `RESEARCH_STATE.md`), `BLOCKED` (cannot proceed, see reason), `PUNTED` (deferred, see reason).
- **deps** — other item_ids that must be DONE first.
- **hand-back trigger** — conditions during the test that require switching to research mode before continuing.

Every item has a `what failure means` field. If the answer is "adjust one config value and rerun," it's a parameter sweep, not a research question.

---

## Tier 1 — architecture-defining (run these first)

These determine whether the three-tier architecture in `ARCHITECTURE.md` stands. A failure in any one of them forces an architecture rewrite.

### T1.1 — sleep_mode_operational_under_podman — DONE ✓

**Previous attempt:** FAIL on 2026-04-17. Root cause (R5 cycle): `--enable-sleep-mode` was missing from the `vllm serve` command. `VLLM_SERVER_DEV_MODE=1` exposes the HTTP routes, but `--enable-sleep-mode` is what makes the engine use `CuMemAllocator` with a "weights" pool — without it, `/sleep` is a no-op at the memory layer. Full writeup in `RESEARCH_STATE.md` R5 cycle and `DECISIONS.md`.

**Question:** With `--enable-sleep-mode` present, does vLLM 0.19.0 Sleep Mode level=1 actually free GPU VRAM and repopulate it on wake, under our rootless podman + TP=2 setup on Blackwell sm_120?

**Procedure:**
1. Start vLLM container for Qwen3-Coder-30B-A3B-AWQ, TP=2, `--gpu-memory-utilization 0.85`, env `VLLM_SERVER_DEV_MODE=1`, vllm args include **`--enable-sleep-mode`**.
2. Verify startup completes (watch for Blackwell crash pattern — see T1.1b below).
3. Run one completion, record TPS and VRAM total across both GPUs.
4. `POST /sleep?level=1` — measure time, record `is_sleeping: true`, record VRAM after sleep, record CPU RAM delta.
5. `POST /wake_up` — measure wake time.
6. Run another completion. Record TPS and TTFT vs pre-sleep.

**Pass:** freed VRAM ≥ 80%, wake ≤ 10s, post-wake TPS within 5% of pre-sleep, CPU RAM increases by approximately the weight size on sleep (~18 GiB for this model).

**What each outcome means — branch here:**
- **PASS** → three-tier architecture stands. Proceed to T1.2.
- **Freed VRAM in 20–60% band** → matches regression bug #32714 symptoms on vLLM v0.14+. Fork to **T1.1a**.
- **Container crashes at startup** with `--enable-sleep-mode` added → matches Blackwell sm_120 bug #21336 pattern. Fork to **T1.1b**.
- **Freed VRAM < 20%** (similar to previous run, where the flag was missing) → our interpretation of the flag requirement is wrong, or there is a rootless-podman / CUDA VMM interaction we haven't identified. **Hand back to research.**
- **Post-wake output is garbage** → do NOT interpret as level=2 (we're using level=1 here). Hand back to research with the full output sample.

**Deps:** none — highest priority, run first.

**Hand-back trigger:** any outcome except PASS, T1.1a's or T1.1b's continuation paths.

### T1.1a — sleep_mode_regression_workaround (CONDITIONAL)

**Triggered by:** T1.1 shows 20–60% freed (regression #32714 signature).

**Question:** Does pinning vLLM to a pre-regression version (0.13.0) fix the memory-free amount while preserving enough feature coverage for our candidate model list?

**Procedure:**
1. Build or pull vLLM 0.13.0 image. Record image digest.
2. Re-run T1.1 procedure against 0.13.0.
3. If 0.13.0 passes: check whether 0.13.0 can load the Tier-2 candidate models (GLM-4.7-Flash, GLM-4.5-Air, Qwen3-Coder-Next-AWQ). Each of these has published `requires_vllm` constraints — verify compatibility explicitly, don't assume.
4. If 0.13.0 cannot load our shortlist: try progressively newer versions (0.14.x, 0.15.x, ...) to find the smallest version that both frees memory correctly and supports the models we need.

**Pass:** some vLLM version in the 0.13.0–0.19.0 range both frees ≥80% VRAM on sleep AND loads all three candidate models without errors.

**What failure means:**
- No version in the range satisfies both → three-tier architecture fallback becomes the active plan. See T1.1c.
- 0.13.0 works but doesn't support GLM-4.7-Flash (requires ≥0.15) → we accept a narrower model shortlist OR run dual vLLM versions per role (complex, probably not worth it).

**Deps:** T1.1 outcome = regression signature.

**Hand-back trigger:** no viable version found; or a candidate model incompatibility surfaces that wasn't in `config/models.yaml`.

### T1.1b — sleep_mode_blackwell_crash_workaround (CONDITIONAL)

**Triggered by:** T1.1 container crashes at startup with `--enable-sleep-mode` added (but loaded fine without it — which we already confirmed in the R4 bench run).

**Question:** Is this bug #21336 on sm_120, and is there a flag combination or vLLM version where `--enable-sleep-mode` + AWQ-Marlin + TP=2 on sm_120 works without crashing?

**Procedure:**
1. Capture the exact crash log (full container stderr + NCCL/CUDA errors).
2. Compare signature to #21336 reports (custom_all_reduce disabled, P2P disabled on sm_120 — already present in our clean run).
3. Try toggles one at a time: `--disable-custom-all-reduce`, `--enforce-eager` (last resort, loses CUDA graphs), `VLLM_USE_V1=0` to force v0 engine.
4. If a toggle makes it load, rerun the memory-freeing measurement on that config.

**Pass:** some flag combination loads and frees ≥80% VRAM on sleep.

**What failure means:** sleep mode is unusable on our hardware+vLLM combination. Fall back to T1.1c (container stop/start for behemoth tier).

**Deps:** T1.1 outcome = startup crash with flag.

**Hand-back trigger:** all toggle combinations crash.

### T1.1c — podman_stop_start_fallback (CONDITIONAL, architecture rewrite path)

**Triggered by:** T1.1a and T1.1b both fail, or T1.1 hand-backs to research result in "sleep mode not viable on our stack" verdict.

**Question:** Does `podman stop` + `podman start` of a pre-warmed vLLM container meet a relaxed swap-time budget for the behemoth tier?

**Procedure:**
1. Start behemoth vLLM container, fully initialize (weights loaded, CUDA graphs captured, first warmup request served).
2. `podman stop <container>` — measure time to stop.
3. `podman start <container>` — measure time to ready (HTTP /health returns 200).
4. Verify the second startup uses page cache (weights should come from DRAM, not disk) — measure weight-load phase specifically.
5. Repeat 5× to get a stable distribution.

**Pass:** stop+start cycle ≤30s median, ≤45s p95, weights come from page cache (not cold-disk).

**What failure means:**
- Stop+start >60s → architecture collapses to two-tier (no behemoth tier). Update `ARCHITECTURE.md` accordingly.
- Weights don't page-cache (disk I/O on restart) → investigate `tmpfs` or explicit page-cache priming. Separate smaller research item.

**Architecture implication:** if T1.1c is the working path, update `ARCHITECTURE.md` to reflect:
- Coder + thinker stay resident (always hot, TP=2, ~40% gpu-mem-util each).
- Behemoth swap path is 30s not 1s.
- Behemoth is only invoked for explicit escalation (user `@behemoth` reference), never auto-woken.

**Deps:** T1.1a and T1.1b both concluded negatively.

**Hand-back trigger:** stop+start time so large the behemoth tier offers no value; or page-cache consistently misses.

---

### T1.2 — concurrent_two_vllm_processes_shared_gpus — DONE (FAIL ✗)

**Superseded by:** T1.2a. See `RESEARCH_STATE.md` R6 log.

---

### T1.2a — tp1_per_gpu_concurrent_decode — DONE (PASS ✓)

**Question:** If we isolate Coder to GPU0 at TP=1 and Thinker to GPU1 at TP=1, do both models hit acceptable concurrency and throughput?

**Result (2026-04-18):** PASS. Perfect 1.000 concurrent isolation.
- Coder (30B A3B) isolated: 251.0 t/s. Concurrent: 251.0 t/s.
- Thinker (27B dense) isolated: 76.5 t/s. Concurrent: 76.5 t/s.
- Prefill contention: ~0x (cache hit dominance).
Conclusion: TP=1-per-GPU cleanly solves the shared-GPU time-slicing issue. Coder single-GPU throughput is extremely healthy.

---

### T1.2b — sleep_mode_sequential_swap — CANCELLED

**Triggered by:** T1.2a fails (T1.2a passed so this is skipped).

---

### T1.2c — mps_eval — CANCELLED

**Triggered by:** T1.2a and T1.2b both fail (skipped).

---

### T1.3 — qwen3_coder_next_awq_tp2_viability — DONE (PASS ✓)

**Question:** Does `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` load and decode acceptably on TP=2 across our two 5090s? (Note: now decoupled from the hot-pair architecture, as Behemoth borrows both GPUs during wakeup).

**Result (2026-04-18):** PASS. 189.5 t/s seq=1 decode, 610 t/s aggregate at seq=4, 13007 t/s prefill at 32k context, 100% tool-call pass rate (9/9) with `--tool-call-parser hermes`.
Conclusion: Behemoth tier is fully viable. Three-tier architecture confirmed.

Parser trial history (failures before working config):
- `--tool-call-parser qwen3_coder` → 9/9 `no_call` (parser does not emit tool calls for this model)
- `--tool-call-parser qwen3_xml` → 9/9 `wrong_tool` (format mismatch)
- `--tool-call-parser qwen3_xml --reasoning-parser qwen3` → 9/9 `exception` (reasoning parser causes hard failure)
- `--tool-call-parser hermes --reasoning-parser qwen3` → 9/9 `no_call` (reasoning parser intercepts XML tool tags into the `reasoning` field)
- `--tool-call-parser hermes` (no reasoning parser) → 9/9 PASS

Also required for successful load: `--gpu-memory-utilization 0.95` (0.85 OOM'd during CUDA graph capture) and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.

**Procedure (as executed — for reproducibility):**
1. Deploy `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, TP=2, `--gpu-memory-utilization 0.95`, context 32768, `--tool-call-parser hermes --enable-auto-tool-choice`. No `--reasoning-parser`.
2. Measure decode TPS at seq=1 and concurrency=4.
3. Measure prefill throughput at 8k, 16k, 32k prompts.
4. Run all Phase 0 tool-call tasks (9 tasks) to verify tool-call parsing works on the Next variant.
5. Do NOT enable MTP spec decode. Measure first without; MTP is a separate item (T3.3).

**Pass (as behemoth):** decode TPS ≥ 40 t/s at seq=1, no prefill pathology, tool calling works.

**What failure means:**
- TPS below 40 → behemoth tier is not viable. Drop to two-tier architecture. Not a research block, a plan change.
- Crashes during load → hybrid attention / vLLM version mismatch. **Hand back to research** to check vLLM release notes.
- Tool-call parsing broken → separate fix path (parser or chat template). Track as a sub-item.

**Deps:** none (independent of T1.1 / T1.2a, though sequencing after T1.1 is cheaper because the vLLM image is the same)

**Hand-back trigger:** load failure, tool-call parser failure.

---

### T1.5 — kvcached_shared_gpu_kv_pool — PARTIAL ⚠ (coder PASS / thinker blocked, re-run after T2.3)

**Result (2026-04-19):**
- Phase A coder: PASS — 250.8 t/s (baseline 251.0). kvcached adds zero overhead for Transformer/MoE models.
- Phase A thinker: BLOCKED — Qwen3.5-27B-AWQ uses hybrid Mamba (SSM) layers. kvcached v0.1.5 raises `ValueError: got MambaSpec` at KV init. Thinker-specific, not a kvcached bug.
- Phase B: FAIL — thinker OOM'd during weight loading. kvcached virtual memory applies to KV cache pages only, not weights. Combined 18+14 GiB saturated the GPU before KV allocation.

kvcached is viable in principle; the Phase B block is entirely due to the current thinker model. **Re-run Phase B after T2.3 selects a non-Mamba thinker**, if combined weights fit under ~28 GiB.

**Original question:** Can `ovg-project/kvcached` enable two vLLM instances to cohabit a single GPU (or a pair, TP=2-on-both) with a shared, elastic KV-cache pool — without the CUDA context time-slicing collapse we hit in T1.2?

**Why this matters:** `kvcached` operates at the GPU memory layer (decouples virtual from physical addressing) rather than the CUDA context layer. If it works on our stack, it opens a third path between (a) the current TP=1-per-GPU isolation and (b) the failed naive concurrent TP=2. It could enable concurrent TP=2 without MPS/root, and make per-role KV budgets dynamically resizable — directly addressing the "distribute KV cache per model efficiently" concern.

**Procedure (spike — small, time-boxed):**
1. Pull `ghcr.io/ovg-project/kvcached-vllm:latest` (tagged `kvcached-v0.1.5-vllm-v0.19.0`). No build step — kvcached ships a pre-built vLLM image. Integration is via env vars: `ENABLE_KVCACHED=true` + `KVCACHED_AUTOPATCH=1`. Do NOT pass `--gpu-memory-utilization`; kvcached manages allocation automatically. Add `--no-enable-prefix-caching` for a clean measurement baseline.
2. Phase A (baseline sanity): single kvcached instance, TP=1, gpu0. Coder (Qwen3-Coder-30B-AWQ) then thinker (Qwen3.5-27B-AWQ) separately. Verify no regression vs T1.2a numbers (seq=1 ≥ 95% of 251/76.5 t/s).
3. Phase B (the point): two kvcached instances, both TP=1 on the SAME GPU (gpu0), sharing the KV pool. Combined weight footprint (~32 GiB) saturates physical VRAM — kvcached must handle this via virtual memory mapping. Run T1.2a-style concurrent-decode workload.
4. Phase C (stretch — only if B passes): two kvcached instances both TP=2 on all GPUs — the original failed T1.2 topology, retried.

**Pass:**
- Phase A: baseline intact (within 5% of T1.2a numbers).
- Phase B: combined TPS ≥ 80% of the sum of isolated TPS (coder 251 + thinker 76.5 = 327.5 → ≥ 262 combined).
- Phase C: combined TPS ≥ 2% (i.e. not the catastrophic collapse seen in T1.2). ≥ 50% would be a genuine architectural win and would reopen TP=2-for-both-hot.

**What failure means:**
- Phase A fails → `kvcached` incompatibility with our engine version or podman setup. Investigate (hand-back). No architectural change.
- Phase B fails → `kvcached` doesn't solve the context-slicing problem (it addresses memory, not scheduling). Architecture stays TP=1-per-GPU. Not a research block.
- Phase C fails but B passes → the CUDA-context problem is orthogonal to KV sharing; memory sharing alone isn't enough for TP=2 concurrent. Accept status quo + note in DECISIONS.md.
- Phase C passes → TP=2-for-both-hot is revived. Major architecture rewrite — hand back to research.

**Deps:** none (orthogonal to everything currently queued; needs working baseline from T1.2a which is DONE).

**Hand-back trigger:** Phase B passes (validates new sharing primitive — operator needs to decide whether to rework architecture around it), or Phase C passes (forces architecture rewrite), or any install/load crash whose cause is outside documented kvcached issues.

---

### T1.4 — th03_token_budget_fix — DONE (FAIL) ✗

**Result (2026-04-18):** FAIL. Model exhausts search budget even at 16k tokens. Not a budget limit, but a reasoning pathology/loop. Architecture-heavy tasks must be routed to Coder or Behemoth.

**Procedure:**

**Procedure:**
1. Start Qwen3.5-27B-AWQ with existing config + `--max-tokens 16384` (or the equivalent server-side limit).
2. Re-run thinker task th03 only (5 repetitions for stability).
3. Verify that `<think>` completes and a non-empty answer is produced.

**Pass:** 5/5 runs produce a non-empty architecture answer.

**What failure means:**
- Still empties the budget → thinker's reasoning length exceeds any reasonable cap for this task → route architecture-heavy tasks to the coder or behemoth, not the thinker.
- Parser issue (reasoning tokens not surfaced) → verify reasoning parser compatibility with raised limit. Minor fix.

**Deps:** none (independent, fast)

**Hand-back trigger:** cap raise doesn't help even at 32k; implies thinker strategy needs change.

---

## Tier 2 — model shortlist (which model fills each role)

Run after T1 is settled (or mid-T1 if cycles allow).

### T2.1 — glm47_flash_mla_verification — DONE (INCONCLUSIVE — MLA active / V1 tool-broken) ⚠

**Question:** In our current vLLM, does GLM-4.7-Flash use MLA (KV ≈54 KB/token) or is it stuck on the GQA-sized path (~98 KB/token) due to the known arch-convertor bug?

**Result (2026-04-18, 8 runs):** INCONCLUSIVE.

- **MLA confirmed active**: bench.logs from all runs (including the first on the standard image) show `Using TRITON_MLA attention backend out of potential backends: ['TRITON_MLA']`. The original reference values (MLA ≈54 KB, GQA ≈98 KB) were wrong for this model: actual MLA footprint ~129 KB/token (47 layers × kv_lora_rank=512 ≈ 94 KB base + CUDA-graph overhead measured via VRAM subtraction); actual GQA fallback would be ~376 KB (16 kv_heads × 128 head_dim × 47 layers). The `cu130-nightly` custom image correctly resolves `Glm4MoeLiteForCausalLM` but does not change the attention backend.
- **V1 engine cannot be disabled**: vLLM nightly forces V1 for this architecture. All six known env vars (`VLLM_V1=0`, `VLLM_USE_V1=0`, `VLLM_V1_ENABLED=0`, `VLLM_USE_V1_ENGINE=0`, `VLLM_ENGINE_ITERATOR_SOURCE=LEGACY`) are reported as "Unknown" and ignored.
- **Tool-calling broken under V1**: EngineDeadError on Tasks 02 and 03 (complex schemas). Task 01 (simple) passes at ~44.7 t/s. Tool sanity locked at 33%.
- **Decision**: GLM-4.7-Flash in cold storage until vLLM V1 stabilizes for this architecture. See `DECISIONS.md`.

**Original pass criterion (now superseded):** KV/token ≤ 60 KB (MLA path confirmed). Revised understanding: correct MLA threshold for this model is < 135 KB (94 KB base + ~40% overhead); the KV measurement methodology (VRAM subtraction) is imprecise due to CUDA-graph memory inclusion. Direct evidence for MLA is the TRITON_MLA backend log message.

**Root cause of tool crash (R8 research):** `glm4_moe_tool_parser.py` streaming path bug (PR #37385, not in our build). See `DECISIONS.md` for full details. Fix path is T2.1b.

---

### T2.1b — glm47_flash_streaming_parser_patch — CANCELLED

**Question:** Does patching `glm4_moe_tool_parser.py` with PR #37385's one-line fix (store streaming tool arguments as JSON string, not Python dict) make GLM-4.7-Flash tool calling reliable under V1?

**Root cause confirmed by R8 research:** `extract_tool_calls_streaming` stores `prev_tool_call_arr[index]["arguments"]` as a Python dict. Finalization code that uses this as a string crashes with TypeError inside the V1 EngineCore subprocess → EngineDeadError propagates to all subsequent requests. PR #37386 (v0.18.0) fixed the non-streaming path (greedy `.*` in `func_arg_regex`); PR #37385 fixes the streaming path but is not yet merged.

**CANCELLED after R9 research (2026-04-18).** The tool parser (`glm4_moe_tool_parser.py`) was reviewed in full (504 lines, all functions) and is correct. The crash is in the EngineCore subprocess (pid=188), not in the tool parser (APIServer pid=1). An EngineCore crash cannot be fixed by patching the parser. Root cause: TRITON_MLA PIECEWISE CUDA graph instability on Blackwell at longer decode lengths. Fix requires a vLLM upstream change. GLM-4.7-Flash is in cold storage until a vLLM release fixes `Glm4MoeLite`/TRITON_MLA on sm_120. See `DECISIONS.md` for full rationale.

**Pass:** ≥2/3 tool sanity tasks pass (Tasks 01–03). EngineDeadError no longer occurs.

**What failure means:**
- Non-streaming test passes but streaming still crashes after patch → the streaming path has an additional unfixed bug. **Hand back to research** with the new container logs showing the updated stack trace.
- Non-streaming test also crashes → the non-streaming path has a separate regression. **Hand back to research** — the fix path is wrong.
- Patch applies but all tasks still fail (wrong_tool, wrong_args) → parser works but model behavior needs investigation (parser not the issue).

**Deps:** T2.1 DONE (complete), R8 research cycle complete.

**Hand-back trigger:** any failure mode not covered by the "what failure means" branches above; or the container log shows a different exception than TypeError (indicating a different bug).

---

### T2.2 — coder_shootout_glm47_vs_qwen3coder30b

**Question:** Does GLM-4.7-Flash-AWQ beat, tie, or lose to Qwen3-Coder-30B-A3B-AWQ as our coder, on the current task suite + infra-shaped additions?

**Procedure:**
1. Both on vLLM TP=2, same context (32768), same quant format (AWQ-INT4), same KV cache settings.
2. Run Phase 0 tool-reliability suite (20 tasks) + Phase 2.1 coder quality suite (10 tasks).
3. Also add 3 infra-shaped smoke tasks (authored as part of this item): a containerfile authoring task, a systemd unit diagnosis, a shell-heavy refactor. See `benchmarks/infra_tasks/` to be created.
4. Score: tool-call PASS rate, task completion rate, human quality 1–5 per task, TTFT p50, decode TPS.
5. Note: for GLM-4.7-Flash, **do not enable MTP** in this test (separate item T3.2); use `--speculative-config.method none` or omit.

**Pass for GLM-4.7-Flash winning:** ≥95% tool-call PASS, quality sum ≥ Qwen3-Coder on ≥60% of tasks, no TPS regression worse than 20%.

**What failure means:**
- GLM-4.7-Flash loses → Qwen3-Coder-30B-AWQ stays as coder. Kill GLM-4.7-Flash candidacy in `DECISIONS.md`.
- GLM-4.7-Flash wins → swap coder baseline. Update `ARCHITECTURE.md` and re-confirm T1.2 (concurrent dual-process) with the new model pair.
- Mixed results (wins on some task types, loses on others) → **hand back to research**; the task suite may need extension and the role definition may need refinement (e.g. "code completion" vs "agentic multi-step").

**Deps:** T1.2 (need TP=2 dual-process viability confirmed), T2.1 (need MLA confirmed working on GLM-4.7-Flash).

**Hand-back trigger:** mixed results that suggest role redefinition.

---

### T2.3 — thinker_shootout_glm45_air_vs_qwen35_27b

**Question:** Does GLM-4.5-Air-AWQ outperform Qwen3.5-27B-AWQ as our thinker, specifically on multi-step planning and tool-heavy reasoning?

**Procedure:**
1. Both on vLLM TP=2. GLM-4.5-Air uses `--tool-call-parser glm45 --reasoning-parser glm45`. Note the group_size=64 Marlin kernel constraint caps TP=2 — which is what we want anyway.
2. Run Phase 2.2 thinker suite (8 tasks, including the fixed th03 from T1.4).
3. Add 3 infra-shaped reasoning tasks: a multi-container dependency resolution, a network tuning recommendation with constraints, an architecture decision between two infra approaches.
4. Human score 1–5 per task.
5. Measure decode TPS at seq=1 (for context-scheduling purposes; not the primary criterion).

**Pass for GLM-4.5-Air winning:** quality mean ≥ Qwen3.5-27B by ≥0.3 points on 1–5 scale, and the model doesn't regress on `<think>` token budget issues.

**What failure means:**
- GLM-4.5-Air loses → Qwen3.5-27B stays thinker. Note in `DECISIONS.md`.
- GLM-4.5-Air wins → swap thinker. Verify T1.2 with new pair.
- Token budget issues reappear on GLM-4.5-Air → same fix path as T1.4 at a raised cap.

**Deps:** T1.4 (th03 fix); T1.2 for confidence that concurrent-dual works.

**Hand-back trigger:** GLM-4.5-Air has undocumented parser issues we cannot resolve from the model card.

---

### T2.5 — coder_shootout_qwen36_35b_a3b_vs_qwen3coder30b — DONE ✓ (PASS)

**Result (2026-04-18):** PASS. 237.1 t/s seq=1 decode. **96.7% tool-call reliability** (29/30 pass). **100% quality completion rate** at `max_tokens=4096`. Requires `qwen3_coder` tool-parser + `qwen3` reasoning-parser + `enable-auto-tool-choice`. Verified BenchClient fix for `delta.reasoning` capture.

**What failure means:**
- Qwen3.6 loses on quality → Qwen3-Coder-30B-AWQ stays baseline. Mark Qwen3.6 ELIMINATED in `DECISIONS.md`.
- Qwen3.6 wins solo but can't run concurrent-dual at TP=1 (weight size makes CUDA graphs tight alongside thinker) → re-test concurrent viability via T1.5 (kvcached) once that spike settles. Gate the coder swap on concurrent numbers, not solo numbers.
- Mixed results → same hand-back pattern as T2.2.

**Deps:** none for solo run. For concurrent-dual eligibility: T1.5 (kvcached spike) or a successful TP=2-for-both-hot revival.

**Hand-back trigger:** mixed results; or CUDA graph capture fails at TP=1 and we need to decide TP=2 vs accept graph-disabled slowdown.

---

### T2.6 — behemoth_archetype_scouting (DESIGN)

**Question:** Which behemoth archetype(s) deserve a slot beyond the currently-settled Qwen3-Coder-Next-80B-A3B? Candidate axes: knowledge-rich (current) vs context-rich (e.g. 50–70B AWQ with 256k+ usable context) vs precision-rich (a mid-size model run at higher precision).

**Why this item    notes: >
      Former thinker baseline. Superseded by Qwen3.6-27B family.
      Hybrid SSM/transformer (Qwen3_5ForConditionalGeneration, GDN/FLA layers).
      --max-num-seqs 1 required to keep CUDA graphs active on single GPU.
as a direction to "leave space for."

**This is a design / scouting item, not a runnable benchmark.** It produces:
1. A candidate archetype taxonomy (2–3 archetypes, short rationale each).
2. A per-archetype 1–2 model shortlist (HF repos, VRAM at TP=2 AWQ, native context, tool-parser status).
3. For each shortlist entry, a list of concrete runnable sub-items (T2.6.1, T2.6.2, …) to be queued only after archetype triage.

**Out of scope here:** running any benchmark. That moves to sub-items.

**Pass:** a written-up design note added to `ARCHITECTURE.md` (behemoth section) and 1+ sub-items queued.

**Deps:** T1.3 (behemoth tier viability is confirmed — ✓ DONE), T2.4 (behemoth value vs coder — gate on "is the tier worth diversifying" evidence before spending cycles).

**Hand-back trigger:** N/A — this is a research-mode item by construction.

---

### T2.4 — behemoth_quality_vs_coder

**Question:** Does the behemoth (Qwen3-Coder-Next-80B-A3B-AWQ) produce meaningfully better results than the coder baseline on hard tasks, enough to justify its wake overhead?

**Procedure:**
1. Identify the 3–5 hardest tasks from our suites where the coder baseline struggles (TPS outlier, quality <3/5, or repeated tool-call failures).
2. Run those tasks on the behemoth (warm — no sleep/wake overhead for the quality test).
3. Human score. Compare to coder-baseline scores on the same tasks.

**Pass:** behemoth quality ≥ coder quality + 1.0 on 1–5 scale on the hard subset.

**What failure means:**
- Behemoth doesn't win on hard tasks → remove behemoth tier; simpler two-tier architecture.
- Behemoth wins but barely → keep as optional, don't wake automatically.
- Behemoth wins clearly → confirms the third tier. Integration step defines wake trigger.

**Deps:** T1.3 (behemoth must actually load and run).

**Hand-back trigger:** tasks where behemoth dramatically outperforms reveal a category we should be testing generally — may require task suite extension.

---

## Tier 3 — optimization axes (once roles are settled)

These tune parameters on the settled role assignments. Lower priority than Tier 2 — running them on the wrong model wastes the sweep.

### T3.1 — kv_cache_q8_q4_on_thinker

**Question:** At what KV cache quantization can the thinker hold 30k–50k context without unacceptable quality loss on reasoning tasks?

**Procedure:**
1. Thinker role endpoint (whichever wins T2.3). Configurations to test:
   - `--kv-cache-dtype auto` (bf16 equiv), context 32k — baseline
   - `--kv-cache-dtype fp8`, context 30k
   - `--kv-cache-dtype fp8`, context 50k
   - `--kv-cache-dtype int4` (if supported), context 50k
2. Run Phase 2.2 thinker suite + infra reasoning tasks.
3. Score quality 1–5. Record per-config VRAM usage and TPS.

**Pass:** fp8 at 30k shows no quality regression vs baseline. Decide whether int4 is worth the quality trade for 50k.

**What failure means:**
- fp8 regresses noticeably → baseline stays; live with shorter context.
- int4 is catastrophic → expected; we stick with fp8.
- fp8 is fine at 50k → great, use that.

**Deps:** T2.3 (need settled thinker).

**Hand-back trigger:** none expected unless a config combo crashes vLLM.

---

### T3.2 — mtp_spec_decode_on_blackwell_sm120

**Question:** Does MTP speculative decoding on GLM-4.7-Flash work on sm_120 (consumer Blackwell) without the 10× regression seen on B200?

**Procedure:**
1. Only run if GLM-4.7-Flash won T2.2.
2. Deploy with `--speculative-config.method mtp --speculative-config.num_speculative_tokens 1`.
3. Benchmark against no-spec-decode baseline on Phase 0 tool tasks.
4. Compare decode TPS (should go up 20–60% if spec decode is healthy).

**Pass:** TPS increases by ≥15%, no quality regression.

**What failure means:**
- TPS drops (the B200 pathology) → disable MTP, add to `DECISIONS.md` as "MTP unusable on sm_120 for this model."
- TPS unchanged → spec decode ineffective on our workload; disable for simplicity.
- TPS increases → keep it on.

**Deps:** T2.2 (only relevant if GLM-4.7-Flash is the coder).

**Hand-back trigger:** unexplained crashes → research vLLM issue tracker for sm_120 MTP status.

---

### T3.3 — qwen3_next_mtp_on_behemoth

**Question:** Same question as T3.2 but for Qwen3-Coder-Next's `qwen3_next_mtp` spec decode method.

**Deps:** T1.3, T2.4 (only if behemoth is kept).

**Hand-back trigger:** same as T3.2.

---

### T3.4 — prefix_cache_survival_across_sleep

**Question:** Does vLLM's CPU-offloaded prefix cache survive a sleep/wake cycle at level 1?

**Procedure:**
1. Deploy any model with `--enable-prefix-caching --cpu-offload-gb 8` (or current equivalent), TP=2.
2. Send a 4k-token prompt, record prefill time.
3. Send same prompt again, verify prefix cache hit (prefill time drops).
4. `POST /sleep?level=1`, then `POST /wake_up`.
5. Send same prompt again, measure prefill time.
6. Pass: post-wake prefill time ≈ pre-sleep cached time, not cold prefill time.

**What failure means:**
- Prefix cache flushed on sleep → wake cost is effectively "full prefill on first request per session." Architecturally fine but user-visible. Note in `ARCHITECTURE.md`.
- Prefix cache survives → good, integration plan can assume warm prefix across swaps.

**Deps:** T1.1.

**Hand-back trigger:** none expected; this is a factual check.

---

### T_CV1 — convergence_startup_timing — OPEN

**Question:** How long does Convergence take to become ready from cold start? Also: what is the practical context ceiling (max `-c` before RAM pressure causes OOM or TPS collapse)?

**Why this matters:** startup time determines always-resident vs on-demand policy; context ceiling determines the maximum query length Convergence can serve before Singularity is required.

**Procedure:**

*Part A — Startup timing (3 cold reps + 1 warm):*
1. Ensure Convergence is not running.
2. Drop page cache: `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`.
3. Record `date +%s%3N`. Start server with production flags (ngl=0, --no-mmap, -t $(nproc), -c 16384).
4. Poll `http://localhost:8002/health` every 500ms. Record timestamp on first 200 OK.
5. Stop server. Repeat steps 2–4 three times. Compute median cold-start time.
6. Run one warm-cache startup (no cache drop after the last cold rep). Record warm time.

*Part B — TPS baseline (ngl=0):*
7. Run `llama-bench` at ngl=0, -t $(nproc), -p 512, -n 128, -r 3. Record tg128 and pp512. This is the T_CV2 bootstrap baseline.

*Part C — Context ceiling sweep:*
8. For each `-c` in [16384, 32768, 65536, 131072]:
   - Start server with that `-c` value.
   - Send a prompt at ~80% of that context length. Record TTFT and TPS.
   - Monitor RAM via `free -h` — stop if free RAM drops below 8GB.
9. Record the last `-c` that completed without OOM or >50% TPS degradation vs `-c 16384`.

**Pass:** startup time and context ceiling captured; no thresholds — these are measurements.

**What the startup time means:**
- < 30s: cold-start on demand acceptable
- 30–90s: tolerable for explicit escalation, not transparent routing
- > 90s: always-resident strongly preferred

**Deps:** none.

**Hand-back trigger:** server crashes during load (GDN support issue in pr-1288 — check architecture errors in log); or context sweep reveals unexpected OOM pattern.

---

### T_CV2 — convergence_thread_count_sweep — OPEN

**Question:** What thread count maximizes token generation speed for Convergence? Baseline at `$(nproc)` = 32 threads gives 13.15 t/s. Hypothesis: MoE expert matrices (~1.28MB each at IQ2_M) may be too small to benefit from 32 threads — cache thrashing and thread coordination overhead may dominate.

**Procedure:**
1. Start Convergence with production command, varying only `-t`.
2. For each thread count in [8, 12, 16, 20, 24, 28, 32], run:
   ```bash
    /srv/ai/projects/ik_llama.cpp/build/bin/llama-bench \
      -m <model_path> \
      -ngl 0 --no-mmap \
      -fa 1 -b 4096 -ub 2048 \
      -t <N> \
      -p 512 -n 128 \
      -r 3
   ```
3. Record `tg128` (token generation at 128 tokens) and `pp512` (prompt processing at 512 tokens) for each `-t` value.
4. Plot or tabulate. Identify optimal for tg and for pp (may differ).

**Pass:** optimal thread count identified. Even if 32 is optimal, that's a valid result.

**What failure means:** all values within 5% of each other → thread count doesn't matter much for this model, keep 32 for simplicity.

**Deps:** T_CV1 (confirm model loads cleanly first).

**Hand-back trigger:** none expected.

---

### T_CV3 — convergence_partial_gpu_expert_offload — OPEN

**Question:** Can we improve Convergence generation speed by offloading some MoE expert layers to GPU? This experiment is intended for scenarios where Arclight/Core are sleeping, freeing up VRAM.

**Context:** Production Convergence is CPU-only (-ngl 0) to avoid conflict. However, when Arclight and Core are inactive, we have ~64GB VRAM available. Offloading early expert layers could improve throughput for long-horizon reasoning tasks.

**Method:** Use `-ngl 999 --n-cpu-moe N` which keeps MoE of the last N layers on CPU (keeping first layers on GPU).

**Procedure:**
1. Establish llama-bench baseline: `-ngl 0`, tg128 = X t/s.
2. Test `-ngl 999 --n-cpu-moe 50` (keep last 50 of 60 layers' experts on CPU, first 10 on GPU): measure tg128.
3. Test `-ngl 999 --n-cpu-moe 45` (first 15 on GPU): measure tg128.
4. Test `-ngl 999 --n-cpu-moe 40` (first 20 on GPU): measure tg128. Watch for OOM (each layer ~1-2GB experts).
5. Stop at first OOM or diminishing returns.

**VRAM budget per expert layer:** ~(expert_dim × hidden_dim × 3 matrices × IQ2_M bits/8). Rough estimate: ~2-3GB per layer for all 3 expert matrices. 10 layers ≈ 20-30GB of the available 52GB idle VRAM. Verify with `nvidia-smi` during test.

**Pass:** tg128 improves by ≥15% with partial offload without OOM.

**What failure means:**
- OOM before meaningful speedup → VRAM budget calculation was wrong; recompute.
- Speed improvement < 5% → DDR5 bandwidth is not the limiting factor for early layers (routing overhead dominates); keep `--cpu-moe` for simplicity.

**Deps:** T_CV2 (establish thread count baseline first).

**Hand-back trigger:** unexpected crash or architecture error — possible that ik_llama.cpp pr-1288 has issues with mixed CPU/GPU expert placement on Qwen3.5 specifically.

---

### T_KV1 — coder_big_context_mode — OPEN

**Question:** What is the maximum usable context for the coder when running TP=2 (thinker sleeping), and what is the throughput/latency profile at that context?

**Context:** With thinker sleeping (level=1, ~4GB GPU1 residual) and coder restarted as TP=2, combined VRAM is ~60GB. At fp8 KV, estimated KV budget is ~37GB → ~60–75K tokens. Verify the estimate and characterize quality at extended context.

**Procedure:**
1. Sleep thinker at level=1.
2. Restart coder with `--tensor-parallel-size 2 --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 --max-model-len 65536`.
3. Confirm startup and report VRAM split via `nvidia-smi`.
4. Send prompts of increasing length (8K, 16K, 32K, 65K tokens). Record TTFT and TPS at each.
5. Test `--swap-space 32` (32GB CPU KV overflow) with `--max-model-len 131072` — measure TTFT degradation at 100K+ token prompts.
6. Report: max context without swap, max context with swap at acceptable TTFT (<60s).

**Pass:** confirmed max context without swap ≥ 60K tokens; TPS at extended context within 20% of 32K baseline.

**What failure means:** KV budget estimate was wrong (VRAM footprint higher than expected) → recompute with actual `nvidia-smi` numbers and re-record in DECISIONS.md.

**Deps:** T1.1 (sleep mode working — DONE).

**Hand-back trigger:** coder TP=2 startup fails (unexpected — report engine version + CUDA error).

---

### T_KV2 — cuda_checkpoint_tp2_hot_restart — OPEN (HIGH PRIORITY)

**Script:** `benchmarks/queue/T_KV2_cuda_checkpoint_tp2_hot_restart.sh` (authored 2026-04-25)

**Question:** Does NVIDIA `cuda-checkpoint` + CRIU work for a TP=2 vLLM process in rootless Podman on our hardware? What is the restore time vs cold start?

**Why high priority:** unlocks ~5s mode switches for Extended Arclight (vs 170–300s cold), making the escalation pattern viable for interactive sessions.

**Prerequisites (one-time host setup):**
```bash
# 1. CRIU
sudo apt install criu
criu check          # verify kernel features; 'criu check --full' for exhaustive audit

# 2. cuda-checkpoint CLI + CRIU hooks (needed for GPU memory in checkpoint)
git clone https://github.com/NVIDIA/cuda-checkpoint /srv/ai/tools/cuda-checkpoint
cd /srv/ai/tools/cuda-checkpoint && make
sudo make install   # installs binary to /usr/local/bin/ + CRIU plugin to /usr/lib/criu/

# 3. Checkpoint directory
mkdir -p /srv/ai/checkpoints/coder-tp2
```

**Checkpoint strategy:** uses `podman container checkpoint --export <archive>` (CRIU-based,
handles rootless user-namespace mapping). With the cuda-checkpoint CRIU plugin installed,
GPU device memory (weights + compiled CUDA graphs) is included in the checkpoint — enabling
fast restore. Without the plugin, GPU state is excluded and restore is slow (no speedup over
cold). The script tests both conditions and reports which was active.

**Procedure:**
```bash
bash benchmarks/queue/T_KV2_cuda_checkpoint_tp2_hot_restart.sh
# Options: --dry-run, --reps N, --ctx N, --gpu-mem F, --skip-cold
```
The script handles: cold-start baseline, checkpoint, 3× restore with timing, cleanup,
metrics.json and summary.md. See script header for full option documentation.

**Pass:** restore time < 30s (10× improvement over cold); inference output matches pre-snapshot behavior.

**What failure means:**
- CRIU fails in rootless Podman → investigate `--unprivileged` CRIU flags or run outside container for this test.
- Multi-GPU (two CUDA contexts) checkpoint fails → report error; fall back to torch.compile disk cache only (~80s warm start).
- Restore succeeds but outputs are wrong → do not use; report to vLLM issue tracker.

**Deps:** T_KV1 (have a warm TP=2 process to snapshot).

**Hand-back trigger:** CRIU or CUDA checkpoint errors not in known docs — research before retrying.

---

### T_KV3 — thinker_tp2_fix_or_replacement — OPEN (GATE for no-Core final decision)

**Question:** Can the thinker run at TP=2 correctly? If not, is there an alternative thinker model that supports TP=2 without GDN state-split errors?

**Why this is a gate:** The "no Core" architecture is provisional. Extended Arclight (thinker TP=2) is the only path to long-context thinker escalation. If TP=2 remains broken for GDN indefinitely, we need a replacement thinker that doesn't have this constraint — or accept that only coder gets Extended mode.

**Two sub-questions to resolve in order:**

**Sub-Q1 — vLLM version fix:** Does current vLLM 0.19+ (with V1 disabled) still reproduce the T2.4g TP=2 quality failure for Qwen3.6-27B?
- T2.4g was run with V1 engine state unknown. V1 disabling may have changed behavior.
- Re-run T2.4g exact procedure (th02 × 3 reps, TP=2 + cp-ON) with current deploy.sh (V1 disabled).
- If now CORRECT: TP=2 thinker works, Extended thinker mode is unblocked. Update DECISIONS.md.
- If still INCORRECT: proceed to Sub-Q2.

**Sub-Q2 — alternative thinker:** Research and test a thinker model that:
- Is not GDN-hybrid (pure Transformer or MLA) — TP=2 shard is mathematically safe
- Has quality ≥ Qwen3.6-27B (4.875/5) on the 8-task thinker suite
- Fits within ~21GB AWQ at TP=1 for normal hot-pair mode
- Candidates to evaluate: strong reasoning models released 2025–2026 with pure Transformer or MLA architecture; community distills of top-tier proprietary models (o1, Claude-3.5-Sonnet style SFT on open base); small-company / individual researcher fine-tunes with documented tool-calling benchmarks

**Deps:** T_KV2 (have checkpoint infrastructure before committing to thinker TP=2 operational use).

**Hand-back trigger:** Any Sub-Q2 candidate research — return to research mode to identify and vet models before running tests.

---

### T_PAR1 — parallel_throughput_sweep — OPEN

**Question:** What is the optimal `--max-num-seqs` for each Arclight model and `-np` for Convergence to maximize aggregate TPS for agentic multi-subagent workloads?

**Context:** OpenCode v1.3+ routes multiple subagents concurrently. Coder and thinker may receive parallel requests. Convergence may receive batch queries during autonomous research runs. Current config (`--max-num-seqs 1` for thinker) is conservative. A3B MoE models have low active-param count, so concurrent requests should batch efficiently.

**Procedure:**

*Part A — vLLM Arclight (run for both coder and thinker separately):*
1. For each `--max-num-seqs` in [1, 2, 4, 8]:
   - Send N concurrent requests simultaneously (matching max-num-seqs value)
   - Record: per-request TTFT, per-request TPS, aggregate TPS, per-request latency
2. Find the knee: point where aggregate TPS gain per additional seq drops below 20%.
3. Note: thinker at seq > 1 may require `--enforce-eager` or larger `--gpu-memory-utilization` to maintain CUDA graphs — verify.

*Part B — ik_llama.cpp Convergence:*
1. For each `-np` in [1, 2, 4]:
   - Send N concurrent requests
   - Record: per-request TPS, aggregate TPS, RAM bandwidth (via `perf stat` or `turbostat`)
2. Note: `-np` increases memory pressure. Watch for OOM at `-np 4` with context > 16K.

**Pass:** optimal N identified for each tier. Even if N=1 is optimal everywhere, that's a valid result.

**What failure means:** aggregate TPS does not improve with N → memory bandwidth is the bottleneck even for batched requests; keep N=1 for predictable per-request latency.

**Deps:** T_CV2 (Convergence thread baseline before adding concurrency).

**Hand-back trigger:** none expected — this is a parameter sweep.

---

### T2.4 — arclight_thinker_qwen36_27b_candidate — DONE (FAIL ✗)

**Question:** Is Qwen3.6-27B-AWQ a better Arclight thinker than Qwen3.5-27B, specifically for quality on reasoning and infra-diagnostic tasks?

**Why Qwen3.6-27B specifically:**
- Dominant benchmarks: AIME 2026 94.1% (+4.9pp vs Gemma4), GPQA Diamond 87.8% (+3.5pp vs Gemma4), SWE-bench Verified 77.2%
- Same parser stack as coder (`--tool-call-parser qwen3_coder --reasoning-parser qwen3`) — proven at 96.7% reliability in T2.5, no new parser risk
- AWQ at 21 GiB from QuantTrio (trusted) → fits GPU1 TP=1 with comfortable KV headroom
- Apache 2.0

**Architecture note (R14):**
- 27B dense, 64 layers: 16 × (3 × Gated DeltaNet + 1 × Gated Attention)
- GDN (Gated DeltaNet) hybrid — NOT Mamba. kvcached still blocked (DeltaNetSpec unsupported) but TP=1 isolated deployment unaffected.
- Vision encoder included in weights; use `--language-model-only` to shed it for extra KV headroom. Verify flag in vLLM 0.19.x — omit if unsupported (budget fine either way).
- `transformers>=5.5.4` required — check vLLM container image version at deployment time.

**Procedure:**
```bash
# Download if not present
pyenv activate hf
HF_HOME=/srv/ai/models hf download QuantTrio/Qwen3.6-27B-AWQ

# Deploy on GPU1 TP=1 (thinker slot)
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 \
  --ctx 32768 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-auto-tool-choice

# Run Phase 2.2 thinker quality suite (8 tasks, max_tokens=4096)
# After run: score human_review.md, update metrics.json
```

**Baseline:** Qwen3.5-27B at 76.5 t/s seq=1, quality mean 4.0/5.

**Pass criteria:**
- Quality mean ≥ 4.0/5 AND at least ties on th02/th05 (the failure modes from T2.3b)
- TPS ≥ 60 t/s seq=1 (allows for GDN overhead vs Qwen3.5-27B)
- Tool call reliability ≥ 90% on thinker suite

**Special attention — th03:** Qwen3.5-27B emits empty output here (thinking budget exhaustion). Check whether Qwen3.6-27B produces a non-empty answer. If yes, this is a meaningful fix.

**Known vLLM issues to watch:**
- #40621 batch inference: affects concurrent multi-request scenarios; thinker role is typically single sequential requests — low risk.
- #40756 MTP crash: only with `--speculative-config`; we do NOT use MTP — not applicable.
- #40725 TP=4 non-English corruption: TP=1 only — not applicable.

**kvcached Phase B note:** GDN hybrid means T1.5 Phase B remains blocked (same as Qwen3.5-27B). No change to kvcached status.

**Deps:** none.

**Hand-back trigger:** tool calling broken (wrong parser? transformers version incompatibility?); or CUDA graph OOM despite --gpu-mem-util 0.90 (try `--max-num-seqs 1` as fallback, same as Qwen3.5-27B fix).

---

### T2.4c — arclight_thinker_qwen36_27b_nvfp4_swap — INCONCLUSIVE

**Question:** Does NVFP4 quantization + bf16 KV cache + TP=2 resolve the confident incorrectness seen in AWQ TP=1 runs?

**Runs completed:**
- 230351Z (partial, th02+th03 only): both 5.0 — operator declared PASS. **Premature — only 2/8 tasks.**
- 232801Z (full 8-task run): th02 semantic error (missed jobs not assigned to any GPU; `-1` assignment = silent wrong output). Mean ~3.94/5 — below 4.0 baseline.

**Result:** NVFP4 + bf16 KV did NOT resolve confident incorrectness. AWQ run 4 (the only run with correct th02) scored ~4.25/5, better than the NVFP4 full run. T2.4c INCONCLUSIVE — do not deploy NVFP4 as thinker until root cause is understood.

**Publisher note:** `sakamakismile/Qwen3.6-27B-NVFP4` is an untrusted publisher. If NVFP4 warrants re-testing after root cause investigation, use `nvidia/` or `bartowski/` quants instead. Deferred to T_NVFP4.

**What this means:** Root cause investigation takes priority (T2.4f → T2.4d → T2.4e). Qwen3.5-27B remains the active thinker baseline.

---

### T2.4f — qwen36_27b_rope_chunkedprefill_audit — DONE ✓
### T2.4d — qwen36_27b_awq_reproducibility — DONE ✓ (th02 correct 3/3; quality 4.875/5 scored R17)
### T2.4e — qwen36_27b_awq_bf16kv_tp2 — DONE (th02 INCORRECT — confounded, see R17)

**Question:** Is AWQ run 4's correct th02 implementation stable, or was it a lucky sample near the model's capability ceiling?

**Why this matters:** Run 4 (AWQ, fp8 KV, ctx=32768, max_tokens=16384) is the only run that produced correct th02 code. All other runs had errors. If run 4 is not reproducible, the confident incorrectness is a model capability issue that no serving config change will fix reliably.

**Procedure:**
1. Run exact run 4 config **three times** with identical parameters:
   - Model: `QuantTrio/Qwen3.6-27B-AWQ`
   - TP=1 (GPU1), fp8 KV, ctx=32768, max_tokens=16384, max-num-seqs 1
   - Apply any corrections from T2.4f (rope_theta, chunked prefill flags)
2. Score th02 in each run: correct (all jobs assigned including misses) / semantic-error / crash
3. Score all 8 tasks in at least one of the 3 runs; track which run if you don't score all three fully.

**Pass:** th02 correct in ≥ 2/3 runs → run 4 is reproducible, move to T2.4e.
**Fail:** th02 correct in ≤ 1/3 runs → capability ceiling confirmed. Model cannot reliably implement this task class at 27B. Pivot to T2.4b (Qwopus SFT) or accept Qwen3.5-27B as the thinker permanently until a better 27B model exists.

**Deps:** T2.4f should run first (config corrections may affect outcome). Can run immediately if T2.4f shows no corrections needed.

---

### T2.4e — qwen36_27b_awq_bf16kv_tp2 — DONE (th02 INCORRECT — confounded, see R17)

**Result (2026-04-25, 114506Z):** th02 INCORRECT — missed jobs silently dropped, not assigned to busiest GPU. 104.8 t/s decode, 109ms TTFT p50. Quality scores pending human review. Run used `--no-enable-chunked-prefill` (DISABLE_CHUNKED_PREFILL=1), changing two variables from T2.4d simultaneously (TP=1→2 AND cp-on→off). Root cause of regression ambiguous. See R17 in RESEARCH_STATE.md.

**Question (original):** Does removing fp8 KV cache noise (by using bf16 KV + TP=2) improve th02 correctness for the AWQ model specifically? Isolates KV precision from the NVFP4 publisher quality issue.

**Why AWQ not NVFP4:** sakamakismile/NVFP4 is an untrusted publisher. T2.4c conflated publisher quality with quantization format. This test uses the same trusted AWQ weights (QuantTrio) but with bf16 KV to isolate the KV dtype variable cleanly.

**Config:**
```bash
./infra/scripts/deploy.sh vllm tp2b QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.85 \
  --ctx 32768 \
  --kv-cache-dtype auto \      # bf16 KV — no quantization noise
  --max-num-seqs 1 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
  # + any rope_theta/chunked-prefill corrections from T2.4f
```

**Requires sleeping coder** (TP=2 borrows GPU0). Sleep Qwen3.6-35B coder first.

**Run once.** Score th02 and th05. If th02 correct: fp8 KV was degrading quality → bf16+TP=2 is the production config (with sleep-mode coordination). If th02 still errors: KV precision is not the issue → capability ceiling or RoPE/chunked-prefill is the root cause.

**Pass criteria:** th02 correct (all jobs assigned, including misses to busiest GPU) AND mean ≥ 4.0/5.
**Deps:** T2.4d should complete first (confirms whether issue is reproducibility or config). Can run in parallel with T2.4d if testing time is limited.

---

### T_NVFP4 — nvfp4_thinker_mass_survey — DEFERRED

**Question:** Is the NVFP4 format (Blackwell-native FP4 tensor cores) worth pursuing for the thinker role, using a trusted publisher?

**Why deferred:** T2.4c used `sakamakismile` (untrusted publisher). Cannot distinguish format benefit from publisher calibration quality. Do not pull additional NVFP4 quants until T2.4d/e/f determine whether the root cause is KV precision (which NVFP4 doesn't directly address) or model capability (which NVFP4 also doesn't address).

**When to un-defer:** After T2.4e result. If AWQ + bf16 KV + TP=2 passes (KV precision was the issue), then NVFP4 is theoretically superior (native FP4 weights + BF16 KV headroom). Pull from trusted publishers at that point:
- `nvidia/Qwen3.6-27B-NVFP4` (if available)
- `bartowski/Qwen3.6-27B-NVFP4` (established publisher)
- Do NOT use `sakamakismile` again.

**What to measure:** TPS seq=1 vs AWQ baseline (77.4 t/s) + th02 quality only (the discriminating task). If TPS ≥ 80 t/s AND th02 correct → worth a full 8-task run.

**Deps:** T2.4e DONE.

---

### T2.4g — qwen36_27b_awq_tp2_chunkedprefill_on — DONE (H-TP2 CONFIRMED 2026-04-25)

**Question:** Does TP=2 + bf16 KV + chunked-prefill ON produce correct th02? This is the single missing cell in the 2×2 factorial (TP × chunked-prefill) that isolates whether the T2.4e regression was caused by TP=2 or by cp-OFF.

**Why MEDIUM, not LOW:** Without this test we cannot attribute the T2.4e failure to a specific cause. The current state is a confound — two variables changed simultaneously (TP=1→2 AND cp-ON→OFF) and the result was incorrect. This ambiguity has downstream consequences:
- If T2.4g is CORRECT: cp-OFF alone caused the failure; TP=2 is safe with GDN. Opens NVFP4 re-evaluation (T_NVFP4) with correct cp setting.
- If T2.4g is INCORRECT: TP=2 itself breaks GDN state across shards, regardless of cp. Fully closes the TP=2 question. T_NVFP4 remains deferred (NVFP4 + TP=2 would also fail).
The deployment decision (TP=1 stays) is not changed by either outcome. But the scientific question of *why* TP=1 works is open until this runs.

**Why not production-blocking:** TP=1 is proven correct and deployed. This test cannot make TP=1 wrong. It can only determine whether TP=2 is also viable (with cp-ON) or definitively broken.

**What it would tell us if CORRECT:** chunked-prefill was the only differentiator; TP=2 itself is fine for GDN. Could inform future config decisions if we ever need higher thinker TPS and are willing to accept sleep-mode overhead.

**What it would tell us if INCORRECT:** TP=2 itself causes th02 regression regardless of chunked-prefill. Fully settles the TP=2 question.

**Config:**
```bash
./infra/scripts/deploy.sh vllm tp2b QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.85 \
  --ctx 32768 \
  --kv-cache-dtype auto \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```

**Requires sleeping coder** (TP=2 borrows GPU0).

**Pass criteria:** th02 correct (all jobs assigned, including misses to busiest GPU).

**Deps:** T2.4d DONE ✓, T2.4e DONE. No blocking deps.

---

### T2.4b — arclight_thinker_qwopus36_27b_candidate — SKIPPED (dep condition met)

**Question:** Does the Qwopus SFT (Claude + Kimi + Qwen3.5 reasoning distillation on Qwen3.6-27B) improve thinker quality over the base Qwen3.6-27B, specifically on multi-step constraint reasoning (th02, th05)?

**Why Qwopus specifically:** The SFT data targets reasoning-chain quality — exactly the failure mode of Gemma4-31B (th02 algorithm logic, th05 distributed consistency reasoning). If the base model misses on these same task types, a reasoning-distilled variant is the logical follow-up.

**Model:** `Jackrong/Qwopus3.6-27B-v1-preview` — SFT on Qwen3.6-27B base with ~12K examples from Claude Distillation + Kimi K2.5 + Qwen3.5 reasoning datasets. v1-preview status, 16-prompt evaluation only.

**Deployment options (no AWQ available):**

Option A — BF16 TP=2 (requires sleeping coder):
```bash
# Stop coder (GPU0), stop thinker (GPU1) to free both GPUs
# Deploy at TP=2 — 27B BF16 ≈ 54GB fits 64GB with --gpu-mem-util 0.85
./infra/scripts/deploy.sh vllm tp2b Jackrong/Qwopus3.6-27B-v1-preview \
  --gpu-mem-util 0.85 --ctx 32768 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --enable-auto-tool-choice
```

Option B — community GGUF (if available by the time T2.4b runs): check HF for quants. Prefer QuantTrio if they release one.

**Pass criteria:** quality mean > T2.4 baseline on th02 and th05 specifically. Mean overall is secondary — the point is whether reasoning distillation fixes the specific failure mode identified.

**Deps:** T2.4 must complete first. Only run T2.4b if:
- T2.4 fails quality bar (mean < 4.0/5), OR
- T2.4 passes overall but still misses on th02/th05 — reasoning distillation directly targets those.
- Skip T2.4b entirely if T2.4 passes on th02/th05.

**Hand-back trigger:** BF16 OOM at TP=2 (unexpected — 54GB should fit in 64GB with 0.85 util); or tool calling completely broken (base Qwen3.6-27B parser stack should transfer, but SFT could have affected template).

---

### T2.3b — arclight_thinker_gemma4_31b_candidate — DONE (REJECTED)

**Question:** Is Gemma4-31B (dense, Apache 2.0, no Mamba) a better Arclight thinker than Qwen3.5-27B, specifically for quality on reasoning and infra-diagnostic tasks?

**Why Gemma4-31B specifically:**
- Dense model: no SSM/Mamba layers → no MambaSpec → kvcached-compatible (FullAttentionSpec)
- GPQA Diamond 84.3%, AIME 2026 89.2%, Arena Elo 1452 (beats Qwen3.5-397B at 1449)
- Apache 2.0 license

**Corrected specs (R13, 2026-04-23):**
- Weight size: **~20 GiB** (not ~16 GiB as originally estimated)
- vLLM image: **`vllm/vllm-openai:gemma4`** — NOT `:latest`. vLLM 0.19.x has tool-call JSON corruption bug (#39468). Pass via `BENCH_IMAGE=vllm/vllm-openai:gemma4`.
- Parser flags: **`--tool-call-parser gemma4 --trust-remote-code`** only. DO NOT add `--reasoning-parser gemma4`: streaming path drops tool calls when model skips reasoning (same root cause as Qwen3-Next-80B). See DECISIONS.md.
- kvcached Phase B (single GPU): **NOT viable**. 22+20=42 GiB > 32 GiB. See note below.

**Procedure:**
```bash
./benchmarks/queue/T2.3b_gemma4_31b_thinker.sh
```
Script handles TP=1→TP=2 fallback, TPS measurement, and the 8-task Phase 2.2 quality suite.
After the run: score `human_review.md` (1–5 per task), update `metrics.json`.

**Result (2026-04-24):** REJECTED as primary Thinker. Mean quality 4.0/5. Excellent task completion (100%, including non-empty th03 where Qwen3.5-27B emits empty), but "Surface-Level Reasoning" on th02/th03/th05 fails the depth-of-reasoning bar. Dense architecture, no-MambaSpec, strong 5/8 task profile → redirected to **coder candidate (T2.3c)** rather than discarded.

**What failure means:**
- Quality lower → Qwen3.5-27B stays thinker. Record task-type breakdown in DECISIONS.md.
- Wins on quality AND produces non-empty th03 → significant win (fixes the known defect).

**kvcached Phase B note:**
Single-GPU cohabitation with Qwen3.6-35B coder is physically impossible at these weight sizes (42 GiB > 32 GiB). The Phase B opportunity for Gemma4 is **cross-GPU KV sharing** (coder GPU0 + thinker GPU1, shared elastic KV pool). This is a different topology from the original T1.5 Phase B design and requires a separate research item before testing.

**Deps:** none (can run immediately — script is ready).

**Hand-back trigger:** unexpected vLLM compatibility failure with Gemma4 on sm_120 Blackwell; or tool calling broken despite correct image and parser flags.

---

---

### T2.3c — arclight_coder_gemma4_31b_candidate — SKIPPED (benchmark evidence, 2026-04-25)

**Skipped without running.** Operator-led benchmark research (ChatGPT with full context, Qwen3.6-35B-A3B model card as primary source) shows Qwen3.6-35B-A3B is clearly superior on the benchmarks most relevant to this workload. Running T2.3c would consume a GPU-day to confirm what external evals already show.

**Key evidence (Qwen card comparison table, vendor-reported but only shared source with both models):**

| Benchmark | Gemma4-31B | Qwen3.6-35B-A3B | Gap |
|-----------|-----------|----------------|-----|
| SWE-bench Verified | 52.0 | **73.4** | +21.4 Qwen |
| Terminal-Bench 2.0 | 42.9 | **51.5** | +8.6 Qwen |
| MCPMark | 18.1 | **37.0** | +18.9 Qwen |
| WideSearch | 35.2 | **60.1** | +24.9 Qwen |
| NL2Repo | 15.5 | **29.4** | +13.9 Qwen |
| LiveCodeBench v6 | 80.0 | 80.4 | tie |
| TAU3-Bench | **67.5** | 67.2 | tie |

Pattern: **Gemma ties on pure coding (LiveCodeBench) but Qwen dominates on agentic, terminal-agent, MCP, and repo-level tasks** — exactly the operator workload. Gemma is not weak; Qwen is simply better for this profile.

**Decision:** Qwen3.6-35B-A3B-AWQ remains the coder. Gemma4-31B is fully retired from Arclight consideration — no role remains.

---

## TESTING_QUEUE — status summary (R17, 2026-04-25)

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| T2.4f | DONE ✓ | — | rope_theta=10M confirmed; cp-OFF OOM at TP=1 confirmed |
| T2.4d | DONE ✓ | — | TP=1 th02 correct 3/3; quality **4.875/5** (R17 scored) |
| T2.4e | DONE (INCONCLUSIVE) | — | th02 INCORRECT; confounded by TP=2+cp-OFF; see R17 |
| T2.3c | SKIPPED | — | Gemma4-31B as coder — benchmark evidence (SWE/Terminal/MCP) shows Qwen clearly better; no test needed |
| T_CV1 | OPEN | MEDIUM | Convergence startup timing — no deps |
| T2.4g | DONE (H-TP2 CONFIRMED) | — | th02 SEMANTIC ERROR 0/3; TP=2 definitively broken for GDN regardless of cp |
| T_NVFP4 | DEFERRED | — | NVFP4 mass-pull — restricted to TP=1 only; no urgency; defer indefinitely |
| T_CV2 | OPEN | LOW | Thread count sweep — after T_CV1 |
| T_CV3 | OPEN | LOW | Partial GPU expert offload — after T_CV2 |
| T2.4b | SKIPPED | — | Qwopus SFT — dep condition met (T2.4d quality ≥ 4.0, th02/th05 pass) |
| T2.4 | DONE (settled by T2.4d) | — | AWQ TP=1: reproducible correct at 4.875/5. |
| T2.4c | INCONCLUSIVE | — | NVFP4 TP=2: full run ~3.94/5 (th02 semantic error). |
| T2.3b | DONE | — | Gemma4-31B as thinker — REJECTED; redirected to T2.3c |
| T1.5 Phase B | DEFERRED | — | kvcached blocked (GDN/DeltaNet unsupported) |

---

## Tier 4 — engine / infra comparison

### T4.1 — sglang_for_a3b_coder

**Question:** Given Sleep Mode and MTP are vLLM-exclusive for our use, is SGLang worth keeping as a comparison point at all?

**Answer from research (tentative):** probably not for anything where we plan multi-model swap. But retain SGLang as a comparison point for:
- Single-model raw-TPS upper bound (reference number, not architecture-relevant).
- Models where vLLM has bugs we can't work around.

**Procedure:** only rerun if a specific model fails on vLLM and we need a fallback. Otherwise PUNTED.

**Status:** PUNTED (revisit if vLLM becomes a bottleneck).

---

### T4.2 — llama_cpp_as_fallback_for_exotic_quants

**Question:** Is llama.cpp needed for GGUF-only quants we can't get in AWQ (e.g. Unsloth UD-* dynamic quants)?

**Status:** PUNTED. Only run if a capability gap appears that AWQ cannot fill. Unsloth UD quants are mostly relevant for memory-tight systems, which we are not.

---

## Tier 5 — integration

### T5.1 — opencode_endpoint_binding_with_subagents

**Question:** Does OpenCode v1.3+ route subagent calls to the correct endpoint without a proxy, and does it handle endpoint failure gracefully?

**Procedure:**
1. Configure `opencode.json` with `vllm-coder` / `vllm-thinker` / `vllm-behemoth` providers bound to our ports.
2. Define three agents with matching model overrides.
3. Run a task that invokes all three (via `spawn_subagent`).
4. Verify correct endpoint routing by inspecting per-endpoint request logs.
5. Kill one endpoint mid-task, verify OpenCode's behavior (timeout, retry, graceful surface to user).

**Pass:** routing is deterministic and failures surface cleanly.

**Deps:** T2.2, T2.3, T2.4 (need settled roles).

**Hand-back trigger:** OpenCode routing does not work as documented.

---

### T5.2 — behemoth_wake_trigger_integration

**Question:** What signals a need to wake the behemoth, and how is it wired?

**Procedure:** design task, not benchmark. Write up in `ARCHITECTURE.md` after T2.4 settles. Likely path: an explicit "escalate" subagent whose model binding is `vllm-behemoth`, operator invokes manually via `@behemoth` reference.

**Status:** DESIGN — no benchmark needed if T2.4 passes.

---

### T5.3 — mcp_servers_firecrawl_searxng

**Question:** Do the firecrawl + searxng MCP servers work reliably with OpenCode + our local LLM stack?

**Procedure:** functional smoke test, not a benchmark. Configure, run 5 common doc-lookup queries, verify answers cite sources. Goal is "does the pipe work," not "which model is better at citing."

**Deps:** T5.1 (need OpenCode integration settled first).

**Hand-back trigger:** unexpected tool-call format issues that suggest the model's tool parser + MCP server aren't speaking the same dialect.

---

## Tier 6 — task suite extension (ongoing, not blocking)

The current task suites are coding-heavy. Infra workload coverage is thin. These items author new tasks and re-score models on the expanded suite.

### T6.1 — infra_shell_and_container_tasks — RERUN NEEDED

Tasks authored (in01–in05): Containerfile debugging, systemd unit troubleshooting, compose authoring, shell idempotency, network tuning. Script at `benchmarks/queue/T6.1_infra_task_suite.sh`.

**Manual run (2026-04-25):** 232 t/s TPS, 100% task completion, confirmed via operator-run session. Results recorded in `config/models.yaml`.

**Rerun required:** The manual run was with uncertain V1 engine state. Now that V1 is disabled by default in `deploy.sh`, rerun the full T6.1 script to get a clean automated baseline. Also needed: confirm whether coder TP=1 (with V1 disabled) recovers ~237 t/s — run T6.1 with TP=1 alongside TP=2 to settle the production config.

**Deliverable:** `results/T6.1_infra_task_suite_*/metrics.json` with V1-disabled config recorded explicitly.

### T6.2 — cross_arch_tasks

Tasks specific to Orange Pi / armbian work: cross-compilation, kernel module questions, cloud-init-like init idempotency.

### T6.3 — ops_tasks_ceph_openstack

Ceph placement group troubleshooting, OpenStack service diagnosis, at least 3 tasks each.

### T6.4 — rag_aware_tasks

Tasks that explicitly require looking up current docs (to test the MCP pipeline end-to-end with the model's tool-call discipline).

**Scheduling:** run T6.x tasks back through T2.2 / T2.3 to re-score settled roles. A role winner on the coding suite may not win on the infra suite — that's the whole point.

---

## Parking lot — explicit non-items

Listed so they don't get re-added as confusion.

- NVFP4 as a standalone phase — see `DECISIONS.md` PROVISIONAL.
- Alternative AWQ publishers — see `DECISIONS.md` PROVISIONAL.
- Abliterated variants — off-topic.
- Dense 70B TP=2 — see `DECISIONS.md` SETTLED.
- Cloud quality baselines — not relevant to our goal (we aim for local-only).
- Cost analysis — not relevant (electricity is already paid).