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

### T1.4 — th03_token_budget_fix

**Question:** Does raising `--max-tokens` to 16384 on the thinker endpoint resolve the `th03_architecture_tradeoffs` empty-output defect?

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

### T2.1 — glm47_flash_mla_verification

**Question:** In our current vLLM, does GLM-4.7-Flash use MLA (KV ≈54 KB/token) or is it stuck on the GQA-sized path (~98 KB/token) due to the known arch-convertor bug?

**Procedure:**
1. Deploy `cyankiwi/GLM-4.7-Flash-AWQ-4bit` with `--tool-call-parser glm47 --reasoning-parser glm45`, TP=2, some context value C.
2. Read vLLM logs for reported KV cache block size or `GPU KV cache size` line.
3. Compute KV/token: `(free_kv_bytes) / (blocks × block_size)`, compare to expected MLA (~54 KB) vs GQA (~98 KB).
4. If KV/token ≈ 98 KB, apply the one-line `glm4_moe_lite` patch in `model_arch_config_convertor.py` (or pin a known-good vLLM version) and retest.

**Pass:** KV/token ≤ 60 KB (MLA path confirmed).

**What failure means:**
- MLA not detected and patch doesn't apply cleanly → wait for upstream fix or pin vLLM version. Note in `DECISIONS.md`.
- MLA works → GLM-4.7-Flash is fully usable, proceed to T2.2.

**Deps:** none. Operationally cheap — verify once per vLLM version bump.

**Hand-back trigger:** patch required but doesn't apply; need research on current upstream status.

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

### T6.1 — infra_shell_and_container_tasks

Author ~5–10 tasks covering: Containerfile debugging, systemd unit troubleshooting, compose-file authoring, shell script idempotency, network tuning (sysctl/ethtool/tc).

**Deliverable:** `benchmarks/infra_tasks/tasks/*.json` + grading rubric.

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