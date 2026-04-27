# Open Queue

Only OPEN and BLOCKED items. Full status table: docs/queue/status.md. Full procedures for DONE items: docs/history/done-items.md.

Legend: **OPEN** = ready to run (deps met). **BLOCKED** = deps or research needed. **DEFERRED** = not now.

---

## HIGH priority

### T_PAR1 — parallel_throughput_sweep — OPEN (Coder + Thinker rerun)

**Convergence portion DONE** (N≥2 crashes pr-1288 — real data). Coder and Thinker portions NOT MEASURED — prior run found no endpoints running; fabricated numbers were written to docs. **Rerun required.**

**Question:** What is the optimal `--max-num-seqs` for coder and thinker to maximize aggregate TPS for parallel OpenCode subagent workloads?

**Script:** `benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh`

**Prerequisites:**
- Coder running on port 30000 (TP=2, production config)
- Thinker running on port 30001 (TP=1 GPU1, production config)
- Convergence already settled (skip with `--skip-convergence`)

**Run:**
```bash
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh --skip-convergence --reps 3
```

For thinker: run once at production `--max-num-seqs 1` (baseline behavior), then once at `--max-num-seqs 4` (reveals true parallel ceiling). Both results are useful.

**What to watch for:**
- Coder aggregate TPS vs N: knee at N=4 or N=2? A3B MoE should batch efficiently.
- Thinker: can it even handle N>1 without OOM on 32GB card? (Prior claim of OOM was fabricated — verify empirically.)
- If thinker OOMs at N>1: record VRAM usage from nvidia-smi to understand the actual limit.

**Pass:** optimal N identified for each tier. N=1 everywhere is a valid result.

**What failure means:**
- Coder OOM at N>1 → record VRAM; note CUDA graph pre-allocation may be the blocker.
- Thinker OOM → confirms serial-only constraint; update status table.
- Aggregate TPS flat across N → memory bandwidth bottleneck; keep N=1.

**Hand-back trigger:** None expected — this is a parameter sweep. If coder crashes with CUDA error (not OOM), hand back to research.

---

### T_KV3 — thinker_tp2_fix_or_replacement — BLOCKED on research (Sub-Q2)

**Sub-Q1 SETTLED (T2.4g):** GDN TP=2 broken regardless of chunked-prefill.

**Sub-Q2 BLOCKED:** Research needed to identify a thinker candidate that:
- Is NOT GDN-hybrid (pure Transformer or MLA) — TP=2 shard is mathematically safe
- Quality ≥ Qwen3.6-27B (4.875/5) on the 8-task thinker suite
- Fits ~21GB AWQ at TP=1 for normal hot-pair mode
- Candidates: strong reasoning models released 2025–2026 with pure Transformer or MLA architecture

**No script yet.** Research mode required first. Return here after research provides a model slug + deploy config.

---

## MEDIUM priority

### T3.4 — prefix_cache_survival_across_sleep — OPEN

**Question:** Does vLLM's CPU-offloaded prefix cache survive a sleep/wake cycle at level=1?

**Procedure:**
1. Deploy any model with `--enable-prefix-caching --cpu-offload-gb 8`, TP=2.
2. Send 4k-token prompt, record prefill time.
3. Send same prompt again — verify cache hit (prefill time drops).
4. `POST /sleep?level=1`, then `POST /wake_up`.
5. Send same prompt again, measure prefill time.

**Pass:** post-wake prefill ≈ pre-sleep cached time (not cold-prefill time).

**What failure means:** Prefix cache flushed on sleep → first request after wake costs full prefill. Architecturally fine but user-visible latency. Note in `docs/arch/current.md`.

**Deps:** T1.1 ✓. Script at `benchmarks/queue/T3.4_prefix_cache_survival.sh` (verify exists or create).

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
