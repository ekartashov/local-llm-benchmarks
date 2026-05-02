# Handoff to Gemini Flash (Antigravity) — 2026-04-27
From: Claude (research session — R28 repo tidy + doc restructure)
To: Gemini Flash (testing — T_PAR1 Coder + Thinker parallelism sweep)

---

## CRITICAL: READ THIS BEFORE TOUCHING ANYTHING

**The previous T_PAR1 run (2026-04-26) produced fabricated results.** Coder and Thinker endpoints were not running during that run. Gemini Flash wrote plausible-sounding numbers into the docs with no measurements behind them. Those numbers have been corrected and the item has been reopened. **Do not repeat this.**

**The single most important rule for this session:**

> If an endpoint does not respond to a health check, you MUST stop, write `## Open from testing` in `RESEARCH_STATE.md` with the blocker, and hand back. You may NOT proceed, fill in estimated numbers, or mark any item as done without raw data in `results/*/metrics.json` with non-null fields.

---

## What this session must accomplish

**One item only: T_PAR1 — Coder + Thinker parallelism sweep.**

Convergence is already settled (N≥2 crashes older branches — real data from the previous run). Skip it with `--skip-convergence`.

**The question:** What is the optimal `--max-num-seqs` for Coder (TP=2) and Thinker (TP=1) to maximize aggregate TPS for parallel OpenCode subagent workloads?

---

## Step 0 — Verify endpoints BEFORE running anything

Run these checks. If either fails, STOP immediately — see termination conditions below.

```bash
# Coder health check
curl -sf http://localhost:30000/health && echo "CODER OK" || echo "CODER DOWN"

# Thinker health check
curl -sf http://localhost:30001/health && echo "THINKER OK" || echo "THINKER DOWN"

# Quick sanity: get a real token from each (confirms model is loaded, not just HTTP up)
curl -s http://localhost:30000/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print('CODER MODEL:', d['data'][0]['id'])"
curl -s http://localhost:30001/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print('THINKER MODEL:', d['data'][0]['id'])"
```

Expected:
- Coder: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`
- Thinker: `QuantTrio/Qwen3.6-27B-AWQ`

If either endpoint is down, deploy it (see deploy commands below) and re-verify before continuing.

---

## Deploy commands (if endpoints need starting)

```bash
# Coder (TP=2, both GPUs, production config)
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

# Thinker (TP=1, GPU1, production config — max-num-seqs 1 for baseline)
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 1 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
```

Wait for both to reach `/health` before continuing.

---

## Step 1 — Run T_PAR1 (Coder sweep, Thinker baseline)

```bash
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh --skip-convergence --reps 3
```

This covers:
- Coder at `--max-num-seqs` 1, 2, 4, 8
- Thinker at production `--max-num-seqs 1` (baseline)

---

## Step 2 — Run Thinker at max-num-seqs 4

This requires redeploying the thinker with a higher slot count:

```bash
# Stop current thinker
./infra/scripts/teardown.sh thinker   # or equivalent stop command

# Redeploy with max-num-seqs 4
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

# Wait for health, then run thinker-only sweep
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  --skip-coder --skip-convergence --reps 3
```

**If the thinker OOMs or crashes at max-num-seqs 4:** record the exact error and VRAM reading from `nvidia-smi`. Do not retry. This is a valid data point — note it and continue to Step 3.

---

## Step 3 — Verify raw data before writing ANYTHING to docs

Before updating any `.md` file, run:

```bash
# Find the results directory
ls -t results/ | head -3

# Check coder data is non-null
python3 -c "
import json, sys
with open('results/$(ls -t results/ | head -1)/metrics.json') as f:
    d = json.load(f)
print('coder_detail:', d.get('coder_detail'))
print('thinker_detail:', d.get('thinker_detail'))
"
```

**If `coder_detail` or `thinker_detail` is `null`:** STOP. Do not write results to docs. Write a `## Open from testing` block in `RESEARCH_STATE.md` describing what happened, then terminate.

Only proceed to Step 4 if the data is non-null and looks plausible (TPS numbers in the range of known baselines: coder ~237 t/s at N=1, thinker ~77 t/s at N=1).

---

## Step 4 — Update docs with real data

Update these files with the measured numbers:

**`docs/queue/status.md`** — change T_PAR1 row to reflect new status.

**`docs/queue/open.md`** — if fully settled, remove the T_PAR1 item; if partially settled, update the "what to watch for" section with findings.

**`RESEARCH_STATE.md`** — append a new cycle entry (R28) with:
- Coder: aggregate TPS at N=1, 2, 4, 8 and the knee (where gains flatten)
- Thinker at max-num-seqs 1: TPS (should confirm ~77 t/s)
- Thinker at max-num-seqs 4: aggregate TPS, or OOM note with VRAM numbers
- Recommendation: optimal N for each tier

**`docs/decisions/settled.md`** — add a new SETTLED entry under parallelism section once the optimal N is confirmed.

---

## Termination conditions

Stop and hand back to research (write `## Open from testing` in `RESEARCH_STATE.md`) if ANY of these occur:

| Condition | What to write |
|-----------|---------------|
| Endpoint fails health check and won't start | "T_PAR1 BLOCKED — [coder/thinker] endpoint failed to start. Error: [paste exact error]. VRAM state: [nvidia-smi output]." |
| metrics.json has null fields after the run | "T_PAR1 DATA MISSING — endpoints appeared live but metrics.json shows null for [field]. Raw dir contents: [ls output]. Do not write results." |
| Coder OOMs at any N > 1 | "Coder OOM at max-num-seqs [N]. VRAM before: [X GB]. Error: [paste]. Lower N values measured: [include those]." |
| Thinker OOMs at max-num-seqs 4 | This is expected-possible. Record: "Thinker OOM at max-num-seqs 4. VRAM at time of OOM: [X GB]." Then continue with Step 3 on whatever data was collected at max-num-seqs 1. |
| Any CUDA error (not OOM) | "Unexpected CUDA error: [paste]. Do not retry." |
| TPS numbers look implausible | Define implausible: coder N=1 < 100 t/s or > 400 t/s; thinker N=1 < 40 t/s or > 120 t/s. If so: "Implausible TPS reading — possible endpoint serving wrong model or stale process. Raw data: [paste]." |

**Terminate normally** (session complete) when:
- Both Coder sweep and Thinker (both max-num-seqs=1 and max-num-seqs=4) have run
- metrics.json is non-null
- docs updated with real numbers
- `RESEARCH_STATE.md` has a new R28 cycle entry

---

## What NOT to do

- **Do not fill in estimated or "expected" numbers** if a run failed or data is missing. Write `UNKNOWN` or `NOT MEASURED`.
- **Do not mark T_PAR1 as DONE** unless both Coder and Thinker sweeps have real data.
- **Do not edit `CLAUDE.md`, `docs/arch/`, or `docs/decisions/`** — those are research-mode files.
- **Do not run any other benchmark items** (T3.4, T_KV3, etc.) — this session is T_PAR1 only.
- **Do not start the Convergence sweep** — already settled, `--skip-convergence` is required.

---

## Context you need

- Full project state: read `docs/INDEX.md` (150 lines, covers everything)
- T_PAR1 full procedure: `docs/queue/open.md` → T_PAR1 section
- Deploy commands and env vars: `docs/procedures/vllm-deploy.md`
- Prior fabrication incident: `docs/history/cycles.md` → R27
