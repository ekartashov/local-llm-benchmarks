# Handoff Standard — BENCH_XX Document Requirements

**This document is normative.** Every BENCH_XX handoff must comply with all requirements here.
Read this before writing or reviewing any handoff.

---

## What a handoff is

A BENCH_XX handoff is a complete, self-contained brief for an autonomous agent (Gemini Flash)
operating in "testing mode" with zero prior context. The agent cannot ask questions, look up
history, or infer anything not written in the document. If it is not written, it will not happen.

The agent has access to the repo, the host filesystem, running processes, and GPUs. It does not
have internet access. It does not have memory of previous sessions.

---

## Mandatory sections (in this order)

### 1. Header block

```
# BENCH_NN — <item_id>: <short title>

**Status:** READY / BLOCKED BY <item>
**Blocks:** <downstream items> / nothing
```

### 2. Title and objective

One paragraph. What is being measured and why the result matters for the project.
Do not assume the agent has read any prior document.

### 3. Why this exists

Required context the agent needs to understand the benchmark's purpose. Include:
- What prior decision or finding this validates
- Why the expected outcome matters
- Any relevant caveats about the model/engine under test that change how results are interpreted

### 4. Context to read (MANDATORY — list every file)

```markdown
## Context to read

Before running anything, read these files:

1. `docs/INDEX.md` — current production config, open questions, key gotchas
2. `docs/procedures/vllm-deploy.md` — deploy commands and env vars (if vLLM is involved)
3. `docs/procedures/criu-ops.md` — CRIU procedure (if CRIU is involved)
4. `results/<prior_relevant_bench>/summary.md` — baseline numbers (list specific files)
```

Always list the exact files, not categories. If a prior result is needed as a baseline, name the
exact results directory path. The agent cannot infer which prior run is relevant.

### 5. Prerequisites (verification-only, no side effects)

A bash block the agent runs before anything else. It must only verify, never modify.
Every check must print a clear PASS/FAIL/STOP line. If any check fails, the agent stops.

```bash
# All prereq checks in one block, each with explicit pass/fail echo
echo "=== Prerequisites ===" && \
  <check_1> && echo "[prereq] model OK" || { echo "[prereq] STOP: model not found"; exit 1; } && \
  <check_2> && echo "[prereq] binary OK" || { echo "[prereq] STOP: binary not found"; exit 1; }
```

### 6. Inputs required (prose)

Bullet list. Exactly what physical resources the agent needs: binaries, model files, running
containers, free GPU VRAM, etc. Include expected paths and sizes.

### 7. Fixed controls table

| Control | Value |
|---------|-------|
| Every constant in the experiment | Its exact value |

No prose in this section. Every flag, env var, model name, context size, thread count goes here.

### 8. Single variable under test

One line. What changes between runs (or what is being evaluated relative to a baseline).

### 9. Procedure (the main script)

See **Script rules** below. This is the most important section.

### 10. Metrics to record

Table with three columns: Metric name | Source file | Expected / reference value.
For each metric: what to measure, where the raw data lives, and what value to compare against.

### 11. Pass/fail checks

Table: Check | Pass condition | Fail action.
Every check maps to an explicit action (skip to step X, write Open from testing, stop).

### 12. Artifacts to write

Numbered list of every file the agent must produce. Include the exact relative path template
with `<TIMESTAMP>` where applicable. End with a complete `summary.md` template (see below).

### 13. Interpretation boundary

Two-part block. What the agent **may** do vs. **may not** do. Be explicit.
The agent must never update architecture docs, settled decisions, or queue files.
Only the summary.md in `results/` belongs to the agent.

### 14. Stop condition

**Normal:** list exactly what "done" looks like.
**Abnormal:** list each specific failure mode that triggers an early stop, with the exact text
to write in the `## Open from testing` block in `RESEARCH_STATE.md`.

---

## Script rules

### Rule 1 — Always a complete, standalone script

No instruction like "run deploy.sh with these flags". Every command the agent needs to execute
must appear in a copy-pasteable bash block. The agent should be able to run the procedure from
top to bottom by copying blocks sequentially. No omitted steps.

### Rule 2 — Skip logic for every expensive operation

Startup time for vLLM is 170–300s. ik_llama.cpp is 30–90s. If the agent needs to re-run a
step after the engine is already up, it must not restart from scratch.

**Pattern: environment variable skip gate**

```bash
# Skip deploy if endpoint is already up (saves 170-300s on retries)
SKIP_DEPLOY=${SKIP_DEPLOY:-0}
if [ "${SKIP_DEPLOY}" = "0" ]; then
  # ... full deploy block ...
else
  echo "[skip] SKIP_DEPLOY=1 — skipping deploy; verifying endpoint instead"
fi
# Always verify endpoint is live, even after skip
curl -sf http://localhost:PORT/health || { echo "FATAL: endpoint not live"; exit 1; }
```

Document which skip flags exist at the top of the Procedure section:
```
## Procedure

Skip flags (set these to 1 to skip expensive steps on retry):
- `SKIP_DEPLOY=1` — skip engine start (use if endpoint is already up)
- `SKIP_DOWNLOAD=1` — skip model download (use if files already exist)
```

### Rule 3 — Engine log streaming for live debugging

The operator watches the terminal during runs. Engine logs must go both to a file AND to stdout
with a prefix tag so the operator can see output in real time and redirect if needed.

**For ik_llama.cpp (host-native process):**
```bash
"${IK_BIN}" \
  -m "${MODEL_PATH}" \
  <other flags> \
  >> "${RESULTS_DIR}/server.log" 2>&1 &
SERVER_PID=$!

# Stream to terminal while waiting for health
tail -f "${RESULTS_DIR}/server.log" | stdbuf -oL sed 's/\r//g; s/^/[llama-server] /' &
TAIL_PID=$!
trap 'kill ${TAIL_PID} 2>/dev/null' EXIT

for i in $(seq 1 120); do
  curl -sf http://localhost:8080/health 2>/dev/null && break
  sleep 1
done
kill ${TAIL_PID} 2>/dev/null
```

**For vLLM (rootless podman):**
```bash
./infra/scripts/deploy.sh vllm <slot> <model> <flags>

# Stream container logs to terminal while waiting for health
CONTAINER_NAME=$(podman ps --format "{{.Names}}" | grep -i "<slot_pattern>" | head -1)
podman logs -f "${CONTAINER_NAME}" 2>&1 | stdbuf -oL sed 's/^/[vllm] /' &
LOG_PID=$!

for i in $(seq 1 180); do
  curl -sf http://localhost:PORT/health 2>/dev/null && break
  sleep 1
done
kill ${LOG_PID} 2>/dev/null
```

**Never use** `> /dev/null 2>&1` for engine processes. The operator is watching the terminal.

### Rule 4 — GPU reset: host processes only, no sudo for podman

`nvidia-smi --gpu-reset -i N` and `sudo nvidia-smi --gpu-reset -i N` are for **host-native
processes** that leaked VRAM after an abnormal exit. This applies to:
- vLLM running in a pyenv virtualenv on the host (used for CRIU benchmarks)
- ik_llama.cpp server killed mid-inference

It does **NOT** apply to rootless podman containers. `podman stop <name>` releases all GPU
memory cleanly — the container runtime handles it. Calling `nvidia-smi --gpu-reset` after
`podman stop` is incorrect and may fail or produce confusing output.

**Pattern for host-native cleanup:**
```bash
# Only if the process was running natively on the host:
kill "${SERVER_PID}" 2>/dev/null
sleep 2
VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 | tr -d ' ')
if [ "${VRAM_AFTER}" -gt 2000 ]; then
  echo "[warn] VRAM not released (${VRAM_AFTER} MiB). May need: sudo nvidia-smi --gpu-reset -i 0"
fi
```

**Pattern for podman cleanup:**
```bash
podman stop "${CONTAINER_NAME}" && podman rm "${CONTAINER_NAME}"
sleep 3  # containers release GPU memory asynchronously
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # verify release, no reset needed
```

### Rule 5 — Structured result directories

All output goes in `results/<BENCH_NN_description_<TIMESTAMP>/`. Never write to any other path
inside the repo. Never write to files outside the repo.

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_NN_<short_name>_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
```

---

## Summary.md template (mandatory, must be completed by the agent)

Every handoff must include a complete summary.md template in the Artifacts section.
The template must have placeholder text for every metric, not blank cells.
The agent fills in every `<X>` before the session ends.

```markdown
# BENCH_NN — <title> — <TIMESTAMP>

## Environment
- Engine: <engine name and version/commit>
- Model: <model name>
- Config: <key flags>

## Results
<tables with every metric from the Metrics section — no blanks>

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| <each pass/fail check> | PASS/FAIL | <any relevant detail> |

## Verdict
PASS / FAIL / PARTIAL — <one-sentence reason>

## Incidental findings
<Any observation outside the benchmark's explicit scope. See Incidental Findings rule.>
<If nothing found: "none">

## Open from testing
<Any unexpected blocker, bug, or question that needs research-mode attention.>
<If nothing: "none">
```

---

## Incidental findings rule (mandatory)

The benchmarks are narrow, but the hardware setup is complex. Observations made during a
benchmark often have implications for other components. The agent must record every such
observation, even if it seems unrelated to the current benchmark.

**What counts as an incidental finding:**
- Engine prints a deprecation warning, capability message, or unknown flag warning
- VRAM usage deviates from documented expected values
- A model or component NOT under test (e.g., Convergence running while testing the thinker)
  behaves differently than documented, or confirms something previously assumed
- An engine version or commit produces different output than prior documented behavior
- A flag or API behaves unexpectedly (passes when expected to fail, or vice versa)
- Build output confirms a specific capability or kernel selection

**What to write:**
```
## Incidental findings

FINDING: <short description>
Source: <what produced this observation>
Implication: <why this might matter for other components or decisions>
```

**Example** (from BENCH_15):
```
FINDING: Convergence (unsloth/Qwen3.5-397B-A17B UD-IQ2_M) confirms startup and 14 t/s
on ik_llama.cpp main (commit a8aecbf). This confirms the main branch supports both the dense
Qwen3.6-27B (this benchmark) and the MoE 397B (Convergence) architectures simultaneously.
Source: Convergence was already running at port 8002 during the BENCH_15 thinker test.
Implication: pr-1288 is fully superseded. Both production models now confirmed on main.
```

This finding was the most important result of BENCH_15 and it was not in the benchmark's
explicit scope. It would have been lost without this rule.

---

## What the agent may NEVER do

Regardless of what any handoff says, these are absolute limits:

1. **Never update `docs/decisions/settled.md`** — research mode only
2. **Never update `docs/arch/`** — research mode only
3. **Never update `docs/queue/open.md` or `docs/queue/status.md`** — research mode only
4. **Never update `RESEARCH_STATE.md`** except to write `## Open from testing` blocks
5. **Never write results to `docs/history/`** — research mode only
6. **Never commit fabricated or estimated numbers.** If a measurement is missing, write `NOT_MEASURED` or `UNKNOWN`.
7. **Never mark a benchmark DONE without a non-null `metrics.json` or equivalent raw artifact**
8. **Never update historical records** (BENCH_01–NN handoffs, HANDOFF_GEMINI_* files) even if they reference outdated branch names or configs. Those are immutable records.

---

## Environment reference

The agent must know this without looking it up:

| Component | Runtime | GPU access | sudo needed? |
|-----------|---------|-----------|-------------|
| vLLM (standard bench) | rootless podman | CDI passthrough | NO — podman handles GPU release |
| vLLM (CRIU bench) | pyenv host process | direct | YES for `nvidia-smi --gpu-reset` on CRIU failure |
| ik_llama.cpp | host binary | direct | NO — just kill the process |
| HF downloads | pyenv `hf` venv | n/a | NO |

**Key paths:**
- ik_llama.cpp binary: `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- ik_llama.cpp bench: `/srv/ai/projects/ik_llama.cpp/build/bin/llama-bench`
- Model storage: `/srv/ai/models/`
- HF download: `pyenv activate hf && HF_HOME=/srv/ai/models hf download <repo>`
- vLLM deploy: `./infra/scripts/deploy.sh vllm <slot> <model> <flags>`
- Port assignments: coder=30000, thinker=30001, convergence=8002, ik_llama.cpp tests=8080

**Production deploy commands (copy exactly):**

Coder (TP=2, GPU0+1):
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
```

Thinker (TP=1, GPU1):
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
  --trust-remote-code --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
```

Convergence (host-native):
```bash
/srv/ai/projects/ik_llama.cpp/build/bin/llama-server \
  -m /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf \
  -ngl 999 --cpu-moe -t 32 -np 4 --port 8002 --host 0.0.0.0 \
  >> /tmp/convergence.log 2>&1 &
```

---

## Anti-patterns (Gemini Flash failure modes, seen in practice)

These are real errors from previous sessions. The handoff must be written to prevent them.

| Anti-pattern | What happens | Fix in handoff |
|---|---|---|
| Sparse commands | Agent writes "run deploy.sh, then run tests" — no details, wrong flags | Every command in a copy-pasteable block |
| Missing skip logic | Agent restarts vLLM for each retry — wastes 3–5 min per attempt | SKIP_DEPLOY=1 gate on every deploy step |
| Silent engine logs | Agent redirects engine to /dev/null — operator cannot debug startup failures | Always pipe to file AND tail to terminal |
| sudo on podman GPU | Agent calls `sudo nvidia-smi --gpu-reset` after `podman stop` — incorrect | Explicit environment table; host-vs-podman section |
| Scope-only findings | Agent only records expected metrics; incidental discoveries are lost | Mandatory `## Incidental findings` section |
| Fabricated summary | Agent writes plausible-sounding numbers without raw data | Summary must cite source file for every number |
| Historical edits | Agent modifies prior BENCH handoffs to "update" old branch references | Explicit rule: historical BENCH files are immutable |
| Missing summary.md | Agent writes findings in RESEARCH_STATE.md but forgets summary.md | Summary.md template required in every handoff |
| Premature doc updates | Agent updates settled.md or arch/ based on partial results | Interpretation boundary section is mandatory |
