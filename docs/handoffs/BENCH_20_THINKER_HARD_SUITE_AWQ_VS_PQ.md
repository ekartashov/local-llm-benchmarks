# BENCH_20 — T_HARD1: Hard Systems Engineering Suite — AWQ vs PrismaQuant Head-to-Head

**Status:** READY
**Blocks:** nothing
**Blocked by:** nothing

---

## Title and objective

Run a 10-task hard systems engineering evaluation suite against both the current production
thinker (PrismaQuant 5.5bit) and the superseded baseline (AWQ 4bit), and save raw responses
for research-mode scoring. The suite tests deep reasoning chains across Linux internals,
distributed systems, networking, Proxmox, OpenStack, Ansible, and Kubernetes — domains
deliberately chosen to require multi-step expert inference rather than pattern retrieval.

The result matters because BENCH_12's quality evaluation (7/8 tasks, th01–th08 suite) showed
parity between PrismaQuant and AWQ. That suite was calibrated during AWQ selection and may be
too easy to reveal differentiation. This benchmark is designed to find the difficulty level at
which the models diverge, if one exists.

---

## Why this exists

PrismaQuant was promoted to production over AWQ on quality grounds despite a 26–33% TPS
regression. The quality evidence (BENCH_12) was based on tasks from the original thinker
selection phase. Those tasks have a known ceiling: th02 (EDF scheduling) is the hardest and
both models passed it. If PrismaQuant's GPTQ calibration genuinely improves reasoning quality
(as the method claims), the difference should be most visible on tasks that require chaining
5+ expert-level concepts where each step is non-obvious.

**This handoff's agent does NOT score responses.** Scoring requires research-mode judgment
against gold answers. The agent's job is to run both models, save raw responses, and write a
summary.md with response paths. Claude in research mode will score using the gold answers in
`benchmarks/phase2_model_selection/tasks/thinker_hard/gold/`.

---

## Context to read

Before running anything, read these files:

1. `docs/INDEX.md` — current production config, open questions, key gotchas
2. `docs/procedures/vllm-deploy.md` — deploy commands and env vars
3. `results/BENCH_12_prismaquant_thinker_*/summary.md` — PrismaQuant quality baseline
4. `benchmarks/phase2_model_selection/tasks/thinker_hard/` — the 10 task files (scan filenames to confirm all 10 are present before starting)

---

## Prerequisites

```bash
echo "=== Prerequisites ===" && \

# 1. Task files present (all 10)
TASK_COUNT=$(ls benchmarks/phase2_model_selection/tasks/thinker_hard/*.json 2>/dev/null | wc -l)
echo "[prereq] thinker_hard tasks found: ${TASK_COUNT}"
[ "${TASK_COUNT}" -ge 10 ] \
  && echo "[prereq] Task files OK" \
  || { echo "[prereq] STOP: expected 10 task files, found ${TASK_COUNT}"; exit 1; } && \

# 2. Gold answers present
GOLD_COUNT=$(ls benchmarks/phase2_model_selection/tasks/thinker_hard/gold/*.json 2>/dev/null | wc -l)
echo "[prereq] Gold answer files found: ${GOLD_COUNT}" && \

# 3. bench.py importable
python3 -c "import benchmarks.phase2_model_selection.bench; print('[prereq] bench.py OK')" \
  || { echo "[prereq] WARN: bench.py import failed — will use curl fallback in procedure"; } && \

# 4. PrismaQuant model available
ls /srv/ai/models/hub/ | grep -qi "prismaquant" \
  && echo "[prereq] PrismaQuant model files OK" \
  || { echo "[prereq] STOP: PrismaQuant model not found at /srv/ai/models/"; exit 1; } && \

# 5. AWQ model available
ls /srv/ai/models/hub/ | grep -qi "Qwen3.6-27B-AWQ\|27B-AWQ" \
  && echo "[prereq] AWQ model files OK" \
  || { echo "[prereq] STOP: AWQ model (QuantTrio/Qwen3.6-27B-AWQ) not found — download first"; exit 1; } && \

# 6. GPU1 VRAM state
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== End prerequisites ==="
```

---

## Inputs required

- `benchmarks/phase2_model_selection/tasks/thinker_hard/*.json` — 10 task files
- `benchmarks/phase2_model_selection/tasks/thinker_hard/gold/*.json` — 10 gold answer files (not used during run; present for research-mode scoring)
- `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm` model files in `/srv/ai/models/`
- `QuantTrio/Qwen3.6-27B-AWQ` model files in `/srv/ai/models/`
- `infra/scripts/deploy.sh` at repo root
- GPU1 with ~24 GB VRAM free
- Port 30001 free

---

## Fixed controls

| Control | Value |
|---------|-------|
| Thinker port | 30001 |
| TP | 1, GPU1 |
| VLLM_USE_V1 | 0 |
| VLLM_ENGINE_ITERATOR_SOURCE | LEGACY (PrismaQuant only) |
| VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS | 1 |
| --gpu-mem-util | 0.90 |
| --ctx | 32768 |
| --kv-cache-dtype | fp8 |
| --enable-chunked-prefill | ON |
| --max-num-seqs | 1 (single request — full model attention per task) |
| --max-tokens per request | 16384 (complex reasoning tasks need room) |
| Concurrency | N=1 (tasks run sequentially) |
| Reps per task | 1 (quality eval, not throughput) |
| Tasks | all 10 in benchmarks/phase2_model_selection/tasks/thinker_hard/ |
| Scoring | NOT DONE in this handoff — research mode scores against gold/ |

---

## Single variable under test

Model: `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm` vs `QuantTrio/Qwen3.6-27B-AWQ`. All
other flags and task prompts identical.

---

## Procedure

Skip flags:
- `SKIP_DEPLOY_PQ=1` — skip PrismaQuant deploy (use if it's already running on port 30001)
- `SKIP_DEPLOY_AWQ=1` — skip AWQ deploy (use if AWQ is already running — unusual)

### Setup

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_20_thinker_hard_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/pq" "${RESULTS_DIR}/awq"
echo "Results dir: ${RESULTS_DIR}"

TASK_DIR="benchmarks/phase2_model_selection/tasks/thinker_hard"
THINKER_URL="http://localhost:30001/v1/chat/completions"
MAX_TOKENS=16384
```

### Step 1 — Deploy PrismaQuant thinker

```bash
SKIP_DEPLOY_PQ=${SKIP_DEPLOY_PQ:-0}
if [ "${SKIP_DEPLOY_PQ}" = "0" ]; then
  EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
  [ -n "${EXISTING}" ] && {
    echo "[deploy] Stopping existing: ${EXISTING}"
    podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 3
  }

  echo "[deploy] Starting PrismaQuant thinker ..."
  VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm gpu1 rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
    --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-seqs 1 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
    >> "${RESULTS_DIR}/deploy_pq.log" 2>&1 &

  sleep 5
  CONTAINER_NAME=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
  [ -n "${CONTAINER_NAME}" ] && \
    podman logs -f "${CONTAINER_NAME}" 2>&1 | stdbuf -oL sed 's/^/[vllm-pq] /' & LOG_PID=$!

  for i in $(seq 1 300); do
    curl -sf http://localhost:30001/health 2>/dev/null && { echo "[deploy] PQ thinker READY"; break; }
    sleep 1
  done
  [ -n "${LOG_PID}" ] && kill "${LOG_PID}" 2>/dev/null

  curl -sf http://localhost:30001/health \
    || { echo "FATAL: PQ thinker did not come up"; exit 1; }
else
  echo "[skip] SKIP_DEPLOY_PQ=1 — verifying endpoint"
  curl -sf http://localhost:30001/health || { echo "FATAL: endpoint not live"; exit 1; }
fi

PQ_MODEL_ID=$(curl -s http://localhost:30001/v1/models \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[deploy] Active model: ${PQ_MODEL_ID}"
echo "${PQ_MODEL_ID}" > "${RESULTS_DIR}/pq/model_id.txt"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "${RESULTS_DIR}/pq/vram.txt"
```

### Step 2 — Run all 10 tasks against PrismaQuant

```bash
echo "[run] Running 10 tasks against PrismaQuant ..."

for TASK_FILE in "${TASK_DIR}"/*.json; do
  TASK_ID=$(python3 -c "import json; d=json.load(open('${TASK_FILE}')); print(d['id'])")
  TASK_NUM=$(basename "${TASK_FILE}" .json | cut -d_ -f1)
  echo "[task] ${TASK_ID} ..."

  SYSTEM_PROMPT=$(python3 -c "import json; d=json.load(open('${TASK_FILE}')); print(d['system_prompt'])")
  USER_MESSAGE=$(python3 -c "import json; d=json.load(open('${TASK_FILE}')); print(d['user_message'])")

  python3 - <<PYEOF > "${RESULTS_DIR}/pq/${TASK_NUM}_${TASK_ID}.json"
import json, urllib.request, time

task = json.load(open("${TASK_FILE}"))
model_id = open("${RESULTS_DIR}/pq/model_id.txt").read().strip()

payload = json.dumps({
    "model": model_id,
    "messages": [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": task["user_message"]}
    ],
    "max_tokens": ${MAX_TOKENS},
    "temperature": 0.0,
    "stream": False
}).encode()

t0 = time.perf_counter()
req = urllib.request.Request(
    "${THINKER_URL}",
    data=payload,
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=600) as resp:
    body = json.loads(resp.read())
elapsed = time.perf_counter() - t0

out = {
    "task_id": task["id"],
    "model": model_id,
    "elapsed_s": round(elapsed, 2),
    "completion_tokens": body["usage"]["completion_tokens"],
    "prompt_tokens": body["usage"]["prompt_tokens"],
    "tps": round(body["usage"]["completion_tokens"] / elapsed, 1),
    "response": body["choices"][0]["message"]["content"],
    "finish_reason": body["choices"][0]["finish_reason"]
}
print(json.dumps(out, indent=2))
PYEOF

  STATUS=$?
  if [ "${STATUS}" = "0" ]; then
    TPS=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/pq/${TASK_NUM}_${TASK_ID}.json')); print(d['tps'])")
    FINISH=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/pq/${TASK_NUM}_${TASK_ID}.json')); print(d['finish_reason'])")
    echo "[task] ${TASK_ID}: done (${TPS} t/s, finish=${FINISH})"
  else
    echo "[task] ${TASK_ID}: FAILED — check ${RESULTS_DIR}/pq/${TASK_NUM}_${TASK_ID}.json"
  fi
done

echo "[run] PrismaQuant: all tasks complete"
ls -la "${RESULTS_DIR}/pq/"
```

### Step 3 — Redeploy AWQ thinker

```bash
SKIP_DEPLOY_AWQ=${SKIP_DEPLOY_AWQ:-0}
if [ "${SKIP_DEPLOY_AWQ}" = "0" ]; then
  echo "[deploy] Switching to AWQ thinker ..."
  EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
  [ -n "${EXISTING}" ] && {
    podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 3
  }
  # Verify VRAM is released
  sleep 2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
    --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-seqs 1 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    >> "${RESULTS_DIR}/deploy_awq.log" 2>&1 &

  sleep 5
  CONTAINER_NAME=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
  [ -n "${CONTAINER_NAME}" ] && \
    podman logs -f "${CONTAINER_NAME}" 2>&1 | stdbuf -oL sed 's/^/[vllm-awq] /' & LOG_PID=$!

  for i in $(seq 1 300); do
    curl -sf http://localhost:30001/health 2>/dev/null && { echo "[deploy] AWQ thinker READY"; break; }
    sleep 1
  done
  [ -n "${LOG_PID}" ] && kill "${LOG_PID}" 2>/dev/null

  curl -sf http://localhost:30001/health \
    || { echo "FATAL: AWQ thinker did not come up"; tail -30 "${RESULTS_DIR}/deploy_awq.log"; exit 1; }
else
  echo "[skip] SKIP_DEPLOY_AWQ=1 — verifying endpoint"
  curl -sf http://localhost:30001/health || { echo "FATAL: endpoint not live"; exit 1; }
fi

AWQ_MODEL_ID=$(curl -s http://localhost:30001/v1/models \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[deploy] Active model: ${AWQ_MODEL_ID}"
echo "${AWQ_MODEL_ID}" > "${RESULTS_DIR}/awq/model_id.txt"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "${RESULTS_DIR}/awq/vram.txt"
```

### Step 4 — Run all 10 tasks against AWQ

```bash
echo "[run] Running 10 tasks against AWQ ..."

for TASK_FILE in "${TASK_DIR}"/*.json; do
  TASK_ID=$(python3 -c "import json; d=json.load(open('${TASK_FILE}')); print(d['id'])")
  TASK_NUM=$(basename "${TASK_FILE}" .json | cut -d_ -f1)
  echo "[task] ${TASK_ID} ..."

  python3 - <<PYEOF > "${RESULTS_DIR}/awq/${TASK_NUM}_${TASK_ID}.json"
import json, urllib.request, time

task = json.load(open("${TASK_FILE}"))
model_id = open("${RESULTS_DIR}/awq/model_id.txt").read().strip()

payload = json.dumps({
    "model": model_id,
    "messages": [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": task["user_message"]}
    ],
    "max_tokens": ${MAX_TOKENS},
    "temperature": 0.0,
    "stream": False
}).encode()

t0 = time.perf_counter()
req = urllib.request.Request(
    "${THINKER_URL}",
    data=payload,
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=600) as resp:
    body = json.loads(resp.read())
elapsed = time.perf_counter() - t0

out = {
    "task_id": task["id"],
    "model": model_id,
    "elapsed_s": round(elapsed, 2),
    "completion_tokens": body["usage"]["completion_tokens"],
    "prompt_tokens": body["usage"]["prompt_tokens"],
    "tps": round(body["usage"]["completion_tokens"] / elapsed, 1),
    "response": body["choices"][0]["message"]["content"],
    "finish_reason": body["choices"][0]["finish_reason"]
}
print(json.dumps(out, indent=2))
PYEOF

  STATUS=$?
  if [ "${STATUS}" = "0" ]; then
    TPS=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/awq/${TASK_NUM}_${TASK_ID}.json')); print(d['tps'])")
    FINISH=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/awq/${TASK_NUM}_${TASK_ID}.json')); print(d['finish_reason'])")
    echo "[task] ${TASK_ID}: done (${TPS} t/s, finish=${FINISH})"
  else
    echo "[task] ${TASK_ID}: FAILED"
  fi
done

echo "[run] AWQ: all tasks complete"
ls -la "${RESULTS_DIR}/awq/"
```

### Step 5 — Restore production thinker (PrismaQuant — MANDATORY)

```bash
echo "[restore] Restoring production PrismaQuant thinker ..."
EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${EXISTING}" ] && podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 3

VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
  >> "${RESULTS_DIR}/deploy_restore.log" 2>&1 &

sleep 5
CONTAINER_NAME=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${CONTAINER_NAME}" ] && \
  podman logs -f "${CONTAINER_NAME}" 2>&1 | stdbuf -oL sed 's/^/[vllm-restore] /' & LOG_PID=$!

for i in $(seq 1 300); do
  curl -sf http://localhost:30001/health 2>/dev/null && { echo "[restore] PRODUCTION THINKER RESTORED"; break; }
  sleep 1
done
[ -n "${LOG_PID}" ] && kill "${LOG_PID}" 2>/dev/null

# Verify correct model is running (must be PrismaQuant, NOT AWQ)
RESTORED_MODEL=$(curl -s http://localhost:30001/v1/models \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[restore] Active model: ${RESTORED_MODEL}"
echo "${RESTORED_MODEL}" | grep -i "PrismaQuant\|prismaquant" \
  && echo "[restore] CORRECT — PrismaQuant confirmed" \
  || echo "[restore] WARNING: unexpected model ID — verify manually"
```

### Step 6 — Print response length summary

```bash
python3 - <<'PYEOF'
import json, os, glob

RESULTS_DIR = sorted(glob.glob("results/BENCH_20_thinker_hard_*"), reverse=True)[0]
print(f"\nResults: {RESULTS_DIR}\n")

tasks = sorted(glob.glob(f"{RESULTS_DIR}/pq/[0-9]*.json"))
task_ids = [json.load(open(t))["task_id"] for t in tasks]

print(f"{'Task':<40} {'PQ tokens':>10} {'PQ fin':>8} {'AWQ tokens':>11} {'AWQ fin':>9}")
print("-" * 82)

for t in tasks:
    pq = json.load(open(t))
    task_id = pq["task_id"]
    awq_path = t.replace("/pq/", "/awq/")
    if os.path.exists(awq_path):
        awq = json.load(open(awq_path))
        print(f"{task_id:<40} {pq['completion_tokens']:>10} {pq['finish_reason']:>8} {awq['completion_tokens']:>11} {awq['finish_reason']:>9}")
    else:
        print(f"{task_id:<40} {pq['completion_tokens']:>10} {pq['finish_reason']:>8} {'MISSING':>11}")

print()
print("NOTE: Do not score responses here. Research mode scores using gold/ files.")
PYEOF
```

---

## Metrics to record

| Metric | Source | Note |
|--------|--------|------|
| Tasks completed (PQ) | count of .json files in results/pq/ | Expected: 10 |
| Tasks completed (AWQ) | count of .json files in results/awq/ | Expected: 10 |
| finish_reason per task per model | each .json file | "stop" = complete; "length" = truncated at 16384 |
| completion_tokens per task per model | each .json file | Longer is not better — note for scorer |
| Any task that timed out (600s) | missing or error .json | Record in Open from testing |

**DO NOT attempt to score responses.** The scoring requires comparing each response against
`benchmarks/phase2_model_selection/tasks/thinker_hard/gold/<task_id>_gold.json` using a
rubric that requires research-mode judgment. Write the raw response files and stop.

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| All 10 PQ tasks complete | 10 .json files in results/pq/ with finish_reason | Record which failed; continue to AWQ run |
| All 10 AWQ tasks complete | 10 .json files in results/awq/ with finish_reason | Record which failed; continue to restore |
| No task truncated (finish_reason=length) | All finish_reason="stop" | Note truncated tasks — scorer needs to know reasoning was cut off |
| Production restored | /health 200 on 30001, PrismaQuant model ID | Redeploy manually; do not leave AWQ in production |

---

## Artifacts to write

1. `results/BENCH_20_thinker_hard_<TIMESTAMP>/pq/model_id.txt`
2. `results/BENCH_20_thinker_hard_<TIMESTAMP>/pq/vram.txt`
3. `results/BENCH_20_thinker_hard_<TIMESTAMP>/pq/<NN>_<task_id>.json` — one per task (10 files)
4. `results/BENCH_20_thinker_hard_<TIMESTAMP>/awq/model_id.txt`
5. `results/BENCH_20_thinker_hard_<TIMESTAMP>/awq/vram.txt`
6. `results/BENCH_20_thinker_hard_<TIMESTAMP>/awq/<NN>_<task_id>.json` — one per task (10 files)
7. `results/BENCH_20_thinker_hard_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_20 — Thinker Hard Suite — AWQ vs PrismaQuant — <TIMESTAMP>

## Environment
- PrismaQuant: rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm, TP=1 GPU1, V0 engine, fp8 KV
- AWQ: QuantTrio/Qwen3.6-27B-AWQ, TP=1 GPU1, V0 engine, fp8 KV
- max_tokens: 16384, temperature: 0.0, max-num-seqs: 1 (sequential)

## Task completion

| Task | PQ tokens | PQ finish | AWQ tokens | AWQ finish |
|------|-----------|-----------|------------|------------|
| th_h01_iou_sqpoll_cgroup | <X> | <stop/length> | <X> | <stop/length> |
| th_h02_tcp_paws_nat | <X> | <stop/length> | <X> | <stop/length> |
| th_h03_raft_asymmetric_partition | <X> | <stop/length> | <X> | <stop/length> |
| th_h04_xsnp_hitm_perf | <X> | <stop/length> | <X> | <stop/length> |
| th_h05_ext4_fsync_durability | <X> | <stop/length> | <X> | <stop/length> |
| th_h06_proxmox_numa_hugepages | <X> | <stop/length> | <X> | <stop/length> |
| th_h07_openstack_dvr_garp | <X> | <stop/length> | <X> | <stop/length> |
| th_h08_ansible_fact_cache_race | <X> | <stop/length> | <X> | <stop/length> |
| th_h09_k8s_hpa_vpa_conflict | <X> | <stop/length> | <X> | <stop/length> |
| th_h10_k8s_pdb_drain_deadlock | <X> | <stop/length> | <X> | <stop/length> |

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| 10 PQ tasks complete | PASS/FAIL | |
| 10 AWQ tasks complete | PASS/FAIL | |
| No truncated responses | PASS/FAIL | List any task with finish_reason=length |
| Production thinker restored | PASS/FAIL | |

## Verdict
PARTIAL/PASS — raw responses collected. Scoring pending research-mode review.

## Incidental findings
<Any observation outside this benchmark's explicit scope. Write one FINDING block per
observation. If nothing unusual observed: "none">

## Open from testing
<Any task that timed out, any deploy failure, any unexpected model behaviour. If none: "none">

## Scoring status
NOT SCORED. Research mode will score each response against:
  benchmarks/phase2_model_selection/tasks/thinker_hard/gold/<task_id>_gold.json
Raw responses: results/BENCH_20_thinker_hard_<TIMESTAMP>/pq/ and /awq/
```

---

## Interpretation boundary

**You may:**
- Record raw responses, token counts, finish reasons
- Note if any model consistently produces shorter or longer responses
- Record truncated responses (finish_reason=length) as a finding — they indicate the model
  reached 16384 tokens without concluding, which may itself be a quality signal
- Note any model that refuses to engage with a task or produces an obviously wrong format

**You may NOT:**
- Score responses against the gold answers
- Conclude which model performed better
- Update `docs/decisions/settled.md`, production config, or queue files
- Update RESEARCH_STATE.md except for the `## Open from testing` block if stopping abnormally

---

## Stop condition

**Normal:** 20 response files written (10 PQ + 10 AWQ), production PrismaQuant thinker
restored on port 30001, `summary.md` written with all completion_tokens and finish_reasons
filled in.

**Abnormal:** Write `## Open from testing` in `RESEARCH_STATE.md` and stop if:
- AWQ model cannot be deployed (missing files, OOM): write `BENCH_20 BLOCKED: AWQ deploy
  failed — <error>. PQ responses are in results/BENCH_20_.../pq/. Run AWQ separately.`
- More than 3 tasks time out (>600s per task): write `BENCH_20 PARTIAL: N tasks timed out
  on <model>. Likely cause: max_tokens=16384 causing very long generation on thinking model.
  Consider reducing to 8192 on retry.`
- Production thinker cannot be restored: write `BENCH_20 CRITICAL: Production thinker not
  restored. Port 30001 is serving <model> or is down. Manual intervention required.`
