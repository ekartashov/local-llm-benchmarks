# BENCH_23 — T_PQ2 Phase 1: PrismaQuant coder 4.75bit (Marlin fallback, no rebuild)

**Status:** READY
**Blocks:** T_PQ2 Phase 2 (CUDA 13.0 container rebuild for NVFP4 MoE)
**Blocked by:** nothing

---

## Title
T_PQ2 Phase 1 — Deploy `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` as a coder candidate: measure TPS, tool-call reliability, and quality on th02 vs the current AWQ baseline.

## Objective
Test whether `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` is a viable replacement for the current AWQ coder (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`). Phase 1 uses compressed-tensors Marlin fallback kernels — no CUDA container rebuild required. The primary question is whether PrismaQuant's GPTQ calibration improves reasoning quality while keeping tool-call reliability at production bar (≥95%). TPS is informational in Phase 1; Phase 2 (with NVFP4 kernels) is where TPS improvement is expected.

## Why this exists

**PrismaQuant vs AWQ quality:** PrismaQuant uses GPTQ calibration with scale sweeps, achieving ~0.33× RTN MSE relative to naive AWQ quantization. For the thinker (BENCH_12), this quality advantage justified a 26–33% TPS regression. For the coder, the TPS picture is different (baseline is much higher: 237–1205 t/s vs thinker's 77 t/s), so quality improvement at some TPS cost may still be acceptable.

**Marlin fallback path (Phase 1):** The 35B A3B model is MoE (Mixture of Experts). SM120 (RTX 5090) has an NVFP4 MoE grouped-GEMM kernel compatibility issue (FlashInfer #38718: compute_120a suffix vs compute_120f needed for TMA WS grouped GEMM). Phase 1 bypasses this by running through compressed-tensors with Marlin dequant kernels — the same path BENCH_12 used for the thinker successfully. CUDA 13.0 was released August 2025 but the FlashInfer issue persists; community Docker fix exists but upstream ETA is late Q2 2026.

**Why NOT MTP on the coder:** MTP `--speculative-config` MUST NOT be added to the coder. BENCH_14 (2026-05-01) proved that `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` breaks tool-call generation for A3B MoE coder: 0/3 tool-call probes produced valid tool calls. This is a hard constraint regardless of what the PrismaQuant model card suggests.

**TP=2 mandatory:** The coder requires TP=2. At TP=1, Qwen3.6-35B-A3B triggers Reasoning Collapse: Triton/FLA kernel shape mismatch in eager mode causes hallucination loops. TP=2 is the only production-viable configuration for this architecture on vLLM 0.19.0.

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — current production config, gotchas, port assignments
2. `docs/procedures/vllm-deploy.md` — deploy commands and env vars for vLLM
3. `docs/decisions/models.md` — AWQ coder production config; why TP=2 mandatory; why MTP forbidden
4. `results/T_PAR1_parallel_throughput_sweep_20260427T180802Z/summary.md` — AWQ coder baseline TPS (N=1: 240.9 t/s, N=4: 709.8 t/s, N=8: 1204.9 t/s)
5. `results/T2.5_coder_shootout_qwen36_35b_a3b_vs_qwen3coder30b_20260418T230329Z/summary.md` — AWQ coder quality baseline

## Prerequisites

```bash
echo "=== BENCH_23 Prerequisites ===" && \

# 1. PrismaQuant coder model downloaded (or will auto-download during deploy)
ls /srv/ai/models/hub/models--rdtand--Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm/ 2>/dev/null \
  && echo "[prereq] PQ coder model cache found" \
  || echo "[prereq] WARNING: PQ coder model not cached — deploy.sh will download (~30-40GB)" && \

# 2. No conflicting container on port 30000
EXISTING=$(podman ps --format "{{.Names}}\t{{.Ports}}" | grep "30000")
[ -z "${EXISTING}" ] \
  && echo "[prereq] Port 30000 free" \
  || echo "[prereq] WARNING: port 30000 occupied by: ${EXISTING} — stop it before deploy" && \

# 3. AWQ baseline metrics exist for comparison
ls results/T_PAR1_parallel_throughput_sweep_*/summary.md 2>/dev/null | head -3 \
  && echo "[prereq] AWQ baseline results found" \
  || echo "[prereq] WARNING: AWQ TPS baseline not found — note 240.9 t/s N=1 from docs" && \

# 4. th02 task prompt available
ls benchmarks/phase2_model_selection/tasks/coder/ 2>/dev/null \
  && echo "[prereq] Coder task dir found" \
  || echo "[prereq] WARNING: coder task dir not found — use th02 prompt from thinker suite" && \

# 5. GPU VRAM baseline
echo "[prereq] VRAM baseline:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

## Inputs required

- `infra/scripts/deploy.sh` — vLLM container deploy script
- Model `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` (HuggingFace download, ~30–40GB disk, pulled by deploy.sh if not cached at `/srv/ai/models/`)
- GPU0 + GPU1 free (32GB each needed; stop Arclight hot-pair before this test)
- AWQ coder TPS baseline from `results/T_PAR1_parallel_throughput_sweep_*/metrics.json` (or use the documented numbers: 240.9 t/s N=1, 709.8 t/s N=4, 1204.9 t/s N=8)
- th02 task prompt from `benchmarks/phase2_model_selection/tasks/` (or construct: EDF scheduling algorithm implementation + test cases)

## Fixed controls

| Control | Value |
|---------|-------|
| Model | rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm |
| Engine | vLLM 0.19.x |
| Placement | TP=2 GPU0+GPU1 (slot: tp2a) |
| VLLM_USE_V1 | 0 (mandatory — V1 unstable on Blackwell sm_120) |
| VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS | 1 (mandatory for TP=2 — prevents OOM at CUDA graph capture) |
| gpu-mem-util | 0.90 |
| Context size | 32768 |
| kv-cache-dtype | fp8 |
| speculative-config | **NOT SET** — MTP forbidden on A3B MoE coder (BENCH_14: 0/3 tool calls at MTP n=1) |
| tool-call-parser | qwen3_coder |
| reasoning-parser | qwen3 |
| Port | 30000 |
| TPS sweep concurrent clients N | 1, 4 |
| TPS sweep max_tokens | 150 |
| TPS sweep temperature | 0.0 |
| Tool-call probes | 5 |
| th02 max_tokens | 8192 |
| th02 temperature | 0.0 |

## Single variable under test

Model quantization: **PrismaQuant 4.75bit** (rdtand, GPTQ calibrated) vs AWQ 4bit baseline (cyankiwi). All other flags and hardware identical.

## Procedure

Skip flags (set to 1 to skip expensive steps on retry):
- `SKIP_DEPLOY=1` — skip deploy (use if PQ coder endpoint already running at port 30000)
- `SKIP_DOWNLOAD=1` — skip model download check (use if model already cached)

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_23_pq2_phase1_coder_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# AWQ baseline reference numbers (from T_PAR1 BENCH_01)
AWQ_TPS_N1=240.9
AWQ_TPS_N4=709.8
AWQ_TOOL_RATE=96.7  # % (T2.5: 29/30 tool calls)

# ===================================================================
# PHASE 1: Download model if not cached
# ===================================================================
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
if [ "${SKIP_DOWNLOAD}" = "0" ]; then
  echo "Checking/downloading rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm..."
  # deploy.sh will handle HF download automatically
  echo "Download will happen during deploy if model not cached."
fi

# ===================================================================
# PHASE 2: Deploy PrismaQuant coder
# ===================================================================
SKIP_DEPLOY=${SKIP_DEPLOY:-0}
if [ "${SKIP_DEPLOY}" = "0" ]; then
  # Stop existing coder if running
  EXISTING=$(podman ps --format "{{.Names}}" | grep -i "coder\|tp2\|35b\|30000" | head -1)
  [ -n "${EXISTING}" ] && { podman stop "${EXISTING}"; podman rm "${EXISTING}"; sleep 3; }

  echo "=== Deploying PrismaQuant coder ==="
  VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm tp2a rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm \
    --trust-remote-code \
    --gpu-mem-util 0.90 \
    --max-model-len 32768 \
    --kv-cache-dtype fp8 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    2>&1 | tee "${RESULTS_DIR}/deploy.log"
  # CRITICAL: NO --speculative-config — MTP breaks tool calls on A3B MoE coder (BENCH_14)

  CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|tp2\|35b" | head -1)
  podman logs -f "${CODER_CONTAINER}" 2>&1 | stdbuf -oL sed 's/^/[coder] /' &
  LOG_PID=$!

  for i in $(seq 1 300); do
    curl -sf http://localhost:30000/health 2>/dev/null && echo "[coder] HEALTH OK" && break
    sleep 1
  done
  kill "${LOG_PID}" 2>/dev/null

  curl -sf http://localhost:30000/health \
    || { echo "FATAL: PrismaQuant coder did not start"; \
         echo "Check deploy.log for: compressed-tensors load error, kernel fallback messages, OOM"; \
         exit 1; }

  # Record: did compressed-tensors load? what kernels selected?
  echo "=== Kernel/quantization selection from deploy log ===" | tee -a "${RESULTS_DIR}/deploy_notes.txt"
  grep -i "compressed\|marlin\|nvfp4\|fp4\|quantization\|kernel\|fallback" "${RESULTS_DIR}/deploy.log" \
    | head -30 | tee -a "${RESULTS_DIR}/deploy_notes.txt"
else
  echo "[skip] SKIP_DEPLOY=1"
  curl -sf http://localhost:30000/health || { echo "FATAL: endpoint not live"; exit 1; }
fi

# Record VRAM after load
echo "=== VRAM after PQ coder load ===" | tee "${RESULTS_DIR}/vram.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram.txt"

# ===================================================================
# PHASE 3: TPS sweep (N=1 and N=4)
# ===================================================================
echo "=== TPS sweep ===" | tee "${RESULTS_DIR}/tps.txt"

# Run the parallel throughput sweep script
# This script sends N concurrent requests and measures aggregate TPS
test -f benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  && SWEEP_SCRIPT=benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  || { echo "WARNING: T_PAR1 sweep script not found — using inline measurement"; }

if [ -f "${SWEEP_SCRIPT:-/dev/null}" ]; then
  for N in 1 4; do
    echo "--- N=${N} ---" | tee -a "${RESULTS_DIR}/tps.txt"
    MODEL=rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm \
    PORT=30000 N_CLIENTS=${N} MAX_TOKENS=150 TEMPERATURE=0 \
    bash "${SWEEP_SCRIPT}" 2>&1 | tee -a "${RESULTS_DIR}/tps.txt"
  done
else
  # Inline TPS measurement (fallback)
  echo "Using inline parallel measurement"
  for N in 1 4; do
    echo "--- N=${N} inline ---" | tee -a "${RESULTS_DIR}/tps.txt"
    PIDS=()
    START_MS=$(date +%s%3N)
    for CLIENT in $(seq 1 ${N}); do
      curl -sf http://localhost:30000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm\",
             \"prompt\":\"Write a Python function to sort a list using quicksort.\",
             \"max_tokens\":150,\"temperature\":0.0}" \
        > "${RESULTS_DIR}/tps_N${N}_client${CLIENT}.json" &
      PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "${PID}"; done
    END_MS=$(date +%s%3N)

    TOTAL_TOKENS=0
    for CLIENT in $(seq 1 ${N}); do
      T=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/tps_N${N}_client${CLIENT}.json')); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
      TOTAL_TOKENS=$((TOTAL_TOKENS + T))
    done
    ELAPSED_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
    AGG_TPS=$(python3 -c "print(round(${TOTAL_TOKENS} / ${ELAPSED_S}, 1)) if ${ELAPSED_S} > 0 else print(0)")

    echo "N=${N}: elapsed=${ELAPSED_S}s tokens=${TOTAL_TOKENS} aggregate_tps=${AGG_TPS}" | tee -a "${RESULTS_DIR}/tps.txt"
  done
fi

# ===================================================================
# PHASE 4: Tool-call reliability (5 probes)
# ===================================================================
echo "=== Tool-call probes (5) ===" | tee "${RESULTS_DIR}/tool_calls.txt"
echo "probe,has_tool_call,tool_name" > "${RESULTS_DIR}/tool_calls.csv"

TOOL_SYSTEM="You are a coding assistant. Use the provided tools to help the user."
TOOL_DEF='[{"type":"function","function":{"name":"execute_code","description":"Execute Python code","parameters":{"type":"object","properties":{"code":{"type":"string","description":"Python code to execute"}},"required":["code"]}}}]'

for PROBE in 1 2 3 4 5; do
  PROBE_PROMPT="Write a Python function to compute the Fibonacci sequence up to n terms, then call execute_code to run it with n=10."
  RESPONSE=$(curl -sf http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm\",
      \"messages\": [
        {\"role\": \"system\", \"content\": \"${TOOL_SYSTEM}\"},
        {\"role\": \"user\", \"content\": \"${PROBE_PROMPT}\"}
      ],
      \"tools\": ${TOOL_DEF},
      \"tool_choice\": \"auto\",
      \"max_tokens\": 512,
      \"temperature\": 0.0
    }" 2>/dev/null)

  echo "${RESPONSE}" > "${RESULTS_DIR}/tool_probe_${PROBE}.json"
  HAS_TOOL=$(echo "${RESPONSE}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d['choices'][0]['message']
    tc = msg.get('tool_calls', [])
    print('YES' if tc else 'NO')
except Exception as e:
    print('ERROR:' + str(e))
" 2>/dev/null)
  TOOL_NAME=$(echo "${RESPONSE}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls', [])
    print(tc[0]['function']['name'] if tc else 'none')
except:
    print('error')
" 2>/dev/null)

  echo "${PROBE},${HAS_TOOL},${TOOL_NAME}" >> "${RESULTS_DIR}/tool_calls.csv"
  echo "Probe ${PROBE}: has_tool=${HAS_TOOL} name=${TOOL_NAME}" | tee -a "${RESULTS_DIR}/tool_calls.txt"
done

PASS_COUNT=$(grep ",YES," "${RESULTS_DIR}/tool_calls.csv" | wc -l)
echo "Tool-call pass rate: ${PASS_COUNT}/5" | tee -a "${RESULTS_DIR}/tool_calls.txt"
echo "tool_call_pass_rate=${PASS_COUNT}/5" >> "${RESULTS_DIR}/tool_calls.txt"

# ===================================================================
# PHASE 5: Quality — th02 EDF scheduling algorithm
# ===================================================================
echo "=== th02 quality task ===" | tee "${RESULTS_DIR}/th02.txt"

# th02: Earliest Deadline First (EDF) scheduling — the primary quality discriminator
# (Same task used in BENCH_12, BENCH_19, BENCH_20 for thinker — adapted for coder role)
TH02_PROMPT="Implement a complete Earliest Deadline First (EDF) scheduler in Python. Requirements:
1. Task class with: task_id, arrival_time, execution_time, deadline attributes
2. EDF scheduler that processes a list of tasks, always selecting the task with the earliest deadline
3. Calculate and return: schedule order, average waiting time, missed deadline count, CPU utilization
4. Include a test with tasks: [(0,3,5), (1,2,4), (2,1,3), (3,4,8)] format: (arrival, exec, deadline)
5. Verify correctness: task (2,1,3) should execute first when available due to deadline=3

Provide complete, runnable Python code with the test included."

TH02_START_MS=$(date +%s%3N)
TH02_RESPONSE=$(curl -sf http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${TH02_PROMPT}\"}],
    \"max_tokens\": 8192,
    \"temperature\": 0.0
  }" 2>/dev/null)
TH02_END_MS=$(date +%s%3N)

echo "${TH02_RESPONSE}" > "${RESULTS_DIR}/th02_response.json"
TH02_TTFT_S=$(python3 -c "print(round(($TH02_END_MS - $TH02_START_MS) / 1000.0, 1))")
TH02_TOKENS=$(echo "${TH02_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens','UNKNOWN'))" 2>/dev/null)
TH02_TEXT=$(echo "${TH02_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:500])" 2>/dev/null)

echo "th02 TTFT: ${TH02_TTFT_S}s, tokens: ${TH02_TOKENS}" | tee -a "${RESULTS_DIR}/th02.txt"
echo "th02 response preview (first 500 chars):" | tee -a "${RESULTS_DIR}/th02.txt"
echo "${TH02_TEXT}" | tee -a "${RESULTS_DIR}/th02.txt"

echo "=== BENCH_23 measurement complete — results in ${RESULTS_DIR} ==="
echo "Now write summary.md — th02 quality scoring requires human review."
```

**Note on th02 quality scoring:** The th02 EDF response must be scored by a human (or research-mode Claude) for semantic correctness:
- Does the scheduler actually prioritize by earliest deadline?
- Is the task (2,1,3) executed first in the test case (deadline=3 is earliest)?
- Is the code runnable Python (no placeholder, no truncation)?
- Is the algorithmic logic correct (not just surface-level correct)?

Score 1–5 using the standard in `docs/decisions/scoring.md`. AWQ coder baseline on th02: PASS (97% tool-call rate, T2.5). Do NOT fabricate a score — write UNSCORED and include the raw text if you cannot score it definitively.

## Metrics to record

| Metric | Source file | Expected / reference value |
|--------|-------------|---------------------------|
| TPS N=1 (t/s) | `tps.txt` | AWQ baseline: 240.9 t/s; PQ Phase 1 may be slightly lower (Marlin) |
| TPS N=4 aggregate (t/s) | `tps.txt` | AWQ baseline: 709.8 t/s |
| TPS regression vs AWQ N=1 (%) | computed | Informational only in Phase 1; Marlin expected slightly slower |
| Tool-call pass rate | `tool_calls.csv` | ≥ 5/5 (95%+ required for pass) |
| th02 quality score (1–5) | `th02.txt` + human review | AWQ equivalent: ~4.5/5; PASS requires correct EDF logic |
| GPU0 VRAM at load (MiB) | `vram.txt` | ~29,000–30,000 MiB (similar to AWQ at 0.90) |
| GPU1 VRAM at load (MiB) | `vram.txt` | ~29,000–30,000 MiB (similar to AWQ at 0.90) |
| Kernel selection (Marlin vs NVFP4) | `deploy_notes.txt` | Expected: Marlin fallback (not NVFP4 in Phase 1) |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Model loads without error | Health 200 at port 30000 | Record OOM or load error; write Open from testing |
| compressed-tensors loaded (not raw weights) | Deploy log shows compressed-tensors or Marlin | Note if raw weights loaded instead (different perf profile) |
| TPS N=1 within 40% of AWQ (≥144 t/s) | PASS | INCONCLUSIVE — record TPS and note |
| Tool-call pass rate ≥ 5/5 | Required for PASS | If < 4/5: FAIL — record which probes failed and response text |
| th02 EDF algorithm semantically correct | Pass: algorithm prioritizes by deadline, task (2,1,3) runs first | FAIL if logic is wrong; UNSCORED if code is incomplete |
| No --speculative-config in deploy | Verified from deploy.log / container inspect | CRITICAL: if spec-config present, stop test, report contamination |

## Artifacts to write

1. `results/BENCH_23_pq2_phase1_coder_<TIMESTAMP>/deploy.log` — full deploy output
2. `results/BENCH_23_pq2_phase1_coder_<TIMESTAMP>/deploy_notes.txt` — kernel selection excerpt
3. `results/BENCH_23_pq2_phase1_coder_<TIMESTAMP>/vram.txt` — VRAM at load
4. `results/BENCH_23_pq2_phase1_coder_<TIMESTAMP>/tps.txt` — TPS sweep results
5. `results/BENCH_23_pq2_phase1_coder_<TIMESTAMP>/tool_calls.csv` — per-probe tool-call results
6. `results/BENCH_23_pq2_phase1_coder_<TIMESTAMP>/th02_response.json` — full th02 raw response
7. `results/BENCH_23_pq2_phase1_coder_<TIMESTAMP>/tool_probe_1-5.json` — individual probe responses
8. `results/BENCH_23_pq2_phase1_coder_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_23 — T_PQ2 Phase 1: PrismaQuant coder 4.75bit — <TIMESTAMP>

## Environment
- Engine: vLLM <version from deploy.log>
- Model: rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm
- Config: TP=2 GPU0+1, V0 engine, fp8 KV, ctx 32768, NO speculative-config
- Kernel path: <Marlin / NVFP4 — from deploy_notes.txt>

## VRAM at load
| GPU | VRAM used (MiB) | VRAM free (MiB) |
|-----|----------------|----------------|
| 0   | <x>            | <x>            |
| 1   | <x>            | <x>            |

## TPS
| N (concurrent) | PQ 4.75bit (t/s) | AWQ 4bit baseline (t/s) | Delta (%) |
|----------------|-----------------|------------------------|-----------|
| 1              | <x>             | 240.9                  | <x>%      |
| 4              | <x>             | 709.8                  | <x>%      |

## Tool-call reliability
| Probe | has_tool_call | tool_name |
|-------|--------------|-----------|
| 1     | YES/NO       | <name>    |
| 2     | YES/NO       | <name>    |
| 3     | YES/NO       | <name>    |
| 4     | YES/NO       | <name>    |
| 5     | YES/NO       | <name>    |
| **Pass rate** | **<N>/5** | |
| AWQ baseline | 29/30 (96.7%) | |

## Quality — th02 EDF scheduling
| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| EDF logic correct (earliest deadline selected) | <x> / UNSCORED | |
| Test case (2,1,3) runs first correctly | PASS / FAIL / UNSCORED | |
| Code is runnable Python | PASS / FAIL | |
| Overall th02 score | <x> / UNSCORED | |
| AWQ baseline th02 score | ~4.5 (T2.5 PASS) | |

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| Model loads | PASS/FAIL | |
| Kernel: Marlin fallback confirmed | PASS/FAIL | From deploy_notes.txt |
| NO speculative-config in deploy | PASS/FAIL | |
| Tool-call pass rate ≥ 5/5 | PASS/FAIL | Actual: <N>/5 |
| th02 EDF semantically correct | PASS/FAIL/UNSCORED | |
| TPS within 40% of AWQ (≥144 t/s at N=1) | PASS/FAIL | Actual: <x> t/s |

## Verdict
PASS / FAIL / PARTIAL — <one sentence>

## Kernel selection note
<Paste or summarize deploy_notes.txt: which quantization path was selected — Marlin, NVFP4, or raw?>
This determines whether Phase 2 (CUDA rebuild for NVFP4) is expected to improve TPS.

## Incidental findings
<Any unexpected deploy error, quantization warning, VRAM anomaly, or other observation.>
<If nothing unusual: "none">

## Open from testing
<Any unexpected blocker or question for research mode.>
<If nothing: "none">
```

## Interpretation boundary

**You may:**
- Record TPS, tool-call pass rate, and VRAM at load
- Note the kernel selection from deploy logs (Marlin vs NVFP4 vs unknown)
- Copy the raw th02 response text into the summary for human scoring
- Write UNSCORED for th02 if you cannot definitively judge EDF algorithm correctness

**You may NOT:**
- Score th02 quality without carefully verifying EDF algorithmic correctness in the code
- Update `docs/decisions/models.md` or promote PrismaQuant coder to production
- Update `docs/arch/current.md` with new coder config
- Add `--speculative-config` to any coder deployment — this is forbidden (BENCH_14 FAIL)
- Compare against thinker PrismaQuant (different role, different architecture, different expected results)

## Stop condition

**Normal:** Model loads, TPS measured at N=1 and N=4, 5 tool-call probes complete, th02 response captured, `summary.md` written with raw th02 text included for human scoring.

**Abnormal:** Write `## Open from testing` in `RESEARCH_STATE.md` if:
- Model fails to load: `BENCH_23_LOAD_FAIL: rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm failed to load at TP=2. Error: [excerpt from deploy.log]. VRAM at failure: GPU0=[x]MiB GPU1=[y]MiB.`
- Tool-call pass rate 0/5 (all fail like AWQ MTP): `BENCH_23_TOOL_FAIL: 0/5 tool calls on PrismaQuant 4.75bit coder. Same failure as BENCH_14 (MTP n=1). This may indicate a compressed-tensors / structured-output interaction. Raw probe responses in tool_probe_*.json.`
- Model loads but produces clearly corrupted output (garbage text, wrong language): `BENCH_23_CORRUPT_OUTPUT: PrismaQuant 4.75bit coder produces corrupted output. Sample: [excerpt]. VRAM: [values]. May indicate quantization format incompatibility.`
