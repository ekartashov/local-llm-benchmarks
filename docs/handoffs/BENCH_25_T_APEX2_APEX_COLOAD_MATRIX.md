# BENCH_25 — T_APEX2: Tri-model Co-load with APEX Coder + Convergence

**Status:** BLOCKED BY BENCH_24 (T_APEX1 must PASS)
**Blocks:** nothing
**Blocked by:** BENCH_24 (T_APEX1)

---

## Title
T_APEX2 — With APEX I-Compact coder on GPU0 (smaller footprint than PQ's 27.9 GB), load Convergence across GPU0+GPU1 headroom and measure whether the tri-model co-load achieves ≥10 t/s Convergence TPS.

## Objective
Validate the I-Compact + fp8 KV topology lock. With PrismaQuant coder (27.9 GB) replaced by APEX I-Compact (17.3 GB), GPU0 gains ~10 GB of freed VRAM for Convergence attention layers. Combined with GPU1 headroom (~2.5 GB while thinker runs), this should support ngl ≈ 81 Convergence layers — projecting ~10 t/s vs the current 4.05 t/s co-load TPS. If achieved, this is a 2.5× improvement and the locked production co-load topology.

## Why this exists

Current co-load TPS is 4.05 t/s for Convergence (BENCH_21: -ngl 15, 15 attention layers on GPU). The bottleneck is VRAM: with PQ coder occupying ~30 GB on GPU0, only ~2 GB remains for Convergence attention layers on GPU0. GPU1 thinker leaves ~2.5 GB. Total ~4.5 GB → only ~15 layers → ~4 t/s.

With APEX I-Compact: GPU0 used ≈ 18.5 GB (weights + KV fp8) → ~13.5 GB headroom + GPU1 ~2.5 GB headroom = ~16 GB combined → ngl ≈ (16,000 − 6,247) / 113 ≈ 86 layers → expected ~10.2 t/s.

**Production VRAM model (BENCH_21):** Fixed overhead = 6,247 MiB. Per-layer cost = 113 MiB. TPS formula: 1 / (0.07148 + (94 − ngl) × 0.00211) t/s.

**Critical:** Use the actual GPU0 VRAM reading from BENCH_24 to compute updated ngl estimate before running. Do not assume 18,500 MiB — use the measured number.

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — current production config, port assignments
2. `docs/arch/convergence.md` — Convergence production launch command, GGML_CUDA_NO_PINNED=1 requirement
3. `results/BENCH_24_apex1_coder_*/summary.md` — **REQUIRED**: get actual GPU0 VRAM used by APEX coder; confirm T_APEX1 PASS before proceeding
4. `results/BENCH_21_*/summary.md` — Convergence VRAM model (6,247 MiB fixed + 113 MiB/layer) and thinker GPU1 headroom (~2,500 MiB)

## Prerequisites

```bash
echo "=== BENCH_25 Prerequisites ===" && \

# 1. APEX server must be running from BENCH_24
curl -sf http://localhost:8080/health 2>/dev/null \
  && echo "[prereq] APEX server OK at :8080" \
  || { echo "[prereq] STOP: APEX coder not running at :8080 — run BENCH_24 first"; exit 1; } && \

# 2. Thinker must be running
curl -sf http://localhost:30001/health 2>/dev/null \
  && echo "[prereq] Thinker OK at :30001" \
  || echo "[prereq] WARNING: Thinker not running — start before step 4" && \

# 3. Convergence must NOT be running yet
ss -tlnp | grep -q ':8002' \
  && echo "[prereq] WARNING: Convergence already running at :8002 — stop it first" \
  || echo "[prereq] port 8002 free for Convergence" && \

# 4. Convergence GGUF available
CONV_GGUF="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
[ -f "${CONV_GGUF}" ] \
  && echo "[prereq] Convergence GGUF OK" \
  || { echo "[prereq] STOP: Convergence GGUF not found at ${CONV_GGUF}"; exit 1; } && \

# 5. VRAM check: get current GPU0 used by APEX coder
echo "[prereq] Current VRAM:" && \
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader && \

# 6. Read GPU0 VRAM from BENCH_24 summary (or compute from current nvidia-smi)
GPU0_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 | tr -d ' ')
GPU1_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
GPU0_FREE=$((32768 - GPU0_USED))
GPU1_FREE=$((32768 - GPU1_USED))
echo "[prereq] GPU0 free for Convergence: ${GPU0_FREE} MiB" && \
echo "[prereq] GPU1 free for Convergence: ${GPU1_FREE} MiB" && \
COMBINED=$((GPU0_FREE + GPU1_FREE))
NGL_EST=$(python3 -c "print(max(0, int(($COMBINED - 6247) / 113)))")
echo "[prereq] Estimated Convergence ngl: ${NGL_EST} layers"
```

## Inputs required

- APEX I-Compact coder running on ik_llama.cpp at port 8080 (from BENCH_24)
- Thinker running on vLLM at port 30001 GPU1
- Convergence GGUF at `/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/`
- ik_llama.cpp binary at `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- Actual GPU0 VRAM reading from BENCH_24 summary.md

## Fixed controls

| Control | Value |
|---------|-------|
| Convergence model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M (same as production) |
| Convergence engine | ik_llama.cpp main (host-native process) |
| Convergence CPU MoE | `--cpu-moe` |
| Convergence threads | `-t 32` |
| Convergence slots | `-np 1` |
| Convergence context | `-c 4096` |
| Convergence KV | default (bf16) |
| Convergence ngl | auto: set to (GPU0_FREE + GPU1_FREE − 6247) / 113 (capped at 94) |
| GGML_CUDA_NO_PINNED | `1` (required for CRIU viability; keeps experts file-backed) |
| Convergence port | `8002` |
| CODER (APEX) | stays running at :8080, no changes |
| THINKER | stays running at :30001, no changes |
| TPS prompt | `"List the three laws of thermodynamics in one sentence each."` |
| TPS measurement | 5 sequential requests, average |
| Thinker correctness probe | th02 EDF task (same as BENCH_24) |

## Single variable under test

**Coder VRAM footprint:** APEX I-Compact (17.3 GB) vs PQ 4.75bit (27.9 GB). All other factors (thinker config, Convergence model, ik_llama.cpp engine) are held constant. The freed GPU0 VRAM goes to Convergence attention layers.

## Procedure

Skip flags:
- `SKIP_CONVERGENCE_DEPLOY=1` — skip Convergence startup (use if already running at :8002)

```bash
set -euo pipefail
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_25_apex2_coload_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
CONV_GGUF="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"

# Baselines
CONV_ISOLATED_TPS=13.99   # T_CV3 baseline
CONV_COLOAD_TPS=4.05      # BENCH_21 with PQ coder (-ngl 15)

# ===================================================================
# PHASE 1: Record pre-Convergence VRAM (APEX coder + thinker baseline)
# ===================================================================
echo "=== Pre-Convergence VRAM ===" | tee "${RESULTS_DIR}/vram.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram.txt"

GPU0_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 | tr -d ' ')
GPU1_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
GPU0_FREE=$((32768 - GPU0_USED))
GPU1_FREE=$((32768 - GPU1_USED))
COMBINED=$((GPU0_FREE + GPU1_FREE))
NGL_TARGET=$(python3 -c "print(min(94, max(0, int(($COMBINED - 6247) / 113))))")
TPS_PROJECTED=$(python3 -c "import math; ngl=${NGL_TARGET}; print(round(1.0 / (0.07148 + (94-ngl)*0.00211), 2)) if ngl > 0 else print(0)")

echo "gpu0_used_mib=${GPU0_USED}" | tee -a "${RESULTS_DIR}/metadata.txt"
echo "gpu1_used_mib=${GPU1_USED}" | tee -a "${RESULTS_DIR}/metadata.txt"
echo "gpu0_free_mib=${GPU0_FREE}" | tee -a "${RESULTS_DIR}/metadata.txt"
echo "gpu1_free_mib=${GPU1_FREE}" | tee -a "${RESULTS_DIR}/metadata.txt"
echo "combined_free_mib=${COMBINED}" | tee -a "${RESULTS_DIR}/metadata.txt"
echo "convergence_ngl_target=${NGL_TARGET}" | tee -a "${RESULTS_DIR}/metadata.txt"
echo "convergence_tps_projected=${TPS_PROJECTED}" | tee -a "${RESULTS_DIR}/metadata.txt"
echo "Pre-load: GPU0 free=${GPU0_FREE} MiB + GPU1 free=${GPU1_FREE} MiB = ${COMBINED} MiB combined → ngl_target=${NGL_TARGET} → projected TPS ${TPS_PROJECTED} t/s"

# ===================================================================
# PHASE 2: Deploy Convergence with computed ngl
# ===================================================================
SKIP_CONVERGENCE_DEPLOY=${SKIP_CONVERGENCE_DEPLOY:-0}
if [ "${SKIP_CONVERGENCE_DEPLOY}" = "0" ]; then
  echo "=== Starting Convergence at ngl=${NGL_TARGET} ===" | tee -a "${RESULTS_DIR}/metadata.txt"
  GGML_CUDA_NO_PINNED=1 "${IK_BIN}" \
    -m "${CONV_GGUF}" \
    -ngl "${NGL_TARGET}" --cpu-moe \
    -b 4096 -ub 2048 -t 32 -np 1 -c 4096 \
    --jinja --host 0.0.0.0 --port 8002 \
    >> "${RESULTS_DIR}/convergence.log" 2>&1 &
  CONV_PID=$!
  echo "convergence_pid=${CONV_PID}" >> "${RESULTS_DIR}/metadata.txt"

  tail -f "${RESULTS_DIR}/convergence.log" | stdbuf -oL sed 's/\r//g; s/^/[convergence] /' &
  TAIL_PID=$!
  trap 'kill ${TAIL_PID} 2>/dev/null; kill ${CONV_PID} 2>/dev/null' EXIT

  echo "Waiting for Convergence health (up to 120s)..."
  for i in $(seq 1 120); do
    curl -sf http://localhost:8002/health 2>/dev/null && echo "[convergence] HEALTH OK (ngl=${NGL_TARGET})" && break
    sleep 1
  done
  kill "${TAIL_PID}" 2>/dev/null

  curl -sf http://localhost:8002/health \
    || { echo "FATAL: Convergence did not start. Check ${RESULTS_DIR}/convergence.log"; exit 1; }

  # Read actual layer split from logs (ik_llama.cpp reports VRAM per device)
  echo "=== GPU layer allocation from startup log ===" | tee "${RESULTS_DIR}/layer_split.txt"
  grep -i "layer\|device\|cuda\|vram\|offload" "${RESULTS_DIR}/convergence.log" | head -30 | tee -a "${RESULTS_DIR}/layer_split.txt"
else
  echo "[skip] SKIP_CONVERGENCE_DEPLOY=1"
  curl -sf http://localhost:8002/health || { echo "FATAL: Convergence not live"; exit 1; }
fi

# Record VRAM with all three models loaded
echo "=== VRAM with tri-model load ===" | tee -a "${RESULTS_DIR}/vram.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram.txt"

# ===================================================================
# PHASE 3: Convergence TPS measurement (5 sequential requests)
# ===================================================================
echo "=== Convergence TPS (5 requests) ===" | tee "${RESULTS_DIR}/convergence_tps.txt"
echo "baseline_isolated=${CONV_ISOLATED_TPS} baseline_pq_coload=${CONV_COLOAD_TPS} projected=${TPS_PROJECTED}" >> "${RESULTS_DIR}/convergence_tps.txt"

CONV_MODEL=$(curl -sf http://localhost:8002/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "convergence")
TEST_PROMPT="List the three laws of thermodynamics in one sentence each."

TOTAL_TOKENS=0
TOTAL_S=0
for REP in 1 2 3 4 5; do
  START_MS=$(date +%s%3N)
  RESP=$(curl -sf http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${CONV_MODEL}\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":80,\"temperature\":0.0}" 2>/dev/null)
  END_MS=$(date +%s%3N)
  ELAPSED_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  TOKENS=$(echo "${RESP}" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(d.get('usage',{}).get('completion_tokens',0))
except: print(0)
" 2>/dev/null || echo 0)
  TPS=$(python3 -c "print(round(${TOKENS} / max(${ELAPSED_S}, 0.1), 2))")
  TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
  TOTAL_S=$(python3 -c "print(${TOTAL_S} + ${ELAPSED_S})")
  echo "rep=${REP} elapsed=${ELAPSED_S}s tokens=${TOKENS} tps=${TPS}" | tee -a "${RESULTS_DIR}/convergence_tps.txt"
  echo "${RESP}" > "${RESULTS_DIR}/conv_rep_${REP}.json"
done
AVG_TPS=$(python3 -c "print(round(${TOTAL_TOKENS} / max(${TOTAL_S}, 0.1), 2))")
echo "avg_tps=${AVG_TPS} total_tokens=${TOTAL_TOKENS} total_s=${TOTAL_S}" | tee -a "${RESULTS_DIR}/convergence_tps.txt"

# ===================================================================
# PHASE 4: Thinker correctness check (th02 while all three models loaded)
# ===================================================================
echo "=== Thinker correctness (th02 probe while tri-model load active) ===" | tee "${RESULTS_DIR}/thinker_check.txt"

THINKER_MODEL=$(curl -sf http://localhost:30001/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "thinker")
TH02_PROMPT="Implement a complete Earliest Deadline First (EDF) scheduler in Python. Requirements:
1. Task class with: task_id, arrival_time, execution_time, deadline attributes
2. EDF scheduler that processes a list of tasks, always selecting the task with the earliest deadline
3. Calculate and return: schedule order, average waiting time, missed deadline count
4. Include a test with tasks: [(0,3,5), (1,2,4), (2,1,3), (3,4,8)]
Provide complete, runnable Python code."

TH02_START_MS=$(date +%s%3N)
TH02_RESP=$(curl -sf http://localhost:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${THINKER_MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${TH02_PROMPT}\"}],
    \"max_tokens\": 4096,
    \"temperature\": 0.0
  }" 2>/dev/null)
TH02_END_MS=$(date +%s%3N)
echo "${TH02_RESP}" > "${RESULTS_DIR}/thinker_th02.json"
TH02_ELAPSED=$(python3 -c "print(round(($TH02_END_MS - $TH02_START_MS) / 1000.0, 1))")
TH02_TOKENS=$(echo "${TH02_RESP}" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(d.get('usage',{}).get('completion_tokens','UNKNOWN'))
except: print('UNKNOWN')
" 2>/dev/null)
echo "thinker th02 elapsed: ${TH02_ELAPSED}s tokens: ${TH02_TOKENS}" | tee -a "${RESULTS_DIR}/thinker_check.txt"

# ===================================================================
# PHASE 5: Bandwidth contention test (simultaneous coder + convergence)
# ===================================================================
echo "=== Bandwidth contention test ===" | tee "${RESULTS_DIR}/contention.txt"

CODER_MODEL=$(curl -sf http://localhost:8080/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "apex-compact")

# Fire coder request and convergence request simultaneously
CODER_START_MS=$(date +%s%3N)
curl -sf http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${CODER_MODEL}\",\"prompt\":\"Write a binary search function in Python.\",\"max_tokens\":100,\"temperature\":0.0}" \
  > "${RESULTS_DIR}/contention_coder.json" &
CODER_PID=$!

curl -sf http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${CONV_MODEL}\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":40,\"temperature\":0.0}" \
  > "${RESULTS_DIR}/contention_conv.json" &
CONV_PID_REQ=$!

wait "${CODER_PID}" 2>/dev/null
CODER_END_MS=$(date +%s%3N)
wait "${CONV_PID_REQ}" 2>/dev/null
CONV_END_MS=$(date +%s%3N)

CODER_ELAPSED=$(python3 -c "print(round(($CODER_END_MS - $CODER_START_MS) / 1000.0, 2))")
CONV_ELAPSED=$(python3 -c "print(round(($CONV_END_MS - $CODER_START_MS) / 1000.0, 2))")
CODER_TOKENS=$(python3 -c "
import json
try:
    d=json.load(open('${RESULTS_DIR}/contention_coder.json'))
    print(d.get('usage',{}).get('completion_tokens',0))
except: print(0)
" 2>/dev/null || echo 0)
CONV_TOKENS=$(python3 -c "
import json
try:
    d=json.load(open('${RESULTS_DIR}/contention_conv.json'))
    print(d.get('usage',{}).get('completion_tokens',0))
except: print(0)
" 2>/dev/null || echo 0)

echo "contention: coder ${CODER_TOKENS}tok/${CODER_ELAPSED}s conv ${CONV_TOKENS}tok/${CONV_ELAPSED}s" | tee -a "${RESULTS_DIR}/contention.txt"
echo "=== Final VRAM ===" && nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram.txt"

echo "=== BENCH_25 complete ==="
echo "Results in: ${RESULTS_DIR}"
```

## Metrics to record

| Metric | Source file | Expected / reference value |
|--------|-------------|---------------------------|
| ngl used for Convergence | `metadata.txt` | Computed from BENCH_24 actual VRAM; target ~81 |
| Convergence TPS average (t/s) | `convergence_tps.txt` | Target ≥ 10 t/s; baseline PQ co-load 4.05 t/s; isolated 13.99 t/s |
| Convergence TPS improvement vs PQ co-load | computed | Target ≥ 2× (≥ 8.1 t/s) |
| GPU0 VRAM after all three loaded (MiB) | `vram.txt` | APEX + KV + Convergence attn layers |
| GPU1 VRAM after all three loaded (MiB) | `vram.txt` | Thinker + Convergence attn layers overflow |
| Actual layer split (GPU0 / GPU1) | `layer_split.txt` | Record as reported by ik_llama.cpp startup |
| Thinker th02 correctness | `thinker_check.txt` | PASS (EDF correct) — confirms no interference |
| Thinker th02 token count | `thinker_check.txt` | Should be ~same as BENCH_23 baseline |
| Contention: coder TPS with Convergence running | `contention.txt` | Expect slight degradation vs isolated (PCIe bandwidth sharing) |
| Contention: Convergence TPS with coder running | `contention.txt` | Should match sequential avg (DDR5 is the bottleneck) |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Convergence starts | `/health` 200 | Record OOM; check GPU0 free MiB was ≥ 6,247 MiB |
| Convergence TPS ≥ 10 t/s | STRONG PASS — topology confirmed | |
| Convergence TPS 8–10 t/s | PARTIAL — acceptable, still 2× improvement | |
| Convergence TPS < 8 t/s | FAIL — investigate ngl vs VRAM mismatch | |
| Thinker th02 correct | Required — EDF scheduler returns correct output | FAIL → record interference; check GPU1 VRAM |
| Contention degradation < 30% for coder | Informational | Note if > 30% — possible PCIe saturation |

## Artifacts to write

1. `results/BENCH_25_apex2_coload_<TIMESTAMP>/metadata.txt` — VRAM math, ngl target, projections
2. `results/BENCH_25_apex2_coload_<TIMESTAMP>/vram.txt` — VRAM at each phase
3. `results/BENCH_25_apex2_coload_<TIMESTAMP>/convergence.log` — startup log with layer allocation
4. `results/BENCH_25_apex2_coload_<TIMESTAMP>/layer_split.txt` — GPU layer allocation excerpt
5. `results/BENCH_25_apex2_coload_<TIMESTAMP>/convergence_tps.txt` — TPS per rep + average
6. `results/BENCH_25_apex2_coload_<TIMESTAMP>/thinker_check.txt` — thinker correctness
7. `results/BENCH_25_apex2_coload_<TIMESTAMP>/contention.txt` — simultaneous request timings
8. `results/BENCH_25_apex2_coload_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_25 — T_APEX2: APEX Coder + Convergence + Thinker Co-load — <TIMESTAMP>

## Environment
- APEX coder: I-Compact on ik_llama.cpp at :8080 (from BENCH_24)
- Thinker: rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm on vLLM at :30001 GPU1
- Convergence: UD-IQ2_M on ik_llama.cpp at :8002 (host-native, GGML_CUDA_NO_PINNED=1)

## VRAM breakdown
| GPU | Before Convergence | After Convergence | Delta (Convergence attn) |
|-----|-------------------|------------------|--------------------------|
| 0   | <x> MiB (APEX)   | <y> MiB          | <y-x> MiB               |
| 1   | <x> MiB (thinker) | <y> MiB          | <y-x> MiB               |

## Convergence layer allocation
- ngl used: <N> layers (target was <target>)
- Layer split: GPU0 = <N0>, GPU1 = <N1> (from startup log)
- GPU0 headroom (pre-Convergence): <x> MiB
- GPU1 headroom (pre-Convergence): <x> MiB

## Convergence TPS
| Rep | Tokens | Elapsed (s) | TPS |
|-----|--------|-------------|-----|
| 1   | <x>    | <x>         | <x> |
| 2   | <x>    | <x>         | <x> |
| 3   | <x>    | <x>         | <x> |
| 4   | <x>    | <x>         | <x> |
| 5   | <x>    | <x>         | <x> |
| **avg** | | | **<x> t/s** |

| Metric | Value |
|--------|-------|
| Avg Convergence TPS (tri-load) | <x> t/s |
| vs PQ co-load baseline (4.05 t/s) | <x>× improvement |
| vs isolated baseline (13.99 t/s) | <x>% of isolated |
| Projected TPS at ngl=<N> | <x> t/s (from VRAM model) |

## Thinker correctness under load
- th02 elapsed: <x>s, tokens: <x>
- EDF logic correct: PASS / FAIL / UNSCORED

## Bandwidth contention
| Component | Isolated TPS | Under contention | Degradation |
|-----------|-------------|------------------|-------------|
| Coder | ~<from BENCH_24> t/s | <x> t/s | <x>% |
| Convergence | <avg from above> t/s | <x> t/s | <x>% |

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| Convergence starts at ngl=<N> | PASS/FAIL | |
| Convergence TPS ≥ 10 t/s | PASS/FAIL | Actual: <x> t/s |
| Thinker th02 correct | PASS/FAIL/UNSCORED | |
| No OOM | PASS/FAIL | |

## Verdict
PASS / FAIL / PARTIAL — <one sentence>

## Incidental findings
<GPU VRAM anomalies, layer split surprise, contention behavior.>
<If nothing: "none">

## Open from testing
<Any blocker for research mode. If TPS < 8 t/s, record ngl used and VRAM readings for revised projection.>
<If nothing: "none">
```

## Interpretation boundary

**You may:**
- Record Convergence TPS, VRAM split, layer allocation, thinker correctness
- Compute improvement ratio vs PQ co-load baseline (4.05 t/s)
- Note contention behavior (simultaneous coder + Convergence decode)

**You may NOT:**
- Update `docs/arch/current.md` with new co-load topology
- Promote this as the production config
- Stop the APEX server (it may be needed for BENCH_26)

## Stop condition

**Normal:** Convergence TPS measured (5 reps), thinker correctness confirmed, contention test done, summary.md written.

**Abnormal:** Write `## Open from testing` in `RESEARCH_STATE.md` if:
- Convergence OOM: `BENCH_25_OOM: Convergence failed to start at ngl=<N>. GPU0 free was <x> MiB, GPU1 free was <y> MiB. May need lower ngl. Reduce to ngl=<N-10> and retry.`
- TPS < 6 t/s: `BENCH_25_TPS_LOW: Convergence TPS=<x> t/s at ngl=<N> (expected ~<proj>). VRAM split GPU0/GPU1=<x>/<y> MiB. Possible BW contention or layer count discrepancy vs VRAM model.`
- Thinker th02 fails: `BENCH_25_THINKER_INTERFERENCE: Thinker th02 incorrect under tri-model load. GPU1 VRAM: <x> MiB. Thinker th02 response: [first 200 chars from thinker_th02.json].`
