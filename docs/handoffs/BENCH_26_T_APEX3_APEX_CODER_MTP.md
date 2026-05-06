# BENCH_26 — T_APEX3: APEX Coder MTP on ik_llama.cpp

**Status:** BLOCKED BY BENCH_24 (T_APEX1 must PASS; MTP head check result must be YES)
**Blocks:** nothing
**Blocked by:** BENCH_24 (T_APEX1)

---

## Title
T_APEX3 — If APEX I-Compact GGUF contains MTP draft heads, measure speculative decoding TPS gain on ik_llama.cpp vs no-MTP baseline; verify tool-call reliability with MTP enabled.

## Objective
Determine whether ik_llama.cpp MTP is net-positive for the APEX coder — unlike vLLM MTP at TP=1 (BENCH_23b: −38.6% N=1, −50.6% N=4 due to SM120 grouped GEMM expert verification overhead). ik_llama.cpp uses its own CUDA kernels, so expert verification cost should be lower. If MTP is net-positive and tool calls remain reliable, this becomes the production APEX coder config.

**Stop immediately if BENCH_24 MTP check result was `MTP_HEADS_FOUND: NO`** — write SKIPPED in summary.md and close as SKIPPED in status.md. No GPU time needed.

## Why this exists

Qwen3.6-35B-A3B was trained with MTP heads (vLLM MTP worked at TP=1, BENCH_23b: 5/5 tool calls, 34.7 t/s). The vLLM MTP overhead is SM120 CUTLASS grouped GEMM expert verification — kernel-bound and specific to FlashInfer. ik_llama.cpp uses its own mul_mat CUDA kernels, so the same kernel-bound overhead does not apply. MTP may be net-positive here. If even +10% TPS with clean tool calls, MTP becomes the default APEX config.

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — port assignments, gotchas
2. `results/BENCH_24_apex1_coder_*/summary.md` — **READ mtp_check.txt result first**: if `MTP_HEADS_FOUND: NO`, stop immediately and write SKIPPED
3. `results/BENCH_24_apex1_coder_*/metadata.txt` — APEX model path, server startup notes

## Prerequisites

```bash
echo "=== BENCH_26 Prerequisites ===" && \

# 0. Check MTP head result from BENCH_24 — STOP if NO
BENCH24_DIR=$(ls -d results/BENCH_24_apex1_coder_* 2>/dev/null | sort | tail -1)
[ -z "${BENCH24_DIR}" ] && { echo "[prereq] STOP: BENCH_24 results not found"; exit 1; }
MTP_STATUS=$(grep "MTP_HEADS_FOUND" "${BENCH24_DIR}/mtp_check.txt" 2>/dev/null | head -1)
echo "[prereq] MTP status from BENCH_24: ${MTP_STATUS}"
echo "${MTP_STATUS}" | grep -q "YES" \
  || { echo "[prereq] STOP: MTP_HEADS_FOUND: NO — T_APEX3 must be closed as SKIPPED. No bench needed."; exit 0; } && \

# 1. APEX server running
curl -sf http://localhost:8080/health 2>/dev/null \
  && echo "[prereq] APEX server OK at :8080" \
  || { echo "[prereq] STOP: APEX coder not running — restart from BENCH_24 procedure"; exit 1; } && \

# 2. Discover MTP flag from ik_llama.cpp help
IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
echo "[prereq] MTP-related flags in llama-server --help:" && \
"${IK_BIN}" --help 2>&1 | grep -i "draft\|specul\|mtp\|lookahead" | head -20 && \

# 3. GPU VRAM
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

## Inputs required

- APEX I-Compact server running at port 8080 (from BENCH_24 — do NOT restart with MTP flags yet)
- APEX model path from `BENCH_24/metadata.txt`
- MTP head tensor names from `BENCH_24/mtp_check.txt` (for flag verification)
- ik_llama.cpp binary at `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`

## Fixed controls

| Control | Value |
|---------|-------|
| Model | APEX I-Compact (same file as BENCH_24) |
| Engine | ik_llama.cpp main |
| Flags (constant) | `-ngl 999 -t 32 -c 32768 --kv-type q8_0 --no-mmap --jinja --port 8080` |
| No-MTP baseline | Use N=1 agg TPS from BENCH_24 `tps.txt` — do NOT re-run |
| MTP variants | `--draft-max 1`, `--draft-max 3` (try flag name from `--help` output) |
| TPS measurement | Same method as BENCH_24: N=1 and N=4 parallel via server + curl |
| Tool-call probes | 5 (same as BENCH_24) |

## Single variable under test

**Speculative decoding depth:** no-MTP (BENCH_24 baseline) vs `--draft-max 1` vs `--draft-max 3`. All other flags identical.

## Procedure

Skip flags:
- `SKIP_MTP1=1` — skip draft-max=1 run (use if already measured)
- `SKIP_MTP3=1` — skip draft-max=3 run

```bash
set -euo pipefail
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_26_apex3_mtp_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"

# Get model path from BENCH_24
BENCH24_DIR=$(ls -d results/BENCH_24_apex1_coder_* 2>/dev/null | sort | tail -1)
APEX_FILE=$(grep "^apex_model_path=" "${BENCH24_DIR}/metadata.txt" 2>/dev/null | cut -d= -f2)
[ -z "${APEX_FILE}" ] && APEX_FILE=$(ls /srv/ai/models/hub/models--mudler--Qwen3.6-35B-A3B-APEX-GGUF/snapshots/*/Qwen3.6-35B-A3B-APEX-I-Compact*.gguf 2>/dev/null | head -1)
echo "apex_model_path=${APEX_FILE}" > "${RESULTS_DIR}/metadata.txt"

# Baseline from BENCH_24 (no-MTP)
BASELINE_N1=$(grep "N=1:" "${BENCH24_DIR}/tps.txt" 2>/dev/null | grep "agg_tps=" | head -1 | grep -o "agg_tps=[0-9.]*" | cut -d= -f2 || echo "UNKNOWN")
echo "baseline_no_mtp_n1_agg_tps=${BASELINE_N1}" >> "${RESULTS_DIR}/metadata.txt"
echo "No-MTP baseline (from BENCH_24): N=1 agg=${BASELINE_N1} t/s"

# Discover correct MTP flag name
MTP_FLAG=$("${IK_BIN}" --help 2>&1 | grep -i "draft-max\|draft_max" | head -1 | awk '{print $1}' | tr -d '[]')
[ -z "${MTP_FLAG}" ] && MTP_FLAG="--draft-max"
echo "mtp_flag=${MTP_FLAG}" >> "${RESULTS_DIR}/metadata.txt"
echo "Using MTP flag: ${MTP_FLAG}"

TPS_PROMPT="Write a Python function that implements merge sort, including docstring and type hints."
TOOL_DEF='[{"type":"function","function":{"name":"execute_code","description":"Execute Python code","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}}]'

run_tps() {
  local N=$1
  local MODEL=$2
  local PORT=$3
  local PIDS=()
  local START_MS=$(date +%s%3N)
  for CLIENT in $(seq 1 ${N}); do
    curl -sf "http://localhost:${PORT}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"prompt\":\"${TPS_PROMPT}\",\"max_tokens\":150,\"temperature\":0.0}" \
      > "/tmp/bench26_N${N}_c${CLIENT}.json" &
    PIDS+=($!)
  done
  for PID in "${PIDS[@]}"; do wait "${PID}" 2>/dev/null; done
  local END_MS=$(date +%s%3N)
  local ELAPSED_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  local TOTAL_TOKENS=0
  for CLIENT in $(seq 1 ${N}); do
    T=$(python3 -c "
import json
try:
    d=json.load(open('/tmp/bench26_N${N}_c${CLIENT}.json'))
    print(d.get('usage',{}).get('completion_tokens',0))
except: print(0)
" 2>/dev/null || echo 0)
    TOTAL_TOKENS=$((TOTAL_TOKENS + T))
  done
  python3 -c "print(round(${TOTAL_TOKENS} / max(${ELAPSED_S},0.1), 1))"
}

run_tool_probes() {
  local MODEL=$1
  local PORT=$2
  local LABEL=$3
  local PASS=0
  echo "probe,has_tool_call" > "${RESULTS_DIR}/tool_calls_${LABEL}.csv"
  for PROBE in 1 2 3 4 5; do
    RESP=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a fibonacci function then call execute_code to test it with n=10.\"}],\"tools\":${TOOL_DEF},\"tool_choice\":\"auto\",\"max_tokens\":512,\"temperature\":0.0}" 2>/dev/null)
    HAS_TOOL=$(echo "${RESP}" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    tc=d['choices'][0]['message'].get('tool_calls',[])
    print('YES' if tc else 'NO')
except: print('ERROR')
" 2>/dev/null)
    echo "${PROBE},${HAS_TOOL}" >> "${RESULTS_DIR}/tool_calls_${LABEL}.csv"
    [ "${HAS_TOOL}" = "YES" ] && PASS=$((PASS+1))
  done
  echo "${PASS}/5"
}

for DRAFT_N in 1 3; do
  SKIP_VAR="SKIP_MTP${DRAFT_N}"
  [ "${!SKIP_VAR:-0}" = "1" ] && { echo "[skip] SKIP_MTP${DRAFT_N}=1"; continue; }

  echo "=== MTP draft-max=${DRAFT_N} ===" | tee -a "${RESULTS_DIR}/results.txt"

  # Stop current server
  EXISTING_PID=$(ss -tlnp 2>/dev/null | grep ':8080' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1)
  [ -n "${EXISTING_PID}" ] && { kill "${EXISTING_PID}" 2>/dev/null; sleep 5; }

  # Start server with MTP
  CUDA_VISIBLE_DEVICES=0 "${IK_BIN}" \
    -m "${APEX_FILE}" \
    -ngl 999 -t 32 -np 4 -c 32768 \
    --kv-type q8_0 --no-mmap --jinja \
    "${MTP_FLAG}" "${DRAFT_N}" \
    --host 0.0.0.0 --port 8080 \
    >> "${RESULTS_DIR}/server_mtp${DRAFT_N}.log" 2>&1 &
  SRV_PID=$!

  tail -f "${RESULTS_DIR}/server_mtp${DRAFT_N}.log" | stdbuf -oL sed 's/\r//g; s/^/[apex-mtp'${DRAFT_N}'] /' &
  TAIL_PID=$!

  for i in $(seq 1 120); do curl -sf http://localhost:8080/health 2>/dev/null && break; sleep 1; done
  kill "${TAIL_PID}" 2>/dev/null
  curl -sf http://localhost:8080/health || { echo "FATAL: Server with MTP ${DRAFT_N} did not start"; continue; }

  MODEL_NAME=$(curl -sf http://localhost:8080/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "apex-compact")

  TPS_N1=$(run_tps 1 "${MODEL_NAME}" 8080)
  TPS_N4=$(run_tps 4 "${MODEL_NAME}" 8080)
  TOOL_PASS=$(run_tool_probes "${MODEL_NAME}" 8080 "mtp${DRAFT_N}")

  DELTA_N1=$(python3 -c "
baseline=${BASELINE_N1}
actual=${TPS_N1}
try:
    if baseline != 'UNKNOWN':
        print(round((float(actual) - float(baseline)) / float(baseline) * 100, 1))
    else:
        print('UNKNOWN')
except: print('UNKNOWN')
" 2>/dev/null)

  echo "draft_max=${DRAFT_N} tps_n1_agg=${TPS_N1} tps_n4_agg=${TPS_N4} tool_pass=${TOOL_PASS} delta_n1_pct=${DELTA_N1}" | tee -a "${RESULTS_DIR}/results.txt"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader >> "${RESULTS_DIR}/results.txt"
done

echo "=== BENCH_26 complete === Results in: ${RESULTS_DIR}"
```

## Metrics to record

| Metric | Source file | Expected / reference value |
|--------|-------------|---------------------------|
| No-MTP baseline N=1 agg TPS | BENCH_24 `tps.txt` | From BENCH_24 actual |
| MTP draft-max=1 N=1 agg TPS | `results.txt` | Hope: > baseline; vLLM MTP was −38.6% |
| MTP draft-max=3 N=1 agg TPS | `results.txt` | Hope: > draft-max=1 |
| MTP draft-max=1 N=4 agg TPS | `results.txt` | Hope: > baseline N=4 |
| TPS delta vs no-MTP N=1 (%) | `results.txt` | Positive = net win; negative = reject |
| Tool-call pass rate at draft-max=1 | `tool_calls_mtp1.csv` | Must be ≥ 4/5 to consider production |
| Tool-call pass rate at draft-max=3 | `tool_calls_mtp3.csv` | Must be ≥ 4/5 to consider production |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| MTP heads found (gate) | YES from BENCH_24 | If NO: close as SKIPPED; no bench needed |
| MTP flag accepted | Server starts with flag | If unknown flag: check `--help`; try `--speculative-max-tokens` |
| MTP TPS > no-MTP | Net positive any draft depth | Record negative delta; MTP not recommended |
| Tool calls ≥ 4/5 with MTP | Required for production recommendation | If < 4/5: MTP breaks tool calls → reject |

## Artifacts to write

1. `results/BENCH_26_apex3_mtp_<TIMESTAMP>/metadata.txt`
2. `results/BENCH_26_apex3_mtp_<TIMESTAMP>/server_mtp1.log` + `server_mtp3.log`
3. `results/BENCH_26_apex3_mtp_<TIMESTAMP>/results.txt` — TPS and delta per variant
4. `results/BENCH_26_apex3_mtp_<TIMESTAMP>/tool_calls_mtp1.csv` + `tool_calls_mtp3.csv`
5. `results/BENCH_26_apex3_mtp_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_26 — T_APEX3: APEX Coder MTP on ik_llama.cpp — <TIMESTAMP>

## MTP head check result (from BENCH_24)
MTP_HEADS_FOUND: YES / NO
If NO: **Status: SKIPPED** — no MTP heads in I-Compact GGUF. T_APEX3 closed.

## TPS comparison
| Config | N=1 agg TPS | N=4 agg TPS | vs no-MTP N=1 | Tool pass |
|--------|-------------|-------------|--------------|-----------|
| No-MTP (BENCH_24) | <x> | <x> | baseline | 5/5 |
| draft-max=1 | <x> | <x> | <±x>% | <N>/5 |
| draft-max=3 | <x> | <x> | <±x>% | <N>/5 |

vLLM MTP baseline (BENCH_23b, TP=1): −38.6% N=1, −50.6% N=4 (kernel-bound)

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| MTP flag accepted | PASS/FAIL | Flag used: <flag name> |
| Net TPS positive (best variant) | PASS/FAIL | Best: draft-max=<N> at <x> t/s |
| Tool calls ≥ 4/5 (best variant) | PASS/FAIL | |

## Verdict
PASS (MTP net-positive + tool calls) / FAIL (MTP regression or tool call failure) / SKIPPED (no heads) — <one sentence>

## Recommended production config
<If PASS: "APEX I-Compact with --draft-max N" or "No-MTP is still better">
<If SKIPPED/FAIL: "No-MTP config from BENCH_24">

## Incidental findings
<none / any startup messages about MTP acceptance rate, draft token overhead, etc.>

## Open from testing
<If MTP flag name is different from --draft-max, record correct name for docs.>
<If nothing: "none">
```

## Interpretation boundary

**You may:** Record TPS delta, tool-call pass rate, flag name used.

**You may NOT:** Update `docs/decisions/models.md` or change the production APEX coder config — that is research mode after reading summary.md.

## Stop condition

**Normal:** If MTP_HEADS_FOUND=NO → write SKIPPED summary and stop. If YES → both draft-max variants tested, TPS and tool calls recorded, summary written.

**Abnormal:** `BENCH_26_MTP_FLAG_UNKNOWN: --draft-max not recognized. llama-server --help output for speculative: [excerpt].`
