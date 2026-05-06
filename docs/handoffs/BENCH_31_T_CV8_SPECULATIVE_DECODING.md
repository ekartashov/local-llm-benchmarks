# BENCH_31 — T_CV8: Convergence Speculative Decoding (MTP head check + bench)

**Status:** READY
**Blocks:** nothing
**Blocked by:** nothing

---

## Title
T_CV8 — Check UD-IQ2_M GGUF for embedded MTP draft heads; if present, benchmark speculative decoding TPS gain on ik_llama.cpp; also research DFlash (small draft model) feasibility.

## Objective
Determine whether Convergence can benefit from speculative decoding. Qwen3.5-397B-A17B was trained with MTP heads. If unsloth's UD-IQ2_M quantization preserves them, ik_llama.cpp can use them for in-model speculative decoding (draft 1–3 tokens ahead). Even +20% TPS would meaningfully improve the 13.99 t/s Convergence baseline.

## Why this exists

MTP (Multi-Token Prediction) heads are embedded additional output heads trained to predict 2–4 tokens ahead. When present in GGUF, ik_llama.cpp uses them for speculative decoding without a separate draft model — predicting N tokens, verifying in one GPU pass, accepting fast paths. For Convergence (DDR5-bound at 13.99 t/s), MTP gains depend on how often draft tokens are accepted; acceptance rate correlates with task predictability.

**If heads absent in UD-IQ2_M:** BENCH_31 closes as SKIPPED for MTP. Note for T_APEX4 (APEX Convergence may preserve heads — check when downloaded).

## Context to read

Before running anything, read these files in order:

1. `docs/arch/convergence.md` — production launch command, model path
2. `results/T_CV3_*/summary.md` or `results/BENCH_27_*/summary.md` — TPS baseline (13.99 t/s)
3. `results/BENCH_24_apex1_coder_*/mtp_check.txt` — if available: methodology for MTP head check (same Python approach)

## Prerequisites

```bash
echo "=== BENCH_31 Prerequisites ===" && \

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
[ -x "${IK_BIN}" ] && echo "[prereq] ik_llama.cpp OK" \
  || { echo "[prereq] STOP: ${IK_BIN} not found"; exit 1; } && \

CONV_SHARD1="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
[ -f "${CONV_SHARD1}" ] && echo "[prereq] Convergence GGUF shard 1 OK" \
  || { echo "[prereq] STOP: GGUF not found"; exit 1; } && \

# MTP flag discovery
echo "[prereq] MTP/speculative flags in ik_llama.cpp --help:" && \
"${IK_BIN}" --help 2>&1 | grep -iE "draft|specul|mtp|lookahead|accept" | head -20 && \

# gguf library
pyenv activate hf 2>/dev/null && \
python3 -c "from gguf import GGUFReader; print('[prereq] gguf library OK')" 2>/dev/null \
  || { pip install gguf -q; python3 -c "from gguf import GGUFReader; print('[prereq] gguf installed')" 2>/dev/null \
       || echo "[prereq] WARNING: gguf library unavailable — will use strings fallback"; } && \

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

## Inputs required

- ik_llama.cpp binary
- Convergence UD-IQ2_M GGUF (all 4 shards, check only needs shard 1 for metadata)
- pyenv `hf` venv with `gguf` library (or `strings` fallback)
- GPU0 free (≥ 1 GB for ngl=15 baseline; ≥ 17 GB for ngl=94)

## Fixed controls

| Control | Value |
|---------|-------|
| Model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M |
| Engine | ik_llama.cpp main |
| ngl | 94 (GPU0 only) for TPS bench; 15 acceptable if GPU0 constrained |
| CPU MoE | `--cpu-moe` |
| Threads | `-t 32` |
| Slots | `-np 1` |
| Context | `-c 4096` |
| Port | `8002` |
| TPS baseline | 13.99 t/s (T_CV3) |
| TPS reps | 5 per config |
| MTP draft variants | N=1, N=3 (if heads found) |
| Pass threshold | MTP TPS ≥ 1.20× baseline (≥ 16.8 t/s) |

## Single variable under test

**MTP draft depth** (N=0 baseline, N=1, N=3) vs baseline `--cpu-moe` without MTP. Secondary: whether UD-IQ2_M actually contains MTP heads.

## Procedure

Skip flags:
- `SKIP_MTP_BENCH=1` — skip MTP bench even if heads found (MTP check only)

```bash
set -euo pipefail
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_31_cv8_speculative_decoding_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
CONV_DIR="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M"
CONV_SHARD1="${CONV_DIR}/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"

BASELINE_TPS=13.99

# ===================================================================
# PHASE 1: MTP head check on UD-IQ2_M
# ===================================================================
echo "=== Phase 1: MTP head check on UD-IQ2_M ===" | tee "${RESULTS_DIR}/mtp_check.txt"

# Check all 4 shards — MTP heads may be in any shard
pyenv activate hf 2>/dev/null
pip install gguf -q 2>/dev/null || true

python3 - "${CONV_DIR}" << 'PYTHON' 2>&1 | tee -a "${RESULTS_DIR}/mtp_check.txt"
import sys, os, glob
model_dir = sys.argv[1]

try:
    from gguf import GGUFReader
except ImportError:
    print("gguf library not available — using strings fallback")
    sys.exit(1)

gguf_files = sorted(glob.glob(os.path.join(model_dir, "*.gguf")))
print(f"Checking {len(gguf_files)} GGUF shard(s): {[os.path.basename(f) for f in gguf_files]}")

all_mtp = []
for f in gguf_files:
    try:
        r = GGUFReader(f, "r")
        tensors = [t.name for t in r.tensors]
        mtp = [t for t in tensors if any(k in t.lower() for k in ['draft', 'mtp', 'specul', 'accept'])]
        if mtp:
            all_mtp.extend(mtp)
            print(f"  {os.path.basename(f)}: MTP tensors found: {len(mtp)}")
            for t in mtp[:10]:
                print(f"    - {t}")
        else:
            print(f"  {os.path.basename(f)}: no MTP tensors ({len(tensors)} total)")
    except Exception as e:
        print(f"  {os.path.basename(f)}: error reading — {e}")

if all_mtp:
    print(f"\nMTP_HEADS_FOUND: YES ({len(all_mtp)} total MTP tensors across all shards)")
else:
    print(f"\nMTP_HEADS_FOUND: NO")
    print("T_CV8 MTP bench: SKIPPED (no embedded MTP heads in UD-IQ2_M)")
    print("Note: If APEX Convergence is downloaded (T_APEX4), check its GGUF — original model may have heads but UD-IQ2_M omits them.")
PYTHON

MTP_FOUND=$(grep -c "MTP_HEADS_FOUND: YES" "${RESULTS_DIR}/mtp_check.txt" 2>/dev/null || echo 0)
echo "mtp_heads_found=${MTP_FOUND}" >> "${RESULTS_DIR}/metadata.txt"

# ===================================================================
# PHASE 2: MTP draft flag discovery
# ===================================================================
echo "=== Phase 2: MTP flag discovery ===" | tee "${RESULTS_DIR}/flag_audit.txt"
"${IK_BIN}" --help 2>&1 | grep -iE "draft|specul|mtp|lookahead" | head -30 | tee -a "${RESULTS_DIR}/flag_audit.txt"

MTP_FLAG=$("${IK_BIN}" --help 2>&1 | grep -oE "\-\-draft-max|\-\-speculative-max-tokens|\-\-draft" | head -1 || echo "--draft-max")
echo "mtp_flag_detected=${MTP_FLAG}" >> "${RESULTS_DIR}/metadata.txt"

# ===================================================================
# PHASE 3: MTP bench (only if heads found)
# ===================================================================
SKIP_MTP_BENCH=${SKIP_MTP_BENCH:-0}

echo "engine,mtp_n,tps_avg,vram_mib,status" > "${RESULTS_DIR}/mtp_sweep.csv"

if [ "${MTP_FOUND}" -gt 0 ] && [ "${SKIP_MTP_BENCH}" = "0" ]; then
  echo "=== Phase 3: MTP bench ==="

  stop_server() {
    local PID=$(ss -tlnp 2>/dev/null | grep ':8002' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1 || true)
    [ -n "${PID}" ] && { kill "${PID}" 2>/dev/null; sleep 5; }
  }

  measure_tps() {
    local REPS=$1
    local MODEL=$(curl -sf http://localhost:8002/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "convergence")
    local TOTAL=0; local TOTAL_S=0
    local PROMPT="List the three laws of thermodynamics in one sentence each."
    for R in $(seq 1 ${REPS}); do
      START_MS=$(date +%s%3N)
      RESP=$(curl -sf http://localhost:8002/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":80,\"temperature\":0.0}" \
        --max-time 120 2>/dev/null)
      END_MS=$(date +%s%3N)
      T=$(echo "${RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
      S=$(python3 -c "print(round(($END_MS - $START_MS)/1000.0,2))")
      TOTAL=$((TOTAL + T)); TOTAL_S=$(python3 -c "print(${TOTAL_S} + ${S})")
    done
    python3 -c "print(round(${TOTAL}/max(${TOTAL_S},0.1),2))"
  }

  for DRAFT_N in 0 1 3; do
    stop_server

    EXTRA=""
    [ "${DRAFT_N}" -gt 0 ] && EXTRA="${MTP_FLAG} ${DRAFT_N}"

    CUDA_VISIBLE_DEVICES=0 GGML_CUDA_NO_PINNED=1 "${IK_BIN}" \
      -m "${CONV_SHARD1}" \
      -ngl 94 --cpu-moe \
      -b 4096 -ub 2048 -t 32 -np 1 -c 4096 \
      ${EXTRA} \
      --jinja --host 0.0.0.0 --port 8002 \
      >> "${RESULTS_DIR}/server_mtp${DRAFT_N}.log" 2>&1 &
    SRV_PID=$!

    tail -f "${RESULTS_DIR}/server_mtp${DRAFT_N}.log" | stdbuf -oL sed 's/\r//g; s/^/[conv-mtp'${DRAFT_N}'] /' &
    TAIL_PID=$!

    STARTED=0
    for i in $(seq 1 120); do
      curl -sf http://localhost:8002/health 2>/dev/null && STARTED=1 && break
      kill -0 "${SRV_PID}" 2>/dev/null || break
      sleep 1
    done
    kill "${TAIL_PID}" 2>/dev/null

    if [ "${STARTED}" = "0" ]; then
      echo "ik,${DRAFT_N},FAIL,FAIL,STARTUP_FAIL" >> "${RESULTS_DIR}/mtp_sweep.csv"
      stop_server; continue
    fi

    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 | tr -d ' ')
    TPS=$(measure_tps 5)
    DELTA=$(python3 -c "print(round((${TPS} - ${BASELINE_TPS}) / ${BASELINE_TPS} * 100, 1))")
    echo "ik,${DRAFT_N},${TPS},${VRAM},OK" >> "${RESULTS_DIR}/mtp_sweep.csv"
    echo "draft-max=${DRAFT_N}: tps=${TPS} t/s delta=${DELTA}% vs baseline=${BASELINE_TPS} t/s"
    stop_server
  done

else
  if [ "${MTP_FOUND}" -eq 0 ]; then
    echo "MTP_BENCH_SKIPPED: MTP_HEADS_FOUND=NO in UD-IQ2_M" | tee "${RESULTS_DIR}/skip_reason.txt"
    echo "ik,N/A,N/A,N/A,SKIPPED_NO_HEADS" >> "${RESULTS_DIR}/mtp_sweep.csv"
  else
    echo "MTP_BENCH_SKIPPED: SKIP_MTP_BENCH=1" | tee "${RESULTS_DIR}/skip_reason.txt"
    echo "ik,N/A,N/A,N/A,SKIPPED_FLAG" >> "${RESULTS_DIR}/mtp_sweep.csv"
  fi
fi

# ===================================================================
# PHASE 4: DFlash feasibility note (research-only, no GPU needed)
# ===================================================================
echo "=== Phase 4: DFlash feasibility research ===" | tee "${RESULTS_DIR}/dflash_notes.txt"
echo "DFlash requires a small draft model of the same architecture family."
echo "Checking for any small Qwen3 or Qwen3.5 MoE GGUF already downloaded..."
find /srv/ai/models/hub/ -name "*.gguf" 2>/dev/null \
  | grep -i "qwen3\|qwen3.5" \
  | grep -iv "397B\|35B\|27B" \
  | head -20 | tee -a "${RESULTS_DIR}/dflash_notes.txt"
echo "Note: DFlash viability requires a compatible small draft model. Record any found above." \
  | tee -a "${RESULTS_DIR}/dflash_notes.txt"

cat "${RESULTS_DIR}/mtp_sweep.csv"
echo "=== BENCH_31 complete === Results in: ${RESULTS_DIR}"
```

## Metrics to record

| Metric | Source file | Expected |
|--------|-------------|---------|
| MTP heads found in UD-IQ2_M | `mtp_check.txt` | YES / NO — unknown |
| MTP flag name used | `flag_audit.txt` | `--draft-max` or actual name |
| Convergence TPS no-MTP baseline | `mtp_sweep.csv` | ~13.99 t/s (T_CV3) |
| Convergence TPS draft-max=1 | `mtp_sweep.csv` | Hope: ≥ 16.8 t/s (+20%) if heads present |
| Convergence TPS draft-max=3 | `mtp_sweep.csv` | May be higher or lower than N=1 |
| TPS delta vs baseline (%) | computed | Positive = MTP net win |
| DFlash candidate models found | `dflash_notes.txt` | Any small Qwen3 GGUF already on disk |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| MTP head check runs without error | gguf library works or strings fallback | Note if both fail |
| If heads found: MTP bench starts | Server health OK with draft flag | Record flag rejection error |
| If heads found: TPS ≥ 16.8 t/s (N=1) | PASS — +20% gain, proceed to production | |
| If heads absent: close as SKIPPED | Write SKIPPED verdict | No further action |

## Artifacts to write

1. `results/BENCH_31_cv8_speculative_decoding_<TIMESTAMP>/mtp_check.txt`
2. `results/BENCH_31_cv8_speculative_decoding_<TIMESTAMP>/flag_audit.txt`
3. `results/BENCH_31_cv8_speculative_decoding_<TIMESTAMP>/mtp_sweep.csv`
4. `results/BENCH_31_cv8_speculative_decoding_<TIMESTAMP>/server_mtp*.log` (if bench ran)
5. `results/BENCH_31_cv8_speculative_decoding_<TIMESTAMP>/dflash_notes.txt`
6. `results/BENCH_31_cv8_speculative_decoding_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_31 — T_CV8: Convergence Speculative Decoding — <TIMESTAMP>

## MTP head check (UD-IQ2_M)
- Shards checked: 4
- MTP_HEADS_FOUND: YES / NO
- If YES: tensor names: <list first 5>
- If NO: Bench status: **SKIPPED** — write T_CV8 as SKIPPED in queue/status.md (research mode)

## MTP bench results (if heads found)
| Engine | draft-max | TPS | VRAM (MiB) | vs baseline |
|--------|-----------|-----|------------|------------|
| ik | 0 (baseline) | <x> t/s | <x> | — |
| ik | 1 | <x> t/s | <x> | <±x>% |
| ik | 3 | <x> t/s | <x> | <±x>% |

Baseline reference: 13.99 t/s (T_CV3). Pass threshold: ≥ 16.8 t/s (+20%).

## DFlash feasibility
Small Qwen3 GGUFs found on disk: <list or "none">
DFlash viable: YES (model found) / NO (no compatible draft model) / UNKNOWN

## MTP flag used
Flag name: <actual flag from --help or --draft-max if default>

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| MTP head check completed | PASS/FAIL | |
| MTP bench TPS ≥ 16.8 t/s | PASS/FAIL/SKIPPED | Actual: <x> t/s |

## Verdict
PASS (MTP net-positive ≥ +20%) / FAIL (regression or tool issues) / SKIPPED (no heads in UD-IQ2_M) — <one sentence>

## Incidental findings
<GGUF structure observations, unexpected tensor count, DFlash model discovered.>
<If nothing: "none">

## Open from testing
<If heads absent, flag for research mode: "Consider checking APEX Convergence GGUF (T_APEX4) when downloaded — base model has MTP heads; UD-IQ2_M may omit them.">
<If bench ran: any anomaly.>
```

## Interpretation boundary

**You may:** Run MTP head check, run MTP bench if heads found, list DFlash candidates on disk.

**You may NOT:** Update `docs/arch/convergence.md` with MTP flag, or promote speculative decoding as production config — that is research mode.

## Stop condition

**Normal:** MTP head check complete. If NO heads → write SKIPPED summary and stop. If YES → bench runs for N=0/1/3, summary written.

**Abnormal:** `BENCH_31_GGUF_READ_ERROR: gguf library failed on all 4 shards. Errors: [from mtp_check.txt]. strings fallback also failed. Cannot confirm MTP head presence — T_CV8 inconclusive.`
