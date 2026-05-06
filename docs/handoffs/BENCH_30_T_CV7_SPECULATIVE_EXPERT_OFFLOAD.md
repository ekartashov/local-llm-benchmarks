# BENCH_30 — T_CV7: Convergence Speculative Expert Offload Engine Eval

**Status:** READY
**Blocks:** nothing
**Blocked by:** nothing

---

## Title
T_CV7 — Audit speculative expert offload (SEO) flag support in ik_llama.cpp vs llama.cpp; measure Convergence TPS delta with SEO enabled on each engine; select one engine for production Convergence SEO.

## Objective
Speculative expert offload (SEO) prefetches MoE expert weights to VRAM during the *previous* token's computation, overlapping PCIe transfer with GPU work. Unlike standard `--cpu-moe` (which moves experts to VRAM only when needed), SEO can eliminate or reduce PCIe stall latency. ik_llama.cpp has a tuned adaptive threshold (32 × total/active experts ≈ 1,638 tokens for Convergence); llama.cpp uses a fixed 32-token threshold. This benchmark determines which engine gives better TPS for Convergence with SEO, and whether SEO produces any meaningful gain over baseline `--cpu-moe`.

## Why this exists

Convergence is DDR5-bandwidth-bound at 13.99 t/s (T_CV3). SEO does not reduce DDR5 reads — it overlaps PCIe *transfer* with GPU compute. If PCIe transfer is on the critical path (VRAM→CPU→VRAM for expert weights), SEO could help. If DDR5 read is the bottleneck (RAM→CPU), SEO does nothing. This benchmark settles which component dominates.

**ik_llama.cpp SEO note:** Adaptive threshold of 1,638 tokens for Convergence (512 total experts, 10 active: 32 × 512/10 ≈ 1,638). Below that threshold, experts stay CPU-side. Above it, SEO prefetch kicks in. Short prompts (<1,638 tokens) will see no difference from baseline.

**Do NOT use `-rtr` flag** — forces MoE mul_mat to CPU on ik_llama.cpp (known bug, separate from SEO).

## Context to read

Before running anything, read these files in order:

1. `docs/arch/convergence.md` — production launch command, model path
2. `results/T_CV3_*/summary.md` or `results/BENCH_27_*/summary.md` — Convergence TPS baseline (13.99 t/s at ngl=999 or 94)
3. `docs/INDEX.md` — gotchas, hardware BW specs (DDR5 ~83 GB/s actual)

## Prerequisites

```bash
echo "=== BENCH_30 Prerequisites ===" && \

# 1. ik_llama.cpp binary
IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
[ -x "${IK_BIN}" ] && echo "[prereq] ik_llama.cpp OK: ${IK_BIN}" \
  || { echo "[prereq] STOP: ${IK_BIN} not found"; exit 1; } && \

# 2. llama.cpp binary (upstream, may be at different path)
LLAMA_BIN=$(which llama-server 2>/dev/null || ls /srv/ai/projects/llama.cpp/build/bin/llama-server 2>/dev/null || echo "NOT_FOUND")
echo "[prereq] llama.cpp binary: ${LLAMA_BIN}" && \
[ "${LLAMA_BIN}" = "NOT_FOUND" ] && echo "[prereq] WARNING: llama.cpp not found — only ik_llama.cpp tested" || true && \

# 3. Audit SEO flags in ik_llama.cpp
echo "[prereq] SEO-related flags in ik_llama.cpp --help:"
"${IK_BIN}" --help 2>&1 | grep -iE "offload|expert|cpu.moe|rtr|prefetch|threshold" | head -30 && \

# 4. Audit SEO flags in llama.cpp (if available)
[ "${LLAMA_BIN}" != "NOT_FOUND" ] && {
  echo "[prereq] SEO-related flags in llama.cpp --help:"
  "${LLAMA_BIN}" --help 2>&1 | grep -iE "offload|expert|cpu.moe|rtr|prefetch|threshold" | head -30
} || true && \

# 5. Convergence GGUF and GPU VRAM
CONV_GGUF="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
[ -f "${CONV_GGUF}" ] && echo "[prereq] Convergence GGUF OK" \
  || { echo "[prereq] STOP: GGUF not found"; exit 1; } && \
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

**IMPORTANT — Step 0 flag audit:** Before writing commands, run the `--help` audit above. The exact SEO flag names may differ from the names in this document. Adapt the procedure to use the actual flag names found. Common candidates:
- ik_llama.cpp: `--expert-offload`, `--expert-offload-threshold`, `--lookup-cache-static`
- llama.cpp: `--expert-offload-device`, `--expert-offload-scale`
- If no SEO flag exists in one engine, note `SEO_NOT_SUPPORTED` for that engine and test only the other.

## Inputs required

- ik_llama.cpp binary at `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- llama.cpp binary (if installed; test skipped if absent)
- Convergence UD-IQ2_M GGUF
- GPU0 free (≥ 17 GB for ngl=94; ≥ 1 GB for ngl=15 baseline)

## Fixed controls

| Control | Value |
|---------|-------|
| Model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M |
| Port | `8002` |
| CPU MoE | `--cpu-moe` (baseline); SEO modifies how experts move to GPU |
| Threads | `-t 32` |
| Slots | `-np 1` |
| Context | `-c 4096` |
| ngl | 94 (GPU0-only) OR 999 if GPU0 fully free (same as T_CV3) |
| TPS prompt | `"List the three laws of thermodynamics in one sentence each."` |
| TPS reps | 5 per config |
| Long-context prompt | A 2,000-token prompt (required to exceed ik_llama.cpp 1,638-token threshold) |
| Long-context reps | 3 |
| -rtr flag | **NEVER USE** (forces all MoE mul_mat to CPU — ik_llama.cpp bug) |

## Single variable under test

**SEO enabled vs disabled** on each engine. Secondary: ik_llama.cpp vs llama.cpp.

## Procedure

Skip flags:
- `SKIP_LLAMA_CPP=1` — skip llama.cpp tests (if not installed or same result expected)
- `SKIP_LONG_CTX=1` — skip 2K-token prompt (tests only short-prompt TPS)

```bash
set -euo pipefail
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_30_cv7_speculative_expert_offload_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
LLAMA_BIN=$(which llama-server 2>/dev/null || ls /srv/ai/projects/llama.cpp/build/bin/llama-server 2>/dev/null || echo "NOT_FOUND")
CONV_GGUF="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"

# Step 1: Audit actual flag names (save to file for reference)
echo "=== ik_llama.cpp SEO flags ===" | tee "${RESULTS_DIR}/flag_audit.txt"
"${IK_BIN}" --help 2>&1 | grep -iE "offload|expert|cpu.moe|prefetch|threshold|rtr" \
  | head -40 | tee -a "${RESULTS_DIR}/flag_audit.txt"

[ "${LLAMA_BIN}" != "NOT_FOUND" ] && {
  echo "=== llama.cpp SEO flags ===" | tee -a "${RESULTS_DIR}/flag_audit.txt"
  "${LLAMA_BIN}" --help 2>&1 | grep -iE "offload|expert|cpu.moe|prefetch|threshold" \
    | head -40 | tee -a "${RESULTS_DIR}/flag_audit.txt"
}

echo "IMPORTANT: Review ${RESULTS_DIR}/flag_audit.txt before continuing."
echo "Update SEO_FLAG_IK and SEO_FLAG_LLAMA below with actual flag names."

# ===================================================================
# CONFIGURE: Set SEO flags based on flag audit output above
# ===================================================================
# AGENT: Update these based on flag_audit.txt output
SEO_FLAG_IK="--expert-offload"          # Adjust if flag name differs in audit
SEO_FLAG_LLAMA="--expert-offload-scale" # Adjust if flag name differs in audit

# Verify flags exist
IK_SEO_SUPPORTED=$("${IK_BIN}" --help 2>&1 | grep -c "${SEO_FLAG_IK#--}" || echo 0)
echo "ik_llama.cpp SEO flag '${SEO_FLAG_IK}' found in --help: ${IK_SEO_SUPPORTED}" | tee -a "${RESULTS_DIR}/flag_audit.txt"

echo "engine,seo_enabled,ctx_size,tps_avg,vram_mib" > "${RESULTS_DIR}/seo_sweep.csv"

short_prompt="List the three laws of thermodynamics in one sentence each."
long_prompt=$(python3 -c "
text = 'The principles of thermodynamics govern energy transformations in physical systems. ' * 200
print(text[:8000])
print('\nSummarize the above in one sentence.')
")

stop_server() {
  local PID=$(ss -tlnp 2>/dev/null | grep ':8002' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1 || true)
  [ -n "${PID}" ] && { kill "${PID}" 2>/dev/null; sleep 5; }
}

measure_tps() {
  local PROMPT=$1
  local REPS=$2
  local TOTAL=0
  local TOTAL_S=0
  local MODEL=$(curl -sf http://localhost:8002/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "convergence")
  for R in $(seq 1 ${REPS}); do
    START_MS=$(date +%s%3N)
    RESP=$(curl -sf http://localhost:8002/v1/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL}\",\"prompt\":$(python3 -c "import json; print(json.dumps(\"${PROMPT}\"))"),\"max_tokens\":80,\"temperature\":0.0}" \
      --max-time 120 2>/dev/null)
    END_MS=$(date +%s%3N)
    T=$(echo "${RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
    S=$(python3 -c "print(round(($END_MS - $START_MS)/1000.0, 2))")
    TOTAL=$((TOTAL + T))
    TOTAL_S=$(python3 -c "print(${TOTAL_S} + ${S})")
  done
  python3 -c "print(round(${TOTAL}/max(${TOTAL_S},0.1), 2))"
}

run_config() {
  local ENGINE=$1     # "ik" or "llama"
  local BINARY=$2
  local SEO_FLAG=$3   # "" for baseline, or the SEO flag string
  local SEO_LABEL=$4  # "baseline" or "seo"
  local NGL=${5:-94}

  echo "--- ${ENGINE} ${SEO_LABEL} ngl=${NGL} ---"
  stop_server

  EXTRA_FLAGS=""
  [ -n "${SEO_FLAG}" ] && EXTRA_FLAGS="${SEO_FLAG}"

  CUDA_VISIBLE_DEVICES=0 GGML_CUDA_NO_PINNED=1 "${BINARY}" \
    -m "${CONV_GGUF}" \
    -ngl "${NGL}" --cpu-moe \
    -b 4096 -ub 2048 -t 32 -np 1 -c 4096 \
    ${EXTRA_FLAGS} \
    --jinja --host 0.0.0.0 --port 8002 \
    >> "${RESULTS_DIR}/${ENGINE}_${SEO_LABEL}_ngl${NGL}.log" 2>&1 &
  SRV_PID=$!

  tail -f "${RESULTS_DIR}/${ENGINE}_${SEO_LABEL}_ngl${NGL}.log" | stdbuf -oL sed 's/\r//g; s/^/['${ENGINE}'-'${SEO_LABEL}'] /' &
  TAIL_PID=$!

  STARTED=0
  for i in $(seq 1 120); do
    curl -sf http://localhost:8002/health 2>/dev/null && STARTED=1 && break
    kill -0 "${SRV_PID}" 2>/dev/null || { echo "Server died"; break; }
    sleep 1
  done
  kill "${TAIL_PID}" 2>/dev/null

  if [ "${STARTED}" = "0" ]; then
    echo "${ENGINE},${SEO_LABEL},short,FAIL,FAIL" >> "${RESULTS_DIR}/seo_sweep.csv"
    stop_server; return
  fi

  VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 | tr -d ' ')
  TPS_SHORT=$(measure_tps "${short_prompt}" 5)
  echo "${ENGINE},${SEO_LABEL},short,${TPS_SHORT},${VRAM}" >> "${RESULTS_DIR}/seo_sweep.csv"
  echo "  short TPS: ${TPS_SHORT} t/s"

  SKIP_LONG_CTX=${SKIP_LONG_CTX:-0}
  if [ "${SKIP_LONG_CTX}" = "0" ]; then
    TPS_LONG=$(measure_tps "${long_prompt}" 3)
    echo "${ENGINE},${SEO_LABEL},long_2k,${TPS_LONG},${VRAM}" >> "${RESULTS_DIR}/seo_sweep.csv"
    echo "  long TPS: ${TPS_LONG} t/s (above ik 1,638-token threshold)"
  fi

  stop_server
}

# ik_llama.cpp: baseline then SEO
run_config "ik" "${IK_BIN}" "" "baseline" 94
[ "${IK_SEO_SUPPORTED}" -gt 0 ] \
  && run_config "ik" "${IK_BIN}" "${SEO_FLAG_IK}" "seo" 94 \
  || echo "ik,seo_not_supported,N/A,N/A,N/A" >> "${RESULTS_DIR}/seo_sweep.csv"

# llama.cpp: baseline then SEO (if available)
SKIP_LLAMA_CPP=${SKIP_LLAMA_CPP:-0}
if [ "${SKIP_LLAMA_CPP}" = "0" ] && [ "${LLAMA_BIN}" != "NOT_FOUND" ]; then
  LLAMA_SEO_SUPPORTED=$("${LLAMA_BIN}" --help 2>&1 | grep -c "${SEO_FLAG_LLAMA#--}" || echo 0)
  run_config "llama" "${LLAMA_BIN}" "" "baseline" 94
  [ "${LLAMA_SEO_SUPPORTED}" -gt 0 ] \
    && run_config "llama" "${LLAMA_BIN}" "${SEO_FLAG_LLAMA}" "seo" 94 \
    || echo "llama,seo_not_supported,N/A,N/A,N/A" >> "${RESULTS_DIR}/seo_sweep.csv"
else
  echo "llama,not_tested,N/A,N/A,N/A" >> "${RESULTS_DIR}/seo_sweep.csv"
fi

echo "=== Results ===" && cat "${RESULTS_DIR}/seo_sweep.csv"
echo "=== BENCH_30 complete === Results in: ${RESULTS_DIR}"
```

## Metrics to record

| Metric | Source file | Expected |
|--------|-------------|---------|
| ik_llama.cpp SEO flag name | `flag_audit.txt` | May differ from `--expert-offload` |
| llama.cpp SEO flag name | `flag_audit.txt` | May differ or not exist |
| ik baseline TPS short prompt | `seo_sweep.csv` | ~13.99 t/s (T_CV3 reference) |
| ik SEO TPS short prompt | `seo_sweep.csv` | Expect same (below 1,638-token threshold) |
| ik SEO TPS long prompt (2K) | `seo_sweep.csv` | Hope: > baseline; depends on PCIe vs DDR5 balance |
| llama baseline vs SEO TPS | `seo_sweep.csv` | Informational; ik has tuned threshold advantage |
| TPS delta SEO vs baseline (%) | computed | Positive = SEO helps; 0 = DDR5 is binding |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| ik baseline TPS ≥ 12 t/s | Confirms ngl=94 working | |
| SEO flag identified | Found in --help | Note `SEO_NOT_SUPPORTED` — report to research mode |
| SEO starts without error | Server health OK with flag | Record flag name that caused error |
| Engine winner identified | One engine shows higher SEO TPS | If tied: prefer ik_llama.cpp (tuned threshold) |

## Artifacts to write

1. `results/BENCH_30_cv7_speculative_expert_offload_<TIMESTAMP>/flag_audit.txt`
2. `results/BENCH_30_cv7_speculative_expert_offload_<TIMESTAMP>/seo_sweep.csv`
3. `results/BENCH_30_cv7_speculative_expert_offload_<TIMESTAMP>/ik_*.log` + `llama_*.log`
4. `results/BENCH_30_cv7_speculative_expert_offload_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_30 — T_CV7: Speculative Expert Offload Engine Eval — <TIMESTAMP>

## Flag audit results
- ik_llama.cpp SEO flag: <actual flag name> (found in --help: YES/NO)
- llama.cpp SEO flag: <actual flag name> (found in --help: YES/NO/NOT_TESTED)
- ik_llama.cpp adaptive threshold (Convergence): 32 × (512/10) ≈ 1,638 tokens

## SEO TPS results
| Engine | SEO | Prompt size | TPS | VRAM (MiB) | vs ik-baseline |
|--------|-----|-------------|-----|------------|----------------|
| ik | OFF (baseline) | short | <x> | <x> | — |
| ik | ON | short | <x> | <x> | <±x>% |
| ik | OFF | long (2K) | <x> | <x> | — |
| ik | ON | long (2K) | <x> | <x> | <±x>% |
| llama | OFF | short | <x> / NOT_TESTED | <x> | |
| llama | ON | short | <x> / NOT_SUPPORTED | <x> | |

## Verdict
SEO net positive: YES / NO / NOT_SUPPORTED
Recommended engine: ik_llama.cpp / llama.cpp / no change
Reasoning: <one sentence — e.g., "DDR5 is the bottleneck; PCIe overlap does not help" or "ik SEO +12% at 2K tokens, PCIe partially on critical path">

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| ik baseline TPS ≥ 12 t/s | PASS/FAIL | |
| SEO flag accepted | PASS/FAIL/NOT_SUPPORTED | Actual flag: <name> |
| Engine winner identified | PASS/INCONCLUSIVE | |

## Incidental findings
<Flag names that differ from docs, unexpected SEO startup behavior, VRAM delta with SEO enabled.>
<If nothing: "none">

## Open from testing
<If SEO not supported on either engine, or if TPS anomaly observed, describe for research mode.>
<If nothing: "none">
```

## Interpretation boundary

**You may:** Record flag names, TPS values, VRAM, compute delta percentages.

**You may NOT:** Update `docs/arch/convergence.md` with SEO flag, or declare a "winner" engine for production — that is research mode after reading summary.

## Stop condition

**Normal:** ik baseline + SEO tested (both short + long prompt), llama.cpp tested or noted as absent, summary written with flag audit results.

**Abnormal:** `BENCH_30_FLAG_UNKNOWN: Neither --expert-offload nor any SEO flag found in ik_llama.cpp --help. flag_audit.txt excerpt: [first 20 lines]. T_CV7 may require a newer ik_llama.cpp build.`
