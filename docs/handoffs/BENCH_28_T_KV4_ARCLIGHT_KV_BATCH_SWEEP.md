# BENCH_28 — T_KV4: Arclight Coder/Thinker KV Max-Size + Batch Sweep

**Status:** READY
**Blocks:** nothing
**Blocked by:** nothing

---

## Title
T_KV4 — Sweep `gpu-mem-util` and `max-num-seqs` for Arclight coder and thinker; record actual KV pool size (MiB), max/avg KV usage, and TPS at each operating point.

## Objective
Establish the safe operating envelope for Arclight coder and thinker: maximum KV pool size without OOM, actual KV utilization under load, and TPS vs batch capacity trade-offs. Produces two tables (coder and thinker) that define safe defaults and aggressive profiles for production tuning.

## Why this exists

Current production defaults (coder: util=0.90, seqs=16; thinker: util=0.90, seqs=4) were chosen conservatively. KV pool size is determined by `(total_VRAM × util − model_weight_floor) ÷ KV_bytes_per_token`. For the PQ coder (29.4 GB model floor at util=0.90 leaves <1 GB for KV), the model floor is the binding constraint — not util. For the thinker (22–23 GB model floor at util=0.73), there's more headroom. This sweep characterizes the real boundaries so we can tune safely.

**GDN note for thinker:** GDN (Gated DeltaNet) maintains O(d) recurrent state, not O(n) KV tokens. Measured 0 MiB KV delta at 50K context (T3.1). The thinker's KV pool is almost entirely unused. High `gpu-mem-util` for the thinker primarily affects recurrent state memory, not KV cache token capacity.

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — production config, port assignments, gotchas #14 (PQ OOM at startup), #17 (thinker GDN KV near-zero)
2. `docs/procedures/vllm-deploy.md` — deploy commands, env vars
3. `results/BENCH_21_*/summary.md` — baseline VRAM readings for coder and thinker at util=0.90
4. `results/BENCH_23_pq2_phase1_coder_*/summary.md` — coder VRAM at load (reference for max-num-seqs=16 baseline)

## Prerequisites

```bash
echo "=== BENCH_28 Prerequisites ===" && \

# 1. Both GPUs available (run when neither model is deployed, or each model in isolation)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader && \

# 2. infra scripts available
[ -f "./infra/scripts/deploy.sh" ] && echo "[prereq] deploy.sh OK" \
  || { echo "[prereq] STOP: deploy.sh not found"; exit 1; } && \

# 3. vLLM version
podman run --rm --entrypoint python3 \
  $(podman images --format "{{.Repository}}:{{.Tag}}" | grep vllm | head -1) \
  -c "import vllm; print('vLLM version:', vllm.__version__)" 2>/dev/null \
  || echo "[prereq] WARNING: could not check vLLM version" && \

# 4. Models cached (will not re-download)
ls /srv/ai/models/hub/models--rdtand--Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm/ 2>/dev/null \
  && echo "[prereq] PQ coder model cached" \
  || echo "[prereq] WARNING: PQ coder not cached — deploy will download" && \
ls /srv/ai/models/hub/models--rdtand--Qwen3.6-27B-PrismaQuant-5.5bit-vllm/ 2>/dev/null \
  && echo "[prereq] PQ thinker model cached" \
  || echo "[prereq] WARNING: PQ thinker not cached — deploy will download"
```

## Inputs required

- `infra/scripts/deploy.sh` (vLLM rootless podman)
- Models cached in `/srv/ai/models/` (PQ coder + PQ thinker)
- GPU0 free for coder sweep; GPU1 free for thinker sweep
- vLLM 0.20+ (V1 engine, `VLLM_USE_V1=0` is no-op — ignored)

## Fixed controls

| Control | Value |
|---------|-------|
| Coder model | rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm |
| Thinker model | rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm |
| Coder GPU | GPU0 (TP=1) |
| Thinker GPU | GPU1 (TP=1) |
| Context size | 32768 (both) |
| KV dtype | fp8 (both, `--kv-cache-dtype fp8`) |
| TPS prompt | `"Write a Python function that sorts a list using quicksort."` |
| TPS max_tokens | 150 |
| TPS reps per point | 3 (N=1 sequential) |
| Tool-call probe | 1 quick probe per point (pass/fail only) |

## Single variable under test

**Coder sweep:** `--max-num-seqs` at values {4, 8, 16, 32} with `--gpu-mem-util` fixed at 0.90. Records KV pool MiB from vLLM startup log and TPS at each seqs value.

**Thinker sweep:** `--gpu-mem-util` at values {0.73 (model floor), 0.80, 0.85, 0.90, 0.95} with `--max-num-seqs 4` fixed. Records KV pool MiB, VRAM used, and TPS.

## Procedure

Skip flags:
- `SKIP_CODER_SWEEP=1` — skip coder max-num-seqs sweep
- `SKIP_THINKER_SWEEP=1` — skip thinker gpu-mem-util sweep

```bash
set -euo pipefail
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_28_kv4_arclight_kv_sweep_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

CODER_MODEL="rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm"
THINKER_MODEL="rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm"
TPS_PROMPT="Write a Python function that sorts a list using quicksort."

echo "seqs_or_util,kv_pool_mib,vram_used_mib,tps_n1_avg,tool_pass" > "${RESULTS_DIR}/coder_sweep.csv"
echo "seqs_or_util,kv_pool_mib,vram_used_mib,tps_n1_avg,tool_pass" > "${RESULTS_DIR}/thinker_sweep.csv"

stop_container() {
  local PORT=$1
  local PATTERN=$2
  local C=$(podman ps --format "{{.Names}}" | grep -i "${PATTERN}" | head -1 2>/dev/null || true)
  [ -n "${C}" ] && { podman stop "${C}" 2>/dev/null; podman rm "${C}" 2>/dev/null; sleep 5; }
  # Wait for VRAM release
  sleep 3
}

deploy_and_measure() {
  local ROLE=$1         # "coder" or "thinker"
  local MODEL=$2
  local SLOT=$3         # "gpu0" or "gpu1"
  local PORT=$4
  local UTIL=$5
  local SEQS=$6
  local SWEEP_VAR=$7    # the value being swept (for CSV label)
  local CSV=$8

  echo "--- ${ROLE}: util=${UTIL} seqs=${SEQS} ---"

  # Deploy
  local DEPLOY_LOG="${RESULTS_DIR}/${ROLE}_${SWEEP_VAR}_deploy.log"

  if [ "${ROLE}" = "coder" ]; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./infra/scripts/deploy.sh vllm "${SLOT}" "${MODEL}" \
      --trust-remote-code \
      --gpu-mem-util "${UTIL}" \
      --max-model-len 32768 \
      --kv-cache-dtype fp8 \
      --max-num-seqs "${SEQS}" \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --enable-auto-tool-choice \
      > "${DEPLOY_LOG}" 2>&1
  else
    ./infra/scripts/deploy.sh vllm "${SLOT}" "${MODEL}" \
      --trust-remote-code \
      --gpu-mem-util "${UTIL}" \
      --max-model-len 32768 \
      --kv-cache-dtype fp8 \
      --enable-chunked-prefill \
      --max-num-seqs "${SEQS}" \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --enable-auto-tool-choice \
      > "${DEPLOY_LOG}" 2>&1
  fi

  CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "${ROLE}" | head -1 2>/dev/null || true)
  [ -z "${CONTAINER}" ] && { echo "DEPLOY_FAILED: ${ROLE} util=${UTIL} seqs=${SEQS}" | tee -a "${RESULTS_DIR}/${CSV}"; echo "${SWEEP_VAR},FAIL,FAIL,FAIL,FAIL" >> "${RESULTS_DIR}/${CSV}"; return; }

  podman logs -f "${CONTAINER}" 2>&1 | stdbuf -oL sed 's/^/['${ROLE}'] /' &
  LOG_PID=$!

  for i in $(seq 1 300); do
    curl -sf "http://localhost:${PORT}/health" 2>/dev/null && break
    sleep 1
  done
  kill "${LOG_PID}" 2>/dev/null

  if ! curl -sf "http://localhost:${PORT}/health" 2>/dev/null; then
    echo "HEALTH_FAIL: ${ROLE} util=${UTIL} seqs=${SEQS}" | tee -a "${DEPLOY_LOG}"
    stop_container "${PORT}" "${ROLE}"
    echo "${SWEEP_VAR},OOM_OR_FAIL,OOM_OR_FAIL,OOM_OR_FAIL,FAIL" >> "${RESULTS_DIR}/${CSV}"
    return
  fi

  # Extract KV pool from logs
  KV_POOL=$(podman logs "${CONTAINER}" 2>/dev/null \
    | grep -i "kv cache\|gpu_memory_utilization\|cache_block\|num_gpu_blocks" \
    | tail -5)
  KV_POOL_MIB=$(podman logs "${CONTAINER}" 2>/dev/null \
    | grep -oP "(?<=gpu_memory_utilization=)[0-9.]+" | head -1 || echo "UNKNOWN")
  # Try to get block count and compute MiB
  BLOCKS=$(podman logs "${CONTAINER}" 2>/dev/null | grep -oP "(?<=num_gpu_blocks=)[0-9]+" | head -1 || echo "0")
  # Each block = 16 tokens * 2 (K+V) * n_layers * n_kv_heads * head_dim * dtype_bytes
  # fp8 = 1 byte; PQ coder: 28 experts * ... actually just record blocks and compute post-hoc
  echo "kv_pool_blocks=${BLOCKS}" >> "${RESULTS_DIR}/metadata.txt"
  KV_POOL_MIB_COMPUTED=$(python3 -c "
blocks = int('${BLOCKS}' or 0)
# Approximate: each block = 16 tokens, fp8 per element
# Qwen3.6-35B-A3B: 64 kv_heads (or GQA), 128 head_dim, 64 layers
# Use 32 as n_kv_heads (GQA) for approximation
block_mib = 16 * 2 * 64 * 32 * 128 / (1024*1024)
total = blocks * block_mib
print(round(total, 0))
" 2>/dev/null || echo "UNKNOWN")

  VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$([ "${SLOT}" = "gpu0" ] && echo 0 || echo 1) | tr -d ' ')

  MODEL_NAME=$(curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "${MODEL}")

  # TPS: 3 sequential N=1 requests
  TOTAL_TOKENS=0
  TOTAL_S=0
  for REP in 1 2 3; do
    START_MS=$(date +%s%3N)
    RESP=$(curl -sf "http://localhost:${PORT}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"${TPS_PROMPT}\",\"max_tokens\":150,\"temperature\":0.0}" 2>/dev/null)
    END_MS=$(date +%s%3N)
    T=$(echo "${RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
    ELAPSED=$(python3 -c "print(round(($END_MS - $START_MS)/1000.0,2))")
    TOTAL_TOKENS=$((TOTAL_TOKENS + T))
    TOTAL_S=$(python3 -c "print(${TOTAL_S} + ${ELAPSED})")
  done
  TPS_AVG=$(python3 -c "print(round(${TOTAL_TOKENS}/max(${TOTAL_S},0.1),1))")

  # Quick tool-call probe (1 probe, pass/fail)
  TOOL_RESP=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a fibonacci function and call execute_code to test it with n=8.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"execute_code\",\"description\":\"Execute code\",\"parameters\":{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\"}},\"required\":[\"code\"]}}}],\"tool_choice\":\"auto\",\"max_tokens\":256,\"temperature\":0.0}" 2>/dev/null)
  TOOL_PASS=$(echo "${TOOL_RESP}" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    tc=d['choices'][0]['message'].get('tool_calls',[])
    print('YES' if tc else 'NO')
except: print('ERROR')
" 2>/dev/null)

  echo "${SWEEP_VAR},${KV_POOL_MIB_COMPUTED},${VRAM_USED},${TPS_AVG},${TOOL_PASS}" >> "${RESULTS_DIR}/${CSV}"
  echo "  Result: kv_pool≈${KV_POOL_MIB_COMPUTED}MiB vram=${VRAM_USED}MiB tps=${TPS_AVG} tool=${TOOL_PASS}"

  stop_container "${PORT}" "${ROLE}"
}

# ===================================================================
# PART A: CODER sweep — max-num-seqs {4, 8, 16, 32} at util=0.90
# ===================================================================
SKIP_CODER_SWEEP=${SKIP_CODER_SWEEP:-0}
if [ "${SKIP_CODER_SWEEP}" = "0" ]; then
  echo "=== CODER max-num-seqs sweep ===" | tee "${RESULTS_DIR}/coder_sweep.log"
  for SEQS in 4 8 16 32; do
    deploy_and_measure "coder" "${CODER_MODEL}" "gpu0" 30000 0.90 "${SEQS}" "seqs${SEQS}" "coder_sweep.csv"
  done
  echo "Coder sweep complete:"
  cat "${RESULTS_DIR}/coder_sweep.csv"
else
  echo "[skip] SKIP_CODER_SWEEP=1"
fi

# ===================================================================
# PART B: THINKER sweep — gpu-mem-util {0.73, 0.80, 0.85, 0.90, 0.95} at seqs=4
# ===================================================================
SKIP_THINKER_SWEEP=${SKIP_THINKER_SWEEP:-0}
if [ "${SKIP_THINKER_SWEEP}" = "0" ]; then
  echo "=== THINKER gpu-mem-util sweep ===" | tee "${RESULTS_DIR}/thinker_sweep.log"
  for UTIL in 0.73 0.80 0.85 0.90 0.95; do
    deploy_and_measure "thinker" "${THINKER_MODEL}" "gpu1" 30001 "${UTIL}" 4 "util${UTIL}" "thinker_sweep.csv"
  done
  echo "Thinker sweep complete:"
  cat "${RESULTS_DIR}/thinker_sweep.csv"
else
  echo "[skip] SKIP_THINKER_SWEEP=1"
fi

# Restore production config at end
echo "=== Restoring production: coder + thinker ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./infra/scripts/deploy.sh vllm gpu0 "${CODER_MODEL}" \
  --trust-remote-code --gpu-mem-util 0.90 --max-model-len 32768 \
  --kv-cache-dtype fp8 --max-num-seqs 16 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

./infra/scripts/deploy.sh vllm gpu1 "${THINKER_MODEL}" \
  --trust-remote-code --gpu-mem-util 0.90 --max-model-len 32768 \
  --kv-cache-dtype fp8 --enable-chunked-prefill --max-num-seqs 4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

echo "=== BENCH_28 complete === Results in: ${RESULTS_DIR}"
```

**Note on KV pool MiB extraction:** The `KV_POOL_MIB_COMPUTED` above is an approximation. Also check the vLLM startup log directly for lines like `"GPU KV cache memory": X MiB` or `"Number of GPU blocks: Y (X MiB)"`. Record the raw log line in the summary.

**Note on coder at seqs=32:** With PQ coder (29.4 GB floor) at util=0.90, the KV pool is already near zero. seqs=32 may OOM during the profiling forward pass — this is expected and should be recorded as OOM, not a failure.

## Metrics to record

| Metric | Source file | Expected |
|--------|-------------|---------|
| Coder KV pool MiB at seqs=4/8/16/32 | `coder_sweep.csv` | seqs=16 is production; seqs=32 may OOM |
| Coder TPS at each seqs value | `coder_sweep.csv` | seqs=16: ~57 t/s (BENCH_23 baseline) |
| Thinker KV pool MiB at each util | `thinker_sweep.csv` | Grows with util; GDN barely uses it |
| Thinker TPS at each util | `thinker_sweep.csv` | ~92 t/s at all utils (model floor is binding) |
| Tool-call pass at each point | both CSVs | Should be YES at all viable configs |
| OOM boundary for coder | `coder_sweep.csv` | Expected: seqs ≥ 32 at util=0.90 |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| At least one coder seqs value deploys | Any seqs {4,8,16,32} PASS | Record minimum viable seqs |
| Production config (seqs=16) deploys | PASS | Critical — must be stable |
| At least 3 thinker util values deploy | util {0.73, 0.80, 0.85} | |
| Tool calls YES at production config | coder seqs=16, thinker util=0.90 | |

## Artifacts to write

1. `results/BENCH_28_kv4_arclight_kv_sweep_<TIMESTAMP>/coder_sweep.csv`
2. `results/BENCH_28_kv4_arclight_kv_sweep_<TIMESTAMP>/thinker_sweep.csv`
3. `results/BENCH_28_kv4_arclight_kv_sweep_<TIMESTAMP>/coder_*_deploy.log` (one per seqs value)
4. `results/BENCH_28_kv4_arclight_kv_sweep_<TIMESTAMP>/thinker_*_deploy.log` (one per util value)
5. `results/BENCH_28_kv4_arclight_kv_sweep_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_28 — T_KV4: Arclight KV Batch Sweep — <TIMESTAMP>

## Coder sweep (util=0.90, varying max-num-seqs)
| max-num-seqs | KV pool (MiB) | VRAM used (MiB) | TPS N=1 | Tool call |
|-------------|--------------|----------------|---------|-----------|
| 4           | <x>          | <x>            | <x>     | YES/NO    |
| 8           | <x>          | <x>            | <x>     | YES/NO    |
| 16 (prod)   | <x>          | <x>            | <x>     | YES/NO    |
| 32          | OOM / <x>    | OOM / <x>      | OOM / <x> | OOM/YES/NO |

Note: KV pool extracted from vLLM startup log. `OOM` = failed during profiling forward pass.

## Thinker sweep (max-num-seqs=4, varying gpu-mem-util)
| gpu-mem-util | KV pool (MiB) | VRAM used (MiB) | TPS N=1 | Tool call |
|-------------|--------------|----------------|---------|-----------|
| 0.73 (floor) | <x>          | <x>            | <x>     | YES/NO    |
| 0.80         | <x>          | <x>            | <x>     | YES/NO    |
| 0.85         | <x>          | <x>            | <x>     | YES/NO    |
| 0.90 (prod)  | <x>          | <x>            | <x>     | YES/NO    |
| 0.95         | <x>          | <x>            | <x>     | YES/NO    |

Note: GDN architecture — thinker KV pool is nearly unused (0 MiB delta at 50K context, T3.1). Larger pool does not improve thinker quality.

## Recommended operating envelopes
- **Coder safe default:** seqs=16, util=0.90 (production config, KV pool=<x> MiB)
- **Coder aggressive:** seqs=<N> (if seqs>16 viable and tool calls pass)
- **Thinker:** util=0.90 is adequate; higher util wastes VRAM on unused KV pool (GDN)

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| Coder seqs=16 deploys + tool call | PASS/FAIL | Production config |
| Thinker util=0.90 deploys + tool call | PASS/FAIL | Production config |
| Production restored at end | PASS/FAIL | |

## Verdict
PASS / PARTIAL — <one sentence on whether any config change is warranted>

## Incidental findings
<Startup OOM patterns, unexpected KV pool sizes, GDN KV utilization confirmation.>
<If nothing: "none">

## Open from testing
<If seqs=16 OOMs or any production config fails, record for research mode.>
<If nothing: "none">
```

## Interpretation boundary

**You may:** Record KV pool MiB, VRAM used, TPS, tool-call pass/fail at each operating point. Note OOM boundary.

**You may NOT:** Change production `gpu-mem-util` or `max-num-seqs` in `docs/procedures/vllm-deploy.md` or `docs/arch/current.md` — research mode decides based on summary.

## Stop condition

**Normal:** Both sweeps complete, CSVs filled, production config restored, summary written.

**Abnormal:** `BENCH_28_RESTORE_FAIL: Production deploy failed after sweep. Last known good: coder seqs=<N> tool=YES, thinker util=<U> tool=YES.`
