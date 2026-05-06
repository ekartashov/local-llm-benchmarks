# BENCH_32 — T_ARCH3: Arclight Full (fp16 KV) vs Extended Arclight Rationale

**Status:** READY
**Blocks:** nothing
**Blocked by:** nothing

---

## Title
T_ARCH3 — Measure thinker quality delta at fp16 vs fp8 KV (with deliberately small KV pool); confirm Extended Arclight coder TP=2 at 64K context; verify the GDN KV illogicality hypothesis for fp16 thinker.

## Objective
Settle two open architecture questions:
1. **Arclight Full (fp16 KV for thinker):** Does fp16 KV improve thinker quality? Hypothesis: NO — GDN (Gated DeltaNet) maintains O(d) recurrent state, not O(n) KV tokens. Measured 0 MiB KV delta at 50K context (T3.1). fp16 KV precision is irrelevant for a model that barely uses KV cache.
2. **Extended Arclight:** Does coder TP=2 at 65K context deliver the expected 238 t/s and correct output?

Settling both allows the architecture document to mark "Arclight Full" as ILLOGICAL (wasted VRAM on KV type for GDN) and confirms Extended Arclight as the production long-context coder path.

## Why this exists

**Thinker GDN KV irrelevance:** T3.1 showed 0 MiB KV cache delta at 50K context for Qwen3.6-27B (GDN architecture). This means the KV pool is almost entirely unused — the recurrent state is O(d), not O(n). Switching from fp8 to fp16 KV wastes 50% of the thinker's KV pool VRAM on a precision upgrade that has no effect on model quality. "Arclight Full" (thinker with fp16 KV) is architecturally illogical for GDN models.

**Extended Arclight verification:** Extended Arclight (coder TP=2, 65K ctx) is documented as settled at 238 t/s (T_KV1). This benchmark confirms it still works with the PrismaQuant coder and current vLLM 0.20.0 (V0/V1 engine changes may affect TP=2 behavior).

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — port assignments, gotchas #3 (TP=2 GDN broken), #4 (uvloop/CRIU), #17 (GDN KV near-zero)
2. `docs/procedures/vllm-deploy.md` — Extended Arclight deploy command, env vars
3. `docs/arch/extended-arclight.md` — CRIU procedure, ghost VRAM cleanup, expected TPS
4. `results/T_KV1_*/summary.md` or `results/BENCH_21_*/summary.md` — Extended Arclight TPS baseline (238 t/s)
5. `results/T3.1_*/summary.md` — GDN KV 0 MiB delta at 50K context

## Prerequisites

```bash
echo "=== BENCH_32 Prerequisites ===" && \

# 1. vLLM available
[ -f "./infra/scripts/deploy.sh" ] && echo "[prereq] deploy.sh OK" \
  || { echo "[prereq] STOP: deploy.sh not found"; exit 1; } && \

# 2. Models cached
ls /srv/ai/models/hub/models--rdtand--Qwen3.6-27B-PrismaQuant-5.5bit-vllm/ 2>/dev/null \
  && echo "[prereq] Thinker model cached" \
  || echo "[prereq] WARNING: thinker model not cached" && \
ls /srv/ai/models/hub/models--rdtand--Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm/ 2>/dev/null \
  && echo "[prereq] Coder model cached" \
  || echo "[prereq] WARNING: coder model not cached" && \

# 3. GPU VRAM
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader && \

# 4. CRIU available (for Extended Arclight)
which criu >/dev/null 2>&1 && criu check && echo "[prereq] CRIU OK" \
  || echo "[prereq] WARNING: CRIU not available — Extended Arclight restart will be cold (100s) not hot (0.28s)" && \

# 5. uvloop patch (for CRIU + vLLM)
grep -r "asyncio\|uvloop" /srv/ai/projects/vllm/vllm/entrypoints/api_server.py 2>/dev/null | head -5 \
  || echo "[prereq] INFO: Cannot check uvloop patch — verify UV_USE_IO_URING=0 is set"
```

## Inputs required

- `infra/scripts/deploy.sh` (vLLM rootless podman)
- Thinker model cached: `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm`
- Coder model cached: `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm`
- GPU1 free for thinker tests; GPU0+1 free for Extended Arclight coder test

## Fixed controls — Part A (thinker KV precision)

| Control | Value |
|---------|-------|
| Thinker model | rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm |
| GPU | GPU1, TP=1 |
| Port | 30001 |
| Context | 32768 |
| max-num-seqs | 4 |
| gpu-mem-util | 0.73 (model floor — minimizes KV pool to stress-test precision effect) |
| KV variants | `fp8` (production) then `bf16` (--kv-cache-dtype auto = bf16 equiv) |
| Quality tasks | th02 (EDF) + 1 hard thinker task |
| Reps | 1 per task (semantic quality, not TPS) |

## Fixed controls — Part B (Extended Arclight)

| Control | Value |
|---------|-------|
| Coder model | rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm |
| GPU | TP=2 GPU0+GPU1 |
| Port | 30000 |
| Context | 65536 |
| gpu-mem-util | 0.90 |
| KV | fp8 |
| TPS reps | 3 at N=1 |
| Coding task | 60K context task (construct from repeated code blocks to fill context) |

## Single variable under test

**Part A:** KV cache dtype (`fp8` vs `bf16`) on the thinker with minimal KV pool (util=0.73). Expected finding: no quality difference (GDN irrelevance confirmed).

**Part B:** Context extension (65K) on coder at TP=2. Expected: 238 t/s decode and no correctness regression.

## Procedure

Skip flags:
- `SKIP_PART_A=1` — skip thinker KV precision test
- `SKIP_PART_B=1` — skip Extended Arclight test

```bash
set -euo pipefail
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_32_arch3_arclight_full_vs_extended_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

THINKER_MODEL="rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm"
CODER_MODEL="rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm"

echo "kv_type,task,quality_score,finish_reason,tokens" > "${RESULTS_DIR}/part_a_results.csv"

TH02_PROMPT="Implement a complete Earliest Deadline First (EDF) scheduler in Python. Requirements:
1. Task class with: task_id, arrival_time, execution_time, deadline attributes
2. EDF scheduler always selecting earliest-deadline task
3. Test with tasks: [(0,3,5), (1,2,4), (2,1,3), (3,4,8)] — task (2,1,3) must run first
Provide complete runnable Python code."

HARD_TASK_PROMPT="Implement a lock-free concurrent queue in Python using atomic operations. The queue must:
1. Support multiple producers and consumers without locks
2. Use compare-and-swap semantics (via ctypes or multiprocessing)
3. Be bounded with a configurable maximum size
4. Handle ABA problem prevention
5. Include a test demonstrating thread-safe concurrent push and pop from 4 threads
Provide complete, runnable Python code with correctness verification."

stop_container() {
  local PATTERN=$1
  local C=$(podman ps --format "{{.Names}}" | grep -i "${PATTERN}" | head -1 2>/dev/null || true)
  [ -n "${C}" ] && { podman stop "${C}" 2>/dev/null; podman rm "${C}" 2>/dev/null; sleep 5; }
}

# ===================================================================
# PART A: Thinker fp8 vs bf16 KV quality comparison
# ===================================================================
SKIP_PART_A=${SKIP_PART_A:-0}
if [ "${SKIP_PART_A}" = "0" ]; then
  echo "=== PART A: Thinker KV precision comparison ===" | tee "${RESULTS_DIR}/part_a.log"

  for KV_TYPE in fp8 bf16; do
    KV_FLAG=$( [ "${KV_TYPE}" = "fp8" ] && echo "--kv-cache-dtype fp8" || echo "" )
    echo "--- KV type: ${KV_TYPE} ---" | tee -a "${RESULTS_DIR}/part_a.log"

    stop_container "thinker\|30001"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./infra/scripts/deploy.sh vllm gpu1 "${THINKER_MODEL}" \
      --trust-remote-code \
      --gpu-mem-util 0.73 \
      --max-model-len 32768 \
      ${KV_FLAG} \
      --enable-chunked-prefill \
      --max-num-seqs 4 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --enable-auto-tool-choice \
      > "${RESULTS_DIR}/thinker_${KV_TYPE}_deploy.log" 2>&1

    CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1 2>/dev/null || true)
    [ -z "${CONTAINER}" ] && {
      echo "DEPLOY_FAIL for ${KV_TYPE} KV — check deploy log"
      echo "${KV_TYPE},th02,DEPLOY_FAIL,FAIL,0" >> "${RESULTS_DIR}/part_a_results.csv"
      continue
    }

    podman logs -f "${CONTAINER}" 2>&1 | stdbuf -oL sed 's/^/[thinker-'${KV_TYPE}'] /' &
    LOG_PID=$!
    for i in $(seq 1 300); do curl -sf http://localhost:30001/health 2>/dev/null && break; sleep 1; done
    kill "${LOG_PID}" 2>/dev/null

    if ! curl -sf http://localhost:30001/health 2>/dev/null; then
      echo "HEALTH_FAIL for ${KV_TYPE} KV"
      echo "${KV_TYPE},th02,HEALTH_FAIL,FAIL,0" >> "${RESULTS_DIR}/part_a_results.csv"
      stop_container "thinker\|30001"; continue
    fi

    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
    echo "kv=${KV_TYPE} VRAM_GPU1=${VRAM}MiB" | tee -a "${RESULTS_DIR}/part_a.log"

    # Extract KV pool from startup log (to demonstrate size difference)
    KV_BLOCKS=$(podman logs "${CONTAINER}" 2>/dev/null | grep -oP "(?<=num_gpu_blocks=)[0-9]+" | head -1 || echo "UNKNOWN")
    echo "kv=${KV_TYPE} kv_blocks=${KV_BLOCKS}" | tee -a "${RESULTS_DIR}/part_a.log"

    THINKER_MODEL_NAME=$(curl -sf http://localhost:30001/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "${THINKER_MODEL}")

    for TASK in "th02" "hard"; do
      PROMPT=$( [ "${TASK}" = "th02" ] && echo "${TH02_PROMPT}" || echo "${HARD_TASK_PROMPT}" )
      RESP=$(curl -sf http://localhost:30001/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"${THINKER_MODEL_NAME}\",
          \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import json; print(json.dumps(\"${PROMPT}\"))")}],
          \"max_tokens\": 8192,
          \"temperature\": 0.0
        }" 2>/dev/null)
      echo "${RESP}" > "${RESULTS_DIR}/thinker_${KV_TYPE}_${TASK}.json"
      FINISH=$(echo "${RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0].get('finish_reason','unknown'))" 2>/dev/null || echo "unknown")
      TOKENS=$(echo "${RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens','?'))" 2>/dev/null || echo "?")
      # Quality scored by human — write UNSCORED placeholder
      echo "${KV_TYPE},${TASK},UNSCORED,${FINISH},${TOKENS}" >> "${RESULTS_DIR}/part_a_results.csv"
      echo "  ${TASK}: finish=${FINISH} tokens=${TOKENS}" | tee -a "${RESULTS_DIR}/part_a.log"
    done

    stop_container "thinker\|30001"
  done
  echo "Part A complete. Quality scoring (th02 + hard task) requires human review."
  cat "${RESULTS_DIR}/part_a_results.csv"
else
  echo "[skip] SKIP_PART_A=1"
fi

# ===================================================================
# PART B: Extended Arclight (coder TP=2, 65K ctx)
# ===================================================================
SKIP_PART_B=${SKIP_PART_B:-0}
if [ "${SKIP_PART_B}" = "0" ]; then
  echo "=== PART B: Extended Arclight — coder TP=2, 65K context ===" | tee "${RESULTS_DIR}/part_b.log"

  # Stop thinker if running (TP=2 needs both GPUs)
  stop_container "thinker\|30001"
  sleep 5

  echo "=== VRAM before Extended Arclight deploy ===" | tee "${RESULTS_DIR}/vram_part_b.txt"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram_part_b.txt"

  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
  ./infra/scripts/deploy.sh vllm tp2a "${CODER_MODEL}" \
    --trust-remote-code \
    --gpu-mem-util 0.90 \
    --max-model-len 65536 \
    --kv-cache-dtype fp8 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    > "${RESULTS_DIR}/extended_arclight_deploy.log" 2>&1

  CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|tp2\|35b" | head -1 2>/dev/null || true)
  if [ -z "${CONTAINER}" ]; then
    echo "DEPLOY_FAIL: Extended Arclight did not start" | tee -a "${RESULTS_DIR}/part_b.log"
    echo "extended_arclight_status=DEPLOY_FAIL" >> "${RESULTS_DIR}/metadata.txt"
  else
    podman logs -f "${CONTAINER}" 2>&1 | stdbuf -oL sed 's/^/[ext-arclight] /' &
    LOG_PID=$!
    for i in $(seq 1 300); do curl -sf http://localhost:30000/health 2>/dev/null && break; sleep 1; done
    kill "${LOG_PID}" 2>/dev/null
  fi

  if curl -sf http://localhost:30000/health 2>/dev/null; then
    echo "=== VRAM after Extended Arclight load ===" | tee -a "${RESULTS_DIR}/vram_part_b.txt"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram_part_b.txt"

    CODER_NAME=$(curl -sf http://localhost:30000/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "${CODER_MODEL}")

    # TPS test: 3 reps N=1
    echo "=== Extended Arclight TPS ===" | tee "${RESULTS_DIR}/part_b_tps.txt"
    TOTAL=0; TOTAL_S=0
    for R in 1 2 3; do
      START_MS=$(date +%s%3N)
      RESP=$(curl -sf http://localhost:30000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${CODER_NAME}\",\"prompt\":\"Write a Python function to sort a list using merge sort with type hints.\",\"max_tokens\":200,\"temperature\":0.0}" 2>/dev/null)
      END_MS=$(date +%s%3N)
      T=$(echo "${RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
      S=$(python3 -c "print(round(($END_MS - $START_MS)/1000.0,2))")
      TOTAL=$((TOTAL + T)); TOTAL_S=$(python3 -c "print(${TOTAL_S} + ${S})")
      echo "rep=${R} tokens=${T} elapsed=${S}s" | tee -a "${RESULTS_DIR}/part_b_tps.txt"
    done
    TPS_AVG=$(python3 -c "print(round(${TOTAL}/max(${TOTAL_S},0.1),1))")
    echo "avg_tps=${TPS_AVG} baseline_expected=238" | tee -a "${RESULTS_DIR}/part_b_tps.txt"

    # 64K context correctness test (pad a coding task to fill context)
    echo "=== 64K context task ===" | tee "${RESULTS_DIR}/part_b_64k.txt"
    PADDING=$(python3 -c "
# Generate a ~60K token prompt (rough: 1 tok ≈ 4 chars)
code = '''# Legacy codebase for refactoring\nimport os\nimport sys\n\ndef process_data(data: list) -> dict:\n    result = {}\n    for item in data:\n        key = str(item)\n        result[key] = item * 2\n    return result\n\n'''
print(code * 2000)
print('Refactor the above code to use dataclasses and type hints. Keep the same logic.')
")
    START_MS=$(date +%s%3N)
    RESP_64K=$(curl -sf http://localhost:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"${CODER_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import json; print(json.dumps(\"${PADDING}\"))")}],
        \"max_tokens\": 512,
        \"temperature\": 0.0
      }" \
      --max-time 600 2>/dev/null)
    END_MS=$(date +%s%3N)
    echo "${RESP_64K}" > "${RESULTS_DIR}/part_b_64k_response.json"
    ELAPSED_64K=$(python3 -c "print(round(($END_MS - $START_MS)/1000.0,1))")
    PROMPT_TOKENS=$(echo "${RESP_64K}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('prompt_tokens','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
    OUT_TOKENS=$(echo "${RESP_64K}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
    FINISH=$(echo "${RESP_64K}" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0].get('finish_reason','unknown'))" 2>/dev/null || echo "unknown")
    echo "64k task: prompt=${PROMPT_TOKENS}tok out=${OUT_TOKENS}tok elapsed=${ELAPSED_64K}s finish=${FINISH}" | tee -a "${RESULTS_DIR}/part_b_64k.txt"
    echo "extended_arclight_status=OK avg_tps=${TPS_AVG}" >> "${RESULTS_DIR}/metadata.txt"
  else
    echo "HEALTH_FAIL: Extended Arclight not responding at :30000"
    echo "extended_arclight_status=HEALTH_FAIL" >> "${RESULTS_DIR}/metadata.txt"
  fi

  # Restore production config
  echo "=== Restoring production: coder TP=1 + thinker ===" | tee -a "${RESULTS_DIR}/part_b.log"
  stop_container "coder\|tp2\|35b\|30000"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./infra/scripts/deploy.sh vllm gpu0 "${CODER_MODEL}" \
    --trust-remote-code --gpu-mem-util 0.90 --max-model-len 32768 \
    --kv-cache-dtype fp8 --max-num-seqs 16 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    >> "${RESULTS_DIR}/restore.log" 2>&1 &

  ./infra/scripts/deploy.sh vllm gpu1 "${THINKER_MODEL}" \
    --trust-remote-code --gpu-mem-util 0.90 --max-model-len 32768 \
    --kv-cache-dtype fp8 --enable-chunked-prefill --max-num-seqs 4 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    >> "${RESULTS_DIR}/restore.log" 2>&1 &

  wait
  echo "Production restore: done"
else
  echo "[skip] SKIP_PART_B=1"
fi

echo "=== BENCH_32 complete === Results in: ${RESULTS_DIR}"
```

**Quality scoring note (Part A):** The two thinker responses (fp8 KV and bf16 KV) must be compared for semantic quality. Score each on th02 (EDF correctness) and hard task (lock-free queue: correct synchronization, ABA prevention, no deadlock). Score 1–5 per `docs/decisions/scoring.md`. The expected result is **identical quality** — GDN does not use KV cache, so precision does not matter. Write UNSCORED if uncertain; note whether the thinking blocks differ in depth or correctness between the two KV conditions.

## Metrics to record

| Metric | Source file | Expected |
|--------|-------------|---------|
| **Part A** | | |
| Thinker th02 score — fp8 KV | `part_a_results.csv` + human review | Expected: ~4.5/5 (same as BENCH_23b) |
| Thinker th02 score — bf16 KV | `part_a_results.csv` + human review | Expected: same as fp8 — no quality gain |
| Thinker hard task score — fp8 vs bf16 | `part_a_results.csv` | Expected: same |
| KV blocks at util=0.73 fp8 vs bf16 | `part_a.log` | bf16 pool should be ~half fp8 block count |
| VRAM GPU1 at util=0.73 fp8 vs bf16 | `part_a.log` | Should be nearly same (model dominates) |
| **Part B** | | |
| Extended Arclight TPS (N=1, TP=2) | `part_b_tps.txt` | Target ~238 t/s (T_KV1 baseline) |
| 64K context task: prompt token count | `part_b_64k.txt` | Should be 55K–64K tokens |
| 64K context task: finish reason | `part_b_64k.txt` | `stop` (not `length` or error) |
| Production restored at end | Manual check | Both coder TP=1 and thinker healthy |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Part A both KV types deploy | Health 200 | Record OOM at util=0.73 |
| Part A th02 quality identical fp8 vs bf16 | Scores within 0.5 | If bf16 significantly better: GDN hypothesis wrong — report |
| Part B coder deploys at TP=2 65K ctx | Health 200 | Record VRAM and error from deploy log |
| Part B TPS ≥ 200 t/s | PASS — Extended Arclight viable | < 150 t/s: investigate engine path (--enforce-eager?) |
| Part B 64K task finish = stop | PASS — context fits | `length`: context overflow, note exact token count |
| Production restored | Both endpoints healthy | Note manual steps needed |

## Artifacts to write

1. `results/BENCH_32_arch3_arclight_full_vs_extended_<TIMESTAMP>/part_a.log`
2. `results/BENCH_32_arch3_arclight_full_vs_extended_<TIMESTAMP>/part_a_results.csv`
3. `results/BENCH_32_arch3_arclight_full_vs_extended_<TIMESTAMP>/thinker_fp8_th02.json` + `_hard.json`
4. `results/BENCH_32_arch3_arclight_full_vs_extended_<TIMESTAMP>/thinker_bf16_th02.json` + `_hard.json`
5. `results/BENCH_32_arch3_arclight_full_vs_extended_<TIMESTAMP>/vram_part_b.txt`
6. `results/BENCH_32_arch3_arclight_full_vs_extended_<TIMESTAMP>/part_b_tps.txt`
7. `results/BENCH_32_arch3_arclight_full_vs_extended_<TIMESTAMP>/part_b_64k.txt` + `part_b_64k_response.json`
8. `results/BENCH_32_arch3_arclight_full_vs_extended_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_32 — T_ARCH3: Arclight Full vs Extended Arclight — <TIMESTAMP>

## Part A: Thinker fp8 vs bf16 KV quality (util=0.73 — minimal KV pool)

### KV pool at model floor (util=0.73)
| KV type | GPU1 VRAM (MiB) | KV blocks | Approximate token capacity |
|---------|----------------|-----------|---------------------------|
| fp8     | <x>            | <x>       | <x>K tokens               |
| bf16    | <x>            | <x>       | <x>K tokens (≈ half fp8)  |

### Quality results
| KV type | Task | Score (1–5) | finish | tokens | Notes |
|---------|------|-------------|--------|--------|-------|
| fp8 (production) | th02 | <x> or UNSCORED | stop/length | <x> | |
| bf16 | th02 | <x> or UNSCORED | stop/length | <x> | |
| fp8 | hard (lock-free queue) | <x> or UNSCORED | stop/length | <x> | |
| bf16 | hard (lock-free queue) | <x> or UNSCORED | stop/length | <x> | |

### Part A verdict
Quality difference fp8 vs bf16 KV: **NONE / MARGINAL / SIGNIFICANT** — <one sentence>
GDN KV illogicality hypothesis: **CONFIRMED / REFUTED** — <one sentence>
Conclusion: Arclight Full (thinker fp16 KV) is **LOGICAL / ILLOGICAL** for this architecture.

---

## Part B: Extended Arclight (coder TP=2, 65K context)

### Deployment
- Status: PASS / DEPLOY_FAIL / HEALTH_FAIL
- VRAM GPU0 after load: <x> MiB | GPU1 after load: <x> MiB

### TPS (N=1)
| Rep | Tokens | Elapsed (s) | TPS |
|-----|--------|-------------|-----|
| 1   | <x>    | <x>         | <x> |
| 2   | <x>    | <x>         | <x> |
| 3   | <x>    | <x>         | <x> |
| **avg** | | | **<x> t/s** (baseline: 238 t/s) |

### 64K context task
- Prompt tokens: <x> (target: 55K–64K)
- Output tokens: <x>
- Elapsed: <x>s
- Finish reason: stop / length / error

### Part B verdict
Extended Arclight viable: **YES / NO** — <one sentence>

---

## Overall pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| Both KV types deploy (Part A) | PASS/FAIL | |
| GDN KV illogicality confirmed (Part A) | PASS/FAIL | Quality difference: NONE / MARGINAL |
| Extended Arclight TPS ≥ 200 t/s | PASS/FAIL | Actual: <x> t/s |
| 64K context task completes | PASS/FAIL | finish_reason: <x> |
| Production restored | PASS/FAIL | |

## Architecture recommendations (for research mode)
- Arclight Full (thinker fp16 KV): **SETTLE AS ILLOGICAL** / **NEEDS FURTHER EVIDENCE** — <reason>
- Extended Arclight: **CONFIRMED VIABLE** / **DEGRADED** — <reason>

## Incidental findings
<KV block count anomaly, VRAM difference, thinking block depth difference between fp8/bf16.>
<If nothing: "none">

## Open from testing
<If GDN hypothesis is REFUTED (bf16 KV quality meaningfully better), this is a significant finding — write detailed Open from testing with response excerpts.>
<If nothing: "none">
```

## Interpretation boundary

**You may:** Record quality scores (with UNSCORED if uncertain), TPS, VRAM, KV block counts, 64K context task timing.

**You may NOT:**
- Update `docs/decisions/settled.md` to mark Arclight Full as ILLOGICAL — that is research mode
- Update `docs/arch/extended-arclight.md` with new TPS numbers
- Update `docs/arch/current.md` with architecture changes
- Score th02 or hard task without carefully verifying algorithmic correctness of the code

## Stop condition

**Normal:** Part A complete (both KV types tested, responses saved for human scoring), Part B complete (TPS measured, 64K task run), production restored, summary written with UNSCORED placeholders ready for human review.

**Abnormal:** Write `## Open from testing` in `RESEARCH_STATE.md` if:
- Part A bf16 scores clearly higher than fp8: `BENCH_32_GDN_HYPOTHESIS_REFUTED: Thinker bf16 KV scored <X> vs fp8 <Y> on [th02/hard]. Thinking block depth was different: fp8=[excerpt], bf16=[excerpt]. GDN KV illogicality assumption may need revision.`
- Part B TPS < 150 t/s: `BENCH_32_EXTENDED_ARCLIGHT_DEGRADED: TP=2 65K ctx TPS=<X> t/s (baseline 238 t/s). Check if --enforce-eager was set (large penalty). Deploy log excerpt: [relevant lines].`
- Part B 64K context overflow: `BENCH_32_64K_OVERFLOW: finish_reason=length at <X> prompt tokens. Context ceil may be lower than 65536 on current vLLM 0.20.0. Max viable context: <X> tokens.`
