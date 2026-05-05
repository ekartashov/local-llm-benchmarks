# BENCH_21 — Co-load VRAM mapping + thinker --max-model-len 131072 upgrade

**Status:** READY
**Blocks:** production config update (correct gpu_mem_util values), T_CRIU3 (VRAM budget for checkpoint storage sizing)
**Blocked by:** nothing

---

## Title
Three-model co-load VRAM mapping: find working gpu_mem_util values for coder TP=2 + thinker TP=1 + Convergence simultaneous operation, and measure Convergence TPS under co-load.

## Objective
Every production benchmark to date has measured each component in isolation. The current gpu_mem_util=0.90 values are single-model measurements and are not valid for co-load. This benchmark finds the minimum gpu_mem_util values that allow all three models to load simultaneously and remain stable, then measures Convergence decode TPS while the two Arclight processes are co-resident. A secondary objective is to confirm that adding `--max-model-len 131072` to the thinker (needed for hard-task reasoning budgets per BENCH_20) costs ~0 extra VRAM as expected from the GDN DeltaNet architecture.

## Why this exists

**GPU1 sharing constraint:** In normal hot-pair mode, GPU1 hosts both the coder TP=2 shard (half of a 35B model) and the thinker (full 27B model). The coder reserves gpu_mem_util * 32GB on EACH GPU. The thinker reserves gpu_mem_util * 32GB on GPU1 only. Both cannot use gpu_mem_util=0.90 simultaneously — the sum would require ~57GB on GPU1. The correct co-load values have never been measured.

**Convergence GPU interaction:** Convergence runs `-ngl 999 --cpu-moe`, placing attention/norm/embed layers on GPU. T_CV5 measured ~7.4GB on GPU0 and ~5.3GB on GPU1 (Convergence alone). Whether and how this interacts with vLLM's GPU reservations has never been measured.

**Context ceiling for hard tasks:** BENCH_20 found that hard reasoning tasks (Raft, cache-coherence, distributed systems) consume 20–28K reasoning tokens inside `<think>`. The 32K default context leaves insufficient headroom. `--max-model-len 131072` is needed in production. This is predicted to cost ~0 extra VRAM on DeltaNet hybrid models (GDN's recurrent state is fixed-size, not paged KV blocks — confirmed T3.1 Phase 1 at 50K context). BENCH_21 verifies this in practice.

**This test determines the actual production co-load configuration.**

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — current production config, gotchas, port assignments
2. `docs/procedures/vllm-deploy.md` — deploy commands and env vars for vLLM
3. `docs/arch/convergence.md` — Convergence launch command (must use `--no-mmap`), model path
4. `results/T_PAR1_parallel_throughput_sweep_20260427T180802Z/summary.md` — coder VRAM at isolation (GPU0=29,756 MiB, GPU1=29,754 MiB at gpu_mem_util=0.90)
5. `results/T_CV5_ngl_sweep_20260427T205900Z/summary.md` — Convergence VRAM at isolation (~7,400 MiB GPU0, ~5,300 MiB GPU1 at ngl=999)
6. `docs/decisions/models.md` — exact production flags for coder and thinker

## Prerequisites

```bash
echo "=== BENCH_21 Prerequisites ===" && \

# 1. All containers stopped (clean slate needed)
RUNNING=$(podman ps --format "{{.Names}}" | wc -l)
echo "[prereq] Running containers: ${RUNNING} (expected 0 for clean slate; stop production models if any)"

# 2. ik_llama.cpp binary exists
test -f /srv/ai/projects/ik_llama.cpp/build/bin/llama-server \
  && echo "[prereq] ik_llama.cpp binary OK" \
  || { echo "[prereq] STOP: ik_llama.cpp binary not found — rebuild or check path"; exit 1; }

# 3. Convergence GGUF model files exist (all 4 shards)
ls /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-000*.gguf 2>/dev/null | wc -l | \
  { read N; [ "${N}" -eq 4 ] && echo "[prereq] Convergence GGUF all 4 shards OK" || echo "[prereq] STOP: expected 4 GGUF shards, found ${N}"; }

# 4. Thinker model cached (or will be downloaded by deploy.sh)
ls /srv/ai/models/hub/models--rdtand--Qwen3.6-27B-PrismaQuant-5.5bit-vllm/ 2>/dev/null \
  && echo "[prereq] Thinker model cache found" || echo "[prereq] WARNING: thinker model may need download (deploy.sh will handle)"

# 5. Coder model cached
ls /srv/ai/models/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit/ 2>/dev/null \
  && echo "[prereq] Coder model cache found" || echo "[prereq] WARNING: coder model may need download (deploy.sh will handle)"

# 6. VRAM baseline (all GPUs should be near 0 MiB)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
echo "[prereq] VRAM snapshot above — should show near-0 used if all containers stopped"
```

## Inputs required

- `infra/scripts/deploy.sh` — vLLM container deploy script
- `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server` — ik_llama.cpp binary for Convergence
- All model caches:
  - Coder: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (HF or local cache)
  - Thinker: `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm` (~19GB disk)
  - Convergence: 4 GGUF shards at path in convergence.md (~123GB total)
- GPU1 VRAM free (32GB — all containers stopped before start)
- A single HTTP test prompt to send to Convergence during TPS measurement

## Fixed controls

| Control | Value |
|---------|-------|
| Coder model | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit |
| Coder placement | TP=2 GPU0+GPU1 |
| Coder flags (fixed) | `--kv-cache-dtype fp8 --ctx 32768 --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice` |
| Coder env | `VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` |
| Thinker model | rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm |
| Thinker placement | TP=1 GPU1 |
| Thinker flags (fixed) | `--trust-remote-code --kv-cache-dtype fp8 --ctx 32768 --enable-chunked-prefill --max-num-seqs 4 --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice --speculative-config '{"method":"mtp","num_speculative_tokens":3}'` |
| Thinker extended ctx flag | `--max-model-len 131072` (primary test goal) |
| Thinker env | `VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY` |
| Convergence model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M |
| Convergence flags | `-ngl 999 --cpu-moe --no-mmap -t 32 -np 4 -c 131072` |
| Convergence port | 8002 |
| TPS test prompt | `"Explain the CAP theorem and its implications for distributed database design."` |
| TPS test max_tokens | 100 |
| TPS test temperature | 0.0 |
| TPS test repetitions | 3 |

## Single variable under test

`--gpu-memory-utilization` values for coder and thinker under co-load (binary search), starting at `coder=0.35, thinker=0.62`.

## Procedure

Skip flags (set to 1 to skip expensive steps on retry):
- `SKIP_CONVERGENCE=1` — skip Convergence start (use if already running at port 8002)
- `SKIP_THINKER=1` — skip thinker start (use if already running at port 30001)
- `SKIP_CODER=1` — skip coder start (use if already running at port 30000)

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_21_coload_vram_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Starting co-load gpu_mem_util values (decrease both if GPU1 OOMs):
CODER_MEM_UTIL=${CODER_MEM_UTIL:-0.35}
THINKER_MEM_UTIL=${THINKER_MEM_UTIL:-0.62}

IK_BIN=/srv/ai/projects/ik_llama.cpp/build/bin/llama-server
CONVERGENCE_GGUF=/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf
TPS_PROMPT="Explain the CAP theorem and its implications for distributed database design."

# === Record VRAM baseline (everything stopped) ===
echo "=== VRAM baseline (pre-load) ===" | tee "${RESULTS_DIR}/vram_log.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram_log.txt"

# ===================================================================
# PHASE 1: Start Convergence
# ===================================================================
SKIP_CONVERGENCE=${SKIP_CONVERGENCE:-0}
if [ "${SKIP_CONVERGENCE}" = "0" ]; then
  echo "=== Starting Convergence ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
  "${IK_BIN}" \
    -m "${CONVERGENCE_GGUF}" \
    -ngl 999 --cpu-moe --no-mmap \
    -b 4096 -ub 2048 -t 32 -np 4 -c 131072 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --jinja --host 0.0.0.0 --port 8002 \
    >> "${RESULTS_DIR}/convergence.log" 2>&1 &
  CONVERGENCE_PID=$!

  tail -f "${RESULTS_DIR}/convergence.log" | stdbuf -oL sed 's/\r//g; s/^/[convergence] /' &
  CONV_TAIL_PID=$!

  for i in $(seq 1 120); do
    curl -sf http://localhost:8002/health 2>/dev/null && echo "[convergence] HEALTH OK" && break
    sleep 1
  done
  kill "${CONV_TAIL_PID}" 2>/dev/null

  curl -sf http://localhost:8002/health \
    || { echo "FATAL: Convergence did not start"; cat "${RESULTS_DIR}/convergence.log" | tail -20; exit 1; }
else
  echo "[skip] SKIP_CONVERGENCE=1"
  curl -sf http://localhost:8002/health || { echo "FATAL: endpoint not live"; exit 1; }
fi

echo "=== VRAM after Convergence ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram_log.txt"

# ===================================================================
# PHASE 2: Start thinker with --max-model-len 131072
# ===================================================================
SKIP_THINKER=${SKIP_THINKER:-0}
if [ "${SKIP_THINKER}" = "0" ]; then
  echo "=== Starting thinker (gpu_mem_util=${THINKER_MEM_UTIL}, max-model-len=131072) ===" | tee -a "${RESULTS_DIR}/vram_log.txt"

  VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
  ./infra/scripts/deploy.sh vllm gpu1 rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
    --trust-remote-code \
    --gpu-mem-util "${THINKER_MEM_UTIL}" \
    --max-model-len 131072 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-seqs 4 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    "--speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":3}" \
    2>&1 | tee "${RESULTS_DIR}/thinker_deploy.log"

  THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker\|gpu1\|27b" | head -1)
  podman logs -f "${THINKER_CONTAINER}" 2>&1 | stdbuf -oL sed 's/^/[thinker] /' &
  LOG_PID=$!

  for i in $(seq 1 180); do
    curl -sf http://localhost:30001/health 2>/dev/null && echo "[thinker] HEALTH OK" && break
    sleep 1
  done
  kill "${LOG_PID}" 2>/dev/null

  curl -sf http://localhost:30001/health \
    || { echo "FATAL: Thinker did not start at gpu_mem_util=${THINKER_MEM_UTIL}"; \
         echo "thinker_start=FAIL gpu_mem_util=${THINKER_MEM_UTIL}" >> "${RESULTS_DIR}/results.txt"; \
         echo "Suggestion: reduce THINKER_MEM_UTIL and retry with SKIP_CONVERGENCE=1 SKIP_THINKER=0"; \
         exit 1; }
else
  echo "[skip] SKIP_THINKER=1"
  curl -sf http://localhost:30001/health || { echo "FATAL: thinker endpoint not live"; exit 1; }
fi

echo "=== VRAM after thinker ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram_log.txt"

# Verify thinker's max-model-len is 131072 (check model config endpoint)
THINKER_CONFIG=$(curl -sf http://localhost:30001/v1/models 2>/dev/null)
echo "Thinker model info: ${THINKER_CONFIG}" | tee -a "${RESULTS_DIR}/vram_log.txt"

# Quick smoke test on thinker
THINKER_SMOKE=$(curl -sf http://localhost:30001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm","prompt":"Hello","max_tokens":5,"temperature":0.0}' 2>/dev/null)
echo "Thinker smoke: ${THINKER_SMOKE}" | tee -a "${RESULTS_DIR}/vram_log.txt"

# ===================================================================
# PHASE 3: Start coder
# ===================================================================
SKIP_CODER=${SKIP_CODER:-0}
if [ "${SKIP_CODER}" = "0" ]; then
  echo "=== Starting coder (gpu_mem_util=${CODER_MEM_UTIL}) ===" | tee -a "${RESULTS_DIR}/vram_log.txt"

  VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
    --gpu-mem-util "${CODER_MEM_UTIL}" \
    --max-model-len 32768 \
    --kv-cache-dtype fp8 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    2>&1 | tee "${RESULTS_DIR}/coder_deploy.log"

  CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|tp2\|35b" | head -1)
  podman logs -f "${CODER_CONTAINER}" 2>&1 | stdbuf -oL sed 's/^/[coder] /' &
  LOG_PID=$!

  for i in $(seq 1 300); do
    curl -sf http://localhost:30000/health 2>/dev/null && echo "[coder] HEALTH OK" && break
    sleep 1
  done
  kill "${LOG_PID}" 2>/dev/null

  curl -sf http://localhost:30000/health \
    || { echo "FATAL: Coder did not start at gpu_mem_util=${CODER_MEM_UTIL}"; \
         echo "coder_start=FAIL gpu_mem_util=${CODER_MEM_UTIL}" >> "${RESULTS_DIR}/results.txt"; \
         echo "Suggestion: reduce CODER_MEM_UTIL and retry with SKIP_CONVERGENCE=1 SKIP_THINKER=1 SKIP_CODER=0"; \
         exit 1; }
else
  echo "[skip] SKIP_CODER=1"
  curl -sf http://localhost:30000/health || { echo "FATAL: coder endpoint not live"; exit 1; }
fi

# ===================================================================
# PHASE 4: Three-model VRAM snapshot
# ===================================================================
echo "=== FINAL three-model co-load VRAM snapshot ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader | tee -a "${RESULTS_DIR}/vram_log.txt"
nvidia-smi --format=csv --query-compute-apps=pid,process_name,used_memory | tee -a "${RESULTS_DIR}/vram_per_process.txt"
echo "=== All running container names ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
podman ps --format "{{.Names}}\t{{.Status}}" | tee -a "${RESULTS_DIR}/vram_log.txt"

# Record configuration used
echo "coder_mem_util=${CODER_MEM_UTIL}" > "${RESULTS_DIR}/results.txt"
echo "thinker_mem_util=${THINKER_MEM_UTIL}" >> "${RESULTS_DIR}/results.txt"
echo "thinker_max_model_len=131072" >> "${RESULTS_DIR}/results.txt"
echo "all_three_loaded=OK" >> "${RESULTS_DIR}/results.txt"

# ===================================================================
# PHASE 5: Convergence TPS under co-load (3 sequential requests)
# ===================================================================
echo "=== Convergence TPS under co-load ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
CONV_TPS_RESULTS="${RESULTS_DIR}/convergence_coload_tps.csv"
echo "rep,ttft_s,tokens_generated,tps" > "${CONV_TPS_RESULTS}"

for REP in 1 2 3; do
  START_MS=$(date +%s%3N)
  RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"convergence\",\"prompt\":\"${TPS_PROMPT}\",\"max_tokens\":100,\"temperature\":0.0}" 2>/dev/null)
  END_MS=$(date +%s%3N)

  ELAPSED_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  TOKENS=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
  TPS=$(python3 -c "
import sys
tokens = '${TOKENS}'
elapsed = ${ELAPSED_S}
try:
    print(round(float(tokens) / elapsed, 2)) if elapsed > 0 else print('N/A')
except:
    print('N/A')
" 2>/dev/null)

  echo "${REP},${ELAPSED_S},${TOKENS},${TPS}" >> "${CONV_TPS_RESULTS}"
  echo "[convergence TPS] rep=${REP} elapsed=${ELAPSED_S}s tokens=${TOKENS} tps=${TPS}"
done

cat "${CONV_TPS_RESULTS}" | tee -a "${RESULTS_DIR}/vram_log.txt"

# Record Convergence TPS isolated baseline for comparison
echo "convergence_isolated_tps=13.99" >> "${RESULTS_DIR}/results.txt"
echo "See results/T_CV3_* for isolated baseline (13.99 t/s)"

echo "=== BENCH_21 complete — results in ${RESULTS_DIR} ==="
```

**Binary search guidance if OOM on GPU1:**

If coder or thinker fails to start (GPU1 OOM), reduce both values and retry. Suggested sequence:
```
# Attempt 1 (default)
CODER_MEM_UTIL=0.35 THINKER_MEM_UTIL=0.62

# Attempt 2 (if GPU1 OOM with thinker+coder)  
CODER_MEM_UTIL=0.33 THINKER_MEM_UTIL=0.60

# Attempt 3 (last resort — minimal KV cache, model weights only)
CODER_MEM_UTIL=0.30 THINKER_MEM_UTIL=0.58

# If attempt 3 fails: co-load is NOT viable. Record as COLOAD_IMPOSSIBLE.
# Document which step fails (thinker load? coder load?) and the exact OOM error.
```

**Important: use SKIP_* flags to avoid restarting components that already loaded:**
```bash
# If thinker loaded at 0.62 but coder OOMs at 0.35 → reduce coder and retry:
CODER_MEM_UTIL=0.33 SKIP_CONVERGENCE=1 SKIP_THINKER=1 bash <this_script>
```

## Metrics to record

| Metric | Source file | Expected / reference value |
|--------|-------------|---------------------------|
| GPU0 VRAM (all three loaded, MiB) | `vram_log.txt` (FINAL snapshot) | ~20–25GB (coder + Convergence; no thinker on GPU0) |
| GPU1 VRAM (all three loaded, MiB) | `vram_log.txt` (FINAL snapshot) | ~28–30GB (coder shard + thinker; tight) |
| VRAM per process breakdown | `vram_per_process.txt` | 4 processes (coder, thinker, convergence + llama-server) |
| Coder gpu_mem_util that worked | `results.txt` | Binary search result |
| Thinker gpu_mem_util that worked | `results.txt` | Binary search result |
| Thinker max-model-len confirmed | `vram_log.txt` (model info) | 131072 |
| VRAM delta from 131072 vs 32768 for thinker | vram_log.txt (thinker-alone vs after ctx upgrade) | ~0 MiB (DeltaNet prediction) |
| Convergence TPS rep 1–3 under co-load | `convergence_coload_tps.csv` | Likely < 13.99 t/s (GPU shared) |
| Convergence isolated TPS reference | settled in docs | 13.99 t/s (T_CV3) |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Thinker starts with --max-model-len 131072 | Health 200 at port 30001 | Record OOM, note VRAM values, write Open from testing |
| Thinker VRAM delta from ctx upgrade | ≤ 100 MiB extra vs 32768 (DeltaNet prediction) | If > 500 MiB: note contradiction of T3.1 finding in Open from testing |
| Coder starts with reduced gpu_mem_util | Health 200 at port 30000 | Binary search per guidance above; if all fail: record COLOAD_IMPOSSIBLE |
| All three co-loaded simultaneously | All three health endpoints return 200 | Record the last failing component and its OOM details |
| Convergence TPS under co-load | Any measured value (informational) | If request times out: record TIMEOUT, note which GPU layers are starved |
| Production restored at end | All three running with correct flags | Restart any that were modified |

## Artifacts to write

1. `results/BENCH_21_coload_vram_<TIMESTAMP>/vram_log.txt` — VRAM snapshots at each phase
2. `results/BENCH_21_coload_vram_<TIMESTAMP>/vram_per_process.txt` — nvidia-smi per-process breakdown
3. `results/BENCH_21_coload_vram_<TIMESTAMP>/convergence_coload_tps.csv` — TPS under co-load
4. `results/BENCH_21_coload_vram_<TIMESTAMP>/results.txt` — key/value config summary
5. `results/BENCH_21_coload_vram_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_21 — Co-load VRAM mapping — <TIMESTAMP>

## Environment
- Coder: cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, vLLM TP=2
- Thinker: rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm, vLLM TP=1 GPU1
- Convergence: unsloth/Qwen3.5-397B-A17B UD-IQ2_M, ik_llama.cpp ngl=999 --cpu-moe

## Co-load outcome
COLOAD_OK / COLOAD_IMPOSSIBLE / PARTIAL (note which component failed)

## Winning gpu_mem_util values
| Model | gpu_mem_util | Notes |
|-------|-------------|-------|
| Coder | <x> | First value that loaded; binary search steps: <list> |
| Thinker | <x> | First value that loaded |

## VRAM snapshot (all three loaded)
| GPU | VRAM used (MiB) | VRAM free (MiB) | Notes |
|-----|----------------|----------------|-------|
| 0   | <x>            | <x>            | Coder shard + Convergence |
| 1   | <x>            | <x>            | Coder shard + Thinker + Convergence |

## Per-process VRAM
<Paste or summarize vram_per_process.txt>

## Thinker ctx upgrade: VRAM delta for --max-model-len 131072 vs 32768
| Phase | GPU1 VRAM used (MiB) |
|-------|---------------------|
| Before thinker (Convergence only) | <x> |
| After thinker (131072 ctx) | <x> |
| Delta | <x> MiB (predicted: ~0 MiB) |

## Convergence TPS under co-load
| Rep | TTFT (s) | Tokens | TPS |
|-----|----------|--------|-----|
| 1   | <x>      | <x>    | <x> |
| 2   | <x>      | <x>    | <x> |
| 3   | <x>      | <x>    | <x> |
| **Mean** | — | — | **<x>** |
| Isolated reference (T_CV3) | — | — | 13.99 |
| Co-load TPS change (%) | — | — | <x>% |

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| Thinker loads with 131072 ctx | PASS/FAIL | |
| VRAM delta ≤ 100 MiB for ctx upgrade | PASS/FAIL | Actual delta: <x> MiB |
| Coder co-loads with thinker | PASS/FAIL | Working value: <x> |
| All three co-loaded | PASS/FAIL | |
| Convergence TPS measured | PASS/FAIL | |

## Verdict
PASS / FAIL / PARTIAL — <one sentence>

## Incidental findings
<Any VRAM anomaly, unexpected engine warning, OOM detail, kernel selection message. One FINDING block per observation.>
<If nothing unusual: "none">

## Open from testing
<Any unexpected blocker, OOM detail, or architectural surprise that needs research-mode attention.>
<If nothing: "none">
```

## Interpretation boundary

**You may:**
- Record all VRAM measurements and TPS values as observed
- Note the binary search results (which values worked, which failed)
- Record whether the thinker's VRAM delta for ctx upgrade was as predicted
- Note Convergence TPS degradation percentage under co-load

**You may NOT:**
- Update `docs/decisions/models.md` or `docs/arch/current.md` with the new gpu_mem_util values
- Conclude whether the co-load configuration is "acceptable" for production
- Update any settled decisions
- Modify production deploy scripts

All production config decisions based on this data are research-mode decisions.

## Stop condition

**Normal:** All three models loaded (or co-load declared impossible after all binary search attempts), VRAM snapshot captured, Convergence TPS measured, `summary.md` written.

**Abnormal:** Write `## Open from testing` in `RESEARCH_STATE.md` if:
- All three binary search attempts fail: write `COLOAD_IMPOSSIBLE: coder + thinker + Convergence cannot coexist at any tested gpu_mem_util; GPU1 budget insufficient. Values tried: [list]. Details: [OOM excerpt].`
- Thinker fails to load with `--max-model-len 131072` even at gpu_mem_util=0.90 alone: write `T_THINKER_CTX_FAIL: max-model-len 131072 causes OOM on thinker TP=1 GPU1 at gpu_mem_util=0.90. VRAM used by thinker alone at 32768 ctx: [x] MiB.`
- Convergence crashes during co-load TPS test: write `COLOAD_CONV_CRASH: Convergence crashed during inference while coder+thinker co-resident. GPU1 VRAM snapshot: [values].`
