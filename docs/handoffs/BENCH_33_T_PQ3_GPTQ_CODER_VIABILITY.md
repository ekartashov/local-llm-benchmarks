# BENCH_33 — T_PQ3: GPTQ-Int4 Coder Viability (Host-Native vLLM 0.20.0 + cu130)

**Status:** READY
**Blocks:** T_PQ4 (FP8 TP=2 path — only relevant if this FAILS on tool calls)

---

## Title and objective

Test whether `Qwen3.5-35B-A3B-GPTQ-Int4` (or the Qwen3.6 variant if available) on host-native vLLM 0.20.0 with CUDA 13.0 kernels delivers ≥ 150 t/s N=1 decode **and** passes ≥ 4/5 tool-call probes at TP=1 on a single RTX 5090.

A pass would make GPTQ-Int4 the new production coder: it matches or exceeds APEX ik_llama.cpp's 185 t/s N=1 while restoring vLLM's continuous-batching N=4 throughput (projected 400–500 t/s vs APEX's 217 t/s). A fail on tool calls closes the INT4-via-vLLM path entirely and redirects to the FP8 TP=2 path (Extended Arclight only).

---

## Why this exists

**The N=4 batching problem:** The current production coder (APEX GGUF I-Compact on ik_llama.cpp, BENCH_24) achieves 185 t/s N=1 but only 217 t/s at N=4 aggregate — a 1.17× scaling factor. vLLM PrismaQuant (BENCH_23) achieves 56.5 t/s N=1 but 459 t/s N=4 (8.1× scaling) because vLLM performs continuous batching: decode steps across N concurrent sequences are fused into one batched GEMM call. ik_llama.cpp maintains separate slots without fused cross-sequence computation. For agentic workloads that fan out 4 parallel subagents to the coder, this is a 2.1× aggregate throughput deficit.

**Why GPTQ-Int4 may work where PrismaQuant is slow:** The SM120 bottleneck for vLLM MoE is FlashInfer's CUTLASS grouped GEMM using compute_120a instead of compute_120f, forcing a slow fallback. PrismaQuant uses a mixed NVFP4/MXFP8/BF16 path that partially hits this bottleneck. GPTQ-Int4 uses Marlin kernels (the same path as AWQ INT4), which are NOT routed through FlashInfer CUTLASS grouped GEMM. Marlin bypasses the SM120 issue entirely. Additionally, vLLM 0.20.0 ships CUDA 13.0 (cu130) kernels by default, which includes the compute_120f fix for any remaining FlashInfer paths. Community measurement: 194–197 t/s on a single RTX 5090 with this model.

**Why AWQ INT4 failed (BENCH_23a) may not predict GPTQ failure:** AWQ uses a different calibration method and weight-packing format from GPTQ. The BENCH_23a failure (2/5 tool calls at TP=1 V1) may be AWQ-format-specific or may be INT4-precision-specific. If precision-specific, GPTQ-Int4 will fail the same way. This test determines which hypothesis is correct.

**Host-native requirement:** The project is moving away from Podman containers because CRIU (for 0.28s model swaps) is incompatible with rootless container environments. All future vLLM deployments use host-native pyenv virtualenvs.

---

## Context to read

Before running anything, read these files:

1. `docs/INDEX.md` — current production config (R35), key gotchas
2. `results/BENCH_24_apex1_coder_20260505T235441Z/summary.md` — APEX baseline (185.0/217.1 t/s, 5/5 tools) — this is the bar to beat
3. `results/BENCH_23_*/summary.md` (most recent BENCH_23 run) — PQ reference (56.5/459 t/s), and AWQ BENCH_23a failure (2/5 tools) for comparison

---

## Prerequisites

```bash
echo "=== BENCH_33 Prerequisites ===" && \

# 1. NVIDIA driver version — need >= 575 for CUDA 13.0 runtime
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d'.' -f1) && \
[ "${DRIVER_VER}" -ge 575 ] && \
  echo "[prereq] driver OK (${DRIVER_VER}.x >= 575)" || \
  { echo "[prereq] STOP: driver ${DRIVER_VER}.x too old for cu130; need >= 575"; exit 1; } && \

# 2. Both RTX 5090 GPUs visible
GPU_COUNT=$(nvidia-smi -L | grep -c "RTX 5090") && \
[ "${GPU_COUNT}" -ge 2 ] && \
  echo "[prereq] GPUs OK (${GPU_COUNT}x RTX 5090)" || \
  { echo "[prereq] STOP: expected 2x RTX 5090, found ${GPU_COUNT}"; exit 1; } && \

# 3. pyenv available
command -v pyenv >/dev/null 2>&1 && \
  echo "[prereq] pyenv OK" || \
  { echo "[prereq] STOP: pyenv not found; install before proceeding"; exit 1; } && \

# 4. hf CLI available in hf venv
pyenv activate hf 2>/dev/null && \
  command -v hf >/dev/null 2>&1 && \
  echo "[prereq] hf CLI OK" || \
  { echo "[prereq] STOP: hf CLI not found in pyenv 'hf' virtualenv"; exit 1; } && \

# 5. GPU0 free for coder (APEX or thinker may be running — check and report)
GPU0_USED=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader | tr -d ' MiB') && \
echo "[prereq] GPU0 VRAM used: ${GPU0_USED} MiB (idle target < 1000; kill APEX/ik_llama.cpp if high)" && \

# 6. Thinker endpoint status (informational — we keep it alive if running)
curl -sf http://localhost:30001/health 2>/dev/null && \
  echo "[prereq] thinker alive on 30001 (good — keep running)" || \
  echo "[prereq] thinker not running on 30001 (OK — not required)" && \

echo "=== Prerequisites complete ==="
```

---

## Inputs required

- **GPU:** RTX 5090 (GPU0, 32 GB GDDR7) — GPU1 may hold thinker; this test uses GPU0 only
- **Model (primary):** `Qwen/Qwen3.6-35B-A3B-GPTQ-Int4` — try first (~17.5 GB download)
- **Model (fallback):** `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` — use if 3.6 not available (~17.5 GB)
- **vLLM:** 0.20.0+ with cu130 kernels, installed in pyenv virtualenv `vllm-host`
- **Existing processes to stop:** any ik_llama.cpp server running on port 8080 (APEX coder)
- **Port:** 8080 (same as APEX coder; stop APEX first, restore at end)

---

## Fixed controls

| Control | Value |
|---------|-------|
| Tensor parallel size | 1 (GPU0 only) |
| Context length | 32768 tokens |
| Tool-call parser | `qwen3_coder` |
| Port | 8080 |
| max-num-seqs | 16 |
| gpu-memory-utilization | 0.90 |
| chunked-prefill | enabled |
| vLLM version | 0.20.0+ (cu130) |
| TPS measurement method | `usage.completion_tokens` from response JSON / elapsed wall time |
| Tool-call probe count | 5 |
| TPS N=1 reps | 3 (use reps 2+3 as warm values) |
| TPS N=4 reps | 3 concurrent requests × 3 rounds |
| th02 prompt | EDF scheduler (same as all prior coder benchmarks) |
| th02 max_tokens | 16000 |
| CUDA_VISIBLE_DEVICES | 0 |

---

## Single variable under test

Quantization format: GPTQ-Int4 (Marlin kernel path) vs APEX GGUF I-Compact (ik_llama.cpp mul_mat path) — at fixed TP=1, fixed context, fixed port, on the same GPU0.

---

## Procedure

Skip flags (set to 1 to skip expensive steps on retry):
- `SKIP_SETUP=1` — skip pyenv virtualenv creation and vLLM install (use if `vllm-host` already exists with correct version)
- `SKIP_DOWNLOAD=1` — skip model download (use if files already on disk)
- `SKIP_DEPLOY=1` — skip server start (use if endpoint already up on port 8080)

---

### Step 0 — Stop APEX coder

```bash
# Kill any running ik_llama.cpp server on port 8080
APEX_PID=$(lsof -ti:8080 2>/dev/null || true)
if [ -n "${APEX_PID}" ]; then
  kill "${APEX_PID}" && echo "[step0] APEX coder stopped (PID ${APEX_PID})"
  sleep 3
else
  echo "[step0] port 8080 already free"
fi
```

---

### Step 1 — Install vLLM host-native (pyenv + pip)

```bash
SKIP_SETUP=${SKIP_SETUP:-0}
if [ "${SKIP_SETUP}" = "0" ]; then

  # Create virtualenv if not exists
  pyenv virtualenv 3.11.12 vllm-host 2>/dev/null || echo "[step1] vllm-host already exists"
  pyenv activate vllm-host
  pip install --upgrade pip --quiet

  # Install vLLM 0.20.0+ with cu130 kernels (CUDA 13.0 default in 0.20.0)
  pip install vllm --quiet
  echo "[step1] vLLM installed: $(python -c 'import vllm; print(vllm.__version__)')"

  # Verify CUDA version seen by torch
  CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
  echo "[step1] torch CUDA: ${CUDA_VER}"
  # cu130 is ideal; cu128 also works but may lack some SM120 optimizations

else
  echo "[skip] SKIP_SETUP=1 — skipping install"
  pyenv activate vllm-host
fi

# Always verify vLLM is importable
python -c "import vllm; print('[step1] vLLM OK:', vllm.__version__)" || \
  { echo "FATAL: vLLM not importable"; exit 1; }
```

---

### Step 2 — Download model

```bash
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
if [ "${SKIP_DOWNLOAD}" = "0" ]; then

  pyenv activate hf
  export HF_HOME=/srv/ai/models

  # Try Qwen3.6 GPTQ first (architecture parity with current production coder)
  echo "[step2] Attempting Qwen3.6-35B-A3B-GPTQ-Int4 download..."
  if hf download Qwen/Qwen3.6-35B-A3B-GPTQ-Int4 2>/dev/null; then
    MODEL_REPO="Qwen/Qwen3.6-35B-A3B-GPTQ-Int4"
    echo "[step2] Using Qwen3.6 GPTQ"
  else
    echo "[step2] Qwen3.6 GPTQ not available, falling back to Qwen3.5..."
    hf download Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 || \
      { echo "FATAL: both model downloads failed"; exit 1; }
    MODEL_REPO="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    echo "[step2] Using Qwen3.5 GPTQ (fallback)"
  fi
  echo "MODEL_REPO=${MODEL_REPO}" > /tmp/bench33_model.env

else
  echo "[skip] SKIP_DOWNLOAD=1 — loading MODEL_REPO from /tmp/bench33_model.env"
  source /tmp/bench33_model.env
fi

# Resolve model path on disk
MODEL_PATH=$(python -c "
import os; hf_home = os.environ.get('HF_HOME', '/srv/ai/models')
repo = '${MODEL_REPO}'.replace('/', '--').replace('--', '--models--', 1)
snapshots = os.path.join(hf_home, 'hub', 'models--' + repo.replace('models--', ''), 'snapshots')
import glob; snaps = sorted(glob.glob(snapshots + '/*/'))
print(snaps[-1].rstrip('/') if snaps else '')
" 2>/dev/null)
[ -n "${MODEL_PATH}" ] && echo "[step2] Model path: ${MODEL_PATH}" || \
  { echo "FATAL: could not resolve model path"; exit 1; }
echo "MODEL_PATH=${MODEL_PATH}" >> /tmp/bench33_model.env
```

---

### Step 3 — Create results directory

```bash
source /tmp/bench33_model.env
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="/srv/ai/projects/local-llm-benchmarks/results/BENCH_33_gptq_coder_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}" >> /tmp/bench33_model.env
echo "TIMESTAMP=${TIMESTAMP}" >> /tmp/bench33_model.env
echo "[step3] Results dir: ${RESULTS_DIR}"
echo "model_repo=${MODEL_REPO}" > "${RESULTS_DIR}/run_config.txt"
echo "model_path=${MODEL_PATH}" >> "${RESULTS_DIR}/run_config.txt"
echo "timestamp=${TIMESTAMP}" >> "${RESULTS_DIR}/run_config.txt"
```

---

### Step 4 — Deploy vLLM coder (GPTQ-Int4, GPU0, port 8080)

```bash
source /tmp/bench33_model.env
pyenv activate vllm-host

SKIP_DEPLOY=${SKIP_DEPLOY:-0}
if [ "${SKIP_DEPLOY}" = "0" ]; then

  export CUDA_VISIBLE_DEVICES=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name coder \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16 \
    --max-model-len 32768 \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --port 8080 \
    >> "${RESULTS_DIR}/server.log" 2>&1 &
  SERVER_PID=$!
  echo "${SERVER_PID}" > "${RESULTS_DIR}/server.pid"

  # Stream log to terminal while waiting for health
  tail -f "${RESULTS_DIR}/server.log" | stdbuf -oL sed 's/\r//g; s/^/[vllm] /' &
  TAIL_PID=$!

  echo "[step4] Waiting for server health (up to 300s)..."
  for i in $(seq 1 300); do
    curl -sf http://localhost:8080/health 2>/dev/null && break
    sleep 1
    [ $((i % 30)) -eq 0 ] && echo "[step4] still waiting... ${i}s"
  done
  kill "${TAIL_PID}" 2>/dev/null

  curl -sf http://localhost:8080/health || { echo "FATAL: server did not start"; exit 1; }
  echo "[step4] Server up"

  # Record VRAM after load
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram.txt"
  echo "[step4] VRAM:"
  cat "${RESULTS_DIR}/vram.txt"

else
  echo "[skip] SKIP_DEPLOY=1 — skipping server start"
  curl -sf http://localhost:8080/health || { echo "FATAL: endpoint not live"; exit 1; }
  source /tmp/bench33_model.env
fi
```

> **STOP if startup fails with OOM or "profiling forward pass" error:** the model may need lower `--max-num-seqs`. Retry with `--max-num-seqs 8` and set `SKIP_DEPLOY=0`. Write to Open from testing if it persists.

> **STOP if startup fails with "CUDA error: no kernel image is available":** cu128 wheels installed instead of cu130 — the SM120 Marlin kernel is missing. Run `pip install vllm --extra-index-url https://wheels.vllm.ai/nightly` and retry.

---

### Step 5 — TPS measurement N=1

```bash
source /tmp/bench33_model.env
pyenv activate vllm-host

# Three sequential requests; record completion_tokens and elapsed time per request
PROMPT="Write a Python function that implements binary search on a sorted list. Include type hints, handle edge cases, and add a brief docstring."

for REP in 1 2 3; do
  START=$(date +%s%N)
  RESP=$(curl -sf http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"coder\",
      \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
      \"max_tokens\": 1024,
      \"temperature\": 0
    }")
  END=$(date +%s%N)
  ELAPSED_MS=$(( (END - START) / 1000000 ))
  TOKENS=$(echo "${RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])")
  TPS=$(python3 -c "print(round(${TOKENS} / (${ELAPSED_MS} / 1000.0), 1))")
  echo "[step5] rep${REP}: ${TOKENS} tokens in ${ELAPSED_MS}ms = ${TPS} t/s"
  echo "${RESP}" > "${RESULTS_DIR}/tps_n1_rep${REP}.json"
done
echo "[step5] N=1 TPS recorded (use reps 2+3 as warm values)"
```

---

### Step 6 — TPS measurement N=4

```bash
source /tmp/bench33_model.env
pyenv activate vllm-host

PROMPT="Write a Python function that implements binary search on a sorted list. Include type hints, handle edge cases, and add a brief docstring."

for ROUND in 1 2 3; do
  START=$(date +%s%N)
  # Fire 4 concurrent requests
  for i in 1 2 3 4; do
    curl -sf http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"coder\",
        \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
        \"max_tokens\": 512,
        \"temperature\": 0
      }" > "${RESULTS_DIR}/tps_n4_round${ROUND}_req${i}.json" &
  done
  wait
  END=$(date +%s%N)
  ELAPSED_MS=$(( (END - START) / 1000000 ))

  TOTAL_TOKENS=0
  for i in 1 2 3 4; do
    T=$(python3 -c "
import json
with open('${RESULTS_DIR}/tps_n4_round${ROUND}_req${i}.json') as f:
    d = json.load(f)
print(d['usage']['completion_tokens'])
")
    TOTAL_TOKENS=$((TOTAL_TOKENS + T))
  done
  AGG_TPS=$(python3 -c "print(round(${TOTAL_TOKENS} / (${ELAPSED_MS} / 1000.0), 1))")
  echo "[step6] round${ROUND}: ${TOTAL_TOKENS} total tokens in ${ELAPSED_MS}ms = ${AGG_TPS} t/s aggregate"
done
echo "[step6] N=4 TPS recorded (use rounds 2+3 as warm values)"
```

---

### Step 7 — Tool-call probes (5×)

```bash
source /tmp/bench33_model.env
pyenv activate vllm-host

TOOLS='[{
  "type": "function",
  "function": {
    "name": "execute_code",
    "description": "Execute a code snippet and return the result",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {"type": "string", "description": "Python code to execute"},
        "language": {"type": "string", "enum": ["python"]}
      },
      "required": ["code"]
    }
  }
}]'

PROMPTS=(
  "Compute the first 10 Fibonacci numbers and call execute_code to print them."
  "Write a function to check if a number is prime, then call execute_code to test it on 17."
  "Sort the list [3,1,4,1,5,9,2,6] using bubble sort and call execute_code to show the steps."
  "Calculate the factorial of 10 using recursion and call execute_code to verify."
  "Find all prime numbers up to 50 using the Sieve of Eratosthenes and call execute_code."
)

PASS=0
for i in 1 2 3 4 5; do
  PROMPT="${PROMPTS[$((i-1))]}"
  RESP=$(curl -sf http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"coder\",
      \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
      \"tools\": ${TOOLS},
      \"tool_choice\": \"auto\",
      \"max_tokens\": 2048,
      \"temperature\": 0
    }")
  echo "${RESP}" > "${RESULTS_DIR}/tool_call_${i}.json"

  HAS_TOOL=$(echo "${RESP}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
choices = d.get('choices', [])
if not choices: print('NO_CHOICE'); exit()
msg = choices[0].get('message', {})
tc = msg.get('tool_calls', [])
print('YES' if tc and tc[0].get('function', {}).get('name') == 'execute_code' else 'NO')
")
  echo "[step7] probe${i}: ${HAS_TOOL}"
  [ "${HAS_TOOL}" = "YES" ] && PASS=$((PASS + 1))
done

echo "[step7] Tool-call pass rate: ${PASS}/5"
echo "tool_call_pass=${PASS}" >> "${RESULTS_DIR}/run_config.txt"

# GATE: if < 3/5, stop immediately
[ "${PASS}" -ge 3 ] || {
  echo "FATAL: tool-call pass rate ${PASS}/5 — below minimum threshold"
  echo "ACTION: write Open from testing block in RESEARCH_STATE.md (see Stop condition)"
  echo "ACTION: restore APEX coder before exiting (Step 9)"
  exit 1
}
```

---

### Step 8 — th02 quality probe (EDF scheduler)

```bash
source /tmp/bench33_model.env
pyenv activate vllm-host

TH02_PROMPT="Implement an Earliest Deadline First (EDF) scheduler in Python. Requirements: Task class with task_id, arrival_time, execution_time, deadline attributes. EDF scheduler that processes tasks selecting the one with earliest deadline. Calculate and return schedule order, average waiting time, missed deadline count, CPU utilization. Test with tasks: [(0,3,5),(1,2,4),(2,1,3),(3,4,8)] format: (arrival, exec, deadline). Verify: task (2,1,3) executes first when available due to deadline=3."

RESP=$(curl -sf http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"coder\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${TH02_PROMPT}\"}],
    \"max_tokens\": 16000,
    \"temperature\": 0
  }" \
  --max-time 300)
echo "${RESP}" > "${RESULTS_DIR}/th02.json"

FINISH=$(echo "${RESP}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(d['choices'][0]['finish_reason'])
")
TOKENS=$(echo "${RESP}" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(d['usage']['completion_tokens'])
")
echo "[step8] th02: finish_reason=${FINISH}, tokens=${TOKENS}"
[ "${FINISH}" = "stop" ] && echo "[step8] th02 PASS (finish_reason=stop)" || \
  echo "[step8] th02 WARNING: finish_reason=${FINISH} — truncated or error"
```

---

### Step 9 — Restore APEX coder

```bash
source /tmp/bench33_model.env

# Stop vLLM server
if [ -f "${RESULTS_DIR}/server.pid" ]; then
  kill $(cat "${RESULTS_DIR}/server.pid") 2>/dev/null
  echo "[step9] vLLM server stopped"
fi
# Confirm port free
sleep 5
lsof -ti:8080 2>/dev/null && kill $(lsof -ti:8080) 2>/dev/null || true

# Restart APEX coder (ik_llama.cpp)
APEX_MODEL="/srv/ai/models/hub/models--mudler--Qwen3.6-35B-A3B-APEX-GGUF/snapshots/$(ls /srv/ai/models/hub/models--mudler--Qwen3.6-35B-A3B-APEX-GGUF/snapshots/ | tail -1)/Qwen3.6-35B-A3B-APEX-I-Compact.gguf"
IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"

CUDA_VISIBLE_DEVICES=0 GGML_CUDA_NO_PINNED=1 \
  "${IK_BIN}" \
  -m "${APEX_MODEL}" \
  -ngl 999 -t 32 -np 4 -c 32768 \
  --kv-type q8_0 --no-mmap --jinja \
  --port 8080 \
  >> /tmp/apex_restore.log 2>&1 &

echo "[step9] APEX restore launched, waiting for health..."
for i in $(seq 1 120); do
  curl -sf http://localhost:8080/health 2>/dev/null && break
  sleep 1
done
curl -sf http://localhost:8080/health && echo "[step9] APEX coder restored" || \
  echo "[step9] WARNING: APEX restore did not come up — check /tmp/apex_restore.log"
```

---

### Step 10 — Write summary

```bash
source /tmp/bench33_model.env
echo "[step10] Writing summary to ${RESULTS_DIR}/summary.md"
```

See **Artifacts to write** below for template.

---

## Metrics to record

| Metric | Source | Reference |
|--------|--------|-----------|
| N=1 TPS warm | `tps_n1_rep2.json` + `tps_n1_rep3.json` (avg reps 2+3) | APEX: 185.0 t/s; PQ: 56.5 t/s |
| N=4 aggregate TPS warm | rounds 2+3 of step 6 | APEX: 217.1 t/s; PQ: 459.3 t/s |
| Tool-call pass rate | `tool_call_N.json` (5 files) | AWQ (fail): 2/5; PQ: 5/5 |
| th02 finish_reason | `th02.json` | Should be `stop` |
| th02 completion_tokens | `th02.json` | PQ BENCH_23: ~10,338 tokens |
| GPU0 VRAM used | `vram.txt` | APEX: ~18,500 MiB; PQ: ~27,900 MiB |
| Model version used | `run_config.txt` | 3.6 preferred, 3.5 fallback |
| vLLM version | server log or `python -c "import vllm; print(vllm.__version__)"` | 0.20.0+ |
| CUDA version | server log | 13.x (cu130) preferred |

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|----------------|-------------|
| Server starts without OOM | Health endpoint responds within 300s | Retry with `--max-num-seqs 8`; if still fails write Open from testing |
| Tool-call pass rate | ≥ 4/5 | If < 3/5: stop, write CLOSED in Open from testing, restore APEX |
| TPS N=1 warm | ≥ 150 t/s | If 100–150: note as "below community report, investigate config"; if < 100: write Open from testing |
| TPS N=4 aggregate warm | ≥ 300 t/s | If 150–300: note; if < 150: write Open from testing (batching broken) |
| th02 finish_reason | `stop` | If `length`: note max_tokens=16000 may still be too low; not a hard fail |
| APEX restored | Port 8080 responds after restore | Write Open from testing if restore fails |

---

## Artifacts to write

1. `results/BENCH_33_gptq_coder_<TIMESTAMP>/server.log` — full vLLM startup + serving log
2. `results/BENCH_33_gptq_coder_<TIMESTAMP>/vram.txt` — `nvidia-smi` VRAM after model load
3. `results/BENCH_33_gptq_coder_<TIMESTAMP>/run_config.txt` — model repo, path, timestamp, tool-call pass count
4. `results/BENCH_33_gptq_coder_<TIMESTAMP>/tps_n1_rep{1,2,3}.json` — raw API responses for N=1
5. `results/BENCH_33_gptq_coder_<TIMESTAMP>/tps_n4_round{1,2,3}_req{1,2,3,4}.json` — raw API responses for N=4
6. `results/BENCH_33_gptq_coder_<TIMESTAMP>/tool_call_{1..5}.json` — raw tool-call probe responses
7. `results/BENCH_33_gptq_coder_<TIMESTAMP>/th02.json` — raw EDF scheduler response
8. `results/BENCH_33_gptq_coder_<TIMESTAMP>/summary.md` — use template below

### summary.md template

```markdown
# BENCH_33 — GPTQ-Int4 Coder Viability — <TIMESTAMP>

## Environment
- Model: <MODEL_REPO> (Qwen3.6 or Qwen3.5 GPTQ-Int4)
- Engine: vLLM <version>, cu<cuda_version>, TP=1 GPU0
- Config: --max-num-seqs 16, gpu-mem-util 0.90, ctx=32768, chunked-prefill

## VRAM
| GPU | Used (MiB) | Free (MiB) |
|-----|-----------|-----------|
| GPU0 | <val> | <val> |

## TPS
| N (concurrent) | GPTQ-Int4 (t/s) | APEX baseline (t/s) | PQ baseline (t/s) |
|----------------|-----------------|---------------------|-------------------|
| 1 (warm avg)   | <val>           | 185.0               | 56.5              |
| 4 (warm agg)   | <val>           | 217.1               | 459.3             |

## Tool-call reliability
| Probe | Tool Call? | Tool Name |
|-------|------------|-----------|
| 1 | YES/NO | <name or —> |
| 2 | YES/NO | |
| 3 | YES/NO | |
| 4 | YES/NO | |
| 5 | YES/NO | |

**Pass rate: <N>/5**

## Quality — th02 EDF scheduling
- finish_reason: <stop|length>
- completion_tokens: <N>
- Reasoning intact: YES/NO (check that EDF logic is correct)

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| TPS N=1 >= 150 t/s | PASS/FAIL | |
| TPS N=4 >= 300 t/s agg | PASS/FAIL | |
| Tool-call rate >= 4/5 | PASS/FAIL | |
| th02 finish_reason=stop | PASS/WARN | |

## Verdict
PASS / FAIL — <one line: what this means for coder selection>
```

---

## Interpretation boundary

**The agent MAY:**
- Record all raw results in `results/BENCH_33_*/`
- Write `summary.md` using the template above
- Write one `## Open from testing` block in `RESEARCH_STATE.md` for any abnormal stop (OOM, tool-call hard fail, batching broken)
- Restore the APEX coder when done

**The agent MAY NOT:**
- Update `docs/queue/open.md`, `docs/queue/status.md`, or any architecture document
- Change the production config in `docs/INDEX.md` or `RESEARCH_STATE.md` (beyond the Open from testing block)
- Draw conclusions about whether GPTQ should replace APEX — that decision belongs to the research session
- Retry with different models (Qwen3.6 FP8, different quantization levels) without being asked

---

## Stop condition

**Normal:** Steps 0–10 complete, `summary.md` written, APEX coder restored and healthy on port 8080.

**Abnormal stops — write this exact block in `RESEARCH_STATE.md` under `## Open from testing`:**

If tool-call pass rate < 3/5:
```
## Open from testing

BENCH_33 T_PQ3 — GPTQ-Int4 tool calls FAIL: <N>/5 pass rate at TP=1 V1.
Same failure mode as AWQ (BENCH_23a). INT4 precision insufficient for tool-call generation
on A3B MoE coder at TP=1. CLOSE T_PQ3.
Next step: T_PQ4 — FP8 TP=2 Extended Arclight path (Qwen3.6-35B-A3B-FP8, both GPUs,
thinker CRIU-swapped). Research session required first to verify FP8 model size fits TP=2.
```

If server OOM during startup:
```
## Open from testing

BENCH_33 T_PQ3 — GPTQ startup OOM on GPU0 (32 GB). Unexpected given INT4 ≈ 17.5 GB.
Tried max-num-seqs 16 and 8. Check vram.txt and server.log for actual allocation.
Research session: determine if GPTQ-Int4 uses more VRAM than expected (activation overhead,
CUDA graphs, VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS needed?).
```

If TPS N=1 < 100 t/s (SM120 Marlin path not working as expected):
```
## Open from testing

BENCH_33 T_PQ3 — GPTQ-Int4 TPS N=1 = <val> t/s, far below community 194-197 t/s.
Marlin kernel path may not be engaging (check server.log for quantization kernel selection).
Check vLLM CUDA version (need cu130; cu128 may use slower kernel). Research session required.
```
