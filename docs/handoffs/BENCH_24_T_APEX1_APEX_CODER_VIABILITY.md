# BENCH_24 — T_APEX1: APEX GGUF Coder Viability (I-Compact + fp8 KV)

**Status:** READY (requires model download first — see Phase 1)
**Blocks:** BENCH_25 (T_APEX2), BENCH_26 (T_APEX3)
**Blocked by:** nothing

---

## Title
T_APEX1 — Deploy `mudler/Qwen3.6-35B-A3B-APEX-GGUF` I-Compact on ik_llama.cpp: measure decode TPS, tool-call reliability, and th02 quality vs the PrismaQuant coder baseline.

## Objective
Determine whether APEX GGUF I-Compact (17.3 GB) on ik_llama.cpp can replace or outperform the current PrismaQuant 4.75bit coder on vLLM (56.5 t/s agg N=1, 120.9 t/s decode N=1). The primary gate is tool-call reliability: ik_llama.cpp uses grammar/jinja template rather than vLLM's `qwen3_coder` parser, and Qwen3.6 coder requires think+tool generation which is more complex than previous ik_llama.cpp models tested.

## Why this exists

**APEX format:** Adaptive Precision for EXpert Models — MoE-aware mixed-precision GGUF. Edge layers (first/last 5) and shared experts use Q6_K; middle-layer routed experts use Q4–Q6. I-series imatrix uses code+reasoning+tool-call calibration data (not wikitext) → dramatically lower KL worst-case than uniform quants. I-Compact at 17.3 GB fits entirely on GPU0, freeing ~12+ GB of headroom for Convergence attention layers in co-load (T_APEX2).

**FlashInfer bypass:** ik_llama.cpp does NOT use FlashInfer CUTLASS grouped GEMM — it uses its own CUDA mul_mat kernels. This bypasses the compute_120a vs compute_120f SM120 bottleneck that caps vLLM PQ coder at ~57 t/s. BW ceiling for I-Compact: 1790 GB/s ÷ 17.3 GB ≈ 103 t/s theoretical. Expected actual: 72–90 t/s.

**Critical risk:** ik_llama.cpp's tool-call path for Qwen3.6-35B-A3B coder is untested. GLM-4.7-Flash passed 5/5 tool calls on ik_llama.cpp (BENCH_16) but has no extended thinking. The coder uses think+tool mode, which is more complex. This is the primary gate — if tool calls fail, the coder is not viable on this engine regardless of TPS.

**Do NOT use `-rtr` flag:** `-rtr` (row-tile-repack) forces MoE mul_mat to CPU on ik_llama.cpp. It is incompatible with APEX co-load use case. Omit it entirely.

**MTP head check:** Qwen3.6-35B-A3B was trained with MTP heads. Whether the APEX GGUF export includes them is unknown — check during this benchmark (feeds T_APEX3).

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — production config, port assignments, gotchas (especially gotcha #14 for PQ OOM pattern and gotcha #15 for AWQ TP=1 V1 fail)
2. `docs/arch/convergence.md` — ik_llama.cpp binary location, launch command pattern
3. `docs/decisions/models.md` — APEX I-series variant table, current coder baseline
4. `results/BENCH_23_pq2_phase1_coder_*/summary.md` — PQ coder baseline: 56.5 t/s agg N=1, 120.9 t/s decode N=1, 459.3 t/s N=4

## Prerequisites

```bash
echo "=== BENCH_24 Prerequisites ===" && \

# 1. ik_llama.cpp binary
IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
IK_BENCH="/srv/ai/projects/ik_llama.cpp/build/bin/llama-bench"
[ -x "${IK_BIN}" ] \
  && echo "[prereq] llama-server OK: ${IK_BIN}" \
  || { echo "[prereq] STOP: llama-server not found at ${IK_BIN}"; exit 1; } && \
[ -x "${IK_BENCH}" ] \
  && echo "[prereq] llama-bench OK: ${IK_BENCH}" \
  || echo "[prereq] WARNING: llama-bench not found — TPS via server-only method" && \

# 2. GPU VRAM baseline (both GPUs should have headroom)
echo "[prereq] VRAM baseline:" && \
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader && \

# 3. No existing process on port 8080
ss -tlnp | grep -q ':8080' \
  && echo "[prereq] WARNING: port 8080 in use — stop existing ik_llama.cpp server first" \
  || echo "[prereq] port 8080 free" && \

# 4. Check if I-Compact already downloaded
APEX_DIR="/srv/ai/models/hub/models--mudler--Qwen3.6-35B-A3B-APEX-GGUF"
APEX_FILE=$(ls "${APEX_DIR}"/snapshots/*/Qwen3.6-35B-A3B-APEX-I-Compact*.gguf 2>/dev/null | head -1)
[ -n "${APEX_FILE}" ] \
  && echo "[prereq] I-Compact GGUF found: ${APEX_FILE}" \
  || echo "[prereq] I-Compact GGUF not downloaded yet — Phase 1 will download (~17 GB)" && \

# 5. gguf library available in hf venv for MTP check
pyenv activate hf 2>/dev/null && \
python3 -c "from gguf import GGUFReader; print('[prereq] gguf library OK')" 2>/dev/null \
  || echo "[prereq] WARNING: gguf library not in hf venv — install with: pip install gguf"
```

## Inputs required

- ik_llama.cpp binary at `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- ik_llama.cpp bench at `/srv/ai/projects/ik_llama.cpp/build/bin/llama-bench`
- `mudler/Qwen3.6-35B-A3B-APEX-GGUF` I-Compact GGUF (17.3 GB, downloaded in Phase 1)
- GPU0 free for ik_llama.cpp (stop any existing coder container before starting)
- GPU1 may have thinker running (no interference expected — different GPU)

## Fixed controls

| Control | Value |
|---------|-------|
| Model | mudler/Qwen3.6-35B-A3B-APEX-GGUF I-Compact |
| Engine | ik_llama.cpp main (llama-server) |
| GPU layers | `-ngl 999` (all layers to GPU0) |
| KV type | `--kv-type q8_0` (fp8 KV) |
| CPU threads | `-t 32` |
| Parallel slots | `-np 4` (supports both N=1 and N=4 tests) |
| Context size | `-c 32768` |
| Jinja template | `--jinja` (required for tool calls) |
| Server port | `--port 8080` |
| Host binding | `--host 0.0.0.0` |
| Memory mapping | `--no-mmap` |
| -rtr flag | **NOT SET** (DO NOT use — forces MoE mul_mat to CPU) |
| TPS benchmark prompt tokens | 0 (decode-only via llama-bench) |
| TPS benchmark output tokens | 128 |
| TPS benchmark reps | 5 |
| Tool-call probes | 5 |
| th02 max_tokens | 8192 |
| th02 temperature | 0.0 |

## Single variable under test

**Engine and quantization format:** ik_llama.cpp + APEX GGUF (I-Compact, ~17.3 GB) vs vLLM + PrismaQuant 4.75bit (~27.9 GB). BW ceiling increases from ~64 t/s to ~103 t/s; FlashInfer grouped GEMM bottleneck is bypassed entirely.

## Procedure

Skip flags (set to 1 to skip expensive steps on retry):
- `SKIP_DOWNLOAD=1` — skip model download (use if I-Compact GGUF already exists at `${APEX_FILE}`)
- `SKIP_BENCH=1` — skip llama-bench TPS (use if already measured; goes straight to server tests)
- `SKIP_DEPLOY=1` — skip server startup (use if llama-server already running at port 8080)

```bash
set -euo pipefail
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_24_apex1_coder_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
IK_BENCH="/srv/ai/projects/ik_llama.cpp/build/bin/llama-bench"
APEX_DIR="/srv/ai/models/hub/models--mudler--Qwen3.6-35B-A3B-APEX-GGUF"

# PQ baseline references (BENCH_23)
PQ_DECODE_N1_TPS=120.9
PQ_AGG_N1_TPS=56.5
PQ_AGG_N4_TPS=459.3

# ===================================================================
# PHASE 1: Download I-Compact if not already present
# ===================================================================
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
if [ "${SKIP_DOWNLOAD}" = "0" ]; then
  APEX_FILE=$(ls "${APEX_DIR}"/snapshots/*/Qwen3.6-35B-A3B-APEX-I-Compact*.gguf 2>/dev/null | head -1)
  if [ -z "${APEX_FILE}" ]; then
    echo "=== Downloading APEX I-Compact (~17 GB) ==="
    pyenv activate hf
    HF_HOME=/srv/ai/models hf download mudler/Qwen3.6-35B-A3B-APEX-GGUF \
      --include "*I-Compact*" \
      2>&1 | tee "${RESULTS_DIR}/download.log"
    APEX_FILE=$(ls "${APEX_DIR}"/snapshots/*/Qwen3.6-35B-A3B-APEX-I-Compact*.gguf 2>/dev/null | head -1)
    [ -z "${APEX_FILE}" ] && { echo "FATAL: Download failed — GGUF not found after download"; exit 1; }
    echo "Download complete: ${APEX_FILE}"
  else
    echo "[skip] I-Compact already at: ${APEX_FILE}"
  fi
else
  APEX_FILE=$(ls "${APEX_DIR}"/snapshots/*/Qwen3.6-35B-A3B-APEX-I-Compact*.gguf 2>/dev/null | head -1)
  [ -z "${APEX_FILE}" ] && { echo "FATAL: SKIP_DOWNLOAD=1 but I-Compact GGUF not found"; exit 1; }
  echo "[skip] Using existing: ${APEX_FILE}"
fi

echo "apex_model_path=${APEX_FILE}" > "${RESULTS_DIR}/metadata.txt"
du -sh "${APEX_FILE}" | tee -a "${RESULTS_DIR}/metadata.txt"

# ===================================================================
# PHASE 2: MTP head check (feeds T_APEX3 — no GPU needed)
# ===================================================================
echo "=== MTP head check ==="
pyenv activate hf
pip install gguf -q 2>/dev/null || true

python3 << 'PYTHON' 2>&1 | tee "${RESULTS_DIR}/mtp_check.txt"
import sys
try:
    from gguf import GGUFReader
    model_path = open("/dev/stdin").readline().strip() if False else None
except ImportError:
    print("MTP_CHECK: gguf library not available")
    sys.exit(0)
PYTHON

# Run actual check with model path
python3 - "${APEX_FILE}" << 'PYTHON' 2>&1 | tee "${RESULTS_DIR}/mtp_check.txt"
import sys
model_path = sys.argv[1]
try:
    from gguf import GGUFReader
    print(f"Loading GGUF metadata from: {model_path}")
    r = GGUFReader(model_path, "r")
    all_tensors = [t.name for t in r.tensors]
    print(f"Total tensors: {len(all_tensors)}")
    # Check for MTP/draft heads
    mtp_keywords = ['draft', 'mtp', 'specul', 'accept', 'verify']
    mtp_tensors = [t for t in all_tensors if any(k in t.lower() for k in mtp_keywords)]
    if mtp_tensors:
        print(f"MTP_HEADS_FOUND: YES ({len(mtp_tensors)} tensors)")
        for t in mtp_tensors[:20]:
            print(f"  - {t}")
    else:
        print("MTP_HEADS_FOUND: NO")
        print("Sample tensor names (first 10):")
        for t in all_tensors[:10]:
            print(f"  - {t}")
    # Also print key metadata
    for key in r.fields:
        if any(k in key for k in ['arch', 'expert', 'layer', 'head', 'context']):
            try:
                print(f"  meta: {key} = {r.fields[key].parts[-1].tolist()[:5]}")
            except:
                pass
except Exception as e:
    print(f"MTP_CHECK_ERROR: {e}")
    print("Falling back to strings-based check:")
    import subprocess
    result = subprocess.run(
        ['strings', model_path],
        capture_output=True, text=True, timeout=60
    )
    lines_with_draft = [l for l in result.stdout.splitlines() if 'draft' in l.lower() or 'mtp' in l.lower()]
    if lines_with_draft:
        print(f"MTP_STRINGS_FOUND: YES")
        for l in lines_with_draft[:10]:
            print(f"  - {l}")
    else:
        print("MTP_STRINGS_FOUND: NO (or strings check failed)")
PYTHON

MTP_FOUND=$(grep -c "MTP_HEADS_FOUND: YES" "${RESULTS_DIR}/mtp_check.txt" 2>/dev/null || echo 0)
echo "mtp_heads_found=${MTP_FOUND}" >> "${RESULTS_DIR}/metadata.txt"
echo "MTP head check: $([ ${MTP_FOUND} -gt 0 ] && echo FOUND || echo NOT FOUND)"

# ===================================================================
# PHASE 3: llama-bench decode TPS (single sequence, no server needed)
# ===================================================================
SKIP_BENCH=${SKIP_BENCH:-0}
if [ "${SKIP_BENCH}" = "0" ] && [ -x "${IK_BENCH}" ]; then
  echo "=== llama-bench decode TPS (single sequence) ==="
  # Decode-only: -p 0 (no prompt tokens), -n 128 (output tokens), -r 5 reps
  # This measures pure generation TPS comparable to vLLM "decode N=1 TPS" (120.9 t/s baseline)
  CUDA_VISIBLE_DEVICES=0 "${IK_BENCH}" \
    -m "${APEX_FILE}" \
    -ngl 999 -t 32 \
    --kv-type q8_0 \
    -p 0 -n 128 -r 5 \
    2>&1 | tee "${RESULTS_DIR}/llama_bench_tps.txt"
  echo "llama-bench complete"
elif [ "${SKIP_BENCH}" = "1" ]; then
  echo "[skip] SKIP_BENCH=1 — skipping llama-bench"
else
  echo "[warn] llama-bench not found — TPS measured via server-only method"
fi

# ===================================================================
# PHASE 4: Deploy ik_llama.cpp server
# ===================================================================
SKIP_DEPLOY=${SKIP_DEPLOY:-0}
if [ "${SKIP_DEPLOY}" = "0" ]; then
  # Stop any existing process on port 8080
  EXISTING_PID=$(ss -tlnp 2>/dev/null | grep ':8080' | awk '{print $NF}' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1)
  [ -n "${EXISTING_PID}" ] && { echo "Stopping existing process ${EXISTING_PID}"; kill "${EXISTING_PID}" 2>/dev/null; sleep 3; }

  echo "=== Starting APEX I-Compact server ==="
  CUDA_VISIBLE_DEVICES=0 "${IK_BIN}" \
    -m "${APEX_FILE}" \
    -ngl 999 -t 32 -np 4 -c 32768 \
    --kv-type q8_0 \
    --no-mmap \
    --jinja \
    --host 0.0.0.0 --port 8080 \
    >> "${RESULTS_DIR}/server.log" 2>&1 &
  SERVER_PID=$!
  echo "server_pid=${SERVER_PID}" >> "${RESULTS_DIR}/metadata.txt"

  # Stream logs to terminal while waiting for health
  tail -f "${RESULTS_DIR}/server.log" | stdbuf -oL sed 's/\r//g; s/^/[apex-coder] /' &
  TAIL_PID=$!
  trap 'kill ${TAIL_PID} 2>/dev/null; kill ${SERVER_PID} 2>/dev/null' EXIT

  echo "Waiting for server health (up to 120s)..."
  for i in $(seq 1 120); do
    curl -sf http://localhost:8080/health 2>/dev/null && echo "[apex-coder] HEALTH OK" && break
    sleep 1
  done
  kill "${TAIL_PID}" 2>/dev/null

  curl -sf http://localhost:8080/health \
    || { echo "FATAL: Server did not start. Check ${RESULTS_DIR}/server.log"; exit 1; }
else
  echo "[skip] SKIP_DEPLOY=1"
  curl -sf http://localhost:8080/health || { echo "FATAL: server not live"; exit 1; }
fi

# Record VRAM after load
echo "=== VRAM after APEX I-Compact load ===" | tee "${RESULTS_DIR}/vram.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram.txt"

# Get model name from server (for API calls)
MODEL_NAME=$(curl -sf http://localhost:8080/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "apex-compact")
echo "model_name_from_api=${MODEL_NAME}" >> "${RESULTS_DIR}/metadata.txt"

# ===================================================================
# PHASE 5: Server TPS — N=1 and N=4 concurrent
# ===================================================================
echo "=== Server TPS sweep (N=1, N=4) ===" | tee "${RESULTS_DIR}/tps.txt"
echo "baseline_pq_decode_n1=${PQ_DECODE_N1_TPS} baseline_pq_agg_n1=${PQ_AGG_N1_TPS} baseline_pq_agg_n4=${PQ_AGG_N4_TPS}" >> "${RESULTS_DIR}/tps.txt"

TPS_PROMPT="Write a Python function that implements merge sort, including docstring and type hints."

for N in 1 4; do
  echo "--- N=${N} concurrent clients ---" | tee -a "${RESULTS_DIR}/tps.txt"
  PIDS=()
  START_MS=$(date +%s%3N)
  for CLIENT in $(seq 1 ${N}); do
    curl -sf http://localhost:8080/v1/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL_NAME}\",
           \"prompt\":\"${TPS_PROMPT}\",
           \"max_tokens\":150,\"temperature\":0.0}" \
      > "${RESULTS_DIR}/tps_N${N}_client${CLIENT}.json" &
    PIDS+=($!)
  done
  for PID in "${PIDS[@]}"; do wait "${PID}" 2>/dev/null; done
  END_MS=$(date +%s%3N)

  TOTAL_TOKENS=0
  for CLIENT in $(seq 1 ${N}); do
    T=$(python3 -c "
import json, sys
try:
    d=json.load(open('${RESULTS_DIR}/tps_N${N}_client${CLIENT}.json'))
    print(d.get('usage',{}).get('completion_tokens',0))
except:
    print(0)
" 2>/dev/null || echo 0)
    TOTAL_TOKENS=$((TOTAL_TOKENS + T))
  done
  ELAPSED_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  AGG_TPS=$(python3 -c "print(round(${TOTAL_TOKENS} / max(${ELAPSED_S}, 0.1), 1))")
  PER_REQ_TPS=$(python3 -c "print(round(${TOTAL_TOKENS} / max(${ELAPSED_S}, 0.1) / ${N}, 1))")

  echo "N=${N}: elapsed=${ELAPSED_S}s tokens=${TOTAL_TOKENS} agg_tps=${AGG_TPS} per_req_tps=${PER_REQ_TPS}" | tee -a "${RESULTS_DIR}/tps.txt"
done

# ===================================================================
# PHASE 6: Tool-call reliability (5 probes with thinking)
# ===================================================================
echo "=== Tool-call probes (5) ===" | tee "${RESULTS_DIR}/tool_calls.txt"
echo "probe,has_think,has_tool_call,tool_name,finish_reason" > "${RESULTS_DIR}/tool_calls.csv"

TOOL_DEF='[{"type":"function","function":{"name":"execute_code","description":"Execute Python code and return the result","parameters":{"type":"object","properties":{"code":{"type":"string","description":"Python code to execute"},"language":{"type":"string","description":"Programming language","default":"python"}},"required":["code"]}}}]'

for PROBE in 1 2 3 4 5; do
  PROBE_PROMPT="Write a Python function to compute the nth Fibonacci number using dynamic programming, then call execute_code to test it with n=10 and n=20."
  RESPONSE=$(curl -sf http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL_NAME}\",
      \"messages\": [
        {\"role\": \"system\", \"content\": \"You are a coding assistant. Use the provided tools to help the user.\"},
        {\"role\": \"user\", \"content\": \"${PROBE_PROMPT}\"}
      ],
      \"tools\": ${TOOL_DEF},
      \"tool_choice\": \"auto\",
      \"max_tokens\": 1024,
      \"temperature\": 0.0
    }" 2>/dev/null)

  echo "${RESPONSE}" > "${RESULTS_DIR}/tool_probe_${PROBE}.json"
  python3 << PYTHON >> "${RESULTS_DIR}/tool_calls.csv" 2>/dev/null
import json
try:
    d = json.load(open("${RESULTS_DIR}/tool_probe_${PROBE}.json"))
    msg = d['choices'][0]['message']
    content = msg.get('content', '') or ''
    has_think = 'YES' if ('<think>' in content or msg.get('reasoning_content')) else 'NO'
    tc = msg.get('tool_calls', [])
    has_tool = 'YES' if tc else 'NO'
    tool_name = tc[0]['function']['name'] if tc else 'none'
    finish = d['choices'][0].get('finish_reason', 'unknown')
    print(f"${PROBE},{has_think},{has_tool},{tool_name},{finish}")
except Exception as e:
    print(f"${PROBE},ERROR,ERROR,error,{e}")
PYTHON
  ROW=$(tail -1 "${RESULTS_DIR}/tool_calls.csv")
  echo "Probe ${PROBE}: ${ROW}" | tee -a "${RESULTS_DIR}/tool_calls.txt"
done

PASS_COUNT=$(grep ",YES," "${RESULTS_DIR}/tool_calls.csv" | grep -c ",YES," || echo 0)
echo "Tool-call pass rate: ${PASS_COUNT}/5" | tee -a "${RESULTS_DIR}/tool_calls.txt"

# ===================================================================
# PHASE 7: th02 quality — EDF scheduling (same as BENCH_23 baseline)
# ===================================================================
echo "=== th02 quality (EDF scheduling) ===" | tee "${RESULTS_DIR}/th02.txt"

TH02_PROMPT="Implement a complete Earliest Deadline First (EDF) scheduler in Python. Requirements:
1. Task class with: task_id, arrival_time, execution_time, deadline attributes
2. EDF scheduler that processes a list of tasks, always selecting the task with the earliest deadline
3. Calculate and return: schedule order, average waiting time, missed deadline count, CPU utilization
4. Include a test with tasks: [(0,3,5), (1,2,4), (2,1,3), (3,4,8)] format: (arrival, exec, deadline)
5. Verify correctness: task (2,1,3) should execute first when available due to deadline=3

Provide complete, runnable Python code with the test included."

TH02_START_MS=$(date +%s%3N)
TH02_RESPONSE=$(curl -sf http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${TH02_PROMPT}\"}],
    \"max_tokens\": 8192,
    \"temperature\": 0.0
  }" 2>/dev/null)
TH02_END_MS=$(date +%s%3N)

echo "${TH02_RESPONSE}" > "${RESULTS_DIR}/th02_response.json"
TH02_ELAPSED_S=$(python3 -c "print(round(($TH02_END_MS - $TH02_START_MS) / 1000.0, 1))")
TH02_TOKENS=$(echo "${TH02_RESPONSE}" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(d.get('usage',{}).get('completion_tokens','UNKNOWN'))
except:
    print('UNKNOWN')
" 2>/dev/null)
TH02_FINISH=$(echo "${TH02_RESPONSE}" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(d['choices'][0].get('finish_reason','unknown'))
except:
    print('unknown')
" 2>/dev/null)
TH02_PREVIEW=$(echo "${TH02_RESPONSE}" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    content = d['choices'][0]['message']['content']
    # Strip think block for preview
    import re
    content = re.sub(r'<think>.*?</think>', '[THINK_BLOCK]', content, flags=re.DOTALL)
    print(content[:600])
except Exception as e:
    print(f'PARSE_ERROR: {e}')
" 2>/dev/null)

echo "th02 elapsed: ${TH02_ELAPSED_S}s, tokens: ${TH02_TOKENS}, finish: ${TH02_FINISH}" | tee -a "${RESULTS_DIR}/th02.txt"
echo "th02 preview (600 chars, think stripped):" | tee -a "${RESULTS_DIR}/th02.txt"
echo "${TH02_PREVIEW}" | tee -a "${RESULTS_DIR}/th02.txt"

echo "=== BENCH_24 measurements complete ==="
echo "Results in: ${RESULTS_DIR}"
echo "Now write summary.md. th02 scoring requires human review."
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader >> "${RESULTS_DIR}/vram.txt"
```

**Note on th02 scoring:** The EDF response must be scored for semantic correctness — does the scheduler select task (2,1,3) first due to earliest deadline=3? Score 1–5 per `docs/decisions/scoring.md`. AWQ coder baseline: ~4.5/5 (T2.5). Write UNSCORED if uncertain about algorithmic correctness.

**Note on tool-call failure mode:** If tool calls return empty `tool_calls` array or the model responds in plain text ignoring the tool, try a second set of probes with explicit thinking trigger: add `"chat_format": "chatml"` or check server logs for template selection. The ik_llama.cpp jinja template for Qwen3 should handle tool calls — if it doesn't, record which template was selected from the server startup log.

## Metrics to record

| Metric | Source file | Expected / reference value |
|--------|-------------|---------------------------|
| llama-bench decode TPS (single-seq) | `llama_bench_tps.txt` | BW ceiling ~103 t/s; expected actual 72–90 t/s; PQ baseline 120.9 t/s decode (different engine) |
| Server agg TPS N=1 | `tps.txt` | Expected ≥ 56.5 t/s (PQ agg baseline) |
| Server agg TPS N=4 | `tps.txt` | Expected ≥ 250 t/s (vs PQ 459.3 t/s — note ik_llama.cpp batching may differ) |
| Tool-call pass rate | `tool_calls.csv` | Target ≥ 4/5; PQ BENCH_23b baseline 5/5 |
| Has thinking block in tool responses | `tool_calls.csv` | `has_think=YES` expected (Qwen3.6 coder uses extended thinking) |
| th02 quality score | `th02.txt` + human review | PQ baseline ~4.5/5; PASS requires correct EDF logic |
| th02 finish reason | `th02.txt` | `stop` (if `length`: increase max_tokens or note truncation) |
| GPU0 VRAM after load (MiB) | `vram.txt` | Expected ~17,715 MiB (I-Compact 17.3 GB) + ~1,000 MiB KV (q8_0, 32K ctx) ≈ 18,700 MiB |
| GPU1 VRAM (unchanged) | `vram.txt` | Should show only thinker (if running) |
| MTP heads found | `mtp_check.txt` | YES / NO — determines T_APEX3 proceed/skip |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Download completes | GGUF file exists, readable | Record download error in Open from testing |
| Server starts | `/health` returns 200 | Record error from server.log; check VRAM OOM |
| No `-rtr` in launch command | Verify from startup log | CRITICAL: if rtr present, stop and relaunch without it |
| TPS N=1 agg ≥ 56.5 t/s | PASS | INCONCLUSIVE — record and continue |
| TPS N=1 agg ≥ 80 t/s | STRONG PASS — meaningful win over PQ | |
| Tool-call pass rate ≥ 4/5 | PASS | 3/5 → one retry with explicit chat format; < 3/5 → FAIL (note jinja template used) |
| Think block present in tool responses | Informational — record YES/NO | |
| th02 EDF semantically correct | PASS: task (2,1,3) first, runnable code | FAIL if wrong; UNSCORED if incomplete |
| GPU0 VRAM ≤ 20,000 MiB after load | Confirms VRAM projection for T_APEX2 | If > 20,000 MiB: update co-load math in summary |

## Artifacts to write

1. `results/BENCH_24_apex1_coder_<TIMESTAMP>/download.log` — HF download output (if downloaded)
2. `results/BENCH_24_apex1_coder_<TIMESTAMP>/metadata.txt` — model path, MTP head status, model name
3. `results/BENCH_24_apex1_coder_<TIMESTAMP>/mtp_check.txt` — full MTP head check output
4. `results/BENCH_24_apex1_coder_<TIMESTAMP>/llama_bench_tps.txt` — llama-bench output
5. `results/BENCH_24_apex1_coder_<TIMESTAMP>/server.log` — ik_llama.cpp server startup + runtime
6. `results/BENCH_24_apex1_coder_<TIMESTAMP>/vram.txt` — GPU VRAM at load and end
7. `results/BENCH_24_apex1_coder_<TIMESTAMP>/tps.txt` — server-based N=1/N=4 TPS
8. `results/BENCH_24_apex1_coder_<TIMESTAMP>/tool_calls.csv` — per-probe tool call results
9. `results/BENCH_24_apex1_coder_<TIMESTAMP>/tool_probe_1-5.json` — raw probe responses
10. `results/BENCH_24_apex1_coder_<TIMESTAMP>/th02_response.json` — full th02 response
11. `results/BENCH_24_apex1_coder_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_24 — T_APEX1: APEX GGUF Coder (I-Compact + fp8 KV) — <TIMESTAMP>

## Environment
- Engine: ik_llama.cpp (commit: run `git -C /srv/ai/projects/ik_llama.cpp rev-parse --short HEAD`)
- Model: mudler/Qwen3.6-35B-A3B-APEX-GGUF I-Compact (<size from du -sh>)
- Config: -ngl 999 -t 32 -np 4 --kv-type q8_0 --no-mmap --jinja, port 8080
- Note: -rtr NOT used (would force MoE to CPU)

## VRAM at load
| GPU | VRAM used (MiB) | VRAM free (MiB) | Notes |
|-----|----------------|----------------|-------|
| 0   | <x>            | <x>            | APEX coder |
| 1   | <x>            | <x>            | Thinker (if running) / free |

## TPS
| Method | N | APEX I-Compact | PQ 4.75bit baseline | Delta (%) |
|--------|---|----------------|---------------------|-----------|
| llama-bench decode | 1 | <x> t/s | 120.9 t/s (vLLM) | <x>% |
| server agg | 1 | <x> t/s | 56.5 t/s | <x>% |
| server agg | 4 | <x> t/s | 459.3 t/s | <x>% |

Note: llama-bench and vLLM decode TPS are comparable (pure generation rate). Server agg TPS includes scheduling overhead.

## Tool-call reliability
| Probe | has_think | has_tool_call | tool_name | finish_reason |
|-------|-----------|--------------|-----------|---------------|
| 1     | YES/NO    | YES/NO       | <name>    | <reason>      |
| 2     | YES/NO    | YES/NO       | <name>    | <reason>      |
| 3     | YES/NO    | YES/NO       | <name>    | <reason>      |
| 4     | YES/NO    | YES/NO       | <name>    | <reason>      |
| 5     | YES/NO    | YES/NO       | <name>    | <reason>      |
| **Pass rate** | | **<N>/5** | | |
| PQ BENCH_23b baseline | | 5/5 | | |

## Quality — th02 EDF scheduling
| Dimension | Score | Notes |
|-----------|-------|-------|
| EDF logic correct (earliest deadline first) | <x>/5 or UNSCORED | |
| Task (2,1,3) runs first (deadline=3 is earliest) | PASS/FAIL/UNSCORED | |
| Code is runnable Python | PASS/FAIL | |
| Thinking present in response | YES/NO | |
| Overall th02 score | <x>/5 or UNSCORED | |

## MTP head check
- Result: MTP_HEADS_FOUND: YES/NO
- Tensor count: <N>
- Implication: T_APEX3 PROCEED / T_APEX3 SKIP (no heads in I-Compact)

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| Server starts | PASS/FAIL | |
| TPS N=1 agg ≥ 56.5 t/s | PASS/FAIL | Actual: <x> t/s |
| Tool-call pass rate ≥ 4/5 | PASS/FAIL | Actual: <N>/5 |
| th02 EDF semantically correct | PASS/FAIL/UNSCORED | |
| GPU0 VRAM ≤ 20,000 MiB | PASS/FAIL | Actual: <x> MiB |

## Verdict
PASS / FAIL / PARTIAL — <one sentence>

## T_APEX2 VRAM projection (fill if PASS)
With I-Compact at <VRAM_USED> MiB on GPU0:
- GPU0 headroom for Convergence: <32,768 - VRAM_USED> MiB
- GPU1 headroom for Convergence: ~2,500 MiB (BENCH_21 thinker measured)
- Combined: <combined> MiB → ngl = (<combined> - 6,247) / 113 = <ngl> layers → ~<tps> t/s

## Incidental findings
<Any unexpected behavior from jinja template, ik_llama.cpp startup messages, KV type confirmation, GPU isolation findings, or other observations.>
<If nothing: "none">

## Open from testing
<Any blocker for T_APEX2 / T_APEX3. If tool calls fail all 5 probes, record exact failure mode and raw JSON of one probe response.>
<If nothing: "none">
```

## Interpretation boundary

**You may:**
- Record TPS, tool-call pass rate, VRAM, MTP head status
- Copy the raw th02 response text into the summary
- Note which jinja template was selected from the server startup log
- Compute the T_APEX2 VRAM projection from the actual GPU0 VRAM at load

**You may NOT:**
- Score th02 without verifying EDF algorithmic correctness in the code
- Update `docs/decisions/models.md` or promote APEX to production
- Update `docs/arch/current.md` with a new coder config
- Conclude on T_APEX2 or T_APEX3 proceed/skip without writing Open from testing first

## Stop condition

**Normal:** llama-bench complete, server running, N=1/N=4 TPS measured, 5 tool-call probes captured, th02 response captured, summary.md written with raw th02 text for human scoring, MTP head status recorded.

**Abnormal:** Write `## Open from testing` in `RESEARCH_STATE.md` if:
- Server fails to start: `BENCH_24_LOAD_FAIL: APEX I-Compact failed on ik_llama.cpp. VRAM at failure: GPU0=[x]MiB. Error excerpt: [from server.log]. Note: check if GGUF format is supported by current ik_llama.cpp build.`
- Tool calls 0/5: `BENCH_24_TOOL_FAIL: APEX I-Compact 0/5 tool calls via ik_llama.cpp jinja. Jinja template used: [from server startup log]. Raw probe 1 response: [first 300 chars of tool_probe_1.json].`
- TPS < 40 t/s (below PQ agg with scheduling overhead): `BENCH_24_TPS_LOW: APEX I-Compact <x> t/s agg N=1 — below PQ 4.75bit agg (56.5 t/s). llama-bench decode: <y> t/s. VRAM: <z> MiB. May indicate compute-bound rather than BW-bound path.`
