# BENCH_16 — T_ENGINE_EVAL: GLM-4.7-Flash on ik_llama.cpp

**Status: READY**
**Blocks: T2.2 re-open (if pass)**
**Blocked by: nothing**

---

## Title
T_ENGINE_EVAL — GLM-4.7-Flash re-evaluation on ik_llama.cpp after vLLM Triton/Blackwell failure

## Objective
Determine whether GLM-4.7-Flash (30B-A3B) is viable on ik_llama.cpp after failing on vLLM 0.19.0 due to a Triton MLA CUDA graph crash on sm_120. If the model loads, passes tool-call smoke test, and reaches ≥150 t/s, re-open T2.2 (GLM vs Qwen3.6-35B coder comparison) using ik_llama.cpp as the engine.

## Why this exists

GLM-4.7-Flash failed on vLLM 0.19.0 with `TRITON_MLA PIECEWISE CUDA graph instability on Blackwell` — a Triton kernel issue specific to sm_120 (RTX 5090). ik_llama.cpp uses its own CUDA kernels, not Triton. The same crash path does not exist. GLM-4.7-Flash is a 30B-A3B MoE model with Multi-head Latent Attention (MLA), sharing the A3B architecture with the current coder. If it is both functional and fast on ik_llama.cpp, it is a legitimate coder candidate and T2.2 must be re-run with the corrected engine.

---

## Prerequisites

### Step 0 — GGUF availability check (run this FIRST; do not download anything yet)

```bash
# Check for known GGUF repositories
pyenv activate hf
HF_HOME=/srv/ai/models

# Search for bartowski (most common GGUF packager)
hf search bartowski/GLM-4.7-Flash-GGUF 2>/dev/null && echo "bartowski FOUND" || echo "bartowski NOT FOUND"
hf search cyankiwi/GLM-4.7-Flash-GGUF 2>/dev/null && echo "cyankiwi FOUND" || echo "cyankiwi NOT FOUND"
hf search unsloth/GLM-4.7-Flash-GGUF 2>/dev/null && echo "unsloth FOUND" || echo "unsloth NOT FOUND"

# Also check HF hub directly
python3 -c "
from huggingface_hub import list_models
results = list(list_models(search='GLM-4.7-Flash GGUF', limit=10))
for r in results:
    print(r.id)
"
```

**If no GGUF found:** record `gguf_available=NO` in `${RESULTS_DIR}/status.txt`, write a one-line note in `RESEARCH_STATE.md` under `## Open from testing`, and stop. This is a prerequisite failure, not a test failure.

**If GGUF found:** record the repository name and proceed to Step 1.

```bash
# This test runs on the HOST.
# Find ik_llama.cpp binary
IK_BIN=$(which llama-server 2>/dev/null || \
         ls /srv/ai/projects/ik_llama.cpp/build/bin/llama-server 2>/dev/null || \
         echo "NOT FOUND")
echo "llama-server binary: ${IK_BIN}"
[ "${IK_BIN}" = "NOT FOUND" ] && echo "STOP — install ik_llama.cpp first" && exit 1

# GPU VRAM baseline
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

---

## Inputs required
- `llama-server` binary from the ik_llama.cpp build already on the host
- `llama-bench` binary in the same directory as `llama-server`
- GLM-4.7-Flash GGUF file (exact path determined during Step 0 availability check — do not proceed if not found)
- `pyenv` with `hf` virtualenv active for HuggingFace Hub searches and downloads
- GPU0 free (or GPU0+1 if model requires tensor-split); production coder must be stopped

---

## Fixed controls

| Control | Value |
|---------|-------|
| Model | GLM-4.7-Flash GGUF (Q4_K_M or Q5_K_M — whichever is available) |
| Engine | ik_llama.cpp (llama-server) |
| GPU layers | `-ngl 999` (all layers to GPU) |
| Context size | `--ctx-size 32768` |
| Server port | 8080 |
| Parallel slots | `-np 1` (single request, TPS measurement) |
| CPU threads | `--threads 8` |
| TP / tensor-split | single GPU GPU0 first; if OOM, try `--tensor-split 0.5,0.5` GPU0+1 |
| Reps (llama-bench) | 3 |

## Single variable under test
**Engine** — ik_llama.cpp vs vLLM 0.19.0 (which crashed). All other factors (model, quantization format) change due to the engine swap.

---

## Procedure

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_16_tengine_eval_glm_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

IK_BIN=$(which llama-server 2>/dev/null || \
         ls /srv/ai/projects/ik_llama.cpp/build/bin/llama-server 2>/dev/null)
IK_BENCH=$(dirname "${IK_BIN}")/llama-bench
```

### Step 1 — Download GGUF

```bash
# Set GGUF_REPO to the repository found in Step 0
GGUF_REPO="bartowski/GLM-4.7-Flash-GGUF"  # adjust if different repo found
echo "gguf_repo=${GGUF_REPO}" >> "${RESULTS_DIR}/status.txt"

pyenv activate hf
# Prefer Q5_K_M for quality; accept Q4_K_M if Q5 not available
HF_HOME=/srv/ai/models hf download "${GGUF_REPO}" --include "*Q5_K_M*" 2>/dev/null \
  || HF_HOME=/srv/ai/models hf download "${GGUF_REPO}" --include "*Q4_K_M*"

# Locate downloaded file
MODEL_PATH=$(ls /srv/ai/models/models--${GGUF_REPO//\//-}/snapshots/*/*.gguf 2>/dev/null \
             | grep -i "Q5_K_M\|Q4_K_M" | head -1)
echo "Model path: ${MODEL_PATH}"
[ -z "${MODEL_PATH}" ] && echo "STOP — model file not found after download" && exit 1
echo "model_path=${MODEL_PATH}" >> "${RESULTS_DIR}/status.txt"
```

### Step 2 — Check model card for MLA-specific flags

```bash
# Check for any ik_llama.cpp MLA-specific flags in the model card or README
# MLA (Multi-head Latent Attention) in ik_llama.cpp may benefit from --mla-attn or similar
"${IK_BIN}" --help 2>&1 | grep -i "mla\|latent" || echo "No MLA-specific flags found in this build"

# Record ik_llama.cpp build info
"${IK_BIN}" --version 2>&1 | tee "${RESULTS_DIR}/ik_version.txt"
```

### Step 3 — Launch ik_llama.cpp server (single GPU first)

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
  | tee "${RESULTS_DIR}/vram_before_load.txt"

"${IK_BIN}" \
  -m "${MODEL_PATH}" \
  -ngl 999 \
  --ctx-size 32768 \
  --port 8080 \
  --host 0.0.0.0 \
  --threads 8 \
  -np 1 \
  >> "${RESULTS_DIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "llama-server PID: ${SERVER_PID}"

# Wait up to 120s
for i in $(seq 1 120); do
  curl -sf http://localhost:8080/health 2>/dev/null && echo "SERVER READY" && break
  sleep 1
done

HEALTH_OK=$(curl -sf http://localhost:8080/health 2>/dev/null && echo 1 || echo 0)
echo "health_ok_single_gpu=${HEALTH_OK}" >> "${RESULTS_DIR}/status.txt"
```

**If single-GPU OOM:** retry with tensor-split across both GPUs.

```bash
# OOM retry — tensor-split GPU0+1
if [ "${HEALTH_OK}" = "0" ]; then
  tail -20 "${RESULTS_DIR}/server.log" | grep -i "oom\|out of memory\|alloc" \
    && echo "OOM confirmed — retrying with --tensor-split 0.5,0.5"
  kill "${SERVER_PID}" 2>/dev/null; sleep 3

  "${IK_BIN}" \
    -m "${MODEL_PATH}" \
    -ngl 999 \
    --tensor-split 0.5,0.5 \
    --ctx-size 32768 \
    --port 8080 \
    --host 0.0.0.0 \
    --threads 8 \
    -np 1 \
    >> "${RESULTS_DIR}/server_tp2.log" 2>&1 &
  SERVER_PID=$!

  for i in $(seq 1 120); do
    curl -sf http://localhost:8080/health 2>/dev/null && echo "SERVER READY (tensor-split)" && break
    sleep 1
  done

  HEALTH_OK=$(curl -sf http://localhost:8080/health 2>/dev/null && echo 1 || echo 0)
  echo "health_ok_tensor_split=${HEALTH_OK}" >> "${RESULTS_DIR}/status.txt"
fi
```

**If still not ready after both attempts:** capture logs and stop.
```bash
if [ "${HEALTH_OK}" = "0" ]; then
  tail -60 "${RESULTS_DIR}/server.log" >> "${RESULTS_DIR}/startup_failure.txt"
  echo "STARTUP_FAILED" > "${RESULTS_DIR}/status.txt"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "${RESULTS_DIR}/startup_failure.txt"
  # Write Open from testing in RESEARCH_STATE.md; skip to Step 7
fi
```

### Step 4 — Record VRAM after model load

```bash
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
  | tee "${RESULTS_DIR}/vram_after_load.txt"
```

### Step 5 — Tool-call smoke test (5 prompts)

```bash
# co01 tool-call suite if available
CO01_DIR="benchmarks/phase2_model_selection/tasks/coder/co01"
PASS_COUNT=0
TOTAL=5

if [ -d "${CO01_DIR}" ]; then
  python3 -m benchmarks.phase2_model_selection.bench \
    --endpoint http://localhost:8080/v1 \
    --results-dir "${RESULTS_DIR}/tool_call_test" \
    --mode tool-call \
    --tasks "${CO01_DIR}" \
    --model local \
    --label "GLM-4.7-Flash-ik" \
    --max-tokens 4096 \
    | tee "${RESULTS_DIR}/tool_call_output.txt"
else
  # Manual tool-call test — 5 single-function invocations
  for i in 1 2 3 4 5; do
    RESULT=$(curl -s http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"local\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Use the get_weather tool to get the weather for London.\"}],
        \"tools\": [{
          \"type\": \"function\",
          \"function\": {
            \"name\": \"get_weather\",
            \"description\": \"Get the current weather\",
            \"parameters\": {
              \"type\": \"object\",
              \"properties\": {\"location\": {\"type\": \"string\"}},
              \"required\": [\"location\"]
            }
          }
        }],
        \"tool_choice\": \"auto\",
        \"max_tokens\": 256
      }")
    TOOL_CALL=$(echo "${RESULT}" | python3 -c "import sys,json; r=json.load(sys.stdin); print('PASS' if r['choices'][0]['message'].get('tool_calls') else 'FAIL')" 2>/dev/null || echo "ERROR")
    echo "Tool-call test ${i}: ${TOOL_CALL}"
    echo "${RESULT}" > "${RESULTS_DIR}/tool_call_${i}.json"
    [ "${TOOL_CALL}" = "PASS" ] && PASS_COUNT=$((PASS_COUNT + 1))
  done
  echo "Tool-call result: ${PASS_COUNT}/${TOTAL}" | tee "${RESULTS_DIR}/tool_call_summary.txt"
fi
```

### Step 6 — TPS measurement (llama-bench, N=1)

```bash
"${IK_BENCH}" \
  -m "${MODEL_PATH}" \
  -ngl 999 \
  -p 512 \
  -n 512 \
  -r 3 \
  | tee "${RESULTS_DIR}/llama_bench_tps.txt"

# Parse TPS (tg = token generation rate)
python3 - <<'EOF'
import re
lines = open("${RESULTS_DIR}/llama_bench_tps.txt".replace("${RESULTS_DIR}", __import__("os").environ.get("RESULTS_DIR",""))).read()
for line in lines.splitlines():
    if '|' in line and 'tg' in line.lower():
        print("TPS row:", line.strip())
EOF
# Reference: coder baseline ~241 t/s (N=1 at 512/512)
# Pass threshold: ≥150 t/s (≥60% of coder baseline)
```

### Step 7 — Qualitative coding quality smoke test (2–3 prompts)

```bash
# Use 2–3 representative coding prompts from T2.x task suite if available
CODER_TASKS=$(ls benchmarks/phase2_model_selection/tasks/coder/co0[2-4]* 2>/dev/null | head -3)

if [ -n "${CODER_TASKS}" ]; then
  python3 -m benchmarks.phase2_model_selection.bench \
    --endpoint http://localhost:8080/v1 \
    --results-dir "${RESULTS_DIR}/quality_smoke" \
    --mode quality \
    --tasks benchmarks/phase2_model_selection/tasks/coder/ \
    --task-filter "co02,co03,co04" \
    --model local \
    --label "GLM-4.7-Flash-ik" \
    --max-tokens 4096 \
    | tee "${RESULTS_DIR}/quality_smoke_output.txt"
else
  echo "Coder task suite not found — skip quality smoke; record as NOT_MEASURED"
  echo "quality=NOT_MEASURED" >> "${RESULTS_DIR}/status.txt"
fi
```

### Step 8 — Stop ik_llama.cpp server

```bash
kill "${SERVER_PID}" 2>/dev/null || pkill -f "llama-server.*8080" 2>/dev/null
sleep 3
echo "llama-server stopped"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
  | tee "${RESULTS_DIR}/vram_after_stop.txt"
```

---

## Metrics to record

| Metric | Source | Notes |
|--------|--------|-------|
| GGUF available | status.txt | YES / NO — if NO, stop |
| GGUF repo | status.txt | bartowski / cyankiwi / other |
| Quantization used | status.txt | Q4_K_M / Q5_K_M |
| Startup success | status.txt | YES (single GPU) / YES (tensor-split) / NO |
| GPU VRAM after load (MiB) | vram_after_load.txt | per GPU |
| Tool-call pass rate | tool_call_summary.txt | X/5 |
| TPS at N=1 (t/s) | llama_bench_tps.txt | coder baseline: ~241 t/s |
| Qualitative coding quality | quality_smoke/ | ACCEPTABLE / POOR / NOT_MEASURED |
| MLA flags used | server.log / ik_version.txt | record any special flags |

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| GGUF available | YES | Record NOT_FOUND in RESEARCH_STATE.md; stop |
| Startup success | Health OK within 120s (single GPU or tensor-split) | Record error class from server.log; stop |
| Tool-call pass rate | ≥ 4/5 | FAIL — model not viable as coder candidate |
| TPS at N=1 | ≥ 150 t/s (≥60% of coder baseline ~241 t/s) | FAIL — too slow; record actual value |
| Qualitative coding quality | ACCEPTABLE | POOR — note for research review |

**Full pass:** tool-call ≥4/5 AND TPS ≥150 t/s. If pass: write "T2.2 re-opened with ik_llama.cpp engine" in `RESEARCH_STATE.md`.

**Any fail:** record outcome in `RESEARCH_STATE.md`. Do NOT re-open T2.2.

---

## Artifacts to write

1. `results/BENCH_16_tengine_eval_glm_<timestamp>/status.txt` — GGUF availability + startup flag
2. `results/BENCH_16_tengine_eval_glm_<timestamp>/vram_before_load.txt`
3. `results/BENCH_16_tengine_eval_glm_<timestamp>/vram_after_load.txt`
4. `results/BENCH_16_tengine_eval_glm_<timestamp>/vram_after_stop.txt`
5. `results/BENCH_16_tengine_eval_glm_<timestamp>/server.log`
6. `results/BENCH_16_tengine_eval_glm_<timestamp>/ik_version.txt`
7. `results/BENCH_16_tengine_eval_glm_<timestamp>/tool_call_summary.txt` (or tool_call_1..5.json)
8. `results/BENCH_16_tengine_eval_glm_<timestamp>/llama_bench_tps.txt`
9. `results/BENCH_16_tengine_eval_glm_<timestamp>/quality_smoke/` (if run)
10. `results/BENCH_16_tengine_eval_glm_<timestamp>/summary.md`:

```markdown
# BENCH_16 — T_ENGINE_EVAL GLM-4.7-Flash ik_llama.cpp — <TIMESTAMP>

## GGUF availability
GGUF_AVAILABLE: YES / NO
Repo: <bartowski/GLM-4.7-Flash-GGUF or other>
Quant: Q4_K_M / Q5_K_M

## Startup
STARTUP_OK (single GPU) / STARTUP_OK (tensor-split) / STARTUP_FAILED
(see server.log if failed)

## VRAM at model load
| GPU | VRAM (MiB) |
|-----|-----------|
| GPU0 | <X> |
| GPU1 | <X> (if tensor-split) |

## Tool-call pass rate
<X>/5

## TPS (N=1)
<X> t/s (coder baseline: ~241 t/s; pass threshold: ≥150 t/s)

## Qualitative coding quality
ACCEPTABLE / POOR / NOT_MEASURED

## MLA flags
<none / --mla-attn / other>

## Verdict
PASS (T2.2 re-opened) / FAIL (reason) / PREREQ_FAIL (no GGUF)

## Notes
<any unexpected errors, architecture warnings, MLA handling messages>
```

**Do NOT write to any file outside `results/BENCH_16_tengine_eval_glm_<timestamp>/`.**

---

## Interpretation boundary

- **You may record** TPS, VRAM, tool-call pass rate, startup outcome, GGUF availability.
- **You may note** that T2.2 should be re-opened if the pass condition is met, and write one line in RESEARCH_STATE.md.
- **You may NOT** update `docs/decisions/settled.md`, declare GLM as a production coder, or modify the coder slot assignment.
- **You may NOT** run a full T2.2 quality comparison — that requires a separate handoff.
- **You may NOT** test other GLM quantizations or context sizes beyond what is specified here.

## Stop condition

**Normal:** status.txt written, tool-call result written, TPS written, server stopped, summary.md written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` and stop if:
- Server crashes with an error other than OOM (architecture not supported, missing GGUF op, MLA kernel crash)
- Tool-call responses are structurally malformed in an unexpected way (not a quality failure — a protocol/parsing failure)
- TPS is below 30 t/s (suggests wrong kernel path, serialization overhead, or architecture misidentified)

**Prerequisite stop:** if no GGUF found in Step 0, stop immediately. Write:
```
## Open from testing
BENCH_16 prerequisite fail: GLM-4.7-Flash GGUF not available on HuggingFace as of <DATE>.
No GGUF found at bartowski/, cyankiwi/, or unsloth/. T2.2 re-evaluation blocked.
```
