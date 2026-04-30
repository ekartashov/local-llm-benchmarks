# BENCH_15 — T_KV3 Path B: 128K Context on ik_llama.cpp (Qwen3.6-27B)

**Status: READY**
**Blocks: nothing**
**Blocked by: nothing** (T3.1 Phase 1 / BENCH_11 proved 0 MiB VRAM delta at 50K; DeltaNet recurrent state is fixed-size regardless of context length)

---

## Title
T_KV3 Path B — 128K context feasibility on ik_llama.cpp with tensor-split for Qwen3.6-27B Q5_K_M

## Objective
Verify that Qwen3.6-27B can serve 128K context on ik_llama.cpp with `--tensor-split 0.5,0.5`, measure actual KV cache VRAM growth at increasing context sizes, confirm recurrent state integrity via th02, and record TPS + TTFT at 128K. Establish whether the production thinker slot can be extended beyond the current 32K ceiling at acceptable cost.

## Why this exists

Qwen3.6-27B uses a GDN (Gated DeltaNet) hybrid architecture. Its recurrent state is fixed-size and lives entirely within each layer. vLLM TP=2 is broken because TP shards the recurrent state across GPUs (T2.4g proved this). ik_llama.cpp `--tensor-split` does layer-split (pipeline parallelism), not tensor parallelism — recurrent state stays on one GPU per layer, so the vLLM TP=2 failure mode does not apply.

T3.1 Phase 1 (BENCH_11) showed 0 MiB VRAM delta from 32K→50K context in vLLM. The same physics applies here: GDN layers have no KV cache growth. The 128K test exercises the full native context limit using ik_llama.cpp directly.

---

## Prerequisites

```bash
# 1. Find ik_llama.cpp binary
IK_BIN=$(which llama-server 2>/dev/null || \
         ls /srv/ai/projects/ik_llama.cpp/build/bin/llama-server 2>/dev/null || \
         echo "NOT FOUND")
echo "llama-server binary: ${IK_BIN}"
[ "${IK_BIN}" = "NOT FOUND" ] && echo "STOP — install ik_llama.cpp first" && exit 1

# 2. Download Q5_K_M GGUF (if not already present)
# pyenv activate hf && HF_HOME=/srv/ai/models hf download unsloth/Qwen3.6-27B-GGUF \
#   --include "*Q5_K_M*"

MODEL_PATH=$(ls /srv/ai/models/models--unsloth--Qwen3.6-27B-GGUF/snapshots/*/Qwen3.6-27B-Q5_K_M.gguf 2>/dev/null | head -1)
echo "Model path: ${MODEL_PATH}"
[ -z "${MODEL_PATH}" ] && echo "STOP — model not found; run HF download first" && exit 1

# 3. GPU VRAM baseline (both GPUs — ik_llama.cpp will use GPU0+1)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
# Note: production coder uses GPU0+1 on port 30000. Stop it before this test.

# 4. Production thinker (vLLM port 30001) — no conflict with ik_llama.cpp port 8080
curl -sf http://localhost:30001/health && echo "THINKER OK (no conflict)" || echo "THINKER DOWN (note — test proceeds regardless)"
```

**This test runs on the HOST.**

## Inputs required
- `llama-server` binary from the ik_llama.cpp build already on the host (used by Convergence at port 8002)
- `llama-bench` binary in the same directory as `llama-server`
- `unsloth/Qwen3.6-27B-GGUF` Q5_K_M file downloaded to `/srv/ai/models/` (see Prerequisites Step 2)
- th02 prompt file at `benchmarks/phase2_model_selection/tasks/thinker/th02*` (or manual prompt text)
- GPU0+1 VRAM free — production coder (TP=2) must be stopped before this test

If the production coder is running (TP=2, GPU0+1), stop it before launching ik_llama.cpp — both claim both GPUs at full VRAM.

```bash
CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|gpu0\|35b" | head -1)
[ -n "${CODER_CONTAINER}" ] && podman stop "${CODER_CONTAINER}" && podman rm "${CODER_CONTAINER}" \
  && echo "Coder stopped" || echo "Coder not running"
sleep 3
```

---

## Fixed controls

| Control | Value |
|---------|-------|
| Model | unsloth/Qwen3.6-27B-GGUF Q5_K_M |
| Engine | ik_llama.cpp (llama-server) |
| Parallelism | `--tensor-split 0.5,0.5` (layer-split, not TP sharding) |
| GPU layers | `-ngl 999` (all layers to GPU) |
| Context size | `--ctx-size 131072` |
| Server port | 8080 |
| Parallel slots | `-np 1` |
| CPU threads | `--threads 8` |
| llama-bench reps | 3 |

## Single variable under test
Context size (1K → 128K) — measuring VRAM growth rate and TPS at each level to confirm the DeltaNet zero-KV-growth property holds in ik_llama.cpp.

---

## Procedure

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_15_tkv3_pathb_128k_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

IK_BIN=$(which llama-server 2>/dev/null || \
         ls /srv/ai/projects/ik_llama.cpp/build/bin/llama-server 2>/dev/null)
IK_BENCH=$(dirname "${IK_BIN}")/llama-bench

MODEL_PATH=$(ls /srv/ai/models/models--unsloth--Qwen3.6-27B-GGUF/snapshots/*/Qwen3.6-27B-Q5_K_M.gguf 2>/dev/null | head -1)
```

### Step 1 — Record GPU VRAM baseline

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
  | tee "${RESULTS_DIR}/vram_baseline.txt"
echo "vram_baseline recorded"
```

### Step 2 — Launch ik_llama.cpp server

```bash
"${IK_BIN}" \
  -m "${MODEL_PATH}" \
  -ngl 999 \
  --tensor-split 0.5,0.5 \
  --ctx-size 131072 \
  --port 8080 \
  --host 0.0.0.0 \
  --threads 8 \
  -np 1 \
  >> "${RESULTS_DIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "llama-server PID: ${SERVER_PID}"

# Wait for server ready (model load may take 30–90s)
for i in $(seq 1 120); do
  curl -sf http://localhost:8080/health 2>/dev/null && echo "SERVER READY" && break
  sleep 1
done

# Check if server is actually up
curl -sf http://localhost:8080/health || { echo "STARTUP FAILED — see server.log"; cat "${RESULTS_DIR}/server.log" | tail -30; }
```

**If server fails to start:**
```bash
# Capture error
tail -50 "${RESULTS_DIR}/server.log" > "${RESULTS_DIR}/startup_failure.txt"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "${RESULTS_DIR}/startup_failure.txt"
echo "STARTUP_FAILED" > "${RESULTS_DIR}/status.txt"
# Write Open from testing in RESEARCH_STATE.md if error is unrecognised
# Skip to Step 8.
```

### Step 3 — Record VRAM after model load

```bash
sleep 5  # let allocations settle
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
  | tee "${RESULTS_DIR}/vram_after_load.txt"

# Parse and display
python3 - <<'EOF'
lines = open("${RESULTS_DIR}/vram_after_load.txt".replace("${RESULTS_DIR}", __import__("os").environ.get("RESULTS_DIR",""))).read().strip().splitlines()
for l in lines:
    idx, used = l.strip().split(", ")
    print(f"GPU{idx} VRAM after load: {used} MiB")
EOF
```

Expected: Q5_K_M for 27B ≈ 19–20 GB total; ~10 GB per GPU at 0.5/0.5 tensor-split.

### Step 4 — th02 quality check (DeltaNet recurrent state gate)

```bash
# th02 prompt — tests DeltaNet gate recurrence with non-trivial causal reasoning
# Use the prompt file if it exists:
TH02_PROMPT_FILE=$(ls benchmarks/phase2_model_selection/tasks/thinker/th02* 2>/dev/null | head -1)

if [ -n "${TH02_PROMPT_FILE}" ]; then
  python3 - <<'PYEOF'
import json, os
prompt_file = os.popen("ls benchmarks/phase2_model_selection/tasks/thinker/th02* 2>/dev/null | head -1").read().strip()
task = json.load(open(prompt_file))
payload = {
    "model": "local",
    "messages": task.get("messages", [{"role": "user", "content": task.get("prompt", "")}]),
    "max_tokens": 16384,
    "stream": False
}
print(json.dumps(payload, indent=2))
PYEOF
fi

# Send th02 prompt via curl
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @<(python3 -c "
import json
payload = {
  'model': 'local',
  'messages': [{'role': 'user', 'content': 'PASTE_TH02_PROMPT_HERE_IF_NO_FILE'}],
  'max_tokens': 16384,
  'stream': False
}
print(json.dumps(payload))
") | python3 -c "
import sys, json
r = json.load(sys.stdin)
content = r['choices'][0]['message']['content']
thinking = '<think>' in content and content.split('<think>')[1].split('</think>')[0].strip() if '<think>' in content else ''
print('THINKING_NONEMPTY:', bool(thinking))
print('ANSWER_SNIPPET:', content[-500:])
" | tee "${RESULTS_DIR}/th02_result.txt"
```

**th02 pass condition:** response contains a non-empty `<think>` block AND answer matches the same correct conclusion as T2.4d baseline. Record CORRECT / FAIL in `${RESULTS_DIR}/th02_result.txt`.

### Step 5 — KV cache VRAM headroom sweep (llama-bench)

```bash
# Sweep at increasing context sizes — measures actual KV growth per level
"${IK_BENCH}" \
  -m "${MODEL_PATH}" \
  -ngl 999 \
  --tensor-split 0.5,0.5 \
  -p 1024,16384,32768,65536,131072 \
  -n 256 \
  -r 3 \
  | tee "${RESULTS_DIR}/llama_bench_kv_sweep.txt"

echo "llama-bench sweep complete"
```

After each benchmark step, also snapshot VRAM (llama-bench restores state between runs; manual VRAM snapshots give the headroom picture):

```bash
for CTX in 1024 16384 32768 65536 131072; do
  echo "--- Sending ${CTX}-token prefill probe ---"
  # Generate a filler prompt of the right length and send to running server
  python3 -c "
import json, math
ctx = ${CTX}
# Generate a prompt that is approximately ctx-1024 tokens (reserve generation budget)
fill_tokens = max(ctx - 1024, 512)
filler = ('The following is a test sequence for context measurement. ' * math.ceil(fill_tokens / 12))[:fill_tokens*4]
payload = {'model': 'local', 'messages': [{'role': 'user', 'content': filler}], 'max_tokens': 64, 'stream': False}
print(json.dumps(payload))
" > /tmp/probe_${CTX}.json

  curl -s http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d @/tmp/probe_${CTX}.json > /dev/null

  VRAM_NOW=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',')
  echo "ctx=${CTX} vram=${VRAM_NOW}" | tee -a "${RESULTS_DIR}/vram_kv_sweep.txt"
  sleep 2
done
```

### Step 6 — TPS measurement at multiple context sizes

```bash
# Parse TPS from llama-bench output
python3 - <<'EOF'
import re
lines = open("${RESULTS_DIR}/llama_bench_kv_sweep.txt".replace("${RESULTS_DIR}", __import__("os").environ.get("RESULTS_DIR",""))).read()
# llama-bench outputs: | model | ... | pp_ms | tg_ms | ...
# Extract tg (token generation) rows
for line in lines.splitlines():
    if '|' in line and ('tg' in line.lower() or 'pp' in line.lower()):
        print(line)
EOF
```

### Step 7 — TTFT at 128K

```bash
# Measure time-to-first-token at 128K context
python3 - <<'EOF'
import json, time, urllib.request

# Generate ~127K token filler prompt
fill = ("Benchmark filler text for extended context measurement. " * 6000)[:500000]
payload = json.dumps({
    "model": "local",
    "messages": [{"role": "user", "content": fill}],
    "max_tokens": 1,
    "stream": True
}).encode()

req = urllib.request.Request(
    "http://localhost:8080/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"}
)
t0 = time.monotonic()
with urllib.request.urlopen(req, timeout=300) as resp:
    for chunk in resp:
        if chunk.strip():
            ttft_ms = round((time.monotonic() - t0) * 1000)
            print(f"TTFT at 128K: {ttft_ms} ms")
            break
EOF
```

Record TTFT result to `${RESULTS_DIR}/ttft_128k.txt`.

### Step 8 — Stop ik_llama.cpp server

```bash
kill "${SERVER_PID}" 2>/dev/null || pkill -f "llama-server.*8080" 2>/dev/null
sleep 3
echo "llama-server stopped"

# Confirm GPU VRAM released
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
  | tee "${RESULTS_DIR}/vram_after_stop.txt"
```

### Step 9 — Verify production thinker still running (sanity check)

```bash
curl -sf http://localhost:30001/health && echo "PRODUCTION THINKER OK" || echo "PRODUCTION THINKER DOWN (note: was it running before this test?)"
# ik_llama.cpp runs on port 8080 — no interference with vLLM port 30001
# If coder was stopped in prerequisite: restore it now
# ./infra/scripts/deploy.sh vllm gpu0,gpu1 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit --tp 2 ...
```

---

## Metrics to record

| Metric | Source | Expected / Reference |
|--------|--------|----------------------|
| GPU0 VRAM after model load (MiB) | vram_after_load.txt | ~10,000 MiB |
| GPU1 VRAM after model load (MiB) | vram_after_load.txt | ~10,000 MiB |
| KV VRAM delta: 1K → 128K (MiB) | vram_kv_sweep.txt | ~0 MiB (DeltaNet prediction) |
| KV growth per 1K tokens (MiB/K) | computed from sweep | ~0 (record empirically) |
| TPS at 1K ctx (t/s) | llama_bench_kv_sweep.txt | record |
| TPS at 16K ctx (t/s) | llama_bench_kv_sweep.txt | record |
| TPS at 32K ctx (t/s) | llama_bench_kv_sweep.txt | record |
| TPS at 64K ctx (t/s) | llama_bench_kv_sweep.txt | record |
| TPS at 128K ctx (t/s) | llama_bench_kv_sweep.txt | record |
| TTFT at 128K (ms) | ttft_128k.txt | record |
| th02 result | th02_result.txt | CORRECT |
| Startup result | status.txt / server.log | STARTUP_OK |

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Startup without OOM | Server ready within 120s | Capture server.log tail; write Open from testing; skip to Step 8 |
| th02 CORRECT | Non-empty `<think>` + correct answer matches T2.4d baseline | FAIL — recurrent state corrupted by layer-split; write Open from testing |
| 128K context served | No OOM or crash during KV sweep at 131072 | Record at what context level OOM occurs |
| KV growth ~0 | VRAM delta 1K→128K < 500 MiB total | If significant growth observed, record actual rate — contradicts DeltaNet prediction |
| TTFT at 128K recorded | Numeric value in ttft_128k.txt | Mark as NOT_MEASURED if request times out |

**Pass:** th02 CORRECT at 128K context + model loads without OOM.
**Fail:** th02 FAIL (recurrent state corrupted even with layer-split) OR OOM during model load.

---

## Artifacts to write

1. `results/BENCH_15_tkv3_pathb_128k_<timestamp>/vram_baseline.txt`
2. `results/BENCH_15_tkv3_pathb_128k_<timestamp>/vram_after_load.txt`
3. `results/BENCH_15_tkv3_pathb_128k_<timestamp>/vram_kv_sweep.txt`
4. `results/BENCH_15_tkv3_pathb_128k_<timestamp>/vram_after_stop.txt`
5. `results/BENCH_15_tkv3_pathb_128k_<timestamp>/llama_bench_kv_sweep.txt`
6. `results/BENCH_15_tkv3_pathb_128k_<timestamp>/th02_result.txt`
7. `results/BENCH_15_tkv3_pathb_128k_<timestamp>/ttft_128k.txt`
8. `results/BENCH_15_tkv3_pathb_128k_<timestamp>/server.log`
9. `results/BENCH_15_tkv3_pathb_128k_<timestamp>/summary.md`:

```markdown
# BENCH_15 — T_KV3 Path B: 128K Context ik_llama.cpp — <TIMESTAMP>

## Startup
STARTUP_OK / OOM_AT_STARTUP / ERROR (see server.log)

## VRAM at model load
| GPU | VRAM (MiB) |
|-----|-----------|
| GPU0 | <X> |
| GPU1 | <X> |
| Total | <X> |

## KV cache VRAM growth
| Context | GPU0 VRAM (MiB) | GPU1 VRAM (MiB) | Delta from 1K |
|---------|----------------|----------------|---------------|
| 1K      | <X>            | <X>            | 0 (baseline)  |
| 16K     | <X>            | <X>            | <±X>          |
| 32K     | <X>            | <X>            | <±X>          |
| 64K     | <X>            | <X>            | <±X>          |
| 128K    | <X>            | <X>            | <±X>          |

## TPS by context size
| Context | TPS (t/s) |
|---------|-----------|
| 1K      | <X>       |
| 16K     | <X>       |
| 32K     | <X>       |
| 64K     | <X>       |
| 128K    | <X>       |

## TTFT at 128K
<X> ms

## th02 result
CORRECT / FAIL

## Verdict
PASS / FAIL / PARTIAL (e.g. loaded but OOM at 128K)

## Notes
<any unexpected errors, kernel warnings, context levels where OOM occurred>
```

**Do NOT write to any file outside `results/BENCH_15_tkv3_pathb_128k_<timestamp>/`.**

---

## Interpretation boundary

- **You may record** VRAM at each context size, TPS, TTFT, th02 outcome, startup result.
- **You may note** whether KV growth matched the DeltaNet zero-growth prediction and at what context level (if any) OOM occurred.
- **You may NOT** update `docs/decisions/settled.md`, production config, or architecture docs.
- **You may NOT** conclude whether ik_llama.cpp should permanently replace vLLM for the thinker role.
- **You may NOT** test other quantizations (Q4_K_M, Q8_0) — that is out of scope for this handoff.

## Stop condition

**Normal:** VRAM sweep written, TPS recorded, TTFT recorded, th02 result recorded, server stopped, summary.md written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` and stop if:
- Server crashes on load with an error that is NOT OOM (architecture not supported, missing op)
- th02 fails with a new, unrecognised error pattern (not the known GDN TP=2 recurrent state failure — that is a different config)
- Context sweep OOMs at a level below 32K (unexpected, suggests Q5_K_M is larger than estimated)
