# BENCH_19 — T_MTP1: MTP Speculative Decoding on PrismaQuant Thinker (n=1,2,3 Sweep)

**Status:** READY
**Blocks:** nothing
**Blocked by:** nothing

---

## Title and objective

Measure the TPS impact of enabling native Multi-Token Prediction (MTP) speculative decoding on the production thinker (`rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm`). Sweep `num_speculative_tokens` n=1, n=2, n=3. The rdtand author states n=3 is optimal for this model. BENCH_13 ran MTP on the old AWQ thinker (superseded 2026-05-01) and found +31.8% at N=1 / +51% at N=4 — that result does not apply to the current production model.

The result matters because the thinker regressed 26–33% vs AWQ when promoted to PrismaQuant (51.3 t/s vs 76.9 t/s at seq=1). MTP could recover a significant fraction of that gap at zero additional VRAM cost (MTP head is already in the model weights).

---

## Why this exists

BENCH_12 established PrismaQuant as production thinker on quality grounds (2026-05-01). TPS regression was accepted because full NVFP4 (CUDA 13.0) will eventually close the gap — but CUDA 13.0 container is deferred. MTP is a near-zero-cost intermediate gain: one extra flag, same model, same hardware.

**Feasibility uncertainty:** PrismaQuant requires `VLLM_ENGINE_ITERATOR_SOURCE=LEGACY` to force the V0 engine path (compressed-tensors quantization is not compatible with V1 on our config). MTP speculative decoding in vLLM was developed primarily for V1. Whether `--speculative-config '{"method":"mtp",...}'` is accepted by the V0 LEGACY iterator path is **unknown and must be verified in Step 2 before any measurement**. If the deploy fails, that is the primary result — record the error and stop.

**Token counting caveat:** The standard T_PAR1 throughput sweep script counts SSE chunks, not actual tokens. When MTP delivers multiple tokens per chunk, this undercounts TPS by ~2×. **All TPS measurements in this benchmark must use `usage.completion_tokens` from the API response.** A custom measurement script is provided in the procedure.

---

## Context to read

Before running anything, read these files:

1. `docs/INDEX.md` — current production config, open questions, key gotchas
2. `docs/procedures/vllm-deploy.md` — deploy commands, env vars, V0 engine flags
3. `results/BENCH_12_prismaquant_thinker_*/summary.md` — PrismaQuant baseline TPS (51.3 t/s seq=1 / 198.9 t/s seq=4). This is the comparison target.
4. `docs/handoffs/BENCH_13_T_MTP1_THINKER_MTP.md` — AWQ MTP result for reference (+31.8% N=1, +51% N=4 on AWQ). Do not cite these as PrismaQuant numbers.

---

## Prerequisites

```bash
echo "=== Prerequisites ===" && \

# 1. PrismaQuant model files present
ls /srv/ai/models/hub/ | grep -i "PrismaQuant\|prismaquant\|Qwen3.6-27B-PrismaQuant" \
  && echo "[prereq] PrismaQuant model files OK" \
  || { echo "[prereq] STOP: PrismaQuant model not found — run: pyenv activate hf && HF_HOME=/srv/ai/models hf download rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm"; exit 1; } && \

# 2. infra/scripts/deploy.sh exists
[ -x ./infra/scripts/deploy.sh ] \
  && echo "[prereq] deploy.sh OK" \
  || { echo "[prereq] STOP: deploy.sh not found or not executable"; exit 1; } && \

# 3. GPU1 mostly free (thinker not running)
VRAM1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
echo "[prereq] GPU1 VRAM used: ${VRAM1} MiB"
[ "${VRAM1}" -gt 25000 ] \
  && echo "[prereq] WARN: GPU1 has a large process running — stop it before deploying thinker" \
  || echo "[prereq] GPU1 headroom OK" && \

# 4. BENCH_12 baseline results present (for comparison)
ls results/BENCH_12_prismaquant_thinker_*/summary.md 2>/dev/null | head -1 \
  && echo "[prereq] BENCH_12 baseline found" \
  || echo "[prereq] WARN: BENCH_12 results not found — baseline TPS from docs: 51.3 t/s / 198.9 t/s"

echo "=== End prerequisites ==="
```

---

## Inputs required

- `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm` model files in `/srv/ai/models/`
- `infra/scripts/deploy.sh` at repo root
- GPU1 with ~24 GB VRAM free
- Port 30001 free
- Python 3.x available on host for the measurement script

---

## Fixed controls

| Control | Value |
|---------|-------|
| Model | rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm |
| Engine | vLLM TP=1 GPU1 |
| VLLM_USE_V1 | 0 (mandatory) |
| VLLM_ENGINE_ITERATOR_SOURCE | LEGACY (mandatory for compressed-tensors) |
| VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS | 1 |
| --hf-overrides | `{"architectures":["Qwen3_5ForCausalLM"]}` (REQUIRED: model config declares VL arch) |
| --gpu-mem-util | 0.90 |
| --ctx | 32768 |
| --kv-cache-dtype | fp8 |
| --enable-chunked-prefill | ON |
| --max-num-seqs | 4 |
| --tool-call-parser | qwen3_coder |
| --reasoning-parser | qwen3 |
| --enable-auto-tool-choice | ON |
| TPS measurement method | `usage.completion_tokens` via API (NOT SSE chunk counting) |
| Reps per N per mtp_n | 5 |
| Concurrency levels tested | N=1, N=4 |
| MTP baseline (no MTP) | BENCH_12: 51.3 t/s (N=1), 198.9 t/s (N=4) |

---

## Single variable under test

`num_speculative_tokens` in `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`, sweeping N = 1, 2, 3. All other flags identical to production config.

---

## Procedure

Skip flags (set to 1 to skip expensive steps on retry):
- `SKIP_DOWNLOAD=1` — skip model download check (use if files already confirmed present)
- `SKIP_DEPLOY=1` — skip deploy and go straight to measurement (use if MTP thinker is already up)

### Setup

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_19_mtp1_prismaquant_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
echo "Results dir: ${RESULTS_DIR}"

# Baseline TPS from BENCH_12 (used for delta calculations)
BASELINE_N1=51.3
BASELINE_N4=198.9
```

### Step 1 — Download model if needed

```bash
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
if [ "${SKIP_DOWNLOAD}" = "0" ]; then
  ls /srv/ai/models/hub/ | grep -qi "prismaquant" \
    && echo "[skip] PrismaQuant model already present" \
    || {
      echo "[download] Downloading rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm ..."
      pyenv activate hf && \
        HF_HOME=/srv/ai/models hf download rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
        >> "${RESULTS_DIR}/download.log" 2>&1
      echo "[download] Done"
    }
else
  echo "[skip] SKIP_DOWNLOAD=1 — skipping download"
fi
```

### Step 2 — Deploy PrismaQuant thinker WITH MTP n=1 (feasibility gate)

**This is the most important step.** If the deploy fails with an error about speculative decoding not supported, or MTP incompatible with LEGACY iterator, or any startup crash — record the error and proceed to Step 7 (restore production, stop).

```bash
SKIP_DEPLOY=${SKIP_DEPLOY:-0}
if [ "${SKIP_DEPLOY}" = "0" ]; then
  # Stop any existing thinker container
  EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
  [ -n "${EXISTING}" ] && {
    echo "[deploy] Stopping existing: ${EXISTING}"
    podman stop "${EXISTING}" && podman rm "${EXISTING}"
    sleep 3
  }

  echo "[deploy] Starting PrismaQuant thinker + MTP n=1 ..."
  VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm gpu1 rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
    --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-seqs 4 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    >> "${RESULTS_DIR}/deploy_mtp_n1.log" 2>&1 &
  DEPLOY_PID=$!

  # Stream container logs to terminal while waiting
  sleep 5
  CONTAINER_NAME=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
  if [ -n "${CONTAINER_NAME}" ]; then
    podman logs -f "${CONTAINER_NAME}" 2>&1 | stdbuf -oL sed 's/^/[vllm] /' &
    LOG_PID=$!
  fi

  # Wait up to 300s for health
  HEALTH_OK=0
  for i in $(seq 1 300); do
    curl -sf http://localhost:30001/health 2>/dev/null && { HEALTH_OK=1; break; }
    sleep 1
  done
  [ -n "${LOG_PID}" ] && kill "${LOG_PID}" 2>/dev/null

  if [ "${HEALTH_OK}" = "0" ]; then
    echo "[FATAL] Thinker did not come up within 300s"
    echo "--- Last 50 lines of deploy log ---"
    tail -50 "${RESULTS_DIR}/deploy_mtp_n1.log"
    echo ""
    echo "ACTION: Check for errors about 'speculative', 'LEGACY', 'compressed_tensors', or 'MTP'."
    echo "If MTP is not supported with V0+LEGACY, write Open from testing and proceed to Step 7."
    exit 1
  fi

  # Verify model identity
  curl -s http://localhost:30001/v1/models \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[deploy] Model:', d['data'][0]['id'])"

  # Record VRAM at n=1 deploy
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "${RESULTS_DIR}/vram_mtp_n1.txt"
  echo "[deploy] MTP n=1 deploy successful"
else
  echo "[skip] SKIP_DEPLOY=1 — verifying endpoint is live"
  curl -sf http://localhost:30001/health || { echo "FATAL: endpoint not live"; exit 1; }
  echo "[skip] Endpoint OK"
fi
```

### Step 3 — Measure TPS with MTP n=1

This script uses `usage.completion_tokens` (not SSE chunk counting) to get accurate TPS under MTP.

```bash
cat > /tmp/bench_mtp_tps.py << 'PYEOF'
#!/usr/bin/env python3
"""
MTP-safe TPS measurement.
Uses usage.completion_tokens — correct when MTP delivers multiple tokens per SSE chunk.
"""
import os, sys, json, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error

ENDPOINT = "http://localhost:30001/v1/chat/completions"
MODEL = os.environ.get("MODEL_ID", "rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm")
RESULTS_DIR = os.environ.get("RESULTS_DIR", ".")
MTP_N = int(os.environ.get("MTP_N", "1"))
REPS = int(os.environ.get("REPS", "5"))
CONCURRENCY_LEVELS = [1, 4]

PROMPT = (
    "Explain the difference between TCP and UDP in exactly three sentences. "
    "Be precise and technical."
)

def single_request():
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256,
        "stream": False
    }).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    tokens = body["usage"]["completion_tokens"]
    tps = tokens / elapsed if elapsed > 0 else 0
    return tps, tokens, elapsed

def measure_concurrent(n):
    """Run N concurrent requests, report aggregate TPS."""
    tps_samples = []
    for _ in range(REPS):
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(single_request) for _ in range(n)]
            results = [f.result() for f in as_completed(futures)]
        t_total = time.perf_counter() - t_start
        total_tokens = sum(r[1] for r in results)
        agg_tps = total_tokens / t_total
        tps_samples.append(agg_tps)
    return {
        "mean": round(statistics.mean(tps_samples), 1),
        "stdev": round(statistics.stdev(tps_samples) if len(tps_samples) > 1 else 0, 1),
        "samples": [round(x, 1) for x in tps_samples],
    }

results = {}
for n in CONCURRENCY_LEVELS:
    print(f"  Measuring N={n} (MTP n={MTP_N}, {REPS} reps) ...", flush=True)
    r = measure_concurrent(n)
    results[f"n{n}"] = r
    print(f"  N={n}: {r['mean']} t/s ± {r['stdev']} (samples: {r['samples']})")

out = {"mtp_n": MTP_N, "concurrency": results, "model": MODEL}
outfile = os.path.join(RESULTS_DIR, f"tps_mtp_n{MTP_N}.json")
with open(outfile, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {outfile}")
PYEOF

echo "[measure] Running TPS measurement for MTP n=1 ..."
MODEL_ID=$(curl -s http://localhost:30001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
MODEL_ID="${MODEL_ID}" MTP_N=1 REPS=5 RESULTS_DIR="${RESULTS_DIR}" python3 /tmp/bench_mtp_tps.py \
  | tee "${RESULTS_DIR}/tps_mtp_n1_stdout.txt"
```

### Step 4 — Quality gate: th02 (run once, applies to all n)

```bash
# Run th02 — the DeltaNet recurrent state correctness gate
# If th02 fails here, DO NOT proceed to n=2 or n=3 — MTP is corrupting the decode path
TH02_PROMPT_FILE=$(ls benchmarks/phase2_model_selection/tasks/thinker/th02* 2>/dev/null | head -1)
if [ -z "${TH02_PROMPT_FILE}" ]; then
  echo "[quality] th02 prompt file not found at benchmarks/phase2_model_selection/tasks/thinker/"
  echo "[quality] Run th02 manually: send the EDF scheduling prompt to http://localhost:30001/v1/chat/completions"
  echo "[quality] Record response in ${RESULTS_DIR}/th02_mtp_n1.txt and score manually"
else
  MODEL_ID=$(curl -s http://localhost:30001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
  python3 -m benchmarks.phase2_model_selection.bench \
    --endpoint http://localhost:30001/v1 \
    --results-dir "${RESULTS_DIR}/th02_n1" \
    --mode quality \
    --tasks benchmarks/phase2_model_selection/tasks/thinker/ \
    --task-filter th02 \
    --model "${MODEL_ID}" \
    --label "PQ-MTP-n1" \
    --max-tokens 16384 \
    2>&1 | tee "${RESULTS_DIR}/th02_n1_stdout.txt"
  echo "[quality] th02 output written to ${RESULTS_DIR}/th02_n1/"
fi

echo ""
echo "MANUAL SCORE REQUIRED: Read ${RESULTS_DIR}/th02_n1/ output."
echo "th02 PASS condition: model reaches the same correct EDF scheduling answer as in BENCH_12."
echo "If FAIL (wrong answer, empty <think> block, or garbled output): STOP — do not proceed to n=2."
read -p "th02 result [PASS/FAIL]: " TH02_RESULT
echo "${TH02_RESULT}" > "${RESULTS_DIR}/th02_result.txt"
[ "${TH02_RESULT}" = "PASS" ] || { echo "[STOP] th02 FAIL — MTP n=1 corrupts decode path. Proceed to Step 7."; exit 0; }
```

### Step 5 — Redeploy with n=2, measure TPS

```bash
echo "[deploy] Redeploying with MTP n=2 ..."
EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${EXISTING}" ] && podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 3

VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  >> "${RESULTS_DIR}/deploy_mtp_n2.log" 2>&1 &

CONTAINER_NAME=$(sleep 5; podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${CONTAINER_NAME}" ] && podman logs -f "${CONTAINER_NAME}" 2>&1 | stdbuf -oL sed 's/^/[vllm-n2] /' & LOG_PID=$!

HEALTH_OK=0
for i in $(seq 1 300); do
  curl -sf http://localhost:30001/health 2>/dev/null && { HEALTH_OK=1; break; }; sleep 1
done
[ -n "${LOG_PID}" ] && kill "${LOG_PID}" 2>/dev/null

if [ "${HEALTH_OK}" = "0" ]; then
  echo "[WARN] MTP n=2 deploy failed — recording and skipping n=2"
  tail -30 "${RESULTS_DIR}/deploy_mtp_n2.log" | tee "${RESULTS_DIR}/deploy_mtp_n2_error.txt"
else
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "${RESULTS_DIR}/vram_mtp_n2.txt"
  MODEL_ID=$(curl -s http://localhost:30001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
  MODEL_ID="${MODEL_ID}" MTP_N=2 REPS=5 RESULTS_DIR="${RESULTS_DIR}" python3 /tmp/bench_mtp_tps.py \
    | tee "${RESULTS_DIR}/tps_mtp_n2_stdout.txt"
fi
```

### Step 6 — Redeploy with n=3, measure TPS

```bash
echo "[deploy] Redeploying with MTP n=3 ..."
EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${EXISTING}" ] && podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 3

VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  >> "${RESULTS_DIR}/deploy_mtp_n3.log" 2>&1 &

CONTAINER_NAME=$(sleep 5; podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${CONTAINER_NAME}" ] && podman logs -f "${CONTAINER_NAME}" 2>&1 | stdbuf -oL sed 's/^/[vllm-n3] /' & LOG_PID=$!

HEALTH_OK=0
for i in $(seq 1 300); do
  curl -sf http://localhost:30001/health 2>/dev/null && { HEALTH_OK=1; break; }; sleep 1
done
[ -n "${LOG_PID}" ] && kill "${LOG_PID}" 2>/dev/null

if [ "${HEALTH_OK}" = "0" ]; then
  echo "[WARN] MTP n=3 deploy failed — recording and skipping n=3"
  tail -30 "${RESULTS_DIR}/deploy_mtp_n3.log" | tee "${RESULTS_DIR}/deploy_mtp_n3_error.txt"
else
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "${RESULTS_DIR}/vram_mtp_n3.txt"
  MODEL_ID=$(curl -s http://localhost:30001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
  MODEL_ID="${MODEL_ID}" MTP_N=3 REPS=5 RESULTS_DIR="${RESULTS_DIR}" python3 /tmp/bench_mtp_tps.py \
    | tee "${RESULTS_DIR}/tps_mtp_n3_stdout.txt"
fi
```

### Step 7 — Restore production thinker (MANDATORY — run regardless of outcome)

```bash
echo "[restore] Restoring production thinker (no MTP) ..."
EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${EXISTING}" ] && podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 3

VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
  >> "${RESULTS_DIR}/deploy_restore.log" 2>&1 &

CONTAINER_NAME=$(sleep 5; podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${CONTAINER_NAME}" ] && podman logs -f "${CONTAINER_NAME}" 2>&1 | stdbuf -oL sed 's/^/[vllm-restore] /' & LOG_PID=$!

for i in $(seq 1 300); do
  curl -sf http://localhost:30001/health 2>/dev/null && { echo "[restore] PRODUCTION THINKER RESTORED"; break; }
  sleep 1
done
[ -n "${LOG_PID}" ] && kill "${LOG_PID}" 2>/dev/null

# Verify correct model
curl -s http://localhost:30001/v1/models \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('[restore] Active model:', d['data'][0]['id'])"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

### Step 8 — Print summary table

```bash
python3 - <<'PYEOF'
import json, os, glob

RESULTS_DIR = os.environ.get("RESULTS_DIR", "")
if not RESULTS_DIR:
    # Find latest BENCH_19 dir
    dirs = sorted(glob.glob("results/BENCH_19_mtp1_prismaquant_*"), reverse=True)
    RESULTS_DIR = dirs[0] if dirs else "."

BASELINE_N1 = 51.3
BASELINE_N4 = 198.9

print(f"\nResults from: {RESULTS_DIR}\n")
print(f"{'MTP n':<8} {'N=1 TPS':>10} {'N=1 delta':>10} {'N=4 TPS':>10} {'N=4 delta':>10}")
print("-" * 52)
print(f"{'baseline':<8} {BASELINE_N1:>10.1f} {'—':>10} {BASELINE_N4:>10.1f} {'—':>10}")

for n in [1, 2, 3]:
    fp = os.path.join(RESULTS_DIR, f"tps_mtp_n{n}.json")
    if not os.path.exists(fp):
        print(f"{'n='+str(n):<8} {'NOT_RUN':>10}")
        continue
    d = json.load(open(fp))
    n1 = d["concurrency"]["n1"]["mean"]
    n4 = d["concurrency"]["n4"]["mean"]
    d1 = round((n1 - BASELINE_N1) / BASELINE_N1 * 100, 1)
    d4 = round((n4 - BASELINE_N4) / BASELINE_N4 * 100, 1)
    print(f"{'n='+str(n):<8} {n1:>10.1f} {d1:>+9.1f}% {n4:>10.1f} {d4:>+9.1f}%")

print()
PYEOF
```

---

## Metrics to record

| Metric | Source file | Expected / reference |
|--------|-------------|----------------------|
| MTP deploy success (n=1) | deploy_mtp_n1.log | Server starts, /health 200 within 300s |
| N=1 TPS (MTP n=1) | tps_mtp_n1.json | Reference: AWQ+MTP was +31.8% → ~67 t/s; PQ may differ |
| N=4 TPS (MTP n=1) | tps_mtp_n1.json | Reference: AWQ+MTP was +51% → ~406 t/s aggregate; PQ may differ |
| N=1 TPS (MTP n=2) | tps_mtp_n2.json | Expected higher than n=1 (author: n=3 optimal) |
| N=4 TPS (MTP n=2) | tps_mtp_n2.json | May plateau or drop at high concurrency |
| N=1 TPS (MTP n=3) | tps_mtp_n3.json | Expected highest per-request TPS |
| N=4 TPS (MTP n=3) | tps_mtp_n3.json | May regress vs n=1 at high concurrency |
| GPU1 VRAM per n | vram_mtp_nN.txt | Baseline ~27,732 MiB; MTP head reuses existing weights — no increase expected |
| th02 quality (n=1) | th02_result.txt | PASS (correct EDF answer, no decode corruption) |
| Production restored | deploy_restore.log + /health | /health 200 on 30001 with PrismaQuant model ID |

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| MTP n=1 deploy success | /health 200 within 300s | Record error; write Open from testing; skip to Step 7 |
| th02 quality (n=1) | Same correct EDF answer as BENCH_12 | STOP — MTP corrupts decode; do not test n=2 or n=3 |
| N=1 TPS improvement (n=1) | ≥ +10% vs baseline 51.3 t/s | Note marginal; still record and continue to n=2 |
| N=4 TPS not catastrophically degraded (n=1) | > −30% vs baseline 198.9 t/s | Note; speculative tokens consume KV slots at high concurrency |
| Best n ≥ +20% vs PrismaQuant baseline | N=1 TPS at best n ≥ 61.6 t/s | If no n meets this: MTP marginal on PrismaQuant; note for research |
| Production thinker restored | /health 200 on port 30001, correct model ID | Redeploy manually; do not leave MTP config in production |

---

## Artifacts to write

1. `results/BENCH_19_mtp1_prismaquant_<TIMESTAMP>/deploy_mtp_n1.log`
2. `results/BENCH_19_mtp1_prismaquant_<TIMESTAMP>/deploy_mtp_n2.log`
3. `results/BENCH_19_mtp1_prismaquant_<TIMESTAMP>/deploy_mtp_n3.log`
4. `results/BENCH_19_mtp1_prismaquant_<TIMESTAMP>/tps_mtp_n1.json`
5. `results/BENCH_19_mtp1_prismaquant_<TIMESTAMP>/tps_mtp_n2.json`
6. `results/BENCH_19_mtp1_prismaquant_<TIMESTAMP>/tps_mtp_n3.json`
7. `results/BENCH_19_mtp1_prismaquant_<TIMESTAMP>/vram_mtp_nN.txt` (one per n)
8. `results/BENCH_19_mtp1_prismaquant_<TIMESTAMP>/th02_n1/` — quality output
9. `results/BENCH_19_mtp1_prismaquant_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_19 — T_MTP1 PrismaQuant Thinker MTP Sweep — <TIMESTAMP>

## Environment
- Engine: vLLM TP=1 GPU1, VLLM_USE_V1=0, VLLM_ENGINE_ITERATOR_SOURCE=LEGACY
- Model: rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm
- Config: --hf-overrides {"architectures":["Qwen3_5ForCausalLM"]}, fp8 KV, cp-ON, max-num-seqs 4

## MTP deploy feasibility
MTP n=1 startup: SUCCESS / FAILED (error: <copy first relevant error line if failed>)

## TPS results

| MTP n | N=1 TPS | N=1 delta vs baseline | N=4 TPS | N=4 delta vs baseline |
|-------|---------|-----------------------|---------|-----------------------|
| baseline (no MTP) | 51.3 t/s | — | 198.9 t/s | — |
| n=1 | <X> t/s | <±%> | <X> t/s | <±%> |
| n=2 | <X> t/s | <±%> | <X> t/s | <±%> |
| n=3 | <X> t/s | <±%> | <X> t/s | <±%> |

Best n: <N> (N=1 TPS: <X> t/s, N=4 TPS: <X> t/s)

## VRAM
| MTP n | GPU1 VRAM used |
|-------|----------------|
| n=1 | <X> MiB |
| n=2 | <X> MiB |
| n=3 | <X> MiB |
Baseline (no MTP): 27,732 MiB

## Quality
th02 (n=1): PASS / FAIL
Notes: <any observed differences in output structure or reasoning quality>

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| MTP n=1 deploy | PASS/FAIL | |
| th02 quality | PASS/FAIL | |
| N=1 TPS ≥ +10% at best n | PASS/FAIL | |
| N=4 TPS not catastrophically degraded | PASS/FAIL | |
| Production thinker restored | PASS/FAIL | |

## Verdict
PASS / FAIL / PARTIAL — <one sentence: best n, measured gain, recommendation>

## Incidental findings
<Any observation outside this benchmark's explicit scope. Write one FINDING block per observation.
If nothing: "none">

## Open from testing
<Any unexpected blocker, bug, or question needing research-mode attention. If none: "none">
```

---

## Interpretation boundary

**You may:**
- Record TPS delta at each n, VRAM, th02 outcome, startup success/failure
- Note which n produced the best N=1 TPS and whether N=4 degraded under MTP speculation
- Record any engine warnings or unexpected flags in Incidental findings

**You may NOT:**
- Update `docs/decisions/settled.md`, production config in INDEX.md, or queue files
- Enable MTP permanently in production (that requires research-mode review)
- Conclude that MTP is safe at N=4 if you only tested N=1
- Test n=2 or n=3 if th02 failed at n=1 — quality gate is blocking

---

## Stop condition

**Normal:** TPS table filled for all attempted n values, th02 scored, production thinker restored at port 30001, `summary.md` written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` and stop if:
- MTP n=1 deploy fails (error about speculative decoding incompatible with LEGACY iterator, or OOM, or startup crash). Write: `BENCH_19 BLOCKED: MTP speculative decoding not supported with VLLM_ENGINE_ITERATOR_SOURCE=LEGACY on vLLM 0.19.0. Error: <copy first relevant error line>.`
- th02 fails (wrong EDF answer, empty `<think>` block). Write: `BENCH_19 BLOCKED: MTP n=1 corrupts PrismaQuant decode path. th02 fail. Do not enable MTP on PrismaQuant until resolved.`
- N=1 TPS with n=1 is worse than −20% vs baseline (unexpected regression, not concurrency effect). Write: `BENCH_19 UNEXPECTED: MTP n=1 regresses TPS on PrismaQuant by <X>%. Investigate before enabling.`
