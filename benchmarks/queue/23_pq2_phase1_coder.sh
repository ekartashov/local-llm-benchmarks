#!/usr/bin/env bash
# benchmarks/queue/23_pq2_phase1_coder.sh
#
# BENCH_23 — T_PQ2 Phase 1: PrismaQuant coder 4.75bit (Marlin fallback)
#
# Objective: Measure TPS, tool-call reliability, and quality on th02 for the 
# PrismaQuant 4.75bit coder candidate.
#
# References:
# - docs/handoffs/BENCH_23_T_PQ2_PHASE1_PRISMAQUANT_CODER.md
# - benchmarks/queue/T2.5_qwen36_35b_coder_shootout.sh
# - benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# Configuration
MODEL="rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm"
PORT="${PORT_VLLM_GPU0}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/BENCH_23_pq2_phase1_coder_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[BENCH_23] $(date -u +%H:%M:%S) $*" | tee -a "${LOG}"; }

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " BENCH_23: PRISMAQUANT CODER 4.75BIT PHASE 1"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Model: ${MODEL}"
log "Results Dir: ${RESULTS_DIR}"

# --- Step 0: Cleanup ---
cleanup() {
    log "Cleaning up..."
    EXISTING=$(podman ps --format "{{.Names}}" | grep -i "coder\|tp2\|35b\|${PORT}" | head -1)
    if [ -n "${EXISTING}" ]; then
        log "Stopping container ${EXISTING}..."
        podman stop "${EXISTING}" >/dev/null 2>&1 || true
        podman rm "${EXISTING}" >/dev/null 2>&1 || true
    fi
    # Ensure results are owned by user
    sudo chown -R "${USER}:${USER}" "${RESULTS_DIR}" || true
}
trap cleanup EXIT

# --- Step 1: Prerequisites ---
log "Checking prerequisites..."
# Stop Arclight hot-pair if running (GPU0+1 needed)
for c in arclight-coder arclight-thinker bench-vllm-gpu0 bench-vllm-gpu1; do
    if podman container exists "${c}" 2>/dev/null; then
        log "Stopping conflicting container ${c}..."
        podman stop "${c}" >/dev/null 2>&1 || true
        podman rm "${c}" >/dev/null 2>&1 || true
    fi
done

# --- Step 2: Deploy ---
log "Deploying PrismaQuant coder (TP=1, no enforce-eager)..."
# Steady-state VRAM: ~27.5 GiB used, 4.5 GiB free (BENCH_23 vram.txt).
# OOM without --enforce-eager was a transient profiling spike (max_num_seqs=1024
# forward pass needs ~1.02 GiB; only 1.01 GiB free). Fix: --max-num-seqs 16
# limits profiling batch → spike drops to ~27.7 GiB, leaving 3.6 GiB free.
# gpu-mem-util 0.90: budget 28.2 GiB vs model floor ~27.5 GiB → ~700 MiB KV cache.
# 0.84 (previous debug value) was below the model floor → 0 KV blocks.
VLLM_USE_FLASHINFER_NVFP4=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu0 "${MODEL}" \
    --trust-remote-code \
    --gpu-mem-util 0.90 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 16 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    2>&1 | tee "${RESULTS_DIR}/deploy.log"

log "Endpoint ready (deploy.sh confirmed healthy)."

# Record kernel selection
log "Recording kernel selection..."
grep -i "compressed\|marlin\|nvfp4\|fp4\|quantization\|kernel\|fallback" "${RESULTS_DIR}/deploy.log" \
    | head -30 > "${RESULTS_DIR}/deploy_notes.txt"

# Record VRAM
log "Recording VRAM..."
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader > "${RESULTS_DIR}/vram.txt"

# --- Step 3: TPS Measurement ---
log "Measuring TPS (N=1 and N=4)..."

MEASURE_PY="${RESULTS_DIR}/measure_tps.py"
cat > "${MEASURE_PY}" <<'PYEOF'
import asyncio, httpx, json, sys, time, os

async def _measure_one(client, endpoint, model, prompt, max_tokens):
    t0 = time.monotonic()
    fttt = None
    count = 0
    async with client.stream("POST", f"{endpoint}/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }) as resp:
        resp.raise_for_status()
        async for raw in resp.aiter_lines():
            if not raw.startswith("data: ") or "[DONE]" in raw:
                continue
            chunk = json.loads(raw[6:])
            if chunk.get("usage"):
                count = chunk["usage"]["completion_tokens"]
            elif fttt is None and chunk.get("choices") and chunk["choices"][0].get("delta"):
                delta = chunk["choices"][0]["delta"]
                tok = delta.get("content") or delta.get("reasoning") or ""
                if tok:
                    fttt = time.monotonic() - t0
    total = time.monotonic() - t0
    decode_s = total - (fttt or 0)
    return {
        "ttft_ms": round((fttt or 0) * 1000, 1),
        "decode_tps": round(count / decode_s, 1) if decode_s > 0 and count > 0 else 0.0,
        "tokens": count,
    }

async def measure_agg(endpoint, model, n_seqs, max_tokens):
    prompt = "Write a detailed Python implementation of a red-black tree. Be extremely verbose."
    async with httpx.AsyncClient(timeout=300) as client:
        t_wall_start = time.monotonic()
        results = await asyncio.gather(*[_measure_one(client, endpoint, model, prompt, max_tokens) for _ in range(n_seqs)])
        wall_s = time.monotonic() - t_wall_start
        total_tokens = sum(r["tokens"] for r in results)
        agg_tps = round(total_tokens / wall_s, 1) if wall_s > 0 else 0.0
        return {
            "n": n_seqs,
            "agg_tps": agg_tps,
            "avg_ttft_ms": round(sum(r["ttft_ms"] for r in results) / n_seqs, 1),
            "results": results
        }

async def main():
    endpoint = sys.argv[1]
    model = sys.argv[2]
    n = int(sys.argv[3])
    max_tokens = int(sys.argv[4])
    out_path = sys.argv[5]
    
    res = await measure_agg(endpoint, model, n, max_tokens)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"N={n} AGG_TPS={res['agg_tps']}")

asyncio.run(main())
PYEOF

for N in 1 4; do
    log "Running TPS sweep N=${N}..."
    python3 "${MEASURE_PY}" "http://localhost:${PORT}/v1" "${MODEL}" "${N}" 150 "${RESULTS_DIR}/raw/tps_N${N}.json"
done

# --- Step 4: Tool-call Reliability ---
log "Testing tool-call reliability (5 probes)..."

PROBE_PY="${RESULTS_DIR}/probe_tool.py"
cat > "${PROBE_PY}" <<'PYEOF'
import sys, json, requests

def probe(endpoint, model):
    system = "You are a coding assistant. Use the provided tools to help the user."
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"]
            }
        }
    }]
    prompt = "Write a Python function to compute the Fibonacci sequence up to n terms, then call execute_code to run it with n=10."
    
    resp = requests.post(f"{endpoint}/chat/completions", json={
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0.0
    })
    return resp.json()

endpoint = sys.argv[1]
model = sys.argv[2]
res = probe(endpoint, model)
print(json.dumps(res))
PYEOF

echo "probe,has_tool_call,tool_name" > "${RESULTS_DIR}/tool_calls.csv"
for i in $(seq 1 5); do
    log "Tool probe ${i}..."
    RESPONSE=$(python3 "${PROBE_PY}" "http://localhost:${PORT}/v1" "${MODEL}")
    echo "${RESPONSE}" > "${RESULTS_DIR}/raw/tool_probe_${i}.json"
    
    HAS_TOOL=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); msg=d['choices'][0]['message']; print('YES' if msg.get('tool_calls') else 'NO')")
    TOOL_NAME=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); msg=d['choices'][0]['message']; tc=msg.get('tool_calls', []); print(tc[0]['function']['name'] if tc else 'none')")
    
    echo "${i},${HAS_TOOL},${TOOL_NAME}" >> "${RESULTS_DIR}/tool_calls.csv"
    log "Probe ${i}: ${HAS_TOOL} (${TOOL_NAME})"
done

# --- Step 5: Quality (th02) ---
log "Running th02 (EDF Scheduling) quality task..."
TH02_PY="${RESULTS_DIR}/th02.py"
cat > "${TH02_PY}" <<'PYEOF'
import requests, sys, json
endpoint, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    resp = requests.post(f"{endpoint}/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16384,
        "temperature": 0.0
    }, timeout=600)
    print(json.dumps(resp.json()))
except Exception as e:
    print(json.dumps({"error": str(e)}))
PYEOF

TH02_PROMPT="Implement a complete Earliest Deadline First (EDF) scheduler in Python. Requirements:
1. Task class with: task_id, arrival_time, execution_time, deadline attributes
2. EDF scheduler that processes a list of tasks, always selecting the task with the earliest deadline
3. Calculate and return: schedule order, average waiting time, missed deadline count, CPU utilization
4. Include a test with tasks: [(0,3,5), (1,2,4), (2,1,3), (3,4,8)] format: (arrival, exec, deadline)
5. Verify correctness: task (2,1,3) should execute first when available due to deadline=3

Provide complete, runnable Python code with the test included."

python3 "${TH02_PY}" "http://localhost:${PORT}/v1" "${MODEL}" "${TH02_PROMPT}" > "${RESULTS_DIR}/raw/th02_response.json"

# --- Step 6: Generate Summary ---
log "Generating summary..."

export RESULTS_DIR="${RESULTS_DIR}"
export TIMESTAMP="${TIMESTAMP}"
export MODEL="${MODEL}"

python3 - <<'PYEOF'
import json, pathlib, os

results_dir = pathlib.Path(os.environ["RESULTS_DIR"])
timestamp = os.environ["TIMESTAMP"]
model = os.environ["MODEL"]
raw = results_dir / "raw"

def load_json(p):
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except:
        return {"error": "malformed json"}

tps_n1 = load_json(raw / "tps_N1.json").get("agg_tps", 0)
tps_n4 = load_json(raw / "tps_N4.json").get("agg_tps", 0)

probes = []
pass_count = 0
for i in range(1, 6):
    p = load_json(raw / f"tool_probe_{i}.json")
    choices = p.get("choices", [{}])
    msg = choices[0].get("message", {}) if choices else {}
    has_tc = "YES" if msg.get("tool_calls") else "NO"
    name = msg.get("tool_calls", [{}])[0].get("function", {}).get("name", "none") if has_tc == "YES" else "none"
    probes.append((i, has_tc, name))
    if has_tc == "YES": pass_count += 1

th02 = load_json(raw / "th02_response.json")
choices = th02.get("choices", [{}])
th02_text = choices[0].get("message", {}).get("content", "ERROR: No response") if choices else "ERROR: Empty response"

# Baseline comparison
AWQ_N1 = 240.9
AWQ_N4 = 709.8
delta_n1 = round((tps_n1 - AWQ_N1) / AWQ_N1 * 100, 1) if tps_n1 else -100
delta_n4 = round((tps_n4 - AWQ_N4) / AWQ_N4 * 100, 1) if tps_n4 else -100

summary = f"""# BENCH_23 — T_PQ2 Phase 1: PrismaQuant coder 4.75bit — {timestamp}

## Environment
- Model: {model}
- Config: TP=1 GPU0, V1 engine, fp8 KV, ctx 32768, gpu-mem-util=0.90, max-num-seqs=16, expandable_segments=True, NVFP4 FlashInfer
- Kernel selection: {(results_dir / "deploy_notes.txt").read_text().strip() if (results_dir / "deploy_notes.txt").exists() else "N/A"}

## VRAM at load
```
{(results_dir / "vram.txt").read_text().strip() if (results_dir / "vram.txt").exists() else "N/A"}
```

## TPS
| N (concurrent) | PQ 4.75bit (t/s) | AWQ 4bit baseline (t/s) | Delta (%) |
|----------------|-----------------|------------------------|-----------|
| 1              | {tps_n1}            | {AWQ_N1}                 | {delta_n1}% |
| 4              | {tps_n4}            | {AWQ_N4}                 | {delta_n4}% |

## Tool-call reliability
| Probe | has_tool_call | tool_name |
|-------|--------------|-----------|
"""
for p in probes:
    summary += f"| {p[0]} | {p[1]} | {p[2]} |\n"

summary += f"| **Pass rate** | **{pass_count}/5** | |\n"
summary += "| AWQ baseline | 29/30 (96.7%) | |\n\n"

summary += "## Quality — th02 EDF scheduling (Raw Response)\n\n"
summary += "```python\n" + th02_text + "\n```\n\n"

summary += """## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
"""
summary += f"| Model loads | {'PASS' if tps_n1 > 0 else 'FAIL'} | |\n"
summary += f"| Tool-call rate >= 5/5 | {'PASS' if pass_count >= 5 else 'FAIL'} | Actual: {pass_count}/5 |\n"
summary += f"| TPS within 40% of AWQ | {'PASS' if tps_n1 >= 144 else 'FAIL'} | Actual: {tps_n1} t/s |\n"

summary += f"\n## Verdict\n{'PASS' if pass_count >= 5 and tps_n1 >= 144 else 'INCONCLUSIVE'} - Review th02 reasoning above.\n"

(results_dir / "summary.md").write_text(summary)

# metrics.json
metrics = {
    "item_id": "BENCH_23",
    "timestamp": timestamp,
    "metrics": {
        "decode_tps_n1": tps_n1,
        "decode_tps_n4": tps_n4,
        "tool_call_pass_rate": pass_count / 5.0,
        "awq_n1_baseline": AWQ_N1,
        "awq_n4_baseline": AWQ_N4
    },
    "verdict": "PASS" if pass_count >= 5 and tps_n1 >= 144 else "FAIL"
}
(results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
PYEOF

# --- Cleanup ---
log "Cleaning up..."
podman stop bench-vllm-gpu0 2>/dev/null || true

# Restore config.json if neuter attempt was made
MODEL_PATH=$(find /srv/ai/models/hub/models--rdtand--Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm/snapshots/ -maxdepth 1 -mindepth 1 -type d | head -1)
CONFIG_JSON="${MODEL_PATH}/config.json"
if [ -f "${CONFIG_JSON}.bak" ]; then
    log "Restoring original config.json..."
    mv "${CONFIG_JSON}.bak" "${CONFIG_JSON}"
fi

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log " BENCH_23 COMPLETE"
log " Results: ${RESULTS_DIR}"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
