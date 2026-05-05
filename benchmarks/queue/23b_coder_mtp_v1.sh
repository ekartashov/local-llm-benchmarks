#!/usr/bin/env bash
# benchmarks/queue/23b_coder_mtp_v1.sh
#
# BENCH_23b — Coder TP=1 V1 Engine + MTP Validation
#
# Objective: Verify if MTP n=1 is compatible with the V1 engine for the 
# 35B-A3B MoE coder, and if it resolves the 60 t/s bottleneck at TP=1.
#
# References:
# - benchmarks/queue/23_pq2_phase1_coder.sh
# - results/T_MTP1_prismaquant_thinker_mtp_20260503T232623Z/summary.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# Configuration
MODEL="rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm"
PORT="${PORT_VLLM_GPU0}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/BENCH_23b_coder_mtp_v1_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[BENCH_23b] $(date -u +%H:%M:%S) $*" | tee -a "${LOG}"; }

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " BENCH_23b: CODER TP=1 V1 ENGINE + MTP N=1"
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
}
trap cleanup EXIT

# --- Step 1: Prerequisites ---
log "Checking prerequisites..."
for c in arclight-coder arclight-thinker bench-vllm-gpu0 bench-vllm-gpu1; do
    if podman container exists "${c}" 2>/dev/null; then
        log "Stopping conflicting container ${c}..."
        podman stop "${c}" >/dev/null 2>&1 || true
        podman rm "${c}" >/dev/null 2>&1 || true
    fi
done

# --- Step 2: Deploy ---
log "Deploying Coder with MTP n=1 on V1 engine..."
# We use VLLM_USE_V1=1 and add speculative-config.
# We keep max-num-seqs 16 and gpu-mem-util 0.90 for stable graph capture.
VLLM_USE_V1=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu0 "${MODEL}" \
    --trust-remote-code \
    --gpu-mem-util 0.90 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 16 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    2>&1 | tee "${RESULTS_DIR}/deploy.log"

log "Endpoint ready (deploy.sh confirmed healthy)."

# Record kernel selection and MTP status
log "Recording logs..."
grep -i "marlin\|mtp\|v1\|engine\|graph\|quantization\|kernel\|speculative" "${RESULTS_DIR}/deploy.log" \
    | head -50 > "${RESULTS_DIR}/deploy_notes.txt"

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
        "max_tokens": 1024,
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

# --- Step 5: Generate Summary ---
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

summary = f"""# BENCH_23b — Coder TP=1 V1 Engine + MTP N=1 — {timestamp}

## Environment
- Model: {model}
- Config: TP=1 GPU0, V1 Engine, MTP n=1, fp8 KV, ctx 32768, gpu-mem-util=0.90, max-num-seqs=16

## TPS Comparison (vs No-MTP Baseline)
| N | No MTP (BENCH_23) | MTP n=1 | Delta |
|---|-------------------|---------|-------|
| 1 | 56.5 t/s          | {tps_n1} | {round((tps_n1-56.5)/56.5*100, 1) if tps_n1 else 0}% |
| 4 | N/A               | {tps_n4} | N/A |

## Tool-call reliability
| Probe | has_tool_call | tool_name |
|-------|--------------|-----------|
"""
for p in probes:
    summary += f"| {p[0]} | {p[1]} | {p[2]} |\n"

summary += f"| **Pass rate** | **{pass_count}/5** | |\n\n"

summary += """## Verdict
Review logs for 'speculative' activation. If TPS > 80 and tool calls pass, MTP is viable for MoE at TP=1.
"""

(results_dir / "summary.md").write_text(summary)
PYEOF

# --- Cleanup ---
log "Cleaning up..."
podman stop bench-vllm-gpu0 2>/dev/null || true

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log " BENCH_23b COMPLETE"
log " Results: ${RESULTS_DIR}"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
