#!/usr/bin/env bash
# benchmarks/queue/BENCH_24_T_APEX1_apex_coder_viability.sh
#
# BENCH_24 — T_APEX1: APEX GGUF Coder Viability
#
# Objective: Evaluate mudler/Qwen3.6-35B-A3B-APEX-GGUF I-Compact on ik_llama.cpp.
#
# References:
# - docs/handoffs/BENCH_24_T_APEX1_APEX_CODER_VIABILITY.md
# - docs/queue/open.md (T_APEX1)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
APEX_FILE="/srv/ai/models/hub/models--mudler--Qwen3.6-35B-A3B-APEX-GGUF/snapshots/42c47e7a396813c593fb12c9307ada5cd8090d4b/Qwen3.6-35B-A3B-APEX-I-Compact.gguf"
PORT=8080

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/BENCH_24_apex1_coder_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[BENCH_24] $(date -u +%H:%M:%S) $*" | tee -a "${LOG}"; }

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " BENCH_24: APEX GGUF CODER VIABILITY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Results Dir: ${RESULTS_DIR}"

# --- Step 0: Cleanup ---
cleanup() {
    log "Cleaning up server..."
    # Kill llama-server if running on PORT
    EXISTING_PID=$(ss -tlnp 2>/dev/null | grep ":${PORT}" | awk '{print $NF}' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1)
    [ -n "${EXISTING_PID}" ] && kill "${EXISTING_PID}" && sleep 2
}
trap cleanup EXIT

# --- Step 1: Prerequisites ---
log "Checking prerequisites..."
[ -f "${IK_BIN}" ] || { log "FATAL: ik_llama.cpp binary not found at ${IK_BIN}"; exit 1; }
[ -f "${APEX_FILE}" ] || { log "FATAL: APEX GGUF model not found at ${APEX_FILE}"; exit 1; }

# Force cleanup of target port before checking
EXISTING_PID=$(ss -tlnp 2>/dev/null | grep ":${PORT}" | awk '{print $NF}' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1 || echo "")
if [ -n "${EXISTING_PID}" ]; then
    log "Port ${PORT} occupied by PID ${EXISTING_PID}. Cleaning up..."
    kill "${EXISTING_PID}" || kill -9 "${EXISTING_PID}"
    sleep 3
fi

ss -tlnp | grep ":${PORT}" >/dev/null 2>&1 && { log "FATAL: Port ${PORT} still occupied after cleanup"; exit 1; } || true
log "Prerequisites OK."

# --- Step 2: Deploy ---
SKIP_DEPLOY=${SKIP_DEPLOY:-0}
if [ "${SKIP_DEPLOY}" = "0" ]; then
    log "Deploying ik_llama.cpp server..."
    CUDA_VISIBLE_DEVICES=0 "${IK_BIN}" \
      -m "${APEX_FILE}" \
      -ngl 999 -t 32 -np 4 -c 32768 \
      -ctk q8_0 -ctv q8_0 \
      --no-mmap \
      --jinja \
      --host 0.0.0.0 --port "${PORT}" \
      >> "${RESULTS_DIR}/server.log" 2>&1 &
    SERVER_PID=$!
    
    # Stream logs to terminal while waiting for health
    tail -f "${RESULTS_DIR}/server.log" | stdbuf -oL sed 's/\r//g; s/^/[llama-server] /' &
    TAIL_PID=$!
    
    for i in $(seq 1 120); do
      curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && log "Server healthy." && break
      sleep 1
    done
    kill "${TAIL_PID}" 2>/dev/null
    
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || { log "FATAL: Server failed to start"; exit 1; }
else
    log "[skip] SKIP_DEPLOY=1 — assuming server already running on ${PORT}"
fi

# Record VRAM
log "Recording VRAM..."
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader > "${RESULTS_DIR}/vram.txt"

# Get Model Name
MODEL_NAME=$(curl -sf "http://localhost:${PORT}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])")

# --- Step 3: TPS Sweep ---
log "Measuring TPS (N=1 and N=4)..."
MEASURE_PY="${RESULTS_DIR}/measure_tps.py"
cat > "${MEASURE_PY}" <<'PYEOF'
import asyncio, httpx, json, sys, time, os

async def _measure_one(client, endpoint, model, prompt, max_tokens):
    t0 = time.monotonic()
    fttt = None
    count = 0
    try:
        resp = await client.post(f"{endpoint}/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        })
        resp.raise_for_status()
        d = resp.json()
        count = d.get("usage", {}).get("completion_tokens", 0)
        # Timings from API if available
        timings = d.get("timings", {})
        fttt = timings.get("prompt_ms", 0) / 1000.0
    except Exception as e:
        print(f"Error: {e}")
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
            "results": results
        }

async def main():
    endpoint, model, n, max_tokens, out_path = sys.argv[1:6]
    res = await measure_agg(endpoint, model, int(n), int(max_tokens))
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"N={n} AGG_TPS={res['agg_tps']}")

async def main_sync():
    await main()

if __name__ == "__main__":
    asyncio.run(main())
PYEOF

for N in 1 4; do
    log "Running TPS sweep N=${N}..."
    python3 "${MEASURE_PY}" "http://localhost:${PORT}/v1" "${MODEL_NAME}" "${N}" 150 "${RESULTS_DIR}/raw/tps_N${N}.json"
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

endpoint, model = sys.argv[1:3]
print(json.dumps(probe(endpoint, model)))
PYEOF

echo "probe,has_think,has_tool_call,tool_name" > "${RESULTS_DIR}/tool_calls.csv"
for i in $(seq 1 5); do
    log "Tool probe ${i}..."
    RESPONSE=$(python3 "${PROBE_PY}" "http://localhost:${PORT}/v1" "${MODEL_NAME}")
    echo "${RESPONSE}" > "${RESULTS_DIR}/raw/tool_probe_${i}.json"
    
    HAS_THINK=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d['choices'][0]['message']; print('YES' if m.get('reasoning_content') else 'NO')")
    HAS_TOOL=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d['choices'][0]['message']; print('YES' if m.get('tool_calls') else 'NO')")
    TOOL_NAME=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d['choices'][0]['message']; tc=m.get('tool_calls', []); print(tc[0]['function']['name'] if tc else 'none')")
    
    echo "${i},${HAS_THINK},${HAS_TOOL},${TOOL_NAME}" >> "${RESULTS_DIR}/tool_calls.csv"
    log "Probe ${i}: think=${HAS_THINK} tool=${HAS_TOOL} (${TOOL_NAME})"
done

# --- Step 5: Quality (th02) ---
log "Running th02 (EDF Scheduling)..."
TH02_PROMPT="Implement a complete Earliest Deadline First (EDF) scheduler in Python. Requirements:
1. Task class with: task_id, arrival_time, execution_time, deadline attributes
2. EDF scheduler that processes a list of tasks, always selecting the task with the earliest deadline
3. Calculate and return: schedule order, average waiting time, missed deadline count, CPU utilization
4. Include a test with tasks: [(0,3,5), (1,2,4), (2,1,3), (3,4,8)] format: (arrival, exec, deadline)
5. Verify correctness: task (2,1,3) should execute first when available due to deadline=3"

python3 -c "import requests, json, sys; resp = requests.post(f'http://localhost:${PORT}/v1/chat/completions', json={'model': '${MODEL_NAME}', 'messages': [{'role': 'user', 'content': sys.stdin.read()}], 'max_tokens': 4096, 'temperature': 0.0}); print(json.dumps(resp.json()))" << EOF > "${RESULTS_DIR}/raw/th02_response.json"
${TH02_PROMPT}
EOF
log "th02 complete."

# --- Step 6: Generate Summary ---
log "Generating summary..."
export RESULTS_DIR="${RESULTS_DIR}"
export TIMESTAMP="${TIMESTAMP}"
python3 - <<'PYEOF'
import json, pathlib, os

results_dir = pathlib.Path(os.environ["RESULTS_DIR"])
raw = results_dir / "raw"

def load_json(p):
    return json.loads(p.read_text()) if p.exists() else {}

tps_n1 = load_json(raw / "tps_N1.json").get("agg_tps", 0)
tps_n4 = load_json(raw / "tps_N4.json").get("agg_tps", 0)

probes = []
pass_count = 0
for i in range(1, 6):
    p = load_json(raw / f"tool_probe_{i}.json")
    msg = p.get("choices", [{}])[0].get("message", {})
    has_tc = "YES" if msg.get("tool_calls") else "NO"
    has_rc = "YES" if msg.get("reasoning_content") else "NO"
    name = msg.get("tool_calls", [{}])[0].get("function", {}).get("name", "none") if has_tc == "YES" else "none"
    probes.append((i, has_rc, has_tc, name))
    if has_tc == "YES": pass_count += 1

th02 = load_json(raw / "th02_response.json")
msg = th02.get("choices", [{}])[0].get("message", {})
th02_content = msg.get("content", "") or ""
th02_reasoning = msg.get("reasoning_content", "") or ""

summary = f"""# BENCH_24 — APEX GGUF Coder Viability — {os.environ.get('TIMESTAMP')}

## Environment
- Model: Qwen3.6-35B-A3B-APEX-I-Compact.gguf
- Engine: ik_llama.cpp (server)
- Config: -ngl 999, q8_0 KV, 32K context

## TPS
| N (concurrent) | APEX I-Compact (t/s) | PQ 4.75bit baseline (t/s) |
|----------------|----------------------|---------------------------|
| 1              | {tps_n1}                 | 56.5                      |
| 4              | {tps_n4}                 | 459.0                     |

## Tool-call reliability
| Probe | Reasoning? | Tool Call? | Tool Name |
|-------|------------|------------|-----------|
"""
for p in probes:
    summary += f"| {p[0]} | {p[1]} | {p[2]} | {p[3]} |\n"

summary += f"\n**Pass rate: {pass_count}/5**\n\n"
summary += "## Quality — th02 EDF scheduling\n"
summary += f"### Reasoning Preview\n```\n{th02_reasoning[:1000]}...\n```\n"
summary += f"### Content Preview\n```python\n{th02_content[:2000]}\n```\n"

summary += """
## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
"""
summary += f"| TPS N=1 >= 56.5 | {'PASS' if tps_n1 >= 56.5 else 'FAIL'} | |\n"
summary += f"| Tool-call rate >= 4/5 | {'PASS' if pass_count >= 4 else 'FAIL'} | Actual: {pass_count}/5 |\n"

summary += f"\n## Verdict\n{'PASS' if pass_count >= 4 and tps_n1 >= 56.5 else 'FAIL'}\n"

(results_dir / "summary.md").write_text(summary)
PYEOF

log "BENCH_24 COMPLETE. Results in ${RESULTS_DIR}"
