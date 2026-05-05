#!/usr/bin/env bash
# benchmarks/queue/23c_coder_mtp_tuned.sh
#
# BENCH_23c — Coder TP=1 V1 Engine + MTP Tuned
#
# Objective: Attempt to resolve the 34 t/s MTP bottleneck by increasing 
# batched token limits as suggested by the vLLM logs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# Configuration
MODEL="rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm"
PORT="${PORT_VLLM_GPU0}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/BENCH_23c_coder_mtp_tuned_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[BENCH_23c] $(date -u +%H:%M:%S) $*" | tee -a "${LOG}"; }

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " BENCH_23c: CODER TP=1 V1 ENGINE + MTP TUNED"
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
log "Deploying Coder with MTP n=1 and increased batched tokens..."
# Tuning: max-num-batched-tokens 8192 to resolve scheduler cap.
# max-num-seqs 64 for more scheduling flexibility.
VLLM_USE_V1=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu0 "${MODEL}" \
    --trust-remote-code \
    --gpu-mem-util 0.90 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    2>&1 | tee "${RESULTS_DIR}/deploy.log"

log "Endpoint ready (deploy.sh confirmed healthy)."

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

# --- Step 4: Summary ---
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

summary = f"""# BENCH_23c — Coder TP=1 V1 Engine + MTP Tuned — {timestamp}

## Environment
- Model: {model}
- Config: TP=1 GPU0, V1 Engine, MTP n=1, fp8 KV, ctx 32768, gpu-mem-util=0.90, max-num-seqs=64, max-num-batched-tokens=8192

## TPS Results
| N | No MTP (BENCH_23) | MTP n=1 (BENCH_23b) | MTP Tuned (BENCH_23c) |
|---|-------------------|---------------------|-----------------------|
| 1 | 56.5 t/s          | 34.7 t/s            | {tps_n1} t/s |
| 4 | N/A               | 192.0 t/s           | {tps_n4} t/s |

## Verdict
If TPS is still < 50, then the bottleneck is kernel-bound (Expert routing overhead) rather than scheduler-bound.
"""

(results_dir / "summary.md").write_text(summary)
PYEOF

# --- Cleanup ---
log "Cleaning up..."
podman stop bench-vllm-gpu0 2>/dev/null || true

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log " BENCH_23c COMPLETE"
log " Results: ${RESULTS_DIR}"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
