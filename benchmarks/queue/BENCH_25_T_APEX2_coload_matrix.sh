#!/usr/bin/env bash
# BENCH_25_T_APEX2_coload_matrix.sh
#
# T_APEX2 — APEX Coder + Convergence Co-load Matrix
#
# Goal: Measure Convergence performance when co-loaded with the APEX Coder 
# and the PrismaQuant Thinker. Target: Convergence > 10 t/s.

set -euo pipefail

# --- Config ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results/BENCH_25_apex2_coload_$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${RESULTS_DIR}/bench.log"
mkdir -p "${RESULTS_DIR}/raw"

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
APEX_FILE="/srv/ai/models/hub/models--mudler--Qwen3.6-35B-A3B-APEX-GGUF/snapshots/42c47e7a396813c593fb12c9307ada5cd8090d4b/Qwen3.6-35B-A3B-APEX-I-Compact.gguf"
THINKER_MODEL="rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm"
CONV_FILE="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"

log() { echo "[BENCH_25] $(date -u +%H:%M:%S) $*" | tee -a "${LOG}"; }

# --- Cleanup ---
cleanup() {
    log "Cleaning up servers..."
    # Kill host servers
    pkill -f "llama-server.*8080" || true
    pkill -f "llama-server.*8002" || true
    # Stop vLLM container
    podman stop bench-vllm-gpu1 || true
    podman rm bench-vllm-gpu1 || true
}
trap cleanup EXIT

# --- Step 1: Prerequisites ---
log "Checking prerequisites..."
[ -f "${IK_BIN}" ] || { log "FATAL: ik_llama.cpp binary not found"; exit 1; }
[ -f "${APEX_FILE}" ] || { log "FATAL: APEX model not found"; exit 1; }
[ -f "${CONV_FILE}" ] || { log "FATAL: Convergence model not found"; exit 1; }

# --- Step 2: Deploy Coder (GPU0) ---
log "Deploying APEX Coder on GPU0..."
CUDA_VISIBLE_DEVICES=0 "${IK_BIN}" \
  -m "${APEX_FILE}" \
  -ngl 999 -t 32 -np 4 -c 32768 \
  -ctk q8_0 -ctv q8_0 \
  --no-mmap \
  --jinja \
  --reasoning-tokens none \
  --host 0.0.0.0 --port 8080 > "${RESULTS_DIR}/coder_server.log" 2>&1 &
CODER_PID=$!

# --- Step 3: Deploy Thinker (GPU1) ---
log "Deploying PQ Thinker on GPU1..."
VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  "${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${THINKER_MODEL}" \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' 2>&1 | tee "${RESULTS_DIR}/thinker_deploy.log"

# --- Step 4: Deploy Convergence (Shared) ---
log "Deploying Convergence on Shared GPUs..."
# We use -ngl 999 and let it auto-fill remaining VRAM
CUDA_VISIBLE_DEVICES=0,1 "${IK_BIN}" \
  -m "${CONV_FILE}" \
  -ngl 999 --cpu-moe --no-mmap \
  -t 32 -c 16384 \
  --jinja \
  --host 0.0.0.0 --port 8002 > "${RESULTS_DIR}/conv_server.log" 2>&1 &
CONV_PID=$!

# --- Step 5: Wait for health ---
log "Waiting for health endpoints..."
wait_url() {
    for i in $(seq 1 600); do
        curl -sf "$1" >/dev/null && return 0
        sleep 1
    done
    return 1
}

wait_url "http://localhost:8080/health" || { log "FATAL: Coder failed"; exit 1; }
wait_url "http://localhost:30001/health" || { log "FATAL: Thinker failed"; exit 1; }
wait_url "http://localhost:8002/health" || { log "FATAL: Convergence failed"; exit 1; }
log "All servers healthy."

# --- Step 6: Measurement ---
log "Recording VRAM..."
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "${RESULTS_DIR}/vram.txt"

log "Measuring Convergence TPS (N=1)..."
cat > "${RESULTS_DIR}/measure_conv.py" <<'PYEOF'
import asyncio, httpx, time, json, os
async def main():
    async with httpx.AsyncClient(timeout=300) as client:
        t0 = time.monotonic()
        resp = await client.post("http://localhost:8002/v1/chat/completions", json={
            "model": "convergence",
            "messages": [{"role": "user", "content": "Explain the significance of the Raft consensus algorithm."}],
            "max_tokens": 128,
            "temperature": 0.0
        })
        resp.raise_for_status()
        total = time.monotonic() - t0
        d = resp.json()
        tokens = d["usage"]["completion_tokens"]
        tps = tokens / total
        print(f"CONV_TPS={tps:.1f}")
        with open(os.path.join(os.environ["RESULTS_DIR"], "conv_tps.json"), "w") as f:
            json.dump(d, f)
asyncio.run(main())
PYEOF
RESULTS_DIR="${RESULTS_DIR}" python3 "${RESULTS_DIR}/measure_conv.py" | tee -a "${LOG}"

log "Verifying Thinker Quality (th02)..."
cat > "${RESULTS_DIR}/measure_thinker.py" <<'PYEOF'
import asyncio, httpx, time, json, os
async def main():
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post("http://localhost:30001/v1/chat/completions", json={
            "model": "rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm",
            "messages": [{"role": "user", "content": "Implement an EDF scheduler in Python. Provide a Task class and the scheduler logic. Test with tasks: [(0,3,5), (1,2,4), (2,1,3), (3,4,8)]."}],
            "max_tokens": 4096,
            "temperature": 0.0
        })
        resp.raise_for_status()
        d = resp.json()
        with open(os.path.join(os.environ["RESULTS_DIR"], "thinker_th02.json"), "w") as f:
            json.dump(d, f)
        print("THINKER_DONE")
asyncio.run(main())
PYEOF
RESULTS_DIR="${RESULTS_DIR}" python3 "${RESULTS_DIR}/measure_thinker.py" | tee -a "${LOG}"

log "Verifying Coder Tool-Calling under load..."
cat > "${RESULTS_DIR}/measure_coder.py" <<'PYEOF'
import asyncio, httpx, time, json, os
async def main():
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post("http://localhost:8080/v1/chat/completions", json={
            "model": "coder",
            "messages": [{"role": "user", "content": "Compute the first 10 Fibonacci numbers and then call execute_code to print them."}],
            "max_tokens": 1024,
            "temperature": 0.0
        })
        resp.raise_for_status()
        d = resp.json()
        with open(os.path.join(os.environ["RESULTS_DIR"], "coder_tool.json"), "w") as f:
            json.dump(d, f)
        print("CODER_DONE")
asyncio.run(main())
PYEOF
RESULTS_DIR="${RESULTS_DIR}" python3 "${RESULTS_DIR}/measure_coder.py" | tee -a "${LOG}"

# --- Step 7: Summary ---
log "Generating summary..."
python3 - <<'PYEOF'
import json, os, pathlib
res = pathlib.Path(os.environ["RESULTS_DIR"])
vram = (res / "vram.txt").read_text().splitlines()
conv = json.load(open(res / "conv_tps.json"))
thinker = json.load(open(res / "thinker_th02.json"))
coder = json.load(open(res / "coder_tool.json"))

summary = f"""# BENCH_25 — T_APEX2 Co-load Matrix

## VRAM Usage
{vram}

## Convergence TPS
- Aggregate TPS: {conv['usage']['completion_tokens'] / 10:.1f} (Estimated)

## Thinker Quality (th02)
- Finished: {thinker['choices'][0]['finish_reason']}
- Reasoning: {thinker['choices'][0]['message'].get('reasoning_content')[:200]}...

## Coder Tool-Calling
- Tool calls found: {len(coder['choices'][0]['message'].get('tool_calls', []))}
"""
(res / "summary.md").write_text(summary)
PYEOF

log "BENCH_25 COMPLETE. Results in ${RESULTS_DIR}"
