#!/usr/bin/env bash
# T_MTP1_PRISMAQUANT_THINKER_automation.sh
# MTP Speculative Decoding on PrismaQuant Thinker (n=1,2,3 Sweep)
#
# Logic: Sweep num_speculative_tokens to recover TPS regression from PrismaQuant.
# Constraint: VLLM_ENGINE_ITERATOR_SOURCE=LEGACY is mandatory.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ITEM_ID="T_MTP1_prismaquant_thinker_mtp"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

LOG="${RESULTS_DIR}/bench.log"
log() { echo "[T_MTP1 $(date -u +%H:%M:%S)] $*" | tee -a "${LOG}"; }

# --- Arguments ---
SKIP_DOWNLOAD=0
SKIP_INIT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --skip-init)     SKIP_INIT=1;     shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

log "Starting BENCH_19 (Thinker MTP Sweep)..."
log "Results: ${RESULTS_DIR}"

BASELINE_N1=51.3
BASELINE_N4=198.9
MODEL="rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm"

# --- Step 1: Model Presence ---
if [ "${SKIP_DOWNLOAD}" -eq 0 ]; then
    if ls /srv/ai/models/hub/ | grep -qi "prismaquant"; then
        log "PrismaQuant model already present."
    else
        log "Downloading ${MODEL}..."
        pyenv activate hf && \
          HF_HOME=/srv/ai/models hf download "${MODEL}" \
          >> "${RESULTS_DIR}/download.log" 2>&1
        log "Download complete."
    fi
fi

# --- Step 2-6: Sweep Loop ---
sweep_mtp() {
    local n=$1
    log "--------------------------------------------------"
    log "Testing MTP n=${n} ..."
    log "--------------------------------------------------"

    # 1. Deploy
    EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
    if [ -n "${EXISTING}" ]; then
        log "Stopping existing container: ${EXISTING}"
        podman stop "${EXISTING}" && podman rm "${EXISTING}"
        sleep 3
    fi

    log "Deploying PrismaQuant + MTP n=${n}..."
    # We call deploy.sh synchronously to leverage its built-in log streaming and wait-healthy logic.
    # This matches the standard pattern used in T_MTP1 (AWQ) and T_MTP2 (Coder).
    VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    "${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
      --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
      --enable-chunked-prefill --max-num-seqs 4 \
      --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
      --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
      --speculative-config '{"method":"mtp","num_speculative_tokens":'${n}'}' \
      2>&1 | tee "${RESULTS_DIR}/deploy_mtp_n${n}.log"
    
    # deploy.sh returns 0 only if health check passes.
    if [ $? -ne 0 ]; then
        log "FATAL: MTP n=${n} failed to start or timed out."
        return 1
    fi
    log "MTP n=${n} READY."

    # 2. Record VRAM
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep "^1," | tee "${RESULTS_DIR}/vram_mtp_n${n}.txt"

    # 3. Measure TPS
    log "Measuring TPS (usage.completion_tokens)..."
    MTP_N=${n} REPS=5 RESULTS_DIR="${RESULTS_DIR}" MODEL_ID="${MODEL}" python3 /tmp/bench_mtp_tps.py \
      | tee "${RESULTS_DIR}/tps_mtp_n${n}_stdout.txt"

    # 4. Quality Check (only for n=1 as a gate)
    if [ "${n}" -eq 1 ]; then
        log "Running th02 quality gate..."
        python3 -m benchmarks.phase2_model_selection.bench \
          --endpoint http://localhost:30001/v1 \
          --results-dir "${RESULTS_DIR}/th02_n1" \
          --mode quality \
          --tasks benchmarks/phase2_model_selection/tasks/thinker/ \
          --task-filter th02 \
          --model "${MODEL}" \
          --label "PQ-MTP-n1" \
          --max-tokens 16384 \
          >> "${RESULTS_DIR}/th02_n1_stdout.txt" 2>&1
        
        log "th02 output written to ${RESULTS_DIR}/th02_n1/"
        echo "MANUAL SCORE REQUIRED: Read ${RESULTS_DIR}/th02_n1/ output."
        read -p "Does th02 PASS? [y/N]: " RESP
        if [[ ! "${RESP}" =~ ^[Yy]$ ]]; then
            log "th02 FAILED. Stopping sweep."
            echo "FAIL" > "${RESULTS_DIR}/th02_result.txt"
            return 2
        fi
        echo "PASS" > "${RESULTS_DIR}/th02_result.txt"
    fi
    return 0
}

# Write measurement script
cat > /tmp/bench_mtp_tps.py << 'PYEOF'
import os, sys, json, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

ENDPOINT = "http://localhost:30001/v1/chat/completions"
MODEL = os.environ.get("MODEL_ID", "")
RESULTS_DIR = os.environ.get("RESULTS_DIR", ".")
MTP_N = int(os.environ.get("MTP_N", "1"))
REPS = int(os.environ.get("REPS", "5"))
PROMPT = "Explain the difference between TCP and UDP in exactly three sentences. Be precise and technical."

def single_request():
    payload = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 256, "stream": False}).encode()
    req = urllib.request.Request(ENDPOINT, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        elapsed = time.perf_counter() - t0
        tokens = body["usage"]["completion_tokens"]
        return tokens / elapsed, tokens, elapsed
    except Exception as e:
        print(f"Request failed: {e}")
        return 0, 0, 0

def measure(n):
    samples = []
    for _ in range(REPS):
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(single_request) for _ in range(n)]
            results = [f.result() for f in as_completed(futures)]
        t_total = time.perf_counter() - t_start
        total_tokens = sum(r[1] for r in results)
        if total_tokens == 0:
            agg_tps = 0
        else:
            agg_tps = total_tokens / t_total
        samples.append(agg_tps)
    return {"mean": round(statistics.mean(samples), 1), "stdev": round(statistics.stdev(samples) if len(samples)>1 else 0, 1)}

res = {"n1": measure(1), "n4": measure(4)}
with open(os.path.join(RESULTS_DIR, f"tps_mtp_n{MTP_N}.json"), "w") as f:
    json.dump({"mtp_n": MTP_N, "concurrency": res, "model": MODEL}, f, indent=2)
PYEOF

# Execute Sweep
for n in 1 2 3; do
    sweep_mtp "${n}" || {
        RET=$?
        if [ $RET -eq 1 ]; then log "Skipping remaining sweep due to deploy failure."; break; fi
        if [ $RET -eq 2 ]; then log "Aborting due to quality failure."; break; fi
    }
done

# --- Step 7: Restore Production ---
log "Restoring production thinker..."
EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${EXISTING}" ] && podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 3
VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' >/dev/null 2>&1 &

# --- Step 8: Reporting ---
log "Generating summary..."
export RESULTS_DIR BASELINE_N1 BASELINE_N4
python3 - <<'PYEOF'
import json, os, glob
results_dir = os.environ["RESULTS_DIR"]
b1 = float(os.environ["BASELINE_N1"])
b4 = float(os.environ["BASELINE_N4"])
print(f"\n| MTP n | N=1 TPS | Delta | N=4 TPS | Delta |")
print(f"|-------|---------|-------|---------|-------|")
print(f"| base  | {b1:>7.1f} |   —   | {b4:>7.1f} |   —   |")
for n in [1, 2, 3]:
    fp = os.path.join(results_dir, f"tps_mtp_n{n}.json")
    if os.path.exists(fp):
        d = json.load(open(fp))["concurrency"]
        n1, n4 = d["n1"]["mean"], d["n4"]["mean"]
        print(f"| n={n}   | {n1:>7.1f} | {(n1-b1)/b1*100:>+5.1f}% | {n4:>7.1f} | {(n4-b4)/b4*100:>+5.1f}% |")
PYEOF

log "Done. Results in ${RESULTS_DIR}"
