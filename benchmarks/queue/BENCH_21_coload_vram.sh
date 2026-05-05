#!/usr/bin/env bash
# BENCH_21 — Co-load VRAM mapping — 2026-05-05
# Three-way co-load: thinker TP=1 (GPU1) + coder TP=1 (GPU0) + Convergence (both GPUs, low ngl).
# Verifies thinker --max-model-len 131072 with MTP n=3 co-resident with coder and Convergence.
#
# NOTE: vLLM on this host is 0.20.0.  VLLM_USE_V1 / VLLM_ENGINE_ITERATOR_SOURCE are UNKNOWN
# env vars in 0.20.0 and are silently ignored — V1 is the ONLY engine.  Do NOT pass them.
#
# VRAM budget (measured):
#   GPU0 total:  31,360 MiB  (~31.3 GiB)
#   GPU1 total:  31,360 MiB  (~31.3 GiB)
#   Thinker alone (util=0.95):  29,278 MiB on GPU1 → 2,834 MiB free on GPU1
#   Coder alone (util=0.80 eager): ~25,060 MiB on GPU0 → ~6,300 MiB free on GPU0
#   Convergence at ngl=999 --cpu-moe: 8,182/8,684 MiB per GPU — does NOT fit after thinker+coder.
#   Convergence at ngl≤18 (estimated): ≤2,200 MiB per GPU — fits within residual margins.
#
# Default CONVERGENCE_NGL=15 is the starting point.  Try 18→20 if 15 succeeds.

set -euo pipefail

# Configuration
CODER_MEM_UTIL=${CODER_MEM_UTIL:-0.80}
THINKER_MEM_UTIL=${THINKER_MEM_UTIL:-0.95}
CONVERGENCE_NGL=${CONVERGENCE_NGL:-15}
SKIP_CONVERGENCE=${SKIP_CONVERGENCE:-0}
SKIP_THINKER=${SKIP_THINKER:-0}
SKIP_CODER=${SKIP_CODER:-0}

# Parse CLI arguments for skipping
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-thinker) SKIP_THINKER=1; shift ;;
    --skip-coder) SKIP_CODER=1; shift ;;
    --skip-convergence) SKIP_CONVERGENCE=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Cleanup existing processes to ensure a clean slate
echo "=== Cleaning up existing processes ==="
pkill -9 llama-server || true
pkill -9 vllm || true
podman stop -a || true
podman rm -f $(podman ps -aq) || true
sleep 5

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_21_coload_vram_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

IK_BIN=/srv/ai/projects/ik_llama.cpp/build/bin/llama-server
CONVERGENCE_GGUF=/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf
TPS_PROMPT="Explain the CAP theorem and its implications for distributed database design."

echo "=== BENCH_21: Co-load VRAM mapping ==="
echo "Coder Mem Util:    ${CODER_MEM_UTIL}"
echo "Thinker Mem Util:  ${THINKER_MEM_UTIL}"
echo "Convergence NGL:   ${CONVERGENCE_NGL}"
echo "Results Dir:       ${RESULTS_DIR}"
echo "---------------------------------------"

# === Record VRAM baseline (everything stopped) ===
echo "=== VRAM baseline (pre-load) ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | tee -a "${RESULTS_DIR}/vram_log.txt"

# PHASE 1: Start thinker with --max-model-len 131072
# --hf-overrides forces Qwen3_5ForCausalLM architecture needed for 128K hybrid attention pool.
# MTP n=3 is the settled production config (BENCH_19).
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is required for ctx>32768 in vLLM 0.20+.
if [ "${SKIP_THINKER}" = "0" ]; then
  echo "=== Starting thinker (gpu_mem_util=${THINKER_MEM_UTIL}, max-model-len=131072) ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm gpu1 rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm \
    --gpu-mem-util "${THINKER_MEM_UTIL}" \
    --ctx 131072 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-seqs 4 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
    "--speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":3}" \
    2>&1 | tee "${RESULTS_DIR}/thinker_deploy.log"
  echo "[thinker] HEALTH OK"
  sleep 5
  echo "=== VRAM after Thinker (Phase 1) ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram_log.txt"
else
  echo "[skip] SKIP_THINKER=1"
fi

# PHASE 2: Start coder on GPU0 only (TP=1).
# --enforce-eager eliminates CUDA graph capture, reducing peak profiling VRAM from ~30.7 GiB
# to ~24.5 GiB.  At util=0.80: budget=25.1 GiB, non-KV overhead~24.5 GiB → KV≈+0.6 GiB.
# Coder has NO MTP (BENCH_14: breaks tool calls on A3B MoE) and NO hf-overrides.
if [ "${SKIP_CODER}" = "0" ]; then
  echo "=== Starting coder (gpu_mem_util=${CODER_MEM_UTIL}, enforce-eager, GPU0 only) ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm gpu0 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
    --gpu-mem-util "${CODER_MEM_UTIL}" \
    --ctx 32768 \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    2>&1 | tee "${RESULTS_DIR}/coder_deploy.log"
  echo "[coder] HEALTH OK"
  sleep 5
  echo "=== VRAM after Coder (Phase 2) ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram_log.txt"
else
  echo "[skip] SKIP_CODER=1"
fi

# PHASE 3: Start Convergence with REDUCED ngl to fit in residual VRAM.
# After thinker (GPU1 free: ~2,834 MiB) and coder (GPU0 free: ~6,300 MiB), Convergence
# must stay within those margins.  ngl=999 uses 8,182/8,684 MiB — does NOT fit.
# Estimated safe ceiling: ngl≤18 (≈1,400 MiB/GPU).  Default CONVERGENCE_NGL=15.
# TPS will be lower than isolated 14 t/s; this is expected for the co-load test.
if [ "${SKIP_CONVERGENCE}" = "0" ]; then
  echo "=== Starting Convergence (ngl=${CONVERGENCE_NGL}, filling residual gaps) ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
  echo "convergence_ngl=${CONVERGENCE_NGL}" >> "${RESULTS_DIR}/results.txt"
  "${IK_BIN}" \
    -m "${CONVERGENCE_GGUF}" \
    -ngl "${CONVERGENCE_NGL}" --cpu-moe \
    -b 4096 -ub 2048 -t 32 -np 4 -c 131072 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --jinja --host 0.0.0.0 --port 8002 \
    >> "${RESULTS_DIR}/convergence.log" 2>&1 &
  CONVERGENCE_PID=$!
  echo "[convergence] Waiting for health OK (PID=${CONVERGENCE_PID})..."
  CONVERGENCE_UP=0
  for i in $(seq 1 120); do
    if curl -sf http://localhost:8002/health 2>/dev/null; then
      echo "[convergence] HEALTH OK"
      CONVERGENCE_UP=1
      break
    fi
    sleep 1
  done
  if [ "${CONVERGENCE_UP}" = "0" ]; then
    echo "[convergence] FAILED to start within 120s — check ${RESULTS_DIR}/convergence.log"
    echo "convergence_status=FAIL" >> "${RESULTS_DIR}/results.txt"
  fi
else
  echo "[skip] SKIP_CONVERGENCE=1"
fi

# ===================================================================
# PHASE 4: Three-model VRAM snapshot
# ===================================================================
echo "=== FINAL three-model co-load VRAM snapshot ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader | tee -a "${RESULTS_DIR}/vram_log.txt"
nvidia-smi --format=csv --query-compute-apps=pid,process_name,used_memory | tee -a "${RESULTS_DIR}/vram_per_process.txt"
echo "=== All running container names ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
podman ps --format "{{.Names}}\t{{.Status}}" | tee -a "${RESULTS_DIR}/vram_log.txt"

# Record configuration used
echo "coder_mem_util=${CODER_MEM_UTIL}" >> "${RESULTS_DIR}/results.txt"
echo "thinker_mem_util=${THINKER_MEM_UTIL}" >> "${RESULTS_DIR}/results.txt"
echo "thinker_max_model_len=131072" >> "${RESULTS_DIR}/results.txt"

# ===================================================================
# PHASE 5: Convergence TPS under co-load (3 sequential requests)
# ===================================================================
echo "=== Convergence TPS under co-load ===" | tee -a "${RESULTS_DIR}/vram_log.txt"
CONV_TPS_RESULTS="${RESULTS_DIR}/convergence_coload_tps.csv"
echo "rep,ttft_s,tokens_generated,tps" > "${CONV_TPS_RESULTS}"

for REP in 1 2 3; do
  START_MS=$(date +%s%3N)
  RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"convergence\",\"prompt\":\"${TPS_PROMPT}\",\"max_tokens\":100,\"temperature\":0.0}" 2>/dev/null || echo "ERROR")
  END_MS=$(date +%s%3N)

  if [ "$RESPONSE" = "ERROR" ]; then
    echo "${REP},ERROR,ERROR,ERROR" >> "${CONV_TPS_RESULTS}"
    continue
  fi

  ELAPSED_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  TOKENS=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); usage=d.get('usage',{}); print(usage.get('completion_tokens', usage.get('output_tokens', 'UNKNOWN')))" 2>/dev/null || echo "UNKNOWN")
  TPS=$(python3 -c "
import sys
tokens = '${TOKENS}'
elapsed = ${ELAPSED_S}
try:
    print(round(float(tokens) / elapsed, 2)) if elapsed > 0 else print('N/A')
except:
    print('N/A')
" 2>/dev/null)

  echo "${REP},${ELAPSED_S},${TOKENS},${TPS}" >> "${CONV_TPS_RESULTS}"
  echo "[convergence TPS] rep=${REP} elapsed=${ELAPSED_S}s tokens=${TOKENS} tps=${TPS}"
done

cat "${CONV_TPS_RESULTS}" | tee -a "${RESULTS_DIR}/vram_log.txt"

# Record Convergence TPS isolated baseline for comparison
echo "convergence_isolated_tps=13.99" >> "${RESULTS_DIR}/results.txt"
echo "See results/T_CV3_* for isolated baseline (13.99 t/s)"

echo "=== BENCH_21 complete — results in ${RESULTS_DIR} ==="
