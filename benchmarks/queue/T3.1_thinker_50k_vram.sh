#!/usr/bin/env bash
# T3.1 Phase 1 — Thinker fp8 KV cache VRAM headroom at 50K context
#
# Objective: Determine whether the thinker (QuantTrio/Qwen3.6-27B-AWQ, TP=1, GPU1)
# can deploy at 51200 token context with fp8 KV cache without OOM on GPU1's 32 GB VRAM.

set -euo pipefail

# ── Flag Parsing ──────────────────────────────────────────────────────────────
SKIP_BASELINE=0
SKIP_DEPLOY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-baseline) SKIP_BASELINE=1; shift ;;
    --skip-deploy)   SKIP_DEPLOY=1;   shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ── Step 0: Robust Cleanup ───────────────────────────────────────────────────
if [ "${SKIP_BASELINE}" -eq 0 ] && [ "${SKIP_DEPLOY}" -eq 0 ]; then
  echo "Step 0: Clearing ghost processes and residue..."

  # Kill any stray vLLM processes
  pkill -u "$USER" -9 -f "vllm|VLLM::" || true

  # Clear shared memory residue
  sudo rm -rf /dev/shm/* 2>/dev/null || true

  # Clear port 30001
  sudo fuser -k 30001/tcp 2>/dev/null || true
fi

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="QuantTrio/Qwen3.6-27B-AWQ"
PROMPT="List the three laws of thermodynamics in one sentence each."
MAX_TOKENS=50
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/T3.1_thinker_50k_vram_${TIMESTAMP}"
TIMINGS_FILE="${RESULTS_DIR}/timings.txt"
STATUS_FILE="${RESULTS_DIR}/status.txt"

mkdir -p "${RESULTS_DIR}"

echo "Starting T3.1 Phase 1 (BENCH_11)..."
echo "Results will be saved to: ${RESULTS_DIR}"

# ── Step 1: Record baseline VRAM and confirm thinker healthy ──────────────────
echo "Step 1: Recording baseline VRAM (32K)..."

# Confirm thinker is running, or deploy it
if ! curl -sf http://localhost:30001/health >/dev/null; then
  if [ "${SKIP_BASELINE}" -eq 1 ] || [ "${SKIP_DEPLOY}" -eq 1 ]; then
    echo "ERROR: Baseline thinker not running and skip-deploy requested."
    exit 1
  fi
  echo "Production thinker not running. Deploying 32K baseline..."
  VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm gpu1 "${MODEL}" \
    --gpu-mem-util 0.90 \
    --ctx 32768 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --max-num-seqs 1 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --served-model-name thinker
fi

VRAM_32K=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
echo "GPU1 VRAM (32K baseline): ${VRAM_32K} MiB"
echo "${VRAM_32K}" > "${RESULTS_DIR}/vram_32k_gpu1_mib.txt"
echo "vram_32k_gpu1_mib=${VRAM_32K}" > "${TIMINGS_FILE}"

# Short inference to confirm baseline health
START_MS=$(date +%s%3N)
BASELINE_RESPONSE=$(curl -sf http://localhost:30001/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"thinker\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0.0}")
END_MS=$(date +%s%3N)
BASELINE_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "Baseline TTFT: ${BASELINE_TTFT_S}s"
echo "${BASELINE_RESPONSE}" > "${RESULTS_DIR}/baseline_response.json"
echo "baseline_ttft_s=${BASELINE_TTFT_S}" >> "${TIMINGS_FILE}"

# ── Step 2: Redeploy thinker at 50K context ───────────────────────────────────
echo "Step 2: Redeploying thinker at 50K context..."

THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1 || true)
if [ -n "${THINKER_CONTAINER}" ]; then
  echo "Stopping: ${THINKER_CONTAINER}"
  podman stop "${THINKER_CONTAINER}" 2>/dev/null || true
  podman rm "${THINKER_CONTAINER}" 2>/dev/null || true
fi

# Deploy at 50K
if [ "${SKIP_DEPLOY}" -eq 0 ]; then
  VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm gpu1 "${MODEL}" \
    --gpu-mem-util 0.90 \
    --ctx 51200 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --max-num-seqs 1 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --served-model-name thinker
fi

# Wait up to 120s for health
HEALTH_OK=0
echo "Waiting for thinker to become healthy (50K)..."
for i in $(seq 1 120); do
  if curl -sf http://localhost:30001/health 2>/dev/null; then
    echo "THINKER READY (50K)"
    HEALTH_OK=1
    break
  fi
  sleep 1
done

echo "health_ok=${HEALTH_OK}" >> "${TIMINGS_FILE}"

# ── Step 3: Handle OOM at startup ─────────────────────────────────────────────
if [ "${HEALTH_OK}" -eq 0 ]; then
  echo "OOM_AT_STARTUP" > "${STATUS_FILE}"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_at_oom.txt"
  echo "GPU VRAM at OOM:"
  cat "${RESULTS_DIR}/vram_at_oom.txt"

  # Collect startup logs
  THINKER_CONTAINER=$(podman ps -a --format "{{.Names}}" | grep -i "thinker" | head -1)
  if [ -n "${THINKER_CONTAINER}" ]; then
    podman logs "${THINKER_CONTAINER}" 2>&1 | tail -40 > "${RESULTS_DIR}/startup_logs.txt"
  fi
  echo "OOM at 50K context. See startup_logs.txt."
else
  # ── Step 4: Record 50K VRAM and run inference ───────────────────────────────
  echo "Step 4: Recording 50K VRAM and running inference..."
  VRAM_50K=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
  echo "GPU1 VRAM (50K context): ${VRAM_50K} MiB"
  echo "${VRAM_50K}" > "${RESULTS_DIR}/vram_50k_gpu1_mib.txt"
  echo "vram_50k_gpu1_mib=${VRAM_50K}" >> "${TIMINGS_FILE}"

  VRAM_DELTA=$((VRAM_50K - VRAM_32K))
  KV_DELTA_PER_KTOK=$(python3 -c "print(round(${VRAM_DELTA} / 18.432, 1))")
  echo "VRAM delta: +${VRAM_DELTA} MiB for +18432 tokens"
  echo "  fp8 KV overhead: ~${KV_DELTA_PER_KTOK} MiB per 1K context tokens"
  echo "vram_delta_mib=${VRAM_DELTA}" >> "${TIMINGS_FILE}"
  echo "vram_mib_per_1k_ctx_fp8=${KV_DELTA_PER_KTOK}" >> "${TIMINGS_FILE}"

  # Inference at 50K context
  START_MS=$(date +%s%3N)
  RESPONSE_50K=$(curl -sf http://localhost:30001/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"thinker\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0.0}")
  END_MS=$(date +%s%3N)
  TTFT_50K_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  echo "TTFT at 50K context: ${TTFT_50K_S}s"
  echo "${RESPONSE_50K}" > "${RESULTS_DIR}/inference_50k.json"
  echo "ttft_50k_s=${TTFT_50K_S}" >> "${TIMINGS_FILE}"

  # Text comparison
  BASELINE_TEXT=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/baseline_response.json'))['choices'][0]['text'])" 2>/dev/null || echo "")
  TEXT_50K=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/inference_50k.json'))['choices'][0]['text'])" 2>/dev/null || echo "")
  if [ "${BASELINE_TEXT}" = "${TEXT_50K}" ]; then
    TEXT_MATCH="identical"
  else
    TEXT_MATCH="differs"
  fi
  echo "Text match vs baseline: ${TEXT_MATCH}"
  echo "text_match=${TEXT_MATCH}" >> "${TIMINGS_FILE}"

  echo "STARTUP_OK" > "${STATUS_FILE}"
fi

# ── Step 5: Restore production thinker ────────────────────────────────────────
if [ "${SKIP_DEPLOY}" -eq 0 ]; then
  echo "Step 5: Restoring production thinker (32K)..."

  THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1 || true)
  if [ -n "${THINKER_CONTAINER}" ]; then
    podman stop "${THINKER_CONTAINER}" 2>/dev/null || true
    podman rm "${THINKER_CONTAINER}" 2>/dev/null || true
  fi

  VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm gpu1 "${MODEL}" \
    --gpu-mem-util 0.90 \
    --ctx 32768 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --max-num-seqs 1 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --served-model-name thinker
fi

for i in $(seq 1 120); do
  if curl -sf http://localhost:30001/health >/dev/null; then
    echo "PRODUCTION THINKER RESTORED"
    break
  fi
  sleep 1
done

echo "Benchmark complete. Results in: ${RESULTS_DIR}"

# ── Generate Summary ─────────────────────────────────────────────────────────
STATUS=$(cat "${STATUS_FILE}")
VRAM_32K_VAL=$(grep "vram_32k_gpu1_mib=" "${TIMINGS_FILE}" | cut -d= -f2 || echo "N/A")
VRAM_50K_VAL=$(grep "vram_50k_gpu1_mib=" "${TIMINGS_FILE}" | cut -d= -f2 || echo "N/A")
VRAM_DELTA_VAL=$(grep "vram_delta_mib=" "${TIMINGS_FILE}" | cut -d= -f2 || echo "N/A")
KV_PER_1K_VAL=$(grep "vram_mib_per_1k_ctx_fp8=" "${TIMINGS_FILE}" | cut -d= -f2 || echo "N/A")
BASELINE_TTFT_VAL=$(grep "baseline_ttft_s=" "${TIMINGS_FILE}" | cut -d= -f2 || echo "N/A")
TTFT_50K_VAL=$(grep "ttft_50k_s=" "${TIMINGS_FILE}" | cut -d= -f2 || echo "N/A")
TEXT_MATCH_VAL=$(grep "text_match=" "${TIMINGS_FILE}" | cut -d= -f2 || echo "N/A")

cat <<EOF > "${RESULTS_DIR}/summary.md"
# T3.1 Phase 1 — Thinker 50K Context VRAM Feasibility — ${TIMESTAMP}

## Result
${STATUS}

| Metric | Value |
|--------|-------|
| GPU1 VRAM at 32K | ${VRAM_32K_VAL} MiB |
| GPU1 VRAM at 50K | ${VRAM_50K_VAL} MiB (or N/A — OOM) |
| VRAM delta | +${VRAM_DELTA_VAL} MiB (or N/A) |
| fp8 KV per 1K tokens | ~${KV_PER_1K_VAL} MiB (or N/A) |
| Baseline TTFT | ${BASELINE_TTFT_VAL}s |
| TTFT at 50K | ${TTFT_50K_VAL}s (or N/A) |
| Text match | ${TEXT_MATCH_VAL} |

## Status
${STATUS}
EOF
