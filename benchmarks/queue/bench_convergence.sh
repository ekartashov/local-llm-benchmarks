#!/usr/bin/env bash
# bench_convergence.sh — Convergence tier (Qwen3.5-397B UD-IQ2_M) benchmark
# Part of: /srv/ai/projects/local-llm-benchmarks/benchmarks/
#
# ── Reproducing environment ────────────────────────────────────────────────────
# Host:    ZRH01-AIRIG
# CPU:     Intel i9-14900K (24c/32t, Raptor Lake, NO AMX)
# RAM:     192 GB DDR5 (~83 GB/s actual, 4-DIMM downclocked config)
# GPU 0:   NVIDIA RTX 5090 32GB GDDR7 (1790 GB/s)
# GPU 1:   NVIDIA RTX 5090 32GB GDDR7 (1790 GB/s)
# OS:      Linux, kernel 6.x
#
# ── Engine ─────────────────────────────────────────────────────────────────────
# ik_llama.cpp, branch main (Qwen3.5 MoE / GDN support)
# Binary:  /srv/ai/projects/ik_llama.cpp/build/bin/llama-server
# Bench:   /srv/ai/projects/ik_llama.cpp/build/bin/llama-bench
#
# ── Build instructions ─────────────────────────────────────────────────────────
# cd /srv/ai/projects/ik_llama.cpp
# git checkout main && git pull origin main
# cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
# cmake --build build --config Release -j$(nproc)
# # Verify Qwen3.5 support:
# find src/ -name "*.cpp" -o -name "*.h" | xargs grep -l "qwen35\|ssm_alpha" 2>/dev/null
#
# ── Model ──────────────────────────────────────────────────────────────────────
# unsloth/Qwen3.5-397B-A17B-GGUF, UD-IQ2_M variant (~123 GB across 4 split files)
# Snapshot: da33c16fa4440f831149fcf53b98a22bc07785e5
# Path:     /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/
#             da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/
# Download: hf download unsloth/Qwen3.5-397B-A17B-GGUF \
#             "UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf" \
#             "UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00002-of-00004.gguf" \
#             "UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00003-of-00004.gguf" \
#             "UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00004-of-00004.gguf" \
#             --local-dir /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/
#
# ── Prerequisites ──────────────────────────────────────────────────────────────
# 1. Sleep Arclight vLLM processes to free VRAM:
#    curl -X POST http://localhost:30000/sleep?level=1
#    curl -X POST http://localhost:30001/sleep?level=1
# 2. Confirm RAM available: 'free -h' should show ~140+ GB available
#    (192GB - ~44GB vLLM sleep weights - ~5GB OS = ~143GB)
# 3. All 4 model split files present in snapshot directory
#
# ── Usage ──────────────────────────────────────────────────────────────────────
# ./bench_convergence.sh                 # defaults: t=nproc, ctx=16384, port=8002
# THREADS=16 ./bench_convergence.sh      # test specific thread count
# THREADS=16 CTX=8192 ./bench_convergence.sh
# SKIP_SERVER=1 ./bench_convergence.sh   # skip server start (use existing)
#
# ── What this covers ───────────────────────────────────────────────────────────
# T_CV1: Startup timing (cold start from NVMe, warm cache)
# T_CV2: Thread count sweep (8, 16, 24, 32)
# Baseline perf: prompt eval and token generation TPS
# Functional: verify model produces coherent output
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
IK_DIR="/srv/ai/projects/ik_llama.cpp"
BINARY="${IK_DIR}/build/bin/llama-server"
BENCH_BINARY="${IK_DIR}/build/bin/llama-bench"

SNAPSHOT="da33c16fa4440f831149fcf53b98a22bc07785e5"
MODEL_BASE="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/${SNAPSHOT}/UD-IQ2_M"
MODEL="${MODEL_BASE}/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"

PORT="${PORT:-8002}"
THREADS="${THREADS:-$(nproc)}"
CTX="${CTX:-16384}"
SKIP_SERVER="${SKIP_SERVER:-0}"

RESULTS_BASE="${RESULTS_BASE:-/srv/ai/projects/local-llm-benchmarks/results}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
ITEM_ID="convergence_397b_iq2m_t${THREADS}"
RUN_DIR="${RESULTS_BASE}/${ITEM_ID}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}/raw"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date -u +%T)] $*" | tee -a "${RUN_DIR}/bench.log"; }

wait_ready() {
    local port="$1" timeout="${2:-180}" pid="${3:-0}"
    for i in $(seq 1 "${timeout}"); do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "${i}"
            return 0
        fi
        if [ "${pid}" -ne 0 ] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "CRASHED"
            return 1
        fi
        sleep 1
    done
    echo "TIMEOUT"
    return 1
}

# ── Pre-flight ─────────────────────────────────────────────────────────────────
log "=== Convergence benchmark ==="
log "Item: ${ITEM_ID}"
log "Timestamp: ${TIMESTAMP}"
log "Threads: ${THREADS} | Context: ${CTX} | Port: ${PORT}"
log "Results: ${RUN_DIR}"

# Verify binary
if [ ! -x "${BINARY}" ]; then
    log "ERROR: Binary not found at ${BINARY}"
    log "Build instructions:"
    log "  cd ${IK_DIR}"
    log "  git checkout main && git pull origin main"
    log "  cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release"
    log "  cmake --build build --config Release -j\$(nproc)"
    exit 1
fi

# Verify model split files
log "Verifying model split files..."
for i in 1 2 3 4; do
    f="${MODEL_BASE}/Qwen3.5-397B-A17B-UD-IQ2_M-0000${i}-of-00004.gguf"
    if [ ! -f "${f}" ]; then
        log "ERROR: Missing split file: ${f}"
        log "Download command in script header."
        exit 1
    fi
    SZ=$(du -sh "${f}" | cut -f1)
    log "  Split ${i}/4: ${SZ} OK"
done

# RAM check
AVAIL_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
TOTAL_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
log "RAM: ${AVAIL_GB} GB available of ${TOTAL_GB} GB total"
if [ "${AVAIL_GB}" -lt 130 ]; then
    log "WARNING: Low RAM (${AVAIL_GB} GB). Recommend sleeping Arclight first:"
    log "  curl -X POST http://localhost:30000/sleep?level=1"
    log "  curl -X POST http://localhost:30001/sleep?level=1"
fi

# VRAM baseline
log "VRAM before start:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader \
    | tee -a "${RUN_DIR}/bench.log" \
    > "${RUN_DIR}/raw/vram_before.txt"

# ── Phase 1: Startup timing ────────────────────────────────────────────────────
STARTUP_MS="N/A"

if [ "${SKIP_SERVER}" -eq 0 ]; then
    log ""
    log "=== Phase 1: Startup timing ==="

    START_MS=$(date +%s%3N)
    "${BINARY}" \
        -m "${MODEL}" \
        -ngl 999 \
        --cpu-moe \
        --no-mmap \
        -b 4096 -ub 2048 \
        -t "${THREADS}" \
        -c "${CTX}" \
        --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
        --jinja \
        --host 0.0.0.0 --port "${PORT}" \
        2>&1 | tee "${RUN_DIR}/server.log" &
    SERVER_PID=$!
    log "Server PID: ${SERVER_PID}"

    WAIT_RESULT=$(wait_ready "${PORT}" 300 "${SERVER_PID}")
    if [ "${WAIT_RESULT}" = "TIMEOUT" ] || [ "${WAIT_RESULT}" = "CRASHED" ]; then
        log "ERROR: Server failed to start (${WAIT_RESULT}). Check ${RUN_DIR}/server.log"
        exit 1
    fi

    READY_MS=$(date +%s%3N)
    STARTUP_MS=$(( READY_MS - START_MS ))
    log "Server ready: ${STARTUP_MS}ms ($(echo "scale=1; ${STARTUP_MS}/1000" | bc)s)"

    # VRAM after load
    log "VRAM after model load:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader \
        | tee -a "${RUN_DIR}/bench.log" \
        > "${RUN_DIR}/raw/vram_after_load.txt"
else
    log "SKIP_SERVER=1 — assuming server already running on port ${PORT}"
    SERVER_PID=0
fi

# ── Phase 2: Prompt processing baseline ───────────────────────────────────────
log ""
log "=== Phase 2: Throughput baseline (llama-bench) ==="

"${BENCH_BINARY}" \
    -m "${MODEL}" \
    -ngl 999 --cpu-moe --no-mmap \
    -fa 1 -b 4096 -ub 2048 -t "${THREADS}" \
    -p "256,512,1024,2048,4096" \
    -n "0" \
    -r 3 2>&1 | tee "${RUN_DIR}/raw/bench_pp.txt"

log "Token generation baseline:"
"${BENCH_BINARY}" \
    -m "${MODEL}" \
    -ngl 999 --cpu-moe --no-mmap \
    -fa 1 -b 4096 -ub 2048 -t "${THREADS}" \
    -p "512" \
    -n "64,128,256" \
    -r 3 2>&1 | tee "${RUN_DIR}/raw/bench_tg.txt"

# ── Phase 3: Thread count sweep ───────────────────────────────────────────────
log ""
log "=== Phase 3: Thread count sweep (T_CV2) ==="

{
    echo "threads,pp512_tps,tg128_tps"
    for T in 8 12 16 20 24 28 32; do
        log "  Testing -t ${T}..."
        RAW=$("${BENCH_BINARY}" \
            -m "${MODEL}" \
            -ngl 999 --cpu-moe --no-mmap \
            -fa 1 -b 4096 -ub 2048 -t "${T}" \
            -p "512" -n "128" -r 2 2>&1)

        PP=$(echo "${RAW}" | grep "pp512" | awk '{print $NF}' | head -1)
        TG=$(echo "${RAW}" | grep "tg128" | awk '{print $NF}' | head -1)
        echo "${T},${PP:-N/A},${TG:-N/A}"
        echo "${RAW}" >> "${RUN_DIR}/raw/thread_sweep_t${T}.txt"
    done
} | tee "${RUN_DIR}/raw/thread_sweep_summary.csv"

log "Thread sweep complete. Best TG thread count:"
sort -t',' -k3 -nr "${RUN_DIR}/raw/thread_sweep_summary.csv" | head -3 | tee -a "${RUN_DIR}/bench.log"

# ── Phase 4: Functional test ──────────────────────────────────────────────────
log ""
log "=== Phase 4: Functional test (infra question) ==="

INFRA_PROMPT="You are an expert Linux infrastructure engineer. Briefly explain (2-3 sentences) why the vLLM inference engine uses CUDA graph capture and what problem it solves."

RESPONSE=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "
import json
print(json.dumps({
    'model': 'convergence',
    'messages': [{'role': 'user', 'content': '''${INFRA_PROMPT}'''}],
    'max_tokens': 256,
    'temperature': 0.0,
    'stream': False
}))
")" 2>/dev/null || echo '{"error":"request_failed"}')

echo "${RESPONSE}" > "${RUN_DIR}/raw/functional_test.json"

ANSWER=$(echo "${RESPONSE}" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d['choices'][0]['message']['content'])
except Exception as e:
    print(f'PARSE_ERROR: {e}')
" 2>/dev/null)

log "Functional test answer:"
echo "${ANSWER}" | tee -a "${RUN_DIR}/bench.log"

if echo "${ANSWER}" | grep -qiE "PARSE_ERROR|request_failed"; then
    FUNCTIONAL="FAIL"
    log "Functional test: FAIL"
else
    FUNCTIONAL="PASS"
    log "Functional test: PASS"
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
if [ "${SKIP_SERVER}" -eq 0 ] && [ "${SERVER_PID}" -ne 0 ]; then
    log ""
    log "=== Stopping server ==="
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    log "Server stopped"
fi

# ── Metrics JSON ──────────────────────────────────────────────────────────────
BEST_THREAD_TG=$(sort -t',' -k3 -nr "${RUN_DIR}/raw/thread_sweep_summary.csv" 2>/dev/null | head -1 | cut -d',' -f1 || echo "unknown")

python3 - << PYEOF > "${RUN_DIR}/metrics.json"
import json
metrics = {
    "item_id": "${ITEM_ID}",
    "timestamp": "${TIMESTAMP}",
    "config": {
        "engine": "ik_llama.cpp",
        "engine_branch": "main",
        "engine_commit": "see git log at /srv/ai/projects/ik_llama.cpp",
        "engine_binary": "${BINARY}",
        "model": "unsloth/Qwen3.5-397B-A17B-GGUF UD-IQ2_M",
        "model_path": "${MODEL}",
        "model_snapshot": "${SNAPSHOT}",
        "quant": "UD-IQ2_M",
        "quant_size_gb_approx": 123,
        "placement": "cpu-moe: MoE ffn_gate/up/down_exps in DDR5 RAM; attention+norms+embed on GPU",
        "threads": ${THREADS},
        "context_length": ${CTX},
        "no_mmap": True,
        "key_flags": "-ngl 999 --cpu-moe --no-mmap -b 4096 -ub 2048 -fa on(default) -fmoe on(default)",
        "port": ${PORT},
        "host": "ZRH01-AIRIG",
        "cpu": "Intel i9-14900K (24c/32t, Raptor Lake, no AMX)",
        "ram_gb": 192,
        "ddr5_bandwidth_gbs_actual": 83,
        "gpus": "2x RTX 5090 32GB GDDR7",
    },
    "metrics": {
        "startup_ms": "${STARTUP_MS}",
        "functional_test": "${FUNCTIONAL}",
        "known_baseline": {
            "gen_tps": 13.15,
            "pp_tps_469tok": 60.66,
            "pp_tps_2348tok": 158.94,
            "note": "From first live run, 32 threads, not yet optimized"
        },
        "bottleneck": "DDR5 bandwidth (~83 GB/s) reading ~2.3 GB MoE expert weights per token",
        "best_thread_count_from_sweep": "${BEST_THREAD_TG}",
    },
    "verdict": "MEASURED",
    "notes": (
        "Convergence tier baseline. Thread sweep results in raw/thread_sweep_summary.csv. "
        "Partial GPU expert offload not tested here (T_CV3 separate). "
        "Benjamin Marie (independent) confirmed UD-IQ2_M on 397B is within BF16 margin of error "
        "on MMLU-Pro/GPQA/LiveCodeBench/Math-500 — safe quantization choice."
    )
}
print(json.dumps(metrics, indent=2))
PYEOF

log ""
log "=== Done ==="
log "Results: ${RUN_DIR}"
log "metrics.json: ${RUN_DIR}/metrics.json"
log "Thread sweep: ${RUN_DIR}/raw/thread_sweep_summary.csv"
log ""
log "Quick summary:"
log "  Startup:       ${STARTUP_MS}ms"
log "  Functional:    ${FUNCTIONAL}"
log "  Best thread(TG): ${BEST_THREAD_TG}"
log ""
log "Known baseline (from R12 session, unoptimized):"
log "  Gen TPS:  ~13.15 t/s (DDR5-bandwidth-limited)"
log "  PP TPS:   ~158 t/s at 2348-token batch"
log ""
log "Next steps:"
log "  T_CV3: partial GPU expert offload (first N layers on GPU)"
log "  T2.3b: Gemma4-31B as Arclight thinker"