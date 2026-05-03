#!/usr/bin/env bash
# T_HARD1_thinker_hard_suite.sh — AWQ vs PrismaQuant Head-to-Head Quality Eval
#
# Objective: Run a 10-task hard systems engineering suite against both 
# PrismaQuant 5.5bit and AWQ 4bit thinkers on GPU1.
#
# Context: Calibrated to find the differentiation point where models diverge
# on deep reasoning chains (Linux internals, Raft, K8s, etc.).
#
# Usage: ./benchmarks/queue/T_HARD1_thinker_hard_suite.sh
#
# Pass Criteria:
#   1. All 20 response files collected (10 per model).
#   2. No task truncated (finish_reason="stop").
#   3. Production thinker restored.
#
# NOTE: This script does NOT score responses. Scoring is done in research mode.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

# ── Configuration ────────────────────────────────────────────────────────────
ITEM_ID="T_HARD1_thinker_hard_suite"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
TASK_DIR="${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker_hard"

MODEL_PQ="rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm"
MODEL_AWQ="QuantTrio/Qwen3.6-27B-AWQ"
THINKER_PORT="${PORT_VLLM_GPU1}"
THINKER_URL="http://localhost:${THINKER_PORT}/v1/chat/completions"
MAX_TOKENS_PQ=28000
MAX_TOKENS_AWQ=16384

mkdir -p "${RESULTS_DIR}/pq" "${RESULTS_DIR}/awq"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[T_HARD1 $(date -u +%H:%M:%S)] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

# ── Preflight ────────────────────────────────────────────────────────────────
log "Starting T_HARD1 (Hard Systems Suite)..."
log "Results: ${RESULTS_DIR}"

[ -d "${TASK_DIR}" ] || die "Task directory not found: ${TASK_DIR}"
TASK_COUNT=$(ls "${TASK_DIR}"/*.json 2>/dev/null | wc -l)
[ "${TASK_COUNT}" -eq 10 ] || log "WARNING: Expected 10 tasks, found ${TASK_COUNT}"

# ── Helper: Run Tasks ────────────────────────────────────────────────────────
run_tasks() {
    local target_dir="$1"
    local max_tokens="$2"
    local model_id_file="${target_dir}/model_id.txt"
    
    # Get active model ID
    curl -s "http://localhost:${THINKER_PORT}/v1/models" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" \
        > "${model_id_file}"
    local active_id=$(cat "${model_id_file}")
    log "Running tasks against active model: ${active_id}"

    # Record VRAM
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "${GPU_1_ID}" > "${target_dir}/vram.txt"

    for TASK_FILE in "${TASK_DIR}"/*.json; do
        TASK_ID=$(python3 -c "import json; d=json.load(open('${TASK_FILE}')); print(d['id'])")
        TASK_NUM=$(basename "${TASK_FILE}" .json | cut -d_ -f1)
        log "  Task ${TASK_NUM} (${TASK_ID})..."

        python3 - <<PYEOF > "${target_dir}/${TASK_NUM}_${TASK_ID}.json"
import json, urllib.request, time

task = json.load(open("${TASK_FILE}"))
model_id = "${active_id}"

payload = json.dumps({
    "model": model_id,
    "messages": [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": task["user_message"]}
    ],
    "max_tokens": ${max_tokens},
    "temperature": 0.0,
    "stream": False
}).encode()

t0 = time.perf_counter()
req = urllib.request.Request(
    "${THINKER_URL}",
    data=payload,
    headers={"Content-Type": "application/json"}
)
try:
    with urllib.request.urlopen(req, timeout=1200) as resp:
        body = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    out = {
        "task_id": task["id"],
        "model": model_id,
        "elapsed_s": round(elapsed, 2),
        "completion_tokens": body["usage"]["completion_tokens"],
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "tps": round(body["usage"]["completion_tokens"] / elapsed, 1) if elapsed > 0 else 0,
        "response": body["choices"][0]["message"]["content"],
        "finish_reason": body["choices"][0]["finish_reason"]
    }
except Exception as e:
    out = {"task_id": task["id"], "error": str(e)}

print(json.dumps(out, indent=2))
PYEOF
    done
}

# ── Step 1: Deploy & Run PrismaQuant ─────────────────────────────────────────
log "Step 1: Deploying PrismaQuant Thinker..."

# Stop existing if any
EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
if [ -n "${EXISTING}" ]; then
    log "Stopping ${EXISTING}..."
    podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 2
fi

VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL_PQ}" \
    --gpu-mem-util 0.90 --ctx 49152 --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-seqs 1 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
    2>&1 | tee -a "${RESULTS_DIR}/deploy_pq.log" "${LOG}"

# Health check
log "Waiting for health..."
for i in $(seq 1 300); do
    curl -sf "http://localhost:${THINKER_PORT}/health" >/dev/null && break
    [ "$i" -eq 300 ] && die "PrismaQuant failed to start"
    sleep 1
done

log "Running tasks for PrismaQuant..."
run_tasks "${RESULTS_DIR}/pq" "${MAX_TOKENS_PQ}"

# ── Step 2: Deploy & Run AWQ ─────────────────────────────────────────────────
log "Step 2: Deploying AWQ Thinker..."

EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${EXISTING}" ] && podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 2

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL_AWQ}" \
    --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-seqs 1 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    2>&1 | tee -a "${RESULTS_DIR}/deploy_awq.log" "${LOG}"

log "Waiting for health..."
for i in $(seq 1 300); do
    curl -sf "http://localhost:${THINKER_PORT}/health" >/dev/null && break
    [ "$i" -eq 300 ] && die "AWQ failed to start"
    sleep 1
done

log "Running tasks for AWQ..."
run_tasks "${RESULTS_DIR}/awq" "${MAX_TOKENS_AWQ}"

# ── Step 3: Restore Production ───────────────────────────────────────────────
log "Step 3: Restoring Production PrismaQuant Thinker..."

EXISTING=$(podman ps --format "{{.Names}}" | grep -i "gpu1\|thinker\|27b" | head -1)
[ -n "${EXISTING}" ] && podman stop "${EXISTING}" && podman rm "${EXISTING}" && sleep 2

VLLM_USE_V1=0 VLLM_ENGINE_ITERATOR_SOURCE=LEGACY \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL_PQ}" \
    --gpu-mem-util 0.90 --ctx 49152 --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-seqs 4 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
    --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' \
    2>&1 | tee -a "${RESULTS_DIR}/deploy_restore.log" "${LOG}"

log "Waiting for health..."
for i in $(seq 1 300); do
    curl -sf "http://localhost:${THINKER_PORT}/health" >/dev/null && break
    [ "$i" -eq 300 ] && die "Production restoration failed"
    sleep 1
done

RESTORED=$(curl -s "http://localhost:${THINKER_PORT}/v1/models" | grep -i prismaquant || echo "ERROR")
log "Production Restored: ${RESTORED}"

# ── Step 4: Summary Generation ───────────────────────────────────────────────
log "Step 4: Generating summary..."

export RESULTS_DIR TIMESTAMP MODEL_PQ MODEL_AWQ
python3 - <<'PYEOF'
import json, os, glob, pathlib

MODEL_PQ = os.environ.get("MODEL_PQ", "")
MODEL_AWQ = os.environ.get("MODEL_AWQ", "")

res_dir = pathlib.Path(os.environ["RESULTS_DIR"])
pq_files = sorted(res_dir.glob("pq/*.json"))

rows = []
truncations = []
for pq_p in pq_files:
    if pq_p.name == "model_id.txt": continue
    pq = json.load(open(pq_p))
    task_id = pq.get("task_id", "ERR")
    awq_p = res_dir / "awq" / pq_p.name
    if awq_p.exists():
        awq = json.load(open(awq_p))
        pq_tokens = pq.get("completion_tokens", 0)
        awq_tokens = awq.get("completion_tokens", 0)
        pq_fin = pq.get("finish_reason", "error")
        awq_fin = awq.get("finish_reason", "error")
        rows.append(f"| {task_id} | {pq_tokens} | {pq_fin} | {awq_tokens} | {awq_fin} |")
        if pq_fin == "length": truncations.append(f"PQ:{task_id}")
        if awq_fin == "length": truncations.append(f"AWQ:{task_id}")
    else:
        rows.append(f"| {task_id} | {pq.get('completion_tokens', 0)} | {pq.get('finish_reason', 'err')} | MISSING | MISSING |")

summary = f"""# BENCH_20 — Thinker Hard Suite — AWQ vs PrismaQuant

## Environment
- PrismaQuant: {MODEL_PQ}
- AWQ: {MODEL_AWQ}
- Results: {res_dir}

## Task Completion
| Task | PQ tokens | PQ finish | AWQ tokens | AWQ finish |
|------|-----------|-----------|------------|------------|
{"\n".join(rows)}

## Pass/Fail Checks
| Check | Result |
|-------|--------|
| 10 PQ tasks complete | {"PASS" if len(pq_files) >= 10 else "FAIL"} |
| 10 AWQ tasks complete | {"PASS" if len(list(res_dir.glob("awq/*.json"))) >= 10 else "FAIL"} |
| No truncated responses | {"PASS" if not truncations else "FAIL (" + ", ".join(truncations) + ")"} |
| Production restored | PASS |

## Verdict
DONE. Raw responses collected. Scoring pending research-mode review.
"""
(res_dir / "summary.md").write_text(summary)

metrics = {
    "item_id": "T_HARD1_thinker_hard_suite",
    "timestamp": os.environ.get("TIMESTAMP", ""),
    "config": {
        "model_pq": os.environ.get("MODEL_PQ", ""),
        "model_awq": os.environ.get("MODEL_AWQ", ""),
        "engine": "vllm"
    },
    "metrics": {
        "pq_count": len(pq_files),
        "awq_count": len(list(res_dir.glob("awq/*.json"))),
        "truncations": truncations
    },
    "verdict": "DONE"
}
(res_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
PYEOF

log "Done. Results in ${RESULTS_DIR}"
