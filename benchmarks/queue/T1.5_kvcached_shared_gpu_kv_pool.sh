#!/usr/bin/env bash
# T1.5_kvcached_shared_gpu_kv_pool.sh
# SPIKE: Can kvcached enable two vLLM instances to share a single GPU's KV pool without collapse?
#
# Phase A — baseline sanity: single kvcached instance (coder then thinker), gpu0 only.
#           Pass: TPS within 5% of T2.5 / T1.2a baselines.
# Phase B — two instances, BOTH on gpu0, sharing KV via kvcached.
#           Pass: worst concurrent/isolated TPS ratio ≥ 0.80.
# Phase C — two instances, BOTH TP=2 (all GPUs) with kvcached. Stretch: original T1.2 failed topology.
#           Pass: any result above catastrophic collapse (combined TPS > 2% of isolated sum).
#
# kvcached image: ghcr.io/ovg-project/kvcached-vllm:latest (tagged kvcached-v0.1.5-vllm-v0.19.0)
# Integration: ENABLE_KVCACHED=true + KVCACHED_AUTOPATCH=1 env vars; no --gpu-memory-utilization.
# Ref: https://github.com/ovg-project/kvcached
#
# Hand-back triggers:
#   - Phase A: load failure (kvcached incompatibility with our stack)
#   - Phase B: PASS (validates shared-GPU topology — operator must decide architecture implications)
#   - Phase C: PASS (revives TP=2-for-both-hot — forces architecture rewrite, hand to research)
#   - Any OOM or crash not covered by documented failure modes
#
# Usage: ./benchmarks/queue/T1.5_kvcached_shared_gpu_kv_pool.sh
#   Requires: kvcached image pulled, models downloaded to MODEL_CACHE.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

ITEM_ID="T1.5_kvcached_shared_gpu_kv_pool"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
LOG="${RESULTS_DIR}/bench.log"
mkdir -p "${RESULTS_DIR}/raw"

KVCACHED_IMAGE="ghcr.io/ovg-project/kvcached-vllm:latest"
CODER_MODEL="QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"  # same as T1.2a for clean comparison
THINKER_MODEL="QuantTrio/Qwen3.5-27B-AWQ"
CTX=32768

# T1.2a isolated baselines (raw vLLM, no kvcached) — regression reference
BASELINE_CODER_TPS=251.0
BASELINE_THINKER_TPS=76.5

# Ports for Phase B (both on gpu0): reuse gpu0 + gpu1 port numbers,
# but both containers will be pinned to GPU_0_ID.
PORT_PHASEAB_CODER="${PORT_VLLM_GPU0}"    # 30000
PORT_PHASEAB_THINKER="${PORT_VLLM_GPU1}"  # 30001

# Ports for Phase C (both TP=2 on all GPUs): reuse tp2a + tp2b slots.
PORT_PHASEC_CODER="${PORT_VLLM_TP2_A}"    # 30000
PORT_PHASEC_THINKER="${PORT_VLLM_TP2_B}"  # 30001

log()  { echo "[T1.5 $(date -u +%H:%M:%S)] $*" | tee -a "${LOG}"; }
die()  { log "FATAL: $*"; exit 1; }

# ── Container helpers ─────────────────────────────────────────────────────────
stop_container() {
    local name="$1"
    podman stop "${name}" 2>/dev/null || true
    podman rm   "${name}" 2>/dev/null || true
}

wait_ready() {
    local url="$1" label="$2" cname="$3" timeout="${4:-600}"
    local deadline=$(( $(date +%s) + timeout ))
    log "Waiting for ${label} (${cname}) at ${url} (timeout ${timeout}s)..."
    while (( $(date +%s) < deadline )); do
        if curl -sf "${url%/v1}/health" >/dev/null 2>&1; then
            log "${label} is ready."
            return 0
        fi
        sleep 5
    done
    log "ERROR: ${label} did not become ready within ${timeout}s."
    log "--- container logs: ${cname} (last 80 lines) ---"
    podman logs --tail 80 "${cname}" 2>&1 | tee -a "${LOG}" || true
    log "--- end container logs ---"
    return 1
}

# Launch a single kvcached-vllm container.
# Usage: launch_kvcached <name> <port> <gpu_list> <tp> <model> [extra vllm args...]
# Note: no --gpu-memory-utilization — kvcached manages memory allocation.
launch_kvcached() {
    local cname="$1" port="$2" gpu_list="$3" tp="$4" model="$5"
    shift 5
    stop_container "${cname}"

    local tp_arg=()
    [[ "${tp}" -gt 1 ]] && tp_arg=(--tensor-parallel-size "${tp}")

    # Build CDI device args from comma-separated gpu_list
    local cdi_args=()
    IFS=',' read -ra _gpus <<< "${gpu_list}"
    for _g in "${_gpus[@]}"; do
        cdi_args+=(--device "nvidia.com/gpu=${_g}")
    done

    podman run -d \
        --name "${cname}" \
        "${cdi_args[@]}" \
        -e "NVIDIA_VISIBLE_DEVICES=${gpu_list}" \
        -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility" \
        -e "HF_HOME=/root/.cache/huggingface" \
        -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}" \
        -e "ENABLE_KVCACHED=true" \
        -e "KVCACHED_AUTOPATCH=1" \
        -p "${port}:8000" \
        --shm-size=4g \
        --restart=no \
        -v "${MODEL_CACHE}:/root/.cache/huggingface:z" \
        "${KVCACHED_IMAGE}" \
        python3 -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --port 8000 \
        --max-model-len "${CTX}" \
        --no-enable-prefix-caching \
        "${tp_arg[@]+"${tp_arg[@]}"}" \
        "$@" \
        2>&1 | tee -a "${LOG}"
}

# ── Measurement helper (inlined Python, same pattern as T1.2a) ────────────────
MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
"""
T1.5 measurement helper. Modes: isolated, concurrent, prefill_concurrent.
"""
import asyncio, httpx, json, sys, time

DECODE_PROMPT = (
    "Write a complete Python implementation of a red-black tree with insert, "
    "delete, search, and in-order traversal. Include full type hints."
)
MAX_DECODE_TOKENS = 1500


async def _measure_one(client, endpoint, model, messages, max_tokens):
    t0 = time.monotonic()
    fttt = None
    count = 0
    async with client.stream("POST", f"{endpoint}/chat/completions", json={
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "stream": True, "temperature": 0.0,
    }) as resp:
        resp.raise_for_status()
        async for raw in resp.aiter_lines():
            if not raw.startswith("data: ") or "[DONE]" in raw:
                continue
            delta = json.loads(raw[6:])["choices"][0]["delta"]
            tok = delta.get("content") or delta.get("reasoning") or ""
            if tok:
                if fttt is None:
                    fttt = time.monotonic() - t0
                count += 1
    total = time.monotonic() - t0
    decode_t = total - (fttt or 0)
    return {
        "ttft_ms":    round((fttt or 0) * 1000, 1),
        "decode_tps": round(count / decode_t, 1) if decode_t > 0 and count > 0 else 0.0,
        "tokens":     count,
    }


async def isolated(endpoint, model, out):
    async with httpx.AsyncClient(timeout=300) as c:
        r = await _measure_one(c, endpoint, model, [{"role": "user", "content": DECODE_PROMPT}], MAX_DECODE_TOKENS)
    json.dump(r, open(out, "w"))
    print(json.dumps(r))


async def concurrent(ep_a, model_a, ep_b, model_b, out_a, out_b):
    async with httpx.AsyncClient(timeout=300) as c:
        ra, rb = await asyncio.gather(
            _measure_one(c, ep_a, model_a, [{"role": "user", "content": DECODE_PROMPT}], MAX_DECODE_TOKENS),
            _measure_one(c, ep_b, model_b, [{"role": "user", "content": DECODE_PROMPT}], MAX_DECODE_TOKENS),
        )
    json.dump(ra, open(out_a, "w"))
    json.dump(rb, open(out_b, "w"))
    print(json.dumps({"a": ra, "b": rb}))


mode = sys.argv[1]
if mode == "isolated":
    asyncio.run(isolated(sys.argv[2], sys.argv[3], sys.argv[4]))
elif mode == "concurrent":
    asyncio.run(concurrent(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                           sys.argv[6], sys.argv[7]))
else:
    print(f"Unknown mode: {mode}", file=sys.stderr); sys.exit(1)
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
# PHASE A — baseline sanity: single kvcached instance, TP=1, gpu0 only
# Coder first, then thinker. Isolated TPS vs T1.2a / T2.5 baselines.
# ─────────────────────────────────────────────────────────────────────────────
log "=== PHASE A: baseline sanity (single kvcached instance, coder then thinker, gpu0) ==="

PHASE_A_CODER_PASS=false
PHASE_A_THINKER_PASS=false
TPS_A_CODER_ISO=0
TPS_A_THINKER_ISO=0

log "Deploying coder (${CODER_MODEL}) with kvcached on gpu0..."
launch_kvcached "bench-t15-coder" "${PORT_PHASEAB_CODER}" "${GPU_0_ID}" 1 "${CODER_MODEL}" \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice

EP_CODER="http://localhost:${PORT_PHASEAB_CODER}/v1"
if wait_ready "${EP_CODER}" "kvcached-coder" "bench-t15-coder"; then
    log "Warming up coder..."
    python3 "${MEASURE_PY}" isolated "${EP_CODER}" "${CODER_MODEL}" /dev/null 2>/dev/null || true

    log "Measuring coder isolated TPS (kvcached)..."
    python3 "${MEASURE_PY}" isolated "${EP_CODER}" "${CODER_MODEL}" \
        "${RESULTS_DIR}/raw/phaseA_coder_isolated.json" | tee -a "${LOG}"
    TPS_A_CODER_ISO=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/phaseA_coder_isolated.json'))['decode_tps'])")
    log "Phase A coder TPS: ${TPS_A_CODER_ISO} t/s (baseline ${BASELINE_CODER_TPS})"

    if python3 -c "exit(0 if float('${TPS_A_CODER_ISO}') >= ${BASELINE_CODER_TPS} * 0.95 else 1)"; then
        PHASE_A_CODER_PASS=true
        log "Phase A coder: PASS"
    else
        log "Phase A coder: FAIL — TPS below 95% of baseline. kvcached overhead is significant."
        log "HAND-BACK: kvcached may not be functional on our stack. Stopping spike."
        stop_container "bench-t15-coder"
        python3 -c "
import json, pathlib
pathlib.Path('${RESULTS_DIR}/metrics.json').write_text(json.dumps({
    'item_id': '${ITEM_ID}', 'timestamp': '${TIMESTAMP}',
    'config': {'engine': 'kvcached-vllm', 'image': '${KVCACHED_IMAGE}'},
    'metrics': {'phase_a_coder_tps': ${TPS_A_CODER_ISO}, 'baseline_tps': ${BASELINE_CODER_TPS}},
    'verdict': 'FAIL',
    'notes': 'Phase A coder TPS regression > 5%. kvcached incompatibility suspected. Hand back to research.'
}, indent=2))
pathlib.Path('${RESULTS_DIR}/summary.md').write_text(
    '# T1.5 kvcached Spike — FAIL\n\nPhase A coder TPS ${TPS_A_CODER_ISO} below 95% threshold of baseline ${BASELINE_CODER_TPS}.\n\nkvcached may be incompatible with our engine version or podman setup.\n\n**Hand back to research.**\n'
)
"
        exit 1
    fi
else
    log "HAND-BACK: kvcached container failed to start. Check container logs above."
    stop_container "bench-t15-coder"
    exit 1
fi
stop_container "bench-t15-coder"

log "Deploying thinker (${THINKER_MODEL}) with kvcached on gpu0..."
launch_kvcached "bench-t15-thinker" "${PORT_PHASEAB_THINKER}" "${GPU_0_ID}" 1 "${THINKER_MODEL}" \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-num-seqs 1

EP_THINKER="http://localhost:${PORT_PHASEAB_THINKER}/v1"
if wait_ready "${EP_THINKER}" "kvcached-thinker" "bench-t15-thinker"; then
    log "Warming up thinker..."
    python3 "${MEASURE_PY}" isolated "${EP_THINKER}" "${THINKER_MODEL}" /dev/null 2>/dev/null || true

    log "Measuring thinker isolated TPS (kvcached, gpu0)..."
    python3 "${MEASURE_PY}" isolated "${EP_THINKER}" "${THINKER_MODEL}" \
        "${RESULTS_DIR}/raw/phaseA_thinker_isolated.json" | tee -a "${LOG}"
    TPS_A_THINKER_ISO=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/phaseA_thinker_isolated.json'))['decode_tps'])")
    log "Phase A thinker TPS: ${TPS_A_THINKER_ISO} t/s (baseline ${BASELINE_THINKER_TPS})"

    if python3 -c "exit(0 if float('${TPS_A_THINKER_ISO}') >= ${BASELINE_THINKER_TPS} * 0.95 else 1)"; then
        PHASE_A_THINKER_PASS=true
        log "Phase A thinker: PASS"
    else
        log "Phase A thinker: WARN — TPS below 95% baseline. Noting but continuing to Phase B."
    fi
else
    log "WARNING: kvcached thinker container did not start. Skipping thinker isolation measurement."
fi
stop_container "bench-t15-thinker"

PHASE_A_OVERALL=$( "${PHASE_A_CODER_PASS}" && echo "PASS" || echo "FAIL" )
log "Phase A overall: ${PHASE_A_OVERALL} (coder=${PHASE_A_CODER_PASS}, thinker=${PHASE_A_THINKER_PASS})"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE B — two kvcached instances, BOTH on gpu0, sharing KV pool
# ─────────────────────────────────────────────────────────────────────────────
log "=== PHASE B: two kvcached instances on same GPU (gpu0) ==="
log "NOTE: Combined weight footprint (~18+14 = ~32 GiB) will saturate physical VRAM."
log "kvcached must manage this via virtual memory mapping for Phase B to succeed."

PHASE_B_PASS=false
TPS_B_CODER_ISO=0
TPS_B_THINKER_ISO=0
TPS_B_CODER_CON=0
TPS_B_THINKER_CON=0

log "Deploying coder + thinker on gpu0 simultaneously (kvcached)..."
launch_kvcached "bench-t15-coder" "${PORT_PHASEAB_CODER}" "${GPU_0_ID}" 1 "${CODER_MODEL}" \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice
launch_kvcached "bench-t15-thinker" "${PORT_PHASEAB_THINKER}" "${GPU_0_ID}" 1 "${THINKER_MODEL}" \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-num-seqs 1

PHASE_B_LOADED=true
if ! wait_ready "${EP_CODER}" "kvcached-coder(B)" "bench-t15-coder"; then
    log "Phase B: coder failed to start. Likely OOM or kvcached virtual memory failure."
    PHASE_B_LOADED=false
fi
if ! wait_ready "${EP_THINKER}" "kvcached-thinker(B)" "bench-t15-thinker"; then
    log "Phase B: thinker failed to start. Likely OOM or kvcached virtual memory failure."
    PHASE_B_LOADED=false
fi

if "${PHASE_B_LOADED}"; then
    log "Both instances loaded. Warming up..."
    python3 "${MEASURE_PY}" concurrent \
        "${EP_CODER}" "${CODER_MODEL}" \
        "${EP_THINKER}" "${THINKER_MODEL}" \
        /dev/null /dev/null 2>/dev/null || true

    log "Measuring Phase B isolated TPS (each endpoint, sequentially)..."
    python3 "${MEASURE_PY}" isolated "${EP_CODER}" "${CODER_MODEL}" \
        "${RESULTS_DIR}/raw/phaseB_coder_isolated.json" | tee -a "${LOG}"
    TPS_B_CODER_ISO=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/phaseB_coder_isolated.json'))['decode_tps'])")

    python3 "${MEASURE_PY}" isolated "${EP_THINKER}" "${THINKER_MODEL}" \
        "${RESULTS_DIR}/raw/phaseB_thinker_isolated.json" | tee -a "${LOG}"
    TPS_B_THINKER_ISO=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/phaseB_thinker_isolated.json'))['decode_tps'])")

    log "Phase B isolated TPS: coder=${TPS_B_CODER_ISO}  thinker=${TPS_B_THINKER_ISO}"

    log "Measuring Phase B concurrent TPS (both simultaneously)..."
    python3 "${MEASURE_PY}" concurrent \
        "${EP_CODER}" "${CODER_MODEL}" \
        "${EP_THINKER}" "${THINKER_MODEL}" \
        "${RESULTS_DIR}/raw/phaseB_coder_concurrent.json" \
        "${RESULTS_DIR}/raw/phaseB_thinker_concurrent.json" | tee -a "${LOG}"
    TPS_B_CODER_CON=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/phaseB_coder_concurrent.json'))['decode_tps'])")
    TPS_B_THINKER_CON=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/phaseB_thinker_concurrent.json'))['decode_tps'])")

    log "Phase B concurrent TPS: coder=${TPS_B_CODER_CON}  thinker=${TPS_B_THINKER_CON}"

    PHASE_B_PASS=$(python3 -c "
coder_iso  = float('${TPS_B_CODER_ISO}')
coder_con  = float('${TPS_B_CODER_CON}')
thinker_iso = float('${TPS_B_THINKER_ISO}')
thinker_con = float('${TPS_B_THINKER_CON}')
ratio_c = coder_con / coder_iso if coder_iso > 0 else 0
ratio_t = thinker_con / thinker_iso if thinker_iso > 0 else 0
worst = min(ratio_c, ratio_t)
print('true' if worst >= 0.80 else 'false')
")
    log "Phase B: ${PHASE_B_PASS} (worst concurrent/isolated ratio)"

    if [[ "${PHASE_B_PASS}" == "true" ]]; then
        log "HAND-BACK TRIGGER: Phase B PASS — kvcached enables shared-GPU topology."
        log "Operator must decide: adopt kvcached for production or stay with TP=1-per-GPU."
    fi
else
    log "Phase B: FAIL — one or both instances failed to load on same GPU."
    log "kvcached virtual memory mapping may not be sufficient for our combined weight footprint."
fi

stop_container "bench-t15-coder"
stop_container "bench-t15-thinker"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE C — stretch: two kvcached instances, both TP=2 on all GPUs
# Only run if Phase B passed (otherwise we know the fundamental issue is
# memory, not topology, so Phase C is moot).
# ─────────────────────────────────────────────────────────────────────────────
PHASE_C_VERDICT="SKIPPED"
TPS_C_CODER_CON=0
TPS_C_THINKER_CON=0

if [[ "${PHASE_B_PASS}" == "true" ]]; then
    log "=== PHASE C: two kvcached instances, BOTH TP=2 on all GPUs (original T1.2 failed topology) ==="

    GPU_BOTH="${GPU_0_ID},${GPU_1_ID}"
    EP_C_CODER="http://localhost:${PORT_PHASEC_CODER}/v1"
    EP_C_THINKER="http://localhost:${PORT_PHASEC_THINKER}/v1"

    launch_kvcached "bench-t15-c-coder" "${PORT_PHASEC_CODER}" "${GPU_BOTH}" 2 "${CODER_MODEL}" \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice
    launch_kvcached "bench-t15-c-thinker" "${PORT_PHASEC_THINKER}" "${GPU_BOTH}" 2 "${THINKER_MODEL}" \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --max-num-seqs 1

    PHASE_C_LOADED=true
    if ! wait_ready "${EP_C_CODER}" "kvcached-tp2-coder(C)" "bench-t15-c-coder"; then
        PHASE_C_LOADED=false
    fi
    if ! wait_ready "${EP_C_THINKER}" "kvcached-tp2-thinker(C)" "bench-t15-c-thinker"; then
        PHASE_C_LOADED=false
    fi

    if "${PHASE_C_LOADED}"; then
        log "Phase C: both TP=2 instances loaded. Running concurrent decode..."
        python3 "${MEASURE_PY}" concurrent \
            "${EP_C_CODER}" "${CODER_MODEL}" \
            "${EP_C_THINKER}" "${THINKER_MODEL}" \
            "${RESULTS_DIR}/raw/phaseC_coder_concurrent.json" \
            "${RESULTS_DIR}/raw/phaseC_thinker_concurrent.json" | tee -a "${LOG}"
        TPS_C_CODER_CON=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/phaseC_coder_concurrent.json'))['decode_tps'])")
        TPS_C_THINKER_CON=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/phaseC_thinker_concurrent.json'))['decode_tps'])")
        log "Phase C concurrent TPS: coder=${TPS_C_CODER_CON}  thinker=${TPS_C_THINKER_CON}"

        # Pass threshold: not a catastrophic collapse (> 2% of Phase A baselines combined)
        MIN_PASS_SUM=$(python3 -c "print(round((${BASELINE_CODER_TPS} + ${BASELINE_THINKER_TPS}) * 0.02, 1))")
        COMBINED_C=$(python3 -c "print(float('${TPS_C_CODER_CON}') + float('${TPS_C_THINKER_CON}'))")
        if python3 -c "exit(0 if float('${COMBINED_C}') > float('${MIN_PASS_SUM}') else 1)"; then
            PHASE_C_VERDICT="PASS"
            log "Phase C: PASS (combined ${COMBINED_C} t/s > ${MIN_PASS_SUM})"
            log "HAND-BACK TRIGGER: Phase C PASS — TP=2-for-both-hot revival possible. Architecture rewrite needed."
        else
            PHASE_C_VERDICT="FAIL"
            log "Phase C: FAIL — catastrophic collapse (${COMBINED_C} t/s ≤ ${MIN_PASS_SUM}). CUDA scheduling is the bottleneck, not KV memory."
        fi
    else
        PHASE_C_VERDICT="FAIL (load failure)"
        log "Phase C: FAIL — one or both TP=2 instances could not start."
    fi

    stop_container "bench-t15-c-coder"
    stop_container "bench-t15-c-thinker"
else
    log "=== PHASE C: SKIPPED (Phase B failed — memory sharing insufficient, topology change irrelevant) ==="
fi

# ─────────────────────────────────────────────────────────────────────────────
# Results — write metrics.json and summary.md
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<PYEOF
import json, pathlib

results_dir = pathlib.Path("${RESULTS_DIR}")

phase_a_coder_pass   = "${PHASE_A_CODER_PASS}" == "true"
phase_a_thinker_pass = "${PHASE_A_THINKER_PASS}" == "true"
phase_b_loaded       = "${PHASE_B_LOADED}" == "true"
phase_b_pass         = "${PHASE_B_PASS}" == "true"
phase_c_verdict      = "${PHASE_C_VERDICT}"

tps_a_coder_iso   = float("${TPS_A_CODER_ISO}")
tps_a_thinker_iso = float("${TPS_A_THINKER_ISO}")
tps_b_coder_iso   = float("${TPS_B_CODER_ISO}")
tps_b_thinker_iso = float("${TPS_B_THINKER_ISO}")
tps_b_coder_con   = float("${TPS_B_CODER_CON}")
tps_b_thinker_con = float("${TPS_B_THINKER_CON}")
tps_c_coder_con   = float("${TPS_C_CODER_CON}")
tps_c_thinker_con = float("${TPS_C_THINKER_CON}")
baseline_coder    = float("${BASELINE_CODER_TPS}")
baseline_thinker  = float("${BASELINE_THINKER_TPS}")

ratio_b_coder   = tps_b_coder_con / tps_b_coder_iso if tps_b_coder_iso > 0 else 0
ratio_b_thinker = tps_b_thinker_con / tps_b_thinker_iso if tps_b_thinker_iso > 0 else 0
kvcached_overhead_coder   = round((1 - tps_a_coder_iso / baseline_coder) * 100, 1) if baseline_coder > 0 else 0
kvcached_overhead_thinker = round((1 - tps_a_thinker_iso / baseline_thinker) * 100, 1) if baseline_thinker > 0 else 0

# Overall verdict
if not phase_a_coder_pass:
    verdict = "FAIL"
    verdict_reason = "Phase A coder TPS regression > 5%. kvcached incompatible with our stack."
elif not phase_b_loaded:
    verdict = "FAIL"
    verdict_reason = "Phase B: instances failed to load on same GPU. OOM or virtual memory exhaustion."
elif not phase_b_pass:
    verdict = "FAIL"
    verdict_reason = f"Phase B: concurrent TPS ratio below 0.80. Shared-GPU topology not viable."
else:
    verdict = "PASS"
    verdict_reason = f"Phase B PASS: shared-GPU topology viable. Phase C: {phase_c_verdict}."

metrics = {
    "item_id":   "${ITEM_ID}",
    "timestamp": "${TIMESTAMP}",
    "config": {
        "engine":        "kvcached-vllm",
        "engine_image":  "${KVCACHED_IMAGE}",
        "coder_model":   "${CODER_MODEL}",
        "thinker_model": "${THINKER_MODEL}",
        "context_length": ${CTX},
        "kvcached_env":  "ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1",
        "note":          "No --gpu-memory-utilization; kvcached manages allocation.",
    },
    "metrics": {
        "phase_a": {
            "coder_tps_isolated":           tps_a_coder_iso,
            "thinker_tps_isolated":         tps_a_thinker_iso,
            "coder_kvcached_overhead_pct":  kvcached_overhead_coder,
            "thinker_kvcached_overhead_pct": kvcached_overhead_thinker,
            "pass": phase_a_coder_pass,
        },
        "phase_b": {
            "loaded":                       phase_b_loaded,
            "coder_tps_isolated":           tps_b_coder_iso,
            "thinker_tps_isolated":         tps_b_thinker_iso,
            "coder_tps_concurrent":         tps_b_coder_con,
            "thinker_tps_concurrent":       tps_b_thinker_con,
            "concurrent_iso_ratio_coder":   round(ratio_b_coder, 3),
            "concurrent_iso_ratio_thinker": round(ratio_b_thinker, 3),
            "pass":                         phase_b_pass,
        },
        "phase_c": {
            "verdict":           phase_c_verdict,
            "coder_tps_con":     tps_c_coder_con,
            "thinker_tps_con":   tps_c_thinker_con,
        },
    },
    "verdict": verdict,
    "notes":   verdict_reason,
}

(results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

def r(v, good, incon, hi=True):
    return "PASS" if (v >= good if hi else v <= good) else ("INCON" if (v >= incon if hi else v <= incon) else "FAIL")

summary = f"""# T1.5 kvcached Shared-GPU KV Pool Spike — {verdict}

**Image** {metrics['config']['engine_image']} | **Date** ${TIMESTAMP:0:10}
**Coder** ${CODER_MODEL} | **Thinker** ${THINKER_MODEL}

## Phase A — baseline sanity (single kvcached instance, gpu0)

| Metric | Measured | Baseline (raw vLLM) | kvcached overhead | Result |
|--------|----------|---------------------|-------------------|--------|
| Coder TPS (isolated) | {tps_a_coder_iso:.1f} t/s | {baseline_coder:.1f} t/s | {kvcached_overhead_coder:.1f}% | {r(tps_a_coder_iso, baseline_coder * 0.95, baseline_coder * 0.80)} |
| Thinker TPS (isolated) | {tps_a_thinker_iso:.1f} t/s | {baseline_thinker:.1f} t/s | {kvcached_overhead_thinker:.1f}% | {r(tps_a_thinker_iso, baseline_thinker * 0.95, baseline_thinker * 0.80)} |

## Phase B — two instances on gpu0, shared KV pool

| Metric | Measured | Pass threshold | Result |
|--------|----------|----------------|--------|
| Instances loaded on same GPU | {"YES" if phase_b_loaded else "NO"} | YES | {"PASS" if phase_b_loaded else "FAIL"} |
| Coder concurrent/isolated ratio | {ratio_b_coder:.3f} ({tps_b_coder_iso:.1f}→{tps_b_coder_con:.1f} t/s) | ≥0.80 | {r(ratio_b_coder, 0.80, 0.60)} |
| Thinker concurrent/isolated ratio | {ratio_b_thinker:.3f} ({tps_b_thinker_iso:.1f}→{tps_b_thinker_con:.1f} t/s) | ≥0.80 | {r(ratio_b_thinker, 0.80, 0.60)} |

## Phase C — both TP=2 on all GPUs (stretch)

Verdict: **{phase_c_verdict}**
{"Coder concurrent TPS: " + str(tps_c_coder_con) + " t/s  |  Thinker concurrent TPS: " + str(tps_c_thinker_con) + " t/s" if phase_c_verdict != "SKIPPED" else "Skipped — Phase B did not pass."}

---

## Overall verdict: {verdict}

{verdict_reason}

{"### HAND-BACK REQUIRED: Phase B PASS — operator must decide on kvcached adoption for production." if phase_b_pass else ""}
{"### HAND-BACK REQUIRED: Phase C PASS — TP=2-for-both-hot is revived. Architecture rewrite needed." if phase_c_verdict == "PASS" else ""}
"""

(results_dir / "summary.md").write_text(summary)
print(f"[T1.5] Verdict: {verdict}")
print(f"[T1.5] Results: ${RESULTS_DIR}/")
PYEOF

log "T1.5 spike complete. Results: ${RESULTS_DIR}/"
