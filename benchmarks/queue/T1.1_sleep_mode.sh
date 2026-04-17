#!/usr/bin/env bash
# T1.1_sleep_mode.sh — verify vLLM Sleep Mode end-to-end under rootless podman
# Tests: VRAM freed ≥80% on sleep, wake latency ≤10s, post-wake TPS ≥95% of pre-sleep.
#
# Verified against vLLM 0.19.0 on 2026-04-17
# Sleep/wake endpoints: POST /sleep?level=1  POST /wake_up
#   Requires BOTH:
#     - env VLLM_SERVER_DEV_MODE=1           (exposes /sleep, /wake_up, /is_sleeping routes)
#     - serve flag --enable-sleep-mode       (initializes engine with CuMemAllocator)
#   Without the serve flag, /sleep is a control-plane no-op (no memory is released).
#   See DECISIONS.md "vLLM Sleep Mode" entry.
#
# Level 1 ONLY. Do not change to level=2 — bug #29341 produces gibberish on wake.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

ITEM_ID="T1.1_sleep_mode_operational_under_podman"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
CONTAINER_NAME="bench-vllm-tp2a"
PORT="${PORT_VLLM_TP2_A}"
BASE_URL="http://localhost:${PORT}"
ENDPOINT="${BASE_URL}/v1"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log()  { echo "[T1.1] $*" | tee -a "${LOG}"; }
die()  { log "FATAL: $*"; exit 1; }

vram_total_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i "${GPU_0_ID},${GPU_1_ID}" \
        | awk '$1 ~ /^[0-9]+$/ { s += $1 } END { print s+0 }'
}

# Write the TPS measurement script to a temp file; reused pre- and post-wake.
MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
"""
Measure decode TPS via a streaming chat completion.
Usage: python measure_tps.py <endpoint> <out_json> <model_name>
"""
import asyncio, httpx, json, sys, time

PROMPT = (
    "Write a complete Python implementation of a red-black tree with insert, "
    "delete, search, and in-order traversal. Include full type hints."
)
MAX_TOKENS = 1500

async def measure(endpoint: str) -> dict:
    t0 = time.monotonic()
    first_token_t: float | None = None
    token_count = 0

    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream("POST", f"{endpoint}/chat/completions", json={
            "model": sys.argv[3],
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "temperature": 0.0,
        }) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_lines():
                if not raw.startswith("data: ") or "[DONE]" in raw:
                    continue
                delta = json.loads(raw[6:])["choices"][0]["delta"]
                # Count both answer tokens and reasoning tokens (--reasoning-parser qwen3
                # routes <think> content to delta.reasoning, not delta.reasoning_content).
                tok = delta.get("content") or delta.get("reasoning") or ""
                if tok:
                    if first_token_t is None:
                        first_token_t = time.monotonic() - t0
                    token_count += 1

    total = time.monotonic() - t0
    decode_t = total - (first_token_t or 0)
    return {
        "ttft_ms":    round((first_token_t or 0) * 1000, 1),
        "decode_tps": round(token_count / decode_t, 1) if decode_t > 0 else 0,
        "tokens":     token_count,
    }

result = asyncio.run(measure(sys.argv[1]))
out_path = sys.argv[2]
with open(out_path, "w") as fh:
    json.dump(result, fh)
print(json.dumps(result))
PYEOF

# ── Step 1: Deploy with Sleep Mode enabled ────────────────────────────────────
#
# Both VLLM_SERVER_DEV_MODE=1 (env) AND --enable-sleep-mode (serve flag) are required.
# The env var alone (as in the 2026-04-17 attempt) exposes the /sleep route but the
# engine still uses the default allocator, so /sleep becomes a no-op. See R5 cycle
# in RESEARCH_STATE.md.
log "Deploying ${MODEL} TP=2 gpu-mem=0.85 VLLM_SERVER_DEV_MODE=1 --enable-sleep-mode ..."
VLLM_SERVER_DEV_MODE=1 \
    "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" \
    --gpu-mem-util 0.85 \
    --enable-sleep-mode \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    2>&1 | tee -a "${LOG}"

# ── Step 2: Pre-sleep baseline ────────────────────────────────────────────────
log "Recording pre-sleep VRAM ..."
VRAM_LOADED=$(vram_total_mib)
log "Pre-sleep VRAM: ${VRAM_LOADED} MiB (GPU0+GPU1)"

log "Measuring pre-sleep decode TPS ..."
python3 "${MEASURE_PY}" "${ENDPOINT}" "${RESULTS_DIR}/raw/presleep.json" "${MODEL}" \
    | tee -a "${LOG}"
PRESLEEP_TPS=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/raw/presleep.json')); print(d['decode_tps'])")
log "Pre-sleep TPS: ${PRESLEEP_TPS}"

# ── Step 3: Sleep ─────────────────────────────────────────────────────────────
log "Sending POST ${BASE_URL}/sleep?level=1 ..."
SLEEP_T0_MS=$(date +%s%3N)
HTTP_STATUS=$(curl -s -o "${RESULTS_DIR}/raw/sleep_response.json" \
    -w "%{http_code}" -X POST "${BASE_URL}/sleep?level=1")
SLEEP_LATENCY_MS=$(( $(date +%s%3N) - SLEEP_T0_MS ))
log "Sleep HTTP status: ${HTTP_STATUS}  latency: ${SLEEP_LATENCY_MS}ms"
[[ "${HTTP_STATUS}" == "200" ]] || die "/sleep returned HTTP ${HTTP_STATUS} — check VLLM_SERVER_DEV_MODE"

sleep 2  # give CUDA allocator a moment to propagate the release

# ── Step 4: Post-sleep VRAM ───────────────────────────────────────────────────
VRAM_ASLEEP=$(vram_total_mib)
log "Post-sleep VRAM: ${VRAM_ASLEEP} MiB (freed $(( VRAM_LOADED - VRAM_ASLEEP )) MiB)"

# ── Step 5: Wake ──────────────────────────────────────────────────────────────
log "Sending POST ${BASE_URL}/wake_up ..."
WAKE_T0_MS=$(date +%s%3N)
HTTP_STATUS=$(curl -s -o "${RESULTS_DIR}/raw/wake_response.json" \
    -w "%{http_code}" -X POST "${BASE_URL}/wake_up")
WAKE_LATENCY_MS=$(( $(date +%s%3N) - WAKE_T0_MS ))
log "Wake HTTP status: ${HTTP_STATUS}  latency: ${WAKE_LATENCY_MS}ms"
[[ "${HTTP_STATUS}" == "200" ]] || die "/wake_up returned HTTP ${HTTP_STATUS}"

# Confirm the endpoint is serving again (wake_up should be synchronous, but guard anyway)
"${REPO_ROOT}/infra/scripts/wait-healthy.sh" "${BASE_URL}/health" 60 "${CONTAINER_NAME}"

# ── Step 6: Post-wake TPS ─────────────────────────────────────────────────────
log "Measuring post-wake decode TPS ..."
python3 "${MEASURE_PY}" "${ENDPOINT}" "${RESULTS_DIR}/raw/postwake.json" "${MODEL}" \
    | tee -a "${LOG}"
POSTWAKE_TPS=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/raw/postwake.json')); print(d['decode_tps'])")
log "Post-wake TPS: ${POSTWAKE_TPS}"

# ── Step 7: Compute metrics, apply thresholds, write results ──────────────────
python3 - <<PYEOF
import json, pathlib

item_id   = "${ITEM_ID}"
timestamp = "${TIMESTAMP}"
vram_loaded  = float("${VRAM_LOADED}")
vram_asleep  = float("${VRAM_ASLEEP}")
wake_s       = float("${WAKE_LATENCY_MS}") / 1000
pre_tps      = float("${PRESLEEP_TPS}")
post_tps     = float("${POSTWAKE_TPS}")
out          = pathlib.Path("${RESULTS_DIR}")

freed = (vram_loaded - vram_asleep) / vram_loaded if vram_loaded else 0
tps_ratio = post_tps / pre_tps if pre_tps else 0

# Thresholds from config/thresholds.yaml T1.1
PASS  = dict(freed=0.80, wake=10,  tps=0.95)
INCON = dict(freed=0.60, wake=20,  tps=0.85)

if freed >= PASS["freed"] and wake_s <= PASS["wake"] and tps_ratio >= PASS["tps"]:
    verdict = "PASS"
elif freed >= INCON["freed"] and wake_s <= INCON["wake"] and tps_ratio >= INCON["tps"]:
    verdict = "INCONCLUSIVE"
else:
    verdict = "FAIL"

metrics = {
    "item_id":   item_id,
    "timestamp": timestamp,
    "config": {
        "engine":          "vllm",
        "engine_version":  "0.19.0",
        "model":           "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ",
        "quantization":    "AWQ-INT4",
        "kv_cache_dtype":  "auto",
        "placement":       "tp=2",
        "context_length":  32768,
        "extra_args":      "--enable-sleep-mode --tool-call-parser qwen3_coder --reasoning-parser qwen3 VLLM_SERVER_DEV_MODE=1",
    },
    "metrics": {
        "vram_loaded_mib":       vram_loaded,
        "vram_asleep_mib":       vram_asleep,
        "freed_vram_fraction":   round(freed,     4),
        "wake_latency_s":        round(wake_s,    2),
        "presleep_decode_tps":   pre_tps,
        "postwake_decode_tps":   post_tps,
        "post_wake_tps_ratio":   round(tps_ratio, 4),
    },
    "verdict": verdict,
    "notes": "",
}

(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

def sym(val, pass_thr, incon_thr, higher_is_better=True):
    if higher_is_better:
        return "PASS" if val >= pass_thr else ("INCON" if val >= incon_thr else "FAIL")
    else:
        return "PASS" if val <= pass_thr else ("INCON" if val <= incon_thr else "FAIL")

summary = f"""# T1.1 Sleep Mode — {verdict}

**vLLM** 0.19.0 | **Model** Qwen3-Coder-30B-A3B-AWQ | **Placement** TP=2 | **Date** {timestamp[:10]}

| Metric | Measured | Pass / Incon threshold | Result |
|--------|----------|------------------------|--------|
| VRAM freed on sleep | {freed:.1%}  ({vram_loaded:.0f} → {vram_asleep:.0f} MiB) | ≥80% / ≥60% | {sym(freed, 0.80, 0.60)} |
| Wake latency        | {wake_s:.1f}s | ≤10s / ≤20s | {sym(wake_s, 10, 20, higher_is_better=False)} |
| Post-wake TPS ratio | {tps_ratio:.3f}  ({pre_tps:.1f} → {post_tps:.1f} t/s) | ≥0.95 / ≥0.85 | {sym(tps_ratio, 0.95, 0.85)} |

**Verdict: {verdict}**

Container left running for T1.2. Run \`just teardown\` when done.
"""
(out / "summary.md").write_text(summary)

print(f"[T1.1] Verdict: {verdict}")
print(f"[T1.1] VRAM freed: {freed:.1%}  ({vram_loaded:.0f} → {vram_asleep:.0f} MiB)")
print(f"[T1.1] Wake latency: {wake_s:.1f}s")
print(f"[T1.1] TPS ratio: {tps_ratio:.3f}  ({pre_tps:.1f} → {post_tps:.1f} t/s)")
print(f"[T1.1] Results: ${RESULTS_DIR}/")
PYEOF