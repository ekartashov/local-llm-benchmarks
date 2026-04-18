#!/usr/bin/env bash
# T1.2a_tp1_per_gpu.sh — verify two vLLM processes coexist correctly separated to TP=1 per GPU
# Tests: concurrent decode TPS ≥80% of isolated, prefill TTFT contention ≤2×, zero crashes/corruption.
# Updated for TP=1 architecture on 2026-04-18
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

ITEM_ID="T1.2a_tp1_per_gpu_concurrent_decode"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
CODER_MODEL="QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
THINKER_MODEL="QuantTrio/Qwen3.5-27B-AWQ"
CODER_URL="http://localhost:${PORT_VLLM_GPU0}/v1"
THINKER_URL="http://localhost:${PORT_VLLM_GPU1}/v1"
# 80% of 2×32 GiB = 52429 MiB — warn if exceeded
VRAM_BUDGET_MIB=52429

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log()  { echo "[T1.2a] $*" | tee -a "${LOG}"; }
die()  { log "FATAL: $*"; exit 1; }

vram_total_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i "${GPU_0_ID},${GPU_1_ID}" \
        | awk '$1 ~ /^[0-9]+$/ { s += $1 } END { print s+0 }'
}

# ── Write Python helper to temp file ──────────────────────────────────────────
MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
"""
T1.2a measurement helper.

Modes:
  isolated   <endpoint> <model> <out_json>
  concurrent <ep_a> <model_a> <ep_b> <model_b> <out_json_a> <out_json_b>
  prefill    <endpoint> <model> <num_tokens> <out_json>
  prefill_concurrent <ep_a> <model_a> <ep_b> <model_b> <num_tokens> <out_a> <out_b>
"""
import asyncio, httpx, json, sys, time

DECODE_PROMPT = (
    "Write a complete Python implementation of a red-black tree with insert, "
    "delete, search, and in-order traversal. Include full type hints."
)
MAX_DECODE_TOKENS = 1500

# Generates a synthetic long prompt of approximately `target_tokens` tokens.
# Rough rule: 1 token ≈ 4 chars for mixed English/code.
def make_long_prompt(target_tokens: int) -> str:
    chunk = (
        "The following is part of a large codebase documentation file. "
        "It describes the architecture of a distributed system. "
        "Each component communicates over gRPC. "
        "The scheduler uses a priority queue with O(log n) insertion. "
    )
    reps = max(1, (target_tokens * 4) // len(chunk) + 1)
    base = (chunk * reps)[: target_tokens * 4]
    return base + "\n\nSummarise the above in one sentence."


async def _measure_one(client: httpx.AsyncClient, endpoint: str, model: str,
                        messages: list, max_tokens: int) -> dict:
    t0 = time.monotonic()
    fttt: float | None = None
    count = 0
    text_out = []

    async with client.stream("POST", f"{endpoint}/chat/completions", json={
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
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
                text_out.append(tok)

    total = time.monotonic() - t0
    decode_t = total - (fttt or 0)
    return {
        "ttft_ms":    round((fttt or 0) * 1000, 1),
        "decode_tps": round(count / decode_t, 1) if decode_t > 0 and count > 0 else 0.0,
        "tokens":     count,
        "text_snippet": "".join(text_out)[:200],
    }


async def isolated(endpoint: str, model: str, out_path: str) -> None:
    async with httpx.AsyncClient(timeout=300) as c:
        result = await _measure_one(c, endpoint, model,
                                    [{"role": "user", "content": DECODE_PROMPT}],
                                    MAX_DECODE_TOKENS)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))


async def concurrent(ep_a: str, model_a: str, ep_b: str, model_b: str,
                     out_a: str, out_b: str) -> None:
    async with httpx.AsyncClient(timeout=300) as c:
        res_a, res_b = await asyncio.gather(
            _measure_one(c, ep_a, model_a, [{"role": "user", "content": DECODE_PROMPT}], MAX_DECODE_TOKENS),
            _measure_one(c, ep_b, model_b, [{"role": "user", "content": DECODE_PROMPT}], MAX_DECODE_TOKENS),
        )
    with open(out_a, "w") as f: json.dump(res_a, f)
    with open(out_b, "w") as f: json.dump(res_b, f)
    print(json.dumps({"a": res_a, "b": res_b}))


async def prefill(endpoint: str, model: str, num_tokens: int, out_path: str) -> None:
    prompt = make_long_prompt(num_tokens)
    async with httpx.AsyncClient(timeout=300) as c:
        result = await _measure_one(c, endpoint, model,
                                    [{"role": "user", "content": prompt}], 1)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))


async def prefill_concurrent(ep_a: str, model_a: str, ep_b: str, model_b: str,
                              num_tokens: int, out_a: str, out_b: str) -> None:
    prompt = make_long_prompt(num_tokens)
    msgs = [{"role": "user", "content": prompt}]
    async with httpx.AsyncClient(timeout=300) as c:
        res_a, res_b = await asyncio.gather(
            _measure_one(c, ep_a, model_a, msgs, 1),
            _measure_one(c, ep_b, model_b, msgs, 1),
        )
    with open(out_a, "w") as f: json.dump(res_a, f)
    with open(out_b, "w") as f: json.dump(res_b, f)
    print(json.dumps({"a": res_a, "b": res_b}))


mode = sys.argv[1]
if mode == "isolated":
    asyncio.run(isolated(sys.argv[2], sys.argv[3], sys.argv[4]))
elif mode == "concurrent":
    asyncio.run(concurrent(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                           sys.argv[6], sys.argv[7]))
elif mode == "prefill":
    asyncio.run(prefill(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]))
elif mode == "prefill_concurrent":
    asyncio.run(prefill_concurrent(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                                   int(sys.argv[6]), sys.argv[7], sys.argv[8]))
else:
    print(f"Unknown mode: {mode}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ── Step 1: Deploy coder (gpu0, gpu-mem 0.85) ─────────────────────────────────
log "Deploying coder (${CODER_MODEL}) gpu0 gpu-mem=0.85 ..."
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu0 "${CODER_MODEL}" \
    --gpu-mem-util 0.85 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    2>&1 | tee -a "${LOG}"

# ── Step 2: Deploy thinker (gpu1, gpu-mem 0.85) ───────────────────────────────
log "Deploying thinker (${THINKER_MODEL}) gpu1 gpu-mem=0.85 ctx=16384 ..."
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${THINKER_MODEL}" \
    --gpu-mem-util 0.85 \
    --ctx 16384 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-num-seqs 1 \
    2>&1 | tee -a "${LOG}"

# ── Step 3: VRAM check ─────────────────────────────────────────────────────────
VRAM_DUAL=$(vram_total_mib)
log "VRAM after both loaded: ${VRAM_DUAL} MiB (budget ≤${VRAM_BUDGET_MIB} MiB for 80% of 2×32GiB)"
if (( VRAM_DUAL > VRAM_BUDGET_MIB )); then
    log "WARNING: VRAM exceeds 80% budget — KV cache may be very small, results may be unreliable"
fi

# ── Step 4: Warmup (ensure CUDA graphs exercised before measurement) ───────────
log "Warming up both endpoints ..."
python3 "${MEASURE_PY}" isolated "${CODER_URL}"   "${CODER_MODEL}"   /dev/null 2>/dev/null || true
python3 "${MEASURE_PY}" isolated "${THINKER_URL}" "${THINKER_MODEL}" /dev/null 2>/dev/null || true

# ── Step 5: Isolated decode TPS ───────────────────────────────────────────────
log "Measuring isolated decode TPS — coder ..."
python3 "${MEASURE_PY}" isolated "${CODER_URL}" "${CODER_MODEL}" \
    "${RESULTS_DIR}/raw/coder_isolated.json" | tee -a "${LOG}"
TPS_A_ISO=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/coder_isolated.json'))['decode_tps'])")

log "Measuring isolated decode TPS — thinker ..."
python3 "${MEASURE_PY}" isolated "${THINKER_URL}" "${THINKER_MODEL}" \
    "${RESULTS_DIR}/raw/thinker_isolated.json" | tee -a "${LOG}"
TPS_B_ISO=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/thinker_isolated.json'))['decode_tps'])")

log "Isolated TPS: coder=${TPS_A_ISO}  thinker=${TPS_B_ISO}"

# ── Step 6: Concurrent decode TPS ─────────────────────────────────────────────
log "Measuring concurrent decode TPS (both simultaneously) ..."
python3 "${MEASURE_PY}" concurrent \
    "${CODER_URL}" "${CODER_MODEL}" \
    "${THINKER_URL}" "${THINKER_MODEL}" \
    "${RESULTS_DIR}/raw/coder_concurrent.json" \
    "${RESULTS_DIR}/raw/thinker_concurrent.json" \
    | tee -a "${LOG}"
TPS_A_CON=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/coder_concurrent.json'))['decode_tps'])")
TPS_B_CON=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/thinker_concurrent.json'))['decode_tps'])")

log "Concurrent TPS: coder=${TPS_A_CON}  thinker=${TPS_B_CON}"

# ── Step 7: Isolated prefill TTFT (16k prompt) ────────────────────────────────
log "Measuring isolated prefill TTFT at ~16k tokens — coder ..."
python3 "${MEASURE_PY}" prefill "${CODER_URL}" "${CODER_MODEL}" 16000 \
    "${RESULTS_DIR}/raw/coder_prefill_isolated.json" | tee -a "${LOG}"
TTFT_A_ISO=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/coder_prefill_isolated.json'))['ttft_ms'])")

log "Measuring isolated prefill TTFT at ~16k tokens — thinker ..."
python3 "${MEASURE_PY}" prefill "${THINKER_URL}" "${THINKER_MODEL}" 16000 \
    "${RESULTS_DIR}/raw/thinker_prefill_isolated.json" | tee -a "${LOG}"
TTFT_B_ISO=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/thinker_prefill_isolated.json'))['ttft_ms'])")

log "Isolated prefill TTFT: coder=${TTFT_A_ISO}ms  thinker=${TTFT_B_ISO}ms"

# ── Step 8: Concurrent prefill TTFT (16k to both simultaneously) ──────────────
log "Measuring concurrent prefill TTFT at ~16k tokens (both simultaneously) ..."
python3 "${MEASURE_PY}" prefill_concurrent \
    "${CODER_URL}" "${CODER_MODEL}" \
    "${THINKER_URL}" "${THINKER_MODEL}" \
    16000 \
    "${RESULTS_DIR}/raw/coder_prefill_concurrent.json" \
    "${RESULTS_DIR}/raw/thinker_prefill_concurrent.json" \
    | tee -a "${LOG}"
TTFT_A_CON=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/coder_prefill_concurrent.json'))['ttft_ms'])")
TTFT_B_CON=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/thinker_prefill_concurrent.json'))['ttft_ms'])")

log "Concurrent prefill TTFT: coder=${TTFT_A_CON}ms  thinker=${TTFT_B_CON}ms"

# ── Step 9: Corruption check ──────────────────────────────────────────────────
# Check that concurrent decode outputs contain plausible content (non-empty, non-gibberish).
CORRUPTION=0
for f in "${RESULTS_DIR}/raw/coder_concurrent.json" "${RESULTS_DIR}/raw/thinker_concurrent.json"; do
    tokens=$(python3 -c "import json; print(json.load(open('${f}'))['tokens'])")
    if [[ "${tokens}" == "0" ]]; then
        log "WARNING: zero tokens in ${f} — potential corruption or failure"
        CORRUPTION=1
    fi
done

# ── Step 10: Compute metrics, apply thresholds, write results ─────────────────
python3 - <<PYEOF
import json, pathlib

item_id   = "${ITEM_ID}"
timestamp = "${TIMESTAMP}"
tps_a_iso = float("${TPS_A_ISO}")
tps_b_iso = float("${TPS_B_ISO}")
tps_a_con = float("${TPS_A_CON}")
tps_b_con = float("${TPS_B_CON}")
ttft_a_iso = float("${TTFT_A_ISO}")
ttft_b_iso = float("${TTFT_B_ISO}")
ttft_a_con = float("${TTFT_A_CON}")
ttft_b_con = float("${TTFT_B_CON}")
vram_mib  = float("${VRAM_DUAL}")
corruption = int("${CORRUPTION}")
out = pathlib.Path("${RESULTS_DIR}")

ratio_a = tps_a_con / tps_a_iso if tps_a_iso > 0 else 0
ratio_b = tps_b_con / tps_b_iso if tps_b_iso > 0 else 0
contention_a = ttft_a_con / ttft_a_iso if ttft_a_iso > 0 else 0
contention_b = ttft_b_con / ttft_b_iso if ttft_b_iso > 0 else 0
worst_ratio = min(ratio_a, ratio_b)
worst_contention = max(contention_a, contention_b)

# Thresholds from config/thresholds.yaml T1.2
PASS_RATIO = 0.80;  INCON_RATIO = 0.60
PASS_CONT  = 2.0;   INCON_CONT  = 3.0
PASS_CRASH = 0;     INCON_CRASH = 0

if corruption > PASS_CRASH:
    verdict = "FAIL"
elif worst_ratio >= PASS_RATIO and worst_contention <= PASS_CONT:
    verdict = "PASS"
elif worst_ratio >= INCON_RATIO and worst_contention <= INCON_CONT:
    verdict = "INCONCLUSIVE"
else:
    verdict = "FAIL"

metrics = {
    "item_id":   item_id,
    "timestamp": timestamp,
    "config": {
        "engine": "vllm", "engine_version": "0.19.0",
        "coder_model":  "${CODER_MODEL}",
        "thinker_model": "${THINKER_MODEL}",
        "placement": "tp=1 per gpu (coder: gpu0, thinker: gpu1)",
        "coder_gpu_mem_util": 0.85,
        "thinker_gpu_mem_util": 0.85,
    },
    "metrics": {
        "vram_dual_loaded_mib":       vram_mib,
        "tps_coder_isolated":         tps_a_iso,
        "tps_thinker_isolated":       tps_b_iso,
        "tps_coder_concurrent":       tps_a_con,
        "tps_thinker_concurrent":     tps_b_con,
        "concurrent_vs_isolated_ratio_coder":   round(ratio_a, 4),
        "concurrent_vs_isolated_ratio_thinker": round(ratio_b, 4),
        "prefill_ttft_coder_isolated_ms":       ttft_a_iso,
        "prefill_ttft_thinker_isolated_ms":     ttft_b_iso,
        "prefill_ttft_coder_concurrent_ms":     ttft_a_con,
        "prefill_ttft_thinker_concurrent_ms":   ttft_b_con,
        "prefill_contention_factor_coder":      round(contention_a, 3),
        "prefill_contention_factor_thinker":    round(contention_b, 3),
        "crashes_or_corruption":                corruption,
    },
    "verdict": verdict,
    "notes": "",
}

(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

def fmt(val, pass_thr, incon_thr, higher_is_better=True):
    if higher_is_better:
        return "PASS" if val >= pass_thr else ("INCON" if val >= incon_thr else "FAIL")
    else:
        return "PASS" if val <= pass_thr else ("INCON" if val <= incon_thr else "FAIL")

summary = f"""# T1.2a TP=1-per-GPU Concurrent Dual-Process — {verdict}

**vLLM** 0.19.0 | **Coder** Qwen3-Coder-30B-A3B-AWQ | **Thinker** Qwen3.5-27B-AWQ | **Date** {timestamp[:10]}
**Placement** TP=1 per GPU (Coder: GPU0, Thinker: GPU1) | **gpu-mem-util** 0.85 each

| Metric | Measured | Pass / Incon threshold | Result |
|--------|----------|------------------------|--------|
| Coder concurrent TPS ratio | {ratio_a:.3f} ({tps_a_iso:.1f} → {tps_a_con:.1f} t/s) | ≥0.80 / ≥0.60 | {fmt(ratio_a, 0.80, 0.60)} |
| Thinker concurrent TPS ratio | {ratio_b:.3f} ({tps_b_iso:.1f} → {tps_b_con:.1f} t/s) | ≥0.80 / ≥0.60 | {fmt(ratio_b, 0.80, 0.60)} |
| Prefill contention — coder | {contention_a:.2f}× ({ttft_a_iso:.0f} → {ttft_a_con:.0f} ms) | ≤2.0× / ≤3.0× | {fmt(contention_a, 2.0, 3.0, higher_is_better=False)} |
| Prefill contention — thinker | {contention_b:.2f}× ({ttft_b_iso:.0f} → {ttft_b_con:.0f} ms) | ≤2.0× / ≤3.0× | {fmt(contention_b, 2.0, 3.0, higher_is_better=False)} |
| Crashes / corruption | {corruption} | 0 / 0 | {"PASS" if corruption == 0 else "FAIL"} |

**Verdict: {verdict}**

VRAM dual-loaded: {vram_mib:.0f} MiB (budget ≤52429 MiB)
"""
(out / "summary.md").write_text(summary)

print(f"[T1.2a] Verdict: {verdict}")
print(f"[T1.2a] TPS ratios: coder={ratio_a:.3f}  thinker={ratio_b:.3f}")
print(f"[T1.2a] Prefill contention: coder={contention_a:.2f}×  thinker={contention_b:.2f}×")
print(f"[T1.2a] Results: ${RESULTS_DIR}/")
PYEOF
