#!/usr/bin/env bash
# T1.3_coder_next_tp2.sh — viability check for Behemoth model
# Tests: load success, decode speed, prefill speed, and basic tool calling on Next-80B
# Target: decode TPS >= 40 at seq=1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

ITEM_ID="T1.3_qwen3_coder_next_awq_tp2_viability"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit"
ENDPOINT="http://localhost:${PORT_VLLM_TP2_C}/v1"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log()  { echo "[T1.3] $*" | tee -a "${LOG}"; }
die()  { log "FATAL: $*"; exit 1; }

# ── Write Python helper to temp file ──────────────────────────────────────────
MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
"""
T1.3 measurement helper.

Modes:
  decode_seq1 <endpoint> <model> <out_json>
  decode_seq4 <endpoint> <model> <out_json>
  prefill     <endpoint> <model> <num_tokens> <out_json>
"""
import asyncio, httpx, json, sys, time

DECODE_PROMPT = (
    "Write a complete Python implementation of a red-black tree with insert, "
    "delete, search, and in-order traversal. Include full type hints. "
    "Do not stop until the implementation is fully complete."
)
MAX_DECODE_TOKENS = 1500

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
                        prompt: str, max_tokens: int) -> dict:
    t0 = time.monotonic()
    fttt: float | None = None
    count = 0
    text_out = []

    async with client.stream("POST", f"{endpoint}/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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

async def decode_concurrent(endpoint: str, model: str, num_seqs: int, out_path: str) -> None:
    async with httpx.AsyncClient(timeout=600) as c:
        coros = [_measure_one(c, endpoint, model, DECODE_PROMPT, MAX_DECODE_TOKENS) for _ in range(num_seqs)]
        results = await asyncio.gather(*coros)
    
    # Calculate aggregate metrics
    avg_tps = sum(r["decode_tps"] for r in results) / num_seqs
    avg_ttft = sum(r["ttft_ms"] for r in results) / num_seqs
    total_tps = sum(r["decode_tps"] for r in results)
    
    aggregated = {
        "avg_ttft_ms": round(avg_ttft, 1),
        "avg_tps_per_seq": round(avg_tps, 1),
        "total_system_tps": round(total_tps, 1),
        "results": results
    }
    
    with open(out_path, "w") as f:
        json.dump(aggregated, f)
    print(json.dumps(aggregated))

async def prefill(endpoint: str, model: str, num_tokens: int, out_path: str) -> None:
    prompt = make_long_prompt(num_tokens)
    async with httpx.AsyncClient(timeout=600) as c:
        result = await _measure_one(c, endpoint, model, prompt, 1)
        
    ttft_s = result["ttft_ms"] / 1000.0
    prefill_tps = num_tokens / ttft_s if ttft_s > 0 else 0
    result["prefill_tps"] = round(prefill_tps, 1)
    
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))

mode = sys.argv[1]
if mode == "decode_seq1":
    asyncio.run(decode_concurrent(sys.argv[2], sys.argv[3], 1, sys.argv[4]))
elif mode == "decode_seq4":
    asyncio.run(decode_concurrent(sys.argv[2], sys.argv[3], 4, sys.argv[4]))
elif mode == "prefill":
    asyncio.run(prefill(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]))
else:
    sys.exit(1)
PYEOF

# ── Step 1: Deploy Behemoth (tp2c, gpu-mem 0.95) ──────────────────────────────
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

log "Deploying behemoth (${MODEL}) on tp2c (TP=2) ..."
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2c "${MODEL}" \
    --gpu-mem-util 0.95 \
    --ctx 32768 \
    --tool-call-parser hermes \
    2>&1 | tee -a "${LOG}"

# ── Step 2: Decode Measurements ───────────────────────────────────────────────
log "Measuring decode TPS at seq=1 ..."
python3 "${MEASURE_PY}" decode_seq1 "${ENDPOINT}" "${MODEL}" "${RESULTS_DIR}/raw/decode_seq1.json" | tee -a "${LOG}"
DEC_TPS_1=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/decode_seq1.json'))['total_system_tps'])")

log "Measuring decode TPS at seq=4 ..."
python3 "${MEASURE_PY}" decode_seq4 "${ENDPOINT}" "${MODEL}" "${RESULTS_DIR}/raw/decode_seq4.json" | tee -a "${LOG}"
DEC_TPS_4=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/decode_seq4.json'))['total_system_tps'])")

# ── Step 3: Prefill Measurements ──────────────────────────────────────────────
for tokens in 8000 16000 32000; do
    log "Measuring prefill throughput at ${tokens} tokens ..."
    python3 "${MEASURE_PY}" prefill "${ENDPOINT}" "${MODEL}" "${tokens}" "${RESULTS_DIR}/raw/prefill_${tokens}.json" | tee -a "${LOG}"
done
PREFILL_TPS_32K=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/prefill_32000.json'))['prefill_tps'])")

# ── Step 4: Tool-Call Verification (all Phase 0 tasks) ───────────────────────
log "Running all Phase 0 tool-call tasks (IDs 01–09, matched by --task-filter '0') ..."
# Use the phase0 bench module against the live endpoint.
python3 -m benchmarks.phase0_tool_reliability.bench \
    --endpoint "${ENDPOINT}" \
    --results-dir "${RESULTS_DIR}/tool_tasks" \
    --tasks "${REPO_ROOT}/benchmarks/phase0_tool_reliability/tasks/" \
    --task-filter "0" \
    2>&1 | tee -a "${LOG}"

# Evaluate the smoke tests success rate from metrics.json
TOOL_PASS_RATE=$(python3 -c "import json; d = json.load(open('${RESULTS_DIR}/tool_tasks/metrics.json')); print(d['metrics']['tool_call_success_rate'])")
PASSED_TASKS=$(python3 -c "import json; d = json.load(open('${RESULTS_DIR}/tool_tasks/metrics.json')); print(d['metrics']['counts']['pass'])")
TOTAL_TASKS=$(python3 -c "import json; d = json.load(open('${RESULTS_DIR}/tool_tasks/metrics.json')); print(d['metrics']['total'])")
log "Tool-call smoke success rate: ${PASSED_TASKS}/${TOTAL_TASKS} (${TOOL_PASS_RATE})"

# ── Step 5: Compute metrics and verdict ───────────────────────────────────────
python3 - <<PYEOF
import json, pathlib

item_id   = "${ITEM_ID}"
timestamp = "${TIMESTAMP}"
tps_seq1  = float("${DEC_TPS_1}")
tps_seq4  = float("${DEC_TPS_4}")
prefill_32k = float("${PREFILL_TPS_32K}")
tool_rate = float("${TOOL_PASS_RATE}")
out = pathlib.Path("${RESULTS_DIR}")

# Thresholds from config/thresholds.yaml T1.3
PASS_TPS = 40.0;     INCON_TPS = 25.0
PASS_TOOL = 0.95;    INCON_TOOL = 0.80

if tps_seq1 >= PASS_TPS and tool_rate >= PASS_TOOL:
    verdict = "PASS"
elif tps_seq1 >= INCON_TPS and tool_rate >= INCON_TOOL:
    verdict = "INCONCLUSIVE"
else:
    verdict = "FAIL"

metrics = {
    "item_id":   item_id,
    "timestamp": timestamp,
    "config": {
        "engine":          "vllm",
        "engine_version":  "0.19.0",
        "model":           "${MODEL}",
        "quantization":    "AWQ-INT4",
        "kv_cache_dtype":  "auto",
        "placement":       "tp=2 (tp2c)",
        "context_length":  32768,
        "extra_args":      "--tool-call-parser hermes --enable-auto-tool-choice --gpu-memory-utilization 0.95",
    },
    "metrics": {
        "decode_tps_seq1": tps_seq1,
        "decode_tps_seq4_total": tps_seq4,
        "prefill_32k_tokens_per_s": prefill_32k,
        "tool_call_pass_rate": tool_rate,
    },
    "verdict": verdict,
    "notes": "MTP spec decode disabled per T1.3 requirements",
}

(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

def strfmt(val, pass_thr, incon_thr):
    return "PASS" if val >= pass_thr else ("INCON" if val >= incon_thr else "FAIL")

summary = f"""# T1.3 Behemoth Next-80B Viability — {verdict}

**vLLM** 0.19.0 | **Model** {metrics['config']['model']} | **Date** {timestamp[:10]}
**Placement** TP=2 (slot tp2c) | **gpu-mem-util** 0.95

| Metric | Measured | Pass / Incon threshold | Result |
|--------|----------|------------------------|--------|
| Decode TPS (seq=1) | {tps_seq1:.1f} t/s | ≥40.0 / ≥25.0 | {strfmt(tps_seq1, 40.0, 25.0)} |
| Tool call pass rate | {tool_rate:.2%} | ≥95% / ≥80% | {strfmt(tool_rate, 0.95, 0.80)} |
| Prefill speed @ 32k | {prefill_32k:.0f} t/s | (Orientation only) | - |
| Decode capacity (seq=4) | {tps_seq4:.1f} t/s total | - | - |

**Verdict: {verdict}**
"""
(out / "summary.md").write_text(summary)

print(f"[T1.3] Verdict: {verdict}")
print(f"[T1.3] Decode seq=1: {tps_seq1:.1f} t/s")
print(f"[T1.3] Tool pass rate: {tool_rate:.2%}")
print(f"[T1.3] Results: ${RESULTS_DIR}/")
PYEOF
