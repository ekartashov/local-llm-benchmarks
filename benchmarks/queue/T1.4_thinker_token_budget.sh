#!/usr/bin/env bash
# T1.4_thinker_token_budget.sh — verify raising max_tokens to 16384 resolves th03 empty-output defect.
# Procedure: deploy Qwen3.5-27B-AWQ on gpu1, run th03 task 5 times with max_tokens=16384.
# Pass: 5/5 runs produce a non-empty answer (reasoning completes + answer text present).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

ITEM_ID="T1.4_th03_token_budget_fix"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="QuantTrio/Qwen3.5-27B-AWQ"
ENDPOINT="http://localhost:${PORT_VLLM_GPU1}/v1"
REPS=5          # run th03 this many times
MAX_TOKENS=16384

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[T1.4] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

# ── Write Python helper ────────────────────────────────────────────────────────
RUNNER_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${RUNNER_PY}"' EXIT
cat > "${RUNNER_PY}" <<'PYEOF'
"""
T1.4 runner — send th03 task N times and record whether each run produces
a non-empty answer (content outside <think>...</think> blocks).
"""
import asyncio, httpx, json, re, sys, time

SYSTEM = (
    "You are a principal engineer. "
    "Reason through each option's tradeoffs systematically before giving a recommendation."
)
USER = (
    "We're building a system that fans out LLM requests to multiple local GPU workers. "
    "Choose between these three architectures and justify your choice:\n\n"
    "A) Central queue (Redis) + worker processes polling it\n"
    "B) Push-based: a router process holds all connections and pushes work directly to idle workers\n"
    "C) Sidecar model: each GPU worker runs its own HTTP server; a Nginx upstream routes by least-connections\n\n"
    "Constraints:\n"
    "- 4 GPU workers, each processing ~150 t/s\n"
    "- Peak load: 20 concurrent requests, each ~2000 tokens\n"
    "- P99 TTFT target: < 800ms\n"
    "- The router itself must not be a bottleneck\n"
    "- Workers can crash and must be handled gracefully\n"
    "- No Kubernetes — bare metal with systemd\n\n"
    "For each option: analyse latency, failure modes, operational complexity, and scalability. "
    "Then recommend one and explain what you'd monitor in production."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()

async def run_once(client: httpx.AsyncClient, endpoint: str, model: str,
                   max_tokens: int, rep: int, out_path: str) -> dict:
    t0 = time.monotonic()
    fttt: float | None = None
    total_tokens = 0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    async with client.stream("POST", f"{endpoint}/chat/completions", json={
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }) as resp:
        resp.raise_for_status()
        async for raw in resp.aiter_lines():
            if not raw.startswith("data: ") or "[DONE]" in raw:
                continue
            chunk = json.loads(raw[6:])
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                if fttt is None:
                    fttt = time.monotonic() - t0
                content_parts.append(delta["content"])
                total_tokens += 1
            if delta.get("reasoning"):
                if fttt is None:
                    fttt = time.monotonic() - t0
                reasoning_parts.append(delta["reasoning"])
                total_tokens += 1

    elapsed = time.monotonic() - t0
    full_content = "".join(content_parts)
    full_reasoning = "".join(reasoning_parts)

    # Determine if there is a real answer. If no separate reasoning stream,
    # strip <think>…</think> from content.
    if reasoning_parts:
        answer = full_content.strip()
    else:
        answer = strip_think(full_content)

    result = {
        "rep": rep,
        "total_tokens": total_tokens,
        "reasoning_tokens": len(reasoning_parts),
        "content_tokens": len(content_parts),
        "ttft_ms": round((fttt or 0) * 1000, 1),
        "elapsed_s": round(elapsed, 2),
        "answer_len": len(answer),
        "non_empty": len(answer) > 50,   # >50 chars = real answer, not whitespace
        "answer_snippet": answer[:300],
        "reasoning_snippet": full_reasoning[:200],
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in
                      ("rep", "non_empty", "answer_len", "total_tokens", "ttft_ms")}))
    return result


async def main(endpoint: str, model: str, reps: int, max_tokens: int, out_dir: str) -> None:
    results = []
    async with httpx.AsyncClient(timeout=600) as c:
        for i in range(1, reps + 1):
            print(f"[T1.4] Running rep {i}/{reps} ...")
            r = await run_once(c, endpoint, model, max_tokens, i, f"{out_dir}/rep_{i:02d}.json")
            results.append(r)

    summary = {
        "reps": reps,
        "non_empty_count": sum(1 for r in results if r["non_empty"]),
        "non_empty_rate": sum(1 for r in results if r["non_empty"]) / reps,
        "mean_total_tokens": sum(r["total_tokens"] for r in results) / reps,
        "mean_ttft_ms": sum(r["ttft_ms"] for r in results) / reps,
    }
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


endpoint, model, reps, max_tokens, out_dir = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
)
asyncio.run(main(endpoint, model, reps, max_tokens, out_dir))
PYEOF

# ── Step 1: Deploy thinker on gpu1 with larger context ────────────────────────
# ctx=32768 so the server's max-model-len does not cap our max_tokens=16384.
# Added VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 to prevent OOM during CUDA graph capture on V1 engine.
log "Deploying thinker (${MODEL}) on gpu1, ctx=32768, gpu-mem-util=0.95 ..."
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
    --gpu-mem-util 0.95 \
    --ctx 32768 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-num-seqs 1 \
    2>&1 | tee -a "${LOG}"

# ── Step 2: Warmup run ─────────────────────────────────────────────────────────
log "Warmup request (discarded) ..."
python3 - <<PYEOF 2>/dev/null || true
import httpx, asyncio, json
async def w():
    async with httpx.AsyncClient(timeout=120) as c:
        await c.post("${ENDPOINT}/chat/completions", json={
            "model": "${MODEL}",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10, "temperature": 0.0,
        })
asyncio.run(w())
PYEOF

# ── Step 3: Run th03 task 5 times ─────────────────────────────────────────────
log "Running th03_architecture_tradeoffs × ${REPS} with max_tokens=${MAX_TOKENS} ..."
python3 "${RUNNER_PY}" \
    "${ENDPOINT}" "${MODEL}" "${REPS}" "${MAX_TOKENS}" \
    "${RESULTS_DIR}/raw" \
    2>&1 | tee -a "${LOG}"

# ── Step 4: Collect and write final metrics ────────────────────────────────────
python3 - <<PYEOF
import json, pathlib

item_id   = "${ITEM_ID}"
timestamp = "${TIMESTAMP}"
out       = pathlib.Path("${RESULTS_DIR}")
raw_dir   = out / "raw"

summary = json.loads((raw_dir / "summary.json").read_text())
non_empty_rate  = summary["non_empty_rate"]
non_empty_count = summary["non_empty_count"]
reps            = summary["reps"]

# Thresholds from config/thresholds.yaml T1.4
PASS_RATE  = 1.00
INCON_RATE = 0.80

if non_empty_rate >= PASS_RATE:
    verdict = "PASS"
elif non_empty_rate >= INCON_RATE:
    verdict = "INCONCLUSIVE"
else:
    verdict = "FAIL"

metrics = {
    "item_id":   item_id,
    "timestamp": timestamp,
    "config": {
        "engine":         "vllm",
        "engine_version": "0.19.0",
        "model":          "${MODEL}",
        "placement":      "tp=1 (gpu1)",
        "context_length": 32768,
        "max_tokens_per_request": ${MAX_TOKENS},
        "extra_args":     "--tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-num-seqs 1",
    },
    "metrics": {
        "non_empty_answer_rate":  non_empty_rate,
        "non_empty_count":        non_empty_count,
        "reps":                   reps,
        "mean_total_tokens":      summary["mean_total_tokens"],
        "mean_ttft_ms":           summary["mean_ttft_ms"],
    },
    "verdict": verdict,
    "notes": "th03_architecture_tradeoffs task. Fix: max_tokens raised to 16384 from 8192.",
}

(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

def fmt(v, p, i): return "PASS" if v >= p else ("INCON" if v >= i else "FAIL")

md = f"""# T1.4 Thinker Token Budget Fix (th03) — {verdict}

**vLLM** 0.19.0 | **Model** Qwen3.5-27B-AWQ | **Date** {timestamp[:10]}
**Placement** TP=1 (gpu1) | **ctx** 32768 | **max_tokens** ${MAX_TOKENS}

| Metric | Measured | Pass / Incon threshold | Result |
|--------|----------|------------------------|--------|
| Non-empty answer rate | {non_empty_count}/{reps} ({non_empty_rate:.0%}) | ≥100% / ≥80% | {fmt(non_empty_rate, 1.00, 0.80)} |
| Mean total tokens | {summary['mean_total_tokens']:.0f} | (orientation) | - |
| Mean TTFT | {summary['mean_ttft_ms']:.0f} ms | (orientation) | - |

**Verdict: {verdict}**

"""
if verdict == "PASS":
    md += "**Action:** Update `qwen35_27b_awq` entry in `config/models.yaml` — add `max_tokens_per_request: 16384` note, remove `outstanding_defect` entry for th03.\n"
elif verdict == "FAIL":
    md += "**Action:** max_tokens raise did not fix th03. Hand back to research — the thinker cannot handle architecture-heavy tasks regardless of budget. Route such tasks to coder or behemoth.\n"
else:
    md += "**Action:** Partial fix. Some runs still empty. Check individual rep JSON files in raw/ for pattern.\n"

(out / "summary.md").write_text(md)
print(f"[T1.4] Verdict: {verdict} ({non_empty_count}/{reps} non-empty)")
PYEOF
