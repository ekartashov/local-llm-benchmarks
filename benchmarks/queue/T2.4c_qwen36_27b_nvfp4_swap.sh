#!/usr/bin/env bash
# T2.4c_qwen36_27b_nvfp4_swap.sh — Qwen3.6-27B-NVFP4 as Arclight thinker alternative
#
# Testing hypothesis: AWQ + FP8 KV cache constraint at TP=1 degraded the model,
# causing "confident incorrectness" in T2.4. Sleeping the coder and allocating
# both GPUs for TP=2 gives us room for NVFP4 weights and BF16 KV cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

# ── Argument Parsing ─────────────────────────────────────────────────────────
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY_RUN=1; shift ;;
        *) shift ;;
    esac
done

ITEM_ID="T2.4c_arclight_thinker_qwen36_27b_nvfp4_swap"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="QuantTrio/Qwen3.6-27B-NVFP4"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "[T2.4c] $*" | tee -a "${LOG}"
    fi
}
die() { log "FATAL: $*"; exit 1; }

# ── Decode measurement helper ──────────────────────────────────────────────────
MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
"""T2.4c decode measurement helper."""
import asyncio, httpx, json, sys, time

DECODE_PROMPT = (
    "Explain the Byzantine Generals Problem and describe two distinct consensus "
    "protocols that solve it. For each protocol, outline the key invariants, "
    "failure modes, and the trade-off between safety and liveness."
)
MAX_DECODE_TOKENS = 2048

async def _measure_one(client, endpoint, model, prompt, max_tokens):
    t0 = time.monotonic()
    fttt = None
    count = 0
    text_out = []
    async with client.stream("POST", f"{endpoint}/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
        "extra_body": {"top_k": 20},
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
        "text_snippet": "".join(text_out)[:300],
    }

async def measure(endpoint, model, num_seqs, out_path):
    async with httpx.AsyncClient(timeout=600) as c:
        results = await asyncio.gather(*[
            _measure_one(c, endpoint, model, DECODE_PROMPT, MAX_DECODE_TOKENS)
            for _ in range(num_seqs)
        ])
    agg = {
        "avg_tps_per_seq": round(sum(r["decode_tps"] for r in results) / num_seqs, 1),
        "total_tps":       round(sum(r["decode_tps"] for r in results), 1),
        "avg_ttft_ms":     round(sum(r["ttft_ms"] for r in results) / num_seqs, 1),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(agg, f)
    print(json.dumps({k: agg[k] for k in ("avg_tps_per_seq", "total_tps", "avg_ttft_ms")}))

mode, endpoint, model, num_seqs, out_path = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
asyncio.run(measure(endpoint, model, num_seqs, out_path))
PYEOF

# ── Preflight ─────────────────────────────────────────────────────────────────
log "=== T2.4c Qwen3.6-27B-NVFP4 TP=2 Hypothesis Swap ==="

ENDPOINT="http://localhost:${PORT_VLLM_TP2_B}/v1"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "WOULD DEPLOY: _deploy_tp2"
else
    log "Stopping conflicting containers if any (freeing both GPUs)..."
    for c in bench-vllm-gpu0 bench-vllm-gpu1 bench-vllm-tp2a bench-vllm-tp2b; do
        if podman container exists "${c}" 2>/dev/null; then
            log "Stopping ${c} ..."
            podman stop "${c}" 2>/dev/null || true
            podman rm   "${c}" 2>/dev/null || true
        fi
    done
    
    log "Deploying on tp2b..."
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
        "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2b "${MODEL}" \
        --gpu-mem-util 0.85 \
        --kv-cache-dtype bf16 \
        --ctx 49152 \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --language-model-only \
        --enable-auto-tool-choice \
        2>&1 | tee -a "${LOG}" || die "Deployment failed."
fi

log "Endpoint: ${ENDPOINT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "WOULD MEASURE: decode TPS seq=1"
    log "WOULD RUN: specialized subset phase2 quality bench"
    DEC_TPS_1="0.0"
else
    # ── Decode TPS ─────────────────────────────────────────────────────────
    log "Measuring decode TPS seq=1 ..."
    python3 "${MEASURE_PY}" measure "${ENDPOINT}" "${MODEL}" 1 \
        "${RESULTS_DIR}/raw/decode_seq1.json" | tee -a "${LOG}" || true
    DEC_TPS_1=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/raw/decode_seq1.json'))['avg_tps_per_seq'])" 2>/dev/null || echo "0.0")

    # subset for th02 and th03
    SUBSET_DIR="${RESULTS_DIR}/target_tasks"
    mkdir -p "${SUBSET_DIR}"
    cp "${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker/th02_algorithm_design.json" "${SUBSET_DIR}/"
    cp "${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker/th03_architecture_tradeoffs.json" "${SUBSET_DIR}/"

    log "Running Phase 2.2 subset quality suite (th02, th03) ..."
    python3 -m benchmarks.phase2_model_selection.bench \
        --endpoint "${ENDPOINT}" \
        --results-dir "${RESULTS_DIR}/phase2_quality" \
        --mode quality \
        --tasks "${SUBSET_DIR}" \
        --model "${MODEL}" \
        --label "Qwen3.6-27B-NVFP4" \
        --max-tokens 16384 \
        2>&1 | tee -a "${LOG}" || {
        log "WARNING: phase2 quality bench exited non-zero — check logs."
    }
fi

# Write metrics
python3 - <<PYEOF
import json, pathlib

out = pathlib.Path("${RESULTS_DIR}")
metrics = {
    "item_id": "${ITEM_ID}",
    "timestamp": "${TIMESTAMP}",
    "config": {
        "engine": "vllm",
        "model": "${MODEL}",
        "placement": "tp=2 (tp2b)",
        "quantization": "NVFP4",
    },
    "metrics": {
        "decode_tps_seq1": float("${DEC_TPS_1}"),
    },
    "verdict": "PENDING_HUMAN_REVIEW",
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

md = f"""# T2.4c Qwen3.6-27B-NVFP4 — Swap Hypothesis

**Model** {metrics['config']['model']} | **Placement** tp=2 | **kv cache** bf16

TPS Target (seq=1): {metrics['metrics']['decode_tps_seq1']} t/s

## Human review required

Open **{out}/phase2_quality/human_review.md** and score exactly the two problematic tasks:
- **th02_algorithm_design**: verify if code avoids \`IndexError\`.
- **th03_architecture_tradeoffs**: verify logic-math consistency.

**Verdict: PENDING HUMAN REVIEW**
"""
(out / "summary.md").write_text(md)
PYEOF

log "Done."
log "Human review: ${RESULTS_DIR}/phase2_quality/human_review.md"
