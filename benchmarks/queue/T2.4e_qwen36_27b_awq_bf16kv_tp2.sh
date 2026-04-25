#!/usr/bin/env bash
# T2.4e_qwen36_27b_awq_bf16kv_tp2.sh
#
# AWQ + bf16 KV + TP=2 — isolates KV precision from T2.4c's untrusted publisher.
# Uses same trusted AWQ weights (QuantTrio) with bf16 KV (kv-cache-dtype auto)
# and TP=2 for VRAM headroom. Single run — decision in one shot.
#
# Key isolation: T2.4c conflated NVFP4 publisher quality with KV precision.
# This test keeps weights fixed (AWQ, trusted) and changes ONLY KV dtype.
#
# Config:
#   Model:       QuantTrio/Qwen3.6-27B-AWQ
#   Placement:   TP=2 (tp2b — borrows both GPUs)
#   KV cache:    auto (bf16 — no quantization noise)
#   ctx:         32768
#   max_tokens:  16384
#   max-num-seqs 1
#
# Requires: coder (GPU0 / tp2a) must be stopped first (script handles this).
#
#   DISABLE_CHUNKED_PREFILL=1              — apply if H2 testing still needed
#     adds --enable-chunked-prefill=False
#
# Decision gate (single run):
#   th02 CORRECT + mean >=4.0 → bf16 KV + TP=2 is the production config. PASS.
#   th02 CORRECT + mean <4.0  → KV fixed th02 but overall quality regressed. INCONCLUSIVE.
#   th02 WRONG                → KV precision is not the root cause. Capability ceiling.
#                               → T2.4b (Qwopus SFT) or accept Qwen3.5-27B baseline.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY_RUN=1; shift ;;
        *) shift ;;
    esac
done

# ── Corrections from T2.4f ─────────────────────────────────────────────────────
# See T2.4f findings.md for when to set these.
# ROPE_THETA_FLAG: override rope_theta if T2.4f H1 showed mismatch, e.g.:
#   export ROPE_THETA_FLAG="--rope-theta 10000000"
# Leave empty if H1 was OK.
ROPE_THETA_FLAG="${ROPE_THETA_FLAG:-}"

# DISABLE_CHUNKED_PREFILL: set to 1 to add --disable-chunked-prefill.
# Set if T2.4d Variant B showed chunked-prefill was the issue AND you want
# to confirm in the bf16 KV + TP=2 config too.
DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-}"

ITEM_ID="T2.4e_qwen36_27b_awq_bf16kv_tp2"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="QuantTrio/Qwen3.6-27B-AWQ"

MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
"""T2.4e TPS measurement helper."""
import asyncio, httpx, json, sys, time

DECODE_PROMPT = (
    "Explain the Byzantine Generals Problem and describe two distinct consensus "
    "protocols that solve it. For each protocol, outline the key invariants, "
    "failure modes, and the trade-off between safety and liveness."
)

async def _one(client, endpoint, model):
    t0 = time.monotonic()
    fttt = None
    count = 0
    text_out = []
    async with client.stream("POST", f"{endpoint}/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": DECODE_PROMPT}],
        "max_tokens": 1024,
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
    }

async def measure(endpoint, model, out_path):
    async with httpx.AsyncClient(timeout=300) as c:
        result = await _one(c, endpoint, model)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))

asyncio.run(measure(sys.argv[1], sys.argv[2], sys.argv[3]))
PYEOF

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "[T2.4e] $*" | tee -a "${LOG}"
    fi
}
die() { log "FATAL: $*"; exit 1; }

log "=== T2.4e Qwen3.6-27B-AWQ + bf16 KV + TP=2 ==="
log "Timestamp:             ${TIMESTAMP}"
log "ROPE_THETA_FLAG:       '${ROPE_THETA_FLAG}'"
log "DISABLE_CHUNKED_PREFILL: '${DISABLE_CHUNKED_PREFILL}'"
log ""
log "Hypothesis: fp8 KV on TP=1 degraded th02 quality. bf16 KV + TP=2 should fix it."
log "Isolation:  same AWQ weights (QuantTrio), only KV dtype changes vs run 4."

# ── Build extra args array ────────────────────────────────────────────────────
EXTRA_ARGS=()
# shellcheck disable=SC2206
[[ -n "${ROPE_THETA_FLAG}" ]]       && EXTRA_ARGS+=(${ROPE_THETA_FLAG})
[[ -n "${DISABLE_CHUNKED_PREFILL}" ]] && EXTRA_ARGS+=(--no-enable-chunked-prefill)

ENDPOINT="http://localhost:${PORT_VLLM_TP2_B}/v1"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "WOULD STOP: bench-vllm-gpu0, bench-vllm-tp2a (free both GPUs for TP=2)"
    log "WOULD DEPLOY: vllm tp2b ${MODEL} --gpu-mem-util 0.85 --ctx 32768"
    log "              --kv-cache-dtype auto --max-num-seqs 1 ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
    log "WOULD MEASURE: decode TPS seq=1"
    log "WOULD RUN: Phase 2.2 quality suite (8 tasks, max_tokens=16384)"
    DEC_TPS="0.0"
else
    # ── Step 1: Stop coder to free both GPUs ─────────────────────────────────
    log ""
    log "--- Step 1: Stop coder container (free both GPUs for TP=2) ---"
    for c in bench-vllm-gpu0 bench-vllm-tp2a; do
        if podman container exists "${c}" 2>/dev/null; then
            log "Stopping ${c}..."
            podman stop "${c}" 2>/dev/null || true
            podman rm   "${c}" 2>/dev/null || true
        fi
    done
    log "Both GPUs free."

    # ── Step 2: Deploy AWQ TP=2, bf16 KV ─────────────────────────────────────
    log ""
    log "--- Step 2: Deploy TP=2 (tp2b), bf16 KV, ctx=32768 ---"
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
        "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2b "${MODEL}" \
        --gpu-mem-util 0.85 \
        --ctx 32768 \
        --kv-cache-dtype auto \
        --max-num-seqs 1 \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        2>&1 | tee -a "${LOG}" \
        || die "Deployment failed."

    log "Deployed on tp2b (TP=2). Endpoint: ${ENDPOINT}"

    # ── Warmup ────────────────────────────────────────────────────────────────
    log "Warmup request (discarded)..."
    python3 -c "
import httpx, asyncio
async def w():
    async with httpx.AsyncClient(timeout=120) as c:
        try:
            await c.post('${ENDPOINT}/chat/completions',
                json={'model': '${MODEL}',
                      'messages': [{'role': 'user', 'content': 'hi'}],
                      'max_tokens': 10})
        except Exception:
            pass
asyncio.run(w())
" 2>/dev/null || log "WARNING: warmup request failed (non-fatal)."

    # ── Step 3: Measure decode TPS seq=1 ─────────────────────────────────────
    log ""
    log "--- Step 3: Decode TPS (seq=1) ---"
    python3 "${MEASURE_PY}" "${ENDPOINT}" "${MODEL}" \
        "${RESULTS_DIR}/raw/tps_seq1.json" \
        | tee -a "${LOG}" || true
    DEC_TPS=$(python3 -c "
import json
try:
    print(json.load(open('${RESULTS_DIR}/raw/tps_seq1.json')).get('decode_tps', 0))
except Exception:
    print('0.0')
" 2>/dev/null || echo "0.0")
    log "TPS seq=1: ${DEC_TPS} t/s  (AWQ fp8 TP=1 baseline: ~77 t/s)"

    # ── Step 4: Phase 2.2 quality suite ──────────────────────────────────────
    log ""
    log "--- Step 4: Phase 2.2 quality suite (8 tasks, max_tokens=16384) ---"
    log "Focus: th02 (correct = all jobs assigned incl. misses to busiest GPU)"
    log "       th05 (distributed consistency edge cases)"
    python3 -m benchmarks.phase2_model_selection.bench \
        --endpoint "${ENDPOINT}" \
        --results-dir "${RESULTS_DIR}/phase2_quality" \
        --mode quality \
        --tasks "${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker/" \
        --model "${MODEL}" \
        --label "Qwen3.6-27B-AWQ-bf16kv-tp2" \
        --max-tokens 16384 \
        2>&1 | tee -a "${LOG}" || {
        log "WARNING: quality bench exited non-zero — check logs."
    }
fi

# ── Write metrics.json and summary.md ────────────────────────────────────────
python3 - <<PYEOF
import json, pathlib

out = pathlib.Path("${RESULTS_DIR}")
ts = "${TIMESTAMP}"
rope_flag = "${ROPE_THETA_FLAG}".strip()
cp_flag = "1" if "${DISABLE_CHUNKED_PREFILL}".strip() else ""

try:
    dec_tps = float(json.load(open(out / "raw/tps_seq1.json")).get("decode_tps", 0))
except Exception:
    dec_tps = 0.0

extra_args_str = " ".join(filter(None, [rope_flag, "--no-enable-chunked-prefill" if cp_flag else ""]))

metrics = {
    "item_id": "${ITEM_ID}",
    "timestamp": ts,
    "config": {
        "engine":         "vllm",
        "engine_version": "0.19.x",
        "model":          "${MODEL}",
        "quantization":   "AWQ-INT4",
        "kv_cache_dtype": "auto (bf16)",
        "placement":      "tp=2 (tp2b)",
        "context_length": 32768,
        "max_tokens":     16384,
        "extra_args":     f"--kv-cache-dtype auto --max-num-seqs 1 --tool-call-parser qwen3_coder --reasoning-parser qwen3{' ' + extra_args_str if extra_args_str else ''}",
    },
    "metrics": {
        "decode_tps_seq1":          dec_tps,
        "quality_mean_8task":       None,  # fill after human review
        "th02_correct":             None,  # True / False
        "th05_correct":             None,  # True / False
        "baseline_tps_fp8_tp1":     77.4,
        "baseline_quality_qwen35":  4.0,
    },
    "verdict": "PENDING_HUMAN_REVIEW",
    "notes": (
        "Primary discriminator: th02. CORRECT = all jobs assigned including misses "
        "(to busiest GPU). SEMANTIC ERROR = misses return -1. "
        "PASS requires th02 correct AND mean >=4.0/5."
    ),
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

tps_ratio = round(dec_tps / 77.4, 2) if dec_tps > 0 else "N/A"

md = f"""# T2.4e Qwen3.6-27B-AWQ + bf16 KV + TP=2

**Model:** QuantTrio/Qwen3.6-27B-AWQ
**Config:** TP=2 (tp2b), bf16 KV (kv-cache-dtype auto), ctx=32768, max_tokens=16384, max-num-seqs 1
**Extra flags:** {extra_args_str if extra_args_str else "none"}
**Timestamp:** {ts}

## What this test isolates

T2.4c used sakamakismile/NVFP4 (untrusted publisher). We could not separate
"NVFP4 format benefit" from "poor calibration". This test uses the same trusted
AWQ weights (QuantTrio) and changes ONLY the KV dtype: fp8 → bf16 (auto).
TP=2 provides VRAM headroom for bf16 KV without fp8 compression.

## Auto-measured metrics

| Metric | This run | Baseline (Qwen3.5-27B) |
|--------|----------|------------------------|
| Decode TPS (seq=1) | {dec_tps:.1f} t/s | 76.5 t/s (ratio {tps_ratio}×) |
| Quality mean (8 tasks) | **PENDING** | 4.0/5 |
| th02 correct | **PENDING** | N/A |
| th05 correct | **PENDING** | N/A |

## Human review required

Score all 8 tasks in **phase2_quality/human_review.md** (1–5 per task).

**th02 — the discriminating task (multi-GPU job scheduler):**
- CORRECT: all jobs assigned, including misses → assigned to **busiest GPU**
- SEMANTIC ERROR: missed jobs returned as -1 (not assigned to any GPU)
  The model will argue this is correct — it is not. Misses must be dispatched.

**th05 — distributed consistency edge cases:**
- Check whether cache version conflicts are handled correctly across hot-swap.

After scoring, update **metrics.json**:
- \`quality_mean_8task\` — mean of all 8 task scores
- \`th02_correct\` — True / False
- \`th05_correct\` — True / False
- \`verdict\` — PASS / FAIL / INCONCLUSIVE

## Decision gate

| th02 result | Mean | Verdict | Next action |
|-------------|------|---------|-------------|
| CORRECT | ≥4.0 | **PASS** | bf16 KV + TP=2 is the production config. Update DECISIONS.md. Mark Qwen3.6-27B as new thinker (with sleep-mode coordination for TP=2). |
| CORRECT | <4.0 | INCONCLUSIVE | KV fixed th02 but overall quality is below baseline. Hand back to research. |
| WRONG | any | **FAIL** | KV precision is not the root cause. Capability ceiling confirmed. Consider T2.4b (Qwopus SFT) or accept Qwen3.5-27B as permanent thinker. |

If PASS: the coder (Qwen3.6-35B-A3B-AWQ) must sleep before deploying thinker at TP=2.
Update ARCHITECTURE.md and DECISIONS.md with this constraint.

## Files

- phase2_quality/human_review.md — 8-task quality run
- raw/tps_seq1.json — TPS measurement
- bench.log — full deploy + run log
"""
(out / "summary.md").write_text(md)
print(f"[T2.4e] TPS seq=1 = {dec_tps:.1f} t/s  (ratio {tps_ratio}× vs bf16 TP=1 baseline ~77)")
print(f"[T2.4e] summary.md → {out}/summary.md")
PYEOF

log ""
log "=== T2.4e complete ==="
log "Results: ${RESULTS_DIR}"
log "Summary: ${RESULTS_DIR}/summary.md"
log ""
log "Next: score phase2_quality/human_review.md (1-5 per task)"
log "      Fill metrics.json: quality_mean_8task, th02_correct, th05_correct, verdict"
log "      PASS (th02 correct + mean >=4.0) → update DECISIONS.md, thinker settled"
log "      FAIL (th02 wrong)               → T2.4b (Qwopus SFT) or keep Qwen3.5-27B"
log ""
log "Note: coder (bench-vllm-gpu0/tp2a) was stopped for this test."
log "      Restart it when done (or after T_NVFP4 decision is made)."
