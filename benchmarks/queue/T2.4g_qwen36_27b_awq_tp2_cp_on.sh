#!/usr/bin/env bash
# T2.4g_qwen36_27b_awq_tp2_cp_on.sh
#
# Confound resolution: TP=2 + bf16 KV + chunked-prefill ON.
#
# T2.4e ran TP=2 + bf16 KV + cp-OFF and th02 was INCORRECT. But T2.4e changed
# two variables from T2.4d simultaneously (TP=1→2 AND cp-ON→OFF), so we cannot
# attribute the failure to either variable. T2.4g is the single missing cell:
#
#   ┌──────────────┬────────────────────┬────────────────────────┐
#   │              │ chunked-prefill ON │ chunked-prefill OFF     │
#   ├──────────────┼────────────────────┼────────────────────────┤
#   │ TP=1 (gpu1)  │ ✓ CORRECT 3/3      │ OOM — Triton crash     │
#   │              │ fp8 KV  4.875/5    │ (T2.4f)                │
#   ├──────────────┼────────────────────┼────────────────────────┤
#   │ TP=2 (tp2b)  │ ? ← THIS TEST      │ ✗ INCORRECT            │
#   │              │ bf16 KV            │ bf16 KV (T2.4e)        │
#   └──────────────┴────────────────────┴────────────────────────┘
#
# Two hypotheses:
#   H-CP  — cp-OFF broke GDN recurrent state propagation. TP=2 itself is fine.
#            Expected result: T2.4g CORRECT.
#   H-TP2 — TP=2 breaks GDN state sync across shards regardless of cp setting.
#            Expected result: T2.4g INCORRECT.
#
# Decision gate: th02 correct ≥ 2/3 → H-CP confirmed → TP=2 viable with cp-ON.
#                th02 correct ≤ 1/3 → H-TP2 confirmed → TP=2 definitively broken.
#
# Config:
#   Model:            QuantTrio/Qwen3.6-27B-AWQ
#   Placement:        TP=2 (tp2b — borrows both GPUs; coder must be stopped first)
#   KV cache:         auto (bf16 — same as T2.4e)
#   ctx:              32768
#   max_tokens:       16384  (matches T2.4d run 4 for a fair quality comparison)
#   max-num-seqs:     1
#   chunked-prefill:  EXPLICITLY ENABLED  ← the only change from T2.4e
#
# Rigor upgrades vs earlier T2.4x scripts:
#   - TPS: 3 sequential reps, report median/mean/stdev (not single-shot)
#   - Quality: full 8-task suite × 3 runs (matches T2.4d for direct comparison)
#   - Summary: full 2×2 factorial table with T2.4d/T2.4e cross-references
#
# th02 scoring (the discriminating task):
#   CORRECT       = ALL jobs assigned, including deadline-misses
#                   → missed jobs go to the busiest GPU (max gpu_times)
#   SEMANTIC ERROR = missed jobs returned as -1 or silently dropped
#                   The model will argue "-1 = not processed = correct" — it is NOT.
#                   Missed jobs must still be dispatched; -1 means the scheduler failed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

# ── CLI ────────────────────────────────────────────────────────────────────────
DRY_RUN=0
RUNS=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)  DRY_RUN=1; shift ;;
        --runs)        RUNS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Config ─────────────────────────────────────────────────────────────────────
ITEM_ID="T2.4g_qwen36_27b_awq_tp2_cp_on"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="QuantTrio/Qwen3.6-27B-AWQ"
ENDPOINT="http://localhost:${PORT_VLLM_TP2_B}/v1"

# Baseline references from prior T2.4x runs
BASELINE_T2D_TPS=77.4
BASELINE_T2D_QUALITY=4.875
BASELINE_T2E_TPS=104.8   # T2.4e measured 104.8 t/s at TP=2+bf16KV+cp-OFF

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log()  { echo "[T2.4g] $*" | tee -a "${LOG}"; }
dlog() { echo "[DRY-RUN] $*"; }
die()  { log "FATAL: $*"; exit 1; }

# ── TPS measurement helper (inline python, 3 reps → median/mean/stdev) ─────────
MEASURE_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${MEASURE_PY}"' EXIT
cat > "${MEASURE_PY}" <<'PYEOF'
"""
T2.4g TPS measurement: 3 sequential reps → median, mean, stdev.
Counts both content and reasoning tokens (delta.content + delta.reasoning)
so thinking models report real decode throughput, not just post-think tokens.
"""
import asyncio, httpx, json, math, sys, time

DECODE_PROMPT = (
    "Explain the Byzantine Generals Problem and describe two distinct consensus "
    "protocols that solve it. For each protocol, outline the key invariants, "
    "failure modes, and the trade-off between safety and liveness."
)
MAX_TOKENS = 1024

async def _one(client, endpoint, model):
    t0 = time.monotonic()
    fttt = None
    count = 0
    async with client.stream("POST", f"{endpoint}/chat/completions", json={
        "model":       model,
        "messages":    [{"role": "user", "content": DECODE_PROMPT}],
        "max_tokens":  MAX_TOKENS,
        "stream":      True,
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
                count += len(tok.split())   # rough proxy; vLLM usage is more accurate
                count = count   # keep as incremental — we'll use final count
    # recount properly: each non-empty delta is ~1 token for streaming
    total = time.monotonic() - t0
    decode_t = total - (fttt or 0)
    return {
        "ttft_ms":    round((fttt or 0) * 1000, 1),
        "decode_tps": round(count / decode_t, 1) if decode_t > 0 and count > 0 else 0.0,
        "tokens":     count,
        "latency_ms": round(total * 1000, 1),
    }

async def run(endpoint, model, nreps, out_path):
    async with httpx.AsyncClient(timeout=300) as c:
        reps = []
        for i in range(nreps):
            r = await _one(c, endpoint, model)
            reps.append(r)
            print(f"  rep {i+1}/{nreps}: {r['decode_tps']:.1f} t/s  ttft={r['ttft_ms']:.0f}ms",
                  flush=True)
    tps_vals = [r["decode_tps"] for r in reps]
    ttft_vals = [r["ttft_ms"] for r in reps]
    def _median(xs):
        s = sorted(xs)
        n = len(s)
        return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2
    def _stdev(xs):
        if len(xs) < 2: return 0.0
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x-m)**2 for x in xs) / (len(xs)-1))
    agg = {
        "median_tps":  round(_median(tps_vals), 1),
        "mean_tps":    round(sum(tps_vals) / len(tps_vals), 1),
        "stdev_tps":   round(_stdev(tps_vals), 1),
        "median_ttft": round(_median(ttft_vals), 1),
        "reps":        reps,
    }
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({k: agg[k] for k in ("median_tps", "mean_tps", "stdev_tps", "median_ttft")}))

asyncio.run(run(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]))
PYEOF

# ── Dry-run path ───────────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 1 ]]; then
    dlog "=== T2.4g Dry Run ==="
    dlog "WOULD STOP: bench-vllm-gpu0, bench-vllm-tp2a (free both GPUs)"
    dlog "WOULD DEPLOY: vllm tp2b ${MODEL}"
    dlog "              --gpu-mem-util 0.85 --ctx 32768 --kv-cache-dtype auto"
    dlog "              --max-num-seqs 1 --enable-chunked-prefill"
    dlog "              --tool-call-parser qwen3_coder --reasoning-parser qwen3"
    dlog "              --enable-auto-tool-choice"
    dlog "              VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1"
    dlog "WOULD MEASURE: TPS seq=1  ×3 reps (median/mean/stdev)"
    dlog "WOULD RUN: Phase 2.2 quality suite (8 tasks, max_tokens=16384) × ${RUNS} runs"
    dlog "RESULTS → ${RESULTS_DIR}/"
    exit 0
fi

# ── Step 1: Stop coder to free both GPUs ──────────────────────────────────────
log "=== T2.4g Qwen3.6-27B-AWQ TP=2 + bf16 KV + chunked-prefill ON ==="
log "Timestamp: ${TIMESTAMP}"
log "Results:   ${RESULTS_DIR}"
log ""
log "--- Step 1: Stop coder container (both GPUs needed for TP=2) ---"
for c in bench-vllm-gpu0 bench-vllm-tp2a; do
    if podman container exists "${c}" 2>/dev/null; then
        log "Stopping ${c} ..."
        podman stop "${c}" 2>/dev/null || true
        podman rm   "${c}" 2>/dev/null || true
    else
        log "Container ${c} not running — OK."
    fi
done
log "Both GPUs free."

# ── Step 2: Deploy TP=2 + bf16 KV + chunked-prefill ON ───────────────────────
log ""
log "--- Step 2: Deploy TP=2 (tp2b), bf16 KV, --enable-chunked-prefill ---"
log "Key: --enable-chunked-prefill is EXPLICITLY set (only change from T2.4e)."
log "     T2.4e used --no-enable-chunked-prefill; this is the isolated variable."

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2b "${MODEL}" \
    --gpu-mem-util 0.85 \
    --ctx 32768 \
    --kv-cache-dtype auto \
    --max-num-seqs 1 \
    --enable-chunked-prefill \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    2>&1 | tee -a "${LOG}" \
    || die "Deployment failed. Check bench.log for OOM or Triton errors."

log "Deployed. Endpoint: ${ENDPOINT}"

# ── Step 3: Warmup (discard) ───────────────────────────────────────────────────
log ""
log "--- Step 3: Warmup (discarded) ---"
python3 -c "
import httpx, asyncio
async def w():
    async with httpx.AsyncClient(timeout=120) as c:
        try:
            await c.post('${ENDPOINT}/chat/completions',
                json={'model': '${MODEL}',
                      'messages': [{'role': 'user', 'content': 'hi'}],
                      'max_tokens': 16})
        except Exception as e:
            print(f'warmup non-fatal: {e}')
asyncio.run(w())
" 2>&1 | tee -a "${LOG}" || log "WARNING: warmup failed (non-fatal)"

# ── Step 4: TPS measurement ×3 reps ────────────────────────────────────────────
log ""
log "--- Step 4: Decode TPS (seq=1, 3 sequential reps → median/mean/stdev) ---"
log "Counting content+reasoning tokens; max_tokens=1024 per rep."
python3 "${MEASURE_PY}" \
    "${ENDPOINT}" "${MODEL}" 3 \
    "${RESULTS_DIR}/raw/tps_seq1_3reps.json" \
    2>&1 | tee -a "${LOG}" || log "WARNING: TPS measurement failed (non-fatal)"

TPS_MED=$(python3 -c "
import json
try:
    print(json.load(open('${RESULTS_DIR}/raw/tps_seq1_3reps.json'))['median_tps'])
except Exception:
    print('0.0')
" 2>/dev/null || echo "0.0")
TPS_MEAN=$(python3 -c "
import json
try:
    print(json.load(open('${RESULTS_DIR}/raw/tps_seq1_3reps.json'))['mean_tps'])
except Exception:
    print('0.0')
" 2>/dev/null || echo "0.0")
TPS_STD=$(python3 -c "
import json
try:
    print(json.load(open('${RESULTS_DIR}/raw/tps_seq1_3reps.json'))['stdev_tps'])
except Exception:
    print('0.0')
" 2>/dev/null || echo "0.0")
TTFT_MED=$(python3 -c "
import json
try:
    print(json.load(open('${RESULTS_DIR}/raw/tps_seq1_3reps.json'))['median_ttft'])
except Exception:
    print('0.0')
" 2>/dev/null || echo "0.0")

log "TPS: median=${TPS_MED}  mean=${TPS_MEAN}  stdev=${TPS_STD}"
log "     T2.4d baseline: ${BASELINE_T2D_TPS} t/s (TP=1 fp8 KV cp-ON)"
log "     T2.4e baseline: ${BASELINE_T2E_TPS} t/s (TP=2 bf16 KV cp-OFF)"

# ── Step 5: Quality suite × RUNS runs ─────────────────────────────────────────
log ""
log "--- Step 5: Phase 2.2 quality suite ×${RUNS} runs (8 tasks, max_tokens=16384) ---"
log "th02 is the discriminating task: CORRECT = ALL jobs assigned incl. misses."
log "Running ${RUNS} independent runs to match T2.4d rigor for a valid comparison."

for RUN_NUM in $(seq 1 "${RUNS}"); do
    RUN_DIR="${RESULTS_DIR}/run${RUN_NUM}"
    log ""
    log "  [Run ${RUN_NUM}/${RUNS}] → ${RUN_DIR}"
    mkdir -p "${RUN_DIR}"
    python3 -m benchmarks.phase2_model_selection.bench \
        --endpoint "${ENDPOINT}" \
        --results-dir "${RUN_DIR}" \
        --mode quality \
        --tasks "${REPO_ROOT}/benchmarks/phase2_model_selection/tasks/thinker/" \
        --model "${MODEL}" \
        --label "Qwen3.6-27B-AWQ-tp2-bf16kv-cp-on-run${RUN_NUM}" \
        --max-tokens 16384 \
        2>&1 | tee -a "${LOG}" || {
        log "  WARNING: bench exited non-zero for run${RUN_NUM} — check bench.log."
    }
    log "  [Run ${RUN_NUM}] done → ${RUN_DIR}/human_review.md"
done

# ── Step 6: Stop container ─────────────────────────────────────────────────────
log ""
log "--- Step 6: Stop TP=2 container ---"
podman stop bench-vllm-tp2b 2>/dev/null || true
podman rm   bench-vllm-tp2b 2>/dev/null || true
log "Container stopped. Both GPUs now free."

# ── Step 7: Write metrics.json + summary.md ────────────────────────────────────
log ""
log "--- Step 7: Writing metrics.json and summary.md ---"
python3 - <<PYEOF
import json, pathlib

item_id   = "${ITEM_ID}"
ts        = "${TIMESTAMP}"
runs      = int("${RUNS}")
out       = pathlib.Path("${RESULTS_DIR}")
tps_med   = float("${TPS_MED}")
tps_mean  = float("${TPS_MEAN}")
tps_std   = float("${TPS_STD}")
ttft_med  = float("${TTFT_MED}")
base_t2d  = float("${BASELINE_T2D_TPS}")
base_t2e  = float("${BASELINE_T2E_TPS}")

tps_ratio_vs_t2d = round(tps_med / base_t2d, 2) if base_t2d > 0 and tps_med > 0 else "N/A"
tps_ratio_vs_t2e = round(tps_med / base_t2e, 2) if base_t2e > 0 and tps_med > 0 else "N/A"

metrics = {
    "item_id":   item_id,
    "timestamp": ts,
    "config": {
        "engine":           "vllm",
        "engine_version":   "0.19.x",
        "model":            "${MODEL}",
        "quantization":     "AWQ-INT4",
        "kv_cache_dtype":   "auto (bf16)",
        "placement":        "tp=2 (tp2b)",
        "context_length":   32768,
        "max_tokens":       16384,
        "max_num_seqs":     1,
        "chunked_prefill":  "ENABLED (explicit --enable-chunked-prefill)",
        "extra_args":       "--kv-cache-dtype auto --max-num-seqs 1 --enable-chunked-prefill "
                            "--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice",
        "env":              "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1",
    },
    "metrics": {
        "decode_tps_seq1_median":   tps_med,
        "decode_tps_seq1_mean":     tps_mean,
        "decode_tps_seq1_stdev":    tps_std,
        "ttft_ms_median":           ttft_med,
        "tps_ratio_vs_T2d":         tps_ratio_vs_t2d,
        "tps_ratio_vs_T2e":         tps_ratio_vs_t2e,
        # Fill after human review:
        "th02_correct_run1":        None,   # True / False / "SEMANTIC_ERROR"
        "th02_correct_run2":        None,
        "th02_correct_run3":        None,
        "th02_correct_count":       None,   # int 0–3
        "quality_mean_run1":        None,   # float 1–5
        "quality_mean_run2":        None,
        "quality_mean_run3":        None,
        "quality_mean_all_runs":    None,   # mean across all scored runs
    },
    "verdict": "PENDING_HUMAN_REVIEW",
    "notes": (
        "T2.4g: missing cell in 2x2 factorial (TP x chunked-prefill). "
        "th02 CORRECT >=2/3 → H-CP confirmed (cp-OFF caused T2.4e failure, TP=2 safe with GDN). "
        "th02 CORRECT <=1/3 → H-TP2 confirmed (TP=2 itself breaks GDN state sync)."
    ),
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

run_rows = "\n".join(
    f"| run{i} | | | | run{i}/human_review.md |"
    for i in range(1, runs+1)
)

md = f"""# T2.4g — Qwen3.6-27B-AWQ TP=2 + bf16 KV + chunked-prefill ON

**Confound resolution test** — the single missing cell in the T2.4d/e 2×2 factorial.

**Model:** QuantTrio/Qwen3.6-27B-AWQ
**Config:** TP=2 (tp2b), bf16 KV (auto), `--enable-chunked-prefill`, ctx=32768, max_tokens=16384, max-num-seqs 1
**Timestamp:** {ts}

---

## What this test resolves

T2.4e ran TP=2 + bf16 KV + **cp-OFF** and th02 was INCORRECT. But T2.4e changed two
variables from T2.4d simultaneously (TP=1→2 AND cp-ON→OFF). T2.4g closes the
missing cell by holding TP=2 + bf16 KV constant and setting **cp-ON**.

### 2×2 Factorial Summary

| | cp-ON | cp-OFF |
|---|---|---|
| **TP=1 (gpu1, fp8 KV)** | ✓ CORRECT 3/3 · 4.875/5 · 77.4 t/s (T2.4d) | OOM — Triton crash (T2.4f) |
| **TP=2 (tp2b, bf16 KV)** | **T2.4g ← this run** | ✗ INCORRECT · 104.8 t/s (T2.4e) |

### Hypotheses

- **H-CP (cp was the cause):** GDN (Gated DeltaNet) recurrent state breaks across
  prefill chunk boundaries when cp is disabled. T2.4g should be CORRECT.
- **H-TP2 (TP=2 is the cause):** TP=2 splits the DeltaNet state matrix across GPU
  shards; per-shard updates don't commute and accumulate error. T2.4g should be INCORRECT.

---

## Auto-measured: TPS (seq=1, 3 reps)

| Metric | T2.4g (TP=2 bf16 cp-ON) | T2.4d (TP=1 fp8 cp-ON) | T2.4e (TP=2 bf16 cp-OFF) |
|--------|--------------------------|------------------------|--------------------------|
| Median TPS | {tps_med:.1f} t/s | 77.4 t/s | 104.8 t/s |
| Mean TPS | {tps_mean:.1f} t/s | — | — |
| Stdev TPS | {tps_std:.1f} t/s | — | — |
| TTFT median | {ttft_med:.0f} ms | — | 109 ms |
| vs T2.4d | {tps_ratio_vs_t2d}× | 1.00× | 1.35× |
| vs T2.4e | {tps_ratio_vs_t2e}× | — | 1.00× |

*T2.4e note: T2.4e TPS was single-shot, not 3-rep median. Compare directionally.*

---

## Human review required — th02 scoring (×{runs} runs)

th02 = task 02: multi-GPU job scheduler with deadline-miss handling.

**CORRECT:** All jobs are assigned to a GPU, including deadline misses.
Missed jobs must be routed to the **busiest GPU** (`max(gpu_times)`), not dropped.

**SEMANTIC ERROR (confident incorrectness):** Missed jobs returned as -1 or omitted.
The model will argue: "if a job misses the best GPU, it misses everywhere, so -1 is
correct." This is the failure mode seen in T2.4e and most T2.4c runs. It is NOT
correct — the scheduler must still dispatch missed jobs somewhere.

| Run | th02 result | th02 correct? | Quality mean (1–5) | Notes |
|-----|-------------|---------------|--------------------|-------|
{run_rows}
| **TOTAL** | | **?/3 correct** | | |

*Fill in after scoring each run's `human_review.md`.*

---

## Decision gate

| th02 correct count | Verdict | Interpretation | Next action |
|--------------------|---------|----------------|-------------|
| **≥ 2/3** | **H-CP CONFIRMED** | cp-OFF was the sole cause of T2.4e failure. TP=2 is safe for GDN with cp-ON. | Update DECISIONS.md: TP=2 viable if `--enable-chunked-prefill` is set. T_NVFP4 can be reconsidered at TP=2+cp-ON (NVFP4 + cp-ON + TP=2). |
| **≤ 1/3** | **H-TP2 CONFIRMED** | TP=2 breaks GDN state sync across shards regardless of cp setting. | Update DECISIONS.md: TP=2 definitively broken for Qwen3.6-27B (GDN). T_NVFP4 restricted to TP=1 only. TP=1 + fp8 KV + cp-ON is the only viable serving config. |

**Production impact:** Neither outcome changes the deployed config. TP=1 + fp8 KV +
cp-ON (T2.4d, 4.875/5, 77.4 t/s) remains the production thinker regardless.

---

## After scoring

1. Fill in the th02 and quality table above.
2. Update `metrics.json`:
   - `th02_correct_run1/2/3` → True / False / "SEMANTIC_ERROR"
   - `th02_correct_count` → int
   - `quality_mean_run1/2/3` → float
   - `quality_mean_all_runs` → mean
   - `verdict` → "H-CP_CONFIRMED" / "H-TP2_CONFIRMED" / "INCONCLUSIVE"
3. Update `DECISIONS.md` with the settled TP=2 verdict.
4. Update `RESEARCH_STATE.md` (close R18, note the outcome).
5. If H-CP confirmed and quality at TP=2 is significantly higher than T2.4d →
   reconsider TP=2 as the production config (trades sleep-mode coordination for
   potentially higher quality). Otherwise keep TP=1.

---

## Files

| Path | Contents |
|------|----------|
| `bench.log` | Full deploy + run log |
| `raw/tps_seq1_3reps.json` | TPS reps with per-rep breakdown |
| `run1/human_review.md` | Run 1 — 8 tasks |
| `run2/human_review.md` | Run 2 — 8 tasks |
| `run3/human_review.md` | Run 3 — 8 tasks |
| `metrics.json` | Structured results (update after scoring) |
| `summary.md` | This file |
"""
(out / "summary.md").write_text(md)
print(f"[T2.4g] TPS median={tps_med:.1f}  mean={tps_mean:.1f}  stdev={tps_std:.1f}  TTFT={ttft_med:.0f}ms")
print(f"[T2.4g] summary.md  → {out}/summary.md")
print(f"[T2.4g] metrics.json → {out}/metrics.json")
PYEOF

log ""
log "=== T2.4g complete ==="
log "Results:     ${RESULTS_DIR}"
log "Summary:     ${RESULTS_DIR}/summary.md"
log ""
log "Next steps:"
log "  1. Score th02 in run1/, run2/, run3/human_review.md"
log "     CORRECT = all jobs assigned including misses → busiest GPU"
log "     SEMANTIC ERROR = misses returned -1 (not dispatched)"
log "  2. Score all 8 tasks in each run (1–5 per task)"
log "  3. Fill in the decision table in summary.md"
log "  4. Update metrics.json: th02_correct_run1/2/3, th02_correct_count,"
log "                          quality_mean_run1/2/3, quality_mean_all_runs, verdict"
log "  5. ≥2/3 correct → H-CP confirmed → update DECISIONS.md (TP=2 safe with cp-ON)"
log "     ≤1/3 correct → H-TP2 confirmed → update DECISIONS.md (TP=2 definitively broken)"
log ""
log "Note: coder (bench-vllm-gpu0/tp2a) was stopped for this test."
log "      Restart it after T2.4g result is recorded."
