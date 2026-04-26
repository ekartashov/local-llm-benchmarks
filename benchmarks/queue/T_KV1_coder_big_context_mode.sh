#!/usr/bin/env bash
# T_KV1_coder_big_context_mode.sh
#
# T_KV1 — Coder Extended Context Mode
#
# Measures the maximum usable context for Arclight Coder when running TP=2
# (thinker sleeping at level=1). Tests prefill TTFT and decode TPS across
# a context sweep, then tests swap-space extension.
#
# ARCHITECTURE CONTEXT:
#   Hot pair: Coder TP=1 GPU0, Thinker TP=1 GPU1
#   Extended Arclight: Thinker sleeps → Coder runs TP=2 across both GPUs
#   Combined VRAM: ~64GB − weights (~23GB) − sleep residual (~4GB) = ~37GB KV
#   fp8 KV budget: ~37GB → estimated 60–75K tokens
#   This test verifies that estimate.
#
# PROCEDURE:
#   1. Sleep thinker at level=1 (must be running on port 30001)
#   2. Stop coder TP=1 container (port 30000)
#   3. Deploy coder TP=2 --max-model-len 65536 --kv-cache-dtype fp8
#   4. Record VRAM split via nvidia-smi
#   5. Context sweep [8192, 16384, 32768, 65536] — TTFT + decode TPS each
#   6. Swap extension: restart with --swap-space 32, --max-model-len 131072
#   7. Test at 98304 tokens (75% of 131K) — TTFT + TPS with KV spill to DRAM
#
# PASS CRITERION:
#   max context without swap >= 60K tokens
#   TPS at 65K context within 20% of 32K baseline
#
# OPTIONS:
#   --skip-sleep    Skip sleeping thinker (assume already sleeping)
#   --skip-swap     Skip the 131K swap-space extension test
#   --dry-run       Print commands without executing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

SKIP_SLEEP=0
SKIP_SWAP=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-sleep) SKIP_SLEEP=1; shift ;;
        --skip-swap)  SKIP_SWAP=1;  shift ;;
        --dry-run|-n) DRY_RUN=1;    shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ITEM_ID="T_KV1_coder_big_context_mode"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
THINKER_PORT="${PORT_VLLM_TP2_B:-30001}"
CODER_TP2_PORT="${PORT_VLLM_TP2_A:-30000}"
CODER_TP2_ENDPOINT="http://localhost:${CODER_TP2_PORT}/v1"

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log()  { echo "[T_KV1 $(date -u +%H:%M:%S)] $*" | tee -a "${LOG}"; }
die()  { log "FATAL: $*"; exit 1; }
run()  { if [[ "${DRY_RUN}" -eq 1 ]]; then echo "[dry-run] $*"; else "$@"; fi; }

log "=== T_KV1 Coder Big Context Mode ==="
log "Results dir: ${RESULTS_DIR}"
log "Model: ${MODEL}"

# ── Prerequisite checks ────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 0 ]]; then
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found — run on host, not in container"
    command -v podman >/dev/null || die "podman not found"
fi

# ── Step 1: Sleep thinker ──────────────────────────────────────────────────────
if [[ "${SKIP_SLEEP}" -eq 0 ]]; then
    log "Sleeping thinker (level=1) on port ${THINKER_PORT}..."
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        curl -sf "http://localhost:${THINKER_PORT}/v1/models" >/dev/null \
            || die "Thinker not responding on port ${THINKER_PORT}. Start it first."
        curl -sf -X POST "http://localhost:${THINKER_PORT}/sleep?level=1" | tee -a "${LOG}"
        # Wait until is_sleeping: true
        for i in $(seq 1 30); do
            sleeping="$(curl -sf "http://localhost:${THINKER_PORT}/is_sleeping" \
                        | python3 -c 'import sys,json; print(json.load(sys.stdin)["is_sleeping"])')"
            [[ "${sleeping}" == "True" ]] && break
            sleep 1
        done
        [[ "${sleeping}" == "True" ]] || die "Thinker failed to sleep after 30s"
        log "Thinker sleeping. VRAM freed."
    fi
else
    log "Skipping sleep (--skip-sleep)."
fi

# ── Step 2: Stop coder TP=1 ────────────────────────────────────────────────────
log "Stopping coder TP=1 container (bench-vllm-tp2a)..."
run podman stop bench-vllm-tp2a 2>/dev/null || true
sleep 3

# ── Step 3: Deploy coder TP=2 with extended context ───────────────────────────
log "Deploying coder TP=2 with --max-model-len 65536, fp8 KV..."
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_V1_ENABLED=0 \
run "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" \
    --gpu-mem-util 0.90 \
    --ctx 65536 \
    --kv-cache-dtype fp8 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice

if [[ "${DRY_RUN}" -eq 0 ]]; then
    log "Waiting for coder TP=2 to be ready..."
    for i in $(seq 1 120); do
        curl -sf "${CODER_TP2_ENDPOINT}/models" >/dev/null 2>&1 && break
        sleep 2
    done
    curl -sf "${CODER_TP2_ENDPOINT}/models" >/dev/null \
        || die "Coder TP=2 did not become ready in 240s"
    log "Coder TP=2 is ready."
fi

# ── Step 4: Record VRAM split ──────────────────────────────────────────────────
log "VRAM state after TP=2 deployment:"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
        --format=csv,noheader | tee -a "${LOG}" > "${RESULTS_DIR}/vram_after_deploy.txt"
fi

# ── Embedded Python: context sweep ────────────────────────────────────────────
SWEEP_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${SWEEP_PY}"' EXIT
cat > "${SWEEP_PY}" <<'PYEOF'
"""
T_KV1 context sweep: for each target context length, send a padded prompt,
measure TTFT and decode TPS, detect OOM errors.
"""
import asyncio, httpx, json, os, sys, time

ENDPOINT    = sys.argv[1]
MODEL       = sys.argv[2]
OUT_PATH    = sys.argv[3]
CTX_TARGETS = [int(x) for x in sys.argv[4].split(",")]

# Build a padded prompt that fills ~70% of the target context.
# Token density: ~3.5 chars/token for code+prose mix (conservative).
CHARS_PER_TOKEN = 3.5
PAD_UNIT = (
    "The following text is a verbatim extract from the Linux kernel documentation "
    "on memory management, included here as context filler for a throughput benchmark. "
    "Page frame reclamation (PFR) occurs when the available free pages in a zone drop "
    "below a watermark threshold. The kswapd daemon wakes up and begins reclaiming "
    "pages from the LRU lists: active_anon, inactive_anon, active_file, inactive_file. "
    "The cost of reclamation depends on whether pages are clean (can be freed immediately) "
    "or dirty (must be written to the backing store before being freed). Swap preference "
    "is controlled by /proc/sys/vm/swappiness. A value of 0 disables anonymous swap "
    "entirely; 200 (kernel 5.8+) aggressively swaps anonymous memory to prefer file cache. "
)  # ~150 tokens

QUESTION = (
    "\n\n---\n\nBased on the above context, answer the following question in one paragraph: "
    "What is the primary trade-off between swappiness=0 and swappiness=200 for a workload "
    "that runs a mix of long-lived ML inference processes and short database queries?"
)

async def sweep_one(client, ctx_tokens, out_dir):
    """Send a prompt targeting ctx_tokens, measure TTFT + decode TPS."""
    target_pad_tokens = int(ctx_tokens * 0.70)
    pad_chars = int(target_pad_tokens * CHARS_PER_TOKEN)
    reps = (pad_chars // len(PAD_UNIT)) + 1
    padded = (PAD_UNIT * reps)[:pad_chars]
    prompt = padded + QUESTION

    t0 = time.monotonic()
    fttt = None
    count = 0
    error = None

    try:
        async with client.stream("POST", f"{ENDPOINT}/chat/completions", json={
            "model":       MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  512,
            "temperature": 0.0,
            "stream":      True,
        }) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                error = f"HTTP {resp.status_code}: {body[:200].decode('utf-8', errors='replace')}"
                raise RuntimeError(error)
            async for raw in resp.aiter_lines():
                if not raw.startswith("data: ") or "[DONE]" in raw:
                    continue
                delta = json.loads(raw[6:])["choices"][0]["delta"]
                tok = delta.get("content") or delta.get("reasoning") or ""
                if tok:
                    if fttt is None:
                        fttt = time.monotonic() - t0
                    count += 1
    except Exception as exc:
        error = str(exc)

    total = time.monotonic() - t0
    decode_s = total - (fttt or 0)
    result = {
        "ctx_target_tokens": ctx_tokens,
        "prompt_chars":       len(prompt),
        "ttft_ms":            round((fttt or 0) * 1000, 1),
        "decode_tps":         round(count / decode_s, 1) if decode_s > 0 and count > 0 else 0.0,
        "output_tokens":      count,
        "total_ms":           round(total * 1000, 1),
        "error":              error,
    }
    fname = os.path.join(out_dir, f"ctx_{ctx_tokens}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    status = f"ttft={result['ttft_ms']:.0f}ms  tps={result['decode_tps']:.1f}"
    if error:
        status = f"ERROR: {error[:80]}"
    print(f"  ctx={ctx_tokens:>6}:  {status}", flush=True)
    return result

async def main():
    all_results = []
    async with httpx.AsyncClient(timeout=600) as client:
        for ctx in CTX_TARGETS:
            print(f"[T_KV1] Sweeping ctx={ctx}...", flush=True)
            r = await sweep_one(client, ctx, os.path.dirname(OUT_PATH))
            all_results.append(r)
            if r["error"] and "OOM" in (r["error"] or ""):
                print(f"  OOM at ctx={ctx} — stopping sweep.", flush=True)
                break
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    # Print summary table
    print("\n  ctx_tokens  | ttft_ms  | tps    | error")
    print("  -----------   --------   ------   -----")
    for r in all_results:
        err = (r["error"] or "")[:30] if r["error"] else "—"
        print(f"  {r['ctx_target_tokens']:>10}  | {r['ttft_ms']:>8.0f} | {r['decode_tps']:>6.1f} | {err}")

asyncio.run(main())
PYEOF

# ── Step 5: Run context sweep (no swap) ───────────────────────────────────────
log "Running context sweep [8192, 16384, 32768, 65536]..."
if [[ "${DRY_RUN}" -eq 0 ]]; then
    python3 "${SWEEP_PY}" \
        "${CODER_TP2_ENDPOINT}" \
        "${MODEL}" \
        "${RESULTS_DIR}/raw/sweep_no_swap.json" \
        "8192,16384,32768,65536" \
        2>&1 | tee -a "${LOG}"
else
    echo "[dry-run] python3 sweep_py ... sweep=[8192,16384,32768,65536]"
fi

# ── Step 6: Swap-space extension (optional) ───────────────────────────────────
if [[ "${SKIP_SWAP}" -eq 0 ]]; then
    log "Restarting coder TP=2 with --swap-space 32, --max-model-len 131072..."
    run podman stop bench-vllm-tp2a 2>/dev/null || true
    sleep 5
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    VLLM_V1_ENABLED=0 \
    run "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" \
        --gpu-mem-util 0.90 \
        --ctx 131072 \
        --kv-cache-dtype fp8 \
        --swap-space 32 \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice

    if [[ "${DRY_RUN}" -eq 0 ]]; then
        log "Waiting for swap-extended coder to be ready..."
        for i in $(seq 1 120); do
            curl -sf "${CODER_TP2_ENDPOINT}/models" >/dev/null 2>&1 && break
            sleep 2
        done
        curl -sf "${CODER_TP2_ENDPOINT}/models" >/dev/null \
            || { log "WARNING: Swap-extended coder did not start — skipping swap test"; SKIP_SWAP=1; }
    fi

    if [[ "${SKIP_SWAP}" -eq 0 ]]; then
        log "Testing swap-extended context [65536, 98304, 131072]..."
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            python3 "${SWEEP_PY}" \
                "${CODER_TP2_ENDPOINT}" \
                "${MODEL}" \
                "${RESULTS_DIR}/raw/sweep_with_swap.json" \
                "65536,98304,131072" \
                2>&1 | tee -a "${LOG}"
        else
            echo "[dry-run] python3 sweep_py ... sweep=[65536,98304,131072] (with swap)"
        fi
    fi
fi

# ── Step 7: metrics.json ───────────────────────────────────────────────────────
log "Writing metrics.json..."
if [[ "${DRY_RUN}" -eq 0 ]]; then
    python3 - <<PYEOF
import json, os, pathlib

results_dir = pathlib.Path("${RESULTS_DIR}")
raw_dir = results_dir / "raw"

no_swap = json.loads((raw_dir / "sweep_no_swap.json").read_text()) \
    if (raw_dir / "sweep_no_swap.json").exists() else []
with_swap = json.loads((raw_dir / "sweep_with_swap.json").read_text()) \
    if (raw_dir / "sweep_with_swap.json").exists() else []

# Find max context without OOM in no-swap run
max_ctx_no_swap = max(
    (r["ctx_target_tokens"] for r in no_swap if not r.get("error")),
    default=None
)
baseline_32k = next((r for r in no_swap if r["ctx_target_tokens"] == 32768), None)
peak_65k     = next((r for r in no_swap if r["ctx_target_tokens"] == 65536), None)

tps_regression = None
if baseline_32k and peak_65k and baseline_32k["decode_tps"] > 0 and not peak_65k.get("error"):
    tps_regression = round(
        (peak_65k["decode_tps"] - baseline_32k["decode_tps"]) / baseline_32k["decode_tps"] * 100, 1
    )

pass_criteria_met = (
    max_ctx_no_swap is not None and
    max_ctx_no_swap >= 60000 and
    (tps_regression is None or tps_regression >= -20)
)

metrics = {
    "item_id": "${ITEM_ID}",
    "timestamp": "${TIMESTAMP}",
    "config": {
        "engine":          "vllm",
        "model":           "${MODEL}",
        "quantization":    "AWQ-INT4",
        "kv_cache_dtype":  "fp8",
        "placement":       "tp=2",
        "ctx_no_swap":     65536,
        "ctx_with_swap":   131072,
        "swap_space_gb":   32,
        "extra_args":      "VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1",
    },
    "metrics": {
        "context_sweep_no_swap": no_swap,
        "context_sweep_with_swap": with_swap,
        "max_ctx_no_swap_tokens": max_ctx_no_swap,
        "tps_regression_32k_to_65k_pct": tps_regression,
    },
    "verdict": "PASS" if pass_criteria_met else ("FAIL" if max_ctx_no_swap is not None else "INCONCLUSIVE"),
    "notes": (
        f"Max ctx no-swap: {max_ctx_no_swap}. "
        f"TPS regression 32K→65K: {tps_regression}%." if tps_regression else
        f"Max ctx no-swap: {max_ctx_no_swap}."
    ),
}
(results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("metrics.json written")
print(f"  verdict:             {metrics['verdict']}")
print(f"  max ctx (no swap):   {max_ctx_no_swap}")
print(f"  TPS regression:      {tps_regression}%")
PYEOF
fi

# ── Step 8: summary.md ────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 0 ]] && [[ -f "${RESULTS_DIR}/metrics.json" ]]; then
    python3 - <<PYEOF
import json, pathlib

m = json.loads(pathlib.Path("${RESULTS_DIR}/metrics.json").read_text())
sweep = m["metrics"].get("context_sweep_no_swap", [])
swap  = m["metrics"].get("context_sweep_with_swap", [])

lines = [
    "# T_KV1 — Coder Big Context Mode",
    f"**Timestamp:** ${TIMESTAMP}",
    f"**Model:** ${MODEL}",
    f"**Verdict:** {m['verdict']}",
    "",
    "## Context Sweep (no swap)",
    "| ctx tokens | TTFT ms | TPS | error |",
    "|-----------|---------|-----|-------|",
]
for r in sweep:
    err = r.get("error") or "—"
    lines.append(f"| {r['ctx_target_tokens']:>10} | {r['ttft_ms']:>7.0f} | {r['decode_tps']:>5.1f} | {err} |")

if swap:
    lines += [
        "",
        "## Context Sweep (swap-space 32GB, max-model-len 131072)",
        "| ctx tokens | TTFT ms | TPS | error |",
        "|-----------|---------|-----|-------|",
    ]
    for r in swap:
        err = r.get("error") or "—"
        lines.append(f"| {r['ctx_target_tokens']:>10} | {r['ttft_ms']:>7.0f} | {r['decode_tps']:>5.1f} | {err} |")

lines += [
    "",
    "## Notes",
    f"- Max usable context (no swap): **{m['metrics']['max_ctx_no_swap_tokens']} tokens**",
    f"- TPS regression 32K→65K: **{m['metrics']['tps_regression_32k_to_65k_pct']}%**",
    "",
    m['notes'],
]
pathlib.Path("${RESULTS_DIR}/summary.md").write_text("\n".join(lines))
print("summary.md written")
PYEOF
fi

log "=== T_KV1 complete. Results: ${RESULTS_DIR}/ ==="
log "Next: record results in TESTING_QUEUE.md, update DECISIONS.md with max context."
log "Hand-back trigger: if max ctx < 60K, KV budget estimate was wrong — re-run nvidia-smi."
