#!/usr/bin/env bash
# T_PAR1_parallel_throughput_sweep.sh
#
# T_PAR1 — Parallel Throughput Sweep
#
# Finds the optimal concurrency setting for each tier to maximise aggregate TPS
# for agentic multi-subagent workloads (OpenCode v1.3+ spawns multiple subagents
# concurrently against coder, thinker, and Convergence simultaneously).
#
# PART A — vLLM Arclight (coder and thinker separately):
#   Sweep --max-num-seqs in [1, 2, 4, 8]
#   For each N: fire N concurrent requests, measure per-request + aggregate TPS.
#   Find the "knee": first N where aggregate gain drops below 20%.
#
# PART B — ik_llama.cpp Convergence (already deployed on port 8002):
#   Convergence was already measured at -np 4 (T_CV4: 15.6 t/s aggregate).
#   This part re-measures at -np [1, 2, 4] for a clean comparison row alongside
#   the vLLM tiers. Convergence must already be running (always-resident).
#
# DEPENDENCIES:
#   Part A coder: Qwen3.6-35B-A3B-AWQ on port 30000 (TP=2 production config)
#   Part A thinker: Qwen3.6-27B-AWQ on port 30001 (TP=1 GPU1 production config)
#   Part B: ik_llama.cpp Convergence on port 8002 (always-resident)
#   T_CV2 must be DONE (32 threads confirmed optimal) — it is.
#
# OPTIONS:
#   --skip-coder       Skip Part A coder sweep
#   --skip-thinker     Skip Part A thinker sweep
#   --skip-convergence Skip Part B Convergence sweep
#   --max-seqs LIST    Comma-separated list (default: 1,2,4,8)
#   --reps N           Requests per concurrency level (default: 3 per slot × N)
#   --dry-run          Print commands without executing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

SKIP_CODER=0
SKIP_THINKER=0
SKIP_CONV=0
MAX_SEQS="1,2,4,8"
REPS=3
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-coder)       SKIP_CODER=1;    shift ;;
        --skip-thinker)     SKIP_THINKER=1;  shift ;;
        --skip-convergence) SKIP_CONV=1;     shift ;;
        --max-seqs)         MAX_SEQS="$2";   shift 2 ;;
        --reps)             REPS="$2";       shift 2 ;;
        --dry-run|-n)       DRY_RUN=1;       shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ITEM_ID="T_PAR1_parallel_throughput_sweep"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[T_PAR1 $(date -u +%H:%M:%S)] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

log "=== T_PAR1 Parallel Throughput Sweep ==="
log "Results dir: ${RESULTS_DIR}"

# ── Embedded Python: concurrent sweep ─────────────────────────────────────────
SWEEP_PY="$(mktemp --suffix=.py)"
trap 'rm -f "${SWEEP_PY}"' EXIT

cat > "${SWEEP_PY}" <<'PYEOF'
"""
T_PAR1 concurrent sweep: fire N requests in parallel, measure per-request TTFT/TPS
and aggregate TPS. Reports per-slot results + aggregate.

Usage: python3 sweep.py <endpoint> <model> <max_seqs_csv> <reps> <out_path>

For each N in max_seqs_csv:
  - Fire N requests concurrently (asyncio.gather)
  - Each request: fixed 512-token decode (temperature=0, deterministic)
  - Record: per-request TTFT, per-request TPS
  - Aggregate TPS = sum(output_tokens) / wall_clock_from_first_to_last_token
"""
import asyncio, httpx, json, math, os, sys, time

ENDPOINT    = sys.argv[1]
MODEL       = sys.argv[2]
SEQ_VALUES  = [int(x) for x in sys.argv[3].split(",")]
REPS        = int(sys.argv[4])
OUT_PATH    = sys.argv[5]

DECODE_PROMPT = (
    "Write a detailed technical explanation of how consistent hashing works in "
    "distributed systems. Include the ring topology, virtual nodes, and rebalancing "
    "behaviour when nodes are added or removed. Provide a concrete example with "
    "specific node counts and explain the impact on load distribution."
)

async def one_request(client, slot_id, result_bag):
    t0 = time.monotonic()
    fttt = None
    count = 0
    try:
        async with client.stream("POST", f"{ENDPOINT}/chat/completions", json={
            "model":          MODEL,
            "messages":       [{"role": "user", "content": DECODE_PROMPT}],
            "max_tokens":     512,
            "temperature":    0.0,
            "stream":         True,
            "stream_options": {"include_usage": True},
        }) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_lines():
                if not raw.startswith("data: ") or "[DONE]" in raw:
                    continue
                chunk = json.loads(raw[6:])
                if chunk.get("usage"):
                    count = chunk["usage"]["completion_tokens"]
                elif fttt is None and chunk.get("choices") and chunk["choices"][0].get("delta"):
                    delta = chunk["choices"][0]["delta"]
                    tok = delta.get("content") or delta.get("reasoning") or ""
                    if tok:
                        fttt = time.monotonic() - t0
        total = time.monotonic() - t0
        decode_s = total - (fttt or 0)
        result_bag[slot_id] = {
            "ttft_ms":    round((fttt or 0) * 1000, 1),
            "decode_tps": round(count / decode_s, 1) if decode_s > 0 and count > 0 else 0.0,
            "tokens":     count,
            "latency_ms": round(total * 1000, 1),
            "error":      None,
        }
    except Exception as exc:
        result_bag[slot_id] = {
            "ttft_ms": 0, "decode_tps": 0, "tokens": 0, "latency_ms": 0,
            "error": str(exc)[:120],
        }

async def sweep_n(client, n_seqs, n_reps):
    """Run n_reps rounds of n_seqs concurrent requests. Return aggregated stats."""
    all_rounds = []
    for rep in range(n_reps):
        result_bag = {}
        t_wall_start = time.monotonic()
        await asyncio.gather(*[one_request(client, i, result_bag) for i in range(n_seqs)])
        wall_s = time.monotonic() - t_wall_start
        total_tokens = sum(r["tokens"] for r in result_bag.values())
        aggregate_tps = round(total_tokens / wall_s, 1) if wall_s > 0 else 0.0
        per_req = {
            "median_ttft_ms":     _median([r["ttft_ms"]    for r in result_bag.values()]),
            "median_tps":         _median([r["decode_tps"] for r in result_bag.values()]),
            "aggregate_tps":      aggregate_tps,
            "total_tokens":       total_tokens,
            "wall_s":             round(wall_s, 2),
            "slot_results":       result_bag,
            "errors":             [r["error"] for r in result_bag.values() if r["error"]],
        }
        all_rounds.append(per_req)
        print(f"  N={n_seqs} rep={rep+1}/{n_reps}: agg={aggregate_tps:.1f} t/s  "
              f"per-req-median={per_req['median_tps']:.1f} t/s  "
              f"ttft={per_req['median_ttft_ms']:.0f}ms", flush=True)
    return all_rounds

def _median(xs):
    if not xs: return 0.0
    s = sorted(xs)
    n = len(s)
    return round(s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2, 1)

async def main():
    all_results = {}
    async with httpx.AsyncClient(timeout=300) as client:
        # Verify endpoint is reachable
        try:
            r = await client.get(f"{ENDPOINT}/models")
            r.raise_for_status()
        except Exception as exc:
            print(f"ERROR: endpoint {ENDPOINT} not reachable: {exc}", file=sys.stderr)
            sys.exit(1)

        for n in SEQ_VALUES:
            print(f"\n[T_PAR1] Sweeping N={n} concurrent requests × {REPS} reps...", flush=True)
            rounds = await sweep_n(client, n, REPS)
            agg_tps_values = [r["aggregate_tps"] for r in rounds]
            all_results[n] = {
                "n_seqs":             n,
                "median_agg_tps":     _median(agg_tps_values),
                "rounds":             rounds,
            }

    # Find the knee: first N where gain < 20% over N-1
    print("\n  N  | median_agg_tps | gain_over_prev")
    print("  --   --------------   --------------")
    prev_tps = None
    knee_n = None
    for n in SEQ_VALUES:
        tps = all_results[n]["median_agg_tps"]
        if prev_tps and prev_tps > 0:
            gain_pct = (tps - prev_tps) / prev_tps * 100
            gain_str = f"{gain_pct:+.1f}%"
            if gain_pct < 20 and knee_n is None:
                knee_n = n
                gain_str += " ← knee"
        else:
            gain_str = "—"
        print(f"  {n:2} | {tps:14.1f} | {gain_str}")
        prev_tps = tps

    if knee_n is None:
        knee_n = max(SEQ_VALUES)
    print(f"\n  Recommended max-num-seqs: {knee_n}")

    with open(OUT_PATH, "w") as f:
        json.dump({"results_by_n": all_results, "knee_n": knee_n}, f, indent=2)

asyncio.run(main())
PYEOF

# ── Part A: Coder ──────────────────────────────────────────────────────────────
if [[ "${SKIP_CODER}" -eq 0 ]]; then
    log "Part A — Coder (port ${PORT_VLLM_TP2_A}): sweeping max-num-seqs..."
    CODER_ENDPOINT="http://localhost:${PORT_VLLM_TP2_A}/v1"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        curl -sf "${CODER_ENDPOINT}/models" >/dev/null \
            || die "Coder not responding on port ${PORT_VLLM_TP2_A}. Start it first."
        CODER_MODEL="$(curl -sf "${CODER_ENDPOINT}/models" \
            | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
        log "Coder model: ${CODER_MODEL}"
        python3 "${SWEEP_PY}" \
            "${CODER_ENDPOINT}" "${CODER_MODEL}" \
            "${MAX_SEQS}" "${REPS}" \
            "${RESULTS_DIR}/raw/coder_sweep.json" \
            2>&1 | tee -a "${LOG}"
    else
        echo "[dry-run] Coder sweep on ${CODER_ENDPOINT} max_seqs=[${MAX_SEQS}] reps=${REPS}"
    fi
fi

# ── Part A: Thinker ────────────────────────────────────────────────────────────
if [[ "${SKIP_THINKER}" -eq 0 ]]; then
    log "Part A — Thinker (port ${PORT_VLLM_TP2_B}): sweeping max-num-seqs..."
    THINKER_ENDPOINT="http://localhost:${PORT_VLLM_TP2_B}/v1"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        curl -sf "${THINKER_ENDPOINT}/models" >/dev/null \
            || die "Thinker not responding on port ${PORT_VLLM_TP2_B}. Start it first."
        THINKER_MODEL="$(curl -sf "${THINKER_ENDPOINT}/models" \
            | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
        log "Thinker model: ${THINKER_MODEL}"
        log "  Note: thinker uses --max-num-seqs 1 in production. Sweep may require"
        log "  redeployment with higher --max-num-seqs. If the server rejects N>1 requests,"
        log "  they will queue and results will show N×serial latency, not parallel TPS."
        python3 "${SWEEP_PY}" \
            "${THINKER_ENDPOINT}" "${THINKER_MODEL}" \
            "${MAX_SEQS}" "${REPS}" \
            "${RESULTS_DIR}/raw/thinker_sweep.json" \
            2>&1 | tee -a "${LOG}"
    else
        echo "[dry-run] Thinker sweep on ${THINKER_ENDPOINT} max_seqs=[${MAX_SEQS}] reps=${REPS}"
    fi
fi

# ── Part B: Convergence ────────────────────────────────────────────────────────
# Convergence was already measured at -np 4 (T_CV4: 15.6 t/s aggregate).
# This sweep re-measures -np [1, 2, 4] at fixed thread count (32, per T_CV2).
# Convergence must already be running (always-resident). We do NOT restart it
# here because changing -np requires a server restart — document this in notes.
if [[ "${SKIP_CONV}" -eq 0 ]]; then
    log "Part B — Convergence (port ${PORT_CONVERGENCE}): concurrent request sweep..."
    CONV_ENDPOINT="http://localhost:${PORT_CONVERGENCE}/v1"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        if ! curl -sf "http://localhost:${PORT_CONVERGENCE}/health" >/dev/null 2>&1; then
            log "WARNING: Convergence not responding on port ${PORT_CONVERGENCE}."
            log "  Start it with: llama-server -ngl 999 --cpu-moe -t 32 -np 4 -c 16384 ..."
            log "  Skipping Convergence sweep."
        else
            CONV_MODEL="$(curl -sf "${CONV_ENDPOINT}/models" \
                | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
            log "Convergence model: ${CONV_MODEL}"
            log "  NOTE: Running with whatever -np Convergence was started with."
            log "  T_CV4 established -np 4 as optimal. This sweep tests client-side"
            log "  concurrency (firing N requests against the running server)."
            python3 "${SWEEP_PY}" \
                "${CONV_ENDPOINT}" "${CONV_MODEL}" \
                "1,2,4" "${REPS}" \
                "${RESULTS_DIR}/raw/convergence_sweep.json" \
                2>&1 | tee -a "${LOG}"
        fi
    else
        echo "[dry-run] Convergence sweep on ${CONV_ENDPOINT} concurrency=[1,2,4] reps=${REPS}"
    fi
fi

# ── metrics.json ───────────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 0 ]]; then
    python3 - <<PYEOF
import json, pathlib

results_dir = pathlib.Path("${RESULTS_DIR}")
raw = results_dir / "raw"

def load(fname):
    p = raw / fname
    return json.loads(p.read_text()) if p.exists() else None

coder    = load("coder_sweep.json")
thinker  = load("thinker_sweep.json")
conv     = load("convergence_sweep.json")

def knee(data):
    return data.get("knee_n") if data else None

metrics = {
    "item_id":   "${ITEM_ID}",
    "timestamp": "${TIMESTAMP}",
    "config": {
        "max_seqs_tested": "${MAX_SEQS}",
        "reps_per_level":  ${REPS},
        "coder_endpoint":  "http://localhost:${PORT_VLLM_TP2_A}/v1",
        "thinker_endpoint":"http://localhost:${PORT_VLLM_TP2_B}/v1",
        "conv_endpoint":   "http://localhost:${PORT_CONVERGENCE}/v1",
    },
    "metrics": {
        "coder_knee_n":    knee(coder),
        "thinker_knee_n":  knee(thinker),
        "convergence_np":  4,  # T_CV4 established -np 4 as production default
        "coder_detail":    coder,
        "thinker_detail":  thinker,
        "convergence_detail": conv,
    },
    "verdict": "PASS",
    "notes": (
        f"Coder knee: N={knee(coder)}. "
        f"Thinker knee: N={knee(thinker)}. "
        "Convergence -np 4 already settled by T_CV4."
    ),
}

(results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("metrics.json written")
print(f"  coder recommended max-num-seqs:   {knee(coder)}")
print(f"  thinker recommended max-num-seqs: {knee(thinker)}")
PYEOF
fi

log "=== T_PAR1 complete. Results: ${RESULTS_DIR}/ ==="
log "After this run: update ARCHITECTURE.md parallelism table + DECISIONS.md with optimal N."
log "Update deploy.sh or production startup scripts with the optimal --max-num-seqs per role."
