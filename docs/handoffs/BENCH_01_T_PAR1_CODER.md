# BENCH_01 — T_PAR1: Coder Concurrent Throughput Sweep

**Status: READY**
**Blocks: nothing**
**Blocked by: nothing**

---

## Title
T_PAR1 — Arclight Coder max concurrent client sweep (N = 1, 2, 4, 8)

## Objective
Measure aggregate tokens/second for the Arclight Coder as N concurrent HTTP clients increases from 1 to 8. Find the N at which throughput peaks or plateaus.

## Why this exists
The previous T_PAR1 run (2026-04-26) recorded coder numbers that were fabricated — the endpoint was not running when Gemini Flash wrote the results. All prior coder parallelism numbers (e.g. "1,196 t/s at N=8") are invalid and must not be cited. This handoff produces the first real measurement.

## Prerequisites

Verify all of the following before touching anything else. If any check fails, stop and write a `## Open from testing` block in `RESEARCH_STATE.md`.

```bash
# 1. Coder health
curl -sf http://localhost:30000/health && echo "CODER OK" || echo "CODER DOWN — STOP"

# 2. Coder model identity
curl -s http://localhost:30000/v1/models \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])"
# Expected: cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit

# 3. Script exists
test -f benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh && echo "SCRIPT OK" || echo "SCRIPT MISSING — STOP"

# 4. Results directory is writable
mkdir -p results && echo "RESULTS DIR OK"
```

If the coder is down, deploy it:
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
```
Wait until `/health` returns 200 before proceeding.

## Inputs required
- Running coder endpoint: `http://localhost:30000`
- Script: `benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh`

## Fixed controls
| Control | Value |
|---------|-------|
| Model | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit |
| Engine | vLLM TP=2 GPU0+1 |
| Context ceiling | 32768 |
| KV cache dtype | fp8 |
| Repetitions per N | 3 |
| Thinker endpoint | skipped (--skip-thinker) |
| Convergence endpoint | skipped (--skip-convergence) |

## Single variable under test
**N** — number of concurrent HTTP clients sending simultaneous completion requests.
N ∈ {1, 2, 4, 8}. All other parameters are fixed.

## Procedure

```bash
# Step 1 — Run coder sweep only
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  --skip-thinker \
  --skip-convergence \
  --reps 3

# Step 2 — Locate the results directory just written
ls -td results/T_PAR1_* | head -1
# Note the directory name (e.g. results/T_PAR1_coder_thinker_parallel_20260427T...)
RESULTS_DIR=$(ls -td results/T_PAR1_* | head -1)

# Step 3 — Verify raw data is non-null before proceeding
python3 - <<'EOF'
import json, sys, os
results_dir = os.popen("ls -td results/T_PAR1_* | head -1").read().strip()
path = f"{results_dir}/metrics.json"
with open(path) as f:
    d = json.load(f)
coder = d.get("coder_detail")
if coder is None:
    print("FAIL: coder_detail is null — DO NOT WRITE RESULTS — STOP")
    sys.exit(1)
print("PASS: coder_detail is present")
print("coder_detail keys:", list(coder.keys()) if isinstance(coder, dict) else type(coder))
EOF
```

If Step 3 prints FAIL: stop immediately. Do not write any summary. Write a `## Open from testing` block.

## Metrics to record

From `metrics.json` → `coder_detail`:

| Metric | Field path | Expected range |
|--------|-----------|----------------|
| N=1 aggregate TPS | `coder_detail.n1_tps` | 190–270 t/s |
| N=2 aggregate TPS | `coder_detail.n2_tps` | — |
| N=4 aggregate TPS | `coder_detail.n4_tps` | — |
| N=8 aggregate TPS | `coder_detail.n8_tps` | — |
| N=1 mean latency ms | `coder_detail.n1_latency_ms` | — |
| Raw sweep JSON | `results/*/raw/coder_sweep.json` | must exist |

Also record peak VRAM during the run:
```bash
# Run this in a separate terminal while the sweep is executing
watch -n2 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# Note the peak memory.used values for GPU 0 and GPU 1
```

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| N=1 TPS plausible | `n1_tps` between 190 and 270 | Write failure note; do not record results |
| No null fields | All n1–n8 fields present and numeric | STOP; write `## Open from testing` |
| Raw JSON exists | `raw/coder_sweep.json` is non-empty | STOP |
| No OOM | No CUDA OOM error in script output | Record OOM at which N; continue with lower-N results |

## Artifacts to write

**Write only these files. Do not update any other file.**

1. `results/T_PAR1_<timestamp>/metrics.json` — written automatically by the script. Do not edit it.
2. `results/T_PAR1_<timestamp>/summary.md` — if not written by script, write manually with this exact content:

```markdown
# T_PAR1 Coder Sweep — <TIMESTAMP>

## Raw numbers

| N (concurrent clients) | Aggregate TPS |
|------------------------|---------------|
| 1 | <n1_tps> |
| 2 | <n2_tps> |
| 4 | <n4_tps> |
| 8 | <n8_tps> |

## VRAM at peak N
GPU 0: <X> MiB
GPU 1: <X> MiB

## Status
MEASURED — raw data in metrics.json
```

**Do NOT write to:**
- `docs/decisions/settled.md`
- `docs/queue/open.md` or `docs/queue/status.md`
- `docs/arch/current.md`
- `RESEARCH_STATE.md`
- Any file outside `results/T_PAR1_<timestamp>/`

## Interpretation boundary

- **You may record** the N values and their TPS numbers exactly as measured.
- **You may note** whether a CUDA OOM occurred and at which N.
- **You may NOT** select an optimal N.
- **You may NOT** recommend a max-num-seqs value.
- **You may NOT** conclude whether concurrent or sequential architecture is better.
- **Ignore** the "Recommended max-num-seqs" line printed by the script at the end — that is for research mode to evaluate.

## Stop condition

**Normal termination:** `metrics.json` is written, `coder_detail` is non-null, all N values have numeric TPS, `summary.md` is written.

**Abnormal termination:** Any of the following — write `## Open from testing` block in `RESEARCH_STATE.md` and stop:
- Coder endpoint fails health check and cannot be started
- `coder_detail` is null in `metrics.json`
- N=1 TPS is outside 190–270 range (endpoint likely serving wrong model or stale process)
- CUDA error other than OOM
