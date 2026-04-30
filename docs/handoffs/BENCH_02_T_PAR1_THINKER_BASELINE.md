# BENCH_02 — T_PAR1: Thinker Baseline Confirmation (max-num-seqs=1)

**Status: READY**
**Blocks: BENCH_03**
**Blocked by: nothing**

---

## Title
T_PAR1 — Arclight Thinker baseline TPS at production max-num-seqs=1

## Objective
Confirm the Thinker's single-request TPS under the production configuration (max-num-seqs=1) and record what happens when N>1 concurrent clients hit a single-slot server.

## Why this exists
The thinker's known baseline is 77.4 t/s from T2.4d, measured in a different test context (quality evaluation, not throughput sweep). This measurement uses the same parallel-sweep harness as the coder sweep (BENCH_01), producing comparable numbers and documenting queuing behaviour at N>1 under production config before any config change is attempted.

## Prerequisites

Verify all of the following. If any fails, stop and write `## Open from testing` in `RESEARCH_STATE.md`.

```bash
# 1. Thinker health
curl -sf http://localhost:30001/health && echo "THINKER OK" || echo "THINKER DOWN — STOP"

# 2. Thinker model identity
curl -s http://localhost:30001/v1/models \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])"
# Expected: QuantTrio/Qwen3.6-27B-AWQ

# 3. Script exists
test -f benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh && echo "SCRIPT OK" || echo "SCRIPT MISSING — STOP"
```

If the thinker is down, deploy it:
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 1 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
```
Wait until `/health` returns 200.

## Inputs required
- Running thinker endpoint: `http://localhost:30001`
- Script: `benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh`

## Fixed controls
| Control | Value |
|---------|-------|
| Model | QuantTrio/Qwen3.6-27B-AWQ |
| Engine | vLLM TP=1 GPU1 |
| max-num-seqs | **1** (production value — must not change for this handoff) |
| Context ceiling | 32768 |
| KV cache dtype | fp8 |
| chunked-prefill | ON |
| Repetitions per N | 3 |
| Coder endpoint | skipped (--skip-coder) |
| Convergence endpoint | skipped (--skip-convergence) |

## Single variable under test
**N** — number of concurrent HTTP clients (N ∈ {1, 2, 4, 8}).
Server max-num-seqs is fixed at 1 throughout. At N>1 the server queues requests serially — this behaviour is expected and is what is being documented.

## Procedure

```bash
# Step 1 — Run thinker sweep only, against production (max-num-seqs=1) config
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  --skip-coder \
  --skip-convergence \
  --reps 3

# Step 2 — Locate the results directory
RESULTS_DIR=$(ls -td results/T_PAR1_* | head -1)
echo "Results in: ${RESULTS_DIR}"

# Step 3 — Verify thinker data is non-null
python3 - <<'EOF'
import json, sys, os
results_dir = os.popen("ls -td results/T_PAR1_* | head -1").read().strip()
path = f"{results_dir}/metrics.json"
with open(path) as f:
    d = json.load(f)
thinker = d.get("thinker_detail")
if thinker is None:
    print("FAIL: thinker_detail is null — DO NOT WRITE RESULTS — STOP")
    sys.exit(1)
print("PASS: thinker_detail is present")
EOF
```

If Step 3 prints FAIL: stop immediately. Do not write any summary. Write `## Open from testing`.

## Metrics to record

From `metrics.json` → `thinker_detail`:

| Metric | Field path | Expected range |
|--------|-----------|----------------|
| N=1 aggregate TPS | `thinker_detail.n1_tps` | 60–100 t/s |
| N=2 aggregate TPS | `thinker_detail.n2_tps` | expected ≈ N=1 (queued) |
| N=4 aggregate TPS | `thinker_detail.n4_tps` | expected ≈ N=1 (queued) |
| N=8 aggregate TPS | `thinker_detail.n8_tps` | expected ≈ N=1 (queued) |
| N=1 mean latency ms | `thinker_detail.n1_latency_ms` | — |

Also record GPU 1 VRAM during the run:
```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| N=1 TPS plausible | `n1_tps` between 60 and 100 | Do not record; write failure note |
| No null fields | All n1–n8 fields present and numeric | STOP; write `## Open from testing` |
| Raw JSON exists | `raw/thinker_sweep.json` non-empty | STOP |
| N>1 TPS ≈ N=1 TPS | Expected: queuing means N=2,4,8 aggregate TPS ≈ N=1 TPS | If N>1 aggregate is dramatically higher than N=1, note as anomaly |

## Artifacts to write

**Write only these files:**

1. `results/T_PAR1_<timestamp>/metrics.json` — written automatically by script. Do not edit.
2. `results/T_PAR1_<timestamp>/summary.md`:

```markdown
# T_PAR1 Thinker Baseline — <TIMESTAMP>
## Config: max-num-seqs=1 (production)

| N (concurrent clients) | Aggregate TPS | Notes |
|------------------------|---------------|-------|
| 1 | <n1_tps> | |
| 2 | <n2_tps> | queued (max-num-seqs=1) |
| 4 | <n4_tps> | queued |
| 8 | <n8_tps> | queued |

## GPU 1 VRAM
<X> MiB

## Status
MEASURED — raw data in metrics.json
```

**Do NOT write to any file outside `results/T_PAR1_<timestamp>/`.**

## Interpretation boundary

- **You may record** TPS at each N exactly as measured.
- **You may note** that N>1 results show queuing because max-num-seqs=1.
- **You may NOT** change the thinker's max-num-seqs. That is BENCH_03.
- **You may NOT** conclude whether the thinker needs more slots.
- **You may NOT** update `docs/decisions/settled.md` or any queue file.

## Stop condition

**Normal:** `metrics.json` written, `thinker_detail` non-null, `summary.md` written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` and stop if:
- Thinker fails health check and cannot be started
- `thinker_detail` is null in `metrics.json`
- N=1 TPS is outside 60–100 t/s range
- Any CUDA error (not OOM)
