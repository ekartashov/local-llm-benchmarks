# BENCH_03 — T_PAR1: Thinker Parallel Ceiling (max-num-seqs=4)

**Status: BLOCKED**
**Blocks: nothing**
**Blocked by: BENCH_02** — must complete with non-null `thinker_detail.n1_tps` in range 60–100 t/s before this handoff may begin.

---

## Title
T_PAR1 — Arclight Thinker true parallel ceiling at max-num-seqs=4

## Objective
Measure Thinker aggregate TPS when the server is configured for up to 4 parallel slots (max-num-seqs=4) and N=4 concurrent clients are firing simultaneously. Determine whether the thinker gains throughput from parallel execution or is bandwidth-bound at N=1.

## Why this exists
BENCH_02 documents the production (max-num-seqs=1) queuing behaviour. This handoff tests whether raising max-num-seqs to 4 yields a real throughput gain or causes OOM. The thinker is deployed on a single 32 GB GPU and runs GDN architecture — the prior (fabricated) claim that it OOMs at N>1 has never been verified.

## Prerequisites

**BENCH_02 must be complete first.** Verify before proceeding:

```bash
# 1. BENCH_02 result must exist and be non-null
python3 - <<'EOF'
import json, os, sys
results_dir = os.popen(
    "ls -td results/T_PAR1_* | head -1"
).read().strip()
path = f"{results_dir}/metrics.json"
with open(path) as f:
    d = json.load(f)
tps = d.get("thinker_detail", {})
n1 = tps.get("n1_tps") if isinstance(tps, dict) else None
if n1 is None:
    print("BLOCKED: BENCH_02 thinker_detail.n1_tps is null. Run BENCH_02 first.")
    sys.exit(1)
if not (60 <= n1 <= 100):
    print(f"BLOCKED: n1_tps={n1} is outside expected range 60–100. Investigate before continuing.")
    sys.exit(1)
print(f"BENCH_02 confirmed: n1_tps={n1}. Proceeding.")
EOF

# 2. Thinker health (must currently be running at max-num-seqs=1)
curl -sf http://localhost:30001/health && echo "THINKER OK" || echo "THINKER DOWN — deploy it first"

# 3. Script exists
test -f benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh && echo "SCRIPT OK" || echo "SCRIPT MISSING — STOP"
```

## Inputs required
- BENCH_02 completed successfully
- Running thinker on port 30001 (may be at max-num-seqs=1 — will be replaced)
- Script: `benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh`

## Fixed controls
| Control | Value |
|---------|-------|
| Model | QuantTrio/Qwen3.6-27B-AWQ |
| Engine | vLLM TP=1 GPU1 |
| max-num-seqs | **4** (changed from BENCH_02's production value of 1) |
| Context ceiling | 32768 |
| KV cache dtype | fp8 |
| chunked-prefill | ON |
| Repetitions per N | 3 |
| Coder endpoint | skipped |
| Convergence endpoint | skipped |

## Single variable under test
**max-num-seqs** — comparing this run (max-num-seqs=4) against BENCH_02 (max-num-seqs=1) at matched N values.

## Procedure

```bash
# Step 1 — Stop current thinker container
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker\|gpu1\|27b" | head -1)
echo "Stopping: ${THINKER_CONTAINER}"
podman stop "${THINKER_CONTAINER}" && podman rm "${THINKER_CONTAINER}"

# Step 2 — Redeploy thinker with max-num-seqs=4
VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

# Step 3 — Wait for health (up to 120 s)
for i in $(seq 1 120); do
  curl -sf http://localhost:30001/health && echo "READY" && break
  sleep 1
done

# Step 4 — Verify model identity at new deployment
curl -s http://localhost:30001/v1/models \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])"
# Expected: QuantTrio/Qwen3.6-27B-AWQ

# Step 5 — Run thinker sweep
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  --skip-coder \
  --skip-convergence \
  --reps 3

# Step 6 — Verify data
python3 - <<'EOF'
import json, sys, os
results_dir = os.popen("ls -td results/T_PAR1_* | head -1").read().strip()
with open(f"{results_dir}/metrics.json") as f:
    d = json.load(f)
t = d.get("thinker_detail")
if t is None:
    print("FAIL: thinker_detail null — STOP")
    sys.exit(1)
print("PASS:", {k: t[k] for k in ["n1_tps","n2_tps","n4_tps"] if k in t})
EOF

# Step 7 — MANDATORY: restore production thinker config (max-num-seqs=1)
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker\|gpu1\|27b" | head -1)
podman stop "${THINKER_CONTAINER}" && podman rm "${THINKER_CONTAINER}"

VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 1 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

# Verify production restore
curl -sf http://localhost:30001/health && echo "PRODUCTION THINKER RESTORED"
```

**Step 7 is mandatory** regardless of whether Step 5 passed or failed.

## Metrics to record

| Metric | Field path | Notes |
|--------|-----------|-------|
| N=1 TPS at max-num-seqs=4 | `thinker_detail.n1_tps` | Compare to BENCH_02 n1_tps |
| N=2 TPS at max-num-seqs=4 | `thinker_detail.n2_tps` | |
| N=4 TPS at max-num-seqs=4 | `thinker_detail.n4_tps` | Key data point |
| VRAM at peak during N=4 run | `nvidia-smi` GPU 1 | Record if OOM occurs |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Deploy at max-num-seqs=4 succeeds | No OOM on startup | Record VRAM at OOM; skip to Step 7 |
| N=1 TPS at new config ≈ BENCH_02 N=1 | Within ±15% of BENCH_02 value | Note discrepancy |
| Data non-null | `thinker_detail` present | STOP after Step 7; write `## Open from testing` |
| Production restored | Final `/health` on port 30001 returns 200 | Do not leave the thinker in max-num-seqs=4 state |

**OOM on startup is a valid, expected result.** If the thinker OOMs when deploying with max-num-seqs=4, record: the CUDA OOM message, VRAM reading from `nvidia-smi` at the time. This is not a failure of the benchmark — it is the answer to the research question. Proceed to Step 7.

## Artifacts to write

1. `results/T_PAR1_<timestamp>/metrics.json` — written by script.
2. `results/T_PAR1_<timestamp>/summary.md`:

```markdown
# T_PAR1 Thinker Parallel — <TIMESTAMP>
## Config: max-num-seqs=4

### Result
<PASS or OOM_ON_STARTUP or OOM_AT_N=X>

### TPS at max-num-seqs=4 (if deploy succeeded)
| N | Aggregate TPS | vs BENCH_02 (max-num-seqs=1) |
|---|---------------|-------------------------------|
| 1 | <n1_tps> | <BENCH_02_n1_tps> |
| 2 | <n2_tps> | <BENCH_02_n2_tps> |
| 4 | <n4_tps> | <BENCH_02_n4_tps> |

### OOM note (if applicable)
GPU 1 VRAM at OOM: <X> MiB
Error: <paste first line of CUDA OOM>

### Production config restored
yes / no
```

**Do NOT write to any file outside `results/T_PAR1_<timestamp>/`.**

## Interpretation boundary

- **You may record** TPS numbers at max-num-seqs=4 and compare them numerically to BENCH_02.
- **You may record** the OOM outcome and VRAM numbers if startup fails.
- **You may NOT** conclude whether max-num-seqs=4 should become production.
- **You may NOT** conclude whether the thinker is memory-bandwidth-bound.
- **You may NOT** update any doc outside the results directory.

## Stop condition

**Normal:** data written, production thinker restored (max-num-seqs=1), health check passing.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` if:
- `thinker_detail` is null after a deploy-and-sweep that appeared to succeed
- Production thinker restore fails (health check on port 30001 not responding after 120 s)
- CUDA error other than OOM
