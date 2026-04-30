# BENCH_13 — T_MTP1: Thinker MTP Speculative Decoding (n=1)

**Status: READY**
**Blocks: BENCH_14 (T_MTP2 coder)**
**Blocked by: nothing**

---

## Title
T_MTP1 — MTP n=1 TPS improvement on production thinker (AWQ, no model swap)

## Objective
Measure the TPS impact of enabling native Multi-Token Prediction speculative decoding (`method=mtp`, `num_speculative_tokens=1`) on the production thinker (QuantTrio/Qwen3.6-27B-AWQ, TP=1, GPU1, max-num-seqs=4). Compare TPS at N=1 and N=4 against the settled AWQ baseline without MTP. Verify quality is not degraded (th02 correctness gate).

## Why this exists
Community benchmark (RTX 3090, vLLM 0.19.1) measured −21.6% TPOT ≡ **+27.5% faster decode rate** with MTP n=1 on Qwen3.6. Qwen3.6-27B has a native MTP head trained alongside the model — vLLM uses it in the same forward pass, adding no memory overhead. This is the lowest-cost possible TPS improvement: one flag addition, existing endpoint, no new model download, no container rebuild.

vLLM #40756 (MTP crash) does NOT apply: that bug requires FP8+TP=4+n=5+25K tokens. Our config is AWQ+TP=1+n=1. Unblocked on vLLM 0.19.0.

## Prerequisites

```bash
# 1. Thinker health
curl -sf http://localhost:30001/health && echo "THINKER OK" || echo "THINKER DOWN — deploy first"

# 2. Thinker model identity
curl -s http://localhost:30001/v1/models \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])"
# Expected: QuantTrio/Qwen3.6-27B-AWQ

# 3. Current thinker max-num-seqs (should be 4 per production config)
curl -s http://localhost:30001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin))"
# If still at max-num-seqs=1: redeploy first with --max-num-seqs 4 (settled production config)

# 4. GPU1 VRAM baseline
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

**This test runs on the HOST.**

## Inputs required
- Production thinker running at `http://localhost:30001` with model `QuantTrio/Qwen3.6-27B-AWQ`
- `infra/scripts/deploy.sh` accessible from the repo root
- `benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh` for the TPS sweep (both steps)
- th02 prompt at `benchmarks/phase2_model_selection/tasks/thinker/th02*`
- GPU1 VRAM headroom to redeploy (script stops and restarts the container)

## Fixed controls

| Control | Value |
|---------|-------|
| Model | QuantTrio/Qwen3.6-27B-AWQ |
| Engine | vLLM TP=1 GPU1 |
| VLLM_USE_V1 | 0 (mandatory) |
| max-num-seqs | 4 |
| Context ceiling | 32768 |
| KV cache dtype | fp8 |
| chunked-prefill | ON |
| MTP n | **1** (single variable under test) |
| Reps per N | 3 |

## Single variable under test
**`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`** — present vs absent. All other flags identical.

---

## Procedure

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_13_mtp1_thinker_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
```

### Step 1 — Record baseline TPS (no MTP, current production thinker)

If thinker is already running at max-num-seqs=4 (production), run the sweep directly:

```bash
# Baseline sweep — no MTP
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  --skip-coder \
  --skip-convergence \
  --reps 3

BASELINE_DIR=$(ls -td results/T_PAR1_* | head -1)
echo "Baseline results: ${BASELINE_DIR}"
cp "${BASELINE_DIR}/metrics.json" "${RESULTS_DIR}/metrics_baseline.json"

python3 - <<'EOF'
import json
d = json.load(open("${RESULTS_DIR}/metrics_baseline.json".replace("${RESULTS_DIR}", __import__("os").environ.get("RESULTS_DIR","results/BENCH_13_mtp1_thinker_placeholder"))))
t = d.get("thinker_detail", {})
print(f"Baseline N=1: {t.get('n1_tps')} t/s  N=4: {t.get('n4_tps')} t/s")
EOF
```

If the thinker is not running, deploy the baseline config first:

```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

for i in $(seq 1 120); do
  curl -sf http://localhost:30001/health && echo "READY" && break; sleep 1
done
```

Then run the baseline sweep as above.

### Step 2 — Stop thinker, redeploy with MTP n=1

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker\|gpu1\|27b" | head -1)
podman stop "${THINKER_CONTAINER}" && podman rm "${THINKER_CONTAINER}"

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'

# Wait up to 120s
for i in $(seq 1 120); do
  curl -sf http://localhost:30001/health && echo "THINKER+MTP READY" && break; sleep 1
done
```

**If deploy fails with an error about MTP / speculative-config:**
- Capture container logs: `podman logs $(podman ps -a --format "{{.Names}}" | grep thinker | head -1) 2>&1 | tail -40`
- Save to `${RESULTS_DIR}/mtp_startup_failure.txt`
- Write `## Open from testing` in RESEARCH_STATE.md
- Proceed to Step 5 to restore production thinker without MTP.

### Step 3 — TPS sweep with MTP n=1

```bash
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  --skip-coder \
  --skip-convergence \
  --reps 3

MTP_DIR=$(ls -td results/T_PAR1_* | head -1)
echo "MTP results: ${MTP_DIR}"
cp "${MTP_DIR}/metrics.json" "${RESULTS_DIR}/metrics_mtp_n1.json"

# Print comparison
python3 - <<'EOF'
import json
b = json.load(open("${RESULTS_DIR}/metrics_baseline.json".replace("${RESULTS_DIR}", __import__("os").popen("ls -td results/BENCH_13_mtp1_thinker_* | head -1").read().strip())))
m = json.load(open("${MTP_DIR}/metrics.json".replace("${MTP_DIR}", __import__("os").popen("ls -td results/T_PAR1_* | head -1").read().strip())))
bt = b.get("thinker_detail", {})
mt = m.get("thinker_detail", {})
for n in ["n1","n2","n4"]:
    bv = bt.get(f"{n}_tps"); mv = mt.get(f"{n}_tps")
    delta = round((mv-bv)/bv*100, 1) if bv and mv else "N/A"
    print(f"N={'1' if n=='n1' else '2' if n=='n2' else '4'}: baseline={bv} t/s  mtp={mv} t/s  delta={delta}%")
EOF
```

### Step 4 — Quality smoke check (th02 only)

```bash
# Run th02 manually — the critical DeltaNet correctness gate
# th02 tests whether recurrent state reasoning is intact
# Use the same prompt as T2.4d (available in benchmarks/phase2_model_selection/tasks/thinker/th02*)

python3 -m benchmarks.phase2_model_selection.bench \
  --endpoint http://localhost:30001/v1 \
  --results-dir "${RESULTS_DIR}/quality_mtp" \
  --mode quality \
  --tasks benchmarks/phase2_model_selection/tasks/thinker/ \
  --task-filter th02 \
  --model QuantTrio/Qwen3.6-27B-AWQ \
  --label "AWQ-MTP-n1" \
  --max-tokens 16384

echo "th02 result written to ${RESULTS_DIR}/quality_mtp/"
```

If the quality bench runner doesn't exist or fails: send th02 prompt manually via curl to
`http://localhost:30001/v1/chat/completions` and record the response.

**th02 pass condition:** model reaches the same correct conclusion as in T2.4d (3/3 reproducible).
A wrong answer or empty `<think>` block means MTP is corrupting the decode path.

### Step 5 — Restore production thinker (MANDATORY)

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
[ -n "${THINKER_CONTAINER}" ] && podman stop "${THINKER_CONTAINER}" && podman rm "${THINKER_CONTAINER}"

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

for i in $(seq 1 120); do
  curl -sf http://localhost:30001/health && echo "PRODUCTION THINKER RESTORED" && break; sleep 1
done
```

---

## Metrics to record

| Metric | Source | AWQ baseline (no MTP) |
|--------|--------|-----------------------|
| N=1 TPS (baseline) | metrics_baseline.json | 76.8 t/s |
| N=1 TPS (MTP n=1) | metrics_mtp_n1.json | — |
| N=1 TPS delta % | computed | **target: +10–30%** |
| N=4 TPS (baseline) | metrics_baseline.json | 269.4 t/s |
| N=4 TPS (MTP n=1) | metrics_mtp_n1.json | — |
| N=4 TPS delta % | computed | may be flat or negative under load |
| th02 result | quality_mtp/ | correct 3/3 |
| GPU1 VRAM with MTP | nvidia-smi after Step 2 deploy | ~27,732 MiB (no change expected) |
| MTP startup success | deploy log | — |

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| MTP deploy succeeds | HEALTH_OK within 120s | Capture error; write Open from testing; skip to Step 5 |
| N=1 TPS improvement | ≥ +10% vs baseline | Note marginal gain; still worth checking N=4 impact |
| N=4 TPS not severely degraded | > −20% vs baseline 269 t/s | Note; at max-num-seqs=4, concurrency may hurt MTP |
| th02 correct | Same correct answer as T2.4d | FAIL — MTP corrupts decode path; do not enable in production |
| Production restored | /health 200 on port 30001 | Redeploy manually |

**If N=1 improves but N=4 degrades:** this is expected behaviour — speculative tokens consume KV cache slots, reducing effective batch size. The trade-off is: faster single-request latency at the cost of parallel throughput. Decision to be made in research review.

---

## Artifacts to write

1. `results/BENCH_13_mtp1_thinker_<timestamp>/metrics_baseline.json` — copy of T_PAR1 baseline
2. `results/BENCH_13_mtp1_thinker_<timestamp>/metrics_mtp_n1.json` — copy of T_PAR1 MTP run
3. `results/BENCH_13_mtp1_thinker_<timestamp>/quality_mtp/` — th02 output
4. `results/BENCH_13_mtp1_thinker_<timestamp>/summary.md`:

```markdown
# BENCH_13 — T_MTP1 Thinker MTP n=1 — <TIMESTAMP>

## TPS comparison
| N | Baseline (no MTP) | MTP n=1 | Delta |
|---|-------------------|---------|-------|
| 1 | 76.8 t/s | <X> t/s | <±%> |
| 2 | 139.3 t/s | <X> t/s | <±%> |
| 4 | 269.4 t/s | <X> t/s | <±%> |

## Quality
th02: CORRECT / FAIL

## GPU1 VRAM with MTP
<X> MiB (baseline: 27,732 MiB)

## MTP startup
SUCCESS / FAILED (see mtp_startup_failure.txt)

## Verdict
PASS / FAIL / MARGINAL
```

**Do NOT write to any file outside `results/BENCH_13_mtp1_thinker_<timestamp>/`.**

---

## Interpretation boundary

- **You may record** TPS delta, VRAM, th02 outcome, startup success.
- **You may note** whether N=1 improved and whether N=4 degraded.
- **You may NOT** update `docs/decisions/settled.md`, production config, or queue files.
- **You may NOT** conclude whether MTP should be permanently enabled in production.
- **You may NOT** test n=2 or n=3 — that is out of scope for this handoff.

## Stop condition

**Normal:** TPS comparison written, th02 result recorded, production thinker restored, summary.md written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` and stop if:
- MTP deploy fails with an unrecognised error (not crash on long context, not OOM)
- th02 fails with a new error pattern (not the known TP=2 GDN recurrent state failure — that is for a different config)
- N=1 TPS is worse than −30% vs baseline (unexpected regression, not just concurrency effect)
