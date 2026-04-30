# BENCH_12 — PrismaQuant Thinker: rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm

**Status: READY**
**Blocks: nothing**
**Blocked by: nothing**

---

## Title
PrismaQuant 27B thinker candidate — TPS + quality evaluation vs AWQ baseline

## Objective
Deploy `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm` on GPU1 (thinker slot), measure TPS at
max-num-seqs=4, run the thinker quality suite (th02/th05/full 8-task), and compare both axes
against the settled AWQ baseline (76.8 t/s at N=1, 269.4 t/s at N=4, 4.875/5 quality).

## Why this model

rdtand is the original PrismaQuant author. The 27B model:
- Uses 349 NVFP4 + 35 MXFP8 + 112 BF16 linears, 5.5 bpp average (~19 GB disk)
- Explicitly handles 64 DeltaNet layers with per-linear allocation
- GPTQ + scale sweep achieves 0.33× RTN baseline MSE (vs naive quantization)
- Dense architecture (no MoE) — avoids all SM120 Grouped GEMM bugs that affect the 35B coder

## Research notes — SM120 NVFP4 compatibility

**Key finding from BENCH_12 research (2026-04-30):**

The garbage-output and TMA WS GEMM bugs on SM120 are specific to MoE Grouped GEMM kernels.
This model is dense — standard GEMM, not Grouped GEMM. Those bugs do not apply.

**Expected kernel path with our VLLM_USE_V1=0 (V0 engine):**
- NVFP4 weights will likely be served via Marlin dequant kernels (V0 path), not native FP4 OMMA
- FlashInfer NVFP4 attention is a V1 engine feature; V0 uses standard attention backend
- This means the native FP4 compute advantage (2× TFLOPS) will NOT be realized
- What IS realized: better quantization quality from GPTQ calibration vs AWQ's RTN baseline

**Expected TPS expectation:** 5.5 bpp has more weight data than AWQ's 4 bpp. Without native FP4
dequant shortcut, decode TPS may be flat or slightly lower than AWQ. TPS improvement would
require V1 engine with FlashInfer — that is a separate research item (MTP / V1 stability).

**Primary success criterion for this test: quality, not TPS.**
If th02 passes and quality ≥ AWQ at matched or acceptable TPS, PrismaQuant 5.5bit is a
candidate for the thinker slot. If TPS degrades >20%, the trade-off needs discussion.

---

## Prerequisites

```bash
# 1. GPU1 VRAM baseline (should be ~0 MiB if thinker is stopped, or ~27.7 GB if running)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# 2. No other process holding GPU1
podman ps --format "{{.Names}}\t{{.Status}}"

# 3. Scripts exist
test -f benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh && echo "SWEEP OK"
test -d benchmarks/phase2_model_selection/tasks/thinker/ && echo "TASKS OK"

# 4. Check vLLM CUDA build version (critical for future coder PrismaQuant decision)
podman run --rm vllm/vllm-openai:latest python3 -c \
  "import torch; print('CUDA:', torch.version.cuda)"
# Record this. cu128 = CUDA 12.8, cu130 = CUDA 13.0.
# Does not block this test (27B is dense, CUDA version only matters for MoE GroupedGEMM).
```

---

## Inputs required
- `infra/scripts/deploy.sh` with the vLLM NVFP4-compatible container image built and available
- `benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh` for the TPS sweep
- Production thinker container stopped before Step 1 (this test occupies GPU1)
- GPU1 VRAM fully free (~21 GB needed for PrismaQuant 5.5bit)
- th02 prompt and full 8-task thinker suite at `benchmarks/phase2_model_selection/tasks/thinker/`
- AWQ baseline `metrics.json` from a recent T_PAR1 run (for direct delta comparison)

---

## Fixed controls

| Control | Value |
|---------|-------|
| Model | rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm |
| Engine | vLLM TP=1 GPU1 |
| VLLM_USE_V1 | 0 (mandatory) |
| Quantization flag | **NOT set** — auto-detect from config.json |
| max-num-seqs | 4 |
| Context ceiling | 32768 |
| KV cache dtype | fp8 |
| chunked-prefill | ON |
| Speculative decoding (MTP) | **SKIP** — V1 engine required; separate research item |
| --trust-remote-code | required for compressed-tensors config.json |
| Reps per N (TPS sweep) | 3 |

## Single variable under test
**Model quantization format** — PrismaQuant 5.5bit NVFP4/MXFP8/BF16 vs settled AWQ 4bit.

---

## Procedure

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_12_prismaquant_thinker_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
MODEL="rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm"
```

### Step 1 — Stop current thinker, record baseline GPU1 VRAM

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker\|gpu1\|27b" | head -1)
if [ -n "${THINKER_CONTAINER}" ]; then
  echo "Stopping: ${THINKER_CONTAINER}"
  podman stop "${THINKER_CONTAINER}" && podman rm "${THINKER_CONTAINER}"
fi

sleep 3
VRAM_IDLE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
echo "GPU1 VRAM (idle): ${VRAM_IDLE} MiB"
echo "vram_idle_gpu1_mib=${VRAM_IDLE}" > "${RESULTS_DIR}/timings.txt"
```

### Step 2 — Deploy PrismaQuant thinker

```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 "${MODEL}" \
  --trust-remote-code \
  --gpu-mem-util 0.90 \
  --ctx 32768 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-seqs 4 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice

# Wait up to 180s (compressed-tensors loading may take longer than AWQ)
HEALTH_OK=0
for i in $(seq 1 180); do
  curl -sf http://localhost:30001/health 2>/dev/null && echo "THINKER READY" && HEALTH_OK=1 && break
  sleep 1
done
echo "health_ok=${HEALTH_OK}" >> "${RESULTS_DIR}/timings.txt"
```

### Step 3 — Handle startup failure

If HEALTH_OK=0:
```bash
# Capture error from container logs
THINKER_CONTAINER=$(podman ps -a --format "{{.Names}}" | grep -i "thinker" | head -1)
podman logs "${THINKER_CONTAINER}" 2>&1 | tail -60 > "${RESULTS_DIR}/startup_failure.txt"
echo "OOM_OR_ERROR" > "${RESULTS_DIR}/status.txt"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_at_failure.txt"
echo "STARTUP FAILED — see startup_failure.txt. Proceeding to Step 6 to restore production thinker."
# Skip to Step 6.
```

**Failure modes to look for in startup_failure.txt:**
- `KeyError: expert_params_mapping` → compressed-tensors vLLM version issue
- `NvFp4 MoE backend ... does not support current device` → SM120 family check (unexpected for dense model; note if seen)
- `CUDA out of memory` → 5.5bit model too large for GPU1 at max-num-seqs=4; retry with max-num-seqs=1
- `FlashInfer warmup crash` → FlashInfer NVFP4 on sm_120; set `VLLM_USE_FLASHINFER_NVFP4=0` and retry

**OOM retry (if and only if OOM at max-num-seqs=4):**
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 "${MODEL}" \
  --trust-remote-code \
  --gpu-mem-util 0.90 \
  --ctx 32768 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```
If max-num-seqs=1 succeeds: run TPS sweep only for N=1 (skip N=2,4). Note configuration in results.

### Step 4 — Record VRAM at load

```bash
VRAM_LOADED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
echo "GPU1 VRAM (PrismaQuant loaded): ${VRAM_LOADED} MiB"
echo "vram_loaded_gpu1_mib=${VRAM_LOADED}" >> "${RESULTS_DIR}/timings.txt"

# Reference: AWQ at max-num-seqs=4 was 27732 MiB
VRAM_DELTA=$((VRAM_LOADED - 27732))
echo "VRAM delta vs AWQ baseline (27732 MiB): ${VRAM_DELTA} MiB"
echo "vram_delta_vs_awq=${VRAM_DELTA}" >> "${RESULTS_DIR}/timings.txt"

# Confirm model identity
curl -s http://localhost:30001/v1/models \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])"
# Expected: rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm
```

### Step 5 — TPS sweep at max-num-seqs=4

```bash
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  --skip-coder \
  --skip-convergence \
  --reps 3

# Locate results
SWEEP_DIR=$(ls -td results/T_PAR1_* | head -1)
echo "Sweep results in: ${SWEEP_DIR}"
echo "sweep_dir=${SWEEP_DIR}" >> "${RESULTS_DIR}/timings.txt"

# Verify non-null
python3 - <<'EOF'
import json, os, sys
d = json.load(open(os.popen("ls -td results/T_PAR1_* | head -1").read().strip() + "/metrics.json"))
t = d.get("thinker_detail")
if t is None:
    print("FAIL: thinker_detail null")
    sys.exit(1)
print("TPS N=1:", t.get("n1_tps"), "N=2:", t.get("n2_tps"), "N=4:", t.get("n4_tps"))
EOF
```

### Step 6 — Quality evaluation (thinker task suite, 8 tasks)

```bash
python3 -m benchmarks.phase2_model_selection.bench \
  --endpoint http://localhost:30001/v1 \
  --results-dir "${RESULTS_DIR}/phase2_quality" \
  --mode quality \
  --tasks benchmarks/phase2_model_selection/tasks/thinker/ \
  --model "${MODEL}" \
  --label "PrismaQuant-5.5bit" \
  --max-tokens 16384
```

**If quality bench fails to start:** verify `benchmarks/phase2_model_selection/bench.py` exists,
or run the quality tasks manually using the prompts in the tasks directory with curl.

### Step 7 — Restore production thinker (MANDATORY regardless of outcome)

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
[ -n "${THINKER_CONTAINER}" ] && podman stop "${THINKER_CONTAINER}" && podman rm "${THINKER_CONTAINER}"

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 \
  --ctx 32768 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-seqs 4 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice

for i in $(seq 1 120); do
  curl -sf http://localhost:30001/health && echo "PRODUCTION THINKER RESTORED" && break
  sleep 1
done
```

---

## Metrics to record

| Metric | Source | AWQ baseline |
|--------|--------|-------------|
| GPU1 VRAM at load (MiB) | timings.txt | 27,732 MiB |
| VRAM delta vs AWQ (MiB) | timings.txt | 0 (reference) |
| N=1 aggregate TPS | T_PAR1 sweep metrics.json | 76.8 t/s |
| N=2 aggregate TPS | T_PAR1 sweep metrics.json | 139.3 t/s |
| N=4 aggregate TPS | T_PAR1 sweep metrics.json | 269.4 t/s |
| Quality mean (8 tasks) | phase2_quality/metrics.json (human review) | 4.875/5 |
| th02 result (correct/fail) | human review — critical | 3/3 correct |
| th05 result | human review | pass |
| CUDA build version | Step 0 nvidia check | record |
| Startup result | status.txt | — |

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Startup success | HEALTH_OK=1 within 180s | Record error class; skip to Step 7 |
| TPS N=1 acceptable | Within ±20% of 76.8 t/s (61–92 t/s) | Note if outside range; still run quality |
| TPS N=4 not severely degraded | > 200 t/s (75% of AWQ 269.4) | Note regression; research discussion needed |
| th02 correct | 3/3 correct (recurrent state not corrupted by quantization) | FAIL — quant damages DeltaNet recurrent state |
| Quality mean | ≥ 4.5/5 (quality improvement expected from GPTQ) | Below AWQ baseline = no case for swap |
| Production restored | /health on 30001 returns 200 after Step 7 | Redeploy manually |

**th02 is the critical correctness gate.** DeltaNet recurrent state lives in the weight matrices —
if 5.5-bit quantization corrupts the recurrent state update, th02 will fail the same way TP=2
failed (T2.4g). A th02 failure here means PrismaQuant 5.5bit damages GDN semantics; try the
6.0 bpp variant or fall back to AWQ.

---

## Artifacts to write

1. `results/BENCH_12_prismaquant_thinker_<timestamp>/timings.txt` — VRAM + startup flag
2. `results/BENCH_12_prismaquant_thinker_<timestamp>/status.txt` — STARTUP_OK / OOM / ERROR
3. `results/T_PAR1_<timestamp>/metrics.json` — written by sweep script
4. `results/BENCH_12_prismaquant_thinker_<timestamp>/phase2_quality/` — quality bench output
5. `results/BENCH_12_prismaquant_thinker_<timestamp>/summary.md`:

```markdown
# BENCH_12 — PrismaQuant 27B Thinker — <TIMESTAMP>

## Startup
STARTUP_OK / OOM_AT_STARTUP / ERROR (see startup_failure.txt)

## VRAM
| Config | GPU1 VRAM |
|--------|-----------|
| AWQ baseline (max-num-seqs=4) | 27,732 MiB |
| PrismaQuant 5.5bit | <X> MiB |
| Delta | <±Y> MiB |

## TPS (max-num-seqs=4)
| N | PrismaQuant 5.5bit | AWQ baseline | Delta |
|---|-------------------|-------------|-------|
| 1 | <t/s> | 76.8 t/s | <±%> |
| 2 | <t/s> | 139.3 t/s | <±%> |
| 4 | <t/s> | 269.4 t/s | <±%> |

## Quality (8-task thinker suite)
| Task | Score | Notes |
|------|-------|-------|
| th02 | CORRECT/FAIL | critical |
| th05 | score | |
| mean | X/5 | AWQ baseline: 4.875/5 |

## CUDA build version
<cu128 / cu130 / other>

## Verdict
PASS / FAIL / INCONCLUSIVE

## Notes
<any NVFP4 kernel warnings, backend selection messages, unexpected errors>
```

---

## Interpretation boundary

- **You may record** TPS, VRAM, quality scores, startup outcome, CUDA version.
- **You may note** whether th02 passed or failed, and compare TPS numerically to AWQ baseline.
- **You may NOT** update `docs/decisions/settled.md`, `docs/arch/current.md`, or queue files.
- **You may NOT** conclude whether PrismaQuant should replace AWQ in production.
- **You may NOT** enable `--speculative-config` / MTP — that requires V1 engine research first.

## Stop condition

**Normal:** STARTUP_OK, TPS sweep written, quality bench written, summary.md written,
production thinker restored.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` and stop if:
- Startup fails with an error other than OOM (note exact error class from startup_failure.txt)
- th02 fails with a different error pattern than the known GDN TP=2 failure (unexpected)
- TPS at N=1 is below 30 t/s (suggests wrong kernel path or serialization overhead)
- Production thinker fails to restore after Step 7
