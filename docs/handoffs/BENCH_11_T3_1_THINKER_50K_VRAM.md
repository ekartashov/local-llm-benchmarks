# BENCH_11 — T3.1 Phase 1: Thinker 50K Context VRAM Feasibility

**Status: READY**
**Blocks: nothing**
**Blocked by: nothing**

---

## Title
T3.1 Phase 1 — Thinker fp8 KV cache VRAM headroom at 50K context (vs production 32K)

## Objective
Determine whether the thinker (QuantTrio/Qwen3.6-27B-AWQ, TP=1, GPU1) can deploy at 51200 token context with fp8 KV cache without OOM on GPU1's 32 GB VRAM. Measure the VRAM delta between production 32K context and 50K context. If deployment succeeds, run a short inference to confirm the model is functional. This is a feasibility gate before any quality evaluation at 50K context.

## Why this exists
Real thinker workloads hit the 32768-token ceiling — long reasoning chains, multi-document synthesis, deep code analysis. T_KV3 (extended context) is BLOCKED on research (GDN TP=2 broken; non-GDN replacement not yet identified). This test answers the immediate sub-question: is 50K context physically feasible at TP=1 within GPU1's 32 GB? The answer directly informs whether T_KV3 Path B (ik_llama.cpp tensor-split) is the only route to extended thinker context.

**Production config:** GPU1 used ~21 GB at 32K context (model weights + CUDA graphs + fp8 KV preallocate). GPU1 has 32 GB total — ~11 GB headroom. Extending from 32K to 50K adds 18432 tokens of KV cache capacity. The expected VRAM increase depends on model architecture; this test measures it empirically.

## Prerequisites

```bash
# 1. Production thinker running (will be stopped)
curl -sf http://localhost:30001/health && echo "THINKER OK" || echo "THINKER DOWN"

# 2. GPU1 VRAM — should be ~21 GB used at production 32K config
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# 3. Nothing else holding GPU1 VRAM
podman ps --format "{{.Names}}\t{{.Status}}"
# If coder is running on GPU0+1 (TP=2), it also occupies GPU1.
# Either: run this test when coder is stopped, or when coder is sleeping (level=1 frees ~92% GPU1 VRAM).
# Both GPUs can coexist — coder uses ~8 GB GPU1 at peak (TP=2 model shard).
# Best to stop coder before Step 2 if GPU1 VRAM is tight.

# 4. Container name
podman ps --format "{{.Names}}\t{{.Status}}" | grep -i "thinker"
```

**This test runs on the HOST.**

**The production thinker will be stopped and redeployed** at 50K context, then restored to production config at the end.

## Inputs required
- Production thinker running on port 30001
- `infra/scripts/deploy.sh`
- No additional tools required

## Fixed controls
| Control | Value |
|---------|-------|
| Model | QuantTrio/Qwen3.6-27B-AWQ |
| Engine | vLLM 0.19.0, TP=1, GPU1 |
| KV cache dtype | fp8 |
| --max-num-seqs | 1 |
| --enable-chunked-prefill | on |
| Test prompt | "List the three laws of thermodynamics in one sentence each." |
| max_tokens | 50 |
| temperature | 0.0 |
| Production context (baseline) | 32768 |
| Test context (variable) | 51200 |

## Single variable under test
**Context length** — `--ctx 51200` vs production `--ctx 32768`. All other flags identical. Measures GPU1 VRAM delta, startup success (OOM vs OK), and inference correctness at 50K context.

## Procedure

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/T3.1_thinker_50k_vram_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
PROMPT="List the three laws of thermodynamics in one sentence each."
```

### Step 1 — Record baseline VRAM and confirm thinker healthy

```bash
# GPU1 VRAM at production 32K context
VRAM_32K=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
echo "GPU1 VRAM (32K baseline): ${VRAM_32K} MiB"
echo "${VRAM_32K}" > "${RESULTS_DIR}/vram_32k_gpu1_mib.txt"
echo "vram_32k_gpu1_mib=${VRAM_32K}" > "${RESULTS_DIR}/timings.txt"

# Short inference to confirm baseline health
START_MS=$(date +%s%3N)
BASELINE_RESPONSE=$(curl -sf http://localhost:30001/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"thinker\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
END_MS=$(date +%s%3N)
BASELINE_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "Baseline TTFT: ${BASELINE_TTFT_S}s"
echo "${BASELINE_RESPONSE}" > "${RESULTS_DIR}/baseline_response.json"
echo "baseline_ttft_s=${BASELINE_TTFT_S}" >> "${RESULTS_DIR}/timings.txt"
```

### Step 2 — Redeploy thinker at 50K context

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
echo "Stopping: ${THINKER_CONTAINER}"
podman stop "${THINKER_CONTAINER}" 2>/dev/null; podman rm "${THINKER_CONTAINER}" 2>/dev/null

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 \
  --ctx 51200 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice

# Wait up to 120s for health
HEALTH_OK=0
for i in $(seq 1 120); do
  if curl -sf http://localhost:30001/health 2>/dev/null; then
    echo "THINKER READY (50K)"
    HEALTH_OK=1
    break
  fi
  sleep 1
done

echo "health_ok=${HEALTH_OK}" >> "${RESULTS_DIR}/timings.txt"
```

### Step 3 — Handle OOM at startup (if HEALTH_OK=0)

```bash
if [ "${HEALTH_OK}" -eq 0 ]; then
  echo "OOM_AT_STARTUP" > "${RESULTS_DIR}/status.txt"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_at_oom.txt"
  echo "GPU VRAM at OOM:"
  cat "${RESULTS_DIR}/vram_at_oom.txt"

  # Collect startup logs for root cause
  THINKER_CONTAINER=$(podman ps -a --format "{{.Names}}" | grep -i "thinker" | head -1)
  podman logs "${THINKER_CONTAINER}" 2>&1 | tail -40 > "${RESULTS_DIR}/startup_logs.txt"
  echo "OOM at 50K context. See startup_logs.txt. Proceeding to Step 5 to restore production thinker."
fi
```

### Step 4 — Record 50K VRAM and run inference (if startup succeeded)

```bash
if [ "${HEALTH_OK}" -eq 1 ]; then
  VRAM_50K=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
  echo "GPU1 VRAM (50K context): ${VRAM_50K} MiB"
  echo "${VRAM_50K}" > "${RESULTS_DIR}/vram_50k_gpu1_mib.txt"
  echo "vram_50k_gpu1_mib=${VRAM_50K}" >> "${RESULTS_DIR}/timings.txt"

  VRAM_DELTA=$((VRAM_50K - VRAM_32K))
  KV_DELTA_PER_KTOK=$(python3 -c "print(round(${VRAM_DELTA} / 18.432, 1))")
  echo "VRAM delta: +${VRAM_DELTA} MiB for +18432 tokens"
  echo "  fp8 KV overhead: ~${KV_DELTA_PER_KTOK} MiB per 1K context tokens"
  echo "vram_delta_mib=${VRAM_DELTA}" >> "${RESULTS_DIR}/timings.txt"
  echo "vram_mib_per_1k_ctx_fp8=${KV_DELTA_PER_KTOK}" >> "${RESULTS_DIR}/timings.txt"

  # Inference at 50K context
  START_MS=$(date +%s%3N)
  RESPONSE_50K=$(curl -sf http://localhost:30001/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"thinker\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
  END_MS=$(date +%s%3N)
  TTFT_50K_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  echo "TTFT at 50K context: ${TTFT_50K_S}s"
  echo "${RESPONSE_50K}" > "${RESULTS_DIR}/inference_50k.json"
  echo "ttft_50k_s=${TTFT_50K_S}" >> "${RESULTS_DIR}/timings.txt"

  # Text comparison with baseline
  BASELINE_TEXT=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/baseline_response.json'))['choices'][0]['text'])" 2>/dev/null)
  TEXT_50K=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/inference_50k.json'))['choices'][0]['text'])" 2>/dev/null)
  [ "${BASELINE_TEXT}" = "${TEXT_50K}" ] && TEXT_MATCH="identical" || TEXT_MATCH="differs"
  echo "Text match vs baseline: ${TEXT_MATCH}"
  echo "text_match=${TEXT_MATCH}" >> "${RESULTS_DIR}/timings.txt"

  echo "STARTUP_OK" > "${RESULTS_DIR}/status.txt"
fi
```

### Step 5 — Restore production thinker

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
[ -n "${THINKER_CONTAINER}" ] && podman stop "${THINKER_CONTAINER}" && podman rm "${THINKER_CONTAINER}"

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 \
  --ctx 32768 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice

for i in $(seq 1 120); do
  curl -sf http://localhost:30001/health && echo "PRODUCTION THINKER RESTORED" && break
  sleep 1
done
echo "Results in: ${RESULTS_DIR}"
```

## Metrics to record

| Metric | Source |
|--------|--------|
| GPU1 VRAM at 32K (MiB) | `timings.txt` |
| GPU1 VRAM at 50K (MiB) | `timings.txt` |
| VRAM delta (MiB) | `timings.txt` |
| fp8 KV overhead per 1K tokens (MiB) | `timings.txt` |
| Startup result | `status.txt` |
| Baseline TTFT (s) | `timings.txt` |
| TTFT at 50K (s) | `timings.txt` |
| Text match | `timings.txt` |

Expected values (if startup succeeds):
- GPU1 VRAM at 32K: ~21,000 MiB (production baseline)
- VRAM delta: estimate 100–400 MiB for +18432 tokens at fp8 (actual is architecture-dependent)
- TTFT at 50K (short prompt): similar to baseline (context length only affects KV allocation, not prefill speed for a short prompt)
- Text match: identical (temperature=0.0)

## Pass/fail checks

| Check | Condition | Action |
|-------|-----------|--------|
| 50K startup does not OOM | `HEALTH_OK=1` | If OOM: record `vram_at_oom.txt` and `startup_logs.txt`; note GPU1 headroom is insufficient |
| Inference returns text | Non-null response at 50K | If null: record error; note may be model issue at higher context |
| Restore exits 0 and health passes | Final production thinker healthy | If not: redeploy production thinker manually |

**OOM at startup is a valid result.** It means the fp8 KV cache for 50K context + model weights + CUDA graphs exceed GPU1's 32 GB at `--gpu-mem-util 0.90`. Record which VRAM limit is binding and how many MiB short.

## Artifacts to write

1. `results/T3.1_thinker_50k_vram_<timestamp>/timings.txt`
2. `results/T3.1_thinker_50k_vram_<timestamp>/baseline_response.json`
3. `results/T3.1_thinker_50k_vram_<timestamp>/vram_32k_gpu1_mib.txt`
4. `results/T3.1_thinker_50k_vram_<timestamp>/status.txt`
5. If STARTUP_OK: `vram_50k_gpu1_mib.txt`, `inference_50k.json`
6. If OOM: `vram_at_oom.txt`, `startup_logs.txt`
7. `results/T3.1_thinker_50k_vram_<timestamp>/summary.md`:

```markdown
# T3.1 Phase 1 — Thinker 50K Context VRAM Feasibility — <TIMESTAMP>

## Result
STARTUP_OK / OOM_AT_STARTUP

| Metric | Value |
|--------|-------|
| GPU1 VRAM at 32K | <MiB> MiB |
| GPU1 VRAM at 50K | <MiB> MiB (or N/A — OOM) |
| VRAM delta | +<MiB> MiB (or N/A) |
| fp8 KV per 1K tokens | ~<MiB> MiB (or N/A) |
| Baseline TTFT | <s> |
| TTFT at 50K | <s> (or N/A) |
| Text match | identical / differs / N/A |

## Status
STARTUP_OK / OOM_AT_STARTUP
```

**Do NOT write to any file outside `results/T3.1_thinker_50k_vram_<timestamp>/`.**

## Interpretation boundary

- **You may record** VRAM delta, fp8 KV overhead per 1K tokens, and startup feasibility.
- **You may note** whether 50K context is viable within GPU1's 32 GB budget.
- **You may NOT** run quality evaluation at 50K context (requires the full thinker task suite — out of scope for this feasibility test).
- **You may NOT** update `docs/arch/current.md`, `docs/decisions/models.md`, or the production config.

## Stop condition

**Normal:** STARTUP_OK or OOM_AT_STARTUP with VRAM data recorded, `summary.md` written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` if:
- Startup fails with an error other than OOM (e.g., CUDA error, chunked-prefill incompatibility at 50K) — note exact error from `startup_logs.txt`.
- TTFT at 50K is >5× baseline TTFT for the same short prompt — unexpected; suggests the engine is doing something unusual at higher context.
