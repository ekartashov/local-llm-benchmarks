# BENCH_10 — T_CRIU3 Phase 2: CRIU KV Cache Preservation (Coder TP=2)

**Status: READY**
**Blocks: nothing**
**Blocked by: nothing** (T_KV2 ✓ — coder CRIU mechanism settled; T3.4 ✓ — prefix cache confirmed working)

---

## Title
T_CRIU3 Phase 2 — CRIU KV cache preservation for vLLM coder (TP=2, fp8, prefix-caching)

## Objective
Determine whether a CRIU checkpoint taken with a populated prefix cache (active KV blocks in GPU VRAM + block-map metadata in CPU process RAM) restores with those blocks intact. If yes, post-restore requests identical to pre-checkpoint requests incur warm (cached) latency rather than cold (full-prefill) latency. This is the key advantage of CRIU over sleep mode: sleep drops KV state on process restart; CRIU should preserve it.

## Why this exists
T_KV2 proved 0.28s hot restart for vLLM coder TP=2. T3.4 proved prefix cache works (cold ~2400ms → warm ~173ms, ~13.9× speedup). The open question: does CRIU capture both the CPU-side prefix cache hash tables (process memory) AND the GPU-side KV block data (VRAM via cuda-checkpoint), such that a restored process answers the same prompt with warm latency? If yes, CRIU provides instant restart AND KV continuity — qualitatively better than sleep mode.

**Note on checkpoint size delta:** vLLM preallocates all KV blocks at startup (up to `gpu-mem-util`). VRAM footprint is constant regardless of cache population. cuda-checkpoint captures all VRAM regardless of block content. There is no meaningful checkpoint size delta between empty and populated KV cache for vLLM — the size equals model weights + all preallocated KV blocks in every case.

## Prerequisites

```bash
# 1. cuda-checkpoint plugin installed (settled from T_KV2)
ls /usr/local/lib/criu/cuda-checkpoint.so 2>/dev/null \
  || ls /usr/lib/criu/cuda-checkpoint.so 2>/dev/null \
  && echo "PLUGIN OK" || echo "PLUGIN MISSING — stop"

# 2. vLLM uvloop patch in place (settled from T_KV2)
# Without this, CRIU fails with: Unknown shit 600 (anon_inode:[io_uring])
# Patch: asyncio.run() instead of uvloop.run() in api_server.py and v1/utils.py

# 3. Free disk space (~40 GB needed — coder TP=2 checkpoint)
df -h /srv/ai/checkpoints 2>/dev/null || df -h /srv/ai

# 4. Coder container name
podman ps --format "{{.Names}}\t{{.Status}}" | grep -E "coder|tp2|bench-vllm"
```

**This test runs on the HOST.** `podman container checkpoint` is a host-side command.

**The coder will be redeployed** with `--enable-prefix-caching` + `UV_USE_IO_URING=0`. Step 9 restores the production coder (without prefix caching) at the end.

## Inputs required
- cuda-checkpoint plugin installed
- ~40 GB free disk space at `/srv/ai/checkpoints`
- `infra/scripts/deploy.sh`

## Fixed controls
| Control | Value |
|---------|-------|
| Model | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit |
| Engine | vLLM 0.19.0, TP=2, GPU0+1 |
| Context ceiling | 32768 |
| KV cache dtype | fp8 |
| Prefix caching | enabled (--enable-prefix-caching) |
| Test prompt | "The quick brown fox jumps over the lazy dog. " × 300 + "Summarize the above in one sentence." |
| max_tokens | 10 |
| temperature | 0.0 |
| Checkpoint path | /srv/ai/checkpoints/coder-tp2-kvcache/checkpoint.tar.gz |

## Single variable under test
**Does CRIU preserve GPU VRAM KV cache blocks and CPU-side prefix cache metadata?** — measured by comparing post-restore TTFT to pre-checkpoint warm (cache hit) and cold (cache miss) baselines.

## Procedure

```bash
mkdir -p /srv/ai/checkpoints/coder-tp2-kvcache
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/T_CRIU3_kvcache_preservation_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
```

### Step 1 — Redeploy coder with prefix caching and UV_USE_IO_URING=0

```bash
CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -E "coder|tp2a|bench-vllm" | head -1)
echo "Stopping: ${CODER_CONTAINER}"
podman stop "${CODER_CONTAINER}" 2>/dev/null; podman rm "${CODER_CONTAINER}" 2>/dev/null

VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --gpu-mem-util 0.90 \
  --ctx 32768 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice

for i in $(seq 1 120); do
  curl -sf http://localhost:30000/health && echo "CODER READY" && break
  sleep 1
done
```

### Step 2 — Write test prompt file

```bash
python3 -c "
import json
base = 'The quick brown fox jumps over the lazy dog. '
prompt = base * 300 + 'Summarize the above in one sentence.'
payload = {'model': 'coder', 'prompt': prompt, 'max_tokens': 10, 'temperature': 0.0}
with open('/tmp/bench_kvcache_payload.json', 'w') as f:
    json.dump(payload, f)
approx_tokens = len(prompt) // 4
print(f'Payload written. Approx tokens: {approx_tokens}')
"
```

### Step 3 — Cold prefill (first call, no cache)

```bash
START_MS=$(date +%s%3N)
COLD_RESPONSE=$(curl -sf http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/bench_kvcache_payload.json)
END_MS=$(date +%s%3N)
COLD_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "Cold TTFT: ${COLD_TTFT_S}s"
echo "${COLD_RESPONSE}" > "${RESULTS_DIR}/cold_response.json"
echo "cold_ttft_s=${COLD_TTFT_S}" > "${RESULTS_DIR}/timings.txt"
```

### Step 4 — Warm prefill (same prompt, cache hit expected)

```bash
START_MS=$(date +%s%3N)
WARM_RESPONSE=$(curl -sf http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/bench_kvcache_payload.json)
END_MS=$(date +%s%3N)
WARM_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
CACHE_RATIO=$(python3 -c "print(round(${WARM_TTFT_S} / ${COLD_TTFT_S}, 3))")
echo "Warm TTFT: ${WARM_TTFT_S}s (ratio vs cold: ${CACHE_RATIO} — expected < 0.15 based on T3.4)"
echo "${WARM_RESPONSE}" > "${RESULTS_DIR}/warm_response.json"
echo "warm_ttft_s=${WARM_TTFT_S}" >> "${RESULTS_DIR}/timings.txt"
echo "warm_cold_ratio=${CACHE_RATIO}" >> "${RESULTS_DIR}/timings.txt"

if python3 -c "import sys; sys.exit(0 if ${CACHE_RATIO} < 0.30 else 1)"; then
  echo "CACHE HIT CONFIRMED (ratio ${CACHE_RATIO} < 0.30)"
else
  echo "WARN: ratio ${CACHE_RATIO} — prefix cache may not be active. Check --enable-prefix-caching flag."
  echo "Proceeding with checkpoint anyway to measure restoration behavior."
fi
```

### Step 5 — Record VRAM (with populated KV blocks)

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_pre_checkpoint.txt"
cat "${RESULTS_DIR}/vram_pre_checkpoint.txt"
```

### Step 6 — Checkpoint (with populated prefix cache)

```bash
CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -E "coder|tp2a|bench-vllm" | head -1)
echo "Checkpointing: ${CODER_CONTAINER}"

CKPT_START_MS=$(date +%s%3N)
podman container checkpoint \
  --export=/srv/ai/checkpoints/coder-tp2-kvcache/checkpoint.tar.gz \
  --tcp-established \
  "${CODER_CONTAINER}"
CKPT_EXIT=$?
CKPT_END_MS=$(date +%s%3N)
CKPT_ELAPSED_S=$(python3 -c "print(round(($CKPT_END_MS - $CKPT_START_MS) / 1000.0, 2))")
CKPT_SIZE_GB=$(du -sh /srv/ai/checkpoints/coder-tp2-kvcache/checkpoint.tar.gz 2>/dev/null | awk '{print $1}')

echo "Checkpoint exit: ${CKPT_EXIT}"
echo "Checkpoint time: ${CKPT_ELAPSED_S}s"
echo "Checkpoint size: ${CKPT_SIZE_GB}"
echo "checkpoint_exit=${CKPT_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "checkpoint_elapsed_s=${CKPT_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"
echo "checkpoint_size_gb=${CKPT_SIZE_GB}" >> "${RESULTS_DIR}/timings.txt"

if [ "${CKPT_EXIT}" -ne 0 ]; then
  echo "CHECKPOINT_FAILED" > "${RESULTS_DIR}/status.txt"
  echo "Checkpoint failed. Redeploying production coder."
  VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
    --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
  for i in $(seq 1 120); do curl -sf http://localhost:30000/health && break; sleep 1; done
  exit 1
fi
```

> **Expected log lines during checkpoint (not errors):**
> `Error toggling CUDA in process ID <PID>: "initialization error"` — normal, API Server process is not a CUDA process.

### Step 7 — Restore

```bash
RESTORE_START_MS=$(date +%s%3N)
podman container restore \
  --import=/srv/ai/checkpoints/coder-tp2-kvcache/checkpoint.tar.gz \
  --tcp-established
RESTORE_EXIT=$?
RESTORE_END_MS=$(date +%s%3N)
RESTORE_ELAPSED_S=$(python3 -c "print(round(($RESTORE_END_MS - $RESTORE_START_MS) / 1000.0, 2))")

echo "Restore exit: ${RESTORE_EXIT}"
echo "Restore time: ${RESTORE_ELAPSED_S}s"
echo "restore_exit=${RESTORE_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "restore_elapsed_s=${RESTORE_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

for i in $(seq 1 60); do
  curl -sf http://localhost:30000/health && echo "HEALTH OK after restore" && break
  sleep 1
done

if ! curl -sf http://localhost:30000/health 2>/dev/null; then
  echo "RESTORE_FAILED" > "${RESULTS_DIR}/status.txt"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "${RESULTS_DIR}/vram_at_failure.txt"
  # Ghost VRAM cleanup
  sudo nvidia-smi --gpu-reset -i 0
  sudo nvidia-smi --gpu-reset -i 1
  VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
    --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
  exit 1
fi
```

### Step 8 — Post-restore KV cache check

```bash
START_MS=$(date +%s%3N)
POST_RESPONSE=$(curl -sf http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/bench_kvcache_payload.json)
END_MS=$(date +%s%3N)
POST_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")

POST_COLD_RATIO=$(python3 -c "print(round(${POST_TTFT_S} / ${COLD_TTFT_S}, 3))")
POST_WARM_RATIO=$(python3 -c "print(round(${POST_TTFT_S} / ${WARM_TTFT_S}, 3))")

echo "Post-restore TTFT: ${POST_TTFT_S}s"
echo "  ratio vs cold: ${POST_COLD_RATIO}  (< 0.30 → cache preserved; > 0.70 → cache lost)"
echo "  ratio vs warm: ${POST_WARM_RATIO}  (≈ 1.0 → cache preserved)"

echo "${POST_RESPONSE}" > "${RESULTS_DIR}/post_restore_response.json"
echo "post_restore_ttft_s=${POST_TTFT_S}" >> "${RESULTS_DIR}/timings.txt"
echo "post_cold_ratio=${POST_COLD_RATIO}" >> "${RESULTS_DIR}/timings.txt"
echo "post_warm_ratio=${POST_WARM_RATIO}" >> "${RESULTS_DIR}/timings.txt"

KV_VERDICT=$(python3 -c "
r = ${POST_COLD_RATIO}
print('KV_CACHE_PRESERVED' if r < 0.30 else ('KV_CACHE_LOST' if r > 0.70 else 'KV_CACHE_PARTIAL'))
")
echo "kv_verdict=${KV_VERDICT}" >> "${RESULTS_DIR}/timings.txt"
echo "KV cache verdict: ${KV_VERDICT}"

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_post_restore.txt"
echo "RESTORE_OK" > "${RESULTS_DIR}/status.txt"
echo "Results in: ${RESULTS_DIR}"
```

### Step 9 — Restore production coder (without prefix caching)

```bash
CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -E "coder|tp2a|bench-vllm" | head -1)
podman stop "${CODER_CONTAINER}" 2>/dev/null; podman rm "${CODER_CONTAINER}" 2>/dev/null

VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --gpu-mem-util 0.90 \
  --ctx 32768 \
  --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice

for i in $(seq 1 120); do
  curl -sf http://localhost:30000/health && echo "PRODUCTION CODER RESTORED" && break
  sleep 1
done
```

## Metrics to record

| Metric | Source |
|--------|--------|
| Cold TTFT (s) | `timings.txt` |
| Warm TTFT (s) | `timings.txt` |
| Warm/cold ratio | `timings.txt` |
| Checkpoint exit code | `timings.txt` |
| Checkpoint time (s) | `timings.txt` |
| Checkpoint size (GB) | `timings.txt` |
| Restore exit code | `timings.txt` |
| Restore time (s) | `timings.txt` |
| Post-restore TTFT (s) | `timings.txt` |
| Post-restore/cold ratio | `timings.txt` |
| KV verdict | `timings.txt` |

Expected values:
- Cold TTFT: ~2400ms (matching T3.4 baseline)
- Warm TTFT: ~173ms (matching T3.4 baseline, ~13.9× speedup)
- Restore time: ~0.28s (matching T_KV2 baseline)
- KV verdict: KV_CACHE_PRESERVED (CRIU captures both CPU process RAM and GPU VRAM)

## Pass/fail checks

| Check | Condition | Action |
|-------|-----------|--------|
| Checkpoint exits 0 | Required | If non-zero: record error; redeploy production coder |
| Restore exits 0 | Required | If non-zero: ghost VRAM cleanup (`sudo nvidia-smi --gpu-reset -i 0/1`); redeploy |
| Health passes after restore | `/health` returns 200 | If not: redeploy production coder |
| Warm TTFT < 30% of cold TTFT | Confirms prefix cache is active pre-checkpoint | If not: note flag may not have taken effect |
| KV verdict | KV_CACHE_PRESERVED expected | Record actual; note if PARTIAL or LOST |

## Artifacts to write

1. `results/T_CRIU3_kvcache_preservation_<timestamp>/timings.txt`
2. `results/T_CRIU3_kvcache_preservation_<timestamp>/cold_response.json`
3. `results/T_CRIU3_kvcache_preservation_<timestamp>/warm_response.json`
4. `results/T_CRIU3_kvcache_preservation_<timestamp>/post_restore_response.json`
5. `results/T_CRIU3_kvcache_preservation_<timestamp>/vram_pre_checkpoint.txt`
6. `results/T_CRIU3_kvcache_preservation_<timestamp>/vram_post_restore.txt`
7. `results/T_CRIU3_kvcache_preservation_<timestamp>/status.txt`
8. `results/T_CRIU3_kvcache_preservation_<timestamp>/summary.md`:

```markdown
# T_CRIU3 Phase 2 — CRIU KV Cache Preservation — <TIMESTAMP>

## Result
RESTORE_OK / CHECKPOINT_FAILED / RESTORE_FAILED

| Metric | Value |
|--------|-------|
| Cold TTFT | <s> |
| Warm TTFT | <s> |
| Warm/cold ratio | <ratio> |
| Checkpoint exit | 0 |
| Checkpoint time | <s> |
| Checkpoint size | <GB> |
| Restore time | <s> |
| Post-restore TTFT | <s> |
| Post-restore/cold ratio | <ratio> |
| KV verdict | KV_CACHE_PRESERVED / KV_CACHE_LOST / KV_CACHE_PARTIAL |

## Status
RESTORE_OK / CHECKPOINT_FAILED / RESTORE_FAILED
```

**Do NOT write to any file outside `results/T_CRIU3_kvcache_preservation_<timestamp>/`.**

## Interpretation boundary

- **You may record** TTFT values, KV verdict, checkpoint/restore times, and checkpoint size.
- **You may note** whether restore time matches T_KV2 coder baseline (~0.28s).
- **You may note** whether KV_CACHE_PRESERVED means CRIU is strictly better than sleep mode for long-context continuations.
- **You may NOT** update `docs/procedures/criu-ops.md` with KV cache guidance.
- **You may NOT** update `docs/arch/current.md` or architecture docs.

## Stop condition

**Normal:** checkpoint succeeded, restore succeeded, KV verdict recorded, `summary.md` written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` if:
- KV_CACHE_PARTIAL — partially preserved cache is an unexpected state; record post_cold_ratio and the exact TTFT values for research mode interpretation.
- Restore exits 0 but health never returns within 60s — same TP=1 concern applies; document GPU state.
