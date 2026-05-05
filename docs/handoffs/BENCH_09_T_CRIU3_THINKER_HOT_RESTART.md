# BENCH_09 — T_CRIU3 Phase 1: Thinker TP=1 CRIU Hot Restart

**Status: READY**
**Blocks: nothing**
**Blocked by: nothing**

---

## Title
T_CRIU3 Phase 1 — CRIU checkpoint and restore of Arclight Thinker (vLLM TP=1, GPU1)

## Objective
Confirm that `podman container checkpoint` + `podman container restore` works for the production thinker (TP=1 on GPU1). Measure checkpoint size and restore time. T_KV2 settled CRIU only for the coder (TP=2, GPU0+1). The thinker uses a different config (single GPU, different CUDA graph profile). This test is required before the thinker can be included in the T_CRIU3 checkpoint library.

## Why this exists
T_KV2 proved 0.28s hot restart for vLLM TP=2 (coder). The thinker uses TP=1 on GPU1 exclusively — the CUDA memory layout, process tree, and CUDA context setup differ from TP=2. Before standardizing CRIU for all vLLM processes (T_CRIU3), the thinker's CRIU feasibility must be confirmed independently. If it works, the thinker can be checkpointed before any model swap, enabling 0.28s restore instead of ~100s cold start.

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — current production config, gotchas, port assignments
2. `docs/procedures/.md` — CRIU procedure, cuda-checkpoint, UV_USE_IO_URING=0 requirement
3. `docs/procedures/vllm-deploy.md` — deploy commands for thinker

## Prerequisites

```bash
# 1. Thinker currently running (production config)
curl -sf http://localhost:30001/health && echo "THINKER OK" || echo "THINKER DOWN"

# 2. cuda-checkpoint plugin installed (settled from T_KV2)
ls /usr/local/lib/criu/cuda-checkpoint.so 2>/dev/null \
  || ls /usr/lib/criu/cuda-checkpoint.so 2>/dev/null \
  && echo "PLUGIN OK" || echo "PLUGIN MISSING — stop"

# 3. vLLM uvloop patch in place (settled from T_KV2)
# Without this, CRIU fails with: Unknown shit 600 (anon_inode:[io_uring])
# Patch: asyncio.run() instead of uvloop.run() in api_server.py and v1/utils.py
# If T_KV2 passed, this is in place. Verify once:
podman inspect $(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1) \
  --format '{{.Config.Env}}' | tr ' ' '\n' | grep -i uv || echo "(env not shown by inspect)"
# Check .md if uncertain.

# 4. Free disk space (~25 GB needed)
df -h /srv/ai/checkpoints 2>/dev/null || df -h /srv/ai

# 5. Container name
podman ps --format "{{.Names}}\t{{.Status}}" | grep -i "thinker"
# Note the exact name — used in checkpoint command
```

**This test runs on the HOST.** `podman container checkpoint` is a host-side command. Claude Code inside claude-box cannot run it.

**The thinker will be stopped and restarted** during this test. The production thinker deploy command does NOT include `UV_USE_IO_URING=0`, which is required for CRIU. Step 1 of the procedure redeploys with it. After the test, the restored thinker will be running with `UV_USE_IO_URING=0` set, which is safe to leave running.

## Inputs required
- Production thinker running on port 30001
- cuda-checkpoint plugin installed
- ~25 GB free disk space
- `infra/scripts/deploy.sh`

## Fixed controls
| Control | Value |
|---------|-------|
| Model | QuantTrio/Qwen3.6-27B-AWQ |
| Engine | vLLM 0.19.0, TP=1, GPU1 |
| Context ceiling | 32768 |
| KV cache dtype | fp8 |
| --max-num-seqs | 1 |
| Test prompt | fixed (see procedure) |
| max_tokens | 50 |
| temperature | 0.0 |
| Checkpoint path | /srv/ai/checkpoints/thinker-tp1/checkpoint.tar.gz |

## Single variable under test
**CRIU feasibility for vLLM TP=1 on GPU1** — does checkpoint succeed, does restore succeed, does inference work after restore?

## Procedure

```bash
mkdir -p /srv/ai/checkpoints/thinker-tp1
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/T_CRIU3_thinker_hot_restart_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
PROMPT="List the three laws of thermodynamics in one sentence each."
```

### Step 1 — Redeploy thinker with UV_USE_IO_URING=0

The production thinker does not include `UV_USE_IO_URING=0`, which is required for CRIU. Stop the current container and redeploy with it:

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
echo "Stopping: ${THINKER_CONTAINER}"
podman stop "${THINKER_CONTAINER}" 2>/dev/null; podman rm "${THINKER_CONTAINER}" 2>/dev/null

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
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
  curl -sf http://localhost:30001/health && echo "THINKER READY" && break
  sleep 1
done
```

### Step 2 — Pre-checkpoint baseline inference

```bash
export UV_USE_IO_URING=0

START_MS=$(date +%s%3N)
PRE_RESPONSE=$(curl -sf http://localhost:30001/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"thinker\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
END_MS=$(date +%s%3N)
PRE_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "Pre-checkpoint TTFT: ${PRE_TTFT_S}s"
echo "${PRE_RESPONSE}" > "${RESULTS_DIR}/pre_checkpoint_response.json"
echo "pre_ttft_s=${PRE_TTFT_S}" > "${RESULTS_DIR}/timings.txt"
```

### Step 3 — Checkpoint

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
echo "Checkpointing: ${THINKER_CONTAINER}"

CKPT_START_MS=$(date +%s%3N)
podman container checkpoint \
  --export=/srv/ai/checkpoints/thinker-tp1/checkpoint.tar.gz \
  --tcp-established \
  "${THINKER_CONTAINER}"
CKPT_EXIT=$?
CKPT_END_MS=$(date +%s%3N)
CKPT_ELAPSED_S=$(python3 -c "print(round(($CKPT_END_MS - $CKPT_START_MS) / 1000.0, 2))")
CKPT_SIZE_GB=$(du -sh /srv/ai/checkpoints/thinker-tp1/checkpoint.tar.gz 2>/dev/null | awk '{print $1}')

echo "Checkpoint exit: ${CKPT_EXIT}"
echo "Checkpoint time: ${CKPT_ELAPSED_S}s"
echo "Checkpoint size: ${CKPT_SIZE_GB}"
echo "checkpoint_exit=${CKPT_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "checkpoint_elapsed_s=${CKPT_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"
echo "checkpoint_size_gb=${CKPT_SIZE_GB}" >> "${RESULTS_DIR}/timings.txt"

if [ "${CKPT_EXIT}" -ne 0 ]; then
  echo "CHECKPOINT_FAILED" > "${RESULTS_DIR}/status.txt"
  echo "Checkpoint failed. Redeploying production thinker."
  VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
  ./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
    --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-seqs 1 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
  for i in $(seq 1 120); do curl -sf http://localhost:30001/health && break; sleep 1; done
  exit 1
fi
```

> **Expected log lines during checkpoint (not errors):**
> `Error toggling CUDA in process ID <PID>: "initialization error"` — normal, API Server process is not a CUDA process.

### Step 4 — Restore

```bash
RESTORE_START_MS=$(date +%s%3N)
podman container restore \
  --import=/srv/ai/checkpoints/thinker-tp1/checkpoint.tar.gz \
  --tcp-established
RESTORE_EXIT=$?
RESTORE_END_MS=$(date +%s%3N)
RESTORE_ELAPSED_S=$(python3 -c "print(round(($RESTORE_END_MS - $RESTORE_START_MS) / 1000.0, 2))")

echo "Restore exit: ${RESTORE_EXIT}"
echo "Restore time: ${RESTORE_ELAPSED_S}s"
echo "restore_exit=${RESTORE_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "restore_elapsed_s=${RESTORE_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

# Wait for health
for i in $(seq 1 60); do
  curl -sf http://localhost:30001/health && echo "HEALTH OK after restore" && break
  sleep 1
done

if ! curl -sf http://localhost:30001/health 2>/dev/null; then
  echo "RESTORE_FAILED" > "${RESULTS_DIR}/status.txt"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "${RESULTS_DIR}/vram_at_failure.txt"
  # If ghost VRAM suspected, run: sudo nvidia-smi --gpu-reset -i 1
  VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
  ./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
    --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-seqs 1 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
  exit 1
fi
```

### Step 5 — Post-restore inference

```bash
START_MS=$(date +%s%3N)
POST_RESPONSE=$(curl -sf http://localhost:30001/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"thinker\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
END_MS=$(date +%s%3N)
POST_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "Post-restore TTFT: ${POST_TTFT_S}s"
echo "${POST_RESPONSE}" > "${RESULTS_DIR}/post_restore_response.json"
echo "post_restore_ttft_s=${POST_TTFT_S}" >> "${RESULTS_DIR}/timings.txt"

# Check text match
PRE_TEXT=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/pre_checkpoint_response.json'))['choices'][0]['text'])")
POST_TEXT=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/post_restore_response.json'))['choices'][0]['text'])")
[ "${PRE_TEXT}" = "${POST_TEXT}" ] && echo "TEXT: identical" || echo "TEXT: differs"
echo "text_match=$([ "${PRE_TEXT}" = "${POST_TEXT}" ] && echo identical || echo differs)" >> "${RESULTS_DIR}/timings.txt"

# VRAM after restore
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_post_restore.txt"
cat "${RESULTS_DIR}/vram_post_restore.txt"

echo "RESTORE_OK" > "${RESULTS_DIR}/status.txt"
echo "Results in: ${RESULTS_DIR}"
```

> The thinker is now running with `UV_USE_IO_URING=0` (safe to leave). No further action needed unless you want to match the exact pre-test config by redeploying without it.

## Metrics to record

| Metric | Source |
|--------|--------|
| Checkpoint exit code | `timings.txt` |
| Checkpoint time (s) | `timings.txt` |
| Checkpoint size (GB) | `timings.txt` |
| Restore exit code | `timings.txt` |
| Restore time (s) | `timings.txt` |
| Pre-checkpoint TTFT (s) | `timings.txt` |
| Post-restore TTFT (s) | `timings.txt` |
| Text match | `timings.txt` |
| GPU1 VRAM after restore (MiB) | `vram_post_restore.txt` |

Expected values:
- Checkpoint size: ~21–23 GB (thinker weights in CUDA VRAM)
- Checkpoint time: < 1 s
- Restore time: ~0.28 s (matching T_KV2 coder result)
- Pre/post TTFT: within 20% of each other

## Pass/fail checks

| Check | Condition | Action |
|-------|-----------|--------|
| Checkpoint exits 0 | Required | If non-zero: record exact error; write `## Open from testing`; redeploy production thinker |
| Restore exits 0 | Required | If non-zero: check for ghost VRAM (`sudo nvidia-smi --gpu-reset -i 1`); redeploy |
| Health passes after restore | `/health` returns 200 | If not: redeploy production thinker |
| Post-restore inference returns text | Non-null response | If null: record error |
| Restore time < 5 s | Expected ~0.28 s | If > 5 s: note — may indicate GPU state serialization difference from coder |

## Artifacts to write

1. `results/T_CRIU3_thinker_hot_restart_<timestamp>/timings.txt` — written by procedure
2. `results/T_CRIU3_thinker_hot_restart_<timestamp>/pre_checkpoint_response.json`
3. `results/T_CRIU3_thinker_hot_restart_<timestamp>/post_restore_response.json`
4. `results/T_CRIU3_thinker_hot_restart_<timestamp>/vram_post_restore.txt`
5. `results/T_CRIU3_thinker_hot_restart_<timestamp>/status.txt`
6. `results/T_CRIU3_thinker_hot_restart_<timestamp>/summary.md`:

```markdown
# T_CRIU3 Phase 1 — Thinker TP=1 CRIU Hot Restart — <TIMESTAMP>

## Result
RESTORE_OK / CHECKPOINT_FAILED / RESTORE_FAILED

| Metric | Value |
|--------|-------|
| Checkpoint exit | 0 |
| Checkpoint time | <s> |
| Checkpoint size | <GB> |
| Restore time | <s> |
| Pre-checkpoint TTFT | <s> |
| Post-restore TTFT | <s> |
| Text match | identical / differs |

## GPU1 VRAM after restore
<MiB> MiB

## Incidental findings
<Any observation outside this benchmark's explicit scope: unexpected VRAM readings, other components
behaving differently than documented, engine warnings about kernels or flags, etc. Write one FINDING
block per observation. If nothing unusual observed: "none">

## Status
RESTORE_OK / CHECKPOINT_FAILED / RESTORE_FAILED
```

**Do NOT write to any file outside `results/T_CRIU3_thinker_hot_restart_<timestamp>/`.**

## Interpretation boundary

- **You may record** checkpoint size, checkpoint time, restore time, and inference correctness.
- **You may note** whether restore time matches the T_KV2 coder baseline (~0.28s).
- **You may NOT** add the thinker to the production CRIU checkpoint library policy.
- **You may NOT** update `docs/arch/current.md` or `docs/procedures/.md`.

## Stop condition

**Normal:** checkpoint succeeded, restore succeeded, inference correct, `summary.md` written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` if:
- Checkpoint fails with `Unknown shit 600 (anon_inode:[io_uring])` — uvloop patch is missing from the thinker's vLLM image; research mode required to verify patch location.
- Checkpoint fails with an unfamiliar fd type not in `docs/procedures/.md` — unknown failure mode; research required.
- Restore exits 0 but `/health` never returns within 60 s — may be a CUDA context reinitializataion issue specific to TP=1; record GPU state and VRAM.
