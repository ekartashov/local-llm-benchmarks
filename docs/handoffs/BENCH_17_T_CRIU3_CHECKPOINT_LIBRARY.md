# BENCH_17 — T_CRIU3: CRIU Checkpoint Library Standardization

**Status: READY**
**Blocks: BENCH_18 (QX_PRELOAD)**
**Blocked by: T_KV2 ✓, T_CRIU2 ✓**

---

## Title
T_CRIU3 — criu_universal_checkpoint_library — Production checkpoints for all active vLLM roles

## Objective
Standardize CRIU checkpoint naming and storage for the two active vLLM endpoints (thinker TP=1
and coder TP=2). Create production checkpoints for both. Verify that each restores cleanly to a
functioning inference endpoint. Measure KV cache size delta between a cold (idle) checkpoint and
a KV-populated checkpoint. Confirm whether KV cache is preserved across restore (TTFT on a
continuation is lower than cold re-prefill of the same context).

This is the checkpoint library that BENCH_18 (QX_PRELOAD) will consume. It must exist before
QX_PRELOAD can be measured.

## Why this exists

Prior settled results:
- **T_KV2:** CRIU restore from warm OS page cache = 0.28s (coder TP=2 baseline).
- **T_CRIU3 Phase 1 (BENCH_09):** Thinker TP=1 checkpoint = 501 MB, restore = 0.43s, inference
  correct post-restore. CRIU feasibility confirmed for both active models.
- **T_CRIU2 (BENCH_07/08):** Convergence (ik_llama.cpp --no-mmap) checkpoint = 8.7 GB, restore =
  7s, first inference = 100s without page-cache pre-warm. QX_PRELOAD mechanism is required for
  Convergence viability.

What is still missing:
- A defined, documented naming convention for checkpoint files.
- Production checkpoints at known paths that scripts can reference.
- The KV cache size delta measurement (queue item T_CRIU3 scope explicitly includes this).
- Confirmation that KV preservation works for the thinker in the same way T_CRIU3 Phase 1
  confirmed inference correctness.

Note on coder TP=2: BENCH_10 confirmed that CRIU restore of coder TP=2 results in SHM IPC
deadlock (ShmRingBuffer written_flag invisible to workers after restore). This is a known,
documented architectural incompatibility. The coder checkpoint in this handoff is therefore
created for completeness of the library (cold-boot snapshot, pre-warm target for QX_PRELOAD),
**not** as a live fast-swap mechanism. Restore-to-inference is not expected to succeed for
coder TP=2. Document the result either way.

---

## Prerequisites

```bash
# 1. Thinker running (production config)
curl -sf http://localhost:30001/health && echo "THINKER OK" || echo "THINKER DOWN — deploy first"

# 2. Coder running (production config)
curl -sf http://localhost:30000/health && echo "CODER OK" || echo "CODER DOWN — deploy first"

# 3. cuda-checkpoint binary available (settled from T_KV2)
which cuda-checkpoint && cuda-checkpoint --version \
  || ls /usr/local/bin/cuda-checkpoint /usr/bin/cuda-checkpoint 2>/dev/null \
  || echo "cuda-checkpoint MISSING — stop"

# 4. CRIU plugin present (required for CUDA memory capture)
ls /usr/local/lib/criu/cuda-checkpoint.so 2>/dev/null \
  || ls /usr/lib/criu/cuda-checkpoint.so 2>/dev/null \
  && echo "CRIU PLUGIN OK" || echo "PLUGIN MISSING — stop"

# 5. Checkpoint storage base — check free space
# Minimum needed: ~1 GB thinker (cold) + ~70 GB coder (cold, all KV blocks preallocated) + headroom
mkdir -p /srv/ai/checkpoints
df -h /srv/ai/checkpoints

# 6. UV_USE_IO_URING=0 requirement (settled from T_KV2 / BENCH_09)
# Thinker must have been deployed with UV_USE_IO_URING=0 for CRIU to succeed.
# Check by looking at the running container environment:
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
podman inspect "${THINKER_CONTAINER}" --format '{{range .Config.Env}}{{.}}\n{{end}}' \
  | grep UV_USE_IO_URING || echo "WARNING: UV_USE_IO_URING not visible — redeploy needed (Step 1)"
```

**This test runs on the HOST.** Claude Code inside claude-box cannot run it.

## Inputs required
- Production thinker running at `http://localhost:30001`, deployed with `UV_USE_IO_URING=0` (Step 1 redeploys if missing)
- Production coder running at `http://localhost:30000`, deployed with `UV_USE_IO_URING=0`
- `cuda-checkpoint` binary available (settled from T_KV2 / BENCH_09)
- CRIU plugin at `/usr/local/lib/criu/cuda-checkpoint.so` or `/usr/lib/criu/cuda-checkpoint.so`
- `/srv/ai/checkpoints/` directory with ≥ 80 GB free (thinker ~501 MiB, coder ~67 GB)
- `infra/scripts/deploy.sh` for Step 7 production restore

---

## Naming convention (implement and document)

```
<base>/<role>/<model_slug>-tp<n>-ctx<k>k[-kv<tokens>k].ckpt
```

| Component | Value |
|-----------|-------|
| `<base>` | `/srv/ai/checkpoints` |
| `<role>` | `thinker`, `coder`, `convergence` |
| `<model_slug>` | short model name, no publisher prefix |
| `-tp<n>` | tensor parallel degree |
| `-ctx<k>k` | `--max-model-len` in thousands |
| `-kv<tokens>k` | optional suffix when KV cache is populated at checkpoint time |

Examples:
- `/srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k.ckpt` — cold/idle thinker checkpoint
- `/srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k-kv16k.ckpt` — thinker with 16K tokens in KV cache
- `/srv/ai/checkpoints/coder/qwen36-35b-a3b-awq-tp2-ctx32k.ckpt` — cold coder checkpoint

Document this convention in `docs/procedures/criu-checkpoint-library.md` (Step 8).

---

## Fixed controls

| Control | Value |
|---------|-------|
| Thinker model | QuantTrio/Qwen3.6-27B-AWQ |
| Thinker engine | vLLM TP=1, GPU1, port 30001 |
| Coder model | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit |
| Coder engine | vLLM TP=2, GPU0+GPU1, port 30000 |
| Checkpoint tool | `cuda-checkpoint` (NVIDIA) |
| Checkpoint base | `/srv/ai/checkpoints/` |
| KV sub-measurement context | ~16K tokens |
| KV continuation prompt | send the same conversation context, ask for a follow-up |

## Single variable under test
**Checkpoint creation and restore success per active vLLM endpoint** — does `cuda-checkpoint` produce a restoreable checkpoint for each active model? Measured on thinker (TP=1, expected OK) and coder (TP=2, expected checkpoint-only due to known SHM IPC bug from BENCH_10). KV cache delta size and TTFT preservation ratio are additional measurements taken during the same session.

---

## Procedure

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_17_criu3_checkpoint_library_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

mkdir -p /srv/ai/checkpoints/thinker
mkdir -p /srv/ai/checkpoints/coder

THINKER_CKPT="/srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k.ckpt"
CODER_CKPT="/srv/ai/checkpoints/coder/qwen36-35b-a3b-awq-tp2-ctx32k.ckpt"
THINKER_KV_CKPT="/srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k-kv16k.ckpt"
```

### Step 1 — Redeploy thinker with UV_USE_IO_URING=0 (if needed)

Skip this step if the thinker was already deployed with `UV_USE_IO_URING=0`.

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
podman stop "${THINKER_CONTAINER}" 2>/dev/null; podman rm "${THINKER_CONTAINER}" 2>/dev/null

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
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
  curl -sf http://localhost:30001/health && echo "THINKER READY" && break; sleep 1
done
```

### Step 2 — Create thinker cold checkpoint

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
echo "Thinker container: ${THINKER_CONTAINER}"

CKPT_START_MS=$(date +%s%3N)
podman container checkpoint \
  --export="${THINKER_CKPT}" \
  --tcp-established \
  "${THINKER_CONTAINER}"
CKPT_EXIT=$?
CKPT_END_MS=$(date +%s%3N)
THINKER_CKPT_TIME_S=$(python3 -c "print(round(($CKPT_END_MS - $CKPT_START_MS) / 1000.0, 2))")
THINKER_CKPT_SIZE_MIB=$(du -m "${THINKER_CKPT}" 2>/dev/null | awk '{print $1}')

echo "thinker_cold_checkpoint_exit=${CKPT_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "thinker_cold_checkpoint_time_s=${THINKER_CKPT_TIME_S}" >> "${RESULTS_DIR}/timings.txt"
echo "thinker_cold_checkpoint_size_mib=${THINKER_CKPT_SIZE_MIB}" >> "${RESULTS_DIR}/timings.txt"
echo "Thinker cold checkpoint: exit=${CKPT_EXIT} time=${THINKER_CKPT_TIME_S}s size=${THINKER_CKPT_SIZE_MIB} MiB"

if [ "${CKPT_EXIT}" -ne 0 ]; then
  echo "THINKER_CHECKPOINT_FAILED" > "${RESULTS_DIR}/status.txt"
  echo "Thinker checkpoint failed. Check error above. Writing Open from testing."
  exit 1
fi
```

> **Expected during checkpoint:** `Error toggling CUDA in process ID <N>: "initialization error"` —
> this is normal; the API server process is not a CUDA process.

### Step 3 — Restore thinker from cold checkpoint, verify inference

```bash
RESTORE_START_MS=$(date +%s%3N)
podman container restore \
  --import="${THINKER_CKPT}" \
  --tcp-established
RESTORE_EXIT=$?
RESTORE_END_MS=$(date +%s%3N)
THINKER_RESTORE_TIME_S=$(python3 -c "print(round(($RESTORE_END_MS - $RESTORE_START_MS) / 1000.0, 2))")

echo "thinker_cold_restore_exit=${RESTORE_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "thinker_cold_restore_time_s=${THINKER_RESTORE_TIME_S}" >> "${RESULTS_DIR}/timings.txt"
echo "Thinker cold restore: exit=${RESTORE_EXIT} time=${THINKER_RESTORE_TIME_S}s"

# Wait for health
for i in $(seq 1 60); do
  curl -sf http://localhost:30001/health && echo "THINKER HEALTH OK after restore" && break; sleep 1
done

if ! curl -sf http://localhost:30001/health 2>/dev/null; then
  echo "THINKER_RESTORE_NO_HEALTH" > "${RESULTS_DIR}/status.txt"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "${RESULTS_DIR}/vram_at_failure.txt"
  exit 1
fi

# First-inference latency after restore
INFER_START_MS=$(date +%s%3N)
INFER_RESPONSE=$(curl -sf http://localhost:30001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"thinker","prompt":"What is the boiling point of water in Celsius?","max_tokens":20,"temperature":0.0}')
INFER_END_MS=$(date +%s%3N)
THINKER_RESTORE_INFER_MS=$(( INFER_END_MS - INFER_START_MS ))
echo "${INFER_RESPONSE}" > "${RESULTS_DIR}/thinker_cold_restore_infer.json"
echo "thinker_cold_restore_first_infer_ms=${THINKER_RESTORE_INFER_MS}" >> "${RESULTS_DIR}/timings.txt"
echo "Thinker first-inference after restore: ${THINKER_RESTORE_INFER_MS}ms"
```

### Step 4 — KV cache sub-measurement: populated checkpoint size and TTFT preservation

This step measures whether checkpointing with a warm KV cache changes the checkpoint size and
whether KV state survives the restore.

```bash
# 4a. Re-warm the thinker with a long-context prompt (~16K tokens)
# Use a repetitive structured payload to hit the token target reliably.
LONG_CONTEXT_PROMPT=$(python3 -c "
base = 'The following is a numbered list of infrastructure engineering principles. '
items = [f'{i}. Principle: idempotency must be enforced at every layer of the stack, '\
         f'and all state transitions must be observable and reversible. ' for i in range(1, 400)]
print(base + ' '.join(items))
")

echo "Sending long-context prompt (~16K tokens) to thinker..."
WARM_START_MS=$(date +%s%3N)
WARM_RESPONSE=$(curl -sf http://localhost:30001/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"thinker\",\"prompt\":$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "${LONG_CONTEXT_PROMPT}"),\"max_tokens\":100,\"temperature\":0.0}")
WARM_END_MS=$(date +%s%3N)
echo "${WARM_RESPONSE}" > "${RESULTS_DIR}/thinker_kv_warm_response.json"
WARM_INFER_MS=$(( WARM_END_MS - WARM_START_MS ))
echo "thinker_kv_warm_infer_ms=${WARM_INFER_MS}" >> "${RESULTS_DIR}/timings.txt"
echo "Long-context prefill complete: ${WARM_INFER_MS}ms"

# 4b. Checkpoint with KV populated
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
KV_CKPT_START_MS=$(date +%s%3N)
podman container checkpoint \
  --export="${THINKER_KV_CKPT}" \
  --tcp-established \
  "${THINKER_CONTAINER}"
KV_CKPT_EXIT=$?
KV_CKPT_END_MS=$(date +%s%3N)
KV_CKPT_TIME_S=$(python3 -c "print(round(($KV_CKPT_END_MS - $KV_CKPT_START_MS) / 1000.0, 2))")
KV_CKPT_SIZE_MIB=$(du -m "${THINKER_KV_CKPT}" 2>/dev/null | awk '{print $1}')
KV_DELTA_MIB=$(( KV_CKPT_SIZE_MIB - THINKER_CKPT_SIZE_MIB ))

echo "thinker_kv_checkpoint_exit=${KV_CKPT_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "thinker_kv_checkpoint_time_s=${KV_CKPT_TIME_S}" >> "${RESULTS_DIR}/timings.txt"
echo "thinker_kv_checkpoint_size_mib=${KV_CKPT_SIZE_MIB}" >> "${RESULTS_DIR}/timings.txt"
echo "thinker_kv_checkpoint_delta_mib=${KV_DELTA_MIB}" >> "${RESULTS_DIR}/timings.txt"
echo "KV checkpoint: exit=${KV_CKPT_EXIT} time=${KV_CKPT_TIME_S}s size=${KV_CKPT_SIZE_MIB} MiB (delta vs cold: ${KV_DELTA_MIB} MiB)"

# 4c. Restore from KV-populated checkpoint
KV_RESTORE_START_MS=$(date +%s%3N)
podman container restore \
  --import="${THINKER_KV_CKPT}" \
  --tcp-established
KV_RESTORE_EXIT=$?
KV_RESTORE_END_MS=$(date +%s%3N)
KV_RESTORE_TIME_S=$(python3 -c "print(round(($KV_RESTORE_END_MS - $KV_RESTORE_START_MS) / 1000.0, 2))")

echo "thinker_kv_restore_exit=${KV_RESTORE_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "thinker_kv_restore_time_s=${KV_RESTORE_TIME_S}" >> "${RESULTS_DIR}/timings.txt"

for i in $(seq 1 60); do
  curl -sf http://localhost:30001/health && echo "THINKER HEALTH OK after KV restore" && break; sleep 1
done

# 4d. Send a continuation prompt — should hit KV cache (low TTFT) if KV preserved
CONTINUATION=$(python3 -c "
import json
base = 'The following is a numbered list of infrastructure engineering principles. '
items = [f'{i}. Principle: idempotency must be enforced at every layer of the stack, '\
         f'and all state transitions must be observable and reversible. ' for i in range(1, 400)]
# Repeat the same prompt (prefix cache match) then ask a follow-up
full = base + ' '.join(items)
print(json.dumps(full + ' Based on the above principles, summarize principle 1 in ten words.'))
")

CONT_START_MS=$(date +%s%3N)
CONT_RESPONSE=$(curl -sf http://localhost:30001/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"thinker\",\"prompt\":${CONTINUATION},\"max_tokens\":50,\"temperature\":0.0}")
CONT_END_MS=$(date +%s%3N)
CONT_INFER_MS=$(( CONT_END_MS - CONT_START_MS ))
echo "${CONT_RESPONSE}" > "${RESULTS_DIR}/thinker_kv_continuation_response.json"
echo "thinker_kv_continuation_infer_ms=${CONT_INFER_MS}" >> "${RESULTS_DIR}/timings.txt"

# KV preservation indicator: continuation TTFT should be lower than warm_infer_ms (no re-prefill)
python3 - <<EOF
warm = ${WARM_INFER_MS}
cont = ${CONT_INFER_MS}
ratio = cont / warm if warm > 0 else 0
kv_preserved = ratio < 0.5  # continuation much faster = cache hit
print(f"Long-context prefill (warm):  {warm} ms")
print(f"Continuation after KV restore: {cont} ms")
print(f"Ratio (continuation/prefill):  {ratio:.2f}")
print(f"KV preservation indicator: {'LIKELY YES (ratio < 0.5)' if kv_preserved else 'LIKELY NO (ratio >= 0.5)'}")
EOF
```

> **Interpretation:** vLLM preallocates all KV blocks at startup regardless of actual cache use —
> the KV-populated checkpoint size delta may be zero or near-zero (VRAM footprint is fixed at
> initialization). The TTFT comparison is the primary KV preservation signal.

### Step 5 — Create coder cold checkpoint

**Note:** Based on BENCH_10 findings, coder TP=2 restore-to-inference will likely fail due to
SHM IPC incompatibility. Create the checkpoint anyway for the QX_PRELOAD pre-warm library.
Attempt restore and document the outcome.

```bash
CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|gpu0\|35b" | head -1)
echo "Coder container: ${CODER_CONTAINER}"

# Coder must be deployed with UV_USE_IO_URING=0 for CRIU to succeed.
# If not, redeploy:
# VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
# ./infra/scripts/deploy.sh vllm gpu0gpu1 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
#   --tensor-parallel-size 2 --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
#   --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
# for i in $(seq 1 120); do curl -sf http://localhost:30000/health && echo "CODER READY" && break; sleep 1; done

CODER_CKPT_START_MS=$(date +%s%3N)
podman container checkpoint \
  --export="${CODER_CKPT}" \
  --tcp-established \
  "${CODER_CONTAINER}"
CODER_CKPT_EXIT=$?
CODER_CKPT_END_MS=$(date +%s%3N)
CODER_CKPT_TIME_S=$(python3 -c "print(round(($CODER_CKPT_END_MS - $CODER_CKPT_START_MS) / 1000.0, 2))")
CODER_CKPT_SIZE_MIB=$(du -m "${CODER_CKPT}" 2>/dev/null | awk '{print $1}')

echo "coder_cold_checkpoint_exit=${CODER_CKPT_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "coder_cold_checkpoint_time_s=${CODER_CKPT_TIME_S}" >> "${RESULTS_DIR}/timings.txt"
echo "coder_cold_checkpoint_size_mib=${CODER_CKPT_SIZE_MIB}" >> "${RESULTS_DIR}/timings.txt"
echo "Coder cold checkpoint: exit=${CODER_CKPT_EXIT} time=${CODER_CKPT_TIME_S}s size=${CODER_CKPT_SIZE_MIB} MiB"
```

### Step 6 — Test coder restore (expected to fail inference, document outcome)

```bash
CODER_RESTORE_START_MS=$(date +%s%3N)
podman container restore \
  --import="${CODER_CKPT}" \
  --tcp-established
CODER_RESTORE_EXIT=$?
CODER_RESTORE_END_MS=$(date +%s%3N)
CODER_RESTORE_TIME_S=$(python3 -c "print(round(($CODER_RESTORE_END_MS - $CODER_RESTORE_START_MS) / 1000.0, 2))")

echo "coder_restore_exit=${CODER_RESTORE_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "coder_restore_time_s=${CODER_RESTORE_TIME_S}" >> "${RESULTS_DIR}/timings.txt"
echo "Coder restore: exit=${CODER_RESTORE_EXIT} time=${CODER_RESTORE_TIME_S}s"

for i in $(seq 1 60); do
  curl -sf http://localhost:30000/health && echo "CODER HEALTH OK (unexpected — note this)" && break; sleep 1
done

CODER_HEALTH=$(curl -sf http://localhost:30000/health 2>&1; echo "exit=$?")

# Attempt inference regardless of health result
CODER_INFER_RESPONSE=$(curl -s --max-time 30 http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"coder","prompt":"def add(a, b):","max_tokens":20,"temperature":0.0}' 2>&1)
echo "${CODER_INFER_RESPONSE}" > "${RESULTS_DIR}/coder_restore_infer.json"
echo "coder_restore_health=${CODER_HEALTH}" >> "${RESULTS_DIR}/timings.txt"

# Collect logs regardless of outcome
podman logs $(podman ps -a --format "{{.Names}}" | grep -i "coder" | head -1) \
  2>&1 | tail -60 > "${RESULTS_DIR}/coder_restore_logs.txt"
```

> **Expected:** health may return 200 (HTTP layer works), but inference will timeout with
> `RPC call to execute_model timed out` — this is the SHM IPC bug documented in BENCH_10.
> Record the exact error. The checkpoint itself is still valid for QX_PRELOAD.

### Step 7 — Restore both production endpoints to normal operation

```bash
# Stop any lingering restored containers
for C in $(podman ps --format "{{.Names}}" | grep -iE "thinker|coder"); do
  podman stop "${C}" && podman rm "${C}"
done

# Restore production thinker
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-seqs 4 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

for i in $(seq 1 120); do
  curl -sf http://localhost:30001/health && echo "PRODUCTION THINKER RESTORED" && break; sleep 1
done

# Restore production coder
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
./infra/scripts/deploy.sh vllm gpu0gpu1 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --tensor-parallel-size 2 --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

for i in $(seq 1 120); do
  curl -sf http://localhost:30000/health && echo "PRODUCTION CODER RESTORED" && break; sleep 1
done
```

### Step 8 — Write naming convention documentation

Create `docs/procedures/criu-checkpoint-library.md` with:
- Naming convention table (from the specification above)
- Inventory of created checkpoints with paths, sizes, and dates
- Per-checkpoint restore viability (thinker: OK; coder: checkpoint-only, inference blocked by SHM IPC bug)
- SSD write cost model: size × dumps/day → GB/day → years to TBW limit (NM790 = 3,000 TBW)

---

## Metrics to record

| Metric | File | Expected / Prior |
|--------|------|-----------------|
| Thinker cold checkpoint size (MiB) | `timings.txt` | ~501 MiB (BENCH_09) |
| Thinker cold checkpoint time (s) | `timings.txt` | ~0.4s |
| Thinker cold restore time (s) | `timings.txt` | ~0.43s (BENCH_09) |
| Thinker first-inference after restore (ms) | `timings.txt` | < 5000ms |
| Thinker KV checkpoint size (MiB) | `timings.txt` | ~same as cold (VRAM preallocated) |
| Thinker KV checkpoint delta vs cold (MiB) | `timings.txt` | 0–50 MiB expected |
| KV continuation TTFT (ms) | `timings.txt` | — |
| KV prefill TTFT (ms) | `timings.txt` | — |
| KV preservation indicator (ratio) | stdout | < 0.5 = cache hit |
| Coder cold checkpoint size (MiB) | `timings.txt` | ~67,000 MiB (BENCH_10) |
| Coder cold checkpoint time (s) | `timings.txt` | ~29s (BENCH_10) |
| Coder restore exit code | `timings.txt` | 0 (restore succeeds) |
| Coder restore time (s) | `timings.txt` | ~24–26s (BENCH_10) |
| Coder inference post-restore | `coder_restore_infer.json` | EXPECTED FAIL (SHM IPC) |

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Thinker cold checkpoint exits 0 | Required | Record error; write `## Open from testing`; stop |
| Thinker cold restore health OK | `/health` 200 within 60s | Record VRAM; redeploy production; write Open from testing |
| Thinker first-inference returns text | Non-null response in <30s | Record error; note if new failure mode vs BENCH_09 |
| Thinker KV checkpoint exits 0 | Required | Record error; proceed to coder steps anyway |
| Coder cold checkpoint exits 0 | Required | If fails differently than BENCH_10: write Open from testing |
| Coder restore exits 0 | Expected | If non-zero: record exact error |
| Production endpoints restored | Both `/health` 200 | Redeploy manually if restore command fails |

**Coder inference failure is expected and is NOT a stop condition.** Document outcome in summary.md.

---

## Artifacts to write

1. `/srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k.ckpt` — cold thinker checkpoint
2. `/srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k-kv16k.ckpt` — KV-populated thinker checkpoint
3. `/srv/ai/checkpoints/coder/qwen36-35b-a3b-awq-tp2-ctx32k.ckpt` — cold coder checkpoint (pre-warm only)
4. `docs/procedures/criu-checkpoint-library.md` — naming convention + inventory
5. `results/BENCH_17_criu3_checkpoint_library_<timestamp>/timings.txt` — all timing values
6. `results/BENCH_17_criu3_checkpoint_library_<timestamp>/thinker_cold_restore_infer.json`
7. `results/BENCH_17_criu3_checkpoint_library_<timestamp>/thinker_kv_warm_response.json`
8. `results/BENCH_17_criu3_checkpoint_library_<timestamp>/thinker_kv_continuation_response.json`
9. `results/BENCH_17_criu3_checkpoint_library_<timestamp>/coder_restore_infer.json`
10. `results/BENCH_17_criu3_checkpoint_library_<timestamp>/coder_restore_logs.txt`
11. `results/BENCH_17_criu3_checkpoint_library_<timestamp>/summary.md`:

```markdown
# BENCH_17 — T_CRIU3 Checkpoint Library — <TIMESTAMP>

## Thinker cold checkpoint
| Metric | Value |
|--------|-------|
| Checkpoint size | <MiB> MiB |
| Checkpoint time | <s> s |
| Restore time | <s> s |
| First-inference latency | <ms> ms |
| Result | RESTORE_OK / RESTORE_FAILED |

## Thinker KV cache sub-measurement
| Metric | Value |
|--------|-------|
| KV checkpoint size | <MiB> MiB |
| Delta vs cold | <MiB> MiB |
| Long-context prefill TTFT | <ms> ms |
| Continuation TTFT after KV restore | <ms> ms |
| KV preservation indicator | LIKELY YES / LIKELY NO |

## Coder cold checkpoint
| Metric | Value |
|--------|-------|
| Checkpoint size | <MiB> MiB |
| Checkpoint time | <s> s |
| Restore time | <s> s |
| Inference post-restore | FAIL (SHM IPC) / UNEXPECTED OK |
| Error observed | <paste first 3 lines of error> |

## Naming convention
Documented in docs/procedures/criu-checkpoint-library.md

## Overall status
PASS / PARTIAL (thinker OK, coder checkpoint-only)
```

**Do NOT write to any file outside `results/BENCH_17_criu3_checkpoint_library_<timestamp>/` and
the `/srv/ai/checkpoints/` subtree and `docs/procedures/criu-checkpoint-library.md`.**

---

## Interpretation boundary

- **You may record** checkpoint sizes, timing, restore outcomes, KV preservation ratio.
- **You may note** whether coder inference fails with the same error as BENCH_10 or a new one.
- **You may NOT** update `docs/decisions/settled.md`, production config, or queue priority.
- **You may NOT** conclude whether CRIU is production-safe for the coder endpoint — that
  decision requires research review of the SHM IPC fix status.
- **You may NOT** run tests beyond what is specified (no additional model configs, no TP=2 thinker).

## Stop condition

**Normal:** thinker cold checkpoint + restore working, KV sub-measurement complete, coder
checkpoint created (inference failure expected and documented), naming convention written,
production endpoints restored, `summary.md` written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` and stop if:
- Thinker checkpoint fails (exit non-zero with unfamiliar error — note if it differs from
  `anon_inode:[io_uring]` which would indicate UV_USE_IO_URING=0 was missing).
- Thinker restore exits 0 but `/health` never returns within 60s — new failure mode vs BENCH_09.
- Coder checkpoint fails with an error materially different from anything seen in BENCH_10
  (new error type, not a timeout or SHM variant).
