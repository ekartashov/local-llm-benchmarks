# BENCH_08 — T_CRIU2: CRIU Checkpoint/Restore for Convergence (mmap, --no-mmap removed)

**Status: DONE ✓**
**Blocks: nothing**
**Blocked by: nothing**

> [!NOTE]
> RESTORE_OK. Checkpoint 8.7 GB in 7.6 s. Restore 7.3 s. First-inference TTFT 100.56 s (page-fault warmup from NVMe), rep-2 36.13 s, rep-3 7.73 s. Text match: identical. **Key finding:** without QX_PRELOAD, CRIU mmap restore-to-interactive (108 s) is slower than cold start (83 s). QX_PRELOAD is a prerequisite for CRIU to benefit Convergence.

---

## Title
T_CRIU2 — CRIU checkpoint/restore of Convergence with mmap enabled (--no-mmap removed)

## Objective
With `--no-mmap` removed, model weights are memory-mapped from the GGUF file rather than loaded into anonymous RAM. Measure checkpoint size, restore time, first-inference latency (page-fault warmup), and inference correctness. Compare against BENCH_07 (--no-mmap) to quantify the trade-off.

## Why this exists
BENCH_07 confirmed CRIU works for ik_llama.cpp. With `--no-mmap`, the checkpoint archive is ~135 GB — 18–20 s restore from NVMe, which is better than 83 s cold start but still slow. When `--no-mmap` is removed, model weights are file-backed (mmap'd from the GGUF on disk). CRIU checkpoints file-backed pages without including them in the archive — the checkpoint shrinks to ~12 GB (GPU state + process metadata only), restoring in sub-second. The cost is that after restore, the first inference triggers page faults to reload weights from disk. This test measures whether that page-fault warmup time is acceptable.

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — current production config, gotchas, port assignments
2. `docs/procedures/criu-ops.md` — CRIU procedure, checkpoint/restore commands, ghost VRAM cleanup
3. `docs/arch/convergence.md` — Convergence deployment procedure and model path
4. Prior CRIU test results: `results/BENCH_07_*/summary.md` (--no-mmap findings to compare against)

## Prerequisites

```bash
# 1. BENCH_07 complete and RESTORE_OK
ls results/T_CRIU2_nommap_*/status.txt 2>/dev/null | xargs grep -l "RESTORE_OK" | head -1 \
  && echo "BENCH_07 COMPLETE" || echo "BENCH_07 NOT COMPLETE — run it first"

# 2. cuda-checkpoint plugin available
ls /usr/local/lib/criu/cuda-checkpoint.so 2>/dev/null \
  || ls /usr/lib/criu/cuda-checkpoint.so 2>/dev/null \
  && echo "PLUGIN OK" || echo "PLUGIN MISSING"

# 3. Convergence is running (production config, --no-mmap)
curl -sf http://localhost:8002/health && echo "CONVERGENCE OK" || echo "CONVERGENCE DOWN"

# 4. GGUF file path is known and accessible from the container
# The deploy.sh mounts model storage — verify the GGUF is on NVMe:
podman inspect $(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1) \
  | python3 -c "import sys,json; mounts=[m for m in json.load(sys.stdin)[0]['Mounts'] if 'ai' in m.get('Source','') or 'model' in m.get('Source','').lower()]; [print(m['Source'], '->', m['Destination']) for m in mounts]"

# 5. Free space for checkpoint (need ~15 GB, much less than BENCH_07)
df -h /var/lib/containers/storage 2>/dev/null || df -h /tmp
```

**Important:** This test deploys Convergence WITHOUT `--no-mmap`. First-inference after restore will be slower than steady-state while the kernel reloads weights from the GGUF file (page-fault warmup). This is expected and is the measurement.

**This test runs on the HOST.** `podman container checkpoint` is a host-side command.

## Inputs required
- BENCH_07 completed with RESTORE_OK
- Running Convergence container on port 8002
- cuda-checkpoint plugin installed
- ~15 GB free disk space for checkpoint archive
- `infra/scripts/deploy.sh`
- Known path to Qwen3.5-397B UD-IQ2_M GGUF file (mounted into the container)

## Fixed controls
| Control | Value |
|---------|-------|
| Model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M |
| Engine | ik_llama.cpp pr-1288 |
| --no-mmap | **OFF** (removed — this is the variable under test) |
| -ngl | production default (999 --cpu-moe) |
| -t (threads) | 32 |
| -np | 4 |
| -c (context) | 4096 |
| Test prompt | same as BENCH_07 |
| max_tokens | 50 |
| temperature | 0.0 |
| Repetitions (post-restore) | 3 (rep 1 = page-fault warmup, rep 2-3 = steady state) |

## Single variable under test
**--no-mmap flag** — removed (mmap enabled). The model GGUF is memory-mapped from disk rather than read into anonymous RAM.

Impact on CRIU:
- Checkpoint size: ~12 GB (GPU state + process metadata; file-backed pages not included)
- Restore time: sub-second (no large archive to read)
- First-inference cost: page faults reload weights from NVMe as accessed (~7.4 GB/s, 123 GB → ~17 s worst-case sequential, but access is sparse during decode)

## Procedure

```bash
CKPT_DIR="/path/to/fast/storage"   # EDIT: same volume used in BENCH_07
CKPT_FILE="${CKPT_DIR}/convergence_mmap.tar.gz"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/T_CRIU2_mmap_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# === Step 1: Redeploy Convergence WITHOUT --no-mmap ===
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
podman stop "${CONV_CONTAINER}" 2>/dev/null; podman rm "${CONV_CONTAINER}" 2>/dev/null

./infra/scripts/deploy.sh ikllamacpp convergence \
  -- -ngl 999 --cpu-moe -t 32 -np 4 -c 4096
# NOTE: --no-mmap is intentionally omitted

for i in $(seq 1 120); do
  curl -sf http://localhost:8002/health && echo "READY (mmap mode)" && break
  sleep 1
done

# === Step 2: Warm inference before checkpoint ===
PROMPT="List the three laws of thermodynamics in one sentence each."

PRE_RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"convergence\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
echo "PRE-CHECKPOINT RESPONSE:"
echo "${PRE_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])"
echo "${PRE_RESPONSE}" > "${RESULTS_DIR}/pre_checkpoint_response.json"

# Time a baseline inference (steady-state, weights warm in page cache)
START_MS=$(date +%s%3N)
curl -sf http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"convergence\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}" > /dev/null
END_MS=$(date +%s%3N)
BASELINE_TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
echo "Baseline TTFT (warm, pre-checkpoint): ${BASELINE_TTFT_S}s"
echo "baseline_ttft_s=${BASELINE_TTFT_S}" >> "${RESULTS_DIR}/timings.txt"

# === Step 3: Checkpoint ===
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
echo "Checkpointing (mmap mode): ${CONV_CONTAINER}"

CKPT_START_MS=$(date +%s%3N)
podman container checkpoint \
  --export="${CKPT_FILE}" \
  --tcp-established \
  "${CONV_CONTAINER}"
CKPT_EXIT=$?
CKPT_END_MS=$(date +%s%3N)
CKPT_ELAPSED_S=$(python3 -c "print(round(($CKPT_END_MS - $CKPT_START_MS) / 1000.0, 1))")
CKPT_SIZE_GB=$(du -sh "${CKPT_FILE}" 2>/dev/null | awk '{print $1}')

echo "Checkpoint exit code: ${CKPT_EXIT}"
echo "Checkpoint time: ${CKPT_ELAPSED_S}s"
echo "Checkpoint size: ${CKPT_SIZE_GB}"
echo "checkpoint_size_gb=${CKPT_SIZE_GB}" >> "${RESULTS_DIR}/timings.txt"
echo "checkpoint_elapsed_s=${CKPT_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

if [ "${CKPT_EXIT}" -ne 0 ]; then
  echo "CHECKPOINT_FAILED" > "${RESULTS_DIR}/status.txt"
  echo "Checkpoint failed. Restoring production Convergence (with --no-mmap)."
  ./infra/scripts/deploy.sh ikllamacpp convergence
  for i in $(seq 1 120); do curl -sf http://localhost:8002/health && echo "PRODUCTION RESTORED" && break; sleep 1; done
  exit 1
fi

# === Step 4: Drop page cache to simulate cold restore ===
# This ensures first post-restore inference actually measures page-fault cost,
# not residual cache from the pre-checkpoint run.
echo "Dropping page cache to simulate cold restore..."
sync && echo 3 > /proc/sys/vm/drop_caches
echo "Page cache dropped."

# === Step 5: Restore ===
echo "Restoring from checkpoint (mmap mode)..."
RESTORE_START_MS=$(date +%s%3N)
podman container restore \
  --import="${CKPT_FILE}" \
  --tcp-established
RESTORE_EXIT=$?
RESTORE_END_MS=$(date +%s%3N)
RESTORE_ELAPSED_S=$(python3 -c "print(round(($RESTORE_END_MS - $RESTORE_START_MS) / 1000.0, 2))")

echo "Restore exit code: ${RESTORE_EXIT}"
echo "Restore time: ${RESTORE_ELAPSED_S}s"
echo "restore_elapsed_s=${RESTORE_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

for i in $(seq 1 120); do
  curl -sf http://localhost:8002/health && echo "HEALTH OK after restore" && break
  sleep 1
done

# === Step 6: Post-restore inference — 3 reps (rep 1 = cold page fault warmup) ===
echo "config,rep,ttft_s,text_match" > "${RESULTS_DIR}/post_restore_reps.csv"
for REP in 1 2 3; do
  START_MS=$(date +%s%3N)
  RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"convergence\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
  END_MS=$(date +%s%3N)
  TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  POST_TEXT=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])")
  PRE_TEXT=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/pre_checkpoint_response.json')); print(d['choices'][0]['text'])")
  MATCH=$( [ "${PRE_TEXT}" = "${POST_TEXT}" ] && echo "identical" || echo "differs" )
  echo "mmap,${REP},${TTFT_S},${MATCH}" >> "${RESULTS_DIR}/post_restore_reps.csv"
  echo "Post-restore rep=${REP}: TTFT=${TTFT_S}s text=${MATCH}"
done

# === Step 7: VRAM after restore ===
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_post_restore.txt"
cat "${RESULTS_DIR}/vram_post_restore.txt"

# === Step 8: MANDATORY — restore production Convergence (WITH --no-mmap) ===
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
podman stop "${CONV_CONTAINER}" 2>/dev/null; podman rm "${CONV_CONTAINER}" 2>/dev/null
./infra/scripts/deploy.sh ikllamacpp convergence
for i in $(seq 1 120); do
  curl -sf http://localhost:8002/health && echo "PRODUCTION RESTORED (--no-mmap)" && break
  sleep 1
done

echo "RESTORE_OK" > "${RESULTS_DIR}/status.txt"
echo "Results in: ${RESULTS_DIR}"
```

> **Note on `drop_caches`:** Step 4 requires root (`echo 3 > /proc/sys/vm/drop_caches`). If the host does not permit this, skip the step and note `page_cache_not_dropped=true` in `timings.txt`. The rep-1 TTFT will then be shorter than true cold, but reps 2–3 still reflect steady-state latency.

## Metrics to record

| Metric | Source |
|--------|--------|
| Checkpoint exit code | shell |
| Checkpoint time (s) | `timings.txt` |
| Checkpoint archive size (GB) | `du -sh` |
| Restore exit code | shell |
| Restore time (s) | `timings.txt` |
| Baseline TTFT pre-checkpoint (s) | `timings.txt` |
| Post-restore TTFT rep 1 (page-fault cold) (s) | `post_restore_reps.csv` |
| Post-restore TTFT rep 2 (s) | `post_restore_reps.csv` |
| Post-restore TTFT rep 3 (s) | `post_restore_reps.csv` |
| GPU VRAM after restore (MiB per GPU) | `vram_post_restore.txt` |
| Inference text match rep 1 | `post_restore_reps.csv` |

Expected values:
- Checkpoint size: ~12 GB (file-backed pages excluded)
- Checkpoint time: ~2–3 s
- Restore time: ~1–2 s (sub-second process restore + brief health check)
- Rep 1 TTFT: elevated vs baseline (page faults; magnitude depends on access pattern)
- Rep 2–3 TTFT: close to baseline (weights warm in page cache again)

## Pass/fail checks

| Check | Condition | Action |
|-------|-----------|--------|
| Checkpoint exits 0 | Required | If non-zero: record error, redeploy production (--no-mmap), write `## Open from testing` |
| Checkpoint size < 20 GB | Expected — confirms file-backing is working | If size ≈ 135 GB: mmap may not have taken effect; note anomaly |
| Restore exits 0 | Required | If non-zero: record error, redeploy production (--no-mmap) |
| Health check passes after restore | `/health` returns 200 | Redeploy if not |
| Post-restore inference succeeds all 3 reps | Non-null text | If any null: record error |
| Production restored with --no-mmap at end | `/health` returns 200 + correct flags | Redeploy if not |

## Artifacts to write

1. `results/T_CRIU2_mmap_<timestamp>/pre_checkpoint_response.json`
2. `results/T_CRIU2_mmap_<timestamp>/post_restore_reps.csv`
3. `results/T_CRIU2_mmap_<timestamp>/vram_post_restore.txt`
4. `results/T_CRIU2_mmap_<timestamp>/timings.txt`
5. `results/T_CRIU2_mmap_<timestamp>/status.txt`
6. `results/T_CRIU2_mmap_<timestamp>/summary.md`:

```markdown
# T_CRIU2 CRIU Checkpoint/Restore — mmap (no --no-mmap) — <TIMESTAMP>

## Result
RESTORE_OK / CHECKPOINT_FAILED / RESTORE_FAILED

## Checkpoint
| Metric | Value |
|--------|-------|
| Archive size | <GB> |
| Checkpoint time | <s> |
| Restore time | <s> |

## Post-restore TTFT (seconds)
| Rep | TTFT (s) | Text match |
|-----|----------|------------|
| Baseline (pre-checkpoint, warm) | <s> | — |
| 1 (post-restore, page-fault cold) | <s> | identical / differs |
| 2 (post-restore, warm) | <s> | identical / differs |
| 3 (post-restore, warm) | <s> | identical / differs |

## GPU VRAM after restore
| GPU | VRAM used (MiB) |
|-----|----------------|
| 0   | <x> |
| 1   | <x> |

## Comparison vs BENCH_07 (--no-mmap)
| Metric | BENCH_07 (--no-mmap) | BENCH_08 (mmap) |
|--------|---------------------|-----------------|
| Checkpoint size | ~135 GB | <x> GB |
| Restore time | ~18 s | <x> s |
| Rep-1 TTFT post-restore | <x> s | <x> s |

## Incidental findings
<Any observation outside this benchmark's explicit scope: unexpected VRAM readings, other components
behaving differently than documented, engine warnings about kernels or flags, etc. Write one FINDING
block per observation. If nothing unusual observed: "none">

## Status
MEASURED / CHECKPOINT_FAILED / RESTORE_FAILED
```

**Do NOT write to any file outside `results/T_CRIU2_mmap_<timestamp>/`.**

## Interpretation boundary

- **You may record** checkpoint size, restore time, and all three post-restore TTFT values.
- **You may note** whether rep-1 TTFT is elevated relative to baseline (expected: yes) and by how much.
- **You may note** the size and restore-time difference versus BENCH_07.
- **You may NOT** recommend whether to switch production from `--no-mmap` to mmap.
- **You may NOT** update `docs/arch/convergence.md` or change the production deploy flags.
- **You may NOT** conclude whether the page-fault warmup cost is "acceptable."

## Stop condition

**Normal:** checkpoint succeeded, restore succeeded, 3 post-restore inferences recorded, production Convergence restored with `--no-mmap`, `summary.md` written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` if:
- Checkpoint exits non-zero (CRIU incompatible with mmap mode — record exact error)
- Checkpoint size ≈ 135 GB (same as --no-mmap — mmap did not take effect; investigate before proceeding)
- Restore exits non-zero (record exact error)
- Rep-1 TTFT > 120 s (page-fault reload time unacceptably long — record value and note)
- Production Convergence cannot be restored after 120 s
