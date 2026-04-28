# BENCH_07 — T_CRIU2: CRIU Checkpoint/Restore for Convergence (--no-mmap, current config)

**Status: DONE ✗**
**Blocks: BENCH_08**
**Blocked by: nothing**

> [!NOTE]
> CHECKPOINT_FAILED (SYSTEM_OOM). CRIU dump of a ~135 GB anonymous-RAM process on 188 GB RAM is physically impossible — VMS spiked to ~351 GB during parasite injection. Pre-checkpoint inference was healthy (8.74 t/s). Proceed to BENCH_08 (mmap variant).

---

## Title
T_CRIU2 — CRIU checkpoint and restore of ik_llama.cpp Convergence container (--no-mmap baseline)

## Objective
Verify that `podman container checkpoint` and `podman container restore` work correctly for the Convergence container running ik_llama.cpp with the current `--no-mmap` flag. Measure checkpoint size, restore time, and confirm inference correctness after restore.

## Why this exists
T_KV2 proved CRIU works for vLLM (0.28s hot restart). Convergence runs on ik_llama.cpp, which is a different process model. With `--no-mmap` (current production), all 123 GB of model weights are loaded into anonymous RAM plus ~12 GB of CUDA state — producing a checkpoint of ~135 GB. This test confirms CRIU is compatible with ik_llama.cpp at all before BENCH_08 explores the smaller-checkpoint mmap variant. If CRIU works here, the always-resident policy (83s cold start) becomes optional.

## Prerequisites

```bash
# 1. Confirm cuda-checkpoint plugin is available (installed during T_KV2)
ls /usr/local/lib/criu/cuda-checkpoint.so 2>/dev/null \
  || ls /usr/lib/criu/cuda-checkpoint.so 2>/dev/null \
  && echo "PLUGIN OK" || echo "PLUGIN MISSING — install before proceeding"

# 2. Convergence is running and healthy
curl -sf http://localhost:8002/health && echo "CONVERGENCE OK" || echo "CONVERGENCE DOWN — start it first"

# 3. Container name
podman ps --format "{{.Names}}\t{{.Status}}" | grep -i "convergence\|ikllamacpp"
# Note exact container name for checkpoint command

# 4. Free space on checkpoint volume (need ~150 GB)
df -h /var/lib/containers/storage 2>/dev/null || df -h /tmp
# Use whichever has space — you will pass the path to --export

# 5. GPU VRAM baseline
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

**This test runs on the HOST, not inside a container.** `podman container checkpoint` is a host-side command. Claude Code cannot run it — this handoff is for operator execution.

**The production Convergence container will be stopped during checkpoint.** It will be restored in the same session. If the restore fails, redeploy manually:
```bash
./infra/scripts/deploy.sh ikllamacpp convergence
```

## Inputs required
- Running Convergence container on port 8002
- cuda-checkpoint plugin installed (from T_KV2)
- ~150 GB free disk space for checkpoint archive
- `infra/scripts/deploy.sh`

## Fixed controls
| Control | Value |
|---------|-------|
| Model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M |
| Engine | ik_llama.cpp pr-1288 |
| --no-mmap | **ON** (current production default) |
| -ngl | production default (999 --cpu-moe) |
| -t (threads) | 32 |
| -np | 4 |
| -c (context) | 4096 |
| Test prompt | fixed (see procedure) |
| max_tokens | 50 |
| temperature | 0.0 |
| Checkpoint destination | operator-chosen path with ≥150 GB free |

## Single variable under test
**CRIU checkpoint/restore feasibility** — does `podman container checkpoint --with-previous` succeed for ik_llama.cpp with anonymous-RAM model weights (`--no-mmap`), and does inference produce correct output after restore?

The measurement is binary (works / does not work) plus timing and size.

## Procedure

```bash
CKPT_DIR="/path/to/fast/storage"   # EDIT: path with ≥150 GB free, prefer NVMe
CKPT_FILE="${CKPT_DIR}/convergence_nommap.tar.gz"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/T_CRIU2_nommap_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# --- Warm inference before checkpoint (confirm baseline works) ---
PROMPT="List the three laws of thermodynamics in one sentence each."
PRE_RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"convergence\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
echo "PRE-CHECKPOINT RESPONSE:"
echo "${PRE_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])"
echo "${PRE_RESPONSE}" > "${RESULTS_DIR}/pre_checkpoint_response.json"

# --- Checkpoint ---
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
echo "Checkpointing: ${CONV_CONTAINER}"

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

if [ "${CKPT_EXIT}" -ne 0 ]; then
  echo "CHECKPOINT_FAILED" > "${RESULTS_DIR}/status.txt"
  echo "Checkpoint failed. Recording error and restoring production Convergence."
  ./infra/scripts/deploy.sh ikllamacpp convergence
  for i in $(seq 1 120); do curl -sf http://localhost:8002/health && echo "PRODUCTION RESTORED" && break; sleep 1; done
  exit 1
fi

echo "checkpoint_size_gb=${CKPT_SIZE_GB}" >> "${RESULTS_DIR}/timings.txt"
echo "checkpoint_elapsed_s=${CKPT_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

# --- Restore ---
echo "Restoring from checkpoint..."
RESTORE_START_MS=$(date +%s%3N)
podman container restore \
  --import="${CKPT_FILE}" \
  --tcp-established
RESTORE_EXIT=$?
RESTORE_END_MS=$(date +%s%3N)
RESTORE_ELAPSED_S=$(python3 -c "print(round(($RESTORE_END_MS - $RESTORE_START_MS) / 1000.0, 1))")

echo "Restore exit code: ${RESTORE_EXIT}"
echo "Restore time: ${RESTORE_ELAPSED_S}s"
echo "restore_elapsed_s=${RESTORE_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

# Wait for health after restore
for i in $(seq 1 120); do
  curl -sf http://localhost:8002/health && echo "HEALTH OK after restore" && break
  sleep 1
done

# --- Post-restore inference ---
POST_RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"convergence\",\"prompt\":\"${PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
echo "POST-RESTORE RESPONSE:"
echo "${POST_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])"
echo "${POST_RESPONSE}" > "${RESULTS_DIR}/post_restore_response.json"

# --- GPU VRAM after restore ---
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_post_restore.txt"
cat "${RESULTS_DIR}/vram_post_restore.txt"

# --- Compare pre and post text (manual check) ---
PRE_TEXT=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/pre_checkpoint_response.json')); print(d['choices'][0]['text'])")
POST_TEXT=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/post_restore_response.json')); print(d['choices'][0]['text'])")
if [ "${PRE_TEXT}" = "${POST_TEXT}" ]; then
  echo "TEXT_MATCH: outputs identical (temperature=0.0 expected)"
  echo "text_match=identical" >> "${RESULTS_DIR}/timings.txt"
else
  echo "TEXT_MISMATCH: outputs differ — review manually"
  echo "text_match=differs" >> "${RESULTS_DIR}/timings.txt"
fi

# --- Record final status ---
echo "RESTORE_OK" > "${RESULTS_DIR}/status.txt"
echo "Results in: ${RESULTS_DIR}"

# --- MANDATORY: confirm production Convergence is running ---
# If restore succeeded, the container is already running at port 8002.
# If you want to verify it matches production config, redeploy:
#   ./infra/scripts/deploy.sh ikllamacpp convergence
# Otherwise confirm health is still up:
curl -sf http://localhost:8002/health && echo "PRODUCTION OK" || {
  echo "Health check failed after restore. Redeploying production."
  ./infra/scripts/deploy.sh ikllamacpp convergence
  for i in $(seq 1 120); do curl -sf http://localhost:8002/health && echo "PRODUCTION RESTORED" && break; sleep 1; done
}
```

## Metrics to record

| Metric | Source |
|--------|--------|
| Checkpoint exit code | shell |
| Checkpoint time (s) | `timings.txt` |
| Checkpoint archive size (GB) | `du -sh` |
| Restore exit code | shell |
| Restore time (s) | `timings.txt` |
| GPU VRAM after restore (MiB per GPU) | `vram_post_restore.txt` |
| Inference text match (identical / differs) | `timings.txt` |

Expected values:
- Checkpoint size: ~135 GB (123 GB model anon RAM + ~12 GB CUDA state)
- Checkpoint time: ~18–20 s (at ~7 GB/s write to NVMe)
- Restore time: ~18–20 s (at ~7.4 GB/s read from NVMe)
- Text match: identical (temperature=0.0)

## Pass/fail checks

| Check | Condition | Action |
|-------|-----------|--------|
| Checkpoint exits 0 | Required | If non-zero: record error, redeploy production, write `## Open from testing` |
| Restore exits 0 | Required | If non-zero: record error, redeploy production, write `## Open from testing` |
| Health check passes after restore | `/health` returns 200 | If fails: redeploy production |
| Post-restore inference succeeds | API returns non-null text | If null: record error |
| Text match at temperature=0.0 | Outputs identical | If differs: note — non-determinism possible even at temp=0; flag but do not fail the test |
| Production Convergence running at end | `/health` returns 200 | Redeploy if not |

## Artifacts to write

1. `results/T_CRIU2_nommap_<timestamp>/pre_checkpoint_response.json` — written by procedure
2. `results/T_CRIU2_nommap_<timestamp>/post_restore_response.json` — written by procedure
3. `results/T_CRIU2_nommap_<timestamp>/vram_post_restore.txt` — written by procedure
4. `results/T_CRIU2_nommap_<timestamp>/timings.txt` — written by procedure
5. `results/T_CRIU2_nommap_<timestamp>/status.txt` — RESTORE_OK or CHECKPOINT_FAILED
6. `results/T_CRIU2_nommap_<timestamp>/summary.md`:

```markdown
# T_CRIU2 CRIU Checkpoint/Restore — --no-mmap — <TIMESTAMP>

## Result
RESTORE_OK / CHECKPOINT_FAILED / RESTORE_FAILED

## Timings
| Phase | Value |
|-------|-------|
| Checkpoint time | <s> |
| Checkpoint size | <GB> |
| Restore time | <s> |

## GPU VRAM after restore
| GPU | VRAM used (MiB) |
|-----|----------------|
| 0   | <x> |
| 1   | <x> |

## Inference correctness
Text match (temperature=0.0): identical / differs

## Status
MEASURED / CHECKPOINT_FAILED / RESTORE_FAILED
```

**Do NOT write to any file outside `results/T_CRIU2_nommap_<timestamp>/`.**

## Interpretation boundary

- **You may record** checkpoint size, checkpoint time, restore time, and inference correctness.
- **You may note** whether the checkpoint size matches the expected ~135 GB.
- **You may NOT** change the production Convergence deploy flags.
- **You may NOT** conclude whether CRIU is "better" than the always-resident policy.
- **You may NOT** update `docs/arch/convergence.md`.

## Stop condition

**Normal:** checkpoint succeeded, restore succeeded, inference correct, production Convergence running, `summary.md` written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` if:
- Checkpoint exits non-zero (CRIU incompatible with ik_llama.cpp at this config — record exact error)
- Restore exits non-zero after successful checkpoint (record exact error)
- Post-restore inference returns null or HTTP error (CRIU restored process state is corrupted)
- Checkpoint size < 10 GB (suggests --no-mmap was not honoured and weights are file-backed — unexpected)
