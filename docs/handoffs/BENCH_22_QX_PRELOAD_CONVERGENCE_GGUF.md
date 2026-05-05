# BENCH_22 — QX_PRELOAD: Convergence GGUF page-cache pre-warm for fast CRIU restore

**Status:** READY
**Blocks:** Convergence CRIU operational policy (always-resident vs on-demand with CRIU)
**Blocked by:** nothing (T_CRIU2 DONE — mmap confirmed)

---

## Title
QX_PRELOAD for Convergence: measure CRIU restore-to-interactive TTFT with and without page-cache pre-warming of the 123GB GGUF model files.

## Objective
BENCH_08 (T_CRIU2) established: with mmap enabled, Convergence CRIU checkpoint = 8.7GB (file-backed pages excluded), restore = 7.3s, but first-inference TTFT = 100s due to 123GB of GGUF pages demand-paged from NVMe on access. Without page-cache pre-warming, CRIU mmap restore is **slower** than the 83s cold start. This benchmark tests whether `cat *.gguf > /dev/null` (which populates the page cache before restore) eliminates the 100s penalty and reduces restore-to-interactive time to ~17–20s, making CRIU viable for Convergence.

The two conditions measured:
- **Cold:** CRIU restore with page cache dropped (simulates first restore after boot or cache eviction). Expected TTFT: ~100s.
- **Warm:** CRIU restore with GGUF files pre-warmed into page cache via `cat`. Expected TTFT: ~17–20s (time to stream 123GB from NVMe at 7,400 MB/s read).

## Why this exists

**Background from T_CRIU2 (BENCH_08):** Convergence loads its 123GB GGUF files using mmap. When Linux maps a file, only the pages actually touched during execution are loaded from disk (demand paging). After a CRIU restore, the process resumes with the mmap addresses valid but no pages loaded (CRIU restores only dirty anonymous pages; file-backed mmap pages are NOT included in the checkpoint archive). The first forward pass triggers ~123GB of page faults against the GGUF files, blocking inference until each page loads from NVMe.

**QX_PRELOAD mechanism:** `posix_fadvise(POSIX_FADV_WILLNEED)` or equivalently `cat file > /dev/null` forces the kernel to read the file into page cache proactively. Once GGUF pages are in page cache, subsequent access via mmap incurs zero I/O latency — the page fault resolves from RAM, not NVMe. Pre-warming 123GB at 7,400 MB/s reads takes ~17s.

**Why this matters:** If warm restore TTFT is ~17–20s (vs 83s cold start), CRIU becomes viable for Convergence and the always-resident policy can be replaced by on-demand CRIU restore. This would free ~12GB of GPU VRAM (the attention/embed layers currently resident on GPU) whenever Convergence is idle, benefiting the Arclight co-load situation.

**Production note:** Current production runs Convergence with `--no-mmap`. This benchmark requires deploying WITHOUT `--no-mmap` (mmap mode). The `--no-mmap` flag is restored at the end of this test. Do NOT modify the production launch command in any doc.

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — current production config, gotchas, port assignments
2. `docs/arch/convergence.md` — production launch command (has `--no-mmap`; this test removes it temporarily), model path
3. `docs/procedures/criu-ops.md` — CRIU procedure for ik_llama.cpp (uses `podman container checkpoint`)
4. `results/BENCH_08_T_CRIU2_CONVERGENCE_MMAP.md` (or equivalent summary) — baseline: cold restore TTFT=100.56s, rep-2=36.1s, rep-3=7.7s; checkpoint=8.7GB in 7.6s

## Prerequisites

```bash
echo "=== BENCH_22 Prerequisites ===" && \

# 1. Convergence GGUF all 4 shards exist
GGUF_DIR=/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M
ls "${GGUF_DIR}"/Qwen3.5-397B-A17B-UD-IQ2_M-000*.gguf 2>/dev/null | wc -l | \
  { read N; [ "${N}" -eq 4 ] && echo "[prereq] GGUF 4 shards OK" \
    || { echo "[prereq] STOP: expected 4 GGUF shards, found ${N}"; exit 1; }; } && \

# 2. Free disk space for checkpoint (~10 GB needed)
AVAIL_GB=$(df /var/lib/containers/storage 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}')
[ "${AVAIL_GB:-0}" -ge 10 ] \
  && echo "[prereq] Disk space OK (${AVAIL_GB}GB free)" \
  || { echo "[prereq] STOP: need ≥10GB free, found ${AVAIL_GB}GB"; exit 1; } && \

# 3. CRIU available
which criu >/dev/null 2>&1 && criu check && echo "[prereq] CRIU OK" \
  || { echo "[prereq] STOP: CRIU not available or check failed"; exit 1; } && \

# 4. cuda-checkpoint plugin
ls /usr/local/lib/criu/cuda-checkpoint.so /usr/lib/criu/cuda-checkpoint.so 2>/dev/null | head -1 | \
  { read F; [ -n "${F}" ] && echo "[prereq] cuda-checkpoint plugin OK: ${F}" \
    || { echo "[prereq] STOP: cuda-checkpoint.so not found"; exit 1; }; } && \

# 5. Baseline VRAM (should be low if Convergence is stopped)
echo "[prereq] VRAM baseline:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader && \

# 6. Total GGUF size (for timing expectations)
du -sh "${GGUF_DIR}"/*.gguf | tail -1
echo "[prereq] GGUF total:"
du -sh "${GGUF_DIR}"
```

**Stop if any prereq fails.** The GGUF must be the exact same files referenced in the production launch command — do not substitute other quantizations.

## Inputs required

- Convergence GGUF files at exact path in convergence.md (~123GB total, 4 shards)
- CRIU installed and `criu check` passes
- `cuda-checkpoint.so` plugin installed
- ≥10GB free disk space for checkpoint archive
- `infra/scripts/deploy.sh` with ik_llama.cpp Convergence config
- Ability to run `echo 3 > /proc/sys/vm/drop_caches` (requires root — see note below)
- Arclight models stopped (freeing GPU0+1 VRAM for Convergence)

## Fixed controls

| Control | Value |
|---------|-------|
| Model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M |
| Engine | ik_llama.cpp main |
| --no-mmap | **OFF** (intentionally removed — this is the mmap test) |
| -ngl | 999 |
| --cpu-moe | ON |
| -t (threads) | 32 |
| -np | 4 |
| -c (context) | 4096 |
| Test prompt | `"List the three laws of thermodynamics in one sentence each."` |
| max_tokens | 50 |
| temperature | 0.0 |
| Repetitions per condition | 3 (rep 1 = first-inference; rep 2–3 = steady state) |
| GGUF pre-warm command | `cat "${GGUF_DIR}"/*.gguf > /dev/null` |
| Expected pre-warm time | ~17s (123GB at 7,400 MB/s) |
| Checkpoint storage path | `/var/lib/containers/storage/convergence_mmap_bench22.tar.gz` (or adjust to free-space location) |

## Single variable under test

Page cache state at restore time: **cold** (drop_caches after checkpoint) vs **warm** (pre-warm GGUF files before restore). All other factors held constant.

## Procedure

Skip flags (set to 1 to skip expensive steps on retry):
- `SKIP_DEPLOY=1` — skip Convergence startup (use if already running in mmap mode at port 8002)
- `SKIP_CHECKPOINT=1` — skip checkpoint step (use if checkpoint file already exists at CKPT_FILE)

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_22_qx_preload_convergence_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

GGUF_DIR=/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M
CKPT_FILE=/var/lib/containers/storage/convergence_mmap_bench22.tar.gz
TEST_PROMPT="List the three laws of thermodynamics in one sentence each."

# ===================================================================
# PHASE 1: Deploy Convergence WITHOUT --no-mmap (mmap mode)
# ===================================================================
SKIP_DEPLOY=${SKIP_DEPLOY:-0}
if [ "${SKIP_DEPLOY}" = "0" ]; then
  # Stop any existing Convergence instance
  CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp\|8002" | head -1)
  if [ -n "${CONV_CONTAINER}" ]; then
    echo "Stopping existing Convergence container: ${CONV_CONTAINER}"
    podman stop "${CONV_CONTAINER}" && podman rm "${CONV_CONTAINER}"
    sleep 3
  fi

  echo "=== Starting Convergence in mmap mode (--no-mmap omitted) ==="
  ./infra/scripts/deploy.sh ikllamacpp convergence \
    -- -ngl 999 --cpu-moe -t 32 -np 4 -c 4096 \
    >> "${RESULTS_DIR}/convergence_deploy.log" 2>&1
  # CRITICAL: the above command MUST NOT include --no-mmap.
  # Verify by checking convergence_deploy.log for the actual command used.

  CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
  podman logs -f "${CONV_CONTAINER}" 2>&1 | stdbuf -oL sed 's/\r//g; s/^/[convergence] /' &
  LOG_PID=$!

  for i in $(seq 1 120); do
    curl -sf http://localhost:8002/health 2>/dev/null && echo "[convergence] HEALTH OK (mmap mode)" && break
    sleep 1
  done
  kill "${LOG_PID}" 2>/dev/null

  curl -sf http://localhost:8002/health \
    || { echo "FATAL: Convergence did not start in mmap mode"; exit 1; }

  # Verify --no-mmap is NOT present in the running command
  podman inspect "${CONV_CONTAINER}" \
    | python3 -c "import sys,json; cfg=json.load(sys.stdin)[0]; cmd=' '.join(cfg.get('Config',{}).get('Cmd',[])); print('CMD:', cmd); print('no-mmap present:', '--no-mmap' in cmd)"
else
  echo "[skip] SKIP_DEPLOY=1"
  curl -sf http://localhost:8002/health || { echo "FATAL: Convergence not live"; exit 1; }
fi

# Run 2 warm-up inferences to fully populate page cache (all GGUF pages touched)
echo "=== Warm-up inferences (populating page cache) ==="
for W in 1 2; do
  curl -sf http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"convergence\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}" \
    > "${RESULTS_DIR}/warmup_${W}.json"
  echo "Warmup ${W} complete"
done

# Pre-checkpoint reference inference
PRECHECK_RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"convergence\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
echo "${PRECHECK_RESPONSE}" > "${RESULTS_DIR}/pre_checkpoint_response.json"
PRE_TEXT=$(echo "${PRECHECK_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])")
echo "Pre-checkpoint text: ${PRE_TEXT}"

# ===================================================================
# PHASE 2: CRIU checkpoint
# ===================================================================
SKIP_CHECKPOINT=${SKIP_CHECKPOINT:-0}
if [ "${SKIP_CHECKPOINT}" = "0" ]; then
  CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
  echo "=== Checkpointing: ${CONV_CONTAINER} ==="
  CKPT_START_MS=$(date +%s%3N)

  podman container checkpoint \
    --export="${CKPT_FILE}" \
    --tcp-established \
    "${CONV_CONTAINER}"
  CKPT_EXIT=$?
  CKPT_END_MS=$(date +%s%3N)
  CKPT_ELAPSED_S=$(python3 -c "print(round(($CKPT_END_MS - $CKPT_START_MS) / 1000.0, 1))")
  CKPT_SIZE_GB=$(du -sh "${CKPT_FILE}" 2>/dev/null | awk '{print $1}')

  echo "Checkpoint exit: ${CKPT_EXIT}, time: ${CKPT_ELAPSED_S}s, size: ${CKPT_SIZE_GB}"
  echo "checkpoint_exit=${CKPT_EXIT}" > "${RESULTS_DIR}/timings.txt"
  echo "checkpoint_elapsed_s=${CKPT_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"
  echo "checkpoint_size=${CKPT_SIZE_GB}" >> "${RESULTS_DIR}/timings.txt"

  if [ "${CKPT_EXIT}" -ne 0 ]; then
    echo "CHECKPOINT_FAILED" > "${RESULTS_DIR}/status.txt"
    echo "Checkpoint failed — check convergence_deploy.log. Was --no-mmap accidentally included?"
    # Restore production Convergence before exiting
    ./infra/scripts/deploy.sh ikllamacpp convergence
    for i in $(seq 1 120); do curl -sf http://localhost:8002/health 2>/dev/null && echo "PRODUCTION RESTORED" && break; sleep 1; done
    exit 1
  fi
  echo "Checkpoint size ${CKPT_SIZE_GB} — expected ~8-9GB for mmap mode (NOT ~135GB which would indicate --no-mmap was active)"
else
  echo "[skip] SKIP_CHECKPOINT=1 — using existing checkpoint at ${CKPT_FILE}"
  ls -lh "${CKPT_FILE}" || { echo "FATAL: checkpoint file not found at ${CKPT_FILE}"; exit 1; }
fi

# ===================================================================
# PHASE 3: TEST A — Cold restore (drop page cache first)
# ===================================================================
echo "=== TEST A: Cold restore (drop page cache) ==="

# NOTE: drop_caches requires root. If host does not permit:
# skip the sync+drop_caches lines and note page_cache_not_dropped=true in timings.txt
sync
echo 3 > /proc/sys/vm/drop_caches \
  && echo "Page cache dropped (cold restore test)" \
  || { echo "WARNING: drop_caches failed (may lack root). Recording page_cache_not_dropped=true"; \
       echo "page_cache_not_dropped=true" >> "${RESULTS_DIR}/timings.txt"; }

RESTORE_A_START_MS=$(date +%s%3N)
podman container restore --import="${CKPT_FILE}" --tcp-established
RESTORE_A_EXIT=$?
RESTORE_A_END_MS=$(date +%s%3N)
RESTORE_A_ELAPSED_S=$(python3 -c "print(round(($RESTORE_A_END_MS - $RESTORE_A_START_MS) / 1000.0, 2))")
echo "restore_A_elapsed_s=${RESTORE_A_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"
echo "restore_A_exit=${RESTORE_A_EXIT}" >> "${RESULTS_DIR}/timings.txt"

for i in $(seq 1 30); do
  curl -sf http://localhost:8002/health 2>/dev/null && echo "[restore A] HEALTH OK" && break
  sleep 1
done

echo "rep,condition,ttft_s,text_match" > "${RESULTS_DIR}/restore_reps.csv"
for REP in 1 2 3; do
  START_MS=$(date +%s%3N)
  RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"convergence\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
  END_MS=$(date +%s%3N)
  TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  POST_TEXT=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null)
  MATCH=$( [ "${PRE_TEXT}" = "${POST_TEXT}" ] && echo "identical" || echo "differs" )
  echo "A_cold,${REP},${TTFT_S},${MATCH}" >> "${RESULTS_DIR}/restore_reps.csv"
  echo "[TEST A rep ${REP}] TTFT=${TTFT_S}s text=${MATCH}"
done

# Stop container before next test
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
podman stop "${CONV_CONTAINER}" 2>/dev/null; podman rm "${CONV_CONTAINER}" 2>/dev/null
sleep 3

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "${RESULTS_DIR}/timings.txt"

# ===================================================================
# PHASE 4: TEST B — Warm restore (pre-warm GGUF files before restore)
# ===================================================================
echo "=== TEST B: Warm restore (pre-warm GGUF files) ==="

# Drop page cache again (ensure cold starting point for fair comparison)
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && echo "Page cache dropped (pre-warm test)" \
  || echo "WARNING: drop_caches failed — warm vs cold comparison may be inaccurate"

# Pre-warm: cat all 4 GGUF shards into /dev/null (populates page cache)
PREWARM_START_MS=$(date +%s%3N)
echo "Pre-warming ${GGUF_DIR}/*.gguf into page cache..."
cat "${GGUF_DIR}"/Qwen3.5-397B-A17B-UD-IQ2_M-*.gguf > /dev/null
PREWARM_END_MS=$(date +%s%3N)
PREWARM_ELAPSED_S=$(python3 -c "print(round(($PREWARM_END_MS - $PREWARM_START_MS) / 1000.0, 1))")
echo "Pre-warm complete: ${PREWARM_ELAPSED_S}s (expected ~17s for 123GB at 7,400 MB/s)"
echo "prewarm_elapsed_s=${PREWARM_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"

# Restore immediately after pre-warm (while pages are in cache)
RESTORE_B_START_MS=$(date +%s%3N)
podman container restore --import="${CKPT_FILE}" --tcp-established
RESTORE_B_EXIT=$?
RESTORE_B_END_MS=$(date +%s%3N)
RESTORE_B_ELAPSED_S=$(python3 -c "print(round(($RESTORE_B_END_MS - $RESTORE_B_START_MS) / 1000.0, 2))")
echo "restore_B_elapsed_s=${RESTORE_B_ELAPSED_S}" >> "${RESULTS_DIR}/timings.txt"
echo "restore_B_exit=${RESTORE_B_EXIT}" >> "${RESULTS_DIR}/timings.txt"

for i in $(seq 1 30); do
  curl -sf http://localhost:8002/health 2>/dev/null && echo "[restore B] HEALTH OK" && break
  sleep 1
done

for REP in 1 2 3; do
  START_MS=$(date +%s%3N)
  RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"convergence\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":50,\"temperature\":0.0}")
  END_MS=$(date +%s%3N)
  TTFT_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  POST_TEXT=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null)
  MATCH=$( [ "${PRE_TEXT}" = "${POST_TEXT}" ] && echo "identical" || echo "differs" )
  echo "B_warm,${REP},${TTFT_S},${MATCH}" >> "${RESULTS_DIR}/restore_reps.csv"
  echo "[TEST B rep ${REP}] TTFT=${TTFT_S}s text=${MATCH}"
done

cat "${RESULTS_DIR}/restore_reps.csv"

# ===================================================================
# MANDATORY: Restore production Convergence (WITH --no-mmap)
# ===================================================================
echo "=== Restoring production Convergence (WITH --no-mmap) ==="
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
podman stop "${CONV_CONTAINER}" 2>/dev/null; podman rm "${CONV_CONTAINER}" 2>/dev/null
./infra/scripts/deploy.sh ikllamacpp convergence
for i in $(seq 1 120); do
  curl -sf http://localhost:8002/health && echo "PRODUCTION RESTORED (--no-mmap)" && break; sleep 1
done
curl -sf http://localhost:8002/health || echo "WARNING: Production Convergence did not restart — start manually"

echo "COMPLETE" > "${RESULTS_DIR}/status.txt"
echo "=== BENCH_22 complete — results in ${RESULTS_DIR} ==="
```

**Note on `drop_caches`:** `echo 3 > /proc/sys/vm/drop_caches` requires root. If the host user cannot run this:
- Skip the drop_caches lines
- Note `page_cache_not_dropped=true` in timings.txt
- Test A rep-1 TTFT will be shorter than true cold (pages may still be in cache from deploy)
- This makes the cold vs warm comparison invalid — note in summary.md

## Metrics to record

| Metric | Source file | Expected / reference value |
|--------|-------------|---------------------------|
| Checkpoint size (GB) | `timings.txt` | ~8–9 GB (mmap mode, file-backed pages excluded) |
| Checkpoint elapsed (s) | `timings.txt` | ~7–8s (reference: BENCH_08 = 7.6s) |
| Restore time TEST A (s) | `timings.txt` | ~7s (restore itself; page faults happen during inference) |
| Restore time TEST B (s) | `timings.txt` | ~7s (same restore; pages already warm) |
| Pre-warm elapsed (s) | `timings.txt` | ~17s (123GB at 7,400 MB/s) |
| TEST A rep 1 TTFT (s) | `restore_reps.csv` | ~100s (BENCH_08 baseline; page-fault cold) |
| TEST A rep 2 TTFT (s) | `restore_reps.csv` | ~36s (BENCH_08 reference) |
| TEST A rep 3 TTFT (s) | `restore_reps.csv` | ~7.7s (BENCH_08 reference) |
| TEST B rep 1 TTFT (s) | `restore_reps.csv` | ~14–20s (pre-warmed page cache — the key metric) |
| TEST B rep 2 TTFT (s) | `restore_reps.csv` | ~7s (steady state) |
| TEST B rep 3 TTFT (s) | `restore_reps.csv` | ~7s (steady state) |
| Text match all reps | `restore_reps.csv` | all identical |
| Pre-warm speedup (TEST A rep-1 / TEST B rep-1) | computed | Target: ≥5× |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Checkpoint exits 0 | Required | Stop test; restore production; write Open from testing |
| Checkpoint size < 20GB | Confirms mmap mode (file-backed pages excluded) | If ~135GB: `--no-mmap` was NOT removed; stop, fix deploy, rerun |
| Restore exits 0 (both A and B) | Required | Record error; restore production |
| Health check passes after restore | `/health` returns 200 | Stop; restore production |
| Text match all reps | All `identical` | If any `differs`: note in summary.md as correctness issue |
| Pre-warm speedup ≥ 3× | TEST B rep-1 / TEST A rep-1 ≥ 3× | Record actual ratio; this is the key result even if < 3× |
| Production restored with `--no-mmap` at end | `/health` 200 + `--no-mmap` in running cmd | Manual restart if needed |

## Artifacts to write

1. `results/BENCH_22_qx_preload_convergence_<TIMESTAMP>/timings.txt` — checkpoint/restore/prewarm times
2. `results/BENCH_22_qx_preload_convergence_<TIMESTAMP>/restore_reps.csv` — per-rep TTFT and text match
3. `results/BENCH_22_qx_preload_convergence_<TIMESTAMP>/pre_checkpoint_response.json` — reference text
4. `results/BENCH_22_qx_preload_convergence_<TIMESTAMP>/status.txt` — COMPLETE / CHECKPOINT_FAILED / RESTORE_FAILED
5. `results/BENCH_22_qx_preload_convergence_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_22 — QX_PRELOAD Convergence GGUF — <TIMESTAMP>

## Environment
- Engine: ik_llama.cpp main (commit: run `git -C /srv/ai/projects/ik_llama.cpp rev-parse --short HEAD`)
- Model: unsloth/Qwen3.5-397B-A17B UD-IQ2_M (4 GGUF shards, ~123GB)
- Mode: mmap (--no-mmap omitted)
- Config: -ngl 999 --cpu-moe -t 32 -np 4 -c 4096

## Checkpoint
| Metric | Value |
|--------|-------|
| Archive size | <GB> (expected: ~8-9GB for mmap mode) |
| Checkpoint elapsed | <s> |

## TEST A — Cold restore (page cache dropped)
| Rep | TTFT (s) | Text match |
|-----|----------|------------|
| 1 (cold) | <s> | identical / differs |
| 2 | <s> | identical / differs |
| 3 | <s> | identical / differs |

## TEST B — Warm restore (GGUF pre-warmed via cat)
| Rep | TTFT (s) | Text match |
|-----|----------|------------|
| 1 (warm) | <s> | identical / differs |
| 2 | <s> | identical / differs |
| 3 | <s> | identical / differs |

## Pre-warm
| Metric | Value |
|--------|-------|
| GGUF pre-warm elapsed | <s> (123GB at NVMe speed) |
| Speedup rep-1 (TEST A / TEST B) | <x>× |

## Comparison
| Metric | TEST A (cold) rep-1 | TEST B (warm) rep-1 | Isolated baseline (BENCH_08 cold) |
|--------|--------------------|--------------------|----------------------------------|
| Restore TTFT | <s> | <s> | 100.56s |
| Speedup vs cold start (83s) | <x>× | <x>× | 0.83× (slower than cold) |

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| Checkpoint size < 20GB (mmap confirmed) | PASS/FAIL | Actual: <x>GB |
| Text match all reps | PASS/FAIL | |
| Pre-warm speedup ≥ 3× | PASS/FAIL | Actual: <x>× |
| Production restored with --no-mmap | PASS/FAIL | |

## Verdict
PASS / FAIL / PARTIAL — <one sentence summarizing whether QX_PRELOAD makes CRIU viable for Convergence>

## Incidental findings
<Any unexpected GGUF loading behavior, page cache observations, GPU VRAM anomalies. One FINDING block per observation.>
<If nothing unusual: "none">

## Open from testing
<Any unexpected result — especially if TEST B rep-1 > 30s (pre-warm not working as expected).>
<If nothing: "none">
```

## Interpretation boundary

**You may:**
- Record all TTFT values for both conditions
- Note the speedup ratio (TEST A rep-1 / TEST B rep-1)
- Note whether the text output is identical across all conditions
- Note the pre-warm elapsed time and whether it matches the 7,400 MB/s NVMe estimate

**You may NOT:**
- Change the production Convergence launch command (keep `--no-mmap`)
- Update `docs/arch/convergence.md` with a recommendation to switch to mmap mode
- Conclude whether the always-resident policy should change — that is research-mode
- Speculate on whether CRIU + QX_PRELOAD will be adopted as the production pattern

## Stop condition

**Normal:** Both TEST A and TEST B complete (3 reps each), pre-warm time recorded, production Convergence restored with `--no-mmap`, `summary.md` written.

**Abnormal:** Write `## Open from testing` in `RESEARCH_STATE.md` if:
- Checkpoint fails: `BENCH_22_CKPT_FAIL: podman container checkpoint exited non-zero in mmap mode. Error: [excerpt]. Note: verify --no-mmap was NOT in the deploy command.`
- Checkpoint size ~135GB: `BENCH_22_MMAP_NOT_ACTIVE: checkpoint size ~135GB (expected ~9GB) — --no-mmap may have been included despite instructions. Check deploy.sh ikllamacpp convergence behavior.`
- Restore fails: `BENCH_22_RESTORE_FAIL: restore exited non-zero after [cold/warm] test. Error: [excerpt].`
- TEST B rep-1 > 60s (pre-warm not working): `BENCH_22_PREWARM_INEFFECTIVE: TEST B rep-1 TTFT=[x]s with pre-warm, TEST A rep-1=[y]s without. Pre-warm time was [z]s. Page cache may not be retained across CRIU restore, or GGUF pages may be evicted before restore completes.`
