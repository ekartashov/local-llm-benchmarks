# BENCH_18 — QX_PRELOAD: NVMe Checkpoint Pre-warm via posix_fadvise

**Status: READY (after BENCH_17)**
**Blocks: nothing**
**Blocked by: BENCH_17 (T_CRIU3 — checkpoint files must exist before pre-warm can be measured)**

---

## Title
QX_PRELOAD — Measure CRIU restore time improvement from OS page-cache pre-warming via `posix_fadvise`

## Objective
Implement the `infra/scripts/preload-checkpoint.sh` pre-warm script using `posix_fadvise(POSIX_FADV_WILLNEED)`. Measure CRIU restore time for the thinker checkpoint under two conditions: cold (page cache dropped, checkpoint read from NVMe) and warm (pre-loaded into page cache before restore). Record the speedup. Derive the projected warm restore time for the Convergence checkpoint by extrapolation.

## Why this exists
T_CRIU2 (BENCH_07/08) showed Convergence mmap checkpoint restore = 7s, but first-inference = 100s without page-cache pre-warm — worse than the 83s cold start. The bottleneck is the OS loading model weights from NVMe after restore. The NM790 reads at 7,400 MB/s. A 123 GB Convergence checkpoint takes ~17s to load from NVMe; the same data pre-loaded into page cache = ~0.28s restore (T_KV2 baseline for a warm-cache restore).

Pre-warming the checkpoint file into page cache before the CRIU restore call is the only mechanism that converts a 100s first-inference into a sub-second one. Without QX_PRELOAD, CRIU is not viable for Convergence.

This handoff measures the mechanism on the small thinker checkpoint (~501 MiB) to confirm it works before applying it to the 123 GB Convergence case.

---

## Prerequisites

```bash
# 1. BENCH_17 completed — thinker checkpoint exists
THINKER_CKPT="/srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k.ckpt"
ls -lh "${THINKER_CKPT}" && echo "CHECKPOINT OK" || echo "STOP — run BENCH_17 first"

# 2. sudo access for page-cache drop (requires root; check before starting)
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' && echo "drop_caches OK" \
  || echo "STOP — sudo required for cold measurement; run as root or with sudo configured"

# 3. Production thinker NOT running (this test stops/restores it)
curl -sf http://localhost:30001/health && echo "THINKER RUNNING — will be stopped during test" \
  || echo "THINKER DOWN — ensure it will be available for restore step"

# 4. ik_llama.cpp / Convergence NOT writing aggressively to disk (avoid NVMe contention during timing)
# Convergence is always-resident but not doing IO during this test normally.
```

**This test runs on the HOST.**

## Inputs required
- Thinker checkpoint at `/srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k.ckpt` (created in BENCH_17)
- `sudo` access to run `echo 3 > /proc/sys/vm/drop_caches`
- `infra/scripts/deploy.sh` for production thinker restore at end of test
- Python 3 with `os.posix_fadvise` available (standard library, Python ≥ 3.3)
- Production thinker container stopped or checkpointed before cold restore step

## Fixed controls

| Control | Value |
|---------|-------|
| Checkpoint under test | `/srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k.ckpt` (~501 MiB) |
| NVMe device | Lexar NM790 (7,400 MB/s read) |
| Pre-warm method | `posix_fadvise(fd, 0, size, POSIX_FADV_WILLNEED)` via Python `os` module |
| Cache drop method | `echo 3 > /proc/sys/vm/drop_caches` (drops page cache + dentries + inodes) |
| Restore command | `podman container restore --import=<checkpoint> --tcp-established` |
| Timing method | `date +%s%3N` before and after restore call (millisecond precision) |
| Convergence checkpoint size (reference) | ~123 GB (from T_CRIU2; extrapolation target) |
| Convergence checkpoint path | from BENCH_17 Convergence checkpoint if created, else estimate only |

## Single variable under test
**OS page-cache state of the checkpoint file** — warm (pre-loaded via `posix_fadvise WILLNEED` before restore) vs cold (page cache dropped via `drop_caches` immediately before restore). All other factors identical.

---

## Procedure

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_18_qxpreload_nvme_preload_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

THINKER_CKPT="/srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k.ckpt"
CHECKPOINT_SIZE_MIB=$(du -m "${THINKER_CKPT}" | awk '{print $1}')
echo "checkpoint_size_mib=${CHECKPOINT_SIZE_MIB}" > "${RESULTS_DIR}/timings.txt"
echo "Checkpoint size: ${CHECKPOINT_SIZE_MIB} MiB"
```

### Step 1 — Implement preload-checkpoint.sh

```bash
cat > infra/scripts/preload-checkpoint.sh << 'SCRIPT'
#!/usr/bin/env bash
# Usage: preload-checkpoint.sh <checkpoint_path>
# Issues posix_fadvise(POSIX_FADV_WILLNEED) to pre-warm a checkpoint file into OS page cache.
# Returns immediately (async) — the kernel loads pages in the background.
set -euo pipefail

CHECKPOINT="${1:-}"
if [ -z "${CHECKPOINT}" ]; then
  echo "Usage: $0 <checkpoint_path>" >&2
  exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: File not found: ${CHECKPOINT}" >&2
  exit 1
fi

SIZE_GB=$(du -sh "${CHECKPOINT}" 2>/dev/null | cut -f1)
START_MS=$(date +%s%3N)

python3 - "${CHECKPOINT}" << 'PYEOF'
import os, sys
path = sys.argv[1]
fd = os.open(path, os.O_RDONLY)
size = os.fstat(fd).st_size
os.posix_fadvise(fd, 0, size, os.POSIX_FADV_WILLNEED)
os.close(fd)
print(f"[preload] WILLNEED issued: {path} ({size/1e9:.2f} GB)")
PYEOF

END_MS=$(date +%s%3N)
echo "[preload] Completed in $(( END_MS - START_MS )) ms (kernel loading pages async)"
SCRIPT

chmod +x infra/scripts/preload-checkpoint.sh
echo "infra/scripts/preload-checkpoint.sh created"
```

Verify it runs without error:

```bash
infra/scripts/preload-checkpoint.sh "${THINKER_CKPT}"
echo "preload_script_ok=1" >> "${RESULTS_DIR}/timings.txt"
```

### Step 2 — Stop production thinker (prepare for restore tests)

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
if [ -n "${THINKER_CONTAINER}" ]; then
  podman stop "${THINKER_CONTAINER}" && podman rm "${THINKER_CONTAINER}"
  echo "Thinker container stopped"
else
  echo "Thinker not running — proceeding"
fi
sleep 3
```

### Step 3 — Cold restore measurement (page cache dropped)

```bash
# Drop OS page cache — forces checkpoint to be read from NVMe
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
echo "Page cache dropped"

# Verify checkpoint is NOT in page cache
python3 - "${THINKER_CKPT}" << 'EOF'
import subprocess, sys
path = sys.argv[1]
# fincore (from util-linux) or /proc/self/pagemap approximation
result = subprocess.run(["fincore", path], capture_output=True, text=True)
if result.returncode == 0:
    print("fincore output:", result.stdout.strip())
else:
    print("fincore not available — proceeding with restore (cache state unverified)")
EOF

COLD_START_MS=$(date +%s%3N)
podman container restore \
  --import="${THINKER_CKPT}" \
  --tcp-established
COLD_RESTORE_EXIT=$?
COLD_END_MS=$(date +%s%3N)
COLD_RESTORE_MS=$(( COLD_END_MS - COLD_START_MS ))
COLD_RESTORE_S=$(python3 -c "print(round(${COLD_RESTORE_MS} / 1000.0, 3))")

echo "Cold restore: exit=${COLD_RESTORE_EXIT} time=${COLD_RESTORE_MS} ms (${COLD_RESTORE_S}s)"
echo "cold_restore_exit=${COLD_RESTORE_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "cold_restore_ms=${COLD_RESTORE_MS}" >> "${RESULTS_DIR}/timings.txt"

# Confirm health then stop (clean up before warm measurement)
for i in $(seq 1 60); do
  curl -sf http://localhost:30001/health 2>/dev/null && echo "THINKER HEALTH OK (cold)" && break; sleep 1
done

THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
[ -n "${THINKER_CONTAINER}" ] && podman stop "${THINKER_CONTAINER}" && podman rm "${THINKER_CONTAINER}"
sleep 3
```

### Step 4 — Warm restore measurement (pre-loaded via posix_fadvise)

```bash
# Trigger posix_fadvise — kernel loads pages into page cache asynchronously
infra/scripts/preload-checkpoint.sh "${THINKER_CKPT}"
PRELOAD_ISSUED_MS=$(date +%s%3N)
echo "preload_issued_ms=${PRELOAD_ISSUED_MS}" >> "${RESULTS_DIR}/timings.txt"

# Wait for pages to be loaded. Estimate: 501 MiB at 7,400 MB/s ≈ 0.07s.
# Wait 2s for safety; the kernel should have loaded all pages by then.
sleep 2

# Confirm pages are in cache
python3 - "${THINKER_CKPT}" << 'EOF'
import subprocess, sys
path = sys.argv[1]
result = subprocess.run(["fincore", path], capture_output=True, text=True)
if result.returncode == 0:
    print("fincore output (after preload):", result.stdout.strip())
else:
    print("fincore not available — proceeding")
EOF

WARM_START_MS=$(date +%s%3N)
podman container restore \
  --import="${THINKER_CKPT}" \
  --tcp-established
WARM_RESTORE_EXIT=$?
WARM_END_MS=$(date +%s%3N)
WARM_RESTORE_MS=$(( WARM_END_MS - WARM_START_MS ))
WARM_RESTORE_S=$(python3 -c "print(round(${WARM_RESTORE_MS} / 1000.0, 3))")

echo "Warm restore: exit=${WARM_RESTORE_EXIT} time=${WARM_RESTORE_MS} ms (${WARM_RESTORE_S}s)"
echo "warm_restore_exit=${WARM_RESTORE_EXIT}" >> "${RESULTS_DIR}/timings.txt"
echo "warm_restore_ms=${WARM_RESTORE_MS}" >> "${RESULTS_DIR}/timings.txt"

# Confirm health
for i in $(seq 1 60); do
  curl -sf http://localhost:30001/health 2>/dev/null && echo "THINKER HEALTH OK (warm)" && break; sleep 1
done
```

### Step 5 — Compute speedup and project Convergence warm restore time

```bash
python3 - << EOF
cold_ms = ${COLD_RESTORE_MS}
warm_ms = ${WARM_RESTORE_MS}
speedup = round(cold_ms / warm_ms, 1) if warm_ms > 0 else "inf"
nvme_speed_gbs = 7.4  # GB/s

# Thinker checkpoint size
thinker_size_gib = ${CHECKPOINT_SIZE_MIB} / 1024
cold_theoretical_s = thinker_size_gib / nvme_speed_gbs

# Convergence checkpoint (~123 GB from T_CRIU2)
convergence_size_gib = 123
convergence_cold_theoretical_s = convergence_size_gib / nvme_speed_gbs

# If warm restore time is consistent with T_KV2 baseline (~0.28s), apply that
# to estimate Convergence warm restore. Otherwise use measured ratio.
t_kv2_warm_baseline_s = 0.28
if warm_ms < 500:
    convergence_warm_projected_s = t_kv2_warm_baseline_s  # consistent with T_KV2
    projection_basis = "T_KV2 warm-cache baseline (0.28s)"
else:
    ratio = warm_ms / cold_ms
    convergence_warm_projected_s = round(convergence_cold_theoretical_s * ratio, 1)
    projection_basis = f"measured ratio ({ratio:.2f})"

print("=== QX_PRELOAD Results ===")
print(f"Thinker checkpoint size:   {thinker_size_gib:.2f} GiB")
print(f"Cold restore (NVMe):       {cold_ms} ms  ({cold_ms/1000:.3f}s)")
print(f"Warm restore (page cache): {warm_ms} ms  ({warm_ms/1000:.3f}s)")
print(f"Speedup:                   {speedup}x")
print(f"")
print(f"Convergence checkpoint:    ~{convergence_size_gib} GB")
print(f"Convergence cold restore:  ~{convergence_cold_theoretical_s:.0f}s (theoretical at {nvme_speed_gbs} GB/s)")
print(f"Convergence warm restore:  ~{convergence_warm_projected_s:.2f}s (projected, basis: {projection_basis})")
print(f"Convergence cold start:    ~83s (T_CV1 baseline)")
print(f"CRIU viable for Convergence (warm < cold start): {'YES' if convergence_warm_projected_s < 30 else 'MARGINAL/NO'}")
EOF
```

Record the output to `${RESULTS_DIR}/summary_computed.txt`.

### Step 6 — Restore production thinker (MANDATORY)

```bash
THINKER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "thinker" | head -1)
[ -n "${THINKER_CONTAINER}" ] && podman stop "${THINKER_CONTAINER}" && podman rm "${THINKER_CONTAINER}"

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
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

| Metric | Source | Reference |
|--------|--------|-----------|
| Checkpoint size (MiB) | `timings.txt` | ~501 MiB (BENCH_17) |
| Cold restore time (ms) | `timings.txt` | NVMe: 501 MiB ÷ 7,400 MB/s ≈ 68ms theoretical; actual may differ |
| Warm restore time (ms) | `timings.txt` | T_KV2 baseline: ~280ms (0.28s) from page cache |
| Speedup factor (cold/warm) | `summary_computed.txt` | Target: ≥ 2× |
| posix_fadvise script exit code | stdout | 0 = success |
| Convergence cold restore (projected) | `summary_computed.txt` | ~17s (123 GB ÷ 7,400 MB/s) |
| Convergence warm restore (projected) | `summary_computed.txt` | ~0.28s (T_KV2 basis) |
| Production thinker restored | health check | 200 OK |

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| `preload-checkpoint.sh` runs without error | Exit 0, prints WILLNEED message | Debug Python os.posix_fadvise error; write Open from testing |
| Cold restore succeeds | `podman restore` exit 0, /health 200 | Record exact error; stop test; write Open from testing |
| Warm restore succeeds | `podman restore` exit 0, /health 200 | Record exact error; stop test; write Open from testing |
| Warm restore faster than cold | warm_ms < cold_ms | If not: posix_fadvise may not be effective on this filesystem; write Open from testing |
| Speedup ≥ 2× | warm_ms × 2 ≤ cold_ms | If <2×: note marginal benefit; record both values |
| Production thinker restored | /health 200 on port 30001 | Redeploy manually |

**Pass:** `preload-checkpoint.sh` works AND warm_ms < cold_ms. Speedup factor recorded.

---

## Artifacts to write

1. `infra/scripts/preload-checkpoint.sh` — the pre-warm script (created in Step 1)
2. `results/BENCH_18_qxpreload_nvme_preload_<timestamp>/timings.txt` — all timing values
3. `results/BENCH_18_qxpreload_nvme_preload_<timestamp>/summary_computed.txt` — speedup + Convergence projection
4. `results/BENCH_18_qxpreload_nvme_preload_<timestamp>/summary.md`:

```markdown
# BENCH_18 — QX_PRELOAD NVMe Checkpoint Pre-warm — <TIMESTAMP>

## Checkpoint
Path: /srv/ai/checkpoints/thinker/qwen36-27b-awq-tp1-ctx32k.ckpt
Size: <MiB> MiB

## Restore times
| Condition | Restore time |
|-----------|-------------|
| Cold (NVMe, page cache dropped) | <X> ms |
| Warm (posix_fadvise WILLNEED) | <X> ms |
| Speedup factor | <X>× |

## posix_fadvise script
WORKS / ERRORS (see timings.txt)

## Convergence projection
Cold restore (theoretical): ~<X>s (123 GB at 7,400 MB/s)
Warm restore (projected): ~<X>s (<basis>)
CRIU viable for Convergence (warm << 83s cold start): YES / MARGINAL / NO

## Production thinker
RESTORED / FAILED

## Verdict
PASS / FAIL / MARGINAL
```

**Do NOT write to any file outside `results/BENCH_18_qxpreload_nvme_preload_<timestamp>/` and `infra/scripts/`.**

---

## Interpretation boundary

- **You may record** cold and warm restore times, speedup factor, and the Convergence projection.
- **You may note** whether the projected Convergence warm restore time is below the 83s cold-start baseline.
- **You may NOT** run a full Convergence CRIU restore cycle — that is out of scope for this handoff.
- **You may NOT** update `docs/decisions/settled.md`, production config, or queue priority.
- **You may NOT** test other checkpoint files or pre-warm strategies (readahead, mlock) — this handoff tests posix_fadvise only.

## Stop condition

**Normal:** preload script created, cold and warm restore times recorded, speedup computed, Convergence projection written, production thinker restored, `summary.md` written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` and stop if:
- `posix_fadvise` raises `OSError` (filesystem does not support FADV_WILLNEED — e.g., tmpfs or a restrictive mount option)
- Cold restore exits non-zero (CRIU failure not seen before — different from T_KV2)
- Warm and cold restore times are within 10% of each other (page-cache pre-warm has no effect; investigate `drop_caches` vs filesystem cache interaction)
