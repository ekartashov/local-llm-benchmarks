# CRIU + cuda-checkpoint Operations

> Enables sub-second mode switches for Extended Arclight. **0.28s hot restart** vs 100.2s cold start (T_KV2, 2026-04-26).

---

## One-time host setup

```bash
# 1. CRIU
sudo apt install criu
criu check           # verify kernel features
criu check --full    # exhaustive audit

# 2. NVIDIA cuda-checkpoint (GPU memory in checkpoint)
git clone https://github.com/NVIDIA/cuda-checkpoint /srv/ai/tools/cuda-checkpoint
cd /srv/ai/tools/cuda-checkpoint && make
sudo make install    # binary → /usr/local/bin/ + CRIU plugin → /usr/lib/criu/

# 3. Checkpoint directory
mkdir -p /srv/ai/checkpoints/coder-tp2

# Prerequisites: driver ≥ 570, newuidmap/newgidmap installed
```

---

## Required vLLM patch (do NOT revert)

CRIU cannot checkpoint processes with `io_uring` file descriptors. vLLM uses `uvloop` which creates `io_uring` rings. Patch must stay applied:

- `vllm/entrypoints/openai/api_server.py` — replace `uvloop.run()` with `asyncio.run()`
- `vllm/v1/utils.py` — replace `uvloop.run()` with `asyncio.run()`

**After any vLLM package update:** verify the patch is still in place. Without it, CRIU will fail with:
```
Error: Unknown shit 600 (anon_inode:[io_uring])
```

---

## Required environment

Always export before any CRIU operation:
```bash
export UV_USE_IO_URING=0   # prevents libuv from creating io_uring rings even if uvloop is imported
```

---

## Checkpoint + restore script

```bash
bash benchmarks/queue/T_KV2_cuda_checkpoint_tp2_hot_restart.sh
# Options:
#   --dry-run        Show commands without running
#   --reps N         Number of restore repetitions (default 3)
#   --ctx N          Context length for test inference (default 32768)
#   --gpu-mem F      GPU memory utilization (default 0.90)
#   --skip-cold      Skip cold-start baseline measurement
```

The script handles:
1. Cold-start baseline measurement
2. Checkpoint dump (via CRIU + cuda-checkpoint plugin)
3. N × restore with timing
4. Post-restore inference TPS verification
5. metrics.json and summary.md generation

---

## Manual checkpoint/restore

```bash
# Checkpoint a running vLLM container
export UV_USE_IO_URING=0
podman container checkpoint --export /srv/ai/checkpoints/coder-tp2/checkpoint.tar.gz \
  arclight-coder

# Restore from checkpoint
podman container restore --import /srv/ai/checkpoints/coder-tp2/checkpoint.tar.gz \
  --name arclight-coder-restored
```

**Note:** Host-native CRIU only. Podman CDI mount-point conflicts break checkpointing inside containers. Run hot-swaps on the host directly, not inside the claude-box container.

---

## Ghost VRAM cleanup (after failed restore)

After a failed CRIU restore, the GPU may retain "ghost" VRAM allocations from the partially-restored process. Clear them before the next deploy:

```bash
sudo nvidia-smi --gpu-reset -i 1   # GPU 1 (thinker GPU)
# or
sudo nvidia-smi --gpu-reset -i 0   # GPU 0 if needed
```

Without this, the next vLLM deploy may OOM even though nvidia-smi shows capacity available.

---

## Expected timing

| Operation | Time | Notes |
|-----------|------|-------|
| Cold start (TP=2 coder, 32K ctx) | ~100s | Full weight load + CUDA graph capture |
| Checkpoint dump | ~0.07s | GPU + CPU state |
| Hot restore | **~0.28s** | GPU weights already in VRAM |
| Post-restore inference startup | ~2s | First request warmup |
| Total Extended Arclight round-trip | ~5–10s | sleep thinker + CRIU restore + wake |

---

## Expected log messages during checkpoint/restore (not errors)

These appear during a normal successful run — do not treat them as failures:

```
Error toggling CUDA in process ID <PID>: "initialization error"
Warning: PID <PID> not a CUDA process
```
**Why:** The script freezes the entire vLLM process tree. The API Server and Engine Core processes do not use CUDA directly (only Workers do). cuda-checkpoint reports this as "initialization error" for those pids and continues to the GPU Workers. Normal.

```
Error toggling CUDA in process ID <PID>: "OS call failed or operation not supported on this OS"
```
**Why:** On restore, GPU state is already being managed by the driver's resume path. The immediate toggle attempt occasionally conflicts with the driver's stabilization phase. The script verifies health via `/health` immediately after and this resolves. Normal.

---

## Failure modes

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| `Unknown shit 600 (anon_inode:[io_uring])` | uvloop patch not applied | Verify api_server.py and v1/utils.py patches |
| Restore succeeds but inference is wrong | Checkpoint corrupted or CUDA state mismatch | Do NOT use; report to vLLM issue tracker |
| OOM after restore | Ghost VRAM from previous failed restore | `sudo nvidia-smi --gpu-reset -i 1` then retry |
| Checkpoint CDI mount errors | Running inside container instead of host | Run CRIU operations on host only |
| CRIU fails in rootless Podman | CDI/namespace conflicts | Confirmed: host-native is the only working path |
