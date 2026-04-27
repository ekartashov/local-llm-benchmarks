# Extended Arclight — CRIU Hot-Restart & Big-Context Mode

> **Summary:** Thinker sleeps (level=1, ~4s) → coder restarts as TP=2 across both GPUs → 65K context available. Mode switch time: **0.28s** via CRIU+cuda-checkpoint (T_KV2, SETTLED 2026-04-26). Swap-space extension BLOCKED (vLLM 0.19.0 doesn't recognize `--swap-space` flag). Thinker Extended mode GATED on T_KV3.

---

## Context ceiling

| Config | Max context | TPS | TTFT at ceiling | Status |
|--------|-------------|-----|-----------------|--------|
| Coder TP=2, fp8 KV, no swap | **65,536 tokens** | 238.2 t/s | 3022ms | SETTLED T_KV1 |
| Coder TP=2, swap-space 32 | 131,072 (target) | — | — | BLOCKED (flag unrecognized) |
| Thinker TP=2 | ~150K (estimated) | — | — | GATED T_KV3 |

TTFT sweep at 65K ctx: 8K=2448ms/187.5 t/s, 16K=710ms/246.9, 32K=1448ms/244.6, 65K=3022ms/238.2. TPS regression vs 32K baseline: 2.6%.

---

## CRIU hot-restart

**Result (T_KV2, 2026-04-26):**
- Cold start: **100.2s**
- CRIU checkpoint dump: **0.072s**
- Hot restart (restore): **0.28s** (358× speedup)
- Post-restore TPS: 210 t/s (within ±10% of 232 t/s cold-start)

### Prerequisites (one-time host setup)
```bash
# 1. CRIU
sudo apt install criu
criu check          # verify kernel features

# 2. cuda-checkpoint CLI + CRIU hooks
git clone https://github.com/NVIDIA/cuda-checkpoint /srv/ai/tools/cuda-checkpoint
cd /srv/ai/tools/cuda-checkpoint && make && sudo make install
# Installs binary to /usr/local/bin/ + CRIU plugin to /usr/lib/criu/

# 3. Checkpoint directory
mkdir -p /srv/ai/checkpoints/coder-tp2
```

### Requirements for stability (all mandatory)

1. **Host-native execution** — Podman CDI mount-point conflicts break CRIU. Run production hot-swaps on the host directly.

2. **uvloop patch in vLLM** — `api_server.py` and `v1/utils.py` must use `asyncio.run()` instead of `uvloop.run()`. CRIU cannot checkpoint processes with `io_uring` file descriptors (`Unknown shit 600 (anon_inode:[io_uring])`). Do NOT revert this patch. Any vLLM package update must re-verify the patch.

3. **`UV_USE_IO_URING=0`** — export before any CRIU operation so `libuv` also stays clean.

4. **VRAM hygiene** — after a failed restore, run `sudo nvidia-smi --gpu-reset -i 1` to clear ghost VRAM leaks. Without this, the next deploy may OOM.

### CRIU checkpoint/restore script
```bash
bash benchmarks/queue/T_KV2_cuda_checkpoint_tp2_hot_restart.sh
# Options: --dry-run, --reps N, --ctx N, --gpu-mem F, --skip-cold
```

---

## Thinker Extended mode (GATED on T_KV3)

Thinker TP=2 is **definitively broken** for GDN (Gated DeltaNet) architecture. H-TP2 CONFIRMED by T2.4g (2×2 factorial complete — TP=2 breaks DeltaNet recurrent state sync regardless of chunked-prefill setting). Cannot use until T_KV3 resolves:
- Sub-Q2: identify a TP=2-capable thinker (non-GDN: pure Transformer or MLA architecture)
- Quality bar ≥ Qwen3.6-27B (4.875/5), fits ~21GB AWQ at TP=1

---

## Sharp edges

- After T_KV1's context sweep, always restore production config: stop the TP=2 extended coder, restart coder TP=1 GPU0 + thinker TP=1 GPU1.
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` is required for coder TP=2 or it OOMs during CUDA graph capture.
- Level=1 sleep retains ~4 GiB GPU VRAM residual (CUDA graphs + caching allocator). This is intentional for <1s wake — cannot be reduced.
