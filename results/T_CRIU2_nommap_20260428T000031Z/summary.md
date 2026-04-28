# T_CRIU2 CRIU Checkpoint/Restore — --no-mmap — 20260428T000031Z

## Result
CHECKPOINT_FAILED (SYSTEM_OOM)

## Pre-checkpoint baseline
Convergence was running and responsive. Pre-checkpoint inference: 8.74 t/s (50 tokens, 5.72 s total).

## Failure: CRIU dump triggered OOM killer

| Metric | Value |
|--------|-------|
| llama-server RSS at dump start | ~121 GB |
| llama-server VMS spike during dump | ~351 GB |
| Available system RAM | ~67 GB (of 188 GB total) |
| CRIU dump exit code | non-zero (OOM kill) |

**Root cause:** CRIU's parasite injection and memory-tracking overhead caused virtual address space to spike to ~351 GB during the dump phase. With the process holding ~121 GB of anonymous RAM (model weights loaded via `--no-mmap`), the CRIU dump buffer exhausted available system RAM. The Linux OOM killer terminated the process before the checkpoint could complete.

**Conclusion:** Checkpointing a ~135 GB anonymous-RAM process on a 188 GB system is physically impossible. `--no-mmap` CRIU for Convergence cannot be done on this hardware. Proceed to BENCH_08 (mmap variant).

## Post-restore TTFT
N/A — checkpoint failed before any restore was attempted.

## GPU VRAM after restore
N/A — checkpoint failed.

## Status
CHECKPOINT_FAILED
