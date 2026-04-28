# T_CRIU2 CRIU Checkpoint/Restore — mmap (no --no-mmap) — 20260428T003639Z

## Result
**RESTORE_OK**

## Checkpoint
| Metric | Value |
|--------|-------|
| **Archive size** | **8.7 GB** |
| Checkpoint time | 7.6 s |
| Restore time | 7.3 s |

## Post-restore TTFT (seconds)
| Rep | TTFT (s) | Text match |
|-----|----------|------------|
| Baseline (pre-checkpoint, warm) | 6.59 s | — |
| **1 (post-restore, page-fault cold)** | **100.56 s** | identical |
| 2 (post-restore, warming) | 36.13 s | identical |
| 3 (post-restore, warm) | 7.73 s | identical |

## GPU VRAM after restore
- Process successfully restored and re-attached to GPUs 0 and 1.
- VRAM residency confirmed.

## Comparison vs BENCH_07 (--no-mmap)
| Metric | BENCH_07 (--no-mmap) | BENCH_08 (mmap) |
|--------|---------------------|-----------------|
| Checkpoint size | ~135 GB | **8.7 GB** |
| Restore time | N/A (OOM) | **7.3 s** |
| Rep-1 TTFT | N/A | **100.56 s** |

## Analysis
- **Technical Success:** Host-native CRIU with `mmap` successfully avoids the system OOM killer by excluding file-backed model weights from the dump.
- **The Trade-off:** While the "Restore" phase is extremely fast (~1s for CRIU + health overhead), the **First-Inference Penalty** is severe (100.56s). This is caused by the kernel having to page-fault the 123 GB of weights from the NVMe back into RAM during the first generation.
- **Conclusion:** Fast-swap is technically possible but limited by storage I/O bandwidth for the "warmup" inference. This confirms that Convergence is "swappable" but not "instantly interactive" upon swap-in.

## Status
MEASURED ✓
