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
Per-GPU MiB values were not captured in a separate file during this run. The process successfully re-attached to both GPUs after restore and completed inference (confirmed via API response). Exact VRAM breakdown was not recorded.

## Comparison vs BENCH_07 (--no-mmap)
| Metric | BENCH_07 (--no-mmap) | BENCH_08 (mmap) |
|--------|---------------------|-----------------|
| Checkpoint size | N/A (never measured — SYSTEM_OOM) | **8.7 GB** |
| Restore time | N/A (SYSTEM_OOM) | **7.3 s** |
| Rep-1 TTFT | N/A | **100.56 s** |

## Key finding
CRIU with mmap is technically feasible: 8.7 GB checkpoint, 7.3 s restore. However, the first-inference penalty (100.56 s) **exceeds** the cold start time (83 s). Without pre-loading the model file into page cache before the CRIU restore (QX_PRELOAD), CRIU mmap is slower than a cold start for time-to-first-response. With QX_PRELOAD (123 GB at 7,400 MB/s ≈ 17 s pre-warm), projected restore-to-interactive would be ~7 s + ~7 s = ~14 s — a 6× improvement over cold start. QX_PRELOAD is therefore a prerequisite for CRIU to benefit Convergence.

## Status
MEASURED ✓
