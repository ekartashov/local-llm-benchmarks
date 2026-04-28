# T_PAR1 Thinker Parallel — 2026-04-27T19:10:30Z
## Config: max-num-seqs=4

### Result
PASS — Scaling confirmed up to N=4.

### TPS at max-num-seqs=4
| N | Aggregate TPS | TTFT (median) | vs BENCH_02 (max-num-seqs=1) |
|---|---------------|---------------|-------------------------------|
| 1 | 76.8 | 73 ms | ~76.9 t/s |
| 2 | 139.3 | 76 ms | +81% TPS (Baseline TTFT ~3.4s) |
| 4 | 269.4 | 140 ms | +250% TPS (Baseline TTFT ~10s) |
| 8 | 269.6 | 3,938 ms | Plateau (Baseline TTFT ~23s) |

### GPU 1 VRAM
27732 MiB

### Production config
NOTE: The coder endpoint (localhost:30000) was not running during this test — coder_detail is null in metrics.json. Coder sweep data is in the separate BENCH_01 run (T_PAR1_parallel_throughput_sweep_20260427T180802Z). Production thinker config (max-num-seqs=1) must be restored before returning to normal operation.
