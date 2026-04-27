# T_CV5 NGL Sweep — 20260427T205900Z

| -ngl | Median TPS | GPU 0 VRAM (MiB) | GPU 1 VRAM (MiB) |
|------|-----------|-----------------|-----------------|
| 0    | 3.70 (T_CV1 baseline) | — | — |
| 10   | 4.10 | 3842 | 1930 |
| 20   | 4.72 | 4344 | 2438 |
| 35   | 6.02 | 5142 | 3138 |
| 50   | 8.38 | 5840 | 3946 |
| 999  | 13.99 (T_CV3 baseline) | — | — |

## Status
MEASURED

## Analysis
The decode TPS shows an accelerating upward curve as more layers are offloaded to GPU. 
- 10 layers: +0.40 t/s (+11% vs CPU-only)
- 20 layers: +1.02 t/s (+28% vs CPU-only)
- 35 layers: +2.32 t/s (+63% vs CPU-only)
- 50 layers: +4.68 t/s (+126% vs CPU-only)

Saturation (defined as 90% of 13.99 = 12.59 t/s) was **NOT** reached at ngl=50. The remaining 14 layers (assuming 64 total) are responsible for a significant portion of the performance gap (13.99 - 8.38 = 5.61 t/s).

Each layer offload consumes ~50 MiB of VRAM across both GPUs (~25 MiB per GPU per layer). 
At ngl=50, total Convergence VRAM overhead is ~9.8 GB (5840+3946).

## Recommendation for BENCH_06
Since saturation was not reached at 50, BENCH_06 cannot use a "saturation ngl" from the measured set.
However, for the purpose of testing MoE offload gains, we should either:
1. Run a follow-up sweep for ngl=[55, 60, 64] to find the true saturation point.
2. Proceed with BENCH_06 using ngl=50 as the "best effort" point to see if expert offload can compensate for the missing attention layers.
