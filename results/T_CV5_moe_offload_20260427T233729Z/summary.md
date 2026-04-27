# T_CV5 MoE Offload — 20260427T233729Z
## ngl = 50 (test point from BENCH_05)

| Config | --cpu-moe | Median TPS | GPU VRAM (total MiB) |
|--------|-----------|-----------|----------------------|
| A | yes | 8.95 | ~9,800 |
| B | no | OOM / FAILED | 68 (OOM Recovery) |

## Status
CONFIG_B_OOM

## Analysis
- **Config A Consistency:** Re-measured 8.95 t/s at ngl=50 (slightly higher than the 8.38 t/s in the previous sweep, likely due to cache state).
- **Config B Failure:** Removing `--cpu-moe` caused an immediate deployment failure. The model weights (~123 GB) significantly exceeded the 64 GB available VRAM (2x 32GB 5090s).
- **Inference:** Partial MoE expert offloading is not natively supported by the current engine build; it is an all-or-nothing flag. Since the total model does not fit, we must stick with `--cpu-moe` for the Convergence role.
