# T_CRIU2 CRIU Checkpoint/Restore — --no-mmap — 20260428T000031Z

## Result
**FAILED: SYSTEM_OOM**

## Analysis
- **Execution:** The benchmark successfully started `llama-server` on the host and performed a baseline inference (8.74 t/s).
- **Failure:** During the `criu dump` phase, the system triggered the Linux **OOM Killer**.
- **Metrics:**
    - `llama-server` RSS: ~121 GB
    - `llama-server` Virtual Memory: **351 GB** (spiked during dump)
    - Available RAM: 188 GB
- **Conclusion:** Dumping a 135 GB model stored in anonymous RAM (`--no-mmap`) is physically impossible with the current 188 GB RAM headroom. The parasite injection and memory tracking overhead exceed the system's capacity.

## Status
FAILED (OOM) - Move to BENCH_08 (mmap)
