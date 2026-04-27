# T_PAR1 Thinker Baseline — 2026-04-27T18:35:25Z
## Config: max-num-seqs=1 (production)

| N (concurrent clients) | Aggregate TPS | TTFT (median) | Notes |
|------------------------|---------------|---------------|-------|
| 1 | 76.9 | 74 ms | |
| 2 | 76.9 | 3,401 ms | queued (max-num-seqs=1) |
| 4 | 76.9 | 10,060 ms | queued |
| 8 | 76.8 | 23,391 ms | queued |

## GPU 1 VRAM
27736 MiB

## Status
MEASURED — raw data in metrics.json
