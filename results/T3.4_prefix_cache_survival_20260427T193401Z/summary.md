# T3.4 Prefix Cache Survival — 2026-04-27T19:47:30Z

## Timings

| Stage | Time (ms) |
|-------|-----------|
| Cold prefill | 2410 |
| Warm (pre-sleep) | 173 |
| Post-wake | FAILED |

## Cache hit ratio (pre-sleep)
173 / 2410 = **0.071**
(Success: 13.9x speedup via prefix cache)

## Post-wake Result
**ABORTED** — Server failed to wake up.

### Error Details
*   **Method**: `POST /wake_up`
*   **Status**: 500 Internal Server Error
*   **Exception**: `'list' object has no attribute 'zero_'`
*   **Context**: Occurs in vLLM V1 engine (`v1/engine/core_client.py`) during worker state restoration.

## Status
**FAIL (WAKE)** — Prefix cache works, but Sleep/Wake is broken for the Qwen3.6-35B-A3B architecture in vLLM V1.
