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
FAIL (WAKE) — Prefix cache confirmed working (0.071 ratio). Sleep/wake cycle broken for Qwen3.6-35B-A3B with --enforce-eager.

## Open from testing
Wake failure: `POST /wake_up` returns HTTP 500, `'list' object has no attribute 'zero_'` in `v1/engine/core_client.py`. This is a new failure mode not seen in T1.1 (which tested a different model/config). The prefix cache result (cold 2410ms → warm 173ms, 13.9× speedup) is valid and clean. The wake bug must be investigated before T3.4 can be closed. Research question: is `--enforce-eager` required for sleep mode on Blackwell, and does it break the wake path for this specific model?
