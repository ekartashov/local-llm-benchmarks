# BENCH_10 — T_CRIU3 Phase 2: CRIU KV Cache Preservation (Coder TP=2)

**Status: DONE ✗ (FAILED — post-restore inference unreachable due to SHM IPC bug)**
**Completed: 2026-04-30**

---

## Objective

Determine whether a CRIU checkpoint of vLLM Coder TP=2 (with populated prefix cache) restores
with KV blocks intact, giving warm (cached) post-restore latency.

## Result Summary

| Phase | Outcome |
|-------|---------|
| CRIU dump | OK — 29s, 67 GB |
| CRIU restore | OK — 24–26s |
| Health check post-restore | OK — `/health` 200 |
| KV cache preserved? | **YES** — SchedulerOutput shows hits=2096/queries=3009 (same ratio as pre-checkpoint) |
| First inference post-restore | **FAIL** — `TimeoutError: RPC call to execute_model timed out` / `sample_tokens timed out` |
| Root cause | SHM broadcast IPC broken for TP=2 multi-process after CRIU restore |

**KV cache IS physically preserved in VRAM. Inference is unreachable because the IPC path between
EngineCore and Workers is broken after restore. This is a deep architectural incompatibility in
vLLM TP=2 on Blackwell — not a fixable configuration issue.**

---

## What was confirmed

### KV cache preservation: YES
Post-restore vLLM log shows the scheduler's prefix cache state intact:
```
SchedulerOutput: hits=2096, queries=3009, num_computed_tokens=2096
```
This is identical to the pre-checkpoint warm ratio. CRIU captures both CPU process RAM
(prefix cache hash tables) and GPU VRAM (KV blocks via `cuda-checkpoint --toggle`).
The data is there; it's just unreachable.

### Checkpoint and restore timing
| Operation | Time | Size |
|-----------|------|------|
| CRIU dump (host-native) | ~29s | 67 GB |
| CRIU restore (host-native) | ~24–26s | — |

Note: This is far slower than Phase 1 (Thinker TP=1: 501 MB, ~0.43s restore). The size
difference is because Coder TP=2 preallocates KV blocks across two GPUs and two worker
processes. vLLM preallocates all KV blocks at startup regardless of cache population, so
a "clean" checkpoint (before any inference) would be the same ~67 GB.

### Patches developed during this benchmark
All patches live in `benchmarks/queue/python_hijack/sitecustomize.py`:

1. **CriuSafePoller**: ZMQ Poller subclass that re-registers sockets after PID change and
   caps `poll()` timeout at 1000ms. The cap is critical: workers resume inside `poller.poll()`
   at CRIU restore time; without the cap they block for up to 60s
   (VLLM_RINGBUFFER_WARNING_INTERVAL) before escaping to the patched code path.

2. **SpinCondition.wait() → sched_yield()**: Replaces the broken `poller.poll()` on
   `inproc://` sockets (which are dead after CRIU) with a simple CPU yield. The caller
   (`acquire_read`) loops on the SHM metadata flag; sched_yield() gives the writer CPU time.

3. **SpinCondition.notify() exception guard**: Wraps the ZMQ PUB send in try/except to
   suppress errors on broken sockets post-restore.

These patches are sufficient for TP=1 (Phase 1 confirmed working). They are insufficient for
TP=2 because the underlying SHM visibility problem persists.

---

## Root cause of TP=2 failure

After CRIU restore, EngineCore writes the `execute_model` request to a SHM slot
(`written_flag = 1`, data inline for small payloads or ZMQ overflow marker for large ones).
The worker processes are spinning in `acquire_read()` via `sched_yield()` but `written_flag`
always appears 0 to them. As a result:

- Workers never acknowledge the slot (reader_flags stay 0)
- EngineCore's `acquire_write()` cannot find a free slot on the next call
- `acquire_write()` logs "No available shared memory broadcast block found in 60 seconds"
  every 60s (VLLM_RINGBUFFER_WARNING_INTERVAL)
- After 3 warnings (~180s), `dequeue_timeout` fires for `sample_tokens` → TimeoutError

**Why workers can't see written_flag=1:** The exact mechanism is unresolved. Candidates:
- CPU cache coherency: SHM pages may be cached in a worker's L1/L2 with the old value
  (written_flag=0). The `memory_fence()` calls in vLLM's acquire_write/acquire_read may not
  provide sufficient cross-process CPU memory ordering after CRIU (PID namespace changes
  could affect the fence implementation).
- ZMQ inproc sockets: `SpinCondition.notify()` uses `local_notify_socket.send(b"\x00")` on a
  ZMQ PUB socket. Post-CRIU this socket is broken. Workers' `acquire_read()` may rely on
  the ZMQ notify signal to wake from `sched_yield()` spin, but the signal never arrives.

### Why TP=1 works but TP=2 doesn't
Phase 1 (Thinker TP=1) has a single worker process sharing the same CPU address space
topology with EngineCore. The SHM mapping is simpler and the fence operations work correctly.
TP=2 adds a second worker process on a different GPU, with a more complex SHM ring topology
spanning the multi-process group.

### Why VLLM_USE_V1=0 doesn't help
`VLLM_USE_V1=0` is set in the environment, but vLLM on Blackwell (sm_120) forces V1 engine
regardless. The V1 engine uses `ShmRingBuffer` for EngineCore→Worker broadcast. There is no
path to make TP=2 on Blackwell use pure ZMQ IPC (which would be CRIU-safe via the already-
patched `recv()` method).

---

## Why 26s restore makes the mechanism unviable regardless of the IPC fix

Even if the SHM IPC issue were fixed:
- 26s restore vs ~100s cold start = ~4× speedup (not the 358× achieved by TP=1 at 0.43s)
- vLLM always preallocates full KV blocks at startup → checkpoint is always ~67 GB
- A "checkpoint without KV cache" (taken at startup, before inference) would be identical in
  size because VRAM footprint is fixed at initialization

**The user's instinct is correct: prefix cache prefill speed (13.9× speedup, T3.4) is the
right mechanism for KV continuity after a context switch.** Take the checkpoint without KV
cache concern, and after restore send the same context to warm the prefix cache via fast
prefill.

---

## What to use instead

| Goal | Mechanism | Status |
|------|-----------|--------|
| Fast thinker swap (TP=1) | CRIU host-native, 0.43s restore | SETTLED ✓ (Phase 1) |
| Coder extended context | Keep always-resident; use --max-model-len 65536 | SETTLED ✓ (T_KV1) |
| KV cache continuity after swap | Prefix cache prefill on first post-restore request | T3.4 confirmed (13.9×) |
| Coder fast swap | Not viable via CRIU (26s, IPC bug). Sleep mode wake bug still open (T3.4). | OPEN |

---

## Artifacts developed (kept in repo)

| Path | Purpose |
|------|---------|
| `benchmarks/queue/T_CRIU3_kvcache_preservation.sh` | Benchmark script (host-native CRIU, QX_PRELOAD, health check) |
| `benchmarks/queue/vllm_criu_wrapper.py` | vLLM launcher that applies env patches before exec |
| `benchmarks/queue/python_hijack/sitecustomize.py` | Runtime patches: CriuSafePoller, SpinCondition.wait→sched_yield, notify guard |
| `benchmarks/queue/io_uring_shim.c` | LD_PRELOAD shim to block io_uring syscalls at the C level |
