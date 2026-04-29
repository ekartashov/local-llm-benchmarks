# CRIU + vLLM: Resolving the io_uring Conflict

This document details the technical resolution for the `Unknown shit 600 (anon_inode:[io_uring])` error encountered when attempting to checkpoint vLLM processes using CRIU.

## The Problem
CRIU (Checkpoint/Restore In Userspace) is fundamentally incompatible with `io_uring` file descriptors. When vLLM (specifically V1 or recent versions using `uvloop`) initializes, it often creates `io_uring` instances for high-performance asynchronous I/O. If these descriptors are open during a CRIU dump, the process fails with:
`Error (criu/proc_parse.c:479): Unknown shit 600 (anon_inode:[io_uring])`

## The Solution: Nuclear Loop Policy Override

Standard environment variables like `UV_USE_IO_URING=0` often fail because they are checked too late, or because some libraries initialize the loop before the variable is processed.

We resolved this by injecting a **pre-emptive Event Loop Policy override** at the very start of the vLLM process tree.

### 1. The Wrapper Script Architecture
Instead of using complex `-c` command line injections which can confuse process trackers like `cuda-checkpoint`, we use a dedicated wrapper script:

**File**: `benchmarks/queue/vllm_criu_wrapper.py`
```python
import asyncio
import runpy

# Force the standard SelectorEventLoop (CRIU-compatible)
# This MUST be done before any other vLLM or uvloop imports.
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

if __name__ == "__main__":
    # Execute vLLM API server as the __main__ module
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
```

### vLLM V1 Engine & shm_broadcast (v0.19.1+)
The newer vLLM V1 engine (default for many architectures like Qwen3) uses a high-performance `shm_broadcast` mechanism for inter-process RPC. This mechanism relies on a shared-memory ring buffer with custom spinlocks (`SpinCondition`) that often become inconsistent after a CRIU restore, leading to `TimeoutError: RPC call to execute_model timed out`.

#### Workaround: Force ZMQ-only Communication
To bypass the brittle shared-memory path, we can force vLLM to use standard ZMQ sockets for nearly all inter-process traffic:
- Set `VLLM_MQ_MAX_CHUNK_BYTES_MB=1` (or a small non-zero value).
- This ensures that any RPC payload larger than the threshold (which includes almost all scheduler outputs) "overflows" to a ZMQ `send_multipart` call, bypassing the `ShmRingBuffer` spinlocks.
- Setting this to `0` causes an `IndexError` in vLLM's internal buffer slicing, so `1` is the recommended minimum.

### Updated Checklist
- [x] Neutralize `io_uring` via `LD_PRELOAD` shim.
- [x] Standardize `asyncio` loop policy to `DefaultEventLoopPolicy`.
- [x] Disable `uvloop` (often bundled with vLLM).
- [x] **New**: Force ZMQ-only RPC via `VLLM_MQ_MAX_CHUNK_BYTES_MB=1` to avoid V1 engine hangs.
- [x] Use `cuda-checkpoint` for NVIDIA 595+ driver stability.

This ensures that the `asyncio` loop policy is set at the absolute entry point of the process tree, including for any workers spawned via `multiprocessing`.

### 2. Supporting vLLM Flags
To further minimize asynchronous noise that might trigger `io_uring`, we added:
- `--no-async-scheduling`: Disables the vLLM V1 high-performance scheduler.
- `VLLM_USE_V1=0`: Forces the legacy vLLM V0 engine path.

### 3. CUDA Toggle Syntax (Driver 595+)
For NVIDIA Driver 595 and `cuda-checkpoint` version 595.58.03, the `--toggle-off` and `--toggle-on` flags are **not supported**.

**Correct Syntax**:
```bash
sudo cuda-checkpoint --toggle --pid <PID>
```

**Verification**:
```bash
sudo cuda-checkpoint --get-state --pid <PID>
# Output: checkpointed
```

When a process is in the `checkpointed` state, its GPU memory mappings are released/suspended in a way that allows CRIU to perform a standard dump **without needing a specialized CRIU plugin library** (no `/usr/lib/criu/` required).

### 4. The Final Boss: Persistent io_uring
Despite the loop policy overrides, some deep-seated C++ libraries (PyTorch/NCCL) still initialized `io_uring` rings, which are invisible to Python-level overrides.

**The Fix**: A dedicated C shim injected via `LD_PRELOAD` that intercepts the `syscall()` function and hard-blocks `io_uring` at the kernel boundary.

**Shim Implementation (`benchmarks/queue/io_uring_shim.c`)**:
```c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/syscall.h>
#include <errno.h>
#include <stdarg.h>

long syscall(long number, ...) {
    if (number == SYS_io_uring_setup || number == SYS_io_uring_enter || number == SYS_io_uring_register) {
        errno = ENOSYS; // Return "Function not implemented"
        return -1;
    }
    // Pass-through to real syscall
    typedef long (*syscall_t)(long, ...);
    static syscall_t real_syscall = NULL;
    if (!real_syscall) real_syscall = (syscall_t)dlsym(RTLD_NEXT, "syscall");
    
    va_list args;
    va_start(args, number);
    long a1 = va_arg(args, long);
    long a2 = va_arg(args, long);
    long a3 = va_arg(args, long);
    long a4 = va_arg(args, long);
    long a5 = va_arg(args, long);
    long a6 = va_arg(args, long);
    va_end(args);
    return real_syscall(number, a1, a2, a3, a4, a5, a6);
}
```

### 5. Milestone: Successful TP=2 Checkpoint/Restore
On April 29, 2026, we achieved the first stable CRIU hot-restart for vLLM TP=2:
- **Checkpoint Time**: 29.1s
- **Checkpoint Size**: 69GB
- **Restore Time**: 14.46s
- **Status**: Process tree fully restored and healthy.

## Summary of Fixes in T_CRIU3
| Fix | Purpose |
|-----|---------|
| `runpy` Injection | Forces standard `asyncio` loop, blocking `io_uring`. |
| `pkill -u $USER` | Silently clears ghost processes without breaking terminal layout. |
| Persistence Recovery | Recreates `/run/nvidia-persistenced/socket` to prevent OCI errors. |
| Redirected Toggling | Silences expected "initialization errors" on non-CUDA API processes. |
