# vLLM Host-Native Hot Restart (CRIU + cuda-checkpoint)

This document describes the "Arclight" Hot-Swap implementation used to achieve sub-second restart times for large models.

## 1. Architecture
The hot-swap mechanism uses **CRIU** (Checkpoint/Restore in Userspace) for CPU/Memory state and **NVIDIA cuda-checkpoint** for GPU state.

- **Process Memory**: Dumped by CRIU to disk as image files.
- **GPU Memory**: Serialized by the NVIDIA driver via the `cuda-checkpoint --toggle` utility.
- **Neutralization**: vLLM is patched to use standard `asyncio` instead of `uvloop` to avoid `io_uring` file descriptors, which are currently incompatible with CRIU.

## 2. Expected Log Messages
During a benchmark run, you will see the following messages. These are **normal** and do not indicate a failure:

### Pre-Dump Errors
> `Error toggling CUDA in process ID <PID>: "initialization error"`
> `Warning: PID <PID> not a CUDA process`
- **Reason**: The script attempts to freeze the entire vLLM process tree. The API Server and Engine Core processes do not use CUDA directly (only the Workers do). `cuda-checkpoint` reports this as an initialization error. 
- **Action**: None needed. The script correctly continues to the GPU Workers.

### Post-Restore Errors
> `Error toggling CUDA in process ID <PID>: "OS call failed or operation not supported on this OS"`
- **Reason**: Upon restoration, the GPU state is already being managed by the driver's resume path. Attempting an immediate toggle sometimes conflicts with the driver's stabilization phase.
- **Action**: None needed. The script verifies health via the `/health` endpoint immediately after.

## 3. Environment Variables
To ensure stability, the following must be set:
- `UV_USE_IO_URING=0`: Disables io_uring in libuv/uvloop.
- `VLLM_V1_ENABLED=1`: Required for Qwen3.6-35B-A3B.

## 4. Maintenance
If the vLLM package is updated, the following files may need to be re-patched if they revert to using `uvloop.run()`:
- `vllm/entrypoints/openai/api_server.py`
- `vllm/v1/utils.py`
