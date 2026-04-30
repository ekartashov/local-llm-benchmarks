import runpy
import sys
import os

# sitecustomize.py (loaded via PYTHONPATH=python_hijack/) patches:
#   - zmq.Poller  → CriuSafePoller (PID-change re-registration)
#   - asyncio     → DefaultEventLoopPolicy (no uvloop / io_uring)
#   - SpinCondition.wait() → sched_yield() busy-wait (no broken inproc:// poll)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S", "600")
os.environ.setdefault("VLLM_RPC_TIMEOUT", "600000")

import asyncio
os.environ["UV_USE_IO_URING"] = "0"
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

if __name__ == "__main__":
    print("[WRAPPER] Starting vLLM API Server...", file=sys.stderr)
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
