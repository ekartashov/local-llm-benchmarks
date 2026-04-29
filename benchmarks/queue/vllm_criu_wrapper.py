import runpy
import sys
import os

# sitecustomize.py handles ZMQ and AsyncIO patches globally.
# We just need to ensure the environment is set up.

# Force V1 engine
os.environ["VLLM_USE_V1"] = "1" 
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Force maximum use of Shared Memory (which we patched to busy-wait)
os.environ["VLLM_MQ_MAX_CHUNK_BYTES_MB"] = "999999"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "120"
os.environ["VLLM_RPC_TIMEOUT"] = "120000"

import asyncio
os.environ["UV_USE_IO_URING"] = "0"
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

if __name__ == "__main__":
    print("[WRAPPER] Starting vLLM API Server...", file=sys.stderr)
    # Execute vLLM API server as the __main__ module
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
