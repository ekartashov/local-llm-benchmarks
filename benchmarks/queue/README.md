# benchmarks/queue/

Shell wrappers for TESTING_QUEUE.md items. One script per item: `T1.1_sleep_mode.sh`,
`T2.1_glm47_mla_verify.sh`, etc. The justfile `just t1.1` target calls the wrapper.

## Why scripts don't exist yet

The exact flag names for Sleep Mode endpoints, MLA detection, and MTP spec decode depend
on the vLLM version actually running in the container. Flags change between minor releases.
Pre-writing scripts against training-data knowledge would embed stale flags that fail silently.

## Protocol for authoring a new T*.sh wrapper

**Run all of these on the host (testing mode), not inside the Claude container.**

1. **Identify the running vLLM version:**
   ```bash
   podman run --rm vllm/vllm-openai:latest vllm --version
   # or, if a container is already running:
   podman exec bench-vllm-tp2 vllm --version
   ```

2. **Check the queue item** in `TESTING_QUEUE.md` for the exact procedure and pass criteria.

3. **Check DECISIONS.md** — confirm no SETTLED decision makes this test redundant.

4. **Verify flag availability against the live binary:**
   ```bash
   # Check whether a flag exists before writing it into the script:
   podman run --rm vllm/vllm-openai:latest vllm serve --help | grep -E "sleep|wake|dev.mode"
   podman run --rm vllm/vllm-openai:latest vllm serve --help | grep -E "tensor.parallel"
   podman run --rm vllm/vllm-openai:latest vllm serve --help | grep -E "speculative"
   podman run --rm vllm/vllm-openai:latest vllm serve --help | grep -E "reasoning.parser"
   ```

5. **Write the minimal wrapper.** Structure:
   ```bash
   #!/usr/bin/env bash
   # T1.X_description.sh — one-line summary of what this tests
   # Verified against vLLM <version> on <date>
   set -euo pipefail
   REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
   source "${REPO_ROOT}/config/hardware.env"
   # ... activate python env ...
   # ... deploy step ...
   # ... benchmark step ...
   # ... teardown step ...
   ```

6. **Dry-run before first real run:** use `--dry-run` if the benchmark script supports it,
   or deploy with `--max-model-len 512` and a trivial prompt to verify the container starts.

7. **Commit the script** once the first real run completes and produces a `results/` entry.

## deploy.sh interface (quick reference)

```bash
# Single model, TP=2 (the default for new queue items):
./infra/scripts/deploy.sh vllm tp2 <hf_repo> \
    --gpu-mem-util 0.85 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3

# Two concurrent models, each at 40% GPU mem (T1.2):
./infra/scripts/deploy.sh vllm tp2a <hf_repo_coder>  --gpu-mem-util 0.40 ...
./infra/scripts/deploy.sh vllm tp2b <hf_repo_thinker> --gpu-mem-util 0.40 ...

# Sleep Mode (requires VLLM_SERVER_DEV_MODE=1):
VLLM_SERVER_DEV_MODE=1 ./infra/scripts/deploy.sh vllm tp2 <hf_repo> --gpu-mem-util 0.85 ...
# Sleep:  curl -s -X POST http://localhost:30000/sleep?level=1
# Wake:   curl -s -X POST http://localhost:30000/wake_up

# Long-context behemoth (T1.3, tp2c slot, port 30002):
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
./infra/scripts/deploy.sh vllm tp2c cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit \
    --gpu-mem-util 0.85 --ctx 32768 --tool-call-parser qwen3_coder
```

Named args consumed by deploy.sh (not forwarded to engine):
- `--ctx N` — sets `--max-model-len N` (default 32768)
- `--gpu-mem-util F` — sets `--gpu-memory-utilization F` (default 0.90)
- `--model-file F` — llama.cpp only

Env vars forwarded to the container: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
`VLLM_SERVER_DEV_MODE`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN`.

## Results convention

Every T*.sh script must write to `results/{item_id}_{timestamp}/`:
- `metrics.json` — structured metrics matching the schema in CLAUDE.md
- `summary.md` — human-readable summary via `lib.reporter`
- `bench.log` — full log (redirect stdout+stderr)
- `raw/` — raw API responses (optional, can be large)

`item_id` must match the TESTING_QUEUE.md entry exactly (e.g. `T1.1_sleep_mode_operational_under_podman`).

## Python env

Scripts run on the host. Activate before running any benchmark:
```bash
pyenv activate hf   # canonical; .venv symlink points to the same env
```
