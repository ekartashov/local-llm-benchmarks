#!/usr/bin/env bash
# precache-models.sh — pre-download all benchmark models into MODEL_CACHE.
#
# Runs huggingface-cli inside a throwaway container (vllm image, entrypoint
# overridden) so no host-side Python/HF tooling is required.
#
# Set HF_TOKEN in the environment before running if models are gated:
#   export HF_TOKEN=hf_...
#   ./infra/scripts/precache-models.sh
#
# To download only specific models, pass their registry keys as arguments:
#   ./infra/scripts/precache-models.sh qwen35_35b_a3b_awq qwen3_coder_30b_awq
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# Use a minimal Python image for downloads — avoids vLLM's CUDA init which
# hangs when the container runs as non-root (required for volume write access).
export HF_IMAGE="python:3.12-slim"
export MODELS_YAML="${REPO_ROOT}/config/models.yaml"
export FILTER_KEYS="${*:-}"   # space-separated registry keys to download (empty = all)

echo "[precache] MODEL_CACHE=${MODEL_CACHE}"
mkdir -p "${MODEL_CACHE}"
export MODEL_CACHE

python3 - <<'PYEOF'
import yaml, subprocess, sys, os
from pathlib import Path

models_yaml  = Path(os.environ["MODELS_YAML"])
model_cache  = os.environ["MODEL_CACHE"]
hf_image     = os.environ["HF_IMAGE"]
filter_keys  = os.environ["FILTER_KEYS"].split()   # empty list = download all

# Forward HF credentials into the container if set on the host.
hf_env: list[str] = []
for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    if val := os.environ.get(var):
        hf_env += ["-e", f"{var}={val}"]

data = yaml.safe_load(models_yaml.read_text())

failures: list[str] = []
for name, spec in data.get("models", {}).items():
    if filter_keys and name not in filter_keys:
        continue
    hf_repo = spec.get("hf_repo")
    if not hf_repo:
        continue
    model_file = spec.get("file")   # single-file GGUF variant

    label = f"file: {model_file}" if model_file else "full repo"
    print(f"[precache] Downloading {hf_repo} ({label}) ...")
    sys.stdout.flush()

    dl_args = ["download", hf_repo]
    if model_file:
        dl_args += ["--include", model_file]

    cmd = [
        "podman", "run", "--rm",
        "--userns=keep-id",          # host user UID = container UID → no permission errors
        "-e", "HF_HOME=/data",       # write cache to the mounted volume root
        *hf_env,
        "-v", f"{model_cache}:/data:z",
        hf_image,
        "sh", "-c",
        f"pip install -q huggingface_hub && hf download {hf_repo}"
        + (f" --include {model_file}" if model_file else ""),
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[precache] WARNING: failed to download {hf_repo}", file=sys.stderr)
        failures.append(hf_repo)
    else:
        print(f"[precache] OK: {hf_repo}")
    sys.stdout.flush()

if failures:
    print(f"\n[precache] {len(failures)} download(s) failed: {failures}", file=sys.stderr)
    sys.exit(1)
print("\n[precache] All models downloaded.")
PYEOF
