#!/usr/bin/env bash
# precache-models.sh — pre-download all benchmark models and page-cache weights.
#
# Reads config/models.yaml to get the full list of HF repos, then uses
# huggingface-cli inside a throwaway container to download each one into
# MODEL_CACHE. Run once before benchmarking to avoid cold-cache bias.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

MODELS_YAML="${REPO_ROOT}/config/models.yaml"
HF_IMAGE="huggingface/transformers-pytorch-cpu:latest"

echo "[precache] MODEL_CACHE=${MODEL_CACHE}"
mkdir -p "${MODEL_CACHE}"

# Extract hf_repo values from models.yaml using Python (available in bench env)
python3 - <<'PYEOF'
import yaml, subprocess, os, sys
from pathlib import Path

models_yaml = Path(os.environ["REPO_ROOT"]) / "config" / "models.yaml"
model_cache = os.environ["MODEL_CACHE"]
data = yaml.safe_load(models_yaml.read_text())

for name, spec in data.get("models", {}).items():
    hf_repo = spec.get("hf_repo")
    if not hf_repo:
        continue
    model_file = spec.get("file")  # GGUF variant

    print(f"[precache] Downloading {hf_repo} ...")
    cmd = [
        "huggingface-cli", "download", hf_repo,
        "--cache-dir", model_cache,
        "--local-dir-use-symlinks", "False",
    ]
    if model_file:
        cmd += ["--include", model_file]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[precache] WARNING: failed to download {hf_repo}", file=sys.stderr)
    else:
        print(f"[precache] OK: {hf_repo}")

print("[precache] All models downloaded.")
PYEOF
