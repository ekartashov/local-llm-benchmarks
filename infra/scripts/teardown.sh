#!/usr/bin/env bash
# teardown.sh — stop and remove all benchmark inference containers.
set -euo pipefail

CONTAINERS=(
    bench-vllm-gpu0
    bench-vllm-gpu1
    bench-sglang-gpu0
    bench-sglang-gpu1
    bench-llamacpp-gpu0
    bench-litellm
)

echo "[teardown] Stopping all bench-* containers..."

for name in "${CONTAINERS[@]}"; do
    if podman container exists "${name}" 2>/dev/null; then
        echo "[teardown] Stopping ${name}..."
        podman stop "${name}" 2>/dev/null || true
        podman rm   "${name}" 2>/dev/null || true
    fi
done

echo "[teardown] Done."
