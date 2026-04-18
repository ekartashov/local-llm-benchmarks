#!/usr/bin/env bash
# teardown.sh — stop and remove all benchmark inference containers.
set -euo pipefail

# Standard named containers plus dynamic discovery of ad-hoc runs
TARGET_PATTERNS=("bench-" "vllm" "sglang" "llamacpp" "litellm")

# Gather list of existing containers matching our target engines
ALL_NAMES=$(podman ps -a --format "{{.Names}}" 2>/dev/null || true)
CONTAINERS=()
for name in $ALL_NAMES; do
    for pat in "${TARGET_PATTERNS[@]}"; do
        if [[ "${name}" == *"${pat}"* ]]; then
            CONTAINERS+=("${name}")
            break
        fi
    done
done

echo "[teardown] Stopping ${#CONTAINERS[@]} matching containers..."

for name in "${CONTAINERS[@]}"; do
    if podman container exists "${name}" 2>/dev/null; then
        echo "[teardown] Stopping ${name}..."
        podman stop "${name}" 2>/dev/null || true
        podman rm   "${name}" 2>/dev/null || true
    fi
done

echo "[teardown] Done."
