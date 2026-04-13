#!/usr/bin/env bash
# teardown.sh — stop and remove all benchmark inference containers.
# Finds all docker compose projects with the "bench-" prefix and brings them down.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[teardown] Stopping all bench-* containers..."

# Down each known engine/gpu compose stack
declare -a STACKS=(
    "infra/compose/vllm-gpu0.yaml:bench-vllm-gpu0"
    "infra/compose/vllm-gpu1.yaml:bench-vllm-gpu1"
    "infra/compose/sglang-gpu0.yaml:bench-sglang-gpu0"
    "infra/compose/sglang-gpu1.yaml:bench-sglang-gpu1"
    "infra/compose/llamacpp-gpu0.yaml:bench-llamacpp-gpu0"
    "infra/compose/litellm.yaml:bench-litellm"
)

for entry in "${STACKS[@]}"; do
    compose_file="${REPO_ROOT}/${entry%%:*}"
    project="${entry##*:}"
    if docker compose -f "${compose_file}" -p "${project}" ps -q 2>/dev/null | grep -q .; then
        echo "[teardown] Stopping ${project}..."
        docker compose -f "${compose_file}" -p "${project}" down --remove-orphans
    fi
done

echo "[teardown] All benchmark containers stopped."
