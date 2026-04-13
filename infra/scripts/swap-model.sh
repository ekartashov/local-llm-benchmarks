#!/usr/bin/env bash
# swap-model.sh <engine> <gpu> <new_model_id> [extra_args...]
#
# Hot-swaps the model on a running engine container.
# Records wall-clock time so Phase 3.3 can measure swap latency.
#
# Usage:
#   ./infra/scripts/swap-model.sh vllm gpu0 Qwen/Qwen3.5-27B --ctx 32768
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENGINE="${1:?Usage: swap-model.sh <engine> <gpu> <new_model_id> [extra_args...]}"
GPU="${2:?}"
NEW_MODEL="${3:?}"
shift 3

echo "[swap-model] Swapping to ${NEW_MODEL} on ${ENGINE}/${GPU}..."
SWAP_START=$(date +%s%3N)  # milliseconds

"${REPO_ROOT}/infra/scripts/deploy.sh" "${ENGINE}" "${GPU}" "${NEW_MODEL}" "$@"

SWAP_END=$(date +%s%3N)
SWAP_MS=$(( SWAP_END - SWAP_START ))
SWAP_S=$(echo "scale=2; ${SWAP_MS}/1000" | bc)

echo "[swap-model] Swap complete in ${SWAP_S}s (${SWAP_MS}ms)"

# Emit a machine-readable line that bench scripts can grep for
echo "SWAP_LATENCY_MS=${SWAP_MS}"
echo "SWAP_LATENCY_S=${SWAP_S}"
