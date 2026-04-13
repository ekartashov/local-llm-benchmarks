#!/usr/bin/env bash
# deploy.sh <engine> <gpu> <model_id> [extra engine args...]
#
# Brings up a single inference engine container on the specified GPU.
# Sources config/hardware.env for port and GPU mappings.
#
# Examples:
#   ./infra/scripts/deploy.sh vllm gpu0 Qwen/Qwen3.5-35B-A3B-AWQ --ctx 114688
#   ./infra/scripts/deploy.sh sglang gpu1 Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ
#   ./infra/scripts/deploy.sh llamacpp gpu0 "" --model-file Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ENGINE="${1:?Usage: deploy.sh <engine> <gpu> <model_id> [extra_args...]}"
GPU="${2:?Usage: deploy.sh <engine> <gpu> <model_id> [extra_args...]}"
MODEL_ID="${3:?Usage: deploy.sh <engine> <gpu> <model_id> [extra_args...]}"
shift 3
EXTRA_ARGS="${*:-}"

# ── Map gpu alias to compose file and port ────────────────────────────────────
case "${ENGINE}-${GPU}" in
    vllm-gpu0)
        COMPOSE_FILE="${REPO_ROOT}/infra/compose/vllm-gpu0.yaml"
        PORT="${PORT_VLLM_GPU0}"
        ;;
    vllm-gpu1)
        COMPOSE_FILE="${REPO_ROOT}/infra/compose/vllm-gpu1.yaml"
        PORT="${PORT_VLLM_GPU1}"
        ;;
    sglang-gpu0)
        COMPOSE_FILE="${REPO_ROOT}/infra/compose/sglang-gpu0.yaml"
        PORT="${PORT_SGLANG_GPU0}"
        ;;
    sglang-gpu1)
        COMPOSE_FILE="${REPO_ROOT}/infra/compose/sglang-gpu1.yaml"
        PORT="${PORT_SGLANG_GPU1}"
        ;;
    llamacpp-gpu0)
        COMPOSE_FILE="${REPO_ROOT}/infra/compose/llamacpp-gpu0.yaml"
        PORT="${PORT_LLAMACPP_GPU0}"
        ;;
    *)
        echo "[deploy] ERROR: Unknown engine/gpu combo: ${ENGINE}/${GPU}" >&2
        echo "         Valid: vllm-gpu0, vllm-gpu1, sglang-gpu0, sglang-gpu1, llamacpp-gpu0" >&2
        exit 1
        ;;
esac

# ── Parse optional --ctx flag from extra args ─────────────────────────────────
CTX_LEN="${CTX_LEN:-32768}"
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ctx)
            CTX_LEN="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}"
EXTRA_ARGS="${*:-}"

echo "[deploy] Engine=${ENGINE}  GPU=${GPU}  Model=${MODEL_ID}  Port=${PORT}  CTX=${CTX_LEN}"
[[ -n "${EXTRA_ARGS}" ]] && echo "[deploy] Extra args: ${EXTRA_ARGS}"

# ── Tear down any existing container on this port ─────────────────────────────
# Use project name derived from compose file to avoid cross-project collisions.
PROJECT_NAME="bench-${ENGINE}-${GPU}"
podman compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" down --remove-orphans 2>/dev/null || true

# ── Launch ────────────────────────────────────────────────────────────────────
export MODEL_ID CTX_LEN EXTRA_ARGS

podman compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" up -d

HEALTH_URL="http://localhost:${PORT}/health"
echo "[deploy] Waiting for ${HEALTH_URL} ..."
"${REPO_ROOT}/infra/scripts/wait-healthy.sh" "${HEALTH_URL}" 300

echo "[deploy] Done. Endpoint: http://localhost:${PORT}/v1"
