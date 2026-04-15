#!/usr/bin/env bash
# deploy.sh <engine> <gpu> <model_id> [extra engine args...]
#
# Brings up a single inference engine container on the specified GPU using CDI
# GPU passthrough (nvidia.com/gpu=N), which works with rootless podman.
#
# Sources config/hardware.env for port and GPU mappings.
#
# Examples:
#   ./infra/scripts/deploy.sh vllm gpu0 QuantTrio/Qwen3.5-35B-A3B-AWQ --ctx 114688
#   ./infra/scripts/deploy.sh sglang gpu1 Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ
#   ./infra/scripts/deploy.sh llamacpp gpu0 "" --model-file Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
#
# Download a model before first run (token required for gated repos):
#   export HF_TOKEN=hf_...
#   podman run --rm --entrypoint hf -e HF_TOKEN -v /srv/ai/models:/root/.cache/huggingface:z \
#     vllm/vllm-openai:latest download <org/repo>
# Or use: ./infra/scripts/precache-models.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ENGINE="${1:?Usage: deploy.sh <engine> <gpu> <model_id> [extra_args...]}"
GPU="${2:?Usage: deploy.sh <engine> <gpu> <model_id> [extra_args...]}"
MODEL_ID="${3:-}"
shift 3

# ── Parse --ctx and --model-file from extra args ──────────────────────────────
CTX_LEN="${CTX_LEN:-32768}"
MODEL_FILE=""
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ctx)        CTX_LEN="$2";    shift 2 ;;
        --model-file) MODEL_FILE="$2"; shift 2 ;;
        *)            REMAINING_ARGS+=("$1"); shift ;;
    esac
done
EXTRA_ARGS=("${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}")

# ── Map engine+gpu to image, port, GPU ID ─────────────────────────────────────
case "${ENGINE}-${GPU}" in
    vllm-gpu0)
        IMAGE="vllm/vllm-openai:latest"
        PORT="${PORT_VLLM_GPU0}"
        GPU_ID="${GPU_0_ID}"
        ;;
    vllm-gpu1)
        IMAGE="vllm/vllm-openai:latest"
        PORT="${PORT_VLLM_GPU1}"
        GPU_ID="${GPU_1_ID}"
        ;;
    sglang-gpu0)
        IMAGE="lmsysorg/sglang:latest"
        PORT="${PORT_SGLANG_GPU0}"
        GPU_ID="${GPU_0_ID}"
        ;;
    sglang-gpu1)
        IMAGE="lmsysorg/sglang:latest"
        PORT="${PORT_SGLANG_GPU1}"
        GPU_ID="${GPU_1_ID}"
        ;;
    llamacpp-gpu0)
        IMAGE="ghcr.io/ggerganov/llama.cpp:server-cuda"
        PORT="${PORT_LLAMACPP_GPU0}"
        GPU_ID="${GPU_0_ID}"
        ;;
    *)
        echo "[deploy] ERROR: Unknown engine/gpu combo: ${ENGINE}/${GPU}" >&2
        echo "         Valid: vllm-gpu0, vllm-gpu1, sglang-gpu0, sglang-gpu1, llamacpp-gpu0" >&2
        exit 1
        ;;
esac

CONTAINER_NAME="bench-${ENGINE}-${GPU}"
CDI_DEVICE="nvidia.com/gpu=${GPU_ID}"

echo "[deploy] Engine=${ENGINE}  GPU=${GPU}  Model=${MODEL_ID:-${MODEL_FILE}}  Port=${PORT}  CTX=${CTX_LEN}"
echo "[deploy] CDI device: ${CDI_DEVICE}  Container: ${CONTAINER_NAME}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[deploy] Extra args: ${EXTRA_ARGS[*]}"

# ── Tear down any existing container with this name ───────────────────────────
if podman container exists "${CONTAINER_NAME}" 2>/dev/null; then
    echo "[deploy] Removing existing container ${CONTAINER_NAME}..."
    podman stop "${CONTAINER_NAME}" 2>/dev/null || true
    podman rm  "${CONTAINER_NAME}" 2>/dev/null || true
fi

# ── Build engine-specific podman run command ──────────────────────────────────
# Common flags shared by all engines.
COMMON=(
    podman run -d
    --name "${CONTAINER_NAME}"
    --device "${CDI_DEVICE}"
    -e "NVIDIA_VISIBLE_DEVICES=${GPU_ID}"
    -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    -e "HF_HOME=/root/.cache/huggingface"
    -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
    -p "${PORT}:8000"
    --shm-size=4g
    --restart=no
)
# Forward HF credentials if set in the caller's environment.
[[ -n "${HF_TOKEN:-}" ]]             && COMMON+=(-e "HF_TOKEN=${HF_TOKEN}")
[[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]] && COMMON+=(-e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}")

case "${ENGINE}" in
    vllm)
        CMD=(
            "${COMMON[@]}"
            -v "${MODEL_CACHE}:/root/.cache/huggingface:z"
            "${IMAGE}"
            --model "${MODEL_ID}"
            --port 8000
            --gpu-memory-utilization 0.90
            --max-model-len "${CTX_LEN}"
            --enable-auto-tool-choice
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        )
        ;;
    sglang)
        CMD=(
            "${COMMON[@]}"
            -v "${MODEL_CACHE}:/root/.cache/huggingface:z"
            --entrypoint python
            "${IMAGE}"
            -m sglang.launch_server
            --model-path "${MODEL_ID}"
            --port 8000
            --mem-fraction-static 0.88
            --context-length "${CTX_LEN}"
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        )
        ;;
    llamacpp)
        [[ -z "${MODEL_FILE}" ]] && {
            echo "[deploy] ERROR: llamacpp requires --model-file <filename>" >&2; exit 1
        }
        CMD=(
            "${COMMON[@]}"
            -v "${MODEL_CACHE}:/models:ro,z"
            "${IMAGE}"
            --model "/models/${MODEL_FILE}"
            --port 8000
            --host 0.0.0.0
            --n-gpu-layers 99
            --ctx-size "${CTX_LEN}"
            --parallel 4
            --cont-batching
            --flash-attn
            --jinja
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        )
        ;;
esac

echo "[deploy] ${CMD[*]}"
"${CMD[@]}"

# ── Stream container logs to stderr in background ────────────────────────────
echo ""
echo "[deploy] ── Live log stream (Ctrl-C does not stop the container) ──"
echo "[deploy] Follow logs manually: podman logs -f ${CONTAINER_NAME}"
echo ""
podman logs -f "${CONTAINER_NAME}" 2>&1 | sed "s/^/  [${ENGINE}-${GPU}] /" &
LOGS_PID=$!

# ── Wait for the health endpoint ──────────────────────────────────────────────
HEALTH_URL="http://localhost:${PORT}/health"
# 900s timeout: large models (35B AWQ) can take 5–10 min to load from page cache;
# first-run downloads take much longer.
echo "[deploy] Waiting for ${HEALTH_URL} (timeout=900s) ..."
"${REPO_ROOT}/infra/scripts/wait-healthy.sh" "${HEALTH_URL}" 900 "${CONTAINER_NAME}"
WAIT_RC=$?

# Stop the background log stream.
kill "${LOGS_PID}" 2>/dev/null || true
wait "${LOGS_PID}" 2>/dev/null || true

if [[ ${WAIT_RC} -ne 0 ]]; then
    echo "" >&2
    echo "[deploy] ── Last 50 log lines ────────────────────────────────────" >&2
    podman logs --tail 50 "${CONTAINER_NAME}" 2>&1 >&2 || true
    exit "${WAIT_RC}"
fi

echo "[deploy] Done. Endpoint: http://localhost:${PORT}/v1"
