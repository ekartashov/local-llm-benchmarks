#!/usr/bin/env bash
# deploy.sh <engine> <placement> <model_id> [extra engine args...]
#
# Brings up a single inference engine container using CDI GPU passthrough
# (nvidia.com/gpu=N), which works with rootless podman.
#
# <placement> values:
#   gpu0        — single GPU 0 (legacy; use for debugging only)
#   gpu1        — single GPU 1 (legacy; use for debugging only)
#   tp2         — both GPUs, TP=2 (alias for tp2a; default for new queue items)
#   tp2a        — both GPUs, TP=2, port slot A (coder, :30000)
#   tp2b        — both GPUs, TP=2, port slot B (thinker, :30001)
#   tp2c        — both GPUs, TP=2, port slot C (behemoth, :30002)
#   convergence — ikllamacpp only: CPU-only, both GPUs unassigned, port :8002
#
# Named args (consumed by this script; not forwarded to the engine):
#   --ctx N          set context length (default: 32768 for vllm; 16384 for ikllamacpp)
#   --gpu-mem-util F set --gpu-memory-utilization F (default: 0.90)
#   --model-file F   llamacpp only: GGUF filename inside MODEL_CACHE
#
# Environment variables forwarded to the container if set in the caller:
#   HF_TOKEN, HUGGING_FACE_HUB_TOKEN
#   VLLM_SERVER_DEV_MODE   — set to 1 to enable Sleep Mode API (/sleep, /wake_up)
#   VLLM_ALLOW_LONG_MAX_MODEL_LEN  — set to 1 for models with >32k native context
#
# Examples:
#   ./infra/scripts/deploy.sh vllm tp2 QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ \
#       --gpu-mem-util 0.40 --tool-call-parser qwen3_coder --reasoning-parser qwen3
#
#   VLLM_SERVER_DEV_MODE=1 \
#   ./infra/scripts/deploy.sh vllm tp2 QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ \
#       --gpu-mem-util 0.85 --tool-call-parser qwen3_coder --reasoning-parser qwen3
#
#   ./infra/scripts/deploy.sh vllm tp2b QuantTrio/Qwen3.5-27B-AWQ \
#       --gpu-mem-util 0.40 --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
#       --max-num-seqs 1
#
#   ./infra/scripts/deploy.sh sglang gpu0 Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ
#
#   ./infra/scripts/deploy.sh llamacpp gpu0 "" \
#       --model-file Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

ENGINE="${1:?Usage: deploy.sh <engine> <placement> <model_id> [extra_args...]}"
PLACEMENT="${2:?Usage: deploy.sh <engine> <placement> <model_id> [extra_args...]}"
MODEL_ID="${3:-}"
shift 3

# ── Parse named args out of extra args ────────────────────────────────────────
CTX_LEN="${CTX_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MODEL_FILE=""
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ctx)           CTX_LEN="$2";      shift 2 ;;
        --gpu-mem-util)  GPU_MEM_UTIL="$2"; shift 2 ;;
        --model-file)    MODEL_FILE="$2";   shift 2 ;;
        *)               REMAINING_ARGS+=("$1"); shift ;;
    esac
done
EXTRA_ARGS=("${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}")

# ── Map engine+placement to image, port, GPU list, TP size ────────────────────
TP_SIZE=1  # overridden for tp2* placements
case "${ENGINE}-${PLACEMENT}" in
    vllm-gpu0)
        IMAGE="${BENCH_IMAGE:-vllm/vllm-openai:latest}"
        PORT="${PORT_VLLM_GPU0}"
        GPU_IDS=("${GPU_0_ID}")
        ;;
    vllm-gpu1)
        IMAGE="${BENCH_IMAGE:-vllm/vllm-openai:latest}"
        PORT="${PORT_VLLM_GPU1}"
        GPU_IDS=("${GPU_1_ID}")
        ;;
    vllm-tp2|vllm-tp2a)
        IMAGE="${BENCH_IMAGE:-vllm/vllm-openai:latest}"
        PORT="${PORT_VLLM_TP2_A}"
        GPU_IDS=("${GPU_0_ID}" "${GPU_1_ID}")
        TP_SIZE=2
        ;;
    vllm-tp2b)
        IMAGE="${BENCH_IMAGE:-vllm/vllm-openai:latest}"
        PORT="${PORT_VLLM_TP2_B}"
        GPU_IDS=("${GPU_0_ID}" "${GPU_1_ID}")
        TP_SIZE=2
        ;;
    vllm-tp2c)
        IMAGE="${BENCH_IMAGE:-vllm/vllm-openai:latest}"
        PORT="${PORT_VLLM_TP2_C}"
        GPU_IDS=("${GPU_0_ID}" "${GPU_1_ID}")
        TP_SIZE=2
        ;;
    sglang-gpu0)
        IMAGE="lmsysorg/sglang:latest"
        PORT="${PORT_SGLANG_GPU0}"
        GPU_IDS=("${GPU_0_ID}")
        ;;
    sglang-gpu1)
        IMAGE="lmsysorg/sglang:latest"
        PORT="${PORT_SGLANG_GPU1}"
        GPU_IDS=("${GPU_1_ID}")
        ;;
    llamacpp-gpu0)
        IMAGE="ghcr.io/ggerganov/llama.cpp:server-cuda"
        PORT="${PORT_LLAMACPP_GPU0}"
        GPU_IDS=("${GPU_0_ID}")
        ;;
    ikllamacpp-convergence)
        IMAGE="${IK_LLAMA_IMAGE:-local-ik-llama:runtime}"
        PORT="${PORT_CONVERGENCE}"
        GPU_IDS=()  # CPU-only: no GPU device passthrough; runs in parallel with Arclight
        CTX_LEN="${CTX_LEN:-16384}"
        ;;
    *)
        echo "[deploy] ERROR: Unknown engine/placement combo: ${ENGINE}/${PLACEMENT}" >&2
        echo "         Valid engine values: vllm, sglang, llamacpp, ikllamacpp" >&2
        echo "         Valid placement values: gpu0, gpu1, tp2, tp2a, tp2b, tp2c, convergence" >&2
        echo "         (tp2* = vllm-only; convergence = ikllamacpp-only)" >&2
        exit 1
        ;;
esac

# ── Build CDI device list and NVIDIA_VISIBLE_DEVICES ──────────────────────────
CONTAINER_NAME="bench-${ENGINE}-${PLACEMENT}"
CDI_DEVICE_ARGS=()
NVIDIA_VISIBLE_STR=""
for _gid in "${GPU_IDS[@]}"; do
    CDI_DEVICE_ARGS+=("--device" "nvidia.com/gpu=${_gid}")
done
[[ ${#GPU_IDS[@]} -gt 0 ]] && NVIDIA_VISIBLE_STR=$(IFS=,; echo "${GPU_IDS[*]}")

echo "[deploy] Engine=${ENGINE}  Placement=${PLACEMENT}  TP=${TP_SIZE}"
echo "[deploy] Model=${MODEL_ID:-${MODEL_FILE:-}}  Port=${PORT}  CTX=${CTX_LEN}  gpu-mem-util=${GPU_MEM_UTIL}"
echo "[deploy] GPUs=${NVIDIA_VISIBLE_STR:-none (CPU-only)}  Container=${CONTAINER_NAME}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[deploy] Extra engine args: ${EXTRA_ARGS[*]}"

# ── Tear down any existing container with this name ───────────────────────────
if podman container exists "${CONTAINER_NAME}" 2>/dev/null; then
    echo "[deploy] Removing existing container ${CONTAINER_NAME}..."
    podman stop "${CONTAINER_NAME}" 2>/dev/null || true
    podman rm   "${CONTAINER_NAME}" 2>/dev/null || true
fi

# ── Build engine-specific podman run command ──────────────────────────────────
COMMON=(
    podman run -d
    --name "${CONTAINER_NAME}"
    "${CDI_DEVICE_ARGS[@]+"${CDI_DEVICE_ARGS[@]}"}"
    -e "HF_HOME=/root/.cache/huggingface"
    -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
    -p "${PORT}:8000"
    --shm-size=4g
    --restart=no
)
# Only inject NVIDIA env vars when GPU devices are actually assigned.
# CPU-only placements (ikllamacpp convergence) must NOT set NVIDIA_VISIBLE_DEVICES —
# the runtime libs are still needed (binary links against them) but no GPU is used.
if [[ -n "${NVIDIA_VISIBLE_STR}" ]]; then
    COMMON+=(-e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_STR}")
    COMMON+=(-e "NVIDIA_DRIVER_CAPABILITIES=compute,utility")
fi

# Forward credentials and well-known vLLM env vars if set by caller.
[[ -n "${HF_TOKEN:-}" ]]                        && COMMON+=(-e "HF_TOKEN=${HF_TOKEN}")
[[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]          && COMMON+=(-e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}")
[[ -n "${VLLM_SERVER_DEV_MODE:-}" ]]            && COMMON+=(-e "VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE}")
[[ -n "${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-}" ]]   && COMMON+=(-e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN}")
[[ -n "${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-}" ]] && COMMON+=(-e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS}")
[[ -n "${VLLM_V1_ENABLED:-}" ]]                  && COMMON+=(-e "VLLM_V1_ENABLED=${VLLM_V1_ENABLED}")
[[ -n "${VLLM_USE_V1:-}" ]]                      && COMMON+=(-e "VLLM_USE_V1=${VLLM_USE_V1}")
[[ -n "${VLLM_V1:-}" ]]                          && COMMON+=(-e "VLLM_V1=${VLLM_V1}")
[[ -n "${VLLM_USE_V1_ENGINE:-}" ]]               && COMMON+=(-e "VLLM_USE_V1_ENGINE=${VLLM_USE_V1_ENGINE}")
[[ -n "${VLLM_ENGINE_ITERATOR_SOURCE:-}" ]]      && COMMON+=(-e "VLLM_ENGINE_ITERATOR_SOURCE=${VLLM_ENGINE_ITERATOR_SOURCE}")
[[ -n "${VLLM_ENABLE_V1_MULTIPROCESSING:-}" ]]  && COMMON+=(-e "VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING}")
[[ -n "${ENABLE_KVCACHED:-}" ]]               && COMMON+=(-e "ENABLE_KVCACHED=${ENABLE_KVCACHED}")
[[ -n "${KVCACHED_AUTOPATCH:-}" ]]            && COMMON+=(-e "KVCACHED_AUTOPATCH=${KVCACHED_AUTOPATCH}")

case "${ENGINE}" in
    vllm)
        # Only add --enable-auto-tool-choice when --tool-call-parser is present.
        # vLLM 0.19 raises TypeError if --enable-auto-tool-choice is set without a parser.
        VLLM_TOOL_ARGS=()
        for _a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
            [[ "$_a" == "--tool-call-parser" ]] && VLLM_TOOL_ARGS=(--enable-auto-tool-choice) && break
        done

        # Inject --tensor-parallel-size for TP=2 placements.
        VLLM_TP_ARGS=()
        [[ "${TP_SIZE}" -gt 1 ]] && VLLM_TP_ARGS=(--tensor-parallel-size "${TP_SIZE}")

        CMD=(
            "${COMMON[@]}"
            -v "${MODEL_CACHE}:/root/.cache/huggingface:z"
            "${IMAGE}"
            --model "${MODEL_ID}"
            --port 8000
            --gpu-memory-utilization "${GPU_MEM_UTIL}"
            --max-model-len "${CTX_LEN}"
            "${VLLM_TP_ARGS[@]+"${VLLM_TP_ARGS[@]}"}"
            "${VLLM_TOOL_ARGS[@]+"${VLLM_TOOL_ARGS[@]}"}"
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
    ikllamacpp)
        # Resolve model path: explicit MODEL_ID arg, else CONVERGENCE_MODEL env var.
        _IK_MODEL="${MODEL_ID:-${CONVERGENCE_MODEL:-}}"
        [[ -z "${_IK_MODEL}" ]] && {
            echo "[deploy] ERROR: ikllamacpp requires model path as 3rd arg or CONVERGENCE_MODEL env var" >&2
            echo "  Example: deploy.sh ikllamacpp convergence \"\${CONVERGENCE_MODEL}\"" >&2
            exit 1
        }
        _IK_BUILD="${IK_LLAMA_BUILD_DIR:-/srv/ai/projects/ik_llama.cpp/build}"
        [[ ! -x "${_IK_BUILD}/bin/llama-server" ]] && {
            echo "[deploy] ERROR: llama-server not found at ${_IK_BUILD}/bin/llama-server" >&2
            echo "  Run: ./infra/scripts/build-ik-llama.sh" >&2
            exit 1
        }
        CMD=(
            "${COMMON[@]}"
            -v "${_IK_BUILD}:/app/build:ro,z"
            -v "${MODEL_CACHE}:/models:ro,z"
            --entrypoint "/app/build/bin/llama-server"
            "${IMAGE}"
            --model "/models/${_IK_MODEL}"
            --port 8000
            --host 0.0.0.0
            -ngl "${IKLLAMACPP_NGL:-0}"
            --no-mmap
            -b 4096 -ub 2048
            -t "${NTHREADS:-$(nproc)}"
            -c "${CTX_LEN}"
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
podman logs -f "${CONTAINER_NAME}" 2>&1 | sed "s/^/  [${ENGINE}-${PLACEMENT}] /" &
LOGS_PID=$!

# ── Wait for the health endpoint ──────────────────────────────────────────────
HEALTH_URL="http://localhost:${PORT}/health"
# 900s timeout: large models (35B AWQ) can take 5–10 min to load from page cache;
# first-run downloads take much longer. TP=2 startup is comparable to single-GPU.
echo "[deploy] Waiting for ${HEALTH_URL} (timeout=900s) ..."
"${REPO_ROOT}/infra/scripts/wait-healthy.sh" "${HEALTH_URL}" 900 "${CONTAINER_NAME}"
WAIT_RC=$?

kill "${LOGS_PID}" 2>/dev/null || true
# Do NOT wait on LOGS_PID. $! is only the PID of the `sed` process at the end of the pipe,
# so `kill` stops sed. But `podman logs -f` is asleep waiting for new logs and won't get
# SIGPIPE until the next log line is produced. If we `wait` here, bash waits for the entire
# pipeline to close, which causes a silent hang!

if [[ ${WAIT_RC} -ne 0 ]]; then
    echo "" >&2
    echo "[deploy] ── Last 50 log lines ────────────────────────────────────" >&2
    podman logs --tail 50 "${CONTAINER_NAME}" 2>&1 >&2 || true
    exit "${WAIT_RC}"
fi

echo "[deploy] Done. Endpoint: http://localhost:${PORT}/v1"
