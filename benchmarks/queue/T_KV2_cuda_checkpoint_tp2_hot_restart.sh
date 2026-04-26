#!/usr/bin/env bash
# benchmarks/queue/T_KV2_cuda_checkpoint_tp2_hot_restart.sh
#
# T_KV2 — CUDA checkpoint + CRIU hot-restart for TP=2 vLLM (Extended Arclight)
#
# Q: Does checkpoint/restore reduce Extended Arclight mode-switch time from
#    170–300s (cold CUDA graph compilation) to < 30s (target: ~5s)?
#
# Why high priority: unlocks fast coder big-context swaps for interactive use.
#
# ── Checkpoint strategy ───────────────────────────────────────────────────────
#
# Path A — podman container checkpoint (CRIU, rootless-safe):
#   • Without cuda-checkpoint CRIU hooks: saves CPU process only.
#     GPU weights lost on restore → vLLM reloads from VRAM; no graph recompile
#     if CUDA context survives, but graphs ARE recompiled if context is lost.
#   • With CRIU hooks installed: saves CPU + GPU state (weights + compiled graphs).
#     Restore skips both weight reload AND graph compilation. Expected: ~5s.
#
# Path B — cuda-checkpoint standalone (fallback if Path A totally fails):
#   Checkpoints GPU context only. Not a full process checkpoint; provides
#   partial data for diagnosis and GPU-only timing.
#
# ── Pass criterion ────────────────────────────────────────────────────────────
#   restore time (median of ${REPS} reps) < 30s  → PASS
#   restore time ≥ 30s but < cold start           → PARTIAL (see summary)
#   restore time ≥ cold start                     → FAIL
#
# ── One-time host prerequisites ───────────────────────────────────────────────
#   sudo apt install criu
#
#   # cuda-checkpoint binary + CRIU hooks (enables GPU memory in checkpoint):
#   git clone https://github.com/NVIDIA/cuda-checkpoint /srv/ai/tools/cuda-checkpoint
#   cd /srv/ai/tools/cuda-checkpoint && make
#   sudo make install     # installs to /usr/local/bin/ and /usr/lib/criu/
#
#   mkdir -p /srv/ai/checkpoints/coder-tp2
#
# ── Options ───────────────────────────────────────────────────────────────────
#   --dry-run        Print commands, do not execute
#   --reps N         Restore reps for timing stability (default: 3)
#   --ctx N          Context length for TP=2 (default: 65536 = Extended Arclight)
#   --gpu-mem F      GPU memory utilization (default: 0.90)
#   --skip-cold      Skip cold-start timing; assume container is already warm

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# ── Parse flags ───────────────────────────────────────────────────────────────
DRY_RUN=0
REPS=3
CTX_LEN=65536
GPU_MEM_UTIL=0.90
SKIP_COLD=0
ROOTFUL=0
PODMAN="podman"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --reps)      REPS="${2:?--reps needs value}"; shift 2 ;;
        --ctx)       CTX_LEN="${2:?--ctx needs value}"; shift 2 ;;
        --gpu-mem)   GPU_MEM_UTIL="${2:?--gpu-mem needs value}"; shift 2 ;;
        --skip-cold) SKIP_COLD=1; shift ;;
        --rootful)   ROOTFUL=1; PODMAN="sudo podman"; shift ;;
        *) echo "[T_KV2] Unknown option: $1" >&2; exit 1 ;;
    esac
done

run()     { [[ "${DRY_RUN}" -eq 1 ]] && { echo "[dry-run] $*"; return 0; }; "$@"; }
section() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo " $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

# ── Constants ─────────────────────────────────────────────────────────────────
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PODMAN_CMD=(${PODMAN:-podman})
RESULTS_DIR="${REPO_ROOT}/results/T_KV2_cuda_checkpoint_tp2_hot_restart_${TIMESTAMP}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/srv/ai/checkpoints/coder-tp2}"
CHECKPOINT_ARCHIVE="${CHECKPOINT_DIR}/checkpoint.tar"
CONTAINER_NAME="bench-vllm-tp2a"
MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
PORT="${PORT_VLLM_TP2_A}"         # 30000
HEALTH_URL="http://localhost:${PORT}/health"

[[ "${DRY_RUN}" -eq 0 ]] && mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

# ── Prerequisite detection ────────────────────────────────────────────────────
section "PREREQUISITES"

DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "unknown")
DRIVER_MAJOR=$(echo "${DRIVER_VER}" | cut -d. -f1)
echo "[T_KV2] NVIDIA driver: ${DRIVER_VER}"
if [[ "${DRIVER_MAJOR}" =~ ^[0-9]+$ ]] && [[ "${DRIVER_MAJOR}" -lt 570 ]]; then
    echo "[T_KV2] WARNING: driver ${DRIVER_VER} < 570 — CUDA checkpoint requires ≥570"
fi

# CRIU
HAVE_CRIU=0
for _p in criu /usr/sbin/criu /sbin/criu; do
    if command -v "${_p}" &>/dev/null; then
        echo "[T_KV2] CRIU: $( "${_p}" --version 2>/dev/null | head -1)"
        HAVE_CRIU=1
        break
    fi
done
if [[ "${HAVE_CRIU}" -eq 0 ]]; then
    echo "[T_KV2] MISSING: criu — install with: sudo apt install criu"
    echo "[T_KV2]   podman checkpoint requires CRIU."
fi

# cuda-checkpoint binary
CUDA_CKPT_BIN=""
for _p in \
    /usr/local/bin/cuda-checkpoint \
    /usr/bin/cuda-checkpoint \
    /srv/ai/tools/cuda-checkpoint/build/cuda-checkpoint; do
    [[ -x "${_p}" ]] && { CUDA_CKPT_BIN="${_p}"; break; }
done
if [[ -n "${CUDA_CKPT_BIN}" ]]; then
    echo "[T_KV2] cuda-checkpoint: ${CUDA_CKPT_BIN}"
    HAVE_CUDA_CKPT=1
else
    echo "[T_KV2] INFO: cuda-checkpoint binary not found."
    echo "[T_KV2]   Build: cd /srv/ai/tools/cuda-checkpoint && make && sudo make install"
    HAVE_CUDA_CKPT=0
fi

# cuda-checkpoint CRIU hooks (needed for GPU memory in checkpoint)
HAVE_CUDA_HOOK=0
for _p in /usr/lib/criu/cuda-checkpoint-hook.so /usr/local/lib/criu/cuda-checkpoint-hook.so; do
    if [[ -f "${_p}" ]]; then
        HAVE_CUDA_HOOK=1
        echo "[T_KV2] CRIU GPU hook: ${_p} (GPU memory will be included)"
        break
    fi
done
if [[ "${HAVE_CUDA_HOOK}" -eq 0 ]]; then
    echo "[T_KV2] INFO: cuda-checkpoint CRIU hook NOT found."
    echo "[T_KV2]   Without hook: GPU memory excluded from checkpoint."
    echo "[T_KV2]   Restore will reload weights + recompile graphs (slow path)."
    echo "[T_KV2]   This run measures the 'no-GPU' baseline."
fi

# criu check (verify kernel support)
if [[ "${HAVE_CRIU}" -eq 1 ]] && [[ "${DRY_RUN}" -eq 0 ]]; then
    echo "[T_KV2] Running: criu check (kernel feature audit)..."
    # Find the binary path again to avoid "command not found" in sudo subshells
    CRIU_PATH=""
    for _p in criu /usr/sbin/criu /sbin/criu; do
        command -v "${_p}" &>/dev/null && { CRIU_PATH="${_p}"; break; }
    done
    if [[ "${ROOTFUL}" -eq 1 ]]; then
        sudo "${CRIU_PATH}" check 2>&1 | tee "${RESULTS_DIR}/criu_check.log" || true
    else
        "${CRIU_PATH}" check 2>&1 | tee "${RESULTS_DIR}/criu_check.log" || {
            echo "[T_KV2] WARNING: criu check failed — some features may be missing."
            echo "[T_KV2]   Run 'criu check --full' for details."
        }
    fi
fi

# newuidmap / newgidmap (required for rootless podman checkpoint)
for _bin in newuidmap newgidmap; do
    command -v "${_bin}" &>/dev/null \
        && echo "[T_KV2] ${_bin}: $(command -v "${_bin}")" \
        || echo "[T_KV2] WARNING: ${_bin} not found — install uidmap package"
done

echo ""
echo "[T_KV2] Config: model=${MODEL} ctx=${CTX_LEN} gpu-mem=${GPU_MEM_UTIL} reps=${REPS}"
echo "[T_KV2] Results dir: ${RESULTS_DIR}"
echo "[T_KV2] Checkpoint archive: ${CHECKPOINT_ARCHIVE}"

# ── Helper: wait for health endpoint ─────────────────────────────────────────
# Returns elapsed milliseconds on stdout; returns 1 on timeout.
wait_healthy() {
    local url="$1"
    local timeout_s="${2:-900}"
    local T_START
    T_START=$(date +%s%3N)
    echo "[T_KV2] Waiting for ${url} (timeout=${timeout_s}s)..." >&2
    for _ in $(seq 1 $(( timeout_s * 2 ))); do
        if curl -sf --max-time 2 "${url}" &>/dev/null; then
            local T_NOW ELAPSED
            T_NOW=$(date +%s%3N)
            ELAPSED=$(( T_NOW - T_START ))
            echo "[T_KV2] Health OK in ${ELAPSED}ms ($(( ELAPSED / 1000 ))s)" >&2
            echo "${ELAPSED}"
            return 0
        fi
        # Bail early if container died
        if ! "${PODMAN_CMD[@]}" container exists "${CONTAINER_NAME}" 2>/dev/null; then
            echo "[T_KV2] ERROR: container ${CONTAINER_NAME} disappeared" >&2
            return 1
        fi
        sleep 0.5
    done
    echo "[T_KV2] TIMEOUT after ${timeout_s}s waiting for ${url}" >&2
    return 1
}

# ── Helper: single inference request (returns TPS) ───────────────────────────
run_inference() {
    local label="$1"
    local out="${RESULTS_DIR}/inference_${label}.json"
    local T_START T_END ELAPSED COMP_TOKENS TPS
    T_START=$(date +%s%3N)
    curl -sf "http://localhost:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"prompt\": \"Write a hello world function in Python.\",
            \"max_tokens\": 64,
            \"temperature\": 0
        }" > "${out}" 2>&1 || { echo "N/A"; return; }
    T_END=$(date +%s%3N)
    ELAPSED=$(( T_END - T_START ))
    TPS=$(python3 -c "
import json, sys
try:
    d = json.load(open('${out}'))
    toks = d.get('usage', {}).get('completion_tokens', 0)
    ms = ${ELAPSED}
    print(f'{toks/(ms/1000):.1f}' if ms > 0 and toks > 0 else 'N/A')
except Exception as e:
    print('N/A')
" 2>/dev/null || echo "N/A")
    echo "[T_KV2] Inference (${label}): ${TPS} t/s (${ELAPSED}ms total)" >&2
    echo "${TPS}"
}

# ── Helper: stop and remove container ────────────────────────────────────────
stop_container() {
    if "${PODMAN_CMD[@]}" container exists "${CONTAINER_NAME}" 2>/dev/null; then
        echo "[T_KV2] Stopping/removing ${CONTAINER_NAME}..."
        "${PODMAN_CMD[@]}" stop "${CONTAINER_NAME}" 2>/dev/null || true
        "${PODMAN_CMD[@]}" rm   "${CONTAINER_NAME}" 2>/dev/null || true
    fi
}

# ── Part 1: Cold start (baseline) ────────────────────────────────────────────
COLD_START_MS=0
COLD_TPS="N/A"

if [[ "${SKIP_COLD}" -eq 0 ]]; then
    section "PART 1: COLD START (BASELINE)"
    run stop_container

    echo "[T_KV2] Starting TP=2 coder (Extended Arclight config)..."
    echo "[T_KV2] CUDA graph compilation expected: 170–300s"
    T_COLD=$(date +%s%3N)

    if [[ "${DRY_RUN}" -eq 0 ]]; then
        T_START=$(date +%s%3N)
        PODMAN="${PODMAN}" \
        VLLM_SERVER_DEV_MODE=1 \
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
        VLLM_V1_ENABLED=0 VLLM_USE_V1=0 VLLM_V1=0 VLLM_USE_V1_ENGINE=0 \
        "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" \
            --gpu-mem-util "${GPU_MEM_UTIL}" \
            --ctx "${CTX_LEN}" \
            --kv-cache-dtype fp8 \
            --max-num-seqs 4 \
            --enable-sleep-mode \
            --tool-call-parser qwen3_coder \
            --reasoning-parser qwen3
        T_END=$(date +%s%3N)
        COLD_START_MS=$(( T_END - T_START ))

        echo "[T_KV2] Cold start: ${COLD_START_MS}ms ($(( COLD_START_MS / 1000 ))s)"

        # Warmup: runs a request to ensure CUDA graphs are captured for common shapes
        echo "[T_KV2] Warmup inference (triggers deferred CUDA graph compilation)..."
        run_inference "warmup" > /dev/null || true
        sleep 3
        COLD_TPS=$(run_inference "cold_baseline")
    else
        echo "[dry-run] VLLM_SERVER_DEV_MODE=1 deploy.sh vllm tp2a ${MODEL} ..."
        COLD_START_MS=180000  # placeholder for dry-run arithmetic
    fi
else
    echo "[T_KV2] --skip-cold: verifying container is healthy at ${HEALTH_URL}..."
    curl -sf --max-time 5 "${HEALTH_URL}" &>/dev/null || {
        echo "[T_KV2] ERROR: no container responding at ${HEALTH_URL}" >&2
        echo "[T_KV2]   Remove --skip-cold to do a full cold start." >&2
        exit 1
    }
    echo "[T_KV2] Container is healthy. Proceeding to checkpoint."
    COLD_TPS=$(run_inference "pre_checkpoint_baseline" 2>/dev/null || echo "N/A")
fi

# ── Part 2: Checkpoint ────────────────────────────────────────────────────────
section "PART 2: CHECKPOINT"

CHECKPOINT_PATH="none"
CHECKPOINT_MS=0
CHECKPOINT_SIZE="N/A"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    rm -f "${CHECKPOINT_ARCHIVE}"

    if [[ "${HAVE_CRIU}" -eq 0 ]]; then
        echo "[T_KV2] CRIU not installed — skipping podman checkpoint." >&2
        echo "[T_KV2] Install criu and re-run." >&2
    else
        echo "[T_KV2] Checkpointing ${CONTAINER_NAME} → ${CHECKPOINT_ARCHIVE} ..."
        echo "[T_KV2] CRIU hook active: $( [[ "${HAVE_CUDA_HOOK}" -eq 1 ]] && echo "YES (GPU memory included)" || echo "NO (CPU state only)" )"

        T_CKPT=$(date +%s%3N)

        # --export writes a portable tar archive (allows multiple restores)
        # --leave-running=false (default) stops the container after checkpoint
        if "${PODMAN_CMD[@]}" container checkpoint \
                --export "${CHECKPOINT_ARCHIVE}" \
                "${CONTAINER_NAME}" \
                > "${RESULTS_DIR}/checkpoint.log" 2>&1; then

            T_CKPT_END=$(date +%s%3N)
            CHECKPOINT_MS=$(( T_CKPT_END - T_CKPT ))
            CHECKPOINT_PATH="podman-checkpoint"
            CHECKPOINT_SIZE=$(du -sh "${CHECKPOINT_ARCHIVE}" 2>/dev/null | awk '{print $1}' || echo "N/A")
            CHECKPOINT_BYTES=$(du -sb "${CHECKPOINT_ARCHIVE}" 2>/dev/null | awk '{print $1}' || echo "0")

            echo "[T_KV2] Checkpoint complete: ${CHECKPOINT_MS}ms"
            echo "[T_KV2] Archive size: ${CHECKPOINT_SIZE}"

            # Heuristic: if archive > 20GB, GPU weights are likely included
            if [[ "${CHECKPOINT_BYTES}" -gt 20000000000 ]]; then
                echo "[T_KV2] Archive > 20GB → GPU memory likely INCLUDED (CRIU hook was active)."
            else
                echo "[T_KV2] Archive < 20GB → GPU memory NOT included (CPU state only)."
                echo "[T_KV2]   Restore will skip CUDA graph recompilation IF the CUDA context"
                echo "[T_KV2]   survives, but weights must reload from VRAM → expect slow restore."
            fi
        else
            echo "[T_KV2] podman container checkpoint FAILED." >&2
            tail -30 "${RESULTS_DIR}/checkpoint.log" >&2 || true
            echo "" >&2
            echo "[T_KV2] Rootless checkpoint failure checklist:" >&2
            echo "  1. criu check --full         (verify kernel features)" >&2
            echo "  2. sysctl kernel.ns_last_pid (user namespace support)" >&2
            echo "  3. ls -la /usr/bin/newuidmap (must exist + setuid)" >&2
            echo "  4. Try: sudo podman container checkpoint ... (root test)" >&2
            echo "" >&2
            CHECKPOINT_PATH="podman-checkpoint-failed"
        fi
    fi

    # Path B: try cuda-checkpoint standalone if Path A failed or as supplementary
    if [[ "${HAVE_CUDA_CKPT}" -eq 1 ]] && [[ "${CHECKPOINT_PATH}" != "podman-checkpoint" ]]; then
        echo "[T_KV2] Attempting Path B: cuda-checkpoint standalone (GPU state only)..."

        # Find vLLM API server PID on host (rootless container processes are visible in host ps)
        VLLM_HOST_PID=$(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null | head -1 || echo "")

        if [[ -z "${VLLM_HOST_PID}" ]]; then
            echo "[T_KV2] Path B: could not find vLLM host PID via pgrep." >&2
            echo "[T_KV2]   Try: ps aux | grep vllm.entrypoints" >&2
            echo "[T_KV2]   Or check namespace PID: podman top ${CONTAINER_NAME}" >&2
            CHECKPOINT_PATH="all-failed"
        else
            echo "[T_KV2] vLLM host PID: ${VLLM_HOST_PID}"
            echo "[T_KV2] NOTE: cuda-checkpoint standalone saves GPU state only." >&2
            echo "[T_KV2]       CPU process state is NOT saved — restore is incomplete." >&2
            echo "[T_KV2]       This path is diagnostic only (no functional restore)." >&2

            GPU_CKPT_DIR="${CHECKPOINT_DIR}/gpu_only"
            mkdir -p "${GPU_CKPT_DIR}"
            T_CKPT=$(date +%s%3N)

            if "${CUDA_CKPT_BIN}" \
                    --pid "${VLLM_HOST_PID}" \
                    --tree \
                    --action dump \
                    --dir "${GPU_CKPT_DIR}" \
                    > "${RESULTS_DIR}/cuda_checkpoint_standalone.log" 2>&1; then
                T_CKPT_END=$(date +%s%3N)
                echo "[T_KV2] cuda-checkpoint dump: $(( T_CKPT_END - T_CKPT ))ms"
                GPU_CKPT_SIZE=$(du -sh "${GPU_CKPT_DIR}" 2>/dev/null | awk '{print $1}' || echo "N/A")
                echo "[T_KV2] GPU checkpoint size: ${GPU_CKPT_SIZE}"
                CHECKPOINT_PATH="cuda-ckpt-standalone-only"
            else
                echo "[T_KV2] cuda-checkpoint dump also failed." >&2
                tail -10 "${RESULTS_DIR}/cuda_checkpoint_standalone.log" >&2 || true
                CHECKPOINT_PATH="all-failed"
            fi
        fi
    fi
else
    echo "[dry-run] Would: podman container checkpoint --export ${CHECKPOINT_ARCHIVE} ${CONTAINER_NAME}"
    CHECKPOINT_PATH="podman-checkpoint"
fi

# ── Part 3: Restore × REPS ───────────────────────────────────────────────────
declare -a RESTORE_MS=()
RESTORE_TPS="N/A"
RESTORE_VERDICT="INCONCLUSIVE"

if [[ "${CHECKPOINT_PATH}" == "podman-checkpoint" ]]; then
    section "PART 3: RESTORE (${REPS} reps)"

    if [[ "${DRY_RUN}" -eq 0 ]]; then
        for rep in $(seq 1 "${REPS}"); do
            echo "[T_KV2] --- Restore rep ${rep}/${REPS} ---"

            # Container must not exist before restore from archive
            stop_container
            sleep 2

            T_RESTORE=$(date +%s%3N)

            if "${PODMAN_CMD[@]}" container restore \
                    --import "${CHECKPOINT_ARCHIVE}" \
                    > "${RESULTS_DIR}/restore_rep${rep}.log" 2>&1; then

                R_MS=$(wait_healthy "${HEALTH_URL}" 600) || {
                    echo "[T_KV2] Rep ${rep}: TIMEOUT waiting for health" >&2
                    podman logs --tail 30 "${CONTAINER_NAME}" >> "${RESULTS_DIR}/restore_rep${rep}.log" 2>&1 || true
                    RESTORE_MS+=("TIMEOUT")
                    stop_container
                    continue
                }
                RESTORE_MS+=("${R_MS}")
                echo "[T_KV2] Rep ${rep}: ready in ${R_MS}ms ($(( R_MS / 1000 ))s)"

                # On the last rep, run a verification inference
                if [[ "${rep}" -eq "${REPS}" ]]; then
                    RESTORE_TPS=$(run_inference "restore_verify_rep${rep}" 2>/dev/null || echo "N/A")
                fi
            else
                echo "[T_KV2] Rep ${rep}: podman restore FAILED." >&2
                tail -20 "${RESULTS_DIR}/restore_rep${rep}.log" >&2 || true
                RESTORE_MS+=("FAIL")
            fi

            # Kill before next rep (next rep re-imports from archive)
            [[ "${rep}" -lt "${REPS}" ]] && { stop_container; sleep 3; }
        done
    else
        echo "[dry-run] Would restore ${REPS} times from ${CHECKPOINT_ARCHIVE}"
        for rep in $(seq 1 "${REPS}"); do
            echo "[dry-run]   rep ${rep}: podman container restore --import ${CHECKPOINT_ARCHIVE}"
        done
    fi
elif [[ "${CHECKPOINT_PATH}" == "cuda-ckpt-standalone-only" ]]; then
    section "PART 3: RESTORE — SKIPPED (GPU-only checkpoint)"
    echo "[T_KV2] cuda-checkpoint standalone saved GPU state only."
    echo "[T_KV2] Full restore requires CRIU + cuda-checkpoint CRIU hooks together."
    echo "[T_KV2] No restore timing possible from this checkpoint type."
    echo "[T_KV2] HAND-BACK: install CRIU + cuda-checkpoint hooks, then re-run."
elif [[ "${CHECKPOINT_PATH}" == "all-failed" ]] || [[ "${CHECKPOINT_PATH}" =~ failed$ ]]; then
    section "PART 3: RESTORE — SKIPPED (checkpoint failed)"
    echo "[T_KV2] No checkpoint to restore from. See Part 2 errors above."
fi

# ── Cleanup ────────────────────────────────────────────────────────────────────
[[ "${DRY_RUN}" -eq 0 ]] && stop_container

# ── Compute results ────────────────────────────────────────────────────────────
section "RESULTS"

RESTORE_VALID=()
for _v in "${RESTORE_MS[@]+"${RESTORE_MS[@]}"}"; do
    [[ "${_v}" =~ ^[0-9]+$ ]] && RESTORE_VALID+=("${_v}")
done

RESTORE_MEDIAN_MS=0
RESTORE_MEDIAN_S="N/A"
SPEEDUP="N/A"

if [[ ${#RESTORE_VALID[@]} -gt 0 ]]; then
    IFS=$'\n' _SORTED=($(sort -n <<<"${RESTORE_VALID[*]}")); unset IFS
    _MID=$(( ${#_SORTED[@]} / 2 ))
    RESTORE_MEDIAN_MS="${_SORTED[${_MID}]}"
    RESTORE_MEDIAN_S=$(python3 -c "print(f'{${RESTORE_MEDIAN_MS}/1000:.1f}')")
    [[ "${COLD_START_MS}" -gt 0 ]] && \
        SPEEDUP=$(python3 -c "print(f'{${COLD_START_MS}/${RESTORE_MEDIAN_MS}:.1f}x')")

    if   [[ "${RESTORE_MEDIAN_MS}" -lt 30000 ]];  then RESTORE_VERDICT="PASS"
    elif [[ "${RESTORE_MEDIAN_MS}" -lt "${COLD_START_MS:-999999}" ]]; then RESTORE_VERDICT="PARTIAL"
    else                                                RESTORE_VERDICT="FAIL"
    fi
fi

COLD_START_S=$(python3 -c "print(f'{${COLD_START_MS}/1000:.1f}')" 2>/dev/null || echo "N/A")
RESTORE_MS_STR="${RESTORE_MS[*]+"${RESTORE_MS[*]}"}"
RESTORE_MS_STR="${RESTORE_MS_STR:-none}"

echo "Cold start:           ${COLD_START_S}s"
echo "Cold TPS baseline:    ${COLD_TPS} t/s"
echo "Checkpoint path:      ${CHECKPOINT_PATH}"
echo "Checkpoint size:      ${CHECKPOINT_SIZE}"
echo "Restore reps (ms):    ${RESTORE_MS_STR}"
echo "Restore median:       ${RESTORE_MEDIAN_S}s"
echo "Speedup vs cold:      ${SPEEDUP}"
echo "Post-restore TPS:     ${RESTORE_TPS} t/s"
echo "Verdict:              ${RESTORE_VERDICT}"

# ── Write metrics.json ────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 0 ]]; then
    RESTORE_JSON=$(python3 -c "
import json
vals = '${RESTORE_MS_STR}'.split()
nums = [int(v) for v in vals if v.isdigit()]
print(json.dumps(nums))
" 2>/dev/null || echo "[]")

    cat > "${RESULTS_DIR}/metrics.json" <<EOJSON
{
  "item_id": "T_KV2_cuda_checkpoint_tp2_hot_restart",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config": {
    "engine": "vllm",
    "engine_version": "0.19.x",
    "model": "${MODEL}",
    "quantization": "AWQ-INT4",
    "kv_cache_dtype": "fp8",
    "placement": "tp=2",
    "context_length": ${CTX_LEN},
    "gpu_memory_utilization": ${GPU_MEM_UTIL},
    "checkpoint_path": "${CHECKPOINT_PATH}",
    "checkpoint_archive": "${CHECKPOINT_ARCHIVE}",
    "criu_hook_active": $([ "${HAVE_CUDA_HOOK}" -eq 1 ] && echo "true" || echo "false"),
    "cuda_checkpoint_available": $([ "${HAVE_CUDA_CKPT}" -eq 1 ] && echo "true" || echo "false")
  },
  "metrics": {
    "cold_start_ms": ${COLD_START_MS},
    "cold_start_s": "${COLD_START_S}",
    "cold_tps": "${COLD_TPS}",
    "checkpoint_ms": ${CHECKPOINT_MS},
    "checkpoint_size": "${CHECKPOINT_SIZE}",
    "restore_ms_reps": ${RESTORE_JSON},
    "restore_ms_median": ${RESTORE_MEDIAN_MS},
    "restore_s_median": "${RESTORE_MEDIAN_S}",
    "restore_tps": "${RESTORE_TPS}",
    "speedup_vs_cold": "${SPEEDUP}"
  },
  "verdict": "${RESTORE_VERDICT}",
  "notes": "PASS = restore < 30s. With CRIU GPU hook: weights + CUDA graphs restored (fast). Without hook: CPU-only checkpoint, GPU reloads on restore (expect slow)."
}
EOJSON

    # ── Write summary.md ──────────────────────────────────────────────────────
    cat > "${RESULTS_DIR}/summary.md" <<EOMD
# T_KV2 — CUDA Checkpoint Hot-Restart for TP=2 Extended Arclight

**Timestamp:** ${TIMESTAMP}
**Model:** \`${MODEL}\` (AWQ-INT4, fp8 KV, ctx=${CTX_LEN})
**Checkpoint path:** ${CHECKPOINT_PATH}
**CRIU GPU hook:** $( [[ "${HAVE_CUDA_HOOK}" -eq 1 ]] && echo "**YES** — GPU memory included in checkpoint" || echo "**NO** — GPU excluded; slow restore expected" )

## Timing

| Metric | Value |
|--------|-------|
| Cold start (CUDA graph compile) | **${COLD_START_S}s** |
| Cold TPS | ${COLD_TPS} t/s |
| Checkpoint time | $(python3 -c "print(f'{${CHECKPOINT_MS}/1000:.1f}')" 2>/dev/null || echo "N/A")s |
| Checkpoint size | ${CHECKPOINT_SIZE} |
| Restore median (${#RESTORE_VALID[@]}/${REPS} valid reps) | **${RESTORE_MEDIAN_S}s** |
| Speedup vs cold | **${SPEEDUP}** |
| Post-restore TPS | ${RESTORE_TPS} t/s |

Restore rep times (ms): \`${RESTORE_MS_STR}\`

## Verdict: ${RESTORE_VERDICT}

| Range | Meaning |
|-------|---------|
| < 30s | **PASS** — Extended Arclight swap viable for interactive use |
| 30–90s | **PARTIAL** — acceptable for explicit escalation, not transparent routing |
| ≥ cold | **FAIL** — no improvement; use torch-compile cache path instead |

## Interpretation

$(if [[ "${CHECKPOINT_PATH}" == "podman-checkpoint" ]]; then
if [[ "${HAVE_CUDA_HOOK}" -eq 1 ]]; then
    echo "CRIU GPU hook was active: checkpoint includes compiled CUDA graphs + model weights."
    echo "If PASS: Extended Arclight swaps can use this checkpoint for ~5s mode switches."
    echo "If FAIL: GPU restore is slow despite hook — investigate VRAM write bandwidth."
else
    echo "CRIU GPU hook was NOT installed: checkpoint is CPU process state only."
    echo "GPU weights were NOT saved. On restore, vLLM reloads from VRAM."
    echo ""
    echo "This slow-restore result is a BASELINE, not the target."
    echo ""
    echo "**Next step to get the fast-restore result:**"
    echo "\`\`\`bash"
    echo "# Build + install cuda-checkpoint CRIU hooks:"
    echo "git clone https://github.com/NVIDIA/cuda-checkpoint /srv/ai/tools/cuda-checkpoint"
    echo "cd /srv/ai/tools/cuda-checkpoint && make && sudo make install"
    echo "# Re-run T_KV2:"
    echo "bash benchmarks/queue/T_KV2_cuda_checkpoint_tp2_hot_restart.sh --skip-cold"
    echo "\`\`\`"
fi
elif [[ "${CHECKPOINT_PATH}" =~ failed$ ]]; then
    echo "Checkpoint FAILED. See checkpoint.log for the specific CRIU error."
    echo ""
    echo "**Diagnosis steps:**"
    echo "\`\`\`bash"
    echo "criu check --full        # shows which kernel features are missing"
    echo "sudo podman container checkpoint --export /tmp/test.tar ${CONTAINER_NAME}  # root test"
    echo "\`\`\`"
    echo ""
    echo "**If rootless checkpoint is fundamentally unsupported on this kernel:**"
    echo "Fall back to torch.compile cache for ~80s warm restart (no full-graph recompile)."
    echo "Document as DECIDED in DECISIONS.md with the specific CRIU error."
elif [[ "${CHECKPOINT_PATH}" == "cuda-ckpt-standalone-only" ]]; then
    echo "Only GPU state was checkpointed (cuda-checkpoint standalone, no CRIU)."
    echo "CPU process state (Python interpreter, vLLM state) was NOT saved."
    echo "Functional restore is not possible from this checkpoint alone."
fi)

## Architecture impact

$(if [[ "${RESTORE_VERDICT}" == "PASS" ]]; then
    echo "**Extended Arclight hot-swap is viable.**"
    echo "Update ARCHITECTURE.md: mode-switch time = ~${RESTORE_MEDIAN_S}s (was 170–300s)."
elif [[ "${RESTORE_VERDICT}" == "PARTIAL" ]]; then
    echo "Partial improvement. Useful for explicit escalation (user triggers @extended)."
    echo "Not suitable for transparent automatic routing."
else
    echo "No improvement. Extended Arclight cold-start cost stands at ~${COLD_START_S}s."
    echo "Consider torch.compile cache as a cheaper ~80s warm-restart path."
    echo "Hand back to research mode with this summary."
fi)
EOMD

    echo ""
    echo "[T_KV2] Written: ${RESULTS_DIR}/metrics.json"
    echo "[T_KV2] Written: ${RESULTS_DIR}/summary.md"
else
    echo "[dry-run] Would write metrics.json + summary.md to ${RESULTS_DIR}/"
fi
