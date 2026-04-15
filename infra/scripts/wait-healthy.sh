#!/usr/bin/env bash
# wait-healthy.sh <health_url> [timeout_seconds] [container_name]
#
# Polls a health endpoint until it returns HTTP 200, or times out.
# If container_name is provided, fails fast when the container exits instead
# of waiting the full timeout.
set -euo pipefail

HEALTH_URL="${1:?Usage: wait-healthy.sh <health_url> [timeout_seconds] [container_name]}"
TIMEOUT="${2:-180}"
CONTAINER_NAME="${3:-}"
INTERVAL=5

echo "[wait-healthy] Polling ${HEALTH_URL} (timeout=${TIMEOUT}s)..."
start=$(date +%s)
while true; do
    if curl -sf --max-time 3 "${HEALTH_URL}" > /dev/null 2>&1; then
        echo "[wait-healthy] Endpoint is healthy."
        exit 0
    fi

    # Fast-fail: detect a crashed container rather than waiting the full timeout.
    if [[ -n "${CONTAINER_NAME}" ]]; then
        status=$(podman container inspect "${CONTAINER_NAME}" \
            --format '{{.State.Status}}' 2>/dev/null || echo "gone")
        if [[ "${status}" == "exited" || "${status}" == "stopped" || "${status}" == "gone" ]]; then
            echo "[wait-healthy] ERROR: container '${CONTAINER_NAME}' exited (status=${status}) — engine crashed before becoming healthy." >&2
            exit 1
        fi
    fi

    now=$(date +%s)
    elapsed=$(( now - start ))
    if (( elapsed >= TIMEOUT )); then
        echo "[wait-healthy] ERROR: timed out after ${TIMEOUT}s waiting for ${HEALTH_URL}" >&2
        exit 1
    fi
    echo "[wait-healthy] Not ready yet (${elapsed}s elapsed). Retrying in ${INTERVAL}s..."
    sleep "${INTERVAL}"
done
