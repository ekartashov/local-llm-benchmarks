#!/usr/bin/env bash
# wait-healthy.sh <health_url> [timeout_seconds]
# Polls a health endpoint until it returns HTTP 200, or times out.
set -euo pipefail

HEALTH_URL="${1:?Usage: wait-healthy.sh <health_url> [timeout_seconds]}"
TIMEOUT="${2:-180}"
INTERVAL=5

echo "[wait-healthy] Polling ${HEALTH_URL} (timeout=${TIMEOUT}s)..."
start=$(date +%s)
while true; do
    if curl -sf --max-time 3 "${HEALTH_URL}" > /dev/null 2>&1; then
        echo "[wait-healthy] Endpoint is healthy."
        exit 0
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
