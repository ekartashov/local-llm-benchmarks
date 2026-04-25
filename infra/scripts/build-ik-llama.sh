#!/usr/bin/env bash
# infra/scripts/build-ik-llama.sh
#
# Compiles ik_llama.cpp (pr-1288) inside a CUDA 12.8 build container.
# The output binary lands at IK_LLAMA_SRC/build/bin/ on the HOST filesystem —
# not inside a container volume. The build container is ephemeral and throwaway;
# only the binary survives it.
#
# This is intentional: the host filesystem holds the binary so it can be
# bind-mounted into the runtime container at any time without needing a registry
# or extra pulls. The build and runtime containers share the same CUDA 12.8.1 tag,
# guaranteeing shared library ABI compatibility.
#
# Usage:
#   ./infra/scripts/build-ik-llama.sh [--rebuild-image] [--clean] [--dry-run]
#
# Options:
#   --rebuild-image   Force rebuild of local-ik-llama:build even if it exists.
#   --clean           Remove IK_LLAMA_SRC/build/ before compiling (full recompile).
#                     Use when switching CUDA versions or cmake options.
#   --dry-run         Print commands, do not execute.
#
# Environment:
#   IK_LLAMA_SRC    Path to ik_llama.cpp git repo (default: /srv/ai/projects/ik_llama.cpp)
#
# Prerequisites:
#   - ik_llama.cpp cloned at IK_LLAMA_SRC (git clone ikawrakow/ik_llama.cpp)
#   - Rootless podman accessible in PATH
#   - pr-1288 branch already fetched OR git remote reachable

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

IK_LLAMA_SRC="${IK_LLAMA_SRC:-/srv/ai/projects/ik_llama.cpp}"
BUILD_IMAGE="local-ik-llama:build"
DRY_RUN=0
REBUILD_IMAGE=0
CLEAN_BUILD=0

for arg in "$@"; do
    case "$arg" in
        --dry-run)       DRY_RUN=1 ;;
        --rebuild-image) REBUILD_IMAGE=1 ;;
        --clean)         CLEAN_BUILD=1 ;;
    esac
done

run() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

# ── 1. Verify source tree ────────────────────────────────────────────────────
echo "[build-ik-llama] Source: ${IK_LLAMA_SRC}"
if [[ ! -d "${IK_LLAMA_SRC}/.git" ]]; then
    echo "[build-ik-llama] ERROR: ${IK_LLAMA_SRC} is not a git repo." >&2
    echo "  Clone it: git clone https://github.com/ikawrakow/ik_llama.cpp ${IK_LLAMA_SRC}" >&2
    exit 1
fi

# ── 2. Ensure pr-1288 branch is checked out ─────────────────────────────────
(
    cd "${IK_LLAMA_SRC}"
    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    if [[ "${CURRENT_BRANCH}" != "pr-1288" ]]; then
        echo "[build-ik-llama] Current branch: ${CURRENT_BRANCH}. Switching to pr-1288..."
        if ! git rev-parse --verify pr-1288 &>/dev/null; then
            echo "[build-ik-llama] Fetching pr-1288 from origin..."
            run git fetch origin pull/1288/head:pr-1288
        fi
        run git checkout pr-1288
    else
        echo "[build-ik-llama] Already on pr-1288."
    fi
)

# ── 3. Optionally wipe the build directory ───────────────────────────────────
if [[ "${CLEAN_BUILD}" -eq 1 ]]; then
    echo "[build-ik-llama] --clean: removing ${IK_LLAMA_SRC}/build/"
    run rm -rf "${IK_LLAMA_SRC}/build"
fi

# ── 4. Build the builder image if absent or forced ───────────────────────────
if [[ "${REBUILD_IMAGE}" -eq 1 ]] || ! podman image exists "${BUILD_IMAGE}" 2>/dev/null; then
    echo "[build-ik-llama] Building container image ${BUILD_IMAGE}..."
    run podman build \
        -f "${REPO_ROOT}/infra/Containerfile.ik-llama-build" \
        -t "${BUILD_IMAGE}" \
        "${REPO_ROOT}"
else
    echo "[build-ik-llama] Image ${BUILD_IMAGE} exists. Pass --rebuild-image to force."
fi

# ── 5. Run compilation inside the container ──────────────────────────────────
# --userns=keep-id: files written to /src/build/ are owned by the calling user
#   on the host. Without this, rootless podman maps the container's root to a
#   subuid range; build artifacts end up with a different owner on the host.
# --security-opt=no-new-privileges: hardening for the ephemeral build container.
# No GPU needed: nvcc compiles without a device present.
# The default CMD in Containerfile.ik-llama-build runs cmake configure + build.
echo "[build-ik-llama] Compiling (bind-mounting ${IK_LLAMA_SRC} → /src)..."
echo "[build-ik-llama] This takes several minutes on the first build."
run podman run --rm \
    --name ik-llama-build-run \
    --userns=keep-id \
    --security-opt=no-new-privileges \
    -v "${IK_LLAMA_SRC}:/src:rw,z" \
    "${BUILD_IMAGE}"

echo ""
echo "[build-ik-llama] Build complete."
if [[ "${DRY_RUN}" -eq 0 ]]; then
    echo "[build-ik-llama] Binaries at ${IK_LLAMA_SRC}/build/bin/:"
    ls -lh "${IK_LLAMA_SRC}/build/bin/" 2>/dev/null | grep -E "llama-(server|bench)" || \
        echo "  (none found — check cmake output above for errors)"
fi
echo ""
echo "[build-ik-llama] Build the runtime image next (if not done):"
echo "  podman build -f infra/Containerfile.ik-llama-runtime -t local-ik-llama:runtime ."
