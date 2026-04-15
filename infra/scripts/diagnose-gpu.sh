#!/usr/bin/env bash
# diagnose-gpu.sh — probe NVIDIA GPU passthrough for rootless podman.
#
# Run this on the HOST (not inside a container) to determine why vLLM
# containers fail with "libcuda.so.1: cannot open shared object file".
#
# It tests three passthrough mechanisms in order of preference:
#   1. CDI  (nvidia.com/gpu=N)            ← rootless-podman native, preferred
#   2. nvidia runtime via --runtime=nvidia ← needs entry in containers.conf
#   3. Raw device mounts                  ← last resort, rarely works rootless
#
# At the end it prints a concrete fix command.
#
# Usage:
#   ./infra/scripts/diagnose-gpu.sh
#   ./infra/scripts/diagnose-gpu.sh --no-pull   # skip image pull if cached
set -uo pipefail

NO_PULL="${1:-}"
PROBE_IMAGE="nvcr.io/nvidia/cuda:12.3.2-base-ubuntu22.04"
PASS="[ OK ]"
FAIL="[FAIL]"
SKIP="[SKIP]"
SEP="──────────────────────────────────────────────────────────────"

echo ""
echo "${SEP}"
echo " GPU passthrough diagnostic for rootless podman"
echo "${SEP}"
echo ""

# ── 0. Host nvidia-smi ───────────────────────────────────────────────────────
echo "=== 0. Host nvidia-smi ==="
if nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null; then
    echo "${PASS} nvidia-smi works on host"
else
    echo "${FAIL} nvidia-smi failed — NVIDIA driver not installed or not loaded"
    echo "       Cannot proceed without a working host driver."
    exit 1
fi
echo ""

# ── 1. Podman version and OCI runtime config ─────────────────────────────────
echo "=== 1. Podman runtime configuration ==="
PODMAN_VERSION=$(podman --version 2>/dev/null || echo "NOT FOUND")
echo "  podman:          ${PODMAN_VERSION}"
CRUN_PATH=$(command -v crun 2>/dev/null || echo "not found")
RUNC_PATH=$(command -v runc 2>/dev/null || echo "not found")
NV_RUNTIME_PATH=$(command -v nvidia-container-runtime 2>/dev/null || echo "not found")
echo "  crun:            ${CRUN_PATH}"
echo "  runc:            ${RUNC_PATH}"
echo "  nvidia-runtime:  ${NV_RUNTIME_PATH}"
echo ""

# Check containers.conf for nvidia runtime entry
echo "  containers.conf search for 'nvidia' runtime:"
for cfg in \
    "${HOME}/.config/containers/containers.conf" \
    "/etc/containers/containers.conf" \
    "/usr/share/containers/containers.conf"; do
    if [[ -f "${cfg}" ]]; then
        if grep -i "nvidia" "${cfg}" 2>/dev/null; then
            echo "    ${PASS} found in ${cfg}"
        else
            echo "    (no nvidia entry in ${cfg})"
        fi
    else
        echo "    (${cfg} not found)"
    fi
done
echo ""

# Check CDI device list
echo "=== 2. CDI (Container Device Interface) ==="
if command -v nvidia-ctk &>/dev/null; then
    echo "  nvidia-ctk found: $(command -v nvidia-ctk)"
    echo "  CDI device list:"
    nvidia-ctk cdi list 2>/dev/null || echo "    (nvidia-ctk cdi list failed)"
else
    echo "  ${SKIP} nvidia-ctk not installed (part of nvidia-container-toolkit)"
fi
echo ""

# ── Pull probe image once ─────────────────────────────────────────────────────
if [[ "${NO_PULL}" != "--no-pull" ]]; then
    echo "=== Pulling probe image (pass --no-pull to skip) ==="
    if podman pull "${PROBE_IMAGE}" 2>&1 | tail -3; then
        echo "${PASS} image pulled"
    else
        echo "${FAIL} could not pull ${PROBE_IMAGE}"
        echo "  Tip: run with a cached image or fix network access."
        exit 1
    fi
    echo ""
fi

# Helper: run a quick nvidia-smi inside a container and report
probe() {
    local label="$1"; shift
    echo -n "  Testing ${label} ... "
    if timeout 30 podman run --rm "$@" \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        "${PROBE_IMAGE}" nvidia-smi -L 2>&1 | grep -q "GPU 0:"; then
        echo "${PASS}"
        return 0
    else
        local out
        out=$(timeout 30 podman run --rm "$@" \
            -e NVIDIA_VISIBLE_DEVICES=0 \
            -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
            "${PROBE_IMAGE}" nvidia-smi -L 2>&1 | head -5)
        echo "${FAIL}"
        echo "    ${out}"
        return 1
    fi
}

echo "=== 3. Passthrough mechanism probes ==="
echo ""

# Test A: CDI (preferred for rootless podman)
CDI_WORKS=0
echo "  [A] CDI — podman --device nvidia.com/gpu=0"
if probe "CDI gpu=0" --device "nvidia.com/gpu=0"; then
    CDI_WORKS=1
fi
echo ""

# Test B: --runtime=nvidia (requires containers.conf entry)
RUNTIME_WORKS=0
echo "  [B] nvidia OCI runtime — podman --runtime=nvidia"
if [[ "${NV_RUNTIME_PATH}" != "not found" ]]; then
    if probe "runtime=nvidia" --runtime=nvidia; then
        RUNTIME_WORKS=1
    fi
else
    echo "  ${SKIP} nvidia-container-runtime not found in PATH"
fi
echo ""

# Test C: raw devices (usually requires privileged for rootless)
RAW_WORKS=0
echo "  [C] Raw device mounts — /dev/nvidia0 + /dev/nvidiactl"
if [[ -e /dev/nvidia0 && -e /dev/nvidiactl ]]; then
    if probe "raw devices" \
        --device /dev/nvidia0 \
        --device /dev/nvidiactl \
        --device /dev/nvidia-uvm \
        --device /dev/nvidia-uvm-tools; then
        RAW_WORKS=1
    fi
else
    echo "  ${SKIP} /dev/nvidia0 not accessible from this user (normal for rootless)"
fi
echo ""

# ── Summary and fix recommendation ───────────────────────────────────────────
echo "${SEP}"
echo " Results"
echo "${SEP}"
echo ""
echo "  CDI works:             $([[ ${CDI_WORKS} -eq 1 ]] && echo YES || echo NO)"
echo "  nvidia runtime works:  $([[ ${RUNTIME_WORKS} -eq 1 ]] && echo YES || echo NO)"
echo "  Raw devices work:      $([[ ${RAW_WORKS} -eq 1 ]] && echo YES || echo NO)"
echo ""

if [[ ${CDI_WORKS} -eq 1 ]]; then
    echo "${PASS} RECOMMENDED: use CDI (update compose files — run the command below)"
    echo ""
    echo "  The compose files need to pass '--device nvidia.com/gpu=0' to podman."
    echo "  Because docker-compose (external provider) doesn't forward CDI devices,"
    echo "  switch deploy.sh to call podman directly instead of through compose:"
    echo ""
    echo "    Edit infra/scripts/deploy.sh to use:"
    echo "      podman run -d --device nvidia.com/gpu=0 ..."
    echo ""
    echo "  OR configure the nvidia runtime so compose's 'runtime: nvidia' works:"
    echo "    sudo nvidia-ctk runtime configure --runtime=podman"
    echo "    # then re-run this diagnostic to verify [B] passes"
    echo ""
elif [[ ${RUNTIME_WORKS} -eq 1 ]]; then
    echo "${PASS} nvidia OCI runtime works via --runtime=nvidia."
    echo "  The problem is that docker-compose (external provider) isn't forwarding"
    echo "  the runtime flag to podman correctly. Fix: register in containers.conf:"
    echo ""
    cat <<'EOF'
    # Add to ~/.config/containers/containers.conf :
    [engine.runtimes]
    nvidia = ["/usr/bin/nvidia-container-runtime"]
EOF
    echo ""
elif [[ ${RAW_WORKS} -eq 1 ]]; then
    echo "${PASS} Raw device mounts work."
    echo "  Update compose files to mount /dev/nvidia* instead of using runtime."
    echo "  See infra/compose/vllm-gpu0.yaml — add a 'devices:' section."
    echo ""
else
    echo "${FAIL} NO GPU passthrough mechanism is working."
    echo ""
    echo "  Most likely fix for rootless podman on this host:"
    echo ""
    echo "    # 1. Install NVIDIA Container Toolkit (if not already installed):"
    echo "    #    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    echo "    # 2. Generate CDI specs (run as root):"
    echo "    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
    echo ""
    echo "    # 3. Configure the nvidia OCI runtime for podman (run as root):"
    echo "    sudo nvidia-ctk runtime configure --runtime=podman"
    echo ""
    echo "    # 4. Re-run this diagnostic:"
    echo "    ./infra/scripts/diagnose-gpu.sh --no-pull"
fi

echo ""
echo "${SEP}"
