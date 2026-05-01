#!/usr/bin/env bash
# T_MTP2_coder_mtp_sweep.sh — Coder MTP n=1 Throughput and Quality Validation
#
# Objective: Measure the TPS impact and quality stability of Multi-Token
# Prediction (MTP) with n=1 on the Arclight production coder (Qwen3.6-35B-A3B-AWQ).
#
# Placement: GPU0+1 TP=2
#
# Usage: ./benchmarks/queue/T_MTP2_coder_mtp_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"

# ── Argument Parsing ─────────────────────────────────────────────────────────
REPS=3
DRY_RUN=0
SKIP_DEPLOY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps)        REPS="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --skip-deploy) SKIP_DEPLOY=1; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

ITEM_ID="T_MTP2_coder_mtp_sweep"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/raw"

LOG="${RESULTS_DIR}/bench.log"
log() { echo "[T_MTP2 $(date -u +%H:%M:%S)] $*" | tee -a "${LOG}"; }

# ── Preflight ────────────────────────────────────────────────────────────────
log "Starting T_MTP2 (Coder MTP Validation)..."
log "Results: ${RESULTS_DIR}"

MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
ENDPOINT="http://localhost:30000/v1"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "[DRY-RUN] Would check BENCH_13 success"
    log "[DRY-RUN] Would record initial VRAM"
else
    # Check BENCH_13
    B13_DIR=$(ls -td "${REPO_ROOT}/results/BENCH_13_mtp1_thinker_"* 2>/dev/null | head -1)
    if [ -z "${B13_DIR}" ]; then
        log "WARNING: BENCH_13 results not found. Proceeding anyway..."
    else
        log "Found BENCH_13: ${B13_DIR}"
    fi
fi

# ── Step 1: Baseline Sweep ───────────────────────────────────────────────────
log "Step 1: Recording baseline TPS (no MTP)..."

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "[DRY-RUN] bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh --skip-thinker --skip-convergence --reps ${REPS}"
else
    # Ensure coder is running
    if ! curl -sf "http://localhost:30000/health" >/dev/null; then
        log "Coder not running. Deploying baseline production config..."
        VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
        "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" \
            --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
            --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
    fi

    bash "${REPO_ROOT}/benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh" \
        --skip-thinker \
        --skip-convergence \
        --reps "${REPS}"

    BASELINE_DIR=$(ls -td "${REPO_ROOT}/results/T_PAR1_parallel_throughput_sweep_"* | head -1)
    log "Baseline sweep complete: ${BASELINE_DIR}"
    cp "${BASELINE_DIR}/metrics.json" "${RESULTS_DIR}/metrics_baseline.json"
fi

# ── Step 2: Deploy with MTP n=1 ──────────────────────────────────────────────
log "Step 2: Deploying Coder with MTP n=1..."

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "[DRY-RUN] podman stop bench-vllm-tp2a"
    log "[DRY-RUN] deploy.sh with MTP flags"
else
    # Identify container
    CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|tp2a\|35b" | head -1)
    if [ -n "${CODER_CONTAINER}" ]; then
        log "Stopping ${CODER_CONTAINER}..."
        podman stop "${CODER_CONTAINER}" && podman rm "${CODER_CONTAINER}"
    fi

    VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    "${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2a "${MODEL}" \
        --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
        --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}'

    # Wait for health
    log "Waiting for Coder+MTP health..."
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:30000/health" >/dev/null; then
            log "Coder+MTP READY"
            break
        fi
        sleep 1
        if [ "$i" -eq 120 ]; then
            log "FATAL: Coder+MTP failed to start within 120s"
            exit 1
        fi
    done

    # Record VRAM
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${RESULTS_DIR}/vram_mtp.txt"
    log "VRAM recorded in vram_mtp.txt"
fi

# ── Step 3: MTP Sweep ────────────────────────────────────────────────────────
log "Step 3: Running TPS sweep with MTP n=1..."

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "[DRY-RUN] bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh --skip-thinker --skip-convergence --reps ${REPS}"
else
    bash "${REPO_ROOT}/benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh" \
        --skip-thinker \
        --skip-convergence \
        --reps "${REPS}"

    MTP_DIR=$(ls -td "${REPO_ROOT}/results/T_PAR1_parallel_throughput_sweep_"* | head -1)
    log "MTP sweep complete: ${MTP_DIR}"
    cp "${MTP_DIR}/metrics.json" "${RESULTS_DIR}/metrics_mtp_n1.json"
fi

# ── Step 4: Quality Smoke Check ──────────────────────────────────────────────
log "Step 4: Tool-call quality smoke check..."

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "[DRY-RUN] 3 probes for tool-call reliability"
else
    PROBE_RESULTS="${RESULTS_DIR}/toolcall_check.txt"
    echo "MTP Tool-call Smoke Check" > "${PROBE_RESULTS}"
    PASS_COUNT=0
    for i in $(seq 1 3); do
        log "Probe $i/3..."
        RESPONSE=$(curl -s "${ENDPOINT}/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "'"${MODEL}"'",
            "messages": [
              {"role": "system", "content": "You are a helpful assistant with access to tools."},
              {"role": "user", "content": "List all Python files in the /src directory using the bash tool."}
            ],
            "tools": [
              {
                "type": "function",
                "function": {
                  "name": "bash",
                  "description": "Run a shell command",
                  "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                  }
                }
              }
            ],
            "tool_choice": "auto",
            "max_tokens": 512,
            "temperature": 0
          }')
        
        VALID=$(echo "${RESPONSE}" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    msg = r['choices'][0]['message']
    if msg.get('tool_calls'):
        tc = msg['tool_calls'][0]
        if tc['function']['name'] == 'bash':
            print('OK')
            sys.exit(0)
except: pass
print('FAIL')
sys.exit(1)
" || echo "FAIL")

        echo "Probe $i: ${VALID}" >> "${PROBE_RESULTS}"
        if [ "${VALID}" == "OK" ]; then
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
    done
    log "Tool-call results: ${PASS_COUNT}/3 PASS"
fi

# ── Step 5: Cleanup ──────────────────────────────────────────────────────────
log "Step 5: Cleanup (stopping benchmark container)..."

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "[DRY-RUN] podman stop and rm benchmark container"
else
    CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|tp2a\|35b" | head -1)
    if [ -n "${CODER_CONTAINER}" ]; then
        log "Stopping ${CODER_CONTAINER}..."
        podman stop "${CODER_CONTAINER}" && podman rm "${CODER_CONTAINER}"
    fi
    log "GPUs free. Production coder MUST BE RESTORED MANUALLY."
fi

# ── Step 6: Reporting ────────────────────────────────────────────────────────
log "Step 6: Generating summary.md and metrics.json..."

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "[DRY-RUN] Writing reports to ${RESULTS_DIR}"
else
    export TIMESTAMP RESULTS_DIR
    python3 - <<'EOF'
import json, os, pathlib

results_dir = pathlib.Path(os.environ["RESULTS_DIR"])
baseline_path = results_dir / "metrics_baseline.json"
mtp_path = results_dir / "metrics_mtp_n1.json"

if baseline_path.exists() and mtp_path.exists():
    baseline = json.load(open(baseline_path))
    mtp = json.load(open(mtp_path))

    bc = baseline.get("metrics", {}).get("coder_detail", {}).get("results_by_n", {})
    mc = mtp.get("metrics", {}).get("coder_detail", {}).get("results_by_n", {})

    ns = sorted(set(int(k) for k in bc))
    comp_rows = []
    for n in ns:
        bv = bc.get(str(n), {}).get("median_agg_tps", 0)
        mv = mc.get(str(n), {}).get("median_agg_tps", 0)
        delta = round((mv-bv)/bv*100, 1) if bv > 0 else 0
        comp_rows.append(f"| {n} | {bv} | {mv} | {delta}% |")
    comp_table = "\n".join(comp_rows)
else:
    comp_table = "| N | Baseline | MTP | Delta |\n|---|---|---|---|\n| ERROR | missing | files | N/A |"

# Quality
probe_path = results_dir / "toolcall_check.txt"
if probe_path.exists():
    probes = open(probe_path).read()
    pass_count = probes.count("OK")
else:
    pass_count = 0

# VRAM
vram_path = results_dir / "vram_mtp.txt"
vram_text = open(vram_path).read() if vram_path.exists() else "N/A"

md = f"""# BENCH_14 — T_MTP2 Coder MTP n=1 Validation

## TPS Comparison
| N | Baseline (no MTP) | MTP n=1 | Delta |
|---|-------------------|---------|-------|
{comp_table}

## Quality
Tool-call probe: {pass_count}/3 well-formed

## GPU0+1 VRAM with MTP
```
{vram_text}
```

## Verdict
**{"PASS" if pass_count == 3 else "FAIL"}**
"""
(results_dir / "summary.md").write_text(md)

final_metrics = {
    "item_id": "T_MTP2_coder_mtp_sweep",
    "timestamp": os.environ["TIMESTAMP"],
    "metrics": {
        "tool_call_pass_rate": pass_count / 3,
    },
    "verdict": "PASS" if pass_count == 3 else "FAIL"
}

# Add TPS if available
if baseline_path.exists() and mtp_path.exists():
    final_metrics["metrics"]["baseline_n1_tps"] = bc.get("1", {}).get("median_agg_tps")
    final_metrics["metrics"]["mtp_n1_tps"] = mc.get("1", {}).get("median_agg_tps")

(results_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))
EOF
fi

log "Done. Results in ${RESULTS_DIR}"
log "REMINDER: Production coder is STOPPED. Restart manually."
