#!/usr/bin/env bash
# T2.4f_qwen36_27b_rope_chunkedprefill_audit.sh
#
# Zero-cost config inspection — deploy Qwen3.6-27B-AWQ, capture startup logs,
# verify rope_theta and chunked-prefill state. Does NOT run the quality suite.
#
# Run this BEFORE T2.4d/T2.4e. Takes ~10 minutes (model load only).
# Output: findings.md with corrections (if any) for T2.4d/T2.4e.
#
# Context (R16):
#   Root cause of Qwen3.6-27B confident incorrectness is unknown.
#   Hypothesis 1 (H1): rope_theta mismatch — vLLM reading wrong value for long
#                       thinking traces at positions 2k-4k tokens.
#   Hypothesis 2 (H2): chunked prefill breaks GDN recurrent state mid-trace.
#   This script collects facts to test both hypotheses before burning test cycles.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY_RUN=1; shift ;;
        *) shift ;;
    esac
done

ITEM_ID="T2.4f_qwen36_27b_rope_chunkedprefill_audit"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="QuantTrio/Qwen3.6-27B-AWQ"

mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/bench.log"
STARTUP_LOG="${RESULTS_DIR}/startup.log"

log() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "[T2.4f] $*" | tee -a "${LOG}"
    fi
}
die() { log "FATAL: $*"; exit 1; }

log "=== T2.4f RoPE theta + Chunked Prefill Config Audit ==="
log "Timestamp: ${TIMESTAMP}"
log "Model: ${MODEL}"
log "Purpose: inspect startup log for H1 (rope_theta mismatch) and H2 (chunked prefill ON)."
log "         No quality suite. Findings gate T2.4d and T2.4e."

# ── Step 1: Read local config.json ─────────────────────────────────────────────
log ""
log "--- Step 1: Read local config.json from model cache ---"

python3 - <<PYEOF 2>&1 | tee -a "${LOG}"
import json, pathlib, sys

hub = pathlib.Path("${MODEL_CACHE}") / "hub"
model_dir = hub / "models--QuantTrio--Qwen3.6-27B-AWQ"
if not model_dir.exists():
    print("         Model may not be downloaded yet, or MODEL_CACHE is wrong.")
    print("         Expected rope_theta for Qwen3.6-27B series: 10000000")
    sys.exit(0)

configs = sorted(model_dir.glob("snapshots/*/config.json"))
if not configs:
    print(f"WARNING: no config.json found under {model_dir}/snapshots/")
    sys.exit(0)

cfg_path = configs[-1]
cfg = json.loads(cfg_path.read_text())
print(f"config.json: {cfg_path}")
print(f"  model_type:              {cfg.get('model_type', 'N/A')}")
# Handle nested text_config (Qwen3 format)
text_cfg = cfg.get("text_config", {})
rope_params = text_cfg.get("rope_parameters", {})
rope_val = (
    rope_params.get("rope_theta") or 
    text_cfg.get("rope_theta") or 
    cfg.get("rope_theta") or 
    cfg.get("rotary_emb_base")
)

print(f"  rope_theta:              {rope_val or 'NOT PRESENT'}")
print(f"  max_position_embeddings: {text_cfg.get('max_position_embeddings', cfg.get('max_position_embeddings', 'N/A'))}")

if rope_val is not None:
    expected = 10_000_000
    delta = abs(float(rope_val) - expected)
    tag = "OK" if delta < 1000 else "MISMATCH"
    print(f"  [{tag}] rope_theta = {rope_val}  (expected {expected})")
    if delta >= 1000:
        print(f"  ACTION: override rope_theta in T2.4d/T2.4e — see findings.md")
else:
    print("  [WARN] rope_theta not found in config.json")
PYEOF

# ── Step 2: Deploy on GPU1 and capture full startup log ─────────────────────────
log ""
log "--- Step 2: Deploy TP=1 GPU1 (ctx=49152, fp8 KV) — standard T2.4 config ---"
log "Full startup output → ${STARTUP_LOG}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "WOULD DEPLOY: vllm gpu1 ${MODEL} --gpu-mem-util 0.90 --ctx 49152 --kv-cache-dtype fp8 --max-num-seqs 1"
    log "WOULD EXTRACT: rope_theta, chunked_prefill from startup log"
    log "WOULD WRITE: findings.md"
else
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
        "${REPO_ROOT}/infra/scripts/deploy.sh" vllm gpu1 "${MODEL}" \
        --gpu-mem-util 0.90 \
        --ctx 49152 \
        --kv-cache-dtype fp8 \
        --max-num-seqs 1 \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        2>&1 | tee -a "${LOG}" "${STARTUP_LOG}" \
        || die "Deployment failed. See ${STARTUP_LOG}."

    log "Deploy complete (server healthy)."

    # ── Step 3: Parse startup log ────────────────────────────────────────────────
    log ""
    log "--- Step 3: Extract rope_theta and chunked-prefill from startup log ---"

    python3 - <<PYEOF 2>&1 | tee -a "${LOG}"
import re, pathlib

text = pathlib.Path("${STARTUP_LOG}").read_text(errors="replace")

# rope_theta: appears in config dumps, model init logs, JSON-ish lines
rope_hits = (
    re.findall(r"rope_theta['\"\s]*[:=]\s*([\d.eE+\-]+)", text, re.IGNORECASE) +
    re.findall(r"rotary_emb_base['\"\s]*[:=]\s*([\d.eE+\-]+)", text, re.IGNORECASE)
)
rope_unique = list(dict.fromkeys(rope_hits))

print("\\n=== H1: rope_theta in vLLM startup log ===")
if rope_unique:
    for v in rope_unique[:5]:
        print(f"  found: {v}")
    vf = float(rope_unique[0])
    tag = "OK" if abs(vf - 10_000_000) < 1000 else "MISMATCH"
    print(f"  [{tag}] vLLM using rope_theta = {vf:.0f}  (expected 10000000)")
else:
    print("  NOT FOUND in log — vLLM likely reads silently from config.json.")
    print("  Cross-check with Step 1 config.json output above.")

# chunked prefill
cp_hits = re.findall(r"(?i)(?:chunked.prefill|enable_chunked_prefill)[^\n]{0,150}", text)
mbt_hits = re.findall(r"(?i)max_num_batched_tokens[^\n]{0,80}", text)

print("\\n=== H2: chunked prefill in vLLM startup log ===")
seen = set()
found = False
for h in cp_hits + mbt_hits:
    h = h.strip()
    if h not in seen:
        seen.add(h)
        print(f"  {h}")
        found = True
if not found:
    print("  NOT FOUND explicitly — vLLM 0.19.x enables chunked prefill by default.")
    print("  Assumption: ON unless log shows 'enable_chunked_prefill=False'.")
PYEOF

    # ── Step 4: Write findings.md ─────────────────────────────────────────────────
    log ""
    log "--- Step 4: Writing findings.md ---"

    python3 - <<PYEOF
import re, pathlib
from datetime import datetime, timezone

out = pathlib.Path("${RESULTS_DIR}")
text = (out / "startup.log").read_text(errors="replace")

# rope_theta
rope_hits = (
    re.findall(r"rope_theta['\"\s]*[:=]\s*([\d.eE+\-]+)", text, re.IGNORECASE) +
    re.findall(r"rotary_emb_base['\"\s]*[:=]\s*([\d.eE+\-]+)", text, re.IGNORECASE)
)
rope_unique = list(dict.fromkeys(rope_hits))

h1_status = "UNKNOWN — not found in log; check config.json output in bench.log"
h1_action = (
    "Cross-check Step 1 config.json rope_theta. If config.json shows 1000000, "
    "vLLM is likely reading it correctly. No flag override needed."
)
if rope_unique:
    vf = float(rope_unique[0])
    if abs(vf - 10_000_000) < 1000:
        h1_status = f"OK — vLLM using rope_theta = {vf:.0f} (matches expected 10000000)"
        h1_action = "No correction needed."
    else:
        h1_status = f"MISMATCH — vLLM using {vf:.0f}, expected 10000000"
        h1_action = (
            f"Add rope_theta override to T2.4d/T2.4e. Candidate flags for vLLM 0.19.x:\n"
            f"  --rope-theta 10000000\n"
            f"  --hf-overrides '{{\"rope_theta\": 10000000}}'\n"
            f"  --override-model-config '{{\"rope_theta\": 10000000}}'\n"
            f"Check vLLM 0.19.x --help for the correct flag name. "
            f"Set ROPE_THETA_FLAG env var in T2.4d/T2.4e scripts accordingly."
        )

# chunked prefill
cp_hits = re.findall(r"(?i)(?:chunked.prefill|enable_chunked_prefill)[^\n]{0,150}", text)
mbt_hits = re.findall(r"(?i)max_num_batched_tokens[^\n]{0,80}", text)
cp_lines = list(dict.fromkeys(h.strip() for h in cp_hits + mbt_hits))[:5]
cp_text = "\n".join(f"  {l}" for l in cp_lines) if cp_lines else "  (not found in log)"

if not cp_lines:
    h2_status = "LIKELY ON — no explicit log entry; vLLM 0.19.x enables chunked prefill by default"
    h2_action = (
        "Run T2.4d in two variants:\n"
        "  Variant A: no extra flags (standard run 4 config)\n"
        "  Variant B: add --disable-chunked-prefill (test H2 directly)\n"
        "Compare th02 correctness between variants."
    )
elif any("false" in l.lower() or "disabled" in l.lower() for l in cp_lines):
    h2_status = "OFF — chunked prefill disabled or not active"
    h2_action = "H2 unlikely. Skip --disable-chunked-prefill variant in T2.4d."
else:
    h2_status = "ON — chunked prefill explicitly enabled"
    h2_action = (
        "Run T2.4d Variant B with --disable-chunked-prefill. "
        "If th02 corrects, GDN recurrence is the root cause."
    )

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
findings = f"""# T2.4f — Config Audit Findings

**Timestamp:** {now}
**Model:** QuantTrio/Qwen3.6-27B-AWQ
**Deploy config:** TP=1 GPU1, ctx=49152, fp8 KV, max-num-seqs 1

---

## H1: RoPE theta mismatch

**Status:** {h1_status}

**Action:** {h1_action}

---

## H2: Chunked prefill × GDN recurrence

**Log lines found:**
{cp_text}

**Status:** {h2_status}

**Action:** {h2_action}

---

## T2.4d recommended variants

Based on these findings, run T2.4d as follows:

**Variant A** (always run — baseline):
  Exact run 4 config, no extra flags.
  ctx=32768, fp8 KV, max_tokens=16384, max-num-seqs 1 × 3 runs.

**Variant B** (run if H2 status is ON or LIKELY ON):
  Same config + --disable-chunked-prefill.
  Run once (1 full 8-task run) to check if th02 corrects.

Apply any H1 rope_theta correction (if MISMATCH) to BOTH variants.

---

## Files

- Startup log: {out}/startup.log
- Full output: {out}/bench.log
"""
(out / "findings.md").write_text(findings)
print(f"[T2.4f] findings.md written: {out}/findings.md")
PYEOF

    # Stop the container — T2.4d needs ctx=32768, requires fresh deploy
    log ""
    log "Stopping bench-vllm-gpu1 (T2.4d uses ctx=32768 — different from this ctx=49152 deploy)..."
    podman stop bench-vllm-gpu1 2>/dev/null || true
    podman rm   bench-vllm-gpu1 2>/dev/null || true
    log "Container stopped."
fi

log ""
log "=== T2.4f complete ==="
log "Results dir:  ${RESULTS_DIR}"
log "Findings:     ${RESULTS_DIR}/findings.md"
log "Startup log:  ${STARTUP_LOG}"
log ""
log "Next steps:"
log "  1. Read findings.md — note H1 and H2 status"
log "  2. Run T2.4d (reproducibility ×3):"
log "     Set ROPE_THETA_FLAG and/or DISABLE_CHUNKED_PREFILL env vars if needed"
log "     ./benchmarks/queue/T2.4d_qwen36_27b_awq_reproducibility.sh"
