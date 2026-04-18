#!/usr/bin/env bash
# T2.1b — GLM-4.7-Flash streaming parser patch (PR #37385).
#
# Rebuilds vllm-glm47 with the one-line fix from PR #37385 already baked into
# infra/Containerfile.vllm_glm47. Then confirms the fix holds by:
#   1. Verifying the patch took effect inside the new image (grep diagnostic).
#   2. Non-streaming sanity (curl Task 02 — write_file, 2 args) to confirm the
#      non-streaming path is clean (expected: tool_call present).
#   3. Three-task streaming tool sanity to confirm the streaming crash is gone.
#
# Pass: ≥ 2/3 streaming tasks pass and EngineDeadError does NOT appear.
# If any step fails in an unexpected way, see the "Hand back to research"
# conditions in TESTING_QUEUE.md T2.1b before re-running.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/config/hardware.env"
source "${REPO_ROOT}/.venv/bin/activate"

ITEM_ID="T2.1b_glm47_flash_streaming_parser_patch"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${REPO_ROOT}/results/${ITEM_ID}_${TIMESTAMP}"
MODEL="cyankiwi/GLM-4.7-Flash-AWQ-4bit"
PATCH_IMAGE="vllm-glm47"
ENDPOINT="http://localhost:${PORT_VLLM_TP2_C}/v1"
CTX=32768
GPU_MEM_UTIL=0.85

mkdir -p "${RESULTS_DIR}/raw"
LOG="${RESULTS_DIR}/bench.log"

log() { echo "[T2.1b] $*" | tee -a "${LOG}"; }
die() { log "FATAL: $*"; exit 1; }

# ── Step 1: Rebuild patched image ─────────────────────────────────────────────
log "Building patched ${PATCH_IMAGE} from infra/Containerfile.vllm_glm47 ..."
log "(This bakes in the PR #37385 fix: sed args_dict → full_args_str in streaming parser)"
podman build -f "${REPO_ROOT}/infra/Containerfile.vllm_glm47" -t "${PATCH_IMAGE}" "${REPO_ROOT}" \
    2>&1 | tee -a "${LOG}"
log "Image build complete."

# ── Step 2: Verify patch was applied inside the new image ─────────────────────
log "=== Patch verification ==="
echo 'PARSER=$(find /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/ -name "glm4_moe_tool_parser.py" 2>/dev/null | head -1)
echo "Parser: $PARSER"
if [ -n "$PARSER" ]; then
  echo "--- args_dict references (should be zero after patch):"
  grep -n "\"arguments\": args_dict" "$PARSER" || echo "  (none — patch confirmed)"
  echo "--- current arguments lines:"
  grep -n "\"arguments\"" "$PARSER" | head -10
else
  echo "MISSING: glm4_moe_tool_parser.py not found in image"
fi' | podman run --rm -i --entrypoint bash "${PATCH_IMAGE}" 2>&1 | tee -a "${LOG}"

# ── Step 3: Patch model config.json for MLA dimensions ────────────────────────
log "Checking model config.json for MLA dimensions ..."
SEARCH_PATTERN="${MODEL/\//--}"
CONFIG_JSON=$(find /srv/ai/models/hub -name config.json | grep -i "${SEARCH_PATTERN}" | head -n 1 || true)
if [[ -n "${CONFIG_JSON}" ]]; then
    if ! grep -q "qk_nope_head_dim" "${CONFIG_JSON}"; then
        log "Patching ${CONFIG_JSON} with MLA dimensions ..."
        sed -i 's/"kv_lora_rank":/"qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "kv_lora_rank":/' "${CONFIG_JSON}"
    else
        log "config.json already has MLA fields."
    fi
else
    log "WARNING: config.json not found for ${SEARCH_PATTERN} — MLA may not enable."
fi

# ── Step 4: Deploy GLM-4.7-Flash with patched image ───────────────────────────
export BENCH_IMAGE="${PATCH_IMAGE}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# Single-process V1: parser exceptions are per-request recoverable, not server-fatal.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

log "Deploying ${MODEL} on tp2c (TP=2, patched image) ..."
"${REPO_ROOT}/infra/scripts/deploy.sh" vllm tp2c "${MODEL}" \
    --gpu-mem-util "${GPU_MEM_UTIL}" \
    --ctx "${CTX}" \
    --tool-call-parser glm47 \
    2>&1 | tee -a "${LOG}"

# ── Step 5: Non-streaming sanity — Task 02 (write_file, 2 args) ───────────────
# This isolates the streaming path as the test axis.
# Pass = tool_call present in the response.
log "=== Non-streaming sanity: Task 02 (write_file, 2 args, stream=false) ==="
NONSTREAM_PAYLOAD='{
  "model": "'"${MODEL}"'",
  "stream": false,
  "messages": [{"role": "user", "content": "Create a file at /tmp/hello.txt with content Hello, world!"}],
  "tools": [{"type": "function", "function": {
    "name": "write_file",
    "description": "Write content to a file.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"}
      },
      "required": ["path", "content"]
    }
  }}]
}'
NONSTREAM_RAW=$(curl -s -X POST "${ENDPOINT}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "${NONSTREAM_PAYLOAD}" 2>&1 || echo '{"error":"curl_failed"}')
echo "${NONSTREAM_RAW}" > "${RESULTS_DIR}/raw/nonstream_task02.json"

NONSTREAM_VERDICT=$(python3 - <<PYEOF
import json, sys
raw = open('${RESULTS_DIR}/raw/nonstream_task02.json').read()
try:
    d = json.loads(raw)
    choices = d.get('choices', [])
    tc = choices[0].get('message', {}).get('tool_calls', []) if choices else []
    if tc:
        fn = tc[0]['function']
        print(f"PASS  tool={fn['name']}  args={fn['arguments'][:80]}")
    else:
        content = choices[0].get('message', {}).get('content', '') if choices else ''
        print(f"FAIL  no tool_calls  content_snippet={content[:80]!r}")
except Exception as e:
    print(f"FAIL  parse_error={e}")
PYEOF
)
log "Non-streaming Task 02: ${NONSTREAM_VERDICT}"

NONSTREAM_OK=0
[[ "${NONSTREAM_VERDICT}" == PASS* ]] && NONSTREAM_OK=1

if [[ "${NONSTREAM_OK}" == "0" ]]; then
    log "WARNING: Non-streaming path FAILED. This is unexpected (PR #37386 should have fixed it)."
    log "         If this is a new regression, hand back to research — the fix path is wrong."
fi

# ── Step 6: Three-task streaming tool sanity ──────────────────────────────────
log "=== Streaming tool sanity (3 tasks) ==="
for t_idx in 01 02 03; do
    log "  Running task ${t_idx} ..."
    python3 -m benchmarks.phase0_tool_reliability.bench \
        --endpoint "${ENDPOINT}" \
        --results-dir "${RESULTS_DIR}/tool_sanity/${t_idx}" \
        --tasks "${REPO_ROOT}/benchmarks/phase0_tool_reliability/tasks/" \
        --task-filter "${t_idx}" \
        --max-tokens 2048 \
        2>&1 | tee -a "${LOG}" || log "  WARNING: task ${t_idx} exited non-zero."
done

TOOL_PASS_RATE="0.0"
TASK_RESULTS=()
if compgen -G "${RESULTS_DIR}/tool_sanity/*/metrics.json" > /dev/null 2>&1; then
    TOOL_PASS_RATE=$(python3 -c "
import json, pathlib
rates = []
for p in sorted(pathlib.Path('${RESULTS_DIR}/tool_sanity/').glob('*/metrics.json')):
    d = json.load(p.open())
    r = d['metrics'].get('tool_call_success_rate', 0.0)
    rates.append(r)
    print(f'  task {p.parent.name}: {r:.0%}', end=' ')
    v = d.get('verdict', 'unknown')
    print(f'({v})')
print(round(sum(rates)/len(rates), 2) if rates else 0.0)
" | tee -a "${LOG}" | tail -1)
fi
log "Streaming tool sanity pass rate: ${TOOL_PASS_RATE}"

# Check for EngineDeadError in logs (the exact failure mode from T2.1)
ENGINE_DEAD=$(grep -c "EngineDeadError\|Engine.*dead\|engine.*dead" "${LOG}" || true)
[[ "${ENGINE_DEAD}" -gt 0 ]] && log "WARNING: EngineDeadError still appears in logs (${ENGINE_DEAD} occurrences)"

# ── Step 7: Write metrics and summary ─────────────────────────────────────────
export ITEM_ID TIMESTAMP RESULTS_DIR MODEL CTX GPU_MEM_UTIL TOOL_PASS_RATE \
       NONSTREAM_VERDICT NONSTREAM_OK ENGINE_DEAD

python3 - <<'PYEOF'
import json, pathlib, os

item_id      = os.environ["ITEM_ID"]
timestamp    = os.environ["TIMESTAMP"]
out          = pathlib.Path(os.environ["RESULTS_DIR"])
model        = os.environ["MODEL"]
ctx          = int(os.environ["CTX"])
gpu_util     = float(os.environ["GPU_MEM_UTIL"])
tool_rate    = float(os.environ["TOOL_PASS_RATE"])
ns_ok        = os.environ["NONSTREAM_OK"] == "1"
engine_dead  = int(os.environ.get("ENGINE_DEAD", "0"))
ns_verdict   = os.environ["NONSTREAM_VERDICT"]

streaming_pass = tool_rate >= 0.67
engine_dead_ok = engine_dead == 0

if streaming_pass and engine_dead_ok:
    verdict = "PASS"
    if not ns_ok:
        notes = "Streaming fix confirmed. Non-streaming path failed unexpectedly — check PR #37386 status."
    else:
        notes = "Streaming parser patch effective. Both paths clean. Proceed to T2.2."
elif not ns_ok and not streaming_pass:
    verdict = "FAIL"
    notes = "Both streaming and non-streaming fail. Hand back to research — fix path is wrong."
elif not streaming_pass and engine_dead > 0:
    verdict = "FAIL"
    notes = f"EngineDeadError still occurring ({engine_dead}x). Streaming parser has additional unfixed bugs. Hand back to research with updated stack trace."
elif not streaming_pass:
    verdict = "INCONCLUSIVE"
    notes = f"Streaming rate {tool_rate:.0%} (need ≥67%). No EngineDeadError — errors are different. Check per-task raw results."
else:
    verdict = "INCONCLUSIVE"
    notes = "Partial pass. Review per-task tool_sanity/ results."

metrics = {
    "item_id":   item_id,
    "timestamp": timestamp,
    "config": {
        "engine":         "vllm",
        "engine_version": "0.19.x-nightly (cu130)",
        "model":          model,
        "quantization":   "AWQ-INT4",
        "placement":      "tp=2 (tp2c)",
        "context_length": ctx,
        "gpu_mem_util":   gpu_util,
        "extra_args":     "--tool-call-parser glm47",
        "env_overrides":  "VLLM_ENABLE_V1_MULTIPROCESSING=0",
        "patch":          "PR #37385 (args_dict → full_args_str) via Containerfile sed",
    },
    "metrics": {
        "nonstream_task02_verdict": ns_verdict.split()[0],
        "streaming_tool_sanity_rate": tool_rate,
        "engine_dead_error_count": engine_dead,
    },
    "verdict": verdict,
    "notes":   notes,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))

PASS_STR  = lambda b: "✓ PASS" if b else "✗ FAIL"
md = f"""# T2.1b GLM-4.7-Flash Streaming Parser Patch — {verdict}

**vLLM** 0.19.x-nightly (cu130) | **Model** {model} | **Date** {timestamp[:10]}
**Placement** TP=2 (tp2c) | **Patch** PR #37385 applied in Containerfile
**Safety net** `VLLM_ENABLE_V1_MULTIPROCESSING=0`

| Test | Result | Threshold |
|------|--------|-----------|
| Non-streaming Task 02 (write_file, 2 args) | {PASS_STR(ns_ok)} | tool_call present |
| Streaming tool sanity (3 tasks) | {tool_rate:.0%} pass | ≥ 67% (2/3) |
| EngineDeadError occurrences | {engine_dead} | 0 |

**Non-streaming detail:** `{ns_verdict}`

**Verdict: {verdict}**

{notes}

"""
if verdict == "PASS":
    md += "**Next step:** T2.2 (coder shootout — GLM-4.7-Flash vs Qwen3-Coder-30B). No rebuild needed.\n"
elif "Hand back" in notes:
    md += "**Next step:** Add an `## Open from testing` entry to `RESEARCH_STATE.md` with the new stack trace from `bench.log`, then switch to research mode.\n"

(out / "summary.md").write_text(md)
print(f"[T2.1b] Verdict: {verdict}")
print(f"[T2.1b] Non-streaming Task 02: {PASS_STR(ns_ok)}")
print(f"[T2.1b] Streaming pass rate: {tool_rate:.0%}  EngineDeadError: {engine_dead}")
print(f"[T2.1b] Results: {out}")
PYEOF
