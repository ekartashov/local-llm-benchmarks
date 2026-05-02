# BENCH_14 — T_MTP2: Coder MTP Speculative Decoding (n=1)

**Status: READY (after T_MTP1 / BENCH_13 passes)**
**Blocks: nothing**
**Blocked by: BENCH_13 (T_MTP1 thinker MTP — confirm approach is stable first)**

---

## Title
T_MTP2 — MTP n=1 TPS improvement on production coder (A3B AWQ, TP=2)

## Objective
Measure the TPS impact of enabling native Multi-Token Prediction speculative decoding (`method=mtp`, `num_speculative_tokens=1`) on the production coder (cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, TP=2, GPU0+1, port 30000). Compare aggregate TPS vs the settled AWQ baseline without MTP. Verify tool-call reliability is not degraded.

## Why this exists
vLLM MTP uses the model's own MTP head in the **same forward pass** — no separate draft model, no second expert-routing pass. This is fundamentally different from llama.cpp speculative decoding (which requires a separate draft model and incurs extra expert loading on A3B). If MTP acceptance rates are reasonable on the A3B architecture, this gives free TPS improvement identical to the thinker case.

BENCH_13 (T_MTP1 thinker) confirms MTP is stable on our vLLM 0.19.x stack before touching the coder.

vLLM #40756 (MTP crash) does NOT apply: conditions are FP8+TP=4+n=5+25K tokens. Our config is AWQ+TP=2+n=1.

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — current production config, gotchas, port assignments
2. `docs/procedures/vllm-deploy.md` — deploy commands and speculative-config flag syntax
3. `docs/decisions/models.md` — coder role requirements (tool-call reliability is a hard constraint)

## Prerequisites

```bash
# 1. Coder health
curl -sf http://localhost:30000/health && echo "CODER OK" || echo "CODER DOWN — deploy first"

# 2. Coder model identity
curl -s http://localhost:30000/v1/models \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])"
# Expected: cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit

# 3. GPU0+1 VRAM baseline
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 4. BENCH_13 completed successfully
# Check: results/BENCH_13_mtp1_thinker_*/summary.md — MTP startup SUCCESS, th02 CORRECT
```

**This test runs on the HOST.**

## Inputs required
- Production coder running at `http://localhost:30000` with model `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`
- `infra/scripts/deploy.sh` accessible from the repo root
- `benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh` for the TPS sweep (both steps)
- GPU0+1 VRAM headroom to redeploy (script stops and restarts the container)
- BENCH_13 (T_MTP1 thinker) completed successfully before running this handoff

## Fixed controls

| Control | Value |
|---------|-------|
| Model | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit |
| Engine | vLLM TP=2 GPU0+1 |
| VLLM_USE_V1 | 0 (mandatory) |
| ctx | 32768 |
| KV cache dtype | fp8 |
| chunked-prefill | OFF (not in production coder config) |
| MTP n | **1** (single variable under test) |
| Reps per N | 3 |

## Single variable under test
**`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`** — present vs absent. All other flags identical.

---

## Procedure

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_14_mtp2_coder_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
```

### Step 1 — Record baseline TPS (no MTP, current production coder)

If coder is already running (production state), run the sweep directly:

```bash
# Baseline sweep — no MTP, skip thinker and convergence
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  --skip-thinker \
  --skip-convergence \
  --reps 3

BASELINE_DIR=$(ls -td results/T_PAR1_* | head -1)
echo "Baseline results: ${BASELINE_DIR}"
cp "${BASELINE_DIR}/metrics.json" "${RESULTS_DIR}/metrics_baseline.json"

python3 - <<'EOF'
import json, os
d = json.load(open(os.popen("ls -td results/BENCH_14_mtp2_coder_* | head -1").read().strip() + "/metrics_baseline.json"))
c = d.get("coder_detail", {})
for k in sorted(c):
    if k.endswith("_tps"):
        print(f"  {k}: {c[k]} t/s")
EOF
```

If the coder is not running, deploy the baseline config first:

```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu0gpu1 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --tensor-parallel-size 2 --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

for i in $(seq 1 120); do
  curl -sf http://localhost:30000/health && echo "READY" && break; sleep 1
done
```

Then run the baseline sweep as above.

### Step 2 — Stop coder, redeploy with MTP n=1

```bash
CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|gpu0\|35b" | head -1)
podman stop "${CODER_CONTAINER}" && podman rm "${CODER_CONTAINER}"

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu0gpu1 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --tensor-parallel-size 2 --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'

# Wait up to 120s
for i in $(seq 1 120); do
  curl -sf http://localhost:30000/health && echo "CODER+MTP READY" && break; sleep 1
done
```

**If deploy fails with an error about MTP / speculative-config:**
- Capture container logs: `podman logs $(podman ps -a --format "{{.Names}}" | grep -i "coder\|35b" | head -1) 2>&1 | tail -40`
- Save to `${RESULTS_DIR}/mtp_startup_failure.txt`
- Write `## Open from testing` in RESEARCH_STATE.md
- Proceed to Step 5 to restore production coder without MTP.

**If MTP starts but VRAM exceeds 30 GiB per GPU:**
- Record VRAM in `${RESULTS_DIR}/vram_mtp.txt`
- Continue with sweep (not an error, just higher pressure)

### Step 3 — TPS sweep with MTP n=1

```bash
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh \
  --skip-thinker \
  --skip-convergence \
  --reps 3

MTP_DIR=$(ls -td results/T_PAR1_* | head -1)
echo "MTP results: ${MTP_DIR}"
cp "${MTP_DIR}/metrics.json" "${RESULTS_DIR}/metrics_mtp_n1.json"

# Print comparison
python3 - <<'EOF'
import json, os
b = json.load(open(os.popen("ls -td results/BENCH_14_mtp2_coder_* | head -1").read().strip() + "/metrics_baseline.json"))
m = json.load(open(os.popen("ls -td results/T_PAR1_* | head -1").read().strip() + "/metrics.json"))
bc = b.get("coder_detail", {})
mc = m.get("coder_detail", {})
ns = sorted(set(k.replace("_tps","") for k in bc if k.endswith("_tps")))
for n in ns:
    bv = bc.get(f"{n}_tps"); mv = mc.get(f"{n}_tps")
    delta = round((mv-bv)/bv*100, 1) if bv and mv else "N/A"
    label = n.replace("n","N=")
    print(f"{label}: baseline={bv} t/s  mtp={mv} t/s  delta={delta}%")
EOF
```

### Step 4 — Tool-call quality smoke check

The coder's primary quality gate is tool-call reliability. Send a representative tool-call prompt and verify the response is well-formed:

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
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
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
msg = r['choices'][0]['message']
if msg.get('tool_calls'):
    tc = msg['tool_calls'][0]
    print('TOOL CALL: OK')
    print(f'  function: {tc[\"function\"][\"name\"]}')
    print(f'  args: {tc[\"function\"][\"arguments\"]}')
else:
    print('TOOL CALL: MISSING — got text response instead')
    print(msg.get('content','')[:200])
" | tee "${RESULTS_DIR}/toolcall_check.txt"
```

Repeat 3 times and record how many out of 3 produce a well-formed tool call.

**Pass condition:** 3/3 tool calls well-formed (function name correct, args valid JSON).
A malformed response or missing tool_calls key means MTP is interfering with the structured output path.

If the full co01 suite is available:
```bash
python3 -m benchmarks.phase2_model_selection.bench \
  --endpoint http://localhost:30000/v1 \
  --results-dir "${RESULTS_DIR}/quality_mtp" \
  --mode quality \
  --tasks benchmarks/phase2_model_selection/tasks/coder/ \
  --task-filter co01 \
  --model cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --label "AWQ-MTP-n1-coder" \
  --max-tokens 4096
```

### Step 5 — Restore production coder (MANDATORY)

```bash
CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|35b" | head -1)
[ -n "${CODER_CONTAINER}" ] && podman stop "${CODER_CONTAINER}" && podman rm "${CODER_CONTAINER}"

VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu0gpu1 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --tensor-parallel-size 2 --gpu-mem-util 0.90 --ctx 32768 --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice

for i in $(seq 1 120); do
  curl -sf http://localhost:30000/health && echo "PRODUCTION CODER RESTORED" && break; sleep 1
done
```

---

## Metrics to record

| Metric | Source | AWQ baseline (no MTP) |
|--------|--------|-----------------------|
| N=1 TPS (baseline) | metrics_baseline.json | ~240 t/s |
| N=1 TPS (MTP n=1) | metrics_mtp_n1.json | — |
| N=1 TPS delta % | computed | **target: +5–25%** |
| N=4 TPS (baseline) | metrics_baseline.json | — |
| N=4 TPS (MTP n=1) | metrics_mtp_n1.json | — |
| N=4 TPS delta % | computed | may be flat or negative under load |
| N=8 TPS (baseline) | metrics_baseline.json | ~1205 t/s |
| N=8 TPS (MTP n=1) | metrics_mtp_n1.json | — |
| Tool-call pass (3 probes) | toolcall_check.txt | 3/3 |
| GPU0+1 VRAM with MTP | nvidia-smi after Step 2 deploy | ~23 GiB/GPU expected |
| MTP startup success | deploy log | — |

**Note on expected gain:** The A3B MoE architecture activates ~3B of 35B parameters per token. MTP head overhead is the same regardless of sparsity, but acceptance rates may differ from a dense model. +5% is the minimum useful signal; anything below that is noise.

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| MTP deploy succeeds | HEALTH_OK within 120s | Capture error; write Open from testing; skip to Step 5 |
| N=1 TPS improvement | ≥ +5% vs baseline | Note marginal / no gain; still worth recording N=4/N=8 impact |
| N=8 TPS not severely degraded | > −20% vs baseline ~1205 t/s | Note; at high concurrency MTP may thrash KV cache |
| Tool-call 3/3 | All 3 probes return well-formed tool_calls | FAIL — MTP corrupts structured output; do not enable |
| Production restored | /health 200 on port 30000 | Redeploy manually |

**If N=1 improves but high-N degrades:** expected — speculative tokens consume KV slots, reducing effective batch headroom under concurrency. Trade-off decision for research review, not for this handoff.

**If gain is <5% at N=1:** MTP acceptance rate on A3B MoE is likely too low for meaningful improvement. This is the expected risk vs thinker (dense). Record accurately; do not retry with n=2 (out of scope).

---

## Artifacts to write

1. `results/BENCH_14_mtp2_coder_<timestamp>/metrics_baseline.json` — copy of T_PAR1 baseline
2. `results/BENCH_14_mtp2_coder_<timestamp>/metrics_mtp_n1.json` — copy of T_PAR1 MTP run
3. `results/BENCH_14_mtp2_coder_<timestamp>/toolcall_check.txt` — 3-probe tool-call output
4. `results/BENCH_14_mtp2_coder_<timestamp>/vram_mtp.txt` — `nvidia-smi` output after MTP deploy
5. `results/BENCH_14_mtp2_coder_<timestamp>/summary.md`:

```markdown
# BENCH_14 — T_MTP2 Coder MTP n=1 — <TIMESTAMP>

## TPS comparison
| N | Baseline (no MTP) | MTP n=1 | Delta |
|---|-------------------|---------|-------|
| 1 | ~240 t/s | <X> t/s | <±%> |
| 4 | <X> t/s | <X> t/s | <±%> |
| 8 | ~1205 t/s | <X> t/s | <±%> |

## Quality
Tool-call probe: <X>/3 well-formed

## GPU0+1 VRAM with MTP
GPU0: <X> MiB  GPU1: <X> MiB (baseline: ~23,000 MiB/card)

## MTP startup
SUCCESS / FAILED (see mtp_startup_failure.txt)

## Incidental findings
<Any observation outside this benchmark's explicit scope: unexpected VRAM readings, other components
behaving differently than documented, engine warnings about kernels or flags, etc. Write one FINDING
block per observation. If nothing unusual observed: "none">

## Open from testing
<only if stopping abnormally — describe the block>

## Verdict
PASS / FAIL / MARGINAL / NO-GAIN
```

**Do NOT write to any file outside `results/BENCH_14_mtp2_coder_<timestamp>/`.**

---

## Interpretation boundary

- **You may record** TPS delta at each N, VRAM, tool-call result, startup success.
- **You may note** whether A3B MoE shows a gain comparable to the dense thinker, and whether high-N degrades.
- **You may NOT** update `docs/decisions/settled.md`, production config, or queue files.
- **You may NOT** conclude whether MTP should be permanently enabled in production.
- **You may NOT** test n=2 or n=3 — that is out of scope for this handoff.

## Stop condition

**Normal:** TPS comparison written, tool-call result recorded, production coder restored, summary.md written.

**Abnormal:** write `## Open from testing` in `RESEARCH_STATE.md` and stop if:
- MTP deploy fails with an unrecognised error
- Tool-call probe fails with a new error pattern (not a simple wrong-answer)
- N=1 TPS is worse than −30% vs baseline (unexpected regression, not just concurrency effect)
- TP=2 + MTP triggers a CUDA error not seen in BENCH_13 (different failure mode than thinker TP=1)
