# BENCH_27 — T_CV6: Convergence Extended Architecture (ngl=94, GPU0-only)

**Status:** READY
**Blocks:** nothing
**Blocked by:** nothing

---

## Title
T_CV6 — Remove coder from GPU0 and run Convergence at full ngl=94 (all 94 attention layers on GPU0) with thinker on GPU1; measure TPS and confirm thinker is unaffected.

## Objective
Validate the "Convergence Extended" topology: instead of Arclight hot-pair + co-load Convergence, dedicate GPU0 entirely to Convergence at near-isolated performance (~13.99 t/s) while thinker stays on GPU1. This is the upper-bound reference for Convergence TPS with thinker present, and determines whether removing the coder from hot-pair is worth it for agentic orchestration sessions that rely heavily on Convergence.

## Why this exists

Current co-load Convergence TPS is 4.05 t/s (BENCH_21, -ngl 15 with PQ coder on GPU0). Convergence VRAM model from BENCH_21: 6,247 MiB fixed + 113 MiB/layer. At ngl=94: 6,247 + 94×113 = **16,869 MiB ≈ 16.9 GB** — fits on GPU0 alone (32 GB) with 15 GB headroom. Expected TPS ≈ 13.99 t/s (same as T_CV3 isolated singularity mode, no GPU contention with thinker since thinker uses GPU1 only).

**Trade-off context:** At N=1, thinker (91.9 t/s) is 63% faster than coder (56.5 t/s), so removing coder degrades agentic batch throughput but coder can be recovered via Extended Arclight (CRIU swap). This benchmark measures the TPS upside before deciding the trade-off is worth it.

## Context to read

Before running anything, read these files in order:

1. `docs/INDEX.md` — port assignments, key gotchas
2. `docs/arch/convergence.md` — Convergence launch command, GGML_CUDA_NO_PINNED=1 requirement, model path
3. `docs/procedures/vllm-deploy.md` — thinker deploy command (needed if thinker is not running)
4. `results/T_CV3_convergence_gpu_expert_offload_*/summary.md` — isolated TPS baseline (13.99 t/s)
5. `results/BENCH_21_*/summary.md` — co-load VRAM model (6,247 MiB fixed, 113 MiB/layer)

## Prerequisites

```bash
echo "=== BENCH_27 Prerequisites ===" && \

# 1. ik_llama.cpp binary
IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
[ -x "${IK_BIN}" ] && echo "[prereq] llama-server OK" \
  || { echo "[prereq] STOP: ${IK_BIN} not found"; exit 1; } && \

# 2. Convergence GGUF
CONV_GGUF="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
[ -f "${CONV_GGUF}" ] && echo "[prereq] Convergence GGUF OK" \
  || { echo "[prereq] STOP: GGUF not found at ${CONV_GGUF}"; exit 1; } && \

# 3. GPU VRAM — GPU0 must have ~17 GB free (coder must be stopped)
GPU0_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0 | tr -d ' ')
echo "[prereq] GPU0 free: ${GPU0_FREE} MiB (need ≥ 17,000 MiB for ngl=94)" && \
[ "${GPU0_FREE}" -ge 17000 ] \
  && echo "[prereq] GPU0 VRAM OK" \
  || echo "[prereq] WARNING: GPU0 only ${GPU0_FREE} MiB free — stop coder before proceeding" && \

# 4. Thinker health check
curl -sf http://localhost:30001/health 2>/dev/null \
  && echo "[prereq] Thinker OK at :30001" \
  || echo "[prereq] WARNING: Thinker not running — start before Phase 2 if needed" && \

# 5. No existing Convergence on port 8002
ss -tlnp | grep -q ':8002' \
  && echo "[prereq] WARNING: Convergence already running at :8002 — stop it first" \
  || echo "[prereq] port 8002 free"
```

## Inputs required

- Coder container stopped (GPU0 must have ≥ 17 GB free)
- Thinker running on vLLM GPU1 at port 30001
- ik_llama.cpp binary at `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- Convergence UD-IQ2_M GGUF (all 4 shards in same directory)

## Fixed controls

| Control | Value |
|---------|-------|
| Model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M (same as production) |
| Engine | ik_llama.cpp main |
| ngl | 94 (all attention layers — GPU0 only) |
| CPU MoE | `--cpu-moe` (MoE experts in DDR5 RAM) |
| GPU assignment | `CUDA_VISIBLE_DEVICES=0` (GPU0 only for Convergence) |
| GGML_CUDA_NO_PINNED | `1` |
| Threads | `-t 32` |
| Slots | `-np 1` |
| Context | `-c 4096` |
| Port | `8002` |
| Isolated TPS baseline | 13.99 t/s (T_CV3) |
| Co-load TPS baseline | 4.05 t/s (BENCH_21, -ngl 15 with PQ coder) |
| TPS test prompt | `"List the three laws of thermodynamics in one sentence each."` |
| TPS reps | 5 sequential requests |
| Thinker correctness | th02 EDF task while Convergence is running |

## Single variable under test

**GPU0 occupancy:** Convergence ngl=94 (full GPU0, no coder) vs baseline co-load (-ngl 15, GPU0 shared with PQ coder). Thinker on GPU1 is unchanged in both conditions.

## Procedure

Skip flags:
- `SKIP_CONVERGENCE_DEPLOY=1` — skip startup (use if already running at :8002 with ngl=94)

```bash
set -euo pipefail
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_27_cv6_convergence_extended_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
CONV_GGUF="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"

# Baselines
ISOLATED_TPS=13.99
COLOAD_TPS=4.05

# ===================================================================
# PHASE 1: Stop coder if running, record baseline VRAM
# ===================================================================
echo "=== Pre-deploy VRAM ===" | tee "${RESULTS_DIR}/vram.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram.txt"

# Stop coder if running on port 30000
CODER_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "coder\|30000" | head -1 2>/dev/null || true)
if [ -n "${CODER_CONTAINER}" ]; then
  echo "Stopping coder container: ${CODER_CONTAINER}"
  podman stop "${CODER_CONTAINER}" && podman rm "${CODER_CONTAINER}" 2>/dev/null || true
  sleep 5
  echo "Coder stopped."
fi

# Also stop any ik_llama.cpp process on 8080 (APEX from BENCH_24 if still running)
APEX_PID=$(ss -tlnp 2>/dev/null | grep ':8080' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1 || true)
[ -n "${APEX_PID}" ] && { kill "${APEX_PID}" 2>/dev/null; sleep 3; echo "Stopped APEX server PID ${APEX_PID}"; }

echo "=== Post-coder-stop VRAM ===" | tee -a "${RESULTS_DIR}/vram.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram.txt"
GPU0_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0 | tr -d ' ')
echo "gpu0_free_mib=${GPU0_FREE}" >> "${RESULTS_DIR}/metadata.txt"
[ "${GPU0_FREE}" -lt 17000 ] && { echo "FATAL: GPU0 only ${GPU0_FREE} MiB free — need 17,000 MiB for ngl=94. Stop all GPU0 processes."; exit 1; }

# ===================================================================
# PHASE 2: Deploy Convergence at ngl=94 on GPU0 only
# ===================================================================
SKIP_CONVERGENCE_DEPLOY=${SKIP_CONVERGENCE_DEPLOY:-0}
if [ "${SKIP_CONVERGENCE_DEPLOY}" = "0" ]; then
  echo "=== Starting Convergence at ngl=94 (GPU0 only) ===" | tee -a "${RESULTS_DIR}/metadata.txt"
  CUDA_VISIBLE_DEVICES=0 GGML_CUDA_NO_PINNED=1 "${IK_BIN}" \
    -m "${CONV_GGUF}" \
    -ngl 94 --cpu-moe \
    -b 4096 -ub 2048 -t 32 -np 1 -c 4096 \
    --jinja --host 0.0.0.0 --port 8002 \
    >> "${RESULTS_DIR}/convergence.log" 2>&1 &
  CONV_PID=$!

  tail -f "${RESULTS_DIR}/convergence.log" | stdbuf -oL sed 's/\r//g; s/^/[convergence] /' &
  TAIL_PID=$!
  trap 'kill ${TAIL_PID} 2>/dev/null; kill ${CONV_PID} 2>/dev/null' EXIT

  echo "Waiting for Convergence health (up to 120s)..."
  for i in $(seq 1 120); do
    curl -sf http://localhost:8002/health 2>/dev/null && echo "[convergence] HEALTH OK (ngl=94)" && break
    sleep 1
  done
  kill "${TAIL_PID}" 2>/dev/null

  curl -sf http://localhost:8002/health \
    || { echo "FATAL: Convergence failed. Check ${RESULTS_DIR}/convergence.log"; exit 1; }

  # Extract actual layer allocation from startup log
  echo "=== GPU layer allocation ===" | tee "${RESULTS_DIR}/layer_alloc.txt"
  grep -i "layer\|offload\|cuda\|vram\|device" "${RESULTS_DIR}/convergence.log" | head -30 | tee -a "${RESULTS_DIR}/layer_alloc.txt"
else
  echo "[skip] SKIP_CONVERGENCE_DEPLOY=1"
  curl -sf http://localhost:8002/health || { echo "FATAL: not live"; exit 1; }
fi

echo "=== VRAM after Convergence load (ngl=94) ===" | tee -a "${RESULTS_DIR}/vram.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram.txt"

# ===================================================================
# PHASE 3: Convergence TPS (5 warm requests)
# ===================================================================
echo "=== Convergence TPS ===" | tee "${RESULTS_DIR}/convergence_tps.txt"
echo "baseline_isolated=${ISOLATED_TPS} baseline_pq_coload=${COLOAD_TPS}" >> "${RESULTS_DIR}/convergence_tps.txt"

CONV_MODEL=$(curl -sf http://localhost:8002/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "convergence")
TEST_PROMPT="List the three laws of thermodynamics in one sentence each."

TOTAL_TPS=0
for REP in 1 2 3 4 5; do
  START_MS=$(date +%s%3N)
  RESP=$(curl -sf http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${CONV_MODEL}\",\"prompt\":\"${TEST_PROMPT}\",\"max_tokens\":80,\"temperature\":0.0}" 2>/dev/null)
  END_MS=$(date +%s%3N)
  ELAPSED_S=$(python3 -c "print(round(($END_MS - $START_MS) / 1000.0, 2))")
  TOKENS=$(echo "${RESP}" | python3 -c "
import sys,json
try: print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))
except: print(0)
" 2>/dev/null || echo 0)
  TPS_R=$(python3 -c "print(round(${TOKENS} / max(${ELAPSED_S},0.1), 2))")
  TOTAL_TPS=$(python3 -c "print(round(${TOTAL_TPS} + ${TPS_R}, 2))")
  echo "rep=${REP} elapsed=${ELAPSED_S}s tokens=${TOKENS} tps=${TPS_R}" | tee -a "${RESULTS_DIR}/convergence_tps.txt"
  echo "${RESP}" > "${RESULTS_DIR}/conv_rep_${REP}.json"
done
AVG_TPS=$(python3 -c "print(round(${TOTAL_TPS}/5, 2))")
echo "avg_tps=${AVG_TPS}" | tee -a "${RESULTS_DIR}/convergence_tps.txt"

# ===================================================================
# PHASE 4: Thinker correctness (th02 while Convergence is running)
# ===================================================================
echo "=== Thinker correctness check (concurrent with Convergence) ===" | tee "${RESULTS_DIR}/thinker_check.txt"

THINKER_MODEL=$(curl -sf http://localhost:30001/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "thinker")
TH02_PROMPT="Implement a complete Earliest Deadline First (EDF) scheduler in Python. Requirements:
1. Task class with: task_id, arrival_time, execution_time, deadline attributes
2. EDF scheduler always selecting earliest-deadline task
3. Test with: [(0,3,5), (1,2,4), (2,1,3), (3,4,8)] — task (2,1,3) should run first (deadline=3)
Provide complete runnable Python code."

TH02_START=$(date +%s%3N)
TH02_RESP=$(curl -sf http://localhost:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${THINKER_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${TH02_PROMPT}\"}],\"max_tokens\":4096,\"temperature\":0.0}" 2>/dev/null)
TH02_END=$(date +%s%3N)
echo "${TH02_RESP}" > "${RESULTS_DIR}/thinker_th02.json"
echo "thinker th02 elapsed: $(python3 -c "print(round(($TH02_END - $TH02_START)/1000.0,1))")s" | tee -a "${RESULTS_DIR}/thinker_check.txt"

# ===================================================================
# PHASE 5: Thinker VRAM headroom at util=0.73 (model floor)
# ===================================================================
echo "=== GPU1 VRAM headroom (thinker model floor) ===" | tee "${RESULTS_DIR}/gpu1_headroom.txt"
GPU1_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=1 | tr -d ' ')
GPU1_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=1 | tr -d ' ')
echo "gpu1_used_mib=${GPU1_USED} gpu1_free_mib=${GPU1_FREE}" | tee -a "${RESULTS_DIR}/gpu1_headroom.txt"
# Compute KV token capacity at fp8 vs bf16
python3 << PYTHON | tee -a "${RESULTS_DIR}/gpu1_headroom.txt"
free_mib = ${GPU1_FREE}
# Thinker: Qwen3.6-27B, 28 layers, 16 KV heads per layer, 128 head_dim
# fp8 KV: 1 byte/element; bf16: 2 bytes/element
# KV cache per token = 2 (K+V) * n_layers * n_kv_heads * head_dim * bytes_per_element
n_layers = 28
n_kv_heads = 16  # GQA, actual KV heads
head_dim = 128
fp8_bytes = 1
bf16_bytes = 2
per_token_fp8 = 2 * n_layers * n_kv_heads * head_dim * fp8_bytes  # bytes
per_token_bf16 = 2 * n_layers * n_kv_heads * head_dim * bf16_bytes
free_bytes = free_mib * 1024 * 1024
tokens_fp8 = int(free_bytes / per_token_fp8)
tokens_bf16 = int(free_bytes / per_token_bf16)
print(f"GPU1 free: {free_mib} MiB")
print(f"KV per token fp8:  {per_token_fp8} bytes")
print(f"KV per token bf16: {per_token_bf16} bytes")
print(f"Token capacity fp8:  {tokens_fp8:,} tokens ({tokens_fp8//1024}K)")
print(f"Token capacity bf16: {tokens_bf16:,} tokens ({tokens_bf16//1024}K)")
print(f"Note: GDN thinker uses O(d) recurrent state not O(n) KV — actual KV pool is barely used (T3.1: 0 MiB delta at 50K).")
PYTHON

echo "=== BENCH_27 complete === Results in: ${RESULTS_DIR}"
echo "IMPORTANT: Coder is now stopped. To restore Arclight hot-pair, restart coder manually."
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "${RESULTS_DIR}/vram.txt"
```

## Metrics to record

| Metric | Source file | Expected / reference value |
|--------|-------------|---------------------------|
| Convergence avg TPS at ngl=94 (t/s) | `convergence_tps.txt` | Target ~13.99 t/s (= T_CV3 isolated) |
| TPS improvement vs co-load baseline | computed | Should be ~3.5× (13.99 / 4.05) |
| GPU0 VRAM after load (MiB) | `vram.txt` | ~16,869 MiB (6,247 + 94×113) |
| GPU0 VRAM free after load (MiB) | `vram.txt` | ~15,900 MiB headroom |
| Actual ngl confirmed from startup log | `layer_alloc.txt` | 94 layers on GPU0 only |
| Thinker th02 correct | `thinker_th02.json` | PASS — EDF logic intact under load |
| GPU1 KV token capacity (fp8) | `gpu1_headroom.txt` | Informational for T_ARCH3 |
| GPU1 KV token capacity (bf16) | `gpu1_headroom.txt` | Informational for T_ARCH3 |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Convergence starts at ngl=94 | `/health` 200 | Check GPU0 VRAM; ngl=94 should fit in 32 GB |
| Convergence avg TPS ≥ 12 t/s | PASS — confirms topology value | < 10 t/s: check layer allocation, PCIe to thinker |
| Convergence avg TPS ≥ 13.5 t/s | STRONG PASS — nearly identical to isolated | |
| Thinker th02 correct | PASS — no PCIe contention impact | FAIL: record GPU1 VRAM |
| GPU0 VRAM ≤ 18,000 MiB | Confirms projection | > 18,000: recompute VRAM model |

## Artifacts to write

1. `results/BENCH_27_cv6_convergence_extended_<TIMESTAMP>/metadata.txt`
2. `results/BENCH_27_cv6_convergence_extended_<TIMESTAMP>/vram.txt`
3. `results/BENCH_27_cv6_convergence_extended_<TIMESTAMP>/convergence.log`
4. `results/BENCH_27_cv6_convergence_extended_<TIMESTAMP>/layer_alloc.txt`
5. `results/BENCH_27_cv6_convergence_extended_<TIMESTAMP>/convergence_tps.txt`
6. `results/BENCH_27_cv6_convergence_extended_<TIMESTAMP>/thinker_check.txt` + `thinker_th02.json`
7. `results/BENCH_27_cv6_convergence_extended_<TIMESTAMP>/gpu1_headroom.txt`
8. `results/BENCH_27_cv6_convergence_extended_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_27 — T_CV6: Convergence Extended (ngl=94, GPU0-only) — <TIMESTAMP>

## Environment
- Convergence: UD-IQ2_M on ik_llama.cpp, CUDA_VISIBLE_DEVICES=0, ngl=94, --cpu-moe
- Thinker: rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm on vLLM GPU1 :30001
- Coder: STOPPED (GPU0 fully freed for Convergence)

## VRAM
| GPU | Pre-Convergence | Post-Convergence | Convergence VRAM delta |
|-----|----------------|-----------------|----------------------|
| 0   | <x> MiB        | <x> MiB         | <x> MiB              |
| 1   | <x> MiB (thinker) | <x> MiB      | <x> MiB (unchanged)  |

## GPU layer allocation (from startup log)
- ngl confirmed: <N> layers
- All on GPU0: YES/NO (note if any layers on GPU1 despite CUDA_VISIBLE_DEVICES=0)

## Convergence TPS
| Rep | Tokens | Elapsed (s) | TPS |
|-----|--------|-------------|-----|
| 1   | <x>    | <x>         | <x> |
| 2   | <x>    | <x>         | <x> |
| 3   | <x>    | <x>         | <x> |
| 4   | <x>    | <x>         | <x> |
| 5   | <x>    | <x>         | <x> |
| **avg** | | | **<x> t/s** |

| Metric | Value |
|--------|-------|
| Avg TPS (ngl=94, GPU0-only) | <x> t/s |
| Isolated baseline (T_CV3) | 13.99 t/s |
| Co-load baseline (BENCH_21) | 4.05 t/s |
| Improvement vs co-load | <x>× |
| % of isolated performance | <x>% |

## Thinker correctness
- th02 elapsed: <x>s, EDF logic correct: PASS / FAIL / UNSCORED

## GPU1 KV headroom (for T_ARCH3)
- GPU1 free at thinker model floor: <x> MiB
- Token capacity fp8: <x>K tokens
- Token capacity bf16: <x>K tokens
- Note: GDN architecture barely uses KV — these numbers are theoretical ceiling only.

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| Convergence starts ngl=94 | PASS/FAIL | |
| Convergence TPS ≥ 12 t/s | PASS/FAIL | Actual: <x> t/s |
| Thinker th02 correct | PASS/FAIL/UNSCORED | |

## Verdict
PASS / FAIL / PARTIAL — <one sentence>

## Incidental findings
<Layer allocation surprise, VRAM vs projection mismatch, PCIe contention evidence.>
<If nothing: "none">

## Open from testing
<If TPS < 10 t/s or any interference: describe for research mode.>
<If nothing: "none">
```

## Interpretation boundary

**You may:** Record TPS, VRAM, layer allocation, thinker correctness. Compute GPU1 KV token capacity (for T_ARCH3 context).

**You may NOT:** Update architecture docs, change the production config, or decide whether "Convergence Extended" should replace Arclight hot-pair — that is research mode.

## Stop condition

**Normal:** Convergence started at ngl=94, 5 TPS reps measured, thinker th02 checked, GPU1 headroom computed, summary written. **IMPORTANT: Coder is stopped at end of this bench — note this in summary.md.**

**Abnormal:** Write `## Open from testing` in `RESEARCH_STATE.md` if:
- OOM at ngl=94: `BENCH_27_OOM: Convergence OOM at ngl=94. GPU0 free was <x> MiB (needed ~17,000 MiB). Startup log: [excerpt].`
- TPS < 10 t/s: `BENCH_27_TPS_LOW: Convergence <x> t/s at ngl=94 (expected ~13.99 t/s). Layer allocation: [from layer_alloc.txt]. PCIe contention with thinker possible (check GPU1 VRAM delta).`
