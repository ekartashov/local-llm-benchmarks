# BENCH_29 — T_KV5: Convergence KV Ceiling Sweep (fp8 vs bf16 to 256K)

**Status:** READY
**Blocks:** nothing
**Blocked by:** nothing

---

## Title
T_KV5 — Ramp Convergence context from 4K to 256K for both fp8 (`--kv-type q8_0`) and bf16 (`--kv-type f16`) KV; record OOM point, TPS decay, and VRAM growth at each breakpoint.

## Objective
Determine the practical KV context ceiling for Convergence UD-IQ2_M on ik_llama.cpp and choose the production KV precision for long-context use cases. fp8 halves KV memory vs bf16; if quality is equivalent (expected — Convergence is a reasoning orchestration model, not a precision-critical code generator), fp8 enables 2× longer effective context within the same VRAM budget.

## Why this exists

Convergence GPU VRAM at ngl=94 (T_CV6): ~17 GB. Available headroom: ~15 GB. KV cache at fp8 for Convergence (Qwen3.5-397B-A17B: 128 KV heads, 128 head_dim, 94 attention layers):
- Per token fp8: 2 × 94 × 128 × 128 × 1 byte = 3,080,192 bytes ≈ 2.94 MB/token
- Per token bf16: 2 × 94 × 128 × 128 × 2 bytes = 6,160,384 bytes ≈ 5.88 MB/token
- At 256K tokens fp8: 256K × 2.94 MB ≈ 752 GB — clearly exceeds VRAM. The real limit will be much lower.
- At 15 GB headroom: fp8 ceiling ≈ 15,360 MB / 2.94 MB/tok ≈ **5,224 tokens**; bf16 ≈ **2,612 tokens**

These numbers suggest the KV ceiling is relatively low for Convergence (large model, many attention layers). This benchmark confirms the real numbers and identifies the OOM point.

**Context:** Convergence is always single-sequence (N=1 is hard architectural limit). Long-context is relevant for orchestration sessions with large tool outputs or multi-doc reasoning.

## Context to read

Before running anything, read these files in order:

1. `docs/arch/convergence.md` — production launch command, GGML_CUDA_NO_PINNED=1, model path
2. `results/BENCH_27_cv6_convergence_extended_*/summary.md` — if available: GPU0 VRAM after ngl=94 load (determines actual headroom for KV); if not yet run, use 15 GB estimate
3. `results/T_CV3_*/summary.md` — isolated TPS baseline (13.99 t/s at -c 4096)

## Prerequisites

```bash
echo "=== BENCH_29 Prerequisites ===" && \

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
[ -x "${IK_BIN}" ] && echo "[prereq] llama-server OK" \
  || { echo "[prereq] STOP: binary not found"; exit 1; } && \

CONV_GGUF="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
[ -f "${CONV_GGUF}" ] && echo "[prereq] Convergence GGUF OK" \
  || { echo "[prereq] STOP: GGUF not found"; exit 1; } && \

# GPU0 should have ≥17 GB free (coder stopped; Convergence will load its own weights)
GPU0_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0 | tr -d ' ')
echo "[prereq] GPU0 free: ${GPU0_FREE} MiB" && \
[ "${GPU0_FREE}" -ge 17000 ] \
  && echo "[prereq] GPU0 VRAM OK for ngl=94" \
  || echo "[prereq] WARNING: only ${GPU0_FREE} MiB — may need to reduce ngl or stop other processes" && \

ss -tlnp | grep -q ':8002' \
  && echo "[prereq] WARNING: Convergence already running on :8002 — stop it first" \
  || echo "[prereq] port 8002 free"
```

## Inputs required

- ik_llama.cpp binary at `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- Convergence UD-IQ2_M GGUF (all 4 shards in same directory)
- GPU0 free (coder and any APEX server stopped)
- Python3 available on host for prompt generation

## Fixed controls

| Control | Value |
|---------|-------|
| Model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M |
| Engine | ik_llama.cpp main |
| ngl | 94 (GPU0 only, `CUDA_VISIBLE_DEVICES=0`) |
| CPU MoE | `--cpu-moe` |
| GGML_CUDA_NO_PINNED | `1` |
| Threads | `-t 32` |
| Slots | `-np 1` |
| Port | `8002` |
| Context ramp | 4096, 8192, 16384, 32768, 65536, 131072, 262144 |
| TPS prompt | fixed-length prompt padded to ~75% of context size (see procedure) |
| max_tokens | 80 (short — only measuring decode TPS) |
| Temperature | 0.0 |
| KV types | `f16` (bf16) then `q8_0` (fp8) |

## Single variable under test

**KV type** (`f16` vs `q8_0`) at each context size. Within each KV type, context size is swept from 4K to 256K until OOM.

## Procedure

Skip flags:
- `SKIP_BF16=1` — skip bf16 sweep (test fp8 only)
- `SKIP_FP8=1` — skip fp8 sweep (test bf16 only)

```bash
set -euo pipefail
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/BENCH_29_kv5_convergence_kv_ceiling_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

IK_BIN="/srv/ai/projects/ik_llama.cpp/build/bin/llama-server"
CONV_GGUF="/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"

echo "kv_type,ctx_tokens,prompt_tokens,output_tokens,elapsed_s,tps,vram_mib,status" \
  > "${RESULTS_DIR}/kv_ceiling_sweep.csv"

stop_server() {
  local PID=$(ss -tlnp 2>/dev/null | grep ':8002' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1 || true)
  [ -n "${PID}" ] && { kill "${PID}" 2>/dev/null; sleep 5; }
}

# Generate a prompt of approximately N tokens (rough estimate: 1 word ≈ 1.3 tokens)
make_prompt() {
  local TARGET_TOKENS=$1
  local WORDS=$((TARGET_TOKENS * 3 / 4))
  python3 -c "
words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', 'and', 'then', 'returns', 'to', 'its', 'den', 'where', 'it', 'rests', 'peacefully', 'under', 'the', 'stars']
result = ' '.join(words * (${WORDS} // len(words) + 1))
print(result[:${WORDS}*5])
print('\nSummarize this text in one sentence.')
"
}

run_sweep() {
  local KV_TYPE=$1
  local KV_FLAG=$2  # --kv-type f16 or --kv-type q8_0
  echo "=== KV sweep: ${KV_TYPE} ===" | tee "${RESULTS_DIR}/sweep_${KV_TYPE}.log"

  for CTX in 4096 8192 16384 32768 65536 131072 262144; do
    echo "--- ctx=${CTX} kv=${KV_TYPE} ---" | tee -a "${RESULTS_DIR}/sweep_${KV_TYPE}.log"

    # Start server for this context size
    stop_server
    CUDA_VISIBLE_DEVICES=0 GGML_CUDA_NO_PINNED=1 "${IK_BIN}" \
      -m "${CONV_GGUF}" \
      -ngl 94 --cpu-moe \
      -b 4096 -ub 2048 -t 32 -np 1 -c "${CTX}" \
      --kv-type "${KV_FLAG#--kv-type }" \
      --jinja --host 0.0.0.0 --port 8002 \
      >> "${RESULTS_DIR}/server_${KV_TYPE}_ctx${CTX}.log" 2>&1 &
    SRV_PID=$!

    # Wait up to 120s for health
    STARTED=0
    for i in $(seq 1 120); do
      curl -sf http://localhost:8002/health 2>/dev/null && STARTED=1 && break
      # Check if process died (OOM during startup)
      kill -0 "${SRV_PID}" 2>/dev/null || { echo "Server died (OOM at ctx=${CTX})"; break; }
      sleep 1
    done

    if [ "${STARTED}" = "0" ]; then
      echo "OOM_STARTUP: ctx=${CTX}" | tee -a "${RESULTS_DIR}/sweep_${KV_TYPE}.log"
      echo "${KV_TYPE},${CTX},N/A,N/A,N/A,N/A,N/A,OOM_STARTUP" >> "${RESULTS_DIR}/kv_ceiling_sweep.csv"
      kill "${SRV_PID}" 2>/dev/null; sleep 3
      break  # Stop sweeping this KV type — larger contexts will also OOM
    fi

    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 | tr -d ' ')

    # Make prompt at ~75% of context size
    PROMPT_TOKENS=$((CTX * 3 / 4))
    PROMPT=$(make_prompt "${PROMPT_TOKENS}")
    CONV_MODEL=$(curl -sf http://localhost:8002/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "convergence")

    START_MS=$(date +%s%3N)
    RESP=$(curl -sf http://localhost:8002/v1/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${CONV_MODEL}\",\"prompt\":$(python3 -c "import json; print(json.dumps('${PROMPT}'))"),\"max_tokens\":80,\"temperature\":0.0}" \
      --max-time 300 2>/dev/null || echo "TIMEOUT")
    END_MS=$(date +%s%3N)

    if [ "${RESP}" = "TIMEOUT" ] || [ -z "${RESP}" ]; then
      echo "OOM_INFERENCE or TIMEOUT: ctx=${CTX}" | tee -a "${RESULTS_DIR}/sweep_${KV_TYPE}.log"
      echo "${KV_TYPE},${CTX},${PROMPT_TOKENS},N/A,N/A,N/A,${VRAM},OOM_INFERENCE" >> "${RESULTS_DIR}/kv_ceiling_sweep.csv"
      stop_server
      break
    fi

    ELAPSED_S=$(python3 -c "print(round(($END_MS - $START_MS)/1000.0,2))")
    OUT_TOKENS=$(echo "${RESP}" | python3 -c "
import sys,json
try: print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))
except: print(0)
" 2>/dev/null || echo 0)
    PROMPT_ACTUAL=$(echo "${RESP}" | python3 -c "
import sys,json
try: print(json.load(sys.stdin).get('usage',{}).get('prompt_tokens',0))
except: print(0)
" 2>/dev/null || echo "${PROMPT_TOKENS}")
    TPS=$(python3 -c "print(round(${OUT_TOKENS}/max(${ELAPSED_S},0.1),2))")

    echo "${KV_TYPE},${CTX},${PROMPT_ACTUAL},${OUT_TOKENS},${ELAPSED_S},${TPS},${VRAM},OK" \
      >> "${RESULTS_DIR}/kv_ceiling_sweep.csv"
    echo "ctx=${CTX} prompt=${PROMPT_ACTUAL}tok out=${OUT_TOKENS}tok elapsed=${ELAPSED_S}s tps=${TPS} vram=${VRAM}MiB" \
      | tee -a "${RESULTS_DIR}/sweep_${KV_TYPE}.log"

    stop_server
  done
}

SKIP_BF16=${SKIP_BF16:-0}
SKIP_FP8=${SKIP_FP8:-0}

[ "${SKIP_BF16}" = "0" ] && run_sweep "bf16" "--kv-type f16"
[ "${SKIP_FP8}" = "0" ]  && run_sweep "fp8"  "--kv-type q8_0"

echo "=== Final sweep results ===" | tee -a "${RESULTS_DIR}/kv_ceiling_sweep.csv"
cat "${RESULTS_DIR}/kv_ceiling_sweep.csv"
echo "=== BENCH_29 complete === Results in: ${RESULTS_DIR}"
```

**Note on prompt generation:** The prompt must fill ~75% of `-c CTX` to actually stress the KV cache. If the server returns `400 context_length_exceeded`, the prompt was too long — reduce to 60% and retry that context size.

## Metrics to record

| Metric | Source file | Expected |
|--------|-------------|---------|
| OOM context size — bf16 | `kv_ceiling_sweep.csv` | Expected OOM around 4K–32K (only ~15 GB KV headroom at ngl=94) |
| OOM context size — fp8 | `kv_ceiling_sweep.csv` | Expected ~2× bf16 ceiling |
| TPS at 4K ctx (baseline) | `kv_ceiling_sweep.csv` | ~13.99 t/s (T_CV3 reference) |
| TPS decay rate per context doubling | computed | DDR5-bound: TPS roughly constant until OOM |
| VRAM growth per context doubling | `kv_ceiling_sweep.csv` | Doubles for bf16, halves for fp8 vs bf16 at same ctx |
| Maximum viable ctx fp8 | `kv_ceiling_sweep.csv` | Target: > bf16 ceiling |
| Maximum viable ctx bf16 | `kv_ceiling_sweep.csv` | Reference value |

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Both KV types produce any result | At least 4K ctx OK for both | If startup OOM at 4K: stop; check GPU0 VRAM |
| fp8 ceiling > bf16 ceiling | Expected (fp8 = 2× capacity) | If equal: investigate KV type flag acceptance |
| TPS at baseline (4K) ≥ 12 t/s | Confirms ngl=94 working | < 10 t/s: check layer allocation |
| OOM boundary identified | Status shows OOM at some ctx | |

## Artifacts to write

1. `results/BENCH_29_kv5_convergence_kv_ceiling_<TIMESTAMP>/kv_ceiling_sweep.csv`
2. `results/BENCH_29_kv5_convergence_kv_ceiling_<TIMESTAMP>/sweep_bf16.log`
3. `results/BENCH_29_kv5_convergence_kv_ceiling_<TIMESTAMP>/sweep_fp8.log`
4. `results/BENCH_29_kv5_convergence_kv_ceiling_<TIMESTAMP>/server_<kv>_ctx<N>.log` (one per ctx size tested)
5. `results/BENCH_29_kv5_convergence_kv_ceiling_<TIMESTAMP>/summary.md`:

```markdown
# BENCH_29 — T_KV5: Convergence KV Ceiling Sweep — <TIMESTAMP>

## Environment
- Model: UD-IQ2_M on ik_llama.cpp, ngl=94, CUDA_VISIBLE_DEVICES=0, GGML_CUDA_NO_PINNED=1
- Sweep: context 4K–256K, KV types: bf16 (f16) and fp8 (q8_0)

## Results
| KV type | ctx | prompt tok | out tok | elapsed (s) | TPS | VRAM (MiB) | status |
|---------|-----|-----------|---------|-------------|-----|------------|--------|
| bf16    | 4096   | <x> | <x> | <x> | <x> | <x> | OK/OOM |
| bf16    | 8192   | <x> | <x> | <x> | <x> | <x> | OK/OOM |
| ...     | ...    | ... | ... | ... | ... | ... | ...    |
| fp8     | 4096   | <x> | <x> | <x> | <x> | <x> | OK/OOM |
| ...     | ...    | ... | ... | ... | ... | ... | ...    |

## KV ceiling summary
| KV type | Max viable context | VRAM at ceiling (MiB) | TPS at ceiling |
|---------|-------------------|-----------------------|----------------|
| bf16    | <x> tokens        | <x>                   | <x> t/s        |
| fp8     | <x> tokens        | <x>                   | <x> t/s        |
| fp8 vs bf16 ratio | <x>× | | |

## TPS decay
- TPS at 4K ctx (reference): <x> t/s
- TPS at largest OK ctx (bf16): <x> t/s
- TPS at largest OK ctx (fp8): <x> t/s

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| fp8 ceiling > bf16 | PASS/FAIL | |
| TPS baseline ≥ 12 t/s | PASS/FAIL | |

## Verdict
PASS / FAIL / PARTIAL — <one sentence>
Recommended production KV type: fp8 / bf16 — <reasoning based on ceiling vs quality trade-off>

## Incidental findings
<KV type confirmation from startup log, unexpected VRAM growth pattern, TPS anomaly at any context size.>
<If nothing: "none">

## Open from testing
<If OOM occurs at unexpectedly low context (< 8K): record VRAM headroom and ngl used for research re-calculation.>
<If nothing: "none">
```

## Interpretation boundary

**You may:** Record OOM points, TPS, VRAM per context level. Note whether `--kv-type q8_0` was accepted (check server startup log for confirmation message).

**You may NOT:** Change the production KV type in `docs/arch/convergence.md` or `docs/procedures/` — that is research mode after reading summary.

## Stop condition

**Normal:** Both KV type sweeps reach OOM point, CSV filled, summary written.

**Abnormal:** `BENCH_29_OOM_BASELINE: Convergence OOM at ctx=4096 for kv_type=<X>. GPU0 free before start: <Y> MiB. ngl used: 94. Check if BENCH_27 left server running (check port 8002).`
