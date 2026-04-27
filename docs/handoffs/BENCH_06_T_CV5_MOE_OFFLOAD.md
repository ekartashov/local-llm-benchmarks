# BENCH_06 — T_CV5: Convergence MoE Expert GPU Offload Test

**Status: BLOCKED**
**Blocks: nothing**
**Blocked by: BENCH_05** — must complete with a full `metrics.json` before this handoff may begin.

---

## Title
T_CV5 — Convergence partial MoE expert offload (--cpu-moe removed at optimal ngl)

## Objective
At the ngl value identified in BENCH_05 as the TPS saturation point, test whether removing --cpu-moe (offloading MoE expert weights to GPU alongside attention layers) increases TPS, and at what VRAM cost.

## Why this exists
BENCH_05 finds the minimum ngl that achieves maximum TPS with --cpu-moe. This handoff answers whether any TPS gain remains by also offloading MoE experts, which are currently held in CPU RAM (~123 GB). Offloading even a fraction of experts to GPU trades VRAM for decode speed. The result determines whether a higher -ngl without --cpu-moe is worth the VRAM cost.

## Prerequisites

**BENCH_05 must be complete.** Verify before proceeding:

```bash
python3 - <<'EOF'
import json, os, sys

dirs = sorted(
    [d for d in os.listdir("results") if d.startswith("T_CV5_ngl_sweep_")],
    reverse=True
)
if not dirs:
    print("BLOCKED: no T_CV5_ngl_sweep results found. Run BENCH_05 first.")
    sys.exit(1)

with open(f"results/{dirs[0]}/metrics.json") as f:
    d = json.load(f)

sweep = d.get("sweep", {})
measured = [ngl for ngl in [10, 20, 35, 50] if str(ngl) in sweep or ngl in sweep]
if len(measured) < 4:
    print(f"BLOCKED: only {len(measured)}/4 ngl values measured in BENCH_05.")
    sys.exit(1)

# Find saturation point: first ngl where tps >= 0.9 * 13.99
baseline = 13.99
target = 0.90 * baseline
sat_ngl = None
for ngl in [10, 20, 35, 50]:
    key = str(ngl) if str(ngl) in sweep else ngl
    tps = sweep[key]["median_tps"]
    if tps >= target:
        sat_ngl = ngl
        break

if sat_ngl is None:
    print("BLOCKED: no ngl value reached 90% of baseline TPS in BENCH_05.")
    print("All ngl values underperformed. Research mode must review before proceeding.")
    sys.exit(1)

print(f"BENCH_05 complete. Saturation ngl = {sat_ngl}. Proceeding.")
EOF
```

Also verify Convergence is running:
```bash
curl -sf http://localhost:8002/health && echo "CONVERGENCE OK" || echo "CONVERGENCE DOWN — start it first"
```

## Inputs required
- BENCH_05 `metrics.json` — provides the saturation ngl value
- Running Convergence container on port 8002
- `infra/scripts/deploy.sh`

## Fixed controls
| Control | Value |
|---------|-------|
| Model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M |
| Engine | ik_llama.cpp pr-1288 |
| -ngl | **saturation ngl from BENCH_05** (fixed for this test) |
| -t (threads) | 32 |
| -np | 1 |
| -c (context) | 4096 |
| Test prompt | same as BENCH_05 |
| max_tokens | 200 |
| temperature | 0.0 |
| Repetitions | 3 |

## Single variable under test
**--cpu-moe** — present (MoE on CPU) vs absent (MoE on GPU).

Two configurations at the saturation ngl:
- Config A: `-ngl <sat_ngl> --cpu-moe` (same as BENCH_05 best point)
- Config B: `-ngl <sat_ngl>` (--cpu-moe removed; MoE experts offload to GPU)

## Procedure

```bash
# Read saturation ngl from BENCH_05
SAT_NGL=$(python3 - <<'EOF'
import json, os
dirs = sorted([d for d in os.listdir("results") if d.startswith("T_CV5_ngl_sweep_")], reverse=True)
with open(f"results/{dirs[0]}/metrics.json") as f:
    d = json.load(f)
sweep = d["sweep"]
baseline = 13.99
target = 0.90 * baseline
for ngl in [10, 20, 35, 50]:
    key = str(ngl) if str(ngl) in sweep else ngl
    if sweep[key]["median_tps"] >= target:
        print(ngl)
        break
EOF
)
echo "Using saturation ngl: ${SAT_NGL}"

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/T_CV5_moe_offload_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

PROMPT="Explain the architecture of a transformer model in detail, covering attention mechanisms, feed-forward layers, positional encoding, and the encoder-decoder structure."

# === Config A: with --cpu-moe (baseline, matching BENCH_05) ===
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
podman stop "${CONV_CONTAINER}" 2>/dev/null; podman rm "${CONV_CONTAINER}" 2>/dev/null
./infra/scripts/deploy.sh ikllamacpp convergence \
  -- -ngl ${SAT_NGL} --cpu-moe -t 32 -np 1 -c 4096
for i in $(seq 1 120); do curl -sf http://localhost:8002/health && break; sleep 1; done

VRAM_A=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
echo "Config A VRAM: ${VRAM_A}"
echo "config,ngl,cpu_moe,rep,tokens,elapsed_s,tps" > "${RESULTS_DIR}/raw.csv"

for REP in 1 2 3; do
  START_MS=$(date +%s%3N)
  RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"convergence\",\"prompt\":\"${PROMPT}\",\"max_tokens\":200,\"temperature\":0.0}")
  END_MS=$(date +%s%3N)
  ELAPSED_S=$(python3 -c "print(($END_MS - $START_MS) / 1000.0)")
  TOKENS=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])")
  TPS=$(python3 -c "print(round(${TOKENS}/${ELAPSED_S}, 2))")
  echo "A,${SAT_NGL},yes,${REP},${TOKENS},${ELAPSED_S},${TPS}" >> "${RESULTS_DIR}/raw.csv"
  echo "Config A rep=${REP}: ${TPS} t/s"
done

# === Config B: without --cpu-moe ===
# NOTE: This will attempt to load MoE expert weights into GPU VRAM.
# If VRAM is insufficient, the deploy will OOM. Record the error and skip to restore.
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
podman stop "${CONV_CONTAINER}" 2>/dev/null; podman rm "${CONV_CONTAINER}" 2>/dev/null

echo "Deploying Config B (no --cpu-moe). This may OOM."
./infra/scripts/deploy.sh ikllamacpp convergence \
  -- -ngl ${SAT_NGL} -t 32 -np 1 -c 4096 || {
    echo "CONFIG_B_DEPLOY_FAILED" >> "${RESULTS_DIR}/raw.csv"
    echo "OOM or deploy error at Config B. Recording and skipping to restore."
  }

# Only measure if deploy succeeded
if curl -sf http://localhost:8002/health 2>/dev/null; then
  VRAM_B=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
  echo "Config B VRAM: ${VRAM_B}"
  for REP in 1 2 3; do
    START_MS=$(date +%s%3N)
    RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"convergence\",\"prompt\":\"${PROMPT}\",\"max_tokens\":200,\"temperature\":0.0}")
    END_MS=$(date +%s%3N)
    ELAPSED_S=$(python3 -c "print(($END_MS - $START_MS) / 1000.0)")
    TOKENS=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])")
    TPS=$(python3 -c "print(round(${TOKENS}/${ELAPSED_S}, 2))")
    echo "B,${SAT_NGL},no,${REP},${TOKENS},${ELAPSED_S},${TPS}" >> "${RESULTS_DIR}/raw.csv"
    echo "Config B rep=${REP}: ${TPS} t/s"
  done
else
  echo "Config B did not come up. Recording as OOM/FAILED."
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "${RESULTS_DIR}/vram_at_oom.txt"
fi

# === MANDATORY: restore production Convergence ===
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
podman stop "${CONV_CONTAINER}" 2>/dev/null; podman rm "${CONV_CONTAINER}" 2>/dev/null
./infra/scripts/deploy.sh ikllamacpp convergence
for i in $(seq 1 120); do curl -sf http://localhost:8002/health && echo "PRODUCTION RESTORED" && break; sleep 1; done
```

## Metrics to record

| Config | --cpu-moe | Median TPS | GPU VRAM total |
|--------|-----------|-----------|----------------|
| A | yes | from `raw.csv` | from `nvidia-smi` |
| B | no | from `raw.csv` or OOM | from `vram_at_oom.txt` |

## Pass/fail checks

| Check | Condition | Action |
|-------|-----------|--------|
| Config A TPS ≈ BENCH_05 saturation TPS | Within ±10% | Note discrepancy |
| Config B OOM | Expected possible | Record VRAM and proceed to restore |
| Production Convergence restored | `/health` returns 200 | Do not leave non-production config running |

Config B OOM is a valid result, not a failure. Record the VRAM at the OOM point.

## Artifacts to write

1. `results/T_CV5_moe_offload_<timestamp>/raw.csv`
2. `results/T_CV5_moe_offload_<timestamp>/vram_at_oom.txt` (if applicable)
3. `results/T_CV5_moe_offload_<timestamp>/summary.md`:

```markdown
# T_CV5 MoE Offload — <TIMESTAMP>
## ngl = <SAT_NGL> (saturation point from BENCH_05)

| Config | --cpu-moe | Median TPS | GPU VRAM (total MiB) |
|--------|-----------|-----------|----------------------|
| A | yes | <tps_A> | <vram_A> |
| B | no | <tps_B or OOM> | <vram_B or vram_at_oom> |

## Status
MEASURED / CONFIG_B_OOM
```

**Do NOT write to any file outside `results/T_CV5_moe_offload_<timestamp>/`.**

## Interpretation boundary

- **You may record** TPS and VRAM for both configs.
- **You may note** whether Config B OOM'd and at what VRAM level.
- **You may NOT** update the production -ngl or --cpu-moe settings in `docs/arch/convergence.md`.
- **You may NOT** conclude whether the MoE offload is "worth it."

## Stop condition

**Normal:** both configs measured (or Config B OOM recorded), `summary.md` written, production Convergence restored.

**Abnormal:** write `## Open from testing` if:
- Config A TPS is >20% below BENCH_05 saturation TPS (environment has changed)
- Production Convergence cannot be restored after 120 s
