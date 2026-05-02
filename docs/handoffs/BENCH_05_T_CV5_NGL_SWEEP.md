# BENCH_05 — T_CV5: Convergence -ngl Value Sweep (--cpu-moe fixed)

**Status: DONE**
**Blocks: BENCH_06**
**Blocked by: nothing**

---

## Title
T_CV5 — Convergence -ngl layer-offload sweep with --cpu-moe fixed

## Objective
Measure Convergence decode TPS and GPU VRAM at -ngl values of 10, 20, 35, and 50 (with --cpu-moe always ON). Find whether the current production value of 999 is necessary, or whether a lower -ngl reaches the same TPS ceiling with less VRAM pressure.

## Why this exists
Only the two extremes have been measured: ngl=0 → 3.7 t/s and ngl=999 --cpu-moe → 13.99 t/s. The shape of the curve between them is unknown. Knowing the saturation point (the lowest ngl that reaches ~14 t/s) determines how much GPU VRAM can be reclaimed if Convergence is co-deployed alongside Extended Arclight modes.

## Prerequisites

```bash
# 1. Convergence container is running and healthy
curl -sf http://localhost:8002/health && echo "CONVERGENCE OK" || echo "CONVERGENCE DOWN — START IT FIRST"

# 2. Verify the container name for stop/restart
podman ps --format "{{.Names}}\t{{.Status}}" | grep -i "convergence\|ikllamacpp\|8002"
# Note the exact container name — used in Step 2 of each sweep iteration

# 3. GPU VRAM baseline (both GPUs before any change)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# 4. Results directory writable
mkdir -p results && echo "OK"
```

If Convergence is down, start it:
```bash
./infra/scripts/deploy.sh ikllamacpp convergence
```
Wait until `curl -sf http://localhost:8002/health` returns 200 (may take up to 90 s).

## Inputs required
- Running Convergence container on port 8002
- Known container name (from prerequisite step 2)
- `infra/scripts/deploy.sh`

## Fixed controls
| Control | Value |
|---------|-------|
| Model | unsloth/Qwen3.5-397B-A17B UD-IQ2_M |
| Engine | ik_llama.cpp older branches |
| --cpu-moe | **ON** (fixed throughout all sweep points) |
| -t (threads) | 32 |
| -np | 1 (single slot — isolates decode TPS from pipelining) |
| -c (context) | 4096 |
| Test prompt | fixed (see procedure) |
| max_tokens | 200 |
| temperature | 0.0 |
| Repetitions per ngl | 3 |

## Single variable under test
**-ngl** — number of model layers offloaded to GPU.
Values tested: 10, 20, 35, 50.
Baselines already measured: 0 → 3.7 t/s (T_CV1), 999 → 13.99 t/s (T_CV3).

## Procedure

Set up the results directory and the fixed test prompt:

```bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="results/T_CV5_ngl_sweep_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Fixed test prompt (same for all iterations)
PROMPT="Explain the architecture of a transformer model in detail, covering attention mechanisms, feed-forward layers, positional encoding, and the encoder-decoder structure."

echo "Results directory: ${RESULTS_DIR}"
```

For each NGL value in `(10 20 35 50)`, run the following block in order. Do not proceed to the next NGL value until the current one is complete.

```bash
NGL=<value>   # replace with 10, then 20, then 35, then 50

# --- Stop current Convergence ---
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
echo "Stopping: ${CONV_CONTAINER}"
podman stop "${CONV_CONTAINER}" && podman rm "${CONV_CONTAINER}"
sleep 3

# --- Deploy with new -ngl ---
# NOTE: pass NGL via CONVERGENCE_EXTRA_FLAGS env var or edit deploy.sh call below
# Check deploy.sh for how to pass extra ik_llama.cpp flags; if uncertain, use:
VLLM_USE_V1=0 \
./infra/scripts/deploy.sh ikllamacpp convergence \
  -- -ngl ${NGL} --cpu-moe -t 32 -np 1 -c 4096

# Wait for health (up to 120 s)
echo "Waiting for health..."
for i in $(seq 1 120); do
  curl -sf http://localhost:8002/health && echo "READY at ngl=${NGL}" && break
  sleep 1
done

# --- Record VRAM immediately after startup ---
VRAM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
echo "ngl=${NGL} VRAM: ${VRAM}"

# --- Run 3 inference requests and record TPS ---
for REP in 1 2 3; do
  START_MS=$(date +%s%3N)
  RESPONSE=$(curl -sf http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"convergence\",
      \"prompt\": \"${PROMPT}\",
      \"max_tokens\": 200,
      \"temperature\": 0.0
    }")
  END_MS=$(date +%s%3N)
  ELAPSED_S=$(python3 -c "print(($END_MS - $START_MS) / 1000.0)")
  TOKENS=$(echo "${RESPONSE}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['usage']['completion_tokens'])
")
  TPS=$(python3 -c "print(round(${TOKENS} / ${ELAPSED_S}, 2))")
  echo "ngl=${NGL} rep=${REP}: ${TOKENS} tokens in ${ELAPSED_S}s = ${TPS} t/s"
  echo "ngl,rep,tokens,elapsed_s,tps" >> "${RESULTS_DIR}/raw.csv" 2>/dev/null || \
    echo "ngl,rep,tokens,elapsed_s,tps" > "${RESULTS_DIR}/raw.csv"
  echo "${NGL},${REP},${TOKENS},${ELAPSED_S},${TPS}" >> "${RESULTS_DIR}/raw.csv"
done

echo "ngl=${NGL} complete."
```

After all four NGL values are measured:

```bash
# Restore production Convergence (ngl=999 --cpu-moe -np 4)
CONV_CONTAINER=$(podman ps --format "{{.Names}}" | grep -i "convergence\|ikllamacpp" | head -1)
podman stop "${CONV_CONTAINER}" 2>/dev/null; podman rm "${CONV_CONTAINER}" 2>/dev/null
./infra/scripts/deploy.sh ikllamacpp convergence
# Wait for health
for i in $(seq 1 120); do
  curl -sf http://localhost:8002/health && echo "PRODUCTION CONVERGENCE RESTORED" && break
  sleep 1
done
```

Build the metrics summary:

```bash
python3 - <<'EOF'
import csv, json, statistics, os

results_dir = sorted(
    [d for d in os.listdir("results") if d.startswith("T_CV5_ngl_sweep_")],
    reverse=True
)[0]
path = f"results/{results_dir}/raw.csv"

data = {}
with open(path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ngl = int(row["ngl"])
        tps = float(row["tps"])
        data.setdefault(ngl, []).append(tps)

results = {}
for ngl, tps_list in sorted(data.items()):
    results[ngl] = {
        "median_tps": statistics.median(tps_list),
        "raw_tps": tps_list
    }

# Add known baselines
results[0]   = {"median_tps": 3.70,  "raw_tps": [], "source": "T_CV1"}
results[999] = {"median_tps": 13.99, "raw_tps": [], "source": "T_CV3"}

with open(f"results/{results_dir}/metrics.json", "w") as f:
    json.dump({
        "item_id": "T_CV5_ngl_sweep",
        "timestamp": results_dir.split("_")[-1],
        "sweep": results
    }, f, indent=2)

print("metrics.json written.")
for ngl in sorted(results.keys()):
    print(f"  ngl={ngl:4d}: {results[ngl]['median_tps']:.2f} t/s")
EOF
```

## Metrics to record

For each -ngl value tested:

| Metric | Source |
|--------|--------|
| Median TPS over 3 reps | `raw.csv` → computed |
| GPU 0 VRAM used (MiB) | `nvidia-smi` immediately after startup |
| GPU 1 VRAM used (MiB) | same |

Reference baselines (do not re-measure):
- ngl=0: 3.70 t/s (T_CV1)
- ngl=999 --cpu-moe: 13.99 t/s (T_CV3)

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|---------------|-------------|
| Deploy succeeds at each ngl | Health check passes within 120 s | Record OOM at which ngl; skip to restore |
| TPS values plausible | All measured values between 3.0 and 20.0 t/s | Flag outlier |
| Baseline consistency | ngl=999 re-measure within ±10% of 13.99 t/s (optional sanity check) | Note drift |
| raw.csv non-empty | Contains entries for all four ngl values | STOP; write `## Open from testing` |

## Artifacts to write

1. `results/T_CV5_ngl_sweep_<timestamp>/raw.csv` — written during procedure
2. `results/T_CV5_ngl_sweep_<timestamp>/metrics.json` — written by summary script
3. `results/T_CV5_ngl_sweep_<timestamp>/summary.md`:

```markdown
# T_CV5 NGL Sweep — <TIMESTAMP>

| -ngl | Median TPS | GPU 0 VRAM (MiB) | GPU 1 VRAM (MiB) |
|------|-----------|-----------------|-----------------|
| 0    | 3.70 (T_CV1 baseline) | — | — |
| 10   | <tps> | <x> | <x> |
| 20   | <tps> | <x> | <x> |
| 35   | <tps> | <x> | <x> |
| 50   | <tps> | <x> | <x> |
| 999  | 13.99 (T_CV3 baseline) | — | — |

## Status
MEASURED
```

**Do NOT write to any file outside `results/T_CV5_ngl_sweep_<timestamp>/`.**

## Interpretation boundary

- **You may record** TPS and VRAM at each ngl value.
- **You may note** at which ngl value TPS appears to plateau (first ngl where TPS ≈ 13.99 t/s).
- **You may NOT** select a new production -ngl value.
- **You may NOT** update `docs/arch/convergence.md` or any production config.
- **You may NOT** conclude whether co-deployment with Extended Arclight is viable.

## Stop condition

**Normal:** all four ngl values measured, `metrics.json` and `summary.md` written, production Convergence restored.

**Abnormal:** write `## Open from testing` if:
- Convergence OOMs at any ngl before 35 (unexpected — record the ngl and VRAM)
- Deploy hangs for > 120 s at any ngl (record which value and stop sweep at that point)
- raw.csv is empty after the sweep
