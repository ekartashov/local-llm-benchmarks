# BENCH_21 — co_load_vram: Three-model VRAM Co-load Verification

**Status:** READY  
**Blocks:** T_COLOAD (new), T5.1 (agent-routing), T2.6 (behemoth archetype sizing)

---

## Objective

Verify that all three production models can be co-resident on the two RTX 5090 GPUs simultaneously:
- **Thinker** (rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm) on GPU1 at 131K context with MTP n=3
- **Coder** (cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit) on GPU0 alone (TP=1, enforce-eager) at 32K context
- **Convergence** (unsloth/Qwen3.5-397B-A17B UD-IQ2_M) filling residual VRAM on both GPUs at low ngl

The result answers: can the full Arclight + Convergence stack be kept always-resident, or must Convergence be torn down when both Arclight endpoints are active?

---

## Why this exists

The current production topology runs coder as TP=2 across both GPUs, which evicts Convergence (they cannot share VRAM). This benchmark tests whether running coder as TP=1 (GPU0 only) leaves enough VRAM on both cards for Convergence to co-exist with a reduced GPU layer count, trading some Convergence TPS for always-resident availability.

**Critical context from the prior 8 failed attempts (2026-05-05):**

The previous session's entire strategy was based on "heterogeneous V1/V0 engine" — using `VLLM_USE_V1=1` for thinker and `VLLM_USE_V1=0` for coder. **This was wrong.** vLLM on this host is version **0.20.0**, in which `VLLM_USE_V1`, `VLLM_ENGINE_ITERATOR_SOURCE`, and all related V0 env vars are **unknown variables that are silently ignored**. Every container runs the V1 engine regardless. The V1/V0 distinction does not exist in 0.20.0.

The actual failures were:
1. **Thinker OOM (early runs)**: Wrong context / util settings. **RESOLVED**: thinker at util=0.95, ctx=131072, MTP n=3 is **confirmed working** in run `results/BENCH_21_coload_vram_20260505T082239Z/`. It uses 29,278 MiB on GPU1 (5.92 GiB KV cache, 44,928 token capacity). Do NOT re-investigate thinker startup.
2. **Coder OOM (all attempts)**: The 35B-A3B-AWQ model weights are 22.4 GiB. The CUDA graph profiling pass during startup temporarily peaks at ~30.7 GiB, exhausting any util ≤ 0.75. **Fix**: `--enforce-eager` eliminates CUDA graph capture and drops the profiling peak to ~24.5 GiB. At util=0.80 + enforce-eager: budget = 25.1 GiB, overhead = 24.5 GiB → KV ≈ +0.6 GiB (loadable).
3. **Convergence ngl=999 does not fit after thinker+coder**: After thinker (GPU1 free: 2,834 MiB) and coder (GPU0 free: ~6,300 MiB), Convergence's current ngl=999 config uses 8,182/8,684 MiB per GPU — far exceeding residual margins. Fix: run Convergence with `CONVERGENCE_NGL=15` (estimated ~1,100 MiB/GPU), which should fit. TPS will be lower than the isolated 13.99 t/s; this is acceptable for the co-load test.

---

## Context to read

Before running anything, read these files:

1. `docs/INDEX.md` — current production config, VRAM gotchas (#8 and the hardware quick-ref)
2. `benchmarks/queue/BENCH_21_coload_vram.sh` — the benchmark script (read the header comments)
3. `results/BENCH_21_coload_vram_20260505T082239Z/vram_log.txt` — thinker-only VRAM baseline
4. `results/BENCH_21_coload_vram_20260505T081859Z/vram_log.txt` — Convergence-only VRAM baseline (8,182/8,684 MiB at ngl=999)

---

## Prerequisites

```bash
echo "=== Prerequisites ===" && \
  test -f /srv/ai/projects/ik_llama.cpp/build/bin/llama-server \
    && echo "[prereq] ik_llama.cpp binary OK" \
    || { echo "[prereq] STOP: ik_llama.cpp binary not found"; exit 1; } && \
  test -f /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf \
    && echo "[prereq] Convergence GGUF OK" \
    || { echo "[prereq] STOP: Convergence GGUF not found"; exit 1; } && \
  python3 -c "import subprocess; r=subprocess.run(['podman','images','--format','{{.Repository}}:{{.Tag}}'],capture_output=True,text=True); lines=r.stdout.splitlines(); found=any('vllm' in l for l in lines); print('[prereq] vLLM image OK' if found else '[prereq] STOP: no vllm image found'); exit(0 if found else 1)" && \
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | awk -F',' '{if ($2+0 < 30000) { print "[prereq] STOP: GPU " $1 " has less than 30 GiB free — run cleanup first"; exit 1 } else { print "[prereq] GPU " $1 " free: " $2 " MiB OK" }}'
```

If any check fails, stop and investigate before continuing.

---

## Inputs required

- **ik_llama.cpp binary**: `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- **Convergence model**: `/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf` (all 4 shards must be present in the same directory)
- **Thinker model**: `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm` (already cached from prior runs)
- **Coder model**: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (already cached from prior runs)
- **GPU VRAM**: Both GPUs must be free (≥31 GiB each) before starting

---

## Fixed controls table

| Control | Value |
|---------|-------|
| Thinker model | `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm` |
| Thinker placement | GPU1 only (`deploy.sh vllm gpu1`) |
| Thinker ctx | 131072 |
| Thinker util | 0.95 |
| Thinker engine flags | `--trust-remote-code --kv-cache-dtype fp8 --enable-chunked-prefill --max-num-seqs 4 --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}' --speculative-config={"method":"mtp","num_speculative_tokens":3}` |
| Thinker env | `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` |
| Coder model | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` |
| Coder placement | GPU0 only (`deploy.sh vllm gpu0`, **TP=1**) |
| Coder ctx | 32768 |
| Coder util | 0.80 |
| Coder engine flags | `--kv-cache-dtype fp8 --enforce-eager` |
| Coder env | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` |
| Convergence model | `unsloth/Qwen3.5-397B-A17B UD-IQ2_M` (4-shard split GGUF) |
| Convergence ngl (starting point) | 15 |
| Convergence flags | `--cpu-moe -b 4096 -ub 2048 -t 32 -np 4 -c 131072` |
| vLLM version | 0.20.0 (V1 engine only — do NOT pass VLLM_USE_V1 or VLLM_ENGINE_ITERATOR_SOURCE) |
| TPS test prompt | "Explain the CAP theorem and its implications for distributed database design." |
| TPS reps | 3 sequential requests to Convergence |

---

## Variable under test

Whether the three models can co-reside on 2×32 GiB GPUs, and at what `CONVERGENCE_NGL` value. Try ngl=15 first; if VRAM fits, try ngl=18 and ngl=20 to find the maximum NGL before OOM.

---

## Procedure

Skip flags (set to 1 to skip expensive steps on retry):
- `SKIP_THINKER=1` — skip thinker deploy (use if bench-vllm-gpu1 is already up at port 30001)
- `SKIP_CODER=1` — skip coder deploy (use if bench-vllm-gpu0 is already up at port 30000)
- `SKIP_CONVERGENCE=1` — skip Convergence (use for Phase A thinker+coder-only test)
- `CONVERGENCE_NGL=N` — override the default ngl=15

### Step 0 — Full cleanup

```bash
pkill -9 llama-server || true
podman stop -a || true
podman rm -f $(podman ps -aq) || true
sleep 5
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
# Both GPUs should show <100 MiB used.  If not, wait another 10s and retry.
```

### Step 1 — Phase A: Thinker + Coder only (no Convergence)

Run this first to confirm thinker+coder co-load works before adding Convergence complexity.

```bash
cd /srv/ai/projects/local-llm-benchmarks
SKIP_CONVERGENCE=1 bash benchmarks/queue/BENCH_21_coload_vram.sh
```

**Expected outcome:**
- Thinker: health OK at port 30001 (takes ~3 min — torch compile cache warm from prior runs)
- Coder: health OK at port 30000 (takes ~4 min)
- VRAM after Phase 2 in `results/BENCH_21_coload_vram_<TS>/vram_log.txt`:
  - GPU1: ~29,300 MiB used (thinker)
  - GPU0: ~25,000–25,500 MiB used (coder, enforce-eager, minimal KV)

**If coder fails with `ValueError: No available memory for cache blocks`:**
The util=0.80 is still insufficient. Try util=0.82:
```bash
SKIP_CONVERGENCE=1 CODER_MEM_UTIL=0.82 bash benchmarks/queue/BENCH_21_coload_vram.sh
```

**If coder fails with `OutOfMemoryError` during profiling:**
This is unexpected since enforce-eager removes CUDA graph overhead. Record the exact error, VRAM reading, and stop — write Open from testing.

### Step 2 — Phase B: Three-model co-load at ngl=15

Only run after Step 1 succeeds. The script was already interrupted during thinker health wait, so containers may be running. Clean up and re-run with Convergence enabled.

```bash
pkill -9 llama-server || true
podman stop -a || true
podman rm -f $(podman ps -aq) || true
sleep 5

CONVERGENCE_NGL=15 bash benchmarks/queue/BENCH_21_coload_vram.sh
```

**Expected VRAM after Phase 3:**
- GPU1: ~29,300 MiB used (thinker) + ~500–700 MiB (Convergence) = ~29,900–30,000 MiB
- GPU0: ~25,000 MiB used (coder) + ~500–700 MiB (Convergence) = ~25,500–25,700 MiB

**If Convergence health timeout at ngl=15:**
Check `results/BENCH_21_coload_vram_<TS>/convergence.log`. If there is an OOM error, try ngl=10:
```bash
pkill -9 llama-server || true
podman stop -a || true; podman rm -f $(podman ps -aq) || true; sleep 5
CONVERGENCE_NGL=10 bash benchmarks/queue/BENCH_21_coload_vram.sh
```

### Step 3 — Find maximum viable NGL (binary search)

If ngl=15 succeeds, try progressively higher values to find the OOM boundary:

```bash
# Try ngl=18
pkill -9 llama-server || true
podman stop -a || true; podman rm -f $(podman ps -aq) || true; sleep 5
CONVERGENCE_NGL=18 bash benchmarks/queue/BENCH_21_coload_vram.sh
```

Record VRAM readings and TPS at each ngl that succeeds. Stop at the first OOM and record the boundary.

### Step 4 — Verify Convergence TPS under co-load

The script automatically runs 3 TPS reps in Phase 5 if Convergence is up. Check `results/BENCH_21_coload_vram_<TS>/convergence_coload_tps.csv`.

Baseline for comparison: isolated Convergence at ngl=999 = **13.99 t/s** (from T_CV3).

---

## Metrics to record

| Metric | Source file | Expected / reference |
|--------|-------------|---------------------|
| GPU1 VRAM after thinker (MiB) | `vram_log.txt` Phase 1 block | ~29,278 MiB (from prior run T082239Z) |
| GPU0 VRAM after coder (MiB) | `vram_log.txt` Phase 2 block | ~25,000–25,500 MiB (first measurement) |
| GPU0 free after coder (MiB) | `vram_log.txt` Phase 2 block | ~5,800–6,300 MiB |
| GPU1 free after coder (MiB) | `vram_log.txt` Phase 2 block | ~2,700–2,900 MiB |
| Convergence VRAM GPU0 (MiB) | Phase 4 snapshot minus Phase 2 GPU0 | TBD — function of ngl |
| Convergence VRAM GPU1 (MiB) | Phase 4 snapshot minus Phase 2 GPU1 | TBD — function of ngl |
| Maximum viable CONVERGENCE_NGL | Determined by binary search | Target: ≥10 |
| Convergence TPS under co-load | `convergence_coload_tps.csv` rep2,rep3 (warm) | Target: ≥4 t/s (vs isolated 13.99) |
| Thinker KV cache tokens (from deploy log) | `thinker_deploy.log` "GPU KV cache size" line | ~44,928 tokens (from T082239Z) |

---

## Pass/fail checks

| Check | Pass condition | Fail action |
|-------|----------------|-------------|
| Thinker health | Port 30001 responds within 300s | Stop; write Open from testing with thinker_deploy.log tail |
| Coder health (util=0.80) | Port 30000 responds within 480s | Retry with util=0.82; if still fails write Open from testing with exact error and OOM line |
| Convergence fits at ngl≤20 | llama-server starts + health OK at port 8002 | Write the OOM ngl value and continue with the highest successful ngl |
| Three-model VRAM stays within budget | GPU0 total used ≤ 31,360 MiB, GPU1 total used ≤ 31,360 MiB (from Phase 4 snapshot) | Record actual overage and OOM ngl boundary |
| Convergence TPS under co-load | TPS (rep2+rep3 average) ≥ 4 t/s | Record actual TPS even if below target; this is informational |

---

## Artifacts to write

1. `results/BENCH_21_coload_vram_<TIMESTAMP>/vram_log.txt` — VRAM snapshots at each phase (written by script)
2. `results/BENCH_21_coload_vram_<TIMESTAMP>/thinker_deploy.log` — vLLM startup log (written by script)
3. `results/BENCH_21_coload_vram_<TIMESTAMP>/coder_deploy.log` — vLLM startup log (written by script)
4. `results/BENCH_21_coload_vram_<TIMESTAMP>/convergence.log` — ik_llama.cpp startup log (written by script)
5. `results/BENCH_21_coload_vram_<TIMESTAMP>/convergence_coload_tps.csv` — TPS measurements (written by script)
6. `results/BENCH_21_coload_vram_<TIMESTAMP>/results.txt` — key=value summary (written by script)
7. `results/BENCH_21_coload_vram_<TIMESTAMP>/summary.md` — **YOU MUST WRITE THIS**

### summary.md template

```markdown
# BENCH_21 — Co-load VRAM mapping — <TIMESTAMP>

## Environment
- vLLM version: 0.20.0 (V1 engine only)
- Thinker: rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm, GPU1, util=0.95, ctx=131072, MTP n=3
- Coder: cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, GPU0 (TP=1), util=<CODER_MEM_UTIL>, ctx=32768, enforce-eager
- Convergence: unsloth/Qwen3.5-397B-A17B UD-IQ2_M, ik_llama.cpp, ngl=<NGL_USED>, --cpu-moe

## VRAM measurements (MiB)

### After Phase 1 (thinker only)
| GPU | Used | Free |
|-----|------|------|
| GPU1 | <value> | <value> |

### After Phase 2 (thinker + coder)
| GPU | Used | Free |
|-----|------|------|
| GPU0 | <value> | <value> |
| GPU1 | <value> | <value> |

### After Phase 3 (all three models, ngl=<NGL>)
| GPU | Used | Free |
|-----|------|------|
| GPU0 | <value> | <value> |
| GPU1 | <value> | <value> |

## Convergence VRAM overhead at each tested ngl
| ngl | GPU0 delta (MiB) | GPU1 delta (MiB) | Health OK? |
|-----|-----------------|-----------------|-----------|
| 15  | <value>         | <value>         | YES/NO    |
| 18  | <value>         | <value>         | YES/NO    |
| 20  | <value>         | <value>         | YES/NO    |

## Convergence TPS under co-load (ngl=<NGL_USED>)
| Rep | Elapsed (s) | Tokens | TPS |
|-----|-------------|--------|-----|
| 1   | <value>     | <value> | <value> |
| 2   | <value>     | <value> | <value> |
| 3   | <value>     | <value> | <value> |

Baseline (isolated, ngl=999): 13.99 t/s

## Pass/fail
| Check | Result | Notes |
|-------|--------|-------|
| Thinker health OK | PASS/FAIL | |
| Coder health OK (util=<X>) | PASS/FAIL | |
| Convergence fits at ngl=15 | PASS/FAIL | |
| Maximum viable NGL | <value> | OOM first seen at ngl=<X> |
| Convergence TPS ≥ 4 t/s | PASS/FAIL | Actual: <value> t/s |

## Verdict
PASS / FAIL / PARTIAL — <one-sentence reason>

## Incidental findings
<Any VRAM deviation from expectations, unexpected log warnings, version surprises. If none: "none">

## Open from testing
<Any blocker or new question. If coder or thinker OOM'd at different values than expected, record
the exact error and VRAM numbers. If none: "none">
```

---

## Interpretation boundary

**The agent MAY:**
- Write `results/BENCH_21_coload_vram_<TS>/summary.md`
- Write an `## Open from testing` block in `RESEARCH_STATE.md` if a blocker is found
- Record incidental findings in summary.md (e.g., unexpected VRAM usage, version mismatches)

**The agent MAY NOT:**
- Modify `docs/decisions/settled.md`, `docs/arch/`, `docs/queue/`, `RESEARCH_STATE.md` (except Open from testing)
- Conclude anything about production architecture changes (e.g., "coder should switch to TP=1") — that is research-mode analysis
- Modify the BENCH_21 script itself
- Update `docs/INDEX.md` or `docs/procedures/`

---

## Stop conditions

**Normal stop:** Both vLLM endpoints are healthy, Convergence loaded at some ngl≥10, TPS measured, summary.md written.

**Abnormal stop (write Open from testing for each):**

1. **Thinker OOM despite util=0.95**: This would be a regression from the confirmed working run `T082239Z`. Write: `THINKER_OOM: thinker failed to load at util=0.95 ctx=131072 MTP n=3 — regression from T082239Z. Last 20 lines of thinker_deploy.log: <paste>`.

2. **Coder OOM at util=0.82**: The model+overhead exceeds even 0.82×31.3 GiB. Write: `CODER_OOM: coder TP=1 GPU0 OOMs at util=0.82 enforce-eager. Available KV reported: <value> GiB. Model overhead exceeds budget. Research needed: can --max-num-seqs be reduced or ctx cut to 16384?`

3. **Convergence cannot fit even at ngl=5**: Three-way co-load is infeasible. Write: `COLOAD_INFEASIBLE: Convergence OOMs at ngl=5 after thinker (GPU1 used: <value> MiB) and coder (GPU0 used: <value> MiB). Per-GPU VRAM residual too small for any GPU layer count. Co-load requires production coder TP=2 → Convergence eviction is unavoidable.`

4. **Script crashes before Phase 4 VRAM snapshot**: Write the phase that failed and the exact error.
