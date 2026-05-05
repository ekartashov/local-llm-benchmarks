# Convergence + Singularity Operational Guide

> **Summary:** Qwen3.5-397B-A17B UD-IQ2_M. Production: `ik_llama.cpp` main, `-ngl 15 --cpu-moe -t 32 -np 1`. 13.99 t/s single-seq. 83s cold start OR **12s CRIU restore** (BENCH_22). ⚠ True concurrent HTTP N≥2: **SETTLED FAIL** (PR #1288).

---

## Performance baseline (all settled T_CV1–T_CV4, 2026-04-26)

| Metric | Value | Source |
|--------|-------|--------|
| Cold start time | 83s median (CPU-bound, not disk) | T_CV1 |
| Warm-cache start | 88s | T_CV1 |
| Context ceiling | 128k tokens at 3.6 t/s | T_CV1 |
| CPU-only TPS (ngl=0) | 3.7 t/s | T_CV1 |
| Hybrid TPS (ngl=999 --cpu-moe) | **13.99 t/s** | T_CV3 |
| Hybrid speedup vs CPU-only | 3.75× | T_CV3 |
| Optimal thread count | **32 threads** (PP scales linearly) | T_CV2 |
| Sequential pipelining (-np 4) | **15.6 t/s aggregate** | T_CV4 |
| True concurrent HTTP (N≥2) | ❌ **SETTLED FAIL** — architectural limit (PR #1288: Qwen3.5-MoE cannot exceed 1 concurrent sequence) | T_PAR1 |
| GPU VRAM consumed (hybrid) | ~12GB split across both 5090s | T_CV3 |
| CRIU Restore (Warm) | **~12s** (1s sync + 11s 1st inference) | BENCH_22 |

**Policy:** On-demand Restore (preferred). The 12s restore time (BENCH_22) enables transparent routing. Convergence can be started on-demand when a request escalates from Arclight, saving 123GB of RAM during idle periods.

**T_CRIU2 results (2026-04-28):**
- `--no-mmap` (current): CRIU dump **impossible** on this hardware. CRIU's parasite injection causes VMS to spike to ~351 GB during dump; OOM killer fires. 188 GB RAM is not enough to hold 135 GB anon-RAM model + CRIU dump overhead simultaneously.
- mmap (`--no-mmap` removed): RESTORE_OK. Checkpoint 8.7 GB in 7.6 s. Restore 7.3 s. First-inference TTFT **100.56 s** (page-fault warmup — kernel reloads 123 GB of model weights on demand from NVMe). Rep-2: 36.1 s. Rep-3: 7.7 s (fully warm).

**Critical finding:** Without QX_PRELOAD, CRIU mmap is **slower** than cold start (100 s vs 83 s restore-to-interactive). Always-resident policy remains correct until QX_PRELOAD is implemented.

**With QX_PRELOAD (BENCH_22, 2026-05-05):** 
Success achieved via **`GGML_CUDA_NO_PINNED=1`**. By disabling pinned host memory, MoE experts (113 GB) remain file-backed.
- **Checkpoint size:** 8.0 GB (reduced from 122 GB).
- **Pre-warm:** `cat` GGUF files + CRIU images into page cache (~37s).
- **Restore-to-Interactive:** **12.0s total**.
This makes on-demand CRIU restoration **VIABLE** and preferred over always-resident mode for RAM-heavy multi-model co-loads.

---

## Production launch commands

### Option A: Always-Resident (Max Performance)
*Best for dedicated sessions. No swap overhead.*
```bash
/srv/ai/projects/ik_llama.cpp/build/bin/llama-server \
  -m /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf \
  -ngl 999 --cpu-moe --no-mmap \
  -b 4096 -ub 2048 -t 32 -np 1 --jinja --host 0.0.0.0 --port 8002
```

### Option B: On-Demand Restore (CRIU Viable)
*Best for co-load environments. 12s wake time.*
```bash
# Required environment
export GGML_CUDA_NO_PINNED=1
export UV_USE_IO_URING=0

/srv/ai/projects/ik_llama.cpp/build/bin/llama-server \
  -m /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf \
  -ngl 15 --cpu-moe \
  -b 4096 -ub 2048 -t 32 -np 1 --jinja --host 0.0.0.0 --port 8002
```

**Model path:** 4 split files must all be in the same directory. Reference only `00001-of-00004.gguf`; loader finds the rest automatically.

**RAM prerequisite:** Sleep both Arclight vLLM processes at level=1 before starting Convergence if you need to free VRAM for attention layers (technically Convergence can coexist with Arclight since its weights are CPU-side, but GPU VRAM is shared for attention).

---

## Deploy script shortcut
```bash
./infra/scripts/deploy.sh ikllamacpp convergence
```

---

## Engine: ik_llama.cpp main

**Why main:** Mainline ik_llama.cpp `main` branch now includes necessary architecture support for MoE models (`LLM_ARCH_QWEN35MOE`, `build_qwen35moe()`) and dense models (`LLM_ARCH_QWEN35`).

**Why NOT vLLM for Convergence:** `--cpu-offload-gb` on a 123GB model involves constant PCIe weight-chunk round-trips per forward pass. ik_llama.cpp's `--cpu-moe` keeps MoE expert weights in RAM and only transfers attention/norm/embed — the correct split for sparse MoE.

**Flag notes:**
- `-fa` is on by default — omit it entirely
- `-fmoe` is gone; fused MoE is default, disable with `-no-fmoe`
- `--cpu-moe` is the clean flag to pin all `ffn_gate/up/down_exps` to CPU RAM

**Rebuild after upstream changes:**
```bash
cd /srv/ai/projects/ik_llama.cpp
git checkout main
git pull origin main
cmake -S . -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) --target llama-server
```

**Verify architecture support:**
```bash
rg "qwen35|ssm_alpha|delta_net" /srv/ai/projects/ik_llama.cpp/src/
# Expected: llama-delta-net.cpp, llama-arch.cpp, llama-build-context.cpp
```

---

## Concurrency caveat (IMPORTANT)

`-np 4` sets the server's internal parallel slot count. **This does not mean 4 concurrent HTTP clients work safely.** T_PAR1 (2026-04-26) showed the server crashed at N≥2 concurrent HTTP requests with `Server disconnected` / `GGML_ASSERT(S > 0)`. **SETTLED FAIL (PR #1288):** Qwen3.5-MoE has an explicit architectural constraint preventing more than one concurrent sequence. This is not a bug to investigate or fix — it is a model design property. Even on `main` branch, N≥2 concurrent HTTP requests are not supported for this model.

Production use: send one request at a time. The `-np 4` gives throughput benefits when a single client's request fills multiple slots (long context, many output tokens), not when multiple clients send simultaneously.

---

## Singularity tier

System-exclusive mode — stops all other tiers before starting.

- **Model:** Qwen3.5-397B at Q3_K_M (~140GB) or Q4_K_M (~180GB)
- **Engine:** same ik_llama.cpp binary
- **Startup:** ~70s warm cache
- **VRAM:** maximize GPU offload for attention layers
- **Recovery:** restart Convergence (~83s cold) then Arclight (~100s each)

DDR5 bandwidth is the bottleneck: ~83 GB/s actual. Per-token read ≈ 2.3GB of expert weights → theoretical ceiling ~36 t/s. Measured ~13 t/s (36% efficiency due to NUMA, thread coordination, routing overhead).

---

## Model selection rationale

**Why UD-IQ2_M over UD-IQ3_XXS (current rationale):**
- UD-IQ2_M (~123GB) + Arclight sleep weights (~44GB) + OS (~4GB) = ~171GB of 192GB. 21GB headroom with `--no-mmap`.
- UD-IQ3_XXS (~140GB) leaves only ~8GB — dangerously tight for `--no-mmap`.
- Both are within benchmark margin of error of BF16 on this 512-expert MoE architecture (Benjamin Marie independent evaluation, H200s).

**This rationale becomes obsolete with CRIU (T_CRIU3):** Once CRIU replaces vLLM sleep mode, Arclight sleep weights (~44GB) no longer reside in RAM. Available headroom becomes 192 − 4 = 188GB → UD-IQ3_XXS (~140GB) fits comfortably. Quality upgrade from IQ2→IQ3 is meaningful for orchestration tasks. Revisit quant selection after T_CRIU2 + T_CRIU3 settle.

**`--no-mmap` vs mmap trade-off for CRIU (Settled BENCH_22):**
- `--no-mmap`: CRIU checkpoint impossible (OOM during dump). **RETIRED.**
- mmap (remove `--no-mmap` + **`GGML_CUDA_NO_PINNED=1`**): **VIABLE.** Restore-to-interactive in 12s. This is the new production standard for Convergence when not in always-resident mode. Text output is 100% deterministic pre/post restore.

**Why 397B over 122B:** TAU2 gap +14.7 points (86.7 vs ~72) in multi-step agentic orchestration — exactly what Convergence is invoked for.
