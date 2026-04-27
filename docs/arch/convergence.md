# Convergence + Singularity Operational Guide

> **Summary:** Qwen3.5-397B-A17B UD-IQ2_M always-resident in RAM. Production: `-ngl 999 --cpu-moe -t 32 -np 4`. 13.99 t/s single-seq (T_CV3), 15.6 t/s sequential pipelining (T_CV4). ⚠ True concurrent HTTP requests crash pr-1288 at N≥2 (T_PAR1). 128K context ceiling (T_CV1). 83s cold start → always-on policy.

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
| True concurrent HTTP (N≥2) | ❌ CRASHES (Server disconnect / GGML_ASSERT) | T_PAR1 |
| GPU VRAM consumed (hybrid) | ~12GB split across both 5090s | T_CV3 |

**Policy:** Always-resident. Never on-demand. The 83s cold start is too high for transparent routing. Convergence must be running before any request that might escalate to it.

---

## Production launch command

```bash
/srv/ai/projects/ik_llama.cpp/build/bin/llama-server \
  -m /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf \
  -ngl 999 \
  --cpu-moe \
  --no-mmap \
  -b 4096 -ub 2048 \
  -t 32 \
  -np 4 \
  -c 131072 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --jinja \
  --host 0.0.0.0 --port 8002
```

**Model path:** 4 split files must all be in the same directory. Reference only `00001-of-00004.gguf`; loader finds the rest automatically.

**RAM prerequisite:** Sleep both Arclight vLLM processes at level=1 before starting Convergence if you need to free VRAM for attention layers (technically Convergence can coexist with Arclight since its weights are CPU-side, but GPU VRAM is shared for attention).

---

## Deploy script shortcut
```bash
./infra/scripts/deploy.sh ikllamacpp convergence
```

---

## Engine: ik_llama.cpp pr-1288

**Why pr-1288:** Mainline ik_llama.cpp HEAD predates Qwen3.5 GDN support. PR #1288 adds `LLM_ARCH_QWEN35MOE`, `build_qwen35moe()`, and `llama-delta-net.cpp` with `ssm_alpha`. Must use this branch.

**Why NOT vLLM for Convergence:** `--cpu-offload-gb` on a 123GB model involves constant PCIe weight-chunk round-trips per forward pass. ik_llama.cpp's `--cpu-moe` keeps MoE expert weights in RAM and only transfers attention/norm/embed — the correct split for sparse MoE.

**Flag notes (pr-1288 differs from assumed):**
- `-fa` is on by default — omit it entirely
- `-fmoe` is gone; fused MoE is default, disable with `-no-fmoe`
- `--cpu-moe` is the clean flag to pin all `ffn_gate/up/down_exps` to CPU RAM

**Rebuild after upstream changes:**
```bash
cd /srv/ai/projects/ik_llama.cpp
git pull origin pull/1288/head
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

**Verify architecture support:**
```bash
rg "qwen35|ssm_alpha|delta_net" /srv/ai/projects/ik_llama.cpp/src/
# Expected: llama-delta-net.cpp, llama-arch.cpp, llama-build-context.cpp
```

---

## Concurrency caveat (IMPORTANT)

`-np 4` sets the server's internal parallel slot count. **This does not mean 4 concurrent HTTP clients work safely.** T_PAR1 (2026-04-26) showed the pr-1288 server crashes at N≥2 concurrent HTTP requests with `Server disconnected` / `GGML_ASSERT(S > 0)`. T_CV4 measured 15.6 t/s by filling slots with sequential requests — not truly concurrent.

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

**Why UD-IQ2_M over UD-IQ3_XXS:**
- UD-IQ2_M (~123GB) + Arclight sleep weights (~44GB) + OS (~4GB) = ~171GB of 192GB. 21GB headroom with `--no-mmap`.
- UD-IQ3_XXS (~140GB) leaves only ~8GB — dangerously tight for `--no-mmap`.
- Both are within benchmark margin of error of BF16 on this 512-expert MoE architecture (Benjamin Marie independent evaluation, H200s).

**Why 397B over 122B:** TAU2 gap +14.7 points (86.7 vs ~72) in multi-step agentic orchestration — exactly what Convergence is invoked for.
