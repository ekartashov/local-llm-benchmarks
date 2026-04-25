<!-- last_updated: 2026-04-25 R19 — Convergence CPU-only, Singularity tier added, Thinker → Qwen3.6-27B -->
# ARCHITECTURE.md

Working architectural source of truth.

> **Status (R12, 2026-04-20):** Three-tier architecture confirmed. Tier naming settled.
> Arclight (hot pair) confirmed T1.1–T1.3. Core (behemoth) confirmed T1.3.
> Convergence (king-behemoth) deployed and measured — 13.15 t/s gen, 158 t/s PP at 2k batch.

---

## Tier naming

| Tier | Name | Theme |
|------|------|-------|
| Coder + Thinker (always hot) | **Arclight** | Steins;Gate operation — fast, electric, concurrent |
| 80B behemoth (asleep in vLLM) | **Core** | Undertale Core — powerful, invoked on escalation |
| 397B king-behemoth (RAM-resident) | **Convergence** | Deeper than the Core — ephemeral, anomalous, omnipotent |
| 397B ultra-behemoth (system-exclusive) | **Singularity** | The end of the world — total system commitment |

---

## Deployment topology

```
┌──────────────────────────────────────────────────────────────────────┐
│  ZRH01-AIRIG (i9-14900K, 192GB DDR5, 2×RTX 5090 32GB, no NVLink)     │
│                                                                      │
│  ┌─────────────────────────┐       ┌─────────────────────────┐       │
│  │  GPU 0  (32GB GDDR7)    │       │  GPU 1  (32GB GDDR7)    │       │
│  │                         │       │                         │       │
│  │  ┌─────────────────┐    │       │  ┌─────────────────┐    │       │
│  │  │ vLLM: ARCLIGHT  │    │       │  │ vLLM: ARCLIGHT  │    │       │
│  │  │ CODER           │    │       │  │ THINKER         │    │       │
│  │  │ Qwen3.6-35B-A3B │    │       │  │ Qwen3.6-27B     │    │       │
│  │  │ AWQ TP=1        │    │       │  │ AWQ TP=1        │    │       │
│  │  │ port 30000      │    │       │  │ port 30001      │    │       │
│  │  │ AWAKE           │    │       │  │ AWAKE           │    │       │
│  │  │ ~23GB VRAM      │    │       │  │ ~21GB VRAM      │    │       │
│  │  └────────────┬────┘    │       │  └────────────┬────┘    │       │
│  │               │ sleep   │       │               │ sleep   │       │
│  │  ┌────────────▼────┐    │       │  ┌────────────▼────┐    │       │
│  │  │ vLLM: CORE      │    │       │  │ vLLM: CORE      │    │       │
│  │  │ Qwen3-Next-80B  │    │       │  │ (TP=2 spans     │    │       │
│  │  │ AWQ TP=2 ════════════╪═══════╪══╡  both GPUs)     │    │       │
│  │  │ port 30002      │    │       │  │ port 30002      │    │       │
│  │  │ ASLEEP L1       │    │       │  │ ASLEEP L1       │    │       │
│  │  │ ~20GB/GPU VRAM  │    │       │  │ ~20GB/GPU VRAM  │    │       │
│  │  └─────────────────┘    │       │  └─────────────────┘    │       │
│  │                         │       │                         │       │
│  │      IDLE VRAM          │       │      IDLE VRAM          │       │
│  │      (during            │       │      (during            │       │
│  │   CONVERGENCE ops)      │       │   CONVERGENCE ops)      │       │
│  └─────────────────────────┘       └─────────────────────────┘       │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────┐       │
│  │  DDR5 RAM  (192 GB)                                       │       │
│  │                                                           │       │
│  │  Arclight sleep weights (level=1):  ~23 + ~21 = ~44 GB    │       │
│  │  Convergence model (UD-IQ2_M):                 ~123 GB    │       │
│  │  OS + CUDA runtime:                              ~4 GB    │       │
│  │  ───────────────────────────────────────────────────────  │       │
│  │  Total in-use (Convergence active):            ~171 GB    │       │
│  │  Headroom:                                      ~21 GB    │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────┐       │
│  │  ik_llama.cpp: CONVERGENCE                                │       │
│  │  Qwen3.5-397B-A17B UD-IQ2_M                               │       │
│  │  port 8002                                                │       │
│  │  CPU-ONLY (ngl=0)                                         │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                                      │
│  OpenCode v1.3+ — native multi-endpoint subagent routing             │
│  Arclight:     port 30000 (coder) + 30001 (thinker)                  │
│  Core:         port 30002                                            │
│  Convergence:  port 8002                                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Three-tier design rationale

### Arclight — concurrent hot pair, TP=1-per-GPU

Coder (GPU0) and Thinker (GPU1) run as completely isolated vLLM processes. Physical GPU isolation eliminates CUDA context time-slicing — the mechanism that collapsed concurrent throughput to ~2% in T1.2. Each model gets full GPU memory bandwidth.

Coder at TP=1 on single 5090 is actually *faster* (237 t/s) than at TP=2 (212 t/s) — allreduce overhead removed. Thinker drops from ~106 t/s to ~76 t/s but this is acceptable; the thinker is not the latency-critical path.

OpenCode spawns both simultaneously. Parallel subagent calls to coder and thinker execute concurrently.

### Core — TP=2 asleep, wakes to full 64GB

Waking Core requires sleeping both Arclight models (to free GPU VRAM). The sleep + wake cycle takes ~15-25s end-to-end. Invoked when both hot models fail to solve a problem, or when an explicit `@core` escalation is requested.

Core runs at TP=2 with `--gpu-memory-utilization 0.95` and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`. After Core session, sleep Core and wake both Arclight models.

### Convergence — ik_llama.cpp, RAM-resident, separate process

Convergence is architecturally independent from the vLLM tier. It runs as a native ik_llama.cpp process and communicates via an OpenAI-compatible HTTP API on port 8002. It does not interact with vLLM's sleep mechanism in production because it is **CPU-only (-ngl 0)**.

The split: In production, **all layers reside in DDR5 RAM**. This eliminates VRAM contention with Arclight, allowing Convergence to stay resident without requiring Arclight to sleep.

**Why CPU-only:** with Arclight filling both GPUs (~23GB GPU0 + ~21GB GPU1), Convergence cannot use GPU without VRAM conflict. CPU-only means it runs truly in parallel, always-on, zero GPU contention.

**Two deployment modes:**
1. **Always-resident:** Start at boot, stay running. Warmest possible response. Uses ~123GB RAM constantly but we have 192GB so this is fine when Arclight is not sleeping.
2. **Cold-start on demand:** Start when requested, kill after session. Free the 123GB RAM for other uses. Cold start: 60-120s from NVMe (measuring needed, T_CV1). With `--no-mmap` the model loads fully before serving.

With Arclight active, RAM budget with Convergence resident:
```
Convergence:           123 GB (pinned, --no-mmap)
Arclight weights:       44 GB (resident in VRAM)
OS + CUDA + headroom:  ~4 GB
Total:                ~171 GB of 192 GB (89%)
```

This is comfortable. Always-resident is the recommended mode.

### Singularity — system-exclusive ultra-behemoth

The 4th and final tier. Takes ALL system resources — stops Arclight + Core + Convergence.
- **Model:** Qwen3.5-397B at Q3_K_M or Q4_K_M (~140-180GB)
- **Engine:** ik_llama.cpp (same binary, different quant)
- **Placement:** Full GPU offload for attention/norm/embed; MoE experts in RAM.
- **Startup:** ik_llama.cpp ~70s warm cache, ~30s with all RAM free.
- **Recovery:** After session, restart Convergence, then Arclight, then Core (~300-400s total for vLLM).

---

## Process startup and swap sequences

### Normal operation (Arclight active)
```
vllm serve qwen3.6-35b   --port 30000 --gpu 0      [running]
vllm serve qwen3.5-27b   --port 30001 --gpu 1      [running]
vllm serve qwen3-next-80b --port 30002 --tp 2      [sleeping L1]
ik_llama-server 397b      --port 8002              [running or cold]
```

### Escalate to Core
```bash
curl -X POST http://localhost:30000/sleep?level=1   # ~4s
curl -X POST http://localhost:30001/sleep?level=1   # ~4s (parallel ok)
curl -X POST http://localhost:30002/wake_up          # ~3-6s
# use Core on port 30002
curl -X POST http://localhost:30002/sleep?level=1   # ~4s after done
curl -X POST http://localhost:30000/wake_up          # ~1s (weights in RAM)
curl -X POST http://localhost:30001/wake_up          # ~1s (weights in RAM)
```
Total round-trip: ~15-25s.

### Escalate to Convergence
Convergence is always-resident on port 8002. No swap required — just route the request there.
If starting from cold:
```bash
# Using the deployment script
./infra/scripts/deploy.sh ikllamacpp convergence
```

### Escalate to Singularity
1. Stop all other tiers.
2. Load the high-quant GGUF (Q3_K_M or Q4_K_M).
3. Maximize GPU offload.

---

## Model registry quick reference

| Tier | Name | Model | Engine | Port | VRAM | RAM |
|------|------|-------|--------|------|------|-----|
| Arclight coder | arclight-coder | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | vLLM TP=1 GPU0 | 30000 | ~23GB | 0 |
| Arclight thinker | arclight-thinker | QuantTrio/Qwen3.6-27B-AWQ | vLLM TP=1 GPU1 | 30001 | ~21GB | 0 |
| Core | core | cyankiwi/Qwen3-Next-80B-A3B-AWQ | vLLM TP=2 | 30002 | ~40GB | ~40GB (L1) |
| Convergence | convergence | unsloth/Qwen3.5-397B UD-IQ2_M | ik_llama.cpp | 8002 | 0 | ~123GB |
| Singularity | singularity | Qwen3.5-397B Q4_K_M | ik_llama.cpp | 8003 | ~30GB | ~180GB |

---

## Engine binaries and paths

### vLLM (Arclight + Core)
- Container image: rootless podman via `infra/compose/`
- Version: vLLM 0.19.x
- Deploy: `infra/scripts/deploy.sh`

### ik_llama.cpp (Convergence)
- Repository: `/srv/ai/projects/ik_llama.cpp`
- Branch: `pr-1288` (Qwen3.5 MoE support)
- Binary: `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- Benchmark binary: `/srv/ai/projects/ik_llama.cpp/build/bin/llama-bench`
- Benchmark script: `/srv/ai/projects/local-llm-benchmarks/benchmarks/bench_convergence.sh`

**To rebuild ik_llama.cpp after pulling new changes to pr-1288:**
```bash
cd /srv/ai/projects/ik_llama.cpp
git pull origin pull/1288/head   # or git fetch + checkout pr-1288
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### Mainline llama.cpp (fallback)
- Repository: `/srv/ai/projects/llama.cpp`
- Branch: master (b8851 or later)
- Supports Qwen3.5 architecture but lacks `-fmoe` optimization
- Use only if pr-1288 has issues

---

## Known unknowns (feed into `TESTING_QUEUE.md`)

1. ~~**Sleep Mode works under rootless podman.**~~ **SETTLED — T1.1 PASS.**
2. ~~**Qwen3-Coder-Next-80B-A3B-AWQ TP=2 decode speed.**~~ **SETTLED — T1.3 PASS.**
3. ~~**GLM-4.7-Flash MLA auto-detection.**~~ **SETTLED — T2.1 INCONCLUSIVE (MLA active / V1 tool-broken).**
4. **CPU prefix cache survival across sleep/wake.** T3.4.
5. **Convergence startup time.** Cold-start from NVMe with `--no-mmap`: expected 60-120s but not measured. T_CV1.
6. **Convergence thread count optimal value.** Baseline at 32 threads; 16 or 24 may be better for small expert matrices. T_CV2.
7. **Convergence partial GPU expert offload.** GPU barely used (~10GB of 64GB available). Offloading first N layers' expert weights to GPU could improve generation speed. T_CV3.
8. ~~**Gemma4-31B as Arclight thinker.**~~ **SETTLED (T2.3b, 2026-04-24): REJECTED as thinker.** Redirected to coder candidate T2.3c.
9. ~~**Gemma4-31B as Arclight coder.**~~ **SKIPPED (2026-04-25): Benchmark evidence shows Qwen3.6-35B-A3B clearly superior.** T2.3c.
10. **kvcached Phase B with non-Mamba thinker.** T1.5 re-run after kvcached-compatible thinker is settled.

## How to change this document

This is the **plan**, not the record. When testing invalidates part of it:
1. Update `RESEARCH_STATE.md` with the finding.
2. In research mode, update this file.
3. Update `TESTING_QUEUE.md` if new questions arise.