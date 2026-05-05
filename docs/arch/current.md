# Architecture: Current Deployment

> **Status (R35, 2026-05-05):** Arclight hot-pair settled. Extended Arclight (65K ctx, 0.28s CRIU restart) settled. Convergence dual-mode settled: always-resident (max TPS) or on-demand CRIU restore (~12s, BENCH_22). Thinker 128K context verified (SETTLED via ik_llama.cpp main). Core RETIRED.

---

## Tier naming

| Tier | Name | Description |
|------|------|-------------|
| Coder + Thinker (always hot) | **Arclight** | Fast, concurrent hot-pair. TP=1-per-GPU. |
| One Arclight TP=2 (escalation) | **Extended Arclight** | One model sleeps; survivor spans both GPUs. |
| 397B in RAM, always-on | **Convergence** | CPU+GPU hybrid, 13.99 t/s, 128K ctx ceiling. |
| 397B system-exclusive | **Singularity** | Stops everything; full GPU for attention layers. |

Core (80B Qwen3-Next) RETIRED 2026-04-25 — Extended Arclight fills its role.

---

## Deployment topology

```
┌──────────────────────────────────────────────────────────────────────┐
│  ZRH01-AIRIG (i9-14900K, 192GB DDR5, 2×RTX 5090 32GB, no NVLink)    │
│                                                                      │
│  ┌────────────────────────┐       ┌────────────────────────┐         │
│  │  GPU 0  (32GB GDDR7)   │       │  GPU 1  (32GB GDDR7)   │         │
│  │                        │       │                        │         │
│  │  vLLM: ARCLIGHT CODER  │       │  vLLM: ARCLIGHT THINKER│         │
│  │  Qwen3.6-35B-A3B-PQ    │       │  Qwen3.6-27B-PQ        │         │
│  │  TP=1, fp8 KV, V1-ON   │       │  TP=1, fp8 KV, MTP-n3  │         │
│  │  port 30000            │       │  port 30001            │         │
│  │  ~22GB VRAM            │       │  ~22GB VRAM            │         │
│  │  ~10GB free (KV budget)│       │  ~10GB free (KV budget)│         │
│  └────────────────────────┘       └────────────────────────┘         │
│                                                                      │
│  DDR5 RAM (192 GB)                                                   │
│  ├─ Arclight sleep weights (level=1):  ~23 + ~21 = ~44 GB            │
│  │  NOTE: obsolete once CRIU replaces sleep mode (T_CRIU3).          │
│  │  With CRIU: 0 bytes retained in RAM → frees 44 GB for            │
│  │  higher-quant Convergence (UD-IQ3_XXS ~140 GB becomes viable).   │
│  ├─ Convergence model (UD-IQ2_M):              ~123 GB               │
│  ├─ OS + CUDA runtime:                           ~4 GB               │
│  └─ Total (Convergence active):               ~171 GB  (89%)         │
│                                                                      │
│  ik_llama.cpp: CONVERGENCE                                           │
│  Qwen3.5-397B-A17B UD-IQ2_M, port 8002                               │
│  -ngl 999 --cpu-moe (attention on GPU, MoE experts in RAM)           │
│  Production: -np 4 -t 32 (sequential pipelining, not true parallel)  │
│                                                                      │
│  OpenCode v1.3+ — native multi-endpoint subagent routing             │
│  Arclight: port 30000 (coder) + 30001 (thinker)                      │
│  Convergence: port 8002                                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Design rationale

### Arclight — concurrent hot pair, TP=1-per-GPU (current mode)
Physical GPU isolation (coder GPU0, thinker GPU1) eliminates CUDA context time-slicing — the mechanism that collapsed concurrent throughput to ~2% in T1.2. Each model gets full GPU memory bandwidth.

**Coder:** TP=1 is the production config as of BENCH_23 (2026-05-05). Logical stability (no reasoning collapse) is achieved by using the **vLLM V1 engine** and **CUDA graph capture**. Performance is capped at ~60 t/s due to SM120 grouped GEMM software maturity.

**Thinker:** TP=1 only for vLLM (GDN broken at TP=2). For extended context (128K), use ik_llama.cpp on the `main` branch with `--tensor-split 0.5,0.5` (SETTLED BENCH_15).

**Concurrency (max-num-seqs):** UNKNOWN for production. T_PAR1 Coder/Thinker rerun required. Prior Gemini Flash numbers (1,196 t/s coder at N=8) were fabricated — no measurement exists.

### Sequential TP=2 / Extended mode — complementary operating mode (NOT an alternative to Arclight)
One model active at a time, spanning both GPUs at TP=2. Others CRIU-checkpointed, restored in 0.28s.

**Both Arclight and Sequential TP=2 must coexist.** They serve distinct invocation patterns:
- Arclight (concurrent hot-pair): agent frameworks that fan out parallel subagents (e.g., OpenCode spawning coder + thinker subagents simultaneously). Concurrency is determined by the framework design and prompts, not our choice.
- Sequential TP=2 / Extended Arclight: deep single-context work (long reasoning chains, large codebase analysis, multi-document synthesis). Full GPU bandwidth, large KV budget, no concurrency needed.

**T_PAR1 data determines usage frequency of each mode**, which informs research priority — but not which to build. Both must be supported. The original Core tier (80B TP=2) was the permanent occupant of the "deep single-context escalation" slot. Extended Arclight fills that role with the same hardware. T_KV3 (Extended Thinker) is the remaining gap in this slot.

See T_CRIU3 in queue for the checkpoint-library implementation that makes 0.28s mode switching reliable.

### Extended Arclight — TP=2 escalation
See [extended-arclight.md](extended-arclight.md) for full procedure.
1. Sleep one Arclight model at level=1 (~4s, frees ~21–23GB on that GPU)
2. Restart the other model with `--tensor-parallel-size 2`
3. CRIU hot-restart enables 0.28s mode switches (T_KV2 SETTLED)

**Coder Extended mode (65K ctx, SETTLED T_KV1):** Sleep thinker → restart coder TP=2 at ctx=65536. KV budget ~37GB at fp8

**Thinker Extended mode:** SETTLED (BENCH_15). Qwen3.6-27B (dense) fully supported at 128K context via ik_llama.cpp `main` branch using layer-split parallelism (`--tensor-split`).

### Convergence — ik_llama.cpp, dual-mode (settled)
See [convergence.md](convergence.md) for full guide.
- Always-resident mode: `-ngl 999 --cpu-moe` (13.99 t/s isolate).
- On-demand mode: CRIU restore is viable (~12s restore-to-interactive) when using `GGML_CUDA_NO_PINNED=1` + GGUF/CRIU prewarm (BENCH_22).
- Co-load profile uses reduced offload (e.g., `-ngl 15`) to fit alongside Arclight.
- Context ceiling: **128k tokens** (T_CV1). Beyond this requires Singularity.
- Concurrent clients: **N=1** (architectural limit, PR #1288). `-np 4` is sequential pipelining only.

---

## Big context modes

| Mode | Who sleeps | Active model | VRAM for KV | Max context |
|------|------------|--------------|-------------|-------------|
| Hot pair (normal) | nobody | coder TP=1 + thinker TP=1 | ~9GB + ~11GB | ~32K each |
| Extended coder | thinker | coder TP=2 | ~37GB at fp8 | **65K** (T_KV1) |
| Extended thinker | coder | thinker TP=2 | ~37GB at fp8 | ~150K* |
| Convergence | nobody (GPU shared) | 397B CPU+GPU hybrid | RAM KV | 128K (T_CV1) |
| Singularity | everyone | 397B high-quant | GPU attention | 32K–128K |

\* Extended thinker GATED on T_KV3.

---

## Process sequences

### Normal operation (Arclight hot pair)
```
vllm serve qwen3.6-35b-pq   --port 30000 --gpu 0 --tp 1 --v1 [running]
vllm serve qwen3.6-27b-pq   --port 30001 --gpu 1 --tp 1      [running]
ik_llama-server 397b        --port 8002                      [always-resident]
```

### Extended Arclight — coder big-context mode
```bash
# Sleep thinker
curl -X POST http://localhost:30001/sleep?level=1           # ~4s

# Restart coder at TP=2 (CRIU hot-restart ~0.28s; cold ~100s)
# See docs/procedures/criu-ops.md for CRIU procedure
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
    --gpu-mem-util 0.90 --ctx 65536 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3

# Restore hot pair
podman stop arclight-coder-tp2
./infra/scripts/deploy.sh vllm gpu0 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit [normal flags]
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ [normal flags]
curl -X POST http://localhost:30001/wake_up
```

---

## Model registry

| Tier | Container/Process | Model | Engine | Port | VRAM | RAM |
|------|------------------|-------|--------|------|------|-----|
| Arclight coder | arclight-coder | rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm | vLLM TP=1 GPU0 (V1) | 30000 | ~22GB | 0 |
| Arclight thinker | arclight-thinker | rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm | vLLM TP=1 GPU1 (V0) | 30001 | ~22GB | 0 |
| Ext. Arclight | — (on-demand) | same as coder | vLLM TP=2 GPU0+1 | 30000 | ~60GB split | 0 |
| Convergence | convergence | unsloth/Qwen3.5-397B UD-IQ2_M | ik_llama.cpp | 8002 | ~12GB (attn) | ~123GB |
| Singularity | singularity | Qwen3.5-397B Q4_K_M | ik_llama.cpp | 8003 | ~30GB | ~180GB |
| ~~Core (RETIRED)~~ | — | ~~Qwen3-Next-80B-A3B-AWQ~~ | suspended | — | — | — |

---

## Engine paths

### vLLM (Arclight)
- Container: rootless podman via `infra/compose/`, version 0.19.x
- Deploy: `infra/scripts/deploy.sh`
- V1 engine: **enabled** for Coder TP=1 stability; **disabled** for Thinker (compressed-tensors path).

### ik_llama.cpp (Convergence)
- Repo: `/srv/ai/projects/ik_llama.cpp`, branch `main`
- Binary: `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- Rebuild instructions:
  ```bash
  cd /srv/ai/projects/ik_llama.cpp && git checkout main && git pull origin main
  cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j$(nproc)
  ```
