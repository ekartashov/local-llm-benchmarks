# Architecture: Current Deployment

> **Status (R27, 2026-04-26):** Arclight hot-pair settled. Extended Arclight (65K ctx, 0.28s CRIU restart) settled. Convergence always-resident settled. Thinker TP=2 gated on T_KV3. Core RETIRED.

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
│  │  Qwen3.6-35B-A3B-AWQ   │       │  Qwen3.6-27B-AWQ       │         │
│  │  TP=1, fp8 KV          │       │  TP=1, fp8 KV, cp-ON   │         │
│  │  port 30000            │       │  port 30001            │         │
│  │  ~23GB VRAM            │       │  ~21GB VRAM            │         │
│  │  ~9GB free (KV cache)  │       │  ~11GB free (KV cache) │         │
│  └────────────────────────┘       └────────────────────────┘         │
│                                                                      │
│  DDR5 RAM (192 GB)                                                   │
│  ├─ Arclight sleep weights (level=1):  ~23 + ~21 = ~44 GB            │
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

### Arclight — concurrent hot pair, TP=1-per-GPU
Physical GPU isolation (coder GPU0, thinker GPU1) eliminates CUDA context time-slicing — the mechanism that collapsed concurrent throughput to ~2% in T1.2. Each model gets full GPU memory bandwidth.

**Coder:** TP=2 is the production config at 232 t/s (T6.1 manual baseline, 2026-04-25). TP=1 suffers Reasoning Collapse (hallucination loops from Triton/FLA kernel shape mismatches in eager mode) and cannot serve as production without a non-eager path.

**Thinker:** TP=1 only. TP=2 confirmed broken for GDN architecture (T2.4g). `--max-num-seqs 1` required for CUDA graph stability on single GPU. Do not change until T_KV3 resolves.

**Concurrency (max-num-seqs):** UNKNOWN for production. T_PAR1 Coder/Thinker rerun required. Prior Gemini Flash numbers (1,196 t/s coder at N=8) were fabricated — no measurement exists.

### Extended Arclight — TP=2 escalation
See [extended-arclight.md](extended-arclight.md) for full procedure.
1. Sleep one Arclight model at level=1 (~4s, frees ~21–23GB on that GPU)
2. Restart the other model with `--tensor-parallel-size 2`
3. CRIU hot-restart enables 0.28s mode switches (T_KV2 SETTLED)

**Coder Extended mode (65K ctx, SETTLED T_KV1):** Sleep thinker → restart coder TP=2 at ctx=65536. KV budget ~37GB at fp8.

**Thinker Extended mode:** GATED on T_KV3. TP=2 broken for GDN until fix or non-GDN replacement found.

### Convergence — ik_llama.cpp, always-resident
See [convergence.md](convergence.md) for full guide.
- Always-resident (83s cold start too high for on-demand). Never kill it without planning restart time.
- `-ngl 999 --cpu-moe`: attention/norm/embed on GPU, MoE experts in RAM. 13.99 t/s.
- Context ceiling: **128k tokens** (T_CV1). Beyond this requires Singularity.
- Concurrent clients: **N=1 only** (pr-1288 crashes at N≥2). `-np 4` is sequential pipelining only.

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
vllm serve qwen3.6-35b   --port 30000 --gpu 0 --tp 1    [running]
vllm serve qwen3.6-27b   --port 30001 --gpu 1 --tp 1    [running]
ik_llama-server 397b     --port 8002                    [always-resident]
```

### Extended Arclight — coder big-context mode
```bash
# Sleep thinker
curl -X POST http://localhost:30001/sleep?level=1           # ~4s

# Restart coder at TP=2 (CRIU hot-restart ~0.28s; cold ~100s)
# See docs/procedures/criu-ops.md for CRIU procedure
VLLM_V1_ENABLED=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
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
| Arclight coder | arclight-coder | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | vLLM TP=2 GPU0+1 | 30000 | ~23GB | 0 |
| Arclight thinker | arclight-thinker | QuantTrio/Qwen3.6-27B-AWQ | vLLM TP=1 GPU1 | 30001 | ~21GB | 0 |
| Ext. Arclight | — (on-demand) | same as coder | vLLM TP=2 GPU0+1 | 30000 | ~60GB split | 0 |
| Convergence | convergence | unsloth/Qwen3.5-397B UD-IQ2_M | ik_llama.cpp | 8002 | ~12GB (attn) | ~123GB |
| Singularity | singularity | Qwen3.5-397B Q4_K_M | ik_llama.cpp | 8003 | ~30GB | ~180GB |
| ~~Core (RETIRED)~~ | — | ~~Qwen3-Next-80B-A3B-AWQ~~ | suspended | — | — | — |

---

## Engine paths

### vLLM (Arclight)
- Container: rootless podman via `infra/compose/`, version 0.19.x
- Deploy: `infra/scripts/deploy.sh`
- V1 engine: **disabled** (`VLLM_V1_ENABLED=0`) — stability fix for Blackwell sm_120

### ik_llama.cpp (Convergence)
- Repo: `/srv/ai/projects/ik_llama.cpp`, branch `pr-1288`
- Binary: `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- Rebuild after pr-1288 pull:
  ```bash
  cd /srv/ai/projects/ik_llama.cpp && git pull origin pull/1288/head
  cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j$(nproc)
  ```
