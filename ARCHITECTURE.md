<!-- last_updated: 2026-04-25 R19 — Core retired; Extended Arclight = escalation; CPU-only Convergence; parallelism & big-context sections added -->
# ARCHITECTURE.md

Working architectural source of truth.

> **Status (R19, 2026-04-25):** Core tier **RETIRED** — escalation now via Extended Arclight (one Arclight sleeps, survivor goes TP=2). Arclight hot pair confirmed (coder TP=1 GPU0 + thinker TP=1 GPU1). Convergence CPU-only confirmed. Thinker TP=2 gated on T_KV3.

---

## Tier naming

| Tier | Name | Theme |
|------|------|-------|
| Coder + Thinker (always hot) | **Arclight** | Steins;Gate operation — fast, electric, concurrent |
| Arclight ×1 TP=2 (escalation mode) | **Extended Arclight** | One half sleeps; the survivor spans both GPUs |
| 397B king-behemoth (RAM-resident) | **Convergence** | Deeper than the Core — ephemeral, anomalous, omnipotent |
| 397B ultra-behemoth (system-exclusive) | **Singularity** | The end of the world — total system commitment |

> **Core tier RETIRED (2026-04-25):** The separate 80B Core model is suspended. Extended Arclight fills the escalation role with zero additional memory overhead. See DECISIONS.md.

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
│  │  └─────────────────┘    │       │  └─────────────────┘    │       │
│  │                         │       │                         │       │
│  │  ~9GB free (KV cache)   │       │  ~11GB free (KV cache)  │       │
│  │  (hot pair mode)        │       │  (hot pair mode)        │       │
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
│  │  CPU-ONLY (ngl=0)  ·  parallel: -np N                     │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                                      │
│  OpenCode v1.3+ — native multi-endpoint subagent routing             │
│  Arclight:     port 30000 (coder) + 30001 (thinker)                  │
│  Convergence:  port 8002                                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Design rationale

### Arclight — concurrent hot pair, TP=1-per-GPU

Coder (GPU0) and Thinker (GPU1) run as completely isolated vLLM processes. Physical GPU isolation eliminates CUDA context time-slicing — the mechanism that collapsed concurrent throughput to ~2% in T1.2. Each model gets full GPU memory bandwidth.

**V1 engine disabled by default** (`VLLM_V1_ENABLED=0` in deploy.sh): V1 was degrading coder TP=1 to ~18 t/s (engine bug, not a TP limitation). With V1 disabled, TP=1 is expected to recover ~237 t/s. Current production config is TP=2 at 232 t/s (confirmed via manual T6.1 run). TP=1 with V1 disabled is untested post-fix — T6.1 rerun will confirm which is optimal. TP=2 remains the hot pair config until that rerun.

**Thinker: TP=1 only.** TP=2 confirmed broken for GDN architecture (T2.4g). Do not change until T_KV3 resolves.

**Concurrent throughput:** vLLM `--max-num-seqs N` controls simultaneous requests per instance. At N=4, aggregate TPS is 2–3× single-seq TPS for small active-param MoE models. Current thinker is seq=1 (CUDA graph stability constraint on single GPU). Sweep target: T_PAR1.

OpenCode spawns both simultaneously. Parallel subagent calls to coder and thinker execute concurrently.

### Extended Arclight — TP=2 escalation (replaces Core)

When a single model needs more VRAM (large KV cache / long context) or higher per-session quality:

1. Sleep one Arclight model at level=1 (frees ~21–23GB on that GPU)
2. Restart the other model with `--tensor-parallel-size 2`

The surviving model now spans both GPUs. VRAM available for KV: 64GB total minus model weights minus sleeping model's ~4GB residual.

**Coder Extended mode** (tested, 232 t/s at TP=2 confirmed):
- Sleep thinker → restart coder TP=2
- KV budget: ~37GB at fp8 → ~60–75K tokens context

**Thinker Extended mode** (GATED on T_KV3):
- Thinker TP=2 is broken for GDN (T2.4g SETTLED). Cannot use until T_KV3 confirms a fix or a replacement thinker that supports TP=2.

**Mode switch time:** ~15–25s cold. With CUDA checkpoint/restore (T_KV2, HIGH PRIORITY), this drops to ~5s — restoring a pre-warmed CUDA process snapshot instead of recompiling graphs. NVIDIA `cuda-checkpoint` confirmed on driver 570+ (our RTX 5090 baseline). TP=2 multi-GPU + rootless Podman untested: see T_KV2.

### Convergence — ik_llama.cpp, RAM-resident, CPU-only

Architecturally independent from the vLLM tier. Runs as a native ik_llama.cpp process on port 8002. CPU-only (ngl=0) — all layers in DDR5 RAM. No VRAM contention with Arclight; runs in parallel, always-on.

**Parallel request handling:** ik_llama.cpp `-np N` enables N concurrent decode sequences. At 13 t/s single-seq (IQ2_M baseline with ngl=999; ngl=0 baseline TBD via T_CV1/T_CV2), `-np 4` yields 2–3× aggregate TPS. Essential for agentic runs where multiple subagents query Convergence simultaneously. See T_PAR1.

**Context ceiling:** `-c` sets max context, bounded by RAM. With ~65GB free above model weights at ngl=0:
- IQ2_M per-token KV (GDN hybrid, ~25% full-attention layers): estimated ~0.3–0.5 MB/token
- Theoretical ceiling: ~130–200K tokens
- Current deployment: `-c 16384` (conservative baseline)
- T_CV1 context sweep will establish the practical ceiling

**Two deployment modes:**
1. **Always-resident:** Start at boot. ~123GB RAM constant. Recommended.
2. **Cold-start on demand:** Start when requested. Cold start time: measuring via T_CV1.

RAM budget with Convergence + Arclight:
```
Convergence:           123 GB (pinned, --no-mmap)
Arclight weights:       44 GB (resident in VRAM — not RAM)
OS + CUDA + headroom:  ~4 GB
Total in-use:         ~171 GB of 192 GB (89%)
```

**Better quant path:** Core (80B) retirement frees ~44GB RAM that was previously reserved for Core sleep weights. With that freed, Q3_K_M (~140GB) fits cleanly alongside Arclight — meaningful quality improvement over IQ2_M for a 397B model.

### Singularity — system-exclusive ultra-behemoth

Takes ALL system resources — stops Arclight + Convergence.
- **Model:** Qwen3.5-397B at Q3_K_M or Q4_K_M (~140–180GB)
- **Engine:** ik_llama.cpp (same binary, different quant)
- **Placement:** Full GPU offload for attention/norm/embed; MoE experts in RAM
- **Startup:** ~70s warm cache
- **Recovery:** Restart Convergence then Arclight (~60–90s; no Core to restart)

---

## Big context modes

| Mode | Who sleeps | Active model | VRAM for KV | Approx max context |
|------|------------|--------------|-------------|-------------------|
| Hot pair (normal) | nobody | coder TP=1 + thinker TP=1 | ~9GB + ~11GB | ~32K each |
| Extended coder | thinker | coder TP=2 | ~37GB at fp8 | ~60–75K |
| Extended thinker | coder | thinker TP=2 | ~37GB at fp8 | ~150K* |
| Convergence | nobody | 397B CPU-only | RAM KV | ~130–200K** |
| Singularity | everyone | 397B high-quant | GPU attention | 32K–128K |

\* Extended thinker GATED on T_KV3 — TP=2 broken for GDN until resolved.
\*\* Convergence context ceiling untested — T_CV1 context sweep pending.

**CPU KV overflow (`--swap-space N`):** KV blocks spill to DRAM when GPU KV is full. Useful for prefill-heavy flows (ingest large doc, generate short answer). Generation TPS penalty proportional to spill fraction.

---

## Parallelism summary

| Tier | Mechanism | Single-seq TPS | Lever | Expected aggregate |
|------|-----------|---------------|-------|-------------------|
| Arclight coder | vLLM `--max-num-seqs` | 232 t/s (TP=2) | N=4 | ~500–700 t/s |
| Arclight thinker | vLLM `--max-num-seqs` | 77 t/s | N=4 | ~150–230 t/s |
| Convergence | ik_llama.cpp `-np` | ~13 t/s (ngl=999 baseline; ngl=0 TBD) | N=4 | ~25–40 t/s |
| Singularity | ik_llama.cpp `-np` | TBD | N=2 | TBD |

See T_PAR1 for the measurement procedure.

---

## Process sequences

### Normal operation (Arclight hot pair)
```
vllm serve qwen3.6-35b   --port 30000 --gpu 0 --tp 1    [running]
vllm serve qwen3.6-27b   --port 30001 --gpu 1 --tp 1    [running]
ik_llama-server 397b     --port 8002                    [running or cold]
```

### Extended Arclight — coder big-context mode
```bash
# Sleep thinker
curl -X POST http://localhost:30001/sleep?level=1           # ~4s

# Restart coder at TP=2 (hot-restart with CUDA checkpoint ~5s; cold ~170–300s)
podman stop arclight-thinker
VLLM_V1_ENABLED=0 VLLM_USE_V1=0 \
./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
    --gpu-mem-util 0.90 --ctx 65536 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3

# Use coder on port 30000 with ~60–75K context

# Restore hot pair
podman stop arclight-coder-tp2
./infra/scripts/deploy.sh vllm gpu0 cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit [normal flags]
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ [normal flags]
curl -X POST http://localhost:30001/wake_up
```
Total round-trip: ~15–25s cold; ~5s with CUDA checkpoint (T_KV2).

### Escalate to Convergence
Always-resident on port 8002. No swap required.
If starting from cold:
```bash
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
| Ext. Arclight coder | — (on-demand) | same as coder | vLLM TP=2 GPU0+1 | 30000 | ~23GB split | 0 |
| Convergence | convergence | unsloth/Qwen3.5-397B UD-IQ2_M | ik_llama.cpp | 8002 | 0 | ~123GB |
| Singularity | singularity | Qwen3.5-397B Q4_K_M | ik_llama.cpp | 8003 | ~30GB | ~180GB |
| ~~Core (RETIRED)~~ | — | ~~Qwen3-Next-80B-A3B-AWQ~~ | suspended | — | — | — |

---

## Engine binaries and paths

### vLLM (Arclight)
- Container image: rootless podman via `infra/compose/`
- Version: vLLM 0.19.x
- Deploy: `infra/scripts/deploy.sh`
- V1 engine: **disabled by default** (`VLLM_V1_ENABLED=0`) — stability fix for Blackwell

### ik_llama.cpp (Convergence)
- Repository: `/srv/ai/projects/ik_llama.cpp`
- Branch: `pr-1288` (Qwen3.5 MoE support)
- Binary: `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`
- Benchmark binary: `/srv/ai/projects/ik_llama.cpp/build/bin/llama-bench`

**To rebuild ik_llama.cpp after pulling new changes to pr-1288:**
```bash
cd /srv/ai/projects/ik_llama.cpp
git pull origin pull/1288/head
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
2. ~~**Qwen3-Coder-Next-80B-A3B-AWQ TP=2 decode speed.**~~ **SETTLED — T1.3 PASS. (Core tier retired.)**
3. ~~**GLM-4.7-Flash MLA auto-detection.**~~ **SETTLED — T2.1 INCONCLUSIVE (MLA active / V1 tool-broken).**
4. **CPU prefix cache survival across sleep/wake.** T3.4.
5. **Convergence startup time + context sweep.** T_CV1 (amended — add context sweep).
6. **Convergence thread count optimal value.** T_CV2.
7. **Convergence partial GPU expert offload.** T_CV3.
8. ~~**Gemma4-31B as Arclight thinker.**~~ **SETTLED (T2.3b, 2026-04-24): REJECTED.**
9. ~~**Gemma4-31B as Arclight coder.**~~ **SKIPPED (2026-04-25): Qwen3.6-35B-A3B clearly superior.**
10. ~~**kvcached Phase B with non-Mamba thinker.**~~ **CLOSED: GDN (Qwen3.6-27B) not supported by kvcached (DeltaNetSpec missing, no upstream timeline). Static `--max-model-len` asymmetry is the available lever for per-model context budgets.**
11. **Coder big-context mode: max usable context with thinker sleeping.** T_KV1.
12. **CUDA checkpoint/restore for TP=2 hot-restart** (HIGH PRIORITY). T_KV2.
13. **Thinker TP=2 fix or alternative model — GATE for finalized no-Core architecture.** T_KV3.
14. **Parallel throughput sweep (--max-num-seqs for vLLM, -np for ik_llama.cpp).** T_PAR1.
15. **Arclight coder TP=1 with V1 disabled: confirm ~237 t/s recovery.** Needed via T6.1 rerun.

## How to change this document

This is the **plan**, not the record. When testing invalidates part of it:
1. Update `RESEARCH_STATE.md` with the finding.
2. In research mode, update this file.
3. Update `TESTING_QUEUE.md` if new questions arise.
