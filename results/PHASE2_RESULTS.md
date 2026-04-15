# Phase 2 — Model Selection Results

**Hardware:** RTX 5090 32 GB GDDR7, vLLM 0.19.0  
**Date:** 2026-04-15

---

## Phase 2.1 — Coder model

**Winner: Qwen3-Coder-30B-A3B-AWQ**

| Model | TPS | TTFT p50 | Completion | Verdict |
|-------|-----|----------|------------|---------|
| Qwen3-Coder-30B-A3B-AWQ | **251 t/s** | 29 ms | 100% | INCONCLUSIVE* |
| Qwen3.5-35B-A3B-AWQ | 22 t/s | — | — | FAIL |

*INCONCLUSIVE because decode_tps threshold (150) is calibrated for concurrency=4; at concurrency=1 251 t/s is well above the threshold. Verdict is effectively PASS.

**Why 35B lost:** bf16 VRAM = ~22 GiB, leaves only 510 MiB headroom — insufficient for CUDA graph profiling (needs 1.03 GiB). `--enforce-eager` required → 10× speed penalty.

**Config (30B):**
```
--tool-call-parser qwen3_coder --reasoning-parser qwen3
CTX: 32768, gpu_memory_utilization: 0.90
```

**Concurrency note (Phase 1.1 re-run):** At concurrency=4, 30B delivers 182 t/s/request (~730 t/s aggregate). CUDA graphs scale efficiently.

---

## Phase 2.2 — Thinker model

**Winner: Qwen3.5-27B-AWQ**

| Model | TPS | TTFT p50 | Completion | Quality | Verdict |
|-------|-----|----------|------------|---------|---------|
| Qwen3.5-27B-AWQ | **76 t/s** | 85 ms | 87.5% (7/8) | **4.0/5** | FAIL* |
| DeepSeek-R1-32B-AWQ | 74 t/s | 88 ms | 100% (8/8) | 2.6/5 | INCONCLUSIVE** |

*FAIL on completion rate (87.5% < 100%) due to th03 always exceeding token budget in `<think>`.  
**INCONCLUSIVE because decode_tps threshold (100) is calibrated for MoE; 74 t/s is strong for a 32B dense model.

**Quality breakdown (human review, 1–5):**

| Task | Qwen3.5-27B | R1-32B | Notes |
|------|-------------|--------|-------|
| th01 heisenbug | 5 | 3 | Qwen: threading.local+lru_cache interaction, probability analysis. R1: surface fix |
| th02 scheduling | 4 | 3 | Qwen: proper EDF + complexity proof. R1: schedule has gap bug |
| th03 architecture | 0 | 2 | Qwen: empty (token budget). R1: shallow bullet points |
| th04 pydantic perf | 4 | 2 | Qwen: correctly blames v1 compat layer internals. R1: wrong diagnosis |
| th05 consistency | 5 | 2 | Qwen: generation_id + key-at-start insight. R1: misses the point on scenario 4 |
| th06 refactor plan | 5 | 2 | Qwen: Strangler Fig + rollback plan. R1: generic 5-step outline |
| th07 closures | 5 | 4 | Both correct; Qwen includes dis.dis() internals + 4 fix options |
| th08 DCL proof | 4 | 3 | Both reasonable; Qwen more nuanced on CPython atomicity |
| **Total** | **32/40 (4.0)** | **21/40 (2.6)** | |

**Why Qwen3.5-27B is viable despite hybrid SSM architecture:**  
Default vLLM CUDA graph profiling needs 1.53 GiB for 51 batch sizes, but only 1.11 GiB free after loading 19.78 GiB model → OOM. Fix: `--max-num-seqs 1` limits profiling to 1 batch size → 0.04 GiB actual. CUDA graphs stay active. Speed: 76 t/s (vs 24 t/s with `--enforce-eager`).

**Config (Qwen3.5-27B):**
```
--tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-num-seqs 1
CTX: 32768, max_tokens: 8192, gpu_memory_utilization: 0.90
```

**Known issue — th03:** `th03_architecture_tradeoffs` consistently exhausts the 8192-token budget inside `<think>` without surfacing an answer. Workaround: run that specific task without `--reasoning-parser` or increase `--max-tokens` to 16384+.

---

## Stack decided so far

| Role | Model | Engine | TPS | Flags |
|------|-------|--------|-----|-------|
| Coder | Qwen3-Coder-30B-A3B-AWQ | vLLM | 251 t/s (seq) / 182 t/s×4 | `--tool-call-parser qwen3_coder` |
| Thinker | Qwen3.5-27B-AWQ | vLLM | 76 t/s | `--tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-num-seqs 1` |

---

## Remaining Phase 2 sub-tests

- **2.3** — Peak mode: Coder-Next vs Coder-30B (requires GGUF on llamacpp; 160B bf16 doesn't fit single GPU)
- **2.4** — Devstral: ELIMINATED. OOM (30.39 GiB, needs ~32 GiB bf16). Even if quantized: lower quality than 30B-AWQ, minimal reasoning. Not worth pursuing.
- **2.5** — Dense + spec-decode vs MoE (speed comparison on thinker path)
