# Model Decisions

Role assignments, why each model won/lost, and production configurations.
Use `rg "<model-name>" docs/decisions/models.md` to find any model quickly.

---

## Active roles

### Arclight Coder: Qwen3.6-35B-A3B-AWQ (cyankiwi)
**SETTLED (T2.5, 2026-04-18).** Supersedes Qwen3-Coder-30B.
- **Performance:** 237.1 t/s seq=1, 232 t/s TP=2 (T6.1 baseline 2026-04-25)
- **Quality:** 96.7% tool-call reliability (29/30), 100% quality completion (10/10)
- **Production config:**
  ```
  Model: cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
  TP=2, GPU0+1
  --gpu-memory-utilization 0.90
  --kv-cache-dtype fp8
  --ctx 32768 (65536 in Extended mode)
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
  VLLM_V1_ENABLED=0
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
  ```
- **Why TP=2 mandatory:** TP=1 on vLLM 0.19.0 triggers Reasoning Collapse (hallucination loops, Triton/FLA kernel shape mismatch in eager mode). TP=1 regresses to ~20 t/s. Not worth debugging.
- **Why coder wins:** SWE-bench Verified 73.4%, Terminal-Bench 51.5%, MCPMark 37.0%, WideSearch 60.1% — dominates every agentic benchmark relevant to this workload. Gemma4-31B ties on pure coding (LiveCodeBench 80.0 vs 80.4) but loses on all agentic tasks.

### Arclight Thinker: Qwen3.6-27B-AWQ (QuantTrio)
**SETTLED (T2.4d + R17 scoring, 2026-04-25).** Supersedes Qwen3.5-27B.
- **Performance:** 77.4 t/s seq=1 (T2.4d)
- **Quality:** 4.875/5 on 8-task thinker suite. th02 correct 3/3 (reproducible). th08=4 (forward-ref bug in eager-init example); all others 5.
- **Production config:**
  ```
  Model: QuantTrio/Qwen3.6-27B-AWQ
  TP=1, GPU1
  --gpu-memory-utilization 0.90
  --kv-cache-dtype fp8
  --enable-chunked-prefill
  --max-num-seqs 1
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
  requires transformers>=5.5.4
  ```
- **Why TP=1 only:** GDN (Gated DeltaNet) hybrid architecture — TP=2 breaks recurrent state sync across GPU shards (H-TP2 confirmed T2.4g). Do not attempt TP=2 until T_KV3 provides a non-GDN replacement.
- **Why this wins over Qwen3.5-27B:** AIME 2026 94.1% (+4.9pp vs Gemma4), GPQA 87.8%. Quality 4.875 vs 4.0. No th03 reasoning budget exhaustion (which was a Qwen3.5-27B defect).

### Convergence: Qwen3.5-397B-A17B UD-IQ2_M (unsloth)
**SETTLED (R12, 2026-04-20).** See docs/arch/convergence.md for full guide.
- Model path: `/srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/.../UD-IQ2_M/`
- 4 split GGUF files, ~123GB total. Reference `00001-of-00004.gguf`.

---

## Suspended / retired roles

### Core (80B): Qwen3-Next-80B-A3B-AWQ (cyankiwi) — SUSPENDED
SUSPENDED (R19, 2026-04-25). Extended Arclight fills the role with zero additional memory overhead. Do not redeploy unless Extended Arclight proves insufficient after T_KV1/T_KV3.
- Last known performance: 189.5 t/s seq=1, 610 t/s at seq=4, 13007 t/s prefill@32k. Tool calls 100% with `--tool-call-parser hermes` (no `--reasoning-parser`).
- Requires: `--gpu-memory-utilization 0.95`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.

### Qwen3.5-27B-AWQ — SUPERSEDED as thinker (retain as fallback)
SUPERSEDED by Qwen3.6-27B. Was 4.0/5 quality, 76.5 t/s. Known defect: th03 always exhausts thinking budget → empty output. Hybrid Mamba/SSM architecture blocks kvcached Phase B. Do not redeploy as primary unless Qwen3.6-27B regresses.

---

## Eliminated candidates

### Gemma4-31B-it-AWQ — FULLY REJECTED from Arclight (thinker AND coder)
SETTLED (T2.3b thinker, T2.3c coder skipped by benchmark evidence, 2026-04-24/25).
- **Thinker REJECTED:** Mean 4.0/5 — matches Qwen3.5-27B baseline but fails depth-of-reasoning bar on th02/th03/th05. "Surface-Level Reasoning": prioritized heuristic secondary goals over primary constraints (th02), recommended naive Nginx for stateful LLM workloads (th03), false "pollution" concern (th05).
- **Coder SKIPPED:** Benchmark evidence (Qwen3.6 card comparison) shows Qwen3.6-35B-A3B clearly superior: SWE-bench +21.4pp, MCPMark +18.9pp, WideSearch +24.9pp. Gemma4 ties only on pure coding benchmarks; loses all agentic/terminal/MCP categories.
- **vLLM image:** Must use `vllm/vllm-openai:gemma4` tag (not `:latest`) — issue #39468 corrupt JSON in 0.19.0.
- **Parser rule:** `--tool-call-parser gemma4 --trust-remote-code` ONLY. Never add `--reasoning-parser gemma4` — causes all `no_call` when model skips thinking.
- **Architecture note:** Dense (no Mamba), ~20 GiB AWQ — was kvcached-compatible, but that's irrelevant since it's eliminated.

### GLM-4.7-Flash (30B-A3B MoE) — COLD STORAGE
MLA confirmed active (TRITON_MLA backend). EngineCore crashes at Tasks 02/03 (complex tool schemas). Root cause: TRITON_MLA PIECEWISE CUDA graph instability on Blackwell at longer decode lengths. Crash is in EngineCore subprocess — cannot patch from the outside. Monitor vLLM releases for `Glm4MoeLiteForCausalLM` + TRITON_MLA + sm_120 stability fix.

### GLM-4.5-Air — does not exist as planned
Z.ai skipped text-only Air at 4.5; released GLM-4.7 flagship + GLM-4.7-Flash. T2.3 (original GLM-4.5-Air test) is therefore void.

### DeepSeek-R1-32B-AWQ — ELIMINATED
Quality 2.6/5, dominated by Qwen3.5-27B 4.0/5 across all task categories. No path to viability.

### Devstral — ELIMINATED
bf16 OOMs at 30.4 GiB. Quality below Qwen3-Coder-30B-AWQ.

### lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled — KILLED without testing
Four hard blockers: no AWQ (GGUF only, BF16=70GB exceeds VRAM for calibration), tool calling unverified (attention-only LoRA), wrong model family for thinker role, Anthropic ToS risk (Claude API distillation data).

### GLM-4.6, GLM-4.7 full — out of reach
357/358B models need ~8× datacenter-class GPUs. Not feasible.

### Qwen3-Coder-30B-A3B-AWQ — SUPERSEDED as coder
SUPERSEDED by Qwen3.6-35B-A3B (better quality, similar TPS). Baseline performance retained as reference: 251 t/s seq=1, ~730 t/s at c=4. Parser `--tool-call-parser qwen3_coder --reasoning-parser qwen3`.

---

## Parser reference (correct stacks, do not deviate without testing)

| Model | Tool-call parser | Reasoning parser | Notes |
|-------|-----------------|-----------------|-------|
| Qwen3.6-35B-A3B | qwen3_coder | qwen3 | Both required + enable-auto-tool-choice |
| Qwen3.6-27B | qwen3_coder | qwen3 | Both required |
| Qwen3-Next-80B | hermes | **NONE** | Reasoning parser causes all no_call |
| Gemma4-31B | gemma4 + trust-remote-code | **NONE** | Reasoning parser causes all no_call |
| GLM-4.7-Flash | glm47 | glm45 | COLD STORAGE — EngineCore crash at complex tools |

---

## NVFP4 (Blackwell-native FP4) — deferred

**T_NVFP4 DEFERRED indefinitely.** T2.4c used `sakamakismile/Qwen3.6-27B-NVFP4` (untrusted publisher). Full 8-task run showed mean ~3.94/5 and th02 semantic error. Cannot distinguish format benefit from publisher calibration quality. Additionally, if ever reconsidered: TP=1 only (TP=2 broken for GDN). Use `nvidia/` or `bartowski/` publishers only; never `sakamakismile`.

AWQ-Marlin is already excellent for A3B MoE on RTX 5090. NVFP4 kernels in vLLM/CUTLASS are still maturing. Re-evaluate if: dense model added where VRAM is tight; TRT-LLM FP4 path accessible; vLLM major version bump with native Blackwell FP4 support.
