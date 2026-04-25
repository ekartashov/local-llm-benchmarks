# DECISIONS.md

Settled and provisional decisions, tiered by confidence. An item here is **not retested** — if you're about to run a benchmark, check this file first.

Legend:
- **SETTLED** — research + reasoning + (where applicable) measured evidence converge. Do not retest.
- **PROVISIONAL** — strong prior from research but contingent on engine/hardware/quant details that can shift. Re-evaluate if the condition listed changes.
- **SUPERSEDED** — was decided, is now invalidated (explains why).

Items removed entirely: the previous "Decisions already settled (do not re-evaluate)" list in the old `CLAUDE.md` is fully migrated here with updated rationale.

---

## SETTLED — tier naming (R12, 2026-04-20)

Three-tier architecture has permanent names. Use these in all docs, config, and scripts going forward.

| Old name | New name | Theme | Character |
|----------|----------|-------|-----------|
| Coder + Thinker (hot pair) | **Arclight** | Steins;Gate operation — fast, electric, concurrent | Always-on pair, energetic |
| Behemoth (80B asleep) | **Core** | Undertale core — slower but powerful, underground | Invoked on escalation |
| King-behemoth (397B in RAM) | **Convergence** | Deeper than the Core — ephemeral, anomalous, omnipotent | Pulled rarely, profound |

---

## SETTLED — model selection

### Qwen3.6-35B-A3B-AWQ is the Arclight coder role winner
**SETTLED (2026-04-18, T2.5 PASS).** 96.7% tool reliability + 100% quality completion rate + 237.1 t/s decode. Supersedes Qwen3-Coder-30B. Requires `--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice`.

### Thinking models require max_tokens ≥ 4096 in quality benchmarks
**SETTLED (2026-04-18).** At 1024 tokens, thinking models exhaust their budget while still inside `<think>`, producing no answer. Raising to 4096 is the floor for any quality-scored task. Default changed in `bench.py`.

### vLLM 0.19 reasoning-parser field is `delta.reasoning`
**SETTLED (2026-04-18).** vLLM v0.19.0+ emits thinking tokens under `delta.reasoning` (OpenAI o1-style), not `delta.reasoning_content`. BenchClient now captures both.

### Convergence tier uses ik_llama.cpp + GGUF, not vLLM
**SETTLED (R12, 2026-04-20).** Qwen3.5-397B-A17B at any quality-preserving quantization (~109–180GB) exceeds single-host VRAM (64GB), requires CPU+GPU hybrid inference, and needs GGUF format. vLLM's `--cpu-offload-gb` path for models this large involves constant PCIe weight transfers per forward pass and is impractical. ik_llama.cpp's `--cpu-moe` flag keeps MoE expert weights in RAM while hosting attention/norm/embedding layers on GPU — the correct architectural split for sparse MoE models. vLLM sleep mode does not apply to Convergence; it is a completely separate process.

### Convergence model: Qwen3.5-397B-A17B UD-IQ2_M (~123GB)
**SETTLED (R12, 2026-04-20).** Benjamin Marie's independent evaluation (MMLU-Pro, GPQA Diamond, LiveCodeBench v6, Math-500 subsets on H200s) found UD-IQ2_M on this specific model to be within benchmark margin of error of BF16. The 512-expert MoE architecture (10 active per token) tolerates 2-bit compression on expert weights far better than dense models — each expert handles a narrow specialization so per-expert compression loss is minimal. UD-IQ2_M (~123GB) is preferred over UD-IQ3_XXS (~140GB) because it enables `--no-mmap` (fully pinned, no NVMe page faults) with comfortable RAM headroom alongside vLLM level=1 sleep weights (~44GB). UD-IQ3_XXS at 140GB leaves only ~8GB free after sleep weights — dangerously tight for `--no-mmap`.

**Contrast:** MiniMax M2.5 quantizes catastrophically at IQ2-IQ4 (community-verified) — do not use MiniMax M2.5 as Convergence. Qwen3.5-397B is one of the best-quantizing models in the current landscape.

### Convergence performance baseline
**MEASURED (R12, 2026-04-20).** On ZRH01-AIRIG with vLLM sleeping (level=1):
- Token generation: **~13.15 t/s** (bottlenecked by DDR5 bandwidth reading MoE expert weights)
- Prompt eval (469 tokens): **60.66 t/s**
- Prompt eval (2348 tokens): **158.94 t/s** (larger batches amortize bandwidth over more tokens)
- GPU VRAM barely consumed — only attention/norm/embedding layers (~8-12GB total across both 5090s)
- Bottleneck is DDR5 bandwidth (~83 GB/s actual on 4-DIMM downclocked config) vs ~2.3GB RAM read per token for full expert sweep

**Thread count not yet optimized** — baseline at `$(nproc)` = 32. Smaller counts (16, 24) may be better due to MoE expert matrix sizes being too small to exploit 32 threads without cache thrashing. See T_CV2.

### ik_llama.cpp pr-1288 is required for Convergence
**SETTLED (R12, 2026-04-20).** Mainline ik_llama.cpp (version 4427, commit 07516cec) predates Qwen3.5 GDN (Gated Delta Network) support. PR #1288 adds `LLM_ARCH_QWEN35MOE`, `build_qwen35moe()`, and `llama-delta-net.cpp` with `ssm_alpha`. Must check out the `pr-1288` branch before building. Mainline llama.cpp (b8851) also supports this architecture as fallback but lacks `-fmoe` optimization.

**Reproduction:**
```bash
cd /srv/ai/projects/ik_llama.cpp
git fetch origin pull/1288/head:pr-1288
git checkout pr-1288
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

**Verify architecture support:**
```bash
find src/ -name "*.cpp" -o -name "*.h" | xargs grep -l "qwen35\|ssm_alpha\|delta_net" 2>/dev/null
# Expected: src/llama-delta-net.cpp, src/llama-arch.cpp, src/llama-build-context.cpp etc.
```

### Convergence launch command (verified working)
**SETTLED (R12, 2026-04-20).** In pr-1288, flash attention and fused MoE are **on by default**. `-fa` requires a value (`-fa on`) but is unnecessary since on is default. `-fmoe` flag is gone; use `-no-fmoe` to disable. `--cpu-moe` is the clean flag that keeps all `ffn_gate/up/down_exps` tensors in CPU RAM.

```bash
/srv/ai/projects/ik_llama.cpp/build/bin/llama-server \
  -m /srv/ai/models/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf \
  -ngl 999 \
  --cpu-moe \
  --no-mmap \
  -b 4096 -ub 2048 \
  -t $(nproc) \
  -c 16384 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --jinja \
  --host 0.0.0.0 --port 8002
```

**Model path:** 4 split files must all be in the same directory. Reference only `00001-of-00004.gguf`; loader finds the rest automatically.

**RAM prerequisite:** sleep both Arclight vLLM processes at level=1 before starting Convergence to free VRAM (though not strictly required for RAM since model is CPU-side) and to free up GPU for the attention layers.

### vLLM Sleep Mode level=1 ~4 GiB residual is a design floor, not a tunable
When a vLLM instance sleeps at level=1, it retains ~4 GiB GPU VRAM. This is the caching allocator instance, captured CUDA graphs, JIT-compiled kernels, and process state — deliberately preserved to enable <1s wake. Cannot be shrunk without breaking the wake-time guarantee. Level=2 offloads more but is unusable (gibberish-on-wake, see separate entry).

### kvcached: zero overhead for coder; re-evaluate after thinker selection
**PROVISIONAL (2026-04-19, T1.5 partial — coder PASS, thinker blocked).** Phase A coder PASS: 250.8 t/s on Qwen3.6-35B-AWQ with kvcached, identical to raw vLLM. Zero overhead for Transformer/MoE models.

Phase B blocked by two issues specific to the current thinker (Qwen3.5-27B-AWQ):
1. **MambaSpec incompatibility:** Qwen3.5-27B has hybrid Mamba (SSM) layers. kvcached v0.1.5 supports `FullAttentionSpec`, `SlidingWindowSpec`, `MLAAttentionSpec` only — raises `ValueError: got MambaSpec` at KV init. This is a thinker-model constraint, not a kvcached bug.
2. **Weight footprint:** kvcached virtual memory applies to KV cache pages only, not weights. Combined weights must still fit in physical VRAM.

**Re-run T1.5 Phase B after thinker model selection.** If the new thinker is pure Transformer/MoE (no Mamba) and combined weights fit comfortably under ~28 GiB, Phase B is worth retrying. Gemma4-31B REJECTED (T2.3b). Qwen3.6-27B uses GDN (not Mamba) but GDN is also not supported by kvcached — T1.5 Phase B remains blocked until kvcached adds DeltaNetSpec support upstream.

### Qwen3.6-27B-AWQ is the Arclight thinker (replaces Qwen3.5-27B)
**SETTLED (2026-04-25, T2.4d + R17 quality scoring).** 3/3 th02 correct, quality **4.875/5** on 8-task suite (scores: th01–th07 all 5, th08=4 — minor forward-ref bug in eager-init example). Exceeds 4.0 bar decisively and beats Qwen3.5-27B (4.0/5). 77.4 t/s seq=1. No reasoning budget exhaustion defects (th03 was a blocker for Qwen3.5-27B).

**Production config:**
- Model: `QuantTrio/Qwen3.6-27B-AWQ`
- Placement: TP=1, GPU1
- `--kv-cache-dtype fp8 --enable-chunked-prefill --max-num-seqs 1`
- `--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice`

**Supersedes:** Qwen3.5-27B-AWQ (retain as fallback only).

### Qwen3.6-27B RoPE theta is 10,000,000
**SETTLED (2026-04-25, T2.4f).** Audit confirmed `rope_theta` matches config.json and vLLM reflects it in logs. No mismatch.

### Chunked-Prefill must stay enabled for Qwen3.6 (GDN) at TP=1
**SETTLED (2026-04-25, T2.4d/g).** Disabling chunked prefill via `--no-enable-chunked-prefill` on GDN architecture causes immediate Triton OOM during kernel warmup at TP=1. Must remain enabled at TP=1. At TP=2, cp setting is irrelevant — TP=2 is broken for GDN regardless (H-TP2 confirmed, T2.4g). Production config uses TP=1 + cp-ON only.

### Qwen3.6-27B-AWQ at TP=2 is definitively broken for GDN — H-TP2 CONFIRMED
**SETTLED (2026-04-25, T2.4g).** T2.4g ran TP=2 + bf16 KV + **chunked-prefill ON** (the missing cell in the 2×2 factorial) and produced th02 SEMANTIC ERROR × 0/3. Combined with T2.4d (TP=1 + cp-ON = CORRECT 3/3), this isolates TP=2 itself as the cause of the GDN correctness regression — independent of chunked-prefill setting.

**Full 2×2 factorial:**

| | cp-ON | cp-OFF |
|---|---|---|
| **TP=1** | ✓ CORRECT 3/3 · 4.875/5 · 77.4 t/s (T2.4d) | OOM — Triton crash (T2.4f) |
| **TP=2** | ✗ INCORRECT 0/3 · 4.69/5 · 98.4 t/s (T2.4g) | ✗ INCORRECT · 104.8 t/s (T2.4e) |

**Root cause:** TP=2 splits the DeltaNet state matrix across GPU shards. Per-shard recurrent state updates do not commute — accumulated error produces a qualitatively different (wrong) solution for state-dependent reasoning tasks.

**Consequence for T_NVFP4:** If NVFP4 is ever reconsidered, it must be tested at TP=1 only. TP=2 is off the table for any GDN model.

**Production config unchanged:** TP=1 + fp8 KV + cp-ON is the correct and only viable config for Qwen3.6-27B-AWQ as thinker.

### Qwen3.6-27B — NVFP4 configuration is REJECTED
**SETTLED (2026-04-25, T2.4c).** NVFP4 quantization on the GDN architecture introduces reasoning pathologies (confident logic errors) that do not exist in the AWQ weights. Do not use NVFP4 for thinker roles until kernels/quantizers improve.

**Run history:**
- T2.4 runs 1–3 (AWQ, fp8 KV, max_tokens ≤ 4096): truncation — token budget exhaustion, not model intelligence.
- T2.4 **run 4** (AWQ, fp8 KV, ctx=32768, max_tokens=16384): **CORRECT** th02 (EDF + priority + best-fit, all jobs assigned including misses). Mean ~4.25/5. This is the only clean passing run.
- T2.4 run 5 (AWQ, fp8 KV, ctx=49152, higher max_tokens): three implementation bugs in th02 (inverted heap semantics, sign error, IndexError). Run 4 correctness did not survive config change.
- T2.4c partial run 230351Z (NVFP4, bf16 KV, TP=2, th02+th03 only): both scored 5.0 — operator declared PASS. Premature: only 2/8 tasks tested.
- T2.4c **full run 232801Z** (NVFP4, bf16 KV, TP=2, all 8 tasks): th02 has semantic error — missed jobs assigned to no GPU (silently wrong; `-1` instead of assigning to busiest GPU). Mean ~3.94/5 — below 4.0 baseline. **NVFP4 + bf16 KV did NOT resolve confident incorrectness.**

**Root cause hypothesis status (closed 2026-04-25 after T2.4g):**
1. **H1 — RoPE theta mismatch: DEAD.** T2.4f confirmed `rope_theta=10,000,000` correct.
2. **H2 — Chunked prefill × GDN recurrence: DEAD.** T2.4g ran TP=2 + cp-ON and still produced SEMANTIC ERROR. cp setting is not the cause.
3. **H3 — Capability ceiling: FALSIFIED at TP=1, CONFIRMED at TP=2.** TP=1 is correct 3/3; TP=2 is incorrect 0/3 regardless of cp setting.
4. **H4 — NVFP4 publisher quality: CONFIRMED CONTRIBUTING.** Explained T2.4c failure; does not explain TP=2 failure (also AWQ).
5. **H-TP2 — TP=2 breaks GDN state sync: CONFIRMED.** The full 2×2 factorial (T2.4d/e/f/g) isolates TP=2 as the sole cause of the th02 regression.

**NVFP4 mass-pull:** If ever reconsidered, TP=1 only. TP=2 is off the table for GDN. See T_NVFP4 in TESTING_QUEUE.md.

**GDN / kvcached note:** DeltaNetSpec not in kvcached v0.1.5 supported list. T1.5 Phase B remains blocked regardless of config. Does not affect TP=1 or TP=2 isolated deployment.

**Deployment flags (confirmed working):** `--tool-call-parser qwen3_coder --reasoning-parser qwen3 --kv-cache-dtype fp8 --max-num-seqs 1`. Requires `transformers>=5.5.4`.

### lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled — do not test
**SETTLED (R14, 2026-04-24).** Killed without testing. Four hard blockers:
1. No AWQ quantization (GGUF only); BF16 = 70GB exceeds 64GB VRAM pool for in-VRAM AWQ calibration.
2. Tool calling unverified — attention-only LoRA preserves base weights; tool format depends on template/post-training which was not validated.
3. 7,800-sample fine-tune of Qwen3.6-35B-A3B (our coder base) — wrong model family for thinker role, and too thin to produce reliable quality uplift.
4. Anthropic ToS: training data generated via Claude API; model card acknowledges compliance risk. Do not use in production contexts without legal review.

---

## SETTLED — infrastructure / tooling

### No Ollama
Adds 10–30% overhead vs raw engine containers; known broken tool parser for Qwen3.5 family (Ollama issue #14493). Qwen3.5 GGUF also doesn't work in Ollama due to separate mmproj vision files. Nothing to measure.

### No KTransformers
CPU offload path depends on AMX instructions. i9-14900K (Raptor Lake) does not have AMX. Project is architecturally unsuited to our CPU.

### Engines are containerized (vLLM) or native binary (ik_llama.cpp)
vLLM/SGLang: rootless podman only. No host installs. ik_llama.cpp for Convergence: native binary on host, built from source at `/srv/ai/projects/ik_llama.cpp`. This is intentional — ik_llama.cpp doesn't have a maintained container image for the pr-1288 branch, and GGUF inference doesn't benefit from container isolation the way a long-running API server does.

### Two vLLM processes on shared GPUs collapse to ~2% throughput
Two parallel vLLM processes sharing the same GPU(s) without NVIDIA MPS experience catastrophic performance degradation. Root cause: GPU-wide CUDA context time-slicing. Do not retest without MPS.

### NVIDIA MPS is skipped
Requires a privileged root daemon on the host which breaks our rootless podman invariant. sm_120 (consumer Blackwell) support also unverified outside datacenter environments.

### Multi-model in a single vLLM process is not supported
vLLM does not support hosting more than one model weight set in one server process. Open upstream feature request with no ETA.

### Sleep Mode level=1 is our swap primitive
Level 1 offloads weights to CPU RAM, discards KV cache, preserves CUDA graphs / allocator / JIT kernels. Wake times 0.1–6s depending on model size.

**Both flags are mandatory:**
- `VLLM_SERVER_DEV_MODE=1` (env): exposes `/sleep`, `/wake_up`, `/is_sleeping` HTTP routes
- `--enable-sleep-mode` (vllm serve flag): makes the engine initialize with `CuMemAllocator` and reserve the "weights" memory pool

**Level 1 only.** Level=2 produces gibberish output on wake (bug #29341).

### System RAM is not VRAM for KV cache
Offloading KV cache to DDR5 costs 50–80% decode speed. We keep working set in GDDR7. System RAM is fine for **weight storage during sleep** (vLLM level=1) and for **Convergence MoE expert weights** (read sequentially during inference, bandwidth-limited not latency-limited).

---

## SETTLED — models

### Qwen3-Coder-30B-A3B-AWQ is the Arclight coder baseline (superseded by Qwen3.6)
Measured: 251 t/s single-request, ~730 t/s aggregate at concurrency=4, single-GPU. Parser `--tool-call-parser qwen3_coder --reasoning-parser qwen3` works.

### Qwen3.6-35B-A3B-AWQ is the Arclight coder of record
SETTLED in T2.5 (2026-04-18). 237.1 t/s single-GPU. 96.7% tool-call reliability. 100% quality completion rate. Supersedes Qwen3-Coder-30B. Apache 2.0.

### Qwen3.5-27B-AWQ — SUPERSEDED as Arclight thinker (retain as fallback)
Was viable at 4.0/5, 76 t/s. Superseded by Qwen3.6-27B (4.875/5, 77.4 t/s, SETTLED 2026-04-25 T2.4d). Defect: `th03_architecture_tradeoffs` always exceeds thinking budget → empty output. Mamba SSM hybrid blocks kvcached Phase B. Do not redeploy as primary thinker unless Qwen3.6-27B regresses.

### Gemma4-31B is NOT the primary Arclight thinker; worth revisiting as coder candidate
**SETTLED (T2.3b, 2026-04-24).** Dense model (no Mamba), ~20 GiB AWQ, pure Transformer / no MambaSpec blocker for kvcached. Strong benchmarks (GPQA 84.3%, AIME 2026 89.2%, Arena Elo 1452). Quality test T2.3b mean 4.0/5 — matches Qwen3.5-27B baseline but fails depth-of-reasoning bar on th02/th03/th05 (see "rejected as primary thinker" below). The 5/8 task profile (excellent on th01, th04, th06, th07, th08) makes it a plausible *coder* role candidate — dense, fast tool execution, 100% task completion, no SSM layers. Not yet tested in coder role.

### --reasoning-parser gemma4 must NOT be combined with --tool-call-parser gemma4
**SETTLED (R13, 2026-04-23).** vLLM 0.19.x streaming code path waits for `</think>` close tag before activating the tool call parser. If Gemma4 skips reasoning and emits tool calls directly, the parser never fires — raw tool tokens appear as text content (all `no_call`). This is the same root cause as Qwen3-Next-80B (80B behemoth requiring `--tool-call-parser hermes` alone with no reasoning-parser). Correct Gemma4 deploy flags: `--tool-call-parser gemma4 --trust-remote-code`. For quality-only tasks (no tool calls), the reasoning parser is also unnecessary since the model's answers are scored by content not by reasoning visibility.

### Gemma4-31B is fully rejected from Arclight (thinker AND coder)
**SETTLED (2026-04-24, T2.3b).** Mean quality 4.0/5. Excels at textbook knowledge (Python closures, DCL, Pydantic — all scored 5), but demonstrates "Surface-Level Reasoning" on production systems tasks:
1. **th02 (Algorithm Logic)**: Prioritized heuristic secondary goals over primary constraints (missed deadline count).
2. **th03 (Systems Architecture)**: Recommended naive Nginx load-balancing for stateful LLM workloads, ignoring token-blindness and Head-of-Line blocking.
3. **th05 (Optimization)**: Recommended discarding valid versioned cache results based on a false "pollution" concern.

Gemma4 is a capable model but Qwen dominates on every benchmark relevant to this workload (agentic, terminal-agent, MCP, repo-level). No remaining path in Arclight. Do not re-evaluate unless a new Gemma generation closes the agentic gap (SWE-bench Verified, Terminal-Bench, MCPMark). Coder decision: benchmark evidence, 2026-04-25 (T2.3c skipped).

### vllm/vllm-openai:gemma4 tag required for Gemma4 models
**SETTLED (R13, 2026-04-23).** vLLM 0.19.0/0.19.1 (`:latest` tag) has issue #39468: tool-call argument strings are wrapped with `<|"|>` chars, producing unparseable JSON. Fixed in the `vllm/vllm-openai:gemma4` Docker tag. Deploy Gemma4 models with `BENCH_IMAGE=vllm/vllm-openai:gemma4`.

### DeepSeek-R1-32B-AWQ is not our thinker
Quality 2.6/5 dominated by Qwen3.5-27B 4.0/5 across task categories. Do not pursue further.

### Devstral is eliminated
bf16 OOMs at 30.4 GiB, quality below Qwen3-Coder-30B-AWQ. No path to viability.

### Core model: Qwen3-Coder-Next-80B-A3B-AWQ (cyankiwi)
SETTLED T1.3 (2026-04-18). 189.5 t/s seq=1 decode, 610 t/s aggregate at seq=4, 13007 t/s prefill@32k. Tool calls 100% reliable with `--tool-call-parser hermes` and **no** `--reasoning-parser`. Requires `--gpu-memory-utilization 0.95` and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.

**Do NOT add `--reasoning-parser qwen3`** to Core — causes hard failure at all parser combinations (reasoning parser intercepts XML-tagged tool calls before tool-call parser sees them).

### Convergence model: Qwen3.5-397B-A17B UD-IQ2_M vs 122B comparison
**SETTLED (R12, 2026-04-20).** 397B@UD-IQ2_M is preferred over 122B@Q4 despite lower bits-per-weight. Both quantize cleanly on the Qwen3.5 architecture. The TAU2 gap (+14.7 points: 86.7 vs ~72) represents genuine behavioral differences in multi-step agentic orchestration and long-horizon reasoning — the exact capability Convergence is invoked for. Since UD-IQ2_M on 397B is within BF16 margin of error, this is effectively 397B@BF16 vs 122B@Q4, and the 397B wins on every reasoning metric that matters for architectural decisions.

### GLM-4.7-Flash in cold storage
MLA confirmed active (TRITON_MLA). EngineCore crashes on Tasks 02/03 (2-arg tools). Root cause: TRITON_MLA PIECEWISE CUDA graph instability on Blackwell at certain decode lengths. Requires upstream vLLM fix. Monitor vLLM releases.

### GLM-4.6-Air does not exist
Z.ai released GLM-4.6V (vision) but skipped text-only Air. Went to GLM-4.7 flagship + GLM-4.7-Flash.

### GLM-4.6 and GLM-4.7 full (357B / 358B) are out of reach
Need ~8× datacenter-class GPUs. Not feasible on 2×5090.

---

## SETTLED — hardware/infra truth

### Convergence DDR5 bandwidth is the generation bottleneck
**SETTLED (R12, 2026-04-20).** Per-token generation reads ~2.3GB of expert weights from DDR5 (10 active experts × 3 matrices × 60 layers × ~1.28MB per matrix at IQ2_M). Actual DDR5 bandwidth is ~83 GB/s (4-DIMM downclocked from theoretical 120 GB/s). Theoretical ceiling: 83/2.3 ≈ 36 t/s. Measured: ~13 t/s. Gap explained by NUMA effects, thread coordination overhead, and expert routing compute. Prompt processing scales much better with batch size because bandwidth is amortized: 158 t/s at 2348-token batch vs 60 t/s at 469-token batch.

### Behemoth (80B A3B MoE) on TP=2 is extremely viable as Core
Verified in T1.3 (2026-04-18). 189.5 t/s seq=1 decode, 610 t/s at seq=4, 13007 t/s prefill at 32k context. Requires `--gpu-memory-utilization 0.95` and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.

### Dense 70B via TP=2 on PCIe x8/x8 no-NVLink → slow
Community-measured: 20–35 t/s. NCCL sync-point overhead accumulates over many layers at large hidden dims.

### MoE with ≤20B active params via TP=2 → fine
A3B decode allreduce is ~60 MB/s at 150 t/s, ~0.2% of x8 PCIe bandwidth.

### Speculative decoding does not help MoE
MoE active-param savings already address the memory-bandwidth bottleneck that spec decode targets.

---

## PROVISIONAL — re-check if condition changes

### NVFP4 as a standalone optimization phase — deprioritized
Blackwell FP4/FP8 kernels in vLLM/CUTLASS are still maturing. AWQ-Marlin is already excellent for A3B MoE on 5090.

**Re-evaluate if:** dense model added where VRAM headroom becomes tight; TRT-LLM FP4 path accessible; vLLM major version bump.

### LiteLLM as a classifier / router — not needed
OpenCode v1.3+ supports native multi-endpoint subagent routing.

**Re-evaluate if:** frontend other than OpenCode needed.

### Alternative AWQ publishers — not pursued
Inter-publisher AWQ quality variance is within noise. QuantTrio and cyankiwi are known-good. Note: `cpatonn/` returned HTTP 401 for Qwen3-Next-80B — do not use `cpatonn/` for that model.

### Convergence thread count — not yet optimized
Baseline at 32 threads. MoE expert matrices are relatively small (~1.28MB each); 32 threads may over-parallelize them causing cache thrashing. Values of 16 or 24 may be better. See T_CV2.

**Re-evaluate after:** T_CV2 thread sweep.

### Convergence GPU layer distribution — not yet optimized
GPU VRAM usage reported as "small amount" during first run — attention/norm/embedding layers occupy only ~8-12GB of available 64GB. Partial expert layer offload (e.g. first 10-15 layers' MoE experts on GPU) could improve generation speed. Requires `-ot` regex approach and careful VRAM budgeting. See T_CV3.

**Re-evaluate after:** T_CV2 establishes thread baseline; T_CV3 tests partial offload.

### Gemma4-31B as primary Arclight thinker — rejected; coder role open
Dense, no Mamba, ~20GB AWQ, kvcached-compatible. Mean 4.0/5 — same as baseline — but fails depth-of-reasoning bar on th02/th03/th05 (naivety on production constraints). Rejected as thinker. Dense architecture and strong completion rate make it a plausible coder candidate — queued as T2.3c.

---

## SUPERSEDED — invalidated by later findings

### OLD: "One-line source patch for glm4_moe_lite"
Superseded — legacy workaround for older images. Current cu130-nightly resolves natively.

### OLD: "No dense 70B with tensor parallelism"
Still true for large hidden dims but doesn't generalize. TP=2 is fine for A3B MoE.

### OLD: bf16 fit test determines viability
Models were eliminated based on bf16 size. We serve AWQ-INT4 or GGUF. Any model where the quant fits is a candidate.

### OLD: "Qwen3-Coder-Next needs GGUF"
Superseded — `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` exists and fits TP=2.

### OLD: "Two concurrent TP=2 processes sharing GPUs"
Superseded — CUDA context time-slicing collapses to ~2% throughput (T1.2).

### OLD: LiteLLM as router
Superseded — OpenCode native routing.

### OLD: "Phase 2 winner: Qwen3.5-27B-AWQ" (standalone)
Still provisionally valid for TP=1 isolated thinker. Gemma4-31B evaluated as alternative (T2.3b) — rejected as thinker, redirected to coder candidate (T2.3c). Qwen3.5-27B remains Arclight thinker baseline.

### OLD: "ik_llama.cpp version 4427 does not support Qwen3.5"
Superseded — PR #1288 branch adds full Qwen3.5 MoE support including GDN layers (`ssm_alpha`, `ssm_beta`, `ssm_out`). Mainline ik_llama.cpp HEAD (commit 07516cec) does not have it yet, but pr-1288 does. Mainline llama.cpp b8851 also has it as fallback.

---

## How to edit this file

When a new decision is made in research mode, add it under the appropriate tier with:
1. One-line claim.
2. The evidence (measured / reasoned-from-math / cited).
3. A "re-evaluate if" clause for PROVISIONAL items, or what superseded it for SUPERSEDED items.

When testing discovers something that contradicts an item here, do NOT edit it directly from testing mode. Record the contradiction in `RESEARCH_STATE.md` under "Open from testing" and hand back to research.