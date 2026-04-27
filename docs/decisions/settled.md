# Settled Decisions

**Before running any benchmark, check here first.** If the answer is already settled, record it in this file and skip the test.

Use `rg "SETTLED" docs/decisions/settled.md` to scan all settled items.
Use `rg "<keyword>" docs/decisions/` to find a specific topic.

---

## Architecture & tier naming (R12, 2026-04-20)

| Old name | New name | Status |
|----------|----------|--------|
| Coder + Thinker hot pair | **Arclight** | SETTLED |
| 80B Core (behemoth) | ~~Core~~ → **RETIRED** | SETTLED |
| 397B king-behemoth | **Convergence** | SETTLED |
| 397B ultra-behemoth | **Singularity** | SETTLED |
| Coder TP=2 escalation | **Extended Arclight** | SETTLED |

Core RETIRED (2026-04-25, R19): Extended Arclight fills the escalation role with zero additional memory overhead.

---

## Model selection (see also docs/decisions/models.md for full candidate history)

### Arclight Coder: Qwen3.6-35B-A3B-AWQ
SETTLED (T2.5, 2026-04-18). 96.7% tool-call reliability, 100% quality completion, 237.1 t/s. Parser: `--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice`. TP=2 mandatory on vLLM 0.19.0 (TP=1 → Reasoning Collapse in eager mode).

### Arclight Thinker: Qwen3.6-27B-AWQ
SETTLED (T2.4d, R17, 2026-04-25). 4.875/5 quality, 77.4 t/s. Config: TP=1 GPU1, fp8 KV, `--enable-chunked-prefill --max-num-seqs 1 --tool-call-parser qwen3_coder --reasoning-parser qwen3`. Requires `transformers>=5.5.4`.

### Convergence: Qwen3.5-397B-A17B UD-IQ2_M
SETTLED (R12, 2026-04-20). ~123GB, always-resident. Engine: ik_llama.cpp pr-1288.

### Core (80B): RETIRED
SETTLED (R19, 2026-04-25). Extended Arclight fills the role. 80B model suspended. Re-evaluate only if Extended Arclight proves insufficient after T_KV1/T_KV3.

---

## Parallelism & GPU sharing

### TP=1-per-GPU is the only viable concurrent model (no NVLink)
SETTLED (T1.2 + T1.2a, 2026-04-18). Two TP=2 processes sharing GPUs collapse to ~2 t/s each (~2%). Root cause: GPU-wide CUDA context time-slicing without MPS. TP=1-per-GPU eliminates allreduce entirely. Do not retry without NVLink.

### MPS skipped
SETTLED. Requires root daemon (breaks rootless podman). sm_120 consumer Blackwell support unverified outside datacenter environments.

### Thinker TP=2 broken for GDN — H-TP2 CONFIRMED
SETTLED (T2.4g, 2026-04-25). Full 2×2 factorial (T2.4d/e/f/g) isolates TP=2 itself as the cause of GDN correctness regression regardless of chunked-prefill setting. Root cause: TP=2 splits the DeltaNet state matrix across GPU shards; recurrent state updates do not commute.

| | cp-ON | cp-OFF |
|---|---|---|
| TP=1 | ✓ CORRECT 3/3, 4.875/5, 77.4 t/s | OOM (T2.4f) |
| TP=2 | ✗ INCORRECT 0/3, 4.69/5, 98.4 t/s | ✗ INCORRECT (T2.4e) |

### Chunked-prefill mandatory for GDN at TP=1
SETTLED (T2.4f, 2026-04-25). `--no-enable-chunked-prefill` on GDN architecture causes immediate Triton OOM at TP=1. Always keep enabled.

### MoE A3B allreduce overhead is negligible on PCIe x8/x8
SETTLED. A3B decode allreduce is ~60 MB/s at 150 t/s — ~0.2% of x8 PCIe bandwidth. Dense 70B is the problem: 20–35 t/s due to sync-point overhead on many layers.

---

## CRIU checkpoint/restore (T_KV2, 2026-04-26)

SETTLED. Host-native CRIU + NVIDIA cuda-checkpoint (driver 570+) enables full CUDA process snapshots. **0.28s hot restart vs 100.2s cold start (358× speedup).**

Requirements:
1. **Host-native execution** (Podman CDI conflicts break CRIU)
2. **uvloop patch**: `api_server.py` + `v1/utils.py` use `asyncio.run()` not `uvloop.run()`
3. **`UV_USE_IO_URING=0`** exported before any CRIU operation
4. **`sudo nvidia-smi --gpu-reset -i 1`** after failed restore (ghost VRAM)

See docs/procedures/criu-ops.md for full procedure.

---

## Extended Arclight / context ceilings (T_KV1, 2026-04-26)

SETTLED. Coder TP=2 at fp8 KV: **65,536 tokens max context** without swap. 238.2 t/s (2.6% regression vs 32K). `--swap-space N` is unrecognized in vLLM 0.19.0 R-V0 engine path — 131K target blocked until vLLM version bump.

---

## Convergence operational parameters (T_CV1–T_CV4, 2026-04-26)

SETTLED. See docs/arch/convergence.md for numbers. Summary:
- Always-resident (83s cold start → never on-demand)
- Production config: `-ngl 999 --cpu-moe -t 32 -np 4` at 13.99 t/s
- Context ceiling: 128k tokens
- **-np 4 is sequential pipelining, NOT true concurrency** — concurrent HTTP crashes at N≥2 (T_PAR1)

---

## vLLM engine flags (critical — do not forget)

### Both flags required for Sleep Mode
SETTLED (R5, 2026-04-17). `VLLM_SERVER_DEV_MODE=1` exposes HTTP routes; `--enable-sleep-mode` makes the engine use `CuMemAllocator`. Without the serve flag, `/sleep` is a no-op at the memory layer.

### Sleep Mode level=1 only
SETTLED. Level=2 produces gibberish on wake (bug #29341). Level=1 frees ~92% VRAM, retains ~4 GiB residual (CUDA graphs — intentional for <1s wake). This residual cannot be reduced at level=1.

### VLLM_V1_ENABLED=0 mandatory on Blackwell
SETTLED. V1 engine causes 10× TPS regression in eager mode and EngineDeadError on TRITON_MLA PIECEWISE path for some architectures. Disable globally.

### VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 required for TP=2
SETTLED (T1.3 + T_KV1). Without this, vLLM OOMs during CUDA graph capture at TP=2 with `--gpu-memory-utilization ≥ 0.85`.

### reasoning-parser must NOT be combined with tool-call parser for skip-think models
SETTLED (R13, T1.3). When a model skips `</think>` and emits tool calls directly, the streaming code path waits for `</think>` before activating the tool-call parser — it never fires, producing all `no_call`. Affects: Qwen3-Next-80B (use `--tool-call-parser hermes` only), Gemma4 (use `--tool-call-parser gemma4 --trust-remote-code` only). Do not add `--reasoning-parser` to these models.

### vLLM 0.19 reasoning-parser streaming field
SETTLED (R10, 2026-04-18). Field is `delta.reasoning` (OpenAI o1-style), not `delta.reasoning_content`. BenchClient captures both.

### max_tokens floor for thinking models
SETTLED (2026-04-18). At 1024 tokens, thinking models exhaust budget inside `<think>` producing no answer. Floor is 4096 tokens for any quality-scored task.

---

## KV cache (elastic / dynamic)

### kvcached: zero overhead for coder; blocked for GDN thinker
PROVISIONAL (T1.5, 2026-04-19). Phase A (coder alone): 250.8 t/s — identical to raw vLLM. Phase B (two instances sharing GPU): blocked. GDN (Qwen3.6-27B) uses DeltaNetSpec which is not in kvcached v0.1.5. Re-evaluate only when kvcached adds DeltaNetSpec (no upstream milestone).

### Dynamic KV rebalancing not possible for GDN
SETTLED (R19, 2026-04-25). GDN layers use a fixed-size recurrent state matrix (~32KB/head/layer), not paged KV blocks. kvcached virtualizes paged KV blocks only. Static `--max-model-len` asymmetry is the only available lever.

---

## Infrastructure

### No Ollama
10–30% overhead vs raw containers. Broken tool parser for Qwen3.5 family (Ollama #14493). Not viable.

### No KTransformers
Depends on AMX instructions. i9-14900K (Raptor Lake) has no AMX. Architecturally incompatible.

### Multi-model in single vLLM process not supported
Upstream: open feature request, no ETA. No workaround available.

### Engines are containerized (vLLM) or native binary (ik_llama.cpp)
vLLM: rootless podman only. ik_llama.cpp: native host binary at `/srv/ai/projects/ik_llama.cpp/build/bin/llama-server`.

### ik_llama.cpp pr-1288 required for Convergence
SETTLED (R12). Mainline ik_llama.cpp HEAD (commit 07516cec) predates Qwen3.5 GDN support. PR #1288 adds `LLM_ARCH_QWEN35MOE` and `llama-delta-net.cpp` with `ssm_alpha`. Must checkout pr-1288 branch.

### DDR5 bandwidth is the Convergence bottleneck
SETTLED (R12). Per-token read: ~2.3GB expert weights. Actual bandwidth: ~83 GB/s. Theoretical ceiling: ~36 t/s. Measured: ~13 t/s (36% efficiency). Gap: NUMA, thread coordination, routing compute.

---

## SUPERSEDED / do not re-evaluate

- Old "one-line source patch for glm4_moe_lite" — superseded by current images
- "Two concurrent TP=2 processes sharing GPUs" — SETTLED FAIL (T1.2)
- "Dense 70B via TP=2 on PCIe x8/x8" — 20–35 t/s, impractical
- "bf16 fit test determines viability" — we serve AWQ-INT4/GGUF; bf16 size is irrelevant
- "Qwen3-Coder-Next needs GGUF" — cyankiwi AWQ exists and fits TP=2
- LiteLLM as router — superseded by OpenCode native routing
- "Qwen3.5-27B-AWQ as thinker" — superseded by Qwen3.6-27B (4.875 vs 4.0, same TPS)
