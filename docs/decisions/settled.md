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

### Arclight Coder: Qwen3.6-35B-A3B-AWQ — SUPERSEDED 2026-05-05
SETTLED (T2.5, 2026-04-18). 96.7% tool-call reliability, 100% quality completion, 237.1 t/s TP=2. Parser: `--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice`. Superseded by PrismaQuant at TP=1 (BENCH_23a proved AWQ TP=1 also fails tool calls at 2/5 even with V1 engine; TP=2 AWQ at 237 t/s remains the reference if PrismaQuant ever reverts).

**AWQ TP=1 V1 = FAIL (BENCH_23a, 2026-05-05):** AWQ (INT4 Marlin) at TP=1 with V1 engine yielded **2/5 tool calls** and truncated th02 (finish_reason=length at 32K). TPS was 59.5 t/s (N=1) / 496.1 t/s (N=4) — faster than PrismaQuant at N=4 (496 vs 459 t/s) but not production-viable due to tool-call failure. Conclusion: the TP=1 reliability fix is specific to PrismaQuant+V1, not just V1 engine alone.

### Arclight Thinker: Qwen3.6-27B PrismaQuant-5.5bit — CURRENT PRODUCTION
SETTLED (BENCH_12 2026-05-01, updated BENCH_19 2026-05-03, hard suite BENCH_20 2026-05-03). Quality parity confirmed on standard suite (7/8 tasks; th08 truncated in both) and on hard 10-task systems engineering suite (PQ 41/50, AWQ 42/50 — tie; PQ 43/50 if task 03 truncation artifact excluded). TPS: **91.9 t/s seq=1 / 314.8 t/s N=4** with MTP n=3 (BENCH_19; baseline was 51.3/198.9 without MTP — BENCH_12). Quality rationale: th02 EDF algorithm correct; DeltaNet not corrupted. Tool-call reliability: **5/5 PASS under MTP n=3** (BENCH_19). Config: `rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm`, TP=1 GPU1, V1 engine (vLLM 0.20.0 only), fp8 KV, `--trust-remote-code --enable-chunked-prefill --max-num-seqs 4 --tool-call-parser qwen3_coder --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. Full NVFP4 path (dense GEMM — unaffected by FlashInfer #38718 MoE grouped-GEMM regression) can be unlocked with a container rebuild to CUDA 13.0 + FlashInfer SM120 support. CUDA 13.0 already released (August 2025); container rebuild not yet scheduled.

**Context sizing for hard task re-runs:** PrismaQuant (and any thinking model) can consume 20–28K reasoning tokens on hard multi-step tasks (Raft, cache-coherence, distributed systems). The 32K default context leaves insufficient headroom. Always deploy with `--max-model-len 131072` and set `max_tokens≥32000` in requests. On this hybrid DeltaNet model, 131K max-model-len costs ~0 extra VRAM (KV pool barely grows with context — confirmed T3.1 Phase 1, 0 MiB delta at 50K).

**Co-load measured point (BENCH_21, 2026-05-05):** Three-way co-load was verified at thinker util=0.95 + coder util=0.80 + Convergence `-ngl 15`, with 701 MiB (GPU0) / 467 MiB (GPU1) headroom and Convergence warm TPS 4.05 t/s. Any higher-ngl / lower-util extrapolation remains OPEN until directly benchmarked.

### Arclight Coder: Qwen3.6-35B-A3B GPTQ-Int4 (groxaxo) — CURRENT PRODUCTION
SETTLED (BENCH_33, 2026-05-06). **SUCCESS.** This config resolves the N=4 aggregate throughput bottleneck of the APEX GGUF stack (217 → 503 t/s) while maintaining a TP=1 footprint. 
- **TPS:** **103.2 t/s (N=1) / 502.9 t/s (N=4)**.
- **Reliability:** **5/5 tool calls PASS**; th02 quality probe PASS.
- **Engine:** vLLM 0.20.1 (Host-Native), Marlin kernels (`gptq_marlin`) confirmed active on SM120.
- **Config:** `groxaxo/Qwen3.6-35B-A3B-GPTQ-Pro-FOEM-4bit-g128`, TP=1 GPU0, `--max-num-seqs 8`, `--gpu-memory-utilization 0.80`, `--quantization gptq`.
- **Note:** Requires manual `config.json` patch to include `quantization_config` block for correct loader engagement.

### Arclight Coder: Qwen3.6-35B-A3B PrismaQuant-4.75bit — SUPERSEDED 2026-05-06
SETTLED (BENCH_23 2026-05-05). Superseded by GPTQ-Int4 (BENCH_33) which provides identical reliability with significantly better aggregate throughput (+9% N=4, but more importantly, runs on the stable Marlin path rather than the experimental SM120 FlashInfer path).

### Arclight Coder: Qwen3.6-35B-A3B APEX GGUF (mudler) — SUPERSEDED 2026-05-06
SETTLED (BENCH_24, 2026-05-06). Superseded by GPTQ-Int4 (BENCH_33). While APEX was faster at N=1 (185 vs 103 t/s), it failed to scale at N=4 (217 vs 503 t/s). GPTQ-Int4 is the superior choice for high-throughput agentic workflows.

### Arclight Thinker: Qwen3.6-27B-AWQ — SUPERSEDED 2026-05-01
SETTLED (T2.4d, R17, 2026-04-25). 4.875/5 quality, 77.4 t/s. Config: TP=1 GPU1, fp8 KV, `--enable-chunked-prefill --max-num-seqs 4 --tool-call-parser qwen3_coder --reasoning-parser qwen3`. Requires `transformers>=5.5.4`. Reference numbers valid for comparisons.

**max-num-seqs upgraded 1→4 (T_PAR1, R30, 2026-04-30):** BENCH_02/03 confirmed max-num-seqs=4 is safe: 269.4 t/s aggregate at N=4 (3.5× vs seqs=1), VRAM delta 4 MiB. The seqs=1 constraint was conservative and empirically unnecessary. At seqs=4, TTFT for N=1 requests is unchanged (73 ms). Production deploy should use `--max-num-seqs 4`.

### Convergence: Qwen3.5-397B-A17B UD-IQ2_M
SETTLED (R12, updated BENCH_22 2026-05-05). ~123GB. Engine: ik_llama.cpp main. Supports **CRIU On-Demand Restore (12s)** when configured with `GGML_CUDA_NO_PINNED=1`.

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

## CRIU checkpoint/restore (T_KV2, T_CRIU3, 2026-04-26 – 2026-04-30)

### TP=1 (thinker): SETTLED PASS
Host-native CRIU + NVIDIA cuda-checkpoint (driver 570+) enables full CUDA process snapshots.
**0.43s hot restart, 501 MB checkpoint, TTFT parity.** KV cache is physically preserved
post-restore (SchedulerOutput prefix hit ratio identical before/after checkpoint).

Requirements:
1. **Host-native execution** (Podman CDI conflicts break CRIU)
2. **uvloop patch**: `api_server.py` + `v1/utils.py` use `asyncio.run()` not `uvloop.run()`
3. **`UV_USE_IO_URING=0`** exported before any CRIU operation
4. **`sudo nvidia-smi --gpu-reset -i 1`** after failed restore (ghost VRAM)
5. **Runtime patches** in `benchmarks/queue/python_hijack/sitecustomize.py`:
   - `CriuSafePoller` — ZMQ Poller re-registers sockets after PID change, caps poll() at 1000ms
   - `SpinCondition.wait()` → `sched_yield()` — avoids broken inproc:// sockets post-restore

See docs/procedures/criu-ops.md for full procedure.

### TP=2 (coder): SETTLED FAIL
CRIU restore of vLLM TP=2 on Blackwell (sm_120) is broken at the SHM broadcast layer.

- Dump/restore itself succeeds: 29s dump, 67 GB checkpoint, 24–26s restore (~4× vs cold start)
- KV cache IS physically preserved in VRAM (confirmed by SchedulerOutput prefix hit ratio)
- Post-restore inference FAILS: `TimeoutError: RPC call to execute_model timed out`
- Root cause: `ShmRingBuffer` (used by V1 EngineCore→Worker broadcast) relies on shared memory
  writes becoming visible across processes. After CRIU restore, workers cannot see EngineCore's
  `written_flag=1` writes — likely a CPU cache coherency failure at the cross-process SHM level.
- `VLLM_USE_V1=0` cannot help: Blackwell sm_120 forces V1 engine regardless of the env var.
- Patches (CriuSafePoller, SpinCondition→sched_yield) are necessary but not sufficient.
- Even if fixed: 26s restore time vs ~100s cold start is only 4× — not worth the complexity.
  vLLM preallocates all KV blocks at startup, so a "clean" checkpoint is still ~67 GB.

### TP=397B (Convergence): SETTLED PASS (BENCH_22, 2026-05-05)
**12s restore-to-interactive achieved.**
- **Configuration**: `GGML_CUDA_NO_PINNED=1` (keeps MoE experts file-backed).
- **Checkpoint**: 8.0 GB (reduced from 122 GB).
- **Optimization**: `QX_PRELOAD` pre-warms GGUF weights + CRIU images into page cache.
- **Result**: ~1s restore + ~11s first-inference TTFT. Validates on-demand restoration for king-behemoth models.

---

## Extended Arclight / context ceilings (T_KV1, 2026-04-26)

SETTLED. Coder TP=2 at fp8 KV: **65,536 tokens max context** without swap. 238.2 t/s (2.6% regression vs 32K). `--swap-space N` is unrecognized in vLLM 0.19.0 R-V0 engine path — 131K target blocked until vLLM version bump.

---

## Convergence operational parameters (T_CV1–T_CV4, 2026-04-26)

SETTLED. See docs/arch/convergence.md for numbers. Summary:
- **On-Demand viable (12s)** or Always-resident (83s cold start).
- Production config: `GGML_CUDA_NO_PINNED=1`, `-ngl 15 --cpu-moe -t 32 -np 1`.
- Context ceiling: 128k tokens.
- **-np 1 enforced for stability** (N≥2 concurrent HTTP requests are permanently limited to N=1 by architectural constraint PR #1288).

---

## vLLM engine flags (critical — do not forget)

### Both flags required for Sleep Mode
SETTLED (R5, 2026-04-17). `VLLM_SERVER_DEV_MODE=1` exposes HTTP routes; `--enable-sleep-mode` makes the engine use `CuMemAllocator`. Without the serve flag, `/sleep` is a no-op at the memory layer.

### Sleep Mode level=1 only
SETTLED. Level=2 produces gibberish on wake (bug #29341). Level=1 frees ~92% VRAM, retains ~4 GiB residual (CUDA graphs — intentional for <1s wake). This residual cannot be reduced at level=1.

### VLLM_USE_V1=0 mandatory on Blackwell
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

### ik_llama.cpp main required for Convergence & Qwen3.6
SETTLED (R12, updated 2026-05-02). Mainline ik_llama.cpp `main` branch now includes the DeltaNet support originally introduced in `pr-1288`, as well as `LLM_ARCH_QWEN35` support for dense models. The stale `pr-1288` branch is superseded.

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
