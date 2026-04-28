# Done Items — Full Procedures

Full procedures for DONE/CANCELLED/SKIPPED/PUNTED items. This is a grep/ripgrep target. For open items, see `docs/queue/open.md`. For current status of all items, see `docs/queue/status.md`.

---

## T1.1 — sleep_mode_operational_under_podman — DONE ✓

**Previous attempt:** FAIL on 2026-04-17. Root cause (R5): `--enable-sleep-mode` was missing from `vllm serve`. `VLLM_SERVER_DEV_MODE=1` exposes HTTP routes but `--enable-sleep-mode` is what makes the engine use `CuMemAllocator`. Without it, `/sleep` is a no-op.

**Result (2026-04-17):** PASS. With `--enable-sleep-mode` added: freed 92.8% VRAM (59 → 4 GiB) in ~4s, wake in 0.9s, post-wake TPS 212.3 t/s (ratio 1.000). CPU RAM increased by ~18 GiB during sleep.

---

## T1.1a — sleep_mode_regression_workaround — CANCELLED

Triggered by: T1.1 shows 20–60% freed (regression #32714 signature). T1.1 passed — fallback not needed.

---

## T1.1b — sleep_mode_blackwell_crash_workaround — CANCELLED

Triggered by: T1.1 container crashes at startup with `--enable-sleep-mode` added. T1.1 passed — Blackwell crash workaround not needed.

---

## T1.1c — podman_stop_start_fallback — CANCELLED

Triggered by: T1.1a and T1.1b both fail. T1.1 passed — podman stop/start fallback not needed.

---

## T1.2 — concurrent_two_vllm_processes_shared_gpus — DONE (FAIL ✗)

**Result (2026-04-17):** FAIL. Two concurrent TP=2 processes sharing GPUs collapsed to 4.2 t/s each (~2% of isolate throughput). Root cause: GPU-wide CUDA context time-slicing without MPS. Superseded by T1.2a. See R6 cycle log.

---

## T1.2a — tp1_per_gpu_concurrent_decode — DONE (PASS ✓)

**Question:** If we isolate Coder to GPU0 at TP=1 and Thinker to GPU1 at TP=1, do both models hit acceptable concurrency and throughput?

**Result (2026-04-18):** PASS. Perfect 1.000 concurrent isolation.
- Coder (30B A3B) isolated: 251.0 t/s. Concurrent: 251.0 t/s.
- Thinker (27B dense) isolated: 76.5 t/s. Concurrent: 76.5 t/s.
- Prefill contention: ~0x (cache hit dominance).

---

## T1.2b — sleep_mode_sequential_swap — CANCELLED

Triggered by: T1.2a fails. T1.2a passed — skipped.

---

## T1.2c — mps_eval — CANCELLED

Triggered by: T1.2a and T1.2b both fail. T1.2a passed — skipped.

---

## T1.3 — qwen3_coder_next_awq_tp2_viability — DONE (PASS ✓)

**Question:** Does `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` load and decode acceptably on TP=2?

**Result (2026-04-18):** PASS. 189.5 t/s seq=1 decode, 610 t/s aggregate at seq=4, 13007 t/s prefill at 32k context, 100% tool-call pass rate (9/9) with `--tool-call-parser hermes`. Three-tier architecture confirmed.

**Parser trial history (failures before working config):**
- `--tool-call-parser qwen3_coder` → 9/9 `no_call`
- `--tool-call-parser qwen3_xml` → 9/9 `wrong_tool`
- `--tool-call-parser qwen3_xml --reasoning-parser qwen3` → 9/9 `exception`
- `--tool-call-parser hermes --reasoning-parser qwen3` → 9/9 `no_call` (reasoning parser intercepts tool call tokens)
- `--tool-call-parser hermes` (no reasoning parser) → **9/9 PASS**

Required: `--gpu-memory-utilization 0.95` (0.85 OOM'd during CUDA graph capture) and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.

**Status:** SUSPENDED (R19, 2026-04-25). Extended Arclight fills the role with zero additional memory overhead. Do not redeploy unless Extended Arclight proves insufficient.

---

## T1.4 — th03_token_budget_fix — DONE (FAIL ✗)

**Question:** Does `--max-tokens 16384` fix Qwen3.5-27B th03 token budget exhaustion?

**Result (2026-04-18):** FAIL. Model exhausts search budget even at 16k tokens. Not a budget limit issue, but a reasoning pathology/loop specific to Qwen3.5-27B. Architecture-heavy tasks must be routed to Coder or Behemoth.

---

## T1.5 — kvcached_shared_gpu_kv_pool — PARTIAL ⚠

**Result (2026-04-19):**
- Phase A coder: PASS — 250.8 t/s (baseline 251.0). kvcached adds zero overhead for Transformer/MoE models.
- Phase A thinker: BLOCKED — Qwen3.5-27B-AWQ uses hybrid Mamba (SSM) layers. kvcached v0.1.5 raises `ValueError: got MambaSpec`. Thinker-specific.
- Phase B: FAIL — OOM during weight loading. kvcached virtualizes KV cache pages only, not weights. Combined 18+14 GiB saturated GPU.
- Phase B **Deferred** — kvcached T1.5 Phase B remains blocked (GDN/DeltaNet unsupported in v0.1.5).

---

## T2.1 — glm47_flash_mla_verification — DONE (INCONCLUSIVE) ⚠

**Question:** In our current vLLM, does GLM-4.7-Flash use MLA?

**Result (2026-04-18):**
- MLA confirmed active: all runs show `Using TRITON_MLA attention backend`. Actual MLA footprint ~129 KB/token.
- V1 engine cannot be disabled: vLLM nightly forces V1 for this architecture. All six known env vars ignored.
- Tool-calling broken under V1: EngineDeadError on Tasks 02 and 03 (complex schemas). Task 01 (simple) passes at ~44.7 t/s. Tool sanity locked at 33%.
- Decision: GLM-4.7-Flash in cold storage until vLLM V1 stabilizes for TRITON_MLA on sm_120.

---

## T2.1b — glm47_flash_streaming_parser_patch — CANCELLED

**Cancelled after R9 research (2026-04-18).** Root cause is TRITON_MLA PIECEWISE CUDA graph instability in the EngineCore subprocess (not in the tool parser). Cannot fix EngineCore crashes by patching the parser. Monitor vLLM releases for `Glm4MoeLite`/TRITON_MLA on sm_120 fix.

---

## T2.2 — coder_shootout_glm47_vs_qwen3coder30b — BLOCKED (then superseded)

**Blocked** on T2.1 (GLM-4.7-Flash tool-calling broken) and GLM going to cold storage. Superseded by T2.5 (Qwen3.6-35B-A3B). Skip until vLLM fixes V1/TRITON_MLA on sm_120.

---

## T2.3 — thinker_shootout_glm45_air_vs_qwen35_27b — BLOCKED

**Blocked** — GLM-4.5-Air does not exist as planned. Z.ai skipped text-only Air at 4.5; released GLM-4.7 flagship + GLM-4.7-Flash. T2.3 is void.

---

## T2.3b — arclight_thinker_gemma4_31b_candidate — DONE (REJECTED)

**Question:** Is Gemma4-31B a better Arclight thinker than Qwen3.5-27B?

**Result (2026-04-24):** REJECTED as primary thinker. Mean quality 4.0/5 — matches Qwen3.5-27B baseline but fails depth-of-reasoning bar on th02/th03/th05 ("Surface-Level Reasoning"). Strong 5/8 task profile (th01, th04, th06, th07, th08 all scored 5), 100% task completion, dense architecture. Redirected to coder candidate T2.3c.

**Specific failures:** th02: prioritized heuristic secondary goals over primary constraints. th03: recommended naive Nginx for stateful LLM workloads. th05: false "pollution" concern.

**Corrected specs:**
- Weight size: ~20 GiB (not ~16 GiB as originally estimated).
- vLLM image: `vllm/vllm-openai:gemma4` NOT `:latest` — issue #39468 corrupt JSON in 0.19.0.
- Parser rule: `--tool-call-parser gemma4 --trust-remote-code` ONLY. Never add `--reasoning-parser gemma4` — causes all `no_call` when model skips thinking.

---

## T2.3c — arclight_coder_gemma4_31b_candidate — SKIPPED (benchmark evidence)

**Skipped without running (2026-04-25).** Benchmark evidence shows Qwen3.6-35B-A3B clearly superior:

| Benchmark | Gemma4-31B | Qwen3.6-35B-A3B | Gap |
|-----------|-----------|----------------|-----|
| SWE-bench Verified | 52.0 | **73.4** | +21.4 Qwen |
| Terminal-Bench 2.0 | 42.9 | **51.5** | +8.6 Qwen |
| MCPMark | 18.1 | **37.0** | +18.9 Qwen |
| WideSearch | 35.2 | **60.1** | +24.9 Qwen |
| LiveCodeBench v6 | 80.0 | 80.4 | tie |

Gemma4 fully retired from Arclight consideration.

---

## T2.4 — arclight_thinker_qwen36_27b_candidate — DONE ✓ (settled by T2.4d)

**Question:** Is Qwen3.6-27B-AWQ a better Arclight thinker than Qwen3.5-27B?

**Result:** PASS. AWQ TP=1: reproducible correct at 4.875/5. See T2.4d.

**Procedure:**
```bash
pyenv activate hf
HF_HOME=/srv/ai/models hf download QuantTrio/Qwen3.6-27B-AWQ

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 --ctx 32768 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-auto-tool-choice
# Run Phase 2.2 thinker quality suite (8 tasks, max_tokens=4096)
```

---

## T2.4b — arclight_thinker_qwopus36_27b_candidate — SKIPPED (dep condition met)

Skipped — T2.4d quality ≥ 4.0 AND th02/th05 pass. Dep condition met. Qwopus SFT not needed.

---

## T2.4c — arclight_thinker_qwen36_27b_nvfp4_swap — INCONCLUSIVE

**Question:** Does NVFP4 + bf16 KV + TP=2 resolve confident incorrectness?

**Runs completed:**
- 230351Z (partial, th02+th03 only): both 5.0 — declared PASS. **Premature — only 2/8 tasks.**
- 232801Z (full 8-task run): th02 semantic error (missed jobs assigned `-1` = silent wrong output). Mean ~3.94/5 — below 4.0 baseline.

**Result:** INCONCLUSIVE. NVFP4 + bf16 KV did NOT resolve confident incorrectness. AWQ run 4 scored ~4.25/5, better than the NVFP4 full run. **Publisher note:** `sakamakismile/Qwen3.6-27B-NVFP4` is an untrusted publisher. If NVFP4 warrants re-testing, use `nvidia/` or `bartowski/` publishers only. Deferred to T_NVFP4.

---

## T2.4d — qwen36_27b_awq_reproducibility — DONE ✓

**Question:** Is AWQ run 4's correct th02 implementation stable (3×)?

**Result (2026-04-25, T094048Z and T102313Z):** PASS. th02 correct 3/3 runs. Quality mean 4.875/5 (th01=5, th02=5, th03=5, th04=5, th05=5, th06=5, th07=5, th08=4). th08 got 4 due to forward-reference bug in eager-init example. H3 (capability ceiling) falsified.

**Config:** QuantTrio/Qwen3.6-27B-AWQ, TP=1 (GPU1), fp8 KV, ctx=32768, max_tokens=16384, max-num-seqs 1, chunked-prefill-ON.

---

## T2.4e — qwen36_27b_awq_bf16kv_tp2 — DONE (INCONCLUSIVE — confounded)

**Question:** Does removing fp8 KV cache noise (bf16 KV + TP=2) improve th02 correctness?

**Result (2026-04-25, T114506Z):** th02 INCORRECT — missed jobs silently dropped, not assigned to busiest GPU. 104.8 t/s decode, 109ms TTFT p50. **Critical confound:** T2.4e changed two variables from T2.4d simultaneously (TP=1→2 AND chunked-prefill on→off). Cannot attribute regression to either cause alone. See T2.4g for resolution.

---

## T2.4f — qwen36_27b_rope_chunkedprefill_audit — DONE ✓

**Question:** Is RoPE theta being set correctly? Does chunked-prefill setting affect correctness?

**Result:** 
- `rope_theta: 10,000,000` correctly parsed by vLLM (matches model config.json). H1 (RoPE theta mismatch) dead.
- `--no-enable-chunked-prefill` at TP=1 causes immediate Triton OOM during kernel warmup. Chunked prefill must remain enabled at TP=1.

---

## T2.4g — qwen36_27b_awq_tp2_chunkedprefill_on — DONE ✓ (H-TP2 CONFIRMED)

**Question:** Does TP=2 + bf16 KV + chunked-prefill ON produce correct th02? (The single missing cell in the 2×2 factorial.)

**Result (2026-04-25):** th02 SEMANTIC ERROR × 0/3. Runs 2 and 3 identical to run1 (temperature=0, 49709 completion tokens all three runs). Quality mean 4.69/5 — strong on all tasks except th02. **H-TP2 CONFIRMED.** TP=2 itself breaks GDN (Gated DeltaNet) recurrent state sync across GPU shards, regardless of chunked-prefill setting.

**Config:**
```bash
./infra/scripts/deploy.sh vllm tp2b QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.85 --ctx 32768 \
  --kv-cache-dtype auto \
  --max-num-seqs 1 --enable-chunked-prefill \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice
```

---

## T2.4h — thinker_tp2_correctness_enforce_eager — DONE ✓

**Question:** Does `--enforce-eager` resolve the GDN state-splitting errors at TP=2?

**Results (2026-04-26):** SUCCESS (Semantic) / FAILED (Performance). Eager mode restores semantic correctness at TP=2 (th02 pass), but at the cost of 10x throughput collapse (~16 t/s). Not production viable.

---

## T2.5 — coder_shootout_qwen36_35b_a3b_vs_qwen3coder30b — DONE ✓

**Question:** Does Qwen3.6-35B-A3B-AWQ beat Qwen3-Coder-30B-A3B-AWQ as our coder?

**Result (2026-04-18):** PASS. 237.1 t/s seq=1 decode. 96.7% tool-call reliability (29/30 pass). 100% quality completion rate at `max_tokens=4096`. Requires `qwen3_coder` tool-parser + `qwen3` reasoning-parser + `enable-auto-tool-choice`. Verified BenchClient fix for `delta.reasoning` capture.

---

## T6.1 — infra_shell_and_container_tasks — DONE ✓

**Question:** What is the production coder baseline on infra-shaped tasks?

**Result:**
- Manual run (2026-04-25): 232 t/s TPS, 100% task completion.
- Rerun (2026-04-26): TP=2 at 20.25 t/s in eager mode (100% pass), TP=1 at 23.6 t/s suffered Reasoning Collapse. TP=2 is the only viable deployment. Automated baseline in `results/T6.1_infra_task_suite_20260426T165210Z/`.

---

## T_CV1 — convergence_startup_timing — DONE ✓

**Result (2026-04-26):** PASS. Startup time ~83s (cold) / ~88s (warm). Context ceiling 128k tokens. Generation 3.7 t/s (ngl=0 CPU-only). Decision: Always-Resident policy adopted.

**Procedure:**

*Part A — Startup timing (3 cold reps + 1 warm):*
1. Ensure Convergence is not running.
2. Drop page cache: `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`.
3. Start server with production flags (ngl=0, --no-mmap, -t $(nproc), -c 16384).
4. Poll `http://localhost:8002/health` every 500ms. Record timestamp on first 200 OK.
5. Repeat ×3 (cold). Then ×1 warm (no cache drop). Compute median.

*Part B — TPS baseline:*
6. `llama-bench` at ngl=0, -t $(nproc), -p 512, -n 128, -r 3.

*Part C — Context ceiling sweep:*
7. For each `-c` in [16384, 32768, 65536, 131072]: start server, send prompt at ~80% context length, record TTFT and TPS.

---

## T_CV2 — convergence_thread_count_sweep — DONE ✓

**Result (2026-04-26):** PASS. 32 threads optimal for the 397B model on pure CPU. PP scales linearly with threads (62 t/s vs 29 t/s at 16 threads). Script: `benchmarks/queue/T_CV2_convergence_thread_sweep.sh`.

---

## T_CV3 — convergence_gpu_expert_offload — DONE ✓

**Result (2026-04-26):** PASS. Achieved 13.99 t/s (Singularity/hybrid mode) via `-ngl 999 --cpu-moe` — attention layers on GPU, MoE experts in RAM. 3.75× speedup over ngl=0 CPU-only (3.7 t/s). Script: `benchmarks/queue/T_CV3_convergence_gpu_expert_offload.sh`.

---

## T_CV4 — convergence_parallel_tps — DONE ✓ (with caveat)

**Result (2026-04-26):** PASS. Aggregate throughput 15.6 t/s at concurrency=4 (1.12x scaling over 13.9 t/s single-seq). Decision: Production config uses `-np 4` as default parallel capacity.

**⚠ Caveat (added in R27/doc-tidy):** T_CV4 measured sequential pipelining into the server's 4 internal slots, NOT true concurrent client requests. T_PAR1 showed truly concurrent HTTP requests crash pr-1288 at N≥2 (GGML_ASSERT). The 15.6 t/s result is valid for sequential workloads. True concurrent capacity is N=1 until pr-1288 bug is fixed upstream.

Script: `benchmarks/queue/T_CV4_convergence_parallel_tps.sh`.

---

## T_KV1 — coder_big_context_mode — DONE ✓

**Question:** What is the maximum usable context for the coder when running TP=2 (thinker sleeping)?

**Status — PASS (2026-04-26):**
- **Max Usable Context**: 65,536 tokens (GPU-only, no swap spill).
- **TTFT (65K)**: 3,022 ms (~21K tokens/sec prefill).
- **TPS (65K)**: 238.2 t/s (minimal 2.6% regression from 32K baseline).
- **Swap Note**: `--swap-space 32` flag failed (unrecognized in vLLM 0.19.0). 131K test skipped.

**Procedure:**
1. Sleep thinker at level=1.
2. Restart coder with `--tensor-parallel-size 2 --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 --max-model-len 65536`.
3. Send prompts of increasing length (8K, 16K, 32K, 65K tokens). Record TTFT and TPS at each.
4. Test `--swap-space 32` with `--max-model-len 131072` — blocked in vLLM 0.19.0.

---

## T_KV2 — cuda_checkpoint_tp2_hot_restart — DONE ✓

**Question:** Does NVIDIA cuda-checkpoint + CRIU work for a TP=2 vLLM process?

**Result (2026-04-26):** PASS. Achieved 0.28s Hot Restart time vs 100.2s Cold Start (358× speedup).
- Technique: Host-native CRIU + cuda-checkpoint plugin + vLLM uvloop (io_uring) neutralization.
- VRAM Cleanup: Automated GPU reset (-i 1) used to clear ghost leaks between runs.
- Stability: Verified with Qwen3.6-35B-A3B (TP=2) and fp8 KV cache.
- Log: `results/T_KV2_host_hot_restart_20260426T023839Z`

**Prerequisites:**
```bash
sudo apt install criu
criu check
git clone https://github.com/NVIDIA/cuda-checkpoint /srv/ai/tools/cuda-checkpoint
cd /srv/ai/tools/cuda-checkpoint && make && sudo make install
mkdir -p /srv/ai/checkpoints/coder-tp2
```

**Script:**
```bash
bash benchmarks/queue/T_KV2_cuda_checkpoint_tp2_hot_restart.sh
# Options: --dry-run, --reps N, --ctx N, --gpu-mem F, --skip-cold
```

**Required patches (do NOT revert):**
- `vllm/entrypoints/openai/api_server.py` — replace `uvloop.run()` with `asyncio.run()`
- `vllm/v1/utils.py` — replace `uvloop.run()` with `asyncio.run()`

Without patch: `Error: Unknown shit 600 (anon_inode:[io_uring])`

## T4.1 — sglang_for_a3b_coder — PUNTED

SGLang weight-loader bug for Qwen3.5 MoE AWQ is permanent (KeyError in qwen3_5.py:1662, per-expert vs fused tensor format, no upstream fix). Sleep Mode and MTP are vLLM-exclusive for our use. Retain SGLang only as a reference comparison if vLLM becomes a bottleneck for a specific model. Otherwise PUNTED.

---

## T_CV5 — convergence_ngl_sweep — DONE ✓

**Question:** What is the optimal -ngl value for Convergence (Qwen3.5-397B) to reach its TPS ceiling with minimum VRAM?

**Result (2026-04-27):** MEASURED. Sweep of ngl [10, 20, 35, 50] showed an accelerating TPS curve, but saturation (90% of 13.99 t/s = 12.6 t/s) was **NOT** reached at ngl=50.

| -ngl | Median TPS | GPU 0 VRAM (MiB) | GPU 1 VRAM (MiB) |
|------|-----------|-----------------|-----------------|
| 0    | 3.70 (T_CV1) | — | — |
| 10   | 4.10 | 3842 | 1930 |
| 20   | 4.72 | 4344 | 2438 |
| 35   | 6.02 | 5142 | 3138 |
| 50   | 8.38 | 5840 | 3946 |
| 999  | 13.99 (T_CV3) | — | — |

**MoE Expert Offload (BENCH_06):** Removing `--cpu-moe` at `ngl=50` caused an immediate OOM/Deployment failure. The model weights (~123 GB) significantly exceed the 64 GB VRAM capacity. Partial expert offload is not natively supported by the engine.

**Analysis:** Performance gain per layer is non-linear and highest for the final layers. The remaining 14 layers (51–64) are responsible for ~40% of the total speedup. Offloading 50 layers consumes ~9.8 GB total VRAM.

**Artifacts:** `results/T_CV5_ngl_sweep_20260427T205900Z/` and `results/T_CV5_moe_offload_20260427T233729Z/`
### T_CRIU2 — Convergence CRIU mmap Success — 2026-04-28
**Status:** DONE ✓

**Procedure:** Host-native CRIU dump/restore of `llama-server` (mmap mode) with FD-sanitized Python launcher.
**Result:** 
- Checkpoint Size: **8.7 GB** (down from 135 GB).
- Restore Time: **7.3 s**.
- Page-Fault Penalty: **100.56 s** (Rep 1) -> 7.73 s (Rep 3).
**Finding:** Fast-swapping massive models is technically viable but bandwidth-limited by the first-inference warmup cost. Sanitized process environment is mandatory for CRIU reliability in agentic sessions.
**Artifacts:** `results/T_CRIU2_mmap_20260428T003639Z/`

---

## T_CRIU3 Phase 1 — thinker_hot_restart_feasibility — DONE ✓

**Question:** Does host-native CRIU + cuda-checkpoint work for the Thinker (TP=1, GPU1)?

**Result (2026-04-28):** PASS. Achieved 0.33s restore time (303× speedup vs cold start).
- **Method:** Host-native CRIU dump/restore + `cuda-checkpoint --toggle` + Python-sanitized launcher.
- **Metrics:** 401 MB checkpoint (CPU-only), TTFT parity (0.71s vs 0.72s), identical text output.
- **Implication:** Unblocks Sequential TP=2 orchestration. All Arclight/Core roles now have a sub-second swap path.
- **Log:** `results/T_CRIU3_thinker_hot_restart_20260428T082547Z/`
