# Queue Status — All Items

One-line status for every item. Load docs/queue/open.md for full specs of OPEN/BLOCKED items.
Full procedures for DONE items: docs/history/done-items.md

| Item | Status | Key result / blocker |
|------|--------|----------------------|
| T1.1 | DONE ✓ | Sleep Mode works. 92.8% VRAM freed, 4s sleep, 0.9s wake. |
| T1.1a | CANCELLED | T1.1 passed — fallback not needed. |
| T1.1b | CANCELLED | T1.1 passed — Blackwell crash workaround not needed. |
| T1.1c | CANCELLED | T1.1 passed — podman stop/start fallback not needed. |
| T1.2 | DONE ✗ | FAIL. Two TP=2 processes → ~2% concurrent throughput (CUDA context time-slicing). |
| T1.2a | DONE ✓ | TP=1-per-GPU: coder 251 t/s, thinker 76.5 t/s, perfect 1.0× isolation. |
| T1.2b | CANCELLED | T1.2a passed. |
| T1.2c | CANCELLED | T1.2a passed. |
| T1.3 | DONE ✓ | Behemoth 189.5 t/s, 100% tool calls with hermes parser. Core SUSPENDED. |
| T1.4 | DONE ✗ | Token budget cap doesn't fix th03. Route arch tasks to coder/behemoth. |
| T1.5 Phase A | DONE ✓ | kvcached zero overhead for coder. |
| T1.5 Phase B | DEFERRED | GDN (DeltaNetSpec) not supported by kvcached v0.1.5. No upstream timeline. |
| T2.1 | DONE ⚠ | MLA active (TRITON_MLA). Tool calls broken (EngineCore crash on V1). Cold storage. |
| T2.1b | CANCELLED | Root cause is EngineCore crash, not parser. Patch won't help. |
| T2.2 | BLOCKED | Depends on T2.1 (GLM-4.7-Flash in cold storage). Skip until vLLM fixes V1/TRITON_MLA on sm_120. |
| T2.3 | BLOCKED | Depends on T2.2 (GLM-4.5-Air doesn't exist as planned). |
| T2.3b | DONE ✗ | Gemma4-31B rejected as thinker. Mean 4.0/5, fails th02/th03/th05 depth-of-reasoning. |
| T2.3c | SKIPPED | Benchmark evidence: Qwen3.6-35B clearly superior on all agentic metrics. |
| T2.4 | DONE ✓ | Settled by T2.4d. AWQ TP=1 correct, 4.875/5. |
| T2.4b | SKIPPED | Dep condition met (T2.4d ≥ 4.0, th02/th05 pass). |
| T2.4c | INCONCLUSIVE | NVFP4 TP=2: mean ~3.94/5, th02 semantic error. Does not fix confident incorrectness. |
| T2.4d | DONE ✓ | TP=1 + fp8 KV + cp-ON: th02 correct 3/3, 4.875/5. Production thinker config confirmed. |
| T2.4e | DONE ⚠ | TP=2 + bf16 KV + cp-OFF: th02 INCORRECT. Confounded by two variable changes. See T2.4g. |
| T2.4f | DONE ✓ | rope_theta 10M correct. cp-OFF at TP=1 → Triton OOM. |
| T2.4g | DONE ✓ | H-TP2 CONFIRMED. TP=2 + cp-ON: th02 INCORRECT 0/3. GDN TP=2 definitively broken. |
| T2.4h | DONE ✓ | --enforce-eager restores semantic correctness at TP=2. TPS ~16 t/s (10× penalty). Not production viable. |
| T2.5 | DONE ✓ | Qwen3.6-35B-A3B-AWQ: 96.7% tool, 100% quality, 237.1 t/s. New coder baseline. |
| T2.6 | OPEN | Behemoth archetype scouting (design item). No script needed. |
| T3.1 | OPEN | KV cache q8/q4 on settled thinker. Deps: T2.3 (blocked). |
| T3.2 | OPEN | MTP spec decode on GLM-4.7-Flash. Deps: T2.2 (blocked). |
| T3.3 | OPEN | Qwen3-Next MTP on behemoth. Deps: T1.3 ✓, T2.4. |
| T3.4 | OPEN | Prefix cache survival across sleep/wake. Deps: T1.1 ✓. Script ready. |
| T4.1 | PUNTED | SGLang for A3B coder. Revisit only if vLLM blocks a model. |
| T5.1 | OPEN | OpenCode endpoint binding with subagents. Deps: T2.2, T2.3 (blocked). |
| T5.2 | OPEN | Behemoth wake trigger integration. Deps: T2.4 ✓. |
| T5.3 | OPEN | MCP servers firecrawl + searxng. Deps: T5.1 (blocked). |
| T6.1 | DONE ✓ | Infra tasks baseline. TP=2 production: 232 t/s (manual), ~20 t/s in eager-test mode. |
| T6.2 | OPEN | Cross-arch tasks (Orange Pi / armbian). No script. |
| T6.3 | OPEN | Ops tasks (Ceph, OpenStack). No script. |
| T6.4 | OPEN | RAG-aware tasks. Deps: T5.3 (blocked). |
| T_CV1 | DONE ✓ | 83s cold start, 128K ctx ceiling, 3.7 t/s CPU-only. Always-resident policy adopted. |
| T_CV2 | DONE ✓ | 32 threads optimal. PP scales linearly (62 t/s vs 29 t/s at 16 threads). |
| T_CV3 | DONE ✓ | 13.99 t/s Singularity mode (-ngl 999 --cpu-moe). 3.75× speedup. |
| T_CV4 | DONE ✓ | 15.6 t/s sequential pipelining at -np 4 (1.12× scaling). Concurrent HTTP: crashes at N≥2 (T_PAR1). |
| T_KV1 | DONE ✓ | 65K context, 238.2 t/s, 3022ms TTFT. --swap-space blocked (flag unrecognized in 0.19.0). |
| T_KV2 | DONE ✓ | 0.28s hot restart (358× vs 100.2s cold). CRIU + cuda-checkpoint settled. |
| T_KV3 | BLOCKED (CRITICAL) | Sub-Q1 settled (GDN TP=2 broken). Sub-Q2: Path A = find non-GDN replacement; Path B = ik_llama.cpp tensor-split on existing 27B. |
| T_PAR1 | OPEN (partial) | Convergence: N≥2 crashes (REAL). Coder/Thinker: UNRELIABLE (fabricated) — rerun required. |
| T_NVFP4 | DEFERRED | Restricted to TP=1; untrusted publisher (sakamakismile) used in T2.4c. Defer indefinitely. |
| T_TRT_LLM | LOW | TensorRT-LLM peak TPS optimization. Post-settlement only — requires stable roles + hours-long compile per model. |
| T_CRIU2 | OPEN | Test CRIU + cuda-checkpoint on ik_llama.cpp / Convergence. Confirms engine-agnostic fast-swap. |
| T_CRIU3 | OPEN | Standardize CRIU checkpoint library for all vLLM processes. Enables sequential TP=2 architecture. |
| T_CV5 | DONE ✓ | NGL sweep complete. Expert offload (no --cpu-moe) OOMs as expected. |
| T_ENGINE_EVAL | OPEN | Re-evaluate GLM-4.7-Flash + other cold-storage models on ik_llama.cpp. vLLM sleep no longer required. |
| QX_PRELOAD | OPEN | Design NVMe checkpoint pre-loading mechanism for warm CRIU restores (7,400 MB/s, 18s cold → 0.28s warm). |
