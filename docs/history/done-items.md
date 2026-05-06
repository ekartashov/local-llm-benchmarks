# Done Items

Full procedures for all DONE/CANCELLED/SKIPPED queue items, newest first. For chronological cycle log, see `docs/history/cycles.md`.

---

## T_APEX1 — apex_coder_viability — DONE ✓ (BENCH_24, 2026-05-06)
- **Result:** **SUCCESS**. APEX I-Compact on ik_llama.cpp delivers **185.0 t/s** (N=1) and **217.1 t/s** (N=4).
- **Tool Calls:** **5/5 PASS**. Grammar-based template correctly parses Qwen3 reasoning blocks.
- **VRAM:** 18.5 GB (weights + fp8 KV). Freeing 9.4GB vs PrismaQuant baseline.
- **Promotion:** APEX GGUF + ik_llama.cpp (Host) is now the primary production coder.

## T_APEX2 — apex_coder_convergence_coload_matrix — DONE ✓ (BENCH_25, 2026-05-06)
- **Result:** **SUCCESS**. Convergence co-load performance reached **13.8 t/s** (98% of isolated speed).
- **Topology:** APEX Coder (GPU0) + PQ Thinker (GPU1) + Shared Convergence (auto-NGL).
- **VRAM:** GPU0: 24.3 GB, GPU1: 30.8 GB. Both cards nearly saturated.
- **Outcome:** Validated the "Golden Topology" for three-model co-resident operation.

---

## T_PQ2 — prismaquant_coder_audit — DONE ✓ (BENCH_23/23a/b/c, 2026-05-05)
- **Result:** **SUCCESS (Stability)**. PrismaQuant A3B MoE is logically stable at TP=1 using vLLM V1 engine.
- **TPS:** 56.5 t/s (N=1) / 459 t/s (N=4). SM120 FlashInfer bottleneck confirmed.
- **Decision:** PrismaQuant promoted to production Coder for precision/VRAM benefits. vLLM V1 engine mandatory.
