# Gemini Flash Session Record — 2026-05-02

**Session type:** Testing (BENCH_15 — T_KV3 Path B)
**Operator:** Gemini Flash 2.0 (testing mode)
**Status:** COMPLETE

---

## What this session ran

**BENCH_15: T_KV3 Path B — 128K context on ik_llama.cpp with Qwen3.6-27B-Q5_K_M**

Prior to this session, ik_llama.cpp was on branch `pr-1288`. The `pr-1288` branch does not support the `qwen35` dense architecture (`LLM_ARCH_QWEN35`). BENCH_15 required migrating to `main` (which merged both DeltaNet MoE and the dense qwen35 support as of commit a8aecbf, 2026-04-30).

**Migration step performed:** ik_llama.cpp rebuilt from `main` branch before running the benchmark.

---

## Results

| Metric | Value |
|--------|-------|
| Engine | ik_llama.cpp main (commit a8aecbf) |
| Model | unsloth/Qwen3.6-27B-Q5_K_M |
| Config | --ctx-size 131072, --tensor-split 0.5,0.5, -np 1 |
| Context tested | 125,022 tokens |
| Prefill TPS | 1,892.9 t/s |
| Prefill time | 66.05s |
| Decode TPS | 49.4 t/s |
| GPU0 VRAM (loaded) | 14,290 MiB |
| GPU1 VRAM (loaded) | 13,978 MiB |
| Total VRAM | ~28,268 MiB |
| th02 quality | PASS (consistent hashing, non-empty `<think>` block) |
| Verdict | **PASS** |

Raw data: `results/T_KV3_pathb_128k_context_20260502T123824Z/`

---

## What was updated

- `docs/decisions/settled.md` — T_KV3 SETTLED; new entry: "ik_llama.cpp main required for Convergence & Qwen3.6"
- `docs/history/done-items.md` — T_KV3 entry added
- `docs/queue/status.md` — T_KV3 status updated to DONE ✓
- `docs/queue/open.md` — T_KV3 item removed
- `RESEARCH_STATE.md` — R31 cycle recorded
- `infra/scripts/build-ik-llama.sh` — updated to target `main` branch

---

## Hand-back to research

No open items from this session. T_KV3 is SETTLED. Convergence and Qwen3.6-27B thinker both confirmed on ik_llama.cpp main.

Next priorities per queue: T_MTP1 rerun on PrismaQuant thinker (BENCH_13 result stale — model was superseded), QX_PRELOAD for CRIU on Convergence.
