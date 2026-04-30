# Handoff to Gemini Flash (Antigravity) — 2026-04-26
From: Claude (Research + scripting session)
To: Gemini Flash (Testing — T_KV1 context sweep, T_PAR1 parallelism sweep)

---

## What happened this session

Two things:

1. **Documentation audit** of the prior Gemini sessions' artifacts. Corrections made:
   - `RESEARCH_STATE.md`: duplicate R21 renamed to R20 (T_KV2 cycle)
   - `TESTING_QUEUE.md`: T_CV1-4 structure fixed (detached header + duplicate blocks); status table updated to R25
   - `DECISIONS.md`: T_KV2 (CRIU) and Convergence operational params split into separate SETTLED sections
   - `ARCHITECTURE.md`: T_KV2 / T_CV1 context ceiling marked settled; parallelism table updated with measured numbers
   - `T6.1_infra_task_suite.sh`: comment added clarifying it tests eager mode (~20 t/s), not the production 232 t/s baseline

2. **New scripts authored** (ready to run):
   - `benchmarks/queue/T_KV1_coder_big_context_mode.sh`
   - `benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh`

Everything committed on `main`. No test runs were performed.

---

## Current architecture (settled — do not re-litigate)

| Tier | Model | Config | Port | Status |
|------|-------|--------|------|--------|
| Arclight Coder | Qwen3.6-35B-A3B-AWQ | vLLM TP=2, fp8 KV, ctx=32768, V1 disabled | 30000 | SETTLED |
| Arclight Thinker | Qwen3.6-27B-AWQ | vLLM TP=1 GPU1, fp8 KV, cp-ON, max-num-seqs 1 | 30001 | SETTLED |
| Extended Arclight | Coder as TP=2 (thinker sleeping) | ctx=65536, CRIU hot-restart=0.28s | 30000 | SETTLED (T_KV2) |
| Convergence | Qwen3.5-397B UD-IQ2_M | ik_llama.cpp -ngl 999 --cpu-moe -t 32 -np 4 | 8002 | SETTLED (T_CV1-4) |

**Critical operational requirements (do not forget):**
- vLLM containers must have `uvloop` patched out (`api_server.py` + `v1/utils.py` use `asyncio.run()`).
- Export `UV_USE_IO_URING=0` before any CRIU checkpoint/restore operation.
- `VLLM_USE_V1=0` must be set for all vLLM deployments.
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` required for coder TP=2 (prevents OOM during graph capture).
- For TP=2 coder: `--gpu-memory-utilization 0.90`.
- After a failed CRIU restore: `sudo nvidia-smi --gpu-reset -i 1` to clear ghost VRAM leaks.

**Thinker TP=2 is definitively broken for GDN (H-TP2 confirmed, T2.4g).** Do not attempt without a new model candidate from T_KV3.

---

## Prioritized run queue

### HIGH — run these next (both scripts ready, all deps satisfied)

**T_KV1 — `benchmarks/queue/T_KV1_coder_big_context_mode.sh`**

```bash
bash benchmarks/queue/T_KV1_coder_big_context_mode.sh
# Options: --skip-sleep (if thinker already sleeping), --skip-swap, --dry-run
```

**Prerequisites:**
- Thinker must be running on port 30001 (the script sleeps it via REST API)
- Coder must be running on port 30000 (script stops it and restarts as TP=2)
- No manual steps needed — script handles the sleep/restart/sweep/cleanup cycle

**What to watch for:**
- Does coder TP=2 with `--max-model-len 65536` start cleanly? (OOM = KV estimate was wrong)
- What does `nvidia-smi` show for VRAM split between the two GPUs after TP=2 deploy?
- TTFT at 65K context (should be < 60s for usable interactive sessions)
- TPS at 65K vs 32K baseline (pass criterion: no more than 20% regression)
- Does `--swap-space 32 --max-model-len 131072` actually start, or does it fail?

**Failure modes:**
- OOM at 65536 context → KV budget estimate was wrong; record actual VRAM numbers from nvidia-smi in TESTING_QUEUE.md and hand back to research
- Startup crashes with CUDA error → report full error; do not retry without research pass
- TPS regression > 20% at 65K → still mark PASS if context works, but note the regression

---

**T_PAR1 — `benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh`**

```bash
bash benchmarks/queue/T_PAR1_parallel_throughput_sweep.sh
# Options: --skip-coder, --skip-thinker, --skip-convergence, --reps 3, --dry-run
```

**Prerequisites:**
- Coder running on port 30000 (TP=2 production config, normal hot-pair mode)
- Thinker running on port 30001 (TP=1 GPU1 production config)
- Convergence running on port 8002 (always-resident)
- **Important:** the thinker is deployed with `--max-num-seqs 1`. For the N=2,4,8 slots the script will fire concurrent requests; with max-num-seqs=1 the server will queue them, so results will show serial latency × N, not true parallel TPS. To get a real parallel result for thinker, redeploy thinker with `--max-num-seqs 4` first, then run `--skip-coder --skip-convergence`. Run the thinker sweep twice: once at production max-num-seqs=1 (documents baseline behavior) and once at max-num-seqs=4 (reveals true parallel ceiling). Both results are useful.

**What to watch for:**
- Coder aggregate TPS vs N: is the knee at N=4 or N=2? (A3B MoE should batch efficiently)
- Does coder aggregate TPS match the ~730 t/s measured at seq=4 in T1.2a? (expected: yes)
- Thinker serial vs parallel: at N=1 should be ~77 t/s; at N=4 does aggregate TPS scale?
- Convergence: already measured at 15.6 t/s (T_CV4). This confirms the number on the same hardware/software state as the Arclight sweep.

---

### MEDIUM — after T_KV1 and T_PAR1

**T_KV3 is BLOCKED on research** — Sub-Q1 already answered by T2.4g (TP=2 broken for GDN with V1 disabled). Sub-Q2 needs a research session to identify a TP=2-capable thinker candidate. Do not attempt T_KV3 test runs until research provides a model slug + deploy config.

---

## Artifact sync rules

After each test run, update these files:

| File | What to update |
|------|----------------|
| `TESTING_QUEUE.md` | Change item status to DONE ✓ / FAIL ✗ / INCONCLUSIVE ⚠; add result block with key numbers |
| `RESEARCH_STATE.md` | Append to `## Open from testing` if anything unexpected; update "What we believe right now" only if a number changes materially |
| `config/models.yaml` | Update measured TPS, config flags for the model tested |
| `DECISIONS.md` | Add SETTLED entry if a question is now closed |

Do not edit `ARCHITECTURE.md` in testing mode — that's research-mode only. If a test result forces an architecture change, write a clear `## Open from testing` block and flag it: "HAND-BACK: architecture change required."

---

## What I need back from you

**After T_KV1:**
- The max usable context without swap (tokens)
- TPS at 32K vs 65K (regression %)
- Whether the swap-space 131K restart succeeded, and TTFT at 98K context
- The exact `nvidia-smi` VRAM numbers after TP=2 coder deployment (used to refine the KV budget estimate in ARCHITECTURE.md)
- If startup fails: full error from bench.log

**After T_PAR1:**
- Coder: aggregate TPS at N=1,2,4,8 → which N is the knee?
- Thinker: aggregate TPS at N=1 (baseline) and at N=4 (if re-deployed with max-num-seqs=4)
- Convergence: does the measured number match T_CV4's 15.6 t/s aggregate?

---

## Known sharp edges

- `T6.1_infra_task_suite.sh` uses `--enforce-eager` (intentional for quality comparison but gives ~20 t/s, not production TPS). Do not quote its TPS as the Arclight Coder baseline — the baseline is 232 t/s from the manual 2026-04-25 run.
- Coder TP=2 requires `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` or it OOMs during graph capture. The T_KV1 script sets this automatically.
- After T_KV1's context sweep, restore the production config: stop the TP=2 extended coder, restart thinker, restart coder TP=2 at ctx=32768. T_PAR1 expects the production hot-pair.
- `T_KV3` is listed HIGH priority in the status table but requires research first — do not start it without a model candidate.
