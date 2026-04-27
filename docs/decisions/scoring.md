# Model & Engine Evaluation Scoring Framework

> **Purpose:** Define how model and engine choices are evaluated. Selection is not a single-variable maximization problem — TPS, quality, context, and TTFT have different weights per role. This document prevents ad-hoc "this one's faster" reasoning and keeps evaluations consistent across sessions.

---

## The core principle

A model/engine combination is evaluated against the **usability profile of its role**, not a universal metric. The same TPS regression that's acceptable for Convergence would be disqualifying for the coder.

---

## Per-role scoring weights

### Arclight Coder

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| TPS (decode) | HIGH | Agent loop velocity — many small tasks in rapid succession. Low TPS → agents block on each code generation step. |
| Tool-call reliability | CRITICAL | 100% parse success required. A parser failure drops a subagent silently. No model passes if this is below ~95%. |
| Quality (task suite) | MEDIUM | Floor: 90%+ on T6.x coding suite. Above the floor, TPS wins over incremental quality gain. |
| Context ceiling | MEDIUM | 32K sufficient for most coding tasks; 65K useful for large file operations. Extended Arclight covers escalation. |
| TTFT | LOW | Latency to first token matters less for coding (user waits for complete output, not first token). |

**Decision rule:** Maximize TPS above the quality floor. Never trade tool-call reliability for speed.

### Arclight Thinker

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Quality (task suite) | CRITICAL | Reasoning tasks have no "good enough" floor below ~4.5/5. Incorrect reasoning that sounds plausible is worse than "I don't know." |
| Context ceiling | HIGH | The thinker is invoked for tasks that exhaust the coder's context. 32K is a known bottleneck. 65K+ is the target (T_KV3). |
| TPS (decode) | MEDIUM | 40–80 t/s range is acceptable. Going from 77 → 100 t/s is not worth a quality regression. Serial depth-first reasoning doesn't benefit from high decode throughput the way parallel tasks do. |
| Tool-call reliability | HIGH | Same as coder: parse failures are silent failures. |
| TTFT | MEDIUM | Long reasoning chains have high TTFT regardless — the thinker is expected to be slow. TTFT matters more for interactive use. |

**Decision rule:** Maximize quality above a TPS floor of ~40 t/s. Context ceiling is a hard constraint, not a nice-to-have.

### Convergence (397B)

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Prefill throughput (TTFT) | HIGH | Convergence typically ingests large contexts (architecture docs, codebases, research summaries). 66 t/s prefill allows fast context loading. This matters more than decode speed. |
| Quality (task suite) | HIGH | Invoked for synthesis and orchestration tasks that Arclight can't handle. Quality floor is high — this is the escalation tier. |
| Context ceiling | HIGH | 128K available (T_CV1). Must stay near this. |
| Decode TPS | LOW | 14 t/s is acceptable for deep analysis. Convergence is not used for interactive chat. |
| Quant headroom | MEDIUM | Higher quant = better quality for 512-expert sparse MoE at this scale. Ram budget determines quant ceiling (see convergence.md). |

**Decision rule:** Maximize quality + prefill throughput. Decode TPS is acceptable at 10–20 t/s.

---

## Engine evaluation criteria

When comparing engines for the same model:

| Criterion | Priority | Notes |
|-----------|----------|-------|
| Architecture correctness | CRITICAL | Does the model produce correct outputs? (e.g., GDN recurrent state in vLLM TP=2 → incorrect) |
| Tool-call reliability | CRITICAL | Measure with the T6.x suite on the specific engine/parser combination |
| TPS | HIGH | Baseline: vLLM current numbers. Any alternative engine must be measured against this. |
| Architecture support breadth | MEDIUM | Can it run models vLLM rejects? (e.g., GLM-4.7-Flash TRITON_MLA, GDN tensor-split) |
| Deployment complexity | LOW | Operational overhead amortizes over time. Don't reject a better engine because setup is harder. |
| CRIU compatibility | MEDIUM | Required for fast model switching. Verified for vLLM (T_KV2). Pending for ik_llama.cpp (T_CRIU2). |

---

## What this framework prevents

- Selecting a model because it has a higher HuggingFace benchmark score with no local measurement
- Rejecting a model because it's "slower" without checking if the speed difference matters for its role
- Claiming a configuration is "better" without specifying better on which dimension for which role
- Accepting a 10% quality regression to gain 20% TPS for the thinker (wrong trade-off)
- Accepting a 30% TPS loss to gain 2% quality for the coder (also wrong trade-off)
