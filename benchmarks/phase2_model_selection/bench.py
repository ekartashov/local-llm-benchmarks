"""
Phase 2 — Model selection benchmark.

Three modes:
  quality          Run coding tasks and record raw outputs for human review.
                   Automated check: task_completion_rate + decode_tps.
                   Outputs a human_review.md alongside metrics.json.
                   Used for sub-tests 2.1 (coder comparison), 2.2 (thinker),
                   2.3 (peak mode).

  tool-reliability Re-run Phase 0 tasks using Phase 0 scorer.
                   Used for sub-test 2.4 (Devstral validation).

  spec-decode      Throughput test only — same as Phase 1 throughput mode.
                   Run twice (once without spec-decode, once with) and compare.
                   Used for sub-test 2.5.

Usage:
    # Quality mode
    python -m benchmarks.phase2_model_selection.bench \\
        --endpoint http://localhost:30000/v1 \\
        --results-dir results/phase2_2.1_qwen35_20260414 \\
        --mode quality \\
        --tasks benchmarks/phase2_model_selection/tasks/quality/ \\
        --label "Qwen3.5-35B-A3B-AWQ"

    # Tool-reliability mode (reuses Phase 0 tasks)
    python -m benchmarks.phase2_model_selection.bench \\
        --endpoint http://localhost:30000/v1 \\
        --results-dir results/phase2_2.4_devstral_20260414 \\
        --mode tool-reliability \\
        --tasks benchmarks/phase0_tool_reliability/tasks/

    # Spec-decode mode
    python -m benchmarks.phase2_model_selection.bench \\
        --endpoint http://localhost:30000/v1 \\
        --results-dir results/phase2_2.5_spec_20260414 \\
        --mode spec-decode \\
        --tasks benchmarks/phase1_engine_selection/tasks/throughput/
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
from rich.console import Console
from rich.table import Table

from lib.client import BenchClient
from lib.metrics import BenchResult, PhaseMetrics

console = Console()

PHASE = "phase2_model_selection"


# ── Task loaders ──────────────────────────────────────────────────────────────

def load_quality_tasks(tasks_dir: Path) -> list[dict[str, Any]]:
    files = sorted(tasks_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No task JSON files in {tasks_dir}")
    tasks = []
    for f in files:
        data = json.loads(f.read_text())
        for key in ("id", "system_prompt", "user_message"):
            if key not in data:
                raise ValueError(f"Task {f.name} missing required field '{key}'")
        tasks.append(data)
    return tasks


# ── Quality mode ──────────────────────────────────────────────────────────────

async def run_quality(
    client: BenchClient,
    tasks: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    label: str,
) -> tuple[PhaseMetrics, list[dict[str, Any]]]:
    """
    Run tasks and collect outputs for human review.
    Returns (PhaseMetrics, list of per-task dicts with prompt + response).
    """
    pm = PhaseMetrics()
    outputs: list[dict[str, Any]] = []

    for task in tasks:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["user_message"]},
        ]
        result = await client.chat(
            messages=messages,
            task_id=task["id"],
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        pm.add(result)

        completed = result.ok and bool(result.raw_text.strip())
        icon = "[green]✓[/green]" if completed else "[red]✗[/red]"
        console.print(
            f"  {icon} {task['id']:40s}  "
            f"ttft={result.ttft_ms:6.0f}ms  tps={result.decode_tps:5.1f}  "
            f"{'ok' if completed else 'EMPTY/ERR'}"
        )

        outputs.append({
            "id": task["id"],
            "description": task.get("description", ""),
            "label": label,
            "user_message": task["user_message"],
            "response": result.raw_text,
            "completed": completed,
            "ttft_ms": result.ttft_ms,
            "decode_tps": result.decode_tps,
            "error": result.error,
        })

    return pm, outputs


def _write_human_review(
    outputs: list[dict[str, Any]],
    results_dir: Path,
    label: str,
) -> None:
    """Write a human_review.md for side-by-side quality evaluation."""
    lines = [
        f"# Human Review — {label}",
        "",
        "**Instructions:** Review each response and mark quality 1–5.",
        "1 = wrong/useless  3 = acceptable  5 = excellent",
        "",
        "| Task | Completed | TTFT | TPS | Quality (fill in) |",
        "|------|-----------|------|-----|-------------------|",
    ]
    for o in outputs:
        lines.append(
            f"| {o['id']} | {'✓' if o['completed'] else '✗'} "
            f"| {o['ttft_ms']:.0f}ms | {o['decode_tps']:.0f} | — |"
        )
    lines += ["", "---", ""]

    for o in outputs:
        lines += [
            f"## {o['id']}",
            "",
            f"**Description:** {o['description']}",
            "",
            "**Prompt:**",
            "```",
            o["user_message"],
            "```",
            "",
            "**Response:**",
            "```",
            o["response"] if o["response"] else f"[ERROR: {o['error']}]",
            "```",
            "",
            f"**Quality score (1–5):** _fill in_",
            "",
            "---",
            "",
        ]

    (results_dir / "human_review.md").write_text("\n".join(lines))


# ── Tool-reliability mode (reuses Phase 0 scorer) ─────────────────────────────

async def run_tool_reliability(
    client: BenchClient,
    tasks_dir: Path,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Reuse Phase 0 task format + scoring. Returns a score-summary dict."""
    from lib.tool_tasks import load_tasks
    from lib.scorer import score_tool_call, summarise_scores

    tasks = load_tasks(tasks_dir)
    scored = []
    for task in tasks:
        result = await client.chat(
            messages=task.to_messages(),
            tools=task.tools,
            task_id=task.id,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        sr = score_tool_call(result, task.expected_tool_calls)
        scored.append(sr)
        icon = "[green]✓[/green]" if sr.passed else "[red]✗[/red]"
        console.print(f"  {icon} {sr.task_id:40s} {sr.score}")

    return summarise_scores(scored)


# ── Spec-decode mode ──────────────────────────────────────────────────────────

async def run_spec_decode(
    client: BenchClient,
    tasks: list[dict[str, Any]],
    model: str,
    max_tokens: int,
) -> PhaseMetrics:
    """
    Plain throughput run — same as Phase 1 throughput mode.
    Deploy twice (with/without spec-decode) and compare externally.
    """
    pm = PhaseMetrics()
    for task in tasks:
        result = await client.chat(
            messages=[
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user", "content": task["user_message"]},
            ],
            task_id=task["id"],
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        pm.add(result)
        console.print(
            f"  {task['id']:35s}  "
            f"ttft={result.ttft_ms:6.0f}ms  tps={result.decode_tps:6.1f}"
        )
    return pm


# ── Engine version ─────────────────────────────────────────────────────────────

async def _get_engine_version(base_url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{base_url.rstrip('/v1')}/version")
            if r.status_code == 200:
                return r.json().get("version", "unknown")
    except Exception:
        pass
    return "unknown"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = Path(args.tasks)

    console.print(f"[bold]Phase 2 — Model selection ({args.mode})[/bold]")
    console.print(f"Endpoint : {args.endpoint}")
    console.print(f"Mode     : {args.mode}")
    console.print(f"Label    : {args.label}")

    client = BenchClient(base_url=args.endpoint, results_dir=results_dir)
    engine_version = await _get_engine_version(args.endpoint)

    thresholds_path = Path(args.thresholds)
    thresholds: dict[str, Any] = {}
    if thresholds_path.exists():
        thresholds = yaml.safe_load(thresholds_path.read_text()) or {}

    extra_metrics: dict[str, Any] = {}
    outputs_for_review: list[dict[str, Any]] = []

    if args.mode == "quality":
        tasks = load_quality_tasks(tasks_dir)
        console.print(f"Loaded {len(tasks)} task(s)\n")
        pm, outputs_for_review = await run_quality(
            client, tasks, model=args.model, max_tokens=args.max_tokens, label=args.label
        )
        timing = pm.to_dict()
        n = timing["n"]
        completed = sum(1 for o in outputs_for_review if o["completed"])
        task_completion_rate = completed / n if n else 0.0
        extra_metrics = {
            **timing,
            "task_completion_rate": task_completion_rate,
        }

        table = Table(title=f"Quality Results — {args.label}")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Pass threshold", justify="right")
        table.add_row("Tasks run", str(n))
        table.add_row("Task completion rate", f"{task_completion_rate:.1%}", "100%")
        table.add_row("Decode tps (mean)", f"{timing['decode_tps_mean']:.1f}", "≥100")
        table.add_row("TTFT p50 (ms)", f"{timing['ttft_p50_ms']:.0f}", "—")
        table.add_row("Error rate", f"{timing['error_rate']:.1%}", "—")
        console.print(table)

        _write_human_review(outputs_for_review, results_dir, args.label)
        console.print(f"[dim]Human review sheet: {results_dir}/human_review.md[/dim]")

    elif args.mode == "tool-reliability":
        console.print(f"Loaded tasks from {tasks_dir}\n")
        score_summary = await run_tool_reliability(
            client, tasks_dir, model=args.model, max_tokens=args.max_tokens
        )
        pass_rate = score_summary.get("tool_call_success_rate", 0.0)
        critical_rate = score_summary.get("critical_error_rate", 0.0)
        extra_metrics = {**score_summary, "task_completion_rate": pass_rate}

        table = Table(title=f"Tool Reliability — {args.label}")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Pass threshold", justify="right")
        table.add_row("Tool call success rate", f"{pass_rate:.1%}", "≥95%")
        table.add_row("Critical errors (engine)", f"{critical_rate:.1%}", "0%")
        for score_val, count in score_summary.get("counts", {}).items():
            table.add_row(f"  {score_val}", str(count), "")
        console.print(table)

    elif args.mode == "spec-decode":
        tasks = load_quality_tasks(tasks_dir)
        console.print(f"Loaded {len(tasks)} task(s)\n")
        pm = await run_spec_decode(
            client, tasks, model=args.model, max_tokens=args.max_tokens
        )
        timing = pm.to_dict()
        extra_metrics = {**timing, "task_completion_rate": 1.0 - timing["error_rate"]}

        table = Table(title=f"Spec-decode throughput — {args.label}")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Decode tps (mean)", f"{timing['decode_tps_mean']:.1f}")
        table.add_row("TTFT p50 (ms)", f"{timing['ttft_p50_ms']:.0f}")
        table.add_row("Error rate", f"{timing['error_rate']:.1%}")
        console.print(table)

    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")

    # ── Verdict ────────────────────────────────────────────────────────────────
    # Phase 2 verdict uses phase0 thresholds for tool-reliability, phase2 otherwise
    verdict_phase = "phase0_tool_reliability" if args.mode == "tool-reliability" else PHASE
    from lib.reporter import _verdict
    verdict = _verdict(verdict_phase, extra_metrics, thresholds)
    color = {"PASS": "green", "FAIL": "red", "INCONCLUSIVE": "yellow"}[verdict]
    console.print(f"\n[bold {color}]Verdict: {verdict}[/bold {color}]")

    if args.mode == "quality":
        console.print("[dim]Note: quality verdict is based on completion rate + speed only. "
                      "Human review of human_review.md is required.[/dim]")

    # ── metrics.json ───────────────────────────────────────────────────────────
    metrics_out: dict[str, Any] = {
        "phase": PHASE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "engine": args.engine,
            "engine_version": engine_version,
            "model": args.model,
            "label": args.label,
            "quantization": args.quantization,
            "gpu": args.gpu_label,
            "gpu_id": args.gpu_id,
            "context_length": args.ctx_len,
            "extra_args": args.extra_args,
            "mode": args.mode,
        },
        "metrics": extra_metrics,
        "verdict": verdict,
        "notes": args.notes,
    }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))

    from lib.reporter import generate_summary
    summary = generate_summary(results_dir, thresholds)
    (results_dir / "summary.md").write_text(summary)
    console.print(f"Results: {results_dir}")

    if verdict == "FAIL":
        raise SystemExit(1)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Phase 2 — Model selection benchmark")
    parser.add_argument("--endpoint", default="http://localhost:30000/v1")
    parser.add_argument("--results-dir", default=f"results/{PHASE}_{ts}")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--mode",
                        choices=["quality", "tool-reliability", "spec-decode"],
                        default="quality")
    parser.add_argument("--label", default="model",
                        help="Human-readable label for this run (e.g. 'Qwen3.5-35B-AWQ')")
    parser.add_argument("--model", default="default")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thresholds", default="config/thresholds.yaml")
    # Metadata
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--quantization", default="unknown")
    parser.add_argument("--gpu-label", default="RTX 5090")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--ctx-len", type=int, default=32768)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--notes", default="")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
