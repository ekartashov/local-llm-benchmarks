"""
Phase 0 — Tool-call reliability benchmark.

Runs all tasks in tasks/ against a live inference endpoint and scores each one
using lib.scorer. Writes raw responses, metrics.json, and summary.md.

Usage:
    python -m benchmarks.phase0_tool_reliability.bench \\
        --endpoint http://localhost:30000/v1 \\
        --results-dir results/phase0_20260414_103000 \\
        --tasks benchmarks/phase0_tool_reliability/tasks/ \\
        [--task-filter 01_read_file] \\
        [--model default] \\
        [--max-tokens 2048] \\
        [--concurrency 1]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from lib.client import BenchClient
from lib.metrics import PhaseMetrics
from lib.scorer import ScoredResult, ToolCallScore, score_tool_call, summarise_scores
from lib.tool_tasks import ToolTask, load_tasks

console = Console()

PHASE = "phase0_tool_reliability"


async def _get_engine_version(base_url: str) -> str:
    """Try to read the engine version from the /version endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{base_url.rstrip('/v1')}/version")
            if r.status_code == 200:
                return r.json().get("version", "unknown")
    except Exception:
        pass
    return "unknown"


async def _get_served_model(base_url: str) -> str:
    """Return the first model ID from /v1/models, or 'default' on failure."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{base_url}/models")
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    return data[0]["id"]
    except Exception:
        pass
    return "default"


async def run_task(
    client: BenchClient,
    task: ToolTask,
    model: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> ScoredResult:
    """Run a single task and return its scored result."""
    async with sem:
        result = await client.chat(
            messages=task.to_messages(),
            tools=task.tools,
            task_id=task.id,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return score_tool_call(result, task.expected_tool_calls)


async def main_async(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = Path(args.tasks)

    console.print(f"[bold]Phase 0 — Tool-call reliability[/bold]")
    console.print(f"Endpoint : {args.endpoint}")
    console.print(f"Tasks dir: {tasks_dir}")
    console.print(f"Results  : {results_dir}")

    tasks = load_tasks(tasks_dir, task_filter=args.task_filter)
    console.print(f"Loaded {len(tasks)} task(s)")

    client = BenchClient(base_url=args.endpoint, results_dir=results_dir)
    engine_version = await _get_engine_version(args.endpoint)

    # Auto-detect model if caller left the default placeholder.
    if args.model == "default":
        args.model = await _get_served_model(args.endpoint)
        console.print(f"Model    : {args.model} [dim](auto-detected)[/dim]")

    sem = asyncio.Semaphore(args.concurrency)
    scored_results: list[ScoredResult] = []
    phase_metrics = PhaseMetrics()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("Running tasks...", total=len(tasks))

        coros = [run_task(client, t, args.model, args.max_tokens, sem) for t in tasks]
        for coro in asyncio.as_completed(coros):
            sr = await coro
            scored_results.append(sr)
            if sr.bench_result:
                phase_metrics.add(sr.bench_result)
            icon = "[green]✓[/green]" if sr.passed else "[red]✗[/red]"
            console.print(f"  {icon} {sr.task_id:40s} {sr.score}")
            progress.advance(overall)

    # ── Results table ──────────────────────────────────────────────────────────
    score_summary = summarise_scores(scored_results)
    pass_rate = score_summary.get("tool_call_success_rate", 0.0)
    critical_rate = score_summary.get("critical_error_rate", 0.0)

    table = Table(title="Phase 0 Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Tasks run", str(score_summary.get("total", 0)))
    table.add_row("Pass rate", f"{pass_rate:.1%}")
    table.add_row("Critical errors (engine)", f"{critical_rate:.1%}")
    for score_val, count in score_summary.get("counts", {}).items():
        table.add_row(f"  {score_val}", str(count))

    timing = phase_metrics.to_dict()
    table.add_row("TTFT p50 (ms)", f"{timing['ttft_p50_ms']:.0f}")
    table.add_row("TTFT p95 (ms)", f"{timing['ttft_p95_ms']:.0f}")
    table.add_row("Decode tps (mean)", f"{timing['decode_tps_mean']:.1f}")
    console.print(table)

    # Verdict
    verdict: str
    if pass_rate >= 0.95 and critical_rate == 0.0:
        verdict = "PASS"
    elif pass_rate >= 0.80 and critical_rate <= 0.03:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"

    color = {"PASS": "green", "FAIL": "red", "INCONCLUSIVE": "yellow"}[verdict]
    console.print(f"\n[bold {color}]Verdict: {verdict}[/bold {color}]")

    # ── Write metrics.json ─────────────────────────────────────────────────────
    metrics_out: dict[str, Any] = {
        "phase": PHASE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "engine": args.engine,
            "engine_version": engine_version,
            "model": args.model,
            "quantization": args.quantization,
            "gpu": args.gpu_label,
            "gpu_id": args.gpu_id,
            "context_length": args.ctx_len,
            "extra_args": args.extra_args,
        },
        "metrics": {
            **score_summary,
            **timing,
        },
        "verdict": verdict,
        "notes": args.notes,
    }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    console.print(f"Metrics written to {metrics_path}")

    # ── Generate summary.md ────────────────────────────────────────────────────
    thresholds_path = Path(args.thresholds)
    thresholds: dict[str, Any] = {}
    if thresholds_path.exists():
        import yaml
        thresholds = yaml.safe_load(thresholds_path.read_text()) or {}

    from lib.reporter import generate_summary
    summary = generate_summary(results_dir, thresholds)
    summary_path = results_dir / "summary.md"
    summary_path.write_text(summary)
    console.print(f"Summary written to {summary_path}")

    # Exit non-zero on FAIL so CI / just phase0 can catch it
    if verdict == "FAIL":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 — Tool-call reliability benchmark")
    parser.add_argument("--endpoint", default="http://localhost:30000/v1")
    parser.add_argument(
        "--results-dir",
        default=f"results/{PHASE}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--tasks",
        default="benchmarks/phase0_tool_reliability/tasks/",
    )
    parser.add_argument("--task-filter", default=None, help="Only run tasks whose id starts with this")
    parser.add_argument("--model", default="default", help="Model name to pass to the API")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent requests")
    parser.add_argument("--thresholds", default="config/thresholds.yaml")
    # Metadata for metrics.json
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
