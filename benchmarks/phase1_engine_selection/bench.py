"""
Phase 1 — Engine selection benchmark.

Two modes:
  throughput   Measure TTFT, decode speed, and latency at configurable concurrency.
               Used for sub-tests 1.1 (vLLM vs SGLang) and 1.4 (llama.cpp comparison).

  prefix-cache Measure KV-cache effectiveness: compare TTFT on cold vs warm requests
               that share a long common prefix. Used for sub-tests 1.2 and 1.3.

Usage:
    python -m benchmarks.phase1_engine_selection.bench \\
        --endpoint http://localhost:30000/v1 \\
        --results-dir results/phase1_1.1_vllm_20260414 \\
        --mode throughput \\
        --tasks benchmarks/phase1_engine_selection/tasks/throughput/ \\
        --concurrency 4

    python -m benchmarks.phase1_engine_selection.bench \\
        --endpoint http://localhost:30000/v1 \\
        --results-dir results/phase1_1.3_vllm_prefix_20260414 \\
        --mode prefix-cache \\
        --tasks benchmarks/phase1_engine_selection/tasks/prefix_cache/ \\
        --warmup-rounds 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
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

PHASE = "phase1_engine_selection"


# ── Task schema for Phase 1 ────────────────────────────────────────────────────

def load_generation_tasks(tasks_dir: Path) -> list[dict[str, Any]]:
    """Load simple generation tasks (no tool schema needed for Phase 1)."""
    files = sorted(tasks_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No task JSON files in {tasks_dir}")
    tasks = []
    for f in files:
        data = json.loads(f.read_text())
        # Minimal required fields
        for key in ("id", "system_prompt", "user_message"):
            if key not in data:
                raise ValueError(f"Task {f.name} missing required field '{key}'")
        tasks.append(data)
    return tasks


# ── Throughput mode ───────────────────────────────────────────────────────────

async def run_throughput(
    client: BenchClient,
    tasks: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    concurrency: int,
) -> PhaseMetrics:
    """
    Run all tasks at the given concurrency, collect timing metrics.
    Each task is run once; increase task count or repeat runs for statistical stability.
    """
    pm = PhaseMetrics()
    sem = asyncio.Semaphore(concurrency)

    async def _run_one(task: dict[str, Any]) -> BenchResult:
        async with sem:
            messages = [
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user", "content": task["user_message"]},
            ]
            return await client.chat(
                messages=messages,
                task_id=task["id"],
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
            )

    results = await asyncio.gather(*[_run_one(t) for t in tasks])
    for r in results:
        pm.add(r)
        status = "ok" if r.ok else f"ERR:{r.error[:40]}"
        console.print(
            f"  {r.task_id:35s}  ttft={r.ttft_ms:6.0f}ms  "
            f"tps={r.decode_tps:6.1f}  {status}"
        )
    return pm


# ── Prefix-cache mode ─────────────────────────────────────────────────────────

async def run_prefix_cache(
    client: BenchClient,
    tasks: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    warmup_rounds: int,
) -> dict[str, Any]:
    """
    Measure KV-cache speedup.

    For each task:
      1. Send `warmup_rounds` requests to populate the cache.
      2. Send one cold measurement (first-ever request, or after a cache flush).
      3. Send one warm measurement immediately after.
      4. Compute speedup = ttft_warm / ttft_cold.

    Returns a dict with per-task cold/warm TTFT and aggregate speedup ratio.
    """
    results: list[dict[str, Any]] = []

    for task in tasks:
        messages = [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["user_message"]},
        ]
        task_id = task["id"]

        # ── Warm-up rounds (populate KV cache) ─────────────────────────────
        for i in range(warmup_rounds):
            warmup_msg = messages[:-1] + [{"role": "user", "content": task.get("warmup_message", task["user_message"])}]
            await client.chat(
                messages=warmup_msg,
                task_id=f"{task_id}_warmup_{i}",
                model=model,
                temperature=0.0,
                max_tokens=64,  # short — just enough to populate the prefix
            )

        # ── Cold measurement ────────────────────────────────────────────────
        # Use a unique user message so it's definitely a cache miss on the full turn
        cold_msgs = messages[:-1] + [
            {"role": "user", "content": task.get("cold_message", task["user_message"] + " (cold)")}
        ]
        cold_result = await client.chat(
            messages=cold_msgs,
            task_id=f"{task_id}_cold",
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
        )

        # ── Warm measurement ────────────────────────────────────────────────
        # Same messages as cold — prefix is now cached
        warm_result = await client.chat(
            messages=cold_msgs,
            task_id=f"{task_id}_warm",
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
        )

        speedup = (
            warm_result.ttft_ms / cold_result.ttft_ms
            if cold_result.ttft_ms > 0
            else 1.0
        )
        results.append({
            "task_id": task_id,
            "ttft_cold_ms": cold_result.ttft_ms,
            "ttft_warm_ms": warm_result.ttft_ms,
            "speedup_ratio": speedup,
            "prefix_tokens": task.get("prefix_tokens", 0),
            "cold_ok": cold_result.ok,
            "warm_ok": warm_result.ok,
        })
        console.print(
            f"  {task_id:35s}  cold={cold_result.ttft_ms:6.0f}ms  "
            f"warm={warm_result.ttft_ms:6.0f}ms  ratio={speedup:.3f}"
        )

    valid = [r for r in results if r["cold_ok"] and r["warm_ok"] and r["ttft_cold_ms"] > 0]
    mean_ratio = statistics.mean(r["speedup_ratio"] for r in valid) if valid else 1.0

    return {
        "per_task": results,
        "prefix_reuse_speedup": mean_ratio,
        "n_valid": len(valid),
    }


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

    console.print(f"[bold]Phase 1 — Engine selection ({args.mode})[/bold]")
    console.print(f"Endpoint   : {args.endpoint}")
    console.print(f"Mode       : {args.mode}")
    console.print(f"Tasks dir  : {tasks_dir}")
    console.print(f"Concurrency: {args.concurrency}")

    tasks = load_generation_tasks(tasks_dir)
    console.print(f"Loaded {len(tasks)} task(s)\n")

    client = BenchClient(base_url=args.endpoint, results_dir=results_dir)
    engine_version = await _get_engine_version(args.endpoint)

    # ── Run ────────────────────────────────────────────────────────────────────
    extra_metrics: dict[str, Any] = {}

    if args.mode == "throughput":
        pm = await run_throughput(
            client, tasks,
            model=args.model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
        timing = pm.to_dict()
        extra_metrics = timing

        table = Table(title="Throughput Results")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Pass threshold", justify="right")
        table.add_row("Tasks", str(timing["n"]))
        table.add_row("TTFT p50 (ms)", f"{timing['ttft_p50_ms']:.0f}", "≤500")
        table.add_row("TTFT p95 (ms)", f"{timing['ttft_p95_ms']:.0f}", "≤1000")
        table.add_row("Decode tps (mean)", f"{timing['decode_tps_mean']:.1f}", "≥150")
        table.add_row("Latency p50 (ms)", f"{timing['latency_p50_ms']:.0f}", "—")
        table.add_row("Error rate", f"{timing['error_rate']:.1%}", "—")
        console.print(table)

    elif args.mode == "prefix-cache":
        cache_results = await run_prefix_cache(
            client, tasks,
            model=args.model,
            max_tokens=args.max_tokens,
            warmup_rounds=args.warmup_rounds,
        )
        extra_metrics = cache_results

        speedup = cache_results["prefix_reuse_speedup"]
        table = Table(title="Prefix Cache Results")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Pass threshold", justify="right")
        table.add_row("Tasks valid", str(cache_results["n_valid"]))
        table.add_row("Prefix reuse speedup (ratio)", f"{speedup:.3f}", "≤0.50")
        console.print(table)

        for row in cache_results["per_task"]:
            console.print(
                f"  {row['task_id']:35s}  "
                f"cold={row['ttft_cold_ms']:.0f}ms  "
                f"warm={row['ttft_warm_ms']:.0f}ms  "
                f"ratio={row['speedup_ratio']:.3f}"
            )
    else:
        raise ValueError(f"Unknown mode: {args.mode!r}. Use 'throughput' or 'prefix-cache'.")

    # ── Verdict ────────────────────────────────────────────────────────────────
    thresholds_path = Path(args.thresholds)
    thresholds: dict[str, Any] = {}
    if thresholds_path.exists():
        thresholds = yaml.safe_load(thresholds_path.read_text()) or {}

    from lib.reporter import _verdict
    verdict = _verdict(PHASE, extra_metrics, thresholds)
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
            "mode": args.mode,
            "concurrency": args.concurrency,
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
    parser = argparse.ArgumentParser(description="Phase 1 — Engine selection benchmark")
    parser.add_argument("--endpoint", default="http://localhost:30000/v1")
    parser.add_argument("--results-dir", default=f"results/{PHASE}_{ts}")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--mode", choices=["throughput", "prefix-cache"], default="throughput")
    parser.add_argument("--model", default="default")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup-rounds", type=int, default=2,
                        help="prefix-cache mode only: rounds to populate KV cache before measuring")
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
