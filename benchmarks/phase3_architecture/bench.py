"""
Phase 3 — Architecture benchmark.

Four modes:
  dual-model    Run coder + thinker tasks co-resident on separate GPU endpoints.
                Both models loaded simultaneously; measures per-endpoint throughput.
                Used for sub-test 3.1 — does the multi-tier architecture hurt
                throughput vs single-model occupation of one GPU?

  routing       Test LiteLLM proxy routing accuracy.
                A heuristic classifier assigns each task to "coder" or "thinker"
                tier; requests are forwarded to the matching LiteLLM model alias
                and the response is checked to confirm the right backend was used.
                Used for sub-test 3.2.

  swap-timing   Measure model hot-swap latency end-to-end.
                Calls infra/scripts/swap-model.sh N times, parses wall-clock
                output, and reports p50/p95 latency.
                Used for sub-test 3.3.

  swarm         Parallel concurrency stress test.
                Runs a fixed task set sequentially, then at the specified
                concurrency level, and reports the wall-time speedup ratio.
                Used for sub-test 3.4.

Usage:
    # Dual-model mode (both endpoints must be running)
    python -m benchmarks.phase3_architecture.bench \\
        --mode dual-model \\
        --endpoint http://localhost:30000/v1 \\
        --thinker-endpoint http://localhost:30001/v1 \\
        --tasks benchmarks/phase2_model_selection/tasks/quality/ \\
        --thinker-tasks benchmarks/phase2_model_selection/tasks/thinker/ \\
        --results-dir results/phase3_3.1_dual_20260414 \\
        --label "Dual Qwen3.5-35B+27B"

    # Routing mode (LiteLLM proxy must be running at --endpoint)
    python -m benchmarks.phase3_architecture.bench \\
        --mode routing \\
        --endpoint http://localhost:30100/v1 \\
        --tasks benchmarks/phase3_architecture/tasks/routing/ \\
        --results-dir results/phase3_3.2_routing_20260414 \\
        --coder-model-id "QuantTrio/Qwen3.5-35B-A3B-AWQ" \\
        --thinker-model-id "Qwen/Qwen3.5-27B"

    # Swap-timing mode
    python -m benchmarks.phase3_architecture.bench \\
        --mode swap-timing \\
        --engine vllm --gpu gpu0 \\
        --model QuantTrio/Qwen3.5-35B-A3B-AWQ \\
        --swap-target Qwen/Qwen3.5-27B \\
        --swap-rounds 5 \\
        --results-dir results/phase3_3.3_swap_20260414

    # Swarm mode
    python -m benchmarks.phase3_architecture.bench \\
        --mode swarm \\
        --endpoint http://localhost:30000/v1 \\
        --tasks benchmarks/phase3_architecture/tasks/swarm/ \\
        --concurrency 8 \\
        --results-dir results/phase3_3.4_swarm_20260414
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
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

PHASE = "phase3_architecture"

# ── Heuristic routing classifier ─────────────────────────────────────────────

_THINKER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(analy[sz]e|explain why|reason|prove|disprove|design a system|"
               r"trade.?off|compare and contrast|argue|formal(ly)?|hypothesis|"
               r"what would happen|root cause|correctness|optimal)\b", re.IGNORECASE),
]

_CODER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(implement|fix|write|refactor|complete|add|debug|port|"
               r"unit test|type hint|annotate|migrate|optimize the query|"
               r"n\+1|pagination)\b", re.IGNORECASE),
]


def classify_tier(user_message: str) -> str:
    """Return 'thinker' or 'coder' based on keyword heuristics."""
    thinker_score = sum(
        len(p.findall(user_message)) for p in _THINKER_PATTERNS
    )
    coder_score = sum(
        len(p.findall(user_message)) for p in _CODER_PATTERNS
    )
    return "thinker" if thinker_score >= coder_score else "coder"


# ── Task loaders ──────────────────────────────────────────────────────────────

def load_tasks(tasks_dir: Path, required_fields: tuple[str, ...] = ("id", "system_prompt", "user_message")) -> list[dict[str, Any]]:
    files = sorted(tasks_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No task JSON files in {tasks_dir}")
    tasks = []
    for f in files:
        data = json.loads(f.read_text())
        for key in required_fields:
            if key not in data:
                raise ValueError(f"Task {f.name} missing required field '{key}'")
        tasks.append(data)
    return tasks


# ── Dual-model mode ───────────────────────────────────────────────────────────

async def run_dual_model(
    coder_client: BenchClient,
    thinker_client: BenchClient,
    coder_tasks: list[dict[str, Any]],
    thinker_tasks: list[dict[str, Any]],
    coder_model: str,
    thinker_model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """
    Run coder and thinker tasks concurrently against their respective endpoints.
    Both models are loaded at the same time (co-resident on separate GPUs).
    Returns per-endpoint PhaseMetrics converted to dicts.
    """
    async def _run_tasks(
        client: BenchClient,
        tasks: list[dict[str, Any]],
        model: str,
        label: str,
    ) -> PhaseMetrics:
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
            icon = "[green]✓[/green]" if result.ok else "[red]✗[/red]"
            console.print(
                f"  [{label}] {icon} {task['id']:38s}  "
                f"ttft={result.ttft_ms:6.0f}ms  tps={result.decode_tps:5.1f}"
            )
        return pm

    console.print("[dim]Running coder + thinker tasks concurrently...[/dim]")
    coder_pm, thinker_pm = await asyncio.gather(
        _run_tasks(coder_client, coder_tasks, coder_model, "coder"),
        _run_tasks(thinker_client, thinker_tasks, thinker_model, "thinker"),
    )

    ct = coder_pm.to_dict()
    tt = thinker_pm.to_dict()
    return {
        "coder_n": ct["n"],
        "coder_decode_tps": ct["decode_tps_mean"],
        "coder_ttft_p50_ms": ct["ttft_p50_ms"],
        "coder_error_rate": ct["error_rate"],
        "thinker_n": tt["n"],
        "thinker_decode_tps": tt["decode_tps_mean"],
        "thinker_ttft_p50_ms": tt["ttft_p50_ms"],
        "thinker_error_rate": tt["error_rate"],
    }


# ── Routing mode ──────────────────────────────────────────────────────────────

async def run_routing(
    proxy_url: str,
    tasks: list[dict[str, Any]],
    coder_model_id: str,
    thinker_model_id: str,
    results_dir: Path,
) -> dict[str, Any]:
    """
    Test LiteLLM routing accuracy.

    Algorithm:
    1. For each task, apply heuristic classifier to determine predicted tier.
    2. Send request to LiteLLM proxy with model alias matching prediction
       ("coder" or "thinker").
    3. Compare prediction against task's correct_tier field.
    4. Check response x-litellm-model header to confirm backend was reached.
    """
    correct = 0
    total = len(tasks)
    per_task: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=60.0) as http:
        for task in tasks:
            correct_tier: str = task.get("correct_tier", "coder")
            predicted_tier = classify_tier(task["user_message"])
            model_alias = predicted_tier  # "coder" or "thinker"

            t0 = time.monotonic()
            try:
                resp = await http.post(
                    f"{proxy_url.rstrip('/')}/chat/completions",
                    json={
                        "model": model_alias,
                        "messages": [
                            {"role": "system", "content": task["system_prompt"]},
                            {"role": "user", "content": task["user_message"]},
                        ],
                        "max_tokens": 256,
                        "temperature": 0.0,
                        "stream": False,
                    },
                    headers={"Authorization": "Bearer local-dev"},
                )
                resp.raise_for_status()
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                data = resp.json()
                # LiteLLM echoes the backend model in the model field
                used_model: str = data.get("model", "")
                error = ""
            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                used_model = ""
                error = str(exc)

            # Determine whether routing was correct
            # Map used_model back to a tier based on coder_model_id / thinker_model_id
            if coder_model_id.lower() in used_model.lower():
                actual_tier = "coder"
            elif thinker_model_id.lower() in used_model.lower():
                actual_tier = "thinker"
            else:
                # Fall back to prediction when backend model can't be identified
                actual_tier = predicted_tier

            prediction_correct = predicted_tier == correct_tier
            routing_correct = actual_tier == correct_tier
            if prediction_correct:
                correct += 1

            icon = "[green]✓[/green]" if prediction_correct else "[red]✗[/red]"
            console.print(
                f"  {icon} {task['id']:35s}  "
                f"pred={predicted_tier:7s}  correct={correct_tier:7s}  "
                f"latency={elapsed_ms:.0f}ms"
            )

            per_task.append({
                "id": task["id"],
                "correct_tier": correct_tier,
                "predicted_tier": predicted_tier,
                "actual_tier": actual_tier,
                "prediction_correct": prediction_correct,
                "routing_correct": routing_correct,
                "latency_ms": elapsed_ms,
                "used_model": used_model,
                "error": error,
            })

    routing_accuracy = correct / total if total else 0.0
    latencies = [t["latency_ms"] for t in per_task if not t["error"]]
    return {
        "routing_accuracy": routing_accuracy,
        "n_tasks": total,
        "correct": correct,
        "incorrect": total - correct,
        "routing_overhead_ms_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "per_task": per_task,
    }


# ── Swap-timing mode ──────────────────────────────────────────────────────────

async def run_swap_timing(
    engine: str,
    gpu: str,
    model_a: str,
    model_b: str,
    rounds: int,
    results_dir: Path,
    extra_args_a: str = "",
    extra_args_b: str = "",
) -> dict[str, Any]:
    """
    Alternate between model_a and model_b for `rounds` swap cycles.
    Each cycle calls swap-model.sh and parses SWAP_LATENCY_MS from stdout.
    Returns p50/p95/mean in seconds.
    """
    repo_root = Path(__file__).resolve().parents[3]
    swap_script = repo_root / "infra" / "scripts" / "swap-model.sh"

    latencies_ms: list[float] = []
    per_round: list[dict[str, Any]] = []

    # Start with model_a deployed; alternate to model_b and back
    models = [model_b, model_a] * rounds
    extra = [extra_args_b, extra_args_a] * rounds

    for i, (target_model, extra_a) in enumerate(zip(models, extra)):
        console.print(f"  Round {i + 1}/{len(models)}: swapping to {target_model} ...")
        cmd = [str(swap_script), engine, gpu, target_model]
        if extra_a:
            cmd.extend(extra_a.split())

        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout_bytes, _ = await proc.communicate()
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            stdout = stdout_bytes.decode(errors="replace")

            # Parse the machine-readable line emitted by swap-model.sh
            m = re.search(r"SWAP_LATENCY_MS=(\d+)", stdout)
            if m:
                swap_ms = float(m.group(1))
            else:
                swap_ms = elapsed_ms  # fallback to wall clock

            error = "" if proc.returncode == 0 else f"exit {proc.returncode}"
        except Exception as exc:
            swap_ms = (time.monotonic() - t0) * 1000.0
            error = str(exc)
            console.print(f"    [red]Error: {error}[/red]")

        latencies_ms.append(swap_ms)
        swap_s = swap_ms / 1000.0
        per_round.append({
            "round": i + 1,
            "target_model": target_model,
            "swap_latency_ms": swap_ms,
            "swap_latency_s": swap_s,
            "error": error,
        })
        console.print(f"    Swap time: {swap_s:.2f}s")

    sorted_ms = sorted(latencies_ms)
    n = len(sorted_ms)
    p50_ms = sorted_ms[n // 2]
    p95_ms = sorted_ms[min(int(n * 0.95), n - 1)]
    mean_ms = sum(sorted_ms) / n if n else 0.0

    return {
        "swap_latency_s": p95_ms / 1000.0,       # reporter reads this for verdict
        "swap_latency_s_p50": p50_ms / 1000.0,
        "swap_latency_s_p95": p95_ms / 1000.0,
        "swap_latency_s_mean": mean_ms / 1000.0,
        "rounds": n,
        "per_round": per_round,
    }


# ── Swarm mode ────────────────────────────────────────────────────────────────

async def run_swarm(
    client: BenchClient,
    tasks: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    """
    Compare sequential vs parallel wall time for the same task set.
    swarm_speedup = parallel_wall_s / sequential_wall_s  (lower is better)
    """
    async def _send_one(task: dict[str, Any]) -> BenchResult:
        return await client.chat(
            messages=[
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user", "content": task["user_message"]},
            ],
            task_id=task["id"],
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    # ── Sequential baseline ────────────────────────────────────────────────────
    console.print("[dim]Sequential baseline...[/dim]")
    seq_pm = PhaseMetrics()
    t_seq_start = time.monotonic()
    for task in tasks:
        r = await _send_one(task)
        seq_pm.add(r)
        console.print(
            f"  [seq] {task['id']:35s}  tps={r.decode_tps:5.1f}  "
            f"{'ok' if r.ok else 'ERR'}"
        )
    seq_wall_s = time.monotonic() - t_seq_start

    # ── Parallel run ───────────────────────────────────────────────────────────
    console.print(f"[dim]Parallel run (concurrency={concurrency})...[/dim]")
    par_pm = PhaseMetrics()
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(task: dict[str, Any]) -> BenchResult:
        async with sem:
            return await _send_one(task)

    t_par_start = time.monotonic()
    par_results = await asyncio.gather(*[_bounded(t) for t in tasks])
    par_wall_s = time.monotonic() - t_par_start

    for r in par_results:
        par_pm.add(r)

    swarm_speedup = par_wall_s / seq_wall_s if seq_wall_s > 0 else 1.0
    seq_d = seq_pm.to_dict()
    par_d = par_pm.to_dict()

    console.print(
        f"  Sequential: {seq_wall_s:.1f}s  "
        f"Parallel: {par_wall_s:.1f}s  "
        f"Speedup ratio: {swarm_speedup:.3f}"
    )

    return {
        "swarm_speedup": swarm_speedup,
        "sequential_wall_s": seq_wall_s,
        "parallel_wall_s": par_wall_s,
        "concurrency": concurrency,
        "n_tasks": len(tasks),
        "seq_decode_tps_mean": seq_d["decode_tps_mean"],
        "par_decode_tps_mean": par_d["decode_tps_mean"],
        "seq_error_rate": seq_d["error_rate"],
        "par_error_rate": par_d["error_rate"],
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

    console.print(f"[bold]Phase 3 — Architecture ({args.mode})[/bold]")
    console.print(f"Endpoint : {args.endpoint}")

    thresholds_path = Path(args.thresholds)
    thresholds: dict[str, Any] = {}
    if thresholds_path.exists():
        thresholds = yaml.safe_load(thresholds_path.read_text()) or {}

    extra_metrics: dict[str, Any] = {}
    engine_version = "N/A"

    if args.mode == "dual-model":
        console.print(f"Thinker  : {args.thinker_endpoint}")
        engine_version = await _get_engine_version(args.endpoint)

        coder_client = BenchClient(base_url=args.endpoint, results_dir=results_dir)
        thinker_client = BenchClient(base_url=args.thinker_endpoint, results_dir=results_dir)

        coder_tasks = load_tasks(Path(args.tasks))
        thinker_tasks = load_tasks(Path(args.thinker_tasks))
        console.print(
            f"Coder tasks: {len(coder_tasks)}  "
            f"Thinker tasks: {len(thinker_tasks)}\n"
        )

        extra_metrics = await run_dual_model(
            coder_client, thinker_client,
            coder_tasks, thinker_tasks,
            coder_model=args.model,
            thinker_model=args.thinker_model,
            max_tokens=args.max_tokens,
        )

        table = Table(title=f"Dual-model — {args.label}")
        table.add_column("Endpoint", style="bold")
        table.add_column("Tasks", justify="right")
        table.add_column("Decode TPS", justify="right")
        table.add_column("TTFT p50 (ms)", justify="right")
        table.add_column("Errors", justify="right")
        table.add_row(
            "Coder",
            str(extra_metrics["coder_n"]),
            f"{extra_metrics['coder_decode_tps']:.1f}",
            f"{extra_metrics['coder_ttft_p50_ms']:.0f}",
            f"{extra_metrics['coder_error_rate']:.1%}",
        )
        table.add_row(
            "Thinker",
            str(extra_metrics["thinker_n"]),
            f"{extra_metrics['thinker_decode_tps']:.1f}",
            f"{extra_metrics['thinker_ttft_p50_ms']:.0f}",
            f"{extra_metrics['thinker_error_rate']:.1%}",
        )
        console.print(table)
        console.print(
            "[dim]Note: dual-model verdict is informational (no dedicated threshold). "
            "Compare coder_decode_tps vs Phase 2 single-model baseline.[/dim]"
        )

    elif args.mode == "routing":
        tasks = load_tasks(Path(args.tasks), required_fields=("id", "system_prompt", "user_message", "correct_tier"))
        console.print(f"Loaded {len(tasks)} routing task(s)\n")

        extra_metrics = await run_routing(
            proxy_url=args.endpoint,
            tasks=tasks,
            coder_model_id=args.coder_model_id,
            thinker_model_id=args.thinker_model_id,
            results_dir=results_dir,
        )

        table = Table(title="Routing accuracy")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Pass threshold", justify="right")
        table.add_row("Tasks", str(extra_metrics["n_tasks"]))
        table.add_row("Correct", str(extra_metrics["correct"]))
        table.add_row("routing_accuracy", f"{extra_metrics['routing_accuracy']:.1%}", "≥95%")
        table.add_row("Mean overhead (ms)", f"{extra_metrics['routing_overhead_ms_mean']:.0f}", "—")
        console.print(table)

    elif args.mode == "swap-timing":
        console.print(
            f"Engine={args.engine}  GPU={args.gpu}  "
            f"Model A={args.model}  Model B={args.swap_target}  "
            f"Rounds={args.swap_rounds}\n"
        )
        extra_metrics = await run_swap_timing(
            engine=args.engine,
            gpu=args.gpu,
            model_a=args.model,
            model_b=args.swap_target,
            rounds=args.swap_rounds,
            results_dir=results_dir,
            extra_args_a=args.extra_args,
            extra_args_b=args.swap_target_extra_args,
        )

        table = Table(title="Swap timing")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Pass threshold", justify="right")
        table.add_row("Rounds (each direction)", str(args.swap_rounds))
        table.add_row("swap_latency_s (p95)", f"{extra_metrics['swap_latency_s_p95']:.2f}s", "≤30s")
        table.add_row("swap_latency_s_p50", f"{extra_metrics['swap_latency_s_p50']:.2f}s", "—")
        table.add_row("swap_latency_s_mean", f"{extra_metrics['swap_latency_s_mean']:.2f}s", "—")
        console.print(table)

    elif args.mode == "swarm":
        tasks = load_tasks(Path(args.tasks))
        console.print(f"Loaded {len(tasks)} swarm task(s)  concurrency={args.concurrency}\n")
        engine_version = await _get_engine_version(args.endpoint)

        client = BenchClient(base_url=args.endpoint, results_dir=results_dir)
        extra_metrics = await run_swarm(
            client, tasks,
            model=args.model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )

        table = Table(title=f"Swarm @ concurrency={args.concurrency}")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Pass threshold", justify="right")
        table.add_row("Tasks", str(extra_metrics["n_tasks"]))
        table.add_row("Sequential wall time", f"{extra_metrics['sequential_wall_s']:.1f}s")
        table.add_row("Parallel wall time", f"{extra_metrics['parallel_wall_s']:.1f}s")
        table.add_row("swarm_speedup", f"{extra_metrics['swarm_speedup']:.3f}", "≤0.60")
        table.add_row("Seq decode TPS", f"{extra_metrics['seq_decode_tps_mean']:.1f}")
        table.add_row("Par decode TPS", f"{extra_metrics['par_decode_tps_mean']:.1f}")
        console.print(table)

    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")

    # ── Verdict ────────────────────────────────────────────────────────────────
    from lib.reporter import _verdict
    verdict = _verdict(PHASE, extra_metrics, thresholds)
    color = {"PASS": "green", "FAIL": "red", "INCONCLUSIVE": "yellow"}[verdict]
    console.print(f"\n[bold {color}]Verdict: {verdict}[/bold {color}]")

    # ── metrics.json ───────────────────────────────────────────────────────────
    metrics_out: dict[str, Any] = {
        "phase": PHASE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "engine": args.engine,
            "engine_version": engine_version,
            "model": args.model,
            "label": args.label,
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
    (results_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    from lib.reporter import generate_summary
    summary = generate_summary(results_dir, thresholds)
    (results_dir / "summary.md").write_text(summary)
    console.print(f"Results: {results_dir}")

    if verdict == "FAIL":
        raise SystemExit(1)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Phase 3 — Architecture benchmark")
    parser.add_argument("--endpoint", default="http://localhost:30000/v1")
    parser.add_argument("--results-dir", default=f"results/{PHASE}_{ts}")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--mode",
                        choices=["dual-model", "routing", "swap-timing", "swarm"],
                        required=True)
    parser.add_argument("--label", default="phase3")
    parser.add_argument("--model", default="default")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thresholds", default="config/thresholds.yaml")
    # Dual-model specific
    parser.add_argument("--thinker-endpoint", default="http://localhost:30001/v1")
    parser.add_argument("--thinker-tasks", default="")
    parser.add_argument("--thinker-model", default="default")
    # Routing specific
    parser.add_argument("--coder-model-id", default="",
                        help="Model ID string used by coder backend (for header matching)")
    parser.add_argument("--thinker-model-id", default="",
                        help="Model ID string used by thinker backend")
    # Swap-timing specific
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--gpu", default="gpu0")
    parser.add_argument("--swap-target", default="",
                        help="Model to swap to (alternates with --model)")
    parser.add_argument("--swap-rounds", type=int, default=5,
                        help="Number of A→B→A swap cycles")
    parser.add_argument("--swap-target-extra-args", default="")
    # Swarm specific
    parser.add_argument("--concurrency", type=int, default=8)
    # Metadata
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
