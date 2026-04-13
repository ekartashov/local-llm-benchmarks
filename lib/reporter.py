"""Generate summary.md from metrics.json; handles the 'compare' subcommand."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

console = Console()


# ── Verdict logic ─────────────────────────────────────────────────────────────


def _verdict(phase: str, metrics: dict[str, Any], thresholds: dict[str, Any]) -> str:
    """Return PASS / INCONCLUSIVE / FAIL for the given phase metrics."""
    phase_thresholds = thresholds.get(phase, {})
    if not phase_thresholds:
        return "INCONCLUSIVE"

    overall = "PASS"
    for key, spec in phase_thresholds.items():
        value = metrics.get(key)
        if value is None:
            # key not present — can't evaluate
            if overall == "PASS":
                overall = "INCONCLUSIVE"
            continue

        pass_val = spec.get("pass")
        incon_val = spec.get("inconclusive")

        if pass_val is None:
            continue

        # Determine directionality: if pass < inconclusive, lower is better.
        # Otherwise higher is better.
        lower_is_better = incon_val is not None and pass_val < incon_val

        if lower_is_better:
            if value <= pass_val:
                result = "PASS"
            elif incon_val is not None and value <= incon_val:
                result = "INCONCLUSIVE"
            else:
                result = "FAIL"
        else:
            if value >= pass_val:
                result = "PASS"
            elif incon_val is not None and value >= incon_val:
                result = "INCONCLUSIVE"
            else:
                result = "FAIL"

        if result == "FAIL":
            return "FAIL"
        if result == "INCONCLUSIVE" and overall == "PASS":
            overall = "INCONCLUSIVE"

    return overall


# ── Markdown generation ───────────────────────────────────────────────────────


def _metric_rows(metrics: dict[str, Any], phase_thresholds: dict[str, Any]) -> list[str]:
    """Produce markdown table rows for each top-level scalar metric."""
    rows: list[str] = []
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        spec = phase_thresholds.get(key, {})
        pass_val = spec.get("pass", "—")
        threshold_str = f"≥{pass_val}" if isinstance(pass_val, (int, float)) else "—"
        rows.append(f"| {key} | {value:.4g} | {threshold_str} |")
    return rows


def generate_summary(results_dir: Path, thresholds: dict[str, Any]) -> str:
    """Read metrics.json in results_dir, return a markdown summary string."""
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        return f"# Summary\n\nNo metrics.json found in {results_dir}\n"

    data: dict[str, Any] = json.loads(metrics_path.read_text())
    phase: str = data.get("phase", "unknown")
    timestamp: str = data.get("timestamp", "")
    config: dict[str, Any] = data.get("config", {})
    metrics: dict[str, Any] = data.get("metrics", {})
    notes: str = data.get("notes", "")

    phase_thresholds = thresholds.get(phase, {})
    verdict = _verdict(phase, metrics, thresholds)

    # Store verdict back into metrics.json
    data["verdict"] = verdict
    metrics_path.write_text(json.dumps(data, indent=2))

    lines: list[str] = [
        f"# {phase} — {verdict}",
        "",
        f"**Timestamp:** {timestamp}  ",
        f"**Engine:** {config.get('engine', '?')} {config.get('engine_version', '')}  ",
        f"**Model:** {config.get('model', '?')} ({config.get('quantization', '?')})  ",
        f"**GPU:** {config.get('gpu', '?')} (id={config.get('gpu_id', '?')})  ",
        f"**Context length:** {config.get('context_length', '?')}  ",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Pass threshold |",
        "|--------|-------|----------------|",
    ]
    lines.extend(_metric_rows(metrics, phase_thresholds))

    # Per-task breakdown (phase0 style)
    per_task: list[dict[str, Any]] = metrics.get("per_task", [])
    if per_task:
        lines += [
            "",
            "## Per-task results",
            "",
            "| Task ID | Score | Reason |",
            "|---------|-------|--------|",
        ]
        for t in per_task:
            reason = t.get("reason", "").replace("|", "\\|")
            lines.append(f"| {t['task_id']} | {t['score']} | {reason} |")

    if notes:
        lines += ["", "## Notes", "", notes]

    lines.append("")
    return "\n".join(lines)


# ── Compare subcommand ────────────────────────────────────────────────────────


def compare(dir_a: Path, dir_b: Path, key: str = "tool_call_success_rate") -> None:
    """Print a side-by-side comparison of two result directories."""

    def _load(d: Path) -> dict[str, Any]:
        p = d / "metrics.json"
        if not p.exists():
            console.print(f"[red]metrics.json not found in {d}[/red]")
            sys.exit(1)
        return json.loads(p.read_text())

    a = _load(dir_a)
    b = _load(dir_b)

    table = Table(title=f"Comparison: {key}")
    table.add_column("Metric", style="bold")
    table.add_column(dir_a.name, justify="right")
    table.add_column(dir_b.name, justify="right")
    table.add_column("Δ (B−A)", justify="right")

    a_metrics: dict[str, Any] = a.get("metrics", {})
    b_metrics: dict[str, Any] = b.get("metrics", {})

    all_keys = sorted(set(a_metrics) | set(b_metrics))
    for k in all_keys:
        av = a_metrics.get(k)
        bv = b_metrics.get(k)
        if not isinstance(av, (int, float)) or not isinstance(bv, (int, float)):
            continue
        delta = bv - av
        delta_str = f"{delta:+.4g}"
        color = "green" if delta >= 0 else "red"
        table.add_row(k, f"{av:.4g}", f"{bv:.4g}", f"[{color}]{delta_str}[/{color}]")

    # Highlight the requested key
    console.print(table)
    for k in (key, "verdict"):
        av2 = a_metrics.get(k) if k != "verdict" else a.get("verdict")
        bv2 = b_metrics.get(k) if k != "verdict" else b.get("verdict")
        if av2 is not None or bv2 is not None:
            console.print(f"[bold]{k}:[/bold] {dir_a.name}={av2}  {dir_b.name}={bv2}")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Report or compare benchmark results")
    subparsers = parser.add_subparsers(dest="cmd")

    # Default: generate summary for a results dir
    parser.add_argument("results_dir", nargs="?", help="Results directory containing metrics.json")
    parser.add_argument(
        "--thresholds",
        default="config/thresholds.yaml",
        help="Path to thresholds.yaml",
    )

    # compare subcommand
    sub = subparsers.add_parser("compare", help="Compare two result directories")
    sub.add_argument("dir_a")
    sub.add_argument("dir_b")
    sub.add_argument("--key", default="tool_call_success_rate")

    args = parser.parse_args(argv)

    if args.cmd == "compare":
        compare(Path(args.dir_a), Path(args.dir_b), args.key)
        return

    if not args.results_dir:
        parser.print_help()
        sys.exit(1)

    thresholds_path = Path(args.thresholds)
    thresholds: dict[str, Any] = {}
    if thresholds_path.exists():
        thresholds = yaml.safe_load(thresholds_path.read_text()) or {}

    results_dir = Path(args.results_dir)
    summary = generate_summary(results_dir, thresholds)

    summary_path = results_dir / "summary.md"
    summary_path.write_text(summary)
    console.print(summary)


if __name__ == "__main__":
    main()
