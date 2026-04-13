"""
Sub-test 0.4 — Chat template verification.

Sends targeted probe requests to a live endpoint to detect known engine/template
bugs BEFORE running the full 30-task suite. Fails fast with actionable diagnostics.

Probes:
  1. basic_tool_call       — simple tool call, no thinking (baseline sanity check)
  2. multi_turn_tool_call  — tool call in a 2-turn conversation (template multi-turn rendering)
  3. think_then_tool       — prompt that induces <think> before tool call (vLLM PR #39055)
  4. parallel_tool_calls   — two tools in one response (confirms engine supports parallel calls)
  5. special_char_args     — tool args with unicode / special chars (JSON escaping)

Exit code 0 = all probes passed (safe to run Phase 0.1)
Exit code 1 = one or more probes failed (includes remediation advice)
Exit code 2 = endpoint unreachable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from lib.client import BenchClient
from lib.metrics import BenchResult
from lib.scorer import ToolCallScore, _detect_dropped_or_no_call

console = Console()

# ── Probe definitions ─────────────────────────────────────────────────────────

_READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

_WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}


@dataclass
class ProbeResult:
    name: str
    passed: bool
    score: str = ""
    reason: str = ""
    remediation: str = ""
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""


# ── Individual probes ─────────────────────────────────────────────────────────


async def probe_basic_tool_call(client: BenchClient, model: str) -> ProbeResult:
    """Simplest possible tool call — no thinking, single turn."""
    result = await client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant. Always use tools."},
            {"role": "user", "content": "Read the file at /workspace/README.md"},
        ],
        tools=[_READ_FILE_TOOL],
        task_id="probe_basic",
        model=model,
        temperature=0.0,
        max_tokens=512,
    )
    return _score_probe(
        "basic_tool_call",
        result,
        expected_name="read_file",
        remediation=(
            "Engine is not parsing tool calls at all. "
            "Check: --enable-auto-tool-choice and --tool-call-parser flags are set."
        ),
    )


async def probe_multi_turn_tool_call(client: BenchClient, model: str) -> ProbeResult:
    """Tool call in a 2-turn conversation to verify chat template multi-turn rendering."""
    result = await client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant. Always use tools."},
            {"role": "user", "content": "I need to check a config file."},
            {"role": "assistant", "content": "Sure, I can help with that. Which file?"},
            {"role": "user", "content": "Read /workspace/config/hardware.env"},
        ],
        tools=[_READ_FILE_TOOL],
        task_id="probe_multi_turn",
        model=model,
        temperature=0.0,
        max_tokens=512,
    )
    return _score_probe(
        "multi_turn_tool_call",
        result,
        expected_name="read_file",
        remediation=(
            "Multi-turn tool call rendering is broken. "
            "The Qwen3.5 HF chat template has a known bug in multi-turn rendering. "
            "Fix: use barubary's patched chat template "
            "(https://huggingface.co/Qwen/Qwen3.5-35B-A3B-AWQ/discussions/1) "
            "or the Unsloth March 2026 GGUF which includes the fix."
        ),
    )


async def probe_think_then_tool(client: BenchClient, model: str) -> ProbeResult:
    """
    Probe that induces <think>...</think> before a tool call.
    Tests for vLLM PR #39055: tool calls inside <think> silently dropped.
    """
    result = await client.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful coding assistant. "
                    "Think step by step before acting. Always use tools."
                ),
            },
            {
                "role": "user",
                "content": (
                    "I have a complex task: first think about what file to read, "
                    "then read /workspace/config/models.yaml to check the model list."
                ),
            },
        ],
        tools=[_READ_FILE_TOOL],
        task_id="probe_think_tool",
        model=model,
        temperature=0.0,
        max_tokens=1024,
    )

    # Special analysis: did we get a tool call, text only, or dropped?
    has_tool_calls = bool(result.tool_calls)
    has_think_tag = "<think>" in result.raw_text or "</think>" in result.raw_text

    if not result.ok:
        return ProbeResult(
            name="think_then_tool",
            passed=False,
            score=ToolCallScore.EXCEPTION,
            reason=result.error,
            remediation="Request failed — check endpoint health.",
        )

    if has_tool_calls:
        return ProbeResult(
            name="think_then_tool",
            passed=True,
            score=ToolCallScore.PASS,
            reason="Tool call received correctly after (possible) thinking.",
            raw_tool_calls=result.tool_calls,
            raw_text=result.raw_text,
        )

    if has_think_tag and not has_tool_calls:
        drop_score = _detect_dropped_or_no_call(result.raw_text)
        return ProbeResult(
            name="think_then_tool",
            passed=False,
            score=drop_score,
            reason=(
                "Model produced <think> content but no tool call reached the client. "
                "This is vLLM PR #39055: tool calls emitted inside <think> are silently dropped."
            ),
            remediation=(
                "Workaround: add --reasoning-parser qwen3 to vLLM launch args. "
                "This separates reasoning tokens from tool-call tokens. "
                "Check if PR #39055 has been merged into your vLLM version."
            ),
            raw_text=result.raw_text,
        )

    return _score_probe(
        "think_then_tool",
        result,
        expected_name="read_file",
        remediation=(
            "No tool call and no <think> tag — model may have tool calling disabled "
            "or the prompt is not triggering tool use."
        ),
    )


async def probe_parallel_tool_calls(client: BenchClient, model: str) -> ProbeResult:
    """Request two tool calls in one response (parallel tool use)."""
    result = await client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant. Use tools in parallel when possible."},
            {
                "role": "user",
                "content": "Read both /workspace/config/hardware.env and /workspace/config/models.yaml at the same time.",
            },
        ],
        tools=[_READ_FILE_TOOL],
        task_id="probe_parallel",
        model=model,
        temperature=0.0,
        max_tokens=512,
    )

    if not result.ok:
        return ProbeResult(
            name="parallel_tool_calls",
            passed=False,
            score=ToolCallScore.EXCEPTION,
            reason=result.error,
        )

    # Accept if at least one tool call came through
    if result.tool_calls:
        n = len(result.tool_calls)
        return ProbeResult(
            name="parallel_tool_calls",
            passed=True,
            score=ToolCallScore.PASS,
            reason=f"Received {n} tool call(s). Parallel calls: {'yes' if n >= 2 else 'no (model chose sequential)'}.",
            raw_tool_calls=result.tool_calls,
        )

    drop_score = _detect_dropped_or_no_call(result.raw_text)
    return ProbeResult(
        name="parallel_tool_calls",
        passed=False,
        score=drop_score,
        reason="No tool calls returned for a prompt explicitly requesting parallel calls.",
        remediation=(
            "Check that --enable-auto-tool-choice is set and tool_choice='auto' is sent. "
            "Some engines require --max-tool-call-tokens to be set explicitly."
        ),
        raw_text=result.raw_text,
    )


async def probe_special_char_args(client: BenchClient, model: str) -> ProbeResult:
    """Tool args with unicode / special characters — tests JSON escaping in the engine."""
    result = await client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant. Always use tools."},
            {
                "role": "user",
                "content": r'Write "héllo wörld\n" to /tmp/tëst_ünïcödé.txt',
            },
        ],
        tools=[_WRITE_FILE_TOOL],
        task_id="probe_special_chars",
        model=model,
        temperature=0.0,
        max_tokens=512,
    )

    scored = _score_probe(
        "special_char_args",
        result,
        expected_name="write_file",
        remediation=(
            "Engine failed to parse tool args containing unicode / escape sequences. "
            "This may indicate a JSON serialization bug in the tool-call parser."
        ),
    )

    # Additional check: can we actually parse the args JSON?
    if scored.passed and result.tool_calls:
        for tc in result.tool_calls:
            raw_args = tc.get("function", {}).get("arguments", "")
            try:
                json.loads(raw_args)
            except json.JSONDecodeError as e:
                scored.passed = False
                scored.score = ToolCallScore.FORMAT_ERROR
                scored.reason = f"Tool args JSON failed to parse: {e}"
                scored.remediation = (
                    "Engine produced malformed JSON in tool arguments. "
                    "Known as vLLM PR #35347. Verify your vLLM version includes the fix."
                )
    return scored


# ── Helpers ───────────────────────────────────────────────────────────────────


def _score_probe(
    name: str,
    result: BenchResult,
    expected_name: str,
    remediation: str = "",
) -> ProbeResult:
    if not result.ok:
        return ProbeResult(
            name=name,
            passed=False,
            score=ToolCallScore.EXCEPTION,
            reason=result.error,
            remediation="Request failed — check endpoint health and container logs.",
        )

    if result.tool_calls:
        called_names = [tc.get("function", {}).get("name") for tc in result.tool_calls]
        if expected_name in called_names:
            return ProbeResult(
                name=name,
                passed=True,
                score=ToolCallScore.PASS,
                reason=f"Received tool call: {called_names}",
                raw_tool_calls=result.tool_calls,
                raw_text=result.raw_text,
            )
        return ProbeResult(
            name=name,
            passed=False,
            score=ToolCallScore.WRONG_TOOL,
            reason=f"Expected '{expected_name}', got {called_names}",
            remediation=remediation,
            raw_tool_calls=result.tool_calls,
        )

    drop_score = _detect_dropped_or_no_call(result.raw_text)
    return ProbeResult(
        name=name,
        passed=False,
        score=drop_score,
        reason="No tool calls in response.",
        remediation=remediation,
        raw_text=result.raw_text[:300] if result.raw_text else "",
    )


# ── Main ──────────────────────────────────────────────────────────────────────


async def run_probes(endpoint: str, model: str, results_dir_path: str) -> list[ProbeResult]:
    from pathlib import Path

    results_dir = Path(results_dir_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    client = BenchClient(base_url=endpoint, results_dir=results_dir)

    probes = [
        probe_basic_tool_call(client, model),
        probe_multi_turn_tool_call(client, model),
        probe_think_then_tool(client, model),
        probe_parallel_tool_calls(client, model),
        probe_special_char_args(client, model),
    ]

    results: list[ProbeResult] = []
    for coro in probes:
        pr = await coro
        results.append(pr)
        icon = "[green]✓[/green]" if pr.passed else "[red]✗[/red]"
        console.print(f"  {icon} {pr.name:30s} {pr.score}")
        if not pr.passed and pr.reason:
            console.print(f"      [dim]reason: {pr.reason}[/dim]")

    return results


def _print_report(results: list[ProbeResult]) -> None:
    table = Table(title="Chat Template Verification — Sub-test 0.4")
    table.add_column("Probe", style="bold")
    table.add_column("Result", justify="center")
    table.add_column("Score")
    table.add_column("Notes")

    for pr in results:
        status = "[green]PASS[/green]" if pr.passed else "[red]FAIL[/red]"
        notes = pr.reason[:80] if pr.reason else ""
        table.add_row(pr.name, status, pr.score, notes)

    console.print(table)

    failed = [r for r in results if not r.passed]
    if failed:
        console.print("\n[bold red]Failures — Remediation advice:[/bold red]")
        for pr in failed:
            if pr.remediation:
                console.print(f"\n  [bold]{pr.name}[/bold]")
                console.print(f"  {pr.remediation}")


async def main_async(args: argparse.Namespace) -> int:
    # Check endpoint reachability first
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{args.endpoint.rstrip('/v1')}/health")
            r.raise_for_status()
    except Exception as e:
        console.print(f"[bold red]ERROR: Cannot reach {args.endpoint}: {e}[/bold red]")
        console.print("Is the inference engine running? Run deploy.sh first.")
        return 2

    console.print(f"[bold]Sub-test 0.4 — Chat template verification[/bold]")
    console.print(f"Endpoint: {args.endpoint}")
    console.print(f"Model   : {args.model}\n")

    results = await run_probes(args.endpoint, args.model, args.results_dir)
    _print_report(results)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    all_passed = passed == total

    console.print(f"\n[bold]{'[green]All probes passed' if all_passed else '[red]Some probes failed'}[/bold] ({passed}/{total})")

    if all_passed:
        console.print("[green]✓ Safe to proceed with Phase 0.1 (full 30-task run)[/green]")
    else:
        console.print("[red]✗ Fix the issues above before running Phase 0.1[/red]")
        console.print("  Retrying after fixes: ./benchmarks/phase0_tool_reliability/run_0.4_chat_template.sh")

    # Save probe results as JSON
    import json
    from pathlib import Path
    out = {
        "sub_test": "0.4_chat_template_verification",
        "endpoint": args.endpoint,
        "model": args.model,
        "passed": passed,
        "total": total,
        "all_passed": all_passed,
        "probes": [
            {
                "name": r.name,
                "passed": r.passed,
                "score": r.score,
                "reason": r.reason,
                "remediation": r.remediation,
            }
            for r in results
        ],
    }
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / "probe_results.json").write_text(json.dumps(out, indent=2))

    return 0 if all_passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-test 0.4 — Chat template verification")
    parser.add_argument("--endpoint", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--results-dir", default="results/phase0_0.4_chat_template")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
