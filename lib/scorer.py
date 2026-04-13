"""Tool-call scoring logic for Phase 0 and any phase that validates tool use."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from lib.metrics import BenchResult


class ToolCallScore:
    PASS = "pass"                # Correct tool call, parsed successfully
    FORMAT_ERROR = "format_error"  # Engine couldn't parse the model's output
    DROPPED = "dropped"          # Engine silently returned text-only (no tool call at all)
    WRONG_TOOL = "wrong_tool"    # Parsed but wrong tool name
    WRONG_ARGS = "wrong_args"    # Right tool, missing or wrong arguments
    NO_CALL = "no_call"          # Model made no tool call attempt (plain text reply)
    EXCEPTION = "exception"      # Request-level error (timeout, 4xx, 5xx)


@dataclass
class ScoredResult:
    task_id: str
    score: str                       # one of ToolCallScore constants
    reason: str = ""
    bench_result: BenchResult | None = None

    @property
    def passed(self) -> bool:
        return self.score == ToolCallScore.PASS

    @property
    def is_engine_error(self) -> bool:
        """DROPPED or FORMAT_ERROR — indicates a broken engine parser, not a model failure."""
        return self.score in (ToolCallScore.DROPPED, ToolCallScore.FORMAT_ERROR)


def score_tool_call(
    result: BenchResult,
    expected_tool_calls: list[dict[str, Any]],
) -> ScoredResult:
    """
    Score a single BenchResult against the expected tool-call spec.

    expected_tool_calls format (from task JSON):
        [{"name": "read_file", "must_include_args": ["path"]}]
    """
    if not result.ok:
        return ScoredResult(
            task_id=result.task_id,
            score=ToolCallScore.EXCEPTION,
            reason=result.error,
            bench_result=result,
        )

    if not expected_tool_calls:
        # No tool call expected — treat any response as PASS
        return ScoredResult(task_id=result.task_id, score=ToolCallScore.PASS, bench_result=result)

    # Check if the model produced any tool calls
    if not result.tool_calls:
        # Distinguish DROPPED (engine swallowed it) from NO_CALL (model didn't try)
        score = _detect_dropped_or_no_call(result.raw_text)
        return ScoredResult(
            task_id=result.task_id,
            score=score,
            reason="No tool calls in response",
            bench_result=result,
        )

    # Validate each expected call against what we received
    for expected in expected_tool_calls:
        exp_name = expected.get("name", "")
        must_args: list[str] = expected.get("must_include_args", [])

        # Find a matching tool call by name
        matching = [tc for tc in result.tool_calls if tc.get("function", {}).get("name") == exp_name]

        if not matching:
            called_names = [tc.get("function", {}).get("name") for tc in result.tool_calls]
            return ScoredResult(
                task_id=result.task_id,
                score=ToolCallScore.WRONG_TOOL,
                reason=f"Expected '{exp_name}', got {called_names}",
                bench_result=result,
            )

        tc = matching[0]
        raw_args: str = tc.get("function", {}).get("arguments", "")

        # Try to parse arguments JSON
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            return ScoredResult(
                task_id=result.task_id,
                score=ToolCallScore.FORMAT_ERROR,
                reason=f"Could not parse tool arguments JSON: {raw_args!r}",
                bench_result=result,
            )

        # Check required argument keys are present and non-empty
        for arg_key in must_args:
            if arg_key not in args or not args[arg_key]:
                return ScoredResult(
                    task_id=result.task_id,
                    score=ToolCallScore.WRONG_ARGS,
                    reason=f"Missing or empty required arg '{arg_key}' in {args}",
                    bench_result=result,
                )

    return ScoredResult(task_id=result.task_id, score=ToolCallScore.PASS, bench_result=result)


def _detect_dropped_or_no_call(raw_text: str) -> str:
    """
    Heuristic: if the raw text contains what looks like a mangled tool-call
    (e.g. JSON blobs with 'function' keys, <tool_call> tags) then the engine
    dropped/failed to parse it. Otherwise the model simply didn't call a tool.
    """
    patterns = [
        r"<tool_call>",
        r'"function"\s*:',
        r'"name"\s*:\s*"[a-z_]+"',
        r"```json",
    ]
    for pattern in patterns:
        if re.search(pattern, raw_text, re.IGNORECASE):
            return ToolCallScore.DROPPED
    return ToolCallScore.NO_CALL


def summarise_scores(scored: list[ScoredResult]) -> dict[str, Any]:
    """Aggregate scored results into a metrics dict suitable for metrics.json."""
    total = len(scored)
    if total == 0:
        return {}

    counts: dict[str, int] = {}
    for s in scored:
        counts[s.score] = counts.get(s.score, 0) + 1

    passed = counts.get(ToolCallScore.PASS, 0)
    engine_errors = counts.get(ToolCallScore.DROPPED, 0) + counts.get(ToolCallScore.FORMAT_ERROR, 0)

    return {
        "total": total,
        "counts": counts,
        "tool_call_success_rate": passed / total,
        "critical_error_rate": engine_errors / total,
        "per_task": [
            {"task_id": s.task_id, "score": s.score, "reason": s.reason} for s in scored
        ],
    }
