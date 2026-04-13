"""Tests for lib.scorer — unit tests that run without a live endpoint."""

from __future__ import annotations

import pytest

from lib.metrics import BenchResult
from lib.scorer import (
    ScoredResult,
    ToolCallScore,
    _detect_dropped_or_no_call,
    score_tool_call,
    summarise_scores,
)


def _make_result(
    task_id: str = "t1",
    tool_calls: list | None = None,
    raw_text: str = "",
    error: str = "",
) -> BenchResult:
    r = BenchResult(task_id=task_id)
    r.tool_calls = tool_calls or []
    r.raw_text = raw_text
    r.error = error
    return r


def _make_tool_call(name: str, args: dict | None = None) -> dict:
    import json
    return {
        "id": "call_0",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args or {}),
        },
    }


# ── score_tool_call ────────────────────────────────────────────────────────────

class TestScoreToolCall:
    def test_pass_correct_call_correct_args(self):
        result = _make_result(tool_calls=[_make_tool_call("read_file", {"path": "/tmp/f"})])
        sr = score_tool_call(result, [{"name": "read_file", "must_include_args": ["path"]}])
        assert sr.score == ToolCallScore.PASS
        assert sr.passed

    def test_exception_on_error(self):
        result = _make_result(error="Connection refused")
        sr = score_tool_call(result, [{"name": "read_file", "must_include_args": ["path"]}])
        assert sr.score == ToolCallScore.EXCEPTION
        assert sr.reason == "Connection refused"

    def test_wrong_tool(self):
        result = _make_result(tool_calls=[_make_tool_call("write_file", {"path": "/x"})])
        sr = score_tool_call(result, [{"name": "read_file", "must_include_args": ["path"]}])
        assert sr.score == ToolCallScore.WRONG_TOOL

    def test_wrong_args_missing_key(self):
        result = _make_result(tool_calls=[_make_tool_call("read_file", {})])
        sr = score_tool_call(result, [{"name": "read_file", "must_include_args": ["path"]}])
        assert sr.score == ToolCallScore.WRONG_ARGS

    def test_wrong_args_empty_value(self):
        result = _make_result(tool_calls=[_make_tool_call("read_file", {"path": ""})])
        sr = score_tool_call(result, [{"name": "read_file", "must_include_args": ["path"]}])
        assert sr.score == ToolCallScore.WRONG_ARGS

    def test_format_error_bad_json(self):
        result = _make_result(tool_calls=[{
            "id": "call_0",
            "type": "function",
            "function": {"name": "read_file", "arguments": "not-json"},
        }])
        sr = score_tool_call(result, [{"name": "read_file", "must_include_args": ["path"]}])
        assert sr.score == ToolCallScore.FORMAT_ERROR
        assert sr.is_engine_error

    def test_no_call_when_text_only(self):
        result = _make_result(raw_text="Sure, let me help you with that.")
        sr = score_tool_call(result, [{"name": "read_file", "must_include_args": ["path"]}])
        assert sr.score == ToolCallScore.NO_CALL

    def test_dropped_when_tool_call_json_in_text(self):
        raw = '{"function": {"name": "read_file"}, "arguments": {}}'
        result = _make_result(raw_text=raw)
        sr = score_tool_call(result, [{"name": "read_file", "must_include_args": ["path"]}])
        assert sr.score == ToolCallScore.DROPPED
        assert sr.is_engine_error

    def test_pass_no_expected_calls(self):
        result = _make_result(raw_text="Here is your answer.")
        sr = score_tool_call(result, [])
        assert sr.score == ToolCallScore.PASS

    def test_pass_multiple_must_include_args(self):
        result = _make_result(
            tool_calls=[_make_tool_call("write_file", {"path": "/x", "content": "hello"})]
        )
        sr = score_tool_call(
            result, [{"name": "write_file", "must_include_args": ["path", "content"]}]
        )
        assert sr.score == ToolCallScore.PASS


# ── _detect_dropped_or_no_call ────────────────────────────────────────────────

class TestDetectDropped:
    def test_detects_tool_call_tag(self):
        assert _detect_dropped_or_no_call("<tool_call>...") == ToolCallScore.DROPPED

    def test_detects_function_key(self):
        assert _detect_dropped_or_no_call('{"function": "x"}') == ToolCallScore.DROPPED

    def test_detects_name_key(self):
        assert _detect_dropped_or_no_call('"name": "read_file"') == ToolCallScore.DROPPED

    def test_detects_code_fence(self):
        assert _detect_dropped_or_no_call("```json\n{...}") == ToolCallScore.DROPPED

    def test_plain_text_is_no_call(self):
        assert _detect_dropped_or_no_call("Here is the answer.") == ToolCallScore.NO_CALL


# ── summarise_scores ──────────────────────────────────────────────────────────

class TestSummariseScores:
    def _make_sr(self, score: str, task_id: str = "t1") -> ScoredResult:
        return ScoredResult(task_id=task_id, score=score)

    def test_empty(self):
        assert summarise_scores([]) == {}

    def test_all_pass(self):
        scored = [self._make_sr(ToolCallScore.PASS, f"t{i}") for i in range(10)]
        out = summarise_scores(scored)
        assert out["tool_call_success_rate"] == 1.0
        assert out["critical_error_rate"] == 0.0

    def test_mixed(self):
        scored = [
            self._make_sr(ToolCallScore.PASS, "t1"),
            self._make_sr(ToolCallScore.PASS, "t2"),
            self._make_sr(ToolCallScore.PASS, "t3"),
            self._make_sr(ToolCallScore.DROPPED, "t4"),  # engine error
            self._make_sr(ToolCallScore.WRONG_TOOL, "t5"),
        ]
        out = summarise_scores(scored)
        assert out["total"] == 5
        assert out["tool_call_success_rate"] == pytest.approx(3 / 5)
        assert out["critical_error_rate"] == pytest.approx(1 / 5)
        assert out["counts"][ToolCallScore.PASS] == 3
        assert out["counts"][ToolCallScore.DROPPED] == 1

    def test_per_task_included(self):
        scored = [self._make_sr(ToolCallScore.PASS, "task_01")]
        out = summarise_scores(scored)
        assert len(out["per_task"]) == 1
        assert out["per_task"][0]["task_id"] == "task_01"
        assert out["per_task"][0]["score"] == ToolCallScore.PASS
