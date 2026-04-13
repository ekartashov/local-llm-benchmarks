"""Tests for verify_chat_template probes — no live endpoint needed."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from lib.metrics import BenchResult
from lib.scorer import ToolCallScore
from benchmarks.phase0_tool_reliability.verify_chat_template import (
    ProbeResult,
    _score_probe,
    probe_basic_tool_call,
    probe_multi_turn_tool_call,
    probe_think_then_tool,
    probe_special_char_args,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(
    tool_calls: list | None = None,
    raw_text: str = "",
    error: str = "",
    task_id: str = "probe_test",
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
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }


# ── _score_probe ──────────────────────────────────────────────────────────────

class TestScoreProbe:
    def test_pass_correct_tool(self):
        result = _make_result(tool_calls=[_make_tool_call("read_file", {"path": "/x"})])
        pr = _score_probe("test", result, expected_name="read_file")
        assert pr.passed
        assert pr.score == ToolCallScore.PASS

    def test_fail_wrong_tool(self):
        result = _make_result(tool_calls=[_make_tool_call("write_file", {"path": "/x"})])
        pr = _score_probe("test", result, expected_name="read_file", remediation="fix it")
        assert not pr.passed
        assert pr.score == ToolCallScore.WRONG_TOOL
        assert pr.remediation == "fix it"

    def test_fail_exception(self):
        result = _make_result(error="Connection refused")
        pr = _score_probe("test", result, expected_name="read_file")
        assert not pr.passed
        assert pr.score == ToolCallScore.EXCEPTION
        assert "Connection refused" in pr.reason

    def test_fail_no_tool_calls(self):
        result = _make_result(raw_text="Sure, let me help!")
        pr = _score_probe("test", result, expected_name="read_file")
        assert not pr.passed
        assert pr.score in (ToolCallScore.NO_CALL, ToolCallScore.DROPPED)

    def test_dropped_when_tool_json_in_text(self):
        result = _make_result(raw_text='"function": "read_file"')
        pr = _score_probe("test", result, expected_name="read_file")
        assert not pr.passed
        assert pr.score == ToolCallScore.DROPPED


# ── probe_basic_tool_call ─────────────────────────────────────────────────────

class TestProbeBasicToolCall:
    @pytest.mark.asyncio
    async def test_pass_when_tool_call_returned(self):
        mock_result = _make_result(tool_calls=[_make_tool_call("read_file", {"path": "/workspace/README.md"})])
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_result)
        pr = await probe_basic_tool_call(mock_client, model="test-model")
        assert pr.passed
        assert pr.name == "basic_tool_call"

    @pytest.mark.asyncio
    async def test_fail_when_no_tool_call(self):
        mock_result = _make_result(raw_text="I can't use tools right now.")
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_result)
        pr = await probe_basic_tool_call(mock_client, model="test-model")
        assert not pr.passed
        assert pr.name == "basic_tool_call"

    @pytest.mark.asyncio
    async def test_fail_on_exception(self):
        mock_result = _make_result(error="timeout")
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_result)
        pr = await probe_basic_tool_call(mock_client, model="test-model")
        assert not pr.passed
        assert pr.score == ToolCallScore.EXCEPTION


# ── probe_think_then_tool ─────────────────────────────────────────────────────

class TestProbeThinkThenTool:
    @pytest.mark.asyncio
    async def test_pass_when_tool_call_after_thinking(self):
        mock_result = _make_result(
            tool_calls=[_make_tool_call("read_file", {"path": "/workspace/config/models.yaml"})],
            raw_text="<think>Let me think...</think>",
        )
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_result)
        pr = await probe_think_then_tool(mock_client, model="test-model")
        assert pr.passed
        assert pr.score == ToolCallScore.PASS

    @pytest.mark.asyncio
    async def test_fail_think_drop_bug(self):
        """Simulates vLLM PR #39055: <think> present but no tool call delivered."""
        mock_result = _make_result(
            tool_calls=[],
            raw_text="<think>I need to read models.yaml</think>",
        )
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_result)
        pr = await probe_think_then_tool(mock_client, model="test-model")
        assert not pr.passed
        assert "PR #39055" in pr.reason
        assert "--reasoning-parser qwen3" in pr.remediation

    @pytest.mark.asyncio
    async def test_fail_exception(self):
        mock_result = _make_result(error="500 Internal Server Error")
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_result)
        pr = await probe_think_then_tool(mock_client, model="test-model")
        assert not pr.passed
        assert pr.score == ToolCallScore.EXCEPTION


# ── probe_special_char_args ───────────────────────────────────────────────────

class TestProbeSpecialCharArgs:
    @pytest.mark.asyncio
    async def test_pass_valid_unicode_args(self):
        import json
        mock_result = _make_result(
            tool_calls=[{
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"path": "/tmp/tëst.txt", "content": "héllo"}),
                },
            }]
        )
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_result)
        pr = await probe_special_char_args(mock_client, model="test-model")
        assert pr.passed

    @pytest.mark.asyncio
    async def test_fail_malformed_json_args(self):
        mock_result = _make_result(
            tool_calls=[{
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "/tmp/tëst.txt", "content": héllo}',  # invalid JSON
                },
            }]
        )
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_result)
        pr = await probe_special_char_args(mock_client, model="test-model")
        assert not pr.passed
        assert pr.score == ToolCallScore.FORMAT_ERROR
        assert "PR #35347" in pr.remediation


# ── ProbeResult ───────────────────────────────────────────────────────────────

class TestProbeResult:
    def test_defaults(self):
        pr = ProbeResult(name="test", passed=True)
        assert pr.score == ""
        assert pr.reason == ""
        assert pr.remediation == ""
        assert pr.raw_tool_calls == []
        assert pr.raw_text == ""

    def test_passed_flag(self):
        assert ProbeResult(name="x", passed=True).passed
        assert not ProbeResult(name="x", passed=False).passed
