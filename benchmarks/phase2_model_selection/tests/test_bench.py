"""Tests for Phase 2 bench — quality, tool-reliability, and spec-decode modes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.metrics import BenchResult, PhaseMetrics
from benchmarks.phase2_model_selection.bench import (
    _write_human_review,
    load_quality_tasks,
    run_quality,
    run_spec_decode,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(task_id: str = "t1", text: str = "def foo(): pass",
                 ttft_ms: float = 150.0, decode_tps: float = 180.0,
                 error: str = "") -> BenchResult:
    r = BenchResult(task_id=task_id)
    r.raw_text = text
    r.ttft_ms = ttft_ms
    r.decode_tps = decode_tps
    r.latency_ms = 800.0
    r.prompt_tokens = 100
    r.completion_tokens = 80
    r.error = error
    return r


def _write_task(d: Path, task_id: str, fname: str | None = None) -> None:
    data = {
        "id": task_id,
        "description": f"Test task {task_id}",
        "system_prompt": "You are helpful.",
        "user_message": "Write some code.",
    }
    (d / (fname or f"{task_id}.json")).write_text(json.dumps(data))


# ── load_quality_tasks ────────────────────────────────────────────────────────

class TestLoadQualityTasks:
    def test_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_task(d, "q01")
            _write_task(d, "q02", "q02_task.json")
            tasks = load_quality_tasks(d)
            assert len(tasks) == 2
            assert all("id" in t for t in tasks)

    def test_sorted_alphabetically(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_task(d, "z", "z.json")
            _write_task(d, "a", "a.json")
            tasks = load_quality_tasks(d)
            assert tasks[0]["id"] == "a"

    def test_empty_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                load_quality_tasks(Path(td))

    def test_missing_field_raises(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "bad.json").write_text(json.dumps({"id": "x", "system_prompt": "s"}))
            with pytest.raises(ValueError, match="user_message"):
                load_quality_tasks(d)


# ── run_quality ───────────────────────────────────────────────────────────────

class TestRunQuality:
    @pytest.fixture
    def tasks(self):
        return [
            {"id": "q01", "description": "Task 1", "system_prompt": "sys", "user_message": "msg1"},
            {"id": "q02", "description": "Task 2", "system_prompt": "sys", "user_message": "msg2"},
        ]

    @pytest.mark.asyncio
    async def test_returns_metrics_and_outputs(self, tasks, tmp_path):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=[
            _make_result("q01", text="def foo(): pass"),
            _make_result("q02", text="class Bar: pass"),
        ])
        pm, outputs = await run_quality(mock_client, tasks, model="test",
                                        max_tokens=512, label="TestModel")
        assert isinstance(pm, PhaseMetrics)
        assert len(outputs) == 2
        assert outputs[0]["id"] == "q01"
        assert outputs[0]["label"] == "TestModel"
        assert outputs[0]["completed"] is True

    @pytest.mark.asyncio
    async def test_empty_response_not_completed(self, tasks, tmp_path):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=[
            _make_result("q01", text=""),
            _make_result("q02", text="some code"),
        ])
        _, outputs = await run_quality(mock_client, tasks, model="test",
                                       max_tokens=512, label="TestModel")
        assert outputs[0]["completed"] is False
        assert outputs[1]["completed"] is True

    @pytest.mark.asyncio
    async def test_error_result_not_completed(self, tasks, tmp_path):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=_make_result("q01", text="", error="500"))
        _, outputs = await run_quality(mock_client, [tasks[0]], model="test",
                                       max_tokens=512, label="TestModel")
        assert outputs[0]["completed"] is False
        assert outputs[0]["error"] == "500"


# ── _write_human_review ───────────────────────────────────────────────────────

class TestWriteHumanReview:
    def test_creates_markdown_file(self, tmp_path):
        outputs = [
            {"id": "q01", "description": "Fix bug", "label": "ModelA",
             "user_message": "Here is the code", "response": "Fixed code",
             "completed": True, "ttft_ms": 150.0, "decode_tps": 200.0, "error": ""},
            {"id": "q02", "description": "Write tests", "label": "ModelA",
             "user_message": "Write tests for foo()", "response": "",
             "completed": False, "ttft_ms": 0.0, "decode_tps": 0.0, "error": "timeout"},
        ]
        _write_human_review(outputs, tmp_path, "ModelA")
        review_path = tmp_path / "human_review.md"
        assert review_path.exists()
        content = review_path.read_text()
        assert "ModelA" in content
        assert "q01" in content
        assert "q02" in content
        assert "Fixed code" in content
        assert "[ERROR: timeout]" in content

    def test_summary_table_included(self, tmp_path):
        outputs = [
            {"id": "t01", "description": "", "label": "M",
             "user_message": "q", "response": "a",
             "completed": True, "ttft_ms": 100.0, "decode_tps": 150.0, "error": ""},
        ]
        _write_human_review(outputs, tmp_path, "M")
        content = (tmp_path / "human_review.md").read_text()
        assert "| t01 |" in content
        assert "Quality (fill in)" in content


# ── run_spec_decode ───────────────────────────────────────────────────────────

class TestRunSpecDecode:
    @pytest.mark.asyncio
    async def test_returns_phase_metrics(self, tmp_path):
        tasks = [
            {"id": "s01", "system_prompt": "sys", "user_message": "msg"},
            {"id": "s02", "system_prompt": "sys", "user_message": "msg"},
        ]
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=[
            _make_result("s01", decode_tps=220),
            _make_result("s02", decode_tps=240),
        ])
        pm = await run_spec_decode(mock_client, tasks, model="test", max_tokens=256)
        assert isinstance(pm, PhaseMetrics)
        assert len(pm.results) == 2
        assert pm.decode_tps_mean() == pytest.approx(230.0)
