"""Tests for Phase 1 bench — throughput and prefix-cache modes without a live endpoint."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lib.metrics import BenchResult, PhaseMetrics
from benchmarks.phase1_engine_selection.bench import (
    load_generation_tasks,
    run_prefix_cache,
    run_throughput,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(task_id: str, ttft_ms: float = 100.0, decode_tps: float = 200.0,
                 latency_ms: float = 500.0, error: str = "") -> BenchResult:
    r = BenchResult(task_id=task_id)
    r.ttft_ms = ttft_ms
    r.decode_tps = decode_tps
    r.latency_ms = latency_ms
    r.prompt_tokens = 200
    r.completion_tokens = 100
    r.raw_text = "def foo(): pass"
    r.error = error
    return r


def _write_task(d: Path, task_id: str, filename: str | None = None) -> None:
    data = {
        "id": task_id,
        "system_prompt": "You are a helpful assistant.",
        "user_message": "Write a hello world function.",
    }
    fname = filename or f"{task_id}.json"
    (d / fname).write_text(json.dumps(data))


# ── load_generation_tasks ─────────────────────────────────────────────────────

class TestLoadGenerationTasks:
    def test_loads_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_task(d, "t01")
            _write_task(d, "t02", "02_task.json")
            tasks = load_generation_tasks(d)
            assert len(tasks) == 2

    def test_sorted_by_filename(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_task(d, "z_task", "z_task.json")
            _write_task(d, "a_task", "a_task.json")
            tasks = load_generation_tasks(d)
            assert tasks[0]["id"] == "a_task"

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                load_generation_tasks(Path(td))

    def test_missing_required_field_raises(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "bad.json").write_text(json.dumps({"id": "x"}))  # missing system_prompt
            with pytest.raises(ValueError, match="system_prompt"):
                load_generation_tasks(d)


# ── run_throughput ────────────────────────────────────────────────────────────

class TestRunThroughput:
    @pytest.fixture
    def tasks(self):
        return [
            {"id": "t01", "system_prompt": "sys", "user_message": "msg1"},
            {"id": "t02", "system_prompt": "sys", "user_message": "msg2"},
        ]

    @pytest.mark.asyncio
    async def test_returns_phase_metrics(self, tasks, tmp_path):
        from lib.client import BenchClient
        mock_client = AsyncMock(spec=BenchClient)
        mock_client.chat = AsyncMock(side_effect=[
            _make_result("t01", ttft_ms=120, decode_tps=210),
            _make_result("t02", ttft_ms=90, decode_tps=180),
        ])
        pm = await run_throughput(mock_client, tasks, model="test", max_tokens=256, concurrency=2)
        assert isinstance(pm, PhaseMetrics)
        assert len(pm.results) == 2
        assert pm.decode_tps_mean() > 0

    @pytest.mark.asyncio
    async def test_error_result_counted(self, tasks, tmp_path):
        from lib.client import BenchClient
        mock_client = AsyncMock(spec=BenchClient)
        mock_client.chat = AsyncMock(side_effect=[
            _make_result("t01"),
            _make_result("t02", error="timeout"),
        ])
        pm = await run_throughput(mock_client, tasks, model="test", max_tokens=256, concurrency=1)
        assert pm.error_rate() == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_concurrency_respected(self, tmp_path):
        """All tasks complete even at concurrency > len(tasks)."""
        from lib.client import BenchClient
        tasks = [{"id": f"t{i:02d}", "system_prompt": "s", "user_message": "m"} for i in range(5)]
        mock_client = AsyncMock(spec=BenchClient)
        mock_client.chat = AsyncMock(return_value=_make_result("tx"))
        pm = await run_throughput(mock_client, tasks, model="test", max_tokens=128, concurrency=10)
        assert len(pm.results) == 5


# ── run_prefix_cache ──────────────────────────────────────────────────────────

class TestRunPrefixCache:
    @pytest.fixture
    def tasks(self):
        return [{
            "id": "pc01",
            "system_prompt": "Long shared prefix " * 200,
            "user_message": "Answer this.",
            "warmup_message": "Warm up.",
            "cold_message": "Cold question.",
            "prefix_tokens": 400,
        }]

    @pytest.mark.asyncio
    async def test_returns_speedup_ratio(self, tasks, tmp_path):
        from lib.client import BenchClient
        mock_client = AsyncMock(spec=BenchClient)
        # warmup x2, then cold, then warm
        mock_client.chat = AsyncMock(side_effect=[
            _make_result("warmup_0", ttft_ms=400),
            _make_result("warmup_1", ttft_ms=380),
            _make_result("pc01_cold", ttft_ms=500),
            _make_result("pc01_warm", ttft_ms=100),  # cache hit!
        ])
        result = await run_prefix_cache(
            mock_client, tasks, model="test", max_tokens=64, warmup_rounds=2
        )
        assert result["prefix_reuse_speedup"] == pytest.approx(100 / 500)
        assert result["n_valid"] == 1
        assert result["per_task"][0]["speedup_ratio"] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_handles_cold_result_error(self, tasks, tmp_path):
        from lib.client import BenchClient
        mock_client = AsyncMock(spec=BenchClient)
        mock_client.chat = AsyncMock(side_effect=[
            _make_result("warmup_0"),
            _make_result("warmup_1"),
            _make_result("cold", ttft_ms=0, error="timeout"),  # failed cold
            _make_result("warm", ttft_ms=80),
        ])
        result = await run_prefix_cache(
            mock_client, tasks, model="test", max_tokens=64, warmup_rounds=2
        )
        # cold failed — not counted as valid
        assert result["n_valid"] == 0

    @pytest.mark.asyncio
    async def test_no_speedup_ratio_1(self, tasks, tmp_path):
        """When cold == warm TTFT, ratio should be 1.0 (no cache benefit)."""
        from lib.client import BenchClient
        mock_client = AsyncMock(spec=BenchClient)
        mock_client.chat = AsyncMock(side_effect=[
            _make_result("warmup_0"),
            _make_result("warmup_1"),
            _make_result("cold", ttft_ms=300),
            _make_result("warm", ttft_ms=300),
        ])
        result = await run_prefix_cache(
            mock_client, tasks, model="test", max_tokens=64, warmup_rounds=2
        )
        assert result["prefix_reuse_speedup"] == pytest.approx(1.0)
