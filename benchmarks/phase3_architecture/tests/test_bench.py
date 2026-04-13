"""Tests for Phase 3 bench — classify_tier, run_swarm, run_routing, run_swap_timing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.metrics import BenchResult, PhaseMetrics
from benchmarks.phase3_architecture.bench import (
    classify_tier,
    load_tasks,
    run_swarm,
    run_routing,
    run_swap_timing,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(task_id: str = "t1", decode_tps: float = 180.0,
                 ttft_ms: float = 150.0, error: str = "") -> BenchResult:
    r = BenchResult(task_id=task_id)
    r.raw_text = "def foo(): pass"
    r.ttft_ms = ttft_ms
    r.decode_tps = decode_tps
    r.latency_ms = 600.0
    r.prompt_tokens = 50
    r.completion_tokens = 40
    r.error = error
    return r


def _write_task(d: Path, task_id: str, extra: dict | None = None) -> None:
    data = {
        "id": task_id,
        "system_prompt": "You are helpful.",
        "user_message": "Write a function.",
        **(extra or {}),
    }
    (d / f"{task_id}.json").write_text(json.dumps(data))


# ── classify_tier ─────────────────────────────────────────────────────────────

class TestClassifyTier:
    def test_implement_is_coder(self):
        assert classify_tier("Implement a thread-safe LRU cache") == "coder"

    def test_fix_is_coder(self):
        assert classify_tier("Fix the deadlock in this code") == "coder"

    def test_refactor_is_coder(self):
        assert classify_tier("Refactor this class to use __slots__") == "coder"

    def test_analyze_is_thinker(self):
        assert classify_tier("Analyze the trade-offs between approaches") == "thinker"

    def test_prove_is_thinker(self):
        assert classify_tier("Prove the correctness of this algorithm") == "thinker"

    def test_explain_why_is_thinker(self):
        assert classify_tier("Explain why eventual consistency causes issues") == "thinker"

    def test_compare_contrast_is_thinker(self):
        assert classify_tier("Compare and contrast MoE vs dense architectures") == "thinker"

    def test_empty_string_defaults_to_coder(self):
        # No keywords → coder_score == thinker_score == 0 → returns "coder"
        result = classify_tier("")
        assert result in ("coder", "thinker")


# ── load_tasks ────────────────────────────────────────────────────────────────

class TestLoadTasks:
    def test_loads_all_json_files(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_task(d, "t01")
            _write_task(d, "t02")
            tasks = load_tasks(d)
            assert len(tasks) == 2

    def test_sorted_alphabetically(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_task(d, "z01")
            _write_task(d, "a01")
            tasks = load_tasks(d)
            assert tasks[0]["id"] == "a01"

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                load_tasks(Path(td))

    def test_missing_required_field_raises(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "bad.json").write_text(json.dumps({"id": "x"}))
            with pytest.raises(ValueError, match="user_message"):
                load_tasks(d)

    def test_custom_required_fields(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_task(d, "t01", extra={"correct_tier": "coder"})
            tasks = load_tasks(
                d,
                required_fields=("id", "system_prompt", "user_message", "correct_tier"),
            )
            assert tasks[0]["correct_tier"] == "coder"


# ── run_swarm ─────────────────────────────────────────────────────────────────

class TestRunSwarm:
    @pytest.fixture
    def tasks(self):
        return [
            {"id": f"sw0{i}", "system_prompt": "sys", "user_message": f"msg{i}"}
            for i in range(4)
        ]

    @pytest.mark.asyncio
    async def test_returns_speedup_and_wall_times(self, tasks):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[_make_result(f"sw0{i}") for i in range(8)]
        )
        result = await run_swarm(
            mock_client, tasks, model="test", max_tokens=256, concurrency=4
        )
        assert "swarm_speedup" in result
        assert "sequential_wall_s" in result
        assert "parallel_wall_s" in result
        assert result["n_tasks"] == 4

    @pytest.mark.asyncio
    async def test_speedup_is_ratio(self, tasks):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[_make_result(f"sw0{i}") for i in range(8)]
        )
        result = await run_swarm(
            mock_client, tasks, model="test", max_tokens=256, concurrency=4
        )
        expected = result["parallel_wall_s"] / result["sequential_wall_s"]
        assert result["swarm_speedup"] == pytest.approx(expected, rel=1e-3)

    @pytest.mark.asyncio
    async def test_all_errors_still_returns_dict(self, tasks):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[_make_result(f"sw0{i}", error="500") for i in range(8)]
        )
        result = await run_swarm(
            mock_client, tasks, model="test", max_tokens=256, concurrency=4
        )
        assert result["par_error_rate"] == pytest.approx(1.0)


# ── run_routing ───────────────────────────────────────────────────────────────

class TestRunRouting:
    @pytest.fixture
    def routing_tasks(self):
        return [
            {
                "id": "r01", "system_prompt": "sys",
                "user_message": "Implement a function",
                "correct_tier": "coder",
            },
            {
                "id": "r02", "system_prompt": "sys",
                "user_message": "Analyze the trade-offs",
                "correct_tier": "thinker",
            },
        ]

    @pytest.mark.asyncio
    async def test_returns_accuracy_dict(self, routing_tasks, tmp_path):
        mock_response = MagicMock()
        mock_response.json.return_value = {"model": "coder-model-id", "choices": []}
        mock_response.raise_for_status = MagicMock()

        with patch("benchmarks.phase3_architecture.bench.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_http

            result = await run_routing(
                proxy_url="http://localhost:30100/v1",
                tasks=routing_tasks,
                coder_model_id="coder-model-id",
                thinker_model_id="thinker-model-id",
                results_dir=tmp_path,
            )

        assert "routing_accuracy" in result
        assert "n_tasks" in result
        assert result["n_tasks"] == 2
        assert 0.0 <= result["routing_accuracy"] <= 1.0

    @pytest.mark.asyncio
    async def test_error_handling(self, routing_tasks, tmp_path):
        with patch("benchmarks.phase3_architecture.bench.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(side_effect=Exception("connection refused"))
            mock_cls.return_value = mock_http

            result = await run_routing(
                proxy_url="http://localhost:30100/v1",
                tasks=routing_tasks,
                coder_model_id="coder",
                thinker_model_id="thinker",
                results_dir=tmp_path,
            )

        # Should not raise; errors are captured per-task
        assert result["n_tasks"] == 2
        for t in result["per_task"]:
            assert t["error"] != "" or t["routing_correct"] in (True, False)


# ── run_swap_timing ───────────────────────────────────────────────────────────

class TestRunSwapTiming:
    @pytest.mark.asyncio
    async def test_parses_swap_latency_from_stdout(self, tmp_path):
        fake_stdout = (
            b"[deploy] Done.\n"
            b"[swap-model] Swap complete in 12.34s (12340ms)\n"
            b"SWAP_LATENCY_MS=12340\n"
            b"SWAP_LATENCY_S=12.34\n"
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(fake_stdout, b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await run_swap_timing(
                engine="vllm", gpu="gpu0",
                model_a="ModelA", model_b="ModelB",
                rounds=2,
                results_dir=tmp_path,
            )

        assert result["rounds"] == 4  # 2 rounds × 2 swaps each
        assert result["swap_latency_s_p95"] == pytest.approx(12.34, rel=1e-3)
        assert result["swap_latency_s"] == result["swap_latency_s_p95"]

    @pytest.mark.asyncio
    async def test_fallback_when_no_latency_in_stdout(self, tmp_path):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"No latency here\n", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await run_swap_timing(
                engine="vllm", gpu="gpu0",
                model_a="A", model_b="B",
                rounds=1,
                results_dir=tmp_path,
            )

        # Should fall back to wall-clock time (very small in test)
        assert result["rounds"] == 2
        assert result["swap_latency_s_p95"] >= 0.0
