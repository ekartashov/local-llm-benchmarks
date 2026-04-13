"""Tests for lib.metrics — unit tests for BenchResult, PhaseMetrics, Stopwatch."""

from __future__ import annotations

import time

import pytest

from lib.metrics import BenchResult, PhaseMetrics, Stopwatch, _percentile


# ── BenchResult ────────────────────────────────────────────────────────────────

class TestBenchResult:
    def test_ok_when_no_error(self):
        r = BenchResult(task_id="t1")
        assert r.ok is True

    def test_not_ok_when_error(self):
        r = BenchResult(task_id="t1", error="timeout")
        assert r.ok is False

    def test_total_tokens(self):
        r = BenchResult(prompt_tokens=100, completion_tokens=50)
        assert r.total_tokens == 150

    def test_defaults(self):
        r = BenchResult()
        assert r.task_id == ""
        assert r.ttft_ms == 0.0
        assert r.decode_tps == 0.0
        assert r.tool_calls == []
        assert r.raw_text == ""


# ── PhaseMetrics ───────────────────────────────────────────────────────────────

class TestPhaseMetrics:
    def _sample_results(self, n: int = 5, error_count: int = 0) -> list[BenchResult]:
        results = []
        for i in range(n):
            r = BenchResult(task_id=f"t{i}")
            r.ttft_ms = float(100 + i * 10)
            r.decode_tps = float(200 + i * 5)
            r.latency_ms = float(500 + i * 20)
            r.prompt_tokens = 100
            r.completion_tokens = 50
            if i < error_count:
                r.error = "fail"
            results.append(r)
        return results

    def test_empty(self):
        pm = PhaseMetrics()
        assert pm.ttft_p50() == 0.0
        assert pm.decode_tps_mean() == 0.0
        assert pm.error_rate() == 0.0

    def test_add_and_ttft(self):
        pm = PhaseMetrics()
        for r in self._sample_results(5):
            pm.add(r)
        assert pm.ttft_p50() > 0

    def test_error_rate(self):
        pm = PhaseMetrics()
        for r in self._sample_results(10, error_count=2):
            pm.add(r)
        assert pm.error_rate() == pytest.approx(0.2)

    def test_to_dict_keys(self):
        pm = PhaseMetrics()
        for r in self._sample_results(3):
            pm.add(r)
        d = pm.to_dict()
        assert "ttft_p50_ms" in d
        assert "decode_tps_mean" in d
        assert "error_rate" in d
        assert "total_tokens" in d


# ── Stopwatch ─────────────────────────────────────────────────────────────────

class TestStopwatch:
    def test_elapsed_increases(self):
        sw = Stopwatch().start()
        time.sleep(0.01)
        ms = sw.elapsed_ms()
        assert ms > 5  # at least 5ms

    def test_split_returns_elapsed(self):
        sw = Stopwatch().start()
        time.sleep(0.01)
        split_ms = sw.split()
        assert split_ms > 0
        assert split_ms in sw.splits

    def test_multiple_splits(self):
        sw = Stopwatch().start()
        sw.split()
        sw.split()
        assert len(sw.splits) == 2
        assert sw.splits[1] >= sw.splits[0]


# ── _percentile ───────────────────────────────────────────────────────────────

class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 99) == 42.0

    def test_p50_even(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        assert _percentile(vals, 50) == 2.0

    def test_p95_small_list(self):
        vals = list(range(1, 21))  # 1..20
        result = _percentile([float(v) for v in vals], 95)
        assert result >= 19.0  # should be near the top
