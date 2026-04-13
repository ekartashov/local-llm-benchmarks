"""Tests for lib.reporter — verdict logic and summary generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from lib.reporter import _verdict, generate_summary


THRESHOLDS: dict = {
    "phase0_tool_reliability": {
        "tool_call_success_rate": {"pass": 0.95, "inconclusive": 0.80},
        "critical_error_rate": {"pass": 0.00, "inconclusive": 0.03},
    },
    "phase1_engine_selection": {
        "decode_tps": {"pass": 150, "inconclusive": 80},
        "ttft_ms": {"pass": 500, "inconclusive": 1000},
    },
}


class TestVerdict:
    def test_pass_all_metrics_above_threshold(self):
        metrics = {"tool_call_success_rate": 0.97, "critical_error_rate": 0.0}
        assert _verdict("phase0_tool_reliability", metrics, THRESHOLDS) == "PASS"

    def test_fail_below_inconclusive(self):
        metrics = {"tool_call_success_rate": 0.70, "critical_error_rate": 0.0}
        assert _verdict("phase0_tool_reliability", metrics, THRESHOLDS) == "FAIL"

    def test_inconclusive_between_pass_and_fail(self):
        metrics = {"tool_call_success_rate": 0.85, "critical_error_rate": 0.0}
        assert _verdict("phase0_tool_reliability", metrics, THRESHOLDS) == "INCONCLUSIVE"

    def test_fail_due_to_critical_errors(self):
        metrics = {"tool_call_success_rate": 0.96, "critical_error_rate": 0.10}
        assert _verdict("phase0_tool_reliability", metrics, THRESHOLDS) == "FAIL"

    def test_inconclusive_if_metric_missing(self):
        metrics = {"tool_call_success_rate": 0.97}  # critical_error_rate missing
        result = _verdict("phase0_tool_reliability", metrics, THRESHOLDS)
        assert result in ("PASS", "INCONCLUSIVE")

    def test_pass_phase1_high_throughput(self):
        metrics = {"decode_tps": 200, "ttft_ms": 300}
        assert _verdict("phase1_engine_selection", metrics, THRESHOLDS) == "PASS"

    def test_fail_phase1_low_throughput(self):
        metrics = {"decode_tps": 50, "ttft_ms": 300}
        assert _verdict("phase1_engine_selection", metrics, THRESHOLDS) == "FAIL"

    def test_lower_is_better_ttft(self):
        # ttft_ms: pass=500, inconclusive=1000 — lower is better
        metrics = {"decode_tps": 200, "ttft_ms": 750}  # ttft between pass and incon
        assert _verdict("phase1_engine_selection", metrics, THRESHOLDS) == "INCONCLUSIVE"

    def test_unknown_phase_returns_inconclusive(self):
        assert _verdict("phase99_unknown", {}, THRESHOLDS) == "INCONCLUSIVE"


class TestGenerateSummary:
    def _write_metrics(self, tmp: Path, data: dict) -> None:
        (tmp / "metrics.json").write_text(json.dumps(data))

    def test_no_metrics_file(self):
        with tempfile.TemporaryDirectory() as td:
            summary = generate_summary(Path(td), THRESHOLDS)
            assert "No metrics.json" in summary

    def test_generates_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write_metrics(d, {
                "phase": "phase0_tool_reliability",
                "timestamp": "2026-04-14T10:00:00Z",
                "config": {
                    "engine": "vllm",
                    "engine_version": "0.19.2",
                    "model": "Qwen/Qwen3.5-35B-A3B-AWQ",
                    "quantization": "AWQ-INT4",
                    "gpu": "RTX 5090",
                    "gpu_id": 0,
                    "context_length": 114688,
                },
                "metrics": {"tool_call_success_rate": 0.97, "critical_error_rate": 0.0},
                "verdict": "PASS",
                "notes": "",
            })
            summary = generate_summary(d, THRESHOLDS)
            assert "phase0_tool_reliability" in summary
            assert "PASS" in summary
            assert "tool_call_success_rate" in summary

    def test_verdict_written_back_to_metrics_json(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write_metrics(d, {
                "phase": "phase0_tool_reliability",
                "timestamp": "2026-04-14T10:00:00Z",
                "config": {},
                "metrics": {"tool_call_success_rate": 0.97, "critical_error_rate": 0.0},
                "verdict": "",
                "notes": "",
            })
            generate_summary(d, THRESHOLDS)
            written = json.loads((d / "metrics.json").read_text())
            assert written["verdict"] == "PASS"

    def test_per_task_table_rendered(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write_metrics(d, {
                "phase": "phase0_tool_reliability",
                "timestamp": "2026-04-14T10:00:00Z",
                "config": {},
                "metrics": {
                    "tool_call_success_rate": 0.5,
                    "critical_error_rate": 0.0,
                    "per_task": [
                        {"task_id": "01_read_file", "score": "pass", "reason": ""},
                        {"task_id": "02_write_file", "score": "wrong_tool", "reason": "got list_dir"},
                    ],
                },
                "verdict": "",
                "notes": "",
            })
            summary = generate_summary(d, THRESHOLDS)
            assert "01_read_file" in summary
            assert "02_write_file" in summary
