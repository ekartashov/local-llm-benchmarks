"""Metric data types and calculators shared across all benchmark phases."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchResult:
    """Holds every measurable outcome from a single chat completion request."""

    task_id: str = ""

    # Timing
    ttft_ms: float = 0.0        # time-to-first-token (ms)
    decode_tps: float = 0.0     # tokens/s during decode phase only
    latency_ms: float = 0.0     # total wall time (ms)

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    # Content
    raw_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # Raw API response (serialisable dict)
    response: dict[str, Any] = field(default_factory=dict)

    # Error (non-empty means the request failed)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.error == ""


@dataclass
class PhaseMetrics:
    """Aggregate metrics across many BenchResult objects."""

    results: list[BenchResult] = field(default_factory=list)

    def add(self, r: BenchResult) -> None:
        self.results.append(r)

    # ── Latency ───────────────────────────────────────────────────────────────

    def ttft_p50(self) -> float:
        return _percentile([r.ttft_ms for r in self.results if r.ok], 50)

    def ttft_p95(self) -> float:
        return _percentile([r.ttft_ms for r in self.results if r.ok], 95)

    def decode_tps_mean(self) -> float:
        vals = [r.decode_tps for r in self.results if r.ok and r.decode_tps > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def latency_p50(self) -> float:
        return _percentile([r.latency_ms for r in self.results if r.ok], 50)

    def latency_p95(self) -> float:
        return _percentile([r.latency_ms for r in self.results if r.ok], 95)

    # ── Throughput ────────────────────────────────────────────────────────────

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.results)

    def error_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if not r.ok) / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.results),
            "ttft_p50_ms": self.ttft_p50(),
            "ttft_p95_ms": self.ttft_p95(),
            "decode_tps_mean": self.decode_tps_mean(),
            "latency_p50_ms": self.latency_p50(),
            "latency_p95_ms": self.latency_p95(),
            "total_tokens": self.total_tokens(),
            "error_rate": self.error_rate(),
        }


class Stopwatch:
    """Lightweight wall-clock timer."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._splits: list[float] = []

    def start(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self._splits = []
        return self

    def split(self) -> float:
        """Record a split and return elapsed ms since start."""
        elapsed = (time.perf_counter() - self._start) * 1000
        self._splits.append(elapsed)
        return elapsed

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    @property
    def splits(self) -> list[float]:
        return list(self._splits)


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]
