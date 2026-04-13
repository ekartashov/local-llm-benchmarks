"""
Shared long system prompt for prefix-cache tasks.
Imported by tasks that need a consistent long prefix.
Estimated: ~2000–2500 tokens — long enough to stress KV cache meaningfully.
"""

SYSTEM_PROMPT = """You are an expert coding assistant with deep knowledge of the following codebase.

## Project: local-llm-benchmarks
A systematic benchmark suite for validating local LLM inference configurations on a dual RTX 5090 workstation.

## File: lib/client.py
```python
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any
from openai import AsyncOpenAI
from lib.metrics import BenchResult, Stopwatch

class BenchClient:
    def __init__(self, base_url: str, results_dir: Path, api_key: str = "local") -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._results_dir = results_dir
        self._raw_dir = results_dir / "raw"
        self._raw_dir.mkdir(parents=True, exist_ok=True)

    async def chat(self, messages, tools=None, task_id="", model="default",
                   temperature=0.0, max_tokens=2048) -> BenchResult:
        result = BenchResult(task_id=task_id)
        sw = Stopwatch().start()
        kwargs = dict(model=model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens, stream=True)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            first_chunk = True
            content_parts = []
            tool_call_chunks = {}
            usage = {}
            async with await self._client.chat.completions.create(**kwargs) as stream:
                async for chunk in stream:
                    if first_chunk:
                        result.ttft_ms = sw.split()
                        first_chunk = False
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue
                    if delta.content:
                        content_parts.append(delta.content)
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_call_chunks:
                                tool_call_chunks[idx] = {"id": tc.id or "", "type": "function",
                                    "function": {"name": "", "arguments": ""}}
                            if tc.function:
                                if tc.function.name:
                                    tool_call_chunks[idx]["function"]["name"] += tc.function.name
                                if tc.function.arguments:
                                    tool_call_chunks[idx]["function"]["arguments"] += tc.function.arguments
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = {"prompt_tokens": chunk.usage.prompt_tokens or 0,
                                 "completion_tokens": chunk.usage.completion_tokens or 0}
            result.latency_ms = sw.elapsed_ms()
            result.raw_text = "".join(content_parts)
            result.tool_calls = list(tool_call_chunks.values())
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)
            decode_time_s = (result.latency_ms - result.ttft_ms) / 1000
            if decode_time_s > 0 and result.completion_tokens > 0:
                result.decode_tps = result.completion_tokens / decode_time_s
        except Exception as exc:
            result.latency_ms = sw.elapsed_ms()
            result.error = str(exc)
        return result
```

## File: lib/metrics.py
```python
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

@dataclass
class BenchResult:
    task_id: str = ""
    ttft_ms: float = 0.0
    decode_tps: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    @property
    def total_tokens(self) -> int: return self.prompt_tokens + self.completion_tokens
    raw_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    @property
    def ok(self) -> bool: return self.error == ""

@dataclass
class PhaseMetrics:
    results: list[BenchResult] = field(default_factory=list)
    def add(self, r: BenchResult) -> None: self.results.append(r)
    def ttft_p50(self) -> float: return _percentile([r.ttft_ms for r in self.results if r.ok], 50)
    def ttft_p95(self) -> float: return _percentile([r.ttft_ms for r in self.results if r.ok], 95)
    def decode_tps_mean(self) -> float:
        vals = [r.decode_tps for r in self.results if r.ok and r.decode_tps > 0]
        return sum(vals) / len(vals) if vals else 0.0
    def to_dict(self):
        return {"n": len(self.results), "ttft_p50_ms": self.ttft_p50(),
                "ttft_p95_ms": self.ttft_p95(), "decode_tps_mean": self.decode_tps_mean(),
                "error_rate": sum(1 for r in self.results if not r.ok) / len(self.results) if self.results else 0.0}

class Stopwatch:
    def __init__(self): self._start = 0.0; self._splits = []
    def start(self): self._start = time.perf_counter(); self._splits = []; return self
    def split(self):
        elapsed = (time.perf_counter() - self._start) * 1000; self._splits.append(elapsed); return elapsed
    def elapsed_ms(self): return (time.perf_counter() - self._start) * 1000

def _percentile(values, pct):
    if not values: return 0.0
    sv = sorted(values)
    idx = min(int(len(sv) * pct / 100), len(sv) - 1)
    return sv[idx]
```

## File: lib/scorer.py
```python
from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import Any
from lib.metrics import BenchResult

class ToolCallScore:
    PASS = "pass"; FORMAT_ERROR = "format_error"; DROPPED = "dropped"
    WRONG_TOOL = "wrong_tool"; WRONG_ARGS = "wrong_args"
    NO_CALL = "no_call"; EXCEPTION = "exception"

@dataclass
class ScoredResult:
    task_id: str; score: str; reason: str = ""; bench_result: BenchResult | None = None
    @property
    def passed(self): return self.score == ToolCallScore.PASS
    @property
    def is_engine_error(self): return self.score in (ToolCallScore.DROPPED, ToolCallScore.FORMAT_ERROR)

def score_tool_call(result, expected_tool_calls):
    if not result.ok: return ScoredResult(result.task_id, ToolCallScore.EXCEPTION, result.error, result)
    if not expected_tool_calls: return ScoredResult(result.task_id, ToolCallScore.PASS, bench_result=result)
    if not result.tool_calls:
        score = _detect_dropped_or_no_call(result.raw_text)
        return ScoredResult(result.task_id, score, "No tool calls", result)
    for expected in expected_tool_calls:
        exp_name = expected.get("name", "")
        must_args = expected.get("must_include_args", [])
        matching = [tc for tc in result.tool_calls if tc.get("function", {}).get("name") == exp_name]
        if not matching:
            called = [tc.get("function", {}).get("name") for tc in result.tool_calls]
            return ScoredResult(result.task_id, ToolCallScore.WRONG_TOOL, f"Expected {exp_name}, got {called}", result)
        raw_args = matching[0].get("function", {}).get("arguments", "")
        try: args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError: return ScoredResult(result.task_id, ToolCallScore.FORMAT_ERROR, f"Bad JSON: {raw_args!r}", result)
        for key in must_args:
            if key not in args or not args[key]: return ScoredResult(result.task_id, ToolCallScore.WRONG_ARGS, f"Missing {key}", result)
    return ScoredResult(result.task_id, ToolCallScore.PASS, bench_result=result)
```

## Architecture decisions
- No dense 70B with tensor parallelism — PCIe x8/x8 without NVLink gives only 20-35 t/s
- System RAM is not VRAM — KV cache in DDR5 = 50-80% speed loss vs GDDR7
- KTransformers not viable — i9-14900K lacks AMX instructions
- Speculative decoding does not help MoE models
- No Ollama — 10-30% overhead vs raw engine containers

## Hardware spec
- CPU: Intel Core i9-14900K (24 cores, Raptor Lake, NO AMX)
- RAM: 192 GB DDR5 (4×48 GB, ~83 GB/s actual due to 4-DIMM downclock)
- GPU 0: NVIDIA RTX 5090 32 GB GDDR7 (1790 GB/s bandwidth, PCIe 5.0 x8)
- GPU 1: NVIDIA RTX 5090 32 GB GDDR7 (1790 GB/s bandwidth, PCIe 5.0 x8)
- No NVLink between GPUs — communication goes through PCIe

Answer the user's question about the codebase concisely and accurately. Keep answers under 150 words."""
