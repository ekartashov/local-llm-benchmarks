"""OpenAI-compatible benchmark client with automatic TTFT and decode-speed instrumentation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from lib.metrics import BenchResult, Stopwatch


class BenchClient:
    """
    Thin wrapper around AsyncOpenAI that:
    - Measures TTFT (time from request send to first SSE chunk with content)
    - Measures decode speed (tokens/s after first chunk)
    - Saves every raw response to results_dir/raw/<task_id>.json
    """

    def __init__(self, base_url: str, results_dir: Path, api_key: str = "local") -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._results_dir = results_dir
        self._raw_dir = results_dir / "raw"
        self._raw_dir.mkdir(parents=True, exist_ok=True)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        task_id: str = "",
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> BenchResult:
        result = BenchResult(task_id=task_id)
        sw = Stopwatch().start()

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            first_chunk = True
            content_parts: list[str] = []
            tool_call_chunks: dict[int, dict[str, Any]] = {}
            usage: dict[str, int] = {}

            async with await self._client.chat.completions.create(**kwargs) as stream:
                async for chunk in stream:
                    if first_chunk:
                        result.ttft_ms = sw.split()
                        first_chunk = False

                    delta = chunk.choices[0].delta if chunk.choices else None

                    # Capture usage before any early-continue — the usage-only
                    # chunk has choices=[] so delta would be None below.
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens or 0,
                            "completion_tokens": chunk.usage.completion_tokens or 0,
                        }

                    if delta is None:
                        continue

                    if delta.content:
                        content_parts.append(delta.content)

                    # vLLM reasoning parser puts <think> tokens in reasoning_content
                    # (a non-standard field surfaced via model_extra). Capture it so
                    # token counts and decode_tps are accurate even in thinking mode.
                    reasoning = (delta.model_extra or {}).get("reasoning_content") if hasattr(delta, "model_extra") else None
                    if reasoning:
                        content_parts.append(reasoning)

                    # Accumulate tool call deltas
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_call_chunks:
                                tool_call_chunks[idx] = {
                                    "id": tc.id or "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc.function:
                                if tc.function.name:
                                    tool_call_chunks[idx]["function"]["name"] += tc.function.name
                                if tc.function.arguments:
                                    tool_call_chunks[idx]["function"]["arguments"] += (
                                        tc.function.arguments
                                    )

            result.latency_ms = sw.elapsed_ms()
            result.raw_text = "".join(content_parts)
            result.tool_calls = list(tool_call_chunks.values())
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)

            # Decode speed: completion tokens / time after first token
            decode_time_s = (result.latency_ms - result.ttft_ms) / 1000
            if decode_time_s > 0 and result.completion_tokens > 0:
                result.decode_tps = result.completion_tokens / decode_time_s

            # Serialise raw response
            raw: dict[str, Any] = {
                "task_id": task_id,
                "request": {
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "temperature": temperature,
                },
                "response": {
                    "content": result.raw_text,
                    "tool_calls": result.tool_calls,
                    "usage": usage,
                },
                "timing": {
                    "ttft_ms": result.ttft_ms,
                    "decode_tps": result.decode_tps,
                    "latency_ms": result.latency_ms,
                },
            }

        except Exception as exc:
            result.latency_ms = sw.elapsed_ms()
            result.error = str(exc)
            raw = {
                "task_id": task_id,
                "error": result.error,
                "timing": {"latency_ms": result.latency_ms},
            }

        self._save_raw(task_id, raw)
        return result

    def _save_raw(self, task_id: str, data: dict[str, Any]) -> None:
        safe_id = task_id.replace("/", "_").replace(" ", "_") or f"task_{int(time.time())}"
        path = self._raw_dir / f"{safe_id}.json"
        path.write_text(json.dumps(data, indent=2))
