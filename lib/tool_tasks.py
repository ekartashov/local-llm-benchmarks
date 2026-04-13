"""Task loader and validator for tool-call benchmark tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator


class ToolTask(BaseModel):
    id: str
    description: str
    system_prompt: str
    user_message: str
    tools: list[dict[str, Any]]
    expected_tool_calls: list[dict[str, Any]] = []
    mock_tool_response: str = ""
    difficulty: str = "simple"
    category: str = ""

    @field_validator("difficulty")
    @classmethod
    def check_difficulty(cls, v: str) -> str:
        allowed = {"simple", "medium", "hard"}
        if v not in allowed:
            raise ValueError(f"difficulty must be one of {allowed}, got {v!r}")
        return v

    def to_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_message},
        ]

    def to_messages_with_mock_response(self, tool_call_id: str = "call_0") -> list[dict[str, Any]]:
        """Return the full conversation up through the mock tool response."""
        msgs = self.to_messages()
        if self.expected_tool_calls and self.mock_tool_response:
            first_expected = self.expected_tool_calls[0]
            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": first_expected["name"],
                        "arguments": json.dumps({"path": "mock"}),
                    },
                }],
            })
            msgs.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": self.mock_tool_response,
            })
        return msgs


def load_tasks(tasks_dir: Path, task_filter: str | None = None) -> list[ToolTask]:
    """
    Load all .json task files from tasks_dir, sorted by filename.
    If task_filter is set, only load tasks whose id starts with task_filter.
    """
    task_files = sorted(tasks_dir.glob("*.json"))
    if not task_files:
        raise FileNotFoundError(f"No task JSON files found in {tasks_dir}")

    tasks: list[ToolTask] = []
    for path in task_files:
        data = json.loads(path.read_text())
        task = ToolTask.model_validate(data)
        if task_filter and not task.id.startswith(task_filter):
            continue
        tasks.append(task)

    if not tasks:
        raise ValueError(f"No tasks matched filter {task_filter!r}")

    return tasks
