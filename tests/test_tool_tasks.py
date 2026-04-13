"""Tests for lib.tool_tasks — task loading and validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from lib.tool_tasks import ToolTask, load_tasks


def _write_task(d: Path, task: dict, filename: str = "01_test.json") -> None:
    (d / filename).write_text(json.dumps(task))


MINIMAL_TASK = {
    "id": "01_test",
    "description": "A minimal test task",
    "system_prompt": "You are a helpful assistant.",
    "user_message": "Do something.",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ],
    "expected_tool_calls": [{"name": "read_file", "must_include_args": ["path"]}],
    "mock_tool_response": "file contents",
    "difficulty": "simple",
    "category": "file_read",
}


class TestToolTaskModel:
    def test_valid_task(self):
        task = ToolTask.model_validate(MINIMAL_TASK)
        assert task.id == "01_test"
        assert task.difficulty == "simple"

    def test_invalid_difficulty(self):
        bad = {**MINIMAL_TASK, "difficulty": "extreme"}
        with pytest.raises(Exception):
            ToolTask.model_validate(bad)

    def test_defaults(self):
        minimal = {**MINIMAL_TASK}
        del minimal["expected_tool_calls"]
        del minimal["mock_tool_response"]
        del minimal["category"]
        task = ToolTask.model_validate(minimal)
        assert task.expected_tool_calls == []
        assert task.mock_tool_response == ""
        assert task.category == ""

    def test_to_messages_structure(self):
        task = ToolTask.model_validate(MINIMAL_TASK)
        msgs = task.to_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == MINIMAL_TASK["user_message"]

    def test_to_messages_with_mock_response(self):
        task = ToolTask.model_validate(MINIMAL_TASK)
        msgs = task.to_messages_with_mock_response()
        # system + user + assistant (tool call) + tool response
        assert len(msgs) == 4
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "tool"
        assert msgs[3]["content"] == MINIMAL_TASK["mock_tool_response"]


class TestLoadTasks:
    def test_load_single_task(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_task(d, MINIMAL_TASK)
            tasks = load_tasks(d)
            assert len(tasks) == 1
            assert tasks[0].id == "01_test"

    def test_load_sorted_by_filename(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i, name in [("z_last", "z_last.json"), ("a_first", "a_first.json")]:
                task = {**MINIMAL_TASK, "id": i}
                _write_task(d, task, name)
            tasks = load_tasks(d)
            # sorted alphabetically: a_first before z_last
            assert tasks[0].id == "a_first"
            assert tasks[1].id == "z_last"

    def test_no_files_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                load_tasks(Path(td))

    def test_task_filter(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in range(3):
                task = {**MINIMAL_TASK, "id": f"task_{i:02d}"}
                _write_task(d, task, f"task_{i:02d}.json")
            tasks = load_tasks(d, task_filter="task_01")
            assert len(tasks) == 1
            assert tasks[0].id == "task_01"

    def test_task_filter_no_match_raises(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_task(d, MINIMAL_TASK)
            with pytest.raises(ValueError, match="No tasks matched"):
                load_tasks(d, task_filter="nonexistent_")
