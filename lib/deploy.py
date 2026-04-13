"""Python wrapper around infra/scripts/deploy.sh for programmatic use in benchmarks."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

# Repo root — works whether the module is imported from any cwd.
_REPO_ROOT = Path(__file__).parent.parent


async def deploy(
    engine: str,
    gpu: str,
    model: str,
    *extra_args: str,
    timeout: int = 300,
) -> None:
    """
    Launch an inference engine container via deploy.sh.

    Args:
        engine:     "vllm" | "sglang" | "llamacpp"
        gpu:        "gpu0" | "gpu1"
        model:      HuggingFace repo id (e.g. "Qwen/Qwen3.5-35B-A3B-AWQ")
        extra_args: forwarded verbatim to deploy.sh (e.g. "--ctx", "114688")
        timeout:    seconds to wait for the health check to pass
    """
    script = _REPO_ROOT / "infra" / "scripts" / "deploy.sh"
    cmd = [str(script), engine, gpu, model, *extra_args]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(_REPO_ROOT),
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    if proc.returncode != 0:
        output = stdout.decode() if stdout else ""
        raise RuntimeError(
            f"deploy.sh exited with code {proc.returncode}.\n{output}"
        )


async def teardown(timeout: int = 60) -> None:
    """Stop all running inference containers via teardown.sh."""
    script = _REPO_ROOT / "infra" / "scripts" / "teardown.sh"
    proc = await asyncio.create_subprocess_exec(
        str(script),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(_REPO_ROOT),
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    if proc.returncode != 0:
        output = stdout.decode() if stdout else ""
        raise RuntimeError(f"teardown.sh exited with code {proc.returncode}.\n{output}")


def deploy_sync(
    engine: str,
    gpu: str,
    model: str,
    *extra_args: str,
    timeout: int = 300,
) -> None:
    """Synchronous convenience wrapper around deploy()."""
    script = _REPO_ROOT / "infra" / "scripts" / "deploy.sh"
    cmd = [str(script), engine, gpu, model, *extra_args]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        timeout=timeout,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"deploy.sh exited with code {result.returncode}.\n{result.stdout}\n{result.stderr}"
        )


def teardown_sync(timeout: int = 60) -> None:
    """Synchronous convenience wrapper around teardown()."""
    script = _REPO_ROOT / "infra" / "scripts" / "teardown.sh"
    result = subprocess.run(
        [str(script)],
        cwd=str(_REPO_ROOT),
        timeout=timeout,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"teardown.sh exited with code {result.returncode}.\n{result.stdout}\n{result.stderr}"
        )
