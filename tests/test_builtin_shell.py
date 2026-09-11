"""`shell` against real subprocesses on this host.

Commands are written per platform rather than mocked. A `shell` tool tested against
a fake process proves nothing about the two things that actually go wrong: output
interleaving across two streams, and a timeout that leaves the tree running.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from azalabscode.events import Event, ToolCallProgress
from azalabscode.toolio import ToolErrorKind
from azalabscode.tools import ToolContext, ToolSet
from azalabscode.tools.builtin.shell import (
    MAX_TIMEOUT_S,
    TIMEOUT_GRACE_S,
    ShellParams,
    ShellTool,
)
from azalabscode.tools.dispatcher import ToolCall, ToolDispatcher

IS_WINDOWS = os.name == "nt"
SHELL = ShellTool()


def cmd(windows: str, posix: str) -> str:
    """Pick the command for this host."""

    return windows if IS_WINDOWS else posix


async def run(ctx: ToolContext, command: str, **kwargs: object):
    params = ShellParams(command=command, **kwargs)  # type: ignore[arg-type]
    error = await SHELL.validate_params(params, ctx)
    if error is not None:
        from azalabscode.toolio import ToolResult

        return ToolResult.failure(error.kind, error.message)
    return await SHELL.run(params, ctx)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


async def test_a_successful_command_returns_its_output(tool_ctx: ToolContext) -> None:
    result = await run(tool_ctx, cmd("Write-Output hello", "echo hello"))

    assert result.ok is True
    assert "hello" in result.text
    assert result.meta["exit_code"] == 0


async def test_stdout_and_stderr_interleave_in_order(tool_ctx: ToolContext) -> None:
    """Spec delta 10. Two pipes cannot be ordered against each other; one append-only
    fd gives the process's own ordering for free."""

    result = await run(
        tool_ctx,
        cmd(
            "Write-Output one; [Console]::Error.WriteLine('two'); Write-Output three",
            "echo one; echo two 1>&2; echo three",
        ),
    )

    assert result.ok is True
    text = result.text
    assert text.index("one") < text.index("two") < text.index("three")


async def test_a_non_zero_exit_is_a_failure_that_still_carries_the_output(
    tool_ctx: ToolContext,
) -> None:
    """Spec 7: `ok=False, kind="exit_status"` so the model treats it as a failure but
    can still read the compiler errors."""

    result = await run(
        tool_ctx,
        cmd("Write-Output failing; exit 3", "echo failing; exit 3"),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.EXIT_STATUS
    assert result.error.details["exit_code"] == 3
    assert "failing" in result.text
    assert "exited with code 3" in result.text


async def test_a_command_with_no_output_says_so(tool_ctx: ToolContext) -> None:
    """Empty text would read as "the tool did not run"."""

    result = await run(tool_ctx, cmd("$null = 1", "true"))
    assert result.ok is True
    assert "no output" in result.text


async def test_the_working_directory_is_honoured(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "sub").mkdir()
    (workspace / "sub" / "marker.txt").write_text("x", encoding="utf-8")

    result = await run(tool_ctx, cmd("Get-ChildItem -Name", "ls"), cwd="sub")

    assert result.ok is True
    assert "marker.txt" in result.text


async def test_a_missing_working_directory_is_caught_before_the_prompt(
    tool_ctx: ToolContext,
) -> None:
    result = await run(tool_ctx, "echo hi", cwd="nope")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.NOT_FOUND


async def test_a_working_directory_outside_the_workspace_is_refused(
    tool_ctx: ToolContext, tmp_path: Path
) -> None:
    result = await run(tool_ctx, "echo hi", cwd=str(tmp_path))
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.PERMISSION


async def test_extra_environment_variables_reach_the_command(
    tool_ctx: ToolContext,
) -> None:
    result = await run(
        tool_ctx,
        cmd("Write-Output $env:AZA_TEST_VAR", "echo $AZA_TEST_VAR"),
        env={"AZA_TEST_VAR": "present"},
    )

    assert result.ok is True
    assert "present" in result.text


async def test_the_ambient_path_survives_an_env_override(tool_ctx: ToolContext) -> None:
    """Replacing the environment wholesale would make every command fail in a way
    that reads as the command's fault."""

    result = await run(
        tool_ctx,
        cmd("Write-Output ($env:PATH.Length -gt 0)", 'test -n "$PATH" && echo True'),
        env={"AZA_TEST_VAR": "x"},
    )
    assert "True" in result.text


async def test_an_empty_command_is_rejected(tool_ctx: ToolContext) -> None:
    result = await run(tool_ctx, "   ")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS


# ---------------------------------------------------------------------------
# Timeouts and the tree kill
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_a_timeout_kills_the_tree_and_returns_the_partial_output(
    tool_ctx: ToolContext,
) -> None:
    """The output produced before the kill is the most useful part of a timeout."""

    result = await run(
        tool_ctx,
        cmd(
            "Write-Output before; Start-Sleep -Seconds 30",
            "echo before; sleep 30",
        ),
        timeout=2.0,
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.TIMEOUT
    assert "before" in result.text
    assert result.meta["timed_out"] is True


@pytest.mark.slow
async def test_the_dispatcher_timeout_leaves_room_for_the_tool_to_report(
    tool_ctx: ToolContext,
) -> None:
    """The tool's own timeout has to fire first, or every slow command comes back as
    a bare dispatcher timeout with no output attached."""

    assert SHELL.timeout_for(ShellParams(command="x", timeout=5.0)) == 5.0 + TIMEOUT_GRACE_S
    assert (
        SHELL.timeout_for(ShellParams(command="x", timeout=MAX_TIMEOUT_S))
        == MAX_TIMEOUT_S + TIMEOUT_GRACE_S
    )


@pytest.mark.slow
async def test_the_dispatcher_cancelling_a_shell_kills_its_process(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """`on_cancel` is what an interrupt at M2 will rely on. A shell that survives its
    cancellation keeps writing to a file the harness has stopped reading."""

    tool = ShellTool()
    d = ToolDispatcher(ToolSet([tool]), context=tool_ctx)
    marker = workspace / "marker.txt"
    command = cmd(
        f"Start-Sleep -Seconds 20; Set-Content -Path '{marker}' -Value done",
        f"sleep 20; echo done > '{marker}'",
    )

    task = asyncio.create_task(
        d.call(ToolCall(call_id="c1", name="shell", arguments={"command": command}))  # type: ignore[arg-type]
    )
    await asyncio.sleep(1.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await d.drain_cleanups()

    await asyncio.sleep(1.0)
    assert not marker.exists(), "the shell survived its cancellation"
    assert tool._live == {}, "the live-process map leaked an entry"


async def test_the_timeout_parameter_is_capped_by_the_schema() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ShellParams(command="x", timeout=MAX_TIMEOUT_S + 1)
    with pytest.raises(ValidationError):
        ShellParams(command="x", timeout=0)


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_a_long_command_emits_progress_events(tool_ctx: ToolContext) -> None:
    """Without this the UI shows nothing for ten minutes and the user cannot tell a
    slow build from a hung one."""

    seen: list[Event] = []

    async def collect(event: Event) -> None:
        seen.append(event)

    tool_ctx.emit = collect
    tool_ctx.config.shell_progress_interval_s = 0.2

    await run(
        tool_ctx,
        cmd(
            "Write-Output tick; Start-Sleep -Milliseconds 900; Write-Output tock",
            "echo tick; sleep 0.9; echo tock",
        ),
        timeout=10.0,
    )

    progress = [e for e in seen if isinstance(e, ToolCallProgress)]
    assert progress, "no progress events were emitted for a ~1s command"
    assert any("tick" in e.text for e in progress)


async def test_a_missing_emitter_is_not_an_error(tool_ctx: ToolContext) -> None:
    """Scripts and tests run with no bus; the tool must tolerate it."""

    assert tool_ctx.emit is None
    result = await run(tool_ctx, cmd("Write-Output ok", "echo ok"))
    assert result.ok is True


async def test_an_emitter_that_raises_does_not_fail_the_call(
    tool_ctx: ToolContext,
) -> None:
    async def broken(event: Event) -> None:
        raise RuntimeError("the bus is down")

    tool_ctx.emit = broken
    tool_ctx.config.shell_progress_interval_s = 0.05

    result = await run(tool_ctx, cmd("Write-Output ok", "echo ok"), timeout=10.0)
    assert result.ok is True


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_shell_hard_blocks_retries() -> None:
    """A partially-applied command is not something a retry can undo (R-T-4, C-7)."""

    from azalabscode.errors import ConfigurationError
    from azalabscode.toolio import RetryPolicy

    assert SHELL.hard_block_retry is True
    assert SHELL.retry.attempts == 0

    class Retrying(ShellTool):
        retry: RetryPolicy = RetryPolicy(attempts=1, unsafe_allow_retry=True)

    with pytest.raises(ConfigurationError, match="hard-blocks retries"):
        Retrying()


def test_shell_is_never_concurrency_safe() -> None:
    assert SHELL.concurrency_safe is False
    assert SHELL.concurrency_safe_for(ShellParams(command="ls")) is False


def test_shell_always_needs_approval() -> None:
    assert SHELL.approval == "always"
    assert SHELL.needs_approval(ShellParams(command="ls")) is True


def test_the_description_names_this_platform_and_its_shell() -> None:
    """Spec delta 9: a model told nothing about the shell writes `ls | head` on a
    Windows box and gets a confusing failure."""

    description = ShellTool.description
    if IS_WINDOWS:
        assert "PowerShell" in description or "cmd.exe" in description
        assert "2>$null" in description or "findstr" in description
    else:
        assert "POSIX" in description
        assert "2>/dev/null" in description


def test_the_description_steers_away_from_reimplementing_the_other_tools() -> None:
    """The §9.3 acceptance check scans for `cat`/`sed -i`/`find` in shell calls, and
    the description is the only lever on that."""

    description = ShellTool.description
    for word in ("read_file over `cat`", "edit_file over `sed -i`", "glob over `find`"):
        assert word in description


def test_the_approval_summary_shows_the_command_verbatim(tool_ctx: ToolContext) -> None:
    summary = SHELL.approval_summary(ShellParams(command="rm -rf build", timeout=30.0), tool_ctx)
    assert "rm -rf build" in summary.detail
    assert summary.danger is True


@pytest.mark.slow
async def test_an_interactive_command_does_not_hang_forever(
    tool_ctx: ToolContext,
) -> None:
    """stdin is /dev/null, so a command that reads it gets EOF rather than blocking
    until the timeout."""

    command = (
        f'& "{sys.executable}" -c "import sys; print(len(sys.stdin.read()))"'
        if IS_WINDOWS
        else f'"{sys.executable}" -c "import sys; print(len(sys.stdin.read()))"'
    )
    result = await run(tool_ctx, command, timeout=20.0)

    assert result.ok is True
    assert "0" in result.text
