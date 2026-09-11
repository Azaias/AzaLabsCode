"""`shell`: run a command, with output merged into one file and the tree killed on exit.

Two design points from the plan, both about things that are wrong in the obvious
implementation:

**Delta 10 -- one output file, not two pipes.** stdout and stderr on separate pipes
cannot be interleaved chronologically: the reader sees whichever pipe it polls, and
an error line ends up in the wrong place relative to the output it explains. Both
streams go to a single append-mode fd instead. That gives correct ordering for free,
takes the drain loop off the hot path (a command producing 50 MB does not need a
Python coroutine keeping up with it), and makes the over-cap case a `seek` rather
than a re-read.

**Delta 9 -- the kill is a tree kill.** Killing the direct child leaves the actual
work running, still holding the output file open. `platform.kill_tree` does the
POSIX process-group signal or the Windows `taskkill /T`; `on_cancel` is what the
dispatcher calls when the *dispatcher's* timeout fires.

Retries are hard-blocked (`hard_block_retry`). A command that half-applied cannot be
undone by running it again.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from azalabscode.events import ToolCallProgress
from azalabscode.permissions import ApprovalPolicy, ApprovalSummary
from azalabscode.toolio import (
    NO_RETRY,
    RetryPolicy,
    ToolDisplay,
    ToolError,
    ToolErrorKind,
    ToolResult,
)
from azalabscode.tools.base import Tool
from azalabscode.tools.context import ToolContext, ToolPathError
from azalabscode.tools.platform import (
    CURRENT_PLATFORM,
    ShellNotFound,
    kill_tree,
    resolve_shell,
    shell_hint,
    spawn,
    split_env,
)

MAX_TIMEOUT_S = 600.0
DEFAULT_TIMEOUT_S = 120.0
TIMEOUT_GRACE_S = 5.0
"""Head-room between the tool's own timeout and the dispatcher's, so the tool gets to
kill the tree and report a proper `exit_status` rather than being cancelled."""

PROGRESS_TAIL_CHARS = 400


def _description() -> str:
    """The model-facing description, rendered for this platform (delta 9).

    A model told nothing about the shell writes `ls | head` on a Windows box and gets
    a confusing failure. The shell is named, and its idioms are stated.
    """

    return f"""\
Run a shell command and return its combined output.

{shell_hint()}

stdout and stderr come back interleaved in the order they were produced, which is \
what you want for reading a build log. The exit code is reported separately: a \
non-zero exit is a failure, and you still get the output.

The command runs with no stdin. Anything interactive -- a prompt, a pager, a \
confirmation -- will hang until the timeout kills it, so pass the non-interactive \
flags (`-y`, `--yes`, `--no-pager`, `--non-interactive`).

Timeout defaults to {DEFAULT_TIMEOUT_S:g}s and is capped at {MAX_TIMEOUT_S:g}s. On \
timeout the whole process tree is killed, not just the shell, and you get the output \
produced so far.

Prefer the dedicated tools where they exist: read_file over `cat`, edit_file over \
`sed -i`, glob over `find`, grep over `grep`. They are faster, they respect the \
workspace boundary, and their output is structured. Use shell for the things only a \
shell does: builds, tests, version control, package managers.

Never chain destructive commands behind a `&&` to save a turn. Each one should be \
approvable on its own.\
"""


class ShellParams(BaseModel):
    """Parameters for `shell`."""

    model_config = {"extra": "forbid"}

    command: str = Field(description="The command line to run.")
    cwd: str | None = Field(
        default=None, description="Working directory, relative to the workspace root."
    )
    timeout: float = Field(
        default=DEFAULT_TIMEOUT_S,
        gt=0,
        le=MAX_TIMEOUT_S,
        description=f"Seconds before the process tree is killed (max {MAX_TIMEOUT_S:g}).",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables, merged over the current ones.",
    )


class ShellTool(Tool):
    """Run a command under the platform's shell."""

    name: ClassVar[str] = "shell"
    description: ClassVar[str] = _description()
    Params: ClassVar[type[BaseModel]] = ShellParams

    approval: ApprovalPolicy = "always"
    timeout: float = MAX_TIMEOUT_S + TIMEOUT_GRACE_S
    retry: RetryPolicy = NO_RETRY
    concurrency_safe: ClassVar[bool] = False
    read_only: ClassVar[bool] = False
    hard_block_retry: ClassVar[bool] = True

    def __init__(self, *, platform: str | None = None) -> None:
        self._platform = platform or CURRENT_PLATFORM
        self._live: dict[str, Any] = {}
        """`call_id -> process`, so `on_cancel` can kill a tree it did not spawn."""
        self._orphans: dict[str, Path] = {}
        """`call_id -> output file` for a cancelled call, cleaned up by `on_cancel`."""
        super().__init__()

    def timeout_for(self, params: BaseModel) -> float:
        """The dispatcher's bound: the per-call timeout plus grace.

        The tool enforces the real timeout itself, because it has to kill the tree and
        return the partial output. The dispatcher's timeout is the backstop for a tool
        that fails to do that.
        """

        assert isinstance(params, ShellParams)
        return min(params.timeout, MAX_TIMEOUT_S) + TIMEOUT_GRACE_S

    async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
        """Check the working directory and that a shell exists, before prompting."""

        assert isinstance(params, ShellParams)
        if not params.command.strip():
            return ToolError(kind=ToolErrorKind.INVALID_PARAMS, message="command must not be empty")
        if params.cwd is not None:
            try:
                cwd = ctx.resolve_path(params.cwd)
            except ToolPathError as exc:
                return exc.as_tool_error()
            if not cwd.is_dir():
                return ToolError(
                    kind=ToolErrorKind.NOT_FOUND,
                    message=f"working directory does not exist: {params.cwd}",
                )
        try:
            resolve_shell(ctx.config.shell_program, platform=self._platform)  # type: ignore[arg-type]
        except ShellNotFound as exc:
            return ToolError(kind=ToolErrorKind.UNAVAILABLE, message=str(exc))
        return None

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Spawn, poll, wait, kill on timeout, report."""

        assert isinstance(params, ShellParams)
        cwd = ctx.resolve_path(params.cwd) if params.cwd else ctx.workspace_root
        limit = min(params.timeout, ctx.config.shell_max_timeout_s, MAX_TIMEOUT_S)
        shell = resolve_shell(ctx.config.shell_program, platform=self._platform)  # type: ignore[arg-type]

        fd, out_name = tempfile.mkstemp(prefix="azalabscode-shell-", suffix=".log")
        out_path = Path(out_name)
        started = time.monotonic()
        proc = None
        timed_out = False

        try:
            proc = await spawn(
                params.command,
                output=fd,
                cwd=cwd,
                env=split_env(ctx.env, params.env),
                shell=shell,
                platform=self._platform,  # type: ignore[arg-type]
            )
            self._live[str(ctx.call_id)] = proc
            timed_out = not await self._wait_with_progress(proc, ctx, out_path, limit)
            if timed_out:
                await kill_tree(proc, platform=self._platform)  # type: ignore[arg-type]
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), 5.0)
        except FileNotFoundError as exc:
            os.close(fd)
            out_path.unlink(missing_ok=True)  # noqa: ASYNC240 - a single stat/unlink on a local temp file; a thread hop costs more than the call
            return ToolResult.failure(
                ToolErrorKind.UNAVAILABLE, f"cannot start shell {shell.program!r}: {exc}"
            )
        except OSError as exc:
            with contextlib.suppress(OSError):
                os.close(fd)
            out_path.unlink(missing_ok=True)  # noqa: ASYNC240 - a single stat/unlink on a local temp file; a thread hop costs more than the call
            self._live.pop(str(ctx.call_id), None)
            return ToolResult.failure(ToolErrorKind.INTERNAL, f"could not run the command: {exc}")
        except asyncio.CancelledError:
            # Deliberately leave the entry in `_live`. The dispatcher's `on_cancel`
            # hook is what kills the tree, and it runs *outside* this task -- a
            # `finally` that popped the entry here would race it, and whether the
            # process died would depend on which coroutine the loop resumed first.
            with contextlib.suppress(OSError):
                os.close(fd)
            self._orphans[str(ctx.call_id)] = out_path
            raise
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)

        self._live.pop(str(ctx.call_id), None)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        output = _read_output(out_path)
        out_path.unlink(missing_ok=True)  # noqa: ASYNC240 - a single stat/unlink on a local temp file; a thread hop costs more than the call
        code = proc.returncode if proc is not None else None

        return self._result(params, output, code, elapsed_ms, timed_out, shell.label, ctx)

    async def _wait_with_progress(
        self, proc: Any, ctx: ToolContext, out_path: Path, limit: float
    ) -> bool:
        """Wait for the process, emitting a progress event roughly every second.

        Returns True if it exited within `limit`, False on timeout. Polling the file
        tail is cheap and, unlike draining a pipe, cannot deadlock: the process is
        never blocked on a reader that stopped reading.
        """

        interval = max(0.05, ctx.config.shell_progress_interval_s)
        deadline = time.monotonic() + limit
        offset = 0
        waiter = asyncio.ensure_future(proc.wait())
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                try:
                    await asyncio.wait_for(asyncio.shield(waiter), min(interval, remaining))
                    return True
                except TimeoutError:
                    offset = await self._emit_progress(ctx, out_path, offset)
        finally:
            if not waiter.done():
                waiter.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await waiter

    async def _emit_progress(self, ctx: ToolContext, out_path: Path, offset: int) -> int:
        """Emit whatever has been appended since `offset`. Returns the new offset."""

        if ctx.emit is None:
            return offset
        try:
            size = out_path.stat().st_size  # noqa: ASYNC240 - a single stat/unlink on a local temp file; a thread hop costs more than the call
        except OSError:
            return offset
        if size <= offset:
            return offset
        try:
            with out_path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read(size - offset)
        except OSError:  # pragma: no cover
            return offset
        text = chunk.decode("utf-8", "replace")
        await ctx.progress(
            ToolCallProgress(
                call_id=str(ctx.call_id),
                tool=self.name,
                text=text[-PROGRESS_TAIL_CHARS:],
            )
        )
        return size

    async def on_cancel(self, params: BaseModel, ctx: ToolContext, reason: str) -> None:
        """Kill the tree when the *dispatcher* cancels or times the call out.

        The tool's own timeout path never gets here. This is the backstop for an
        interrupt or a dispatcher timeout, which cancel the task while the process is
        still running.

        The process is also *reaped* here. A killed-but-unwaited child leaves an
        `asyncio` transport open, which surfaces later as a `ResourceWarning`
        attributed to whatever happened to be running when the collector got to it.
        """

        # Deregister and kill *first*. The temp-file cleanup below can raise on
        # Windows -- the process still holds the log open until it is actually dead --
        # and doing it first would abort this hook before the kill, leaving the
        # command running and the entry in `_live`.
        key = str(ctx.call_id)
        proc = self._live.pop(key, None)
        leftover = self._orphans.pop(key, None)

        if proc is not None:
            with contextlib.suppress(Exception):
                await kill_tree(proc, platform=self._platform)  # type: ignore[arg-type]
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), 5.0)
            _close_transport(proc)

        if leftover is not None:
            with contextlib.suppress(OSError):
                leftover.unlink(missing_ok=True)

    def _result(
        self,
        params: ShellParams,
        output: str,
        code: int | None,
        elapsed_ms: float,
        timed_out: bool,
        shell_label: str,
        ctx: ToolContext,
    ) -> ToolResult:
        display = ToolDisplay(
            kind="shell",
            data={
                "command": params.command,
                "exit_code": code,
                "output": output[-8000:],
                "timed_out": timed_out,
                "shell": shell_label,
            },
        )
        meta = {
            "command": params.command,
            "exit_code": code,
            "elapsed_ms": elapsed_ms,
            "shell": shell_label,
            "timed_out": timed_out,
        }

        if timed_out:
            body = (
                f"Command timed out after {params.timeout:g}s and its process tree was "
                f"killed.\n\nOutput before the kill:\n{output or '(none)'}"
            )
            result = ToolResult.failure(ToolErrorKind.TIMEOUT, body, meta=meta)
            result.display = display
            return result

        if code:
            body = f"Command exited with code {code}.\n\n{output or '(no output)'}"
            result = ToolResult(
                ok=False,
                content=ToolResult.ok_text(body).content,
                error=ToolError(
                    kind=ToolErrorKind.EXIT_STATUS,
                    message=f"exit code {code}",
                    details={"exit_code": code},
                ),
                display=display,
                meta=meta,
            )
            return result

        return ToolResult.ok_text(
            output if output else "(command produced no output; exit code 0)",
            display=display,
            meta=meta,
        )

    def approval_summary(self, params: BaseModel, ctx: ToolContext) -> ApprovalSummary:
        """The command itself, verbatim. There is nothing more useful to show."""

        assert isinstance(params, ShellParams)
        where = params.cwd or ctx.display_path(ctx.workspace_root)
        return ApprovalSummary(
            title=f"shell: {params.command.splitlines()[0][:120]}",
            detail=f"in {where}, timeout {params.timeout:g}s\n\n{params.command}",
            danger=True,
        )


def _close_transport(proc: Any) -> None:
    """Close the subprocess transport of a reaped process.

    `asyncio.subprocess.Process` has no public close. Killing a child without
    closing its transport leaves a `ResourceWarning` that surfaces at an arbitrary
    later point, which under `filterwarnings = ["error"]` fails an unrelated test.
    """

    transport = getattr(proc, "_transport", None)
    if transport is not None:
        with contextlib.suppress(Exception):
            transport.close()


def _read_output(path: Path) -> str:
    """Read the merged output file, tolerating any encoding."""

    try:
        raw = path.read_bytes()
    except OSError:  # pragma: no cover
        return ""
    return raw.decode("utf-8", "replace").replace("\r\n", "\n")


__all__ = ["DEFAULT_TIMEOUT_S", "MAX_TIMEOUT_S", "ShellParams", "ShellTool"]
