"""OS strategy for spawning and killing subprocesses (spec delta 9).

Spec 7 specifies `start_new_session=True` and a `killpg` on the process group.
Both are POSIX-only, and the primary platform here is Windows. This module is the
seam: `resolve_shell()`, `spawn()` and `kill_tree()` each take an explicit
`platform` argument that defaults to the host, so the branch that cannot run here
is still reachable from a test with the syscalls injected.

Two behaviours differ and neither is incidental:

- **Grouping.** POSIX puts the child in a new session so a signal reaches the whole
  tree. Windows has no session concept; `CREATE_NEW_PROCESS_GROUP` detaches the
  child from the parent's Ctrl-C group, and the tree is walked by `taskkill /T`.
- **The shell.** `asyncio.create_subprocess_shell` hard-codes `cmd.exe` on Windows.
  We exec an explicit shell program instead -- `pwsh`, falling back to
  `powershell`, falling back to `cmd` -- so the command the model wrote is
  interpreted by the shell the model was told about.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

type Platform = Literal["posix", "windows"]

CURRENT_PLATFORM: Platform = "windows" if os.name == "nt" else "posix"
"""The host platform. Every function here takes it as an overridable default."""

DEFAULT_KILL_GRACE_S = 3.0
"""Seconds between the polite signal and the fatal one."""

SIGTERM_NUM = int(signal.SIGTERM)
SIGKILL_NUM = int(getattr(signal, "SIGKILL", 9))
"""Signal numbers as ints.

`signal.SIGKILL` does not exist on Windows, so naming it directly makes the POSIX
branch unimportable there -- and that branch has to stay drivable from a test on
either host. SIGKILL is 9 on every POSIX system; the fallback is only ever used by
a test driving the POSIX branch with an injected `killpg`.
"""


@dataclass(frozen=True)
class ShellSpec:
    """A shell program and the argument vector that makes it run one command string.

    `label` is what the model is told it is talking to; it appears in the `shell`
    tool description, which is rendered per platform for exactly this reason.
    """

    program: str
    args: tuple[str, ...]
    label: str

    def argv(self, command: str) -> list[str]:
        """The full argument vector for running `command`."""

        return [self.program, *self.args, command]


# Non-interactive flags matter more than they look. A profile that prints a banner
# corrupts the merged output stream, and a shell that stops to prompt turns a 120 s
# timeout into a guaranteed kill.
_PWSH_ARGS = ("-NoLogo", "-NoProfile", "-NonInteractive", "-Command")
_CMD_ARGS = ("/d", "/s", "/c")
_SH_ARGS = ("-c",)

_WINDOWS_CANDIDATES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("pwsh", _PWSH_ARGS, "PowerShell 7 (pwsh)"),
    ("powershell", _PWSH_ARGS, "Windows PowerShell 5.1"),
    ("cmd", _CMD_ARGS, "cmd.exe"),
)

_POSIX_CANDIDATES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("bash", _SH_ARGS, "bash"),
    ("sh", _SH_ARGS, "sh"),
)


class ShellNotFound(RuntimeError):
    """No usable shell on PATH. Only reachable on a very unusual box."""


def resolve_shell(
    preferred: str | None = None,
    *,
    platform: Platform = CURRENT_PLATFORM,
    which: Callable[[str], str | None] = shutil.which,
) -> ShellSpec:
    """Pick the shell to run commands through.

    `preferred` names a program (`"pwsh"`, `"bash"`, an absolute path); it wins if it
    resolves. Otherwise the platform's candidate list is walked in order. `which` is
    injected so a test can pretend `pwsh` is missing without touching PATH.
    """

    candidates = _WINDOWS_CANDIDATES if platform == "windows" else _POSIX_CANDIDATES

    if preferred:
        for name, args, label in candidates:
            if Path(preferred).name.lower() in (name, f"{name}.exe"):
                found = which(preferred) or which(name)
                if found:
                    return ShellSpec(program=found, args=args, label=label)
        found = which(preferred)
        if found:
            args = _PWSH_ARGS if platform == "windows" else _SH_ARGS
            return ShellSpec(program=found, args=args, label=Path(found).name)

    for name, args, label in candidates:
        found = which(name)
        if found:
            return ShellSpec(program=found, args=args, label=label)

    if platform == "posix" and Path("/bin/sh").exists():
        return ShellSpec(program="/bin/sh", args=_SH_ARGS, label="sh")
    raise ShellNotFound(
        f"no shell found on PATH for platform {platform!r}; "
        f"tried {', '.join(name for name, _, _ in candidates)}"
    )


def creation_kwargs(platform: Platform = CURRENT_PLATFORM) -> dict[str, Any]:
    """The platform-specific keyword arguments that group a child for later killing.

    POSIX: a new session, so `killpg` reaches every descendant.
    Windows: a new process group, so the child does not receive the parent's Ctrl-C
    and `taskkill /T` has a group to walk.
    """

    if platform == "windows":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


async def spawn(
    command: str,
    *,
    output: int,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    shell: ShellSpec | None = None,
    platform: Platform = CURRENT_PLATFORM,
    exec_: Callable[..., Any] = asyncio.create_subprocess_exec,
) -> asyncio.subprocess.Process:
    """Start `command` under a shell, with stdout and stderr on one file descriptor.

    `output` is a single writable fd used for *both* streams (spec delta 10): two
    pipes cannot be interleaved chronologically, and one append-only fd gives
    correct ordering, takes the read loop off the hot path, and makes the over-cap
    case a `seek`.

    stdin is `/dev/null`. An interactive command must fail fast rather than block
    until the timeout kills it.
    """

    spec = shell or resolve_shell(platform=platform)
    return await exec_(
        *spec.argv(command),
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=output,
        cwd=os.fspath(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        **creation_kwargs(platform),
    )


@dataclass
class KillOutcome:
    """What `kill_tree` actually did, for the record and for tests.

    `method` is `"already_exited"`, `"signal"` (POSIX) or `"taskkill"` (Windows).
    `escalated` means the polite attempt did not land inside the grace window and
    the fatal one was used.
    """

    method: str
    escalated: bool = False
    returncode: int | None = None
    detail: str = ""
    steps: list[str] = field(default_factory=list)


async def _wait_briefly(proc: Any, limit: float) -> bool:
    """Wait up to `limit` seconds for the process to exit. True if it did."""

    if limit <= 0:
        return proc.returncode is not None
    try:
        await asyncio.wait_for(asyncio.shield(_wait(proc)), limit)
    except TimeoutError:
        return False
    return True


async def _wait(proc: Any) -> int:
    result = proc.wait()
    if asyncio.iscoroutine(result):
        return await result
    return result


async def _default_taskkill(pid: int) -> tuple[int, str]:
    """Run `taskkill /F /T /PID`, returning `(returncode, combined output)`."""

    exe = shutil.which("taskkill") or "taskkill"
    proc = await asyncio.create_subprocess_exec(
        exe,
        "/F",
        "/T",
        "/PID",
        str(pid),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode("utf-8", "replace").strip()


def _no_killpg(pid: int, sig: int) -> None:  # pragma: no cover - Windows only
    """Stand-in for `os.killpg` on Windows, where process groups are not signalled.

    The POSIX branch is never taken on Windows in production; a test that drives it
    injects a real fake. Raising here rather than silently succeeding means a
    mis-selected branch is loud.
    """

    raise NotImplementedError("os.killpg is not available on this platform")


_KILLPG: Callable[[int, int], None] = getattr(os, "killpg", _no_killpg)
_GETPGID: Callable[[int], int] = getattr(os, "getpgid", lambda pid: pid)


async def kill_tree(
    proc: Any,
    *,
    platform: Platform = CURRENT_PLATFORM,
    grace: float = DEFAULT_KILL_GRACE_S,
    killpg: Callable[[int, int], None] = _KILLPG,
    getpgid: Callable[[int], int] = _GETPGID,
    taskkill: Callable[[int], Any] = _default_taskkill,
) -> KillOutcome:
    """Terminate `proc` and every process it started.

    Killing only the direct child leaves the actual work running: a `pytest` under
    a `pwsh` under the harness is two levels down, and it keeps the output file
    open after the tool has reported a timeout.

    The syscall seams are injected so the branch for the platform this is not
    running on is still exercisable. Both branches are idempotent -- a process that
    is already gone returns `already_exited` rather than raising.
    """

    if proc.returncode is not None:
        return KillOutcome(method="already_exited", returncode=proc.returncode)

    pid = proc.pid
    if platform == "windows":
        return await _kill_tree_windows(proc, pid, grace=grace, taskkill=taskkill)
    return await _kill_tree_posix(proc, pid, grace=grace, killpg=killpg, getpgid=getpgid)


async def _kill_tree_posix(
    proc: Any,
    pid: int,
    *,
    grace: float,
    killpg: Callable[[int, int], None],
    getpgid: Callable[[int], int],
) -> KillOutcome:
    steps: list[str] = []
    try:
        pgid = getpgid(pid)
    except (ProcessLookupError, OSError):
        pgid = pid
        steps.append("getpgid-failed")

    try:
        killpg(pgid, SIGTERM_NUM)
        steps.append("SIGTERM")
    except (ProcessLookupError, OSError) as exc:
        # The group went away between the check and the signal. Nothing to escalate.
        steps.append(f"SIGTERM-failed:{type(exc).__name__}")
        with contextlib.suppress(Exception):
            await _wait_briefly(proc, 0.05)
        return KillOutcome(
            method="signal", returncode=proc.returncode, detail=str(exc), steps=steps
        )

    if await _wait_briefly(proc, grace):
        return KillOutcome(method="signal", returncode=proc.returncode, steps=steps)

    try:
        killpg(pgid, SIGKILL_NUM)
        steps.append("SIGKILL")
    except (ProcessLookupError, OSError) as exc:
        steps.append(f"SIGKILL-failed:{type(exc).__name__}")
        return KillOutcome(
            method="signal",
            escalated=True,
            returncode=proc.returncode,
            detail=str(exc),
            steps=steps,
        )

    await _wait_briefly(proc, grace)
    return KillOutcome(method="signal", escalated=True, returncode=proc.returncode, steps=steps)


async def _kill_tree_windows(
    proc: Any,
    pid: int,
    *,
    grace: float,
    taskkill: Callable[[int], Any],
) -> KillOutcome:
    steps: list[str] = []
    detail = ""
    try:
        rc, out = await taskkill(pid)
        steps.append(f"taskkill:{rc}")
        detail = out
    except Exception as exc:
        steps.append(f"taskkill-failed:{type(exc).__name__}")
        detail = str(exc)

    if await _wait_briefly(proc, grace):
        return KillOutcome(
            method="taskkill", returncode=proc.returncode, detail=detail, steps=steps
        )

    # taskkill did not land -- fall back to terminating the direct child, which at
    # least stops it holding the output file open.
    with contextlib.suppress(Exception):
        proc.kill()
        steps.append("proc.kill")
    await _wait_briefly(proc, grace)
    return KillOutcome(
        method="taskkill",
        escalated=True,
        returncode=proc.returncode,
        detail=detail,
        steps=steps,
    )


def describe_platform(
    *,
    platform: Platform = CURRENT_PLATFORM,
    shell: ShellSpec | None = None,
) -> str:
    """One line naming the shell the model's commands will run under.

    Interpolated into the `shell` tool description so the model does not write
    `ls | head` on a box where the shell is `pwsh` (spec delta 9).
    """

    try:
        spec = shell or resolve_shell(platform=platform)
        label = spec.label
    except ShellNotFound:  # pragma: no cover - no shell on PATH
        label = "an unknown shell"
    os_name = "Windows" if platform == "windows" else "a POSIX system"
    return f"{os_name}, commands run through {label}"


def shell_hint(*, platform: Platform = CURRENT_PLATFORM, shell: ShellSpec | None = None) -> str:
    """Platform-specific guidance appended to the `shell` tool description."""

    spec_label = (shell or _try_resolve(platform)).label
    if platform == "windows":
        if "cmd" in spec_label.lower():
            return (
                "This is cmd.exe: use `dir`, `type`, `findstr`, and `&&` for chaining. "
                "POSIX syntax such as `ls`, `cat`, `2>/dev/null` or `$(...)` will not work."
            )
        return (
            "This is PowerShell: `ls`/`cat`/`rm` are aliases with different flags, "
            "redirect with `2>$null` rather than `2>/dev/null`, use `$env:NAME` for "
            "environment variables, and prefer `Select-Object -First N` over `head`."
        )
    return (
        "This is a POSIX shell: normal `sh` syntax applies, including pipes, "
        "`2>/dev/null` and `$(...)`."
    )


def _try_resolve(platform: Platform) -> ShellSpec:
    try:
        return resolve_shell(platform=platform)
    except ShellNotFound:  # pragma: no cover - no shell on PATH
        return ShellSpec(program="sh", args=_SH_ARGS, label="an unknown shell")


def split_env(
    base: Mapping[str, str] | None, overrides: Mapping[str, str] | None
) -> dict[str, str]:
    """Merge `overrides` onto `base` (defaulting to the current environment).

    A tool never replaces the environment wholesale: dropping PATH would make every
    command fail in a way that reads as the command's fault.
    """

    out = dict(base if base is not None else os.environ)
    if overrides:
        out.update({str(k): str(v) for k, v in overrides.items()})
    return out


def as_command_list(argv: Sequence[str]) -> str:
    """Render an argv for a human-readable log line."""

    return " ".join(str(a) for a in argv)


__all__ = [
    "CURRENT_PLATFORM",
    "DEFAULT_KILL_GRACE_S",
    "SIGKILL_NUM",
    "SIGTERM_NUM",
    "KillOutcome",
    "Platform",
    "ShellNotFound",
    "ShellSpec",
    "as_command_list",
    "creation_kwargs",
    "describe_platform",
    "kill_tree",
    "resolve_shell",
    "shell_hint",
    "spawn",
    "split_env",
]
