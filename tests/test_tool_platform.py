"""`tools/platform.py`: shell resolution, spawning, and `kill_tree` on both platforms.

The exit test for M1 asks for `kill_tree` tested on Windows *and* POSIX. Only one of
those can run for real on any given machine, so the coverage is split in two:

- **Both branches, always.** `kill_tree` takes `platform`, `killpg`, `getpgid` and
  `taskkill` as injected seams, so the POSIX signal escalation and the Windows
  `taskkill` escalation are both driven here regardless of host. This is what
  actually pins the *logic*: SIGTERM then SIGKILL after the grace window, taskkill
  then `proc.kill()`, and `already_exited` for a process that is gone.
- **The host branch, for real.** A genuine `pwsh`/`sh` process tree is spawned and
  killed, asserting the child is dead afterwards. The other platform's real-syscall
  path is marked `skipif` and is recorded as unverified in progress.md.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from azalabscode.tools.platform import (
    CURRENT_PLATFORM,
    SIGKILL_NUM,
    SIGTERM_NUM,
    KillOutcome,
    Platform,
    ShellNotFound,
    ShellSpec,
    creation_kwargs,
    describe_platform,
    kill_tree,
    resolve_shell,
    shell_hint,
    spawn,
    split_env,
)

IS_WINDOWS = os.name == "nt"


# ---------------------------------------------------------------------------
# A process double, so both branches are drivable on either host
# ---------------------------------------------------------------------------


class FakeProc:
    """A process that exits only when told to, or after N kill signals.

    `returncode` starts as `None` (running). `die_after` is how many kill attempts it
    takes to actually stop, which is what distinguishes the escalation path from the
    polite one.
    """

    def __init__(self, pid: int = 4242, *, die_after: int = 1, returncode: int | None = None):
        self.pid = pid
        self.returncode = returncode
        self.die_after = die_after
        self.kills = 0
        self.signals: list[int] = []
        self.hard_killed = False

    def receive(self, sig: int) -> None:
        self.signals.append(sig)
        self.kills += 1
        if self.kills >= self.die_after:
            self.returncode = -sig

    def kill(self) -> None:
        self.hard_killed = True
        self.returncode = -9

    async def wait(self) -> int:
        # Never returns while running: the caller's grace window is what times out.
        while self.returncode is None:
            await asyncio.sleep(0.005)
        return self.returncode


# ---------------------------------------------------------------------------
# resolve_shell
# ---------------------------------------------------------------------------


def test_windows_prefers_pwsh_then_powershell_then_cmd() -> None:
    present = {"pwsh": "C:/pwsh.exe", "powershell": "C:/ps.exe", "cmd": "C:/cmd.exe"}
    spec = resolve_shell(platform="windows", which=lambda n: present.get(n))
    assert spec.program == "C:/pwsh.exe"

    del present["pwsh"]
    assert resolve_shell(platform="windows", which=lambda n: present.get(n)).program == "C:/ps.exe"

    del present["powershell"]
    spec = resolve_shell(platform="windows", which=lambda n: present.get(n))
    assert spec.program == "C:/cmd.exe"
    assert spec.args == ("/d", "/s", "/c")


def test_posix_prefers_bash_then_sh() -> None:
    present = {"bash": "/bin/bash", "sh": "/bin/sh"}
    assert resolve_shell(platform="posix", which=lambda n: present.get(n)).program == "/bin/bash"
    del present["bash"]
    assert resolve_shell(platform="posix", which=lambda n: present.get(n)).program == "/bin/sh"


def test_a_preferred_shell_wins_when_it_resolves() -> None:
    present = {"pwsh": "C:/pwsh.exe", "cmd": "C:/cmd.exe"}
    spec = resolve_shell("cmd", platform="windows", which=lambda n: present.get(n))
    assert spec.program == "C:/cmd.exe"


def test_no_shell_at_all_raises_rather_than_returning_something_broken() -> None:
    with pytest.raises(ShellNotFound):
        resolve_shell(platform="windows", which=lambda n: None)


def test_creation_kwargs_differ_by_platform() -> None:
    """POSIX groups by session; Windows by process group. Neither flag exists on the
    other platform, so mixing them is an immediate TypeError from the spawner."""

    assert creation_kwargs("posix") == {"start_new_session": True}
    assert "creationflags" in creation_kwargs("windows")
    assert "start_new_session" not in creation_kwargs("windows")


def test_the_description_names_the_shell_the_model_is_talking_to() -> None:
    spec = ShellSpec(program="pwsh", args=("-Command",), label="PowerShell 7 (pwsh)")
    assert "PowerShell 7" in describe_platform(platform="windows", shell=spec)
    assert "Windows" in describe_platform(platform="windows", shell=spec)
    assert "2>$null" in shell_hint(platform="windows", shell=spec)
    assert "2>/dev/null" in shell_hint(platform="posix")


def test_cmd_gets_a_different_hint_from_powershell() -> None:
    cmd = ShellSpec(program="cmd", args=("/c",), label="cmd.exe")
    hint = shell_hint(platform="windows", shell=cmd)
    assert "findstr" in hint
    assert "PowerShell" not in hint


def test_split_env_merges_over_the_base_rather_than_replacing_it() -> None:
    merged = split_env({"PATH": "/usr/bin", "KEEP": "1"}, {"EXTRA": "2"})
    assert merged == {"PATH": "/usr/bin", "KEEP": "1", "EXTRA": "2"}
    assert split_env({"A": "1"}, None) == {"A": "1"}


def test_split_env_defaults_to_the_real_environment() -> None:
    """Dropping PATH would make every command fail in a way that reads as the
    command's fault, so the default base is the live environment."""

    assert "PATH" in {k.upper() for k in split_env(None, {"X": "1"})}


# ---------------------------------------------------------------------------
# kill_tree -- POSIX branch, driven on any host
# ---------------------------------------------------------------------------


async def test_posix_kill_sends_sigterm_to_the_process_group() -> None:
    proc = FakeProc(pid=99, die_after=1)
    seen: list[tuple[int, int]] = []

    def killpg(pgid: int, sig: int) -> None:
        seen.append((pgid, sig))
        proc.receive(sig)

    outcome = await kill_tree(
        proc, platform="posix", grace=1.0, killpg=killpg, getpgid=lambda pid: 1234
    )

    assert seen == [(1234, SIGTERM_NUM)]
    assert outcome.method == "signal"
    assert outcome.escalated is False
    assert outcome.steps == ["SIGTERM"]


async def test_posix_kill_escalates_to_sigkill_after_the_grace_window() -> None:
    """A process that ignores SIGTERM is the reason the escalation exists."""

    proc = FakeProc(pid=99, die_after=2)
    seen: list[int] = []

    def killpg(pgid: int, sig: int) -> None:
        seen.append(sig)
        proc.receive(sig)

    outcome = await kill_tree(
        proc, platform="posix", grace=0.05, killpg=killpg, getpgid=lambda pid: 1234
    )

    assert seen == [SIGTERM_NUM, SIGKILL_NUM]
    assert outcome.escalated is True
    assert proc.returncode is not None


async def test_posix_kill_signals_the_group_not_the_pid() -> None:
    """Signalling the pid leaves the grandchildren running, which is the whole bug
    `start_new_session` plus `killpg` exists to prevent."""

    proc = FakeProc(pid=77, die_after=1)
    targets: list[int] = []

    def killpg(pgid: int, sig: int) -> None:
        targets.append(pgid)
        proc.receive(sig)

    await kill_tree(proc, platform="posix", grace=1.0, killpg=killpg, getpgid=lambda pid: 555)
    assert targets == [555]
    assert 77 not in targets


async def test_posix_kill_falls_back_to_the_pid_when_getpgid_fails() -> None:
    proc = FakeProc(pid=77, die_after=1)
    targets: list[int] = []

    def boom(pid: int) -> int:
        raise ProcessLookupError(pid)

    def killpg(pgid: int, sig: int) -> None:
        targets.append(pgid)
        proc.receive(sig)

    outcome = await kill_tree(proc, platform="posix", grace=1.0, killpg=killpg, getpgid=boom)
    assert targets == [77]
    assert "getpgid-failed" in outcome.steps


async def test_posix_kill_of_a_vanished_group_does_not_raise() -> None:
    """A race between the check and the signal is normal and must not fail the call."""

    proc = FakeProc(pid=77)

    def killpg(pgid: int, sig: int) -> None:
        raise ProcessLookupError(pgid)

    outcome = await kill_tree(
        proc, platform="posix", grace=0.05, killpg=killpg, getpgid=lambda pid: 5
    )
    assert outcome.method == "signal"
    assert any("SIGTERM-failed" in s for s in outcome.steps)


# ---------------------------------------------------------------------------
# kill_tree -- Windows branch, driven on any host
# ---------------------------------------------------------------------------


async def test_windows_kill_uses_taskkill_with_the_tree_flag() -> None:
    proc = FakeProc(pid=1234, die_after=1)
    calls: list[int] = []

    async def taskkill(pid: int) -> tuple[int, str]:
        calls.append(pid)
        proc.receive(SIGTERM_NUM)
        return 0, f"SUCCESS: sent termination signal to process {pid}"

    outcome = await kill_tree(proc, platform="windows", grace=1.0, taskkill=taskkill)

    assert calls == [1234]
    assert outcome.method == "taskkill"
    assert outcome.escalated is False
    assert "SUCCESS" in outcome.detail


async def test_windows_kill_falls_back_to_proc_kill_when_taskkill_does_not_land() -> None:
    proc = FakeProc(pid=1234, die_after=99)

    async def taskkill(pid: int) -> tuple[int, str]:
        return 128, "ERROR: the process could not be terminated"

    outcome = await kill_tree(proc, platform="windows", grace=0.05, taskkill=taskkill)

    assert proc.hard_killed is True
    assert outcome.escalated is True
    assert "proc.kill" in outcome.steps


async def test_windows_kill_survives_taskkill_being_missing() -> None:
    """A box without taskkill on PATH must still get the direct child killed."""

    proc = FakeProc(pid=1234, die_after=99)

    async def taskkill(pid: int) -> tuple[int, str]:
        raise FileNotFoundError("taskkill")

    outcome = await kill_tree(proc, platform="windows", grace=0.05, taskkill=taskkill)
    assert proc.hard_killed is True
    assert any("taskkill-failed" in s for s in outcome.steps)


# ---------------------------------------------------------------------------
# kill_tree -- shared behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", ["posix", "windows"])
async def test_killing_an_already_exited_process_is_a_no_op(platform: Platform) -> None:
    """Idempotence matters: the tool's timeout path and the dispatcher's `on_cancel`
    can both fire for one call."""

    proc = FakeProc(returncode=0)

    async def taskkill(pid: int) -> tuple[int, str]:  # pragma: no cover - must not run
        raise AssertionError("taskkill was called on a dead process")

    def killpg(pgid: int, sig: int) -> None:  # pragma: no cover - must not run
        raise AssertionError("killpg was called on a dead process")

    outcome = await kill_tree(
        proc, platform=platform, taskkill=taskkill, killpg=killpg, getpgid=lambda p: p
    )
    assert outcome == KillOutcome(method="already_exited", returncode=0)


# ---------------------------------------------------------------------------
# Real processes, on this host
# ---------------------------------------------------------------------------


async def _spawn_sleeper(seconds: float, fd: int) -> asyncio.subprocess.Process:
    """A real shell command that outlives the test unless it is killed."""

    if IS_WINDOWS:
        command = f"Start-Sleep -Seconds {seconds}"
    else:
        command = f"sleep {seconds}"
    return await spawn(command, output=fd, platform=CURRENT_PLATFORM)


async def test_spawn_merges_stdout_and_stderr_into_one_fd() -> None:
    """Delta 10: one fd for both streams, so ordering is the process's own."""

    fd, name = tempfile.mkstemp()
    try:
        if IS_WINDOWS:
            command = "Write-Output one; [Console]::Error.WriteLine('two'); Write-Output three"
        else:
            command = "echo one; echo two 1>&2; echo three"
        proc = await spawn(command, output=fd, platform=CURRENT_PLATFORM)
        await asyncio.wait_for(proc.wait(), 30)
        os.close(fd)
        text = Path(name).read_text(encoding="utf-8", errors="replace")
    finally:
        Path(name).unlink(missing_ok=True)

    assert "one" in text and "two" in text and "three" in text
    assert text.index("one") < text.index("two") < text.index("three")


async def test_spawn_gives_the_child_no_stdin() -> None:
    """An interactive command must fail fast rather than block until the timeout."""

    fd, name = tempfile.mkstemp()
    try:
        command = '$i = [Console]::In.ReadToEnd(); Write-Output "read:$i"' if IS_WINDOWS else "cat"
        proc = await spawn(command, output=fd, platform=CURRENT_PLATFORM)
        # Exits promptly on EOF rather than hanging.
        await asyncio.wait_for(proc.wait(), 30)
        os.close(fd)
    finally:
        Path(name).unlink(missing_ok=True)


@pytest.mark.slow
async def test_kill_tree_really_kills_a_real_process_on_this_host() -> None:
    """The host branch end to end: spawn a sleeper, kill it, assert it is gone."""

    fd, name = tempfile.mkstemp()
    try:
        proc = await _spawn_sleeper(60, fd)
        assert proc.returncode is None
        outcome = await kill_tree(proc, grace=5.0)
        await asyncio.wait_for(proc.wait(), 10)
        assert proc.returncode is not None
        assert outcome.method == ("taskkill" if IS_WINDOWS else "signal")
        os.close(fd)
    finally:
        Path(name).unlink(missing_ok=True)


@pytest.mark.slow
@pytest.mark.skipif(IS_WINDOWS, reason="POSIX-only: real killpg on a real session")
async def test_posix_kill_tree_reaps_a_grandchild_process() -> None:
    """The behaviour the whole POSIX branch exists for: a `sh` that started a `sleep`
    must not leave the `sleep` running when the `sh` is killed."""

    fd, name = tempfile.mkstemp()
    marker = Path(name).with_suffix(".marker")
    try:
        command = f"(sleep 60; touch {marker}) & echo started; wait"
        proc = await spawn(command, output=fd, platform="posix")
        await asyncio.sleep(0.5)
        await kill_tree(proc, platform="posix", grace=3.0)
        await asyncio.wait_for(proc.wait(), 10)
        await asyncio.sleep(1.0)
        assert not marker.exists(), "the grandchild survived the group kill"
        os.close(fd)
    finally:
        Path(name).unlink(missing_ok=True)
        marker.unlink(missing_ok=True)


@pytest.mark.slow
@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only: real taskkill /T on a real tree")
async def test_windows_kill_tree_reaps_a_grandchild_process() -> None:
    """`taskkill /T` walks the tree; killing only the shell would leave the child
    Python process running and holding the output file open."""

    fd, name = tempfile.mkstemp()
    marker = Path(name).with_suffix(".marker")
    py = sys.executable.replace("\\", "/")
    child = f"import time, pathlib; time.sleep(60); pathlib.Path(r'{marker}').write_text('x')"
    try:
        command = f"& '{py}' -c \"{child}\""
        proc = await spawn(command, output=fd, platform="windows")
        await asyncio.sleep(1.0)
        await kill_tree(proc, platform="windows", grace=5.0)
        await asyncio.wait_for(proc.wait(), 15)
        await asyncio.sleep(1.0)
        assert not marker.exists(), "the grandchild survived the tree kill"
        os.close(fd)
    finally:
        Path(name).unlink(missing_ok=True)
        marker.unlink(missing_ok=True)


def test_the_event_loop_policy_can_spawn_subprocesses() -> None:
    """Spec delta 22, restated here because this is the module that would break.

    `WindowsSelectorEventLoopPolicy` makes `create_subprocess_shell` raise
    `NotImplementedError`, and the failure reads as a `shell` bug.
    """

    from azalabscode.sync import assert_subprocess_capable_loop_policy

    assert_subprocess_capable_loop_policy()


def test_subprocess_creation_flags_are_accepted_by_the_real_spawner() -> None:
    """The flags are platform-specific; passing the wrong set is a TypeError at spawn
    time, which this catches without needing a real async loop."""

    kwargs = creation_kwargs(CURRENT_PLATFORM)
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    proc.wait(timeout=30)
    assert proc.returncode == 0
