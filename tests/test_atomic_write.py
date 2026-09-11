"""The atomic-write suite, aimed squarely at Windows.

Three things are checked, and the first two are Windows-specific failures rather
than theoretical ones:

1. **A held handle on the destination does not lose the write.** `os.replace` raises
   `PermissionError` (WinError 5) when the destination is open in another handle --
   an editor, a `tail`, the harness's own reader. Without the retry loop the
   checkpoint is simply gone. `test_a_replace_over_a_held_handle_fails_without_a_retry`
   is the control: it shows the bare `os.replace` failing on the same inputs, so the
   retry test is not measuring nothing.

2. **A thousand writes leave one file.** Every exit path unlinks the temp file, so
   the destination directory never accumulates `.tmp` residue, and a reader that
   globs the directory never sees a partial checkpoint.

3. **A reader sees the old file or the new one, never a splice.** The temp file lives
   in the destination directory, because `os.replace` is atomic only within a volume.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from azalabscode.control.atomic import (
    TEMP_SUFFIX,
    atomic_write_bytes,
    atomic_write_bytes_async,
    atomic_write_text,
    sweep_temp_files,
)
from azalabscode.errors import CheckpointError

WINDOWS = sys.platform == "win32"


def residue(directory: Path) -> list[str]:
    """Every temp file left behind in `directory`."""

    return sorted(p.name for p in directory.iterdir() if p.suffix == TEMP_SUFFIX)


# ---------------------------------------------------------------------------
# The basics
# ---------------------------------------------------------------------------


def test_a_write_lands_and_leaves_nothing_behind(tmp_path: Path) -> None:
    """The happy path, stated so a failure elsewhere is never ambiguous."""

    target = tmp_path / "session.json"
    atomic_write_bytes(target, b'{"run_id": "abc"}')

    assert target.read_bytes() == b'{"run_id": "abc"}'
    assert [p.name for p in tmp_path.iterdir()] == ["session.json"]


def test_it_creates_the_destination_directory(tmp_path: Path) -> None:
    """A first checkpoint should not need the directory to exist already."""

    target = tmp_path / "runs" / "abc" / "session.json"
    atomic_write_bytes(target, b"{}")

    assert target.read_bytes() == b"{}"


def test_the_temp_file_is_in_the_destination_directory(tmp_path: Path) -> None:
    """`os.replace` is atomic only within a volume, and the system temp dir is often
    on another one. Observed by watching the directory while the write runs."""

    target = tmp_path / "session.json"
    seen: list[list[str]] = []

    original = os.replace

    def watching(src, dst):  # type: ignore[no-untyped-def]
        seen.append(sorted(p.name for p in tmp_path.iterdir()))
        return original(src, dst)

    import azalabscode.control.atomic as atomic

    atomic.os.replace = watching  # type: ignore[assignment]
    try:
        atomic_write_bytes(target, b"payload")
    finally:
        atomic.os.replace = original  # type: ignore[assignment]

    assert seen, "os.replace was never called"
    assert any(name.endswith(TEMP_SUFFIX) for name in seen[0]), seen[0]


def test_text_is_written_with_the_line_endings_it_was_given(tmp_path: Path) -> None:
    """No universal-newline translation: a checkpoint is bytes, not a text document."""

    target = tmp_path / "notes.txt"
    atomic_write_text(target, "a\nb\n")
    assert target.read_bytes() == b"a\nb\n"

    atomic_write_text(target, "a\nb\n", newline="\r\n")
    assert target.read_bytes() == b"a\r\nb\r\n"


# ---------------------------------------------------------------------------
# The Windows retry loop
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not WINDOWS, reason="a held handle only blocks os.replace on Windows")
def test_a_replace_over_a_held_handle_fails_without_a_retry(tmp_path: Path) -> None:
    """The control. Without this, the retry test could be passing for no reason."""

    target = tmp_path / "session.json"
    target.write_text("old", encoding="utf-8")
    source = tmp_path / "new.tmp"
    source.write_text("new", encoding="utf-8")

    with open(target, encoding="utf-8") as _held, pytest.raises(PermissionError):
        os.replace(source, target)

    assert target.read_text(encoding="utf-8") == "old"


async def test_a_write_waits_out_a_held_handle_and_still_lands(tmp_path: Path) -> None:
    """The retry loop, against a handle that is released while the write is trying.

    The handle is closed from a timer thread rather than after a fixed wait, so the
    test does not depend on how many retries happen to fit in a window -- which is
    the same reason the kill test uses a marker file instead of a sleep.
    """

    target = tmp_path / "session.json"
    target.write_bytes(b"old")

    handle = open(target, "rb")  # noqa: SIM115, ASYNC230 - held open on purpose, closed below
    released = threading.Timer(0.15, handle.close)
    released.start()
    try:
        started = time.monotonic()
        await atomic_write_bytes_async(target, b"new")
        elapsed = time.monotonic() - started
    finally:
        released.cancel()
        handle.close()

    assert target.read_bytes() == b"new"
    assert residue(tmp_path) == []
    if WINDOWS:
        assert elapsed >= 0.1, "the write cannot have succeeded before the handle was released"


def test_a_handle_that_is_never_released_raises_rather_than_hanging(tmp_path: Path) -> None:
    """A bounded retry loop. `CheckpointError` names the destination and the wait.

    The alternative -- retrying forever -- turns "somebody left the file open" into
    a run that never checkpoints again and never says why.
    """

    target = tmp_path / "session.json"
    target.write_bytes(b"old")

    with open(target, "rb"):
        if not WINDOWS:
            pytest.skip("a held handle does not block os.replace on this platform")
        with pytest.raises(CheckpointError) as caught:
            atomic_write_bytes(target, b"new", attempts=4, max_backoff_s=0.01)

    assert "session.json" in str(caught.value)
    assert target.read_bytes() == b"old", "a failed write must not damage the old file"
    assert residue(tmp_path) == [], "a failed write must not leave a temp file"


# ---------------------------------------------------------------------------
# Residue over volume
# ---------------------------------------------------------------------------


def test_a_thousand_writes_leave_exactly_one_file(tmp_path: Path) -> None:
    """The exit test's wording. One file at the end, and it is the last thing written."""

    target = tmp_path / "session.json"
    for index in range(1000):
        atomic_write_bytes(target, f'{{"seq": {index}}}'.encode(), fsync=False)

    assert [p.name for p in tmp_path.iterdir()] == ["session.json"]
    assert target.read_bytes() == b'{"seq": 999}'


async def test_a_thousand_concurrent_writes_leave_exactly_one_file(tmp_path: Path) -> None:
    """The same, from many tasks at once. Each write is a whole document or nothing.

    The controller serializes its own writes under the checkpoint lock, so this is
    about the primitive rather than about the controller: two threads racing on
    `os.replace` must still leave one intact file, not a splice of two.
    """

    target = tmp_path / "session.json"
    payloads = [f'{{"seq": {i}}}'.encode() for i in range(200)]

    await asyncio.gather(
        *(atomic_write_bytes_async(target, payload, fsync=False) for payload in payloads)
    )

    assert [p.name for p in tmp_path.iterdir()] == ["session.json"]
    assert target.read_bytes() in payloads, "the file is a splice of two writes"


def test_a_reader_never_sees_a_partial_document(tmp_path: Path) -> None:
    """Interleave reads with writes of different lengths and check every read parses."""

    import json

    target = tmp_path / "session.json"
    atomic_write_bytes(target, json.dumps({"seq": 0, "pad": ""}).encode())

    for index in range(1, 300):
        payload = json.dumps({"seq": index, "pad": "x" * (index * 37)}).encode()
        atomic_write_bytes(target, payload, fsync=False)
        loaded = json.loads(target.read_bytes())
        assert loaded["seq"] == index
        assert len(loaded["pad"]) == index * 37


# ---------------------------------------------------------------------------
# The orphan sweep
# ---------------------------------------------------------------------------


def test_the_sweep_removes_orphans_and_nothing_else(tmp_path: Path) -> None:
    """For the one case the write path cannot control: a death between mkstemp and
    replace, which is exactly what the kill test does on purpose."""

    (tmp_path / "session.json").write_bytes(b"{}")
    (tmp_path / ".session.json-abc.tmp").write_bytes(b"half")
    (tmp_path / ".session.json-def.tmp").write_bytes(b"half")
    (tmp_path / "values").mkdir()

    assert sweep_temp_files(tmp_path) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["session.json", "values"]


def test_sweeping_a_directory_that_is_not_there_is_not_an_error(tmp_path: Path) -> None:
    """A run that has never checkpointed has no directory to sweep."""

    assert sweep_temp_files(tmp_path / "nope") == 0
