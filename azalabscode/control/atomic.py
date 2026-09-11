"""Atomic file replacement, and the Windows retry loop that makes it one.

`providers/models_cache.py` has the short version of this: `mkstemp` in the
destination directory, write, `os.replace`. That is enough for a cache, where a
failed write is a cache miss. It is not enough for a checkpoint.

Three things are different here, and each of them is a failure that has been
observed rather than imagined:

* **`os.replace` fails on Windows when the destination has an open handle.** Not
  rarely -- an editor with `session.json` open, a `tail`, the harness's own reader,
  or the previous write's handle still closing. `os.replace` raises
  `PermissionError` (`WinError 5`) and the checkpoint is lost. The retry loop is
  mandatory, not defensive: a bounded, jittered wait is the only thing between "the
  user had the file open" and "the run cannot be saved".

* **The temp file lives in the destination directory.** `os.replace` is atomic only
  within a volume, and the system temp directory is routinely on another one. The
  prefix is a dot plus the destination's name, so a leftover is attributable.

* **Nothing is left behind.** Every exit path unlinks the temp file, including the
  one where `os.replace` exhausted its retries. Over a thousand writes the
  destination directory holds exactly one file. `sweep_temp_files` exists for the
  case that is not reachable from here: a temp file orphaned by a process death
  between `mkstemp` and `replace`.

The documented limit: there is no directory `fsync` on Windows, so a power loss can
lose the newest checkpoint. It can never corrupt one -- the destination is only ever
replaced by a file that has already been written and flushed.
"""

from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path

from azalabscode.errors import CheckpointError

DEFAULT_ATTEMPTS = 24
"""Replace attempts before giving up. With the backoff below this is ~2.5 s."""

INITIAL_BACKOFF_S = 0.002
MAX_BACKOFF_S = 0.2
JITTER = 0.25

TEMP_SUFFIX = ".tmp"

_RETRYABLE_WINERRORS = frozenset({5, 32, 33})
"""Access denied, sharing violation, lock violation. The three ways a live handle
on the destination shows up as an `OSError` from `os.replace`."""


def _is_retryable(error: OSError) -> bool:
    """Whether `error` from `os.replace` is worth waiting out.

    A `PermissionError` is always one: on Windows it is what an open handle on the
    destination produces, and on POSIX it means the directory is not writable, which
    a retry costs nothing to confirm. The `winerror` set catches the sharing and
    lock violations that arrive as a plain `OSError`.
    """

    if isinstance(error, PermissionError):
        return True
    winerror = getattr(error, "winerror", None)
    return winerror in _RETRYABLE_WINERRORS


def _backoff(attempt: int, *, initial: float, maximum: float) -> float:
    """Exponential backoff with jitter, so two writers do not resonate."""

    base = min(maximum, initial * (2**attempt))
    return base * (1.0 - JITTER * random.random())


def atomic_write_bytes(
    path: str | os.PathLike[str],
    data: bytes,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    initial_backoff_s: float = INITIAL_BACKOFF_S,
    max_backoff_s: float = MAX_BACKOFF_S,
    fsync: bool = True,
) -> Path:
    """Write `data` to `path` so that a reader sees either the old file or the new one.

    Blocking. Call it from a thread (`atomic_write_bytes_async`) when a run is live:
    a 2 MB checkpoint written on the event loop is a 2 MB stall in the UI (R-U-4).

    Raises `CheckpointError` if the replace could not be completed within
    `attempts`; the temp file is removed either way.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}-", suffix=TEMP_SUFFIX
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        _replace_with_retry(
            tmp,
            target,
            attempts=attempts,
            initial_backoff_s=initial_backoff_s,
            max_backoff_s=max_backoff_s,
        )
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def _replace_with_retry(
    tmp: Path,
    target: Path,
    *,
    attempts: int,
    initial_backoff_s: float,
    max_backoff_s: float,
) -> None:
    """`os.replace` in a bounded jittered retry loop. The Windows half of the job."""

    last: OSError | None = None
    started = time.monotonic()
    for attempt in range(max(1, attempts)):
        try:
            os.replace(tmp, target)
        except OSError as error:
            if not _is_retryable(error):
                raise
            last = error
            time.sleep(_backoff(attempt, initial=initial_backoff_s, maximum=max_backoff_s))
        else:
            return

    waited = time.monotonic() - started
    raise CheckpointError(
        f"could not replace {target} after {attempts} attempt(s) over {waited:.1f}s; "
        f"the destination is held open by another process ({last})"
    ) from last


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str = "\n",
    **kwargs: object,
) -> Path:
    """`atomic_write_bytes` for text, writing exactly the line endings given."""

    body = text.replace("\r\n", "\n").replace("\n", newline) if newline != "\n" else text
    return atomic_write_bytes(path, body.encode(encoding), **kwargs)  # type: ignore[arg-type]


async def atomic_write_bytes_async(
    path: str | os.PathLike[str],
    data: bytes,
    **kwargs: object,
) -> Path:
    """`atomic_write_bytes` on a worker thread.

    The checkpoint lock is held across this call, which is deliberate: two agents
    must not interleave their writes. The event loop stays free, so a fan-out
    workflow's other agents keep streaming while the file goes down.
    """

    import asyncio

    return await asyncio.to_thread(atomic_write_bytes, path, data, **kwargs)  # type: ignore[arg-type]


def sweep_temp_files(directory: str | os.PathLike[str], *, older_than_s: float = 0.0) -> int:
    """Delete orphaned `.tmp` files in `directory`. Returns how many went.

    Nothing this module writes leaves one behind. This is for the case it cannot
    control: a process killed between `mkstemp` and `os.replace`, which is exactly
    what the kill test does on purpose.
    """

    root = Path(directory)
    if not root.is_dir():
        return 0
    now = time.time()
    removed = 0
    for entry in root.iterdir():
        if not entry.name.startswith(".") or entry.suffix != TEMP_SUFFIX:
            continue
        try:
            if older_than_s and now - entry.stat().st_mtime < older_than_s:
                continue
            entry.unlink()
        except OSError:  # pragma: no cover - a temp file another process still holds
            continue
        removed += 1
    return removed


__all__ = [
    "DEFAULT_ATTEMPTS",
    "INITIAL_BACKOFF_S",
    "MAX_BACKOFF_S",
    "TEMP_SUFFIX",
    "atomic_write_bytes",
    "atomic_write_bytes_async",
    "atomic_write_text",
    "sweep_temp_files",
]
