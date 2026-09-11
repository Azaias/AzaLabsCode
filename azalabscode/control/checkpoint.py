"""Turning a `Session` into bytes on disk, and large values into files beside it.

The split between this module and `atomic.py` is the split between *what* is
written and *how*. `Checkpointer` decides the paths, the spilling and the
serialization; `atomic` guarantees that a reader sees either the old file or the new
one and never a half-written one.

The serialization is deliberately a **separate call** from the write. `_fold` runs
synchronously under the checkpoint lock with no awaits -- a fold that suspended
would let another agent interleave one of its own -- so `serialize()` is called
there and `write_async()` after it, still inside the lock. The bytes handed to the
thread are therefore a snapshot of the moment the fold ended, not of whenever the
thread happened to be scheduled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from azalabscode.control.atomic import atomic_write_bytes, atomic_write_bytes_async
from azalabscode.control.session import (
    SPILL_THRESHOLD_BYTES,
    VALUES_DIRNAME,
    Session,
    ValueRef,
    session_file,
)


class Checkpointer:
    """Where a session is written, and what is small enough to go inside it.

    A checkpointer with no `session_dir` and no `path` still works: `serialize()`
    returns bytes and `value_ref()` keeps everything inline. That is the in-memory
    run, and it is the default -- a run only touches disk when it is told where.
    """

    def __init__(
        self,
        *,
        session_dir: str | Path | None = None,
        path: str | Path | None = None,
        spill_threshold: int = SPILL_THRESHOLD_BYTES,
        indent: int | None = 2,
    ) -> None:
        self.session_dir = Path(session_dir) if session_dir is not None else None
        self._explicit_path = Path(path) if path is not None else None
        self.spill_threshold = spill_threshold
        self.indent = indent
        self.writes = 0
        """How many checkpoints have gone to disk. What a status bar counts."""

    # -- paths --------------------------------------------------------------

    @property
    def path(self) -> Path | None:
        """Where an automatic checkpoint goes, or `None` for an in-memory run."""

        if self._explicit_path is not None:
            return self._explicit_path
        if self.session_dir is not None:
            return session_file(self.session_dir)
        return None

    @property
    def enabled(self) -> bool:
        """Whether safe points write to disk without being asked (R-C-10)."""

        return self.path is not None

    def dir_for(self, target: Path) -> Path:
        """The directory a spilled value belongs beside, given a destination file."""

        return self.session_dir if self.session_dir is not None else target.parent

    # -- values -------------------------------------------------------------

    def value_ref(self, value: Any) -> ValueRef:
        """Wrap a node input or output, spilling it to `values/` when it is large.

        Spilling needs a directory, so an in-memory run keeps everything inline
        regardless of size. That is the right trade: nothing is being written, so
        nothing is being kept small.
        """

        blob = json.dumps(value, default=str)
        size = len(blob.encode("utf-8"))
        if size <= self.spill_threshold or self.session_dir is None:
            return ValueRef(inline=value, size=size)

        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
        relative = f"{VALUES_DIRNAME}/{digest}.json"
        target = self.session_dir / relative
        if not target.exists():
            atomic_write_bytes(target, blob.encode("utf-8"))
        return ValueRef(path=relative, size=size)

    # -- the document -------------------------------------------------------

    def serialize(self, session: Session) -> bytes:
        """The bytes for `session`. Synchronous: it is called inside `_fold`."""

        return session.dumps(indent=self.indent)

    def write(self, data: bytes, path: str | Path | None = None) -> Path:
        """Write pre-serialized bytes. Blocking; prefer `write_async` in a run."""

        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no checkpoint path: pass one, or configure a session_dir")
        atomic_write_bytes(target, data)
        self.writes += 1
        return target

    async def write_async(self, data: bytes, path: str | Path | None = None) -> Path:
        """Write pre-serialized bytes on a worker thread.

        Called with the checkpoint lock held, which is what serializes two agents'
        writes. The loop stays free, so a 2 MB checkpoint does not stall the UI.
        """

        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no checkpoint path: pass one, or configure a session_dir")
        await atomic_write_bytes_async(target, data)
        self.writes += 1
        return target


__all__ = ["Checkpointer"]
