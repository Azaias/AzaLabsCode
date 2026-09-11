"""Atomic writes, line-ending preservation, and unified diffs.

Shared by `write_file` and `edit_file` rather than duplicated: the atomic-write
critical section is the one piece of this layer where a subtle difference between
two copies would be a data-loss bug.

The write is `mkstemp` in the **destination directory** -- `os.replace` is only
atomic within a volume, so a temp file in `%TEMP%` is a cross-volume move that is
not atomic at all -- then write, `fsync`, then `os.replace` in a bounded retry loop.
On Windows `os.replace` raises `PermissionError` while any handle to the destination
is open, which an editor, a virus scanner or a watcher does routinely. The retry is
mandatory there, not defensive.
"""

from __future__ import annotations

import difflib
import os
import tempfile
import time
from pathlib import Path

REPLACE_ATTEMPTS = 8
REPLACE_INITIAL_DELAY_S = 0.01
MAX_DIFF_LINES = 400
"""Diff lines kept in an approval summary. A 20 000-line diff is not reviewable and
would dominate a model request."""


def detect_line_ending(text: str) -> str:
    """The dominant line ending in `text`. Defaults to `\\n` for text with none."""

    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def apply_line_ending(text: str, ending: str) -> str:
    """Re-encode `text`'s line endings as `ending`, from any starting mix."""

    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    if ending == "\n":
        return normalised
    return normalised.replace("\n", ending)


def read_text(path: Path) -> tuple[str, str, str]:
    """Read a file for editing: `(text, encoding, line ending)`.

    Text is returned with the file's original line endings intact, because an edit
    has to match what is actually in the file.
    """

    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        encoding = "latin-1"
    return text, encoding, detect_line_ending(text)


def atomic_write(path: Path, data: bytes, *, attempts: int = REPLACE_ATTEMPTS) -> int:
    """Write `data` to `path` atomically. Returns the byte count.

    The temp file is created in the destination directory so the final `os.replace`
    is a same-volume rename. On failure the temp file is removed; a crashed write
    never leaves a partial `path` and never leaves residue.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, path, attempts=attempts)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return len(data)


def _replace_with_retry(src: Path, dst: Path, *, attempts: int) -> None:
    """`os.replace` with a bounded backoff for Windows' open-handle `PermissionError`."""

    delay = REPLACE_INITIAL_DELAY_S
    last: OSError | None = None
    for _ in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:  # Windows: destination handle is open
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 0.5)
        except OSError as exc:
            last = exc
            break
    raise OSError(
        f"could not replace {dst} after {attempts} attempts: {last}. "
        f"Another process is holding the file open."
    ) from last


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> int:
    """Write `text` verbatim -- line endings included -- atomically."""

    return atomic_write(path, text.encode(encoding, errors="strict"))


def unified_diff(
    before: str,
    after: str,
    *,
    path: str,
    context: int = 3,
    max_lines: int = MAX_DIFF_LINES,
) -> str:
    """A unified diff of two texts, bounded in length.

    Line endings are normalised before diffing so a CRLF file does not produce a
    diff in which every line differs. The write itself preserves the real endings.
    """

    a = before.replace("\r\n", "\n").splitlines(keepends=True)
    b = after.replace("\r\n", "\n").splitlines(keepends=True)
    lines = list(difflib.unified_diff(a, b, fromfile=f"a/{path}", tofile=f"b/{path}", n=context))
    if not lines:
        return ""
    if len(lines) > max_lines:
        kept = lines[:max_lines]
        kept.append(f"... [{len(lines) - max_lines} more diff lines omitted]\n")
        lines = kept
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)


def diff_stats(before: str, after: str) -> dict[str, int]:
    """Added and removed line counts, for `ToolResult.display` and telemetry."""

    a = before.replace("\r\n", "\n").splitlines()
    b = after.replace("\r\n", "\n").splitlines()
    added = removed = 0
    for line in difflib.ndiff(a, b):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return {"added": added, "removed": removed}


def nearest_candidate(haystack: str, needle: str, *, limit: int = 3) -> list[str]:
    """Lines that look like `needle` but are not it, for a failed exact match.

    Whitespace-normalised comparison, because the overwhelmingly common cause of a
    failed `edit_file` is indentation the model reconstructed rather than copied.
    """

    target = " ".join(needle.split())
    if not target:
        return []
    first = target.split("\n")[0][:120]
    scored: list[tuple[float, str]] = []
    for line in haystack.replace("\r\n", "\n").split("\n"):
        normalised = " ".join(line.split())
        if not normalised:
            continue
        ratio = difflib.SequenceMatcher(None, normalised, first).ratio()
        if ratio > 0.6:
            scored.append((ratio, line))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [line for _, line in scored[:limit]]


__all__ = [
    "MAX_DIFF_LINES",
    "REPLACE_ATTEMPTS",
    "apply_line_ending",
    "atomic_write",
    "detect_line_ending",
    "diff_stats",
    "nearest_candidate",
    "read_text",
    "unified_diff",
    "write_text_atomic",
]
