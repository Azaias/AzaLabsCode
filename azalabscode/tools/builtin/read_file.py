"""`read_file`: the tool every other file tool depends on.

It is also the tool that sets the read-before-write state (spec delta 13), which is
what stops `write_file` and `edit_file` clobbering contents the model has never
seen. A read that fails does not record anything.

Two deliberate departures from spec 7:

- **The byte cap errors rather than truncates** (delta 8). A ~100-byte error telling
  the model to narrow its window costs almost nothing; 400 000 characters of clipped
  file costs 100k tokens and is usually the wrong 400 000 characters anyway.
- **`max_result_size_chars` is `inf`.** Spilling a file read to disk that the model
  then has to `read_file` back is circular. The line window and the byte cap are the
  real bounds.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from azalabscode.content import ImagePart, Part, TextPart
from azalabscode.permissions import ApprovalPolicy, ApprovalSummary
from azalabscode.toolio import NO_RETRY, RetryPolicy, ToolDisplay, ToolErrorKind, ToolResult
from azalabscode.tools.base import Tool
from azalabscode.tools.context import ToolContext, ToolPathError

DESCRIPTION = """\
Read a text file from the workspace and return its contents with line numbers.

Use this before editing any file: write_file and edit_file both refuse to touch a \
file you have not read, because a change made from a guess about the contents \
overwrites whatever is actually there.

Output is `   12| text`, one line per source line. The line numbers are for your \
reference when choosing an offset; they are not part of the file, so never include \
them in the `old` string you pass to edit_file.

Reads at most `limit` lines starting at line `offset` (1-based). The default window \
is 2000 lines. If the file is longer you get a note saying so and the offset to use \
for the next page.

Also reads images (png, jpg, gif, webp), which are returned to you as an image \
rather than as text. Other binary files are refused: read them with the shell if \
you really need to.

Common mistakes: passing a directory instead of a file; passing a line range as \
`offset=10, limit=20` when you meant lines 10 to 20 (that is `offset=10, limit=11`); \
re-reading an unchanged file you have already read this session.\
"""

MAX_LINE_CHARS = 2_000
"""A single line longer than this is clipped. A minified bundle is one 3 MB line and
would otherwise defeat the line window entirely."""

DEFAULT_LIMIT = 2_000

_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Extensions worth refusing before reading a byte. The null-byte scan below is the
# real test; this just avoids loading a 2 GB archive to discover it is binary.
_BINARY_EXTENSIONS = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".bin",
        ".o",
        ".a",
        ".lib",
        ".zip",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".tar",
        ".jar",
        ".war",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wav",
        ".flac",
        ".ogg",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".pack",
        ".idx",
        ".class",
        ".wasm",
    }
)

_NULL_SCAN_BYTES = 8_192


class ReadFileParams(BaseModel):
    """Parameters for `read_file`."""

    model_config = {"extra": "forbid"}

    path: str = Field(description="File to read, absolute or relative to the workspace root.")
    offset: int = Field(default=1, ge=1, description="1-based line number to start reading from.")
    limit: int = Field(
        default=DEFAULT_LIMIT, ge=1, le=50_000, description="Maximum number of lines to return."
    )


class ReadFileTool(Tool):
    """Read a text or image file, line-numbered, within a bounded window."""

    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = DESCRIPTION
    Params: ClassVar[type[BaseModel]] = ReadFileParams

    approval: ApprovalPolicy = "never"
    timeout: float = 10.0
    retry: RetryPolicy = NO_RETRY
    max_result_size_chars: int | float = float("inf")
    concurrency_safe: ClassVar[bool] = True
    read_only: ClassVar[bool] = True

    async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
        """Reject the call before it runs for anything a path check can see."""

        assert isinstance(params, ReadFileParams)
        try:
            path = ctx.resolve_path(params.path)
        except ToolPathError as exc:
            return exc.as_tool_error()

        if path.is_dir():
            return self._error(
                ToolErrorKind.INVALID_PARAMS,
                f"{ctx.display_path(path)} is a directory, not a file. Use glob to list "
                f"its contents.",
            )
        if not path.exists():
            return self._error(
                ToolErrorKind.NOT_FOUND,
                f"no such file: {params.path}. Check the path with glob before reading.",
            )
        if path.is_symlink() and not path.resolve().exists():
            return self._error(ToolErrorKind.NOT_FOUND, f"{params.path} is a broken symlink")
        return None

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Read the file."""

        assert isinstance(params, ReadFileParams)
        path = ctx.resolve_path(params.path)
        rel = ctx.display_path(path)

        try:
            stat = path.stat()
        except OSError as exc:
            return ToolResult.failure(ToolErrorKind.NOT_FOUND, f"cannot stat {rel}: {exc}")

        if not stat.st_size and path.suffix.lower() not in _IMAGE_TYPES:
            ctx.note_read(path, offset=params.offset, limit=params.limit)
            return ToolResult.ok_text(
                f"{rel} is empty (0 bytes).",
                display=ToolDisplay(kind="file", data={"path": rel, "lines": 0, "empty": True}),
                meta={"path": str(path), "bytes": 0, "lines": 0},
            )

        suffix = path.suffix.lower()
        if suffix in _IMAGE_TYPES:
            return self._read_image(path, rel, stat.st_size, suffix, ctx)

        if suffix in _BINARY_EXTENSIONS:
            return ToolResult.failure(
                ToolErrorKind.UNSUPPORTED,
                f"{rel} is a binary file ({suffix}); read_file only handles text and images",
            )

        try:
            raw = path.read_bytes()
        except OSError as exc:
            return ToolResult.failure(ToolErrorKind.PERMISSION, f"cannot read {rel}: {exc}")

        if b"\x00" in raw[:_NULL_SCAN_BYTES]:
            return ToolResult.failure(
                ToolErrorKind.UNSUPPORTED,
                f"{rel} contains null bytes and appears to be binary; read_file only "
                f"handles text and images",
            )

        text, encoding, warning = _decode(raw)
        lines, line_ending = _split_lines(text)
        total = len(lines)

        start = params.offset - 1
        if start >= total and total > 0:
            return ToolResult.failure(
                ToolErrorKind.INVALID_PARAMS,
                f"offset {params.offset} is past the end of {rel}, which has {total} lines",
            )
        window = lines[start : start + params.limit]
        rendered, clipped = _render(window, start)

        if len(rendered) > ctx.config.max_read_bytes:
            return ToolResult.failure(
                ToolErrorKind.INVALID_PARAMS,
                f"lines {params.offset}-{start + len(window)} of {rel} are "
                f"{len(rendered)} characters, over the {ctx.config.max_read_bytes} "
                f"character limit for one read. Re-read with a smaller limit "
                f"(try limit={max(1, params.limit // 4)}), or use grep to find the "
                f"part you need.",
            )

        notes: list[str] = []
        if warning:
            notes.append(warning)
        if clipped:
            notes.append(
                f"{clipped} line(s) were longer than {MAX_LINE_CHARS} characters and were clipped."
            )
        end = start + len(window)
        truncated = end < total
        if truncated:
            notes.append(
                f"Showing lines {params.offset}-{end} of {total}. Continue with offset={end + 1}."
            )

        body = rendered
        if notes:
            body = f"{rendered}\n\n[{' '.join(notes)}]" if rendered else f"[{' '.join(notes)}]"

        ctx.note_read(path, offset=params.offset, limit=params.limit, truncated=truncated)

        return ToolResult.ok_text(
            body,
            display=ToolDisplay(
                kind="file",
                data={
                    "path": rel,
                    "offset": params.offset,
                    "lines": len(window),
                    "total_lines": total,
                    "truncated": truncated,
                    "encoding": encoding,
                    "line_ending": line_ending,
                },
            ),
            meta={
                "path": str(path),
                "bytes": stat.st_size,
                "lines": len(window),
                "total_lines": total,
                "encoding": encoding,
                "truncated": truncated,
            },
        )

    def _read_image(
        self, path: Path, rel: str, size: int, suffix: str, ctx: ToolContext
    ) -> ToolResult:
        if size > ctx.config.max_image_bytes:
            return ToolResult.failure(
                ToolErrorKind.UNSUPPORTED,
                f"{rel} is {size} bytes, over the {ctx.config.max_image_bytes}-byte image limit",
            )
        try:
            data = path.read_bytes()
        except OSError as exc:
            return ToolResult.failure(ToolErrorKind.PERMISSION, f"cannot read {rel}: {exc}")

        ctx.note_read(path)
        parts: list[Part] = [
            TextPart(text=f"{rel} ({size} bytes, {_IMAGE_TYPES[suffix]}):"),
            ImagePart(
                media_type=_IMAGE_TYPES[suffix],
                data_b64=base64.b64encode(data).decode("ascii"),
            ),
        ]
        return ToolResult(
            ok=True,
            content=parts,
            display=ToolDisplay(kind="image", data={"path": rel, "bytes": size}),
            meta={"path": str(path), "bytes": size, "media_type": _IMAGE_TYPES[suffix]},
        )

    @staticmethod
    def _error(kind: ToolErrorKind, message: str) -> Any:
        from azalabscode.toolio import ToolError

        return ToolError(kind=kind, message=message)

    def approval_summary(self, params: BaseModel, ctx: ToolContext) -> ApprovalSummary:
        """`read_file` is never gated, but a UI may still render this."""

        assert isinstance(params, ReadFileParams)
        return ApprovalSummary(title=f"read_file {params.path}", detail="", danger=False)


def _decode(raw: bytes) -> tuple[str, str, str]:
    """Decode bytes to text, returning `(text, encoding, warning)`.

    UTF-8 first, latin-1 as the fallback with a warning. Latin-1 decodes any byte
    sequence, so the fallback never fails -- which means the warning is the only
    signal that the text may be wrong, and it is attached to the result.
    """

    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8"), "utf-8", ""
    except UnicodeDecodeError:
        return (
            raw.decode("latin-1"),
            "latin-1",
            "This file is not valid UTF-8; it was decoded as latin-1 and some "
            "characters may be wrong.",
        )


def _split_lines(text: str) -> tuple[list[str], str]:
    """Split into lines and report the dominant line ending.

    The ending is reported rather than normalised away, because `write_file` and
    `edit_file` need it: rewriting a CRLF file with LF endings is a whole-file diff
    that hides the one line that actually changed.
    """

    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    ending = "\r\n" if crlf > lf else "\n"
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalised.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines, ending


def _render(lines: list[str], start: int) -> tuple[str, int]:
    """Line-number the window. Returns `(text, clipped line count)`."""

    width = max(4, len(str(start + len(lines))))
    out: list[str] = []
    clipped = 0
    for i, line in enumerate(lines, start=start + 1):
        if len(line) > MAX_LINE_CHARS:
            line = f"{line[:MAX_LINE_CHARS]}... [clipped, {len(line)} chars]"
            clipped += 1
        out.append(f"{i:>{width}}| {line}")
    return "\n".join(out), clipped


__all__ = ["DESCRIPTION", "MAX_LINE_CHARS", "ReadFileParams", "ReadFileTool"]
