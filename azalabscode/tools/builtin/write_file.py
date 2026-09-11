"""`write_file`: create a file, or replace one whose current contents have been read.

The read-before-write rule (spec delta 13) is enforced twice, and the second time is
the one that matters. `validate_params` checks it before the approval prompt, so a
human is never asked to approve a write that was going to be refused. `run` re-checks
the mtime in a critical section with no awaits between the check and the rename, so a
file modified while the human was deciding is not silently overwritten.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from azalabscode.permissions import ApprovalPolicy, ApprovalSummary
from azalabscode.toolio import NO_RETRY, RetryPolicy, ToolDisplay, ToolErrorKind, ToolResult
from azalabscode.tools.base import Tool
from azalabscode.tools.context import ToolContext, ToolPathError
from azalabscode.tools.fileio import (
    apply_line_ending,
    detect_line_ending,
    diff_stats,
    read_text,
    unified_diff,
    write_text_atomic,
)

DESCRIPTION = """\
Write a complete file, creating it or replacing what is there.

Parent directories are created for you. The write is atomic: readers see either the \
old file or the new one, never a half-written one.

If the file already exists you must have read it first with read_file, and it must \
not have changed since. This is not a formality -- writing a file you have not read \
destroys whatever was in it, and "I know what is in that file" is wrong often enough \
to be worth a round trip.

Prefer edit_file for changing part of an existing file. write_file rewrites the whole \
thing, so a one-line change becomes a whole-file diff that is much harder to review \
and much easier to get wrong.

`content` is written exactly as given. Include the trailing newline if the file \
should have one.\
"""


class WriteFileParams(BaseModel):
    """Parameters for `write_file`."""

    model_config = {"extra": "forbid"}

    path: str = Field(description="File to write, absolute or relative to the workspace root.")
    content: str = Field(description="The complete new contents of the file.")


class WriteFileTool(Tool):
    """Create or replace a file atomically."""

    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = DESCRIPTION
    Params: ClassVar[type[BaseModel]] = WriteFileParams

    approval: ApprovalPolicy = "always"
    timeout: float = 10.0
    retry: RetryPolicy = NO_RETRY
    concurrency_safe: ClassVar[bool] = False
    read_only: ClassVar[bool] = False

    async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
        """Path, directory and read-before-write checks, before any prompt (delta 11)."""

        assert isinstance(params, WriteFileParams)
        try:
            path = ctx.resolve_path(params.path)
        except ToolPathError as exc:
            return exc.as_tool_error()

        if path.is_dir():
            from azalabscode.toolio import ToolError

            return ToolError(
                kind=ToolErrorKind.INVALID_PARAMS,
                message=f"{ctx.display_path(path)} is a directory, not a file",
            )
        return ctx.check_read_before_write(path)

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Write the file."""

        assert isinstance(params, WriteFileParams)
        path = ctx.resolve_path(params.path)
        rel = ctx.display_path(path)

        existed = path.exists()
        before = ""
        encoding = "utf-8"
        ending = detect_line_ending(params.content)
        if existed:
            # Re-check inside the critical section. Between validate_params and here
            # a human may have spent a minute in the approval modal.
            stale = ctx.check_read_before_write(path)
            if stale is not None:
                return ToolResult(
                    ok=False,
                    content=ToolResult.failure(stale.kind, stale.message).content,
                    error=stale,
                )
            try:
                before, encoding, ending = read_text(path)
            except OSError as exc:
                return ToolResult.failure(
                    ToolErrorKind.PERMISSION, f"cannot read {rel} before writing: {exc}"
                )

        # Honour the endings the model sent for a new file; preserve the file's own
        # for an existing one, so a one-line change is a one-line diff.
        content = params.content if not existed else apply_line_ending(params.content, ending)

        try:
            written = write_text_atomic(path, content, encoding=encoding)
        except (OSError, UnicodeEncodeError) as exc:
            return ToolResult.failure(ToolErrorKind.PERMISSION, f"cannot write {rel}: {exc}")

        ctx.read_state.clear(path)
        ctx.note_read(path)

        diff = unified_diff(before, content, path=rel) if existed else ""
        stats = diff_stats(before, content)
        verb = "Updated" if existed else "Created"
        lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)

        return ToolResult.ok_text(
            f"{verb} {rel} ({written} bytes, {lines} lines).",
            display=ToolDisplay(
                kind="diff",
                data={
                    "path": rel,
                    "existed": existed,
                    "diff": diff,
                    "bytes": written,
                    "added": stats["added"],
                    "removed": stats["removed"],
                },
            ),
            meta={
                "path": str(path),
                "bytes": written,
                "existed": existed,
                "lines": lines,
                "encoding": encoding,
            },
        )

    def approval_summary(self, params: BaseModel, ctx: ToolContext) -> ApprovalSummary:
        """Full content for a new file, a unified diff for an overwrite (R-C-6)."""

        assert isinstance(params, WriteFileParams)
        try:
            path = ctx.resolve_path(params.path)
        except ToolPathError:
            return ApprovalSummary(
                title=f"write_file {params.path}", detail="(path is invalid)", danger=True
            )
        rel = ctx.display_path(path)

        if not path.exists():
            preview = params.content[:2000]
            suffix = "" if len(params.content) <= 2000 else " ... [truncated]"
            return ApprovalSummary(
                title=f"write_file {rel} (new file)",
                detail=f"{len(params.content)} characters:\n{preview}{suffix}",
                danger=False,
            )

        try:
            before, _, ending = read_text(path)
        except OSError:
            before, ending = "", "\n"
        diff = unified_diff(before, apply_line_ending(params.content, ending), path=rel)
        return ApprovalSummary(
            title=f"write_file {rel} (overwrite)",
            detail=f"replaces {len(before)} characters with {len(params.content)}",
            diff=diff,
            danger=True,
        )


__all__ = ["DESCRIPTION", "WriteFileParams", "WriteFileTool"]
