"""`edit_file`: exact string replacement, unique unless `replace_all`.

Uniqueness is the whole design. A model that supplies a string appearing three times
does not know which one it means, and picking the first is how an edit lands in the
wrong function. The error names the count and, when the match failed entirely, the
nearest whitespace-normalised candidates -- because a reconstructed indentation is
the overwhelmingly common cause and the model cannot see it in its own output.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

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
from azalabscode.tools.fileio import (
    apply_line_ending,
    diff_stats,
    nearest_candidate,
    read_text,
    unified_diff,
    write_text_atomic,
)

DESCRIPTION = """\
Replace an exact string in a file with another one.

Read the file first. edit_file refuses to touch a file you have not read this \
session, or one that changed after you read it.

`old` must appear exactly once. If it appears zero times or more than once the edit \
is refused and you are told the count -- pass a longer `old` with surrounding lines \
until it is unique, or set replace_all=True if you genuinely mean every occurrence.

`old` must match the file byte for byte, apart from line endings, which are handled \
for you. Copy it out of the read_file output; do not retype it. Two things break \
this most often: including the `   12| ` line-number prefix from read_file (it is not \
in the file), and reconstructing indentation instead of copying it.

To delete text, pass an empty `new`. To insert, include an anchor line in both `old` \
and `new`.

The result and the approval prompt both show a unified diff of the change.\
"""


class EditFileParams(BaseModel):
    """Parameters for `edit_file`."""

    model_config = {"extra": "forbid"}

    path: str = Field(description="File to edit, absolute or relative to the workspace root.")
    old: str = Field(description="Exact text to replace. Must be unique unless replace_all.")
    new: str = Field(description="Replacement text. Empty string deletes the match.")
    replace_all: bool = Field(
        default=False, description="Replace every occurrence instead of requiring exactly one."
    )

    @model_validator(mode="after")
    def _old_is_not_empty(self) -> EditFileParams:
        if not self.old:
            raise ValueError("'old' must not be empty; use write_file to create a file")
        if self.old == self.new:
            raise ValueError("'old' and 'new' are identical; the edit would do nothing")
        return self


class EditFileTool(Tool):
    """Exact-match string replacement with a unified diff."""

    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = DESCRIPTION
    Params: ClassVar[type[BaseModel]] = EditFileParams

    approval: ApprovalPolicy = "always"
    timeout: float = 10.0
    retry: RetryPolicy = NO_RETRY
    concurrency_safe: ClassVar[bool] = False
    read_only: ClassVar[bool] = False

    async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
        """Everything that can fail without touching the file (delta 11).

        Match counting happens here too, so `manual` mode never prompts for an edit
        whose `old` string is not in the file.
        """

        assert isinstance(params, EditFileParams)
        try:
            path = ctx.resolve_path(params.path)
        except ToolPathError as exc:
            return exc.as_tool_error()

        rel = ctx.display_path(path)
        if path.is_dir():
            return ToolError(
                kind=ToolErrorKind.INVALID_PARAMS, message=f"{rel} is a directory, not a file"
            )
        if not path.exists():
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND,
                message=f"no such file: {params.path}. Use write_file to create it.",
            )

        stale = ctx.check_read_before_write(path)
        if stale is not None:
            return stale

        try:
            text, _, _ = read_text(path)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION, message=f"cannot read {rel}: {exc}")

        return _match_error(text, params, rel)

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Apply the edit."""

        assert isinstance(params, EditFileParams)
        path = ctx.resolve_path(params.path)
        rel = ctx.display_path(path)

        stale = ctx.check_read_before_write(path)
        if stale is not None:
            return ToolResult(
                ok=False,
                content=ToolResult.failure(stale.kind, stale.message).content,
                error=stale,
            )

        try:
            text, encoding, ending = read_text(path)
        except OSError as exc:
            return ToolResult.failure(ToolErrorKind.PERMISSION, f"cannot read {rel}: {exc}")

        error = _match_error(text, params, rel)
        if error is not None:
            return ToolResult(
                ok=False,
                content=ToolResult.failure(error.kind, error.message).content,
                error=error,
            )

        normalised = text.replace("\r\n", "\n")
        old = params.old.replace("\r\n", "\n")
        new = params.new.replace("\r\n", "\n")
        count = normalised.count(old)
        updated = (
            normalised.replace(old, new) if params.replace_all else normalised.replace(old, new, 1)
        )
        on_disk = apply_line_ending(updated, ending)

        try:
            written = write_text_atomic(path, on_disk, encoding=encoding)
        except (OSError, UnicodeEncodeError) as exc:
            return ToolResult.failure(ToolErrorKind.PERMISSION, f"cannot write {rel}: {exc}")

        ctx.read_state.clear(path)
        ctx.note_read(path)

        replaced = count if params.replace_all else 1
        diff = unified_diff(normalised, updated, path=rel)
        stats = diff_stats(normalised, updated)

        return ToolResult.ok_text(
            f"Edited {rel}: {replaced} replacement(s), "
            f"+{stats['added']}/-{stats['removed']} lines.\n\n{diff}",
            display=ToolDisplay(
                kind="diff",
                data={
                    "path": rel,
                    "diff": diff,
                    "replacements": replaced,
                    "added": stats["added"],
                    "removed": stats["removed"],
                },
            ),
            meta={
                "path": str(path),
                "bytes": written,
                "replacements": replaced,
                "encoding": encoding,
            },
        )

    def approval_summary(self, params: BaseModel, ctx: ToolContext) -> ApprovalSummary:
        """A unified diff of what the edit would do (R-C-6)."""

        assert isinstance(params, EditFileParams)
        try:
            path = ctx.resolve_path(params.path)
        except ToolPathError:
            return ApprovalSummary(
                title=f"edit_file {params.path}", detail="(path is invalid)", danger=True
            )
        rel = ctx.display_path(path)
        try:
            text, _, _ = read_text(path)
        except OSError:
            return ApprovalSummary(
                title=f"edit_file {rel}", detail="(file could not be read)", danger=True
            )

        normalised = text.replace("\r\n", "\n")
        old = params.old.replace("\r\n", "\n")
        new = params.new.replace("\r\n", "\n")
        count = normalised.count(old)
        updated = (
            normalised.replace(old, new) if params.replace_all else normalised.replace(old, new, 1)
        )
        stats = diff_stats(normalised, updated)
        return ApprovalSummary(
            title=f"edit_file {rel}",
            detail=f"{count} match(es), +{stats['added']}/-{stats['removed']} lines",
            diff=unified_diff(normalised, updated, path=rel),
            danger=True,
        )


def _match_error(text: str, params: EditFileParams, rel: str) -> ToolError | None:
    """Count matches and build the "not unique" / "not found" error (spec 7).

    Comparison is on LF-normalised text so a CRLF file does not fail every match.
    """

    normalised = text.replace("\r\n", "\n")
    old = params.old.replace("\r\n", "\n")
    count = normalised.count(old)

    if count == 0:
        candidates = nearest_candidate(normalised, old)
        hint = ""
        if candidates:
            shown = "\n".join(f"  {c}" for c in candidates)
            hint = (
                f"\nThe closest lines in the file are:\n{shown}\n"
                f"If one of these is what you meant, copy it exactly -- the difference "
                f"is probably whitespace."
            )
        return ToolError(
            kind=ToolErrorKind.INVALID_PARAMS,
            message=(
                f"the 'old' string does not appear in {rel}. Re-read the file and copy "
                f"the text exactly, without the read_file line-number prefix.{hint}"
            ),
            details={"path": rel, "matches": 0},
        )

    if count > 1 and not params.replace_all:
        return ToolError(
            kind=ToolErrorKind.INVALID_PARAMS,
            message=(
                f"the 'old' string appears {count} times in {rel}, so this edit is "
                f"ambiguous. Extend 'old' with the surrounding lines until it is unique, "
                f"or pass replace_all=True to change all {count}."
            ),
            details={"path": rel, "matches": count},
        )

    return None


__all__ = ["DESCRIPTION", "EditFileParams", "EditFileTool"]
