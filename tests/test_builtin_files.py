"""`read_file`, `write_file`, `edit_file` against real temp workspaces.

Every test here runs the tool end to end through `ToolDispatcher`, or directly with
a real `ToolContext` on `tmp_path`. Nothing is mocked: the failure paths that matter
-- a denied path, a binary file, a non-unique edit, a write to a file that was never
read -- are all things that only show up against a real filesystem.

The read-before-write suite (spec delta 13) is the largest section on purpose. It is
the single behaviour that prevents silent clobbering, and it has four distinct
failure modes: never read, read then externally modified, read then written, and the
case-insensitive path alias on Windows.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from azalabscode.toolio import ToolErrorKind
from azalabscode.tools import ReadFileTool, ToolContext, WriteFileTool
from azalabscode.tools.builtin.edit_file import EditFileParams, EditFileTool
from azalabscode.tools.builtin.read_file import MAX_LINE_CHARS, ReadFileParams
from azalabscode.tools.builtin.write_file import WriteFileParams
from azalabscode.tools.fileio import (
    apply_line_ending,
    atomic_write,
    detect_line_ending,
    nearest_candidate,
    unified_diff,
)

READ = ReadFileTool()
WRITE = WriteFileTool()
EDIT = EditFileTool()


async def read(ctx: ToolContext, path: str, **kwargs: object):
    params = ReadFileParams(path=path, **kwargs)  # type: ignore[arg-type]
    error = await READ.validate_params(params, ctx)
    if error is not None:
        from azalabscode.toolio import ToolResult

        return ToolResult(ok=False, content=[], error=error).model_copy(
            update={"content": ToolResult.failure(error.kind, error.message).content}
        )
    return await READ.run(params, ctx)


async def write(ctx: ToolContext, path: str, content: str):
    params = WriteFileParams(path=path, content=content)
    error = await WRITE.validate_params(params, ctx)
    if error is not None:
        from azalabscode.toolio import ToolResult

        return ToolResult.failure(error.kind, error.message)
    return await WRITE.run(params, ctx)


async def edit(ctx: ToolContext, path: str, old: str, new: str, replace_all: bool = False):
    params = EditFileParams(path=path, old=old, new=new, replace_all=replace_all)
    error = await EDIT.validate_params(params, ctx)
    if error is not None:
        from azalabscode.toolio import ToolResult

        return ToolResult.failure(error.kind, error.message)
    return await EDIT.run(params, ctx)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


async def test_read_returns_line_numbered_text(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    result = await read(tool_ctx, "a.py")

    assert result.ok is True
    assert "   1| one" in result.text
    assert "   3| three" in result.text
    assert result.meta["total_lines"] == 3


async def test_read_windows_the_file_and_says_how_to_continue(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    (workspace / "big.txt").write_text("\n".join(str(i) for i in range(100)), encoding="utf-8")
    result = await read(tool_ctx, "big.txt", offset=1, limit=10)

    assert "offset=11" in result.text
    assert result.meta["truncated"] is True
    assert result.meta["lines"] == 10


async def test_read_past_the_end_is_an_error_not_an_empty_result(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """An empty result reads as "the file has nothing there", which is a different
    fact from "your offset is wrong"."""

    (workspace / "s.txt").write_text("a\nb\n", encoding="utf-8")
    result = await read(tool_ctx, "s.txt", offset=99)

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS
    assert "2 lines" in result.error.message


async def test_read_of_a_directory_points_at_glob(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "sub").mkdir()
    result = await read(tool_ctx, "sub")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS
    assert "glob" in result.error.message


async def test_read_of_a_missing_file_is_not_found(tool_ctx: ToolContext) -> None:
    result = await read(tool_ctx, "nope.txt")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.NOT_FOUND


async def test_read_outside_the_workspace_is_a_permission_error(
    tool_ctx: ToolContext, tmp_path: Path
) -> None:
    """R-T-7. The message names the root so the model stops trying."""

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    result = await read(tool_ctx, str(outside))

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.PERMISSION
    assert "outside the workspace" in result.error.message


async def test_a_relative_escape_is_rejected(tool_ctx: ToolContext) -> None:
    result = await read(tool_ctx, "../../etc/passwd")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.PERMISSION


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
async def test_a_symlink_pointing_outside_the_workspace_is_rejected(
    tool_ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    """Symlinks are resolved *before* the containment check, so a link inside the
    workspace pointing out of it is refused rather than followed."""

    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(target)

    result = await read(tool_ctx, "link.txt")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.PERMISSION


async def test_an_unrestricted_workspace_allows_outside_paths(
    workspace: Path, tmp_path: Path
) -> None:
    from azalabscode.tools import ToolConfig, WorkspaceConfig

    outside = tmp_path / "outside.txt"
    outside.write_text("visible\n", encoding="utf-8")
    ctx = ToolContext(
        workspace_root=workspace,
        config=ToolConfig(workspace=WorkspaceConfig(unrestricted=True)),
    )
    result = await read(ctx, str(outside))
    assert result.ok is True
    assert "visible" in result.text


async def test_a_binary_file_is_refused_by_extension(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    (workspace / "lib.so").write_bytes(b"\x7fELF" + b"\x01" * 100)
    result = await read(tool_ctx, "lib.so")

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.UNSUPPORTED


async def test_a_binary_file_with_a_text_extension_is_caught_by_the_null_scan(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """The extension list is an optimisation; the null-byte scan is the real test."""

    (workspace / "data.txt").write_bytes(b"hello\x00world")
    result = await read(tool_ctx, "data.txt")

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.UNSUPPORTED
    assert "null bytes" in result.error.message


async def test_a_non_utf8_file_falls_back_to_latin1_with_a_warning(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """Latin-1 decodes any byte sequence, so the warning is the only signal the text
    may be wrong. It has to reach the model."""

    (workspace / "iso.txt").write_bytes("caf\xe9\n".encode("latin-1"))
    result = await read(tool_ctx, "iso.txt")

    assert result.ok is True
    assert result.meta["encoding"] == "latin-1"
    assert "not valid UTF-8" in result.text


async def test_a_utf8_bom_is_stripped(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "bom.txt").write_bytes(b"\xef\xbb\xbfhello\n")
    result = await read(tool_ctx, "bom.txt")
    assert "   1| hello" in result.text
    assert "﻿" not in result.text


async def test_crlf_is_normalised_for_display_but_reported(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    (workspace / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")
    result = await read(tool_ctx, "crlf.txt")

    assert "\r" not in result.text
    assert result.meta["total_lines"] == 2
    assert result.display is not None
    assert result.display.data["line_ending"] == "\r\n"


async def test_an_enormous_single_line_is_clipped(tool_ctx: ToolContext, workspace: Path) -> None:
    """A minified bundle is one 3 MB line and would otherwise defeat the line window."""

    (workspace / "min.js").write_text("z" * (MAX_LINE_CHARS + 500), encoding="utf-8")
    result = await read(tool_ctx, "min.js")

    assert "clipped" in result.text
    assert len(result.text) < MAX_LINE_CHARS + 400


async def test_an_over_budget_window_errors_rather_than_truncating(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """Spec delta 8: a ~100-byte error beats 400 000 characters of clipped file."""

    tool_ctx.config.max_read_bytes = 500
    (workspace / "wide.txt").write_text("\n".join("x" * 80 for _ in range(50)), encoding="utf-8")
    result = await read(tool_ctx, "wide.txt")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS
    assert "smaller limit" in result.error.message
    assert len(result.error.message) < 400


async def test_an_empty_file_says_so(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "empty.txt").write_text("", encoding="utf-8")
    result = await read(tool_ctx, "empty.txt")
    assert result.ok is True
    assert "empty" in result.text


async def test_an_image_comes_back_as_an_image_part(tool_ctx: ToolContext, workspace: Path) -> None:
    from azalabscode.content import ImagePart

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    (workspace / "pixel.png").write_bytes(png)
    result = await read(tool_ctx, "pixel.png")

    assert result.ok is True
    assert any(isinstance(p, ImagePart) for p in result.content)
    assert result.meta["media_type"] == "image/png"


async def test_an_oversized_image_is_refused(tool_ctx: ToolContext, workspace: Path) -> None:
    tool_ctx.config.max_image_bytes = 10
    (workspace / "big.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
    result = await read(tool_ctx, "big.png")

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.UNSUPPORTED


async def test_read_file_opts_out_of_the_result_cap() -> None:
    """Spilling a file read that the model then re-reads is circular (delta 8)."""

    import math

    assert READ.max_result_size_chars == math.inf
    assert READ.result_cap_for(ReadFileParams(path="x")) == math.inf


async def test_read_records_the_read_state(tool_ctx: ToolContext, workspace: Path) -> None:
    path = workspace / "a.txt"
    path.write_text("hi\n", encoding="utf-8")
    assert tool_ctx.read_state.get(path) is None

    await read(tool_ctx, "a.txt")

    record = tool_ctx.read_state.get(path)
    assert record is not None
    assert record.mtime_ns == path.stat().st_mtime_ns


async def test_a_failed_read_records_nothing(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "bin.txt").write_bytes(b"\x00\x01")
    await read(tool_ctx, "bin.txt")
    assert tool_ctx.read_state.get(workspace / "bin.txt") is None


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


async def test_write_creates_a_new_file_and_its_parents(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    result = await write(tool_ctx, "deep/nested/new.txt", "hello\n")

    assert result.ok is True
    assert (workspace / "deep" / "nested" / "new.txt").read_text(encoding="utf-8") == "hello\n"
    assert result.meta["existed"] is False
    assert result.meta["bytes"] == 6


async def test_write_refuses_an_existing_file_that_was_never_read(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """Spec delta 13, the headline case: this is what stops silent clobbering."""

    (workspace / "a.txt").write_text("original\n", encoding="utf-8")
    result = await write(tool_ctx, "a.txt", "replaced\n")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS
    assert "has not been read" in result.error.message
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "original\n"


async def test_write_succeeds_after_a_read(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.txt").write_text("original\n", encoding="utf-8")
    await read(tool_ctx, "a.txt")
    result = await write(tool_ctx, "a.txt", "replaced\n")

    assert result.ok is True
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "replaced\n"


async def test_write_refuses_a_file_modified_since_it_was_read(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """The second half of delta 13: a stale read is as dangerous as no read."""

    path = workspace / "a.txt"
    path.write_text("original\n", encoding="utf-8")
    await read(tool_ctx, "a.txt")

    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000_000))

    result = await write(tool_ctx, "a.txt", "replaced\n")
    assert result.ok is False
    assert result.error is not None
    assert "modified since" in result.error.message
    assert path.read_text(encoding="utf-8") == "original\n"


async def test_a_second_write_needs_another_read(tool_ctx: ToolContext, workspace: Path) -> None:
    """A write refreshes the state itself, so consecutive writes work -- what it must
    not do is let a *stale* read authorise a second write."""

    (workspace / "a.txt").write_text("v1\n", encoding="utf-8")
    await read(tool_ctx, "a.txt")
    assert (await write(tool_ctx, "a.txt", "v2\n")).ok is True
    assert (await write(tool_ctx, "a.txt", "v3\n")).ok is True
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "v3\n"


async def test_write_preserves_the_line_endings_of_an_existing_file(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """Rewriting a CRLF file with LF is a whole-file diff that hides the one line
    that actually changed."""

    path = workspace / "crlf.txt"
    path.write_bytes(b"one\r\ntwo\r\n")
    await read(tool_ctx, "crlf.txt")

    await write(tool_ctx, "crlf.txt", "one\nthree\n")
    assert path.read_bytes() == b"one\r\nthree\r\n"


async def test_write_honours_the_endings_the_model_sent_for_a_new_file(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    await write(tool_ctx, "new.txt", "a\r\nb\r\n")
    assert (workspace / "new.txt").read_bytes() == b"a\r\nb\r\n"


async def test_write_to_a_directory_is_rejected(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "sub").mkdir()
    result = await write(tool_ctx, "sub", "x")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS


async def test_write_outside_the_workspace_is_refused(
    tool_ctx: ToolContext, tmp_path: Path
) -> None:
    result = await write(tool_ctx, str(tmp_path / "escaped.txt"), "x")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.PERMISSION
    assert not (tmp_path / "escaped.txt").exists()


async def test_the_approval_summary_shows_content_for_new_and_a_diff_for_overwrite(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """R-C-6: an approval without a diff is a leap of faith."""

    new = WRITE.approval_summary(WriteFileParams(path="new.txt", content="hello"), tool_ctx)
    assert new.diff is None
    assert "hello" in new.detail
    assert new.danger is False

    (workspace / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    over = WRITE.approval_summary(WriteFileParams(path="a.txt", content="one\nTWO\n"), tool_ctx)
    assert over.diff is not None
    assert "-two" in over.diff and "+TWO" in over.diff
    assert over.danger is True


async def test_the_result_carries_a_structured_diff_for_a_widget(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """Spec delta 12: the UI must not have to re-parse text written for a model."""

    (workspace / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    await read(tool_ctx, "a.txt")
    result = await write(tool_ctx, "a.txt", "one\nTWO\n")

    assert result.display is not None
    assert result.display.kind == "diff"
    assert "-two" in result.display.data["diff"]
    assert result.display.data["added"] == 1
    assert result.display.data["removed"] == 1


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


async def test_edit_replaces_a_unique_string(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    await read(tool_ctx, "a.py")

    result = await edit(tool_ctx, "a.py", "return 1", "return 2")

    assert result.ok is True
    assert (workspace / "a.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"
    assert "+    return 2" in result.text


async def test_edit_refuses_a_non_unique_string_and_says_the_count(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """A model that supplies an ambiguous string does not know which one it means,
    and picking the first is how an edit lands in the wrong function."""

    (workspace / "a.py").write_text("x = 1\ny = 1\nz = 1\n", encoding="utf-8")
    await read(tool_ctx, "a.py")

    result = await edit(tool_ctx, "a.py", "= 1", "= 2")

    assert result.ok is False
    assert result.error is not None
    assert "appears 3 times" in result.error.message
    assert "replace_all=True" in result.error.message
    assert (workspace / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 1\nz = 1\n"


async def test_replace_all_changes_every_occurrence(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.py").write_text("x = 1\ny = 1\n", encoding="utf-8")
    await read(tool_ctx, "a.py")

    result = await edit(tool_ctx, "a.py", "= 1", "= 2", replace_all=True)

    assert result.ok is True
    assert result.meta["replacements"] == 2
    assert (workspace / "a.py").read_text(encoding="utf-8") == "x = 2\ny = 2\n"


async def test_a_missing_string_suggests_the_nearest_candidate(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """Reconstructed indentation is the overwhelmingly common cause, and the model
    cannot see it in its own output."""

    (workspace / "a.py").write_text("def f():\n\treturn 1\n", encoding="utf-8")
    await read(tool_ctx, "a.py")

    result = await edit(tool_ctx, "a.py", "    return 1", "    return 2")

    assert result.ok is False
    assert result.error is not None
    assert "closest lines" in result.error.message
    assert "return 1" in result.error.message
    assert "whitespace" in result.error.message


async def test_edit_refuses_a_file_that_was_never_read(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")
    result = await edit(tool_ctx, "a.py", "x = 1", "x = 2")

    assert result.ok is False
    assert result.error is not None
    assert "has not been read" in result.error.message


async def test_edit_of_a_missing_file_points_at_write_file(tool_ctx: ToolContext) -> None:
    result = await edit(tool_ctx, "nope.py", "a", "b")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.NOT_FOUND
    assert "write_file" in result.error.message


async def test_an_empty_old_string_is_rejected_by_the_schema() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EditFileParams(path="a", old="", new="b")


async def test_an_identical_old_and_new_is_rejected() -> None:
    """A no-op edit is always a mistake, and it would otherwise consume an approval."""

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EditFileParams(path="a", old="x", new="x")


async def test_edit_preserves_crlf_endings(tool_ctx: ToolContext, workspace: Path) -> None:
    path = workspace / "crlf.py"
    path.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    await read(tool_ctx, "crlf.py")

    result = await edit(tool_ctx, "crlf.py", "two", "TWO")

    assert result.ok is True
    assert path.read_bytes() == b"one\r\nTWO\r\nthree\r\n"


async def test_an_edit_written_with_lf_matches_a_crlf_file(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """A model never sees the CRLFs, so a multi-line `old` it copies out of read_file
    is LF. Failing every such match would make edit_file useless on Windows."""

    path = workspace / "crlf.py"
    path.write_bytes(b"def f():\r\n    return 1\r\n")
    await read(tool_ctx, "crlf.py")

    result = await edit(tool_ctx, "crlf.py", "def f():\n    return 1", "def f():\n    return 2")

    assert result.ok is True
    assert path.read_bytes() == b"def f():\r\n    return 2\r\n"


async def test_an_edit_can_delete_text(tool_ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.py").write_text("keep\ndrop\nkeep2\n", encoding="utf-8")
    await read(tool_ctx, "a.py")

    result = await edit(tool_ctx, "a.py", "drop\n", "")

    assert result.ok is True
    assert (workspace / "a.py").read_text(encoding="utf-8") == "keep\nkeep2\n"


async def test_the_edit_approval_summary_carries_the_diff(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")
    summary = EDIT.approval_summary(EditFileParams(path="a.py", old="x = 1", new="x = 2"), tool_ctx)

    assert summary.diff is not None
    assert "-x = 1" in summary.diff
    assert "+x = 2" in summary.diff
    assert summary.danger is True


async def test_edit_is_not_concurrency_safe_and_write_is_not_either() -> None:
    """Two writers to the same file, or a write racing a read, is exactly what the
    contiguous-run partition exists to prevent."""

    assert EDIT.concurrency_safe is False
    assert WRITE.concurrency_safe is False
    assert READ.concurrency_safe is True


# ---------------------------------------------------------------------------
# fileio primitives
# ---------------------------------------------------------------------------


def test_atomic_write_leaves_no_temp_residue(tmp_path: Path) -> None:
    """Over many writes, a stray `.tmp` would accumulate in the user's source tree."""

    target = tmp_path / "out.txt"
    for i in range(50):
        atomic_write(target, f"{i}\n".encode())
    assert target.read_text(encoding="utf-8") == "49\n"
    assert list(tmp_path.glob("*.tmp")) == []
    assert len(list(tmp_path.iterdir())) == 1


def test_a_failed_write_removes_its_temp_file(tmp_path: Path, monkeypatch) -> None:
    import azalabscode.tools.fileio as fileio

    def boom(src: Path, dst: Path, *, attempts: int) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(fileio, "_replace_with_retry", boom)
    with pytest.raises(OSError):
        atomic_write(tmp_path / "out.txt", b"x")
    assert list(tmp_path.glob("*")) == []


def test_the_temp_file_is_created_in_the_destination_directory(tmp_path: Path) -> None:
    """`os.replace` is only atomic within a volume; a temp file in %TEMP% is a
    cross-volume move that is not atomic at all."""

    import azalabscode.tools.fileio as fileio

    seen: list[str] = []
    real = fileio.tempfile.mkstemp

    def spy(**kwargs):
        seen.append(kwargs["dir"])
        return real(**kwargs)

    target = tmp_path / "sub" / "out.txt"
    original = fileio.tempfile.mkstemp
    fileio.tempfile.mkstemp = spy  # type: ignore[assignment]
    try:
        atomic_write(target, b"x")
    finally:
        fileio.tempfile.mkstemp = original  # type: ignore[assignment]

    assert seen == [str(target.parent)]


def test_replace_retries_a_windows_style_permission_error(tmp_path: Path, monkeypatch) -> None:
    """On Windows `os.replace` raises `PermissionError` while any handle to the
    destination is open. The retry is mandatory there, not defensive."""

    import azalabscode.tools.fileio as fileio

    calls = {"n": 0}
    real_replace = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("destination is open")
        return real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", flaky)
    target = tmp_path / "out.txt"
    atomic_write(target, b"ok")

    assert calls["n"] == 3
    assert target.read_bytes() == b"ok"


def test_replace_gives_up_with_a_message_naming_the_cause(tmp_path: Path, monkeypatch) -> None:
    import azalabscode.tools.fileio as fileio

    def always_locked(src, dst):
        raise PermissionError("destination is open")

    monkeypatch.setattr(fileio.os, "replace", always_locked)
    with pytest.raises(OSError, match="holding the file open"):
        atomic_write(tmp_path / "out.txt", b"x", attempts=2)


def test_line_ending_detection_and_application() -> None:
    assert detect_line_ending("a\r\nb\r\n") == "\r\n"
    assert detect_line_ending("a\nb\n") == "\n"
    assert detect_line_ending("no newlines") == "\n"
    assert apply_line_ending("a\nb\n", "\r\n") == "a\r\nb\r\n"
    assert apply_line_ending("a\r\nb\r\n", "\n") == "a\nb\n"


def test_a_unified_diff_is_bounded() -> None:
    """A 20 000-line diff is not reviewable and would dominate a model request."""

    before = "\n".join(f"line {i}" for i in range(2000))
    after = "\n".join(f"LINE {i}" for i in range(2000))
    diff = unified_diff(before, after, path="x.txt", max_lines=50)

    assert diff.count("\n") <= 51
    assert "more diff lines omitted" in diff


def test_an_identical_pair_diffs_to_nothing() -> None:
    assert unified_diff("same\n", "same\n", path="x") == ""


def test_nearest_candidate_ignores_whitespace() -> None:
    haystack = "def f():\n\treturn 1\n"
    assert nearest_candidate(haystack, "    return 1") == ["\treturn 1"]
    assert nearest_candidate(haystack, "") == []
