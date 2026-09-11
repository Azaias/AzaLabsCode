"""`ToolContext`: path resolution, workspace containment, and the read-state LRU.

Two policies live here because every file tool needs them, and a bug in either is a
bug in all of them at once: `resolve_path` is R-T-7's containment check, and
`ReadState` is spec delta 13's read-before-write tracking.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from azalabscode.contracts import Delegator
from azalabscode.ids import CallId
from azalabscode.toolio import ToolErrorKind
from azalabscode.tools import (
    ReadRecord,
    ReadState,
    ToolConfig,
    ToolContext,
    ToolPathError,
    WorkspaceConfig,
)

IS_WINDOWS = os.name == "nt"


# ---------------------------------------------------------------------------
# resolve_path (R-T-7)
# ---------------------------------------------------------------------------


def test_a_relative_path_resolves_against_the_workspace_root(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    assert tool_ctx.resolve_path("src/app.py") == workspace / "src" / "app.py"


def test_an_absolute_path_inside_the_workspace_is_accepted(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    target = workspace / "a.txt"
    assert tool_ctx.resolve_path(str(target)) == target


def test_the_workspace_root_itself_is_inside_the_workspace(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    assert tool_ctx.resolve_path(str(workspace)) == workspace


@pytest.mark.parametrize("raw", ["../outside.txt", "src/../../outside.txt", "a/../../b"])
def test_a_relative_escape_is_rejected(tool_ctx: ToolContext, raw: str) -> None:
    with pytest.raises(ToolPathError) as exc:
        tool_ctx.resolve_path(raw)
    assert exc.value.kind is ToolErrorKind.PERMISSION


def test_an_absolute_path_outside_the_workspace_is_rejected(
    tool_ctx: ToolContext, tmp_path: Path
) -> None:
    with pytest.raises(ToolPathError) as exc:
        tool_ctx.resolve_path(str(tmp_path / "elsewhere.txt"))
    assert exc.value.kind is ToolErrorKind.PERMISSION


def test_a_sibling_directory_sharing_a_prefix_is_still_outside(workspace: Path) -> None:
    """`/ws` and `/ws-other` share a string prefix. A naive `startswith` check on the
    root would let the second through."""

    sibling = workspace.parent / f"{workspace.name}-other"
    sibling.mkdir()
    ctx = ToolContext(workspace_root=workspace)

    with pytest.raises(ToolPathError):
        ctx.resolve_path(str(sibling / "x.txt"))


def test_an_empty_path_is_an_invalid_parameter(tool_ctx: ToolContext) -> None:
    for raw in ("", "   "):
        with pytest.raises(ToolPathError) as exc:
            tool_ctx.resolve_path(raw)
        assert exc.value.kind is ToolErrorKind.INVALID_PARAMS


def test_must_exist_turns_a_missing_file_into_not_found(tool_ctx: ToolContext) -> None:
    with pytest.raises(ToolPathError) as exc:
        tool_ctx.resolve_path("nope.txt", must_exist=True)
    assert exc.value.kind is ToolErrorKind.NOT_FOUND


def test_unrestricted_lets_a_path_leave_the_workspace(workspace: Path, tmp_path: Path) -> None:
    ctx = ToolContext(
        workspace_root=workspace,
        config=ToolConfig(workspace=WorkspaceConfig(unrestricted=True)),
    )
    assert ctx.resolve_path(str(tmp_path / "x.txt")) == tmp_path / "x.txt"


@pytest.mark.skipif(IS_WINDOWS, reason="symlink creation needs privileges on Windows")
def test_a_symlink_is_resolved_before_the_containment_check(
    tool_ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    """Resolving *after* the check would follow a link out of the workspace, which is
    the whole hole R-T-7's "symlinks are resolved before the check" closes."""

    outside = tmp_path / "secret"
    outside.mkdir()
    (workspace / "link").symlink_to(outside)

    with pytest.raises(ToolPathError) as exc:
        tool_ctx.resolve_path("link/file.txt")
    assert exc.value.kind is ToolErrorKind.PERMISSION


def test_display_path_is_workspace_relative(tool_ctx: ToolContext, workspace: Path) -> None:
    """The model sees `src/app.py`, not a machine-specific absolute path."""

    assert tool_ctx.display_path(workspace / "src" / "app.py") == "src/app.py"


def test_display_path_falls_back_to_absolute_for_an_outside_path(
    tool_ctx: ToolContext, tmp_path: Path
) -> None:
    assert tool_ctx.display_path(tmp_path / "x") == (tmp_path / "x").as_posix()


def test_display_path_uses_forward_slashes_on_every_platform(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    assert "\\" not in tool_ctx.display_path(workspace / "a" / "b" / "c.py")


# ---------------------------------------------------------------------------
# ReadState
# ---------------------------------------------------------------------------


def test_a_record_survives_a_round_trip(tmp_path: Path) -> None:
    state = ReadState()
    record = ReadRecord(mtime_ns=1, size=2)
    state.record(tmp_path / "a.txt", record)
    assert state.get(tmp_path / "a.txt") is record


def test_a_path_is_normalised_before_lookup(tmp_path: Path) -> None:
    """`src/../src/app.py` and `src/app.py` are the same file."""

    state = ReadState()
    state.record(tmp_path / "src" / "app.py", ReadRecord(mtime_ns=1, size=2))
    assert state.get(tmp_path / "src" / ".." / "src" / "app.py") is not None


@pytest.mark.skipif(not IS_WINDOWS, reason="case-insensitive lookup is a Windows concern")
def test_case_differences_are_the_same_file_on_windows(tmp_path: Path) -> None:
    """A read of `SRC\\App.py` must satisfy a write of `src/app.py`, or the
    read-before-write rule blocks a legitimate edit."""

    state = ReadState()
    state.record(tmp_path / "src" / "App.py", ReadRecord(mtime_ns=1, size=2))
    assert state.get(tmp_path / "SRC" / "app.py") is not None


def test_clearing_forgets_a_path(tmp_path: Path) -> None:
    state = ReadState()
    state.record(tmp_path / "a.txt", ReadRecord(mtime_ns=1, size=2))
    state.clear(tmp_path / "a.txt")
    assert state.get(tmp_path / "a.txt") is None


def test_the_lru_is_bounded_and_evicts_the_oldest(tmp_path: Path) -> None:
    """A long session reads a lot; an unbounded map is a slow leak."""

    state = ReadState(max_entries=3)
    for i in range(5):
        state.record(tmp_path / f"{i}.txt", ReadRecord(mtime_ns=i, size=i))

    assert len(state) == 3
    assert state.get(tmp_path / "0.txt") is None
    assert state.get(tmp_path / "4.txt") is not None


def test_reading_a_path_again_makes_it_the_most_recent(tmp_path: Path) -> None:
    state = ReadState(max_entries=2)
    state.record(tmp_path / "a.txt", ReadRecord(mtime_ns=1, size=1))
    state.record(tmp_path / "b.txt", ReadRecord(mtime_ns=2, size=2))
    state.get(tmp_path / "a.txt")  # touch a
    state.record(tmp_path / "c.txt", ReadRecord(mtime_ns=3, size=3))

    assert state.get(tmp_path / "a.txt") is not None
    assert state.get(tmp_path / "b.txt") is None


def test_membership_ignores_non_path_values() -> None:
    state = ReadState()
    assert 42 not in state


# ---------------------------------------------------------------------------
# check_read_before_write (spec delta 13)
# ---------------------------------------------------------------------------


def test_a_new_file_needs_no_prior_read(tool_ctx: ToolContext, workspace: Path) -> None:
    """There is nothing to clobber."""

    assert tool_ctx.check_read_before_write(workspace / "new.txt") is None


def test_an_existing_unread_file_is_refused(tool_ctx: ToolContext, workspace: Path) -> None:
    path = workspace / "a.txt"
    path.write_text("x", encoding="utf-8")

    error = tool_ctx.check_read_before_write(path)
    assert error is not None
    assert error.details["reason"] == "not_read"
    assert "read_file" in error.message


def test_a_read_file_passes(tool_ctx: ToolContext, workspace: Path) -> None:
    path = workspace / "a.txt"
    path.write_text("x", encoding="utf-8")
    tool_ctx.note_read(path)

    assert tool_ctx.check_read_before_write(path) is None


def test_a_file_modified_after_the_read_is_refused(tool_ctx: ToolContext, workspace: Path) -> None:
    path = workspace / "a.txt"
    path.write_text("x", encoding="utf-8")
    tool_ctx.note_read(path)

    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    error = tool_ctx.check_read_before_write(path)
    assert error is not None
    assert error.details["reason"] == "stale_read"
    assert "modified since" in error.message


def test_note_read_on_a_missing_file_records_nothing(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    tool_ctx.note_read(workspace / "gone.txt")
    assert tool_ctx.read_state.get(workspace / "gone.txt") is None


# ---------------------------------------------------------------------------
# for_call
# ---------------------------------------------------------------------------


def test_for_call_shares_the_read_state_by_reference(
    tool_ctx: ToolContext, workspace: Path
) -> None:
    """A file read by one call must satisfy the write issued by the next."""

    path = workspace / "a.txt"
    path.write_text("x", encoding="utf-8")

    reader = tool_ctx.for_call(call_id=CallId("c1"))
    reader.note_read(path)

    writer = tool_ctx.for_call(call_id=CallId("c2"))
    assert writer.check_read_before_write(path) is None


def test_for_call_rebinds_the_ids_without_touching_the_original(
    tool_ctx: ToolContext,
) -> None:
    derived = tool_ctx.for_call(call_id=CallId("c9"), agent_id="main/0")  # type: ignore[arg-type]

    assert derived.call_id == "c9"
    assert derived.agent_id == "main/0"
    assert tool_ctx.agent_id == "main"
    assert derived.workspace_root == tool_ctx.workspace_root


def test_for_call_carries_the_delegator_through(tool_ctx: ToolContext) -> None:
    class Fake:
        def available_specs(self) -> list[str]:
            return ["x"]

        async def delegate(self, spec_name, task, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    delegator = Fake()
    assert isinstance(delegator, Delegator)

    derived = tool_ctx.for_call(call_id=CallId("c1"), delegator=delegator)  # type: ignore[arg-type]
    assert derived.delegator is delegator
    assert tool_ctx.delegator is None


async def test_progress_is_a_no_op_without_an_emitter(tool_ctx: ToolContext) -> None:
    from azalabscode.events import ToolCallProgress

    await tool_ctx.progress(ToolCallProgress(call_id="c1", tool="shell"))


async def test_progress_swallows_a_failing_emitter(tool_ctx: ToolContext) -> None:
    """Observability must never fail a tool call."""

    from azalabscode.events import Event, ToolCallProgress

    async def broken(event: Event) -> None:
        raise RuntimeError("bus is down")

    tool_ctx.emit = broken
    await tool_ctx.progress(ToolCallProgress(call_id="c1", tool="shell"))


# ---------------------------------------------------------------------------
# ToolConfig
# ---------------------------------------------------------------------------


def test_the_config_round_trips_through_json() -> None:
    """It is part of the session document at M3 (R-X-4)."""

    config = ToolConfig(shell_program="pwsh", allow_private_network=True)
    assert ToolConfig.model_validate_json(config.model_dump_json()) == config


def test_the_defaults_are_the_safe_ones() -> None:
    config = ToolConfig()
    assert config.workspace.unrestricted is False
    assert config.allow_private_network is False
    assert config.workspace.follow_symlinks is True


def test_the_context_resolves_its_root_on_construction(tmp_path: Path) -> None:
    """A workspace root that is not resolved would fail its own containment check
    whenever the path contains a symlink or a Windows short name."""

    nested = tmp_path / "a" / ".." / "a"
    (tmp_path / "a").mkdir()
    ctx = ToolContext(workspace_root=nested)
    assert ctx.workspace_root == (tmp_path / "a").resolve()
