"""`ToolContext`: everything a tool is allowed to know about the run it is inside.

A plain dataclass carrying leaf types only (R-T-10). Nothing here names a provider,
a workflow or a controller, which is what makes an MCP adapter a later addition
rather than a rewrite.

Two pieces of policy live here because every file tool needs them and none of them
should implement them twice:

- **`resolve_path`** (R-T-7). Symlinks are resolved *before* the containment check,
  so a symlink inside the workspace pointing at `/etc` is rejected rather than
  followed.
- **`ReadState`** (spec delta 13). A bounded LRU of what has been read. `write_file`
  and `edit_file` refuse a path that was never read, or was read before an external
  modification. This is the behaviour that stops an agent overwriting a file whose
  current contents it has never seen.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import Field

from azalabscode.contracts import Delegator
from azalabscode.errors import HarnessError
from azalabscode.events import Event
from azalabscode.ids import MAIN_AGENT, AgentId, CallId, NodeId, RunId, new_call_id
from azalabscode.schema import HarnessModel
from azalabscode.toolio import ToolError, ToolErrorKind

NO_RUN: RunId = RunId("")
"""Placeholder run id for a context built outside a run -- a script or a test."""

DEFAULT_READ_STATE_ENTRIES = 512
"""How many read records to keep. Bounded because a long session reads a lot."""


class ToolPathError(HarnessError):
    """A path could not be resolved or left the workspace.

    Raised inside a tool and converted to a `ToolResult` by that tool or by the
    dispatcher; it never escapes the tool layer.
    """

    def __init__(self, kind: ToolErrorKind, message: str) -> None:
        self.kind = kind
        self.message = message
        super().__init__(message)

    def as_tool_error(self) -> ToolError:
        """The data form, for putting into a `ToolResult`."""

        return ToolError(kind=self.kind, message=self.message)


@dataclass
class ReadRecord:
    """What `read_file` saw the last time it looked at a path."""

    mtime_ns: int
    size: int
    offset: int = 1
    limit: int = 0
    truncated: bool = False
    """True when the read covered only part of the file, so a whole-file write is
    being made from a partial view."""


class ReadState:
    """Bounded LRU of `path -> ReadRecord`, keyed by normalised absolute path.

    Case-normalised via `os.path.normcase`, because on Windows `SRC\\App.py` and
    `src/app.py` are the same file and a read of one must satisfy a write of the
    other.
    """

    def __init__(self, max_entries: int = DEFAULT_READ_STATE_ENTRIES) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, ReadRecord] = OrderedDict()

    @staticmethod
    def key(path: Path | str) -> str:
        """The normalised lookup key for a path."""

        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def record(self, path: Path | str, record: ReadRecord) -> None:
        """Note that `path` was read."""

        k = self.key(path)
        self._entries.pop(k, None)
        self._entries[k] = record
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def get(self, path: Path | str) -> ReadRecord | None:
        """The last read of `path`, if it is still in the window."""

        k = self.key(path)
        record = self._entries.get(k)
        if record is not None:
            self._entries.move_to_end(k)
        return record

    def clear(self, path: Path | str) -> None:
        """Forget a path. Called after a write, so the next write must re-read."""

        self._entries.pop(self.key(path), None)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, path: object) -> bool:
        if not isinstance(path, str | Path):
            return False
        return self.key(path) in self._entries


class WorkspaceConfig(HarnessModel):
    """Workspace containment policy (R-T-7)."""

    unrestricted: bool = False
    """When true, paths outside the root are allowed. Off by default, deliberately:
    the safe direction costs a rejected call, the unsafe one costs a file."""
    follow_symlinks: bool = True
    """Resolve symlinks before the containment check. Turning this off does not make
    the check laxer -- it makes an unresolvable link an error."""


class ToolConfig(HarnessModel):
    """Run-wide tool configuration. Serializable: it is part of the session.

    Everything here is a policy knob a human sets once, not something a model can
    influence per call.
    """

    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)

    # shell
    shell_program: str | None = None
    """`"pwsh"`, `"bash"`, an absolute path, or `None` for the platform default."""
    shell_max_timeout_s: float = 600.0
    shell_default_timeout_s: float = 120.0
    shell_progress_interval_s: float = 1.0

    # web
    allow_private_network: bool = False
    """Off by default: an agent that can fetch `http://169.254.169.254/` can read
    cloud instance credentials (R-T-7's spirit applied to the network)."""
    max_fetch_bytes: int = 10 * 1024 * 1024
    max_redirects: int = 5
    user_agent: str = "azalabscode/0.1 (+https://github.com/)"

    # search
    search_backend: str = "serper"
    search_timeout_s: float = 20.0

    # files
    max_read_bytes: int = 400_000
    """A single `read_file` window above this errors rather than truncating: a
    ~100-byte error costs far less than 25k tokens of clipped file (spec delta 8)."""
    max_image_bytes: int = 5 * 1024 * 1024


@dataclass
class ToolContext:
    """The per-call environment handed to `Tool.run`.

    Constructed by the dispatcher from run-wide state plus the ids of this call.
    `read_state`, `config` and `session_dir` are shared across calls in a run;
    `call_id`, `agent_id` and `node_id` are not.
    """

    workspace_root: Path
    run_id: RunId = NO_RUN
    agent_id: AgentId = MAIN_AGENT
    call_id: CallId = field(default_factory=new_call_id)
    node_id: NodeId | None = None
    config: ToolConfig = field(default_factory=ToolConfig)
    read_state: ReadState = field(default_factory=ReadState)
    session_dir: Path | None = None
    env: Mapping[str, str] | None = None
    emit: Callable[[Event], Awaitable[None]] | None = None
    """Progress events. `None` in tests and headless scripts; tools must tolerate it."""
    delegator: Delegator | None = None
    """Set only for an agent whose spec allows delegation."""
    extras: dict[str, Any] = field(default_factory=dict)
    """Escape hatch for a workflow that ships its own tools. Core never reads it."""

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).expanduser().resolve()

    def for_call(
        self,
        *,
        call_id: CallId,
        agent_id: AgentId | None = None,
        node_id: NodeId | None = None,
        delegator: Delegator | None = None,
    ) -> ToolContext:
        """A copy bound to one call, sharing the mutable run-wide state.

        `read_state` is shared by reference on purpose: a file read by one call must
        satisfy the write issued by the next.
        """

        return ToolContext(
            workspace_root=self.workspace_root,
            run_id=self.run_id,
            agent_id=agent_id if agent_id is not None else self.agent_id,
            call_id=call_id,
            node_id=node_id if node_id is not None else self.node_id,
            config=self.config,
            read_state=self.read_state,
            session_dir=self.session_dir,
            env=self.env,
            emit=self.emit,
            delegator=delegator if delegator is not None else self.delegator,
            extras=self.extras,
        )

    async def progress(self, event: Event) -> None:
        """Emit an event if there is anywhere to emit it. Never raises."""

        if self.emit is None:
            return
        try:
            await self.emit(event)
        except Exception:
            return

    # -- paths --------------------------------------------------------------

    def resolve_path(self, raw: str, *, must_exist: bool = False) -> Path:
        """Resolve a model-supplied path and enforce workspace containment (R-T-7).

        Relative paths are taken against `workspace_root`. `~` expands. Symlinks are
        resolved *before* the containment check, so a link inside the workspace
        pointing outside it is rejected rather than followed.

        Raises `ToolPathError`, which the caller turns into a `ToolResult`.
        """

        if not isinstance(raw, str) or not raw.strip():
            raise ToolPathError(ToolErrorKind.INVALID_PARAMS, "path must be a non-empty string")

        candidate = Path(raw.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate

        try:
            resolved = (
                candidate.resolve()
                if self.config.workspace.follow_symlinks
                else Path(os.path.abspath(candidate))
            )
        except OSError as exc:
            raise ToolPathError(
                ToolErrorKind.INVALID_PARAMS, f"cannot resolve path {raw!r}: {exc}"
            ) from exc

        if not self.config.workspace.unrestricted and not self._within_root(resolved):
            raise ToolPathError(
                ToolErrorKind.PERMISSION,
                f"path {raw!r} resolves to {resolved} which is outside the workspace "
                f"root {self.workspace_root}; only paths inside the workspace may be used",
            )

        if must_exist and not resolved.exists():
            raise ToolPathError(ToolErrorKind.NOT_FOUND, f"no such file or directory: {raw}")

        return resolved

    def _within_root(self, resolved: Path) -> bool:
        root = os.path.normcase(os.fspath(self.workspace_root))
        target = os.path.normcase(os.fspath(resolved))
        if target == root:
            return True
        return target.startswith(root.rstrip(os.sep) + os.sep)

    def display_path(self, path: Path) -> str:
        """A path as it should appear to the model: workspace-relative when it can be."""

        try:
            return path.relative_to(self.workspace_root).as_posix()
        except ValueError:
            return path.as_posix()

    # -- read-before-write --------------------------------------------------

    def note_read(
        self, path: Path, *, offset: int = 1, limit: int = 0, truncated: bool = False
    ) -> None:
        """Record that `path` was just read, with the file's current mtime and size."""

        try:
            st = path.stat()
        except OSError:
            return
        self.read_state.record(
            path,
            ReadRecord(
                mtime_ns=st.st_mtime_ns,
                size=st.st_size,
                offset=offset,
                limit=limit,
                truncated=truncated,
            ),
        )

    def check_read_before_write(self, path: Path) -> ToolError | None:
        """Enforce spec delta 13. `None` means the write may proceed.

        A file that does not exist needs no prior read -- there is nothing to
        clobber. An existing file needs a read, and that read must be newer than the
        file's last modification, or the model is editing contents it has not seen.
        """

        if not path.exists():
            return None

        record = self.read_state.get(path)
        rel = self.display_path(path)
        if record is None:
            return ToolError(
                kind=ToolErrorKind.INVALID_PARAMS,
                message=(
                    f"{rel} exists but has not been read in this session. "
                    f"Call read_file on it first, then retry this write."
                ),
                details={"path": rel, "reason": "not_read"},
            )

        try:
            st = path.stat()
        except OSError as exc:  # pragma: no cover - raced deletion
            return ToolError(kind=ToolErrorKind.NOT_FOUND, message=f"cannot stat {rel}: {exc}")

        if st.st_mtime_ns != record.mtime_ns:
            return ToolError(
                kind=ToolErrorKind.INVALID_PARAMS,
                message=(
                    f"{rel} has been modified since it was last read. "
                    f"Call read_file on it again before writing, or your change will "
                    f"overwrite someone else's."
                ),
                details={"path": rel, "reason": "stale_read"},
            )

        return None


__all__ = [
    "DEFAULT_READ_STATE_ENTRIES",
    "NO_RUN",
    "ReadRecord",
    "ReadState",
    "ToolConfig",
    "ToolContext",
    "ToolPathError",
    "WorkspaceConfig",
]
