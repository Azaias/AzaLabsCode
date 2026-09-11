"""The session document: everything a run is, written down (spec 6.2, delta 19).

A `Session` is a *projection* of a live `Controller`, never a second source of
truth. `Controller.session()` builds one on demand from live attributes; nothing
holds one between safe points. The alternative -- a `Session` the controller mutates
alongside its own fields -- means two structures that agree until the day someone
adds a field to one of them.

Delta 19 adds to spec 6.2: `updated_at`, `resume_state`, `rng_seed`, `usage_total`;
`config_type`, `config_hash` and `graph_hash` on `WorkflowRef`; and the per-agent
counters, which already live on `AgentState` because the agent loop maintains them.

Two things are worth knowing before reading further.

**`WorkflowRef` is the whole reconstruction recipe.** `load()` calls
`build(config)` and gets back a run body. The provider is constructed *inside*
`build` (delta 21), so a scripted run is reproducible from `(import_path, config)`
alone and nothing about the live process leaks into the file.

**Large values are spilled, not inlined.** A node output over
`SPILL_THRESHOLD_BYTES` goes to `<session_dir>/values/` and the session holds a
`ValueRef` naming it. `session.json` stays small enough to read in a diff, which is
the difference between a debuggable checkpoint and a 40 MB blob.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from azalabscode.cancellation import StepKind
from azalabscode.errors import ConfigurationError, WorkflowNotImportable
from azalabscode.ids import MAIN_AGENT
from azalabscode.messages import Usage
from azalabscode.permissions import DEFAULT_MODE, ApprovalRequest, PermissionMode
from azalabscode.runstate import NodeStatus, RunState
from azalabscode.schema import VersionedModel
from azalabscode.workflows.state import AgentState

SESSION_FILENAME = "session.json"
VALUES_DIRNAME = "values"
SPILL_THRESHOLD_BYTES = 32 * 1024
"""Node outputs larger than this go to `values/` so `session.json` stays diffable."""


def canonical_json(value: Any) -> str:
    """Sorted, separator-tight JSON. The input to every hash in this module."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_config(config: dict[str, Any]) -> str:
    """A stable digest of a workflow config, so drift is visible in the file."""

    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]


def _import_attr(spec: str, *, what: str) -> Any:
    """Resolve `"package.module:attribute"`, raising `WorkflowNotImportable`."""

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise WorkflowNotImportable(spec, f"expected 'package.module:{what}'")
    try:
        module = importlib.import_module(module_name)
    except Exception as error:  # any import failure is the same failure to the caller
        raise WorkflowNotImportable(spec, f"{type(error).__name__}: {error}") from error
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise WorkflowNotImportable(
            spec, f"module {module_name!r} has no attribute {attribute!r}"
        ) from error


class WorkflowRef(VersionedModel):
    """How to rebuild the workflow this session belongs to (spec 6.2, delta 19).

    `graph_hash` covers `(node_id, node_class, state_type, output_type)` and
    deliberately excludes prompts and config: editing a system prompt must not
    invalidate a saved session.
    """

    import_path: str = ""
    """`"package.module:build"`. Empty means the session is inspectable but not
    resumable without an explicit `build` passed to `load()`."""
    config: dict[str, Any] = Field(default_factory=dict)
    config_type: str | None = None
    """`"package.module:ConfigModel"`, validated on load (R-C-11)."""
    config_hash: str = ""
    graph_hash: str = ""

    @classmethod
    def of(
        cls,
        import_path: str,
        config: dict[str, Any] | None = None,
        *,
        config_type: str | None = None,
        graph_hash: str = "",
    ) -> WorkflowRef:
        """Build a reference, hashing the config so drift shows up in the file."""

        payload = dict(config or {})
        return cls(
            import_path=import_path,
            config=payload,
            config_type=config_type,
            config_hash=hash_config(payload),
            graph_hash=graph_hash,
        )

    def resolve(self) -> Any:
        """Import the `build` callable. Raises `WorkflowNotImportable` (R-C-11)."""

        builder = _import_attr(self.import_path, what="build")
        if not callable(builder):
            raise WorkflowNotImportable(self.import_path, "the resolved attribute is not callable")
        return builder

    def validated_config(self) -> Any:
        """The config, validated through `config_type` when one is recorded.

        Returns the raw dict when no `config_type` is set. A config that no longer
        validates is a `ConfigurationError` at load time rather than a `TypeError`
        five lines into `build` (R-C-11).
        """

        if not self.config_type:
            return self.config
        model = _import_attr(self.config_type, what="ConfigModel")
        validate = getattr(model, "model_validate", None)
        if validate is None:
            raise ConfigurationError(
                f"{self.config_type} is not a pydantic model; it cannot validate a config"
            )
        try:
            return validate(self.config)
        except Exception as error:  # pydantic own error is the message
            raise ConfigurationError(
                f"the saved config no longer validates against {self.config_type}: {error}"
            ) from error


class ValueRef(VersionedModel):
    """A node input or output: inline if small, a file in `values/` if not.

    Kept as one type rather than a union so a reader never has to ask which shape it
    got: `resolve(session_dir)` returns the value either way.
    """

    inline: Any = None
    path: str | None = None
    """Relative to the session directory, when the value was spilled."""
    size: int = 0
    """Serialized size in bytes, recorded for both shapes so a UI can show it."""

    @property
    def spilled(self) -> bool:
        """True when the value lives in a file rather than in the document."""

        return self.path is not None

    def resolve(self, session_dir: str | Path | None = None) -> Any:
        """The value itself, reading it back from `values/` if it was spilled."""

        if self.path is None:
            return self.inline
        if session_dir is None:
            raise ValueError(f"{self.path} was spilled but no session directory was given")
        return json.loads((Path(session_dir) / self.path).read_text(encoding="utf-8"))


class NodeRecord(VersionedModel):
    """One graph node's lifecycle and its memoized output (R-W-6).

    The memo is keyed by `(node_id, attempt)`: `nodes` maps a node id to its record
    and the record carries the attempt, so a resumed run can tell "this node
    completed" from "an earlier attempt of this node completed".
    """

    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    attempt: int = 0
    input: ValueRef | None = None
    output: ValueRef | None = None
    state: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def completed(self) -> bool:
        """True when this node must not be executed again (R-W-6)."""

        return self.status is NodeStatus.COMPLETED


class InflightStep(VersionedModel):
    """A step that was running when the checkpoint was taken (R-C-13).

    Snapshotted inside the checkpoint lock, so a safe point declared by agent A
    correctly records agent B mid-`shell`. What `load()` does with one depends
    entirely on `kind`:

    * `model_call` -- dropped and re-issued. Nothing was half-appended (spec C-1).
    * `tool_call` -- every call id it covers with no result becomes
      `ToolError(kind="interrupted")` and is **never** re-executed.
    * `delegate` -- resumed, not errored (delta 16). `child_agent_id` is how the
      resumed parent finds the child's transcript.
    """

    step_id: str
    kind: StepKind
    agent_id: str
    node_id: str | None = None
    call_id: str | None = None
    call_ids: list[str] = Field(default_factory=list)
    child_agent_id: str | None = None
    description: str = ""
    duration_ms: float = 0.0


class Session(VersionedModel):
    """The serializable document that fully describes a run (spec 6.2, delta 19)."""

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    workflow: WorkflowRef = Field(default_factory=WorkflowRef)
    run_state: RunState = RunState.CREATED
    resume_state: RunState = RunState.RUNNING
    """What `resume()` should return to. `PAUSING` and `INTERRUPTING` are transient
    and collapse to `RUNNING`; `WAITING_APPROVAL` survives, because the thing the run
    was waiting for did not go away while the process was dead."""
    permission_mode: PermissionMode = DEFAULT_MODE
    main_agent: str = str(MAIN_AGENT)
    gated_tools: list[str] = Field(default_factory=list)
    """Tool names whose `ApprovalPolicy` is not `never`, so a reloaded gate keeps
    R-C-7 without the tool registry having been rebuilt yet."""
    agents: dict[str, AgentState] = Field(default_factory=dict)
    nodes: dict[str, NodeRecord] = Field(default_factory=dict)
    pending_approvals: list[ApprovalRequest] = Field(default_factory=list)
    inflight: list[InflightStep] = Field(default_factory=list)
    event_seq: int = 0
    usage_total: Usage = Field(default_factory=Usage)
    rng_seed: int | None = None
    custom: dict[str, Any] = Field(default_factory=dict)

    # -- disk ---------------------------------------------------------------

    def dumps(self, *, indent: int | None = 2) -> bytes:
        """The bytes that go on disk. Indented, because a checkpoint gets read."""

        return self.model_dump_json(indent=indent).encode("utf-8")

    @classmethod
    def loads(cls, data: bytes | str) -> Session:
        """Parse a session document. Raises pydantic's error on a bad one."""

        return cls.model_validate_json(data)

    @classmethod
    def load(cls, path: str | Path) -> Session:
        """Read a session document from disk."""

        return cls.loads(Path(path).read_bytes())

    # -- convenience --------------------------------------------------------

    def agent(self, agent_id: str = str(MAIN_AGENT)) -> AgentState | None:
        """One agent's restored state."""

        return self.agents.get(agent_id)

    def completed_nodes(self) -> list[str]:
        """Node ids that must not be executed again (R-W-6)."""

        return sorted(node_id for node_id, record in self.nodes.items() if record.completed)

    def inflight_for(self, agent_id: str) -> list[InflightStep]:
        """Steps that were running for one agent when the process stopped."""

        return [step for step in self.inflight if step.agent_id == agent_id]


def session_file(session_dir: str | Path) -> Path:
    """The document's path inside a session directory."""

    return Path(session_dir) / SESSION_FILENAME


def values_dir(session_dir: str | Path) -> Path:
    """Where spilled values live."""

    return Path(session_dir) / VALUES_DIRNAME


__all__ = [
    "SESSION_FILENAME",
    "SPILL_THRESHOLD_BYTES",
    "VALUES_DIRNAME",
    "InflightStep",
    "NodeRecord",
    "Session",
    "ValueRef",
    "WorkflowRef",
    "canonical_json",
    "hash_config",
    "session_file",
    "values_dir",
]
