"""The inversion protocols. Every cross-layer seam in the harness is one of these.

Spec 4.1 draws arrows pointing down, and spec 4.3 has `tools` calling
`control.PermissionGate.check()` and `workflows` handing safe points to `control`.
Both of those are upward imports; they cannot both be true. This module is the
resolution: the *protocol* lives at the bottom of the graph, the implementation
lives in the layer that owns the decision, and the consumer depends on the protocol
only. Structural typing means nothing has to be registered or inherited.

    PermissionGate     control.gate.RuntimePermissionGate   ->  tools.dispatcher
    RunControl         control.controller.Controller        ->  workflows.runner
    Delegator          workflows.agent_loop.AgentLoop       ->  tools.builtin.delegate
    ApprovalHandler    tui / control.approval_handlers      ->  control.gate
    EventSink          anything that consumes events        ->  control, tui

`tools` also ships `AllowAllGate` and `DenyAllGate` so the tool layer is usable
standalone at M1, before `control` exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from azalabscode.cancellation import CancelReason, StepKind
from azalabscode.errors import ConfigurationError
from azalabscode.events import Event
from azalabscode.ids import AgentId, CallId, NodeId, RunId, StepId
from azalabscode.permissions import (
    ApprovalRequest,
    ApprovalSummary,
    Decision,
    PermissionMode,
)
from azalabscode.runstate import AgentPhase
from azalabscode.schema import VersionedModel

# ---------------------------------------------------------------------------
# Safe points
# ---------------------------------------------------------------------------


class SafePointKind(StrEnum):
    """The points at which a run may be halted or written down.

    `approval_park` is a safe point in its own right because a run waiting on a human
    is exactly the state R-C-9 requires to survive a save/kill/load cycle.
    """

    TURN_START = "turn_start"
    AFTER_MODEL_CALL = "after_model_call"
    AFTER_TOOL_BATCH = "after_tool_batch"
    NODE_ENTERED = "node_entered"
    NODE_COMPLETED = "node_completed"
    APPROVAL_PARK = "approval_park"
    CUSTOM = "custom"


SAFE_POINT_KINDS: frozenset[SafePointKind] = frozenset(SafePointKind)


class SafePoint(VersionedModel):
    """A point at which the run is consistent enough to be written down or halted.

    Built by `NodeContext.checkpoint()` and handed to `RunControl.safe_point()`.
    Spec 4.5 assigns the two halves deliberately: `workflows` decides *where* the
    safe points are, `control` decides what to *do* at one.

    `snapshot` is the node's declared `State` already dumped to JSON-safe values.
    Dumping in the workflow layer rather than the control layer is what keeps
    `SerializationError` attributable to a node and a field path (R-W-5).
    """

    kind: SafePointKind = SafePointKind.CUSTOM
    node_id: str | None = None
    agent_id: str | None = None
    attempt: int = 0
    snapshot: dict[str, Any] | None = None
    durable: bool = True
    """Whether this point should be written to disk, not just folded into memory."""
    park: bool = False
    """Whether the caller should wait at the pause gate after the fold completes.

    Parking happens strictly *after* the write and strictly *outside* the checkpoint
    lock. An agent parked while holding that lock deadlocks every other checkpoint,
    and the run never reaches PAUSED.
    """


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@runtime_checkable
class StepHandleLike(Protocol):
    """The cancellable unit of work, as `control` needs to see it.

    `workflows.step.StepHandle` implements it. The rule the implementation must
    honour is in `cancellation`: set `cancel_reason` *before* calling
    `task.cancel()`, and let the first reason win so repeated interrupts are
    idempotent (R-C-1).
    """

    step_id: StepId
    agent_id: AgentId
    node_id: NodeId | None
    kind: StepKind
    call_id: CallId | None
    call_ids: list[str]
    """Every call id this step covers. A tool batch has several; a model call none.

    What makes an in-flight batch reconcilable after a process death: the ids with no
    result are the ones that become `ToolError(kind="interrupted")` (R-C-13)."""
    child_agent_id: str | None
    """For a `delegate` step, the child it is waiting on -- which is how a resumed
    parent finds the child's transcript instead of starting a new one (delta 16)."""
    description: str
    """One line naming what is running. `SaveTimeout` renders it (delta 17)."""
    cancel_reason: CancelReason | None

    @property
    def duration_ms(self) -> float:
        """How long this step has been running, live or final."""
        ...

    def request_cancel(self, reason: CancelReason) -> bool:
        """Request cancellation. Returns False if a reason was already recorded."""
        ...


@runtime_checkable
class RunControl(Protocol):
    """What `workflows` may ask of `control`, and nothing more.

    `Controller` satisfies this structurally, so `workflows` never imports `control`
    and import-linter contract 3 holds without a `TYPE_CHECKING` escape hatch.
    """

    @property
    def permission_mode(self) -> PermissionMode:
        """The run's current mode, as a read-only view."""
        ...

    @property
    def permission_gate(self) -> PermissionGate:
        """The gate to inject into a `ToolDispatcher`.

        A workflow rebuilt by `load()` constructs its own provider and dispatcher
        from `(import_path, config)` alone (spec delta 21), so it needs a gate and
        cannot be handed one by the caller -- there is no caller, only a config dict.
        Returning the *protocol* keeps the direction right: `workflows` receives a
        `PermissionGate`, not a `RuntimePermissionGate`.
        """
        ...

    async def safe_point(self, sp: SafePoint) -> None:
        """Fold a safe point into the session, write it, then park if asked.

        May raise `SerializationError`, which the runner must *not* catch: a run
        that cannot be saved should fail rather than continue doing unsaveable work.
        """
        ...

    async def enter_agent(self, agent_id: AgentId, parent_id: AgentId | None = None) -> None:
        """Register an agent as active.

        Called synchronously inside `spawn()`/`delegate()` *before* the child task is
        created, together with flipping the parent to `blocked_on_child`. That
        ordering is what removes the spawn race: there is no instant in which the
        child is unregistered and the parent is already quiescent.
        """
        ...

    async def exit_agent(self, agent_id: AgentId) -> None:
        """Mark an agent finished and drop it from the quiescence count."""
        ...

    async def phase(self, agent_id: AgentId, phase: AgentPhase) -> None:
        """Record an agent's quiescence phase."""
        ...

    def register_step(self, handle: StepHandleLike) -> None:
        """Track a step as in-flight so a checkpoint taken by another agent records it."""
        ...

    def unregister_step(self, handle: StepHandleLike) -> None:
        """Stop tracking a step that finished."""
        ...

    @property
    def rng_seed(self) -> int | None:
        """The run's seed, so a node that samples samples the same way after a resume.

        `Session.rng_seed` round-trips it; `NodeContext.rng` is the one consumer.
        """
        ...

    def emitter_for(self, agent_id: AgentId | str, node_id: str | None = None) -> Any:
        """An `EventEmitter` bound to one agent and node.

        Typed `Any` for symmetry with `restored_agent`: `events` sits below this
        module, so naming `EventEmitter` here would be legal but would put a second
        import of it on a protocol whose whole job is to name as little as possible.
        The one caller is the runner, which knows what it is getting.
        """
        ...

    async def enter_node(self, node_id: str) -> None:
        """Register a node as an execution context for quiescence.

        Not `enter_agent`: a node is not an agent, has no transcript and must not
        appear in `Session.agents` or emit `AgentSpawned`. It still has to be counted,
        because a graph of `Func` nodes has no agents at all and a `pause()` on one
        would otherwise never reach PAUSED -- quiescence over an empty set is
        vacuously true.
        """
        ...

    async def exit_node(self, node_id: str) -> None:
        """Drop a node from the quiescence count."""
        ...

    def node_state(self, node_id: str, *, attempt: int = 0) -> dict[str, Any] | None:
        """The state envelope a previous process checkpointed for this node (R-W-5).

        `None` for a node that has never taken a safe point. The runner restores the
        node's declared `State` from it before running the body, which is what makes
        "incomplete nodes restart from their last checkpoint" (R-W-6) true rather
        than aspirational.
        """
        ...

    def restored_agent(self, agent_id: AgentId) -> Any | None:
        """The state a `load()` restored for this agent, or `None` for a fresh run.

        Typed `Any` deliberately: the state is `workflows.state.AgentState`, which
        sits *above* this module in the layer graph, so naming it here would invert
        the dependency this whole file exists to prevent. The one caller is the agent
        loop, which owns the type and knows what it is getting.
        """
        ...

    # -- the node memo (R-W-6) ---------------------------------------------

    def node_completed(self, node_id: str, *, attempt: int = 0) -> bool:
        """Whether this attempt of this node already produced an output.

        A runner asks this before running anything. `True` means the node must not
        be executed again -- its effects happened in a previous process and its
        output is memoized.
        """
        ...

    def node_output(self, node_id: str, *, attempt: int = 0) -> Any:
        """The memoized output of a completed node. `KeyError` if there is none."""
        ...

    async def node_started(
        self,
        node_id: str,
        *,
        attempt: int = 0,
        input: Any = None,
        node_class: str = "",
    ) -> Any:
        """Record a node as running. `node_class` is what a stage widget draws."""
        ...

    async def node_finished(self, node_id: str, output: Any = None, *, attempt: int = 0) -> Any:
        """Memoize a node's output so a resume never runs it again (R-W-6)."""
        ...

    async def node_failed(self, node_id: str, error: str, *, attempt: int = 0) -> Any:
        """Record a node as failed. Deliberately *not* memoized: a failure is retried,
        not reused, and an `on_child_error="continue"` parent turns it into a value of
        its own instead (R-W-7)."""
        ...


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


@runtime_checkable
class PermissionGate(Protocol):
    """What `tools` may ask about permission, and nothing more.

    Spec 4.3 has the dispatcher calling into `control` directly; that is the upward
    import spec 4.1 forbids. The dispatcher depends on this protocol and `control`
    injects the implementation (spec delta 3).
    """

    @property
    def mode(self) -> PermissionMode:
        """Current permission mode."""
        ...

    async def check(
        self,
        *,
        tool_name: str,
        needs_approval: bool,
        summary: ApprovalSummary,
        params: dict[str, Any],
        agent_id: AgentId,
        call_id: CallId,
        node_id: NodeId | None = None,
    ) -> Decision:
        """Decide whether one call may proceed.

        Returns a `Decision` for *every* call, gated or not, so the dispatcher has a
        single code path. May block on a human in `manual` mode, during which the
        agent's phase is `waiting_approval` -- quiescent, so a pause can still land.
        """
        ...

    def visible_tool_names(self, agent_id: AgentId, requested: Sequence[str]) -> list[str]:
        """Filter a toolset for one agent.

        In `manual` mode a subagent sees only `never`-policy tools (R-C-7). The
        filtered names are what goes into the model request, so the model does not
        waste a turn asking for something it cannot have.
        """
        ...


@runtime_checkable
class ApprovalHandler(Protocol):
    """The seam between a pending approval and a human (R-C-8).

    Implementations: `TUIApprovalHandler` (mounts a modal), `StdinApprovalHandler`
    (headless), `DenyAllHandler`, `QueueApprovalHandler` (tests). A `manual`-mode
    controller with none registered raises `ConfigurationError` at `start()` --
    before any spend -- rather than hanging on the first destructive call.
    """

    async def request(self, request: ApprovalRequest) -> Decision | None:
        """Present a request.

        Return a `Decision` to resolve immediately, or `None` to resolve later out
        of band via `Controller.resolve_approval()`. The TUI returns `None`: the
        modal outlives the call.
        """
        ...

    async def cancel(self, request_id: str, reason: str) -> None:
        """Withdraw a request that no longer needs an answer."""
        ...


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


class DelegateOutcome(VersionedModel):
    """What a subagent hands back to its parent."""

    agent_id: str
    final_text: str = ""
    ok: bool = True
    error: str | None = None
    turns: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Delegator(Protocol):
    """What the `delegate` tool may ask of the agent layer.

    Specs resolve **by name**, so `tools` never imports `AgentSpec` and the tool
    layer stays independent of `workflows` (spec delta 5).
    """

    def available_specs(self) -> list[str]:
        """Names of the agent specs this agent is allowed to delegate to."""
        ...

    async def delegate(
        self,
        spec_name: str,
        task: str,
        *,
        tools: Sequence[str] | None = None,
        model: str | None = None,
        max_turns: int | None = None,
    ) -> DelegateOutcome:
        """Run a child agent to completion and return its result."""
        ...


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@runtime_checkable
class EventSink(Protocol):
    """Anything that consumes the event stream: the TUI, a recorder, a test."""

    async def handle(self, event: Event) -> None:
        """Receive one event. Must not raise; the bus does not retry."""
        ...


def require_approval_handler(mode: PermissionMode, handler: ApprovalHandler | None) -> None:
    """Enforce R-C-8: a `manual` run needs somewhere to send its prompts.

    Called from `Controller.start()`. Raising here, rather than on the first
    destructive call, is the difference between a config error and a mid-run hang.
    """

    if mode is PermissionMode.MANUAL and handler is None:
        raise ConfigurationError(
            "permission_mode is 'manual' but no ApprovalHandler is registered; "
            "register one (StdinApprovalHandler for headless runs, TUIApprovalHandler "
            "under a HarnessApp) or start the run in 'auto' mode"
        )


__all__ = [
    "SAFE_POINT_KINDS",
    "AgentId",
    "ApprovalHandler",
    "CallId",
    "DelegateOutcome",
    "Delegator",
    "EventSink",
    "NodeId",
    "PermissionGate",
    "RunControl",
    "RunId",
    "SafePoint",
    "SafePointKind",
    "StepHandleLike",
    "require_approval_handler",
]
