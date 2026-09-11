"""The four inversion protocols.

At M0 there is nothing to plug into them yet, so what is tested is the thing that
would otherwise silently rot: that each protocol is satisfied *structurally*, by an
object that never imports the harness's own base classes. That is the whole point
-- `Controller` will satisfy `RunControl` without `workflows` ever importing
`control`, and a minimal stand-in proves the protocol is shaped so that it can.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from azalabscode.cancellation import CancelReason, StepKind
from azalabscode.contracts import (
    SAFE_POINT_KINDS,
    ApprovalHandler,
    DelegateOutcome,
    Delegator,
    EventSink,
    PermissionGate,
    RunControl,
    SafePoint,
    SafePointKind,
    StepHandleLike,
    require_approval_handler,
)
from azalabscode.errors import ConfigurationError
from azalabscode.events import Event, NodeStarted
from azalabscode.permissions import (
    ApprovalRequest,
    ApprovalSummary,
    Decision,
    PermissionMode,
)
from azalabscode.runstate import AgentPhase


class StubStep:
    """The shape `workflows.step.StepHandle` will have."""

    def __init__(self) -> None:
        self.step_id = "step_1"
        self.agent_id = "main"
        self.node_id = "root/agent"
        self.kind = StepKind.TOOL_CALL
        self.call_id = "call_1"
        self.call_ids: list[str] = ["call_1"]
        self.child_agent_id: str | None = None
        self.description = "1 tool call(s)"
        self.cancel_reason: CancelReason | None = None

    @property
    def duration_ms(self) -> float:
        """`SaveTimeout` renders this, so the protocol names it (delta 17)."""

        return 12.5

    def request_cancel(self, reason: CancelReason) -> bool:
        """First reason wins, so repeated interrupts are idempotent (R-C-1)."""

        if self.cancel_reason is not None:
            return False
        self.cancel_reason = reason
        return True


class StubControl:
    """The shape `control.controller.Controller` will have."""

    def __init__(self) -> None:
        self.safe_points: list[SafePoint] = []
        self.agents: dict[str, AgentPhase] = {}
        self.steps: list[Any] = []
        self.nodes: dict[tuple[str, int], Any] = {}
        self._mode = PermissionMode.AUTO
        self._gate = StubGate()

    @property
    def permission_mode(self) -> PermissionMode:
        return self._mode

    @property
    def permission_gate(self) -> Any:
        return self._gate

    def node_completed(self, node_id: str, *, attempt: int = 0) -> bool:
        return (node_id, attempt) in self.nodes

    def node_output(self, node_id: str, *, attempt: int = 0) -> Any:
        return self.nodes[node_id, attempt]

    async def node_started(
        self,
        node_id: str,
        *,
        attempt: int = 0,
        input: Any = None,
        node_class: str = "",
    ) -> Any:
        return None

    async def node_finished(self, node_id: str, output: Any = None, *, attempt: int = 0) -> Any:
        self.nodes[node_id, attempt] = output
        return None

    async def node_failed(self, node_id: str, error: str, *, attempt: int = 0) -> Any:
        return None

    def node_state(self, node_id: str, *, attempt: int = 0) -> dict[str, Any] | None:
        """The envelope a safe point wrote for this node, or `None` (M5)."""

        return None

    async def enter_node(self, node_id: str) -> None:
        """A node is an execution context for quiescence, not an agent (M5)."""

        self.agents[node_id] = AgentPhase.RUNNING

    async def exit_node(self, node_id: str) -> None:
        self.agents.pop(node_id, None)

    def emitter_for(self, agent_id: str, node_id: str | None = None) -> Any:
        return None

    async def safe_point(self, sp: SafePoint) -> None:
        self.safe_points.append(sp)

    async def enter_agent(self, agent_id: str, parent_id: str | None = None) -> None:
        self.agents[agent_id] = AgentPhase.RUNNING

    async def exit_agent(self, agent_id: str) -> None:
        self.agents[agent_id] = AgentPhase.FINISHED

    async def phase(self, agent_id: str, phase: AgentPhase) -> None:
        self.agents[agent_id] = phase

    def register_step(self, handle: Any) -> None:
        self.steps.append(handle)

    def unregister_step(self, handle: Any) -> None:
        self.steps.remove(handle)

    def restored_agent(self, agent_id: str) -> Any | None:
        """`None` for a run that never came back from a checkpoint (M3)."""

        return None

    @property
    def rng_seed(self) -> int | None:
        """Round-trips through the session; `NodeContext.rng` is the one consumer (M5)."""

        return None


class StubGate:
    """The shape `control.gate.RuntimePermissionGate` will have."""

    def __init__(self, mode: PermissionMode = PermissionMode.AUTO) -> None:
        self._mode = mode
        self.checked: list[str] = []

    @property
    def mode(self) -> PermissionMode:
        return self._mode

    async def check(
        self,
        *,
        tool_name: str,
        needs_approval: bool,
        summary: ApprovalSummary,
        params: dict[str, Any],
        agent_id: str,
        call_id: str,
        node_id: str | None = None,
    ) -> Decision:
        self.checked.append(tool_name)
        if needs_approval and self._mode is PermissionMode.MANUAL:
            return Decision.deny("no handler in this stub")
        return Decision.approve(by="auto")

    def visible_tool_names(self, agent_id: str, requested: Sequence[str]) -> list[str]:
        if self._mode is PermissionMode.MANUAL and agent_id != "main":
            return [n for n in requested if n in {"read_file", "grep", "glob"}]
        return list(requested)


class StubDelegator:
    """The shape `workflows.agent_loop.AgentLoop` will have."""

    def available_specs(self) -> list[str]:
        return ["explorer", "reviewer"]

    async def delegate(
        self,
        spec_name: str,
        task: str,
        *,
        tools: Sequence[str] | None = None,
        model: str | None = None,
        max_turns: int | None = None,
    ) -> DelegateOutcome:
        return DelegateOutcome(agent_id="main/0", final_text=f"{spec_name}: {task}", turns=1)


class StubHandler:
    """The shape an `ApprovalHandler` will have."""

    def __init__(self) -> None:
        self.seen: list[ApprovalRequest] = []

    async def request(self, request: ApprovalRequest) -> Decision | None:
        self.seen.append(request)
        return None

    async def cancel(self, request_id: str, reason: str) -> None:
        self.seen = [r for r in self.seen if r.request_id != request_id]


class StubSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def handle(self, event: Event) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Structural conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stub", "protocol"),
    [
        (StubStep(), StepHandleLike),
        (StubControl(), RunControl),
        (StubGate(), PermissionGate),
        (StubDelegator(), Delegator),
        (StubHandler(), ApprovalHandler),
        (StubSink(), EventSink),
    ],
    ids=[
        "StepHandleLike",
        "RunControl",
        "PermissionGate",
        "Delegator",
        "ApprovalHandler",
        "EventSink",
    ],
)
def test_each_protocol_is_satisfied_structurally(stub: object, protocol: type) -> None:
    """No inheritance, no registration: this is what keeps the layers apart."""

    assert isinstance(stub, protocol)


def test_a_partial_implementation_is_not_accepted() -> None:
    class Incomplete:
        def available_specs(self) -> list[str]:
            return []

    assert not isinstance(Incomplete(), Delegator)


# ---------------------------------------------------------------------------
# Behaviour the protocols are shaped around
# ---------------------------------------------------------------------------


def test_the_first_cancel_reason_wins() -> None:
    """R-C-1: `interrupt()` is idempotent, so a second reason must not overwrite."""

    step = StubStep()
    assert step.request_cancel(CancelReason.USER_INTERRUPT) is True
    assert step.request_cancel(CancelReason.TIMEOUT) is False
    assert step.cancel_reason is CancelReason.USER_INTERRUPT


async def test_a_run_control_records_phases_and_safe_points() -> None:
    control = StubControl()
    await control.enter_agent("main")
    await control.phase("main", AgentPhase.BLOCKED_IO)
    await control.safe_point(SafePoint(kind=SafePointKind.AFTER_MODEL_CALL, agent_id="main"))
    await control.exit_agent("main")

    assert control.agents["main"] is AgentPhase.FINISHED
    assert control.safe_points[0].kind is SafePointKind.AFTER_MODEL_CALL


async def test_the_gate_returns_a_decision_for_every_call() -> None:
    """Gated or not, so the dispatcher has one code path rather than two."""

    gate = StubGate(PermissionMode.AUTO)
    decision = await gate.check(
        tool_name="read_file",
        needs_approval=False,
        summary=ApprovalSummary(title="read_file a.py"),
        params={"path": "a.py"},
        agent_id="main",
        call_id="c1",
    )
    assert decision.approved is True
    assert gate.checked == ["read_file"]


def test_manual_mode_filters_a_subagents_toolset() -> None:
    """R-C-7: in `manual` mode a subagent sees only `never`-policy tools."""

    gate = StubGate(PermissionMode.MANUAL)
    requested = ["read_file", "grep", "write_file", "shell"]
    assert gate.visible_tool_names("main", requested) == requested
    assert gate.visible_tool_names("main/0", requested) == ["read_file", "grep"]


async def test_delegation_resolves_specs_by_name() -> None:
    """Spec delta 5: resolving by name is what keeps `tools` free of `AgentSpec`."""

    delegator = StubDelegator()
    assert "explorer" in delegator.available_specs()
    outcome = await delegator.delegate("explorer", "find the config loader")
    assert outcome.final_text == "explorer: find the config loader"
    assert outcome.ok is True


async def test_an_approval_handler_may_defer_its_decision() -> None:
    """The TUI returns `None`: the modal outlives the call and resolves out of band."""

    handler = StubHandler()
    request = ApprovalRequest(
        run_id="r",
        agent_id="main",
        call_id="c",
        tool="shell",
        summary=ApprovalSummary(title="shell"),
    )
    assert await handler.request(request) is None
    assert handler.seen == [request]
    await handler.cancel(request.request_id, "no longer needed")
    assert handler.seen == []


async def test_an_event_sink_receives_events() -> None:
    sink = StubSink()
    await sink.handle(NodeStarted(node_id="n"))
    assert sink.events[0].node_id == "n"


# ---------------------------------------------------------------------------
# R-C-8
# ---------------------------------------------------------------------------


def test_manual_mode_without_a_handler_is_a_configuration_error() -> None:
    """Raising at `start()` rather than at the first destructive call is the
    difference between a config error and a mid-run hang."""

    with pytest.raises(ConfigurationError) as excinfo:
        require_approval_handler(PermissionMode.MANUAL, None)
    message = str(excinfo.value)
    assert "StdinApprovalHandler" in message
    assert "auto" in message


def test_auto_mode_needs_no_handler() -> None:
    require_approval_handler(PermissionMode.AUTO, None)
    require_approval_handler(PermissionMode.MANUAL, StubHandler())


# ---------------------------------------------------------------------------
# Safe points
# ---------------------------------------------------------------------------


def test_safe_point_defaults_are_durable_and_non_parking() -> None:
    """Parking is opt-in because it must happen after the write and outside the
    checkpoint lock; a default-on flag would invite getting that order wrong."""

    sp = SafePoint()
    assert sp.durable is True
    assert sp.park is False
    assert sp.kind is SafePointKind.CUSTOM


def test_every_safe_point_kind_in_the_plan_exists() -> None:
    assert {k.value for k in SAFE_POINT_KINDS} == {
        "turn_start",
        "after_model_call",
        "after_tool_batch",
        "node_entered",
        "node_completed",
        "approval_park",
        "custom",
    }


def test_a_safe_point_snapshot_is_plain_json() -> None:
    """The node dumps its own state, which is what keeps `SerializationError`
    attributable to a node and a field path (R-W-5)."""

    sp = SafePoint(snapshot={"turn": 2, "messages": [], "pending": None})
    assert SafePoint.model_validate_json(sp.model_dump_json()) == sp
