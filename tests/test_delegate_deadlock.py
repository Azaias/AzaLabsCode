"""The delegate deadlock regression, and the rest of the subagent contract.

The deadlock, stated once: a parent awaiting a child is not parked and never will
be. Under spec 4.4's original rule -- PAUSED when "the count of agents parked at the
gate equals the count of active agents" -- a pause during a delegate can never
land: the child parks (1), two agents are active (2), and `pause()` waits forever
for a parent that is blocked on the very child that is waiting for the pause to be
lifted.

Spec delta 14 fixes it by making `blocked_on_child` a *quiescent* phase, so the
count that matters is "agents in a non-quiescent phase" and the parent does not
hold the pause up. `test_pausing_during_a_delegate_reaches_paused` is that fix as a
test; `test_the_naive_rule_would_have_deadlocked_here` states the counts the old
rule would have compared, so the regression cannot be quietly undone by making
`blocked_on_child` non-quiescent again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from azalabscode.events import ModelCallStarted, ToolCallStarted
from azalabscode.ids import MAIN_AGENT
from azalabscode.messages import (
    AssistantMessage,
    ToolResultMessage,
    assert_transcript_valid,
)
from azalabscode.permissions import PermissionMode
from azalabscode.runstate import QUIESCENT_PHASES, TERMINAL_STATES, AgentPhase, RunState
from azalabscode.toolio import ToolErrorKind
from azalabscode.tools.builtin.delegate import DelegateTool
from azalabscode.workflows import AgentSpec
from tests.harness import Rig, build_rig, calls, echo_call, says

CHILD = "main/0"


def delegate_call(task: str, spec: str = "explore") -> tuple[str, dict[str, object]]:
    """One `delegate` call."""

    return ("delegate", {"task": task, "spec": spec})


MAIN_SPEC = AgentSpec(
    name="main",
    model="fake/model",
    system_prompt="you may delegate",
    allow_delegate=True,
    subagents=["explore"],
)
CHILD_SPEC = AgentSpec(name="explore", model="fake/model", system_prompt="explore", max_turns=6)


@pytest.fixture
async def rig_factory(workspace: Path):
    """Rigs with the `delegate` tool wired and one subagent spec available."""

    made: list[Rig] = []

    def make(turns, **kwargs) -> Rig:
        kwargs.setdefault("spec", MAIN_SPEC)
        kwargs.setdefault("specs", {"explore": CHILD_SPEC})
        kwargs.setdefault("extra_tools", [DelegateTool()])
        rig = build_rig(turns, workspace=workspace, **kwargs)
        made.append(rig)
        return rig

    try:
        yield make
    finally:
        for rig in made:
            if rig.controller.state not in TERMINAL_STATES:
                await rig.controller.cancel("test teardown")
            await rig.aclose()


# ---------------------------------------------------------------------------
# The happy path, so a regression is never ambiguous
# ---------------------------------------------------------------------------


async def test_a_delegate_runs_a_child_and_returns_its_answer(rig_factory) -> None:
    """R-W-4: the child gets its own transcript and reports back into the parent's."""

    rig = rig_factory(
        [
            calls(delegate_call("survey the repo")),
            says("the child found three files"),
            says("passing that on"),
        ]
    )
    await rig.controller.start()
    result = await rig.controller.wait(timeout=5.0)

    assert result.final_text == "passing that on"
    assert sorted(rig.controller.agents) == ["main", CHILD]

    child_state = rig.controller.agent(CHILD)
    assert child_state is not None
    assert child_state.parent_id == "main"
    assert child_state.final_text == "the child found three files"

    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert results[0].result.ok
    assert "three files" in results[0].result.text
    assert_transcript_valid(rig.transcript, agent_id="main")
    assert_transcript_valid(child_state.messages, agent_id=CHILD)


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------


async def test_pausing_during_a_delegate_reaches_paused(rig_factory) -> None:
    """**The regression.** A pause while a child is mid-call must reach PAUSED.

    The parent is `blocked_on_child` -- quiescent by delta 14 -- and the child parks
    at its next safe point. If `blocked_on_child` were ever made non-quiescent
    again, this test would hang rather than fail, which is why the wait is bounded.
    """

    rig = rig_factory(
        [
            calls(delegate_call("survey the repo")),
            says("child is thinking about it", chunk_delay_s=0.02),
            says("done"),
        ]
    )
    await rig.controller.start()
    await rig.wait_until(lambda: rig.controller.phase_of(CHILD) is not None)
    await rig.wait_event(lambda e: isinstance(e, ModelCallStarted) and e.agent_id == CHILD)

    await rig.controller.pause()
    await rig.wait_state(RunState.PAUSED, timeout=5.0)

    assert rig.controller.phase_of(MAIN_AGENT) is AgentPhase.BLOCKED_ON_CHILD
    assert rig.controller.phase_of(CHILD) is AgentPhase.PARKED
    assert rig.controller.nonquiescent == 0

    await rig.controller.resume()
    result = await rig.controller.wait(timeout=5.0)
    assert result.final_text == "done"


def test_the_naive_rule_would_have_deadlocked_here() -> None:
    """The counts spec 4.4's original rule compares, written down.

    Parked (1) never equals active (2) while the parent is blocked on the child, so
    the original formulation cannot reach PAUSED from the state the test above
    reaches it from. This is the assertion that stops `blocked_on_child` from
    quietly becoming non-quiescent.
    """

    assert AgentPhase.BLOCKED_ON_CHILD in QUIESCENT_PHASES
    phases = {MAIN_AGENT: AgentPhase.BLOCKED_ON_CHILD, CHILD: AgentPhase.PARKED}
    parked = sum(1 for p in phases.values() if p is AgentPhase.PARKED)
    active = len(phases)
    nonquiescent = sum(1 for p in phases.values() if p not in QUIESCENT_PHASES)

    assert parked != active, "the spec 4.4 rule: 1 parked, 2 active, never equal"
    assert nonquiescent == 0, "the delta 14 rule: nothing is doing anything"


async def test_a_child_waiting_for_approval_does_not_hold_a_pause_up(rig_factory) -> None:
    """`waiting_approval` is quiescent too, for the same reason.

    In `manual` mode a subagent is refused an approval-gated tool outright (R-C-7),
    so the request that would block comes from `main`. The parent is what waits, and
    a pause must still land while it does.
    """

    from azalabscode.control import QueueApprovalHandler
    from tests.harness import touch_call

    handler = QueueApprovalHandler()
    rig = rig_factory(
        [calls(touch_call("x")), says("done")],
        mode=PermissionMode.MANUAL,
        handler=handler,
    )
    await rig.controller.start()
    request = await handler.next_request()
    await rig.wait_state(RunState.WAITING_APPROVAL)

    await rig.controller.pause()
    await rig.wait_state(RunState.PAUSED, timeout=5.0)
    assert rig.controller.phase_of(MAIN_AGENT) is AgentPhase.WAITING_APPROVAL

    await rig.controller.resume()
    assert rig.controller.state is RunState.WAITING_APPROVAL
    await rig.controller.approve(request.request_id)
    await rig.controller.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# Interrupt and the agent tree
# ---------------------------------------------------------------------------


async def test_interrupting_the_parent_cancels_the_child_too(rig_factory) -> None:
    """Spec 6.4: structured concurrency does the unwinding."""

    rig = rig_factory(
        [
            calls(delegate_call("a long survey")),
            calls(echo_call("child work", sleep=5.0)),
            says("main carries on"),
        ]
    )
    await rig.controller.start()
    await rig.wait_event(lambda e: isinstance(e, ToolCallStarted) and e.tool == "echo")

    await rig.controller.interrupt()
    result = await rig.controller.wait(timeout=5.0)

    assert result.final_text == "main carries on"
    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert results[0].result.error is not None
    assert results[0].result.error.kind is ToolErrorKind.CANCELLED
    assert rig.echo.finished == [], "the child's tool must not have completed"
    assert_transcript_valid(rig.transcript, agent_id="main")


async def test_interrupting_the_child_leaves_the_parent_alone(rig_factory) -> None:
    """Spec 6.4: `interrupt(target="main/0")` cancels the child's step, not the parent's.

    The child absorbs the cancellation, takes its next model call, and the parent
    still receives an answer.
    """

    rig = rig_factory(
        [
            calls(delegate_call("a long survey")),
            says("child is rambling on and on", chunk_delay_s=0.05),
            says("child recovered"),
            says("main got: recovered"),
        ]
    )
    await rig.controller.start()
    await rig.wait_event(lambda e: isinstance(e, ModelCallStarted) and e.agent_id == CHILD)
    await asyncio.sleep(0.06)

    outcome = await rig.controller.interrupt(target=CHILD)
    assert outcome.hit

    result = await rig.controller.wait(timeout=5.0)
    assert result.final_text == "main got: recovered"

    child_state = rig.controller.agent(CHILD)
    assert child_state is not None
    assert child_state.final_text == "child recovered"
    assert [m for m in child_state.messages if isinstance(m, AssistantMessage) and m.cancelled]

    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert results[0].result.ok, "the parent's delegate step was never cancelled"


async def test_the_child_is_registered_before_its_task_exists(rig_factory) -> None:
    """The spawn race, closed by calling `enter_agent` before `create_task`.

    If the order were reversed there would be an instant with the child unregistered
    and the parent already `blocked_on_child` -- both quiescent, so a pause landing
    there would declare PAUSED over an agent about to start spending money.
    """

    rig = rig_factory([calls(delegate_call("survey")), says("child answer"), says("done")])
    seen: list[tuple[str, int]] = []

    original = rig.controller.phase

    async def spy(agent_id, phase):  # type: ignore[no-untyped-def]
        await original(agent_id, phase)
        seen.append((f"{agent_id}:{phase}", len(rig.controller.agents)))

    rig.controller.phase = spy  # type: ignore[method-assign]
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    blocked = [n for n, count in seen if n == f"main:{AgentPhase.BLOCKED_ON_CHILD}"]
    assert blocked, "the parent must be marked blocked_on_child"
    counts = [count for n, count in seen if n == f"main:{AgentPhase.BLOCKED_ON_CHILD}"]
    assert all(c == 2 for c in counts), "the child was already registered"


async def test_a_subagent_gets_its_own_read_state(rig_factory) -> None:
    """Trap 2 from M1: sharing the parent's read record would let a child write a
    file it has never looked at."""

    rig = rig_factory([calls(delegate_call("survey")), says("child answer"), says("done")])
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    main = rig.main
    assert main is not None and main.children
    child = main.children[0]
    assert child.dispatcher is not rig.dispatcher
    assert child.dispatcher.context.read_state is not rig.dispatcher.context.read_state
    assert child.dispatcher.context.workspace_root == rig.dispatcher.context.workspace_root
    assert child.dispatcher.gate is rig.dispatcher.gate


async def test_delegation_is_refused_when_the_spec_forbids_it(rig_factory) -> None:
    """R-W-4: no delegator on the context means the tool refuses, by design.

    The `delegate` tool does not have to be filtered out of the toolset -- it
    reports "delegation is not enabled for this agent", which is the message the
    model can act on.
    """

    rig = rig_factory(
        [calls(delegate_call("survey")), says("fine, doing it myself")],
        spec=AgentSpec(name="main", model="fake/model", allow_delegate=False),
    )
    await rig.controller.start()
    result = await rig.controller.wait(timeout=5.0)

    assert result.final_text == "fine, doing it myself"
    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert results[0].result.error is not None
    assert results[0].result.error.kind is ToolErrorKind.UNAVAILABLE
    assert list(rig.controller.agents) == ["main"]


async def test_a_subagent_in_manual_mode_cannot_see_gated_tools(rig_factory) -> None:
    """R-C-7 through the real gate: the child's request carries a filtered toolset."""

    from azalabscode.control import QueueApprovalHandler

    rig = rig_factory(
        [calls(delegate_call("survey")), says("child answer"), says("done")],
        mode=PermissionMode.MANUAL,
        handler=QueueApprovalHandler(),
    )
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    child_requests = [r for r in rig.provider.requests if r.metadata.get("agent_id") == CHILD]
    assert child_requests
    names = {t.name for t in child_requests[0].tools}
    assert "touch" not in names, "an approval-gated tool must be hidden from a subagent"
    assert "echo" in names

    main_requests = [r for r in rig.provider.requests if r.metadata.get("agent_id") == "main"]
    assert "touch" in {t.name for t in main_requests[0].tools}
