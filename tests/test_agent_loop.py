"""`AgentLoop` behaviours the control scenarios do not reach.

The scenarios drive the loop through pause, interrupt and approval. These cover the
turn itself: what goes into the request, what comes back out, and the bookkeeping
M3 will serialize.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from azalabscode.contracts import Delegator
from azalabscode.events import Checkpoint, ModelDelta
from azalabscode.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from azalabscode.permissions import PermissionMode
from azalabscode.providers.testing import ScriptedToolCall, ScriptedTurn
from azalabscode.toolio import ToolErrorKind
from azalabscode.workflows import AgentSpec
from tests.harness import Rig, build_rig, calls, echo_call, says


@pytest.fixture
async def rig_factory(workspace: Path):
    """Rigs, torn down whatever happens."""

    made: list[Rig] = []

    def make(turns, **kwargs) -> Rig:
        rig = build_rig(turns, workspace=workspace, **kwargs)
        made.append(rig)
        return rig

    try:
        yield make
    finally:
        for rig in made:
            await rig.aclose()


async def test_the_first_request_carries_the_system_prompt_and_the_task(rig_factory) -> None:
    """Seeding happens once, on the way in, and only when the transcript is empty."""

    rig = rig_factory([says("hello")], task="count the files")
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    first = rig.provider.requests[0]
    assert isinstance(first.messages[0], SystemMessage)
    assert first.messages[0].content == "be useful"
    assert isinstance(first.messages[1], UserMessage)
    assert first.messages[1].text == "count the files"
    assert first.metadata["agent_id"] == "main"


async def test_the_request_carries_the_toolset_the_gate_allows(rig_factory) -> None:
    """`schemas_for` runs the gate, so R-C-7's filter reaches the model request."""

    rig = rig_factory([says("hello")])
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    names = {t.name for t in rig.provider.requests[0].tools}
    assert names == {"echo", "touch"}


async def test_a_spec_toolset_narrows_what_the_model_sees(rig_factory) -> None:
    """`AgentSpec.tools` is the workflow author's filter; the gate's is on top."""

    rig = rig_factory(
        [says("hello")],
        spec=AgentSpec(name="main", model="fake/model", tools=["echo"]),
    )
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    assert {t.name for t in rig.provider.requests[0].tools} == {"echo"}


async def test_malformed_tool_arguments_come_back_as_a_structured_error(rig_factory) -> None:
    """R-P-4: the model gets `invalid_params`, never an exception."""

    rig = rig_factory(
        [
            ScriptedTurn(
                tool_calls=[
                    ScriptedToolCall(call_id="bad_1", name="echo", arguments_chunks=['{"value": '])
                ],
                finish_reason="tool_calls",
            ),
            says("I will fix the JSON"),
        ]
    )
    await rig.controller.start()
    result = await rig.controller.wait(timeout=5.0)

    assert result.final_text == "I will fix the JSON"
    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert results[0].result.error is not None
    assert results[0].result.error.kind is ToolErrorKind.INVALID_PARAMS
    assert rig.echo.started == []


async def test_an_unknown_tool_is_reported_with_the_ones_that_exist(rig_factory) -> None:
    """The model can only correct itself if it is told what is available."""

    rig = rig_factory(
        [
            ScriptedTurn(
                tool_calls=[ScriptedToolCall(call_id="u1", name="nosuchtool", arguments={})],
                finish_reason="tool_calls",
            ),
            says("understood"),
        ]
    )
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert results[0].result.error is not None
    assert results[0].result.error.kind is ToolErrorKind.UNAVAILABLE
    assert "echo" in results[0].result.text


async def test_the_turn_budget_is_carried_in_the_agent_state(rig_factory) -> None:
    """Trap 1: one budget per turn, its decisions memoized by call id (delta 8).

    The decisions are what make a resumed turn byte-stable: the same call elides,
    and the transcript the resumed run builds matches the one that was saved.
    """

    rig = rig_factory([calls(echo_call("a"), echo_call("b")), says("done")])
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    state = rig.controller.agent("main")
    assert state is not None
    assert len(state.result_budget) == 2
    assert all(size >= 0 for size in state.result_budget.values())


async def test_an_over_budget_turn_elides_a_result_rather_than_dropping_it(
    rig_factory,
) -> None:
    """Delta 8: the model is told what was omitted, so it does not call again blind."""

    rig = rig_factory(
        [calls(echo_call("x" * 400), echo_call("y" * 400)), says("done")],
        spec=AgentSpec(name="main", model="fake/model", max_tool_results_chars=420),
    )
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert results[0].result.ok
    assert results[1].result.error is not None
    assert results[1].result.error.kind is ToolErrorKind.BUDGET
    assert "limit" in results[1].result.text


async def test_usage_accumulates_onto_the_agent_state(rig_factory) -> None:
    """What the status bar reads, and what the session records per agent."""

    from azalabscode.messages import Usage

    rig = rig_factory(
        [
            calls(echo_call("a"), usage=Usage(prompt_tokens=10, completion_tokens=2)),
            says("done", usage=Usage(prompt_tokens=20, completion_tokens=3)),
        ]
    )
    await rig.controller.start()
    result = await rig.controller.wait(timeout=5.0)

    assert result.usage.prompt_tokens == 30
    assert result.usage.completion_tokens == 5
    state = rig.controller.agent("main")
    assert state is not None and state.usage.total_tokens == 35


async def test_every_turn_takes_the_three_safe_points(rig_factory) -> None:
    """The checkpoint rhythm M3's `save()` hangs off: turn start, model, tools."""

    rig = rig_factory([calls(echo_call("a")), says("done")])
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    kinds = [str(f.kind) for f in rig.controller.folds]
    assert kinds == [
        "turn_start",
        "after_model_call",
        "after_tool_batch",
        "turn_start",
        "after_model_call",
    ]
    assert len(rig.of_type(Checkpoint)) == len(kinds)


async def test_streamed_text_is_emitted_as_deltas(rig_factory) -> None:
    """R-U-4's input: core emits every fragment; the widget coalesces, not the loop."""

    rig = rig_factory([says("one two three")])
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    deltas = rig.of_type(ModelDelta)
    assert [d.text for d in deltas if d.text] == ["one ", "two ", "three"]
    assert all(d.agent_id == "main" for d in deltas)


async def test_the_agent_loop_is_a_delegator(rig_factory) -> None:
    """Structural typing again: the `delegate` tool never imports this class."""

    rig = rig_factory([says("hi")])
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)
    assert isinstance(rig.main, Delegator)


async def test_child_ids_are_allocated_in_call_order(rig_factory) -> None:
    """Spec 6.3: a checkpointed counter bumped with no await in between."""

    rig = rig_factory([says("hi")])
    await rig.controller.start()
    await rig.controller.wait(timeout=5.0)

    state = rig.controller.agent("main")
    assert state is not None
    assert [state.next_child_id() for _ in range(3)] == ["main/0", "main/1", "main/2"]
    assert state.child_seq == 3


async def test_a_resumed_transcript_is_not_re_seeded(rig_factory, workspace) -> None:
    """M3 depends on this: an agent with messages already keeps them."""

    from azalabscode.workflows import AgentLoop, AgentState

    rig = rig_factory([says("hi")])
    prior = AgentState(agent_id="main", messages=[UserMessage.of("earlier work")])
    loop = AgentLoop(
        AgentSpec(name="main", model="fake/model", system_prompt="ignored"),
        provider=rig.provider,
        dispatcher=rig.dispatcher,
        state=prior,
    )
    result = await loop.run("this task is not used")

    assert result.final_text == "hi"
    assert not any(isinstance(m, SystemMessage) for m in prior.messages)
    assert isinstance(prior.messages[0], UserMessage)
    assert prior.messages[0].text == "earlier work"


async def test_the_loop_runs_without_a_controller(rig_factory) -> None:
    """`control` is optional: a script can drive an agent with no run around it."""

    from azalabscode.workflows import AgentLoop

    rig = rig_factory([calls(echo_call("a")), says("done")])
    loop = AgentLoop(
        AgentSpec(name="solo", model="fake/model"),
        provider=rig.provider,
        dispatcher=rig.dispatcher,
    )
    result = await loop.run("go")

    assert result.final_text == "done"
    assert rig.echo.finished == ["a"]
    assert [type(m).__name__ for m in loop.state.messages] == [
        "UserMessage",
        "AssistantMessage",
        "ToolResultMessage",
        "AssistantMessage",
    ]


async def test_a_subagent_in_auto_mode_sees_the_whole_toolset(rig_factory) -> None:
    """R-C-7's other half, and spec C-6: the restriction lifts in `auto`."""

    rig = rig_factory([says("hi")], mode=PermissionMode.AUTO)
    names = rig.controller.gate.visible_tool_names("main/0", ["echo", "touch"])
    assert names == ["echo", "touch"]

    await rig.controller.set_permission_mode(PermissionMode.MANUAL)
    assert rig.controller.gate.visible_tool_names("main/0", ["echo", "touch"]) == ["echo"]
    assert rig.controller.gate.visible_tool_names("main", ["echo", "touch"]) == [
        "echo",
        "touch",
    ]


async def test_a_cancelled_assistant_message_reaches_the_next_request(rig_factory) -> None:
    """Spec decision 4: the model is shown its own cut-off output, marked."""

    import asyncio

    from azalabscode.events import ModelCallStarted

    rig = rig_factory([says("I was saying something long", chunk_delay_s=0.05), says("second")])
    await rig.controller.start()
    await rig.wait_type(ModelCallStarted)
    await asyncio.sleep(0.06)
    await rig.controller.interrupt()
    await rig.controller.wait(timeout=5.0)

    second = rig.provider.requests[-1]
    partials = [m for m in second.messages if isinstance(m, AssistantMessage) and m.cancelled]
    assert len(partials) == 1
