"""The runner and the seven node types, against a real `Controller`.

What is checked here is the behaviour the graph promises and the agent loop cannot:
the memo (R-W-6), the failure policy (R-W-7), the safe-point rhythm, and the
quiescence bookkeeping that lets a graph with no agents in it reach PAUSED at all.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from azalabscode import (
    AgentNode,
    AgentPhase,
    AgentSpec,
    FakeProvider,
    Func,
    ModelCall,
    NodeCompleted,
    NodeFailed,
    NodeRef,
    NodeStarted,
    RunState,
    Workflow,
)
from tests.graphrig import Boom, SteppingNode, answers, build_graph_rig, explode


def echo(value: Any) -> Any:
    return value


def shout(value: Any) -> str:
    return str(value).upper()


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


async def test_a_linear_graph_runs_in_order_and_returns_its_output() -> None:
    wf = Workflow("linear", input="hello")
    first = wf.func("first", shout)
    wf.output(wf.func("second", lambda text: f"{text}!", input=first))

    rig = build_graph_rig(wf)
    try:
        assert await rig.controller.run(timeout=5) == "HELLO!"
        assert rig.controller.state is RunState.COMPLETED
        assert [e.node_id for e in rig.of_type(NodeStarted)] == ["first", "second"]
        assert rig.completed_nodes() == ["first", "second"]
    finally:
        await rig.aclose()


async def test_without_an_explicit_output_the_last_node_wins() -> None:
    wf = Workflow("linear", input="a")
    wf.func("only", shout)

    rig = build_graph_rig(wf)
    try:
        assert await rig.controller.run(timeout=5) == "A"
    finally:
        await rig.aclose()


async def test_node_started_carries_the_node_class() -> None:
    """What a `StagePipeline` widget draws (spec 8.2)."""

    wf = Workflow("w", input="x")
    wf.func("f", echo)

    rig = build_graph_rig(wf)
    try:
        await rig.controller.run(timeout=5)
        started = rig.of_type(NodeStarted)
        assert [e.node_class for e in started if e.node_class] == ["Func"]
    finally:
        await rig.aclose()


async def test_a_func_may_take_the_context() -> None:
    wf = Workflow("w", input="x")
    wf.func("f", lambda ctx, value: f"{ctx.node_id}:{value}")

    rig = build_graph_rig(wf)
    try:
        assert await rig.controller.run(timeout=5) == "f:x"
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# FanOut, Gather, Map, Subgraph
# ---------------------------------------------------------------------------


async def test_fan_out_runs_branches_concurrently_and_keeps_declaration_order() -> None:
    """Completion order must not decide the list order (M5, `run_children`)."""

    order: list[str] = []

    def slow(delay: float, label: str) -> Any:
        async def body(_: Any) -> str:
            await asyncio.sleep(delay)
            order.append(label)
            return label

        return Func(body)

    wf = Workflow("w", input=None)
    wf.fan_out("branches", {"slow": slow(0.05, "slow"), "fast": slow(0.0, "fast")})

    rig = build_graph_rig(wf)
    try:
        assert await rig.controller.run(timeout=5) == ["slow", "fast"]
        assert order == ["fast", "slow"]  # they really did run concurrently
    finally:
        await rig.aclose()


async def test_gather_joins_several_references() -> None:
    wf = Workflow("w", input="x")
    a = wf.func("a", lambda _: "A")
    b = wf.func("b", lambda _: "B")
    wf.output(wf.gather("join", [a, b]))

    rig = build_graph_rig(wf)
    try:
        assert await rig.controller.run(timeout=5) == ["A", "B"]
    finally:
        await rig.aclose()


async def test_map_runs_the_template_once_per_item_with_indexed_ids() -> None:
    wf = Workflow("w", input=["a", "b", "c"])
    wf.map("each", Func(shout), over=wf.input)

    rig = build_graph_rig(wf)
    try:
        assert await rig.controller.run(timeout=5) == ["A", "B", "C"]
        assert rig.completed_nodes() == ["each", "each/0", "each/1", "each/2"]
    finally:
        await rig.aclose()


async def test_map_bounded_by_max_concurrency() -> None:
    live = 0
    peak = 0

    async def body(_: Any) -> int:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return peak

    wf = Workflow("w", input=[1, 2, 3, 4])
    wf.map("each", Func(body), over=wf.input, max_concurrency=2)

    rig = build_graph_rig(wf)
    try:
        await rig.controller.run(timeout=5)
        assert peak == 2
    finally:
        await rig.aclose()


async def test_map_refuses_a_non_list_input() -> None:
    wf = Workflow("w", input="not a list")
    wf.map("each", Func(echo), over=wf.input)

    rig = build_graph_rig(wf)
    try:
        with pytest.raises(TypeError, match="needs a list"):
            await rig.controller.run(timeout=5)
        assert rig.controller.state is RunState.FAILED
    finally:
        await rig.aclose()


async def test_a_subgraph_runs_inlined_with_prefixed_ids_and_its_own_input() -> None:
    inner = Workflow("inner")
    doubled = inner.func("double", lambda text: f"{text}{text}")
    inner.output(inner.func("mark", lambda text: f"<{text}>", input=doubled))

    outer = Workflow("outer", input="x")
    outer.output(outer.subgraph("sub", inner))

    rig = build_graph_rig(outer)
    try:
        assert await rig.controller.run(timeout=5) == "<xx>"
        assert rig.completed_nodes() == ["sub", "sub/double", "sub/mark"]
    finally:
        await rig.aclose()


async def test_a_subgraph_sees_the_node_input_not_the_workflow_input() -> None:
    inner = Workflow("inner")
    inner.output(inner.func("id", echo))

    outer = Workflow("outer", input="outer-input")
    upstream = outer.func("upstream", lambda _: "inner-input")
    outer.output(outer.subgraph("sub", inner, input=upstream))

    rig = build_graph_rig(outer)
    try:
        assert await rig.controller.run(timeout=5) == "inner-input"
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# Failure policy (R-W-7)
# ---------------------------------------------------------------------------


async def test_a_failing_node_fails_the_run_by_default() -> None:
    wf = Workflow("w", input="x")
    wf.func("boom", explode)

    rig = build_graph_rig(wf)
    try:
        with pytest.raises(Boom):
            await rig.controller.run(timeout=5)
        assert rig.controller.state is RunState.FAILED
        assert [e.node_id for e in rig.of_type(NodeFailed)] == ["boom"]
    finally:
        await rig.aclose()


async def test_on_child_error_continue_turns_a_failure_into_a_value() -> None:
    """R-W-7: the join downstream sees a hole it can recognise, not a dead run."""

    wf = Workflow("w", input="x")
    wf.fan_out(
        "branches",
        {"ok": Func(shout), "bad": Func(explode)},
        on_child_error="continue",
    )

    rig = build_graph_rig(wf)
    try:
        result = await rig.controller.run(timeout=5)
        assert result[0] == "X"
        assert result[1]["failed"] is True
        assert result[1]["error_type"] == "Boom"
        assert result[1]["node_id"] == "branches/bad"
        assert rig.controller.state is RunState.COMPLETED
    finally:
        await rig.aclose()


async def test_a_task_group_failure_is_unwrapped_to_the_child_exception() -> None:
    """A crashed branch must not arrive as an ExceptionGroup wrapping it."""

    wf = Workflow("w", input="x")
    wf.fan_out("branches", {"a": Func(shout), "bad": Func(explode)})

    rig = build_graph_rig(wf)
    try:
        with pytest.raises(Boom):
            await rig.controller.run(timeout=5)
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# The memo (R-W-6)
# ---------------------------------------------------------------------------


async def test_a_completed_node_is_not_run_again_within_one_process() -> None:
    """The memo is checked before anything else in `execute`."""

    calls: list[int] = []

    def once(_: Any) -> str:
        calls.append(1)
        return "value"

    wf = Workflow("w", input=None)
    wf.func("once", once)

    rig = build_graph_rig(wf)
    try:
        await rig.controller.run(timeout=5)
        # A second walk over the same controller: every node is already memoized.
        from azalabscode import Runner

        runner = Runner(wf.compile())
        assert await runner.run(rig.controller) == "value"
        assert calls == [1]
    finally:
        await rig.aclose()


async def test_node_outputs_are_recorded_in_the_session() -> None:
    wf = Workflow("w", input="x")
    wf.func("f", shout)

    rig = build_graph_rig(wf)
    try:
        await rig.controller.run(timeout=5)
        session = rig.controller.session()
        assert session.nodes["f"].output is not None
        assert session.nodes["f"].output.inline == "X"
        assert session.completed_nodes() == ["f"]
    finally:
        await rig.aclose()


async def test_node_state_is_checkpointed_onto_the_record() -> None:
    """R-W-5: the runner stores the declared `State`, not just validates it."""

    wf = Workflow("w", input=None)
    wf.node("stepper", SteppingNode(steps=2))

    rig = build_graph_rig(wf)
    try:
        await rig.controller.run(timeout=5)
        state = rig.controller.node_state("stepper")
        assert state is not None
        assert state["state"]["done"] == 2
        assert state["state"]["trail"] == ["s0", "s1"]
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# Pause on a graph with no agents (M5, handoff item 3)
# ---------------------------------------------------------------------------


async def test_a_graph_of_func_nodes_reaches_paused() -> None:
    """Without `enter_node` there is nothing to count and nothing to park."""

    node = SteppingNode(steps=4, gated=True)
    wf = Workflow("w", input=None)
    wf.node("stepper", node)

    rig = build_graph_rig(wf)
    try:
        await rig.controller.start()
        await node.step()
        await rig.wait_until(lambda: len(node.ran) >= 1)

        await rig.controller.pause()
        node.gated = False  # from here the node runs freely to its next safe point
        node.gate.set()
        assert await rig.wait_state(RunState.PAUSED) is RunState.PAUSED
        assert rig.controller.nonquiescent == 0
        assert len(node.ran) < 4  # parked before finishing

        await rig.controller.resume()
        assert await rig.finish() == ["s0", "s1", "s2", "s3"]
    finally:
        await rig.aclose()


async def test_a_pause_between_two_nodes_does_not_walk_on() -> None:
    """The runner's own quiescence key: quiescence over an empty set is vacuous."""

    started: list[str] = []

    def mark(name: str) -> Any:
        async def body(_: Any) -> str:
            started.append(name)
            return name

        return body

    wf = Workflow("w", input=None)
    first = wf.func("first", mark("first"))
    wf.func("second", mark("second"), input=first)

    rig = build_graph_rig(wf)
    try:
        await rig.controller.pause()
        await rig.controller.start()
        assert await rig.wait_state(RunState.PAUSED) is RunState.PAUSED
        assert started == []  # parked before the first node body

        await rig.controller.resume()
        assert await rig.finish() == "second"
        assert started == ["first", "second"]
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# AgentNode and ModelCall
# ---------------------------------------------------------------------------


async def test_an_agent_node_runs_a_loop_and_returns_its_final_text(tmp_path: Any) -> None:
    from azalabscode import ToolContext, ToolDispatcher

    provider = answers("the answer")
    wf = Workflow("w", input="do the thing", provider=provider)
    controller_holder: dict[str, Any] = {}

    rig = build_graph_rig(wf)
    controller_holder["c"] = rig.controller
    wf.dispatcher = ToolDispatcher(
        [], context=ToolContext(workspace_root=tmp_path), gate=rig.controller.gate
    )
    wf.node("writer", AgentNode(AgentSpec(name="scribe", model="fake/model")))
    rig.controller.bind_workflow(wf)

    try:
        assert await rig.controller.run(timeout=5) == "the answer"
        assert "scribe" in rig.controller.agents
        assert rig.controller.agents["scribe"].messages
    finally:
        await rig.aclose()


async def test_a_model_call_streams_under_the_node_id() -> None:
    """A fan-out branch has to be routable by `agent_id` without being an agent."""

    from azalabscode import ModelDelta

    wf = Workflow("w", input="q", provider=answers("one two"))
    wf.node("call", ModelCall("fake/model"))

    rig = build_graph_rig(wf)
    try:
        await rig.controller.run(timeout=5)
        deltas = rig.of_type(ModelDelta)
        assert deltas
        assert {e.agent_id for e in deltas} == {"call"}
        assert {e.node_id for e in deltas} == {"call"}
    finally:
        await rig.aclose()


async def test_a_fan_out_of_model_calls_gives_each_branch_its_own_provider() -> None:
    """M5 trap 5: one shared script interleaves across the whole run, not per branch."""

    wf = Workflow("w", input="q")
    wf.fan_out(
        "models",
        {
            "a": ModelCall("m/a", provider=answers("alpha")),
            "b": ModelCall("m/b", provider=answers("beta")),
        },
    )

    rig = build_graph_rig(wf)
    try:
        assert await rig.controller.run(timeout=5) == ["alpha", "beta"]
    finally:
        await rig.aclose()


async def test_node_completed_events_name_every_node() -> None:
    wf = Workflow("w", input="q")
    wf.map("each", Func(shout), over=["a", "b"])

    rig = build_graph_rig(wf)
    try:
        await rig.controller.run(timeout=5)
        assert {e.node_id for e in rig.of_type(NodeCompleted)} == {
            "each",
            "each/0",
            "each/1",
        }
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# The runner's own quiescence entry -- companions that make the pause tests
# above unable to be green for the wrong reason.
# ---------------------------------------------------------------------------


async def test_the_runner_and_the_node_hold_separate_quiescence_entries() -> None:
    """The node is `running`; the runner that handed it the work is not.

    A runner claiming to be `running` for the length of every node would make PAUSED
    unreachable; a runner with no entry at all leaves the quiescence map empty
    between two nodes, where quiescence is vacuously true and a pause would be
    declared over a run that is still walking.
    """

    holder: dict[str, Any] = {}
    seen: dict[str, Any] = {}

    async def body(_: Any) -> str:
        controller = holder["controller"]
        seen["node"] = controller.phase_of("f")
        seen["runner"] = controller.phase_of("@w")
        seen["nonquiescent"] = controller.nonquiescent
        return "done"

    wf = Workflow("w", input=None)
    wf.func("f", body)

    rig = build_graph_rig(wf)
    holder["controller"] = rig.controller
    try:
        await rig.controller.run(timeout=5)
        assert seen["node"] is AgentPhase.RUNNING
        assert seen["runner"] is AgentPhase.BLOCKED_ON_CHILD
        assert seen["nonquiescent"] == 1
    finally:
        await rig.aclose()


async def test_a_pause_waits_for_a_node_that_is_still_working() -> None:
    """R-C-3 on a graph: PAUSED means the node reached a safe point, not that it was asked to."""

    node = SteppingNode(steps=2, gated=True)
    wf = Workflow("w", input=None)
    wf.node("stepper", node)

    rig = build_graph_rig(wf)
    try:
        await rig.controller.start()
        await rig.wait_until(node.reached.is_set)  # blocked at the gate, phase running

        await rig.controller.pause()
        await asyncio.sleep(0.05)
        assert rig.controller.state is RunState.PAUSING
        assert rig.controller.nonquiescent >= 1
        assert node.ran == []

        node.gated = False
        node.gate.set()
        assert await rig.wait_state(RunState.PAUSED) is RunState.PAUSED
        assert rig.controller.nonquiescent == 0

        await rig.controller.resume()
        await rig.finish()
    finally:
        await rig.aclose()


async def test_a_run_of_memoized_nodes_still_parks() -> None:
    """The runner's own safe point, and the only case nothing else covers.

    A node whose output is memoized returns without entering, without a context and
    therefore without a safe point of its own. A graph that is entirely memoized --
    a resumed run replaying its completed prefix -- would otherwise walk from end to
    end ignoring a pause that had already been requested.
    """

    from azalabscode import Runner

    ran: list[str] = []

    wf = Workflow("w", input=None)
    wf.func("a", lambda _: ran.append("a") or "A")
    wf.func("b", lambda _: ran.append("b") or "B", input=NodeRef("a"))

    rig = build_graph_rig(wf)
    try:
        await rig.controller.node_finished("a", "A")
        await rig.controller.node_finished("b", "B")
        await rig.controller.pause()

        runner = Runner(wf.compile())
        task = asyncio.create_task(runner.run(rig.controller))
        await rig.wait_until(lambda: rig.controller.phase_of("@w") is AgentPhase.PARKED)
        assert not task.done()
        assert ran == []

        await rig.controller.resume()
        assert await asyncio.wait_for(task, timeout=5) == "B"
        assert ran == []  # memoized: neither body ran (R-W-6)
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# R-W-8 and the model-driven half of R-W-4
# ---------------------------------------------------------------------------


async def test_the_single_agent_loop_is_a_short_workflow(tmp_path: Any) -> None:
    """R-W-8: the simple case stays simple. Everything below is the user's code.

    Eleven lines including tool registration and provider setup, and it is the shape
    M6's coding agent will have -- an `AgentNode` over `default_registry()` with a
    real provider substituted for the fake.
    """

    from azalabscode import ToolContext, ToolDispatcher, read_only_registry

    def build(_config: Any) -> Workflow:
        wf = Workflow("agent", input="summarize the repo")
        wf.provider = answers("done")
        wf.dispatcher = ToolDispatcher(
            read_only_registry(), context=ToolContext(workspace_root=tmp_path)
        )
        wf.agent("main", AgentSpec(name="assistant", model="fake/model", max_turns=8))
        return wf

    rig = build_graph_rig(build({}))
    try:
        assert await rig.controller.run(timeout=5) == "done"
    finally:
        await rig.aclose()


async def test_an_agent_node_may_delegate_through_the_delegate_tool(tmp_path: Any) -> None:
    """R-W-4's model-driven half, inside a graph.

    The child's ids come from `AgentState.next_child_id` -- `<agent>/<n>` -- which is
    a different namespace from the node's `<node>/agent/<n>`, and this is where the
    two meet without aliasing.
    """

    from azalabscode import ToolContext, ToolDispatcher
    from azalabscode.providers.testing import ScriptedToolCall, ScriptedTurn
    from azalabscode.tools.builtin.delegate import DelegateTool

    provider = FakeProvider(
        [
            ScriptedTurn(
                tool_calls=[
                    ScriptedToolCall(
                        call_id="c1",
                        name="delegate",
                        arguments={"spec": "explorer", "task": "look around"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(text="the child found nothing"),
            ScriptedTurn(text="child answer"),
        ]
    )
    explorer = AgentSpec(name="explorer", model="fake/model")
    boss = AgentSpec(
        name="boss",
        model="fake/model",
        allow_delegate=True,
        subagents=["explorer"],
        tools=["delegate"],
    )

    wf = Workflow("w", input="go", specs={"explorer": explorer})
    wf.provider = provider
    wf.dispatcher = ToolDispatcher([DelegateTool()], context=ToolContext(workspace_root=tmp_path))
    wf.agent("lead", boss, specs={"explorer": explorer})

    rig = build_graph_rig(wf)
    try:
        await rig.controller.run(timeout=5)
        assert sorted(rig.controller.agents) == ["boss", "boss/0"]
        assert rig.controller.agents["boss/0"].parent_id == "boss"
    finally:
        await rig.aclose()


async def test_a_runner_walks_a_graph_with_no_run_around_it() -> None:
    """`control=None`: no memo, no safe points, no quiescence. The standalone shape."""

    from azalabscode import Runner

    wf = Workflow("w", input="hello")
    wf.output(wf.func("shout", shout))
    assert await Runner(wf.compile()).run() == "HELLO"
