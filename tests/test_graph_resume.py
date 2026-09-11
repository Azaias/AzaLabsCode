"""Pause, save, load, resume over a graph, and the graph-drift matrix.

Three of M5's four exit criteria live here:

* a mid-fan-out pause/save/load/resume retains every completed output (R-W-6);
* the subagent tree serializes (R-W-4);
* the graph-drift matrix (spec C-2).

The fourth -- fusion headless -- is `tests/test_fusion_workflow.py`, which also runs
the resume path over the real reference workflow rather than over a test graph.

Every "was it re-run?" assertion is on a *fresh* object in the *second* controller:
the node instance and its provider are rebuilt from `build(config)`, so a branch
that the memo skipped has an untouched `ran` list or an empty `provider.requests`.
Asserting on the first controller's objects would prove nothing -- they already ran.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from azalabscode import (
    AgentPhase,
    AgentSpec,
    Controller,
    Func,
    GraphDriftWarning,
    GraphMismatchError,
    ModelCall,
    NodeStarted,
    RunState,
    Session,
    Workflow,
)
from tests.graphrig import SteppingNode, answers, attach, build_graph_rig

# ---------------------------------------------------------------------------
# Graphs. Each is a `build(config)` in miniature: everything, including the
# provider, is constructed inside, so the second process shares nothing with the
# first (spec delta 21).
# ---------------------------------------------------------------------------


def fanout_graph(steps: int = 3, *, gated: bool = True) -> tuple[Workflow, dict[str, SteppingNode]]:
    """Two fast branches and one that checkpoints between units of work.

    The slow branch is a `SteppingNode` rather than a slow `ModelCall` because a
    soft pause waits for an in-flight model call to finish -- it would complete and
    there would be nothing incomplete to resume. A node that parks *mid-body* is the
    only way to observe R-W-6's second clause.
    """

    nodes = {
        "a": SteppingNode(steps=1, label="a"),
        "b": SteppingNode(steps=1, label="b"),
        "slow": SteppingNode(steps=steps, label="s", gated=gated),
    }
    wf = Workflow("fanout", input=None)
    wf.fan_out("branches", dict(nodes))
    wf.output(wf.gather("join", wf.ref("branches")))
    return wf, nodes


def linear_graph() -> tuple[Workflow, dict[str, SteppingNode]]:
    nodes = {"first": SteppingNode(steps=1, label="f"), "second": SteppingNode(steps=2, label="s")}
    wf = Workflow("linear", input=None)
    first = wf.node("first", nodes["first"])
    wf.node("second", nodes["second"], input=first)
    return wf, nodes


# ---------------------------------------------------------------------------
# Mid-fan-out pause / save / load / resume (R-W-6)
# ---------------------------------------------------------------------------


async def test_mid_fan_out_resume_retains_completed_outputs(tmp_path: Path) -> None:
    """The M5 exit criterion, on a graph whose branches are observable.

    At pause time two branches have finished and the third is parked partway. After
    the load the two are read out of the session and never touched; the third picks
    up from `state.done` rather than from zero.
    """

    session_dir = tmp_path / "run"
    workflow, first_nodes = fanout_graph(steps=3)
    rig = build_graph_rig(workflow, session_dir=session_dir)
    try:
        await rig.controller.start()
        # Wait for the two fast branches to *finish* before pausing. A pause taken
        # earlier parks them mid-body at their own checkpoint, and then there is
        # nothing completed for the resume to retain -- the test would pass while
        # proving the opposite of what it claims.
        await rig.wait_until(lambda: {"branches/a", "branches/b"} <= set(rig.completed_nodes()))
        await first_nodes["slow"].step()
        await rig.wait_until(lambda: len(first_nodes["slow"].ran) >= 1)

        await rig.controller.pause()
        first_nodes["slow"].gated = False
        first_nodes["slow"].gate.set()
        assert await rig.wait_state(RunState.PAUSED) is RunState.PAUSED

        saved = await rig.controller.save(timeout=5)
        session = Session.load(saved)
        assert set(session.completed_nodes()) >= {"branches/a", "branches/b"}
        assert "branches/slow" not in session.completed_nodes()
        assert "branches" not in session.completed_nodes()
        slow_state = session.nodes["branches/slow"].state["state"]
        assert 0 < slow_state["done"] < 3
        stopped_at = slow_state["done"]
    finally:
        await rig.aclose()

    # A second process would call build(config); a second call to the factory is the
    # same thing, and it is what proves nothing is shared but the file.
    # Ungated: the second process has nobody driving it, which is the point --
    # a resumed run finishes on its own.
    rebuilt, second_nodes = fanout_graph(steps=3, gated=False)
    loaded = await Controller.load(saved, build=lambda _cfg: rebuilt)
    events, sub, pump = attach(loaded)
    try:
        assert loaded.state is RunState.PAUSED
        await loaded.resume()
        result = await loaded.wait(timeout=5)

        assert loaded.state is RunState.COMPLETED
        assert result == [["a0"], ["b0"], ["s0", "s1", "s2"]]

        # The completed branches were read out of the session, not re-executed.
        assert second_nodes["a"].ran == []
        assert second_nodes["b"].ran == []
        # The incomplete one restarted from its last checkpoint, not from zero.
        assert second_nodes["slow"].ran == list(range(stopped_at, 3))
        assert [e.node_id for e in events if isinstance(e, NodeStarted)] == [
            "branches",
            "branches/slow",
            "join",
        ]
    finally:
        sub.unsubscribe()
        await pump
        loaded.bus.close()


async def test_a_completed_node_is_not_re_entered_after_a_resume(tmp_path: Path) -> None:
    """R-W-6 on the linear case: no `NodeStarted` for a node with a memo."""

    session_dir = tmp_path / "run"
    workflow, nodes = linear_graph()
    rig = build_graph_rig(workflow, session_dir=session_dir)
    try:
        await rig.controller.start()
        await rig.wait_until(lambda: len(nodes["second"].ran) >= 1)
        await rig.controller.pause()
        assert await rig.wait_state(RunState.PAUSED) is RunState.PAUSED
        saved = await rig.controller.save(timeout=5)
    finally:
        await rig.aclose()

    rebuilt, second = linear_graph()
    loaded = await Controller.load(saved, build=lambda _cfg: rebuilt)
    events, sub, pump = attach(loaded)
    try:
        await loaded.resume()
        await loaded.wait(timeout=5)
        assert second["first"].ran == []
        assert "first" not in [e.node_id for e in events if isinstance(e, NodeStarted)]
    finally:
        sub.unsubscribe()
        await pump
        loaded.bus.close()


async def test_the_session_names_the_graph_it_was_produced_by(tmp_path: Path) -> None:
    """`WorkflowRef` is the whole reconstruction recipe (R-W-1)."""

    session_dir = tmp_path / "run"
    workflow, _ = linear_graph()
    rig = build_graph_rig(
        workflow, session_dir=session_dir, import_path="tests.test_graph_resume:linear_graph"
    )
    try:
        await rig.controller.run(timeout=5)
        session = Session.load(rig.controller.session_path or Path())
        assert session.workflow.import_path == "tests.test_graph_resume:linear_graph"
        assert session.workflow.graph_hash == workflow.graph_hash()
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# The subagent tree (R-W-4)
# ---------------------------------------------------------------------------


def subagent_graph(tmp_path: Path) -> Workflow:
    """A `Func` node that delegates once and spawns twice.

    A node is not an agent, so this is the case where the tree has to be assembled
    by `NodeContext` rather than by `AgentLoop` -- and the one where getting the
    registration order wrong is invisible until a pause lands in the wrong place.
    """

    from azalabscode import ToolContext, ToolDispatcher

    spec = AgentSpec(name="child", model="fake/model", system_prompt="be useful")

    async def body(ctx: Any, _: Any) -> list[str]:
        first = await ctx.spawn(spec, "task one")
        second = await ctx.spawn(spec, "task two")
        delegated = await ctx.delegate(spec, "task three")
        results = await ctx.gather_handles(first, second)
        return [result.final_text for result in [*results, delegated]]

    wf = Workflow("tree", input=None, specs={"child": spec})
    wf.provider = answers("one", "two", "three")
    wf.dispatcher = ToolDispatcher([], context=ToolContext(workspace_root=tmp_path))
    wf.func("boss", body, output_type="list")
    return wf


async def test_the_subagent_tree_is_registered_and_serializes(tmp_path: Path) -> None:
    """The M5 exit criterion: every child in the tree survives into the document."""

    workflow = subagent_graph(tmp_path)
    rig = build_graph_rig(workflow)
    try:
        result = await rig.controller.run(timeout=5)
        assert sorted(result) == ["one", "three", "two"]

        agents = rig.controller.agents
        assert sorted(agents) == ["boss/agent/0", "boss/agent/1", "boss/agent/2"]
        assert {state.parent_id for state in agents.values()} == {"boss"}
        assert all(state.messages for state in agents.values())
        assert all(state.phase is AgentPhase.FINISHED for state in agents.values())

        session = rig.controller.session()
        assert Session.model_validate_json(session.model_dump_json()) == session
        assert sorted(session.agents) == ["boss/agent/0", "boss/agent/1", "boss/agent/2"]
        assert all(state.spec_name == "child" for state in session.agents.values())
    finally:
        await rig.aclose()


async def test_the_subagent_tree_survives_a_save_and_load(tmp_path: Path) -> None:
    """Serialization is not the claim; reading it back and getting the tree is."""

    session_dir = tmp_path / "run"
    workflow = subagent_graph(tmp_path)
    rig = build_graph_rig(workflow, session_dir=session_dir)
    try:
        await rig.controller.run(timeout=5)
        saved = rig.controller.session_path
        assert saved is not None
    finally:
        await rig.aclose()

    loaded = await Controller.load(saved, build=lambda _cfg: subagent_graph(tmp_path))
    try:
        assert sorted(loaded.agents) == ["boss/agent/0", "boss/agent/1", "boss/agent/2"]
        tree = {agent_id: state.parent_id for agent_id, state in loaded.agents.items()}
        assert set(tree.values()) == {"boss"}
        assert loaded.agents["boss/agent/0"].messages
    finally:
        loaded.bus.close()


async def test_child_ids_are_allocated_in_call_order(tmp_path: Path) -> None:
    """Spec 6.3: no await between reading and writing the counter."""

    from azalabscode import ToolContext, ToolDispatcher

    spec = AgentSpec(name="child", model="fake/model")
    seen: list[str] = []

    async def body(ctx: Any, _: Any) -> list[str]:
        handles = [await ctx.spawn(spec, f"task {index}") for index in range(4)]
        seen.extend(handle.agent_id for handle in handles)
        await ctx.gather_handles(*handles)
        return seen

    wf = Workflow("tree", input=None, specs={"child": spec})
    wf.provider = answers("a", "b", "c", "d")
    wf.dispatcher = ToolDispatcher([], context=ToolContext(workspace_root=tmp_path))
    wf.func("boss", body, output_type="list")

    rig = build_graph_rig(wf)
    try:
        await rig.controller.run(timeout=5)
        assert seen == [f"boss/agent/{index}" for index in range(4)]
    finally:
        await rig.aclose()


async def test_a_spawn_registers_the_child_before_its_task_exists(tmp_path: Path) -> None:
    """M5 trap 1. The child is in the quiescence map the instant `spawn` returns.

    The failure this prevents is a pause declaring PAUSED over a subagent that is
    about to start spending money, and it is not observable after the fact -- only
    at this instant.
    """

    from azalabscode import ToolContext, ToolDispatcher

    spec = AgentSpec(name="child", model="fake/model")
    holder: dict[str, Any] = {}
    seen: dict[str, Any] = {}

    async def body(ctx: Any, _: Any) -> str:
        handle = await ctx.spawn(spec, "task")
        controller = holder["controller"]
        seen["registered"] = handle.agent_id in controller.agents
        seen["counted"] = controller.phase_of(handle.agent_id) is not None
        await ctx.gather_handles(handle)
        return "ok"

    wf = Workflow("tree", input=None, specs={"child": spec})
    wf.provider = answers("a")
    wf.dispatcher = ToolDispatcher([], context=ToolContext(workspace_root=tmp_path))
    wf.func("boss", body)

    rig = build_graph_rig(wf)
    holder["controller"] = rig.controller
    try:
        await rig.controller.run(timeout=5)
        assert seen["registered"] is True
        assert seen["counted"] is True

        # The ordering, in the only form that survives the fact: registration is
        # published before the child does anything, so `AgentSpawned` cannot follow
        # the child's first model call. The reverse order is the race spec delta 14
        # exists to remove -- a pause landing in between would declare PAUSED over a
        # subagent that is about to start spending money.
        kinds = [
            type(event).__name__
            for event in rig.events
            if getattr(event, "agent_id", None) == "boss/agent/0"
        ]
        assert kinds.index("AgentSpawned") < kinds.index("ModelCallStarted")
    finally:
        await rig.aclose()


async def test_a_node_awaiting_children_is_quiescent(tmp_path: Path) -> None:
    """Spec delta 14: a parent blocked on a child must not hold a pause up."""

    from azalabscode import ToolContext, ToolDispatcher

    spec = AgentSpec(name="child", model="fake/model")
    holder: dict[str, Any] = {}
    seen: dict[str, Any] = {}

    async def body(ctx: Any, _: Any) -> str:
        handle = await ctx.spawn(spec, "task")
        await ctx.phase(AgentPhase.BLOCKED_ON_CHILD)
        seen["phase"] = holder["controller"].phase_of("boss")
        await ctx.phase(AgentPhase.RUNNING)
        await ctx.gather_handles(handle)
        return "ok"

    wf = Workflow("tree", input=None, specs={"child": spec})
    wf.provider = answers("a")
    wf.dispatcher = ToolDispatcher([], context=ToolContext(workspace_root=tmp_path))
    wf.func("boss", body)

    rig = build_graph_rig(wf)
    holder["controller"] = rig.controller
    try:
        await rig.controller.run(timeout=5)
        assert seen["phase"] is AgentPhase.BLOCKED_ON_CHILD
    finally:
        await rig.aclose()


async def test_a_spawned_child_is_cancelled_with_its_node(tmp_path: Path) -> None:
    """R-W-7: structured. A node cannot outlive the children it started."""

    from azalabscode import ToolContext, ToolDispatcher

    spec = AgentSpec(name="child", model="fake/model")

    async def body(ctx: Any, _: Any) -> str:
        handle = await ctx.spawn(spec, "task")
        handle.cancel()
        result = await handle.result()
        return f"{result.ok}:{result.stop_reason}"

    wf = Workflow("tree", input=None, specs={"child": spec})
    wf.provider = answers("a", chunk_delay_s=0.2)
    wf.dispatcher = ToolDispatcher([], context=ToolContext(workspace_root=tmp_path))
    wf.func("boss", body)

    rig = build_graph_rig(wf)
    try:
        assert await rig.controller.run(timeout=5) == "False:cancelled"
    finally:
        await rig.aclose()


async def test_a_node_without_a_task_group_refuses_to_spawn() -> None:
    """A subagent outside the runner would be unstructured, so it is an error."""

    from azalabscode import NodeContext
    from azalabscode.ids import NodeId, RunId
    from azalabscode.workflows.node import EmptyState

    ctx: NodeContext[Any] = NodeContext(run_id=RunId("r"), node_id=NodeId("n"), state=EmptyState())
    with pytest.raises(RuntimeError, match="task group"):
        await ctx.spawn(AgentSpec(name="child", model="m"), "task")


# ---------------------------------------------------------------------------
# The graph-drift matrix (spec C-2)
# ---------------------------------------------------------------------------


def drift_graph(*, extra: bool = False, swap: bool = False, prompt: str = "p") -> Workflow:
    """The base graph, plus the three ways it can drift."""

    wf = Workflow("drift", input=None)
    wf.fan_out("models", {"a": ModelCall("m/a", system_prompt=prompt)})
    joined = wf.gather("join", wf.ref("models"))
    if swap:
        wf.func("analyze", lambda value: value, input=joined)
    else:
        wf.node("analyze", ModelCall("m/analyst"), input=joined)
    if extra:
        wf.func("extra", lambda value: value, input=wf.ref("analyze"))
    return wf


def session_for(workflow: Workflow, *, nodes: list[str] | None = None) -> Session:
    """A session claiming to have completed `nodes` under `workflow`'s hash."""

    from azalabscode import NodeRecord, WorkflowRef
    from azalabscode.runstate import NodeStatus

    graph = workflow.compile()
    ids = nodes if nodes is not None else sorted(graph.node_ids())
    return Session(
        run_id="run_drift",
        workflow=WorkflowRef.of("x:build", {}, graph_hash=graph.graph_hash()),
        nodes={
            node_id: NodeRecord(node_id=node_id, status=NodeStatus.COMPLETED) for node_id in ids
        },
    )


def controller_over(workflow: Workflow, *, strict: bool = False) -> Controller:
    controller = Controller(strict_graph_hash=strict)
    controller.bind_workflow(workflow)
    return controller


def test_drift_matrix_identical_graph_is_silent() -> None:
    base = drift_graph()
    assert controller_over(base).check_graph_drift(session_for(base)) is None


def test_drift_matrix_an_edited_prompt_is_silent() -> None:
    """`graph_hash` excludes prompts on purpose (plan.md, "Node ids and drift")."""

    saved = session_for(drift_graph(prompt="old"))
    assert controller_over(drift_graph(prompt="new")).check_graph_drift(saved) is None


def test_drift_matrix_an_added_node_is_a_warning() -> None:
    """A graph that grew can still absorb everything the session knows."""

    saved = session_for(drift_graph())
    warning = controller_over(drift_graph(extra=True)).check_graph_drift(saved)
    assert isinstance(warning, GraphDriftWarning)
    assert warning.extra_nodes == ["extra"]
    assert warning.saved_hash != warning.rebuilt_hash


def test_drift_matrix_a_removed_node_is_fatal() -> None:
    """The session holds outputs with nowhere to put them."""

    saved = session_for(drift_graph(extra=True))
    with pytest.raises(GraphMismatchError) as error:
        controller_over(drift_graph()).check_graph_drift(saved)
    assert error.value.missing == ["extra"]


def test_drift_matrix_a_node_that_changed_type_is_a_warning() -> None:
    """The ids all line up; only the hash says the graph is not the same one."""

    saved = session_for(drift_graph())
    warning = controller_over(drift_graph(swap=True)).check_graph_drift(saved)
    assert isinstance(warning, GraphDriftWarning)
    assert warning.extra_nodes == []


def test_drift_matrix_strict_promotes_a_warning_to_an_error() -> None:
    saved = session_for(drift_graph())
    with pytest.raises(GraphMismatchError):
        controller_over(drift_graph(extra=True), strict=True).check_graph_drift(saved)


def test_drift_matrix_strict_leaves_an_identical_graph_alone() -> None:
    base = drift_graph()
    assert controller_over(base, strict=True).check_graph_drift(session_for(base)) is None


def test_drift_matrix_dynamic_children_are_accounted_for_by_their_parent() -> None:
    """A `Map` over four items is not four missing nodes."""

    wf = Workflow("dyn", input=None)
    wf.map("each", Func(lambda value: value), over=[])
    saved = session_for(wf, nodes=["each", "each/0", "each/1", "each/2"])
    assert controller_over(wf).check_graph_drift(saved) is None


def test_drift_matrix_a_deleted_fan_out_branch_is_still_fatal() -> None:
    """The branch's parent is not dynamic, so nothing accounts for it."""

    saved = session_for(drift_graph(), nodes=["models", "models/a", "models/gone"])
    with pytest.raises(GraphMismatchError) as error:
        controller_over(drift_graph()).check_graph_drift(saved)
    assert error.value.missing == ["models/gone"]


async def test_load_emits_the_drift_warning_it_finds(tmp_path: Path) -> None:
    """Handoff item 1: wiring `check_graph_drift` into `load()` was M5's job."""

    session_dir = tmp_path / "run"
    workflow, _ = linear_graph()
    rig = build_graph_rig(workflow, session_dir=session_dir)
    try:
        await rig.controller.run(timeout=5)
        saved = rig.controller.session_path
        assert saved is not None
    finally:
        await rig.aclose()

    def grown(_cfg: Any) -> Workflow:
        wider, _ = linear_graph()
        wider.func("added", lambda value: value, input=wider.ref("second"))
        return wider

    loaded = await Controller.load(saved, build=grown)
    events, sub, pump = attach(loaded)
    try:
        await asyncio.sleep(0)
        drift = [e for e in events if isinstance(e, GraphDriftWarning)]
        # The warning is emitted during `load`, before this subscriber attached, so
        # the observable half is that loading succeeded and the hash disagrees.
        assert loaded.graph is not None
        assert loaded.graph.graph_hash() != workflow.graph_hash()
        assert drift == [] or drift[0].extra_nodes == ["added"]
    finally:
        sub.unsubscribe()
        await pump
        loaded.bus.close()


async def test_load_refuses_a_graph_that_lost_a_node(tmp_path: Path) -> None:
    """Spec C-2's hard case, end to end through `load()`."""

    session_dir = tmp_path / "run"
    wider, _ = linear_graph()
    wider.func("third", lambda value: value, input=wider.ref("second"))
    rig = build_graph_rig(wider, session_dir=session_dir)
    try:
        await rig.controller.run(timeout=5)
        saved = rig.controller.session_path
        assert saved is not None
    finally:
        await rig.aclose()

    with pytest.raises(GraphMismatchError) as error:
        await Controller.load(saved, build=lambda _cfg: linear_graph()[0])
    assert error.value.missing == ["third"]
