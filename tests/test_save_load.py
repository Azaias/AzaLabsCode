"""`save` and `load` in one interpreter: the half of M3 the kill test cannot isolate.

The kill test proves the whole cycle survives a process death. It cannot say *which*
part failed when it fails, and it cannot reach the states a crash does not happen to
land in. This file drives each piece on its own:

* a save from every run state R-C-10 says it must be callable from,
* `SaveTimeout` naming the blocking step rather than hanging behind a long tool,
* R-C-9's approval surviving a save/load and its resolution reaching the re-issued
  call,
* R-C-13's interrupted tool call, answered once and never re-executed,
* R-W-6's memoized node outputs,
* spec delta 16's resumed delegate, which the kill test does not cover at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from azalabscode.control import Controller, QueueApprovalHandler, WorkflowRef
from azalabscode.control.session import Session
from azalabscode.errors import CheckpointError, SaveTimeout
from azalabscode.events import Checkpoint
from azalabscode.ids import MAIN_AGENT, CallId
from azalabscode.messages import ToolResultMessage, assert_transcript_valid
from azalabscode.permissions import PermissionMode
from azalabscode.runstate import TERMINAL_STATES, RunState
from azalabscode.toolio import ToolErrorKind
from azalabscode.tools.builtin.delegate import DelegateTool
from azalabscode.workflows.agent_loop import AgentSpec
from tests.harness import (
    BOUND,
    Rig,
    build_rig,
    calls,
    echo_call,
    says,
    touch_call,
)


@pytest.fixture
async def rigs(workspace: Path, tmp_path: Path):
    """Builds rigs with a session directory and guarantees teardown."""

    made: list[Rig] = []

    def make(turns, *, name: str = "run", **kwargs) -> Rig:
        session_dir = tmp_path / "sessions" / name
        kwargs.setdefault("session_dir", session_dir)
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


_PARENT_SPEC = AgentSpec(
    name="main",
    model="fake/model",
    system_prompt="be terse",
    allow_delegate=True,
    subagents=["explorer"],
)
_CHILD_SPEC = AgentSpec(name="explorer", model="fake/model", system_prompt="explore")


def _delegate_turns():
    """The script, in the order the two agents actually consume it.

    `FakeProvider` hands out unkeyed turns in order across the whole run, not per
    agent, so the child's turns sit between the parent's two. Getting this wrong
    gives the child the parent's answer, which looks like a delegation bug and is
    not one.
    """

    return [
        calls(("delegate", {"task": "look at the thing", "spec": "explorer"})),
        calls(echo_call("child-step", sleep=30.0)),
        says("child answer"),
        says("the child said so"),
    ]


def _resumed_delegate_turns():
    """What the *second* process is asked for: the child finishes, then the parent.

    A separate script rather than a replay of the first: the resumed run issues
    different requests (the child's transcript now carries an `interrupted` result),
    and matching those by hash is the kill test's job, not this one's.
    """

    return [says("child answer"), says("the child said so")]


# ---------------------------------------------------------------------------
# Autosave at every safe point
# ---------------------------------------------------------------------------


async def test_every_durable_safe_point_writes_the_session(rigs) -> None:
    """R-C-10's "auto-checkpointing after every safe point is on by default".

    The rhythm is the M2 one -- turn_start, after_model_call, after_tool_batch -- and
    every one of them now leaves a file that parses.
    """

    rig = rigs([calls(echo_call("one")), says("done")])
    await rig.controller.start()
    await rig.finish()

    folds = rig.controller.folds
    assert folds and all(f.to_disk for f in folds), "a durable safe point that wrote nothing"
    assert {str(f.kind) for f in folds} >= {"turn_start", "after_model_call", "after_tool_batch"}

    written = [e for e in rig.of_type(Checkpoint) if e.to_disk]
    # One per fold, plus the terminal one: without that last write the newest file
    # on disk would say RUNNING for a run that finished.
    assert len(written) == len(folds) + 1
    assert all(e.path for e in written)
    assert written[-1].kind == "terminal"

    session = Session.load(rig.controller.session_path)
    assert session.run_state is RunState.COMPLETED
    assert [m.role for m in session.agents["main"].messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


async def test_autosave_can_be_turned_off_without_losing_explicit_saves(rigs) -> None:
    """`autosave=False` is the in-memory run that still answers `save()`."""

    rig = rigs([says("done")], autosave=False)
    await rig.controller.start()
    await rig.finish()

    assert not rig.controller.session_path.exists()
    assert all(not f.to_disk for f in rig.controller.folds)

    await rig.controller.save()
    assert Session.load(rig.controller.session_path).run_state is RunState.COMPLETED


async def test_a_save_with_nowhere_to_go_says_so(workspace: Path) -> None:
    """No `session_dir` and no path: a clear error, not an `AttributeError`."""

    controller = Controller(lambda control: asyncio.sleep(0))
    with pytest.raises(CheckpointError, match="session_dir"):
        await controller.save()


# ---------------------------------------------------------------------------
# Save from every run state (spec 11 scenario 7, the disk version)
# ---------------------------------------------------------------------------


async def test_save_works_from_every_run_state(rigs, tmp_path: Path) -> None:
    """R-C-10: "callable in any state". All of CREATED, RUNNING, PAUSING, PAUSED,
    WAITING_APPROVAL, and a terminal state, each producing a loadable document."""

    handler = QueueApprovalHandler()
    rig = rigs(
        [calls(echo_call("one", sleep=0.05)), calls(touch_call("two")), says("done")],
        mode=PermissionMode.MANUAL,
        handler=handler,
    )
    saves = tmp_path / "saves"
    seen: dict[RunState, Session] = {}

    async def snapshot(label: str) -> None:
        state = rig.controller.state
        path = await rig.controller.save(saves / f"{label}.json", timeout=BOUND)
        seen[state] = Session.load(path)

    await snapshot("created")
    await rig.controller.start()
    await snapshot("running")

    await rig.wait_until(lambda: bool(rig.echo.started))
    await rig.controller.pause()
    if rig.controller.state is RunState.PAUSING:
        await snapshot("pausing")
    await rig.wait_state(RunState.PAUSED)
    await snapshot("paused")

    await rig.controller.resume()
    request = await handler.next_request()
    await rig.wait_state(RunState.WAITING_APPROVAL)
    await snapshot("waiting_approval")

    await rig.controller.approve(request.request_id)
    await rig.finish()
    await snapshot("completed")

    assert {
        RunState.CREATED,
        RunState.RUNNING,
        RunState.PAUSED,
        RunState.WAITING_APPROVAL,
        RunState.COMPLETED,
    } <= set(seen)
    for state, session in seen.items():
        assert session.run_state is state
        for agent_id, agent in session.agents.items():
            assert_transcript_valid(agent.messages, agent_id=agent_id)
    assert seen[RunState.WAITING_APPROVAL].pending_approvals, "R-C-9 needs the request in the file"


async def test_a_save_during_a_running_step_waits_for_the_next_safe_point(rigs) -> None:
    """R-C-10: a RUNNING run is checkpointed at its next safe point, and the call
    awaits that. The document therefore never claims a tool finished when it had not."""

    rig = rigs([calls(echo_call("slow", sleep=0.2)), says("done")])
    await rig.controller.start()
    await rig.wait_until(lambda: bool(rig.echo.started))

    assert rig.controller._needs_safe_point()
    session = Session.load(await rig.controller.save(timeout=BOUND))

    assert rig.echo.finished == ["slow"], "the save landed before the tool finished"
    assert session.inflight == [], "a safe point has nothing in flight, by definition"
    await rig.finish()


# ---------------------------------------------------------------------------
# SaveTimeout (delta 17)
# ---------------------------------------------------------------------------


async def test_a_save_behind_a_long_tool_times_out_and_names_it(rigs) -> None:
    """Delta 17. `save()` must not hang for the tool's whole timeout, and the message
    is what spec C-12's status bar renders -- so it names the step, not the run."""

    rig = rigs([calls(echo_call("very-slow", sleep=30.0)), says("done")])
    await rig.controller.start()
    await rig.wait_until(lambda: bool(rig.echo.started))

    with pytest.raises(SaveTimeout) as caught:
        await rig.controller.save(timeout=0.15)

    assert caught.value.timeout_s == 0.15
    assert "main" in caught.value.blocking
    assert "tool call" in caught.value.blocking
    assert "0.15s" in str(caught.value)

    await rig.controller.cancel("done with this test")


async def test_a_timed_out_save_leaves_no_pending_request_behind(rigs) -> None:
    """A second save must not be resolved by the first one's abandoned future."""

    rig = rigs([calls(echo_call("very-slow", sleep=30.0)), says("done")])
    await rig.controller.start()
    await rig.wait_until(lambda: bool(rig.echo.started))

    with pytest.raises(SaveTimeout):
        await rig.controller.save(timeout=0.1)
    assert rig.controller._pending_saves == []

    with pytest.raises(SaveTimeout):
        await rig.controller.save(timeout=0.1)
    await rig.controller.cancel("done with this test")


async def test_the_blocking_description_names_the_agent_when_no_step_is_running(rigs) -> None:
    """The other half of C-12's message: an agent between two safe points."""

    rig = rigs([says("done")])
    assert "never reached a safe point" in rig.controller.blocking_description()


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


async def test_load_returns_a_paused_controller_that_never_auto_starts(rigs) -> None:
    """R-C-11, exactly: PAUSED, and nothing runs until `resume()`."""

    rig = rigs([calls(echo_call("one")), says("done")])
    await rig.controller.start()
    await rig.finish()

    loaded = await Controller.load(rig.controller.session_path, build=lambda _: _never_runs)
    try:
        assert loaded.state is RunState.PAUSED
        assert loaded.run_id == rig.controller.run_id
        assert loaded.permission_mode is rig.controller.permission_mode
        assert sorted(loaded.agents) == ["main"]
        assert loaded.bus.seq >= Session.load(rig.controller.session_path).event_seq
    finally:
        await loaded.cancel("test teardown")


async def _never_runs(control: Any) -> None:
    """A body that would fail the test if `load()` ever started it by itself."""

    raise AssertionError("load() must not auto-start the run (R-C-11)")


async def test_a_loaded_run_keeps_its_completed_node_outputs(rigs) -> None:
    """R-W-6 in one process: the memo survives the round trip through JSON."""

    rig = rigs([says("done")])
    await rig.controller.node_started("prepare")
    await rig.controller.node_finished("prepare", {"value": [1, 2, 3]})
    await rig.controller.save()

    loaded = await Controller.load(rig.controller.session_path, build=lambda _: _never_runs)
    try:
        assert loaded.node_completed("prepare")
        assert loaded.node_output("prepare") == {"value": [1, 2, 3]}
        assert not loaded.node_completed("prepare", attempt=1), "a retry is a different attempt"
        assert not loaded.node_completed("summarize")
        with pytest.raises(KeyError):
            loaded.node_output("summarize")
    finally:
        await loaded.cancel("test teardown")


async def test_a_spilled_node_output_is_read_back_through_the_session_directory(rigs) -> None:
    """A large output lives in `values/` and a loaded controller still returns it."""

    rig = rigs([says("done")])
    payload = {"text": "x" * 40_000}
    await rig.controller.node_finished("big", payload)
    await rig.controller.save()

    session = Session.load(rig.controller.session_path)
    assert session.nodes["big"].output is not None
    assert session.nodes["big"].output.spilled

    loaded = await Controller.load(rig.controller.session_path, build=lambda _: _never_runs)
    try:
        assert loaded.node_output("big") == payload
    finally:
        await loaded.cancel("test teardown")


async def test_load_rejects_a_workflow_it_cannot_import(rigs) -> None:
    """R-C-11's failure path, at the boundary rather than inside `build`."""

    rig = rigs([says("done")], workflow=WorkflowRef.of("tests.gone:build", {}))
    await rig.controller.save()

    from azalabscode.errors import WorkflowNotImportable

    with pytest.raises(WorkflowNotImportable, match=r"tests\.gone:build"):
        await Controller.load(rig.controller.session_path)


# ---------------------------------------------------------------------------
# R-C-13: an interrupted tool call is never re-executed
# ---------------------------------------------------------------------------


async def test_an_inflight_tool_call_reloads_as_interrupted_and_never_re_runs(rigs) -> None:
    """R-C-13 across a simulated process death, in one interpreter.

    The session is written by hand from a live controller mid-batch -- which is what
    a crash leaves -- and then loaded. The call becomes `interrupted`, the tool is
    never asked to run it again, and the transcript is valid.
    """

    rig = rigs([calls(echo_call("slow", sleep=30.0)), says("done")])
    await rig.controller.start()
    await rig.wait_until(lambda: bool(rig.echo.started))

    # The snapshot a crash would have left: taken with the batch step registered.
    crashed = rig.controller.session()
    assert [s.kind for s in crashed.inflight] == ["tool_call"]
    call_ids = crashed.inflight[0].call_ids
    assert call_ids

    path = rig.controller.session_path.parent / "crashed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(crashed.dumps())
    await rig.controller.cancel("simulated process death")

    loaded = await Controller.load(path, build=lambda _: _never_runs)
    try:
        state = loaded.agent(MAIN_AGENT)
        assert state is not None
        results = [m for m in state.messages if isinstance(m, ToolResultMessage)]
        assert [r.call_id for r in results] == call_ids
        assert results[0].result.error is not None
        assert results[0].result.error.kind is ToolErrorKind.INTERRUPTED
        assert "effect unknown" in results[0].result.error.message
        assert state.open_call_ids == []
        assert state.pending_results == {}
        assert_transcript_valid(state.messages, agent_id="main")

        report = loaded.resume_report
        assert report is not None
        assert report.interrupted_calls == call_ids
        assert report.recovered_results == []
    finally:
        await loaded.cancel("test teardown")


async def test_a_result_that_landed_before_the_death_beats_the_interrupted_error(rigs) -> None:
    """`pending_results` is the record of what actually finished (R-C-13's exception).

    A `shell` that completed a millisecond before the kill must not be reported to
    the model as interrupted: its effect is known, and telling the model otherwise
    invites it to run the command twice.
    """

    rig = rigs([calls(echo_call("a"), echo_call("b", sleep=30.0)), says("done")])
    await rig.controller.start()
    await rig.wait_until(lambda: rig.echo.finished == ["a"])

    crashed = rig.controller.session()
    main = crashed.agents["main"]
    assert set(main.pending_results) == {main.open_call_ids[0]}

    path = rig.controller.session_path.parent / "half.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(crashed.dumps())
    await rig.controller.cancel("simulated process death")

    loaded = await Controller.load(path, build=lambda _: _never_runs)
    try:
        state = loaded.agent(MAIN_AGENT)
        assert state is not None
        results = [m for m in state.messages if isinstance(m, ToolResultMessage)]
        assert len(results) == 2
        assert results[0].result.error is None, "the completed call kept its real result"
        assert results[0].result.text == "echo: a"
        assert results[1].result.error is not None
        assert results[1].result.error.kind is ToolErrorKind.INTERRUPTED

        report = loaded.resume_report
        assert report is not None
        assert report.recovered_results == [results[0].call_id]
        assert report.interrupted_calls == [results[1].call_id]
    finally:
        await loaded.cancel("test teardown")


async def test_an_inflight_model_call_is_dropped_rather_than_repaired(rigs) -> None:
    """Spec C-1: nothing was half-appended, so there is nothing to reconcile."""

    rig = rigs([says("slow one", delay_s=30.0), says("done")])
    await rig.controller.start()
    await rig.wait_until(lambda: bool(rig.controller.inflight))

    crashed = rig.controller.session()
    assert [s.kind for s in crashed.inflight] == ["model_call"]
    path = rig.controller.session_path.parent / "midstream.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(crashed.dumps())
    await rig.controller.cancel("simulated process death")

    loaded = await Controller.load(path, build=lambda _: _never_runs)
    try:
        report = loaded.resume_report
        assert report is not None
        assert len(report.dropped_model_calls) == 1
        assert report.interrupted_calls == []
        state = loaded.agent(MAIN_AGENT)
        assert state is not None
        assert [m.role for m in state.messages] == ["system", "user"]
        assert state.model_call_seq == 0, "the call never completed, so it never counted"
    finally:
        await loaded.cancel("test teardown")


# ---------------------------------------------------------------------------
# R-C-9: an approval survives save -> load
# ---------------------------------------------------------------------------


async def test_a_pending_approval_survives_a_save_and_load(rigs) -> None:
    """R-C-9's first half: same request, restored, and the run waits for it again.

    `load()` returns PAUSED because R-C-11 says it never auto-starts. `resume()` is
    what puts the run back into WAITING_APPROVAL -- the thing it was waiting for did
    not go away while the process was dead.
    """

    handler = QueueApprovalHandler()
    rig = rigs(
        [calls(touch_call("two")), says("done")],
        mode=PermissionMode.MANUAL,
        handler=handler,
    )
    await rig.controller.start()
    request = await handler.next_request()
    await rig.wait_state(RunState.WAITING_APPROVAL)

    # A distinct path: `cancel()` below writes a terminal checkpoint over the
    # session file, and a process that dies does no such thing. This is the snapshot
    # a crash would have left.
    path = await rig.controller.save(
        rig.controller.session_path.parent / "snap.json", timeout=BOUND
    )
    saved = Session.load(path)
    assert [r.request_id for r in saved.pending_approvals] == [request.request_id]
    assert saved.resume_state is RunState.WAITING_APPROVAL
    await rig.controller.cancel("simulated process death")

    resumed_handler = QueueApprovalHandler()
    loaded = await Controller.load(
        path, build=lambda _: _never_runs, approval_handler=resumed_handler
    )
    try:
        assert loaded.state is RunState.PAUSED
        restored = loaded.pending_approvals
        assert [r.request_id for r in restored] == [request.request_id]
        assert restored[0].tool == request.tool
        assert restored[0].params == request.params
        assert restored[0].summary == request.summary
    finally:
        await loaded.cancel("test teardown")


async def test_a_restored_approval_resolves_and_reaches_the_re_issued_call(
    workspace: Path, tmp_path: Path
) -> None:
    """R-C-9's second half: "resolution proceeds normally".

    The call the request was blocking never ran -- it was at the gate -- so it comes
    back as `not started`, the model re-issues it, and the human's decision is
    applied to the re-issued call instead of prompting a second time. Without the
    carried decision, `resolve_approval` on a reloaded run would be a no-op with
    respect to the work.
    """

    handler = QueueApprovalHandler()
    session_dir = tmp_path / "approval"
    first = build_rig(
        [calls(touch_call("two")), says("done")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
        handler=handler,
        session_dir=session_dir,
    )
    try:
        await first.controller.start()
        request = await handler.next_request()
        await first.wait_state(RunState.WAITING_APPROVAL)
        path = await first.controller.save(session_dir / "snap.json", timeout=BOUND)
    finally:
        await first.controller.cancel("simulated process death")
        await first.aclose()

    # The resumed process: a fresh rig whose controller comes from the file, and a
    # handler that would fail the test if it were asked a second time.
    second_handler = QueueApprovalHandler()
    loaded = await Controller.load(
        path, build=lambda _: _never_runs, approval_handler=second_handler
    )
    try:
        restored = loaded.pending_approvals[0]
        assert await loaded.approve(restored.request_id, by="the user") is True
        assert loaded.pending_approvals == []
        assert loaded.gate.carried_decisions == 1

        # Bounded: without the carried decision this call blocks on a handler nobody
        # is going to answer, and an unbounded wait for a regression takes the whole
        # suite down with no output.
        async with asyncio.timeout(BOUND):
            decision = await loaded.gate.check(
                tool_name=request.tool,
                needs_approval=True,
                summary=request.summary,
                params=request.params,
                agent_id=MAIN_AGENT,
                call_id=CallId("a-new-call-id-the-model-just-minted"),
            )
        assert decision.approved
        assert decision.by == "the user"
        assert second_handler.seen == [], "the human was asked twice"
    finally:
        await loaded.cancel("test teardown")


async def test_a_carried_denial_is_honoured_too(workspace: Path, tmp_path: Path) -> None:
    """A denial is a decision as much as an approval, and must survive the same way."""

    handler = QueueApprovalHandler()
    session_dir = tmp_path / "denial"
    rig = build_rig(
        [calls(touch_call("two")), says("done")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
        handler=handler,
        session_dir=session_dir,
    )
    try:
        await rig.controller.start()
        request = await handler.next_request()
        path = await rig.controller.save(session_dir / "snap.json", timeout=BOUND)
    finally:
        await rig.controller.cancel("simulated process death")
        await rig.aclose()

    loaded = await Controller.load(
        path, build=lambda _: _never_runs, approval_handler=QueueApprovalHandler()
    )
    try:
        await loaded.deny(loaded.pending_approvals[0].request_id, "not that file")
        async with asyncio.timeout(BOUND):
            decision = await loaded.gate.check(
                tool_name=request.tool,
                needs_approval=True,
                summary=request.summary,
                params=request.params,
                agent_id=MAIN_AGENT,
                call_id=CallId("new-call"),
            )
        assert not decision.approved
        assert decision.reason == "not that file"
    finally:
        await loaded.cancel("test teardown")


async def test_the_blocked_call_comes_back_as_not_started_not_interrupted(
    workspace: Path, tmp_path: Path
) -> None:
    """A call at the gate had definitely not run. Saying otherwise would be false.

    `interrupted` means "effect unknown, never re-run this". A call the gate was
    still holding has no effect and *should* be re-issued once the human answers, so
    it gets the `cancelled`-kind "not started" result instead (R-C-9 vs R-C-13).
    """

    handler = QueueApprovalHandler()
    rig = build_rig(
        [calls(touch_call("two")), says("done")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
        handler=handler,
        session_dir=tmp_path / "blocked",
    )
    try:
        await rig.controller.start()
        await handler.next_request()
        path = await rig.controller.save(tmp_path / "blocked" / "snap.json", timeout=BOUND)
    finally:
        await rig.controller.cancel("simulated process death")
        await rig.aclose()

    loaded = await Controller.load(
        path, build=lambda _: _never_runs, approval_handler=QueueApprovalHandler()
    )
    try:
        state = loaded.agent(MAIN_AGENT)
        assert state is not None
        results = [m for m in state.messages if isinstance(m, ToolResultMessage)]
        assert len(results) == 1
        assert results[0].result.error is not None
        assert results[0].result.error.kind is ToolErrorKind.CANCELLED
        assert "not started" in results[0].result.error.message

        report = loaded.resume_report
        assert report is not None
        assert report.not_run_calls == [results[0].call_id]
        assert report.interrupted_calls == []
        assert rig.touch.started == [], "the gated tool never ran in the first place"
    finally:
        await loaded.cancel("test teardown")


# ---------------------------------------------------------------------------
# Spec delta 16: an in-flight delegate is resumed, not errored
# ---------------------------------------------------------------------------


async def test_an_inflight_delegate_is_resumed_rather_than_errored(
    workspace: Path, tmp_path: Path
) -> None:
    """Delta 16, end to end: the child keeps its transcript and its id.

    The asymmetry with R-C-13 is the point. A leaf tool call in flight at process
    death becomes `interrupted` and is never re-run, because its effect is unknown.
    A delegate has no effect of its own -- everything the child did is guarded by the
    child's own safe points -- so re-entering it costs nothing and loses nothing.
    """

    rig = build_rig(
        _delegate_turns(),
        workspace=workspace,
        spec=_PARENT_SPEC,
        specs={"explorer": _CHILD_SPEC},
        extra_tools=[DelegateTool()],
        session_dir=tmp_path / "delegate",
    )
    try:
        await rig.controller.start()
        await rig.wait_until(lambda: "main/0" in rig.controller.agents)
        await rig.wait_until(lambda: bool(rig.echo.started))

        crashed = rig.controller.session()
        kinds = [s.kind for s in crashed.inflight]
        assert "delegate" in kinds and "tool_call" in kinds
        delegate_step = next(s for s in crashed.inflight if s.kind == "delegate")
        assert delegate_step.child_agent_id == "main/0"

        path = tmp_path / "delegate" / "snap.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(crashed.dumps())
    finally:
        await rig.controller.cancel("simulated process death")
        await rig.aclose()

    loaded = await Controller.load(path, build=lambda _: _never_runs)
    try:
        report = loaded.resume_report
        assert report is not None
        assert len(report.resumed_delegates) == 1

        main = loaded.agent(MAIN_AGENT)
        assert main is not None
        assert len(main.resume_delegates) == 1
        entry = main.resume_delegates[0]
        assert entry.child_agent_id == "main/0"
        assert entry.spec_name == "explorer"
        assert entry.task == "look at the thing"
        # Deliberately unanswered: the resumed delegate answers it for real.
        assert main.open_call_ids == [entry.call_id]
        assert [m for m in main.messages if isinstance(m, ToolResultMessage)] == []

        # The child kept its own transcript, and its own interrupted call.
        kid = loaded.agent("main/0")
        assert kid is not None
        kid_results = [m for m in kid.messages if isinstance(m, ToolResultMessage)]
        assert len(kid_results) == 1
        assert kid_results[0].result.error is not None
        assert kid_results[0].result.error.kind is ToolErrorKind.INTERRUPTED
        assert_transcript_valid(kid.messages, agent_id="main/0")
    finally:
        await loaded.cancel("test teardown")


async def test_a_resumed_delegate_re_enters_the_recorded_child_id(
    workspace: Path, tmp_path: Path
) -> None:
    """The loop drains `resume_delegates` before its next model call.

    Allocating a fresh child id here would orphan the transcript the whole mechanism
    exists to reuse, and collide with the saved sibling on the next resume (spec 6.3).
    """

    turns = _delegate_turns()
    rig = build_rig(
        turns,
        workspace=workspace,
        spec=_PARENT_SPEC,
        specs={"explorer": _CHILD_SPEC},
        extra_tools=[DelegateTool()],
        session_dir=tmp_path / "d2",
    )
    try:
        await rig.controller.start()
        await rig.wait_until(lambda: bool(rig.echo.started))
        crashed = rig.controller.session()
        path = tmp_path / "d2" / "snap.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(crashed.dumps())
    finally:
        await rig.controller.cancel("simulated process death")
        await rig.aclose()

    resumed = build_rig(
        _resumed_delegate_turns(),
        workspace=workspace,
        spec=_PARENT_SPEC,
        specs={"explorer": _CHILD_SPEC},
        extra_tools=[DelegateTool()],
        session_dir=tmp_path / "d2-resumed",
    )
    body = resumed.controller._body
    assert body is not None
    loaded = await Controller.load(path, build=lambda _: body)
    try:
        await loaded.resume()
        await loaded.wait(timeout=BOUND)

        assert sorted(loaded.agents) == ["main", "main/0"], "no second child was allocated"
        main = loaded.agent(MAIN_AGENT)
        assert main is not None
        assert main.child_seq == 1, "the counter must not advance for a resumed child"
        assert main.resume_delegates == []

        results = [m for m in main.messages if isinstance(m, ToolResultMessage)]
        assert len(results) == 1
        assert results[0].name == "delegate"
        assert results[0].result.error is None
        assert results[0].result.meta.get("resumed") is True
        assert_transcript_valid(main.messages, agent_id="main")
        assert main.final_text == "the child said so"
    finally:
        if loaded.state not in TERMINAL_STATES:
            await loaded.cancel("test teardown")
        await resumed.aclose()


# ---------------------------------------------------------------------------
# R-W-6, in one interpreter
# ---------------------------------------------------------------------------


async def test_a_resumed_run_skips_completed_nodes_and_reruns_incomplete_ones(
    tmp_path: Path,
) -> None:
    """R-W-6 without a subprocess, so a failure names the mechanism rather than the OS.

    The body is the shape M5's `Runner` will have: ask `node_completed`, take the
    memo if there is one, run the body otherwise. `executed` is what proves the memo
    is doing the work -- a node whose body ran twice appears twice.
    """

    executed: list[str] = []
    session_dir = tmp_path / "nodes"

    def make_body(fail_at: str | None):
        async def body(control: Any) -> dict[str, Any]:
            values: dict[str, Any] = {}
            for name in ("one", "two", "three"):
                if control.node_completed(name):
                    values[name] = control.node_output(name)
                    continue
                await control.node_started(name)
                executed.append(name)
                if name == fail_at:
                    raise RuntimeError(f"the process died during {name}")
                values[name] = f"{name}-output"
                await control.node_finished(name, values[name])
            return values

        return body

    first = Controller(
        make_body("three"), session_dir=session_dir, permission_mode=PermissionMode.AUTO
    )
    with pytest.raises(RuntimeError):
        await first.run(timeout=BOUND)
    assert first.state is RunState.FAILED
    assert executed == ["one", "two", "three"]

    assert first.session_path is not None
    second = await Controller.load(first.session_path, build=lambda _: make_body(None))
    try:
        await second.resume()
        result = await second.wait(timeout=BOUND)
    finally:
        if second.state not in TERMINAL_STATES:
            await second.cancel("test teardown")

    assert result == {"one": "one-output", "two": "two-output", "three": "three-output"}
    assert executed == ["one", "two", "three", "three"], "a completed node was re-executed"
    assert second.node_output("one") == "one-output"
    assert second.node_completed("two")
