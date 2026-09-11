"""`HarnessApp` and the widgets, driven through `App.run_test()` (R-U-2, R-U-6).

The rig underneath is `tests/harness.py` -- a real `Controller`, a real `AgentLoop`,
a `FakeProvider` and two recording fake tools -- so every binding here is exercised
against the same control layer M2 and M3 tested, not a double.

Two habits carried over from M3, both learned the expensive way:

* **Every wait is bounded.** There is no `pytest-timeout` in this project, so an
  unbounded wait for a state that never arrives hangs the suite with no output. The
  helpers below all take a bound.
* **"Works in any state" is checked in every state.** M3's `save()` deadlocked in
  exactly one state, and reading the code did not find it. `ctrl+s` is driven from
  RUNNING, WAITING_APPROVAL, PAUSED and COMPLETED for the same reason.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from azalabscode.control import Session
from azalabscode.errors import SaveTimeout
from azalabscode.events import (
    ApprovalRequested,
    Checkpoint,
    Event,
    EventBus,
    MessageInjected,
    ModelDelta,
    RunStateChanged,
    RunWarning,
)
from azalabscode.ids import MAIN_AGENT
from azalabscode.permissions import (
    ApprovalRequest,
    ApprovalSummary,
    Decision,
    PermissionMode,
)
from azalabscode.runstate import RunState
from azalabscode.tui import (
    ApprovalModal,
    EventRouter,
    HarnessApp,
    RunStatusBar,
    StreamPane,
    Transcript,
)
from azalabscode.tui.routing import ANY, matches
from azalabscode.tui.widgets.approval_modal import DENIED_AT_MODAL
from tests.harness import BOUND, Rig, build_rig, calls, echo_call, says, touch_call

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def wait_until(predicate: Callable[[], bool], *, timeout: float = BOUND) -> None:
    """Poll `predicate` until it holds. Always bounded."""

    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def approval_request(**overrides: Any) -> ApprovalRequest:
    """A plausible pending request, for widgets tested without a run behind them."""

    fields: dict[str, Any] = {
        "run_id": "run_test",
        "agent_id": "main",
        "call_id": "call_1",
        "tool": "edit_file",
        "params": {"path": "src/app.py", "old": "a", "new": "b"},
        "summary": ApprovalSummary(
            title="edit_file src/app.py",
            detail="1 replacement",
            diff="--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b\n",
            danger=False,
        ),
    }
    fields.update(overrides)
    return ApprovalRequest(**fields)


class RigApp(HarnessApp):
    """The default layout, plus a handle on the transcript for assertions."""

    @property
    def transcript(self) -> Transcript:
        """The main agent's transcript."""

        return self.query_one(Transcript)


# ---------------------------------------------------------------------------
# Routing (R-U-2)
# ---------------------------------------------------------------------------


class Collector:
    """A minimal `EventConsumer`, for router tests with no widgets involved."""

    def __init__(self, agent: str = ANY, node: str = ANY) -> None:
        self.agent_filter = agent
        self.node_filter = node
        self.seen: list[Event] = []

    def handle_event(self, event: Event) -> None:
        self.seen.append(event)


def test_filters_select_by_agent_and_node() -> None:
    """Spec 8.1: widgets are registered for `(agent_id | "*", node_id | "*")`."""

    everything = Collector()
    just_main = Collector(agent="main")
    just_node = Collector(node="stage/1")

    run_level = RunStateChanged(old=RunState.CREATED, new=RunState.RUNNING)
    from_main = ModelDelta(call_id="c", text="x", agent_id="main")
    from_child = ModelDelta(call_id="c", text="x", agent_id="main/0", node_id="stage/1")

    assert matches(everything, run_level)
    assert matches(everything, from_child)
    assert matches(just_main, from_main)
    assert not matches(just_main, from_child)
    assert not matches(just_main, run_level), "a run-level event has no agent to match"
    assert matches(just_node, from_child)
    assert not matches(just_node, from_main)


def test_a_raising_widget_does_not_stop_the_stream() -> None:
    """One broken widget must not freeze every other widget in the app."""

    class Broken(Collector):
        def handle_event(self, event: Event) -> None:
            raise RuntimeError("widget bug")

    router = EventRouter()
    broken, healthy = Broken(), Collector()
    router.register(broken)
    router.register(healthy)

    for index in range(3):
        router.dispatch(ModelDelta(call_id="c", text=str(index)))

    assert len(healthy.seen) == 3
    assert len(router.errors) == 3
    assert isinstance(router.errors[0].cause, RuntimeError)


# ---------------------------------------------------------------------------
# StreamPane
# ---------------------------------------------------------------------------


async def test_deltas_are_coalesced_into_one_write(workspace: Path) -> None:
    """R-U-4's "coalesced in the UI, not in core", at the smallest scale that shows it."""

    pane = StreamPane(agent_id="main")

    class OnePane(HarnessApp):
        def compose(self) -> ComposeResult:
            yield pane

    rig = build_rig([says("unused")], workspace=workspace)
    app = OnePane(rig.controller)
    async with app.run_test():
        for index in range(20):
            pane.handle_event(ModelDelta(call_id="c", text=f"{index} ", agent_id="main"))
        assert pane.text == "", "a delta must not reach the screen before the timer"
        assert pane.stats.deltas == 20

        pane.flush_now()
        assert pane.stats.flushes == 1
        assert pane.text == "".join(f"{index} " for index in range(20))
    await rig.aclose()


async def test_an_agent_bound_pane_clears_between_calls(workspace: Path) -> None:
    """A pane following an agent shows the current answer, not every answer."""

    pane = StreamPane(agent_id="main")

    class OnePane(HarnessApp):
        def compose(self) -> ComposeResult:
            yield pane

    rig = build_rig([says("unused")], workspace=workspace)
    async with OnePane(rig.controller).run_test():
        from azalabscode.events import ModelCallStarted

        pane.handle_event(ModelCallStarted(call_id="c1", model="m", agent_id="main"))
        pane.handle_event(ModelDelta(call_id="c1", text="first", agent_id="main"))
        pane.flush_now()
        assert pane.text == "first"

        pane.handle_event(ModelCallStarted(call_id="c2", model="m", agent_id="main"))
        assert pane.text == ""
        assert pane.stats.deltas == 1, "statistics describe the run, not the screen"
    await rig.aclose()


# ---------------------------------------------------------------------------
# The app end to end (R-U-2)
# ---------------------------------------------------------------------------


async def test_a_run_reaches_the_transcript(workspace: Path) -> None:
    """R-U-2: the app subscribes on mount and routes to the mounted widgets."""

    rig = build_rig(
        [calls(echo_call("one")), says("all done")],
        workspace=workspace,
        mode=PermissionMode.AUTO,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await rig.controller.start()
        await rig.controller.wait(timeout=BOUND)
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)
        await pilot.pause()

        transcript = app.transcript
        for pane in transcript.panes.values():
            pane.flush_now()
        assert "all done" in transcript.text()

        blocks = transcript.tool_blocks
        assert len(blocks) == 1
        block = next(iter(blocks.values()))
        assert block.tool == "echo"
        assert block.status == "ok"
        assert "echo: one" in block.detail
    await rig.aclose()


async def test_the_status_bar_reads_the_controller_live(workspace: Path) -> None:
    """Spec 8.1's always-mounted bar: state, mode, agents, tokens, checkpoints."""

    rig = build_rig([says("hi")], workspace=workspace)
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        bar = app.query_one(RunStatusBar)
        assert "CREATED" in _bar_text(bar)

        await rig.controller.start()
        await rig.controller.wait(timeout=BOUND)
        await pilot.pause()

        text = _bar_text(bar)
        assert "COMPLETED" in text
        assert "auto" in text
        assert "agents 0/1" in text
        assert bar.checkpoints > 0, "Checkpoint events should have reached the bar"
    await rig.aclose()


def _bar_text(bar: RunStatusBar) -> str:
    return "  ".join(body for body, _ in bar.segments())


# ---------------------------------------------------------------------------
# Approvals (R-U-6)
# ---------------------------------------------------------------------------


async def _run_to_approval(rig: Rig, app: HarnessApp, pilot: Any) -> ApprovalModal:
    """Start the run and wait for the modal the base app mounts."""

    await rig.controller.start()
    await wait_until(lambda: isinstance(app.screen, ApprovalModal))
    await pilot.pause()
    modal = app.screen
    assert isinstance(modal, ApprovalModal)
    return modal


async def test_the_base_app_mounts_the_modal_with_no_ui_code(workspace: Path) -> None:
    """R-U-6. Nothing in this test writes a widget; the base app does it all."""

    rig = build_rig(
        [calls(touch_call("x")), says("done")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        modal = await _run_to_approval(rig, app, pilot)

        assert modal.request.tool == "touch"
        assert rig.controller.state is RunState.WAITING_APPROVAL
        assert rig.controller.approval_handler is app.approval_handler

        await pilot.press("y")
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)
        assert rig.touch.finished == ["x"]
    await rig.aclose()


async def test_denying_at_the_modal_reaches_the_model(workspace: Path) -> None:
    """`n` denies with a reason, and the reason is what the tool error carries."""

    rig = build_rig(
        [calls(touch_call("x")), says("understood")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await _run_to_approval(rig, app, pilot)
        await pilot.press("n")
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)

        assert rig.touch.started == [], "a denied call must never run"
        results = [m for m in rig.transcript if getattr(m, "role", "") == "tool"]
        assert results and DENIED_AT_MODAL in results[0].text
    await rig.aclose()


async def test_escape_denies_rather_than_leaving_the_gate_parked(workspace: Path) -> None:
    """A modal that can be dismissed without answering is a hang, not a cancel."""

    rig = build_rig(
        [calls(touch_call("x")), says("ok")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await _run_to_approval(rig, app, pilot)
        await pilot.press("escape")
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)
        assert rig.touch.started == []
    await rig.aclose()


async def test_switching_to_auto_takes_the_modal_down(workspace: Path) -> None:
    """`ctrl+t` resolves pending requests (R-C-5); the modal must not survive it.

    There is no event for a *withdrawal*, so this is the path that only works
    because `TUIApprovalHandler.cancel` calls back into the app.
    """

    rig = build_rig(
        [calls(touch_call("x")), says("ok")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await _run_to_approval(rig, app, pilot)
        await pilot.press("ctrl+t")
        await wait_until(lambda: not isinstance(app.screen, ApprovalModal))
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)

        assert rig.controller.permission_mode is PermissionMode.AUTO
        assert rig.touch.finished == ["x"], "an auto-approved call should have run"
    await rig.aclose()


async def test_two_pending_requests_are_shown_one_at_a_time(workspace: Path) -> None:
    """Concurrent agents can each be waiting. The queue drains as each is answered."""

    rig = build_rig([says("unused")], workspace=workspace)
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        first = approval_request(call_id="call_1")
        second = approval_request(call_id="call_2")
        await app.approval_handler.request(first)
        await app.approval_handler.request(second)
        app.on_approval_requested(first)
        app.on_approval_requested(second)
        await pilot.pause()

        assert isinstance(app.screen, ApprovalModal)
        assert app.screen.request.call_id == "call_1"
        await pilot.press("y")
        await wait_until(
            lambda: isinstance(app.screen, ApprovalModal) and app.screen.request.call_id == "call_2"
        )
    await rig.aclose()


async def test_a_subclass_may_replace_the_modal(workspace: Path) -> None:
    """R-U-6's escape hatch: overriding `on_approval_requested` replaces the render
    without replacing the handler, and the gate is still parked either way."""

    seen: list[ApprovalRequest] = []

    class Custom(HarnessApp):
        def compose(self) -> ComposeResult:
            yield Static("custom")

        def on_approval_requested(self, request: ApprovalRequest) -> None:
            seen.append(request)

    rig = build_rig(
        [calls(touch_call("x")), says("ok")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
    )
    app = Custom(rig.controller)
    async with app.run_test() as pilot:
        await rig.controller.start()
        await wait_until(lambda: bool(seen))
        await pilot.pause()

        assert not isinstance(app.screen, ApprovalModal)
        assert rig.controller.state is RunState.WAITING_APPROVAL

        app.resolve_approval(seen[0].request_id, Decision.approve(by="custom"))
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)
    await rig.aclose()


# ---------------------------------------------------------------------------
# Bindings (spec 8.1)
# ---------------------------------------------------------------------------


async def test_ctrl_p_pauses_and_resumes(workspace: Path) -> None:
    """R-C-3 and R-C-1 through the keyboard."""

    rig = build_rig(
        [calls(echo_call("slow", sleep=0.6)), says("done")],
        workspace=workspace,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await rig.controller.start()
        # Wait for the tool, not just for RUNNING: a run that finishes before the
        # keystroke lands proves nothing about the binding.
        await wait_until(lambda: bool(rig.echo.started))

        await pilot.press("ctrl+p")
        await wait_until(lambda: rig.controller.state in (RunState.PAUSING, RunState.PAUSED))
        await rig.wait_state(RunState.PAUSED)

        await pilot.press("ctrl+p")
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)
    await rig.aclose()


async def test_ctrl_l_toggles_the_event_log(workspace: Path) -> None:
    """The `ctrl+l` pane exists and carries every event except the deltas."""

    rig = build_rig([calls(echo_call("one")), says("done")], workspace=workspace)
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        log = app.event_log
        assert log is not None
        assert log.display is False

        await rig.controller.run(timeout=BOUND)
        await pilot.press("ctrl+l")
        await pilot.pause()

        assert log.display is True
        assert log.suppressed > 0, "deltas must be suppressed, not logged"
        assert log.written > 0
    await rig.aclose()


async def test_ctrl_c_twice_cancels_the_run(workspace: Path) -> None:
    """One press warns, two cancel. Quitting the UI is not cancelling the run."""

    rig = build_rig(
        [calls(echo_call("slow", sleep=0.5)), says("never")],
        workspace=workspace,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await rig.controller.start()
        await rig.wait_state(RunState.RUNNING)

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert rig.controller.state is not RunState.CANCELLED
        assert "ctrl+c again" in _bar_text(app.query_one(RunStatusBar))

        await pilot.press("ctrl+c")
        await wait_until(lambda: rig.controller.state is RunState.CANCELLED)
    await rig.aclose()


async def test_escape_interrupts_the_current_step(workspace: Path) -> None:
    """R-C-4 through the keyboard: the target agent's step is cancelled and the run
    continues rather than dropping to a pause (spec C-4)."""

    rig = build_rig(
        [calls(echo_call("slow", sleep=0.6)), says("after the interrupt")],
        workspace=workspace,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await rig.controller.start()
        await wait_until(lambda: bool(rig.echo.started))

        await pilot.press("escape")
        await wait_until(lambda: bool(rig.echo.cancelled))
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)
        assert rig.result is not None and "after the interrupt" in rig.result.final_text
    await rig.aclose()


async def test_submit_injection_appends_a_user_message(workspace: Path) -> None:
    """The second half of spec 8.1's `escape` flow: the optional message.

    Injected separately from the interrupt because the interrupt is what makes room
    for it: by the time `escape` has cancelled the *last* step of a run, the run is
    finished and there is no agent left to inject into. The user types while the
    agent is still working, which is what this drives.
    """

    rig = build_rig(
        [
            calls(echo_call("first", sleep=0.6)),
            calls(echo_call("second", sleep=0.6)),
            says("done"),
        ],
        workspace=workspace,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await rig.controller.start()
        await wait_until(lambda: bool(rig.echo.started))
        await pilot.press("escape")
        await wait_until(lambda: rig.echo.started[-1] == "second", timeout=BOUND * 2)

        app.submit_injection("try something else")
        await wait_until(lambda: bool(rig.of_type(MessageInjected)))

        state = rig.controller.agent(MAIN_AGENT)
        assert state is not None
        assert [m.text for m in state.pending_injections] == ["try something else"]

        app.submit_injection("   ")  # empty = interrupt without a message
        await asyncio.sleep(0.05)
        assert len(rig.of_type(MessageInjected)) == 1

        await rig.controller.cancel()
    await rig.aclose()


async def test_a_targetless_interrupt_in_a_fanout_is_shown_not_swallowed(
    workspace: Path,
) -> None:
    """Spec C-4. A fan-out has no `main`, so `escape` cancels nothing -- and the
    user has to be told, which is what `interrupt_target()` exists to fix."""

    rig = build_rig([says("unused")], workspace=workspace)
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        assert app.interrupt_target() is None
        await pilot.press("escape")
        await wait_until(lambda: bool(rig.of_type(RunWarning)))
        await pilot.pause()

        warning = rig.of_type(RunWarning)[0]
        assert warning.code == "interrupt_no_target"
        assert warning.message in _bar_text(app.query_one(RunStatusBar))
    await rig.aclose()


async def test_the_interrupt_target_hook_redirects_escape(workspace: Path) -> None:
    """A fusion app names the selected pane's agent; the base app must honour it."""

    rig = build_rig(
        [calls(echo_call("slow", sleep=0.5)), says("done")],
        workspace=workspace,
    )

    class Targeted(RigApp):
        def interrupt_target(self) -> str | None:
            return str(MAIN_AGENT)

    app = Targeted(rig.controller)
    async with app.run_test() as pilot:
        await rig.controller.start()
        await wait_until(lambda: bool(rig.echo.started))
        await pilot.press("escape")
        await wait_until(lambda: bool(rig.echo.cancelled))
        assert not rig.of_type(RunWarning)
    await rig.aclose()


# ---------------------------------------------------------------------------
# Save and open (R-C-10, R-C-11, spec C-12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [RunState.CREATED, RunState.RUNNING, RunState.WAITING_APPROVAL, RunState.PAUSED],
)
async def test_ctrl_s_saves_from_every_state(
    workspace: Path, tmp_path: Path, state: RunState
) -> None:
    """M3's lesson, applied to the binding: `save()` deadlocked in exactly one
    state and reading the code did not find it. Drive the key from all of them."""

    session_dir = tmp_path / f"session-{state.value}"
    session_dir.mkdir()
    rig = build_rig(
        [calls(touch_call("x", sleep=0.05)), says("done")],
        workspace=workspace,
        mode=PermissionMode.MANUAL if state is RunState.WAITING_APPROVAL else PermissionMode.AUTO,
        session_dir=session_dir,
        autosave=False,
    )
    app = RigApp(rig.controller, session_dir=session_dir)

    async with app.run_test() as pilot:
        if state is not RunState.CREATED:
            await rig.controller.start()
        if state is RunState.WAITING_APPROVAL:
            await rig.wait_state(RunState.WAITING_APPROVAL)
        elif state is RunState.PAUSED:
            await rig.controller.pause()
            await rig.wait_state(RunState.PAUSED)
        elif state is RunState.RUNNING:
            await rig.wait_state(RunState.RUNNING)

        target = session_dir / "session.json"
        bar = app.query_one(RunStatusBar)
        await pilot.press("ctrl+s")
        # Wait on the notice, not on the file: the file can appear from the safe
        # point the save is waiting for, a moment before `save()` itself returns.
        await wait_until(lambda: "saved" in _bar_text(bar), timeout=BOUND * 2)

        assert target.exists()
        assert Session.load(target).run_state is not None
    await rig.aclose()


async def test_a_save_timeout_renders_what_is_blocking_it(workspace: Path, tmp_path: Path) -> None:
    """Spec C-12 and delta 17: `SaveTimeout.blocking` is what the bar shows.

    The message is the difference between "save failed" and "save is waiting for
    main's tool call, which has been running for five minutes -- interrupt it?".
    """

    rig = build_rig(
        [calls(echo_call("slow", sleep=1.0)), says("done")],
        workspace=workspace,
        session_dir=tmp_path,
        autosave=False,
    )
    app = RigApp(rig.controller, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await rig.controller.start()
        await wait_until(lambda: bool(rig.echo.started))

        with pytest.raises(SaveTimeout) as caught:
            await rig.controller.save(tmp_path / "explicit.json", timeout=0.05)
        assert "main:" in caught.value.blocking

        app._notify(f"save is waiting for {caught.value.blocking}")
        await pilot.pause()
        assert "save is waiting for main:" in _bar_text(app.query_one(RunStatusBar))

        await rig.controller.cancel()
    await rig.aclose()


async def test_the_bar_names_the_blocking_step_while_pausing(workspace: Path) -> None:
    """C-12's other half: pressing pause during a slow step must not look like a
    no-op, so PAUSING is displayed with `blocking_description()` beside it."""

    rig = build_rig(
        [calls(echo_call("slow", sleep=0.4)), says("done")],
        workspace=workspace,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await rig.controller.start()
        await wait_until(lambda: bool(rig.echo.started))
        await rig.controller.pause()
        await pilot.pause()

        if rig.controller.state is RunState.PAUSING:
            bar = app.query_one(RunStatusBar)
            text = _bar_text(bar)
            assert "PAUSING" in text
            assert "main:" in text
        await rig.controller.resume()
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)
    await rig.aclose()


async def test_ctrl_o_attaches_to_the_loaded_controller(workspace: Path, tmp_path: Path) -> None:
    """R-C-11 through the keyboard, and trap 2 from the M3 handoff: a loaded run
    has agents and no events for them, so the app must seed from the controller."""

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    rig = build_rig(
        [calls(touch_call("x")), says("done")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
        session_dir=session_dir,
        # Autosave off: a terminal checkpoint would overwrite the snapshot taken
        # below with a COMPLETED run, and the test would then load and assert
        # against the wrong document while still passing most of its assertions.
        autosave=False,
    )
    app = RigApp(rig.controller, session_dir=session_dir)

    async with app.run_test() as pilot:
        await _run_to_approval(rig, app, pilot)
        saved = await rig.controller.save(session_dir / "session.json")
        assert saved.exists()

        original = rig.controller
        await pilot.press("escape")  # answer the modal so the run can be abandoned
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)

        await pilot.press("ctrl+o")
        await wait_until(lambda: app.controller is not original, timeout=BOUND * 2)
        await pilot.pause()

        loaded = app.controller
        assert loaded.state is RunState.PAUSED, "load never auto-starts (R-C-11)"
        assert loaded.agents, "a loaded run has agents before any event arrives"
        assert loaded.pending_approvals, "the pending request survived the round trip"
        assert isinstance(app.screen, ApprovalModal), "and the modal was seeded from it"
        assert app.query_one(RunStatusBar).controller is loaded
    await rig.aclose()


async def test_attaching_a_second_controller_stops_following_the_first(
    workspace: Path,
) -> None:
    """`attach()` is one call, not a teardown sequence a subclass must remember."""

    first = build_rig([says("one")], workspace=workspace)
    second = build_rig([says("two")], workspace=workspace)
    app = RigApp(first.controller)

    async with app.run_test() as pilot:
        await app.attach(second.controller)
        await pilot.pause()

        before = app.events_seen
        await first.controller.emit(Checkpoint(kind="custom", agent_id=None, node_id=None))
        await asyncio.sleep(0.05)
        assert app.events_seen == before, "the old bus is no longer followed"

        await second.controller.emit(Checkpoint(kind="custom"))
        await wait_until(lambda: app.events_seen > before)
    await first.aclose()
    await second.aclose()


# ---------------------------------------------------------------------------
# The modal's own rendering
# ---------------------------------------------------------------------------


async def test_the_modal_shows_everything_spec_8_1_lists(workspace: Path) -> None:
    """Tool name, human summary, params, diff, agent id."""

    rig = build_rig([says("unused")], workspace=workspace)
    app = RigApp(rig.controller)
    request = approval_request()

    async with app.run_test() as pilot:
        app.on_approval_requested(request)
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ApprovalModal)

        rendered = _screen_text(modal)
        assert "edit_file src/app.py" in rendered
        assert "1 replacement" in rendered
        assert "agent main" in rendered
        assert modal.query_one("#approval-diff")
        assert modal.query_one("#approval-params")
    await rig.aclose()


def _screen_text(screen: Any) -> str:
    """Every `Label`/`Static` string on a screen, joined. Cheaper than a snapshot
    and it fails with a readable message rather than a diff of escape codes."""

    from textual.widgets import Label

    parts: list[str] = []
    for widget in screen.query(Label):
        parts.append(str(widget.render()))
    for widget in screen.query(Static):
        parts.append(str(widget.render()))
    return "\n".join(parts)


async def test_a_dangerous_request_is_marked(workspace: Path) -> None:
    """`ApprovalSummary.danger` is the irreversible subset; a UI must style it."""

    rig = build_rig([says("unused")], workspace=workspace)
    app = RigApp(rig.controller)
    request = approval_request(
        tool="shell",
        summary=ApprovalSummary(title="shell rm -rf build", danger=True),
        params={"command": "rm -rf build"},
    )
    async with app.run_test() as pilot:
        app.on_approval_requested(request)
        await pilot.pause()
        assert "irreversible" in _screen_text(app.screen)
    await rig.aclose()


# ---------------------------------------------------------------------------
# Transcript detail
# ---------------------------------------------------------------------------


async def test_an_injected_message_is_highlighted(workspace: Path) -> None:
    """R-C-4's injected message is the one line a user will look for."""

    rig = build_rig([says("unused")], workspace=workspace)
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        transcript = app.transcript
        transcript.handle_event(
            MessageInjected(message_id="m1", text="stop and read this", agent_id="main")
        )
        await pilot.pause()
        assert any(block.has_class("-injected") for block in transcript.blocks)
    await rig.aclose()


async def test_the_transcript_caps_its_blocks(workspace: Path) -> None:
    """A transcript is a view, not a store; the session holds the history."""

    rig = build_rig([says("unused")], workspace=workspace)
    transcript = Transcript(agent_id="main", max_blocks=5)

    class OneTranscript(HarnessApp):
        def compose(self) -> ComposeResult:
            yield transcript

    app = OneTranscript(rig.controller)
    async with app.run_test() as pilot:
        for index in range(12):
            transcript.handle_event(
                MessageInjected(message_id=f"m{index}", text=str(index), agent_id="main")
            )
        await pilot.pause()
        assert len(transcript.blocks) == 5
    await rig.aclose()


async def test_an_approval_updates_the_matching_tool_block(workspace: Path) -> None:
    """`ApprovalResolved` carries a request id, not a call id; the transcript has
    to keep the mapping or the block never leaves "awaiting approval"."""

    rig = build_rig(
        [calls(touch_call("x")), says("done")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await _run_to_approval(rig, app, pilot)
        transcript = app.transcript
        block = next(iter(transcript.tool_blocks.values()))
        assert block.status == "awaiting approval"

        await pilot.press("y")
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)
        await pilot.pause()
        assert block.status == "ok"
    await rig.aclose()


# ---------------------------------------------------------------------------
# The seam itself
# ---------------------------------------------------------------------------


async def test_the_handler_defers_every_decision(workspace: Path) -> None:
    """R-U-6: `request()` returns `None` so the modal outlives the call."""

    rig = build_rig([says("unused")], workspace=workspace)
    app = RigApp(rig.controller)
    async with app.run_test():
        decision = await app.approval_handler.request(approval_request())
        assert decision is None
        assert app.approval_handler.pending
    await rig.aclose()


async def test_the_handler_cannot_be_swapped_under_a_pending_request(
    workspace: Path,
) -> None:
    """Replacing it would leave the gate parked on a future nobody can resolve."""

    rig = build_rig(
        [calls(touch_call("x")), says("done")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
    )
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        await _run_to_approval(rig, app, pilot)
        with pytest.raises(ValueError, match="while a request is pending"):
            rig.controller.set_approval_handler(None)
        await pilot.press("n")
        await wait_until(lambda: rig.controller.state is RunState.COMPLETED)
    await rig.aclose()


async def test_the_app_does_not_install_a_handler_over_an_existing_one(
    workspace: Path,
) -> None:
    """A headless-configured run keeps its own handler when a UI attaches."""

    from azalabscode.control import QueueApprovalHandler

    handler = QueueApprovalHandler()
    rig = build_rig(
        [says("unused")],
        workspace=workspace,
        mode=PermissionMode.MANUAL,
        handler=handler,
    )
    app = RigApp(rig.controller)
    async with app.run_test():
        assert rig.controller.approval_handler is handler
    await rig.aclose()


async def test_the_bus_seq_is_not_assumed_to_start_at_one(workspace: Path) -> None:
    """M3 handoff trap 3: a reloaded run's events continue from the session."""

    rig = build_rig([says("unused")], workspace=workspace)
    bus = EventBus(run_id="run_test")
    bus.seed_seq(500)
    app = RigApp(rig.controller)
    async with app.run_test() as pilot:
        log = app.event_log
        assert log is not None
        log.handle_event(ApprovalRequested(request=approval_request(), seq=501))
        await pilot.pause()
        assert log.written == 1
    await rig.aclose()
