"""The eight control scenarios from spec 11, in memory.

Spec 11 lists them: pause-during-stream, pause-during-tool, interrupt-during-stream,
interrupt-during-tool, interrupt-with-injection, mode switch with a pending
approval, save/load in every state, and the subprocess kill test. The last two need
disk and a second interpreter, so they live in `tests/test_save_load.py` and
`tests/test_kill.py`; the comment where they used to be says which test is which.

Every wait is bounded, and every scenario ends by letting the run finish, so a
scenario that leaves the controller wedged fails loudly rather than leaking a task
into the next test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from azalabscode.control import (
    Controller,
    DenyAllHandler,
    QueueApprovalHandler,
)
from azalabscode.errors import ConfigurationError
from azalabscode.events import (
    ApprovalRequested,
    ApprovalResolved,
    MessageInjected,
    ModelCallCancelled,
    ModelCallStarted,
    PermissionModeChanged,
    RunWarning,
    ToolCallStarted,
)
from azalabscode.ids import MAIN_AGENT
from azalabscode.messages import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
    assert_transcript_valid,
)
from azalabscode.permissions import Decision, PermissionMode
from azalabscode.runstate import TERMINAL_STATES, AgentPhase, RunState
from azalabscode.toolio import ToolErrorKind
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
async def rig_factory(workspace: Path):
    """Builds rigs and guarantees they are torn down, even on a failure."""

    made: list[Rig] = []

    def make(turns, **kwargs) -> Rig:
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
# 1. pause during a model stream
# ---------------------------------------------------------------------------


async def test_scenario_1_pause_during_a_model_stream(rig_factory) -> None:
    """R-C-3: the in-flight call runs to completion, then the run reaches PAUSED.

    Pause is a request, not a cancellation. The distinction is the whole of C-12:
    the user sees PAUSING for as long as the model is still talking.
    """

    rig = rig_factory([says("thinking about it", chunk_delay_s=0.02)])
    await rig.controller.start()
    await rig.wait_type(ModelCallStarted)

    await rig.controller.pause()
    assert rig.controller.state is RunState.PAUSING

    await rig.wait_state(RunState.PAUSED)
    assert rig.controller.nonquiescent == 0
    assert rig.controller.phase_of(MAIN_AGENT) is AgentPhase.PARKED
    # The model call was not cancelled: its output is on the transcript.
    assert not rig.of_type(ModelCallCancelled)

    await rig.controller.resume()
    result = await rig.finish()
    assert result.final_text == "thinking about it"
    assert rig.controller.state is RunState.COMPLETED


async def test_a_hard_pause_cancels_the_stream_instead(rig_factory) -> None:
    """`pause(hard=True)` is interrupt semantics without the injection (R-C-3)."""

    rig = rig_factory([says("a very long answer indeed", chunk_delay_s=0.05), says("done")])
    await rig.controller.start()
    await rig.wait_type(ModelCallStarted)
    await asyncio.sleep(0.06)

    await rig.controller.pause(hard=True)
    await rig.wait_state(RunState.PAUSED)

    cancelled = rig.of_type(ModelCallCancelled)
    assert cancelled and cancelled[0].reason == "pause_hard"

    await rig.controller.resume()
    result = await rig.finish()
    assert result.final_text == "done"


# ---------------------------------------------------------------------------
# 2. pause during a tool call
# ---------------------------------------------------------------------------


async def test_scenario_2_pause_during_a_tool_call(rig_factory) -> None:
    """The tool finishes. A pause never leaves a half-run effect behind."""

    rig = rig_factory([calls(echo_call("slow", sleep=0.2)), says("finished")])
    await rig.controller.start()
    await rig.wait_type(ToolCallStarted)

    await rig.controller.pause()
    assert rig.controller.state is RunState.PAUSING
    assert rig.controller.phase_of(MAIN_AGENT) is AgentPhase.BLOCKED_IO

    await rig.wait_state(RunState.PAUSED)
    assert rig.echo.finished == ["slow"], "pause must not cancel a running tool"

    await rig.controller.resume()
    await rig.finish()
    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert len(results) == 1
    assert results[0].result.ok


async def test_a_hard_pause_cancels_the_tool_and_the_model_sees_it(rig_factory) -> None:
    """The other half of R-C-3, and the tool's `on_cancel` really runs."""

    rig = rig_factory([calls(echo_call("slow", sleep=5.0)), says("gave up")])
    await rig.controller.start()
    await rig.wait_type(ToolCallStarted)

    await rig.controller.pause(hard=True)
    await rig.wait_state(RunState.PAUSED)

    assert rig.echo.finished == []
    assert rig.echo.cancelled == ["slow"], "the dispatcher's detached cleanup must run"

    await rig.controller.resume()
    await rig.finish()
    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert results[0].result.error is not None
    assert results[0].result.error.kind is ToolErrorKind.CANCELLED


# ---------------------------------------------------------------------------
# 3. interrupt during a model stream
# ---------------------------------------------------------------------------


async def test_scenario_3_interrupt_during_a_model_stream(rig_factory) -> None:
    """R-C-4 and delta 15: the partial is kept, marked, and the run continues.

    The run does not change coarse state: interrupting a RUNNING run leaves it
    RUNNING (spec C-4). The agent resumes at its next model call.
    """

    rig = rig_factory(
        [
            says("I am going to ramble for quite a while", chunk_delay_s=0.05),
            says("second answer"),
        ]
    )
    await rig.controller.start()
    await rig.wait_type(ModelCallStarted)
    await asyncio.sleep(0.08)

    outcome = await rig.controller.interrupt()
    assert outcome.hit

    result = await rig.finish()
    assert result.final_text == "second answer"
    assert rig.controller.state is RunState.COMPLETED

    partials = [m for m in rig.transcript if isinstance(m, AssistantMessage) and m.cancelled]
    assert len(partials) == 1
    assert partials[0].text
    assert partials[0].tool_calls == [], "delta 15: a cancelled call keeps no tool calls"


async def test_a_cancelled_response_of_only_tool_calls_is_discarded(rig_factory) -> None:
    """Delta 15's second half: with the calls dropped there is nothing left to keep."""

    rig = rig_factory(
        [
            calls(echo_call("a"), echo_call("b"), chunk_delay_s=0.05),
            says("moving on"),
        ]
    )
    await rig.controller.start()
    await rig.wait_type(ModelCallStarted)
    await asyncio.sleep(0.06)
    await rig.controller.interrupt()

    await rig.finish()
    assert not [m for m in rig.transcript if isinstance(m, AssistantMessage) and m.cancelled]
    assert rig.echo.started == [], "no dispatched call, so no tool ran"
    cancelled = rig.of_type(ModelCallCancelled)
    assert cancelled and cancelled[0].kept_partial is False


async def test_keep_cancelled_output_false_discards_the_partial(rig_factory, workspace) -> None:
    """`AgentSpec.keep_cancelled_output=False` leaves only the event behind."""

    from azalabscode.workflows import AgentSpec

    rig = rig_factory(
        [says("rambling on and on", chunk_delay_s=0.05), says("done")],
        spec=AgentSpec(name="main", model="fake/model", keep_cancelled_output=False),
    )
    await rig.controller.start()
    await rig.wait_type(ModelCallStarted)
    await asyncio.sleep(0.06)
    await rig.controller.interrupt()
    await rig.finish()

    assert not [m for m in rig.transcript if isinstance(m, AssistantMessage) and m.cancelled]
    assert rig.of_type(ModelCallCancelled)


# ---------------------------------------------------------------------------
# 4. interrupt during a tool batch
# ---------------------------------------------------------------------------


async def test_scenario_4_interrupt_during_a_tool_batch(rig_factory) -> None:
    """The plan's worked example: r1 done, r2 running, r3 never started.

    r1 keeps its real result -- which is only possible because the dispatcher hands
    each result over as it lands, rather than at the end of a batch that is about to
    be cancelled. r2 and r3 come back as `cancelled`, and the transcript is valid.
    """

    rig = rig_factory(
        [
            calls(
                echo_call("fast"),
                echo_call("slow", sleep=5.0),
                touch_call("never"),
            ),
            says("after the interrupt"),
        ],
        max_parallel=10,
    )
    await rig.controller.start()
    await rig.wait_until(lambda: rig.echo.finished == ["fast"])

    await rig.controller.interrupt()
    result = await rig.finish()

    assert result.final_text == "after the interrupt"
    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    requested = [
        c.call_id for m in rig.transcript if isinstance(m, AssistantMessage) for c in m.tool_calls
    ]
    assert [m.call_id for m in results] == requested
    assert results[0].result.ok and results[0].result.text == "echo: fast"
    for message in results[1:]:
        assert message.result.error is not None
        assert message.result.error.kind is ToolErrorKind.CANCELLED
    assert rig.touch.started == [], "the third call must never have run"
    assert_transcript_valid(rig.transcript, agent_id="main")


# ---------------------------------------------------------------------------
# 5. interrupt with injection
# ---------------------------------------------------------------------------


async def test_scenario_5_interrupt_with_an_injected_message(rig_factory) -> None:
    """R-C-4: the message lands *after* the cancelled response and before the next call.

    Spec decision 4 depends on that order: the model is meant to read its own
    cut-off output followed by the instruction that supersedes it.
    """

    rig = rig_factory(
        [says("I will start by reading every file", chunk_delay_s=0.05), says("ok, doing that")]
    )
    await rig.controller.start()
    await rig.wait_type(ModelCallStarted)
    await asyncio.sleep(0.08)

    outcome = await rig.controller.interrupt(message="stop, just answer the question")
    assert outcome.injected_message_id is not None

    await rig.finish()

    roles = [(type(m).__name__, getattr(m, "cancelled", None)) for m in rig.transcript]
    injected = [m for m in rig.transcript if isinstance(m, UserMessage) and m.injected]
    assert len(injected) == 1, roles
    assert injected[0].text == "stop, just answer the question"

    index = rig.transcript.index(injected[0])
    before = rig.transcript[index - 1]
    assert isinstance(before, AssistantMessage) and before.cancelled

    events = rig.of_type(MessageInjected)
    assert events and events[0].text == "stop, just answer the question"
    # The second request saw it.
    assert any(
        isinstance(m, UserMessage) and m.injected for m in rig.provider.requests[-1].messages
    )


async def test_interrupting_a_paused_run_injects_and_stays_paused(rig_factory) -> None:
    """Spec C-4: interrupt does not change the run's coarse state."""

    rig = rig_factory([says("first", chunk_delay_s=0.02), says("second")])
    await rig.controller.start()
    await rig.controller.pause()
    await rig.wait_state(RunState.PAUSED)

    await rig.controller.interrupt(message="while you were out")
    assert rig.controller.state is RunState.PAUSED

    await rig.controller.resume()
    await rig.finish()
    assert [m.text for m in rig.transcript if isinstance(m, UserMessage) and m.injected] == [
        "while you were out"
    ]


async def test_interrupting_an_unknown_agent_warns_and_does_nothing(rig_factory) -> None:
    """Spec C-4's fan-out case, where a targetless interrupt has nothing to cancel."""

    rig = rig_factory([says("hello")])
    await rig.controller.start()
    await rig.wait_type(ModelCallStarted)

    outcome = await rig.controller.interrupt(target="fusion/3")
    assert outcome.cancelled_steps == 0
    assert outcome.warning is not None

    warning = await rig.wait_type(RunWarning)
    assert warning.code == "interrupt_no_target"
    await rig.finish()


async def test_a_second_interrupt_is_idempotent(rig_factory) -> None:
    """R-C-1: the first reason wins and a repeat interrupt cancels nothing new."""

    rig = rig_factory([says("rambling", chunk_delay_s=0.1), says("done")])
    await rig.controller.start()
    await rig.wait_type(ModelCallStarted)

    first = await rig.controller.interrupt()
    second = await rig.controller.interrupt()
    assert first.cancelled_steps == 1
    assert second.cancelled_steps == 0
    await rig.finish()


# ---------------------------------------------------------------------------
# 6. mode switch with a pending approval
# ---------------------------------------------------------------------------


async def test_scenario_6_switching_to_auto_releases_a_pending_approval(rig_factory) -> None:
    """R-C-5: the switch expresses intent to stop being asked, so pending calls run."""

    handler = QueueApprovalHandler()
    rig = rig_factory(
        [calls(touch_call("dangerous")), says("done")],
        mode=PermissionMode.MANUAL,
        handler=handler,
    )
    await rig.controller.start()

    request = await handler.next_request()
    await rig.wait_state(RunState.WAITING_APPROVAL)
    assert rig.controller.phase_of(MAIN_AGENT) is AgentPhase.WAITING_APPROVAL
    assert len(rig.controller.pending_approvals) == 1

    resolved = await rig.controller.set_permission_mode(PermissionMode.AUTO)
    assert resolved == 1

    await rig.finish()
    assert rig.touch.finished == ["dangerous"]
    assert rig.controller.pending_approvals == []

    changed = rig.of_type(PermissionModeChanged)
    assert changed and changed[0].pending_resolved == 1
    resolutions = rig.of_type(ApprovalResolved)
    assert resolutions and resolutions[0].decision.by == "mode_switch"
    assert resolutions[0].request_id == request.request_id


async def test_an_approval_can_be_granted_by_hand(rig_factory) -> None:
    """R-C-6, the ordinary path: the step blocks until a human answers."""

    handler = QueueApprovalHandler()
    rig = rig_factory(
        [calls(touch_call("file.txt")), says("written")],
        mode=PermissionMode.MANUAL,
        handler=handler,
    )
    await rig.controller.start()

    request = await handler.next_request()
    assert request.tool == "touch"
    assert request.summary.title
    assert rig.touch.started == [], "nothing runs before the human answers"

    await rig.controller.approve(request.request_id)
    result = await rig.finish()

    assert result.final_text == "written"
    assert rig.touch.finished == ["file.txt"]
    assert rig.of_type(ApprovalRequested)


async def test_a_denial_reaches_the_model_as_a_structured_error(rig_factory) -> None:
    """R-C-6: `deny(reason)` becomes `ToolError(kind="denied")` with the reason."""

    handler = QueueApprovalHandler()
    rig = rig_factory(
        [calls(touch_call("file.txt")), says("understood")],
        mode=PermissionMode.MANUAL,
        handler=handler,
    )
    await rig.controller.start()
    request = await handler.next_request()
    await rig.controller.deny(request.request_id, "not that file")
    await rig.finish()

    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert results[0].result.error is not None
    assert results[0].result.error.kind is ToolErrorKind.DENIED
    assert "not that file" in results[0].result.text
    assert rig.touch.started == []


async def test_manual_mode_without_a_handler_refuses_to_start(rig_factory) -> None:
    """R-C-8: a `ConfigurationError` before any spend, not a hang at the first call."""

    rig = rig_factory([says("hi")], mode=PermissionMode.MANUAL, handler=None)
    with pytest.raises(ConfigurationError):
        await rig.controller.start()
    assert rig.controller.state is RunState.CREATED


async def test_a_deny_all_handler_makes_an_unattended_run_fail_visibly(rig_factory) -> None:
    """Spec C-5: headless runs choose `auto`, stdin, or refusal -- never a hang."""

    rig = rig_factory(
        [calls(touch_call("x")), says("fine")],
        mode=PermissionMode.MANUAL,
        handler=DenyAllHandler(),
    )
    await rig.controller.start()
    await rig.finish()

    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    assert results[0].result.error is not None
    assert results[0].result.error.kind is ToolErrorKind.DENIED


async def test_an_interrupt_withdraws_a_pending_approval(rig_factory) -> None:
    """The modal must come down when the call it belongs to is cancelled."""

    handler = QueueApprovalHandler()
    rig = rig_factory(
        [calls(touch_call("x")), says("moved on")],
        mode=PermissionMode.MANUAL,
        handler=handler,
    )
    await rig.controller.start()
    request = await handler.next_request()

    await rig.controller.interrupt()
    await rig.finish()

    assert [r[0] for r in handler.withdrawn] == [request.request_id]
    assert rig.controller.pending_approvals == []


# ---------------------------------------------------------------------------
# 7 and 8 live on disk now
# ---------------------------------------------------------------------------
#
# M2 stood these in for their spec 11 entries in memory: *fold* in every state
# rather than save in every state, and R-C-13 driven by an interrupt rather than by
# a process death. M3 has the real ones, and two tests asserting the same thing at
# different fidelities is how a suite rots -- so these were replaced, not added to.
#
#   7. save/load in every state
#        tests/test_save_load.py::test_save_works_from_every_run_state
#        tests/test_save_load.py::test_every_durable_safe_point_writes_the_session
#   8. the subprocess kill test
#        tests/test_kill.py::test_the_kill_test_reaches_the_same_final_output
#        tests/test_kill.py::test_the_slow_tool_starts_exactly_once_across_both_processes
#        tests/test_save_load.py::test_an_inflight_tool_call_reloads_as_interrupted_and_never_re_runs
#
# What is *not* duplicated there is scenario 4 -- an interrupt mid-batch, where the
# process stays alive and the fill is `cancelled` rather than `interrupted`. That
# distinction is the whole difference between R-C-4 and R-C-13, and it is tested
# above.


# ---------------------------------------------------------------------------
# Controller-level behaviour the scenarios lean on
# ---------------------------------------------------------------------------


async def test_the_run_completes_normally_with_tools(rig_factory) -> None:
    """The happy path, so a scenario failure is never ambiguous."""

    rig = rig_factory([calls(echo_call("a"), echo_call("b")), says("both read")])
    await rig.controller.start()
    result = await rig.finish()

    assert result.final_text == "both read"
    assert result.turns == 2
    assert sorted(rig.echo.finished) == ["a", "b"]
    assert rig.controller.state is RunState.COMPLETED
    assert rig.controller.nonquiescent == 0
    assert_transcript_valid(rig.transcript, agent_id="main")


async def test_pause_and_resume_are_idempotent(rig_factory) -> None:
    """R-C-1. Pausing twice and resuming a running run are no-ops, not errors."""

    rig = rig_factory([says("hi", chunk_delay_s=0.02)])
    await rig.controller.resume()  # not running yet
    await rig.controller.start()
    await rig.controller.pause()
    await rig.controller.pause()
    await rig.wait_state(RunState.PAUSED)
    await rig.controller.resume()
    await rig.controller.resume()
    await rig.finish()
    assert rig.controller.state is RunState.COMPLETED


async def test_cancelling_a_run_ends_it(rig_factory) -> None:
    """`cancel()` cuts every step and the run lands in CANCELLED."""

    rig = rig_factory([calls(echo_call("slow", sleep=5.0)), says("never")])
    await rig.controller.start()
    await rig.wait_type(ToolCallStarted)
    await rig.controller.cancel()

    assert rig.controller.state is RunState.CANCELLED
    assert rig.echo.finished == []


async def test_a_failing_body_fails_the_run(rig_factory) -> None:
    """A provider error the loop cannot retry away ends the run as FAILED (R-W-5)."""

    from azalabscode.errors import ProviderError, ProviderErrorKind

    rig = rig_factory(
        [
            says("", error=ProviderError(kind=ProviderErrorKind.AUTH, message="bad key")),
            says("", error=ProviderError(kind=ProviderErrorKind.AUTH, message="bad key")),
        ]
    )
    await rig.controller.start()
    with pytest.raises(Exception, match="bad key"):
        await rig.finish()
    assert rig.controller.state is RunState.FAILED


async def test_a_mid_stream_failure_is_retried_once(rig_factory) -> None:
    """Spec C-8: discard the partial, re-issue the whole call, once."""

    from azalabscode.errors import ProviderError, ProviderErrorKind
    from azalabscode.events import ModelCallFailed

    rig = rig_factory(
        [
            says(
                "half an ans",
                error=ProviderError(kind=ProviderErrorKind.SERVER, message="boom"),
                error_after_chunks=1,
            ),
            says("a whole answer"),
        ]
    )
    await rig.controller.start()
    result = await rig.finish()

    assert result.final_text == "a whole answer"
    failures = rig.of_type(ModelCallFailed)
    assert len(failures) == 1 and failures[0].will_retry is True
    assert len([m for m in rig.transcript if isinstance(m, AssistantMessage)]) == 1


async def test_the_controller_is_a_run_control(rig_factory) -> None:
    """Structural typing is what keeps `workflows` from importing `control`."""

    from azalabscode.contracts import PermissionGate, RunControl

    rig = rig_factory([says("hi")])
    gate = rig.controller.gate
    assert isinstance(rig.controller, RunControl)
    assert isinstance(gate, PermissionGate)


async def test_a_controller_with_no_body_cannot_start() -> None:
    """A clear error rather than a run that silently does nothing."""

    controller = Controller(permission_mode=PermissionMode.AUTO)
    with pytest.raises(ValueError, match="no run body"):
        await controller.start()


async def test_a_denied_call_does_not_stop_the_turns(rig_factory) -> None:
    """The model is told and carries on; a denial is data, not an exception."""

    rig = rig_factory(
        [calls(touch_call("x"), echo_call("y")), says("carried on")],
        mode=PermissionMode.MANUAL,
        handler=DenyAllHandler(),
    )
    await rig.controller.start()
    result = await rig.finish()

    assert result.final_text == "carried on"
    results = [m for m in rig.transcript if isinstance(m, ToolResultMessage)]
    requested = [
        c.call_id for m in rig.transcript if isinstance(m, AssistantMessage) for c in m.tool_calls
    ]
    assert [m.call_id for m in results] == requested
    assert results[1].result.ok


async def test_bounded_turns_stop_the_loop(rig_factory) -> None:
    """`max_turns` is a hard bound, and it is reported rather than raised."""

    from azalabscode.workflows import AgentSpec

    rig = rig_factory(
        [calls(echo_call("a")), calls(echo_call("b")), calls(echo_call("c"))],
        spec=AgentSpec(name="main", model="fake/model", max_turns=2),
    )
    await rig.controller.start()
    result = await rig.finish()

    assert result.stop_reason == "max_turns"
    assert result.turns == 2
    assert rig.controller.state is RunState.COMPLETED


async def test_an_approval_decision_records_who_made_it(rig_factory) -> None:
    """`Decision.by` is what the event log and the inspector show."""

    handler = QueueApprovalHandler()
    rig = rig_factory(
        [calls(touch_call("x")), says("ok")],
        mode=PermissionMode.MANUAL,
        handler=handler,
    )
    await rig.controller.start()
    request = await handler.next_request()
    await rig.controller.resolve_approval(
        request.request_id, Decision.approve(by="izaiah", reason="looks fine")
    )
    await rig.finish()

    resolved = rig.of_type(ApprovalResolved)
    assert resolved and resolved[0].by == "izaiah"


async def test_resolving_an_unknown_request_is_false_not_an_error(rig_factory) -> None:
    """Idempotence again: a modal answered twice must not raise."""

    rig = rig_factory([says("hi")])
    assert await rig.controller.approve("req_nonexistent") is False


async def test_waiting_for_a_state_that_never_comes_times_out(rig_factory) -> None:
    """The bound exists so a quiescence bug fails a test instead of hanging the suite."""

    rig = rig_factory([says("hi")])
    with pytest.raises(TimeoutError):
        await rig.controller.wait_for_state(RunState.PAUSED, timeout=0.05)
    assert BOUND > 0
