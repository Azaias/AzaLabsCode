"""The cancellation suite: absorb, re-raise, and the `uncancel()` count.

Three questions, and the whole control layer depends on getting all three right:

1. Was this cancellation *ours*? `cancel_reason is None` says no.
2. Is the reason one the run continues after? `RECOVERABLE_REASONS` says.
3. Did an enclosing scope cancel us as well? The cancel count says, and swallowing
   that one breaks `TaskGroup.__aexit__` and `asyncio.timeout.__aexit__`, both of
   which reconcile on it in 3.12.

Every test here is bounded by an `asyncio.timeout`: there is no `pytest-timeout` in
this project, so a hang would take the whole suite down with no output.
"""

from __future__ import annotations

import asyncio

import pytest

from azalabscode.cancellation import CancelReason, StepKind, StepOutcome
from azalabscode.contracts import StepHandleLike
from azalabscode.ids import MAIN_AGENT, AgentId, NodeId, StepId
from azalabscode.workflows.step import (
    StepHandle,
    run_inline_step,
    run_step,
    should_absorb,
)

BOUND = 5.0
"""Seconds any single test may take. Generous; a failure here means a hang."""


def handle(kind: StepKind = StepKind.MODEL_CALL, agent: str = "main") -> StepHandle:
    """A step handle for one test."""

    return StepHandle(agent_id=AgentId(agent), kind=kind)


async def forever() -> str:
    """A body that never finishes on its own."""

    await asyncio.sleep(3600)
    return "unreachable"  # pragma: no cover


# ---------------------------------------------------------------------------
# The handle contract
# ---------------------------------------------------------------------------


def test_the_handle_satisfies_the_control_facing_protocol() -> None:
    """`control` reaches a step only through `StepHandleLike`."""

    assert isinstance(handle(), StepHandleLike)


def test_the_first_cancel_reason_wins() -> None:
    """R-C-1: repeated interrupts are idempotent, and the first reason is the truth."""

    h = handle()
    assert h.request_cancel(CancelReason.USER_INTERRUPT) is True
    assert h.request_cancel(CancelReason.PAUSE_HARD) is False
    assert h.cancel_reason is CancelReason.USER_INTERRUPT


async def test_a_cancel_that_beats_the_task_is_delivered_by_bind() -> None:
    """The window between "the controller decided" and "the task exists".

    `request_cancel` before `bind` has no task to cancel. If `bind` did not deliver
    it, the step would run to completion after being interrupted.
    """

    h = handle()
    h.request_cancel(CancelReason.USER_INTERRUPT)

    async with asyncio.timeout(BOUND):
        result = await run_step(forever, handle=h)

    assert result.cancelled
    assert h.outcome is StepOutcome.CANCELLED


def test_the_reason_is_set_before_the_task_is_cancelled() -> None:
    """The ordering spec 4.4 insists on, asserted directly.

    A handler reading `cancel_reason` from inside `except CancelledError` must find
    it already there, so the write has to happen before `cancel()`.
    """

    seen: list[CancelReason | None] = []

    class Spy:
        def done(self) -> bool:
            return False

        def cancel(self) -> bool:
            seen.append(h.cancel_reason)
            return True

    h = handle()
    h.task = Spy()  # type: ignore[assignment]
    h.request_cancel(CancelReason.PAUSE_HARD)

    assert seen == [CancelReason.PAUSE_HARD]


# ---------------------------------------------------------------------------
# Absorb
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    [CancelReason.USER_INTERRUPT, CancelReason.PAUSE_HARD, CancelReason.TIMEOUT],
)
async def test_a_recoverable_cancellation_is_absorbed(reason: CancelReason) -> None:
    """The run continues after these three; the step reports rather than raises."""

    h = handle()

    async def interrupt() -> None:
        await asyncio.sleep(0)
        h.request_cancel(reason)

    async with asyncio.timeout(BOUND):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(interrupt())
            result = await run_step(forever, handle=h)

    assert result.cancelled
    assert result.reason is reason
    assert result.value is None


async def test_cancelling_a_step_does_not_cancel_the_agent_awaiting_it() -> None:
    """Spec 4.4's reason for the child-task shape.

    Interrupting one step must leave the agent free to take its next one, so the
    awaiting task's cancel count has to come back untouched.
    """

    task = asyncio.current_task()
    assert task is not None
    entry = task.cancelling()

    h = handle()

    async def interrupt() -> None:
        await asyncio.sleep(0)
        h.request_cancel(CancelReason.USER_INTERRUPT)

    async with asyncio.timeout(BOUND):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(interrupt())
            await run_step(forever, handle=h)

        assert task.cancelling() == entry
        # The agent is still usable: a second step runs to completion.
        second = await run_step(lambda: _value("ok"), handle=handle())

    assert second.completed
    assert second.value == "ok"


async def _value(v: str) -> str:
    return v


# ---------------------------------------------------------------------------
# Re-raise
# ---------------------------------------------------------------------------


async def test_a_cancellation_with_no_reason_is_re_raised() -> None:
    """Rule 2. Absence of a reason is the authoritative signal that it is not ours.

    This is a `TaskGroup` unwinding, a `wait_for` above us, or interpreter shutdown.
    Absorbing it would leave the task tree half torn down.
    """

    h = handle()

    async def cancel_the_task_directly() -> None:
        await asyncio.sleep(0)
        assert h.task is not None
        h.task.cancel()

    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(BOUND):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(cancel_the_task_directly())
                await run_step(forever, handle=h)

    assert h.cancel_reason is None


@pytest.mark.parametrize(
    "reason",
    [CancelReason.PARENT_FAILED, CancelReason.RUN_CANCELLED, CancelReason.SHUTDOWN],
)
async def test_an_unrecoverable_reason_is_re_raised(reason: CancelReason) -> None:
    """Rule 2 again: the reason exists but the run is ending, so it propagates."""

    h = handle()

    async def interrupt() -> None:
        await asyncio.sleep(0)
        h.request_cancel(reason)

    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(BOUND):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(interrupt())
                await run_step(forever, handle=h)


# ---------------------------------------------------------------------------
# The uncancel count
# ---------------------------------------------------------------------------


async def test_an_inline_step_absorbs_and_reconciles_the_cancel_count() -> None:
    """Rule 3, the absorbing half.

    Cancelling an inline step cancels the caller, so the count grows by one.
    Absorbing means putting it back with `uncancel()`, or the next
    `asyncio.timeout` in this task would raise `TimeoutError` for no reason.
    """

    task = asyncio.current_task()
    assert task is not None
    entry = task.cancelling()
    h = handle()

    async def body() -> str:
        h.request_cancel(CancelReason.USER_INTERRUPT)
        await asyncio.sleep(3600)
        return "unreachable"  # pragma: no cover

    result = await run_inline_step(body, handle=h)

    assert result.cancelled
    assert task.cancelling() == entry, "the count must come back to where it started"


async def test_an_inline_step_re_raises_when_an_enclosing_scope_cancelled_too() -> None:
    """Rule 3, the re-raising half. This is the bug the rule exists to prevent.

    Two cancellations are outstanding: ours and somebody else's. Swallowing ours
    would consume the other one's delivery as well, and the enclosing scope would
    wait forever for an unwind that already happened.
    """

    task = asyncio.current_task()
    assert task is not None
    entry = task.cancelling()
    h = handle()

    async def body() -> str:
        task.cancel()  # an enclosing scope, before us
        h.request_cancel(CancelReason.USER_INTERRUPT)  # and then us
        await asyncio.sleep(3600)
        return "unreachable"  # pragma: no cover

    with pytest.raises(asyncio.CancelledError):
        await run_inline_step(body, handle=h)

    assert task.cancelling() == entry + 2, "no count may be consumed on the re-raise path"
    task.uncancel()
    task.uncancel()
    assert task.cancelling() == entry


def test_should_absorb_is_false_without_a_reason() -> None:
    """The predicate in isolation, outside any task."""

    assert should_absorb(handle(), entry_cancelling=0) is False


# ---------------------------------------------------------------------------
# Exception groups: a delegate step wraps its child in a TaskGroup
# ---------------------------------------------------------------------------


async def test_a_real_child_failure_outranks_the_cancellation() -> None:
    """A crashed subagent must not be reported as a clean interrupt."""

    h = handle(StepKind.DELEGATE)

    async def body() -> str:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_boom())
            tg.create_task(asyncio.sleep(3600))
        return "unreachable"  # pragma: no cover

    with pytest.raises(BaseExceptionGroup) as caught:
        async with asyncio.timeout(BOUND):
            await run_step(body, handle=h)

    assert any(isinstance(e, RuntimeError) for e in caught.value.exceptions)
    assert h.outcome is StepOutcome.FAILED


async def test_a_group_of_only_cancellations_follows_the_normal_rules() -> None:
    """A delegate cancelled cleanly surfaces a group with nothing but cancellations."""

    h = handle(StepKind.DELEGATE)

    async def body() -> str:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(asyncio.sleep(3600))
            tg.create_task(asyncio.sleep(3600))
        return "unreachable"  # pragma: no cover

    async def interrupt() -> None:
        await asyncio.sleep(0.01)
        h.request_cancel(CancelReason.USER_INTERRUPT)

    async with asyncio.timeout(BOUND):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(interrupt())
            result = await run_step(body, handle=h)

    assert result.cancelled


async def _boom() -> None:
    await asyncio.sleep(0)
    raise RuntimeError("the child failed")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


async def test_an_exception_propagates_by_default() -> None:
    """A model call that raises is a failed run, not a failed step."""

    h = handle()
    with pytest.raises(RuntimeError):
        async with asyncio.timeout(BOUND):
            await run_step(_boom, handle=h)
    assert h.outcome is StepOutcome.FAILED


async def test_capture_errors_returns_the_exception_instead() -> None:
    """The opt-in for a caller that wants to decide for itself."""

    h = handle()
    async with asyncio.timeout(BOUND):
        result = await run_step(_boom, handle=h, capture_errors=True)

    assert result.failed
    assert isinstance(result.error, RuntimeError)


# ---------------------------------------------------------------------------
# Registration with control
# ---------------------------------------------------------------------------


class RecordingControl:
    """Just enough `RunControl` to observe registration."""

    def __init__(self) -> None:
        self.registered: list[StepId] = []
        self.unregistered: list[StepId] = []

    def register_step(self, h: StepHandleLike) -> None:
        self.registered.append(h.step_id)

    def unregister_step(self, h: StepHandleLike) -> None:
        self.unregistered.append(h.step_id)


async def test_a_step_registers_and_unregisters_even_when_cancelled() -> None:
    """`inflight` must not leak a step that was interrupted."""

    control = RecordingControl()
    h = handle()
    h.request_cancel(CancelReason.USER_INTERRUPT)

    async with asyncio.timeout(BOUND):
        await run_step(forever, handle=h, control=control)  # type: ignore[arg-type]

    assert control.registered == [h.step_id]
    assert control.unregistered == [h.step_id]


async def test_a_step_unregisters_when_the_body_raises() -> None:
    """The same on the failure path, which is the one people forget."""

    control = RecordingControl()
    h = handle()

    with pytest.raises(RuntimeError):
        async with asyncio.timeout(BOUND):
            await run_step(_boom, handle=h, control=control)  # type: ignore[arg-type]

    assert control.unregistered == [h.step_id]


async def test_a_completed_step_records_its_outcome_and_duration() -> None:
    """The bookkeeping the event log and the status bar read."""

    h = StepHandle(
        agent_id=MAIN_AGENT,
        kind=StepKind.TOOL_CALL,
        node_id=NodeId("agent"),
        call_ids=["c1", "c2"],
        description="2 tool calls",
    )

    async with asyncio.timeout(BOUND):
        result = await run_step(lambda: _value("done"), handle=h)

    assert result.completed
    assert result.value == "done"
    assert h.outcome is StepOutcome.COMPLETED
    assert h.duration_ms >= 0.0
    assert h.call_ids == ["c1", "c2"]
