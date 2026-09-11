"""Cancellable steps, and the one rule the whole control layer rests on.

A *step* is the granularity of interrupt: one model call, one tool batch, one
delegated subtree. It runs in a task of its own so it can be cancelled without
cancelling the agent around it (spec 4.4), and it is tracked by a `StepHandle` that
`control` can reach through the `StepHandleLike` protocol.

The rule, restated from `azalabscode.cancellation`:

    A step absorbs `CancelledError` only when it set `cancel_reason` itself *and*
    reconciling the cancel count leaves the awaiting task exactly as it found it.

Three rules follow, and each one is a mistake that's easy to make:

1. **Never `except Exception` around a step body.** In 3.12 `CancelledError` is a
   `BaseException`, so that clause would not catch it -- and if it did, it would be
   wrong.
2. **`cancel_reason is None` means the cancellation is not ours.** A `TaskGroup`
   unwinding after a sibling failure, an interpreter shutdown, a `wait_for` above
   us: all of them cancel without a reason. Re-raise unconditionally.
   `is_recoverable(None)` is `False` and that is the authoritative check.
3. **Absorb only after reconciling the cancel count.** `asyncio.timeout` and
   `TaskGroup.__aexit__` both reconcile on `Task.cancelling()` in 3.12. Swallowing a
   cancellation that an enclosing scope also requested makes both of them
   misbehave: the timeout does not raise `TimeoutError`, the task group does not
   finish unwinding.

`run_step` runs the body in a child task, which is the shape spec 4.4 asks for and
the shape the agent loop uses. `run_inline_step` runs it in the caller's own task,
where cancelling the step *is* cancelling the caller, and absorbing therefore means
calling `uncancel()`. Both go through `should_absorb`, which is where the rule
actually lives.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from azalabscode.cancellation import (
    CancelReason,
    StepKind,
    StepOutcome,
    is_recoverable,
)
from azalabscode.contracts import RunControl
from azalabscode.ids import AgentId, CallId, NodeId, StepId, new_step_id


@dataclass
class StepHandle:
    """One cancellable unit of work. Satisfies `contracts.StepHandleLike`.

    `request_cancel` sets `cancel_reason` **before** calling `task.cancel()`, and the
    first reason wins so repeated interrupts are idempotent (R-C-1). A cancel that
    arrives before the task exists is remembered and delivered by `bind`, which
    closes the window between "the controller decided to interrupt" and "the step
    task was created".
    """

    agent_id: AgentId
    kind: StepKind
    step_id: StepId = field(default_factory=new_step_id)
    node_id: NodeId | None = None
    call_id: CallId | None = None
    call_ids: list[str] = field(default_factory=list)
    """Every call id this step covers. A tool batch has several; a model call none.

    This is what makes an in-flight batch reconcilable after a process death: the
    ids with no result in `AgentState.pending_results` are the ones that become
    `ToolError(kind="interrupted")` (R-C-13).
    """
    child_agent_id: str | None = None
    """For a `delegate` step, the child agent it is waiting on.

    Recorded so a checkpoint taken mid-delegate can name the child, and a resumed
    parent re-enters *that* agent's transcript rather than starting a new one
    (spec delta 16). A leaf tool call leaves it `None`.
    """
    description: str = ""
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    cancel_reason: CancelReason | None = None
    outcome: StepOutcome | None = None
    task: asyncio.Task[Any] | None = None

    @property
    def done(self) -> bool:
        """True once an outcome has been recorded."""

        return self.outcome is not None

    @property
    def duration_ms(self) -> float:
        """Elapsed time, live or final."""

        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return (end - self.started_at) * 1000.0

    def request_cancel(self, reason: CancelReason) -> bool:
        """Record `reason` and cancel the task. Returns False if already requested.

        The order matters: a handler that reads `cancel_reason` from inside its
        `except CancelledError` must find it already set, so the reason has to be
        written before `cancel()` schedules the throw.
        """

        if self.cancel_reason is not None:
            return False
        self.cancel_reason = reason
        self._deliver()
        return True

    def bind(self, task: asyncio.Task[Any] | None) -> None:
        """Attach the task running this step, delivering any cancel that beat it."""

        self.task = task
        if self.cancel_reason is not None:
            self._deliver()

    def _deliver(self) -> None:
        task = self.task
        if task is not None and not task.done():
            task.cancel()

    def finish(self, outcome: StepOutcome) -> None:
        """Record the outcome. Idempotent: the first outcome wins."""

        if self.outcome is None:
            self.outcome = outcome
            self.finished_at = time.monotonic()

    def __repr__(self) -> str:
        return (
            f"<StepHandle {self.kind} agent={self.agent_id} "
            f"reason={self.cancel_reason} outcome={self.outcome}>"
        )


@dataclass
class StepResult[T]:
    """What a step produced, or why it did not."""

    handle: StepHandle
    outcome: StepOutcome
    value: T | None = None
    reason: CancelReason | None = None
    error: BaseException | None = None

    @property
    def completed(self) -> bool:
        """True when the body ran to completion."""

        return self.outcome is StepOutcome.COMPLETED

    @property
    def cancelled(self) -> bool:
        """True when the step was cancelled and the cancellation was absorbed."""

        return self.outcome is StepOutcome.CANCELLED

    @property
    def failed(self) -> bool:
        """True when the body raised and the caller asked for the error back."""

        return self.outcome is StepOutcome.FAILED


def should_absorb(handle: StepHandle, *, entry_cancelling: int) -> bool:
    """Whether this `CancelledError` is ours to swallow. **The rule lives here.**

    `entry_cancelling` is `Task.cancelling()` sampled before the step started, which
    is what makes the check work under an enclosing `asyncio.timeout` that has
    already cancelled us once for its own reasons.

    Three outcomes:

    * The reason is absent or not recoverable -- somebody else's cancellation.
      Re-raise, and do not touch the count.
    * The awaiting task's count is unchanged: only the *child* step task was
      cancelled, so there is nothing to reconcile and the cancellation is ours.
    * The count grew by exactly one and the handle owns the current task -- an
      inline step. `uncancel()` puts the count back where it was; absorb only if
      that lands on `entry_cancelling`. Any larger growth means an enclosing scope
      cancelled us as well, and swallowing it would break that scope.
    """

    if not is_recoverable(handle.cancel_reason):
        return False

    task = asyncio.current_task()
    if task is None:  # pragma: no cover - a step always runs on a task
        return True

    now = task.cancelling()
    if now == entry_cancelling:
        return True
    if now == entry_cancelling + 1 and handle.task is task:
        return task.uncancel() == entry_cancelling
    return False


def _entry_count() -> int:
    """`Task.cancelling()` for the running task, or 0 outside one."""

    task = asyncio.current_task()
    return task.cancelling() if task is not None else 0


def _split_group(group: BaseExceptionGroup[BaseException]) -> BaseException | None:
    """The non-cancellation half of an exception group, or `None` if there is none.

    A `delegate` step wraps its child in a `TaskGroup`, so cancelling it surfaces
    either a bare `CancelledError` or a group. A real child failure outranks the
    cancellation: losing it would turn a crashed subagent into a clean interrupt.
    """

    _, rest = group.split(asyncio.CancelledError)
    return rest


async def run_step[T](
    factory: Callable[[], Coroutine[Any, Any, T]],
    *,
    handle: StepHandle,
    control: RunControl | None = None,
    capture_errors: bool = False,
) -> StepResult[T]:
    """Run `factory()` as a step in a task of its own.

    The child-task shape is what lets `interrupt` cancel one step without cancelling
    the agent that is awaiting it (spec 4.4): the controller cancels `handle.task`,
    the awaiting agent's own cancel count never moves, and `should_absorb` sees an
    unchanged count.

    `capture_errors=False` (the default) lets a real exception propagate: a model
    call that raises is a failed run, not a failed step. The tool path never needs
    it, because the dispatcher returns failures as data.
    """

    entry = _entry_count()
    task = asyncio.create_task(factory(), name=f"step:{handle.kind}:{handle.step_id}")
    handle.bind(task)
    if control is not None:
        control.register_step(handle)

    try:
        try:
            value = await task
        except asyncio.CancelledError:
            handle.finish(StepOutcome.CANCELLED)
            if not should_absorb(handle, entry_cancelling=entry):
                raise
            return StepResult(
                handle=handle,
                outcome=StepOutcome.CANCELLED,
                reason=handle.cancel_reason,
            )
        except BaseExceptionGroup as group:
            rest = _split_group(group)
            if rest is not None:
                handle.finish(StepOutcome.FAILED)
                if capture_errors:
                    return StepResult(handle=handle, outcome=StepOutcome.FAILED, error=rest)
                raise rest from None
            handle.finish(StepOutcome.CANCELLED)
            if not should_absorb(handle, entry_cancelling=entry):
                raise
            return StepResult(
                handle=handle,
                outcome=StepOutcome.CANCELLED,
                reason=handle.cancel_reason,
            )
        except Exception as exc:
            handle.finish(StepOutcome.FAILED)
            if capture_errors:
                return StepResult(handle=handle, outcome=StepOutcome.FAILED, error=exc)
            raise
        handle.finish(StepOutcome.COMPLETED)
        return StepResult(handle=handle, outcome=StepOutcome.COMPLETED, value=value)
    finally:
        if control is not None:
            control.unregister_step(handle)


async def run_inline_step[T](
    factory: Callable[[], Coroutine[Any, Any, T]],
    *,
    handle: StepHandle,
    control: RunControl | None = None,
    capture_errors: bool = False,
) -> StepResult[T]:
    """Run `factory()` in the caller's own task rather than a child task.

    The difference that matters is the cancel count: cancelling an inline step
    cancels the caller, so absorbing means calling `uncancel()` to put the count
    back. `should_absorb` does that only when the growth is exactly the one this
    handle caused; if an enclosing scope cancelled us too, the step re-raises and
    leaves the count alone.

    Use this where the body must share the caller's context (a context manager it
    entered, a `TaskGroup` it owns). The agent loop uses `run_step`.
    """

    entry = _entry_count()
    handle.bind(asyncio.current_task())
    if control is not None:
        control.register_step(handle)

    try:
        try:
            value = await factory()
        except asyncio.CancelledError:
            handle.finish(StepOutcome.CANCELLED)
            if not should_absorb(handle, entry_cancelling=entry):
                raise
            return StepResult(
                handle=handle,
                outcome=StepOutcome.CANCELLED,
                reason=handle.cancel_reason,
            )
        except BaseExceptionGroup as group:
            rest = _split_group(group)
            if rest is not None:
                handle.finish(StepOutcome.FAILED)
                if capture_errors:
                    return StepResult(handle=handle, outcome=StepOutcome.FAILED, error=rest)
                raise rest from None
            handle.finish(StepOutcome.CANCELLED)
            if not should_absorb(handle, entry_cancelling=entry):
                raise
            return StepResult(
                handle=handle,
                outcome=StepOutcome.CANCELLED,
                reason=handle.cancel_reason,
            )
        except Exception as exc:
            handle.finish(StepOutcome.FAILED)
            if capture_errors:
                return StepResult(handle=handle, outcome=StepOutcome.FAILED, error=exc)
            raise
        handle.finish(StepOutcome.COMPLETED)
        return StepResult(handle=handle, outcome=StepOutcome.COMPLETED, value=value)
    finally:
        handle.bind(None)
        if control is not None:
            control.unregister_step(handle)


__all__ = [
    "StepHandle",
    "StepKind",
    "StepOutcome",
    "StepResult",
    "run_inline_step",
    "run_step",
    "should_absorb",
]
