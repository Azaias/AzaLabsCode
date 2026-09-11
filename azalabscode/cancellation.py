"""Cancellation vocabulary.

The single most important rule in the harness lives here in prose, and is enforced
in `workflows/step.py` (M2):

    A step absorbs `CancelledError` only when it set `cancel_reason` itself *and*
    `asyncio.current_task().uncancel()` returns 0. A cancellation with no reason is
    never ours -- it is a `TaskGroup` unwinding or an interpreter shutdown -- and
    must be re-raised unconditionally.

Both halves matter. The reason distinguishes a user interrupt from structured
teardown; the uncancel count catches the case where an enclosing scope cancelled us
as well, where swallowing the exception would break `TaskGroup.__aexit__` and
`asyncio.timeout.__aexit__`, which reconcile on that count in 3.12.
"""

from __future__ import annotations

from enum import StrEnum


class CancelReason(StrEnum):
    """Why a step was cancelled. Set *before* `task.cancel()`, never after."""

    USER_INTERRUPT = "user_interrupt"
    """`Controller.interrupt()`. The run continues at the agent's next model call."""

    PAUSE_HARD = "pause_hard"
    """`Controller.pause(hard=True)`. In-flight steps are cut rather than awaited."""

    TIMEOUT = "timeout"
    """The dispatcher's per-tool timeout, or a model-call deadline."""

    PARENT_FAILED = "parent_failed"
    """A sibling or parent node failed and structured concurrency is unwinding."""

    RUN_CANCELLED = "run_cancelled"
    """`Controller.cancel()`. The run is over."""

    SHUTDOWN = "shutdown"
    """Interpreter or event-loop teardown. Never absorbed."""


class StepKind(StrEnum):
    """What a cancellable step was doing, which decides how it is reconciled."""

    MODEL_CALL = "model_call"
    """Dropped and re-issued on resume: nothing was half-appended (spec C-1)."""

    TOOL_CALL = "tool_call"
    """Never re-executed. Becomes `ToolError(kind="interrupted")` (R-C-13)."""

    DELEGATE = "delegate"
    """Resumed, not errored: a delegate has no external effect of its own."""

    NODE = "node"
    """A whole node body; restarts from its last checkpoint (R-W-6)."""


class StepOutcome(StrEnum):
    """How a step finished. Recorded on the step handle and in the event log."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    """The process died mid-step; discovered at load, never observed live."""


RECOVERABLE_REASONS: frozenset[CancelReason] = frozenset(
    {
        CancelReason.USER_INTERRUPT,
        CancelReason.PAUSE_HARD,
        CancelReason.TIMEOUT,
    }
)
"""Reasons after which the run continues.

A step cancelled for one of these absorbs the `CancelledError` and reports a
`StepOutcome`. Any other reason propagates: the run is ending, and swallowing the
cancellation would leave the task tree in an inconsistent state.
"""


def is_recoverable(reason: CancelReason | None) -> bool:
    """True when a step cancelled for `reason` should absorb rather than re-raise.

    `None` -- no reason recorded -- is never recoverable. Absence of a reason is the
    authoritative signal that the cancellation came from outside the harness.
    """

    return reason is not None and reason in RECOVERABLE_REASONS


__all__ = [
    "RECOVERABLE_REASONS",
    "CancelReason",
    "StepKind",
    "StepOutcome",
    "is_recoverable",
]
