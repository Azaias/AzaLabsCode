"""Run, agent and node lifecycle enums, plus the legal-transition table.

The *machine* lives in `control/state.py`; only the vocabulary is here, because
`events` must be able to carry a `RunStateChanged` without importing the control
layer. `AgentPhase` is here for the same reason and for a second one: quiescence is
defined as a property of phases (spec delta 14), and both `workflows` (which sets
phases) and `control` (which counts them) need the definition to be the same one.
"""

from __future__ import annotations

from enum import StrEnum


class RunState(StrEnum):
    """Coarse state of a whole run (R-C-2)."""

    CREATED = "created"
    RUNNING = "running"
    PAUSING = "pausing"
    """A pause was requested; agents are walking to their next safe point."""
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    INTERRUPTING = "interrupting"
    """Transient. The run returns to whatever state it was in (spec C-4)."""
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES: frozenset[RunState] = frozenset(
    {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}
)

LEGAL_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.RUNNING, RunState.CANCELLED, RunState.FAILED}),
    RunState.RUNNING: frozenset(
        {
            RunState.PAUSING,
            RunState.WAITING_APPROVAL,
            RunState.INTERRUPTING,
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.PAUSING: frozenset(
        {RunState.PAUSED, RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.PAUSED: frozenset(
        {
            RunState.RUNNING,
            RunState.WAITING_APPROVAL,
            RunState.INTERRUPTING,
            RunState.FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.WAITING_APPROVAL: frozenset(
        {
            RunState.RUNNING,
            RunState.PAUSING,
            RunState.PAUSED,
            RunState.INTERRUPTING,
            RunState.FAILED,
            RunState.CANCELLED,
        }
    ),
    # INTERRUPTING returns to whatever state it came from, so every non-terminal
    # state is reachable from it.
    RunState.INTERRUPTING: frozenset(
        {
            RunState.RUNNING,
            RunState.PAUSED,
            RunState.PAUSING,
            RunState.WAITING_APPROVAL,
            RunState.FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
}
"""Transitions permitted by the state machine (spec 6.1)."""


def is_legal_transition(old: RunState, new: RunState) -> bool:
    """True when `old -> new` is a transition the state machine allows.

    A self-transition is always legal and always a no-op: `Controller` methods are
    idempotent where meaningful (R-C-1), so `pause()` on a paused run must not be an
    error.
    """

    return old is new or new in LEGAL_TRANSITIONS[old]


class AgentPhase(StrEnum):
    """What one agent is doing. Quiescence is defined over these (spec delta 14).

    PAUSED is reached when a pause has been requested and the count of
    non-quiescent agents reaches zero -- not when "parked equals active", which
    races on spawn and deadlocks whenever a subagent parks while its parent awaits
    it.
    """

    RUNNING = "running"
    """Executing Python between two safe points. Not quiescent."""

    BLOCKED_IO = "blocked_io"
    """Inside a model call or a tool call. Not quiescent: it has an open effect."""

    BLOCKED_ON_CHILD = "blocked_on_child"
    """Awaiting a delegate or a spawned handle. Quiescent.

    It must be, or a subagent parked at the gate deadlocks a parent that will never
    return. It is safe because a delegate step has no external effect of its own,
    and every effect the child has is guarded by the child's own safe points.
    """

    PARKED = "parked"
    """Waiting at the pause gate. Quiescent."""

    WAITING_APPROVAL = "waiting_approval"
    """Blocked on a human decision. Quiescent."""

    FINISHED = "finished"
    """Done, failed or cancelled. Quiescent."""


QUIESCENT_PHASES: frozenset[AgentPhase] = frozenset(
    {
        AgentPhase.BLOCKED_ON_CHILD,
        AgentPhase.PARKED,
        AgentPhase.WAITING_APPROVAL,
        AgentPhase.FINISHED,
    }
)


def is_quiescent(phase: AgentPhase) -> bool:
    """True when an agent in `phase` does not block the run from reaching PAUSED."""

    return phase in QUIESCENT_PHASES


class NodeStatus(StrEnum):
    """Lifecycle of one graph node. Outputs are memoized by (node_id, attempt)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    """Never re-executed on resume (R-W-6)."""
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


__all__ = [
    "LEGAL_TRANSITIONS",
    "QUIESCENT_PHASES",
    "TERMINAL_STATES",
    "AgentPhase",
    "NodeStatus",
    "RunState",
    "is_legal_transition",
    "is_quiescent",
]
