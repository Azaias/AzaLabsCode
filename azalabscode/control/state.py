"""The run state machine and the per-agent record the controller folds into.

`azalabscode.runstate` holds the vocabulary -- the enums and the legal-transition
table -- because `events` has to carry a `RunStateChanged` without importing
upward. The *machine* is here, where the policy is: which transitions are attempted,
what a self-transition means, and who is told.

`AgentState` is re-exported here for convenience but *defined* in
`workflows.state`: the agent loop is what reads and writes it, and `workflows` may
not import `control`. M3 embeds it in `Session` unchanged.
"""

from __future__ import annotations

import asyncio

from azalabscode.errors import HarnessError
from azalabscode.events import EventEmitter, RunStateChanged
from azalabscode.runstate import TERMINAL_STATES, RunState, is_legal_transition
from azalabscode.workflows.state import AgentState


class IllegalTransition(HarnessError):
    """A run-state transition the machine does not allow (R-C-2, spec 6.1).

    Raised rather than logged: an illegal transition means the controller's own
    bookkeeping is wrong, and continuing from there produces a session document that
    describes a run that never happened.
    """

    def __init__(self, old: RunState, new: RunState) -> None:
        self.old = old
        self.new = new
        super().__init__(f"illegal run-state transition: {old} -> {new}")


class RunStateMachine:
    """The coarse run state, its legal transitions, and who is told about them.

    A self-transition is legal and is a no-op that still emits, so a UI that missed
    an event has a chance to re-sync. Anything else not in the table raises.
    """

    def __init__(
        self,
        *,
        emitter: EventEmitter | None = None,
        state: RunState = RunState.CREATED,
    ) -> None:
        self._state = state
        self._emitter = emitter
        self._waiters: list[asyncio.Future[RunState]] = []
        self.history: list[tuple[RunState, RunState, str | None]] = []

    @property
    def state(self) -> RunState:
        """The current run state."""

        return self._state

    @property
    def terminal(self) -> bool:
        """True once the run can never transition again."""

        return self._state in TERMINAL_STATES

    def can(self, new: RunState) -> bool:
        """Whether `new` is reachable from here."""

        return is_legal_transition(self._state, new)

    async def transition(self, new: RunState, reason: str | None = None) -> bool:
        """Move to `new`. Returns False for a no-op self-transition.

        The state is written **before** the event is emitted: a subscriber that
        reacts by reading `controller.state` must not see the old one.
        """

        old = self._state
        if not is_legal_transition(old, new):
            raise IllegalTransition(old, new)
        if old is new:
            return False

        self._state = new
        self.history.append((old, new, reason))
        self._wake(new)
        if self._emitter is not None:
            await self._emitter.emit(RunStateChanged(old=old, new=new, reason=reason))
        return True

    def restore(self, state: RunState, reason: str | None = None) -> None:
        """Adopt `state` without checking the transition table.

        A load is not a transition. The run being adopted was PAUSED (or waiting on
        an approval) in another process, and there is no legal path from CREATED to
        that in spec 6.1 -- nor should there be, because inventing one would let a
        live run take the same shortcut. Emits nothing: `Controller._restore` sends a
        single `RunStateChanged` describing the load, which is one event rather than
        the three a walk through the table would produce.
        """

        old = self._state
        self._state = state
        self.history.append((old, state, reason or "restored from a checkpoint"))
        self._wake(state)

    def _wake(self, state: RunState) -> None:
        waiters, self._waiters = self._waiters, []
        for future in waiters:
            if not future.done():
                future.set_result(state)

    async def wait_for(self, *states: RunState, timeout: float | None = None) -> RunState:  # noqa: ASYNC109 - the bound is the point; there is no ambient deadline
        """Block until the run is in one of `states`. Returns the state reached.

        Bounded by `timeout` on purpose: this project has no `pytest-timeout`, so a
        test that waits for a state that never arrives would hang the whole suite.
        """

        async def _wait() -> RunState:
            while self._state not in states:
                future: asyncio.Future[RunState] = asyncio.get_running_loop().create_future()
                self._waiters.append(future)
                await future
            return self._state

        if timeout is None:
            return await _wait()
        async with asyncio.timeout(timeout):
            return await _wait()


__all__ = ["AgentState", "IllegalTransition", "RunStateMachine"]
