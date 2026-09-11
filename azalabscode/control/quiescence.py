"""Quiescence: the counter that decides when a paused run is actually paused.

Spec 4.4 defines PAUSED as "the count of agents parked at the gate equals the count
of active agents". That formulation has two failure modes, and both of them are
reachable in the coding agent:

* **It races on spawn.** Between `create_task(child)` and the child registering
  itself there is an instant where the parent is parked and the child does not
  exist, so parked == active and the run is declared PAUSED while a subagent is
  about to start making model calls.
* **It deadlocks on delegate.** A parent awaiting a child is not parked and never
  will be -- it is blocked on the child. If the child parks, parked is 1 and active
  is 2, so PAUSED is never reached and `pause()` hangs forever.

Spec delta 14 replaces it: every agent has a *phase*, some phases are quiescent
(`azalabscode.runstate.QUIESCENT_PHASES`), and

    PAUSED  <=>  a pause was requested  and  no agent is in a non-quiescent phase.

`blocked_on_child` is quiescent, which is what fixes the deadlock. It is safe
because a delegate step has no external effect of its own: every effect the child
has is guarded by the child's own safe points.

The spawn race is closed elsewhere, in `Controller.enter_agent`, which the agent
loop calls **before** creating the child's task.
"""

from __future__ import annotations

import asyncio

from azalabscode.ids import AgentId
from azalabscode.runstate import AgentPhase, is_quiescent


class QuiescenceTracker:
    """Per-agent phases, the derived non-quiescent count, and the pause gate.

    Deliberately has no opinion about run state or events: it counts, and it opens
    and closes a gate. `Controller` reads the count and decides what it means.
    """

    def __init__(self) -> None:
        self.phases: dict[str, AgentPhase] = {}
        self._open = asyncio.Event()
        self._open.set()
        self._quiet = asyncio.Event()
        self._quiet.set()
        self.pause_requested = False

    # -- membership ---------------------------------------------------------

    def enter(self, agent_id: AgentId | str, phase: AgentPhase = AgentPhase.RUNNING) -> None:
        """Register an agent as active.

        Called synchronously before a child's task is created, so there is never an
        instant in which the child is unregistered and its parent is already
        quiescent.
        """

        self.phases[str(agent_id)] = phase
        self._recount()

    def exit(self, agent_id: AgentId | str) -> None:
        """Drop an agent from the count. Idempotent."""

        self.phases.pop(str(agent_id), None)
        self._recount()

    def set_phase(self, agent_id: AgentId | str, phase: AgentPhase) -> AgentPhase | None:
        """Record a phase and return the previous one, or `None` if it was new."""

        key = str(agent_id)
        old = self.phases.get(key)
        self.phases[key] = phase
        self._recount()
        return old

    def phase_of(self, agent_id: AgentId | str) -> AgentPhase | None:
        """One agent's phase, or `None` if it is not registered."""

        return self.phases.get(str(agent_id))

    # -- the count ----------------------------------------------------------

    @property
    def nonquiescent(self) -> int:
        """How many agents are still doing something that blocks a pause."""

        return sum(1 for phase in self.phases.values() if not is_quiescent(phase))

    @property
    def quiescent(self) -> bool:
        """True when nothing is in flight. Recomputed, never cached."""

        return self.nonquiescent == 0

    @property
    def active(self) -> int:
        """How many agents are registered at all."""

        return len(self.phases)

    def nonquiescent_agents(self) -> list[str]:
        """Which agents are holding the pause up. What C-12's status bar renders."""

        return [a for a, phase in self.phases.items() if not is_quiescent(phase)]

    def _recount(self) -> None:
        if self.quiescent:
            self._quiet.set()
        else:
            self._quiet.clear()

    async def wait_quiescent(self, *, timeout: float | None = None) -> None:  # noqa: ASYNC109 - the bound is the point; there is no ambient deadline
        """Block until no agent is non-quiescent."""

        if timeout is None:
            await self._quiet.wait()
            return
        async with asyncio.timeout(timeout):
            await self._quiet.wait()

    # -- the gate -----------------------------------------------------------

    @property
    def open(self) -> bool:
        """True when agents may run past a safe point."""

        return self._open.is_set()

    def request_pause(self) -> bool:
        """Close the gate. Returns False if a pause was already pending."""

        if self.pause_requested:
            return False
        self.pause_requested = True
        self._open.clear()
        return True

    def release(self) -> bool:
        """Open the gate and wake every parked agent. Returns False if not paused."""

        if not self.pause_requested:
            self._open.set()
            return False
        self.pause_requested = False
        self._open.set()
        return True

    async def wait_open(self) -> None:
        """Park until the gate opens.

        The caller must already have set its own phase to a quiescent one, and must
        not be holding the checkpoint lock: an agent parked while holding it
        deadlocks every other checkpoint and PAUSED is never reached.
        """

        await self._open.wait()


__all__ = ["QuiescenceTracker"]
