"""`AgentHandle`: what `spawn()` returns (R-W-4, spec 6.4).

`delegate` blocks and returns a result; `spawn` returns this and the parent carries
on. The two differ in exactly one place that matters to the control layer: a
delegating parent is `blocked_on_child` for the whole child's life, while a spawning
parent stays `running` until it actually awaits `result()`.

The task belongs to the caller's `TaskGroup`, not to this object. That is what makes
`spawn` structured (R-W-7): a node that spawns and forgets still cannot outlive its
node, because the group waits, and a failing child still propagates.

This module is a leaf inside `workflows` -- it names no `AgentLoop` and no
`AgentResult` -- so `agent_loop` and `context` can both import it without a cycle.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from azalabscode.cancellation import CancelReason


@dataclass
class AgentHandle:
    """A concurrently running subagent."""

    agent_id: str
    task: asyncio.Task[Any]
    step: Any = None
    """The `StepHandle` covering the child, so an interrupt can name a reason."""
    spec_name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """True once the child has finished, failed or been cancelled."""

        return self.task.done()

    async def result(self) -> Any:
        """Await the child. Re-raises whatever it raised.

        The caller should flip itself to `blocked_on_child` around this; `NodeContext`
        and `AgentLoop` both do. Awaiting while `running` is not wrong, only slower
        to pause: the run cannot reach PAUSED while any context claims to be busy.
        """

        return await self.task

    def cancel(self, reason: CancelReason = CancelReason.USER_INTERRUPT) -> bool:
        """Cancel the child. Returns False if it had already finished.

        Goes through the `StepHandle` when there is one, so `cancel_reason` is set
        *before* the throw and the child's own absorb rule sees it (R-C-1).
        """

        if self.task.done():
            return False
        if self.step is not None:
            return bool(self.step.request_cancel(reason))
        self.task.cancel()
        return True


__all__ = ["AgentHandle"]
