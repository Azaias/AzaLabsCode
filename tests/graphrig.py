"""A rig for driving a real `Controller` over a real `Runner`.

The graph counterpart of `tests/harness.py`. Everything is real except the model:
the same controller, the same runner, the same node types a workflow author uses.

Every wait is bounded. There is no `pytest-timeout` in this project, so an
unbounded wait for a state that never arrives hangs the whole suite with no output
-- which is exactly the failure mode a quiescence bug produces.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from azalabscode import (
    Controller,
    Event,
    EventBus,
    FakeProvider,
    Graph,
    Node,
    PermissionMode,
    RunState,
    SafePointKind,
    ScriptedTurn,
    Subscription,
    Workflow,
)

BOUND = 5.0
"""Seconds any bounded wait here may take before it is called a hang."""


def answers(*texts: str, chunk_delay_s: float = 0.0) -> FakeProvider:
    """A provider with one turn per text. One per branch: see M5 trap 5."""

    return FakeProvider([ScriptedTurn(text=text, chunk_delay_s=chunk_delay_s) for text in texts])


class StepState(BaseModel):
    """Progress through `SteppingNode`, checkpointed one step at a time."""

    done: int = 0
    trail: list[str] = []


class SteppingNode(Node):
    """A node that checkpoints between units of work.

    The thing a `ModelCall` cannot be in a test: interruptible *mid-node*, at a
    point the test controls, with state that proves where it stopped. A resumed run
    restarts it from `state.done`, which is R-W-6's second clause -- "incomplete ones
    restart from their last checkpoint" -- and there is no other way to observe it.
    """

    State: ClassVar[type[BaseModel]] = StepState
    output_type = "list"

    def __init__(self, steps: int = 3, *, label: str = "s", gated: bool = False):
        self.steps = steps
        self.label = label
        self.gated = gated
        self.gate = asyncio.Event()
        """Set by the test to release exactly one step. The node clears it again."""
        self.reached = asyncio.Event()
        """Set by the node when it is blocked at the gate, so the test never races."""
        self.ran: list[int] = []
        """Which step indices this *instance* executed. Empty after a pure resume."""

    async def step(self, *, timeout: float = BOUND) -> None:
        """Let one step through, and wait until the node is blocked again or done."""

        async with asyncio.timeout(timeout):
            await self.reached.wait()
            self.reached.clear()
            self.gate.set()

    async def run(self, ctx: Any, input: Any) -> list[str]:
        while ctx.state.done < self.steps:
            if self.gated:
                self.reached.set()
                await self.gate.wait()
                self.gate.clear()
            self.ran.append(ctx.state.done)
            ctx.state.trail = [*ctx.state.trail, f"{self.label}{ctx.state.done}"]
            ctx.state.done += 1
            await ctx.checkpoint(SafePointKind.CUSTOM)
        return list(ctx.state.trail)


class Boom(RuntimeError):
    """A node failure a test asked for."""


def explode(_: Any) -> Any:
    """A `Func` body that always raises."""

    raise Boom("this node was asked to fail")


@dataclass
class GraphRig:
    """A controller with a graph bound, plus the machinery to observe it."""

    controller: Controller
    workflow: Workflow | Graph
    events: list[Event] = field(default_factory=list)
    _sub: Subscription | None = None
    _pump: asyncio.Task[None] | None = None

    def of_type(self, kind: type[Event]) -> list[Any]:
        """Every collected event of one class."""

        return [event for event in self.events if isinstance(event, kind)]

    async def wait_event(
        self, predicate: Callable[[Event], bool], *, timeout: float = BOUND
    ) -> Event:
        """Wait for an event matching `predicate`. Bounded."""

        async with asyncio.timeout(timeout):
            while True:
                for event in list(self.events):
                    if predicate(event):
                        return event
                await asyncio.sleep(0.005)

    async def wait_type(self, kind: type[Event], *, timeout: float = BOUND) -> Any:
        """Wait for the first event of a class."""

        return await self.wait_event(lambda e: isinstance(e, kind), timeout=timeout)

    async def wait_state(self, *states: RunState, timeout: float = BOUND) -> RunState:
        """Wait for the run to reach one of `states`."""

        return await self.controller.wait_for_state(*states, timeout=timeout)

    async def wait_until(self, predicate: Callable[[], bool], *, timeout: float = BOUND) -> None:
        """Wait for an arbitrary condition. Bounded."""

        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.005)

    async def finish(self, *, timeout: float = BOUND) -> Any:
        """Await the run and return its result."""

        return await self.controller.wait(timeout=timeout)

    def completed_nodes(self) -> list[str]:
        """Node ids with a memoized output, sorted."""

        return sorted(
            node_id for node_id, record in self.controller.nodes.items() if record.completed
        )

    async def aclose(self) -> None:
        """Stop the event pump and close the bus."""

        if self._sub is not None:
            self._sub.unsubscribe()
        if self._pump is not None:
            await self._pump
        self.controller.bus.close()


def build_graph_rig(
    workflow: Workflow | Graph,
    *,
    session_dir: Path | None = None,
    mode: PermissionMode = PermissionMode.AUTO,
    autosave: bool = True,
    run_id: str = "run_graph",
    import_path: str = "",
    config: dict[str, Any] | None = None,
    strict_graph_hash: bool = False,
    bus: EventBus | None = None,
) -> GraphRig:
    """Bind a workflow to a controller and start collecting its events."""

    bus = bus if bus is not None else EventBus()
    controller = Controller(
        run_id=run_id,
        bus=bus,
        permission_mode=mode,
        session_dir=session_dir,
        autosave=autosave,
        strict_graph_hash=strict_graph_hash,
    )
    controller.bind_workflow(workflow, import_path=import_path, config=config)

    rig = GraphRig(controller=controller, workflow=workflow)
    sub = bus.subscribe(name="graphrig")
    rig._sub = sub

    async def pump() -> None:
        async for event in sub:
            rig.events.append(event)

    rig._pump = asyncio.ensure_future(pump())
    return rig


def attach(controller: Controller) -> tuple[list[Event], Subscription, asyncio.Task[None]]:
    """Collect a loaded controller's events without rebuilding a rig."""

    collected: list[Event] = []
    sub = controller.bus.subscribe(name="loaded")

    async def pump() -> None:
        async for event in sub:
            collected.append(event)

    return collected, sub, asyncio.ensure_future(pump())


def texts(provider: FakeProvider) -> Sequence[Any]:
    """Every request a fake provider received. Zero means the node was memoized."""

    return provider.requests
