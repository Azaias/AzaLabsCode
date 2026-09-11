"""A rig for driving a real `Controller` + `AgentLoop` under `FakeProvider`.

Shared by the control scenarios, the quiescence property test and the delegate
regression. Everything here is real except the model and the tools: the same
controller, the same gate, the same dispatcher and the same loop the coding agent
will use at M6.

Every wait is bounded. There is no `pytest-timeout` in this project, so an
unbounded wait for a state that never arrives hangs the whole suite with no
output -- which is exactly the failure mode a quiescence bug produces.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from azalabscode.contracts import ApprovalHandler
from azalabscode.control import Controller
from azalabscode.events import Event, EventBus, Subscription
from azalabscode.ids import MAIN_AGENT, AgentId
from azalabscode.permissions import ApprovalPolicy, PermissionMode
from azalabscode.providers.testing import (
    FakeProvider,
    ScriptedToolCall,
    ScriptedTurn,
)
from azalabscode.runstate import RunState
from azalabscode.toolio import NO_RETRY, RetryPolicy, ToolResult
from azalabscode.tools.base import Tool
from azalabscode.tools.context import ToolContext
from azalabscode.tools.dispatcher import ToolDispatcher
from azalabscode.workflows.agent_loop import AgentLoop, AgentResult, AgentSpec

BOUND = 5.0
"""Seconds any bounded wait in a test may take before it is called a hang."""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class EchoParams(BaseModel):
    """Parameters for the fake tools."""

    model_config = {"extra": "forbid"}

    value: str = Field(default="ok", description="Text to echo back.")
    sleep: float = Field(default=0.0, ge=0.0, description="Seconds to take.")


class RecordingTool(Tool):
    """A tool that records every start, finish and cancellation.

    The counters are what the R-C-13 scenario asserts against: an interrupted call
    must never be executed a second time, and "never" is only checkable if the tool
    counts.
    """

    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "Echo a value back. Test double."
    Params: ClassVar[type[BaseModel]] = EchoParams

    approval: ApprovalPolicy = "never"
    timeout: float = 30.0
    retry: RetryPolicy = NO_RETRY
    concurrency_safe: ClassVar[bool] = True
    read_only: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self.started: list[str] = []
        self.finished: list[str] = []
        self.cancelled: list[str] = []

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Sleep, then echo."""

        assert isinstance(params, EchoParams)
        self.started.append(params.value)
        if params.sleep:
            await asyncio.sleep(params.sleep)
        self.finished.append(params.value)
        return ToolResult.ok_text(f"echo: {params.value}")

    async def on_cancel(self, params: BaseModel, ctx: ToolContext, reason: str) -> None:
        """Record the cleanup the dispatcher runs detached."""

        assert isinstance(params, EchoParams)
        self.cancelled.append(params.value)


class TouchTool(RecordingTool):
    """The destructive one: `approval="always"`, and not concurrency-safe."""

    name: ClassVar[str] = "touch"
    description: ClassVar[str] = "Pretend to modify a file. Test double."

    approval: ApprovalPolicy = "always"
    concurrency_safe: ClassVar[bool] = False
    read_only: ClassVar[bool] = False


# ---------------------------------------------------------------------------
# Script helpers
# ---------------------------------------------------------------------------


def says(text: str, **kwargs: Any) -> ScriptedTurn:
    """A turn that answers with text and stops."""

    return ScriptedTurn(text=text, **kwargs)


_CALL_SEQ = itertools.count()


def calls(*specs: tuple[str, dict[str, Any]], text: str = "", **kwargs: Any) -> ScriptedTurn:
    """A turn that asks for tool calls, in the order given.

    Call ids are globally unique, not per-turn: a transcript containing two turns
    that both used `call_0` violates the invariant before the loop has done anything
    wrong, and the resulting failure points at the wrong thing.
    """

    return ScriptedTurn(
        text=text,
        tool_calls=[
            ScriptedToolCall(call_id=f"call_{next(_CALL_SEQ)}", name=name, arguments=args)
            for name, args in specs
        ],
        finish_reason="tool_calls",
        **kwargs,
    )


def echo_call(value: str, sleep: float = 0.0) -> tuple[str, dict[str, Any]]:
    """One `echo` call."""

    return ("echo", {"value": value, "sleep": sleep})


def touch_call(value: str, sleep: float = 0.0) -> tuple[str, dict[str, Any]]:
    """One `touch` call -- the approval-gated one."""

    return ("touch", {"value": value, "sleep": sleep})


# ---------------------------------------------------------------------------
# The rig
# ---------------------------------------------------------------------------


@dataclass
class Rig:
    """A controller, an agent loop and the machinery to observe both."""

    controller: Controller
    provider: FakeProvider
    dispatcher: ToolDispatcher
    echo: RecordingTool
    touch: TouchTool
    events: list[Event] = field(default_factory=list)
    loops: dict[str, AgentLoop] = field(default_factory=dict)
    result: AgentResult | None = None
    _sub: Subscription | None = None
    _pump: asyncio.Task[None] | None = None

    # -- events -------------------------------------------------------------

    def of_type(self, kind: type[Event]) -> list[Any]:
        """Every collected event of one class."""

        return [e for e in self.events if isinstance(e, kind)]

    async def wait_event(
        self,
        predicate: Callable[[Event], bool],
        *,
        timeout: float = BOUND,
    ) -> Event:
        """Wait for an event matching `predicate`. Bounded.

        Polls rather than hooking the bus, because the point is to observe the
        stream a real subscriber sees, including anything published before the wait
        started.
        """

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

    # -- lifecycle ----------------------------------------------------------

    @property
    def main(self) -> AgentLoop | None:
        """The main agent's loop, once the body has built it."""

        return self.loops.get(str(MAIN_AGENT))

    @property
    def transcript(self) -> list[Any]:
        """The main agent's messages."""

        state = self.controller.agent(MAIN_AGENT)
        return list(state.messages) if state is not None else []

    async def finish(self, *, timeout: float = BOUND) -> Any:
        """Await the run and return its result."""

        return await self.controller.wait(timeout=timeout)

    async def aclose(self) -> None:
        """Stop the event pump and close the bus."""

        if self._sub is not None:
            self._sub.unsubscribe()
        if self._pump is not None:
            await self._pump
        self.controller.bus.close()


def build_rig(
    turns: Sequence[ScriptedTurn],
    *,
    workspace: Path,
    task: str = "do the thing",
    mode: PermissionMode = PermissionMode.AUTO,
    handler: ApprovalHandler | None = None,
    spec: AgentSpec | None = None,
    specs: Mapping[str, AgentSpec] | None = None,
    tools: Iterable[Tool] | None = None,
    extra_tools: Iterable[Tool] | None = None,
    max_parallel: int = 10,
    session_dir: Path | None = None,
    autosave: bool = True,
    workflow: Any = None,
) -> Rig:
    """Wire a controller, a fake provider, two fake tools and an agent loop together.

    The run body builds the `AgentLoop` from the `RunControl` it is handed, which is
    the shape M5's `Runner` will have: the loop never sees a `Controller`, only the
    protocol.
    """

    echo = RecordingTool()
    touch = TouchTool()
    toolset = list(tools) if tools is not None else [echo, touch]
    toolset.extend(extra_tools or [])
    provider = FakeProvider(list(turns))
    bus = EventBus()

    agent_spec = spec or AgentSpec(name="main", model="fake/model", system_prompt="be useful")

    controller = Controller(
        run_id="run_test",
        bus=bus,
        permission_mode=mode,
        approval_handler=handler,
        gated_tools={t.name for t in toolset if t.approval != "never"},
        session_dir=session_dir,
        autosave=autosave,
        workflow=workflow,
    )
    dispatcher = ToolDispatcher(
        toolset,
        context=ToolContext(workspace_root=workspace),
        gate=controller.gate,
        emitter=bus.emitter(),
        max_parallel=max_parallel,
    )
    rig = Rig(
        controller=controller, provider=provider, dispatcher=dispatcher, echo=echo, touch=touch
    )

    async def body(control: Any) -> AgentResult:
        loop = AgentLoop(
            agent_spec,
            provider=provider,
            dispatcher=dispatcher,
            control=control,
            emitter=control.emitter_for(MAIN_AGENT),
            agent_id=MAIN_AGENT,
            specs=specs,
        )
        rig.loops[str(MAIN_AGENT)] = loop
        rig.result = await loop.run(task)
        return rig.result

    controller.set_body(body)

    sub = bus.subscribe(name="rig")
    rig._sub = sub

    async def pump() -> None:
        async for event in sub:
            rig.events.append(event)

    rig._pump = asyncio.ensure_future(pump())
    return rig


def child_loop(rig: Rig, agent_id: str) -> AgentLoop | None:
    """A subagent's loop, by id, once it exists."""

    main = rig.main
    if main is None:
        return None
    for child in main.children:
        if str(child.agent_id) == agent_id:
            return child
    return None


def agent_ids(rig: Rig) -> list[str]:
    """Every agent the controller knows about."""

    return sorted(rig.controller.agents)


def only_agent(rig: Rig, agent_id: AgentId | str = MAIN_AGENT) -> Any:
    """One agent's state, asserted to exist."""

    state = rig.controller.agent(agent_id)
    assert state is not None, f"no agent {agent_id!r}"
    return state
