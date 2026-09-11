"""`ToolingAgentNode`: an agent node that builds its own toolset, and can be interactive.

Both tool-using reference workflows -- the coding agent (R-A-1) and the inspector
(R-A-3) -- need the same two things that `AgentNode` deliberately does not do, so they
share this node rather than each growing its own.

**It builds its own `ToolDispatcher`, at run time.** `AgentNode` takes one, which is
right for a workflow assembled by a script that already has a controller. A workflow
rebuilt by `Controller.load()` has no such caller: spec delta 21 says the graph must
be reconstructible from `(import_path, config)` alone, and a dispatcher needs the
run's `PermissionGate` -- which only exists once a `Controller` does. `RunControl`
exposes `permission_gate` for exactly this, and `run()` is the first moment it can be
read. Building the toolset there also means the workspace root is resolved in the
process that will actually touch the files.

**It can run its loop once per prompt.** An interactive session is not one agent run:
it is one agent, one transcript, and a series of tasks. `AgentLoop.run()` seeds the
transcript only when it is empty, so every prompt after the first arrives as a
*pending injection* -- the same mechanism `escape` uses (R-C-4), drained at the loop's
next turn boundary. That is also what makes `send()` while the agent is working mean
"read this at your next turn" rather than "wait until you are finished".

Three properties fall out of that and are worth stating because breaking them is
quiet:

* Between prompts the node is `blocked_on_child` and its agent has exited, so the run
  is quiescent and `ctrl+p` reaches PAUSED. An idle session that could not be paused
  or saved would be a session that could not be closed.
* A prompt taken off the queue goes through `ctx.checkpoint(park=True)` before the
  loop is re-entered, so a prompt typed into a paused run does not quietly restart it.
* Re-entering the loop re-emits `AgentSpawned`/`AgentFinished` for the same agent id.
  That is honest -- each prompt is a run of the agent -- and `AgentTree` folds the
  repeat rather than drawing a second node.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from azalabscode import (
    AgentLoop,
    AgentPhase,
    AgentSpec,
    HarnessModel,
    Node,
    Provider,
    ToolContext,
    ToolDispatcher,
    ToolSet,
    default_registry,
    render_prompt,
)

WAKE = "wake"
"""Queue sentinel: "look again". The prompt itself is queued as an injection."""


class ToolingAgentState(HarnessModel):
    """The node's own state. The transcript lives in `Session.agents`, not here."""

    started: bool = False
    agent_id: str = ""
    prompts_answered: int = 0
    """How many times the loop has been run. Restored, so a resumed interactive
    session does not claim to be on its first prompt."""


class ToolingAgentNode(Node):
    """One agent, its tools, and optionally a series of prompts."""

    kind: ClassVar[str] = "ToolingAgentNode"
    State: ClassVar[type[HarnessModel]] = ToolingAgentState
    output_type: str = "str"
    dynamic_children: ClassVar[bool] = True
    """Subagents get node ids of the form `<agent_id>/<n>` at run time."""

    def __init__(
        self,
        spec: AgentSpec,
        *,
        provider: Provider,
        workspace: str | Path = ".",
        tool_names: Sequence[str] | None = None,
        exclude_tools: Sequence[str] = (),
        specs: Mapping[str, AgentSpec] | None = None,
        agent_id: str = "main",
        interactive: bool = False,
        max_parallel: int = 10,
        render: Callable[[Any], str] = render_prompt,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.workspace = Path(workspace)
        self.tool_names = list(tool_names) if tool_names is not None else None
        self.exclude_tools = tuple(exclude_tools)
        self.specs = dict(specs or {})
        self.agent_id = agent_id
        self.interactive = interactive
        self.max_parallel = max_parallel
        self.render = render

        self.prompts: asyncio.Queue[str | None] = asyncio.Queue()
        """Wake-ups from the session. `None` ends the session."""
        self.loop: AgentLoop | None = None
        """The live loop, once the node is running. What a CLI reaches for."""
        self.toolset: ToolSet | None = None
        """The tools this node built, for `--print-tools` and for tests."""

    def describe(self) -> str:
        """One line for `Graph.describe()`."""

        return f"ToolingAgentNode({self.agent_id}, {self.spec.model})"

    # -- the toolset --------------------------------------------------------

    def build_toolset(self) -> ToolSet:
        """The built-ins this agent may call.

        `web_search` registers itself only when a search backend exists, so a machine
        with no `SERPER_API_KEY` gets a toolset without it rather than a tool that
        fails on first use. Nothing here may assume it is present.
        """

        toolset = default_registry(include=self.tool_names, exclude=self.exclude_tools)
        self.toolset = toolset
        return toolset

    def build_dispatcher(self, ctx: Any) -> ToolDispatcher:
        """A dispatcher wired to the run's gate, the workspace and the event bus."""

        emitter = ctx.emitter.bind(agent_id=self.agent_id) if ctx.emitter else None
        workspace = self.workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        context = ToolContext(
            workspace_root=workspace,
            emit=emitter.emit if emitter is not None else None,
        )
        return ToolDispatcher(
            self.build_toolset(),
            context=context,
            gate=ctx.control.permission_gate if ctx.control is not None else None,
            emitter=emitter,
            max_parallel=self.max_parallel,
            turn_budget=self.spec.max_tool_results_chars,
        )

    def build_loop(self, ctx: Any) -> AgentLoop:
        """The `AgentLoop` this node runs. Separated so a test can inspect it."""

        emitter = ctx.emitter.bind(agent_id=self.agent_id) if ctx.emitter else None
        return AgentLoop(
            self.spec,
            provider=self.provider,
            dispatcher=self.build_dispatcher(ctx),
            control=ctx.control,
            emitter=emitter,
            agent_id=self.agent_id,  # type: ignore[arg-type]
            node_id=str(ctx.node_id),  # type: ignore[arg-type]
            specs={**ctx.specs, **self.specs},
        )

    # -- the session --------------------------------------------------------

    def wake(self) -> None:
        """Tell the node to look for work. Safe from any task, never blocks."""

        self.prompts.put_nowait(WAKE)

    def close(self) -> None:
        """End the session after the current prompt. Idempotent."""

        self.prompts.put_nowait(None)

    async def run(self, ctx: Any, input: Any) -> Any:
        """Run the loop, once, or once per prompt in an interactive session."""

        loop = self.build_loop(ctx)
        self.loop = loop
        ctx.state.started = True
        ctx.state.agent_id = self.agent_id
        ctx.agent_id = loop.agent_id
        await ctx.checkpoint(park=False)

        task = self.render(input) if input is not None else None
        final = ""
        await ctx.phase(AgentPhase.BLOCKED_ON_CHILD)
        try:
            while True:
                result = await loop.run(task)
                final = result.final_text
                ctx.state.prompts_answered += 1
                if not self.interactive:
                    break
                if not await self._wait_for_prompt(loop):
                    break
                # A prompt typed into a paused run must not restart it, and the
                # transcript it is about to extend should be on disk first.
                await ctx.checkpoint(park=True)
                task = None
        finally:
            await ctx.phase(AgentPhase.RUNNING)
        return final

    async def _wait_for_prompt(self, loop: AgentLoop) -> bool:
        """Block until there is something to answer. False means the session ended.

        The injections are checked *before* the queue and again after every wake-up:
        a message sent while the loop was finishing its last turn is already on the
        agent's queue, and the wake-up that follows it must not be read as a second
        prompt.
        """

        while True:
            if loop.state.pending_injections:
                return True
            signal = await self.prompts.get()
            if signal is None:
                return False


__all__ = ["WAKE", "ToolingAgentNode", "ToolingAgentState"]
