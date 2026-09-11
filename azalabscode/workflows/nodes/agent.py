"""`AgentNode`: the model-to-tool loop as one graph node (R-W-3, R-W-8).

The cycle lives inside the node, which is exactly what R-W-2 asks for: the graph
stays a DAG, and the node is the checkpoint boundary. Everything underneath is
`AgentLoop`, unchanged -- this class is the adapter between a node's `(ctx, input)`
and an agent's `(task)`, plus the two pieces of bookkeeping the graph needs.

**The agent id defaults to `spec.name`, not to the node id.** Both are strings in
the same quiescence map, so making them equal would have the agent's `enter_agent`
overwrite the node's own entry and its `exit_agent` delete it, leaving the node
running with nothing registered. Defaulting to `spec.name` also gives the coding
agent `main` for free, which is the id spec C-4's targetless interrupt looks for.
`Workflow.compile()` refuses a graph where an agent id collides with a node id.

**The node is `blocked_on_child` while the loop runs.** The agent registers itself
and takes its own safe points; a node claiming to be `running` for the whole life of
its agent would mean the run could never reach PAUSED.

Output is the agent's final text by default. The full `AgentResult` is available
with `output="result"`, and either way the transcript, the usage and the outcome are
already in the session under the agent's id -- nothing is lost by the simpler shape,
and the simpler shape composes with `ModelCall` without an accessor in between.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal

from azalabscode.ids import AgentId, NodeId
from azalabscode.providers.base import Provider
from azalabscode.runstate import AgentPhase
from azalabscode.schema import HarnessModel
from azalabscode.tools.dispatcher import ToolDispatcher
from azalabscode.workflows.agent_loop import AgentLoop, AgentSpec
from azalabscode.workflows.node import Node
from azalabscode.workflows.nodes.model_call import render_prompt


class AgentNodeState(HarnessModel):
    """The node's own state. The agent's lives in `Session.agents`, not here."""

    started: bool = False
    agent_id: str = ""


class AgentNode(Node):
    """Run an agent loop to completion."""

    kind: ClassVar[str] = "AgentNode"
    State: ClassVar[type[HarnessModel]] = AgentNodeState
    output_type: str = "str"
    dynamic_children: ClassVar[bool] = True
    """Subagents get node ids of the form `<agent_id>/<n>` at runtime."""

    def __init__(
        self,
        spec: AgentSpec,
        *,
        provider: Provider | None = None,
        dispatcher: ToolDispatcher | None = None,
        agent_id: str | None = None,
        specs: Mapping[str, AgentSpec] | None = None,
        render: Callable[[Any], str] = render_prompt,
        output: Literal["text", "result"] = "text",
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.dispatcher = dispatcher
        self.agent_id = agent_id or spec.name
        self.specs = dict(specs or {})
        self.render = render
        self.output = output
        if output == "result":
            self.output_type = "AgentResult"

    def describe(self) -> str:
        return f"AgentNode({self.agent_id}, {self.spec.model})"

    def build_loop(self, ctx: Any) -> AgentLoop:
        """The `AgentLoop` this node runs. Separated so a test can inspect it."""

        provider = self.provider or ctx.provider
        dispatcher = self.dispatcher or ctx.tools
        if provider is None or dispatcher is None:
            raise RuntimeError(
                f"node {ctx.node_id} needs a provider and a ToolDispatcher: pass them to "
                "AgentNode(...) or give the Workflow defaults"
            )
        agent_id = AgentId(self.agent_id)
        emitter = ctx.emitter.bind(agent_id=str(agent_id)) if ctx.emitter else None
        return AgentLoop(
            self.spec,
            provider=provider,
            dispatcher=dispatcher,
            control=ctx.control,
            emitter=emitter,
            agent_id=agent_id,
            node_id=NodeId(str(ctx.node_id)),
            specs={**ctx.specs, **self.specs},
        )

    async def run(self, ctx: Any, input: Any) -> Any:
        """Run the loop. The node is quiescent throughout; the agent is not."""

        loop = self.build_loop(ctx)
        ctx.state.started = True
        ctx.state.agent_id = str(loop.agent_id)
        ctx.agent_id = loop.agent_id
        await ctx.checkpoint(park=False)

        await ctx.phase(AgentPhase.BLOCKED_ON_CHILD)
        try:
            result = await loop.run(self.render(input) if input is not None else None)
        finally:
            await ctx.phase(AgentPhase.RUNNING)

        if self.output == "result":
            return result.model_dump(mode="json")
        return result.final_text


__all__ = ["AgentNode", "AgentNodeState"]
