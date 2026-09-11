"""The coding agent (spec 9.1, R-A-1): one agent, every built-in tool, subagents.

The graph is one node. Spec 9.1 asks for `AgentNode(AgentSpec(model=..., tools=all
builtin + delegate, allow_delegate=True, parallel_tool_calls=True))` with the
workspace root at the working directory, and that is what this is -- built on
`ToolingAgentNode` rather than `AgentNode` because a daily-use coding agent is an
interactive *session*, not a single task, and because the dispatcher has to be built
from the run's gate at run time (see `workflows/tooling.py`).

**The node is called `loop` and the agent is called `main`.** They are separate
namespaces sharing one quiescence map, and `Workflow.compile()` refuses a graph where
they collide -- an agent whose `exit_agent` deleted the node's entry would leave the
node running with nothing registered. `main` is also the agent id spec C-4's
targetless interrupt looks for, which is why the agent keeps it.

**The three subagent specs are spec 9.1's.** `explore` and `review` are read-only and
cheap; `edit` has the full toolset and is only useful in `auto` mode, which is
⚠C-6 and is why it is not in `subagents` by default -- a delegated agent in `manual`
mode has its write tools filtered out by the gate (R-C-7), so an `edit` subagent under
`manual` would be handed a task it cannot do. Name it in `subagents` when running in
`auto`.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from azalabscode import (
    AgentSpec,
    FakeProvider,
    OpenRouterProvider,
    Provider,
    Script,
    Workflow,
)
from workflows.coding_agent.prompts import (
    EDIT_PROMPT,
    EXPLORE_PROMPT,
    REVIEW_PROMPT,
    SYSTEM_PROMPT,
)
from workflows.tooling import ToolingAgentNode

NODE_ID = "loop"
"""The node's name. Not `main`: that is the agent id, and the two share a namespace."""

AGENT_ID = "main"

ALL_TOOLS = [
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "shell",
    "web_fetch",
    "web_search",
    "delegate",
]
"""Spec 9.1's "all built-in tools + delegate". `web_search` drops out of the built
toolset when there is no search backend; nothing here may assume it is present."""

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
DEFAULT_CHEAP_MODEL = "anthropic/claude-haiku-4.5"


class CodingAgentConfig(BaseModel):
    """Everything `build` needs, and everything the session stores (R-W-1)."""

    model_config = {"extra": "forbid"}

    task: str = ""
    """The first prompt. Later ones arrive as injections and are not config."""
    model: str = DEFAULT_MODEL
    subagent_model: str = DEFAULT_CHEAP_MODEL
    workspace: str = "."
    """Root the tools may touch (R-T-7). Resolved in the process that runs, not here."""
    tools: list[str] | None = None
    system_prompt: str = SYSTEM_PROMPT
    subagents: list[str] = Field(default_factory=lambda: ["explore", "review"])
    """Which specs `delegate` may name. `edit` is opt-in: see ⚠C-6."""
    interactive: bool = True
    """Whether the session waits for another prompt after each answer."""
    max_turns: int = 60
    max_parallel: int = 10
    max_tokens: int | None = None
    temperature: float | None = None

    script: dict[str, Any] | None = None
    """A `Script` document, inline. When set, no network and no key: the agent runs
    against a `FakeProvider`. Inline rather than a path so a session written by one
    process rebuilds in another (spec delta 21)."""

    def tool_names(self) -> list[str]:
        """The toolset this run asks for."""

        return list(ALL_TOOLS if self.tools is None else self.tools)

    def workspace_path(self) -> str:
        """The workspace, defaulting to the working directory (spec 9.1)."""

        return self.workspace or os.getcwd()

    def spec(self) -> AgentSpec:
        """The main agent."""

        return AgentSpec(
            name=AGENT_ID,
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tool_names(),
            max_turns=self.max_turns,
            parallel_tool_calls=True,
            allow_delegate="delegate" in self.tool_names(),
            subagents=list(self.subagents),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def subagent_specs(self) -> dict[str, AgentSpec]:
        """Spec 9.1's three: `explore`, `review`, `edit`."""

        read_only = ["read_file", "glob", "grep", "web_fetch", "web_search"]
        return {
            "explore": AgentSpec(
                name="explore",
                model=self.subagent_model,
                system_prompt=EXPLORE_PROMPT,
                tools=read_only,
                max_turns=20,
                parallel_tool_calls=True,
            ),
            "review": AgentSpec(
                name="review",
                model=self.subagent_model,
                system_prompt=REVIEW_PROMPT,
                tools=read_only,
                max_turns=20,
                parallel_tool_calls=True,
            ),
            "edit": AgentSpec(
                name="edit",
                model=self.model,
                system_prompt=EDIT_PROMPT,
                tools=["read_file", "write_file", "edit_file", "glob", "grep"],
                max_turns=30,
                parallel_tool_calls=True,
            ),
        }


def provider_for(cfg: CodingAgentConfig) -> Provider:
    """The provider, constructed inside `build` (spec delta 21)."""

    if cfg.script is not None:
        return FakeProvider(Script.model_validate(cfg.script))
    return OpenRouterProvider()


def build(config: CodingAgentConfig | dict[str, Any]) -> Workflow:
    """The importable factory (R-W-1). One interactive agent in a working directory."""

    cfg = (
        config
        if isinstance(config, CodingAgentConfig)
        else CodingAgentConfig.model_validate(config)
    )
    wf = Workflow("coding_agent", input=cfg.task, specs=cfg.subagent_specs())
    node = ToolingAgentNode(
        cfg.spec(),
        provider=provider_for(cfg),
        workspace=cfg.workspace_path(),
        tool_names=cfg.tool_names(),
        specs=cfg.subagent_specs(),
        agent_id=AGENT_ID,
        interactive=cfg.interactive,
        max_parallel=cfg.max_parallel,
    )
    wf.output(wf.node(NODE_ID, node))
    return wf


def agent_node(workflow: Any) -> ToolingAgentNode:
    """The `ToolingAgentNode` inside a built or loaded coding-agent graph.

    A CLI needs it to send prompts, and after `Controller.load()` the node object is
    the *rebuilt* one -- the controller made it from `(import_path, config)`, so the
    only way to reach it is through the graph it belongs to.
    """

    graph = getattr(workflow, "graph", workflow)
    node = graph.entry(NODE_ID).node
    if not isinstance(node, ToolingAgentNode):  # pragma: no cover - a graph that is not ours
        raise TypeError(f"node {NODE_ID!r} is a {type(node).__name__}, not a ToolingAgentNode")
    return node


IMPORT_PATH = "workflows.coding_agent.workflow:build"
CONFIG_TYPE = "workflows.coding_agent.workflow:CodingAgentConfig"

__all__ = [
    "AGENT_ID",
    "ALL_TOOLS",
    "CONFIG_TYPE",
    "DEFAULT_CHEAP_MODEL",
    "DEFAULT_MODEL",
    "IMPORT_PATH",
    "NODE_ID",
    "CodingAgentConfig",
    "agent_node",
    "build",
    "provider_for",
]
