"""Tool observability: one agent, every tool, and a live view of every call (spec 9.3).

The graph is the smallest one in the repo -- a single `ToolingAgentNode` -- because
R-A-3 is not about the graph. It is about what the *UI* can show of one agent's tool
use: a table of every call with status and duration, a drill-down into params, result,
error and timing, and the raw event sequence behind it.

That makes this workflow the honest test of two claims the rest of the system makes.
R-X-3 says every tool call, approval and result emits a typed event and that no core
behaviour is observable only through logs; the inspector renders nothing but events,
so anything missing from the stream is missing from the screen. Spec delta 12 says a
result carries three renderings -- `content` for the model, `display` for widgets,
`meta` for telemetry; the drill-down shows the second and the third, and never parses
the first.

`build(config)` is the whole contract (R-W-1): the session stores
`("workflows.inspector.workflow:build", config)` and rebuilds the graph from it. The
provider is constructed in here for the same reason fusion's is (spec delta 21).
"""

from __future__ import annotations

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
from workflows.tooling import ToolingAgentNode

SYSTEM_PROMPT = """You are answering a question about a codebase or a directory, and
every step you take is being watched in a tool-call inspector.

Use the tools rather than guessing. Prefer `glob` and `grep` to find things and
`read_file` to confirm them; read before you conclude. Say what you found and where,
with paths and line numbers, and say plainly when the answer is that something is not
there."""

NODE_ID = "agent"
"""The node's name. Not `main`: that is the agent id, and a node id equal to an agent
id collides in the quiescence map (`Workflow.compile()` refuses it)."""

AGENT_ID = "main"

DEFAULT_TOOLS = ["read_file", "glob", "grep", "shell", "web_fetch", "web_search"]
"""Read-mostly by default. The inspector is for watching tool use, not for editing;
`write_file`, `edit_file` and `delegate` are available by naming them in `tools`."""


class InspectorConfig(BaseModel):
    """Everything `build` needs, and everything the session stores (R-W-1)."""

    model_config = {"extra": "forbid"}

    task: str = ""
    """What to ask the agent. Part of the config so a resumed run asks the same thing."""
    model: str = "anthropic/claude-haiku-4.5"
    workspace: str = "."
    tools: list[str] | None = Field(default=None)
    """Tool names, or `None` for `DEFAULT_TOOLS`. `[]` means no tools at all."""
    system_prompt: str = SYSTEM_PROMPT
    max_turns: int = 30
    max_parallel: int = 10
    temperature: float | None = None
    max_tokens: int | None = None

    script: dict[str, Any] | None = None
    """A `Script` document, inline. When set, no network: the agent runs against a
    `FakeProvider`. Inline rather than a path so the config stays self-contained and a
    session written by one process rebuilds in another (spec delta 21)."""

    def tool_names(self) -> list[str]:
        """The toolset this run asks for."""

        return list(DEFAULT_TOOLS if self.tools is None else self.tools)

    def spec(self) -> AgentSpec:
        """The agent, as the session will store it."""

        return AgentSpec(
            name=AGENT_ID,
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tool_names(),
            max_turns=self.max_turns,
            parallel_tool_calls=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


def provider_for(cfg: InspectorConfig) -> Provider:
    """The provider, constructed inside `build` (spec delta 21)."""

    if cfg.script is not None:
        return FakeProvider(Script.model_validate(cfg.script))
    return OpenRouterProvider()


def build(config: InspectorConfig | dict[str, Any]) -> Workflow:
    """The importable factory (R-W-1). One agent with every tool it was given."""

    cfg = config if isinstance(config, InspectorConfig) else InspectorConfig.model_validate(config)
    wf = Workflow("inspector", input=cfg.task)
    node = ToolingAgentNode(
        cfg.spec(),
        provider=provider_for(cfg),
        workspace=cfg.workspace,
        tool_names=cfg.tool_names(),
        agent_id=AGENT_ID,
        interactive=False,
        max_parallel=cfg.max_parallel,
    )
    wf.output(wf.node(NODE_ID, node))
    return wf


IMPORT_PATH = "workflows.inspector.workflow:build"
CONFIG_TYPE = "workflows.inspector.workflow:InspectorConfig"

__all__ = [
    "AGENT_ID",
    "CONFIG_TYPE",
    "DEFAULT_TOOLS",
    "IMPORT_PATH",
    "NODE_ID",
    "SYSTEM_PROMPT",
    "InspectorConfig",
    "build",
    "provider_for",
]
