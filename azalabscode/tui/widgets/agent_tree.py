"""`AgentTree`: the agent tree, live, with the main agent at the root (R-U-3, R-U-5).

Consumes `AgentSpawned`, `AgentFinished` and `AgentPhaseChanged`, which between them
carry everything the tree needs: the edge (`parent_id`), the label (`spec_summary`),
whether the child was delegated or spawned, the live phase, and the outcome.

**Subagents are visually distinct, which is R-U-5's whole content.** Depth does part
of it, but a tree of identical rows is not distinct enough to read at a glance, so a
child is dimmed relative to its parent, a delegated child is marked `↳` against a
spawned child's `⇉`, and the phase is coloured -- running, blocked, parked, waiting
on approval, finished, failed. The agent id is always shown, because it is what the
user types into an interrupt.

**A spawn event can arrive before its parent's.** It should not -- `enter_agent` is
called on the parent first -- but a widget that assumes an arrival order and drops
the event when it is wrong shows a tree that is quietly missing a branch. Orphans
are parked under the root and re-parented if their parent turns up.

Selecting a node posts `AgentTree.AgentSelected`, which is what a layout wires to a
`Transcript` (spec 8.2: "selecting an agent focuses its `Transcript`").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.message import Message
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from azalabscode.events import (
    AgentFinished,
    AgentPhaseChanged,
    AgentSpawned,
    Event,
)
from azalabscode.tui.routing import ANY
from azalabscode.tui.widgets.base import EventWidget

PHASE_STYLE: dict[str, str] = {
    "running": "green",
    "blocked_io": "cyan",
    "blocked_on_child": "blue",
    "waiting_approval": "magenta",
    "parked": "yellow",
    "finished": "dim",
}

OUTCOME_STYLE: dict[str, str] = {
    "completed": "dim",
    "end_turn": "dim",
    "max_turns": "yellow",
    "cancelled": "yellow",
    "failed": "bold red",
}

DELEGATED_MARK = "↳"
SPAWNED_MARK = "⇉"


@dataclass
class AgentInfo:
    """One agent as the tree knows it. Folded from events alone (R-U-1)."""

    agent_id: str
    parent_id: str | None = None
    spec_summary: str = ""
    delegated: bool = False
    phase: str = "running"
    outcome: str | None = None
    result_summary: str = ""
    tokens: int = 0

    @property
    def finished(self) -> bool:
        """Whether the agent has left the run."""

        return self.outcome is not None or self.phase == "finished"

    def label(self, *, root: bool) -> Text:
        """The tree row: mark, agent id, spec, status."""

        text = Text()
        if not root:
            text.append(f"{DELEGATED_MARK if self.delegated else SPAWNED_MARK} ", style="dim")
        text.append(self.agent_id, style="bold" if root else "")
        if self.spec_summary:
            text.append(f"  {self.spec_summary}", style="dim")
        status = self.outcome if self.finished else self.phase
        style = (
            OUTCOME_STYLE.get(status or "", "dim")
            if self.finished
            else PHASE_STYLE.get(self.phase, "")
        )
        text.append(f"  {status}", style=style)
        if self.tokens:
            text.append(f"  {self.tokens} tok", style="dim")
        return text


class AgentTree(EventWidget, Tree[str]):
    """The live agent tree (R-U-3). The node's data is the agent id."""

    DEFAULT_CSS = """
    AgentTree {
        height: 1fr;
    }
    """

    class AgentSelected(Message):
        """An agent row was selected. `agent_id` is empty for the run root."""

        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id
            super().__init__()

    def __init__(
        self,
        *,
        label: str = "run",
        agent_id: str = ANY,
        node_id: str = ANY,
        **kwargs: object,
    ) -> None:
        super().__init__(label, **kwargs)  # type: ignore[arg-type]
        self.agent_filter = agent_id
        self.node_filter = node_id
        self.agents: dict[str, AgentInfo] = {}
        """Every agent seen, by id."""
        self._rows: dict[str, TreeNode[str]] = {}
        self.show_root = True
        self.guide_depth = 3
        self.root.expand()

    # -- events -------------------------------------------------------------

    def handle_event(self, event: Event) -> None:
        """Fold one agent-lifecycle event into the tree."""

        if isinstance(event, AgentSpawned):
            info = self._info(event.agent_id or "")
            if info is None:
                return
            # Both fields are *sticky*, because a second `AgentSpawned` for the same
            # agent can carry less than the first did. A delegated child is registered
            # by its parent (`delegated=True`, `parent_id` set) and then again by its
            # own loop's `_enter`, which knows neither; folding the later event
            # verbatim would move the child to the root and change its mark.
            info.parent_id = event.parent_id or info.parent_id
            info.spec_summary = event.spec_summary or info.spec_summary
            info.delegated = info.delegated or event.delegated
            info.phase = "running"
            # A second `AgentSpawned` for an agent that already finished is a
            # re-entered loop, not a new agent: the coding agent's session runs one
            # `AgentLoop` once per prompt under the same id.
            info.outcome = None
            self._sync(info)
        elif isinstance(event, AgentPhaseChanged):
            info = self._info(event.agent_id or "")
            if info is None:
                return
            info.phase = event.new.value
            self._sync(info)
        elif isinstance(event, AgentFinished):
            info = self._info(event.agent_id or "")
            if info is None:
                return
            info.phase = "finished"
            info.outcome = event.outcome
            info.result_summary = event.result_summary
            if event.usage is not None:
                info.tokens = event.usage.total_tokens
            self._sync(info)

    def seed(self, agents: Mapping[str, Any]) -> None:
        """Draw a tree from `Controller.agents`, for a run that came back from disk.

        A loaded run has agents and no events for them (they belong to the process
        that wrote the session), so a tree built from the stream alone is empty until
        the run does something new.
        """

        for agent_id, state in agents.items():
            info = self._info(agent_id)
            if info is None:
                continue
            info.parent_id = getattr(state, "parent_id", None)
            info.spec_summary = getattr(state, "spec_name", "") or info.spec_summary
            phase = getattr(state, "phase", None)
            info.phase = getattr(phase, "value", str(phase or "running"))
            outcome = getattr(state, "outcome", None)
            info.outcome = outcome if info.phase == "finished" else None
            usage = getattr(state, "usage", None)
            info.tokens = getattr(usage, "total_tokens", 0) or 0
            self._sync(info)

    def clear_agents(self) -> None:
        """Drop every agent. Used when the app attaches to another run."""

        self.agents.clear()
        self._rows.clear()
        self.root.remove_children()

    # -- the tree -----------------------------------------------------------

    def _info(self, agent_id: str) -> AgentInfo | None:
        if not agent_id:
            return None
        info = self.agents.get(agent_id)
        if info is None:
            info = AgentInfo(agent_id=agent_id)
            self.agents[agent_id] = info
        return info

    def _sync(self, info: AgentInfo) -> None:
        """Create or move the node for `info`, then relabel it."""

        node = self._rows.get(info.agent_id)
        parent = self._parent_node(info)
        if node is None:
            node = parent.add(info.label(root=info.parent_id is None), data=info.agent_id)
            node.expand()
            self._rows[info.agent_id] = node
            self._adopt_orphans(info.agent_id)
        elif node.parent is not parent:
            # Re-parenting: the spawn arrived before its parent did. Textual has no
            # move, so the subtree is rebuilt under the right parent.
            self._reparent(info, parent)
            node = self._rows[info.agent_id]
        node.set_label(info.label(root=info.parent_id is None))

    def _parent_node(self, info: AgentInfo) -> TreeNode[str]:
        if info.parent_id and info.parent_id in self._rows:
            return self._rows[info.parent_id]
        return self.root

    def _reparent(self, info: AgentInfo, parent: TreeNode[str]) -> None:
        old = self._rows.pop(info.agent_id)
        children = [str(child.data) for child in old.children if child.data]
        old.remove()
        node = parent.add(info.label(root=info.parent_id is None), data=info.agent_id)
        node.expand()
        self._rows[info.agent_id] = node
        for child_id in children:
            child = self.agents.get(child_id)
            if child is not None:
                self._rows.pop(child_id, None)
                self._sync(child)

    def _adopt_orphans(self, parent_id: str) -> None:
        """Move any agent that named `parent_id` before its node existed."""

        for info in list(self.agents.values()):
            if info.parent_id == parent_id and info.agent_id in self._rows:
                node = self._rows[info.agent_id]
                if node.parent is self.root:
                    self._reparent(info, self._rows[parent_id])

    # -- selection ----------------------------------------------------------

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        """Publish the selected agent id (spec 8.2: focus that agent's transcript)."""

        event.stop()
        self.post_message(self.AgentSelected(str(event.node.data or "")))

    @property
    def selected_agent(self) -> str | None:
        """The agent under the cursor, or `None` at the root."""

        node = self.cursor_node
        return str(node.data) if node is not None and node.data else None

    def children_of(self, agent_id: str) -> list[str]:
        """Agent ids whose parent is `agent_id`. For tests and for `--headless`."""

        return [info.agent_id for info in self.agents.values() if info.parent_id == agent_id]


__all__ = ["OUTCOME_STYLE", "PHASE_STYLE", "AgentInfo", "AgentTree"]
