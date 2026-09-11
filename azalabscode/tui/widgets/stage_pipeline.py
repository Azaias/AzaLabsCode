"""`StagePipeline`: the graph's stages as a row of chevrons (R-U-3).

Consumes `NodeStarted`, `NodeCompleted` and `NodeFailed`. Fusion's is
`models ▶ analyze ▶ synthesize`; any graph's is whatever its top-level nodes are.

Three things this widget has to get right, all of them consequences of how M5's
runner emits node events:

* **A branch is not a stage.** `NodeStarted` fires for every node at every level,
  including `models/gpt-4o` and `each/3`. Rolling those into their parent is what
  makes a four-model fan-out one chevron reading `models 4/4` instead of five
  chevrons, and the roll-up is by node-id prefix because that is exactly what the
  runner's ids encode.
* **The runner has a node id of its own.** `@fusion` shows up on `Checkpoint`
  events and is not a graph node. Anything starting with `@` is skipped.
* **Order is the graph's, not the stream's.** A pipeline that ordered its chevrons
  by arrival would reorder itself when a later stage happened to emit first. Pass
  `stages=` from the workflow when the order matters; without it the widget falls
  back to arrival order, which is right for a linear graph and is all it can know.

Clicking a chevron posts `StagePipeline.StageSelected`, which is what fusion's app
wires to "show this stage's pane" (spec 8.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Static

from azalabscode.events import (
    Event,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
)
from azalabscode.tui.routing import ANY
from azalabscode.tui.widgets.base import EventWidget

RUNNER_PREFIX = "@"
"""The runner's own quiescence key prefix. Not a graph node."""

SEPARATOR = "▸"

STATE_STYLE: dict[str, str] = {
    "pending": "dim",
    "running": "bold cyan",
    "ok": "bold green",
    "failed": "bold red",
}

STATE_MARK: dict[str, str] = {
    "pending": "○",
    "running": "◐",
    "ok": "●",
    "failed": "✗",
}


def top_level(node_id: str) -> str:
    """The stage a node id belongs to: everything before the first `/`."""

    return node_id.split("/", 1)[0]


@dataclass
class StageState:
    """One stage, folded from the node events of it and its children."""

    node_id: str
    label: str = ""
    state: str = "pending"
    node_class: str = ""
    children_started: int = 0
    children_done: int = 0
    children_failed: int = 0
    duration_ms: float = 0.0
    error: str = ""
    order: int = 0
    child_ids: set[str] = field(default_factory=set)

    def render(self) -> Text:
        """The chevron: mark, label, and the child counter when there is one."""

        text = Text()
        style = STATE_STYLE.get(self.state, "")
        text.append(f"{STATE_MARK.get(self.state, '○')} ", style=style)
        text.append(self.label or self.node_id, style=style)
        if self.child_ids:
            done = self.children_done + self.children_failed
            text.append(f" {done}/{len(self.child_ids)}", style="dim")
        if self.state == "ok" and self.duration_ms:
            text.append(f" {self.duration_ms / 1000:.1f}s", style="dim")
        if self.state == "failed" and self.error:
            text.append(f" {self.error[:40]}", style="red")
        return text


class StageChip(Static):
    """One chevron. A widget rather than a span so that clicking one is possible."""

    DEFAULT_CSS = """
    StageChip {
        width: auto;
        padding: 0 1;
    }
    StageChip.-running {
        background: $panel;
    }
    """

    def __init__(self, stage: StageState) -> None:
        super().__init__(stage.render(), markup=False)
        self.stage = stage

    def redraw(self) -> None:
        """Re-render from the stage state."""

        self.update(self.stage.render())
        self.set_class(self.stage.state == "running", "-running")

    def on_click(self) -> None:
        """Ask the pipeline to publish this stage."""

        self.post_message(StagePipeline.StageSelected(self.stage.node_id))


class StagePipeline(EventWidget, Horizontal):
    """A horizontal row of stage chevrons with live state (R-U-3)."""

    DEFAULT_CSS = """
    StagePipeline {
        height: 1;
        width: 1fr;
    }
    StagePipeline > Static.-sep {
        width: auto;
        color: $text-muted;
    }
    """

    class StageSelected(Message):
        """A chevron was clicked, or `select()` was called."""

        def __init__(self, node_id: str) -> None:
            self.node_id = node_id
            super().__init__()

    def __init__(
        self,
        stages: Sequence[str] | Sequence[tuple[str, str]] = (),
        *,
        agent_id: str = ANY,
        node_id: str = ANY,
        discover: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.agent_filter = agent_id
        self.node_filter = node_id
        self.discover = discover
        """Whether a stage not in `stages` may be added when its node starts."""
        self.stages: dict[str, StageState] = {}
        self._chips: dict[str, StageChip] = {}
        for index, entry in enumerate(stages):
            stage_id, label = entry if isinstance(entry, tuple) else (entry, entry)
            self.stages[stage_id] = StageState(node_id=stage_id, label=label, order=index)

    def compose(self) -> ComposeResult:
        """The declared stages. Discovered ones are mounted as they arrive."""

        for index, stage in enumerate(self._ordered()):
            if index:
                yield Static(f" {SEPARATOR} ", classes="-sep")
            chip = StageChip(stage)
            self._chips[stage.node_id] = chip
            yield chip

    def _ordered(self) -> list[StageState]:
        return sorted(self.stages.values(), key=lambda stage: stage.order)

    # -- events -------------------------------------------------------------

    def handle_event(self, event: Event) -> None:
        """Fold one node event into its stage."""

        if not isinstance(event, NodeStarted | NodeCompleted | NodeFailed):
            return
        node_id = event.node_id or ""
        if not node_id or node_id.startswith(RUNNER_PREFIX):
            return
        stage_id = top_level(node_id)
        stage = self.stages.get(stage_id)
        if stage is None:
            if not self.discover:
                return
            stage = StageState(node_id=stage_id, label=stage_id, order=len(self.stages))
            self.stages[stage_id] = stage
            self._mount_chip(stage)

        if node_id != stage_id:
            self._fold_child(stage, node_id, event)
        else:
            self._fold_stage(stage, event)
        chip = self._chips.get(stage_id)
        if chip is not None:
            chip.redraw()

    def _fold_stage(self, stage: StageState, event: Event) -> None:
        if isinstance(event, NodeStarted):
            stage.state = "running"
            stage.node_class = event.node_class
        elif isinstance(event, NodeCompleted):
            stage.state = "failed" if stage.children_failed else "ok"
            stage.duration_ms = event.duration_ms
        elif isinstance(event, NodeFailed):
            stage.state = "failed"
            stage.error = event.error

    def _fold_child(self, stage: StageState, node_id: str, event: Event) -> None:
        stage.child_ids.add(node_id)
        if isinstance(event, NodeStarted):
            stage.children_started += 1
            if stage.state == "pending":
                stage.state = "running"
        elif isinstance(event, NodeCompleted):
            stage.children_done += 1
        elif isinstance(event, NodeFailed):
            stage.children_failed += 1

    def _mount_chip(self, stage: StageState) -> None:
        chip = StageChip(stage)
        self._chips[stage.node_id] = chip
        if self.is_mounted:
            if len(self._chips) > 1:
                self.mount(Static(f" {SEPARATOR} ", classes="-sep"))
            self.mount(chip)

    # -- selection ----------------------------------------------------------

    def select(self, node_id: str) -> None:
        """Publish a stage as if it had been clicked."""

        self.post_message(self.StageSelected(node_id))

    def state_of(self, stage_id: str) -> str:
        """One stage's state: `pending`, `running`, `ok` or `failed`."""

        stage = self.stages.get(stage_id)
        return stage.state if stage is not None else "pending"

    def render_line_text(self) -> str:
        """The whole pipeline as plain text, for tests and `--headless` output."""

        return f" {SEPARATOR} ".join(stage.render().plain for stage in self._ordered())


__all__ = [
    "RUNNER_PREFIX",
    "STATE_MARK",
    "STATE_STYLE",
    "StageChip",
    "StagePipeline",
    "StageState",
    "top_level",
]
