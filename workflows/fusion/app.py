"""Fusion's TUI (R-A-2, spec 9.2): N panes across the top, the pipeline below.

    ┌ claude-haiku ┬ gpt ┬ gemini ┐   SplitPanes of StreamPane, one per branch
    ├──────────────┴─────┴────────┤
    │ ● models 3/3 ▸ ◐ analyze ▸ ○ synthesize │   StagePipeline
    ├─────────────────────────────┤
    │ the active stage's output   │   StreamPane for analyze / synthesize
    └─────────────────────────────┘

Three things about fusion make this app different from a single-agent one, and all
three come from the graph rather than from the UI:

**Fusion has no agents.** `controller.agents` is empty for a whole run -- every node
is a `ModelCall`, and a `ModelCall` is not an agent. Anything keyed off the agent
tree renders nothing, so every widget here is keyed off node ids. That includes
`interrupt_target()`: spec C-4 warns that a targetless `escape` in a fan-out cancels
nothing, and the override below is the answer -- a `ModelCall`'s step is registered
under its *node id*, so interrupting `models/claude-haiku-4-5` cancels that branch
and leaves the other three streaming.

**A branch streams under `node_id = <branch node id>`.** M5 made `ModelCall` emit
every delta with both `agent_id` and `node_id` set to the node id precisely so a
pane can be routed to a branch without inventing a second routing key.

**A restored run has outputs and no events.** After `ctrl+o` the completed branches
exist only as memoized node outputs in the session (R-A-2: "panes for completed
models are restored from session, incomplete ones re-stream"), so `seed_from_
controller` reads `Controller.node_output` rather than waiting for deltas that
belong to the process that wrote the file.

The panes are `tail=True`. M4 measured why: a fixed-size pane never asks Textual to
re-arrange the screen, and at four streams that is the difference between 46 ms and
53 ms -- and at six, between 62 ms and 89 ms with the loop blocked for 44 ms.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from azalabscode import Controller, NodeStarted
from azalabscode.tui import (
    EventLog,
    HarnessApp,
    RunStatusBar,
    SplitPanes,
    StagePipeline,
    StreamPane,
)
from workflows.fusion.workflow import FusionConfig

BRANCH_NODE = "models"
ANALYZE_NODE = "analyze"
SYNTH_NODE = "synthesize"

STAGE_PANES = (ANALYZE_NODE, SYNTH_NODE)
"""Stages with output of their own. `models` is the top row, not a bottom pane."""


class FusionApp(HarnessApp):
    """Spec 9.2's layout, built from the config the run was started with."""

    CSS = (
        HarnessApp.CSS
        + """
    #branches {
        height: 55%;
        border-bottom: solid $primary;
    }
    #branches StreamPane {
        border: round $panel;
    }
    #stage-area {
        height: 1fr;
    }
    #pipeline {
        background: $panel;
    }
    """
    )

    def __init__(self, controller: Controller, config: FusionConfig, **kwargs: Any) -> None:
        super().__init__(controller, **kwargs)
        self.config = config
        self.branches = config.branches()
        self.stage_panes: dict[str, StreamPane] = {}
        self.branch_panes: dict[str, StreamPane] = {}
        self.pinned_stage: str | None = None
        """Set once the user clicks a chevron: the app stops auto-advancing."""

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        panes: list[StreamPane] = []
        for slug, model in self.branches:
            node_id = f"{BRANCH_NODE}/{slug}"
            pane = StreamPane(node_id=node_id, title=model, tail=True)
            self.branch_panes[node_id] = pane
            panes.append(pane)
        yield SplitPanes(*panes, id="branches")

        yield StagePipeline(
            [(BRANCH_NODE, "models"), (ANALYZE_NODE, "analyze"), (SYNTH_NODE, "synthesize")],
            discover=False,
            id="pipeline",
        )

        with Vertical(id="stage-area"):
            for node_id in STAGE_PANES:
                pane = StreamPane(node_id=node_id, title=node_id, tail=True)
                pane.display = node_id == ANALYZE_NODE
                self.stage_panes[node_id] = pane
                yield pane
            note = Static("the model panes are above", id="stage-note")
            note.display = False
            yield note

        log = EventLog(id="event-log")
        log.display = False
        yield log
        yield RunStatusBar(self.controller, id="status-bar")

    @property
    def pipeline(self) -> StagePipeline:
        """The stage chevrons."""

        return self.query_one("#pipeline", StagePipeline)

    # -- stage selection ----------------------------------------------------

    def show_stage(self, node_id: str) -> None:
        """Put one stage's pane in the bottom area."""

        note = self.query_one("#stage-note", Static)
        for stage_id, pane in self.stage_panes.items():
            pane.display = stage_id == node_id
        note.display = node_id == BRANCH_NODE

    def on_stage_pipeline_stage_selected(self, message: StagePipeline.StageSelected) -> None:
        """A clicked chevron pins the bottom area to that stage."""

        message.stop()
        self.pinned_stage = message.node_id
        self.show_stage(message.node_id)

    def dispatch(self, event: Any) -> None:
        """Follow the run: the stage that starts becomes the visible one.

        Spec 9.2 asks for "each stage visible as it runs". Auto-advance stops the
        moment the user picks a stage: a screen that jumps away from what someone is
        reading is worse than one that needs a click.
        """

        if (
            isinstance(event, NodeStarted)
            and self.pinned_stage is None
            and event.node_id in self.stage_panes
        ):
            self.show_stage(event.node_id)
        super().dispatch(event)

    # -- hooks --------------------------------------------------------------

    def interrupt_target(self) -> str | None:
        """Which branch `escape` cancels (spec C-4).

        A `ModelCall`'s step is registered under its node id, so the node id *is* the
        interrupt target even though the branch is not an agent. Preference order:
        the pinned stage, then the first branch still streaming, then the analysis
        stage. A targetless interrupt would cancel nothing and emit
        `interrupt_no_target`, which is the case C-4 warns about.
        """

        if self.pinned_stage:
            return self.pinned_stage
        for node_id, pane in self.branch_panes.items():
            if not pane.finished:
                return node_id
        for node_id, pane in self.stage_panes.items():
            if pane.text and not pane.finished:
                return node_id
        return None

    def seed_from_controller(self, controller: Controller) -> None:
        """Restore completed branches from the session (R-A-2)."""

        super().seed_from_controller(controller)
        for node_id, pane in {**self.branch_panes, **self.stage_panes}.items():
            if not controller.node_completed(node_id):
                continue
            output = controller.node_output(node_id)
            if isinstance(output, str) and output:
                pane.set_text(output, status="completed")

    def panes_text(self) -> dict[str, str]:
        """Every pane's flushed text by node id. For `--dump` and for tests."""

        for pane in [*self.branch_panes.values(), *self.stage_panes.values()]:
            pane.flush_now()
        return {
            node_id: pane.text
            for node_id, pane in {**self.branch_panes, **self.stage_panes}.items()
        }


__all__ = ["ANALYZE_NODE", "BRANCH_NODE", "STAGE_PANES", "SYNTH_NODE", "FusionApp"]
