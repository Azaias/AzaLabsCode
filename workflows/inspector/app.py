"""The inspector's TUI (R-A-3, spec 9.3): transcript left, tool calls right.

    ┌ transcript ────────┬ tool calls ──────────┐
    │ streamed output    │ agent tool  params … │
    │ collapsed tools    │ ...                  │
    │                    ├ detail ──────────────┤
    │                    │ params, result,      │
    │                    │ error, timing, the   │
    │                    │ raw event trail      │
    └────────────────────┴──────────────────────┘

R-A-3 asks for "a live table of every tool call with status, duration, and drill-down
to params, result, error, and the raw event sequence", plus the JSONL viewer. The
table is `ToolCallList`, the drill-down is `ToolCallDetail` following its cursor, and
the JSONL viewer is `ctrl+l` -- the event-log pane the base app already owns, which is
the same stream `JsonlRecorder` writes and therefore the same thing R-A-3 asks to see.

The only wiring this app does is the one message: the table posts `Selected` as the
cursor moves and the detail view is pointed at that record. Both widgets fold the
event stream independently, so neither is a view of the other's internals.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical

from azalabscode import Controller
from azalabscode.tui import (
    EventLog,
    HarnessApp,
    PromptInput,
    RunStatusBar,
    ToolCallDetail,
    ToolCallList,
    Transcript,
)
from workflows.inspector.workflow import AGENT_ID, InspectorConfig


class InspectorApp(HarnessApp):
    """Spec 9.3's layout: one agent, watched."""

    CSS = (
        HarnessApp.CSS
        + """
    #columns {
        height: 1fr;
    }
    #left {
        width: 55%;
        border-right: solid $primary;
    }
    #right {
        width: 45%;
    }
    #calls {
        height: 50%;
        border-bottom: solid $primary;
    }
    #detail {
        height: 1fr;
    }
    """
    )

    def __init__(self, controller: Controller, config: InspectorConfig, **kwargs: Any) -> None:
        super().__init__(controller, **kwargs)
        self.config = config

    def compose(self) -> ComposeResult:
        with Horizontal(id="columns"):
            with Vertical(id="left"):
                yield Transcript(agent_id=AGENT_ID, id="transcript")
            with Vertical(id="right"):
                yield ToolCallList(id="calls")
                yield ToolCallDetail(id="detail")
        yield PromptInput(id="prompt")
        log = EventLog(id="event-log")
        log.display = False
        yield log
        yield RunStatusBar(self.controller, id="status-bar")

    # -- the one piece of wiring -------------------------------------------

    @property
    def calls(self) -> ToolCallList:
        """The tool-call table."""

        return self.query_one("#calls", ToolCallList)

    @property
    def detail(self) -> ToolCallDetail:
        """The drill-down."""

        return self.query_one("#detail", ToolCallDetail)

    @property
    def transcript(self) -> Transcript:
        """The agent's transcript."""

        return self.query_one("#transcript", Transcript)

    def on_tool_call_list_selected(self, message: ToolCallList.Selected) -> None:
        """Follow the table's cursor with the drill-down (R-A-3)."""

        message.stop()
        self.detail.show(message.record)

    def interrupt_target(self) -> str | None:
        """The inspector has exactly one agent, and it is not always called `main`."""

        return AGENT_ID


__all__ = ["InspectorApp"]
