"""`Transcript`: one agent's conversation, built from events alone (R-U-1).

The widget never sees an `AssistantMessage`. It sees `ModelCallStarted`, a stream
of `ModelDelta`, and `ModelCallCompleted`, and it reconstructs the turn from those
-- which is what makes the UI layer replaceable: the only inputs are the event
stream and the `Controller` API.

Layout is a column of blocks in arrival order:

* a `StreamPane` per model call, so the batching in R-U-4 is not re-implemented
  here and a transcript under load costs the same as a fusion pane;
* a `Collapsible` per tool call, collapsed by default, whose title carries the
  status and duration and whose body carries params, result and error;
* a highlighted line for each injected user message (R-C-4), because a message the
  user typed mid-run is the one thing in the transcript they will look for.

Blocks are capped. A transcript is a view, not a store: the session document holds
the authoritative history, and an unbounded column of widgets in a long coding run
is a layout cost paid on every repaint forever.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Collapsible, Static

from azalabscode.events import (
    ApprovalRequested,
    ApprovalResolved,
    Event,
    MessageInjected,
    ModelCallCancelled,
    ModelCallCompleted,
    ModelCallFailed,
    ModelCallStarted,
    ModelDelta,
    ToolCallCancelled,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallProgress,
    ToolCallRequested,
    ToolCallStarted,
)
from azalabscode.tui.routing import ANY
from azalabscode.tui.widgets.base import EventWidget
from azalabscode.tui.widgets.stream_pane import StreamPane
from azalabscode.tui.widgets.tool_calls import failure_status

MAX_BLOCKS = 400
"""Blocks kept before the oldest is removed."""

PARAM_SUMMARY_CHARS = 60
"""How much of a call's params fits in a collapsed title."""


def summarize_params(params: dict[str, Any]) -> str:
    """A one-line rendering of a call's params for a collapsed title.

    Declaration order, not importance order: R-U-1 forbids the widget from knowing
    anything tool-specific, and a `Params` model lists its fields with the ones a
    reader cares about first anyway.
    """

    if not params:
        return ""
    parts = []
    for key, value in params.items():
        rendered = value if isinstance(value, str) else json.dumps(value, default=str)
        rendered = rendered.replace("\n", " ")
        if len(rendered) > PARAM_SUMMARY_CHARS:
            rendered = rendered[: PARAM_SUMMARY_CHARS - 1] + "…"
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


class ToolCallBlock(Collapsible):
    """One tool call: title is the status line, body is params, result and error.

    Consumes `ToolCallRequested`, `ToolCallStarted`, `ToolCallProgress`,
    `ToolCallCompleted`, `ToolCallFailed`, `ToolCallCancelled`, `ApprovalRequested`
    and `ApprovalResolved`.
    """

    DEFAULT_CSS = """
    ToolCallBlock {
        margin: 0 0 0 0;
    }
    ToolCallBlock.-failed > CollapsibleTitle {
        color: $error;
    }
    ToolCallBlock.-awaiting > CollapsibleTitle {
        color: $warning;
    }
    """

    STATUS_MARK: ClassVar[dict[str, str]] = {
        "requested": "·",
        "awaiting approval": "?",
        "running": "•",
        "ok": "✓",
        "failed": "✗",
        "denied": "✗",
        "cancelled": "∅",
    }

    def __init__(self, call_id: str, tool: str, params: dict[str, Any]) -> None:
        self._body = Static("", markup=False)
        super().__init__(self._body, title=f"{tool}", collapsed=True)
        self.call_id = call_id
        self.tool = tool
        self.params = dict(params)
        self.status = "requested"
        self.duration_ms = 0.0
        self.detail = ""
        self._progress = ""
        self._retitle()

    def _retitle(self) -> None:
        mark = self.STATUS_MARK.get(self.status, "·")
        summary = summarize_params(self.params)
        timing = f" {self.duration_ms / 1000:.1f}s" if self.duration_ms else ""
        head = f"{mark} {self.tool}"
        if summary:
            head += f"({summary})"
        self.title = f"{head} — {self.status}{timing}"

    def _rebody(self) -> None:
        text = Text(no_wrap=False)
        text.append("params\n", style="bold dim")
        text.append(json.dumps(self.params, indent=2, default=str) + "\n")
        if self._progress:
            text.append("\nprogress\n", style="bold dim")
            text.append(self._progress + "\n")
        if self.detail:
            style = "red" if self.status in {"failed", "denied"} else ""
            text.append(f"\n{self.status}\n", style="bold dim")
            text.append(self.detail, style=style)
        self._body.update(text)

    def apply(self, event: Event) -> None:
        """Fold one event into this block."""

        if isinstance(event, ToolCallRequested):
            self.params = dict(event.params)
            if event.parse_error:
                self.status = "failed"
                self.detail = f"malformed arguments: {event.parse_error}"
        elif isinstance(event, ApprovalRequested):
            self.status = "awaiting approval"
            self.add_class("-awaiting")
            self.detail = event.request.summary.title
            if event.request.summary.diff:
                self.detail += "\n" + event.request.summary.diff
        elif isinstance(event, ApprovalResolved):
            self.remove_class("-awaiting")
            if not event.decision.approved:
                self.status = "denied"
                self.detail = event.decision.reason or "denied"
                self.add_class("-failed")
            else:
                self.status = "running"
                self.detail = ""
        elif isinstance(event, ToolCallStarted):
            self.status = "running"
            self.remove_class("-awaiting")
        elif isinstance(event, ToolCallProgress):
            self._progress = (self._progress + event.text)[-2000:]
        elif isinstance(event, ToolCallCompleted):
            self.status = "ok" if event.result.ok else failure_status(event.result.error)
            self.duration_ms = event.duration_ms
            self.detail = _result_detail(event.result)
        elif isinstance(event, ToolCallFailed):
            # A denial is a decision, not a fault (`failure_status`), and the block
            # says which -- the same three-way split `ToolCallList` makes.
            self.status = failure_status(event.error)
            self.add_class("-failed")
            self.duration_ms = event.duration_ms
            self.detail = f"{event.error.kind}: {event.error.message}"
        elif isinstance(event, ToolCallCancelled):
            self.status = "cancelled"
            self.detail = event.reason
        self._retitle()
        self._rebody()


def _result_detail(result: Any) -> str:
    """The body text for a completed call.

    `ToolResult.display` is the structured rendering widgets are meant to read
    (spec delta 12); `content` is the model's copy and is never re-parsed here.
    When a tool supplied a `display` its `kind` is named so the reader knows a
    richer view exists -- `DiffView` renders it properly at M6.
    """

    display = getattr(result, "display", None)
    text = getattr(result, "text", "") or ""
    if display is not None:
        diff = display.data.get("diff") if isinstance(display.data, dict) else None
        if isinstance(diff, str) and diff:
            return diff
        return f"[{display.kind}]\n{text}"
    return text


class Transcript(EventWidget, VerticalScroll):
    """One agent's turns, tool calls and injected messages, in order (R-U-3)."""

    DEFAULT_CSS = """
    Transcript {
        height: 1fr;
        scrollbar-size-vertical: 1;
    }
    Transcript > .-injected {
        color: $warning;
        text-style: bold;
    }
    Transcript > .-agent-label {
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(
        self,
        *,
        agent_id: str,
        node_id: str = ANY,
        subagent: bool = False,
        max_blocks: int = MAX_BLOCKS,
        autoscroll: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.agent_filter = agent_id
        self.node_filter = node_id
        self.subagent = subagent
        self.max_blocks = max_blocks
        self.autoscroll = autoscroll
        self.blocks: list[Widget] = []
        self._panes: dict[str, StreamPane] = {}
        self._tools: dict[str, ToolCallBlock] = {}
        self._by_request: dict[str, ToolCallBlock] = {}
        self._scroll_wanted = False

    # -- events -------------------------------------------------------------

    def handle_event(self, event: Event) -> None:
        """Fold one event into the column."""

        if isinstance(event, ModelCallStarted):
            self._pane_for(event.call_id).handle_event(event)
        elif isinstance(
            event, ModelDelta | ModelCallCompleted | ModelCallCancelled | ModelCallFailed
        ):
            pane = self._panes.get(event.call_id)
            if pane is not None:
                pane.handle_event(event)
        elif isinstance(event, ToolCallRequested):
            self._tool_for(event.call_id, event.tool, event.params).apply(event)
        elif isinstance(event, ApprovalRequested):
            block = self._tool_for(event.request.call_id, event.request.tool, event.request.params)
            self._by_request[event.request.request_id] = block
            block.apply(event)
        elif isinstance(event, ApprovalResolved):
            block = self._by_request.pop(event.request_id, None)
            if block is not None:
                block.apply(event)
        elif isinstance(
            event,
            ToolCallStarted
            | ToolCallProgress
            | ToolCallCompleted
            | ToolCallFailed
            | ToolCallCancelled,
        ):
            block = self._tools.get(event.call_id)
            if block is not None:
                block.apply(event)
        elif isinstance(event, MessageInjected):
            self._add_injected(event.text)

    # -- blocks -------------------------------------------------------------

    def on_mount(self) -> None:
        """Start the autoscroll timer.

        Scrolling is deferred to a timer rather than done per event for the same
        reason the deltas are: a `scroll_end` per token is a layout per token.
        """

        self.set_interval(1 / 10, self._settle_scroll)

    def _settle_scroll(self) -> None:
        if self._scroll_wanted and self.autoscroll:
            self._scroll_wanted = False
            self.scroll_end(animate=False)

    def _append(self, widget: Widget) -> None:
        self.blocks.append(widget)
        if self.is_mounted:
            self.mount(widget)
        while len(self.blocks) > self.max_blocks:
            oldest = self.blocks.pop(0)
            self._forget(oldest)
            if oldest.is_mounted:
                oldest.remove()
        self._scroll_wanted = True

    def _forget(self, widget: Widget) -> None:
        for call_id, pane in list(self._panes.items()):
            if pane is widget:
                del self._panes[call_id]
        for call_id, block in list(self._tools.items()):
            if block is widget:
                del self._tools[call_id]
        for request_id, block in list(self._by_request.items()):
            if block is widget:
                del self._by_request[request_id]

    def _pane_for(self, call_id: str) -> StreamPane:
        pane = self._panes.get(call_id)
        if pane is None:
            pane = StreamPane(
                agent_id=self.agent_filter,
                call_id=call_id,
                subagent=self.subagent,
            )
            self._panes[call_id] = pane
            self._append(pane)
        return pane

    def _tool_for(self, call_id: str, tool: str, params: dict[str, Any]) -> ToolCallBlock:
        block = self._tools.get(call_id)
        if block is None:
            block = ToolCallBlock(call_id, tool, params)
            self._tools[call_id] = block
            self._append(block)
        return block

    def _add_injected(self, text: str) -> None:
        line = Static(Text(f"▸ {text}"), markup=False, classes="-injected")
        self._append(line)

    # -- introspection ------------------------------------------------------

    @property
    def panes(self) -> dict[str, StreamPane]:
        """Live stream panes by `call_id`."""

        return dict(self._panes)

    @property
    def tool_blocks(self) -> dict[str, ToolCallBlock]:
        """Live tool blocks by `call_id`."""

        return dict(self._tools)

    def text(self) -> str:
        """Everything the panes have flushed, in order. For tests and for `--dump`."""

        return "\n".join(pane.text for pane in self._panes.values() if pane.text)


__all__ = ["MAX_BLOCKS", "ToolCallBlock", "Transcript", "summarize_params"]
