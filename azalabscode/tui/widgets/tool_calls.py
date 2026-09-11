"""`ToolCallList` and `ToolCallDetail`: every tool call, and one of them up close.

Spec 8.2 gives these two widgets one row in its table because they are one view
split in half: a table with status and duration, and a drill-down showing params,
result, error, timing and the raw event trail. R-A-3's whole acceptance criterion is
that the pair works.

The state they share is `ToolCallRecord`, and folding an event into a record is a
*function of the event stream alone* (R-U-1). That matters more than it looks: the
same fold has to produce the same row whether the events arrive live, or are replayed
out of a JSONL file, or are handed over by a second widget. So the record is a plain
dataclass with an `apply()` and no widget in it, and both widgets own records rather
than deriving one from the other.

**The event trail is the point of the detail view, not decoration.** A call that was
requested, waited on approval for nine seconds, started, emitted progress and then
failed tells a story the status column cannot. The trail is bounded per call
(`MAX_TRAIL`) because a `shell` running for ten minutes emits a progress event a
second and the drill-down is a view, not a log -- `EventLog` and `JsonlRecorder` are
where the whole stream lives.

`ToolResult.display` is what the detail view renders (spec delta 12). `content` is
the model's copy; re-parsing it here would be the widget guessing at a rendering the
tool already did properly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from rich.text import Text
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import DataTable, Static

from azalabscode.events import (
    ApprovalRequested,
    ApprovalResolved,
    Event,
    ToolCallCancelled,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallProgress,
    ToolCallRequested,
    ToolCallStarted,
)
from azalabscode.toolio import ToolError, ToolErrorKind, ToolResult
from azalabscode.tui.routing import ANY
from azalabscode.tui.widgets.base import EventWidget

MAX_ROWS = 500
"""Calls the table keeps. The oldest row goes when the cap is reached."""

MAX_TRAIL = 60
"""Events kept per call for the drill-down."""

PARAM_CHARS = 44
"""How much of the params fits in the table's params column."""

STATUS_STYLE: dict[str, str] = {
    "requested": "dim",
    "awaiting approval": "yellow",
    "running": "cyan",
    "ok": "green",
    "failed": "bold red",
    "denied": "red",
    "cancelled": "yellow",
}

TOOL_CALL_EVENTS = (
    ToolCallRequested,
    ApprovalRequested,
    ApprovalResolved,
    ToolCallStarted,
    ToolCallProgress,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallCancelled,
)
"""Every event class that carries a `call_id` these widgets fold."""


def call_id_of(event: Event, requests: Mapping[str, str] | None = None) -> str | None:
    """The `call_id` an event is about, or `None` if it is not about a call.

    Two events do not carry one directly. `ApprovalRequested` has it on the request.
    `ApprovalResolved` has *only* a `request_id`, deliberately -- a decision is about
    a request, and a resolution can arrive from a mode switch that never saw the call
    -- so the caller passes the `request_id -> call_id` index it built when the
    request arrived. Without that index a denial silently never reaches its row.
    """

    if isinstance(event, ApprovalRequested):
        return event.request.call_id
    if isinstance(event, ApprovalResolved):
        return (requests or {}).get(event.request_id)
    return getattr(event, "call_id", None)


def failure_status(error: ToolError | None) -> str:
    """`denied`, `cancelled` or `failed` for a call that did not succeed.

    A denial and a cancellation are outcomes of a *decision*, not faults, and the row
    should say which. The distinction is only in the error kind: the gate denies, the
    dispatcher then reports an ordinary failed call carrying `kind="denied"`, so a
    widget reading `ToolCallFailed` alone would call every denial a failure.
    """

    if error is None:
        return "failed"
    if error.kind is ToolErrorKind.DENIED:
        return "denied"
    if error.kind in {ToolErrorKind.CANCELLED, ToolErrorKind.INTERRUPTED}:
        return "cancelled"
    return "failed"


def summarize(params: dict[str, Any], limit: int = PARAM_CHARS) -> str:
    """A one-line rendering of a call's params, in declaration order."""

    if not params:
        return ""
    parts = []
    for key, value in params.items():
        rendered = value if isinstance(value, str) else json.dumps(value, default=str)
        rendered = rendered.replace("\n", " ")
        if len(rendered) > limit:
            rendered = rendered[: limit - 1] + "…"
        parts.append(f"{key}={rendered}")
    joined = ", ".join(parts)
    return joined if len(joined) <= limit else joined[: limit - 1] + "…"


@dataclass
class ToolCallRecord:
    """One tool call, folded from its events. The row and the drill-down share it."""

    call_id: str
    tool: str = ""
    agent_id: str = ""
    node_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "requested"
    duration_ms: float = 0.0
    result: ToolResult | None = None
    error: ToolError | None = None
    detail: str = ""
    """Free text for the current status: a denial reason, a parse error, a summary."""
    progress: str = ""
    request_id: str | None = None
    trail: list[Event] = field(default_factory=list)
    approval_wait_ms: float = 0.0
    _requested_at: Any = None
    _approved_at: Any = None

    @property
    def finished(self) -> bool:
        """Whether the call reached a terminal status."""

        return self.status in {"ok", "failed", "denied", "cancelled"}

    def apply(self, event: Event) -> None:
        """Fold one event. Unknown event classes are recorded in the trail only."""

        if len(self.trail) >= MAX_TRAIL:
            del self.trail[0]
        self.trail.append(event)

        if event.agent_id:
            self.agent_id = event.agent_id
        if event.node_id:
            self.node_id = event.node_id

        if isinstance(event, ToolCallRequested):
            self.tool = event.tool
            self.params = dict(event.params)
            self._requested_at = event.ts
            if event.parse_error:
                self.status = "failed"
                self.detail = f"malformed arguments: {event.parse_error}"
        elif isinstance(event, ApprovalRequested):
            self.tool = event.request.tool or self.tool
            self.params = dict(event.request.params) or self.params
            self.request_id = event.request.request_id
            self.status = "awaiting approval"
            self.detail = event.request.summary.title
            self._approved_at = event.ts
        elif isinstance(event, ApprovalResolved):
            if self._approved_at is not None:
                self.approval_wait_ms = (event.ts - self._approved_at).total_seconds() * 1000.0
            if event.decision.approved:
                self.status = "running"
                self.detail = f"approved by {event.decision.by or 'user'}"
            else:
                self.status = "denied"
                self.detail = event.decision.reason or "denied"
        elif isinstance(event, ToolCallStarted):
            self.tool = event.tool or self.tool
            self.status = "running"
        elif isinstance(event, ToolCallProgress):
            self.progress = (self.progress + event.text)[-4000:]
        elif isinstance(event, ToolCallCompleted):
            self.status = "ok" if event.result.ok else failure_status(event.result.error)
            self.duration_ms = event.duration_ms
            self.result = event.result
            self.error = event.result.error
            self.detail = ""
        elif isinstance(event, ToolCallFailed):
            self.status = failure_status(event.error)
            self.duration_ms = event.duration_ms
            self.error = event.error
            self.detail = f"{event.error.kind}: {event.error.message}"
        elif isinstance(event, ToolCallCancelled):
            self.status = "cancelled"
            self.detail = event.reason

    def duration_text(self) -> str:
        """The duration column: blank until there is one."""

        if self.duration_ms:
            return f"{self.duration_ms / 1000:.2f}s"
        return "—"

    def cells(self) -> tuple[Text, Text, Text, Text, Text]:
        """The table row: agent, tool, params, status, duration."""

        return (
            Text(self.agent_id or "—", style="dim"),
            Text(self.tool or "?"),
            Text(summarize(self.params), style="dim"),
            Text(self.status, style=STATUS_STYLE.get(self.status, "")),
            Text(self.duration_text(), style="dim"),
        )


class ToolCallList(EventWidget, DataTable[Any]):
    """A live table of every tool call: status, duration, and a cursor (R-U-3).

    Consumes `ToolCallRequested`, `ApprovalRequested`, `ApprovalResolved`,
    `ToolCallStarted`, `ToolCallProgress`, `ToolCallCompleted`, `ToolCallFailed` and
    `ToolCallCancelled`.

    Moving the cursor posts `ToolCallList.Selected`, which is what a layout wires to
    a `ToolCallDetail`. The widget does not hold the detail view itself: R-A-3 puts
    them side by side, the coding agent puts the detail in an overlay, and a table
    that owned its drill-down could not do both.
    """

    DEFAULT_CSS = """
    ToolCallList {
        height: 1fr;
    }
    """

    COLUMNS: ClassVar[tuple[tuple[str, str, int | None], ...]] = (
        ("agent", "agent", 10),
        ("tool", "tool", 12),
        ("params", "params", PARAM_CHARS + 2),
        ("status", "status", 17),
        ("duration", "time", 8),
    )

    class Selected(Message):
        """The cursor moved onto a call, or a row was clicked."""

        def __init__(self, record: ToolCallRecord) -> None:
            self.record = record
            super().__init__()

    def __init__(
        self,
        *,
        agent_id: str = ANY,
        node_id: str = ANY,
        max_rows: int = MAX_ROWS,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.agent_filter = agent_id
        self.node_filter = node_id
        self.max_rows = max_rows
        self.records: dict[str, ToolCallRecord] = {}
        """Every call seen, by `call_id`, in arrival order."""
        self.cursor_type = "row"
        self.zebra_stripes = True
        self._by_request: dict[str, str] = {}
        self._buffered: list[Event] = []

    def on_mount(self) -> None:
        """Add the columns, then replay anything that arrived before the screen did.

        `DataTable.add_column` measures its label against the app's console, so it
        cannot run in `__init__` -- there is no active app there. A widget built and
        registered before the pump starts would otherwise drop its first events, so
        they are buffered rather than discarded.
        """

        for key, label, width in self.COLUMNS:
            self.add_column(label, key=key, width=width)
        buffered, self._buffered = self._buffered, []
        for event in buffered:
            self.handle_event(event)

    # -- events -------------------------------------------------------------

    def handle_event(self, event: Event) -> None:
        """Fold one event into its row, adding the row if this is the first."""

        if not isinstance(event, TOOL_CALL_EVENTS):
            return
        if not self.is_mounted:
            self._buffered.append(event)
            del self._buffered[: max(0, len(self._buffered) - self.max_rows * 4)]
            return
        call_id = call_id_of(event, self._by_request)
        if not call_id:
            return
        if isinstance(event, ApprovalRequested):
            self._by_request[event.request.request_id] = call_id
        record = self.records.get(call_id)
        if record is None:
            record = ToolCallRecord(call_id=call_id)
            self.records[call_id] = record
            record.apply(event)
            self._add_row(record)
        else:
            record.apply(event)
            self._update_row(record)

    def _add_row(self, record: ToolCallRecord) -> None:
        self.add_row(*record.cells(), key=record.call_id)
        self._trim()

    def _update_row(self, record: ToolCallRecord) -> None:
        if record.call_id not in self.rows:
            return
        for (key, _label, _width), cell in zip(self.COLUMNS, record.cells(), strict=True):
            self.update_cell(record.call_id, key, cell)

    def _trim(self) -> None:
        while len(self.rows) > self.max_rows:
            oldest = next(iter(self.rows))
            self.remove_row(oldest)
            self.records.pop(str(oldest.value), None)

    # -- selection ----------------------------------------------------------

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Publish the highlighted record so a detail view can follow the cursor."""

        self._publish(str(event.row_key.value or ""))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Publish on click or enter as well, so a mouse works without the cursor."""

        self._publish(str(event.row_key.value or ""))

    def _publish(self, call_id: str) -> None:
        record = self.records.get(call_id)
        if record is not None:
            self.post_message(self.Selected(record))

    @property
    def selected(self) -> ToolCallRecord | None:
        """The record under the cursor, if any."""

        rows = self.ordered_rows
        if self.cursor_row < 0 or self.cursor_row >= len(rows):
            return None
        return self.records.get(str(rows[self.cursor_row].key.value or ""))

    def ordered(self) -> list[ToolCallRecord]:
        """Every record in arrival order. For `--headless` summaries and tests."""

        return list(self.records.values())


class ToolCallDetail(EventWidget, VerticalScroll):
    """One call up close: params, result, error, timing and the event trail (R-U-3).

    Consumes the same events as `ToolCallList`, but only for the call it is showing:
    a drill-down on a running `shell` updates as the progress arrives instead of
    freezing at the moment it was opened.

    `show(record)` points it at a record. It does not own the record -- the table
    does -- so both widgets stay in step without either one polling the other.
    """

    DEFAULT_CSS = """
    ToolCallDetail {
        height: 1fr;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        *,
        agent_id: str = ANY,
        node_id: str = ANY,
        show_trail: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.agent_filter = agent_id
        self.node_filter = node_id
        self.show_trail = show_trail
        self.record: ToolCallRecord | None = None
        self._body = Static("", markup=False)

    def compose(self) -> Any:
        """One `Static`, rebuilt on every change. The content is a screenful."""

        yield self._body

    def on_mount(self) -> None:
        """Draw whatever was set before the widget was mounted."""

        self.refresh_body()

    def show(self, record: ToolCallRecord | None) -> None:
        """Point the view at a record, or clear it."""

        self.record = record
        self.refresh_body()

    def handle_event(self, event: Event) -> None:
        """Redraw when the shown call changes. Other calls are ignored here."""

        if self.record is None or not isinstance(event, TOOL_CALL_EVENTS):
            return
        if isinstance(event, ApprovalResolved):
            if event.request_id != self.record.request_id:
                return
        elif call_id_of(event) != self.record.call_id:
            return
        # The record is folded by whoever owns it -- normally the table, which is
        # registered before this widget. Redrawing is all that is left to do.
        self.refresh_body()

    def refresh_body(self) -> None:
        """Rebuild the rendered text from the current record."""

        self._body.update(self.render_record())

    def render_record(self) -> Text:
        """The whole drill-down as one `Text`. Separated out so a test can read it."""

        record = self.record
        text = Text(no_wrap=False)
        if record is None:
            text.append("no call selected", style="dim")
            return text

        text.append(f"{record.tool}", style="bold")
        text.append(f"  {record.call_id}\n", style="dim")
        text.append(f"{record.agent_id or '—'}", style="dim")
        if record.node_id:
            text.append(f"  node {record.node_id}", style="dim")
        text.append("  ")
        text.append(record.status, style=STATUS_STYLE.get(record.status, ""))
        text.append(f"  {record.duration_text()}\n", style="dim")
        if record.approval_wait_ms:
            text.append(f"waited {record.approval_wait_ms / 1000:.1f}s for approval\n", style="dim")
        if record.detail:
            text.append(record.detail + "\n")

        text.append("\nparams\n", style="bold dim")
        text.append(json.dumps(record.params, indent=2, default=str) + "\n")

        if record.progress:
            text.append("\nprogress\n", style="bold dim")
            text.append(record.progress[-2000:] + "\n")

        if record.error is not None:
            text.append("\nerror\n", style="bold dim")
            text.append(f"{record.error.kind}: {record.error.message}\n", style="red")
            if record.error.details:
                text.append(json.dumps(record.error.details, indent=2, default=str) + "\n")

        if record.result is not None:
            text.append("\nresult\n", style="bold dim")
            display = record.result.display
            if display is not None:
                text.append(f"[{display.kind}] ", style="cyan")
                text.append(json.dumps(display.data, indent=2, default=str)[:4000] + "\n")
            body = record.result.text
            if body:
                text.append(body[:4000] + "\n")
            if record.result.meta:
                text.append("meta ", style="dim")
                text.append(json.dumps(record.result.meta, default=str)[:1000] + "\n", style="dim")

        if self.show_trail and record.trail:
            text.append("\nevents\n", style="bold dim")
            for event in record.trail:
                text.append(f"{event.seq:>6} ", style="dim")
                text.append(f"{event.type}\n", style="cyan")
        return text


__all__ = [
    "MAX_ROWS",
    "MAX_TRAIL",
    "STATUS_STYLE",
    "TOOL_CALL_EVENTS",
    "ToolCallDetail",
    "ToolCallList",
    "ToolCallRecord",
    "call_id_of",
    "failure_status",
    "summarize",
]
