"""Running the inspector without a TUI (R-U-7).

The same controller and the same graph as the TUI path; the difference is what is
subscribed to the bus. Headless that is a `CallSummary`, so `--headless` still answers
R-A-3's question -- what did it call, how long did each take, what failed -- in a form
a terminal can print.

**The summary folds the events itself rather than borrowing `ToolCallRecord`.** The
widget's fold is richer (it keeps the whole event trail and the structured display)
and it lives in `azalabscode.tui`, which imports Textual. A headless run must not pay
for a UI it will not draw, and import contract 5 forbids a reference workflow from
reaching into `azalabscode.tui.*` anyway. The two folds agree because they are folds
of the *same event contract* (R-X-3), not because one calls the other; the test
asserts they produce the same statuses over one run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from azalabscode import (
    ApprovalRequested,
    ApprovalResolved,
    Controller,
    Event,
    EventBus,
    PermissionMode,
    Subscription,
    ToolCallCancelled,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallRequested,
    ToolCallStarted,
    ToolErrorKind,
)
from workflows.inspector.workflow import CONFIG_TYPE, IMPORT_PATH, InspectorConfig, build


def controller_for(
    config: InspectorConfig | dict[str, Any],
    *,
    session_dir: str | Path | None = None,
    bus: EventBus | None = None,
    mode: PermissionMode = PermissionMode.AUTO,
    approval_handler: Any = None,
    autosave: bool = True,
    run_id: str | None = None,
) -> Controller:
    """A controller with the inspector graph bound and its rebuild recipe recorded."""

    cfg = config if isinstance(config, InspectorConfig) else InspectorConfig.model_validate(config)
    controller = Controller(
        run_id=run_id,
        bus=bus,
        permission_mode=mode,
        approval_handler=approval_handler,
        session_dir=session_dir,
        autosave=autosave,
    )
    controller.bind_workflow(
        build(cfg),
        import_path=IMPORT_PATH,
        config=cfg.model_dump(mode="json"),
        config_type=CONFIG_TYPE,
    )
    return controller


def failure_status(error: Any) -> str:
    """`denied`, `cancelled` or `failed`, from the error kind.

    The same three-way split the widget makes: a denial and a cancellation are
    outcomes of a decision, not faults, and a summary that calls every one of them a
    failure is a summary that cries wolf.
    """

    kind = getattr(error, "kind", None)
    if kind is ToolErrorKind.DENIED:
        return "denied"
    if kind in {ToolErrorKind.CANCELLED, ToolErrorKind.INTERRUPTED}:
        return "cancelled"
    return "failed"


@dataclass
class CallRow:
    """One tool call as the headless summary knows it."""

    call_id: str
    tool: str = ""
    agent_id: str = ""
    status: str = "requested"
    duration_ms: float = 0.0
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether the call succeeded."""

        return self.status == "ok"

    def duration_text(self) -> str:
        """The duration column."""

        return f"{self.duration_ms / 1000:.2f}s" if self.duration_ms else "-"


class CallSummary:
    """Every tool call in a run, folded from the event stream (R-X-3)."""

    def __init__(self) -> None:
        self.rows: dict[str, CallRow] = {}
        self._by_request: dict[str, str] = {}

    def handle_event(self, event: Event) -> None:
        """Fold one event, if it is about a tool call."""

        if isinstance(event, ToolCallRequested):
            row = self._row(event.call_id)
            row.tool = event.tool
            row.agent_id = event.agent_id or ""
            if event.parse_error:
                row.status = "failed"
                row.detail = f"malformed arguments: {event.parse_error}"
        elif isinstance(event, ApprovalRequested):
            row = self._row(event.request.call_id)
            row.tool = row.tool or event.request.tool
            row.status = "awaiting approval"
            self._by_request[event.request.request_id] = event.request.call_id
        elif isinstance(event, ApprovalResolved):
            call_id = self._by_request.get(event.request_id)
            if call_id is None:
                return
            row = self._row(call_id)
            if event.decision.approved:
                row.status = "running"
            else:
                row.status = "denied"
                row.detail = event.decision.reason or "denied"
        elif isinstance(event, ToolCallStarted):
            self._row(event.call_id).status = "running"
        elif isinstance(event, ToolCallCompleted):
            row = self._row(event.call_id)
            row.status = "ok" if event.result.ok else failure_status(event.result.error)
            row.duration_ms = event.duration_ms
        elif isinstance(event, ToolCallFailed):
            row = self._row(event.call_id)
            row.status = failure_status(event.error)
            row.duration_ms = event.duration_ms
            row.detail = f"{event.error.kind}: {event.error.message}"
        elif isinstance(event, ToolCallCancelled):
            row = self._row(event.call_id)
            row.status = "cancelled"
            row.detail = event.reason

    def _row(self, call_id: str) -> CallRow:
        row = self.rows.get(call_id)
        if row is None:
            row = CallRow(call_id=call_id)
            self.rows[call_id] = row
        return row

    def statuses(self) -> dict[str, str]:
        """`call_id -> status`. What the widget's fold is compared against."""

        return {call_id: row.status for call_id, row in self.rows.items()}

    def failures(self) -> Iterator[CallRow]:
        """Calls that did not succeed. What a CI check looks at."""

        return (row for row in self.rows.values() if not row.ok)

    def render(self) -> str:
        """The summary as text, for `--headless`."""

        if not self.rows:
            return "no tool calls"
        width = max(len(row.tool) for row in self.rows.values()) or 4
        lines = [
            f"{row.agent_id or '-':<10} {row.tool:<{width}}  "
            f"{row.status:<18} {row.duration_text():>8}  {row.detail}".rstrip()
            for row in self.rows.values()
        ]
        return "\n".join(lines)


class CallWatcher:
    """Subscribes a `CallSummary` to a bus for the length of a `with` block.

    The same shape as `JsonlRecorder`, and for the same reason: the subscription has
    to outlive the run's last event and be drained before the summary is printed.
    """

    def __init__(self, bus: EventBus, summary: CallSummary | None = None) -> None:
        self.bus = bus
        self.summary = summary if summary is not None else CallSummary()
        self._sub: Subscription | None = None
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self._sub = self.bus.subscribe(name="inspector-summary")
        self._task = asyncio.create_task(self._pump(), name="inspector-summary")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._sub is not None:
            self._sub.unsubscribe()
        if self._task is not None:
            await self._task

    async def _pump(self) -> None:
        assert self._sub is not None
        async for event in self._sub:
            self.summary.handle_event(event)


async def run_headless(
    config: InspectorConfig | dict[str, Any],
    *,
    session_dir: str | Path | None = None,
    timeout: float | None = None,  # noqa: ASYNC109 - the bound is the caller's; Controller.run takes it
) -> tuple[str, CallSummary]:
    """Run with no UI. Returns the agent's answer and the call summary."""

    controller = controller_for(config, session_dir=session_dir)
    async with CallWatcher(controller.bus) as watcher:
        answer = await controller.run(timeout=timeout)
    return str(answer or ""), watcher.summary


__all__ = ["CallRow", "CallSummary", "CallWatcher", "controller_for", "run_headless"]
