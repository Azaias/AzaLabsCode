"""`RunStatusBar`: run state, mode, agents, spend, checkpoints, and what is blocking.

Spec 8.1 requires it always mounted, and spec C-12 is the reason it is worth more
than a decoration: pressing pause during a five-minute `shell` and seeing nothing
happen looks like a bug, so `PAUSING` is displayed as a distinct state alongside
`Controller.blocking_description()` -- `main: shell (312.4s)`. The user then knows
whether to wait or to hard-pause.

Everything except the checkpoint counters is a **live read** off the `Controller`.
That is deliberate. Every one of those fields also has an event (R-X-3), but a bar
rebuilt from events has to be right about every one of them forever, and a bar that
reads `controller.state` cannot drift. The events are used only as a hint that
something changed; a slow timer covers whatever they miss, including elapsed times
that no event will ever fire for.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import RenderResult
from textual.widgets import Static

from azalabscode.control.controller import Controller
from azalabscode.events import (
    Checkpoint,
    Event,
    EventsDropped,
    RunWarning,
)
from azalabscode.runstate import RunState
from azalabscode.tui.widgets.base import EventWidget

REFRESH_INTERVAL_S = 0.25
"""How often the bar redraws on its own. Fast enough that a duration counting up
looks live, slow enough to be free next to a 30 fps stream pane."""

STATE_STYLE = {
    RunState.CREATED: "dim",
    RunState.RUNNING: "bold green",
    RunState.PAUSING: "bold yellow",
    RunState.PAUSED: "bold yellow",
    RunState.WAITING_APPROVAL: "bold magenta",
    RunState.INTERRUPTING: "bold yellow",
    RunState.COMPLETED: "bold green",
    RunState.FAILED: "bold red",
    RunState.CANCELLED: "bold red",
}

WAITING_STATES = frozenset({RunState.PAUSING, RunState.INTERRUPTING, RunState.WAITING_APPROVAL})
"""States in which "what is it waiting for" is the question the user has."""


def format_tokens(count: int) -> str:
    """`1234` as `1.2k`. Exact counts below a thousand, where they are readable."""

    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.2f}M"


class RunStatusBar(EventWidget, Static):
    """One line: state, mode, agents, tokens, cost, checkpoints, notice.

    Consumes `Checkpoint`, `RunWarning` and `EventsDropped`; reads everything else
    from the `Controller`. Filters on `ANY` so run-level events, which carry no
    `agent_id`, reach it (see `tui.routing.matches`).
    """

    DEFAULT_CSS = """
    RunStatusBar {
        height: 1;
        dock: bottom;
        background: $panel;
        color: $text;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        controller: Controller,
        *,
        refresh_interval_s: float = REFRESH_INTERVAL_S,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.controller = controller
        self.refresh_interval_s = refresh_interval_s
        self.checkpoints = 0
        """Safe points folded, as counted by `Checkpoint` events."""
        self.disk_writes = 0
        """Of those, the ones that reached disk. After M3 these are different
        numbers: `Checkpointer.writes` counts writes, `Controller.folds` counts
        safe points."""
        self.last_checkpoint_path: str | None = None
        self.dropped = 0
        self.notice = ""
        """A line that overrides the right-hand side: a `SaveTimeout`'s `blocking`
        string, an `interrupt_no_target` warning, whatever the user must not miss."""
        self.notice_style = "bold yellow"

    def rebind(self, controller: Controller) -> None:
        """Point the bar at a different run, after `ctrl+o` loads a session."""

        self.controller = controller
        self.checkpoints = 0
        self.disk_writes = 0
        self.last_checkpoint_path = None
        self.dropped = 0
        self.notice = ""

    def on_mount(self) -> None:
        """Redraw on a slow timer so elapsed durations advance without an event."""

        self.set_interval(self.refresh_interval_s, self.refresh)

    def set_notice(self, message: str, *, style: str = "bold yellow") -> None:
        """Show a message until the next one replaces it or `clear_notice` runs."""

        self.notice = message
        self.notice_style = style
        if self.is_mounted:
            self.refresh()

    def clear_notice(self) -> None:
        """Drop the current notice."""

        self.notice = ""
        if self.is_mounted:
            self.refresh()

    def handle_event(self, event: Event) -> None:
        """Count checkpoints and surface warnings. Everything else is a live read."""

        if isinstance(event, Checkpoint):
            self.checkpoints += 1
            if event.to_disk:
                self.disk_writes += 1
                self.last_checkpoint_path = event.path
        elif isinstance(event, RunWarning):
            self.set_notice(event.message)
        elif isinstance(event, EventsDropped):
            self.dropped += event.count
        if self.is_mounted:
            self.refresh()

    def segments(self) -> list[tuple[str, str]]:
        """The bar as `(text, style)` pairs. Separated out so a test can read it."""

        controller = self.controller
        state = controller.state
        usage = controller.usage()
        active = sum(1 for agent in controller.agents.values() if agent.phase.value != "finished")

        parts: list[tuple[str, str]] = [
            (state.value.upper(), STATE_STYLE.get(state, "")),
            (controller.permission_mode.value, "cyan"),
            (f"agents {active}/{len(controller.agents)}", ""),
            (f"{format_tokens(usage.total_tokens)} tok", ""),
        ]
        if usage.cost_usd is not None:
            parts.append((f"${usage.cost_usd:.4f}", ""))
        parts.append((f"ckpt {self.checkpoints} (disk {self.disk_writes})", "dim"))

        pending = len(controller.pending_approvals)
        if pending:
            parts.append((f"{pending} awaiting approval", "bold magenta"))
        if self.dropped:
            parts.append((f"{self.dropped} deltas dropped", "dim yellow"))

        if self.notice:
            parts.append((self.notice, self.notice_style))
        elif state in WAITING_STATES:
            parts.append((controller.blocking_description(), "yellow"))
        return parts

    def render(self) -> RenderResult:
        """Join the segments with a separator."""

        text = Text(no_wrap=True, overflow="ellipsis")
        for index, (body, style) in enumerate(self.segments()):
            if index:
                text.append("  │  ", style="dim")
            text.append(body, style=style)
        return text


__all__ = ["REFRESH_INTERVAL_S", "RunStatusBar", "format_tokens"]
