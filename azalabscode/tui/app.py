"""`HarnessApp`: the Textual base every workflow's TUI subclasses (R-U-2).

It does five things, and a subclass usually overrides only the first:

1. **Layout.** `compose()` yields a `Transcript` for the main agent and a
   `RunStatusBar`. A workflow app replaces `compose` and keeps everything else.
2. **Routing.** One subscription to the controller's bus, one pump task, and every
   `EventWidget` in the tree gets the events matching its `(agent_id, node_id)`
   filters (spec 8.1). The pump is the *only* thread of control that touches
   widgets, so no widget needs a lock.
3. **Approvals.** It installs a `TUIApprovalHandler` on the controller and pushes an
   `ApprovalModal` when `ApprovalRequested` arrives, so a workflow author writes no
   UI code to get a working permission prompt (R-U-6).
4. **Bindings.** Spec 8.1's table, with every action implemented against the
   `Controller` API and nothing else (R-U-1).
5. **Attachment.** `attach()` binds to a controller; `ctrl+o` loads a session and
   attaches to the *new* controller, rebinding the subscription and re-seeding the
   widgets from `controller.agents` rather than waiting for events that belong to
   the process that wrote the file.

Three rules worth stating because breaking them is silent:

* **Never take a `Controller` lock from a key handler.** `set_permission_mode` holds
  `_command_lock` while it auto-resolves pending approvals, and resolution calls
  back into the controller; `asyncio.Lock` is not reentrant. Every action here goes
  through a public `Controller` method, and the ones that can wait go through a
  worker so a 120 s `save()` cannot stall the message pump.
* **A loaded run has agents but no events.** `load()` registers every agent the
  session names before any body starts, and those `AgentSpawned` events belong to
  the previous process. `attach()` seeds from `controller.agents` first, then
  follows the stream.
* **`Session.event_seq` seeds the new bus.** A reloaded run's `seq` continues from
  where the file left off. Nothing here may assume it starts at 1.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import RichLog

from azalabscode.control.controller import Controller
from azalabscode.errors import CheckpointError, HarnessError, SaveTimeout
from azalabscode.events import (
    ApprovalRequested,
    ApprovalResolved,
    Event,
    EventBus,
    ModelDelta,
    Subscription,
)
from azalabscode.permissions import ApprovalRequest, Decision, PermissionMode
from azalabscode.runstate import RunState
from azalabscode.tui.approval import TUIApprovalHandler
from azalabscode.tui.bindings import DOUBLE_PRESS_WINDOW_S, HARNESS_BINDINGS
from azalabscode.tui.routing import EventRouter
from azalabscode.tui.widgets.approval_modal import ApprovalModal
from azalabscode.tui.widgets.base import EventWidget
from azalabscode.tui.widgets.prompt_input import PromptInput
from azalabscode.tui.widgets.status_bar import RunStatusBar
from azalabscode.tui.widgets.transcript import Transcript

EVENT_LOG_ID = "event-log"
STATUS_BAR_ID = "status-bar"
MAIN_TRANSCRIPT_ID = "main-transcript"

EVENT_LOG_LINES = 2000
"""Lines the `ctrl+l` pane keeps."""


class EventLog(EventWidget, RichLog):
    """The `ctrl+l` pane: one line per event, newest last.

    `ModelDelta` is excluded. A four-way fan-out at 200 tokens/s is 800 deltas a
    second, the panes already show that text, and a log that scrolls faster than it
    can be read is not an observability tool. Everything else is here, which is
    exactly the guarantee R-X-3 makes.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(max_lines=EVENT_LOG_LINES, markup=False, **kwargs)  # type: ignore[arg-type]
        self.suppressed = 0
        """Deltas not logged."""
        self.written = 0
        """Lines written. Counted here rather than read off `RichLog.lines`, which
        is empty until the widget has been given a size and its deferred renders
        have run -- a real property of the widget, and a confusing thing to assert
        against."""

    def handle_event(self, event: Event) -> None:
        """Append one line, unless it is a delta."""

        if isinstance(event, ModelDelta):
            self.suppressed += 1
            return
        line = Text()
        line.append(f"{event.seq:>6} ", style="dim")
        line.append(f"{event.type:<24}", style="cyan")
        if event.agent_id:
            line.append(f" {event.agent_id}", style="dim")
        detail = _log_detail(event)
        if detail:
            line.append(f"  {detail}")
        self.written += 1
        self.write(line)


def _log_detail(event: Event) -> str:
    """A short right-hand column for the event log, per event class."""

    for field in ("tool", "code", "kind", "new", "call_id", "message"):
        value = getattr(event, field, None)
        if value:
            return f"{field}={value}"
    return ""


class HarnessApp(App[Any]):
    """The Textual base class (R-U-2). Subclass it and override `compose`."""

    BINDINGS = HARNESS_BINDINGS

    COMMAND_PALETTE_BINDING = "ctrl+backslash"
    """Textual binds its command palette to `ctrl+p` as a *priority system* binding,
    which silently wins over the app's own. Spec 8.1 gives `ctrl+p` to pause/resume,
    and a pause key that opens a search box is the kind of bug that looks like the
    controller ignoring you. The palette moves rather than being switched off: it is
    genuinely useful once M6 adds commands to it."""

    CSS = """
    Screen {
        layers: base overlay;
    }
    #event-log {
        height: 40%;
        dock: bottom;
        border-top: solid $primary;
        background: $surface;
    }
    """

    def __init__(
        self,
        controller: Controller,
        *,
        session_dir: str | Path | None = None,
        queue_size: int | None = None,
        install_approval_handler: bool = True,
        autostart: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.controller = controller
        self.autostart = autostart
        """Whether the app starts the run itself once it is mounted and subscribed.

        Off by default because a test starts the run when it is ready to observe it.
        A CLI wants it on, and it has to happen *after* `on_mount` has attached: a
        run started before the subscription exists spends its first events on nobody.
        `Controller.start()` is idempotent, so an app told to autostart a run that is
        already going does nothing."""
        self.session_dir = Path(session_dir) if session_dir is not None else None
        self.queue_size = queue_size
        self.router = EventRouter()
        self.approval_handler = TUIApprovalHandler(on_withdraw=self._on_withdraw)
        self._install_handler = install_approval_handler
        self._sub: Subscription | None = None
        self._pump: asyncio.Task[None] | None = None
        self._modal: ApprovalModal | None = None
        self._modal_request_id: str | None = None
        self._queued_approvals: list[ApprovalRequest] = []
        self._cancel_armed_at = 0.0
        self._interrupt_target: str | None = None
        self.events_seen = 0

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """The default single-agent layout. Subclasses replace this wholesale."""

        yield Transcript(agent_id=str(self.controller.main_agent), id=MAIN_TRANSCRIPT_ID)
        log = EventLog(id=EVENT_LOG_ID)
        log.display = False
        yield log
        yield RunStatusBar(self.controller, id=STATUS_BAR_ID)

    @property
    def status_bar(self) -> RunStatusBar | None:
        """The mounted status bar, if the layout has one."""

        found = self.query(RunStatusBar)
        return found.first(RunStatusBar) if found else None

    @property
    def event_log(self) -> EventLog | None:
        """The mounted event-log pane, if the layout has one."""

        found = self.query(EventLog)
        return found.first(EventLog) if found else None

    # -- attachment ---------------------------------------------------------

    async def on_mount(self) -> None:
        """Attach to the controller the app was constructed with."""

        await self.attach(self.controller)

    async def on_ready(self) -> None:
        """Start the run, if this app was asked to (`autostart`).

        Textual fires `on_ready` after `on_mount` and after the first paint, which is
        the first moment at which the subscription exists and a widget can be written
        to.
        """

        if self.autostart:
            await self.controller.start()

    async def attach(self, controller: Controller) -> None:
        """Bind to `controller`: install the handler, subscribe, seed, then follow.

        Idempotent in the sense that matters: attaching a second controller detaches
        the first cleanly, so `ctrl+o` is one call rather than a teardown sequence a
        subclass has to remember.
        """

        await self.detach()
        self.controller = controller
        if self._install_handler and controller.approval_handler is None:
            controller.set_approval_handler(self.approval_handler)

        self.refresh_consumers()
        bar = self.status_bar
        if bar is not None:
            bar.rebind(controller)

        self._sub = controller.bus.subscribe(maxsize=self.queue_size, name="tui")
        self.seed_from_controller(controller)
        self._pump = asyncio.create_task(self._drain(self._sub), name="tui:pump")

    async def detach(self) -> None:
        """Stop following the current controller. Safe to call when not attached."""

        if self._sub is not None:
            self._sub.unsubscribe()
            self._sub = None
        if self._pump is not None:
            self._pump.cancel()
            await asyncio.gather(self._pump, return_exceptions=True)
            self._pump = None

    def seed_from_controller(self, controller: Controller) -> None:
        """Draw what already exists before the first event arrives.

        A run that came back from `load()` has agents, a transcript, pending
        approvals and a state, and *no* events for any of it -- those belong to the
        process that wrote the session. A widget tree built from the stream alone
        shows an empty screen after `ctrl+o`.
        """

        for request in controller.pending_approvals:
            self.on_approval_requested(request)
        bar = self.status_bar
        if bar is not None:
            bar.refresh()

    def refresh_consumers(self) -> None:
        """Re-scan the widget tree and register every top-level `EventWidget`.

        Top-level only: a `Transcript` owns its child panes and forwards to them, so
        registering those children as well would deliver every delta twice.
        """

        self.router.clear()
        for widget in self._top_level_consumers():
            self.router.register(widget)  # type: ignore[arg-type]

    def _top_level_consumers(self) -> Iterator[Widget]:
        def walk(widget: Widget) -> Iterator[Widget]:
            for child in widget.children:
                if isinstance(child, EventWidget):
                    yield child
                else:
                    yield from walk(child)

        if self.is_mounted:
            yield from walk(self.screen)

    def register_consumer(self, widget: EventWidget) -> None:
        """Register a widget mounted after `attach()`."""

        self.router.register(widget)  # type: ignore[arg-type]

    def unregister_consumer(self, widget: EventWidget) -> None:
        """Stop delivering events to a widget."""

        self.router.unregister(widget)  # type: ignore[arg-type]

    # -- the pump -----------------------------------------------------------

    async def _drain(self, sub: Subscription) -> None:
        """Read the subscription until the bus closes. The only widget writer."""

        async for event in sub:
            self.dispatch(event)

    def dispatch(self, event: Event) -> None:
        """Handle one event at app level, then route it to the widgets."""

        self.events_seen += 1
        if isinstance(event, ApprovalRequested):
            self.on_approval_requested(event.request)
        elif isinstance(event, ApprovalResolved):
            self.approval_handler.resolved(event.request_id)
            self._dismiss_modal(event.request_id)
        self.router.dispatch(event)

    # -- approvals ----------------------------------------------------------

    def on_approval_requested(self, request: ApprovalRequest) -> None:
        """Show the request (R-U-6). Override to render it some other way.

        Overriding this method replaces the modal without replacing the handler:
        the gate is still parked on its future and the run is still in
        `WAITING_APPROVAL`, so an override that never calls `resolve_approval` is a
        hang. Call `self.resolve_approval(request_id, decision)` from whatever the
        override puts on screen.
        """

        if self._modal is not None:
            # One modal at a time. Concurrent agents can each be waiting; the queue
            # is drained as each is answered.
            if request.request_id != self._modal_request_id and not any(
                queued.request_id == request.request_id for queued in self._queued_approvals
            ):
                self._queued_approvals.append(request)
            return
        self._push_approval_modal(request)

    def _push_approval_modal(self, request: ApprovalRequest) -> None:
        modal = ApprovalModal(request)
        self._modal = modal
        self._modal_request_id = request.request_id

        def answered(decision: Decision | None) -> None:
            self._modal = None
            self._modal_request_id = None
            if decision is not None:
                self.resolve_approval(request.request_id, decision)
            self._drain_approval_queue()

        self.push_screen(modal, answered)

    def _drain_approval_queue(self) -> None:
        while self._queued_approvals:
            request = self._queued_approvals.pop(0)
            if request.request_id in self.approval_handler.pending:
                self._push_approval_modal(request)
                return

    def _dismiss_modal(self, request_id: str) -> None:
        """Take the modal down when the request was resolved somewhere else."""

        self._queued_approvals = [
            queued for queued in self._queued_approvals if queued.request_id != request_id
        ]
        if self._modal is not None and self._modal_request_id == request_id:
            modal, self._modal, self._modal_request_id = self._modal, None, None
            modal.dismiss(None)  # type: ignore[arg-type]

    def _on_withdraw(self, request_id: str, reason: str) -> None:
        """The gate withdrew a request. Called from the gate's task, never blocking."""

        self._dismiss_modal(request_id)
        self._notify(f"approval withdrawn: {reason}")

    def resolve_approval(self, request_id: str, decision: Decision) -> None:
        """Send a decision back to the controller, off the message pump."""

        self.run_worker(
            self.controller.resolve_approval(request_id, decision),
            name="resolve-approval",
            group="controller",
        )

    # -- hooks a subclass overrides -----------------------------------------

    def interrupt_target(self) -> str | None:
        """Which agent `escape` interrupts (spec C-4).

        `None` means the controller's default, which is `main`. A fan-out workflow
        has no `main` agent, so a targetless interrupt there cancels nothing and
        emits `RunWarning(code="interrupt_no_target")` -- which this app renders
        rather than swallowing, but which is still not what the user meant. Fusion's
        app overrides this to name the selected pane's agent.
        """

        return None

    def injection_input(self) -> Widget | None:
        """The widget `escape` focuses so the user can type a message (spec 8.1).

        The default finds a mounted `PromptInput`, so a subclass that composes one
        gets the whole `escape` flow -- interrupt, focus, type, enter -- without
        overriding anything. With no input at all the flow is the cancel half only:
        the step is interrupted and the run continues, which is R-C-4's behaviour
        without the optional message.
        """

        found = self.query(PromptInput)
        return found.first(PromptInput) if found else None

    def send_target(self) -> str | None:
        """Which agent `send()` delivers to. `None` means the controller's default.

        The same hook as `interrupt_target()` and for the same reason (spec C-4): a
        workflow whose main agent is not called `main` -- or which has no agents at
        all, as fusion does -- has to say where a typed message goes.
        """

        return self.interrupt_target()

    def session_open_path(self) -> Path | None:
        """Where `ctrl+o` loads from.

        Defaults to the controller's own `session_path`, then `session_dir/
        session.json`. A subclass with a file picker returns the chosen path.
        """

        if self.controller.session_path is not None:
            return self.controller.session_path
        if self.session_dir is not None:
            return self.session_dir / "session.json"
        return None

    def session_save_path(self) -> Path | None:
        """Where `ctrl+s` writes. `None` lets the controller choose its own."""

        if self.session_dir is not None:
            return self.session_dir / "session.json"
        return None

    # -- actions ------------------------------------------------------------

    def _notify(self, message: str, *, style: str = "bold yellow") -> None:
        bar = self.status_bar
        if bar is not None:
            bar.set_notice(message, style=style)

    def action_toggle_pause(self) -> None:
        """`ctrl+p`: pause a running run, resume a paused one (R-C-3, R-C-1)."""

        self.run_worker(self._toggle_pause(), name="toggle-pause", group="controller")

    async def _toggle_pause(self) -> None:
        controller = self.controller
        if controller.state in (RunState.PAUSED, RunState.PAUSING):
            await controller.resume()
        else:
            await controller.pause()
            if controller.state is RunState.PAUSING:
                self._notify(f"pausing — waiting for {controller.blocking_description()}")

    def action_interrupt(self) -> None:
        """`escape`: cancel the target agent's current step, then offer injection."""

        self._interrupt_target = self.interrupt_target()
        self.run_worker(self._interrupt(None), name="interrupt", group="controller")
        widget = self.injection_input()
        if widget is not None:
            if isinstance(widget, PromptInput):
                widget.set_interrupt_mode(True)
            widget.focus()

    def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
        """Route a `PromptInput` submission to `send` or to `submit_injection`.

        Spec 8.2's two verbs, wired once here rather than in each workflow's app: a
        subclass that mounts a `PromptInput` gets both behaviours without writing a
        handler, the same way it gets the approval modal (R-U-6).
        """

        message.stop()
        if message.interrupt:
            self.submit_injection(message.text)
        else:
            self.send(message.text)

    def send(self, text: str) -> None:
        """Queue a message for the agent without cancelling its current step.

        The normal-mode half of spec 8.2's `PromptInput` contract. Empty text is a
        no-op: there is no interrupt to pair it with.
        """

        if not text.strip():
            return
        self.run_worker(self._send(text), name="send", group="controller")

    async def _send(self, text: str) -> None:
        message_id = await self.controller.send(text, target=self.send_target())
        if message_id is None:
            self._notify("message not delivered: no such agent", style="bold red")

    def submit_injection(self, text: str) -> None:
        """Inject `text` into the agent the last `escape` targeted (R-C-4).

        Empty text is spec 8.1's "enter with empty input = interrupt without
        message": the interrupt already happened on `escape`, so there is nothing
        left to do.
        """

        if not text.strip():
            return
        self.run_worker(self._interrupt(text), name="inject", group="controller")

    async def _interrupt(self, message: str | None) -> None:
        result = await self.controller.interrupt(message, target=self._interrupt_target)
        if result.warning:
            self._notify(result.warning)
        elif message is not None and not result.queued:
            self._notify("message not delivered: no such agent", style="bold red")

    def action_toggle_permission_mode(self) -> None:
        """`ctrl+t`: switch between `manual` and `auto` (R-C-5)."""

        self.run_worker(self._toggle_mode(), name="toggle-mode", group="controller")

    async def _toggle_mode(self) -> None:
        controller = self.controller
        target = (
            PermissionMode.AUTO
            if controller.permission_mode is PermissionMode.MANUAL
            else PermissionMode.MANUAL
        )
        # Through the public method: it holds `_command_lock` while auto-resolving
        # pending requests, and reaching into the gate from here would deadlock.
        resolved = await controller.set_permission_mode(target)
        if resolved:
            self._notify(f"{target.value}: {resolved} pending request(s) approved")
        else:
            self._notify(f"permission mode: {target.value}")

    def action_save_session(self) -> None:
        """`ctrl+s`: write the session, or say what is blocking it (R-C-10, C-12)."""

        self.run_worker(self._save(), name="save", group="controller")

    async def _save(self) -> None:
        try:
            path = await self.controller.save(self.session_save_path())
        except SaveTimeout as exc:
            self._notify(f"save is waiting for {exc.blocking}")
        except (CheckpointError, HarnessError, OSError) as exc:
            self._notify(f"save failed: {exc}", style="bold red")
        else:
            self._notify(f"saved {path}", style="green")

    def action_open_session(self) -> None:
        """`ctrl+o`: load a session and attach to the controller it produces."""

        self.run_worker(self._open(), name="open", group="controller")

    async def _open(self) -> None:
        path = self.session_open_path()
        if path is None:
            self._notify("no session path: pass session_dir, or override session_open_path")
            return
        try:
            controller = await Controller.load(
                path,
                bus=EventBus(),
                approval_handler=self.approval_handler,
            )
        except (HarnessError, OSError, ValueError) as exc:
            self._notify(f"open failed: {exc}", style="bold red")
            return
        await self.attach(controller)
        report = controller.resume_report
        self._notify(
            f"loaded {path.name}: {report.summary()}" if report else f"loaded {path.name}",
            style="green",
        )

    def action_toggle_event_log(self) -> None:
        """`ctrl+l`: show or hide the event-log pane."""

        log = self.event_log
        if log is None:
            self._notify("this layout has no event log")
            return
        log.display = not log.display

    def action_request_cancel(self) -> None:
        """`ctrl+c`: warn once, cancel the run on a second press (spec 8.1)."""

        now = time.monotonic()
        if now - self._cancel_armed_at <= DOUBLE_PRESS_WINDOW_S:
            self._cancel_armed_at = 0.0
            self.run_worker(self._cancel(), name="cancel", group="controller")
            return
        self._cancel_armed_at = now
        self._notify("press ctrl+c again to cancel the run", style="bold red")

    async def _cancel(self) -> None:
        await self.controller.cancel("cancelled from the TUI")
        self._notify("run cancelled", style="bold red")

    # -- shutdown -----------------------------------------------------------

    async def on_unmount(self) -> None:
        """Stop the pump. The controller is not cancelled: quitting the UI is not
        cancelling the run, and `ctrl+c` twice is how a user says otherwise."""

        await self.detach()


__all__ = [
    "EVENT_LOG_ID",
    "MAIN_TRANSCRIPT_ID",
    "STATUS_BAR_ID",
    "EventLog",
    "HarnessApp",
]
