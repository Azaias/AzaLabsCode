"""The coding agent's TUI (R-A-1, spec 9.1).

    ┌ transcript (selected agent) ─────┬ agents ──────┐
    │ streamed output, collapsed tools │ main         │
    │                                  │ ↳ main/0     │
    │                                  ├ tool calls ──┤
    │        ┌ diff overlay ┐          │ edit_file …  │
    │        └──────────────┘          │ grep …       │
    ├──────────────────────────────────┴──────────────┤
    │ prompt                                          │
    │ RUNNING │ manual │ agents 2/2 │ 4.1k tok │ ...  │
    └─────────────────────────────────────────────────┘

Spec 9.1's list: transcript, streamed output, diff view on edits, approval modal,
subagent tree, status bar, and the keybindings. The modal and the bindings come free
from `HarnessApp` (R-U-6); this file is the layout plus three pieces of wiring.

**A transcript per agent, switched by the tree.** Spec 8.2 says selecting an agent
focuses its `Transcript`, which means there has to be one. They are created when
`AgentSpawned` arrives and registered with the router by hand -- `refresh_consumers()`
only walks the tree that existed at `attach()` time, so a widget mounted mid-run has
to say so or it silently receives nothing.

**The diff view is an overlay that appears when there is a diff.** An edit is the one
thing in a coding session that is worth interrupting the reading for, and a docked
diff pane would spend most of a session empty. `ctrl+d` toggles it. It is an app-level
binding, so it does not work while the approval modal is up -- which is fine, because
the modal shows the diff itself.

**The prompt box is the session's mouth.** `PromptInput` posts, `HarnessApp` routes to
`send` or `submit_injection`, and both are overridden here to go through
`CodingSession` so the node is woken as well as the controller told.
"""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical

from azalabscode import AgentSpawned, Controller
from azalabscode.tui import (
    HARNESS_BINDINGS,
    AgentTree,
    DiffView,
    EventLog,
    HarnessApp,
    PromptInput,
    RunStatusBar,
    ToolCallDetail,
    ToolCallList,
    Transcript,
)
from workflows.coding_agent.session import CodingSession
from workflows.coding_agent.workflow import AGENT_ID


class CodingAgentApp(HarnessApp):
    """Spec 9.1's layout over a `CodingSession`."""

    BINDINGS: ClassVar[list[BindingType]] = [
        *HARNESS_BINDINGS,
        Binding("ctrl+d", "toggle_diff", "Diff", show=True),
        Binding("ctrl+g", "toggle_detail", "Call detail", show=False),
    ]

    CSS = (
        HarnessApp.CSS
        + """
    #body {
        height: 1fr;
    }
    #left {
        width: 62%;
        border-right: solid $primary;
    }
    #right {
        width: 38%;
    }
    #agents {
        height: 40%;
        border-bottom: solid $primary;
    }
    #calls {
        height: 1fr;
    }
    #detail {
        height: 50%;
        display: none;
        border-top: solid $primary;
    }
    #diff {
        layer: overlay;
        width: 70%;
        height: 60%;
        offset: 8 4;
        background: $surface;
        display: none;
    }
    #prompt {
        dock: bottom;
    }
    """
    )

    def __init__(self, session: CodingSession, **kwargs: Any) -> None:
        super().__init__(session.controller, **kwargs)
        self.session = session
        self.transcripts: dict[str, Transcript] = {}
        self.selected_agent = AGENT_ID

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="left"):
                main = Transcript(agent_id=AGENT_ID, id="transcript-main")
                self.transcripts[AGENT_ID] = main
                yield main
            with Vertical(id="right"):
                yield AgentTree(label=self.session.config.model, id="agents")
                yield ToolCallList(id="calls")
                yield ToolCallDetail(id="detail")
        yield DiffView(id="diff")
        yield PromptInput(id="prompt")
        log = EventLog(id="event-log")
        log.display = False
        yield log
        yield RunStatusBar(self.controller, id="status-bar")

    @property
    def agent_tree(self) -> AgentTree:
        """The subagent tree (R-U-5).

        Not `tree`: `DOMNode.tree` is Textual's own debugging view of the widget
        hierarchy, and shadowing it would break `App.tree` for every caller.
        """

        return self.query_one("#agents", AgentTree)

    @property
    def calls(self) -> ToolCallList:
        """The tool-call table."""

        return self.query_one("#calls", ToolCallList)

    @property
    def detail(self) -> ToolCallDetail:
        """The tool-call drill-down."""

        return self.query_one("#detail", ToolCallDetail)

    @property
    def diff(self) -> DiffView:
        """The diff overlay."""

        return self.query_one("#diff", DiffView)

    @property
    def prompt(self) -> PromptInput:
        """The prompt box."""

        return self.query_one("#prompt", PromptInput)

    async def on_mount(self) -> None:
        """Attach, then put the cursor in the prompt box."""

        await super().on_mount()
        self.call_after_refresh(self.prompt.focus)

    # -- events -------------------------------------------------------------

    def dispatch(self, event: Any) -> None:
        """Give every new agent a transcript, and open the diff view on an edit."""

        if isinstance(event, AgentSpawned) and event.agent_id:
            self.ensure_transcript(event.agent_id)
        view = self._diff_view()
        before = len(view.history) if view is not None else 0
        super().dispatch(event)
        if view is not None and len(view.history) > before:
            view.display = True

    def _diff_view(self) -> DiffView | None:
        """The mounted diff overlay, if the layout still has one."""

        found = self.query(DiffView)
        return found.first(DiffView) if found else None

    def ensure_transcript(self, agent_id: str) -> Transcript:
        """A `Transcript` for `agent_id`, mounted hidden and registered by hand."""

        existing = self.transcripts.get(agent_id)
        if existing is not None:
            return existing
        transcript = Transcript(agent_id=agent_id, subagent=agent_id != AGENT_ID)
        transcript.display = agent_id == self.selected_agent
        self.transcripts[agent_id] = transcript
        self.query_one("#left", Vertical).mount(transcript)
        self.register_consumer(transcript)
        return transcript

    def show_agent(self, agent_id: str) -> None:
        """Focus one agent's transcript (spec 8.2)."""

        if agent_id not in self.transcripts:
            return
        self.selected_agent = agent_id
        for other, transcript in self.transcripts.items():
            transcript.display = other == agent_id

    def on_agent_tree_agent_selected(self, message: AgentTree.AgentSelected) -> None:
        """Tree selection switches the transcript on the left."""

        message.stop()
        if message.agent_id:
            self.show_agent(message.agent_id)

    def on_tool_call_list_selected(self, message: ToolCallList.Selected) -> None:
        """Keep the drill-down on whatever the table's cursor is on."""

        message.stop()
        self.detail.show(message.record)

    def seed_from_controller(self, controller: Controller) -> None:
        """Draw the agents a loaded run already has (they have no events)."""

        super().seed_from_controller(controller)
        if not self.is_mounted:
            return
        self.agent_tree.seed(controller.agents)
        for agent_id in controller.agents:
            self.ensure_transcript(agent_id)

    # -- the two verbs ------------------------------------------------------

    def send(self, text: str) -> None:
        """Normal-mode submit: inject without cancelling, and wake the session."""

        if not text.strip():
            return
        self.run_worker(self._send_to_session(text), name="send", group="controller")

    async def _send_to_session(self, text: str) -> None:
        if await self.session.send(text) is None:
            self._notify("message not delivered: the session is not running", style="bold red")

    def submit_injection(self, text: str) -> None:
        """Post-`escape` submit: the interrupt already happened; deliver the message."""

        if not text.strip():
            return
        self.run_worker(self._inject_to_session(text), name="inject", group="controller")

    async def _inject_to_session(self, text: str) -> None:
        await self.session.interrupt(text)

    def interrupt_target(self) -> str | None:
        """`escape` targets the main agent; a subagent is interrupted by selecting it."""

        return self.selected_agent or AGENT_ID

    # -- actions ------------------------------------------------------------

    def action_toggle_diff(self) -> None:
        """`ctrl+d`: show or hide the diff overlay."""

        view = self.diff
        if view.empty:
            self._notify("no edits yet")
            return
        view.display = not view.display

    def action_toggle_detail(self) -> None:
        """`ctrl+g`: show or hide the tool-call drill-down."""

        detail = self.detail
        detail.display = not detail.display
        if detail.display:
            detail.show(self.calls.selected)

    async def on_unmount(self) -> None:
        """Close the session so the run can finish, then stop the pump.

        Quitting the UI is not cancelling the run (that is `ctrl+c` twice), but an
        interactive session waits for a prompt that is never coming once the terminal
        is gone, and a process that cannot exit is worse than a run that stops.
        """

        self.session.close()
        await super().on_unmount()


__all__ = ["CodingAgentApp"]
