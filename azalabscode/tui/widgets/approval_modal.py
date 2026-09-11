"""`ApprovalModal`: the one screen a human has to read before something irreversible.

R-U-6 is the requirement: a workflow author writes no UI code and still gets a
usable approval prompt. `HarnessApp` pushes this screen when `ApprovalRequested`
arrives, and the decision goes back through `Controller.resolve_approval` -- not
through the handler's return value, because the modal outlives the `request()` call
by design (the handler returns `None`, the gate parks on its future).

Spec 8.1 lists what the modal must show: tool name, a human summary, params
(collapsible), a diff for file edits, and the agent id. All five are here. The diff
is rendered inline with per-line colour rather than by a `DiffView`, which is an M6
widget: an approval a user cannot read is an approval they will rubber-stamp, and
that is worth thirty lines now rather than a cross-reference to a widget that does
not exist yet.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Collapsible, Label, Static

from azalabscode.permissions import ApprovalRequest, Decision
from azalabscode.tui.bindings import forwarded_to_app

DENIED_AT_MODAL = "denied at the approval prompt"
"""Reason attached to a `n`/`escape` denial. It reaches the model verbatim as a
`denied`-kind tool error, so it is written for the model, not for a log."""


def render_diff(diff: str) -> Text:
    """Colour a unified diff by line prefix.

    Deliberately not a syntax highlighter. The question the user is answering is
    "should this change happen", and the answer comes from which lines are added and
    removed, not from how the language colours its keywords.
    """

    text = Text(no_wrap=False)
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            text.append(line + "\n", style="bold")
        elif line.startswith("@@"):
            text.append(line + "\n", style="cyan")
        elif line.startswith("+"):
            text.append(line + "\n", style="green")
        elif line.startswith("-"):
            text.append(line + "\n", style="red")
        else:
            text.append(line + "\n", style="dim")
    return text


class ApprovalModal(ModalScreen[Decision]):
    """A single pending `ApprovalRequest`, answered with `y` or `n`.

    Dismisses with a `Decision`. `escape` denies rather than cancelling the screen:
    a modal that can be dismissed without answering leaves the gate parked forever,
    which is a hang with a very confusing cause.
    """

    BINDINGS: ClassVar[list[Any]] = [
        Binding("y", "approve", "Approve", show=True),
        Binding("n", "deny", "Deny", show=True),
        Binding("escape", "deny", "Deny", show=False),
        # Spec 8.1's table, borrowed. A modal stops binding lookup, and the state a
        # user most wants to switch mode or save from is the one with a modal up.
        # `escape` is dropped: here it denies.
        *forwarded_to_app(drop=frozenset({"escape"})),
    ]

    DEFAULT_CSS = """
    ApprovalModal {
        align: center middle;
        background: $background 60%;
    }
    ApprovalModal > Vertical {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    ApprovalModal .-title {
        text-style: bold;
    }
    ApprovalModal .-danger {
        color: $error;
        text-style: bold;
    }
    ApprovalModal .-agent {
        color: $text-muted;
    }
    ApprovalModal .-keys {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, request: ApprovalRequest, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.request = request

    def compose(self) -> ComposeResult:
        """Tool, summary, agent, params, diff, keys -- spec 8.1's list, in order."""

        summary = self.request.summary
        with Vertical():
            yield Label(summary.title or self.request.tool, classes="-title")
            if summary.danger:
                yield Label("irreversible", classes="-danger")
            yield Label(
                f"agent {self.request.agent_id} · call {self.request.call_id}",
                classes="-agent",
            )
            if summary.detail:
                yield Static(Text(summary.detail), markup=False)
            with VerticalScroll():
                if summary.diff:
                    yield Static(render_diff(summary.diff), markup=False, id="approval-diff")
                # A diff is the thing to read when there is one, so params start
                # collapsed; with no diff the params *are* the request.
                with Collapsible(title="params", collapsed=summary.diff is not None):
                    yield Static(
                        Text(json.dumps(self.request.params, indent=2, default=str)),
                        markup=False,
                        id="approval-params",
                    )
            yield Label("[y] approve   [n] deny", classes="-keys")

    def action_approve(self) -> None:
        """Allow the call."""

        self.dismiss(Decision.approve(by="tui"))

    def action_deny(self) -> None:
        """Refuse the call, with a reason the model will see."""

        self.dismiss(Decision.deny(DENIED_AT_MODAL, by="tui"))


__all__ = ["DENIED_AT_MODAL", "ApprovalModal", "render_diff"]
