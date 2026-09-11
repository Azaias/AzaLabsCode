"""`DiffView`: the unified diff of an edit, highlighted by file type (R-U-3).

Two events put a diff on screen and they arrive at opposite ends of a decision:
`ApprovalRequested` carries the diff of an edit that has *not happened yet* (the
tool builds it in `approval_summary`), and `ToolCallCompleted` carries the diff of
one that has, in `ToolResult.display` (spec delta 12). The widget shows either, and
labels which, because "about to happen" and "happened" are not the same thing to
look at and the diff text alone does not say.

The colouring is two layers. The gutter -- `+`, `-`, the hunk headers -- is coloured
by the diff's own grammar, which is what tells you whether a line is arriving or
leaving. The body of each line is handed to Pygments through Rich's `Syntax`, with
the lexer guessed from the path's extension, which is what makes a 40-line hunk of
Python readable. `ApprovalModal.render_diff` does the first layer only and stays
that way: a modal is a decision, and a decision does not need a syntax highlighter.

A lexer that Pygments does not have, or a path with no extension, falls back to the
gutter-only rendering rather than failing. Nothing about a diff should be able to
take the UI down.
"""

from __future__ import annotations

from typing import Any

from rich.syntax import Syntax
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from azalabscode.events import (
    ApprovalRequested,
    Event,
    ToolCallCompleted,
)
from azalabscode.tui.routing import ANY
from azalabscode.tui.widgets.base import EventWidget

DIFF_TOOLS = frozenset({"edit_file", "write_file"})
"""Tools whose results carry a diff. Spec 8.2 names both."""

MAX_HISTORY = 50
"""Diffs kept for `previous()`/`next()`."""

MAX_DIFF_CHARS = 200_000
"""A diff larger than this is truncated before it is rendered. A whole-file rewrite
of a generated file is a legitimate edit and an unreadable diff; the widget's job is
to stay responsive, and the full text is in the transcript and the session."""

THEME = "ansi_dark"
"""Rich's terminal-palette theme. A fixed colour scheme would fight the app's."""


def syntax_for(path: str | None) -> Syntax | None:
    """A `Syntax` whose lexer matches `path`'s extension, or `None`.

    Built once per diff and reused per line: constructing one per line makes
    Pygments guess the lexer hundreds of times for one hunk.
    """

    if not path:
        return None
    try:
        lexer = Syntax.guess_lexer(path)
        if not lexer or lexer == "default":
            return None
        return Syntax("", lexer, theme=THEME)
    except Exception:  # pragma: no cover - a Pygments lookup failure is not a UI error
        return None


def highlight_diff(diff: str, path: str | None = None) -> Text:
    """Render a unified diff: coloured gutter, syntax-highlighted body."""

    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n[...diff truncated...]"
    syntax = syntax_for(path)
    out = Text(no_wrap=False)
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            out.append(line + "\n", style="bold")
        elif line.startswith("@@"):
            out.append(line + "\n", style="cyan")
        elif line[:1] in {"+", "-", " "} and line:
            prefix, body = line[0], line[1:]
            style = {"+": "bold green", "-": "bold red", " ": "dim"}[prefix]
            out.append(prefix, style=style)
            out.append_text(_body(body, syntax, prefix))
            out.append("\n")
        else:
            out.append(line + "\n", style="dim")
    return out


def _body(body: str, syntax: Syntax | None, prefix: str) -> Text:
    """One line's payload, highlighted when there is a lexer for it."""

    if syntax is None:
        return Text(body, style={"+": "green", "-": "red"}.get(prefix, ""))
    try:
        fragment = syntax.highlight(body)
    except Exception:  # pragma: no cover - a lexer that raises on one line
        return Text(body)
    fragment.rstrip()
    return fragment


class DiffView(EventWidget, VerticalScroll):
    """The most recent edit, as a diff (R-U-3).

    Consumes `ApprovalRequested` (a pending edit) and `ToolCallCompleted` for
    `edit_file` and `write_file` (a completed one).

    Mount it in a layout as a pane, or `display = False` it and let the app show it
    on demand -- the coding agent overlays it, R-A-3 docks it. Either way the widget
    keeps folding events, so an overlay opened after an edit already has the diff.
    """

    DEFAULT_CSS = """
    DiffView {
        height: 1fr;
        padding: 0 1;
        border: round $primary;
    }
    DiffView.-pending {
        border: round $warning;
    }
    """

    def __init__(
        self,
        *,
        agent_id: str = ANY,
        node_id: str = ANY,
        max_history: int = MAX_HISTORY,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.agent_filter = agent_id
        self.node_filter = node_id
        self.max_history = max_history
        self.history: list[tuple[str, str, bool]] = []
        """`(path, diff, pending)` newest last."""
        self.index = -1
        """Which entry is shown. `-1` is "nothing yet"."""
        self._body = Static("", markup=False)

    def compose(self) -> Any:
        """One `Static`; the whole diff is rebuilt whenever it changes."""

        yield self._body

    def on_mount(self) -> None:
        """Draw whatever arrived before the widget was mounted."""

        self.redraw()

    # -- content ------------------------------------------------------------

    @property
    def current(self) -> tuple[str, str, bool] | None:
        """The `(path, diff, pending)` on screen."""

        if 0 <= self.index < len(self.history):
            return self.history[self.index]
        return None

    @property
    def empty(self) -> bool:
        """Whether any diff has been seen."""

        return not self.history

    def show_diff(self, path: str, diff: str, *, pending: bool = False) -> None:
        """Add a diff and show it. Called by `handle_event`, and directly by tests."""

        self.history.append((path, diff, pending))
        del self.history[: max(0, len(self.history) - self.max_history)]
        self.index = len(self.history) - 1
        self.redraw()

    def previous(self) -> None:
        """Step back through the history."""

        if self.index > 0:
            self.index -= 1
            self.redraw()

    def next(self) -> None:
        """Step forward through the history."""

        if self.index < len(self.history) - 1:
            self.index += 1
            self.redraw()

    def clear(self) -> None:
        """Drop everything. Used when the app rebinds to another run."""

        self.history.clear()
        self.index = -1
        self.redraw()

    def redraw(self) -> None:
        """Rebuild the rendered diff from `current`."""

        self._body.update(self.render_diff())
        entry = self.current
        self.set_class(bool(entry and entry[2]), "-pending")
        if entry is not None:
            position = f"{self.index + 1}/{len(self.history)}"
            self.border_title = f"{'pending ' if entry[2] else ''}{entry[0]}  {position}"
        else:
            self.border_title = "diff"

    def render_diff(self) -> Text:
        """The rendered diff. Separated out so a test can read it without a screen."""

        entry = self.current
        if entry is None:
            return Text("no edits yet", style="dim")
        path, diff, _pending = entry
        if not diff:
            return Text(f"{path}: created (no previous content to diff)", style="dim")
        return highlight_diff(diff, path)

    # -- events -------------------------------------------------------------

    def handle_event(self, event: Event) -> None:
        """Show the diff an approval or a completed edit carries."""

        if isinstance(event, ApprovalRequested):
            summary = event.request.summary
            if summary.diff:
                self.show_diff(
                    str(event.request.params.get("path", event.request.tool)),
                    summary.diff,
                    pending=True,
                )
        elif isinstance(event, ToolCallCompleted):
            if event.tool not in DIFF_TOOLS:
                return
            display = event.result.display
            if display is None or display.kind != "diff":
                return
            diff = display.data.get("diff")
            if isinstance(diff, str) and diff:
                self.show_diff(str(display.data.get("path", event.tool)), diff)


__all__ = [
    "DIFF_TOOLS",
    "MAX_DIFF_CHARS",
    "DiffView",
    "highlight_diff",
    "syntax_for",
]
