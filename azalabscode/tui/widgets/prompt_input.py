"""`PromptInput`: the multi-line box the user types into (R-U-3, spec 8.1).

Spec 8.2: "submits `send(text)` in normal mode, `interrupt(text)` after `escape`."
So the widget has two modes and it says which one it is in, because the difference
between "the agent will read this after its current step" and "the agent's current
step is being cancelled to read this" is not one to leave implicit.

`enter` submits and `alt+enter` inserts a newline. That is the opposite of a
`TextArea`'s default and it is the right way round for a prompt: most prompts are
one line, and a box that needs a key combination to send is a box that gets typed
into and abandoned. The multi-line half still works, which is what R-U-3 asks for --
pasting a stack trace in and sending it is the case that matters.

`escape` leaves the box rather than re-firing the app's interrupt binding. A
`TextArea` swallows most keys, so without this the only way out of the prompt is the
mouse.

The widget posts `PromptInput.Submitted`; it never touches the controller. The app
decides what a submission means, which is what keeps R-U-1 true of the whole layer.
"""

from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.message import Message
from textual.widgets import TextArea

SEND_PLACEHOLDER = "message the agent — enter to send, alt+enter for a newline"
INTERRUPT_PLACEHOLDER = "interrupted — enter to send with the interrupt, escape to leave"


class PromptInput(TextArea):
    """A multi-line prompt box with a send mode and an interrupt mode (R-U-3)."""

    DEFAULT_CSS = """
    PromptInput {
        height: auto;
        max-height: 8;
        border: round $primary;
        padding: 0 1;
    }
    PromptInput.-interrupt {
        border: round $warning;
    }
    """

    BINDINGS: ClassVar[list[object]] = []

    class Submitted(Message):
        """The user pressed enter. `interrupt` says which mode the box was in."""

        def __init__(self, text: str, *, interrupt: bool) -> None:
            self.text = text
            self.interrupt = interrupt
            super().__init__()

    class Escaped(Message):
        """The user pressed escape inside the box and left it."""

    def __init__(self, *, placeholder: str = SEND_PLACEHOLDER, **kwargs: object) -> None:
        super().__init__(soft_wrap=True, tab_behavior="focus", **kwargs)  # type: ignore[arg-type]
        self.send_placeholder = placeholder
        self.placeholder = placeholder
        self.interrupt_mode = False
        """True between an `escape` interrupt and the next submission (spec 8.1)."""
        self.submissions = 0
        """How many times the box has been submitted. For tests."""

    # -- modes --------------------------------------------------------------

    def set_interrupt_mode(self, on: bool) -> None:
        """Switch the box between `send` and `interrupt` (spec 8.1's `escape` flow)."""

        self.interrupt_mode = on
        self.set_class(on, "-interrupt")
        self.placeholder = INTERRUPT_PLACEHOLDER if on else self.send_placeholder
        self.border_title = "interrupt" if on else ""

    # -- keys ---------------------------------------------------------------

    async def _on_key(self, event: events.Key) -> None:
        """`enter` submits, `alt+enter` is a newline, `escape` leaves the box.

        Overriding the private handler is deliberate: `TextArea._on_key` claims
        `enter` for a newline insert and calls `prevent_default()`, so a binding
        never sees it. Everything this method does not claim goes straight back to
        the base implementation.
        """

        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.submit()
            return
        if event.key in {"alt+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.set_interrupt_mode(False)
            self.post_message(self.Escaped())
            self.screen.set_focus(None)
            return
        await super()._on_key(event)

    # -- submission ---------------------------------------------------------

    def submit(self) -> None:
        """Post the current text and clear the box.

        An empty box still posts, because spec 8.1 gives it a meaning: "enter with
        empty input = interrupt without message". The app is where that is decided.
        """

        text = self.text
        interrupt = self.interrupt_mode
        self.clear()
        self.set_interrupt_mode(False)
        self.submissions += 1
        self.post_message(self.Submitted(text, interrupt=interrupt))


__all__ = ["INTERRUPT_PLACEHOLDER", "SEND_PLACEHOLDER", "PromptInput"]
