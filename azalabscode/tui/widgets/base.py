"""What every event-consuming widget shares.

A mixin rather than a base class, and one with no `__init__`, so it composes with
whatever Textual widget a consumer wants to be (`Static`, `VerticalScroll`, a
`Screen`) without fighting that class's constructor signature.
"""

from __future__ import annotations

from azalabscode.events import Event
from azalabscode.tui.routing import ANY


class EventWidget:
    """Mixin giving a widget the two filters and the callback the router needs.

    Set `agent_filter` / `node_filter` in the widget's own `__init__` after calling
    `super().__init__(...)`. They default to `ANY`, which is right for run-level
    widgets (the status bar) and wrong for per-agent ones, so every per-agent widget
    sets them explicitly.
    """

    agent_filter: str = ANY
    node_filter: str = ANY

    def handle_event(self, event: Event) -> None:
        """Consume one event.

        Called from the app's pump on the event loop thread, synchronously. It must
        not await and it must not block: the pump is what keeps every other widget
        current, and a widget that sleeps here stalls the whole UI. Buffer instead,
        and flush on a timer -- `StreamPane` is the worked example.
        """

        raise NotImplementedError


__all__ = ["EventWidget"]
