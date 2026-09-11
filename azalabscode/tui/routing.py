"""Event routing: which widget sees which event (R-U-2).

Spec 8.1 says the app "routes each event to widgets registered for
`(agent_id | "*", node_id | "*")`". That is all this module is, and it is
deliberately free of any Textual import: if M4's perf test had gone the other way
(plan decision D5), the Rich-based render loop would have kept this file and
replaced everything around it.

Two rules the router enforces that a widget cannot enforce for itself:

* **A raising widget must not take the pump down.** One misbehaving consumer would
  otherwise stop every other widget from ever seeing another event, and the app
  would look frozen rather than broken. Exceptions are counted and recorded on the
  router so the app can surface them; `BaseException` is never caught, so a
  `CancelledError` still unwinds the pump task the way it must.
* **Registration order is delivery order**, so a status bar registered before a
  transcript is updated first, and a widget added mid-run is served last.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from azalabscode.events import Event

ANY = "*"
"""Filter value meaning "every agent" or "every node"."""


@runtime_checkable
class EventConsumer(Protocol):
    """What the router needs from a widget.

    Structural on purpose: a consumer does not have to inherit from anything in
    this package, which is what lets a workflow ship its own widget at M6 without
    importing a base class it does not otherwise need.
    """

    agent_filter: str
    """An `agent_id`, or `ANY`."""

    node_filter: str
    """A `node_id`, or `ANY`."""

    def handle_event(self, event: Event) -> None:
        """Consume one event. Must not block; must not await."""
        ...


def matches(consumer: EventConsumer, event: Event) -> bool:
    """Whether `consumer` is registered for `event`'s `(agent_id, node_id)`.

    An event with no `agent_id` (a run-state change, a checkpoint) reaches only
    consumers filtering on `ANY`: a transcript bound to `main` has nothing to do
    with a run-level event, and a status bar filters on `ANY` precisely so that it
    does.
    """

    if consumer.agent_filter != ANY and consumer.agent_filter != (event.agent_id or ""):
        return False
    return consumer.node_filter == ANY or consumer.node_filter == (event.node_id or "")


class RouterError(Exception):
    """One consumer raised while handling one event. Recorded, never propagated."""

    def __init__(self, consumer: object, event: Event, cause: BaseException) -> None:
        self.consumer = consumer
        self.event = event
        self.cause = cause
        super().__init__(f"{type(consumer).__name__} raised on {event.type}: {cause!r}")


class EventRouter:
    """Fan-out of events to registered consumers, filtered by agent and node."""

    def __init__(self) -> None:
        self._consumers: list[EventConsumer] = []
        self.delivered = 0
        """Consumer callbacks made. Counts deliveries, not events."""
        self.seen = 0
        """Events routed, whether or not anything was registered for them."""
        self.errors: list[RouterError] = []
        """Every failure, newest last. Bounded by `MAX_ERRORS`."""

    MAX_ERRORS = 100

    def __len__(self) -> int:
        return len(self._consumers)

    def __iter__(self) -> Iterator[EventConsumer]:
        return iter(list(self._consumers))

    def register(self, consumer: EventConsumer) -> None:
        """Add a consumer. Registering the same object twice is a no-op."""

        if consumer not in self._consumers:
            self._consumers.append(consumer)

    def unregister(self, consumer: EventConsumer) -> None:
        """Remove a consumer. Unregistering an unknown object is a no-op."""

        if consumer in self._consumers:
            self._consumers.remove(consumer)

    def clear(self) -> None:
        """Drop every consumer. Used when the app rebinds to a new controller."""

        self._consumers.clear()

    def dispatch(self, event: Event) -> int:
        """Deliver `event` to every matching consumer. Returns the delivery count."""

        self.seen += 1
        count = 0
        for consumer in list(self._consumers):
            if not matches(consumer, event):
                continue
            try:
                consumer.handle_event(event)
            except Exception as exc:  # a widget bug must not stop the stream
                if len(self.errors) >= self.MAX_ERRORS:
                    del self.errors[0]
                self.errors.append(RouterError(consumer, event, exc))
                continue
            count += 1
        self.delivered += count
        return count


__all__ = ["ANY", "EventConsumer", "EventRouter", "RouterError", "matches"]
