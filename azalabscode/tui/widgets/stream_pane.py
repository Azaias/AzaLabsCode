"""`StreamPane`: one model call's streamed output, coalesced onto the frame.

This is the widget R-U-4 is about. Core never coalesces on the emit path -- a
recorder has to see the true stream -- so the coalescing that keeps a four-way
fan-out readable happens here: every `ModelDelta` lands in a list, and the buffer
becomes content once per painted frame.

Three things are worth knowing before changing anything in this file.

**The 33 ms timer asks for a repaint; `render()` does the write.** Spec 8.3 calls
for a 33 ms batch timer and that is what bounds how often the widget asks to be
redrawn, but the buffer is drained in `render()`, at the moment the compositor asks
for the content. The two are not the same thing. Textual's screen timer is
independent of ours, so draining on our timer made the two delays *add*: a delta
arriving just after a flush waited a whole period and then a whole frame. Draining
at paint time makes them overlap, which is worth roughly 30 ms of R-U-4's budget
and most of the tail.

**The timer is not a stopwatch.** On Windows the loop clock resolves to about
15.6 ms, so a 33 ms interval fires somewhere between 31 and 47 ms. Nothing here
depends on the interval being accurate; it depends only on it being *bounded*, and
the latency the perf test asserts on is measured, not assumed.

**Asking for a layout is the expensive part, not the write.** `refresh(layout=True)`
invalidates the screen's arrangement, and the cost grows faster than the number of
panes doing it. `tail=True` exists to avoid it -- see the `StreamPane` docstring.

`stats.latencies_ms` therefore holds, per painted frame, the age of the oldest
delta that frame carried: the whole path from the event's `ts` through the bus, the
pump, the batch timer and the layout. What it does not include is the terminal
write itself, which under `App.run_test()` has no terminal to go to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from rich.text import Text
from textual.app import RenderResult
from textual.widgets import Static

from azalabscode.events import (
    Event,
    EventsDropped,
    ModelCallCancelled,
    ModelCallCompleted,
    ModelCallFailed,
    ModelCallStarted,
    ModelDelta,
)
from azalabscode.messages import Usage
from azalabscode.tui.routing import ANY
from azalabscode.tui.widgets.base import EventWidget

FLUSH_INTERVAL_S = 1 / 30
"""Spec 8.3's 33 ms batch timer, written as a frame rate because that is what it
is: 30 writes a second is well under what a terminal can show and well over what a
reader can follow."""

MAX_RETAINED_CHARS = 200_000
"""Cap on the text one pane keeps. A pane is a view of a stream, not a store of it;
the transcript and the session hold the authoritative copy. Without a cap, a long
fan-out run grows a `Text` that is re-wrapped on every repaint."""


@dataclass
class StreamStats:
    """What one pane did, for the perf test and for a debug overlay.

    Kept on the widget rather than in the test so the same numbers are available to
    a running app -- R-U-4 is a property of the shipped widget, not of a fixture.
    """

    deltas: int = 0
    """Delta events accepted."""
    flushes: int = 0
    """Drains: times the buffer became content. Usually one per painted frame. The
    coalescing ratio is `deltas / flushes`."""
    paints: int = 0
    """Frames that carried at least one new delta."""
    chars: int = 0
    """Characters of text accepted, before any truncation."""
    dropped: int = 0
    """Deltas the *bus* dropped upstream, as reported by `EventsDropped`. Should be
    zero: R-U-4 says coalescing happens in the UI, not in core."""
    latencies_ms: list[float] = field(default_factory=list)
    """One sample per paint: age of the oldest unrendered fragment when it landed
    on screen."""

    @property
    def max_latency_ms(self) -> float:
        """Worst sample, or 0.0 when nothing has been painted."""

        return max(self.latencies_ms, default=0.0)

    @property
    def coalescing_ratio(self) -> float:
        """Deltas per write. 1.0 means the batching did nothing."""

        return self.deltas / self.flushes if self.flushes else 0.0

    def percentile(self, q: float) -> float:
        """Nearest-rank percentile of the latency samples, `q` in [0, 1]."""

        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))
        return ordered[index]


class StreamPane(EventWidget, Static):
    """Incremental text for one `call_id`, or for whatever one agent is streaming.

    Consumes `ModelCallStarted`, `ModelDelta`, `ModelCallCompleted`,
    `ModelCallCancelled`, `ModelCallFailed` and `EventsDropped` (R-U-3's "documents
    which events it consumes").

    Bind it to a `call_id` for a strictly per-call pane, or leave `call_id` unset
    and bind it to an agent, in which case it follows that agent from one call to
    the next and clears at each `ModelCallStarted`.

    **`tail=True` is the mode an N-up fusion pane wants, and it is not cosmetic.**
    A pane whose height follows its content has to call `refresh(layout=True)` on
    every write, and a screen re-arrangement 30 times a second per pane is the most
    expensive thing in the R-U-4 pipeline. It grows faster than the pane count:
    measured on this machine at 200 deltas/s per stream, four panes cost 53 ms
    against a tail pane's 46 ms -- both comfortable -- and six cost 89 ms against
    62 ms, with the event loop blocked for 44 ms at a stretch instead of not at all.
    A tail pane fills its container, renders the last screenful itself, never
    changes size, and therefore never asks for a layout. `test_tui_perf.py`'s
    `test_asking_for_a_layout_is_what_costs` re-measures all of this.

    The default stays `tail=False`, because that is what stacking inside a
    `Transcript` needs: there each pane is one finished turn in a scrollable column
    and must be as tall as its content. A transcript pays the layout cost only
    while its newest turn is streaming, and it has one such pane, not six.
    """

    DEFAULT_CSS = """
    StreamPane {
        height: auto;
        width: 1fr;
        padding: 0 1;
    }
    StreamPane.-tail {
        height: 1fr;
        overflow: hidden hidden;
    }
    StreamPane.-subagent {
        border-left: outer $accent;
        color: $text-muted;
    }
    StreamPane.-cancelled {
        border-left: outer $warning;
    }
    StreamPane.-failed {
        border-left: outer $error;
    }
    """

    def __init__(
        self,
        *,
        agent_id: str = ANY,
        call_id: str | None = None,
        node_id: str = ANY,
        title: str = "",
        subagent: bool = False,
        tail: bool = False,
        flush_interval_s: float = FLUSH_INTERVAL_S,
        max_retained_chars: int = MAX_RETAINED_CHARS,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.agent_filter = agent_id
        self.node_filter = node_id
        self.call_id = call_id
        self.tail = tail
        self.flush_interval_s = flush_interval_s
        self.max_retained_chars = max_retained_chars
        self.stats = StreamStats()
        self.usage: Usage | None = None
        self.finished = False
        self.status = ""
        """Empty while streaming, then `completed`, `cancelled` or `failed`."""

        self._text = ""
        self._reasoning = ""
        self._pending: list[str] = []
        self._pending_reasoning: list[str] = []
        self._pending_since: datetime | None = None
        self._awaiting_paint: datetime | None = None
        self._truncated = False

        if title:
            self.border_title = title
        if subagent:
            self.add_class("-subagent")
        if tail:
            self.add_class("-tail")

    # -- content ------------------------------------------------------------

    @property
    def text(self) -> str:
        """Everything flushed so far. Deltas still in the buffer are not included."""

        return self._text

    @property
    def reasoning(self) -> str:
        """Flushed reasoning text."""

        return self._reasoning

    @property
    def pending(self) -> int:
        """Fragments waiting for the next timer tick."""

        return len(self._pending) + len(self._pending_reasoning)

    # -- events -------------------------------------------------------------

    def handle_event(self, event: Event) -> None:
        """Route one event. Buffers deltas; everything else is cheap and immediate."""

        if isinstance(event, ModelDelta):
            self._on_delta(event)
        elif isinstance(event, ModelCallStarted):
            self._on_started(event)
        elif isinstance(event, ModelCallCompleted):
            self._on_finished(event, "completed", usage=event.usage)
        elif isinstance(event, ModelCallCancelled):
            self._on_finished(event, "cancelled")
        elif isinstance(event, ModelCallFailed):
            if not event.will_retry:
                self._on_finished(event, "failed")
        elif isinstance(event, EventsDropped):
            self.stats.dropped += event.count

    def _for_this_pane(self, call_id: str) -> bool:
        return self.call_id is None or self.call_id == call_id

    def _on_started(self, event: ModelCallStarted) -> None:
        if not self._for_this_pane(event.call_id):
            return
        if self.call_id is None and self._text:
            # An agent-bound pane follows its agent from one call to the next. The
            # transcript is what keeps the history; a pane keeps the current answer.
            self.clear()
        self.finished = False
        self.status = ""
        self.remove_class("-cancelled", "-failed")

    def _on_delta(self, event: ModelDelta) -> None:
        if not self._for_this_pane(event.call_id):
            return
        if event.text is not None:
            self._pending.append(event.text)
            self.stats.chars += len(event.text)
        elif event.reasoning is not None:
            self._pending_reasoning.append(event.reasoning)
            self.stats.chars += len(event.reasoning)
        else:
            # A tool-call argument delta. The transcript renders the call once it is
            # complete; a half-assembled JSON fragment is noise in a text pane.
            return
        self.stats.deltas += 1
        if self._pending_since is None:
            self._pending_since = event.ts

    def _on_finished(self, event: Event, status: str, *, usage: Usage | None = None) -> None:
        call_id = getattr(event, "call_id", "")
        if not self._for_this_pane(call_id):
            return
        self.finished = True
        self.status = status
        if usage is not None:
            self.usage = usage
        if status == "cancelled":
            self.add_class("-cancelled")
        elif status == "failed":
            self.add_class("-failed")
        self.flush_now()

    # -- the batch timer ----------------------------------------------------

    def on_mount(self) -> None:
        """Start the batch timer. Deltas buffered before this are not lost."""

        self.set_interval(self.flush_interval_s, self._tick)

    def _tick(self) -> None:
        """Ask for a repaint if anything is waiting. Does not move the content.

        The timer's only job is to bound how often the widget asks to be redrawn.
        The buffer is drained in `render()`, at the moment the compositor asks for
        the content -- which is the whole point, and it is worth spelling out.

        The batch timer and Textual's screen timer are independent, so their delays
        used to *add*: a delta that arrived just after a flush waited a full timer
        period, and then whatever the next frame cost. Under a loaded CPU that
        stacking put the measured latency over R-U-4's 100 ms budget. Draining at
        paint time instead means every delta that has arrived by the time the frame
        is composed is in that frame, so the two waits overlap rather than sum.
        """

        if (self._pending or self._pending_reasoning) and self.is_mounted:
            # `layout=True` invalidates the screen's arrangement, and at 30 refreshes
            # a second across several panes that is the dominant cost in the whole
            # pipeline, and it gets worse faster than the pane count does. A tail
            # pane's box never changes size, so it never asks for one. See the
            # class docstring for the measurements.
            self.refresh(layout=not self.tail)

    def flush_now(self) -> bool:
        """Move everything buffered into the content now, and ask for a repaint.

        Public because the pane flushes on a call ending as well as on the timer,
        and because a test that wants a deterministic screen has to be able to say
        "now" rather than sleep for a frame and hope.
        """

        wrote = self._drain()
        if wrote and self.is_mounted:
            self.refresh(layout=not self.tail)
        return wrote

    def _drain(self) -> bool:
        """Merge the buffer into the content. Returns True if there was anything."""

        if not self._pending and not self._pending_reasoning:
            return False
        if self._pending:
            self._text += "".join(self._pending)
            self._pending.clear()
        if self._pending_reasoning:
            self._reasoning += "".join(self._pending_reasoning)
            self._pending_reasoning.clear()
        if len(self._text) > self.max_retained_chars:
            self._text = self._text[-self.max_retained_chars :]
            self._truncated = True
        self.stats.flushes += 1
        since = self._pending_since
        if since is not None and (self._awaiting_paint is None or since < self._awaiting_paint):
            self._awaiting_paint = since
        self._pending_since = None
        return True

    def set_text(self, text: str, *, status: str = "") -> None:
        """Replace the pane's content outright, for a run restored from disk.

        R-A-2 asks for panes of completed models to be *restored from the session*
        after a load, and there are no events to rebuild them from -- the deltas
        belong to the process that wrote the file. The node's memoized output is what
        the session has, so this is how it gets on screen.

        `status` marks the pane as it would have been marked by the call ending, so a
        restored pane and a streamed one look the same.
        """

        self._pending.clear()
        self._pending_reasoning.clear()
        self._pending_since = None
        self._text = text[-self.max_retained_chars :]
        self._truncated = len(text) > self.max_retained_chars
        self.finished = bool(status)
        self.status = status
        if status == "cancelled":
            self.add_class("-cancelled")
        elif status == "failed":
            self.add_class("-failed")
        if self.is_mounted:
            self.refresh(layout=not self.tail)

    def clear(self) -> None:
        """Drop the pane's content. Statistics are kept: they describe the run."""

        self._text = ""
        self._reasoning = ""
        self._pending.clear()
        self._pending_reasoning.clear()
        self._pending_since = None
        self._truncated = False
        self.usage = None
        if self.is_mounted:
            self.refresh(layout=True)

    # -- rendering ----------------------------------------------------------

    def render(self) -> RenderResult:
        """Drain the buffer and build the content, closing the latency interval.

        The compositor calls this when it needs the widget's content, so it is both
        the freshest point at which the buffer can be merged and the last point in
        the pipeline this process controls. Measuring here rather than at the timer
        is the difference between measuring the widget's own batching and measuring
        what R-U-4 actually asks about.
        """

        self._drain()
        if self._awaiting_paint is not None:
            elapsed = (datetime.now(UTC) - self._awaiting_paint).total_seconds() * 1000.0
            self.stats.latencies_ms.append(elapsed)
            self.stats.paints += 1
            self._awaiting_paint = None

        return self._tail_content() if self.tail else self._full_content()

    def _full_content(self) -> Text:
        """Everything the pane holds, for a pane whose height follows its content."""

        content = Text(no_wrap=False)
        if self._reasoning:
            content.append(self._reasoning, style="dim italic")
            if self._text:
                content.append("\n")
        if self._truncated:
            content.append("[...earlier output trimmed...]\n", style="dim")
        content.append(self._text)
        if self.status == "cancelled":
            content.append("\n[interrupted]", style="bold yellow")
        elif self.status == "failed":
            content.append("\n[failed]", style="bold red")
        return content

    def _tail_content(self) -> Text:
        """The last screenful, wrapped here rather than clipped by the compositor.

        Textual clips an oversized widget from the *bottom*, which in a live stream
        shows the first screenful forever. The tail has to be selected before the
        content is handed over.

        The slice bounds the work: only the last `height * (width + 1)` characters
        are ever wrapped, so a pane's per-frame cost is a function of its size and
        not of how long the run has been going.
        """

        width = max(1, self.content_size.width)
        height = max(1, self.content_size.height)
        body = self._text[-(height * (width + 1) + width) :]
        if self.status == "cancelled":
            body += "\n[interrupted]"
        elif self.status == "failed":
            body += "\n[failed]"
        text = Text(body, no_wrap=False)
        lines = text.wrap(self.app.console, width)
        return Text("\n").join(lines[-height:])


__all__ = ["FLUSH_INTERVAL_S", "MAX_RETAINED_CHARS", "StreamPane", "StreamStats"]
