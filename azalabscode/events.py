"""Typed events, the bus that carries them, and the JSONL recorder.

R-X-3 is the requirement this module exists to satisfy: every model call, tool
call, node transition, agent spawn, run-state transition, approval, checkpoint and
injected message emits a typed event, and no core behavior is observable only
through logs. The TUI consumes nothing else (R-U-1), which is also what keeps the
UI replaceable.

Backpressure policy, from spec 4.4: each subscriber has its own bounded queue.
When one fills, `ModelDelta` events are coalesced into the tail where possible and
dropped otherwise, with an `EventsDropped` marker delivered to that subscriber as
soon as it drains. Lifecycle events are never dropped -- publishing blocks instead.
A slow UI may lose streaming text; it may not lose the fact that a tool ran.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Literal, Self, final

from pydantic import Field, TypeAdapter

from azalabscode.content import ReasoningPart, TextPart, ToolCallPart
from azalabscode.errors import ProviderError
from azalabscode.messages import Usage
from azalabscode.permissions import ApprovalRequest, Decision, PermissionMode
from azalabscode.runstate import AgentPhase, RunState
from azalabscode.schema import VersionedModel
from azalabscode.toolio import ToolError, ToolResult

DEFAULT_QUEUE_SIZE = 10_000


class Event(VersionedModel):
    """Fields every event carries.

    `seq` is assigned by the bus at publish time and is monotonic per run, so a
    recorded JSONL log has a total order even though timestamps can tie.
    """

    type: str
    seq: int = 0
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str = ""
    agent_id: str | None = None
    node_id: str | None = None


# --- run lifecycle ---------------------------------------------------------


class RunStateChanged(Event):
    """A run-state transition (R-C-2). Emitted for every transition, including no-ops
    that a user action implied, so the status bar never goes stale."""

    type: Literal["run_state_changed"] = "run_state_changed"
    old: RunState
    new: RunState
    reason: str | None = None


class PermissionModeChanged(Event):
    """The run's permission mode was switched (R-C-5)."""

    type: Literal["permission_mode_changed"] = "permission_mode_changed"
    old: PermissionMode
    new: PermissionMode
    pending_resolved: int = 0
    """Approval requests auto-approved by a switch to `auto`."""


class Checkpoint(Event):
    """A safe point was folded into the session, and possibly written to disk."""

    type: Literal["checkpoint"] = "checkpoint"
    kind: str = "custom"
    to_disk: bool = False
    path: str | None = None
    duration_ms: float = 0.0


class RunWarning(Event):
    """Something a user should know about that is not an error.

    Not in spec 6.5. Added for spec C-4: a targetless `interrupt()` in a fan-out
    workflow with no `main` agent has nothing to cancel, and the spec says to "emit
    a warning event and do nothing". Without a class for it, the only record would
    be a log line, which R-X-3 forbids as the sole channel. `code` is a stable
    string a UI can switch on; `message` is for a human.
    """

    type: Literal["run_warning"] = "run_warning"
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class GraphDriftWarning(Event):
    """A loaded session's graph does not match the rebuilt one, but not fatally.

    Extra node ids or a changed `graph_hash`. Missing ids raise `GraphMismatchError`
    instead. Promotable to an error with `strict_graph_hash=True` (spec C-2).
    """

    type: Literal["graph_drift_warning"] = "graph_drift_warning"
    saved_hash: str
    rebuilt_hash: str
    extra_nodes: list[str] = Field(default_factory=list)


# --- nodes and agents ------------------------------------------------------


class NodeStarted(Event):
    """A node began executing."""

    type: Literal["node_started"] = "node_started"
    attempt: int = 0
    node_class: str = ""


class NodeCompleted(Event):
    """A node produced its output. The output itself is in the session, not here."""

    type: Literal["node_completed"] = "node_completed"
    attempt: int = 0
    duration_ms: float = 0.0
    output_summary: str = ""


class NodeFailed(Event):
    """A node raised. `on_child_error="continue"` turns this into the child's output."""

    type: Literal["node_failed"] = "node_failed"
    attempt: int = 0
    error: str = ""
    error_type: str = ""


class AgentSpawned(Event):
    """A subagent was registered in the agent tree (R-W-4)."""

    type: Literal["agent_spawned"] = "agent_spawned"
    parent_id: str | None = None
    spec_summary: str = ""
    delegated: bool = False
    """True for `delegate` (blocking), false for `spawn` (concurrent handle)."""


class AgentFinished(Event):
    """A subagent completed, failed or was cancelled."""

    type: Literal["agent_finished"] = "agent_finished"
    result_summary: str = ""
    usage: Usage | None = None
    outcome: str = "completed"


class AgentPhaseChanged(Event):
    """An agent's quiescence phase changed.

    Not in spec 6.5. Added because PAUSED is defined as a count over these phases
    (spec delta 14), and a pause that never arrives is otherwise undebuggable.
    """

    type: Literal["agent_phase_changed"] = "agent_phase_changed"
    old: AgentPhase
    new: AgentPhase
    nonquiescent: int = 0


# --- model calls -----------------------------------------------------------


class ModelCallStarted(Event):
    """A request was handed to a provider."""

    type: Literal["model_call_started"] = "model_call_started"
    call_id: str
    model: str
    message_count: int = 0
    tool_names: list[str] = Field(default_factory=list)
    attempt: int = 0


class ModelDelta(Event):
    """One streamed fragment. The only event class that is ever dropped or coalesced.

    Exactly one of `text`, `reasoning` or `tool_call` is set. Deltas are coalesced by
    the *bus* under backpressure and by the *widget* on a render timer (R-U-4); core
    never coalesces on the emit path, so a recorder sees the true stream.
    """

    type: Literal["model_delta"] = "model_delta"
    call_id: str
    text: str | None = None
    reasoning: str | None = None
    tool_call_index: int | None = None
    tool_call_delta: str | None = None

    @property
    def coalesce_key(self) -> tuple[str, str, int | None] | None:
        """Key two deltas must share to be mergeable, or `None` if unmergeable."""

        if self.text is not None:
            return (self.call_id, "text", None)
        if self.reasoning is not None:
            return (self.call_id, "reasoning", None)
        if self.tool_call_delta is not None:
            return (self.call_id, "tool_call", self.tool_call_index)
        return None


class ModelCallCompleted(Event):
    """A model call finished normally."""

    type: Literal["model_call_completed"] = "model_call_completed"
    call_id: str
    usage: Usage | None = None
    finish_reason: str | None = None
    duration_ms: float = 0.0


class ModelCallFailed(Event):
    """A model call ended in a provider error.

    Emitted for the first attempt too when the agent loop retries a mid-stream
    failure (spec C-8), so the UI shows what happened rather than a silent stall.
    """

    type: Literal["model_call_failed"] = "model_call_failed"
    call_id: str
    error: ProviderError
    attempt: int = 0
    will_retry: bool = False


class ModelCallCancelled(Event):
    """A model call was cancelled by an interrupt or a hard pause."""

    type: Literal["model_call_cancelled"] = "model_call_cancelled"
    call_id: str
    reason: str
    kept_partial: bool = False
    """Whether the partial output was retained as `AssistantMessage(cancelled=True)`."""


# --- tool calls ------------------------------------------------------------


class ToolCallRequested(Event):
    """The model asked for a tool call. Emitted before the permission check."""

    type: Literal["tool_call_requested"] = "tool_call_requested"
    call_id: str
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    parse_error: str | None = None


class ApprovalRequested(Event):
    """A destructive call is waiting on a human (R-C-6)."""

    type: Literal["approval_requested"] = "approval_requested"
    request: ApprovalRequest


class ApprovalResolved(Event):
    """A pending approval was decided, by a human or by a switch to `auto` mode."""

    type: Literal["approval_resolved"] = "approval_resolved"
    request_id: str
    decision: Decision
    by: str | None = None


class ToolCallStarted(Event):
    """A tool's `run` was entered."""

    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: str
    tool: str
    attempt: int = 0


class ToolCallProgress(Event):
    """Incremental output from a long-running tool.

    Not in spec 6.5. `shell` polls its merged output file at roughly 1 Hz; without
    this the UI shows nothing for ten minutes and the user cannot tell a slow build
    from a hung one.
    """

    type: Literal["tool_call_progress"] = "tool_call_progress"
    call_id: str
    tool: str
    text: str = ""
    elapsed_ms: float = 0.0


class ToolCallCompleted(Event):
    """A tool call returned `ok=True`."""

    type: Literal["tool_call_completed"] = "tool_call_completed"
    call_id: str
    tool: str
    result: ToolResult
    duration_ms: float = 0.0


class ToolCallFailed(Event):
    """A tool call returned `ok=False`. Never an exception: failures are data."""

    type: Literal["tool_call_failed"] = "tool_call_failed"
    call_id: str
    tool: str
    error: ToolError
    duration_ms: float = 0.0


class ToolCallCancelled(Event):
    """A tool call was cancelled in flight."""

    type: Literal["tool_call_cancelled"] = "tool_call_cancelled"
    call_id: str
    tool: str
    reason: str


# --- misc ------------------------------------------------------------------


class MessageInjected(Event):
    """`Controller.interrupt(message=...)` appended a user message (R-C-4)."""

    type: Literal["message_injected"] = "message_injected"
    message_id: str
    text: str = ""


class EventsDropped(Event):
    """Marker delivered to a subscriber that fell behind. Never itself dropped."""

    type: Literal["events_dropped"] = "events_dropped"
    count: int


AnyEvent = Annotated[
    RunStateChanged
    | PermissionModeChanged
    | Checkpoint
    | RunWarning
    | GraphDriftWarning
    | NodeStarted
    | NodeCompleted
    | NodeFailed
    | AgentSpawned
    | AgentFinished
    | AgentPhaseChanged
    | ModelCallStarted
    | ModelDelta
    | ModelCallCompleted
    | ModelCallFailed
    | ModelCallCancelled
    | ToolCallRequested
    | ApprovalRequested
    | ApprovalResolved
    | ToolCallStarted
    | ToolCallProgress
    | ToolCallCompleted
    | ToolCallFailed
    | ToolCallCancelled
    | MessageInjected
    | EventsDropped,
    Field(discriminator="type"),
]
"""Discriminated union of every event. Use `EVENT_ADAPTER` to parse one from JSON."""

EVENT_ADAPTER: TypeAdapter[Any] = TypeAdapter(AnyEvent)

type EventPredicate = Callable[[Event], bool]


class Subscription:
    """One subscriber's bounded view of the event stream.

    Async-iterable. Iteration ends when the bus is closed and the backlog is drained,
    so a recorder task can simply `async for` and exit cleanly.
    """

    def __init__(self, bus: EventBus, *, maxsize: int, name: str) -> None:
        self._bus = bus
        self._maxsize = maxsize
        self.name = name
        self._items: deque[Event] = deque()
        self._wakeup = asyncio.Event()
        self._space = asyncio.Event()
        self._space.set()
        self._closed = False
        self._pending_dropped = 0
        self.dropped_total = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def full(self) -> bool:
        """True when the queue is at its bound."""

        return len(self._items) >= self._maxsize

    def _append(self, event: Event) -> None:
        self._items.append(event)
        if self.full:
            self._space.clear()
        self._wakeup.set()

    def _try_coalesce(self, event: Event) -> bool:
        """Merge a `ModelDelta` into the tail of the queue if they are adjacent.

        Returns True when the event was absorbed. Only ever called for deltas, and
        only when the queue is full: coalescing a stream that is keeping up would
        lose the timing information a recorder is there to capture.
        """

        if not isinstance(event, ModelDelta) or not self._items:
            return False
        tail = self._items[-1]
        if not isinstance(tail, ModelDelta):
            return False
        key = event.coalesce_key
        if key is None or key != tail.coalesce_key:
            return False
        if event.text is not None:
            tail.text = (tail.text or "") + event.text
        elif event.reasoning is not None:
            tail.reasoning = (tail.reasoning or "") + event.reasoning
        elif event.tool_call_delta is not None:
            tail.tool_call_delta = (tail.tool_call_delta or "") + event.tool_call_delta
        tail.seq = event.seq
        return True

    def _drop(self, count: int = 1) -> None:
        self._pending_dropped += count
        self.dropped_total += count

    def close(self) -> None:
        """Stop iteration once the backlog is drained."""

        self._closed = True
        self._wakeup.set()
        self._space.set()

    async def get(self) -> Event | None:
        """Next event, or `None` once the bus is closed and the backlog is empty."""

        while True:
            if self._items:
                event = self._items.popleft()
                if not self.full:
                    self._space.set()
                if self._pending_dropped and len(self._items) < self._maxsize:
                    count = self._pending_dropped
                    self._pending_dropped = 0
                    self._append(
                        EventsDropped(
                            count=count,
                            run_id=event.run_id,
                            seq=event.seq,
                        )
                    )
                return event
            if self._closed:
                return None
            self._wakeup.clear()
            await self._wakeup.wait()

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Event]:
        while True:
            event = await self.get()
            if event is None:
                return
            yield event

    def unsubscribe(self) -> None:
        """Detach from the bus."""

        self._bus._remove(self)
        self.close()


@final
class EventBus:
    """Fan-out of typed events to bounded per-subscriber queues.

    Owns sequence-number allocation. On resume the bus is seeded from
    `Session.event_seq` so a reloaded run's log continues rather than restarting at
    zero and colliding with the events already on disk.
    """

    def __init__(self, *, run_id: str = "", default_queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self.run_id = run_id
        self.default_queue_size = default_queue_size
        self._subs: list[Subscription] = []
        self._seq = 0
        self._closed = False

    @property
    def seq(self) -> int:
        """The last sequence number assigned."""

        return self._seq

    def seed_seq(self, value: int) -> None:
        """Continue numbering from a loaded session's `event_seq`."""

        self._seq = max(self._seq, value)

    def next_seq(self) -> int:
        """Allocate the next sequence number."""

        self._seq += 1
        return self._seq

    def subscribe(self, *, maxsize: int | None = None, name: str = "") -> Subscription:
        """Attach a new subscriber with its own bounded queue."""

        sub = Subscription(self, maxsize=maxsize or self.default_queue_size, name=name)
        self._subs.append(sub)
        return sub

    def _remove(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)

    async def publish(self, event: Event) -> None:
        """Deliver `event` to every subscriber, assigning its sequence number.

        Blocks only when a lifecycle event meets a full queue. `ModelDelta` never
        blocks: it coalesces into the tail, or is dropped and counted.
        """

        if self._closed:
            return
        if not event.run_id:
            event.run_id = self.run_id
        event.seq = self.next_seq()
        for sub in list(self._subs):
            if not sub.full:
                sub._append(event)
                continue
            if isinstance(event, ModelDelta):
                if not sub._try_coalesce(event):
                    sub._drop()
                continue
            # Lifecycle event: wait for room rather than lose it.
            while sub.full and not sub._closed and not self._closed:
                await sub._space.wait()
            if not sub._closed:
                sub._append(event)

    def publish_nowait(self, event: Event) -> None:
        """Publish without awaiting, dropping any event that would block.

        For synchronous call sites only -- currently just the `__del__`-adjacent
        paths in tests. Prefer `publish`.
        """

        if self._closed:
            return
        if not event.run_id:
            event.run_id = self.run_id
        event.seq = self.next_seq()
        for sub in list(self._subs):
            if sub.full:
                if not (isinstance(event, ModelDelta) and sub._try_coalesce(event)):
                    sub._drop()
                continue
            sub._append(event)

    def emitter(
        self,
        *,
        agent_id: str | None = None,
        node_id: str | None = None,
    ) -> EventEmitter:
        """An emitter that stamps `run_id`, `agent_id` and `node_id` onto events."""

        return EventEmitter(self, agent_id=agent_id, node_id=node_id)

    def close(self) -> None:
        """Close the bus; subscribers finish their backlog and stop iterating."""

        self._closed = True
        for sub in list(self._subs):
            sub.close()

    def record(self, path: str | Path, *, name: str = "recorder") -> JsonlRecorder:
        """A JSONL recorder over this bus (spec 6.5).

        Use as an async context manager; it subscribes on entry and flushes on exit.
        """

        return JsonlRecorder(self, Path(path), name=name)


class EventEmitter:
    """A bus view bound to one agent and node.

    Handed to `NodeContext` so a node emits events without knowing its own ids, and
    so a subagent's events are attributed to the subagent rather than to whoever
    happens to publish them.
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        agent_id: str | None = None,
        node_id: str | None = None,
    ) -> None:
        self.bus = bus
        self.agent_id = agent_id
        self.node_id = node_id

    def bind(self, *, agent_id: str | None = None, node_id: str | None = None) -> EventEmitter:
        """A new emitter with some ids replaced."""

        return EventEmitter(
            self.bus,
            agent_id=agent_id if agent_id is not None else self.agent_id,
            node_id=node_id if node_id is not None else self.node_id,
        )

    async def emit(self, event: Event) -> None:
        """Stamp and publish an event."""

        if event.agent_id is None:
            event.agent_id = self.agent_id
        if event.node_id is None:
            event.node_id = self.node_id
        await self.bus.publish(event)


class JsonlRecorder:
    """Writes every event to a JSONL file, one JSON object per line.

    The observability hook the intent asks for: a run's whole event stream on disk,
    replayable, diffable, and cheap enough to leave on. Line-buffered so a killed
    process still leaves a readable log up to its last flush.
    """

    def __init__(self, bus: EventBus, path: Path, *, name: str = "recorder") -> None:
        self.bus = bus
        self.path = path
        self.name = name
        self.count = 0
        self._sub: Subscription | None = None
        self._task: asyncio.Task[None] | None = None
        self._file: Any = None

    async def __aenter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1, newline="\n")
        self._sub = self.bus.subscribe(name=self.name)
        self._task = asyncio.create_task(self._pump(), name=f"jsonl-recorder:{self.name}")
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
        if self._file is not None:
            self._file.close()
            self._file = None

    async def _pump(self) -> None:
        assert self._sub is not None
        async for event in self._sub:
            self._file.write(event.model_dump_json() + "\n")
            self.count += 1

    @staticmethod
    def read(path: str | Path) -> list[Any]:
        """Parse a recorded log back into typed events."""

        out: list[Any] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    out.append(EVENT_ADAPTER.validate_python(json.loads(line)))
        return out


def delta_from_part(
    call_id: str,
    part: TextPart | ReasoningPart | ToolCallPart,
    *,
    index: int | None = None,
) -> ModelDelta:
    """Build a `ModelDelta` from a content fragment.

    Convenience for the agent loop, which has parts in hand and does not want to
    know which `ModelDelta` field each one maps to.
    """

    if isinstance(part, TextPart):
        return ModelDelta(call_id=call_id, text=part.text)
    if isinstance(part, ReasoningPart):
        return ModelDelta(call_id=call_id, reasoning=part.text)
    return ModelDelta(
        call_id=call_id,
        tool_call_index=index,
        tool_call_delta=part.raw_arguments,
    )


__all__ = [
    "DEFAULT_QUEUE_SIZE",
    "EVENT_ADAPTER",
    "AgentFinished",
    "AgentPhaseChanged",
    "AgentSpawned",
    "AnyEvent",
    "ApprovalRequested",
    "ApprovalResolved",
    "Checkpoint",
    "Event",
    "EventBus",
    "EventEmitter",
    "EventPredicate",
    "EventsDropped",
    "GraphDriftWarning",
    "JsonlRecorder",
    "MessageInjected",
    "ModelCallCancelled",
    "ModelCallCompleted",
    "ModelCallFailed",
    "ModelCallStarted",
    "ModelDelta",
    "NodeCompleted",
    "NodeFailed",
    "NodeStarted",
    "PermissionModeChanged",
    "RunStateChanged",
    "RunWarning",
    "Subscription",
    "ToolCallCancelled",
    "ToolCallCompleted",
    "ToolCallFailed",
    "ToolCallProgress",
    "ToolCallRequested",
    "ToolCallStarted",
    "delta_from_part",
]
