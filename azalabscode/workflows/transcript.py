"""The transcript invariant, and the only code allowed to append a tool result.

The invariant, from `azalabscode.messages`:

    For every `ToolCallPart` in an `AssistantMessage` there is exactly one later
    `ToolResultMessage` with the same `call_id`, and the results appear in *call*
    order, not completion order.

The way to break it is to append each result as it arrives. Three parallel reads
finish in whatever order the filesystem felt like, and an interrupt between the
second and the third leaves a transcript with a hole in the middle -- which every
provider rejects, and which no later repair can distinguish from a result that was
deliberately omitted.

So results are never appended as they complete. Each one lands in a `TurnResults`
buffer keyed by `call_id`, and `finalize_turn` materialises the whole batch in call
order in one go, filling any hole with a structured error. There is no window in
which the transcript is invalid, because the transcript is only ever written at a
point where it is complete.

`repair_transcript` is the same operation applied to a transcript that came back
from disk with holes in it (R-C-13); it is what `load()` calls at M3.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field

from azalabscode.content import ToolCallPart
from azalabscode.messages import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    assert_transcript_valid,
    open_call_ids,
)
from azalabscode.toolio import ToolError, ToolErrorKind, ToolResult

type ResultFill = Callable[[ToolCallPart], ToolResult]
"""Builds the result for a call that never produced one."""

CANCELLED_MESSAGE = "the call was cancelled before it finished"
INTERRUPTED_MESSAGE = "process terminated during execution; effect unknown"
NOT_RUN_MESSAGE = "the call was not started before the turn was interrupted"


def cancelled_fill(call: ToolCallPart) -> ToolResult:
    """The result an in-flight call gets when an interrupt cuts the batch (R-C-4).

    `cancelled`, not `interrupted`: the process is alive, so whether the tool had an
    effect is knowable -- `shell` has already killed its process tree by the time
    this is built.
    """

    return _failure(ToolErrorKind.CANCELLED, CANCELLED_MESSAGE, call)


def not_run_fill(call: ToolCallPart) -> ToolResult:
    """The result a call that never started gets. Also `cancelled`: nothing happened.

    Distinguished from `cancelled_fill` only by its message, which is what tells the
    model it may safely re-issue this particular call.
    """

    return _failure(ToolErrorKind.CANCELLED, NOT_RUN_MESSAGE, call)


def interrupted_fill(call: ToolCallPart) -> ToolResult:
    """The result a call in flight at process death gets (R-C-13).

    Never re-executed: the effect is unknown, and re-running a half-applied `shell`
    is how one bad interrupt becomes two.
    """

    return _failure(ToolErrorKind.INTERRUPTED, INTERRUPTED_MESSAGE, call)


def _failure(kind: ToolErrorKind, message: str, call: ToolCallPart) -> ToolResult:
    result = ToolResult.failure(kind, message)
    result.meta = {"call_id": call.call_id, "tool": call.name}
    return result


@dataclass
class TurnResults:
    """The results of one turn's tool calls, held until the whole turn can be written.

    `calls` fixes the order; `results` fills in. Nothing here touches the transcript:
    that is `finalize_turn`'s single job, and keeping it single is what makes the
    invariant a property of one function rather than of every caller.
    """

    calls: list[ToolCallPart] = field(default_factory=list)
    results: dict[str, ToolResult] = field(default_factory=dict)

    @classmethod
    def for_calls(cls, calls: Iterable[ToolCallPart]) -> TurnResults:
        """A buffer expecting exactly `calls`, in that order."""

        return cls(calls=list(calls))

    @property
    def call_ids(self) -> list[str]:
        """Every call id this turn requested, in call order."""

        return [c.call_id for c in self.calls]

    def record(self, call_id: str, result: ToolResult) -> None:
        """Store one result. The first result for a call id wins.

        First-wins matters on the interrupt path: a call that completed keeps its
        real result even if the batch is later filled in wholesale.
        """

        self.results.setdefault(call_id, result)

    def has(self, call_id: str) -> bool:
        """Whether a result has been recorded for this call."""

        return call_id in self.results

    def missing(self) -> list[ToolCallPart]:
        """Calls with no result yet, in call order."""

        return [c for c in self.calls if c.call_id not in self.results]

    @property
    def complete(self) -> bool:
        """True when every call has a result."""

        return not self.missing()

    def messages(self, *, fill: ResultFill = cancelled_fill) -> list[ToolResultMessage]:
        """One `ToolResultMessage` per call, in call order, holes filled.

        Completion order is deliberately not consulted: `results` is a mapping and
        `calls` is the sequence, so there is no way for arrival order to leak in.
        """

        out: list[ToolResultMessage] = []
        for call in self.calls:
            result = self.results.get(call.call_id)
            if result is None:
                result = fill(call)
            out.append(ToolResultMessage(call_id=call.call_id, name=call.name, result=result))
        return out


def finalize_turn(
    messages: list[Message],
    turn: TurnResults,
    *,
    fill: ResultFill = cancelled_fill,
    agent_id: str = "?",
    validate: bool = True,
) -> list[ToolResultMessage]:
    """Append a turn's results to `messages` in call order and check the invariant.

    The validation is not paranoia about this function -- it is the assertion that
    whatever *built* the turn did not hand over calls the transcript never
    requested. It runs at every safe point in tests and under `-X dev`.
    """

    written = turn.messages(fill=fill)
    messages.extend(written)
    if validate:
        assert_transcript_valid(messages, agent_id=agent_id)
    return written


def repair_transcript(
    messages: list[Message],
    *,
    fill: ResultFill = interrupted_fill,
    pending: dict[str, ToolResult] | None = None,
    defer: Collection[str] = (),
    agent_id: str = "?",
) -> list[ToolResultMessage]:
    """Answer every unanswered call, so a reloaded transcript is valid (R-C-13).

    `pending` is `AgentState.pending_results`: results that completed before the
    process died but were never materialised, because the turn they belonged to had
    not finished. Those are used verbatim; everything else gets `fill`, which
    defaults to the `interrupted` error the model is told about and which is never
    re-executed.

    `defer` names calls that must be left unanswered because something is going to
    answer them for real -- an in-flight `delegate`, which a resume re-runs rather
    than errors (spec delta 16). Deferring is safe: an unanswered *trailing* call is
    a turn in progress, which the invariant permits by design. Deferring a call that
    is followed by an answered one is not, so those are filled anyway rather than
    silently corrupting the order.

    Appending at the end is safe for the invariant: results stay in call order
    relative to each other, and every one of them still follows the assistant
    message that requested it.
    """

    open_ids = open_call_ids(messages)
    if not open_ids:
        return []

    deferred = _trailing_subset(open_ids, defer)
    parts = _calls_by_id(messages)
    written: list[ToolResultMessage] = []
    for call_id in open_ids:
        if call_id in deferred:
            continue
        call = parts.get(call_id) or ToolCallPart(call_id=call_id, name="", raw_arguments="")
        result = (pending or {}).get(call_id)
        if result is None:
            result = fill(call)
        message = ToolResultMessage(call_id=call_id, name=call.name, result=result)
        messages.append(message)
        written.append(message)

    assert_transcript_valid(messages, agent_id=agent_id)
    return written


def _trailing_subset(open_ids: Sequence[str], defer: Collection[str]) -> frozenset[str]:
    """The deferrable ids that form an unbroken tail of `open_ids`.

    A deferred call must not be overtaken by an answered one, or the results land
    out of call order and the invariant breaks at the next append. Only the tail can
    be left open, so anything earlier is filled even if the caller asked to defer it.
    """

    if not defer:
        return frozenset()
    wanted = set(defer)
    tail: list[str] = []
    for call_id in reversed(open_ids):
        if call_id not in wanted:
            break
        tail.append(call_id)
    return frozenset(tail)


def _calls_by_id(messages: Sequence[Message]) -> dict[str, ToolCallPart]:
    """Every tool call in the transcript, keyed by call id."""

    out: dict[str, ToolCallPart] = {}
    for message in messages:
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                out[call.call_id] = call
    return out


def interrupted_error(call_id: str) -> ToolError:
    """The error a call in flight at process death carries (R-C-13)."""

    return ToolError(
        kind=ToolErrorKind.INTERRUPTED,
        message=INTERRUPTED_MESSAGE,
        details={"call_id": call_id},
    )


__all__ = [
    "CANCELLED_MESSAGE",
    "INTERRUPTED_MESSAGE",
    "NOT_RUN_MESSAGE",
    "ResultFill",
    "TurnResults",
    "cancelled_fill",
    "finalize_turn",
    "interrupted_error",
    "interrupted_fill",
    "not_run_fill",
    "repair_transcript",
]
