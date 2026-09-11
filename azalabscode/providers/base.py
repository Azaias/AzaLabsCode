"""The provider protocol, its request and stream types, and the stream accumulator.

R-P-1 is the shape of this module: `stream()` is the only required method, and the
non-streaming `complete()` is a helper written on top of it rather than a second
code path that can drift.

R-P-2 is the discipline: nothing here names OpenRouter. Provider-specific knobs
travel in `ModelRequest.provider_options`, which every other layer treats as an
opaque bag (spec C-9). Reasoning is the one exception -- it is abstracted into
`ReasoningConfig` because effort and budget are common across vendors and a
workflow should not have to know which one it is talking to.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from azalabscode.content import (
    Part,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    parse_tool_arguments,
)
from azalabscode.errors import ProviderCallError, ProviderError
from azalabscode.ids import new_call_id
from azalabscode.messages import AssistantMessage, Message, Usage
from azalabscode.schema import HarnessModel, VersionedModel
from azalabscode.toolio import ToolSchema

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ReasoningConfig(HarnessModel):
    """Abstracted reasoning controls.

    `effort` and `max_tokens` are the two knobs every vendor that exposes reasoning
    exposes in some form; the provider maps them. `exclude` asks for reasoning to
    happen but not be returned, which is what a fusion pane wants when it only has
    room for the answer.
    """

    effort: Literal["minimal", "low", "medium", "high"] | None = None
    max_tokens: int | None = None
    exclude: bool = False


class ModelRequest(VersionedModel):
    """One call to a model. Provider-agnostic by construction (R-P-2)."""

    model: str
    messages: list[Message]
    tools: list[ToolSchema] = Field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None
    reasoning: ReasoningConfig | None = None
    parallel_tool_calls: bool | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    """Passed through to the provider verbatim. Opaque to every other layer."""
    metadata: dict[str, str] = Field(default_factory=dict)
    """`run_id`, `agent_id`, `node_id` for logging. Never sent to the model."""

    def fingerprint(self) -> str:
        """A stable hash of the parts of the request that determine the response.

        Used by `FakeProvider(match="by_request_hash")` (spec delta 21): the resumed
        process re-issues an interrupted call, rebuilds a transcript that hashes to
        the recorded key, and gets the same scripted response even though the call
        *index* has shifted.

        Deliberately excludes `metadata` (run and agent ids differ across processes)
        and message ids and timestamps (ULIDs are minted fresh on every construction).
        """

        import hashlib
        import json

        payload = {
            "model": self.model,
            "messages": [_fingerprint_message(m) for m in self.messages],
            "tools": sorted(t.name for t in self.tools),
            "tool_choice": self.tool_choice,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _fingerprint_message(message: Any) -> dict[str, Any]:
    """Reduce a message to the fields that affect the model's response."""

    role = message.role
    if role == "system":
        return {"role": role, "content": message.content}
    if role == "tool":
        return {"role": role, "call_id": message.call_id, "text": message.result.text}
    parts: list[dict[str, Any]] = []
    for part in message.content:
        if part.type == "text":
            parts.append({"t": "text", "v": part.text})
        elif part.type == "reasoning":
            parts.append({"t": "reasoning", "v": part.text})
        elif part.type == "tool_call":
            parts.append({"t": "call", "id": part.call_id, "n": part.name, "a": part.raw_arguments})
        elif part.type == "image":
            parts.append({"t": "image", "v": part.media_type})
        else:
            parts.append({"t": part.type})
    return {"role": role, "parts": parts}


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------


class TextDelta(HarnessModel):
    """A fragment of visible model output."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ReasoningDelta(HarnessModel):
    """A fragment of model reasoning."""

    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str
    signature: str | None = None


class ToolCallStart(HarnessModel):
    """A tool call began. `index` is the slot; deltas that follow reference it."""

    type: Literal["tool_call_start"] = "tool_call_start"
    index: int
    call_id: str
    name: str


class ToolCallDelta(HarnessModel):
    """A fragment of a tool call's JSON arguments."""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    arguments_delta: str


class ToolCallEnd(HarnessModel):
    """A tool call's arguments are complete and can be parsed."""

    type: Literal["tool_call_end"] = "tool_call_end"
    index: int


class UsageReport(HarnessModel):
    """Token and cost accounting. Named `UsageReport` to keep `Usage` the message type."""

    type: Literal["usage"] = "usage"
    usage: Usage


class Finish(HarnessModel):
    """The model stopped, and why."""

    type: Literal["finish"] = "finish"
    reason: str


class StreamError(HarnessModel):
    """Terminal error event. The stream yields this rather than raising (R-P-5).

    A failure after the first byte is the caller's decision, not the provider's: the
    agent loop discards the partial and retries the whole call once (spec C-8).
    """

    type: Literal["error"] = "error"
    error: ProviderError


StreamEvent = Annotated[
    TextDelta
    | ReasoningDelta
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | UsageReport
    | Finish
    | StreamError,
    Field(discriminator="type"),
]
"""Discriminated union of everything a provider stream can yield."""

type StreamEventUnion = (
    TextDelta
    | ReasoningDelta
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | UsageReport
    | Finish
    | StreamError
)
"""The bare union, for annotating variables where a discriminator is not needed."""


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


class ModelPricing(HarnessModel):
    """USD per token, as the provider reports it. `None` means unknown, not free."""

    prompt: float | None = None
    completion: float | None = None
    image: float | None = None
    request: float | None = None


class ModelInfo(VersionedModel):
    """Capability metadata for one model (R-P-7).

    Every field is optional. Missing metadata degrades to "unknown" and never blocks
    a call: a model the catalogue has not caught up with is still a model that works.
    """

    id: str
    name: str | None = None
    context_length: int | None = None
    max_output_tokens: int | None = None
    supports_tools: bool | None = None
    supports_reasoning: bool | None = None
    supports_images: bool | None = None
    pricing: ModelPricing | None = None
    fetched_at: float | None = None
    """Unix seconds, for TTL checks against the on-disk cache."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    """Model access. The only interface the workflow layer knows about."""

    name: str

    def stream(self, request: ModelRequest) -> AsyncIterator[StreamEventUnion]:
        """Stream one model call.

        Never raises for a provider-side failure: yields a terminal `StreamError`
        instead. Cancelling the iterator must close the underlying connection
        promptly (R-P-6).
        """
        ...

    async def model_info(self, model: str) -> ModelInfo | None:
        """Capability metadata, or `None` when the provider cannot say."""
        ...

    async def aclose(self) -> None:
        """Release transport resources. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


class _ToolCallBuffer:
    """Argument deltas for one tool-call slot, buffered until the call ends."""

    __slots__ = ("call_id", "chunks", "closed", "index", "name")

    def __init__(self, index: int, call_id: str, name: str) -> None:
        self.index = index
        self.call_id = call_id or new_call_id()
        self.name = name
        self.chunks: list[str] = []
        self.closed = False

    @property
    def raw(self) -> str:
        return "".join(self.chunks)

    def to_part(self) -> ToolCallPart:
        """Parse the buffered arguments into a part.

        A parse failure is recorded on the part, never raised (R-P-4). The agent loop
        turns it into a structured tool error so the model can correct itself.
        """

        raw = self.raw
        arguments, parse_error = parse_tool_arguments(raw)
        return ToolCallPart(
            call_id=self.call_id,
            name=self.name,
            arguments=arguments,
            raw_arguments=raw,
            parse_error=parse_error,
        )


class StreamAccumulator:
    """Folds a stream of `StreamEvent`s into one `AssistantMessage`.

    Shared by `complete()` and the agent loop so that the streaming and
    non-streaming paths can never disagree about what a response contained.

    Tool calls are ordered by their `index`, not by arrival: providers interleave
    argument deltas across slots, and the transcript invariant is defined in *call*
    order.
    """

    def __init__(self, model: str = "") -> None:
        self.model = model
        self.text_chunks: list[str] = []
        self.reasoning_chunks: list[str] = []
        self.reasoning_signature: str | None = None
        self.tool_calls: dict[int, _ToolCallBuffer] = {}
        self.usage: Usage | None = None
        self.finish_reason: str | None = None
        self.error: ProviderError | None = None
        self.saw_any_output = False

    def feed(self, event: StreamEventUnion) -> None:
        """Fold one event in."""

        match event:
            case TextDelta():
                self.text_chunks.append(event.text)
                self.saw_any_output = True
            case ReasoningDelta():
                self.reasoning_chunks.append(event.text)
                if event.signature:
                    self.reasoning_signature = event.signature
                self.saw_any_output = True
            case ToolCallStart():
                existing = self.tool_calls.get(event.index)
                if existing is None:
                    self.tool_calls[event.index] = _ToolCallBuffer(
                        event.index, event.call_id, event.name
                    )
                else:
                    # Some providers repeat the header with the id or name filled in
                    # only on a later chunk. Take whichever value is non-empty.
                    if event.call_id:
                        existing.call_id = event.call_id
                    if event.name:
                        existing.name = event.name
                self.saw_any_output = True
            case ToolCallDelta():
                buffer = self.tool_calls.get(event.index)
                if buffer is None:
                    buffer = _ToolCallBuffer(event.index, new_call_id(), "")
                    self.tool_calls[event.index] = buffer
                buffer.chunks.append(event.arguments_delta)
                self.saw_any_output = True
            case ToolCallEnd():
                buffer = self.tool_calls.get(event.index)
                if buffer is not None:
                    buffer.closed = True
            case UsageReport():
                self.usage = event.usage
            case Finish():
                self.finish_reason = event.reason
            case StreamError():
                self.error = event.error

    @property
    def text(self) -> str:
        """Visible text accumulated so far."""

        return "".join(self.text_chunks)

    def parts(self) -> list[Part]:
        """Content parts in transcript order: reasoning, then text, then tool calls."""

        parts: list[Part] = []
        reasoning = "".join(self.reasoning_chunks)
        if reasoning:
            parts.append(ReasoningPart(text=reasoning, signature=self.reasoning_signature))
        text = self.text
        if text:
            parts.append(TextPart(text=text))
        for index in sorted(self.tool_calls):
            parts.append(self.tool_calls[index].to_part())
        return parts

    def message(
        self, *, cancelled: bool = False, drop_tool_calls: bool = False
    ) -> AssistantMessage:
        """Build the assistant message.

        `drop_tool_calls` implements spec delta 15: a cancelled call drops *every*
        tool call, not only the structurally incomplete ones. None of them were
        dispatched, and keeping one would require inventing a result and would tell
        the model it ran something it did not.
        """

        parts: list[Part] = self.parts()
        if drop_tool_calls:
            parts = [p for p in parts if not isinstance(p, ToolCallPart)]
        return AssistantMessage(
            content=parts,
            model=self.model,
            usage=None if cancelled else self.usage,
            finish_reason=self.finish_reason,
            cancelled=cancelled,
        )

    def has_only_tool_calls(self) -> bool:
        """True when the response contained tool calls and nothing else.

        A cancelled response like this is discarded entirely: with the tool calls
        dropped there is nothing left worth keeping (spec delta 15).
        """

        return bool(self.tool_calls) and not self.text and not self.reasoning_chunks


async def complete(provider: Provider, request: ModelRequest) -> AssistantMessage:
    """Non-streaming helper (R-P-1). Raises `ProviderCallError` on a stream error.

    Raising here rather than returning a marker is deliberate: a caller that chose
    the non-streaming path has no error event to inspect, so an exception is the only
    honest signal. The streaming path keeps the "errors are data" rule.
    """

    accumulator = StreamAccumulator(model=request.model)
    async for event in provider.stream(request):
        accumulator.feed(event)
    if accumulator.error is not None:
        raise ProviderCallError(accumulator.error)
    return accumulator.message()


__all__ = [
    "Finish",
    "ModelInfo",
    "ModelPricing",
    "ModelRequest",
    "Provider",
    "ReasoningConfig",
    "ReasoningDelta",
    "StreamAccumulator",
    "StreamError",
    "StreamEvent",
    "StreamEventUnion",
    "TextDelta",
    "ToolCallDelta",
    "ToolCallEnd",
    "ToolCallStart",
    "UsageReport",
    "complete",
]
