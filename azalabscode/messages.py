"""The transcript: message types and token accounting.

A transcript is a `list[Message]`. The invariant the whole control layer is built
to preserve is stated here because this is where it is visible:

    For every `ToolCallPart` in an `AssistantMessage` there is exactly one later
    `ToolResultMessage` with the same `call_id`, and the results appear in *call*
    order, not completion order.

`assert_transcript_valid` is the executable form. It runs at every safe point in
tests and under `-X dev`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import Field

from azalabscode.content import (
    Part,
    TextPart,
    ToolCallPart,
    text_of,
    tool_calls_of,
)
from azalabscode.ids import new_message_id
from azalabscode.schema import HarnessModel, VersionedModel
from azalabscode.toolio import ToolResult


class Usage(HarnessModel):
    """Token and cost accounting for one model call.

    Every field is optional-by-zero rather than `None`, so usages add without
    special cases; `cost_usd` is the exception, because a missing cost and a zero
    cost are different claims and only the provider knows which it meant.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion. Cached tokens are already counted in prompt."""

        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        """Sum two usages. Cost is `None` only when neither side reported one."""

        cost: float | None
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=cost,
        )


class _BaseMessage(VersionedModel):
    """Fields every message carries."""

    id: str = Field(default_factory=new_message_id)
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SystemMessage(_BaseMessage):
    """The agent's system prompt. Plain text: no model accepts parts here."""

    role: Literal["system"] = "system"
    content: str


class UserMessage(_BaseMessage):
    """Input from the human, or a task handed to a subagent.

    `injected=True` marks a message delivered by `Controller.interrupt()` mid-run
    rather than at a turn boundary. The coding agent's system prompt tells the model
    that an injected message following a cancelled assistant message takes
    precedence over the thought it was cut off from (spec decision 4).
    """

    role: Literal["user"] = "user"
    content: list[Part]
    injected: bool = False

    @property
    def text(self) -> str:
        """The text content of this message."""

        return text_of(self.content)

    @classmethod
    def of(cls, text: str, *, injected: bool = False) -> UserMessage:
        """A plain-text user message."""

        return cls(content=[TextPart(text=text)], injected=injected)


class AssistantMessage(_BaseMessage):
    """One model response: text, reasoning and tool calls, plus what it cost.

    `cancelled=True` means an interrupt cut this response short and
    `AgentSpec.keep_cancelled_output` was true (the default). The partial text stays
    in the transcript and is sent back to the model, so it knows what it had already
    produced. Every `ToolCallPart` is dropped when a call is cancelled -- not just
    the structurally incomplete ones (spec delta 15) -- because none were
    dispatched, and keeping one would need a fabricated result and would tell the
    model it ran something it did not.
    """

    role: Literal["assistant"] = "assistant"
    content: list[Part] = Field(default_factory=list)
    model: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None
    cancelled: bool = False

    @property
    def text(self) -> str:
        """The text content of this message, excluding reasoning."""

        return text_of(self.content)

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """Tool calls requested by this message, in the order the model emitted them."""

        return tool_calls_of(self.content)


class ToolResultMessage(_BaseMessage):
    """The result of one tool call, keyed to the call it answers."""

    role: Literal["tool"] = "tool"
    call_id: str
    name: str = ""
    """The tool name, duplicated here so a transcript reads without cross-referencing."""
    result: ToolResult

    @property
    def text(self) -> str:
        """The model-facing text of the result."""

        return self.result.text


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]
"""Discriminated union of every message type."""

type Transcript = list[Message]


class TranscriptError(ValueError):
    """The transcript invariant does not hold. Always a harness bug."""


def open_call_ids(messages: list[Any]) -> list[str]:
    """Call ids requested by an assistant message with no result message yet.

    Order matches call order. This is what `repair_transcript` backfills on load and
    what the agent loop's `_finalize_turn` fills holes in.
    """

    requested: list[str] = []
    answered: set[str] = set()
    for message in messages:
        if isinstance(message, AssistantMessage):
            requested.extend(call.call_id for call in message.tool_calls)
        elif isinstance(message, ToolResultMessage):
            answered.add(message.call_id)
    return [call_id for call_id in requested if call_id not in answered]


def assert_transcript_valid(messages: list[Any], *, agent_id: str = "?") -> None:
    """Raise `TranscriptError` unless the transcript invariant holds.

    Checks four things, in the order they go wrong in practice:

    1. no result answers a call that was never made,
    2. no call is answered twice,
    3. results appear after the assistant message that requested them,
    4. results for one assistant turn appear in call order, not completion order.

    A trailing set of unanswered calls is *not* an error here: that is a turn in
    progress. It becomes an error only at the point the next model request is built,
    which is where the loop calls `open_call_ids` instead.
    """

    pending: list[str] = []
    seen_calls: set[str] = set()
    answered: set[str] = set()

    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                if call.call_id in seen_calls:
                    raise TranscriptError(
                        f"agent {agent_id}: duplicate tool call id {call.call_id!r} "
                        f"at message {index}"
                    )
                seen_calls.add(call.call_id)
                pending.append(call.call_id)
        elif isinstance(message, ToolResultMessage):
            if message.call_id not in seen_calls:
                raise TranscriptError(
                    f"agent {agent_id}: result at message {index} answers unknown call "
                    f"{message.call_id!r}"
                )
            if message.call_id in answered:
                raise TranscriptError(
                    f"agent {agent_id}: duplicate result for call {message.call_id!r} "
                    f"at message {index}"
                )
            if message.call_id not in pending:
                raise TranscriptError(
                    f"agent {agent_id}: result for call {message.call_id!r} at message "
                    f"{index} precedes the assistant message that requested it"
                )
            expected = pending[0]
            if message.call_id != expected:
                raise TranscriptError(
                    f"agent {agent_id}: results out of call order at message {index}: "
                    f"expected {expected!r}, got {message.call_id!r}"
                )
            pending.pop(0)
            answered.add(message.call_id)


def usage_total(messages: list[Any]) -> Usage:
    """Sum the usage of every assistant message in a transcript."""

    total = Usage()
    for message in messages:
        if isinstance(message, AssistantMessage) and message.usage is not None:
            total = total + message.usage
    return total


__all__ = [
    "AssistantMessage",
    "Message",
    "SystemMessage",
    "ToolResultMessage",
    "Transcript",
    "TranscriptError",
    "Usage",
    "UserMessage",
    "assert_transcript_valid",
    "open_call_ids",
    "usage_total",
]
