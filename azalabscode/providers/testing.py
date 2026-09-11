"""`FakeProvider`: scripted, deterministic model responses (R-P-8).

Public API, not a test helper, because a workflow author writing their own
workflow needs the same thing the harness's own tests need: a run that costs
nothing and produces the same bytes every time.

The matching strategy is the interesting part. `by_index` is the obvious one and it
breaks the kill test: after a save/kill/load the resumed process re-issues the model
call that was in flight, so the call *index* shifts by one and every subsequent
scripted turn is off by one. `by_request_hash` (the default, spec delta 21) keys
each turn on a hash of the request itself, so the resumed process rebuilds a
transcript that hashes to the recorded key and gets the same response back.

An unmatched request raises `ScriptExhausted`. A fake that silently improvises is
worse than no fake: the kill test would pass while proving nothing.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from azalabscode.errors import ProviderError, ScriptExhausted
from azalabscode.messages import Usage
from azalabscode.providers.base import (
    Finish,
    ModelInfo,
    ModelRequest,
    ReasoningDelta,
    StreamError,
    StreamEventUnion,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageReport,
)
from azalabscode.schema import VersionedModel

MatchMode = Literal["by_request_hash", "by_index"]


class ScriptedToolCall(VersionedModel):
    """A tool call the scripted turn should emit.

    `arguments_chunks` lets a script exercise the assembly path: split one call's
    JSON across several deltas, or emit deliberately malformed JSON to test that it
    arrives as `parse_error` rather than an exception (R-P-4).
    """

    call_id: str
    name: str
    arguments: dict[str, Any] | None = None
    arguments_chunks: list[str] | None = None

    def chunks(self) -> list[str]:
        """The argument fragments to emit, in order."""

        if self.arguments_chunks is not None:
            return self.arguments_chunks
        return [json.dumps(self.arguments or {})]


class ScriptedTurn(VersionedModel):
    """One scripted model response."""

    key: str | None = None
    """Request fingerprint this turn answers, for `by_request_hash`.

    `None` means "match any request not claimed by a keyed turn", consumed in order.
    """
    text: str = ""
    text_chunks: list[str] | None = None
    """Explicit fragmentation. Defaults to one chunk per whitespace-delimited word,
    which is close enough to a real stream to exercise the UI's coalescing."""
    reasoning: str = ""
    tool_calls: list[ScriptedToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage | None = None
    delay_s: float = 0.0
    """Sleep before the first event: lets a test pause or interrupt mid-call."""
    chunk_delay_s: float = 0.0
    """Sleep between chunks."""
    chunk_rate_hz: float = 0.0
    """Chunks per second, self-correcting. Takes precedence over `chunk_delay_s`.

    Spec delta 23. `chunk_delay_s` cannot express a rate on Windows: the loop clock
    resolves to about 15.6 ms, so `chunk_delay_s=0.005` -- the value spec 8.3's
    200 tokens/s asks for -- actually yields about 70/s, and even with the 1 ms
    system timer it stops at 160/s. Sleeping to an absolute schedule instead of a
    fixed interval delivers the requested average whatever the clock does: a tick
    that overslept emits the chunks it owes back to back and the next deadline is
    still on the original grid. Real SSE arrives in bursts for the same reason, so
    this is also the more faithful shape.
    """
    error: ProviderError | None = None
    """Terminal error. With `error_after_chunks`, a mid-stream failure (spec C-8)."""
    error_after_chunks: int = 0

    def text_fragments(self) -> list[str]:
        """The text fragments to emit, in order."""

        if self.text_chunks is not None:
            return self.text_chunks
        if not self.text:
            return []
        words = self.text.split(" ")
        return [w if i == len(words) - 1 else w + " " for i, w in enumerate(words)]


class Script(VersionedModel):
    """A whole scripted run. Committed as JSON and referenced by a workflow config."""

    match: MatchMode = "by_request_hash"
    turns: list[ScriptedTurn] = Field(default_factory=list)
    model_info: dict[str, ModelInfo] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Script:
        """Read a script from a JSON file."""

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        """Write a script to a JSON file."""

        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


class FakeProvider:
    """Replays a `Script`. Stateless with respect to resume when keyed by hash.

    `call_index` exists only for `by_index` mode, and the agent loop seeds it from
    the checkpointed `model_call_seq` so that even the indexed mode survives a
    reload without the provider itself holding resume state.
    """

    name = "fake"

    def __init__(
        self,
        turns: Sequence[ScriptedTurn] | Script | None = None,
        *,
        match: MatchMode | None = None,
        script_path: str | Path | None = None,
        call_index: int = 0,
    ) -> None:
        if script_path is not None:
            script = Script.load(script_path)
        elif isinstance(turns, Script):
            script = turns
        else:
            script = Script(turns=list(turns or []))
        self.script = script
        self.match: MatchMode = match or script.match
        self.call_index = call_index
        self.requests: list[ModelRequest] = []
        """Every request received, for assertions."""
        self._consumed_unkeyed = 0
        self._closed = False

    # -- selection ----------------------------------------------------------

    def _select(self, request: ModelRequest) -> ScriptedTurn:
        """Pick the turn that answers `request`, or raise `ScriptExhausted`."""

        if self.match == "by_index":
            if self.call_index >= len(self.script.turns):
                raise ScriptExhausted(
                    f"FakeProvider(match='by_index') has {len(self.script.turns)} turn(s) "
                    f"but was asked for call {self.call_index}"
                )
            return self.script.turns[self.call_index]

        fingerprint = request.fingerprint()
        for turn in self.script.turns:
            if turn.key == fingerprint:
                return turn

        unkeyed = [t for t in self.script.turns if t.key is None]
        if self._consumed_unkeyed < len(unkeyed):
            turn = unkeyed[self._consumed_unkeyed]
            self._consumed_unkeyed += 1
            return turn

        raise ScriptExhausted(
            f"FakeProvider(match='by_request_hash') has no turn for request "
            f"{fingerprint} (model={request.model}, {len(request.messages)} messages). "
            "Record the turn, or add an unkeyed turn to the script."
        )

    def key_for(self, request: ModelRequest) -> str:
        """The fingerprint a turn would need to answer `request`.

        Used when recording a script: run once, print the keys, paste them in.
        """

        return request.fingerprint()

    # -- provider protocol --------------------------------------------------

    async def stream(self, request: ModelRequest) -> AsyncGenerator[StreamEventUnion, None]:
        """Replay the matching turn as a stream of events."""

        self.requests.append(request)
        turn = self._select(request)
        self.call_index += 1

        if turn.delay_s:
            await asyncio.sleep(turn.delay_s)

        emitted = 0
        error = turn.error
        cut_after = turn.error_after_chunks if error is not None else 0
        started = time.perf_counter()

        async def tick() -> bool:
            """Emit-one-chunk bookkeeping. Returns True when the turn should cut out."""

            nonlocal emitted
            emitted += 1
            if turn.chunk_rate_hz > 0:
                # Sleep to an absolute deadline on the original grid, not for a
                # fixed interval: a tick the clock overslept is paid back by the
                # next ones rather than compounding into a slower stream.
                due = started + emitted / turn.chunk_rate_hz
                remaining = due - time.perf_counter()
                if remaining > 0:
                    await asyncio.sleep(remaining)
            elif turn.chunk_delay_s:
                await asyncio.sleep(turn.chunk_delay_s)
            return cut_after > 0 and emitted >= cut_after

        if error is not None and cut_after == 0:
            yield StreamError(error=error)
            return

        if turn.reasoning:
            yield ReasoningDelta(text=turn.reasoning)
            if await tick():
                assert error is not None
                yield StreamError(error=error)
                return

        for fragment in turn.text_fragments():
            yield TextDelta(text=fragment)
            if await tick():
                assert error is not None
                yield StreamError(error=error)
                return

        for index, call in enumerate(turn.tool_calls):
            yield ToolCallStart(index=index, call_id=call.call_id, name=call.name)
            for chunk in call.chunks():
                yield ToolCallDelta(index=index, arguments_delta=chunk)
                if await tick():
                    assert error is not None
                    yield StreamError(error=error)
                    return
            yield ToolCallEnd(index=index)

        if turn.usage is not None:
            yield UsageReport(usage=turn.usage)
        yield Finish(reason=turn.finish_reason)

    async def model_info(self, model: str) -> ModelInfo | None:
        """Scripted metadata, or a permissive default so callers are not blocked."""

        if model in self.script.model_info:
            return self.script.model_info[model]
        return ModelInfo(
            id=model,
            context_length=128_000,
            supports_tools=True,
            supports_reasoning=False,
            supports_images=False,
        )

    async def aclose(self) -> None:
        """No transport to release. Idempotent."""

        self._closed = True


def turn(
    text: str = "",
    *,
    tool_calls: Sequence[ScriptedToolCall] | None = None,
    key: str | None = None,
    **kwargs: Any,
) -> ScriptedTurn:
    """Shorthand for a scripted turn, for tests that read better as one line."""

    return ScriptedTurn(text=text, tool_calls=list(tool_calls or []), key=key, **kwargs)


def tool_call(name: str, arguments: dict[str, Any], *, call_id: str = "") -> ScriptedToolCall:
    """Shorthand for a scripted tool call."""

    from azalabscode.ids import new_call_id

    return ScriptedToolCall(call_id=call_id or new_call_id(), name=name, arguments=arguments)


__all__ = [
    "FakeProvider",
    "MatchMode",
    "Script",
    "ScriptedToolCall",
    "ScriptedTurn",
    "tool_call",
    "turn",
]
