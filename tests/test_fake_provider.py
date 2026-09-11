"""`FakeProvider` (R-P-8) and the request fingerprint the kill test depends on.

The point of `by_request_hash` (spec delta 21) is stated once here as an executable
scenario: after a save/kill/load the resumed process re-issues the model call that
was in flight, so the call *index* shifts. An index-keyed script silently returns
the wrong turn from then on. A hash-keyed script returns the right one, because the
rebuilt transcript hashes to the same key.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from azalabscode.content import ReasoningPart, TextPart, ToolCallPart
from azalabscode.errors import ProviderError, ProviderErrorKind, ScriptExhausted
from azalabscode.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from azalabscode.providers.base import (
    ModelRequest,
    Provider,
    StreamAccumulator,
    StreamError,
    complete,
)
from azalabscode.providers.testing import (
    FakeProvider,
    Script,
    ScriptedToolCall,
    ScriptedTurn,
    tool_call,
    turn,
)
from azalabscode.toolio import ToolResult, ToolSchema


def request_of(*messages, model: str = "x/y", tools: list[str] | None = None) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=list(messages) or [UserMessage.of("hello")],
        tools=[
            ToolSchema(name=name, description="", parameters={"type": "object"})
            for name in (tools or [])
        ],
    )


async def run(provider: FakeProvider, request: ModelRequest) -> StreamAccumulator:
    accumulator = StreamAccumulator(model=request.model)
    async for event in provider.stream(request):
        accumulator.feed(event)
    return accumulator


def test_fake_provider_satisfies_the_protocol() -> None:
    assert isinstance(FakeProvider([]), Provider)


async def test_replays_text_with_realistic_fragmentation() -> None:
    provider = FakeProvider([turn("hello there world")])
    accumulator = await run(provider, request_of())
    assert accumulator.text == "hello there world"
    assert accumulator.finish_reason == "stop"


async def test_explicit_chunking_is_honoured() -> None:
    provider = FakeProvider([ScriptedTurn(text_chunks=["a", "b", "c"])])
    events = [e async for e in provider.stream(request_of())]
    text_events = [e for e in events if e.type == "text_delta"]
    assert [e.text for e in text_events] == ["a", "b", "c"]


async def test_replays_reasoning_tool_calls_and_usage() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn(
                reasoning="I should read the file.",
                text="Reading.",
                tool_calls=[
                    ScriptedToolCall(call_id="c1", name="read_file", arguments={"path": "a.py"}),
                    ScriptedToolCall(
                        call_id="c2", name="grep", arguments_chunks=['{"pat', 'tern":"TODO"}']
                    ),
                ],
                finish_reason="tool_calls",
                usage=Usage(prompt_tokens=10, completion_tokens=4, cost_usd=0.001),
            )
        ]
    )
    accumulator = await run(provider, request_of(tools=["read_file", "grep"]))

    parts = accumulator.parts()
    assert isinstance(parts[0], ReasoningPart)
    assert isinstance(parts[1], TextPart)
    calls = [p for p in parts if isinstance(p, ToolCallPart)]
    assert [c.name for c in calls] == ["read_file", "grep"]
    assert calls[1].arguments == {"pattern": "TODO"}
    assert accumulator.finish_reason == "tool_calls"
    assert accumulator.usage is not None
    assert accumulator.usage.cost_usd == 0.001


async def test_malformed_scripted_arguments_reach_the_model_as_a_parse_error() -> None:
    """The fake must be able to reproduce R-P-4, or the loop's handling is untested."""

    provider = FakeProvider(
        [ScriptedTurn(tool_calls=[ScriptedToolCall(call_id="c", name="w", arguments_chunks=["{"])])]
    )
    accumulator = await run(provider, request_of())
    call = accumulator.parts()[0]
    assert isinstance(call, ToolCallPart)
    assert call.parse_error is not None


async def test_terminal_error_before_any_output() -> None:
    error = ProviderError(kind=ProviderErrorKind.RATE_LIMIT, message="429")
    provider = FakeProvider([ScriptedTurn(error=error)])
    events = [e async for e in provider.stream(request_of())]
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].error.kind is ProviderErrorKind.RATE_LIMIT


async def test_mid_stream_error_after_n_chunks() -> None:
    """Spec C-8's scenario: partial output, then a failure the loop must decide about."""

    provider = FakeProvider(
        [
            ScriptedTurn(
                text_chunks=["one ", "two ", "three "],
                error=ProviderError(kind=ProviderErrorKind.NETWORK, message="cut"),
                error_after_chunks=2,
            )
        ]
    )
    accumulator = await run(provider, request_of())
    assert accumulator.text == "one two "
    assert accumulator.error is not None
    assert accumulator.error.kind is ProviderErrorKind.NETWORK


async def test_delays_let_a_test_interrupt_mid_call() -> None:
    provider = FakeProvider([ScriptedTurn(text="slow", delay_s=5.0)])

    task = asyncio.create_task(run(provider, request_of()))
    await asyncio.sleep(0.05)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 1.0


async def test_chunk_delay_is_applied_between_fragments() -> None:
    provider = FakeProvider([ScriptedTurn(text_chunks=["a", "b", "c"], chunk_delay_s=0.02)])
    started = time.monotonic()
    await run(provider, request_of())
    assert time.monotonic() - started >= 0.05


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


async def test_by_index_matching_walks_the_script_in_order() -> None:
    provider = FakeProvider([turn("first"), turn("second")], match="by_index")
    assert (await run(provider, request_of())).text == "first"
    assert (await run(provider, request_of())).text == "second"


async def test_by_index_matching_raises_when_the_script_runs_out() -> None:
    provider = FakeProvider([turn("only")], match="by_index")
    await run(provider, request_of())
    with pytest.raises(ScriptExhausted, match="by_index"):
        await run(provider, request_of())


async def test_by_request_hash_returns_the_same_turn_for_the_same_transcript() -> None:
    request = request_of(SystemMessage(content="sys"), UserMessage.of("do the thing"))
    key = request.fingerprint()
    provider = FakeProvider([turn("keyed answer", key=key)], match="by_request_hash")

    assert (await run(provider, request)).text == "keyed answer"
    assert (await run(provider, request)).text == "keyed answer", "keys are not consumed"


async def test_by_request_hash_survives_a_shifted_call_index() -> None:
    """Spec delta 21 in one scenario: the kill test's exact failure mode.

    Turn 1 is re-issued after a reload, so the *index* of turn 2 shifts from 1 to 2.
    Hash matching still returns turn 2, because the transcript is the same.
    """

    first = request_of(UserMessage.of("step one"))
    second = request_of(
        UserMessage.of("step one"),
        AssistantMessage(content=[TextPart(text="did step one")], model="x/y"),
        UserMessage.of("step two"),
    )
    script = Script(
        match="by_request_hash",
        turns=[
            turn("did step one", key=first.fingerprint()),
            turn("did step two", key=second.fingerprint()),
        ],
    )

    live = FakeProvider(script)
    assert (await run(live, first)).text == "did step one"

    # Process dies here; the reloaded run re-issues the first call, then continues.
    resumed = FakeProvider(script)
    assert (await run(resumed, first)).text == "did step one"
    assert (await run(resumed, second)).text == "did step two"
    assert resumed.call_index == 2

    indexed = FakeProvider(script, match="by_index")
    await run(indexed, first)
    assert (await run(indexed, second)).text == "did step two", "sanity: index 1 is turn 2"
    with pytest.raises(ScriptExhausted):
        await run(indexed, second)


async def test_unkeyed_turns_are_consumed_in_order_as_a_fallback() -> None:
    provider = FakeProvider([turn("a"), turn("b")], match="by_request_hash")
    assert (await run(provider, request_of(UserMessage.of("one")))).text == "a"
    assert (await run(provider, request_of(UserMessage.of("two")))).text == "b"


async def test_an_unmatched_request_raises_loudly() -> None:
    """A silently improvising fake would let the kill test pass while proving nothing."""

    provider = FakeProvider([turn("x", key="not-the-real-hash")], match="by_request_hash")
    with pytest.raises(ScriptExhausted) as excinfo:
        await run(provider, request_of())
    assert "no turn for request" in str(excinfo.value)


def test_key_for_reports_the_fingerprint_a_script_needs() -> None:
    request = request_of()
    provider = FakeProvider([])
    assert provider.key_for(request) == request.fingerprint()


# ---------------------------------------------------------------------------
# Fingerprint stability
# ---------------------------------------------------------------------------


def test_fingerprint_ignores_ids_timestamps_and_run_metadata() -> None:
    """Ids and timestamps are minted fresh in the resumed process; the hash must not
    depend on them, or nothing would ever match after a reload."""

    def build() -> ModelRequest:
        return ModelRequest(
            model="x/y",
            messages=[
                SystemMessage(content="sys"),
                UserMessage.of("hello"),
                AssistantMessage(
                    content=[
                        TextPart(text="ok"),
                        ToolCallPart(
                            call_id="c1",
                            name="grep",
                            arguments={"p": 1},
                            raw_arguments='{"p":1}',
                        ),
                    ],
                    model="x/y",
                ),
                ToolResultMessage(call_id="c1", name="grep", result=ToolResult.ok_text("hit")),
            ],
        )

    a, b = build(), build()
    assert a.messages[0].id != b.messages[0].id, "ids really are fresh each time"
    assert a.fingerprint() == b.fingerprint()

    a.metadata = {"run_id": "r1", "agent_id": "main"}
    b.metadata = {"run_id": "r2", "agent_id": "main/0"}
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_with_content_model_and_toolset() -> None:
    base = request_of(UserMessage.of("hello"))
    assert base.fingerprint() != request_of(UserMessage.of("goodbye")).fingerprint()
    assert base.fingerprint() != request_of(UserMessage.of("hello"), model="z/w").fingerprint()
    assert (
        base.fingerprint() != request_of(UserMessage.of("hello"), tools=["read_file"]).fingerprint()
    )


def test_fingerprint_is_stable_across_tool_ordering_but_not_membership() -> None:
    """Tool order in a request is not semantically meaningful; membership is."""

    a = request_of(tools=["a", "b"])
    b = request_of(tools=["b", "a"])
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != request_of(tools=["a"]).fingerprint()


def test_fingerprint_distinguishes_tool_results() -> None:
    def with_result(text: str) -> ModelRequest:
        return request_of(
            UserMessage.of("go"),
            AssistantMessage(
                content=[ToolCallPart(call_id="c", name="grep", arguments={}, raw_arguments="{}")],
                model="x/y",
            ),
            ToolResultMessage(call_id="c", name="grep", result=ToolResult.ok_text(text)),
        )

    assert with_result("hit").fingerprint() != with_result("miss").fingerprint()


# ---------------------------------------------------------------------------
# Scripts on disk
# ---------------------------------------------------------------------------


def test_scripts_survive_a_file_round_trip(tmp_path: Path) -> None:
    """The kill test's script is committed JSON referenced by a workflow's config."""

    script = Script(
        turns=[
            turn("hello", key="k1"),
            ScriptedTurn(tool_calls=[tool_call("read_file", {"path": "a.py"})]),
        ]
    )
    path = tmp_path / "script.json"
    script.save(path)
    assert Script.load(path) == script

    provider = FakeProvider(script_path=path)
    assert provider.script == script


async def test_model_info_degrades_to_a_permissive_default() -> None:
    provider = FakeProvider([])
    info = await provider.model_info("anything/at-all")
    assert info is not None
    assert info.supports_tools is True


async def test_complete_works_against_the_fake() -> None:
    provider = FakeProvider([turn("done")])
    message = await complete(provider, request_of())
    assert message.text == "done"


async def test_requests_are_recorded_for_assertions() -> None:
    provider = FakeProvider([turn("a"), turn("b")])
    await run(provider, request_of(UserMessage.of("one")))
    await run(provider, request_of(UserMessage.of("two")))
    assert len(provider.requests) == 2
    first = provider.requests[0].messages[0]
    assert isinstance(first, UserMessage)
    assert first.text == "one"
