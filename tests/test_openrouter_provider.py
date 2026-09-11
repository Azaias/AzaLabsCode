"""`OpenRouterProvider` against recorded SSE fixtures via `respx` (R-P-3..R-P-7).

The fixtures in `tests/fixtures/*.sse` are shaped like real OpenRouter traffic,
including the parts that are easy to forget and that break naive parsers: the
`: OPENROUTER PROCESSING` keep-alive comment, a final usage chunk with an empty
`choices` array, tool-call arguments split mid-token across deltas, and a stream
that ends without a `finish_reason`.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
import respx

from azalabscode.content import ReasoningPart, TextPart, ToolCallPart
from azalabscode.errors import ProviderCallError, ProviderErrorKind
from azalabscode.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from azalabscode.providers.base import (
    Finish,
    ModelRequest,
    Provider,
    ReasoningConfig,
    StreamAccumulator,
    StreamError,
    TextDelta,
    UsageReport,
    complete,
)
from azalabscode.providers.models_cache import ModelsCache, parse_openrouter_models
from azalabscode.providers.openrouter import (
    DEFAULT_BASE_URL,
    OpenRouterProvider,
    parse_retry_after,
    to_openrouter_message,
    usage_from_openrouter,
)
from azalabscode.toolio import ToolResult, ToolSchema

COMPLETIONS = f"{DEFAULT_BASE_URL}/chat/completions"
MODELS = f"{DEFAULT_BASE_URL}/models"


def read_fixture(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")


def sse_response(name: str) -> httpx.Response:
    return httpx.Response(
        200,
        text=read_fixture(name),
        headers={"content-type": "text/event-stream"},
    )


def request_of(*, tools: bool = False, model: str = "anthropic/claude-sonnet-4") -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=[SystemMessage(content="be brief"), UserMessage.of("what is a ULID?")],
        tools=(
            [
                ToolSchema(
                    name="read_file",
                    description="Read a file.",
                    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                )
            ]
            if tools
            else []
        ),
    )


@pytest.fixture
def provider(isolated_cache: Path) -> OpenRouterProvider:
    return OpenRouterProvider(api_key="test-key", http2=False)


async def collect(provider: OpenRouterProvider, request: ModelRequest) -> list:
    return [event async for event in provider.stream(request)]


# ---------------------------------------------------------------------------
# The protocol itself
# ---------------------------------------------------------------------------


def test_openrouter_provider_satisfies_the_protocol(provider: OpenRouterProvider) -> None:
    """R-P-1: structural conformance, not inheritance."""

    assert isinstance(provider, Provider)


def test_request_and_stream_types_name_no_vendor() -> None:
    """R-P-2: nothing OpenRouter-specific leaks into the provider-agnostic types."""

    for model in (ModelRequest, ReasoningConfig):
        blob = json.dumps(model.model_json_schema())
        assert "openrouter" not in blob.lower()


# ---------------------------------------------------------------------------
# R-P-3: text, reasoning, tool calls, usage, finish reason
# ---------------------------------------------------------------------------


@respx.mock
async def test_text_stream(provider: OpenRouterProvider) -> None:
    respx.post(COMPLETIONS).mock(return_value=sse_response("text_stream.sse"))

    events = await collect(provider, request_of())
    accumulator = StreamAccumulator(model="anthropic/claude-sonnet-4")
    for event in events:
        accumulator.feed(event)

    assert accumulator.text == "A ULID is a sortable identifier."
    assert accumulator.finish_reason == "stop"
    assert accumulator.error is None
    assert accumulator.usage is not None
    assert accumulator.usage.prompt_tokens == 18
    assert accumulator.usage.completion_tokens == 7
    assert accumulator.usage.cached_tokens == 4
    assert accumulator.usage.cost_usd == pytest.approx(0.000123)

    message = accumulator.message()
    assert isinstance(message, AssistantMessage)
    assert message.text == "A ULID is a sortable identifier."
    assert message.tool_calls == []


@respx.mock
async def test_keepalive_comments_and_blank_lines_are_ignored(
    provider: OpenRouterProvider,
) -> None:
    """The `: OPENROUTER PROCESSING` line is a comment, not a data frame."""

    respx.post(COMPLETIONS).mock(return_value=sse_response("text_stream.sse"))
    events = await collect(provider, request_of())
    assert all(not isinstance(e, StreamError) for e in events)
    assert sum(isinstance(e, TextDelta) for e in events) == 3


@respx.mock
async def test_reasoning_stream(provider: OpenRouterProvider) -> None:
    """Both the plain `reasoning` field and `reasoning_details[]` are surfaced."""

    respx.post(COMPLETIONS).mock(return_value=sse_response("reasoning_stream.sse"))

    accumulator = StreamAccumulator(model="x/thinker")
    async for event in provider.stream(request_of(model="x/thinker")):
        accumulator.feed(event)

    parts = accumulator.parts()
    reasoning = [p for p in parts if isinstance(p, ReasoningPart)]
    assert len(reasoning) == 1
    assert reasoning[0].text == "Let me check that. Confirmed."
    assert reasoning[0].signature == "sig-abc"
    assert next(p for p in parts if isinstance(p, TextPart)).text == "Yes."
    assert accumulator.usage is not None
    assert accumulator.usage.reasoning_tokens == 9


@respx.mock
async def test_tool_call_assembly_from_argument_deltas(provider: OpenRouterProvider) -> None:
    """R-P-3: arguments split mid-token across deltas reassemble per `index`."""

    respx.post(COMPLETIONS).mock(return_value=sse_response("tool_calls_stream.sse"))

    accumulator = StreamAccumulator(model="x/tooler")
    async for event in provider.stream(request_of(tools=True, model="x/tooler")):
        accumulator.feed(event)

    calls = [p for p in accumulator.parts() if isinstance(p, ToolCallPart)]
    assert [c.name for c in calls] == ["read_file", "grep"]
    assert [c.call_id for c in calls] == ["call_aaa", "call_bbb"]
    assert calls[0].arguments == {"path": "src/app.py"}
    assert calls[1].arguments == {"pattern": "TODO"}
    assert all(c.parse_error is None for c in calls)
    assert accumulator.finish_reason == "tool_calls"


@respx.mock
async def test_tool_calls_are_ordered_by_index_not_arrival(
    provider: OpenRouterProvider,
) -> None:
    """The transcript invariant is defined in call order, so assembly must be too."""

    body = (
        "".join(
            f"data: {c}\n\n"
            for c in (
                '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"second","type":"function","function":{"name":"b","arguments":"{}"}}]}}]}',
                '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"first","type":"function","function":{"name":"a","arguments":"{}"}}]}}]}',
                '{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            )
        )
        + "data: [DONE]\n\n"
    )
    respx.post(COMPLETIONS).mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    accumulator = StreamAccumulator()
    async for event in provider.stream(request_of(tools=True)):
        accumulator.feed(event)

    calls = [p for p in accumulator.parts() if isinstance(p, ToolCallPart)]
    assert [c.call_id for c in calls] == ["first", "second"]


# ---------------------------------------------------------------------------
# R-P-4: malformed tool JSON is data
# ---------------------------------------------------------------------------


@respx.mock
async def test_malformed_tool_arguments_become_a_parse_error(
    provider: OpenRouterProvider,
) -> None:
    respx.post(COMPLETIONS).mock(return_value=sse_response("malformed_tool_args.sse"))

    accumulator = StreamAccumulator()
    async for event in provider.stream(request_of(tools=True)):
        accumulator.feed(event)

    calls = [p for p in accumulator.parts() if isinstance(p, ToolCallPart)]
    assert len(calls) == 1
    assert calls[0].parse_error is not None
    assert calls[0].arguments is None
    assert calls[0].raw_arguments == '{"path": "a.txt", "content"'
    assert calls[0].ok is False


@respx.mock
async def test_exactly_one_finish_event_per_stream(provider: OpenRouterProvider) -> None:
    """OpenRouter repeats `finish_reason` on the trailing usage chunk.

    Passing both through tells the agent loop one response ended twice, which ends
    a turn that has not ended. Caught against the live API, pinned here.
    """

    body = (
        "".join(
            f"data: {c}\n\n"
            for c in (
                '{"choices":[{"index":0,"delta":{"content":"hi"}}]}',
                '{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                '{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":1,"completion_tokens":1}}',
            )
        )
        + "data: [DONE]\n\n"
    )
    respx.post(COMPLETIONS).mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    events = await collect(provider, request_of())
    assert sum(isinstance(e, Finish) for e in events) == 1
    assert sum(isinstance(e, UsageReport) for e in events) == 1


@respx.mock
async def test_a_stream_that_ends_without_a_finish_reason_still_closes_its_calls(
    provider: OpenRouterProvider,
) -> None:
    """A truncated stream must not leave a tool call unparsed and unfinished."""

    respx.post(COMPLETIONS).mock(return_value=sse_response("truncated_stream.sse"))

    events = await collect(provider, request_of(tools=True))
    assert isinstance(events[-1], Finish)
    accumulator = StreamAccumulator()
    for event in events:
        accumulator.feed(event)
    calls = [p for p in accumulator.parts() if isinstance(p, ToolCallPart)]
    assert calls[0].arguments == {"pattern": "**/*.py"}


# ---------------------------------------------------------------------------
# R-P-5 and spec C-8: retry before the first byte only
# ---------------------------------------------------------------------------


@respx.mock
async def test_retries_a_500_before_the_first_byte(provider: OpenRouterProvider) -> None:
    provider.initial_backoff_s = 0.0
    route = respx.post(COMPLETIONS).mock(
        side_effect=[
            httpx.Response(500, text="upstream boom"),
            sse_response("text_stream.sse"),
        ]
    )

    accumulator = StreamAccumulator()
    async for event in provider.stream(request_of()):
        accumulator.feed(event)

    assert route.call_count == 2
    assert accumulator.error is None
    assert accumulator.text.startswith("A ULID")


@respx.mock
async def test_retries_a_connect_error_before_the_first_byte(
    provider: OpenRouterProvider,
) -> None:
    provider.initial_backoff_s = 0.0
    route = respx.post(COMPLETIONS).mock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            sse_response("text_stream.sse"),
        ]
    )

    events = await collect(provider, request_of())
    assert route.call_count == 2
    assert not any(isinstance(e, StreamError) for e in events)


@respx.mock
async def test_gives_up_after_max_attempts_and_yields_an_error_event(
    provider: OpenRouterProvider,
) -> None:
    provider.initial_backoff_s = 0.0
    route = respx.post(COMPLETIONS).mock(return_value=httpx.Response(503, text="unavailable"))

    events = await collect(provider, request_of())
    assert route.call_count == provider.max_attempts
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].error.kind is ProviderErrorKind.SERVER
    assert events[0].error.status_code == 503


@respx.mock
async def test_auth_failure_is_not_retried(provider: OpenRouterProvider) -> None:
    """A 401 will not become a 200 on the next try; retrying only wastes time."""

    route = respx.post(COMPLETIONS).mock(
        return_value=httpx.Response(401, json={"error": {"message": "No auth credentials found"}})
    )

    events = await collect(provider, request_of())
    assert route.call_count == 1
    assert isinstance(events[0], StreamError)
    assert events[0].error.kind is ProviderErrorKind.AUTH
    assert events[0].error.message == "No auth credentials found"


@respx.mock
async def test_rate_limit_honours_retry_after(provider: OpenRouterProvider) -> None:
    route = respx.post(COMPLETIONS).mock(
        side_effect=[
            httpx.Response(429, text="slow down", headers={"retry-after": "0.05"}),
            sse_response("text_stream.sse"),
        ]
    )

    started = time.monotonic()
    events = await collect(provider, request_of())
    elapsed = time.monotonic() - started

    assert route.call_count == 2
    assert not any(isinstance(e, StreamError) for e in events)
    # One Windows timer tick of slack. `asyncio.sleep` schedules against the loop
    # clock, whose resolution is ~15.6 ms here, so a 50 ms sleep measured with
    # `time.monotonic` legitimately comes back a few milliseconds short. Asserting
    # the exact bound made this test fail about one run in twenty.
    assert elapsed >= 0.05 - 0.016


@respx.mock
async def test_mid_stream_failure_is_not_retried(provider: OpenRouterProvider) -> None:
    """R-P-5 / spec C-8: once a token has arrived, the provider is committed."""

    route = respx.post(COMPLETIONS).mock(return_value=sse_response("error_mid_stream.sse"))

    events = await collect(provider, request_of())

    assert route.call_count == 1
    assert isinstance(events[-1], StreamError)
    assert events[-1].error.kind is ProviderErrorKind.SERVER
    accumulator = StreamAccumulator()
    for event in events:
        accumulator.feed(event)
    assert accumulator.text == "Starting the answ"
    assert accumulator.error is not None


def test_retry_after_parsing() -> None:
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert parse_retry_after("-1") is None


# ---------------------------------------------------------------------------
# R-P-6: cancellation closes the connection promptly
# ---------------------------------------------------------------------------


@respx.mock
async def test_cancelling_the_iterator_closes_the_response(
    provider: OpenRouterProvider,
) -> None:
    """R-P-6: breaking out of the loop must not leak the response."""

    respx.post(COMPLETIONS).mock(return_value=sse_response("text_stream.sse"))

    stream = provider.stream(request_of())
    first = await anext(stream)
    assert isinstance(first, TextDelta)

    started = time.monotonic()
    await stream.aclose()
    assert time.monotonic() - started < 1.0


@respx.mock
async def test_cancelling_the_consuming_task_closes_the_response(
    provider: OpenRouterProvider,
) -> None:
    """The same, through `task.cancel()`, which is how interrupt actually arrives."""

    async def slow_body() -> httpx.Response:
        return sse_response("text_stream.sse")

    respx.post(COMPLETIONS).mock(return_value=await slow_body())

    seen: list[object] = []
    done = asyncio.Event()

    async def consume() -> None:
        async for event in provider.stream(request_of()):
            seen.append(event)
            done.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(done.wait(), timeout=2)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 1.0
    assert seen


@respx.mock
async def test_usage_is_absent_rather_than_guessed_after_cancellation(
    provider: OpenRouterProvider,
) -> None:
    """R-P-6: the provider does not invent a usage figure for a partial call."""

    respx.post(COMPLETIONS).mock(return_value=sse_response("text_stream.sse"))

    accumulator = StreamAccumulator(model="x/y")
    stream = provider.stream(request_of())
    accumulator.feed(await anext(stream))
    await stream.aclose()

    assert accumulator.usage is None
    assert accumulator.message(cancelled=True).usage is None
    assert accumulator.message(cancelled=True).cancelled is True


# ---------------------------------------------------------------------------
# R-P-7: model metadata, cached, degrading to unknown
# ---------------------------------------------------------------------------


@respx.mock
async def test_model_info_is_fetched_parsed_and_cached(provider: OpenRouterProvider) -> None:
    route = respx.get(MODELS).mock(
        return_value=httpx.Response(200, text=read_fixture("models.json"))
    )

    info = await provider.model_info("anthropic/claude-sonnet-4")
    assert info is not None
    assert info.context_length == 200_000
    assert info.max_output_tokens == 64_000
    assert info.supports_tools is True
    assert info.supports_images is True
    assert info.pricing is not None
    assert info.pricing.prompt == pytest.approx(3e-6)

    again = await provider.model_info("anthropic/claude-sonnet-4")
    assert again == info
    assert route.call_count == 1, "second lookup should come from the cache"


@respx.mock
async def test_model_info_degrades_to_none_when_the_endpoint_fails(
    provider: OpenRouterProvider,
) -> None:
    """R-P-7: missing metadata never blocks a call."""

    respx.get(MODELS).mock(return_value=httpx.Response(500))
    assert await provider.model_info("anything") is None


@respx.mock
async def test_model_info_returns_none_for_an_unknown_model(
    provider: OpenRouterProvider,
) -> None:
    respx.get(MODELS).mock(return_value=httpx.Response(200, text=read_fixture("models.json")))
    assert await provider.model_info("nobody/nothing") is None


def test_model_catalogue_parsing_tolerates_junk() -> None:
    """A malformed entry is skipped; a malformed field becomes `None`."""

    payload = json.loads(read_fixture("models.json"))
    models = parse_openrouter_models(payload)
    assert set(models) == {"anthropic/claude-sonnet-4", "x/minimal"}
    minimal = models["x/minimal"]
    assert minimal.context_length is None
    assert minimal.pricing is not None
    assert minimal.pricing.prompt is None


def test_models_cache_expires_and_survives_corruption(tmp_path: Path) -> None:
    cache = ModelsCache(provider="openrouter", cache_dir=tmp_path, ttl_s=0.0)
    models = parse_openrouter_models(json.loads(read_fixture("models.json")))
    cache.store(models)
    assert cache.load() is None, "a zero TTL means always stale"

    fresh = ModelsCache(provider="openrouter", cache_dir=tmp_path, ttl_s=3600)
    assert fresh.load() is not None
    cache.path.write_text("{not json", encoding="utf-8")
    broken = ModelsCache(provider="openrouter", cache_dir=tmp_path, ttl_s=3600)
    assert broken.load() is None
    assert not list(tmp_path.glob("*.tmp")), "no temp residue"


# ---------------------------------------------------------------------------
# Request bodies and message conversion
# ---------------------------------------------------------------------------


def test_request_body_carries_the_streaming_and_usage_flags(
    provider: OpenRouterProvider,
) -> None:
    body = provider.build_body(request_of(tools=True))
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["usage"] == {"include": True}
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert body["tool_choice"] == "auto"


def test_provider_options_pass_through_verbatim_and_win(
    provider: OpenRouterProvider,
) -> None:
    """Spec C-9: the escape hatch is opaque, and a user who sets it means it."""

    request = request_of()
    request.provider_options = {
        "provider": {"order": ["anthropic"], "allow_fallbacks": False},
        "transforms": ["middle-out"],
        "temperature": 0.9,
    }
    body = provider.build_body(request)
    assert body["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
    assert body["transforms"] == ["middle-out"]
    assert body["temperature"] == 0.9


def test_reasoning_config_is_mapped_not_passed_through(provider: OpenRouterProvider) -> None:
    request = request_of()
    request.reasoning = ReasoningConfig(effort="high", max_tokens=4000, exclude=True)
    body = provider.build_body(request)
    assert body["reasoning"] == {"effort": "high", "max_tokens": 4000, "exclude": True}


def test_headers_include_referer_and_title_when_configured(isolated_cache: Path) -> None:
    provider = OpenRouterProvider(
        api_key="k", referer="https://example.test", title="azalabscode", http2=False
    )
    headers = provider._headers()
    assert headers["Authorization"] == "Bearer k"
    assert headers["HTTP-Referer"] == "https://example.test"
    assert headers["X-Title"] == "azalabscode"


def test_message_conversion_covers_every_role() -> None:
    system = to_openrouter_message(SystemMessage(content="be brief"))
    assert system == [{"role": "system", "content": "be brief"}]

    user = to_openrouter_message(UserMessage.of("hello"))
    assert user == [{"role": "user", "content": "hello"}]

    assistant = to_openrouter_message(
        AssistantMessage(
            content=[
                ReasoningPart(text="thinking"),
                TextPart(text="ok"),
                ToolCallPart(
                    call_id="c1",
                    name="grep",
                    arguments={"pattern": "x"},
                    raw_arguments='{"pattern":"x"}',
                ),
            ],
            model="x/y",
        )
    )
    assert assistant[0]["role"] == "assistant"
    assert assistant[0]["content"] == "ok"
    assert assistant[0]["reasoning"] == "thinking"
    assert assistant[0]["tool_calls"][0]["id"] == "c1"
    assert assistant[0]["tool_calls"][0]["function"]["arguments"] == '{"pattern":"x"}'

    result = to_openrouter_message(
        ToolResultMessage(call_id="c1", name="grep", result=ToolResult.ok_text("a.py:1:x"))
    )
    assert result == [{"role": "tool", "tool_call_id": "c1", "content": "a.py:1:x"}]


def test_multimodal_user_content_becomes_blocks() -> None:
    from azalabscode.content import ImagePart

    wire = to_openrouter_message(
        UserMessage(
            content=[TextPart(text="see"), ImagePart(media_type="image/png", data_b64="eA==")]
        )
    )
    blocks = wire[0]["content"]
    assert blocks[0] == {"type": "text", "text": "see"}
    assert blocks[1]["image_url"]["url"] == "data:image/png;base64,eA=="


def test_an_empty_tool_result_says_so_rather_than_sending_an_empty_string() -> None:
    """Several models loop on an empty `tool` message, re-issuing the same call."""

    wire = to_openrouter_message(
        ToolResultMessage(call_id="c", name="glob", result=ToolResult(ok=True, content=[]))
    )
    assert wire[0]["content"] == "(the tool produced no output)"


def test_an_image_bearing_tool_result_appends_a_user_message() -> None:
    """The `tool` role is text-only on every route OpenRouter fronts."""

    from azalabscode.content import ImagePart

    wire = to_openrouter_message(
        ToolResultMessage(
            call_id="c",
            name="read_file",
            result=ToolResult(
                ok=True,
                content=[
                    TextPart(text="read logo.png"),
                    ImagePart(media_type="image/png", data_b64="eA=="),
                ],
            ),
        )
    )
    assert len(wire) == 2
    assert wire[0]["role"] == "tool"
    assert wire[1]["role"] == "user"


def test_usage_conversion_tolerates_missing_and_junk_fields() -> None:
    assert usage_from_openrouter({}).total_tokens == 0
    usage = usage_from_openrouter({"prompt_tokens": "12", "completion_tokens": None, "cost": "bad"})
    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 0
    assert usage.cost_usd is None


# ---------------------------------------------------------------------------
# The non-streaming helper (R-P-1)
# ---------------------------------------------------------------------------


@respx.mock
async def test_complete_is_built_on_stream(provider: OpenRouterProvider) -> None:
    respx.post(COMPLETIONS).mock(return_value=sse_response("text_stream.sse"))

    message = await complete(provider, request_of())
    assert message.text == "A ULID is a sortable identifier."
    assert message.finish_reason == "stop"
    assert message.usage is not None


@respx.mock
async def test_complete_raises_where_stream_yields_an_error(
    provider: OpenRouterProvider,
) -> None:
    """A non-streaming caller has no error event to inspect, so it gets an exception."""

    provider.initial_backoff_s = 0.0
    respx.post(COMPLETIONS).mock(return_value=httpx.Response(503, text="down"))

    with pytest.raises(ProviderCallError) as excinfo:
        await complete(provider, request_of())
    assert excinfo.value.error.kind is ProviderErrorKind.SERVER


async def test_missing_api_key_is_an_error_event_not_a_crash(
    isolated_cache: Path, no_api_key: None
) -> None:
    provider = OpenRouterProvider(http2=False)
    events = [e async for e in provider.stream(request_of())]
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].error.kind is ProviderErrorKind.AUTH
    assert "OPENROUTER_API_KEY" in events[0].error.message


async def test_aclose_is_idempotent(provider: OpenRouterProvider) -> None:
    await provider.aclose()
    await provider.aclose()


# ---------------------------------------------------------------------------
# The accumulator, independently of transport
# ---------------------------------------------------------------------------


def test_accumulator_drops_every_tool_call_on_cancellation() -> None:
    """Spec delta 15: none of them were dispatched, so none of them are kept."""

    accumulator = StreamAccumulator(model="x/y")
    accumulator.feed(TextDelta(text="I will read "))
    accumulator.feed(TextDelta(text="two files."))
    from azalabscode.providers.base import ToolCallDelta, ToolCallStart

    accumulator.feed(ToolCallStart(index=0, call_id="a", name="read_file"))
    accumulator.feed(ToolCallDelta(index=0, arguments_delta='{"path":"a"}'))
    accumulator.feed(ToolCallStart(index=1, call_id="b", name="read_file"))
    accumulator.feed(ToolCallDelta(index=1, arguments_delta='{"path":'))

    kept = accumulator.message(cancelled=True, drop_tool_calls=True)
    assert kept.cancelled is True
    assert kept.tool_calls == []
    assert kept.text == "I will read two files."

    intact = accumulator.message()
    assert [c.call_id for c in intact.tool_calls] == ["a", "b"]


def test_accumulator_knows_when_only_tool_calls_were_produced() -> None:
    """Such a response is discarded entirely on cancellation (spec delta 15)."""

    from azalabscode.providers.base import ToolCallStart

    accumulator = StreamAccumulator()
    accumulator.feed(ToolCallStart(index=0, call_id="a", name="grep"))
    assert accumulator.has_only_tool_calls() is True

    accumulator.feed(TextDelta(text="here"))
    assert accumulator.has_only_tool_calls() is False


def test_accumulator_fills_in_a_late_arriving_call_id() -> None:
    """Some routes send the header with an empty id and fill it on a later chunk."""

    from azalabscode.providers.base import ToolCallStart

    accumulator = StreamAccumulator()
    accumulator.feed(ToolCallStart(index=0, call_id="", name=""))
    accumulator.feed(ToolCallStart(index=0, call_id="late", name="grep"))
    call = accumulator.parts()[0]
    assert isinstance(call, ToolCallPart)
    assert call.call_id == "late"
    assert call.name == "grep"


def test_accumulator_mints_a_call_id_when_the_provider_omits_one() -> None:
    """A tool call with no id cannot be matched to a result; one is minted."""

    from azalabscode.providers.base import ToolCallDelta

    accumulator = StreamAccumulator()
    accumulator.feed(ToolCallDelta(index=3, arguments_delta="{}"))
    call = accumulator.parts()[0]
    assert isinstance(call, ToolCallPart)
    assert call.call_id.startswith("call_")


def test_accumulator_records_usage_from_a_usage_event() -> None:
    from azalabscode.messages import Usage

    accumulator = StreamAccumulator()
    accumulator.feed(UsageReport(usage=Usage(prompt_tokens=5, completion_tokens=6)))
    assert accumulator.usage is not None
    assert accumulator.usage.total_tokens == 11
