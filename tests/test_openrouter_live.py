"""The live half of M0's exit test: a real OpenRouter completion.

Marked `network` and deselected by default, because a test suite that needs an API
key and an internet connection to pass is a test suite people stop running.

    pytest -m network

Skipped, not failed, when no key is present.
"""

from __future__ import annotations

import pytest

from azalabscode.messages import SystemMessage, UserMessage
from azalabscode.providers.base import (
    Finish,
    ModelRequest,
    StreamAccumulator,
    StreamError,
    TextDelta,
    complete,
)
from azalabscode.providers.openrouter import OpenRouterProvider

MODEL = "anthropic/claude-haiku-4.5"

pytestmark = pytest.mark.network


@pytest.fixture
async def live(openrouter_api_key: str, isolated_cache):
    provider = OpenRouterProvider(
        api_key=openrouter_api_key,
        referer="https://github.com/azalabs/azalabscode",
        title="azalabscode-tests",
    )
    try:
        yield provider
    finally:
        await provider.aclose()


def prompt(text: str) -> ModelRequest:
    return ModelRequest(
        model=MODEL,
        messages=[
            SystemMessage(content="You are concise. Answer in at most two sentences."),
            UserMessage.of(text),
        ],
        max_tokens=200,
    )


async def test_streams_a_real_completion(live: OpenRouterProvider) -> None:
    """R-P-3 against the live API: text, a finish reason, and usage."""

    accumulator = StreamAccumulator(model=MODEL)
    deltas = 0
    finishes = 0
    async for event in live.stream(prompt("Explain a ULID in one sentence.")):
        accumulator.feed(event)
        if isinstance(event, TextDelta):
            deltas += 1
        if isinstance(event, Finish):
            finishes += 1
        assert not isinstance(event, StreamError), f"provider error: {event.error}"

    assert deltas > 1, "the response should arrive in more than one fragment"
    assert finishes == 1, "exactly one Finish per stream"
    assert accumulator.text.strip()
    assert accumulator.finish_reason == "stop"
    assert accumulator.usage is not None
    assert accumulator.usage.prompt_tokens > 0
    assert accumulator.usage.completion_tokens > 0
    assert accumulator.usage.cost_usd is not None, "usage: {include: true} was requested"


async def test_streams_a_real_tool_call(live: OpenRouterProvider) -> None:
    """R-P-3: tool-call assembly against a real model's argument deltas."""

    from azalabscode.content import ToolCallPart
    from azalabscode.toolio import ToolSchema

    request = ModelRequest(
        model=MODEL,
        messages=[
            SystemMessage(content="Use the read_file tool when asked to read a file."),
            UserMessage.of("Read the file src/app.py and tell me what it does."),
        ],
        tools=[
            ToolSchema(
                name="read_file",
                description="Read a file from the working directory.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ],
        tool_choice="required",
        max_tokens=200,
    )

    accumulator = StreamAccumulator(model=MODEL)
    async for event in live.stream(request):
        accumulator.feed(event)

    calls = [p for p in accumulator.parts() if isinstance(p, ToolCallPart)]
    assert calls, "tool_choice=required should force a call"
    assert calls[0].name == "read_file"
    assert calls[0].parse_error is None
    assert calls[0].arguments is not None
    assert "app.py" in str(calls[0].arguments.get("path", ""))
    assert calls[0].call_id


async def test_complete_helper_against_the_live_api(live: OpenRouterProvider) -> None:
    message = await complete(live, prompt("Say the word 'ready' and nothing else."))
    assert "ready" in message.text.lower()
    assert message.usage is not None


async def test_model_info_against_the_live_catalogue(live: OpenRouterProvider) -> None:
    """R-P-7 end to end, including the on-disk cache."""

    info = await live.model_info(MODEL)
    assert info is not None
    assert info.context_length and info.context_length > 10_000
    assert info.supports_tools is True

    assert await live.model_info("definitely/not-a-real-model") is None


async def test_a_bad_model_id_is_an_error_event_not_an_exception(
    live: OpenRouterProvider,
) -> None:
    """The streaming path never raises; the caller decides what to do."""

    request = prompt("hello")
    request.model = "definitely/not-a-real-model"
    events = [event async for event in live.stream(request)]
    assert any(isinstance(e, StreamError) for e in events)
