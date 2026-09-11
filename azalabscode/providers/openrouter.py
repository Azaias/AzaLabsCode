"""OpenRouter provider. Every OpenRouter-specific decision in the harness is here.

The three behaviors that are easy to get wrong and expensive to get wrong:

**Retry only before the first byte** (R-P-5, spec C-8). Once a token has arrived,
the call has partially happened; retrying it silently doubles the spend and can
return different text. So the retry loop wraps *opening* the stream, and the moment
the first event is yielded the loop is committed. A later failure ends the stream
with a `StreamError` and the agent loop decides.

**Cancellation closes the connection** (R-P-6). The response is opened inside an
`AsyncExitStack` owned by the generator, so cancelling the iterator runs
`aclose()` on the response as the generator unwinds. Usage after a cancellation is
reported as absent rather than guessed.

**Malformed tool arguments are data** (R-P-4). Assembly buffers argument deltas per
`index` and parses at the end; a parse failure becomes `ToolCallPart.parse_error`
and travels to the model as a structured tool error.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections.abc import AsyncGenerator, Iterable
from contextlib import AsyncExitStack
from typing import Any

import httpx

from azalabscode.errors import ProviderError, ProviderErrorKind
from azalabscode.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
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
from azalabscode.providers.models_cache import ModelsCache, parse_openrouter_models

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
DEFAULT_MAX_ATTEMPTS = 3
API_KEY_ENV = "OPENROUTER_API_KEY"


class OpenRouterProvider:
    """Streaming chat completions against OpenRouter.

    Construct inside a workflow's `build(config)` rather than at module scope
    (spec delta 21): a session records `(import_path, config)` and nothing else, so
    the provider has to be reconstructible from the config alone.
    """

    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        referer: str | None = None,
        title: str | None = None,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        initial_backoff_s: float = 0.5,
        max_backoff_s: float = 8.0,
        client: httpx.AsyncClient | None = None,
        models_cache: ModelsCache | None = None,
        http2: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get(API_KEY_ENV) or ""
        self.base_url = base_url.rstrip("/")
        self.referer = referer
        self.title = title
        self.max_attempts = max(1, max_attempts)
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout),
            http2=http2,
            follow_redirects=True,
        )
        self._models = models_cache or ModelsCache(provider="openrouter")
        self._models_lock = asyncio.Lock()
        self._closed = False

    # -- headers and bodies -------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title
        return headers

    def build_body(self, request: ModelRequest) -> dict[str, Any]:
        """Render a `ModelRequest` into an OpenRouter request body.

        `provider_options` is merged last and wins, which is the point of an escape
        hatch: a user who sets `{"provider": {"order": [...]}}` means it.
        """

        body: dict[str, Any] = {
            "model": request.model,
            "messages": [m for msg in request.messages for m in to_openrouter_message(msg)],
            "stream": True,
            "stream_options": {"include_usage": True},
            "usage": {"include": True},
        }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
            body["tool_choice"] = request.tool_choice
            if request.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = request.parallel_tool_calls
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop"] = request.stop
        if request.seed is not None:
            body["seed"] = request.seed
        if request.reasoning is not None:
            reasoning: dict[str, Any] = {}
            if request.reasoning.effort is not None:
                reasoning["effort"] = request.reasoning.effort
            if request.reasoning.max_tokens is not None:
                reasoning["max_tokens"] = request.reasoning.max_tokens
            if request.reasoning.exclude:
                reasoning["exclude"] = True
            if reasoning:
                body["reasoning"] = reasoning
        body.update(request.provider_options)
        return body

    # -- streaming ----------------------------------------------------------

    async def stream(self, request: ModelRequest) -> AsyncGenerator[StreamEventUnion, None]:
        """Stream one completion. Yields a terminal `StreamError` instead of raising."""

        if not self.api_key:
            yield StreamError(
                error=ProviderError(
                    kind=ProviderErrorKind.AUTH,
                    message=(
                        f"no OpenRouter API key; set {API_KEY_ENV} or pass api_key= to "
                        "OpenRouterProvider"
                    ),
                    provider=self.name,
                )
            )
            return

        body = self.build_body(request)
        url = f"{self.base_url}/chat/completions"

        async with AsyncExitStack() as stack:
            response, error = await self._open_stream_with_retries(stack, url, body)
            if error is not None:
                yield StreamError(error=error)
                return
            assert response is not None
            async for event in self._iter_sse(response):
                yield event

    async def _open_stream_with_retries(
        self,
        stack: AsyncExitStack,
        url: str,
        body: dict[str, Any],
    ) -> tuple[httpx.Response | None, ProviderError | None]:
        """Open the SSE response, retrying transport failures before the first byte.

        Returns `(response, None)` or `(None, error)`. Everything retried here
        happened before any token was produced, so a retry is free of the
        double-spend problem that makes mid-stream retries unacceptable.
        """

        last: ProviderError | None = None
        for attempt in range(1, self.max_attempts + 1):
            # Each attempt owns its own context manager. Ownership transfers to the
            # caller's stack only on success, so a retried attempt cannot leave a
            # half-open response registered for teardown at the end of the stream.
            manager = self._client.stream("POST", url, json=body, headers=self._headers())
            try:
                response = await manager.__aenter__()
            except httpx.HTTPError as exc:
                last = ProviderError(
                    kind=ProviderErrorKind.NETWORK,
                    message=f"{type(exc).__name__}: {exc}",
                    provider=self.name,
                )
            else:
                if response.status_code < 400:
                    stack.push_async_exit(manager)
                    return response, None
                last = await self._error_from_response(response)
                await manager.__aexit__(None, None, None)

            if attempt >= self.max_attempts or not last.retryable:
                return None, last
            await asyncio.sleep(self._backoff(attempt, last.retry_after))
        return None, last  # pragma: no cover - loop always returns

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with jitter, capped, honouring `Retry-After`."""

        if retry_after is not None:
            return min(retry_after, self.max_backoff_s)
        base = min(self.initial_backoff_s * (2 ** (attempt - 1)), self.max_backoff_s)
        return base * (0.5 + random.random() * 0.5)

    async def _error_from_response(self, response: httpx.Response) -> ProviderError:
        """Classify an HTTP error response, reading the body for a usable message."""

        try:
            raw = await response.aread()
            text = raw.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            text = ""
        message = text.strip()[:500] or f"HTTP {response.status_code}"
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                err = payload.get("error")
                if isinstance(err, dict) and isinstance(err.get("message"), str):
                    message = err["message"]
                elif isinstance(err, str):
                    message = err
        except ValueError:
            pass

        status = response.status_code
        if status in (401, 403):
            kind = ProviderErrorKind.AUTH
        elif status == 429:
            kind = ProviderErrorKind.RATE_LIMIT
        elif status >= 500:
            kind = ProviderErrorKind.SERVER
        elif status in (400, 404, 422):
            kind = ProviderErrorKind.MODEL
        else:
            kind = ProviderErrorKind.UNKNOWN

        return ProviderError(
            kind=kind,
            message=message,
            status_code=status,
            retry_after=parse_retry_after(response.headers.get("retry-after")),
            provider=self.name,
            request_id=response.headers.get("x-request-id"),
        )

    async def _iter_sse(self, response: httpx.Response) -> AsyncGenerator[StreamEventUnion, None]:
        """Parse the SSE body into stream events.

        Failures from here on are *not* retried: the first byte has arrived. The
        stream ends with a `StreamError` and the caller decides (spec C-8).
        """

        open_indices: set[int] = set()
        finished = False
        try:
            async for line in response.aiter_lines():
                line = line.rstrip("\r")
                if not line or line.startswith(":"):
                    # Blank lines separate SSE frames; ":" lines are keep-alive
                    # comments, which OpenRouter sends during long model queues.
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except ValueError:
                    continue
                for event in self._events_from_chunk(payload, open_indices):
                    if isinstance(event, Finish):
                        if finished:
                            # OpenRouter repeats `finish_reason` on the trailing
                            # usage chunk. A stream yields exactly one `Finish`, or
                            # the agent loop sees one response end twice.
                            continue
                        finished = True
                    yield event
                    if isinstance(event, StreamError):
                        # A `StreamError` is terminal. Closing out the open tool
                        # calls and appending a synthetic `Finish` after one would
                        # tell the caller the call succeeded, which is exactly the
                        # thing spec C-8 hands to the agent loop to decide.
                        return
        except httpx.HTTPError as exc:
            yield StreamError(
                error=ProviderError(
                    kind=ProviderErrorKind.NETWORK,
                    message=f"stream interrupted: {type(exc).__name__}: {exc}",
                    provider=self.name,
                )
            )
            return

        for index in sorted(open_indices):
            yield ToolCallEnd(index=index)
        if not finished:
            yield Finish(reason="stop")

    def _events_from_chunk(
        self, payload: dict[str, Any], open_indices: set[int]
    ) -> Iterable[StreamEventUnion]:
        """Translate one SSE JSON chunk into zero or more stream events."""

        events: list[StreamEventUnion] = []

        error = payload.get("error")
        if isinstance(error, dict):
            events.append(
                StreamError(
                    error=ProviderError(
                        kind=_kind_from_code(error.get("code")),
                        message=str(error.get("message", "provider error")),
                        status_code=(
                            error.get("code") if isinstance(error.get("code"), int) else None
                        ),
                        provider=self.name,
                    )
                )
            )
            return events

        for choice in payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                events.append(ReasoningDelta(text=reasoning))
            reasoning_details = delta.get("reasoning_details")
            if isinstance(reasoning_details, list):
                for detail in reasoning_details:
                    if not isinstance(detail, dict):
                        continue
                    text = detail.get("text") or detail.get("summary")
                    if isinstance(text, str) and text:
                        events.append(
                            ReasoningDelta(
                                text=text,
                                signature=(
                                    detail.get("signature")
                                    if isinstance(detail.get("signature"), str)
                                    else None
                                ),
                            )
                        )

            content = delta.get("content")
            if isinstance(content, str) and content:
                events.append(TextDelta(text=content))

            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                index = call.get("index", 0)
                if not isinstance(index, int):
                    index = 0
                function = call.get("function") or {}
                raw_name = function.get("name")
                raw_id = call.get("id")
                name = raw_name if isinstance(raw_name, str) else ""
                call_id = raw_id if isinstance(raw_id, str) else ""
                first_seen = index not in open_indices
                # Re-emitting the header when a later chunk fills in an id or name
                # the first one omitted is harmless: the accumulator merges headers
                # for a slot and takes whichever value is non-empty.
                if first_seen or call_id or name:
                    open_indices.add(index)
                    events.append(ToolCallStart(index=index, call_id=call_id, name=name))
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    events.append(ToolCallDelta(index=index, arguments_delta=arguments))

            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                for index in sorted(open_indices):
                    events.append(ToolCallEnd(index=index))
                open_indices.clear()
                events.append(Finish(reason=finish_reason))

        usage = payload.get("usage")
        if isinstance(usage, dict):
            events.append(UsageReport(usage=usage_from_openrouter(usage)))

        return events

    # -- metadata -----------------------------------------------------------

    async def model_info(self, model: str) -> ModelInfo | None:
        """Capability metadata for a model, cached on disk with a TTL (R-P-7)."""

        cached = self._models.get(model)
        if cached is not None:
            return cached
        async with self._models_lock:
            cached = self._models.get(model)
            if cached is not None:
                return cached
            models = await self._fetch_models()
            if models is None:
                return None
            self._models.store(models)
            return models.get(model)

    async def _fetch_models(self) -> dict[str, ModelInfo] | None:
        """Fetch `/models`. Any failure returns `None`: metadata never blocks a call."""

        try:
            response = await self._client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            if response.status_code >= 400:
                return None
            return parse_openrouter_models(response.json())
        except (httpx.HTTPError, ValueError):
            return None

    async def aclose(self) -> None:
        """Close the HTTP client, if this provider owns it. Idempotent."""

        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def to_openrouter_message(message: Message) -> list[dict[str, Any]]:
    """Convert one harness message into OpenRouter wire messages.

    Returns a list because an assistant turn's tool calls and the tool results that
    answer them are separate wire messages, and because a `ToolResultMessage`
    carrying an image needs a follow-up user message: the `tool` role accepts text
    only on every route OpenRouter fronts.
    """

    match message:
        case SystemMessage():
            return [{"role": "system", "content": message.content}]

        case UserMessage():
            return [{"role": "user", "content": _content_blocks(message.content)}]

        case AssistantMessage():
            out: dict[str, Any] = {"role": "assistant"}
            text = message.text
            out["content"] = text if text else None
            reasoning = [p for p in message.content if p.type == "reasoning"]
            if reasoning:
                out["reasoning"] = "".join(p.text for p in reasoning)
            calls = message.tool_calls
            if calls:
                out["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.raw_arguments or json.dumps(call.arguments or {}),
                        },
                    }
                    for call in calls
                ]
            return [out]

        case ToolResultMessage():
            wire: list[dict[str, Any]] = [
                {
                    "role": "tool",
                    "tool_call_id": message.call_id,
                    "content": message.result.text or _empty_result_text(message),
                }
            ]
            images = [p for p in message.result.content if p.type == "image"]
            if images:
                wire.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{p.media_type};base64,{p.data_b64}"},
                            }
                            for p in images
                        ],
                    }
                )
            return wire

    raise TypeError(f"unsupported message type: {type(message).__name__}")


def _empty_result_text(message: ToolResultMessage) -> str:
    """Text for a result with no text content.

    An empty string on the `tool` role makes several models loop, re-issuing the
    same call. Saying so explicitly is cheaper than the retry.
    """

    if message.result.error is not None:
        return f"[{message.result.error.kind}] {message.result.error.message}"
    return "(the tool produced no output)"


def _content_blocks(parts: list[Any]) -> Any:
    """Render user content parts as OpenRouter content blocks.

    Text-only content collapses to a plain string, which is what every route
    accepts; multimodal content becomes the block list.
    """

    if all(p.type == "text" for p in parts):
        return "".join(p.text for p in parts)
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if part.type == "text":
            blocks.append({"type": "text", "text": part.text})
        elif part.type == "image":
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{part.media_type};base64,{part.data_b64}"},
                }
            )
    return blocks


def usage_from_openrouter(payload: dict[str, Any]) -> Usage:
    """Convert a `usage` object, tolerating the several shapes OpenRouter returns."""

    def as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    details = payload.get("prompt_tokens_details") or {}
    completion_details = payload.get("completion_tokens_details") or {}
    cost = payload.get("cost")
    try:
        cost_usd = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost_usd = None

    return Usage(
        prompt_tokens=as_int(payload.get("prompt_tokens")),
        completion_tokens=as_int(payload.get("completion_tokens")),
        cached_tokens=as_int(details.get("cached_tokens")),
        reasoning_tokens=as_int(completion_details.get("reasoning_tokens")),
        cost_usd=cost_usd,
    )


def parse_retry_after(value: str | None) -> float | None:
    """Parse a `Retry-After` header. Seconds only; HTTP-date form is ignored."""

    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _kind_from_code(code: Any) -> ProviderErrorKind:
    """Classify an in-stream error code."""

    if not isinstance(code, int):
        return ProviderErrorKind.UNKNOWN
    if code in (401, 403):
        return ProviderErrorKind.AUTH
    if code == 429:
        return ProviderErrorKind.RATE_LIMIT
    if code >= 500:
        return ProviderErrorKind.SERVER
    if code >= 400:
        return ProviderErrorKind.MODEL
    return ProviderErrorKind.UNKNOWN


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "OpenRouterProvider",
    "parse_retry_after",
    "to_openrouter_message",
    "usage_from_openrouter",
]
