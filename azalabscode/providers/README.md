# `azalabscode.providers` — model access

One protocol, two implementations, and one rule: nothing above this layer is allowed to
know which provider it is talking to.

```python
from azalabscode.providers import ModelRequest, OpenRouterProvider

provider = OpenRouterProvider()                 # reads OPENROUTER_API_KEY
request = ModelRequest(model="anthropic/claude-haiku-4.5", messages=[...], tools=[...])
async for event in provider.stream(request):
    ...                                          # TextDelta | ReasoningDelta | ToolCall* | Finish
await provider.aclose()
```

## The protocol

`Provider` has three members: `stream(ModelRequest) -> AsyncIterator[StreamEvent]`,
`model_info(model)`, and `aclose()`. `complete()` is a module-level helper that just
drains `stream()` — a convenience built on the one real path, not a second path of its
own. (A provider that streamed and completed through separate code would have two sets
of bugs instead of one.)

`StreamEvent` is the union `TextDelta | ReasoningDelta | ToolCallStart | ToolCallDelta |
ToolCallEnd | Finish | UsageReport | StreamError`. `StreamAccumulator` folds that
stream into a single `AssistantMessage`, which is what the agent loop actually wants. A
caller that only needs the tokens can ignore the accumulator.

## `OpenRouterProvider`

An `httpx.AsyncClient` over HTTP/2: it parses SSE, buffers tool-call deltas per
`index`, and sets `usage: {include: true}` so the cost arrives on the same stream as
the text.

Four behaviors are why this file is longer than a plain SDK call:

- **Retries only fire before the first byte arrives.** Once a token has been delivered,
  retrying would duplicate output. Backoff is exponential with jitter, and it honors
  `Retry-After`.
- **Cancelling the iterator closes the response within a second.** An interrupt that
  leaves a socket quietly draining in the background isn't really an interrupt.
- **Malformed tool-call JSON becomes a `ToolCallPart` with a `parse_error`** rather than
  an exception. The model wrote something invalid — that's a fact to hand back to the
  agent loop, not a reason to crash.
- **`provider_options` passes straight through, untouched,** and is opaque to every
  other layer. This is the entire mechanism that keeps OpenRouter-specific details out
  of `workflows`.

Model metadata comes through `ModelsCache`, which is disk-backed with a TTL. If a
metadata fetch fails it reports "unknown" rather than blocking the run.

## `FakeProvider`

A scripted provider, and the reason most of the test suite needs no network. A `Script`
is a list of `ScriptedTurn`s, each carrying text, reasoning, and `ScriptedToolCall`s;
`turn()`, `tool_call()`, and `complete()` are the short builders for writing them.

By default it matches turns `by_request_hash`: a turn is keyed on a hash of
`{messages, tools, model}`. That way a resumed process that re-issues an interrupted
call still gets the same response, even though the call's *index* has shifted.
`by_index` is still available for linear tests; it indexes from the checkpointed
`model_call_seq`, which keeps the provider stateless. An unmatched hash raises
`ScriptExhausted`, so a test that has drifted off its expected path fails loudly
instead of quietly taking a different one.

`chunk_rate_hz` paces a scripted stream against an absolute deadline rather than a fixed
sleep, because the Windows loop clock only resolves to about 15.6 ms.

## Adding a provider

Implement the three protocol members, then construct your provider inside your
workflow's `build(config)`. Nothing else has to change, because `workflows`, `control`,
and `tui` only ever name `Provider`. The one requirement the protocol can't express in
its types: cancellation must be prompt, and it must not leave a connection open.
