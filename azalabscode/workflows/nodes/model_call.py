"""`ModelCall`: one model call, no tools (R-W-3).

The simplest thing that costs money. It is not `AgentLoop` with `max_turns=1`: there
is no toolset, no gate, no tool batch and no transcript to keep, so the whole
machinery those need would be dead weight, and a fan-out of eight of them would pay
for it eight times.

What it does keep is the parts that are not optional:

* the call runs as a **step**, so an interrupt cancels it without cancelling the
  graph around it, and a checkpoint taken elsewhere records it as in flight;
* the node is `blocked_io` for the duration -- not quiescent, because there is an
  open effect;
* every delta is emitted under `agent_id = <node_id>`, which is what lets a
  `StreamPane` route a fan-out branch to its own pane (R-U-3) without the branch
  being an agent.

A cancelled call returns whatever text arrived. Discarding it would throw away the
only thing the run has to show for the money it spent, and spec decision 4 keeps the
partial for exactly that reason.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable
from typing import Any, ClassVar

from azalabscode.cancellation import CancelReason, StepKind
from azalabscode.errors import ProviderCallError
from azalabscode.events import (
    ModelCallCancelled,
    ModelCallCompleted,
    ModelCallFailed,
    ModelCallStarted,
    ModelDelta,
)
from azalabscode.ids import AgentId, CallId, new_call_id
from azalabscode.messages import SystemMessage, UserMessage
from azalabscode.providers.base import (
    ModelRequest,
    Provider,
    ReasoningConfig,
    StreamAccumulator,
    StreamError,
    TextDelta,
    ToolCallDelta,
)
from azalabscode.runstate import AgentPhase
from azalabscode.schema import HarnessModel
from azalabscode.workflows.node import Node
from azalabscode.workflows.step import StepHandle, run_step


class ModelCallState(HarnessModel):
    """What survives a checkpoint: how many attempts, and the text so far."""

    attempts: int = 0
    text: str = ""


def render_prompt(value: Any) -> str:
    """Turn a node input into a prompt.

    A string is itself. Anything else is JSON, indented, because a list of four
    model answers read as `['...', '...']` on one line is unreadable to a model for
    the same reason it is unreadable to a person.
    """

    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n\n".join(f"[{index}]\n{render_prompt(item)}" for index, item in enumerate(value))
    try:
        return json.dumps(value, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


class ModelCall(Node):
    """One completion, streamed, with no tools attached."""

    kind: ClassVar[str] = "ModelCall"
    State: ClassVar[type[HarnessModel]] = ModelCallState
    output_type: str = "str"

    def __init__(
        self,
        model: str,
        *,
        provider: Provider | None = None,
        system_prompt: str = "",
        render: Callable[[Any], str] = render_prompt,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: ReasoningConfig | None = None,
        provider_options: dict[str, Any] | None = None,
        retries: int = 1,
    ) -> None:
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt
        self.render = render
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.provider_options = dict(provider_options or {})
        self.retries = retries

    def describe(self) -> str:
        return f"ModelCall({self.model})"

    def build_request(self, prompt: str) -> ModelRequest:
        """The request this node would send for `prompt`."""

        messages: list[Any] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(UserMessage.of(prompt))
        return ModelRequest(
            model=self.model,
            messages=messages,
            tools=[],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            reasoning=self.reasoning,
            provider_options=dict(self.provider_options),
        )

    async def run(self, ctx: Any, input: Any) -> str:
        """Stream one completion and return its text."""

        provider = self.provider or ctx.provider
        if provider is None:
            raise RuntimeError(
                f"node {ctx.node_id} has no provider: pass one to ModelCall(...) or give "
                "the Workflow a default (a provider must be built inside build(config), "
                "spec delta 21)"
            )

        request = self.build_request(self.render(input))
        attempts = max(1, self.retries + 1)
        for attempt in range(attempts):
            ctx.state.attempts = attempt + 1
            call_id = CallId(new_call_id())
            accumulator = StreamAccumulator(model=self.model)
            handle = StepHandle(
                agent_id=AgentId(str(ctx.node_id)),
                kind=StepKind.MODEL_CALL,
                node_id=ctx.node_id,
                call_id=call_id,
                description=f"model call {self.model}",
            )

            await _emit(
                ctx,
                ModelCallStarted(
                    call_id=str(call_id),
                    model=self.model,
                    message_count=len(request.messages),
                    tool_names=[],
                    attempt=attempt,
                ),
            )
            started = time.monotonic()
            await ctx.phase(AgentPhase.BLOCKED_IO)
            try:
                step = await run_step(
                    # `partial`, not a lambda: the accumulator and the call id are
                    # loop variables and a lambda would capture the last one.
                    functools.partial(_stream, ctx, provider, request, accumulator, call_id),
                    handle=handle,
                    control=ctx.control,
                )
            finally:
                await ctx.phase(AgentPhase.RUNNING)

            ctx.state.text = accumulator.message(cancelled=True).text

            if step.cancelled:
                await _emit(
                    ctx,
                    ModelCallCancelled(
                        call_id=str(call_id),
                        reason=str(step.reason or CancelReason.USER_INTERRUPT),
                        kept_partial=bool(ctx.state.text),
                    ),
                )
                return ctx.state.text

            if accumulator.error is not None:
                will_retry = attempt + 1 < attempts
                await _emit(
                    ctx,
                    ModelCallFailed(
                        call_id=str(call_id),
                        error=accumulator.error,
                        attempt=attempt,
                        will_retry=will_retry,
                    ),
                )
                if will_retry:
                    continue
                raise ProviderCallError(accumulator.error)

            message = accumulator.message()
            await _emit(
                ctx,
                ModelCallCompleted(
                    call_id=str(call_id),
                    usage=message.usage,
                    finish_reason=message.finish_reason,
                    duration_ms=(time.monotonic() - started) * 1000.0,
                ),
            )
            ctx.state.text = message.text
            return message.text

        raise AssertionError("unreachable: the retry loop always returns or raises")


async def _stream(
    ctx: Any,
    provider: Provider,
    request: ModelRequest,
    accumulator: StreamAccumulator,
    call_id: CallId,
) -> None:
    """Consume one stream, folding and emitting as it goes.

    The accumulator belongs to the caller, so a cancellation mid-stream leaves
    everything folded so far reachable.
    """

    async for event in provider.stream(request):
        accumulator.feed(event)
        delta = _delta_for(event, call_id)
        if delta is not None:
            await _emit(ctx, delta)
        if isinstance(event, StreamError):
            return


def _delta_for(event: Any, call_id: CallId) -> ModelDelta | None:
    """The `ModelDelta` for one stream event, or `None` if it is not a fragment."""

    if isinstance(event, TextDelta):
        return ModelDelta(call_id=str(call_id), text=event.text)
    if isinstance(event, ToolCallDelta):
        return ModelDelta(
            call_id=str(call_id),
            tool_call_index=event.index,
            tool_call_delta=event.arguments_delta,
        )
    if getattr(event, "type", "") == "reasoning_delta":
        return ModelDelta(call_id=str(call_id), reasoning=event.text)
    return None


async def _emit(ctx: Any, event: Any) -> None:
    """Emit under this node's id as *both* the agent and the node.

    The agent id is what `StreamPane` routes on (R-U-3), and a fan-out branch has to
    be routable even though it is not an agent.
    """

    if ctx.emitter is None:
        return
    if event.agent_id is None:
        event.agent_id = str(ctx.node_id)
    if event.node_id is None:
        event.node_id = str(ctx.node_id)
    await ctx.emitter.emit(event)


__all__ = ["ModelCall", "ModelCallState", "render_prompt"]
