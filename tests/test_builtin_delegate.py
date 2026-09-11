"""`delegate` against a fake `Delegator`.

The point of this file is as much a layering assertion as a behavioural one: the
tool must work knowing nothing about `AgentSpec`, `AgentLoop` or anything else in
`workflows`. Specs resolve by *name* through `contracts.Delegator` (spec delta 5),
and the fake here implements that protocol in twenty lines -- which is the proof
that the seam is narrow enough.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from azalabscode.contracts import DelegateOutcome, Delegator
from azalabscode.toolio import ToolErrorKind
from azalabscode.tools import DelegateParams, DelegateTool, ToolContext, ToolSet
from azalabscode.tools.dispatcher import ToolCall, ToolDispatcher

DELEGATE = DelegateTool()


@dataclass
class FakeDelegator:
    """A `Delegator` that records what it was asked to do."""

    specs: list[str] = field(default_factory=lambda: ["explorer", "reviewer"])
    outcome: DelegateOutcome | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def available_specs(self) -> list[str]:
        return list(self.specs)

    async def delegate(
        self,
        spec_name: str,
        task: str,
        *,
        tools: Sequence[str] | None = None,
        model: str | None = None,
        max_turns: int | None = None,
    ) -> DelegateOutcome:
        self.calls.append(
            {
                "spec": spec_name,
                "task": task,
                "tools": list(tools) if tools else None,
                "model": model,
                "max_turns": max_turns,
            }
        )
        return self.outcome or DelegateOutcome(
            agent_id="main/0", final_text="the child's answer", turns=3
        )


async def run(ctx: ToolContext, **kwargs: object):
    params = DelegateParams(**kwargs)  # type: ignore[arg-type]
    error = await DELEGATE.validate_params(params, ctx)
    if error is not None:
        from azalabscode.toolio import ToolResult

        return ToolResult.failure(error.kind, error.message)
    return await DELEGATE.run(params, ctx)


# ---------------------------------------------------------------------------


def test_the_fake_satisfies_the_protocol() -> None:
    """If a twenty-line dataclass cannot satisfy it, the seam is too wide."""

    assert isinstance(FakeDelegator(), Delegator)


async def test_a_delegated_task_returns_the_child_final_text(tool_ctx: ToolContext) -> None:
    delegator = FakeDelegator()
    tool_ctx.delegator = delegator

    result = await run(tool_ctx, task="Survey the auth module", spec="explorer")

    assert result.ok is True
    assert result.text == "the child's answer"
    assert delegator.calls[0]["spec"] == "explorer"
    assert delegator.calls[0]["task"] == "Survey the auth module"
    assert result.meta["agent_id"] == "main/0"
    assert result.meta["turns"] == 3


async def test_the_spec_defaults_to_the_first_available(tool_ctx: ToolContext) -> None:
    delegator = FakeDelegator(specs=["only_one"])
    tool_ctx.delegator = delegator

    await run(tool_ctx, task="do it")

    assert delegator.calls[0]["spec"] == "only_one"


async def test_restrictions_are_passed_through(tool_ctx: ToolContext) -> None:
    delegator = FakeDelegator()
    tool_ctx.delegator = delegator

    await run(
        tool_ctx,
        task="read one file",
        spec="explorer",
        tools=["read_file", "glob"],
        model="anthropic/claude-haiku-4.5",
        max_turns=5,
    )

    call = delegator.calls[0]
    assert call["tools"] == ["read_file", "glob"]
    assert call["model"] == "anthropic/claude-haiku-4.5"
    assert call["max_turns"] == 5


async def test_delegation_without_a_delegator_is_unavailable(tool_ctx: ToolContext) -> None:
    """`default_registry` always registers `delegate`; whether this agent may use it
    is only knowable from the context, so the refusal lives here."""

    assert tool_ctx.delegator is None
    result = await run(tool_ctx, task="anything")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.UNAVAILABLE
    assert "do the work yourself" in result.error.message


async def test_an_unknown_spec_lists_the_available_ones(tool_ctx: ToolContext) -> None:
    tool_ctx.delegator = FakeDelegator()
    result = await run(tool_ctx, task="x", spec="nonexistent")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS
    assert "explorer" in result.error.message
    assert "reviewer" in result.error.message


async def test_no_configured_specs_is_unavailable(tool_ctx: ToolContext) -> None:
    tool_ctx.delegator = FakeDelegator(specs=[])
    result = await run(tool_ctx, task="x")

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.UNAVAILABLE


async def test_a_failed_child_is_reported_as_a_failure(tool_ctx: ToolContext) -> None:
    tool_ctx.delegator = FakeDelegator(
        outcome=DelegateOutcome(agent_id="main/1", ok=False, error="hit its turn limit")
    )
    result = await run(tool_ctx, task="x")

    assert result.ok is False
    assert result.error is not None
    assert "hit its turn limit" in result.error.message
    assert result.display is not None
    assert result.display.data["ok"] is False


async def test_a_silent_child_says_so_rather_than_returning_nothing(
    tool_ctx: ToolContext,
) -> None:
    """An empty result reads as "the tool did not run"."""

    tool_ctx.delegator = FakeDelegator(outcome=DelegateOutcome(agent_id="main/2", final_text=""))
    result = await run(tool_ctx, task="x")

    assert result.ok is True
    assert "no text" in result.text


async def test_an_empty_task_is_rejected_by_the_schema() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DelegateParams(task="")


async def test_delegate_reaches_the_child_through_the_dispatcher(
    tool_ctx: ToolContext,
) -> None:
    """The delegator is per-agent, so the dispatcher threads it in per call rather
    than the tool holding one."""

    delegator = FakeDelegator()
    d = ToolDispatcher(ToolSet([DelegateTool()]), context=tool_ctx)

    result = await d.call(
        ToolCall(call_id="c1", name="delegate", arguments={"task": "go"}),  # type: ignore[arg-type]
        delegator=delegator,
    )

    assert result.ok is True
    assert delegator.calls[0]["task"] == "go"


async def test_the_same_dispatcher_refuses_when_no_delegator_is_threaded(
    tool_ctx: ToolContext,
) -> None:
    d = ToolDispatcher(ToolSet([DelegateTool()]), context=tool_ctx)
    result = await d.call(
        ToolCall(call_id="c1", name="delegate", arguments={"task": "go"})  # type: ignore[arg-type]
    )

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.UNAVAILABLE


def test_delegate_never_needs_approval_and_never_retries() -> None:
    """A delegate has no external effect of its own: every effect happens inside the
    child, where it gets its own approval and its own safe point."""

    assert DELEGATE.approval == "never"
    assert DELEGATE.retry.attempts == 0
    assert DELEGATE.hard_block_retry is True


def test_delegate_is_concurrency_safe() -> None:
    """`ctx.spawn` at M5 runs several children at once; the tool must not serialise
    them."""

    assert DELEGATE.concurrency_safe is True


def test_the_description_tells_the_model_the_child_starts_blind() -> None:
    """The single most common way delegation fails is a task like "check the other
    thing", which the child cannot resolve."""

    assert "no knowledge of this conversation" in DelegateTool.description
    assert "self-contained" in DelegateTool.description
