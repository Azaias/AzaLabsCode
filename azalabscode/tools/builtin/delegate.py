"""`delegate`: hand a task to a subagent.

A thin wrapper over `contracts.Delegator`, which `AgentLoop` implements at M2. Specs
resolve **by name** (spec delta 5), so this module never imports `AgentSpec` and the
tool layer stays independent of `workflows` -- the thing import-linter contract 2
checks.

`approval="never"` and it is not retryable. A delegate has no external effect of its
own; every effect happens inside the child, where it gets its own approval and its
own safe point. That is also why an in-flight delegate is *resumed* rather than
errored after a process death (spec delta 16), unlike a leaf tool call.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from azalabscode.contracts import DelegateOutcome
from azalabscode.permissions import ApprovalPolicy, ApprovalSummary
from azalabscode.toolio import (
    NO_RETRY,
    RetryPolicy,
    ToolDisplay,
    ToolError,
    ToolErrorKind,
    ToolResult,
)
from azalabscode.tools.base import Tool
from azalabscode.tools.context import ToolContext

DESCRIPTION = """\
Hand a self-contained task to a subagent and get its answer back.

The subagent starts with no knowledge of this conversation. Everything it needs must \
be in `task`: what to look at, what to produce, and what "done" means. A task like \
"check the other thing" will fail, because it has no idea what the other thing is.

Use this when a task is large and separable -- surveying an unfamiliar part of the \
codebase, reading a long file to answer one question, running an investigation whose \
intermediate steps you do not need to see. The subagent's exploration stays out of \
your context; you get its conclusion.

Do not use it for work you could do in one or two tool calls yourself, or for \
anything requiring back-and-forth: you get one answer, with no opportunity to ask a \
follow-up question.

`spec` names which kind of subagent to run; the available names are listed in the \
parameter. `tools` narrows what it may use. In manual permission mode a subagent \
only gets tools that never need approval, so a task requiring edits will come back \
unfinished.\
"""


class DelegateParams(BaseModel):
    """Parameters for `delegate`."""

    model_config = {"extra": "forbid"}

    task: str = Field(
        min_length=1, description="The complete, self-contained task for the subagent."
    )
    spec: str | None = Field(
        default=None, description="Which agent spec to run. Defaults to the first available."
    )
    tools: list[str] | None = Field(
        default=None, description="Restrict the subagent to these tool names."
    )
    model: str | None = Field(default=None, description="Override the subagent's model.")
    max_turns: int | None = Field(
        default=None, ge=1, le=200, description="Cap the subagent's model↔tool iterations."
    )


class DelegateTool(Tool):
    """Run a subagent through the `Delegator` protocol."""

    name: ClassVar[str] = "delegate"
    description: ClassVar[str] = DESCRIPTION
    Params: ClassVar[type[BaseModel]] = DelegateParams

    approval: ApprovalPolicy = "never"
    timeout: float = 3600.0
    """Effectively unbounded: the child's own steps carry the real timeouts, and a
    dispatcher timeout here would cancel a subtree mid-effect."""
    retry: RetryPolicy = NO_RETRY
    concurrency_safe: ClassVar[bool] = True
    read_only: ClassVar[bool] = False
    hard_block_retry: ClassVar[bool] = True

    async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
        """Check there is a delegator and that the named spec exists."""

        assert isinstance(params, DelegateParams)
        if ctx.delegator is None:
            return ToolError(
                kind=ToolErrorKind.UNAVAILABLE,
                message=(
                    "delegation is not enabled for this agent; do the work yourself "
                    "with the other tools"
                ),
            )

        available = list(ctx.delegator.available_specs())
        if not available:
            return ToolError(
                kind=ToolErrorKind.UNAVAILABLE,
                message="no subagent specs are configured for this run",
            )
        if params.spec is not None and params.spec not in available:
            return ToolError(
                kind=ToolErrorKind.INVALID_PARAMS,
                message=(
                    f"no subagent spec named {params.spec!r}; "
                    f"available: {', '.join(sorted(available))}"
                ),
            )
        return None

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Run the child to completion and return its final text."""

        assert isinstance(params, DelegateParams)
        assert ctx.delegator is not None
        spec = params.spec or ctx.delegator.available_specs()[0]

        outcome: DelegateOutcome = await ctx.delegator.delegate(
            spec,
            params.task,
            tools=params.tools,
            model=params.model,
            max_turns=params.max_turns,
        )

        display = ToolDisplay(
            kind="delegate",
            data={
                "spec": spec,
                "agent_id": outcome.agent_id,
                "turns": outcome.turns,
                "ok": outcome.ok,
            },
        )
        meta = {
            "spec": spec,
            "agent_id": outcome.agent_id,
            "turns": outcome.turns,
            **outcome.meta,
        }

        if not outcome.ok:
            message = outcome.error or "the subagent did not finish successfully"
            result = ToolResult.failure(
                ToolErrorKind.INTERNAL,
                f"subagent {outcome.agent_id} failed: {message}",
                meta=meta,
            )
            result.display = display
            return result

        return ToolResult.ok_text(
            outcome.final_text or "(the subagent produced no text)",
            display=display,
            meta=meta,
        )

    def approval_summary(self, params: BaseModel, ctx: ToolContext) -> ApprovalSummary:
        """Never gated, but rendered in a transcript."""

        assert isinstance(params, DelegateParams)
        return ApprovalSummary(
            title=f"delegate to {params.spec or 'default'}",
            detail=params.task[:500],
            danger=False,
        )


__all__ = ["DESCRIPTION", "DelegateParams", "DelegateTool"]
