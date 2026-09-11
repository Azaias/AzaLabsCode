"""`ToolDispatcher`: batching, the gate, timeouts, retries, caps and the budget.

The contiguous-run batching test (spec delta 6) is the exit criterion this file
carries. It is tested two ways, because they fail differently:

- **`partition_runs` as a pure function.** Asserts the partition itself, including
  the boundary cases -- a turn that starts unsafe, one that ends unsafe, one that is
  entirely safe, and the empty turn.
- **Observed concurrency at runtime.** Asserts that calls in one safe run genuinely
  overlap, that an unsafe call overlaps with nothing, and that results come back in
  *call* order regardless of completion order. A partition that is correct on paper
  and serialised in practice passes the first test and fails the second.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, Field

from azalabscode.content import TextPart, ToolCallPart
from azalabscode.events import EventBus, EventEmitter, ToolCallFailed
from azalabscode.ids import CallId
from azalabscode.permissions import ApprovalPolicy, Decision, PermissionMode
from azalabscode.toolio import (
    DEFAULT_MAX_RESULT_CHARS,
    RetryPolicy,
    ToolError,
    ToolErrorKind,
    ToolResult,
)
from azalabscode.tools import (
    AllowAllGate,
    DenyAllGate,
    RecordingGate,
    Tool,
    ToolCall,
    ToolContext,
    ToolDispatcher,
    ToolSet,
    TurnBudget,
    partition_runs,
)
from azalabscode.tools.dispatcher import Prepared

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class EchoParams(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = "hi"
    delay: float = Field(default=0.0, ge=0)
    unsafe: bool = False


class EchoTool(Tool):
    """A concurrency-safe tool that records when it starts and stops."""

    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "Echo the text back."
    Params: ClassVar[type[BaseModel]] = EchoParams

    approval: ApprovalPolicy = "never"
    timeout: float = 5.0
    concurrency_safe: ClassVar[bool] = True
    read_only: ClassVar[bool] = True

    def __init__(self) -> None:
        self.timeline: list[tuple[str, str]] = []
        self.runs = 0
        super().__init__()

    def is_concurrency_safe(self, params: BaseModel) -> bool:
        assert isinstance(params, EchoParams)
        return not params.unsafe

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(params, EchoParams)
        self.runs += 1
        self.timeline.append(("start", params.text))
        if params.delay:
            await asyncio.sleep(params.delay)
        self.timeline.append(("stop", params.text))
        return ToolResult.ok_text(params.text)


class MutateParams(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = "w"


class MutateTool(Tool):
    """An approval-gated, non-concurrency-safe tool."""

    name: ClassVar[str] = "mutate"
    description: ClassVar[str] = "Change something."
    Params: ClassVar[type[BaseModel]] = MutateParams

    approval: ApprovalPolicy = "always"
    timeout: float = 5.0
    concurrency_safe: ClassVar[bool] = False

    def __init__(self) -> None:
        self.timeline: list[tuple[str, str]] = []
        super().__init__()

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(params, MutateParams)
        self.timeline.append(("start", params.text))
        await asyncio.sleep(0.02)
        self.timeline.append(("stop", params.text))
        return ToolResult.ok_text(params.text)


class BoomTool(Tool):
    """Raises whatever it is told to, to exercise R-T-2's safety net."""

    name: ClassVar[str] = "boom"
    description: ClassVar[str] = "Raise."
    approval: ApprovalPolicy = "never"
    timeout: float = 5.0
    concurrency_safe: ClassVar[bool] = True

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        raise ZeroDivisionError("the tool has a bug")


class SlowTool(Tool):
    """Sleeps past its timeout and records whether cleanup ran."""

    name: ClassVar[str] = "slow"
    description: ClassVar[str] = "Sleep."
    approval: ApprovalPolicy = "never"
    timeout: float = 0.05
    concurrency_safe: ClassVar[bool] = True

    def __init__(self) -> None:
        self.cleaned: list[str] = []
        self.cancelled = False
        super().__init__()

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ToolResult.ok_text("never")  # pragma: no cover

    async def on_cancel(self, params: BaseModel, ctx: ToolContext, reason: str) -> None:
        self.cleaned.append(reason)


class FlakyTool(Tool):
    """Fails with a retryable kind until the Nth attempt."""

    name: ClassVar[str] = "flaky"
    description: ClassVar[str] = "Fail then succeed."
    approval: ApprovalPolicy = "never"
    timeout: float = 5.0
    retry: RetryPolicy = RetryPolicy(attempts=2, initial_backoff_s=0.001, max_backoff_s=0.002)
    concurrency_safe: ClassVar[bool] = True

    def __init__(self, succeed_on: int = 3) -> None:
        self.calls = 0
        self.succeed_on = succeed_on
        super().__init__()

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        self.calls += 1
        if self.calls >= self.succeed_on:
            return ToolResult.ok_text("recovered")
        return ToolResult.failure(ToolErrorKind.NETWORK, "transient")


class BigParams(BaseModel):
    model_config = {"extra": "forbid"}

    size: int = 100


class BigTool(Tool):
    """Produces an arbitrarily large text result, for the cap and budget tests."""

    name: ClassVar[str] = "big"
    description: ClassVar[str] = "Produce output."
    Params: ClassVar[type[BaseModel]] = BigParams
    approval: ApprovalPolicy = "never"
    timeout: float = 5.0
    concurrency_safe: ClassVar[bool] = True

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(params, BigParams)
        return ToolResult.ok_text("x" * params.size)


def make_dispatcher(tools: list[Tool], ctx: ToolContext, **kwargs: Any) -> ToolDispatcher:
    return ToolDispatcher(ToolSet(tools), context=ctx, **kwargs)


def call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCall:
    return ToolCall(call_id=CallId(call_id), name=name, arguments=arguments)


async def drain(sub: Any) -> list[Any]:
    """Every event a subscription is currently holding.

    `Subscription.get()` blocks when the backlog is empty, so drain on the length
    rather than waiting for a sentinel the bus will not send until it closes.
    """

    return [await sub.get() for _ in range(len(sub))]


# ---------------------------------------------------------------------------
# partition_runs -- the pure function (spec delta 6)
# ---------------------------------------------------------------------------


def prep(*safe: bool) -> list[Prepared]:
    return [Prepared(call=call("echo", f"c{i}"), concurrency_safe=s) for i, s in enumerate(safe)]


def shape(items: list[Prepared]) -> list[tuple[bool, int]]:
    return [(b.concurrent, len(b)) for b in partition_runs(items)]


def test_an_all_safe_turn_is_one_concurrent_batch() -> None:
    assert shape(prep(True, True, True, True)) == [(True, 4)]


def test_an_unsafe_call_splits_the_run_around_it() -> None:
    """The case spec 4.3 gets wrong: five reads with one edit among them should be
    two concurrent runs plus the edit, not eight serial calls."""

    assert shape(prep(True, True, False, True, True)) == [(True, 2), (False, 1), (True, 2)]


def test_consecutive_unsafe_calls_each_run_alone() -> None:
    assert shape(prep(False, False)) == [(False, 1), (False, 1)]


def test_a_turn_starting_or_ending_unsafe_keeps_its_trailing_run() -> None:
    assert shape(prep(False, True, True)) == [(False, 1), (True, 2)]
    assert shape(prep(True, True, False)) == [(True, 2), (False, 1)]


def test_the_empty_turn_partitions_to_nothing() -> None:
    assert partition_runs([]) == []


def test_a_lone_safe_call_is_still_marked_concurrent() -> None:
    """Keeps the partition a pure function of the safety flags rather than of length."""

    assert shape(prep(True)) == [(True, 1)]


def test_the_partition_preserves_every_call_in_order() -> None:
    items = prep(True, False, True, True, False)
    flattened = [p for b in partition_runs(items) for p in b.items]
    assert [p.call.call_id for p in flattened] == [p.call.call_id for p in items]


# ---------------------------------------------------------------------------
# Batching at runtime
# ---------------------------------------------------------------------------


async def test_a_safe_run_actually_overlaps(tool_ctx: ToolContext) -> None:
    """A correct partition that then serialises is the failure this catches."""

    echo = EchoTool()
    d = make_dispatcher([echo], tool_ctx)
    calls = [call("echo", f"c{i}", text=str(i), delay=0.05) for i in range(4)]

    results = await d.dispatch(calls)

    assert [r.text for r in results] == ["0", "1", "2", "3"]
    # Every start precedes every stop: the four ran together.
    starts = [i for i, (kind, _) in enumerate(echo.timeline) if kind == "start"]
    stops = [i for i, (kind, _) in enumerate(echo.timeline) if kind == "stop"]
    assert max(starts) < min(stops)


async def test_an_unsafe_call_separates_the_runs_around_it(tool_ctx: ToolContext) -> None:
    """Write-after-read ordering comes free from the partition (delta 6)."""

    echo = EchoTool()
    mutate = MutateTool()
    d = make_dispatcher([echo, mutate], tool_ctx)

    results = await d.dispatch(
        [
            call("echo", "c0", text="r1", delay=0.02),
            call("echo", "c1", text="r2", delay=0.02),
            call("mutate", "c2", text="w"),
            call("echo", "c3", text="r3", delay=0.02),
        ]
    )

    assert [r.text for r in results] == ["r1", "r2", "w", "r3"]
    # The mutate ran strictly between the two read runs.
    assert echo.timeline.index(("stop", "r2")) < echo.timeline.index(("start", "r3"))
    assert mutate.timeline == [("start", "w"), ("stop", "w")]


async def test_results_come_back_in_call_order_not_completion_order(
    tool_ctx: ToolContext,
) -> None:
    """The transcript invariant at M2 matches results to calls by position; a
    completion-ordered list would silently pair the wrong result with the wrong call."""

    echo = EchoTool()
    d = make_dispatcher([echo], tool_ctx)

    results = await d.dispatch(
        [
            call("echo", "c0", text="slow", delay=0.08),
            call("echo", "c1", text="fast", delay=0.0),
        ]
    )

    assert [r.text for r in results] == ["slow", "fast"]
    # "fast" genuinely finished first, so this is not a vacuous assertion.
    assert echo.timeline.index(("stop", "fast")) < echo.timeline.index(("stop", "slow"))


async def test_concurrency_is_per_call_not_per_tool(tool_ctx: ToolContext) -> None:
    """Spec delta 7: one `echo` marked unsafe splits the run even though the tool
    class is concurrency-safe."""

    echo = EchoTool()
    d = make_dispatcher([echo], tool_ctx)
    prepared = [
        d.prepare(call("echo", "c0", text="a")),
        d.prepare(call("echo", "c1", text="b", unsafe=True)),
        d.prepare(call("echo", "c2", text="c")),
    ]
    assert [p.concurrency_safe for p in prepared] == [True, False, True]
    assert shape(prepared) == [(True, 1), (False, 1), (True, 1)]


async def test_a_predicate_that_raises_fails_closed(tool_ctx: ToolContext) -> None:
    """A tool whose own safety assessment crashed does not get to run in parallel."""

    class Cranky(EchoTool):
        def is_concurrency_safe(self, params: BaseModel) -> bool:
            raise RuntimeError("cannot decide")

    tool = Cranky()
    assert tool.concurrency_safe_for(EchoParams()) is False
    assert tool.concurrency_safe_for(None) is False


async def test_the_semaphore_bounds_concurrency_within_a_run(
    tool_ctx: ToolContext,
) -> None:
    echo = EchoTool()
    d = make_dispatcher([echo], tool_ctx, max_parallel=2)
    calls = [call("echo", f"c{i}", text=str(i), delay=0.03) for i in range(6)]

    await d.dispatch(calls)

    # With a limit of two, not all six can be running at once.
    running = 0
    peak = 0
    for kind, _ in echo.timeline:
        running += 1 if kind == "start" else -1
        peak = max(peak, running)
    assert peak <= 2


# ---------------------------------------------------------------------------
# Preparation failures -- nothing reaches the gate
# ---------------------------------------------------------------------------


async def test_an_unknown_tool_names_what_is_available(tool_ctx: ToolContext) -> None:
    d = make_dispatcher([EchoTool()], tool_ctx)
    result = await d.call(call("nope", "c1"))
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.UNAVAILABLE
    assert "echo" in result.error.message


async def test_malformed_tool_json_becomes_invalid_params(tool_ctx: ToolContext) -> None:
    """R-P-4: the provider hands the parse error through as data; the dispatcher turns
    it into something the model can fix."""

    d = make_dispatcher([EchoTool()], tool_ctx)
    part = ToolCallPart(
        call_id="c1", name="echo", raw_arguments="{bad", parse_error="invalid JSON: x"
    )
    result = await d.call(ToolCall.from_part(part))
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS
    assert "well-formed JSON" in result.error.message


async def test_an_unknown_parameter_is_reported_rather_than_dropped(
    tool_ctx: ToolContext,
) -> None:
    """`Params` forbids extras, so a model inventing an argument is told about it."""

    d = make_dispatcher([EchoTool()], tool_ctx)
    result = await d.call(call("echo", "c1", text="hi", nonsense=1))
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS
    assert "nonsense" in result.error.message


async def test_the_validation_message_omits_the_input_echo(tool_ctx: ToolContext) -> None:
    """Pydantic's default rendering echoes the input, which for a `write_file` is the
    whole file. The message must stay small."""

    d = make_dispatcher([EchoTool()], tool_ctx)
    result = await d.call(call("echo", "c1", text="x" * 5000, delay=-1))
    assert result.error is not None
    assert len(result.error.message) < 500
    assert "https://" not in result.error.message


async def test_a_preparation_failure_never_reaches_the_gate(tool_ctx: ToolContext) -> None:
    gate = RecordingGate()
    d = make_dispatcher([EchoTool()], tool_ctx, gate=gate)
    await d.call(call("nope", "c1"))
    await d.call(call("echo", "c2", bogus=1))
    assert gate.checks == []


# ---------------------------------------------------------------------------
# validate_params, the gate, and their ordering (spec delta 11)
# ---------------------------------------------------------------------------


async def test_validate_params_runs_before_the_gate(tool_ctx: ToolContext) -> None:
    """A human must not be asked to approve an edit that was always going to fail."""

    order: list[str] = []

    class Checked(MutateTool):
        async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
            order.append("validate")
            return ToolError(kind=ToolErrorKind.NOT_FOUND, message="nope")

    class Watching(RecordingGate):
        async def check(self, **kwargs: Any) -> Decision:
            order.append("gate")
            return await super().check(**kwargs)

    d = make_dispatcher([Checked()], tool_ctx, gate=Watching())
    result = await d.call(call("mutate", "c1"))

    assert order == ["validate"]
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.NOT_FOUND


async def test_the_gate_is_consulted_for_every_call_gated_or_not(
    tool_ctx: ToolContext,
) -> None:
    """One code path: `never`-policy calls go through the gate too, so a future
    policy can see them."""

    gate = RecordingGate()
    d = make_dispatcher([EchoTool(), MutateTool()], tool_ctx, gate=gate)
    await d.dispatch([call("echo", "c0", text="a"), call("mutate", "c1")])

    assert [c.tool_name for c in gate.checks] == ["echo", "mutate"]
    assert [c.needs_approval for c in gate.checks] == [False, True]


async def test_a_denial_reaches_the_model_with_its_reason(tool_ctx: ToolContext) -> None:
    mutate = MutateTool()
    gate = RecordingGate(default=Decision.deny("not on my watch"))
    d = make_dispatcher([mutate], tool_ctx, gate=gate)
    result = await d.call(call("mutate", "c1"))

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.DENIED
    assert result.error.message == "not on my watch"
    assert "not on my watch" in result.text
    assert mutate.timeline == [], "a denied call must not run"


async def test_a_tool_hidden_from_this_agent_is_unavailable(tool_ctx: ToolContext) -> None:
    """R-C-7 defence in depth: even if the model asks, a filtered tool does not run."""

    gate = RecordingGate(hidden={"mutate"}, permission_mode=PermissionMode.MANUAL)
    mutate = MutateTool()
    d = make_dispatcher([EchoTool(), mutate], tool_ctx, gate=gate)

    result = await d.call(call("mutate", "c1"))
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.UNAVAILABLE
    assert mutate.timeline == []
    assert d.schemas_for() == [EchoTool.schema()]


async def test_a_summary_that_raises_does_not_block_the_run(tool_ctx: ToolContext) -> None:
    class BadSummary(MutateTool):
        def approval_summary(self, params: BaseModel, ctx: ToolContext) -> Any:
            raise RuntimeError("summary is broken")

    gate = RecordingGate()
    d = make_dispatcher([BadSummary()], tool_ctx, gate=gate)
    result = await d.call(call("mutate", "c1"))

    assert result.ok is True
    assert gate.checks[0].summary.danger is True
    assert "could not render" in gate.checks[0].summary.detail


async def test_a_validate_params_that_raises_becomes_an_internal_error(
    tool_ctx: ToolContext,
) -> None:
    class Cranky(EchoTool):
        async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
            raise RuntimeError("pre-check is broken")

    d = make_dispatcher([Cranky()], tool_ctx)
    result = await d.call(call("echo", "c1"))
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INTERNAL


# ---------------------------------------------------------------------------
# Exceptions, timeouts, cancellation
# ---------------------------------------------------------------------------


async def test_an_unanticipated_exception_becomes_an_internal_error_with_a_traceback(
    tool_ctx: ToolContext,
) -> None:
    """R-T-2's safety net: a tool bug is a failed result, not a crashed run."""

    d = make_dispatcher([BoomTool()], tool_ctx)
    result = await d.call(call("boom", "c1"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INTERNAL
    assert "ZeroDivisionError" in result.error.message
    assert "ZeroDivisionError" in result.meta["traceback"]
    assert "the tool has a bug" in result.text


async def test_a_timeout_cancels_the_tool_and_runs_its_cleanup(
    tool_ctx: ToolContext,
) -> None:
    """R-T-5: the dispatcher enforces the timeout, and `shell` kills its tree from
    the `on_cancel` hook this exercises."""

    slow = SlowTool()
    d = make_dispatcher([slow], tool_ctx)
    result = await d.call(call("slow", "c1"))

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.TIMEOUT
    assert slow.cancelled is True
    assert slow.cleaned == ["timeout"]


async def test_cancellation_propagates_rather_than_becoming_a_result(
    tool_ctx: ToolContext,
) -> None:
    """`workflows.step` owns the absorb-or-re-raise decision at M2 and needs the
    `uncancel()` count to make it. Swallowing it here would take that away."""

    slow = SlowTool()
    slow.timeout = 30.0
    d = make_dispatcher([slow], tool_ctx)

    task = asyncio.create_task(d.call(call("slow", "c1")))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert slow.cleaned == ["cancelled"]


async def test_the_step_layer_gets_a_factory_for_the_cancelled_result() -> None:
    result = ToolDispatcher.cancelled_result(call("echo", "c9"))
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.CANCELLED

    result = ToolDispatcher.interrupted_result(call("echo", "c9"))
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INTERRUPTED
    assert "effect unknown" in result.error.message


# ---------------------------------------------------------------------------
# Retries (R-T-4)
# ---------------------------------------------------------------------------


async def test_a_retryable_failure_is_retried_up_to_the_policy(
    tool_ctx: ToolContext,
) -> None:
    flaky = FlakyTool(succeed_on=3)
    d = make_dispatcher([flaky], tool_ctx)
    result = await d.call(call("flaky", "c1"))

    assert result.ok is True
    assert flaky.calls == 3


async def test_retries_stop_at_the_configured_attempt_count(
    tool_ctx: ToolContext,
) -> None:
    flaky = FlakyTool(succeed_on=99)
    d = make_dispatcher([flaky], tool_ctx)
    result = await d.call(call("flaky", "c1"))

    assert result.ok is False
    assert flaky.calls == 3  # the first try plus two retries


async def test_a_non_retryable_kind_is_not_retried(tool_ctx: ToolContext) -> None:
    class Denied(FlakyTool):
        async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
            self.calls += 1
            return ToolResult.failure(ToolErrorKind.INVALID_PARAMS, "wrong")

    tool = Denied()
    d = make_dispatcher([tool], tool_ctx)
    await d.call(call("flaky", "c1"))
    assert tool.calls == 1


def test_an_approval_gated_tool_cannot_declare_retries() -> None:
    """R-T-4 at construction: a misconfigured non-idempotent retry is a bug to find
    before the run costs money."""

    from azalabscode.errors import ConfigurationError

    class Bad(MutateTool):
        retry: RetryPolicy = RetryPolicy(attempts=3)

    with pytest.raises(ConfigurationError, match="R-T-4"):
        Bad()


def test_the_explicit_override_makes_it_legal() -> None:
    class Deliberate(MutateTool):
        retry: RetryPolicy = RetryPolicy(attempts=1, unsafe_allow_retry=True)

    assert Deliberate().retry.attempts == 1


def test_shell_style_hard_block_refuses_retries_even_with_the_override() -> None:
    from azalabscode.errors import ConfigurationError

    class Hard(EchoTool):
        hard_block_retry: ClassVar[bool] = True
        retry: RetryPolicy = RetryPolicy(attempts=1, unsafe_allow_retry=True)

    with pytest.raises(ConfigurationError, match="hard-blocks retries"):
        Hard()


async def test_the_dispatcher_re_enforces_the_retry_rule_at_call_time(
    tool_ctx: ToolContext,
) -> None:
    """`retry` is an instance attribute; something could reassign it after
    construction. The dispatcher is the second line."""

    tool = FlakyTool(succeed_on=99)
    tool.approval = "always"  # now non-idempotent, with attempts still at 2
    d = make_dispatcher([tool], tool_ctx)
    await d.call(call("flaky", "c1"))
    assert tool.calls == 1


# ---------------------------------------------------------------------------
# Result caps and the turn budget (R-T-6, spec delta 8)
# ---------------------------------------------------------------------------


async def test_an_over_cap_result_is_spilled_and_summarised(
    tool_ctx: ToolContext, tmp_path: Path
) -> None:
    tool_ctx.session_dir = tmp_path / "session"
    tool = BigTool()
    tool.max_result_size_chars = 100
    d = make_dispatcher([tool], tool_ctx)

    result = await d.call(call("big", "c1", size=5000))

    assert "<persisted-output>" in result.text
    assert "5000 characters" in result.text
    assert result.meta["output_truncated"] is True
    spilled = Path(result.meta["output_path"])
    assert spilled.exists()
    assert spilled.read_text(encoding="utf-8") == "x" * 5000


async def test_a_result_under_the_cap_is_untouched(tool_ctx: ToolContext) -> None:
    tool = BigTool()
    tool.max_result_size_chars = 100
    d = make_dispatcher([tool], tool_ctx)
    result = await d.call(call("big", "c1", size=50))
    assert result.text == "x" * 50
    assert "output_truncated" not in result.meta


async def test_read_file_style_unbounded_caps_opt_out_entirely(
    tool_ctx: ToolContext,
) -> None:
    """Spilling a file read to disk that the model then re-reads is circular."""

    tool = BigTool()
    tool.max_result_size_chars = math.inf
    d = make_dispatcher([tool], tool_ctx)
    result = await d.call(call("big", "c1", size=200_000))
    assert len(result.text) == 200_000


async def test_the_default_cap_is_the_system_ceiling(tool_ctx: ToolContext) -> None:
    tool = BigTool()
    assert tool.max_result_size_chars == DEFAULT_MAX_RESULT_CHARS


async def test_the_turn_budget_elides_a_result_that_does_not_fit(
    tool_ctx: ToolContext,
) -> None:
    """Eight parallel results just under the per-tool cap is what this bounds."""

    d = make_dispatcher([BigTool()], tool_ctx)
    budget = TurnBudget(limit=1000)

    results = await d.dispatch(
        [call("big", "c0", size=800), call("big", "c1", size=800)], budget=budget
    )

    assert results[0].ok is True
    assert results[1].ok is False
    assert results[1].error is not None
    assert results[1].error.kind is ToolErrorKind.BUDGET
    assert "Re-run this call on its own" in results[1].text


def test_the_budget_decision_is_memoized_by_call_id() -> None:
    """Byte-stability across a resume: the same call id gets the same verdict even if
    completion order changed."""

    budget = TurnBudget(limit=100)
    big = ToolResult.ok_text("y" * 200)

    first = budget.charge(big, call_id="c1")
    second = budget.charge(big, call_id="c1")
    assert first.ok is False and second.ok is False

    small = ToolResult.ok_text("y" * 10)
    a = budget.charge(small, call_id="c2")
    b = budget.charge(small, call_id="c2")
    assert a.ok is True and b.ok is True
    assert budget.spent == 10, "a repeat charge must not spend the budget twice"


def test_an_unbounded_budget_charges_nothing() -> None:
    budget = TurnBudget(limit=math.inf)
    assert budget.charge(ToolResult.ok_text("z" * 10_000), call_id="c1").ok is True
    assert budget.remaining() == math.inf


def test_resetting_the_budget_starts_a_fresh_turn() -> None:
    budget = TurnBudget(limit=100)
    budget.charge(ToolResult.ok_text("y" * 60), call_id="c1")
    budget.reset()
    assert budget.spent == 0
    assert budget.charge(ToolResult.ok_text("y" * 60), call_id="c1").ok is True


def test_the_cap_keeps_non_text_parts(tmp_path: Path) -> None:
    """An `ImagePart` is not what blew the cap and must survive the spill."""

    from azalabscode.content import ImagePart
    from azalabscode.tools.budget import apply_result_cap

    result = ToolResult(
        ok=True,
        content=[TextPart(text="x" * 500), ImagePart(media_type="image/png", data_b64="AA==")],
    )
    capped = apply_result_cap(result, cap=100, call_id="c1", spill_directory=tmp_path)

    assert any(isinstance(p, ImagePart) for p in capped.content)
    assert "<persisted-output>" in capped.content[0].text  # type: ignore[union-attr]


def test_a_spill_directory_that_cannot_be_written_still_produces_a_result() -> None:
    """Losing the overflow is bad; failing the call because the disk is full is worse."""

    from azalabscode.tools.budget import apply_result_cap

    result = ToolResult.ok_text("x" * 500)
    capped = apply_result_cap(result, cap=100, call_id="c1", spill_directory=None)

    assert "<persisted-output>" in capped.text
    assert "discarded" in capped.text
    assert "output_path" not in capped.meta


# ---------------------------------------------------------------------------
# Events (R-X-3)
# ---------------------------------------------------------------------------


async def test_a_call_emits_requested_started_and_completed(
    tool_ctx: ToolContext,
) -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe(name="t")
    d = make_dispatcher([EchoTool()], tool_ctx, emitter=EventEmitter(bus, agent_id="main"))

    await d.call(call("echo", "c1", text="hi"))
    await asyncio.sleep(0)

    seen = [e.type for e in await drain(sub)]
    assert seen == ["tool_call_requested", "tool_call_started", "tool_call_completed"]


async def test_a_failure_emits_tool_call_failed_with_the_error(
    tool_ctx: ToolContext,
) -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe(name="t")
    d = make_dispatcher([BoomTool()], tool_ctx, emitter=EventEmitter(bus, agent_id="main"))

    await d.call(call("boom", "c1"))
    await asyncio.sleep(0)

    failed = [e for e in await drain(sub) if isinstance(e, ToolCallFailed)]
    assert len(failed) == 1
    assert failed[0].error.kind is ToolErrorKind.INTERNAL
    assert failed[0].agent_id == "main"


async def test_a_retry_emits_one_started_event_per_attempt(
    tool_ctx: ToolContext,
) -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe(name="t")
    d = make_dispatcher(
        [FlakyTool(succeed_on=3)], tool_ctx, emitter=EventEmitter(bus, agent_id="main")
    )

    await d.call(call("flaky", "c1"))
    await asyncio.sleep(0)

    attempts = [e.attempt for e in await drain(sub) if e.type == "tool_call_started"]
    assert attempts == [0, 1, 2]


# ---------------------------------------------------------------------------
# Gates shipped with the tool layer
# ---------------------------------------------------------------------------


async def test_allow_all_and_deny_all_satisfy_the_protocol() -> None:
    from azalabscode.contracts import PermissionGate

    assert isinstance(AllowAllGate(), PermissionGate)
    assert isinstance(DenyAllGate(), PermissionGate)
    assert isinstance(RecordingGate(), PermissionGate)

    assert AllowAllGate().visible_tool_names("main", ["a", "b"]) == ["a", "b"]  # type: ignore[arg-type]
    assert DenyAllGate().visible_tool_names("main", ["a", "b"]) == []  # type: ignore[arg-type]
    assert DenyAllGate().mode is PermissionMode.MANUAL


async def test_the_default_gate_is_allow_all(tool_ctx: ToolContext) -> None:
    """So the tool layer is usable standalone at M1, before `control` exists."""

    d = ToolDispatcher(ToolSet([EchoTool()]), context=tool_ctx)
    assert isinstance(d.gate, AllowAllGate)
    assert (await d.call(call("echo", "c1", text="ok"))).ok is True
