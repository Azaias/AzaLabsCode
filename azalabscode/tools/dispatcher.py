"""`ToolDispatcher`: everything a tool is not trusted to do for itself.

Schema validation, the pre-check, the permission call, the timeout, retries, output
caps, the per-turn budget, contiguous-run batching, and conversion of any escaping
exception into `ToolError(kind="internal")` with the traceback in `meta` (R-T-2).
A tool that enforced its own timeout would be a tool that could forget to.

**Batching (spec delta 6).** Spec 4.3 says run a turn's calls concurrently only if
*every* tool is parallel-safe, otherwise run them all in order. That serialises five
`read_file`s because one `edit_file` sits among them. Instead the turn is
partitioned into maximal contiguous runs of concurrency-safe calls; each safe run
executes concurrently under a semaphore, each unsafe call runs alone, and run order
is preserved. Write-after-read ordering comes free: an `edit_file` between two reads
still separates them.

**Cancellation.** `CancelledError` is re-raised, never converted. The dispatcher
runs the tool's `on_cancel` cleanup first -- which is what kills a `shell` process
tree -- and then lets the cancellation propagate. Absorbing it here would break the
`uncancel()`-count discipline that `workflows.step` owns at M2. The step layer calls
`cancelled_result()` to build the `ToolError(kind="cancelled")` the model sees.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, final

from pydantic import BaseModel, ValidationError

from azalabscode.content import ToolCallPart
from azalabscode.contracts import Delegator, PermissionGate
from azalabscode.events import (
    EventEmitter,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallRequested,
    ToolCallStarted,
)
from azalabscode.ids import MAIN_AGENT, AgentId, CallId, NodeId
from azalabscode.permissions import ApprovalSummary
from azalabscode.toolio import (
    DEFAULT_TURN_RESULT_BUDGET,
    ToolError,
    ToolErrorKind,
    ToolResult,
)
from azalabscode.tools.base import Tool, ToolSet, describe_validation_error
from azalabscode.tools.budget import TurnBudget, apply_result_cap, spill_dir_for
from azalabscode.tools.context import ToolContext, ToolPathError
from azalabscode.tools.gates import AllowAllGate

type ResultCallback = Callable[[ToolCall, ToolResult], None]
"""Called as each call finishes, so a cancelled batch does not lose what completed."""

DEFAULT_MAX_PARALLEL = 10
"""Concurrent calls inside one safe run. Ten is enough to hide latency and few
enough that a model asking for forty file reads does not open forty handles."""


@dataclass
class ToolCall:
    """One tool call as the dispatcher receives it.

    Decoupled from `ToolCallPart` so a caller that is not an agent loop -- a test, a
    script, a workflow node -- can dispatch without building a message.
    """

    call_id: CallId
    name: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str = ""
    parse_error: str | None = None

    @classmethod
    def from_part(cls, part: ToolCallPart) -> ToolCall:
        """Build from the model's tool-call part, carrying its parse error through."""

        return cls(
            call_id=CallId(part.call_id),
            name=part.name,
            arguments=part.arguments,
            raw_arguments=part.raw_arguments,
            parse_error=part.parse_error,
        )


@dataclass
class Prepared:
    """A call resolved against the toolset, before anything has run.

    Preparing once is what lets batching ask "is this call concurrency-safe?" without
    parsing the parameters twice, and what makes a parse failure fail closed for
    batching purposes.
    """

    call: ToolCall
    tool: Tool | None = None
    params: BaseModel | None = None
    error: ToolError | None = None
    concurrency_safe: bool = False

    @property
    def ready(self) -> bool:
        """True when there is a tool and validated params to run."""

        return self.error is None and self.tool is not None and self.params is not None


@dataclass
class Batch:
    """One unit of the partitioned turn: a concurrent run, or a single lone call."""

    items: list[Prepared]
    concurrent: bool

    def __len__(self) -> int:
        return len(self.items)


def partition_runs(prepared: Sequence[Prepared]) -> list[Batch]:
    """Split a turn into maximal contiguous runs of concurrency-safe calls (delta 6).

    A safe run of one is still marked concurrent -- it behaves identically and it
    keeps the partition a pure function of the safety flags, which is what the test
    asserts against.
    """

    batches: list[Batch] = []
    run: list[Prepared] = []
    for item in prepared:
        if item.concurrency_safe:
            run.append(item)
            continue
        if run:
            batches.append(Batch(items=run, concurrent=True))
            run = []
        batches.append(Batch(items=[item], concurrent=False))
    if run:
        batches.append(Batch(items=run, concurrent=True))
    return batches


@final
class ToolDispatcher:
    """Runs tool calls on behalf of an agent turn.

    Holds the toolset, the gate, the base `ToolContext` and the concurrency limit.
    One dispatcher serves a whole run; per-call context is derived in `call`.
    """

    def __init__(
        self,
        tools: ToolSet | Sequence[Tool],
        *,
        context: ToolContext,
        gate: PermissionGate | None = None,
        emitter: EventEmitter | None = None,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
        turn_budget: int | float = DEFAULT_TURN_RESULT_BUDGET,
        spill_dir: Any = None,
    ) -> None:
        self.tools = tools if isinstance(tools, ToolSet) else ToolSet(list(tools))
        self.context = context
        self.gate: PermissionGate = gate or AllowAllGate()
        self.emitter = emitter
        self.max_parallel = max_parallel
        self.turn_budget_limit = turn_budget
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._spill_dir = spill_dir
        self._cleanups: set[asyncio.Task[None]] = set()

    # -- schemas -----------------------------------------------------------

    def schemas_for(
        self, agent_id: AgentId = MAIN_AGENT, names: list[str] | None = None
    ) -> list[Any]:
        """Tool schemas this agent may see, filtered through the gate (R-C-7).

        The filtered set is what goes into the model request, so the model does not
        spend a turn asking for a tool it cannot have.
        """

        requested = names if names is not None else self.tools.names()
        visible = self.gate.visible_tool_names(agent_id, requested)
        return self.tools.schemas(list(visible))

    # -- preparation -------------------------------------------------------

    def prepare(self, call: ToolCall, *, agent_id: AgentId = MAIN_AGENT) -> Prepared:
        """Resolve, validate and classify a call without running it.

        Every failure here is terminal for the call: an unknown tool, a malformed
        argument string, a parameter that does not fit the schema, or a tool this
        agent may not see. None of them reach the gate, because there is nothing to
        approve.
        """

        if call.parse_error is not None:
            return Prepared(
                call=call,
                error=ToolError(
                    kind=ToolErrorKind.INVALID_PARAMS,
                    message=(
                        f"the arguments for {call.name} were not valid JSON "
                        f"({call.parse_error}); re-issue the call with well-formed JSON"
                    ),
                    details={"raw_arguments": call.raw_arguments[:500]},
                ),
            )

        tool = self.tools.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self.tools.names())) or "(none)"
            return Prepared(
                call=call,
                error=ToolError(
                    kind=ToolErrorKind.UNAVAILABLE,
                    message=f"no tool named {call.name!r}; available tools: {known}",
                ),
            )

        visible = self.gate.visible_tool_names(agent_id, [call.name])
        if call.name not in visible:
            return Prepared(
                call=call,
                tool=tool,
                error=ToolError(
                    kind=ToolErrorKind.UNAVAILABLE,
                    message=(
                        f"{call.name} is not available to agent {agent_id!r} under the "
                        f"current permission mode; do not call it again in this run"
                    ),
                ),
            )

        try:
            params = tool.parse_params(call.arguments or {})
        except ValidationError as exc:
            return Prepared(
                call=call,
                tool=tool,
                error=ToolError(
                    kind=ToolErrorKind.INVALID_PARAMS,
                    message=describe_validation_error(exc, call.name),
                    details={"errors": exc.error_count()},
                ),
            )
        except Exception as exc:
            return Prepared(
                call=call,
                tool=tool,
                error=ToolError(
                    kind=ToolErrorKind.INTERNAL,
                    message=f"could not build parameters for {call.name}: {exc}",
                ),
            )

        return Prepared(
            call=call,
            tool=tool,
            params=params,
            concurrency_safe=tool.concurrency_safe_for(params),
        )

    # -- single call -------------------------------------------------------

    async def call(
        self,
        call: ToolCall,
        *,
        agent_id: AgentId = MAIN_AGENT,
        node_id: NodeId | None = None,
        delegator: Delegator | None = None,
        budget: TurnBudget | None = None,
        prepared: Prepared | None = None,
    ) -> ToolResult:
        """Run one call end to end and return its result. Never raises for a failure.

        `CancelledError` is the single exception that propagates, after the tool's
        cleanup has run.
        """

        started = time.monotonic()
        item = prepared if prepared is not None else self.prepare(call, agent_id=agent_id)

        await self._emit(
            ToolCallRequested(
                call_id=str(call.call_id),
                tool=call.name,
                params=call.arguments or {},
                parse_error=call.parse_error,
            ),
            agent_id=agent_id,
            node_id=node_id,
        )

        if item.error is not None:
            return await self._finish(_failed(item.error, started), item, agent_id, node_id, budget)

        assert item.tool is not None and item.params is not None
        tool, params = item.tool, item.params
        ctx = self.context.for_call(
            call_id=call.call_id, agent_id=agent_id, node_id=node_id, delegator=delegator
        )

        # Pre-check before the prompt (delta 11): a human should not be asked to
        # approve an edit that was always going to fail.
        try:
            pre = await tool.validate_params(params, ctx)
        except ToolPathError as exc:
            pre = exc.as_tool_error()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            pre = _internal_error(tool.name, exc, "validate_params")
        if pre is not None:
            return await self._finish(_failed(pre, started), item, agent_id, node_id, budget)

        # The gate. Called for every call, gated or not, so there is one code path.
        try:
            needs_approval = tool.needs_approval(params)
        except Exception:
            needs_approval = True
        summary = self._summary(tool, params, ctx)

        decision = await self.gate.check(
            tool_name=tool.name,
            needs_approval=needs_approval,
            summary=summary,
            params=params.model_dump(mode="json"),
            agent_id=agent_id,
            call_id=call.call_id,
            node_id=node_id,
        )
        if not decision.approved:
            reason = decision.reason or "the call was denied"
            return await self._finish(
                _failed(ToolError(kind=ToolErrorKind.DENIED, message=reason), started),
                item,
                agent_id,
                node_id,
                budget,
            )

        result = await self._run_with_retries(tool, params, ctx, item, agent_id, node_id)
        result.duration_ms = (time.monotonic() - started) * 1000.0
        return await self._finish(result, item, agent_id, node_id, budget)

    async def _run_with_retries(
        self,
        tool: Tool,
        params: BaseModel,
        ctx: ToolContext,
        item: Prepared,
        agent_id: AgentId,
        node_id: NodeId | None,
    ) -> ToolResult:
        attempts = self._allowed_attempts(tool)
        attempt = 0
        while True:
            await self._emit(
                ToolCallStarted(call_id=str(ctx.call_id), tool=tool.name, attempt=attempt),
                agent_id=agent_id,
                node_id=node_id,
            )
            result = await self._run_once(tool, params, ctx)
            if result.ok or result.error is None:
                return result
            attempt += 1
            if attempt > attempts or not tool.retry.should_retry(result.error, attempt):
                return result
            await asyncio.sleep(tool.retry.delay_for(attempt))

    def _allowed_attempts(self, tool: Tool) -> int:
        """Retries this tool actually gets, enforcing R-T-4 at call time.

        `Tool.__init__` already raises on an illegal declaration. This is the second
        line: `retry` is an instance attribute and could have been reassigned after
        construction, and a tool that changes the world must not retry because of it.
        """

        if tool.hard_block_retry:
            return 0
        if tool.retry.attempts <= 0:
            return 0
        if tool.approval != "never" and not tool.retry.unsafe_allow_retry:
            return 0
        return tool.retry.attempts

    async def _run_once(self, tool: Tool, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """One attempt, under the timeout, with every exception contained."""

        timeout = self._timeout_for(tool, params)
        task = asyncio.ensure_future(tool.run(params, ctx))
        try:
            if timeout is None or timeout <= 0:
                return await task
            return await asyncio.wait_for(task, timeout)
        except TimeoutError:
            await self._cleanup(tool, params, ctx, "timeout")
            return ToolResult.failure(
                ToolErrorKind.TIMEOUT,
                f"{tool.name} exceeded its {timeout:g}s timeout and was terminated",
                meta={"timeout_s": timeout},
            )
        except asyncio.CancelledError:
            task.cancel()
            # Detached, not awaited. Awaiting cleanup from inside a task that is
            # already being cancelled does not work: the pending cancellation is
            # delivered at the first await, so the cleanup coroutine is thrown into
            # before its body runs and a `shell` process survives its own interrupt.
            # A task of its own is the only thing the cancellation cannot reach.
            self._cleanup_detached(tool, params, ctx, "cancelled")
            raise
        except ToolPathError as exc:
            return ToolResult.failure(exc.kind, exc.message)
        except Exception as exc:
            error = _internal_error(tool.name, exc, "run")
            result = ToolResult.failure(error.kind, error.message)
            result.error = error
            result.meta = {"traceback": traceback.format_exc()}
            return result

    def _timeout_for(self, tool: Tool, params: BaseModel) -> float | None:
        try:
            return tool.timeout_for(params)
        except Exception:
            return tool.timeout

    async def _cleanup(self, tool: Tool, params: BaseModel, ctx: ToolContext, reason: str) -> None:
        """Run the tool's cancellation hook. This is what kills a `shell` tree.

        Failures are swallowed: cleanup that raises must not replace the timeout or
        the cancellation the caller is already handling.
        """

        try:
            await tool.on_cancel(params, ctx, reason)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    def _cleanup_detached(
        self, tool: Tool, params: BaseModel, ctx: ToolContext, reason: str
    ) -> None:
        """Start cleanup in a task of its own and keep a reference to it.

        The reference matters: asyncio holds only a weak reference to a running task,
        so a cleanup nobody is awaiting can be garbage-collected mid-flight, leaving
        the process it was about to kill running.

        `drain_cleanups()` waits for these. The agent loop at M2 calls it before
        declaring a step finished; a test calls it before asserting the process is
        gone.
        """

        task = asyncio.ensure_future(self._cleanup(tool, params, ctx, reason))
        self._cleanups.add(task)
        task.add_done_callback(self._cleanups.discard)

    async def drain_cleanups(self) -> None:
        """Wait for every detached cancellation cleanup to finish.

        The set is emptied here rather than left to the done-callbacks. Looping until
        the set drains would busy-spin: `gather` over already-finished tasks returns
        without yielding long enough for a `call_soon` callback to run, so the loop
        starves the very callbacks it is waiting on.
        """

        pending = list(self._cleanups)
        self._cleanups.clear()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _summary(self, tool: Tool, params: BaseModel, ctx: ToolContext) -> ApprovalSummary:
        try:
            return tool.approval_summary(params, ctx)
        except Exception as exc:
            return ApprovalSummary(
                title=tool.name,
                detail=f"(could not render a summary: {type(exc).__name__}: {exc})",
                danger=True,
            )

    async def _finish(
        self,
        result: ToolResult,
        item: Prepared,
        agent_id: AgentId,
        node_id: NodeId | None,
        budget: TurnBudget | None,
    ) -> ToolResult:
        """Cap, charge, emit. The single exit for every path through `call`."""

        cap = (
            item.tool.result_cap_for(item.params)
            if item.tool is not None and item.params is not None
            else None
        )
        if cap is not None:
            result = apply_result_cap(
                result,
                cap=cap,
                call_id=str(item.call.call_id),
                spill_directory=spill_dir_for(self.context.session_dir, self._spill_dir),
            )
        if budget is not None:
            result = budget.charge(result, call_id=str(item.call.call_id))

        if result.ok:
            await self._emit(
                ToolCallCompleted(
                    call_id=str(item.call.call_id),
                    tool=item.call.name,
                    result=result,
                    duration_ms=result.duration_ms,
                ),
                agent_id=agent_id,
                node_id=node_id,
            )
        else:
            assert result.error is not None
            await self._emit(
                ToolCallFailed(
                    call_id=str(item.call.call_id),
                    tool=item.call.name,
                    error=result.error,
                    duration_ms=result.duration_ms,
                ),
                agent_id=agent_id,
                node_id=node_id,
            )
        return result

    async def _emit(self, event: Any, *, agent_id: AgentId, node_id: NodeId | None) -> None:
        if self.emitter is None:
            return
        event.agent_id = str(agent_id)
        if node_id is not None:
            event.node_id = str(node_id)
        await self.emitter.emit(event)

    # -- a whole turn ------------------------------------------------------

    async def dispatch(
        self,
        calls: Sequence[ToolCall],
        *,
        agent_id: AgentId = MAIN_AGENT,
        node_id: NodeId | None = None,
        delegator: Delegator | None = None,
        budget: TurnBudget | None = None,
        on_result: ResultCallback | None = None,
    ) -> list[ToolResult]:
        """Run a turn's calls with contiguous-run batching, results in call order.

        The returned list is index-aligned with `calls` regardless of completion
        order. That alignment is what the transcript invariant at M2 depends on: a
        result is matched to its call by position and id, never by who finished
        first.

        `on_result` is called with each `(call, result)` the moment that call
        finishes, from inside the concurrency scope. It exists because the return
        value is unreachable when the batch is cancelled -- the `TaskGroup` unwinds
        and nobody reads the finished tasks -- and a call that *did* complete before
        an interrupt must keep its real result (R-C-4). It is synchronous so it
        cannot introduce a suspension point between a call finishing and its result
        being recorded.
        """

        prepared = [self.prepare(c, agent_id=agent_id) for c in calls]
        budget = budget if budget is not None else TurnBudget(limit=self.turn_budget_limit)
        results: dict[int, ToolResult] = {}
        index_of = {id(p): i for i, p in enumerate(prepared)}

        for batch in partition_runs(prepared):
            if batch.concurrent and len(batch) > 1:
                async with asyncio.TaskGroup() as tg:
                    tasks = {
                        index_of[id(item)]: tg.create_task(
                            self._call_limited(
                                item,
                                agent_id=agent_id,
                                node_id=node_id,
                                delegator=delegator,
                                budget=budget,
                                on_result=on_result,
                            ),
                            name=f"tool:{item.call.name}:{item.call.call_id}",
                        )
                        for item in batch.items
                    }
                for i, task in tasks.items():
                    results[i] = task.result()
            else:
                for item in batch.items:
                    result = await self.call(
                        item.call,
                        agent_id=agent_id,
                        node_id=node_id,
                        delegator=delegator,
                        budget=budget,
                        prepared=item,
                    )
                    results[index_of[id(item)]] = result
                    _notify(on_result, item.call, result)

        return [results[i] for i in range(len(prepared))]

    async def _call_limited(
        self,
        item: Prepared,
        *,
        agent_id: AgentId,
        node_id: NodeId | None,
        delegator: Delegator | None,
        budget: TurnBudget,
        on_result: ResultCallback | None = None,
    ) -> ToolResult:
        async with self._semaphore:
            result = await self.call(
                item.call,
                agent_id=agent_id,
                node_id=node_id,
                delegator=delegator,
                budget=budget,
                prepared=item,
            )
        _notify(on_result, item.call, result)
        return result

    # -- what the step layer needs -----------------------------------------

    @staticmethod
    def cancelled_result(
        call: ToolCall, reason: str = "the call was cancelled before it finished"
    ) -> ToolResult:
        """The result a cancelled call is recorded as (R-C-4).

        Built here rather than inside `call` because the dispatcher re-raises
        `CancelledError`: deciding whether a cancellation is ours to absorb is
        `workflows.step`'s job at M2, and it needs the `uncancel()` count to decide.
        """

        return ToolResult.failure(
            ToolErrorKind.CANCELLED, reason, meta={"call_id": str(call.call_id)}
        )

    @staticmethod
    def interrupted_result(call: ToolCall) -> ToolResult:
        """The result a call in flight at process death is restored as (R-C-13).

        Never re-executed: the effect is unknown, and a retry of a half-applied
        `shell` is how one bad interrupt becomes two.
        """

        return ToolResult.failure(
            ToolErrorKind.INTERRUPTED,
            "process terminated during execution; effect unknown",
            meta={"call_id": str(call.call_id)},
        )


def _notify(callback: ResultCallback | None, call: ToolCall, result: ToolResult) -> None:
    """Hand one finished result to the caller's callback, if there is one.

    A callback that raises must not take the batch down with it: the result is
    already real, and losing the rest of the turn to a reporting bug would be worse
    than losing the report.
    """

    if callback is None:
        return
    try:
        callback(call, result)
    except Exception:  # pragma: no cover - a callback that raises
        return


def _failed(error: ToolError, started: float) -> ToolResult:
    """A `ToolResult` carrying `error`, with the elapsed time filled in."""

    result = ToolResult.failure(error.kind, error.message)
    result.error = error
    result.duration_ms = (time.monotonic() - started) * 1000.0
    return result


def _internal_error(tool_name: str, exc: BaseException, where: str) -> ToolError:
    return ToolError(
        kind=ToolErrorKind.INTERNAL,
        message=f"{tool_name}.{where} raised {type(exc).__name__}: {exc}",
        details={"exception": type(exc).__name__},
    )


__all__ = [
    "DEFAULT_MAX_PARALLEL",
    "Batch",
    "Prepared",
    "ResultCallback",
    "ToolCall",
    "ToolDispatcher",
    "partition_runs",
]
