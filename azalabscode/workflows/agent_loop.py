"""`AgentLoop`: the model-to-tool cycle, and the `Delegator` a subagent hangs off.

Spec 4.3 in executable form, with the deltas the plan calls for:

1. Build a `ModelRequest` from the transcript and the toolset the gate will let this
   agent see; stream it; fold the events into an `AssistantMessage`.
2. Safe point.
3. Dispatch the turn's tool calls -- contiguous-run batching lives in the dispatcher
   (delta 6) -- recording each result the instant it lands.
4. Materialise the whole turn's results in call order (`workflows.transcript`).
5. Safe point. Repeat until the model stops asking for tools.

Four things here are not obvious and all four are load-bearing:

* **A step is a task of its own** (`workflows.step`), so an interrupt cancels the
  model call without cancelling the agent that will take the next one.
* **Results are recorded through `on_result`, not from the return value.** A
  cancelled batch never returns: the `TaskGroup` unwinds and nobody reads the
  finished tasks. Without the callback, a `read_file` that completed a millisecond
  before the interrupt would be reported to the model as cancelled.
* **`enter_agent` runs before `create_task`.** The child is registered, and the
  parent flipped to `blocked_on_child`, synchronously inside `delegate()`. Any other
  order leaves an instant in which the child does not exist and the parent is
  already quiescent, which is a pause declaring PAUSED over a subagent that is about
  to start spending money.
* **A subagent gets its own read state.** It shares the workspace, the gate and the
  toolset, but not the record of which files have been read: a file the *parent*
  read is not a file the *child* has seen, and read-before-write exists to stop a
  model writing over contents it has never looked at.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import time
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from azalabscode.cancellation import CancelReason, StepKind
from azalabscode.content import ToolCallPart
from azalabscode.contracts import (
    DelegateOutcome,
    RunControl,
    SafePoint,
    SafePointKind,
)
from azalabscode.errors import ProviderCallError
from azalabscode.events import (
    EventEmitter,
    ModelCallCancelled,
    ModelCallCompleted,
    ModelCallFailed,
    ModelCallStarted,
    ModelDelta,
    ToolCallCancelled,
)
from azalabscode.ids import MAIN_AGENT, AgentId, CallId, NodeId, new_call_id
from azalabscode.messages import (
    AssistantMessage,
    SystemMessage,
    Usage,
    UserMessage,
    usage_total,
)
from azalabscode.providers.base import (
    ModelRequest,
    Provider,
    ReasoningConfig,
    StreamAccumulator,
    StreamError,
    StreamEventUnion,
    TextDelta,
    ToolCallDelta,
)
from azalabscode.runstate import AgentPhase
from azalabscode.schema import HarnessModel, VersionedModel
from azalabscode.toolio import (
    DEFAULT_TURN_RESULT_BUDGET,
    ToolDisplay,
    ToolErrorKind,
    ToolResult,
)
from azalabscode.tools.budget import TurnBudget
from azalabscode.tools.context import ReadState
from azalabscode.tools.dispatcher import ToolCall, ToolDispatcher
from azalabscode.workflows.state import AgentState
from azalabscode.workflows.step import StepHandle, run_step
from azalabscode.workflows.transcript import (
    TurnResults,
    cancelled_fill,
    finalize_turn,
)


class AgentSpec(HarnessModel):
    """What an agent is: a prompt, a model, a toolset and a few limits (spec 6.4).

    Serializable, and part of the session: a resumed run rebuilds its agents from
    these. The subagent list is names, not specs, so the `delegate` tool can resolve
    by name without `tools` importing this class (spec delta 5).
    """

    name: str = "main"
    model: str = ""
    system_prompt: str = ""
    tools: list[str] | None = None
    """Tool names this agent may use, or `None` for everything the dispatcher has.
    The gate filters this again per agent (R-C-7)."""
    max_turns: int = 40
    keep_cancelled_output: bool = True
    """Spec decision 4. An interrupted response stays in the transcript, marked, with
    every tool call stripped (delta 15), and is sent back to the model next turn."""
    parallel_tool_calls: bool | None = None
    stream_error_retries: int = 1
    """Spec C-8: a mid-stream provider failure discards the partial and retries the
    whole call once. The provider will not do it -- retrying after the first byte is
    the caller's decision."""
    allow_delegate: bool = False
    subagents: list[str] = Field(default_factory=list)
    max_tool_results_chars: int = DEFAULT_TURN_RESULT_BUDGET
    temperature: float | None = None
    max_tokens: int | None = None
    tool_choice: str = "auto"
    reasoning: ReasoningConfig | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        """One line for an event payload."""

        return f"{self.name} ({self.model})"


class AgentResult(VersionedModel):
    """What an agent hands back when its loop ends."""

    agent_id: str
    final_text: str = ""
    turns: int = 0
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str = "end_turn"
    """`end_turn`, `max_turns`, `cancelled`."""
    ok: bool = True
    error: str | None = None

    def as_outcome(self, **meta: Any) -> DelegateOutcome:
        """The `contracts.DelegateOutcome` form the `delegate` tool returns."""

        return DelegateOutcome(
            agent_id=self.agent_id,
            final_text=self.final_text,
            ok=self.ok,
            error=self.error,
            turns=self.turns,
            meta={"stop_reason": self.stop_reason, **meta},
        )


class AgentLoop:
    """One agent: a transcript, a provider, a toolset, and the loop between them.

    Implements `contracts.Delegator`, which is how the `delegate` tool reaches back
    into this layer without `tools` importing it.
    """

    def __init__(
        self,
        spec: AgentSpec,
        *,
        provider: Provider,
        dispatcher: ToolDispatcher,
        control: RunControl | None = None,
        emitter: EventEmitter | None = None,
        agent_id: AgentId = MAIN_AGENT,
        node_id: NodeId | None = None,
        parent_id: AgentId | None = None,
        state: AgentState | None = None,
        specs: Mapping[str, AgentSpec] | None = None,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.dispatcher = dispatcher
        self.control = control
        self.emitter = emitter
        self.agent_id = agent_id
        self.node_id = node_id
        self.parent_id = parent_id
        self.specs: Mapping[str, AgentSpec] = specs or {}
        self._state_was_given = state is not None
        self.state = state or AgentState(
            agent_id=str(agent_id),
            parent_id=str(parent_id) if parent_id is not None else None,
            spec_name=spec.name,
        )
        self.children: list[AgentLoop] = []

    # -- the loop -----------------------------------------------------------

    async def run(self, task: str | None = None) -> AgentResult:
        """Run to completion: model, tools, model, until the model stops asking.

        Registers the agent on the way in and deregisters it on the way out, so the
        quiescence count is right even if the loop raises.
        """

        await self._enter(task)
        try:
            result = await self._loop()
        except BaseException as exc:
            self.state.outcome = "failed"
            self.state.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            await self._exit()
        return result

    async def _enter(self, task: str | None) -> None:
        self._adopt_restored_state()
        if not self.state.messages:
            seeded: list[Any] = []
            if self.spec.system_prompt:
                seeded.append(SystemMessage(content=self.spec.system_prompt))
            if task:
                seeded.append(UserMessage.of(task))
            self.state.messages = seeded
        if self.control is not None:
            await self.control.enter_agent(
                self.agent_id,
                self.parent_id,
                spec_summary=self.spec.summary(),  # type: ignore[call-arg]
                state=self.state,  # type: ignore[call-arg]
            )

    def _adopt_restored_state(self) -> None:
        """Take over the transcript a `load()` reconciled for this agent, if there is one.

        Adopting the *object* rather than copying it is what keeps the controller and
        the loop looking at one state (D-M2-1). A loop constructed with an explicit
        `state=` keeps it: the caller has already decided, and a restored transcript
        arriving underneath would silently replace it.

        The seeding below then does nothing, because the restored messages are not
        empty -- which is the whole reason a resumed agent is not re-told its task.
        """

        if self.control is None or self.state.messages or self._state_was_given:
            return
        restored = self.control.restored_agent(self.agent_id)
        if isinstance(restored, AgentState):
            self.state = restored

    async def _exit(self) -> None:
        # Anything injected after the loop's last decision still belongs on the
        # transcript: it is what a resumed run would answer first.
        self.state.take_injections()
        self.state.usage = usage_total(self.state.messages)
        if self.control is not None:
            await self.control.exit_agent(self.agent_id)

    async def _loop(self) -> AgentResult:
        stop_reason = "max_turns"
        final_text = ""

        await self._resume_delegates()

        for turn_index in range(self.spec.max_turns):
            self.state.turn = turn_index
            await self._safe_point(SafePointKind.TURN_START)
            self.state.take_injections()

            assistant, cancelled = await self._model_step()
            if cancelled:
                # R-C-4: the agent resumes at its next model call. The injected
                # message, if there was one, is now on the transcript behind the
                # cancelled response, which is the order spec decision 4 wants.
                self.state.take_injections()
                continue
            if assistant is None:  # pragma: no cover - only on an absorbed failure
                stop_reason = "cancelled"
                break

            self.state.messages = [*self.state.messages, assistant]
            await self._safe_point(SafePointKind.AFTER_MODEL_CALL)

            calls = assistant.tool_calls
            if not calls:
                if self.state.pending_injections:
                    # The user injected while this turn was finishing. "Resume at the
                    # agent's next model call" (R-C-4) means there has to *be* one:
                    # ending the run here would drop the instruction on the floor.
                    self.state.take_injections()
                    continue
                final_text = assistant.text
                stop_reason = "end_turn"
                break

            await self._tool_step(calls)
            await self._safe_point(SafePointKind.AFTER_TOOL_BATCH)

        self.state.final_text = final_text
        self.state.outcome = stop_reason
        self.state.usage = usage_total(self.state.messages)
        return AgentResult(
            agent_id=str(self.agent_id),
            final_text=final_text,
            turns=self.state.turn + 1,
            usage=self.state.usage,
            stop_reason=stop_reason,
            ok=True,
        )

    # -- the model call -----------------------------------------------------

    async def _model_step(self) -> tuple[AssistantMessage | None, bool]:
        """One model call, with the C-8 mid-stream retry. Returns `(message, cut)`.

        `cut` is True when an interrupt cancelled the call. The partial response is
        kept as `AssistantMessage(cancelled=True)` with **every** tool call dropped
        (delta 15): none of them were dispatched, and keeping one would need a
        fabricated result and would tell the model it ran something it did not.
        """

        attempts = max(1, self.spec.stream_error_retries + 1)
        for attempt in range(attempts):
            call_id = CallId(new_call_id())
            accumulator = StreamAccumulator(model=self.spec.model)
            request = self.build_request()
            handle = StepHandle(
                agent_id=self.agent_id,
                kind=StepKind.MODEL_CALL,
                node_id=self.node_id,
                call_id=call_id,
                description=f"model call {self.spec.model}",
            )

            await self._emit(
                ModelCallStarted(
                    call_id=str(call_id),
                    model=self.spec.model,
                    message_count=len(request.messages),
                    tool_names=[t.name for t in request.tools],
                    attempt=attempt,
                )
            )
            started = time.monotonic()
            await self._phase(AgentPhase.BLOCKED_IO)
            try:
                step = await run_step(
                    # `partial`, not a lambda: the request, the accumulator and the call
                    # id are loop variables, and a lambda would capture the last one.
                    functools.partial(self._stream, request, accumulator, call_id),
                    handle=handle,
                    control=self.control,
                )
            finally:
                await self._phase(AgentPhase.RUNNING)
            self.state.model_call_seq += 1

            if step.cancelled:
                return await self._keep_cancelled(accumulator, call_id, step.reason), True

            if accumulator.error is not None:
                will_retry = attempt + 1 < attempts
                await self._emit(
                    ModelCallFailed(
                        call_id=str(call_id),
                        error=accumulator.error,
                        attempt=attempt,
                        will_retry=will_retry,
                    )
                )
                if will_retry:
                    continue
                raise ProviderCallError(accumulator.error)

            message = accumulator.message()
            await self._emit(
                ModelCallCompleted(
                    call_id=str(call_id),
                    usage=message.usage,
                    finish_reason=message.finish_reason,
                    duration_ms=(time.monotonic() - started) * 1000.0,
                )
            )
            return message, False

        raise AssertionError("unreachable: the retry loop always returns or raises")

    async def _stream(
        self,
        request: ModelRequest,
        accumulator: StreamAccumulator,
        call_id: CallId,
    ) -> None:
        """Consume one provider stream, folding and emitting as it goes.

        The accumulator is owned by the caller so that a cancellation mid-stream
        leaves the partial response reachable: everything this coroutine has folded
        is still there after it is thrown into.
        """

        async for event in self.provider.stream(request):
            accumulator.feed(event)
            delta = _delta_for(event, call_id)
            if delta is not None:
                await self._emit(delta)
            if isinstance(event, StreamError):
                return

    async def _keep_cancelled(
        self,
        accumulator: StreamAccumulator,
        call_id: CallId,
        reason: CancelReason | None,
    ) -> AssistantMessage | None:
        """Decide what survives an interrupted model call (R-C-4, delta 15)."""

        kept: AssistantMessage | None = None
        if self.spec.keep_cancelled_output and not accumulator.has_only_tool_calls():
            candidate = accumulator.message(cancelled=True, drop_tool_calls=True)
            if candidate.content:
                self.state.messages = [*self.state.messages, candidate]
                kept = candidate
        await self._emit(
            ModelCallCancelled(
                call_id=str(call_id),
                reason=str(reason or CancelReason.USER_INTERRUPT),
                kept_partial=kept is not None,
            )
        )
        return kept

    def build_request(self) -> ModelRequest:
        """The next `ModelRequest` for this agent.

        The toolset goes through the gate (`schemas_for`), so a subagent in `manual`
        mode never sees a tool it would be refused (R-C-7).
        """

        return ModelRequest(
            model=self.spec.model,
            messages=list(self.state.messages),
            tools=self.dispatcher.schemas_for(self.agent_id, self.spec.tools),
            tool_choice=self.spec.tool_choice,
            max_tokens=self.spec.max_tokens,
            temperature=self.spec.temperature,
            reasoning=self.spec.reasoning,
            parallel_tool_calls=self.spec.parallel_tool_calls,
            provider_options=dict(self.spec.provider_options),
            metadata={
                "agent_id": str(self.agent_id),
                "node_id": str(self.node_id) if self.node_id is not None else "",
            },
        )

    # -- the tool batch -----------------------------------------------------

    async def _tool_step(self, calls: Sequence[ToolCallPart]) -> None:
        """Run one turn's tool calls and write the whole turn to the transcript.

        The budget is constructed here, once per turn, and its decisions are kept in
        the agent's state so a resumed turn elides exactly the result the interrupted
        one did (spec delta 8, trap 1).
        """

        turn = TurnResults.for_calls(calls)
        self.state.open_call_ids = turn.call_ids
        budget = TurnBudget(
            limit=self.spec.max_tool_results_chars,
            decisions=dict(self.state.result_budget),
        )

        def record(call: ToolCall, result: ToolResult) -> None:
            turn.record(str(call.call_id), result)
            self.state.pending_results[str(call.call_id)] = result

        tool_calls = [ToolCall.from_part(part) for part in calls]
        handle = StepHandle(
            agent_id=self.agent_id,
            kind=StepKind.TOOL_CALL,
            node_id=self.node_id,
            call_ids=turn.call_ids,
            description=f"{len(tool_calls)} tool call(s)",
        )

        await self._phase(AgentPhase.BLOCKED_IO)
        try:
            step = await run_step(
                lambda: self.dispatcher.dispatch(
                    tool_calls,
                    agent_id=self.agent_id,
                    node_id=self.node_id,
                    delegator=self if self.spec.allow_delegate else None,
                    budget=budget,
                    on_result=record,
                ),
                handle=handle,
                control=self.control,
            )
        finally:
            # A cancelled `shell` is killed by a detached cleanup task in the
            # dispatcher. Draining before the step is declared finished is what makes
            # "the process is gone" true by the time the next safe point is written.
            await self.dispatcher.drain_cleanups()
            await self._phase(AgentPhase.RUNNING)

        if step.cancelled:
            for missing in turn.missing():
                await self._emit(
                    ToolCallCancelled(
                        call_id=missing.call_id,
                        tool=missing.name,
                        reason=str(step.reason or CancelReason.USER_INTERRUPT),
                    )
                )

        finalize_turn(
            self.state.messages,
            turn,
            fill=cancelled_fill,
            agent_id=str(self.agent_id),
        )
        self.state.result_budget = dict(budget.decisions)
        self.state.pending_results = {}
        self.state.open_call_ids = []

    # -- delegation (contracts.Delegator) -----------------------------------

    def available_specs(self) -> list[str]:
        """Subagent spec names this agent may delegate to."""

        if not self.spec.allow_delegate:
            return []
        return [name for name in self.spec.subagents if name in self.specs]

    async def delegate(
        self,
        spec_name: str,
        task: str,
        *,
        tools: Sequence[str] | None = None,
        model: str | None = None,
        max_turns: int | None = None,
    ) -> DelegateOutcome:
        """Run a child agent to completion and hand its answer back (R-W-4).

        The parent is `blocked_on_child` throughout, which is a **quiescent** phase.
        It has to be: a subagent parked at the pause gate would otherwise deadlock a
        parent that is never going to return, and pause would hang forever (spec
        delta 14). It is safe because a delegate step has no external effect of its
        own -- every effect the child has is guarded by the child's own safe points.
        """

        child = self._build_child(spec_name, tools=tools, model=model, max_turns=max_turns)
        return await self._run_delegate(child, task, spec_name=spec_name)

    async def _run_delegate(
        self, child: AgentLoop, task: str, *, spec_name: str
    ) -> DelegateOutcome:
        """Register a child, run it as a step, and turn its result into an outcome.

        Shared by `delegate()` and the resume path, so a delegate that came back from
        a checkpoint takes exactly the same registration ordering and the same
        cancellation handling as one that never stopped.
        """

        self.children.append(child)

        # Register the child and flip the parent *before* the task exists. Any other
        # order leaves an instant in which the child is unregistered and the parent
        # is already quiescent, and a pause landing there declares PAUSED too early.
        if self.control is not None:
            await self.control.enter_agent(
                child.agent_id,
                self.agent_id,
                spec_summary=child.spec.summary(),  # type: ignore[call-arg]
                delegated=True,  # type: ignore[call-arg]
                state=child.state,  # type: ignore[call-arg]
            )
            await self.control.phase(self.agent_id, AgentPhase.BLOCKED_ON_CHILD)

        handle = StepHandle(
            agent_id=self.agent_id,
            kind=StepKind.DELEGATE,
            node_id=self.node_id,
            child_agent_id=str(child.agent_id),
            description=f"delegate to {spec_name}",
        )
        try:
            step = await run_step(
                lambda: _run_child(child, task),
                handle=handle,
                control=self.control,
            )
        finally:
            await self._phase(AgentPhase.RUNNING)

        if step.cancelled or step.value is None:
            return DelegateOutcome(
                agent_id=str(child.agent_id),
                ok=False,
                error="the subagent was cancelled before it finished",
                meta={"stop_reason": "cancelled"},
            )
        return step.value.as_outcome(spec=spec_name)

    async def _resume_delegates(self) -> None:
        """Re-run the delegate calls a `load()` left open (spec delta 16).

        A leaf tool call in flight at process death becomes an `interrupted` error and
        is never re-run, because its effect is unknown. A delegate is the exception:
        it has no effect of its own, so re-entering it costs nothing and the child
        picks up its own transcript from its own last safe point. The parent's call
        was deliberately left unanswered by `repair_transcript`; this answers it, and
        `finalize_turn` re-checks the invariant on the way out.
        """

        entries = list(self.state.resume_delegates)
        if not entries:
            return
        self.state.resume_delegates = []

        parts = {
            call.call_id: call
            for message in self.state.messages
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
        }
        turn = TurnResults.for_calls(
            [parts[entry.call_id] for entry in entries if entry.call_id in parts]
        )
        for entry in entries:
            if entry.call_id not in parts:
                continue
            child = self._build_child(
                entry.spec_name or next(iter(self.specs), ""),
                tools=entry.tools,
                model=entry.model,
                max_turns=entry.max_turns,
                agent_id=AgentId(entry.child_agent_id),
            )
            outcome = await self._run_delegate(
                child, entry.task, spec_name=entry.spec_name or child.spec.name
            )
            turn.record(entry.call_id, _delegate_result(outcome, entry.spec_name))

        finalize_turn(
            self.state.messages,
            turn,
            fill=cancelled_fill,
            agent_id=str(self.agent_id),
        )

    def _build_child(
        self,
        spec_name: str,
        *,
        tools: Sequence[str] | None,
        model: str | None,
        max_turns: int | None,
        agent_id: AgentId | None = None,
    ) -> AgentLoop:
        base = self.specs[spec_name]
        updates: dict[str, Any] = {}
        if tools is not None:
            updates["tools"] = list(tools)
        if model is not None:
            updates["model"] = model
        if max_turns is not None:
            updates["max_turns"] = max_turns
        child_spec = base.model_copy(update=updates) if updates else base

        # A resumed delegate re-enters the id the checkpoint recorded rather than
        # allocating a new one; anything else orphans the child's transcript and
        # collides with the saved sibling on the next resume (spec 6.3).
        child_id = agent_id if agent_id is not None else AgentId(self.state.next_child_id())
        node_id = (
            NodeId(f"{self.node_id}/agent/{child_id.rsplit('/', 1)[-1]}")
            if self.node_id is not None
            else None
        )
        return AgentLoop(
            child_spec,
            provider=self.provider,
            dispatcher=self._child_dispatcher(),
            control=self.control,
            emitter=self.emitter.bind(agent_id=str(child_id)) if self.emitter else None,
            agent_id=child_id,
            node_id=node_id,
            parent_id=self.agent_id,
            specs=self.specs,
        )

    def _child_dispatcher(self) -> ToolDispatcher:
        """A dispatcher sharing everything except the read-before-write record.

        Trap 2 from the M1 handoff: `ToolContext.for_call` shares `read_state` by
        reference, which is right *within* an agent -- a file read by one call must
        satisfy the write issued by the next. Across agents it is wrong: the child
        has its own context window and has not seen the file, so it must read it
        before it may write it.
        """

        context = dataclasses.replace(self.dispatcher.context, read_state=ReadState())
        return ToolDispatcher(
            self.dispatcher.tools,
            context=context,
            gate=self.dispatcher.gate,
            emitter=self.dispatcher.emitter,
            max_parallel=self.dispatcher.max_parallel,
            turn_budget=self.dispatcher.turn_budget_limit,
        )

    # -- plumbing -----------------------------------------------------------

    async def _safe_point(self, kind: SafePointKind) -> None:
        """Declare a safe point, and park there if a pause is pending.

        `park=True` on every one of the agent's own checkpoints is what makes R-C-3
        true: a pause reaches PAUSED at the next safe point of *every* active agent.
        The controller returns immediately when no pause is pending.
        """

        if self.control is None:
            return
        await self.control.safe_point(
            SafePoint(
                kind=kind,
                agent_id=str(self.agent_id),
                node_id=str(self.node_id) if self.node_id is not None else None,
                attempt=self.state.turn,
                durable=True,
                park=True,
            )
        )

    async def _phase(self, phase: AgentPhase) -> None:
        if self.control is not None:
            await self.control.phase(self.agent_id, phase)

    async def _emit(self, event: Any) -> None:
        if self.emitter is None:
            return
        if event.agent_id is None:
            event.agent_id = str(self.agent_id)
        if event.node_id is None and self.node_id is not None:
            event.node_id = str(self.node_id)
        await self.emitter.emit(event)


def _delegate_result(outcome: DelegateOutcome, spec_name: str) -> ToolResult:
    """The `ToolResult` a resumed delegate writes back onto the parent's transcript.

    Shaped like `tools.builtin.delegate`'s own result, because the model must not be
    able to tell a resumed delegate from one that never stopped. It is built here
    rather than by re-dispatching the call: re-dispatching would allocate a *new*
    child id and abandon the transcript this whole path exists to reuse.
    """

    meta = {
        "spec": spec_name,
        "agent_id": outcome.agent_id,
        "turns": outcome.turns,
        "resumed": True,
    }
    display = ToolDisplay(
        kind="delegate",
        data={
            "spec": spec_name,
            "agent_id": outcome.agent_id,
            "turns": outcome.turns,
            "ok": outcome.ok,
            "resumed": True,
        },
    )
    if not outcome.ok:
        result = ToolResult.failure(
            ToolErrorKind.INTERNAL,
            f"subagent {outcome.agent_id} failed: "
            f"{outcome.error or 'the subagent did not finish successfully'}",
            meta=meta,
        )
        result.display = display
        return result
    return ToolResult.ok_text(
        outcome.final_text or "(the subagent produced no text)", display=display, meta=meta
    )


async def _run_child(child: AgentLoop, task: str) -> AgentResult:
    """Run a subagent inside its own `TaskGroup`.

    The task group is what makes cancelling the delegate step cancel the child too
    (spec 6.4): structured concurrency does the unwinding, and `run_step` splits the
    resulting group so a real child failure outranks the cancellation.
    """

    async with asyncio.TaskGroup() as group:
        task_handle = group.create_task(child.run(task), name=f"agent:{child.agent_id}")
    return task_handle.result()


def _delta_for(event: StreamEventUnion, call_id: CallId) -> ModelDelta | None:
    """The `ModelDelta` for one stream event, or `None` if it is not a fragment."""

    if isinstance(event, TextDelta):
        return ModelDelta(call_id=str(call_id), text=event.text)
    if isinstance(event, ToolCallDelta):
        return ModelDelta(
            call_id=str(call_id),
            tool_call_index=event.index,
            tool_call_delta=event.arguments_delta,
        )
    if event.type == "reasoning_delta":
        return ModelDelta(call_id=str(call_id), reasoning=event.text)
    return None


__all__ = ["AgentLoop", "AgentResult", "AgentSpec"]
