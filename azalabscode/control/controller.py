"""`Controller`: the object that owns a run and everything a human can do to it.

It is the implementation of `contracts.RunControl`, so `workflows` drives it without
importing `control` (spec delta 4), and it owns the four things spec 4.5 puts in
this layer: the run state machine, quiescence, the permission gate, and checkpoint
*timing* (the runner declares safe points; this decides what to do at one).

The two orderings that are not negotiable:

**Write first, park second.** A safe point folds state under the checkpoint lock and
only then parks at the pause gate, *outside* the lock. An agent that parked while
holding it would deadlock every other agent's checkpoint, and PAUSED would never be
reached. This is the deadlock plan.md calls out by name.

**Register the child before creating its task.** `enter_agent` is called
synchronously by `delegate`/`spawn` before `create_task`, together with flipping the
parent to `blocked_on_child`. Otherwise there is an instant in which the child is
unregistered and the parent is already quiescent, and a pause landing in that window
declares a run paused while a subagent is about to make model calls.

M3 adds the durable half: `_fold` serializes under the same lock, the bytes go down
through a thread, and `save`/`load` bracket the run.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, final

from azalabscode.cancellation import CancelReason
from azalabscode.contracts import (
    ApprovalHandler,
    SafePoint,
    SafePointKind,
    StepHandleLike,
    require_approval_handler,
)
from azalabscode.control.checkpoint import Checkpointer
from azalabscode.control.gate import RuntimePermissionGate, ToolGatePredicate
from azalabscode.control.quiescence import QuiescenceTracker
from azalabscode.control.resume import (
    ResumeReport,
    check_graph_drift,
    reconcile,
    resume_state_for,
)
from azalabscode.control.session import (
    InflightStep,
    NodeRecord,
    Session,
    WorkflowRef,
)
from azalabscode.control.state import AgentState, RunStateMachine
from azalabscode.errors import CheckpointError, SaveTimeout, SerializationError
from azalabscode.events import (
    AgentFinished,
    AgentPhaseChanged,
    AgentSpawned,
    Checkpoint,
    Event,
    EventBus,
    EventEmitter,
    GraphDriftWarning,
    MessageInjected,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PermissionModeChanged,
    RunStateChanged,
    RunWarning,
)
from azalabscode.ids import MAIN_AGENT, AgentId, RunId, StepId, new_run_id
from azalabscode.messages import (
    Usage,
    UserMessage,
    assert_transcript_valid,
    open_call_ids,
    usage_total,
)
from azalabscode.permissions import (
    DEFAULT_MODE,
    ApprovalRequest,
    Decision,
    PermissionMode,
)
from azalabscode.runstate import AgentPhase, NodeStatus, RunState
from azalabscode.schema import json_safe
from azalabscode.workflows.builder import Workflow
from azalabscode.workflows.graph import Graph, as_graph
from azalabscode.workflows.runner import Runner

DEFAULT_SAVE_TIMEOUT = 120.0
"""Seconds `save()` waits for a safe point before raising `SaveTimeout` (delta 17).

Long enough that a normal tool call is never the reason a save fails, short enough
that a 600 s `shell` does not silently hold the UI. The exception names the blocking
step, which is what spec C-12's status bar renders."""

type RunBody = Callable[[Any], Awaitable[Any]]
"""What a run executes: a coroutine function taking the `RunControl` (this object).

M5 replaces the hand-written body with `Runner.run(workflow, ...)`. The signature is
`RunControl`-shaped rather than `Controller`-shaped so that nothing written against
it can reach for a control-layer method.
"""


@dataclass
class InterruptResult:
    """What `interrupt()` actually did, so a caller can tell "nothing to cancel" apart.

    Spec C-4: in a fan-out workflow with no `main` agent, a targetless interrupt has
    nothing to cancel. It emits a `RunWarning` and does nothing, and this is how the
    caller finds out.
    """

    target: str
    cancelled_steps: int = 0
    injected_message_id: str | None = None
    queued: bool = False
    """True when the injected message waits for the turn boundary rather than being
    appended straight away."""
    warning: str | None = None

    @property
    def hit(self) -> bool:
        """True when at least one in-flight step was cancelled."""

        return self.cancelled_steps > 0


@dataclass
class FoldRecord:
    """One safe point as the controller folded it, kept in memory for inspection.

    The disk write is separate and conditional (`SafePoint.durable`, a configured
    session path). This list is what a test asserts the *rhythm* against, and what a
    UI would draw a checkpoint timeline from, without re-reading the file.
    """

    kind: SafePointKind
    agent_id: str | None
    node_id: str | None
    inflight: list[str] = field(default_factory=list)
    run_state: RunState = RunState.RUNNING
    to_disk: bool = False
    path: str | None = None


@dataclass
class _SaveRequest:
    """A `save()` waiting for the next safe point (R-C-10).

    The future is resolved inside the checkpoint lock, immediately after the bytes
    have gone down, so an awaiting caller is never told a save completed before the
    file exists.
    """

    path: Path
    future: asyncio.Future[Path]


@final
class Controller:
    """Owns a run: its state, its agents, its permissions, its safe points.

    Every command is `async` and idempotent where that means anything (R-C-1):
    pausing a paused run, resuming a running one and interrupting twice are all
    no-ops rather than errors.
    """

    def __init__(
        self,
        body: RunBody | None = None,
        *,
        run_id: RunId | str | None = None,
        bus: EventBus | None = None,
        permission_mode: PermissionMode = DEFAULT_MODE,
        approval_handler: ApprovalHandler | None = None,
        gated_tools: ToolGatePredicate | Collection[str] = (),
        main_agent: AgentId = MAIN_AGENT,
        validate_transcripts: bool = True,
        session_dir: str | Path | None = None,
        session_path: str | Path | None = None,
        workflow: WorkflowRef | None = None,
        rng_seed: int | None = None,
        autosave: bool = True,
        created_at: datetime | None = None,
        strict_graph_hash: bool = False,
    ) -> None:
        self.run_id = RunId(str(run_id)) if run_id is not None else new_run_id()
        self.bus = bus if bus is not None else EventBus(run_id=str(self.run_id))
        if not self.bus.run_id:
            self.bus.run_id = str(self.run_id)
        self.emitter = self.bus.emitter()
        self.main_agent = main_agent
        self.validate_transcripts = validate_transcripts

        self._body = body
        self._machine = RunStateMachine(emitter=self.emitter)
        self._quiescence = QuiescenceTracker()
        self.agents: dict[str, AgentState] = {}
        self.gate = RuntimePermissionGate(
            run_id=self.run_id,
            mode=permission_mode,
            handler=approval_handler,
            gated_tools=gated_tools,
            main_agent=main_agent,
            emitter=self.emitter,
            on_request=self._on_approval_requested,
            on_resolved=self._on_approval_resolved,
        )
        self._handler = approval_handler
        self._inflight: dict[StepId, StepHandleLike] = {}
        self._cp_lock = asyncio.Lock()
        self._command_lock = asyncio.Lock()
        self._task: asyncio.Task[Any] | None = None
        self._result: Any = None
        self._error: BaseException | None = None
        self._cancelling = False
        self._agents_seen = False
        self.folds: list[FoldRecord] = []
        """Every safe point folded, in order."""

        # -- the durable half (M3) ------------------------------------------
        self.created_at = created_at or datetime.now(UTC)
        self.workflow = workflow if workflow is not None else WorkflowRef()
        self.rng_seed = rng_seed
        self.autosave = autosave
        """Whether a durable safe point writes to disk without being asked (R-C-10)."""
        self.checkpointer = Checkpointer(session_dir=session_dir, path=session_path)
        self.nodes: dict[str, NodeRecord] = {}
        """Node lifecycle and memoized outputs (R-W-6). M5's runner fills it."""
        self.custom: dict[str, Any] = {}
        self.gated_tool_names: list[str] = sorted(gated_tools) if not callable(gated_tools) else []
        """Recorded in the session so a reloaded gate keeps R-C-7 before the tool
        registry has been rebuilt. A predicate cannot be written down, so a run that
        passes one keeps its filtering only for as long as the process lives."""
        self._pending_saves: list[_SaveRequest] = []
        self._restored: dict[str, AgentState] = {}
        self.resume_report: ResumeReport | None = None
        """What `load()` reconciled, or `None` for a run that started fresh."""
        self._resume_target: RunState | None = None
        self.last_saved: Path | None = None
        self.strict_graph_hash = strict_graph_hash
        """Spec C-2: promote a `GraphDriftWarning` to a `GraphMismatchError`. Off by
        default -- a graph that *grew* can still absorb everything the session knows,
        and refusing to load one would make adding a node to a workflow a one-way
        door for every session already on disk."""
        self.graph: Graph | None = None
        """The compiled workflow, once one is bound. `None` for a hand-written body."""
        self.runner: Runner | None = None

    # -- introspection ------------------------------------------------------

    @property
    def state(self) -> RunState:
        """The coarse run state (R-C-2)."""

        return self._machine.state

    @property
    def permission_mode(self) -> PermissionMode:
        """The run's mode, as `RunControl` exposes it to the workflow layer."""

        return self.gate.mode

    @property
    def permission_gate(self) -> RuntimePermissionGate:
        """The gate, for a workflow body that has to build its own dispatcher.

        A run rebuilt by `load()` constructs its provider and dispatcher inside
        `build(config)` (spec delta 21), so there is nobody to hand it a gate. It
        reaches for this one through the `PermissionGate` protocol.
        """

        return self.gate

    @property
    def pending_approvals(self) -> list[ApprovalRequest]:
        """Requests waiting on a human. Part of the session at M3 (R-C-9)."""

        return self.gate.pending

    @property
    def nonquiescent(self) -> int:
        """Agents whose phase blocks a pause. PAUSED requires this to be zero."""

        return self._quiescence.nonquiescent

    @property
    def inflight(self) -> list[StepHandleLike]:
        """Steps running right now, whichever agent they belong to."""

        return list(self._inflight.values())

    @property
    def result(self) -> Any:
        """What the run body returned, once it has."""

        return self._result

    @property
    def error(self) -> BaseException | None:
        """What the run body raised, if it did."""

        return self._error

    def agent(self, agent_id: AgentId | str = MAIN_AGENT) -> AgentState | None:
        """One agent's state."""

        return self.agents.get(str(agent_id))

    def phase_of(self, agent_id: AgentId | str) -> AgentPhase | None:
        """One agent's quiescence phase."""

        return self._quiescence.phase_of(agent_id)

    def blocking_agents(self) -> list[str]:
        """Which agents a pending pause is waiting on (spec C-12's status bar)."""

        return self._quiescence.nonquiescent_agents()

    def blocking_description(self) -> str:
        """What is holding a save or a pause up, in one line (delta 17, spec C-12).

        Names the *step*, not the agent, because "waiting for main" tells a user
        nothing and "main: shell (312.4s)" tells them whether to wait or interrupt.
        """

        steps = [
            f"{handle.agent_id}: {handle.description or handle.kind} "
            f"({handle.duration_ms / 1000:.1f}s)"
            for handle in self._inflight.values()
        ]
        if steps:
            return "; ".join(steps)
        blocked = self.blocking_agents()
        if blocked:
            return "; ".join(f"agent {agent_id} between safe points" for agent_id in blocked)
        return "no in-flight step; the run never reached a safe point"

    def restored_agent(self, agent_id: AgentId | str) -> Any | None:
        """The `AgentState` a `load()` restored for this agent, claimed once.

        The agent loop calls this on the way in and adopts what it gets, so the
        session and the loop share one object rather than two that drift (D-M2-1).
        Claimed once on purpose: a second `AgentLoop` for the same id inside one run
        is a new agent, not the restored one, and must not inherit its transcript.
        """

        return self._restored.pop(str(agent_id), None)

    @property
    def approval_handler(self) -> ApprovalHandler | None:
        """Who answers a pending approval, or `None` for a run that cannot be asked."""

        return self._handler

    def set_approval_handler(self, handler: ApprovalHandler | None) -> None:
        """Register the handler after construction (M4).

        A TUI is built *around* a controller -- the app needs the run to attach to,
        and the handler needs the app to show a modal on -- so the handler cannot be
        passed to `__init__` without an ordering knot. This sets both the copy
        `start()` checks for R-C-8 and the one the gate actually calls; setting only
        one of the two is the bug this method exists to make impossible.

        Refused once the run is under way: swapping the handler out from under a
        pending request would leave the gate parked on a future whose only resolver
        has been discarded.
        """

        if self.gate.pending:
            raise ValueError("cannot replace the approval handler while a request is pending")
        self._handler = handler
        self.gate.handler = handler

    async def wait_for_state(self, *states: RunState, timeout: float = 10.0) -> RunState:  # noqa: ASYNC109 - the bound is the point; there is no ambient deadline
        """Block until the run reaches one of `states`. Always bounded."""

        return await self._machine.wait_for(*states, timeout=timeout)

    # -- lifecycle ----------------------------------------------------------

    def set_body(self, body: RunBody) -> None:
        """Give the run something to execute, before `start()`.

        Exists because the body usually needs the controller: an agent loop is
        constructed with the `RunControl` it will report to, so the controller has to
        exist first. Passing the body to `__init__` is the other order and works only
        when the body closes over nothing.
        """

        if self.state is not RunState.CREATED:
            raise ValueError(f"the run has already started (state={self.state})")
        self._body = body

    def bind_workflow(
        self,
        workflow: Workflow | Graph,
        *,
        import_path: str = "",
        config: dict[str, Any] | None = None,
        config_type: str | None = None,
        input: Any = None,
    ) -> Graph:
        """Use a compiled graph as the run body, and record how to rebuild it (M5).

        This is where `graph_hash` gets into the session. It is computed from the
        graph rather than passed in, so a `WorkflowRef` on disk can never claim a hash
        the graph does not have -- which is the only way spec C-2's comparison means
        anything.

        `import_path` and `config` default to whatever the controller already carries,
        so `load()` can call this without restating what it just read off the file.
        """

        graph = as_graph(workflow)
        self.graph = graph
        self.runner = Runner(graph, input=input)
        self.workflow = WorkflowRef.of(
            import_path or self.workflow.import_path,
            config if config is not None else dict(self.workflow.config),
            config_type=config_type or self.workflow.config_type,
            graph_hash=graph.graph_hash(),
        )
        self._body = self.runner.run
        return graph

    def check_graph_drift(self, session: Session) -> GraphDriftWarning | None:
        """Compare a saved session against the bound graph (spec C-2).

        Missing ids are fatal; extra ids or a changed hash are a warning unless
        `strict_graph_hash`. Dynamic children -- `map/3`, `writer/agent/0` -- are
        accounted for by the parent that declares it makes them, so a `Map` over four
        items does not read as four missing nodes.
        """

        if self.graph is None:
            return None
        covered = {node_id for node_id in session.nodes if self.graph.accounts_for(node_id)}
        return check_graph_drift(
            session,
            node_ids=set(self.graph.node_ids()) | covered,
            graph_hash=self.graph.graph_hash(),
            strict=self.strict_graph_hash,
        )

    async def start(self) -> None:
        """Begin the run. Idempotent: starting a started run does nothing.

        Enforces R-C-8 *before* anything is spent: a `manual`-mode run with no
        `ApprovalHandler` would block on its first destructive call and look like a
        hang, so it is a `ConfigurationError` here instead.

        A run that came back from `load()` is PAUSED rather than CREATED, and
        starting it is legal: the pause gate is closed, so the body runs as far as
        its first safe point and parks there. `resume()` opens the gate.
        """

        if self._task is not None or self._machine.terminal:
            return
        require_approval_handler(self.gate.mode, self._handler)
        if self._body is None:
            raise ValueError("this Controller has no run body to start")

        if self.state is RunState.CREATED:
            await self._machine.transition(RunState.RUNNING, reason="start")
            if self._quiescence.pause_requested:
                await self._machine.transition(
                    RunState.PAUSING, reason="pause requested before start"
                )
        self._task = asyncio.create_task(self._run_body(), name=f"run:{self.run_id}")

    async def wait(self, *, timeout: float | None = None) -> Any:  # noqa: ASYNC109 - the bound is the point; there is no ambient deadline
        """Await the run. Re-raises whatever the body raised."""

        if self._task is None:
            return self._result
        if timeout is None:
            await asyncio.gather(self._task, return_exceptions=True)
        else:
            async with asyncio.timeout(timeout):
                await asyncio.gather(self._task, return_exceptions=True)
        if self._error is not None:
            raise self._error
        return self._result

    async def run(self, *, timeout: float | None = None) -> Any:  # noqa: ASYNC109 - the bound is the point; there is no ambient deadline
        """`start()` then `wait()`. The convenience a script wants."""

        await self.start()
        return await self.wait(timeout=timeout)

    async def _run_body(self) -> None:
        assert self._body is not None
        try:
            result = await self._body(self)
        except asyncio.CancelledError:
            await self._finish(RunState.CANCELLED, "cancelled")
            if not self._cancelling:
                raise
        except Exception as exc:
            # Recorded rather than propagated: `wait()` re-raises it, so a caller that
            # never awaits the run does not get an "exception was never retrieved"
            # warning for something the controller already recorded as FAILED.
            self._error = exc
            await self._finish(RunState.FAILED, f"{type(exc).__name__}: {exc}")
        else:
            self._result = result
            await self._finish(RunState.COMPLETED, "the workflow finished")

    async def _finish(self, state: RunState, reason: str) -> None:
        """Move to a terminal state, going via RUNNING if an interrupt is in flight."""

        async with self._command_lock:
            if self._machine.terminal:
                return
            if not self._machine.can(state):
                await self._machine.transition(RunState.RUNNING, reason="settling")
            await self._machine.transition(state, reason=reason)
        await self.gate.cancel_all(f"the run is {state}")
        await self._checkpoint_terminal()

    async def _checkpoint_terminal(self) -> None:
        """Write the run's last checkpoint, after it has reached a terminal state.

        Without this the newest file on disk says RUNNING for a run that finished
        minutes ago, and `load()` would faithfully resume a completed workflow. There
        is no safe point after the body returns -- the agent that would have declared
        one has already exited -- so the write is direct rather than folded, and it
        deliberately does not append a `FoldRecord`: the safe-point rhythm is a
        property of the agent loop and a terminal write is not one of its beats.

        It also releases anything still waiting in `save()`. A run that ended is not
        going to reach another safe point, and a caller blocked on one would sit
        there until its timeout for no reason.
        """

        targets = self._save_targets(durable=True)
        if not targets:
            return
        written: list[Path] = []
        async with self._cp_lock:
            data = self.checkpointer.serialize(self.session())
            for target in targets:
                await self.checkpointer.write_async(data, target)
                written.append(target)
            self.last_saved = written[0]
            self._release_pending_saves(written)
        await self.emitter.emit(Checkpoint(kind="terminal", to_disk=True, path=str(written[0])))

    async def cancel(self, reason: str = "cancelled by the user") -> None:
        """Cancel the run: every in-flight step, then the body itself."""

        if self._machine.terminal:
            return
        self._cancelling = True
        for handle in list(self._inflight.values()):
            handle.request_cancel(CancelReason.RUN_CANCELLED)
        await self.gate.cancel_all(reason)
        self._quiescence.release()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        else:
            await self._finish(RunState.CANCELLED, reason)

    # -- pause and resume ---------------------------------------------------

    async def pause(self, *, hard: bool = False) -> None:
        """Request a halt at every active agent's next safe point (R-C-3).

        In-flight model and tool calls run to completion, bounded by their own
        timeouts. `hard=True` cancels them instead, which is interrupt semantics
        without the injection -- spec C-12's escape hatch for a five-minute `shell`.
        """

        async with self._command_lock:
            if self._machine.terminal:
                return
            self._quiescence.request_pause()
            if self.state in (RunState.RUNNING, RunState.WAITING_APPROVAL):
                await self._machine.transition(RunState.PAUSING, reason="pause requested")
            if hard:
                for handle in list(self._inflight.values()):
                    if self._quiescence.phase_of(handle.agent_id) is AgentPhase.BLOCKED_IO:
                        handle.request_cancel(CancelReason.PAUSE_HARD)
        await self._settle_pause()

    async def resume(self) -> None:
        """Continue a paused run (R-C-1). A run that is not paused is unaffected.

        A run with a request still pending resumes into `WAITING_APPROVAL`, not
        `RUNNING`: the thing it is waiting for did not go away while it was paused.
        A run that came back from `load()` has no body task yet, so this starts one
        (R-C-11: "`load` never auto-starts; `resume()` continues").
        """

        async with self._command_lock:
            if self._machine.terminal:
                return
            if self.state in (RunState.PAUSED, RunState.PAUSING):
                target = self._resume_target or RunState.RUNNING
                if self.gate.pending:
                    target = RunState.WAITING_APPROVAL
                elif target is RunState.WAITING_APPROVAL:
                    target = RunState.RUNNING
                await self._machine.transition(target, reason="resumed")
            self._resume_target = None
            self._quiescence.release()
        # Outside the command lock: `start()` takes no lock of its own, but creating
        # the body task is the one thing here that can run arbitrary user code.
        if self._task is None and self._body is not None:
            await self.start()

    async def _settle_pause(self) -> None:
        """Declare PAUSED once the last non-quiescent agent goes quiet (delta 14).

        Quiescence over an *empty* agent set is vacuously true, and a run whose body
        task has been created but has not yet reached its first `enter_agent` has an
        empty set. Declaring PAUSED there is wrong twice over: no agent is parked, so
        nothing is actually halted, and an `interrupt(message=...)` aimed at `main`
        lands on an agent that does not exist yet and is dropped. A run that has
        never registered an agent and still has a live body is *starting*, not
        paused.
        """

        if not self._quiescence.pause_requested or not self._quiescence.quiescent:
            return
        if not self._agents_seen and self._task is not None and not self._task.done():
            return
        if self.state in (RunState.PAUSING, RunState.WAITING_APPROVAL, RunState.INTERRUPTING):
            await self._machine.transition(RunState.PAUSED, reason="all agents quiescent")

    # -- interrupt ----------------------------------------------------------

    async def interrupt(
        self, message: str | None = None, target: AgentId | str | None = None
    ) -> InterruptResult:
        """Cancel the target agent's current step and optionally inject a message.

        The run's coarse state does not change (spec C-4): interrupting a RUNNING run
        continues it, interrupting a PAUSED one leaves it paused. `INTERRUPTING` is
        transient and the run returns to whatever it was in.

        Interrupting a child (`target="main/0"`) leaves the parent alone; the parent
        is blocked on the child and will collect whatever the child returns.
        """

        async with self._command_lock:
            target_id = AgentId(str(target)) if target is not None else self.main_agent
            outcome = InterruptResult(target=str(target_id))
            if self._machine.terminal:
                outcome.warning = f"the run is {self.state}"
                return outcome

            previous = self.state
            interrupting = self._machine.can(RunState.INTERRUPTING)
            if interrupting:
                await self._machine.transition(
                    RunState.INTERRUPTING, reason=f"interrupt {target_id}"
                )

            for handle in list(self._inflight.values()):
                if str(handle.agent_id) == str(target_id) and handle.request_cancel(
                    CancelReason.USER_INTERRUPT
                ):
                    outcome.cancelled_steps += 1

            if message is not None:
                message_id, queued = await self._inject(target_id, message)
                outcome.injected_message_id = message_id
                outcome.queued = queued

            if outcome.cancelled_steps == 0 and str(target_id) not in self.agents:
                outcome.warning = f"no agent {str(target_id)!r} is running"
                await self.emitter.emit(
                    RunWarning(
                        code="interrupt_no_target",
                        message=outcome.warning,
                        detail={"target": str(target_id)},
                        agent_id=str(target_id),
                    )
                )

            if interrupting and self.state is RunState.INTERRUPTING:
                if self._machine.can(previous):
                    await self._machine.transition(previous, reason="interrupt applied")
                else:  # pragma: no cover - previous is always reachable in practice
                    await self._machine.transition(RunState.RUNNING, reason="interrupt applied")

        await self._settle_pause()
        return outcome

    async def _inject(self, agent_id: AgentId, text: str) -> tuple[str | None, bool]:
        """Queue an injected user message for `agent_id` (R-C-4).

        Queued rather than appended, always. Two reasons, and the second is the one
        that bites: a user message between an assistant tool-call message and its
        results is rejected by every provider, and an interrupt that cut a model call
        short has not yet written the `cancelled` assistant message the injection is
        supposed to follow. The agent loop drains the queue at its next turn
        boundary, which is exactly where R-C-4 says the agent resumes.
        """

        state = self.agents.get(str(agent_id))
        if state is None:
            return None, False
        message = UserMessage.of(text, injected=True)
        state.pending_injections = [*state.pending_injections, message]
        await self.emitter.emit(
            MessageInjected(message_id=message.id, text=text, agent_id=str(agent_id))
        )
        return message.id, True

    async def send(self, text: str, target: AgentId | str | None = None) -> str | None:
        """Queue a user message for an agent **without cancelling anything**.

        Spec 8.1 gives `PromptInput` two verbs: `send(text)` in normal mode and
        `interrupt(text)` after `escape`. They differ in exactly one respect and it
        is the important one -- `interrupt` cancels the agent's current step so the
        message is read now, `send` lets the step finish and is read at the next turn
        boundary. Without this method a UI's "send" would have to be an interrupt
        with nothing to interrupt, which cancels a model call the user was happy to
        let finish.

        Returns the injected message's id, or `None` when there is no such agent --
        which also emits a `RunWarning`, because a message typed into a box and
        silently dropped is the worst of the available behaviours.
        """

        async with self._command_lock:
            target_id = AgentId(str(target)) if target is not None else self.main_agent
            if self._machine.terminal:
                await self.emitter.emit(
                    RunWarning(
                        code="send_terminal_run",
                        message=f"the run is {self.state}; the message was not delivered",
                        agent_id=str(target_id),
                    )
                )
                return None
            message_id, queued = await self._inject(target_id, text)
            if not queued:
                await self.emitter.emit(
                    RunWarning(
                        code="send_no_target",
                        message=f"no agent {str(target_id)!r} to deliver the message to",
                        detail={"target": str(target_id)},
                        agent_id=str(target_id),
                    )
                )
            return message_id

    # -- permissions --------------------------------------------------------

    async def set_permission_mode(self, mode: PermissionMode) -> int:
        """Switch mode, returning how many pending approvals it auto-resolved (R-C-5).

        Switching to `auto` approves everything pending, because the user's action
        expresses intent to stop being asked. That can release a destructive call the
        user had forgotten about; spec C-5 accepts the cost and asks the UI to warn.
        """

        async with self._command_lock:
            old = self.gate.mode
            resolved = await self.gate.set_mode(mode)
            if old is not mode:
                await self.emitter.emit(
                    PermissionModeChanged(old=old, new=mode, pending_resolved=resolved)
                )
            if (
                self.state is RunState.WAITING_APPROVAL
                and not self.gate.pending
                and not self._quiescence.pause_requested
            ):
                await self._machine.transition(RunState.RUNNING, reason="approvals resolved")
        return resolved

    async def resolve_approval(self, request_id: str, decision: Decision) -> bool:
        """Answer a pending approval (R-C-6). False if there was nothing to answer."""

        return await self.gate.resolve(request_id, decision)

    async def approve(self, request_id: str, *, by: str = "user") -> bool:
        """Approve one request."""

        return await self.resolve_approval(request_id, Decision.approve(by=by))

    async def deny(self, request_id: str, reason: str, *, by: str = "user") -> bool:
        """Deny one request. `reason` reaches the model as a denied-kind tool error."""

        return await self.resolve_approval(request_id, Decision.deny(reason, by=by))

    async def _on_approval_requested(self, request: ApprovalRequest) -> None:
        """Gate hook: the run is now waiting on a human.

        The agent goes `waiting_approval`, which is quiescent -- so a pause can still
        land while a modal is up -- and a safe point is taken so that the pending
        request is part of the checkpoint (R-C-9).
        """

        await self.phase(AgentId(request.agent_id), AgentPhase.WAITING_APPROVAL)
        # No `_command_lock` here or in `_on_approval_resolved`: `set_permission_mode`
        # holds it while it auto-resolves pending requests, and `asyncio.Lock` is not
        # reentrant, so taking it again would deadlock the mode switch.
        if self.state is RunState.RUNNING:
            await self._machine.transition(
                RunState.WAITING_APPROVAL, reason=f"{request.tool} needs approval"
            )
        await self.safe_point(
            SafePoint(
                kind=SafePointKind.APPROVAL_PARK,
                agent_id=request.agent_id,
                node_id=request.node_id,
                durable=True,
                park=False,
            )
        )

    async def _on_approval_resolved(self, request: ApprovalRequest, decision: Decision) -> None:
        """Gate hook: a request was answered, by a human or by a switch to `auto`."""

        await self.phase(AgentId(request.agent_id), AgentPhase.BLOCKED_IO)
        if (
            self.state is RunState.WAITING_APPROVAL
            and not self.gate.pending
            and not self._quiescence.pause_requested
        ):
            await self._machine.transition(RunState.RUNNING, reason="approval resolved")

    # -- RunControl ---------------------------------------------------------

    async def safe_point(self, sp: SafePoint) -> None:
        """Fold a safe point into the session, then park if asked.

        Write first, park second, and park *outside* the lock. An agent parked while
        holding `_cp_lock` deadlocks every other checkpoint and the run never reaches
        PAUSED.

        `_fold` may raise `SerializationError`, and the runner must not catch it: a
        run that cannot be written down should fail rather than keep doing work that
        can never be saved (R-W-5).
        """

        started = time.monotonic()
        written: list[Path] = []
        async with self._cp_lock:
            self._fold(sp)
            targets = self._save_targets(durable=sp.durable)
            if targets:
                # Serialized here, inside the lock and after the synchronous fold, so
                # the bytes are a snapshot of the moment the fold ended rather than of
                # whenever the writer thread happened to be scheduled.
                data = self.checkpointer.serialize(self.session())
                for target in targets:
                    await self.checkpointer.write_async(data, target)
                    written.append(target)
                self.last_saved = written[0]
            self._release_pending_saves(written)
            if written and self.folds:
                self.folds[-1].to_disk = True
                self.folds[-1].path = str(written[0])

        await self.emitter.emit(
            Checkpoint(
                kind=str(sp.kind),
                to_disk=bool(written),
                path=str(written[0]) if written else None,
                duration_ms=(time.monotonic() - started) * 1000.0,
                agent_id=sp.agent_id,
                node_id=sp.node_id,
            )
        )
        if sp.park:
            # A node's quiescence key is its node id, and a node safe point leaves
            # `agent_id` unset for exactly that reason (M5). Falling back to the main
            # agent would park a key nothing registered, on a graph that may have no
            # agents in it at all.
            await self._park(AgentId(sp.agent_id or sp.node_id or str(self.main_agent)))

    def _fold(self, sp: SafePoint) -> None:
        """Mutate the in-memory session. Synchronous, and holds the checkpoint lock.

        No awaits: a fold that suspended would let another agent interleave a fold of
        its own, and the two would disagree about what was in flight. `inflight` is
        snapshotted here rather than maintained lazily, which is what makes a safe
        point taken by agent A correctly record agent B mid-`shell`.
        """

        if sp.snapshot is not None and not json_safe(sp.snapshot):
            raise SerializationError(
                sp.node_id or "?", "snapshot", "the node's state is not JSON-serializable"
            )

        # The snapshot is stored on the node's record so a resumed run can restore an
        # *incomplete* node from its last safe point (R-W-6's second half). Only onto
        # a record that already exists: the runner's own between-nodes safe points
        # carry a key that is not a node and must not mint one.
        if sp.snapshot is not None and sp.node_id is not None:
            record = self.nodes.get(sp.node_id)
            if record is not None and record.attempt == sp.attempt:
                record.state = dict(sp.snapshot)

        for agent_id, state in self.agents.items():
            state.open_call_ids = open_call_ids(state.messages)
            state.usage = usage_total(state.messages)
            if self.validate_transcripts:
                assert_transcript_valid(state.messages, agent_id=agent_id)

        self.folds.append(
            FoldRecord(
                kind=sp.kind,
                agent_id=sp.agent_id,
                node_id=sp.node_id,
                inflight=[str(h.step_id) for h in self._inflight.values()],
                run_state=self.state,
            )
        )

    async def _park(self, agent_id: AgentId) -> None:
        """Wait at the pause gate. A no-op when no pause is pending."""

        if not self._quiescence.pause_requested:
            return
        await self.phase(agent_id, AgentPhase.PARKED)
        try:
            await self._quiescence.wait_open()
        finally:
            await self.phase(agent_id, AgentPhase.RUNNING)

    async def enter_agent(
        self,
        agent_id: AgentId,
        parent_id: AgentId | None = None,
        *,
        spec_summary: str = "",
        delegated: bool = False,
        state: AgentState | None = None,
    ) -> None:
        """Register an agent as active (R-W-4).

        Called synchronously inside `delegate()`/`spawn()` **before** the child's task
        is created, so there is no instant in which the child is unregistered and the
        parent is already quiescent.
        """

        key = str(agent_id)
        if state is not None:
            # The agent loop owns its `AgentState` and hands it over here, so the
            # session and the loop mutate one object rather than two that drift.
            # The parent is filled in when the state does not already name one: a
            # tree whose edges are only in the caller's head does not serialize.
            if state.parent_id is None and parent_id is not None:
                state.parent_id = str(parent_id)
            self.agents[key] = state
        elif key not in self.agents:
            self.agents[key] = AgentState(
                agent_id=key,
                parent_id=str(parent_id) if parent_id is not None else None,
                spec_name=spec_summary,
            )
        self._agents_seen = True
        self._quiescence.enter(key, AgentPhase.RUNNING)
        self.agents[key].phase = AgentPhase.RUNNING
        await self.emitter.emit(
            AgentSpawned(
                agent_id=key,
                parent_id=str(parent_id) if parent_id is not None else None,
                spec_summary=spec_summary,
                delegated=delegated,
            )
        )

    async def exit_agent(self, agent_id: AgentId) -> None:
        """Mark an agent finished and drop it from the quiescence count."""

        key = str(agent_id)
        state = self.agents.get(key)
        if state is not None:
            state.phase = AgentPhase.FINISHED
        self._quiescence.exit(key)
        await self.emitter.emit(
            AgentFinished(
                agent_id=key,
                result_summary=(state.final_text[:200] if state is not None else ""),
                usage=state.usage if state is not None else None,
                outcome=(state.outcome or "completed") if state is not None else "completed",
            )
        )
        await self._settle_pause()

    async def phase(self, agent_id: AgentId, phase: AgentPhase) -> None:
        """Record an agent's quiescence phase, and settle a pending pause if it lands.

        The state is mutated before the event is emitted, so this is safe to call
        from a `finally` while a cancellation is pending: the bookkeeping is already
        done by the time anything can suspend.
        """

        key = str(agent_id)
        old = self._quiescence.set_phase(key, phase)
        state = self.agents.get(key)
        if state is not None:
            state.phase = phase
        if old is not phase:
            await self.emitter.emit(
                AgentPhaseChanged(
                    agent_id=key,
                    old=old if old is not None else phase,
                    new=phase,
                    nonquiescent=self._quiescence.nonquiescent,
                )
            )
        await self._settle_pause()

    async def enter_node(self, node_id: str) -> None:
        """Register a node as an execution context for quiescence (M5).

        Deliberately not `enter_agent`: a node has no transcript, does not belong in
        `Session.agents` and must not emit `AgentSpawned`. It still has to be counted
        and it still has to be parkable, because a graph of `Func` nodes has no agents
        at all and quiescence over an empty set is vacuously true -- a pause landing
        between two such nodes would declare PAUSED over a run that is still walking.
        """

        self._quiescence.enter(node_id, AgentPhase.RUNNING)
        self._agents_seen = True

    async def exit_node(self, node_id: str) -> None:
        """Drop a node from the quiescence count and settle any pending pause."""

        self._quiescence.exit(node_id)
        await self._settle_pause()

    def register_step(self, handle: StepHandleLike) -> None:
        """Track a step as in-flight so a checkpoint by another agent records it."""

        self._inflight[handle.step_id] = handle

    def unregister_step(self, handle: StepHandleLike) -> None:
        """Stop tracking a finished step."""

        self._inflight.pop(handle.step_id, None)

    # -- nodes (R-W-6) ------------------------------------------------------

    def node_record(self, node_id: str) -> NodeRecord | None:
        """One node's record, or `None` if it has never run."""

        return self.nodes.get(node_id)

    def node_completed(self, node_id: str, *, attempt: int = 0) -> bool:
        """Whether this attempt of this node already produced an output (R-W-6).

        The attempt is part of the question, not decoration: a node that completed on
        attempt 0 and is being retried as attempt 1 has to run again, and a resumed
        run must be able to tell those apart.
        """

        record = self.nodes.get(node_id)
        return record is not None and record.completed and record.attempt == attempt

    def node_output(self, node_id: str, *, attempt: int = 0) -> Any:
        """The memoized output of a completed node. Raises `KeyError` if there is none.

        This is what makes "completed nodes are not re-executed" mean something: the
        runner asks for the output instead of running the node, and gets the same
        value the first process produced -- read back from `values/` if it was
        spilled.
        """

        if not self.node_completed(node_id, attempt=attempt):
            raise KeyError(f"node {node_id!r} (attempt {attempt}) has no memoized output")
        record = self.nodes[node_id]
        if record.output is None:
            return None
        return record.output.resolve(self.checkpointer.session_dir)

    async def node_started(
        self,
        node_id: str,
        *,
        attempt: int = 0,
        input: Any = None,
        node_class: str = "",
    ) -> NodeRecord:
        """Record a node as running.

        A *new* attempt overwrites the record; re-entering the same attempt after a
        resume keeps the state the previous process checkpointed, which is the only
        thing that lets an incomplete node restart from its last safe point instead of
        from scratch (R-W-6).
        """

        previous = self.nodes.get(node_id)
        record = NodeRecord(
            node_id=node_id,
            status=NodeStatus.RUNNING,
            attempt=attempt,
            input=self.checkpointer.value_ref(input) if input is not None else None,
            started_at=datetime.now(UTC),
            state=dict(previous.state)
            if previous is not None and previous.attempt == attempt
            else {},
        )
        self.nodes[node_id] = record
        await self.emitter.emit(
            NodeStarted(node_id=node_id, attempt=attempt, node_class=node_class)
        )
        return record

    def node_state(self, node_id: str, *, attempt: int = 0) -> dict[str, Any] | None:
        """The state envelope checkpointed for this node, or `None` (R-W-5).

        Read by the runner before a node body starts. A record from a different
        attempt is not this node's state, so it is not offered.
        """

        record = self.nodes.get(node_id)
        if record is None or record.attempt != attempt or not record.state:
            return None
        return dict(record.state)

    async def node_finished(
        self, node_id: str, output: Any = None, *, attempt: int = 0
    ) -> NodeRecord:
        """Memoize a node's output so a resume never runs it again (R-W-6)."""

        record = self.nodes.get(node_id)
        if record is None or record.attempt != attempt:
            record = NodeRecord(node_id=node_id, attempt=attempt, started_at=datetime.now(UTC))
        record.status = NodeStatus.COMPLETED
        record.output = self.checkpointer.value_ref(output)
        record.finished_at = datetime.now(UTC)
        self.nodes[node_id] = record
        await self.emitter.emit(
            NodeCompleted(node_id=node_id, attempt=attempt, output_summary=_summarize(output))
        )
        return record

    async def node_failed(self, node_id: str, error: str, *, attempt: int = 0) -> NodeRecord:
        """Record a node as failed. Not memoized: a failure is retried, not reused."""

        record = self.nodes.get(node_id)
        if record is None or record.attempt != attempt:
            record = NodeRecord(node_id=node_id, attempt=attempt)
        record.status = NodeStatus.FAILED
        record.error = error
        record.finished_at = datetime.now(UTC)
        self.nodes[node_id] = record
        await self.emitter.emit(NodeFailed(node_id=node_id, attempt=attempt, error=error))
        return record

    # -- the session document -----------------------------------------------

    def session(self) -> Session:
        """Project the live controller into a `Session` (spec 6.2, delta 19).

        A projection, not a mirror. Nothing here is stored between calls, so there is
        no second copy of the run to drift from the first -- adding a field to the
        controller and forgetting it here produces a session that is missing it,
        which a round-trip test catches, rather than one that is quietly stale.
        """

        return Session(
            run_id=str(self.run_id),
            created_at=self.created_at,
            updated_at=datetime.now(UTC),
            workflow=self.workflow,
            run_state=self.state,
            resume_state=resume_state_for(self.state),
            permission_mode=self.gate.mode,
            main_agent=str(self.main_agent),
            gated_tools=list(self.gated_tool_names),
            agents=dict(self.agents),
            nodes=dict(self.nodes),
            pending_approvals=list(self.gate.pending),
            inflight=self._inflight_records(),
            event_seq=self.bus.seq,
            usage_total=self.usage(),
            rng_seed=self.rng_seed,
            custom=dict(self.custom),
        )

    def usage(self) -> Usage:
        """Tokens and cost across every agent in the run (delta 19)."""

        total = Usage()
        for state in self.agents.values():
            total = total + state.usage
        return total

    def _inflight_records(self) -> list[InflightStep]:
        """Snapshot the live step handles. Called inside the checkpoint lock.

        Taken here rather than maintained lazily, which is what makes a safe point
        declared by agent A correctly record agent B mid-`shell`.
        """

        return [
            InflightStep(
                step_id=str(handle.step_id),
                kind=handle.kind,
                agent_id=str(handle.agent_id),
                node_id=str(handle.node_id) if handle.node_id is not None else None,
                call_id=str(handle.call_id) if handle.call_id is not None else None,
                call_ids=list(getattr(handle, "call_ids", []) or []),
                child_agent_id=getattr(handle, "child_agent_id", None),
                description=getattr(handle, "description", ""),
                duration_ms=float(getattr(handle, "duration_ms", 0.0)),
            )
            for handle in self._inflight.values()
        ]

    # -- save and load ------------------------------------------------------

    @property
    def session_path(self) -> Path | None:
        """Where an automatic checkpoint goes, or `None` for an in-memory run."""

        return self.checkpointer.path

    async def save(
        self,
        path: str | Path | None = None,
        *,
        timeout: float = DEFAULT_SAVE_TIMEOUT,  # noqa: ASYNC109 - the bound is the deliverable; delta 17
    ) -> Path:
        """Write the session atomically, from any run state (R-C-10, delta 17).

        A run with nothing in flight is written immediately: there is no work to
        interleave with, so the current state *is* a consistent one. A run with a
        live step is checkpointed at its next safe point and this call awaits that,
        which is the only way to write a document that does not claim a tool call
        finished when it had not.

        Raises `SaveTimeout` naming the blocking step rather than hanging behind a
        600 s `shell`. That message is what spec C-12's status bar renders, and it is
        the difference between "save failed" and "save is waiting for main's `shell`,
        which has been running for five minutes -- interrupt it?".
        """

        target = Path(path) if path is not None else self.session_path
        if target is None:
            raise CheckpointError(
                "save() needs a path: pass one, or construct the Controller with a session_dir"
            )

        if not self._needs_safe_point():
            async with self._cp_lock:
                data = self.checkpointer.serialize(self.session())
                await self.checkpointer.write_async(data, target)
            self.last_saved = target
            await self.emitter.emit(
                Checkpoint(kind="save", to_disk=True, path=str(target), duration_ms=0.0)
            )
            return target

        request = _SaveRequest(path=target, future=asyncio.get_running_loop().create_future())
        self._pending_saves.append(request)
        try:
            async with asyncio.timeout(timeout):
                return await request.future
        except TimeoutError:
            if request in self._pending_saves:
                self._pending_saves.remove(request)
            raise SaveTimeout(timeout, self.blocking_description()) from None

    def _needs_safe_point(self) -> bool:
        """Whether a save has to wait rather than write straight away.

        The test is quiescence, not the in-flight set. A step registered while its
        agent is `waiting_approval` is *not* mutating anything -- the dispatcher is
        blocked at the gate -- and the next safe point will not arrive until a human
        answers. Waiting for it there is how `save()` hangs in the state a user is
        most likely to call it from, which is the bug this line exists to not have.
        A paused, approval-blocked, unstarted or finished run is written immediately.
        """

        if self._machine.terminal or self.state is RunState.CREATED:
            return False
        if self._task is None or self._task.done():
            return False
        return self.nonquiescent > 0

    def _save_targets(self, *, durable: bool) -> list[Path]:
        """Every path a checkpoint must write to, automatic and awaited, deduped."""

        targets: list[Path] = []
        auto = self.session_path
        if durable and self.autosave and auto is not None:
            targets.append(auto)
        for request in self._pending_saves:
            if request.path not in targets:
                targets.append(request.path)
        return targets

    def _release_pending_saves(self, written: Collection[Path]) -> None:
        """Resolve every awaited save whose path is now on disk. Holds the lock.

        Called after the bytes are down, so a caller is never told a save completed
        before the file exists. Synchronous: `set_result` does not suspend, which is
        what keeps the whole write-and-release sequence inside one lock hold.
        """

        if not self._pending_saves:
            return
        done = set(written)
        remaining: list[_SaveRequest] = []
        for request in self._pending_saves:
            if request.path in done and not request.future.done():
                request.future.set_result(request.path)
            elif request.path not in done:
                remaining.append(request)
        self._pending_saves = remaining

    @classmethod
    async def load(
        cls,
        path: str | Path,
        *,
        bus: EventBus | None = None,
        approval_handler: ApprovalHandler | None = None,
        build: Callable[[Any], Any] | None = None,
        session_dir: str | Path | None = None,
        autosave: bool = True,
        validate_transcripts: bool = True,
        strict_graph_hash: bool = False,
        input: Any = None,
    ) -> Controller:
        """Rebuild a run from a session document, PAUSED and ready to `resume()`.

        Never auto-starts (R-C-11). The workflow comes from `(import_path, config)`
        alone -- `build(config)` constructs its own provider and dispatcher (delta
        21) -- so nothing about the process that wrote the file is needed to read it.
        An unimportable path or a config that no longer validates raises here, with
        the path and the reason in the message, rather than five lines into `build`.

        `build` may be passed directly for a session with no `import_path`, which is
        how a test drives a resume without a module on disk.
        """

        source = Path(path)
        session = Session.load(source)
        config = session.workflow.validated_config()
        builder = build if build is not None else _builder_for(session.workflow)
        body = builder(config) if builder is not None else None

        controller = cls(
            body,
            run_id=session.run_id,
            bus=bus,
            permission_mode=session.permission_mode,
            approval_handler=approval_handler,
            gated_tools=session.gated_tools,
            main_agent=AgentId(session.main_agent),
            validate_transcripts=validate_transcripts,
            session_dir=session_dir if session_dir is not None else source.parent,
            session_path=source,
            workflow=session.workflow,
            rng_seed=session.rng_seed,
            autosave=autosave,
            created_at=session.created_at,
            strict_graph_hash=strict_graph_hash,
        )

        # A `build(config)` that returns a `Workflow` is the M5 shape: the controller
        # makes the body itself, which is also the only way it can hold the graph the
        # drift check needs. A callable body is the M2 shape and still works.
        if isinstance(body, (Workflow, Graph)):
            controller.bind_workflow(
                body,
                import_path=session.workflow.import_path,
                config=dict(session.workflow.config),
                config_type=session.workflow.config_type,
                input=input,
            )
            drift = controller.check_graph_drift(session)
            if drift is not None:
                await controller.emitter.emit(drift)

        await controller._restore(session, source)
        return controller

    async def _restore(self, session: Session, source: Path) -> None:
        """Adopt a reconciled session. The other half of `load()`.

        Order matters twice over. Reconciliation runs *before* any agent is
        registered, so nothing can observe a half-repaired transcript. Every agent
        the session names is registered *before* the body starts, because quiescence
        over an empty set is vacuously true and a run whose agents appear lazily
        declares itself paused while a subagent is about to spend money (trap 8).
        """

        self.resume_report = reconcile(session)
        self.nodes = dict(session.nodes)
        self.custom = dict(session.custom)
        self.bus.seed_seq(session.event_seq)
        self._resume_target = session.resume_state

        for agent_id, state in session.agents.items():
            self.agents[agent_id] = state
            if state.phase is AgentPhase.FINISHED:
                continue
            self._restored[agent_id] = state
            self._quiescence.enter(agent_id, AgentPhase.PARKED)
            state.phase = AgentPhase.PARKED
        self._agents_seen = bool(self._restored)

        self.gate.restore_pending(session.pending_approvals)

        # The gate is closed before the state is adopted, so a `start()` between the
        # two would park at the first safe point rather than run past it.
        self._quiescence.request_pause()
        self._machine.restore(RunState.PAUSED, reason=f"loaded from {source}")
        await self.emitter.emit(
            RunStateChanged(
                old=RunState.CREATED,
                new=RunState.PAUSED,
                reason=f"loaded from {source}: {self.resume_report.summary()}",
            )
        )

    # -- events -------------------------------------------------------------

    async def emit(self, event: Event) -> None:
        """Publish an event on this run's bus."""

        await self.emitter.emit(event)

    def emitter_for(self, agent_id: AgentId | str, node_id: str | None = None) -> EventEmitter:
        """An emitter bound to one agent, for an agent loop to carry around."""

        return self.bus.emitter(agent_id=str(agent_id), node_id=node_id)


def _builder_for(workflow: WorkflowRef) -> Callable[[Any], RunBody] | None:
    """Resolve a session's `build` callable, or `None` when it names no workflow.

    A session with no `import_path` is still loadable -- inspecting a checkpoint is
    a legitimate thing to do, and so is resuming one whose body is passed in by hand.
    It just cannot be started without one, which `start()` already says.
    """

    if not workflow.import_path:
        return None
    return workflow.resolve()


def _summarize(value: Any, limit: int = 200) -> str:
    """A node output as one short line, for the `NodeCompleted` event payload."""

    if value is None:
        return ""
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = [
    "DEFAULT_SAVE_TIMEOUT",
    "Controller",
    "FoldRecord",
    "InterruptResult",
    "RunBody",
]
