"""`NodeContext`: what a node is handed, and the only way it reaches the run.

Spec 6.3 lists the fields. Three of them are decisions rather than plumbing:

* **`state`** is the node's declared `State`, already restored. A node reads and
  mutates it freely and calls `checkpoint()` when it wants the mutation to survive.
  The runner never inspects it -- it dumps it, and a dump that is not JSON-safe is a
  `SerializationError` naming this node (R-W-5).
* **`checkpoint()`** is the safe point (spec 4.5). `workflows` decides *where* they
  are; `control` decides what to do at one. It parks by default, which is what makes
  `pause()` reach PAUSED on a graph with no agents in it at all.
* **`delegate` / `spawn`** are the graph-level half of R-W-4. `AgentLoop` has its
  own pair for the model-driven `delegate` tool; these are for a node that wants a
  subagent without being one.

**`spawn` is `async` here, and spec 6.3 writes it `def`.** Registering the child has
to happen before its task exists -- that is the entire content of spec delta 14's
spawn race -- and registration emits an event, so it awaits. A synchronous `spawn`
could only fire the registration off as a task of its own, which reintroduces
exactly the instant the delta exists to remove. The concurrency is unchanged: the
handle comes back before the child has done anything.

The **snapshot envelope** is `{"state": ..., "child_seq": n}`. The counter has to be
checkpointed with the state or two resumed spawns collide with a saved sibling
(spec 6.3), and it is not part of the node's declared `State` because the node did
not declare it and should not have to.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from azalabscode.cancellation import CancelReason, StepKind
from azalabscode.contracts import RunControl, SafePoint, SafePointKind
from azalabscode.events import EventEmitter
from azalabscode.ids import AgentId, NodeId, RunId
from azalabscode.permissions import DEFAULT_MODE, PermissionMode
from azalabscode.providers.base import Provider
from azalabscode.runstate import AgentPhase
from azalabscode.tools.context import ReadState
from azalabscode.tools.dispatcher import ToolDispatcher
from azalabscode.workflows.agent_loop import AgentLoop, AgentResult, AgentSpec
from azalabscode.workflows.graph import Env, Graph
from azalabscode.workflows.handle import AgentHandle
from azalabscode.workflows.step import StepHandle, run_inline_step

STATE_KEY = "state"
CHILD_SEQ_KEY = "child_seq"


@dataclass
class NodeContext[S: BaseModel]:
    """One node's view of the run (spec 6.3)."""

    run_id: RunId
    node_id: NodeId
    state: S
    control: RunControl | None = None
    emitter: EventEmitter | None = None
    provider: Provider | None = None
    tools: ToolDispatcher | None = None
    agent_id: AgentId | None = None
    """The agent this node *is*, for an `AgentNode`. `None` for every other kind."""
    attempt: int = 0
    specs: dict[str, AgentSpec] = field(default_factory=dict)
    task_group: asyncio.TaskGroup | None = None
    """The group a `spawn` joins. The runner opens one per node (R-W-7)."""
    custom: dict[str, Any] = field(default_factory=dict)
    """Workflow-level scratch. Shared across nodes; must stay JSON-serializable."""
    child_seq: int = 0
    """Subagents this node has started. Checkpointed with the state (spec 6.3)."""
    rng_seed: int | None = None
    graph: Graph | None = None
    """The compiled graph, so a container node can find the children it owns."""
    env: Env | None = None
    """The resolution environment, scoped to this node's input for its children."""
    runner: Any = None
    """The `Runner`, so a container can execute a child through the same memoized,
    checkpointed path a top-level node takes. Typed `Any` because `runner` imports
    this module and the reverse import would be a cycle."""

    # -- introspection ------------------------------------------------------

    @property
    def events(self) -> EventEmitter | None:
        """Spec 6.3 calls this `events`; the field is `emitter` everywhere else."""

        return self.emitter

    @property
    def permission_mode(self) -> PermissionMode:
        """The run's mode, read-only (spec 6.3)."""

        return self.control.permission_mode if self.control is not None else DEFAULT_MODE

    @property
    def rng(self) -> random.Random:
        """A generator seeded from `(rng_seed, node_id, attempt)`.

        Deterministic across a resume, and different per node, so a node that
        samples gets the same sample after a reload without every node in the graph
        sharing one stream whose position depends on execution order.
        """

        seed = f"{self.rng_seed}:{self.node_id}:{self.attempt}"
        return random.Random(seed)

    # -- state --------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The envelope written at a safe point: declared state plus the counter."""

        return {
            STATE_KEY: self.state.model_dump(mode="json"),
            CHILD_SEQ_KEY: self.child_seq,
        }

    def adopt(self, envelope: dict[str, Any] | None, state_type: type[BaseModel]) -> None:
        """Restore from an envelope written by an earlier process.

        A malformed or outdated envelope is ignored rather than fatal: the node has
        not completed (or the runner would not be here), so re-running it from
        defaults is correct, and refusing to load would strand the whole session on
        one node's state having changed shape.
        """

        if not envelope:
            return
        raw = envelope.get(STATE_KEY)
        if isinstance(raw, dict):
            # A state that no longer validates restarts from defaults rather than
            # stranding the whole session on one node's shape having changed.
            with contextlib.suppress(Exception):
                self.state = state_type.model_validate(raw)  # type: ignore[assignment]
        seq = envelope.get(CHILD_SEQ_KEY)
        if isinstance(seq, int) and seq > self.child_seq:
            self.child_seq = seq

    # -- safe points --------------------------------------------------------

    async def checkpoint(
        self,
        kind: SafePointKind = SafePointKind.CUSTOM,
        *,
        durable: bool = True,
        park: bool = True,
    ) -> None:
        """Declare a safe point and park there if a pause is pending (spec 4.5).

        `agent_id` is deliberately left unset: the quiescence key for a node is its
        node id, and `Controller.safe_point` parks on `agent_id or node_id`. An
        `AgentNode`'s own loop takes its safe points under its agent id instead.
        """

        if self.control is None:
            return
        await self.control.safe_point(
            SafePoint(
                kind=kind,
                node_id=str(self.node_id),
                attempt=self.attempt,
                snapshot=self.snapshot(),
                durable=durable,
                park=park,
            )
        )

    async def phase(self, phase: AgentPhase) -> None:
        """Set this node's quiescence phase.

        A node that hands work to something with its own safe points -- a subagent, a
        child node -- must go `blocked_on_child` first, or the run can never reach
        PAUSED while it waits.
        """

        if self.control is not None:
            await self.control.phase(AgentId(str(self.node_id)), phase)

    # -- subagents (R-W-4) --------------------------------------------------

    async def delegate(
        self,
        spec: AgentSpec | str,
        task: str,
        *,
        tools: Sequence[str] | None = None,
        model: str | None = None,
    ) -> AgentResult:
        """Run a subagent to completion and return its result (spec 6.3).

        The node is `blocked_on_child` throughout -- quiescent, for the same reason a
        delegating agent is (spec delta 14): a child parked at the pause gate would
        otherwise deadlock a parent that is never going to return.
        """

        child = await self._start(spec, task, tools=tools, model=model, delegated=True)
        await self.phase(AgentPhase.BLOCKED_ON_CHILD)
        try:
            return await child.result()
        finally:
            await self.phase(AgentPhase.RUNNING)

    async def spawn(
        self,
        spec: AgentSpec | str,
        task: str,
        *,
        tools: Sequence[str] | None = None,
        model: str | None = None,
    ) -> AgentHandle:
        """Start a subagent concurrently and return a handle (R-W-4).

        The parent stays `running`: it has work of its own to do, and claiming to be
        blocked would let a pause land while the parent is still between safe points.
        Flip to `blocked_on_child` yourself before awaiting `handle.result()`, or use
        `gather_handles`, which does it.
        """

        return await self._start(spec, task, tools=tools, model=model, delegated=False)

    async def gather_handles(self, *handles: AgentHandle) -> list[AgentResult]:
        """Await several spawned children, quiescent while waiting."""

        await self.phase(AgentPhase.BLOCKED_ON_CHILD)
        try:
            return [await handle.result() for handle in handles]
        finally:
            await self.phase(AgentPhase.RUNNING)

    async def _start(
        self,
        spec: AgentSpec | str,
        task: str,
        *,
        tools: Sequence[str] | None,
        model: str | None,
        delegated: bool,
    ) -> AgentHandle:
        """Build a child, register it, then create its task. In that order.

        The order is the whole point (spec delta 14): `enter_agent` runs *before*
        `create_task`, so there is no instant in which the child is unregistered and
        its parent is already quiescent.
        """

        if self.task_group is None:
            raise RuntimeError(
                f"node {self.node_id} has no task group; subagents may only be started "
                "from inside a node body run by the Runner (R-W-7 wants them structured)"
            )
        if self.provider is None or self.tools is None:
            raise RuntimeError(
                f"node {self.node_id} has no provider or dispatcher, so it cannot run a "
                "subagent; give the Workflow a default provider or pass one to the node"
            )

        child_spec = self._resolve_spec(spec, tools=tools, model=model)
        index = self.child_seq
        self.child_seq = index + 1  # no await between the read and the write (spec 6.3)
        child_id = AgentId(f"{self.node_id}/agent/{index}")

        loop = AgentLoop(
            child_spec,
            provider=self.provider,
            dispatcher=self._child_dispatcher(),
            control=self.control,
            emitter=self.emitter.bind(agent_id=str(child_id)) if self.emitter else None,
            agent_id=child_id,
            node_id=NodeId(str(child_id)),
            parent_id=AgentId(str(self.node_id)),
            specs=self.specs,
        )
        if self.control is not None:
            await self.control.enter_agent(
                child_id,
                AgentId(str(self.node_id)),
                spec_summary=child_spec.summary(),  # type: ignore[call-arg]
                delegated=delegated,  # type: ignore[call-arg]
                state=loop.state,  # type: ignore[call-arg]
            )

        step = StepHandle(
            agent_id=AgentId(str(self.node_id)),
            kind=StepKind.DELEGATE,
            node_id=self.node_id,
            child_agent_id=str(child_id),
            description=f"{'delegate' if delegated else 'spawn'} to {child_spec.name}",
        )
        # The handle must be live before it is handed out. A task cancelled before
        # its first step never enters its own coroutine -- `coro.throw()` on an
        # unstarted coroutine raises at the top -- so `run_inline_step` would never
        # bind, never absorb, and `handle.cancel()` immediately after `spawn()` would
        # take the *node* down instead of the child. Waiting for the child to reach
        # its first suspension closes that window, and nothing can cancel it in
        # between: there is no await between `create_task` and this one.
        running = asyncio.Event()
        task_obj = self.task_group.create_task(
            _run_agent(loop, task, step, self.control, running), name=f"agent:{child_id}"
        )
        step.bind(task_obj)
        await running.wait()
        return AgentHandle(
            agent_id=str(child_id), task=task_obj, step=step, spec_name=child_spec.name
        )

    def _resolve_spec(
        self, spec: AgentSpec | str, *, tools: Sequence[str] | None, model: str | None
    ) -> AgentSpec:
        base = self.specs[spec] if isinstance(spec, str) else spec
        updates: dict[str, Any] = {}
        if tools is not None:
            updates["tools"] = list(tools)
        if model is not None:
            updates["model"] = model
        return base.model_copy(update=updates) if updates else base

    def _child_dispatcher(self) -> ToolDispatcher:
        """A dispatcher sharing everything but the read-before-write record.

        A file the *node* read is not a file the child has seen, and read-before-write
        exists to stop a model overwriting contents it never looked at.
        """

        assert self.tools is not None
        context = dataclasses.replace(self.tools.context, read_state=ReadState())
        return ToolDispatcher(
            self.tools.tools,
            context=context,
            gate=self.tools.gate,
            emitter=self.tools.emitter,
            max_parallel=self.tools.max_parallel,
            turn_budget=self.tools.turn_budget_limit,
        )


async def _run_agent(
    loop: AgentLoop,
    task: str,
    step: StepHandle,
    control: RunControl | None,
    running: asyncio.Event,
) -> AgentResult:
    """Run a subagent as a tracked, cancellable step.

    `run_inline_step` rather than `run_step`: this coroutine is already a task of its
    own inside the node's group, and nesting another task under it would put a second
    cancellation boundary between the handle and the agent.

    `running` is set on the first line, before anything can suspend, so `spawn()`
    knows the coroutine has actually been entered.
    """

    running.set()
    result = await run_inline_step(
        lambda: loop.run(task), handle=step, control=control, capture_errors=False
    )
    if result.cancelled or result.value is None:
        return AgentResult(
            agent_id=str(loop.agent_id),
            ok=False,
            error="the subagent was cancelled before it finished",
            stop_reason="cancelled",
        )
    return result.value


def cancel_all(handles: Sequence[AgentHandle], reason: CancelReason) -> int:
    """Cancel every unfinished handle. Returns how many were still running."""

    return sum(1 for handle in handles if handle.cancel(reason))


__all__ = ["CHILD_SEQ_KEY", "STATE_KEY", "AgentHandle", "NodeContext", "cancel_all"]
