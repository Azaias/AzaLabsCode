"""`Runner`: executes a compiled graph against a `RunControl`.

It is the run body. `Controller` calls it with itself; the runner never sees a
`Controller`, only the protocol (import-linter contract 3).

Everything a node does *not* have to think about is here:

* **The memo (R-W-6).** Before anything else, `node_completed(node_id, attempt)`. A
  completed node is not run; its output is read back from the session -- from
  `values/` if it was spilled. This is the whole content of "a resumed run retains
  every completed output", and it is one branch at the top of `execute`.
* **The quiescence entry.** A node is registered with `enter_node` for its lifetime.
  A graph of `Func` nodes has no agents at all, and without this a `pause()` on one
  would never reach PAUSED -- there would be nothing to count and nothing to park.
* **The safe points.** `node_entered` before the body, `node_completed` after it.
  Both park, which is what makes `pause()` land on a graph rather than only on an
  agent.
* **The failure story (R-W-7).** `on_error="continue"` turns a raise into a
  `NodeFailure` *value*; the default lets it fly and the run goes FAILED.

**The runner has a quiescence entry of its own**, keyed `@<graph name>`. Between two
nodes there is no node registered, and quiescence over an empty set is vacuously
true -- a pause landing there would declare PAUSED while the runner walks on to the
next node. The runner is `running` between nodes, `blocked_on_child` during one, and
parks at a safe point before each. `@` cannot appear in a builder name, so the key
cannot collide with a node id.

Each node body runs inside its own `TaskGroup`, which is what makes `ctx.spawn`
structured (R-W-7): a node cannot outlive its spawned children, and a child that
raises takes the node down.
"""

from __future__ import annotations

import asyncio
from typing import Any, final

from azalabscode.contracts import RunControl, SafePoint, SafePointKind
from azalabscode.ids import NodeId, RunId
from azalabscode.runstate import AgentPhase
from azalabscode.workflows.context import NodeContext
from azalabscode.workflows.graph import Env, Graph, NodeEntry, as_graph
from azalabscode.workflows.node import NodeFailure
from azalabscode.workflows.nodes.containers import ChildErrorPolicy, unwrap_group

GRAPH_KEY_PREFIX = "@"
"""Marks the runner's own quiescence key. A builder name may not contain it."""


@final
class Runner:
    """Walks a graph, once, under a `RunControl`."""

    def __init__(
        self,
        graph: Any,
        *,
        input: Any = None,
        attempt: int = 0,
    ) -> None:
        self.graph: Graph = as_graph(graph)
        self.input = input if input is not None else self.graph.input_value
        self.attempt = attempt
        self.env = Env(input=self.input)
        self.custom: dict[str, Any] = {}
        self.control: RunControl | None = None
        self.contexts: dict[str, NodeContext[Any]] = {}
        """Every context built this run, by node id. What a test inspects."""

    @property
    def key(self) -> str:
        """The runner's own quiescence key."""

        return f"{GRAPH_KEY_PREFIX}{self.graph.name}"

    # -- the body -----------------------------------------------------------

    async def run(self, control: RunControl | None = None) -> Any:
        """Execute the graph and return its output. This is the `RunBody`.

        `control=None` runs the graph with no run around it -- no memo, no safe
        points, no quiescence. That is the same standalone shape the tool layer has
        behind `AllowAllGate` and the agent loop has with `control=None`, and it is
        what a script that just wants the answer should be able to write.
        """

        self.control = control
        if control is None:
            for entry in self.graph.roots():
                await self.execute(entry, self.env)
            return self._output()

        await control.enter_node(self.key)
        try:
            for entry in self.graph.roots():
                await self._park_point()
                await control.phase(self.key, AgentPhase.BLOCKED_ON_CHILD)  # type: ignore[arg-type]
                try:
                    await self.execute(entry, self.env)
                finally:
                    await control.phase(self.key, AgentPhase.RUNNING)  # type: ignore[arg-type]
            await self._park_point()
        finally:
            await control.exit_node(self.key)

        return self._output()

    def _output(self) -> Any:
        """The graph's declared output, or the last top-level node's."""

        if self.graph.output is not None:
            return self.graph.output.resolve(self.env)
        return self.env.outputs.get(self.graph.order[-1]) if self.graph.order else None

    async def _park_point(self) -> None:
        """A safe point between two nodes, with nothing of its own to snapshot."""

        if self.control is None:
            return
        await self.control.safe_point(
            SafePoint(
                kind=SafePointKind.CUSTOM,
                node_id=self.key,
                attempt=self.attempt,
                durable=True,
                park=True,
            )
        )

    # -- one node -----------------------------------------------------------

    async def execute(
        self,
        entry: NodeEntry,
        env: Env,
        *,
        on_error: ChildErrorPolicy = "fail",
    ) -> Any:
        """Run one node, or return its memoized output, and record the result.

        Called by the top-level walk and by every container node, so a fan-out branch
        and a top-level node take exactly the same path -- which is why a branch that
        completed before a kill is not re-run after the load.
        """

        control = self.control
        node_id = entry.node_id

        if control is not None and control.node_completed(node_id, attempt=self.attempt):
            value = control.node_output(node_id, attempt=self.attempt)
            env.set(node_id, value)
            return value

        try:
            value = await self._run_node(entry, env)
        except Exception as error:
            if on_error != "continue":
                raise
            if control is not None:
                await control.node_failed(node_id, f"{type(error).__name__}: {error}")
            failure = NodeFailure.of(node_id, error).model_dump(mode="json")
            env.set(node_id, failure)
            if control is not None:
                await control.node_finished(node_id, failure, attempt=self.attempt)
            return failure

        env.set(node_id, value)
        return value

    async def _run_node(self, entry: NodeEntry, env: Env) -> Any:
        """The uncaught path: enter, restore, run, memoize, checkpoint, exit."""

        control = self.control
        node_id = entry.node_id
        value_in = entry.input.resolve(env)
        ctx = self._context(entry, env)
        self.contexts[node_id] = ctx

        if control is not None:
            await control.enter_node(node_id)
        try:
            if control is not None:
                ctx.adopt(control.node_state(node_id, attempt=self.attempt), entry.node.State)
                await control.node_started(
                    node_id,
                    attempt=self.attempt,
                    input=value_in,
                    node_class=entry.node.kind,
                )
            await ctx.checkpoint(SafePointKind.NODE_ENTERED)

            try:
                async with asyncio.TaskGroup() as group:
                    ctx.task_group = group
                    body = group.create_task(entry.node.run(ctx, value_in), name=f"body:{node_id}")
                value = body.result()
            except BaseExceptionGroup as group_error:
                raise unwrap_group(group_error) from None
            finally:
                ctx.task_group = None

            if control is not None:
                await control.node_finished(node_id, value, attempt=self.attempt)
            await ctx.checkpoint(SafePointKind.NODE_COMPLETED)
            return value
        except Exception as error:
            if control is not None:
                await control.node_failed(node_id, f"{type(error).__name__}: {error}")
            raise
        finally:
            if control is not None:
                await control.exit_node(node_id)

    def _context(self, entry: NodeEntry, env: Env) -> NodeContext[Any]:
        """Build the context one node runs with."""

        control = self.control
        node_id = entry.node_id
        emitter = None
        if control is not None:
            emitter = control.emitter_for(node_id, node_id)
        return NodeContext(
            run_id=RunId(str(getattr(control, "run_id", ""))),
            node_id=NodeId(node_id),
            state=entry.node.State(),
            control=control,
            emitter=emitter,
            provider=self.graph.provider,
            tools=self.graph.dispatcher,
            attempt=self.attempt,
            specs=dict(self.graph.specs),
            custom=self.custom,
            rng_seed=getattr(control, "rng_seed", None),
            graph=self.graph,
            env=env,
            runner=self,
        )


__all__ = ["GRAPH_KEY_PREFIX", "Runner"]
