"""`FanOut`, `Gather`, `Map`, `Subgraph`: the nodes that run other nodes (R-W-3).

Every one of them runs its children **through the runner**, not by calling
`child.run()`. That is not indirection for its own sake: the runner is where the
memo lives (R-W-6), where `NodeStarted`/`NodeCompleted` come from, where the node's
own quiescence entry is opened, and where its safe points get taken. A container
that called its children directly would produce children with no records, and a
resumed fan-out would re-run the three branches that had already finished -- which
is precisely what the mid-fan-out resume test exists to catch.

Two rules hold across all four:

* **The container is `blocked_on_child` while its children run.** It has handed the
  work to something with safe points of its own. A container claiming to be
  `running` for the length of a fan-out is a run that can never reach PAUSED.
* **`on_child_error="continue"` turns a failure into a value** (R-W-7). The child's
  output becomes a `NodeFailure`, the siblings keep going, and the join downstream
  sees a list with a recognisable hole rather than a dead run. The default is
  `"fail"`, and a structured `TaskGroup` failure is unwrapped so the caller gets the
  child's exception rather than a group wrapping it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from azalabscode.runstate import AgentPhase
from azalabscode.workflows.graph import ConstRef, Env, NodeEntry, Ref
from azalabscode.workflows.node import Node

ChildErrorPolicy = Literal["fail", "continue"]


async def run_children(
    ctx: Any,
    entries: Sequence[NodeEntry],
    env: Env,
    *,
    on_child_error: ChildErrorPolicy = "fail",
    max_concurrency: int = 0,
) -> list[Any]:
    """Run `entries` concurrently under one `TaskGroup`, in declaration order out.

    `max_concurrency=0` means unbounded; anything else is a semaphore. The results
    come back in the order the entries were given, never completion order, so a
    fan-out's list index is the branch index whatever the models do.
    """

    if not entries:
        return []

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

    async def one(entry: NodeEntry) -> Any:
        if semaphore is None:
            return await ctx.runner.execute(entry, env, on_error=on_child_error)
        async with semaphore:
            return await ctx.runner.execute(entry, env, on_error=on_child_error)

    await ctx.phase(AgentPhase.BLOCKED_ON_CHILD)
    try:
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(one(entry), name=f"node:{entry.node_id}") for entry in entries
                ]
        except BaseExceptionGroup as group_error:
            raise unwrap_group(group_error) from None
    finally:
        await ctx.phase(AgentPhase.RUNNING)
    return [task.result() for task in tasks]


def unwrap_group(group: BaseExceptionGroup[BaseException]) -> BaseException:
    """The one exception worth re-raising out of a `TaskGroup` failure.

    A real child failure outranks a cancellation: losing it would turn a crashed
    branch into a clean interrupt. With several real failures the first is raised and
    the rest ride along on `__context__` via the group, which is still attached.
    """

    _, real = group.split(asyncio.CancelledError)
    if real is None:
        return group.exceptions[0] if group.exceptions else group
    flat = _flatten(real)
    return flat[0] if flat else group


def _flatten(group: BaseExceptionGroup[BaseException]) -> list[BaseException]:
    out: list[BaseException] = []
    for exc in group.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            out.extend(_flatten(exc))
        else:
            out.append(exc)
    return out


class FanOut(Node):
    """Run N named children concurrently on the same input (R-W-3).

    The children are entries in the graph with `parent` set to this node, so they
    have real ids, real records and real memos. The output is their outputs in
    declaration order.
    """

    kind: ClassVar[str] = "FanOut"
    output_type: str = "list"
    concurrent: ClassVar[bool] = True

    def __init__(
        self,
        *,
        on_child_error: ChildErrorPolicy = "fail",
        max_concurrency: int = 0,
    ) -> None:
        self.on_child_error: ChildErrorPolicy = on_child_error
        self.max_concurrency = max_concurrency

    async def run(self, ctx: Any, input: Any) -> list[Any]:
        """Run every branch on `input` and return their outputs in order."""

        entries = ctx.graph.children_of(str(ctx.node_id))
        return await run_children(
            ctx,
            entries,
            ctx.env.child(input),
            on_child_error=self.on_child_error,
            max_concurrency=self.max_concurrency,
        )


class Gather(Node):
    """Join several references into one list (R-W-3).

    All the work is in the input reference -- `wf.gather("join", [a, b])` compiles to
    a node whose input is a `SeqRef` -- so the body is the identity. It is still a
    node rather than a bare reference because the join is where a workflow usually
    wants a checkpoint, and because a `Gather` id in the session is how a resumed run
    knows the join happened.
    """

    kind: ClassVar[str] = "Gather"
    output_type: str = "list"

    async def run(self, ctx: Any, input: Any) -> Any:
        """Return the joined value, flattened to a list when it is a sequence."""

        if isinstance(input, (list, tuple)):
            return list(input)
        return input


class Map(Node):
    """Run one node once per element of a runtime list (R-W-3).

    Child ids are `<map_id>/<index>` -- the index in the input list, which *is* the
    monotonic counter spec 6.3 asks for and is stable across a resume because the
    list itself comes from a memoized upstream node. A counter carried in the session
    would say the same thing less directly and could disagree with the list.
    """

    kind: ClassVar[str] = "Map"
    output_type: str = "list"
    concurrent: ClassVar[bool] = True
    dynamic_children: ClassVar[bool] = True

    def __init__(
        self,
        template: Node,
        *,
        on_child_error: ChildErrorPolicy = "fail",
        max_concurrency: int = 0,
    ) -> None:
        self.template = template
        self.on_child_error: ChildErrorPolicy = on_child_error
        self.max_concurrency = max_concurrency

    def describe(self) -> str:
        return f"Map({self.template.describe()})"

    def hash_fields(self) -> tuple[str, str, str]:
        """Carries the template's identity: the children have no static entries of
        their own, so this is the only place their class reaches `graph_hash`."""

        base = super().hash_fields()
        template = self.template.hash_fields()
        return (f"Map<{template[0]},{template[1]},{template[2]}>", base[1], base[2])

    async def run(self, ctx: Any, input: Any) -> list[Any]:
        """Run the template once per item. `input` must be a sequence."""

        if not isinstance(input, (list, tuple)):
            raise TypeError(
                f"node {ctx.node_id} is a Map and needs a list; got {type(input).__name__}"
            )
        parent = str(ctx.node_id)
        entries = [
            NodeEntry(
                node_id=f"{parent}/{index}",
                node=self.template,
                input=ConstRef(item),
                parent=parent,
                name=str(index),
            )
            for index, item in enumerate(input)
        ]
        return await run_children(
            ctx,
            entries,
            ctx.env.child(input),
            on_child_error=self.on_child_error,
            max_concurrency=self.max_concurrency,
        )


class Subgraph(Node):
    """Run a nested workflow as one node (R-W-3).

    The inner graph's nodes were inlined into the outer graph at build time with
    their ids prefixed, so they are ordinary entries with ordinary memos; this node
    walks them in the inner topological order. `InputRef` inside the subgraph
    resolves to *this* node's input, which is what `Env.child()` is for.

    Sequential rather than concurrent: a subgraph is a pipeline. Concurrency inside
    one is spelled `FanOut`, in the inner workflow, where it is visible.
    """

    kind: ClassVar[str] = "Subgraph"
    output_type: str = "any"

    def __init__(
        self,
        order: Sequence[str],
        output: Ref | None = None,
        *,
        graph_name: str = "",
    ) -> None:
        self.order = tuple(order)
        self.output = output
        self.graph_name = graph_name

    def describe(self) -> str:
        return f"Subgraph({self.graph_name or len(self.order)})"

    async def run(self, ctx: Any, input: Any) -> Any:
        """Run the inner nodes in order and return the inner output."""

        env = ctx.env.child(input)
        last: Any = None
        await ctx.phase(AgentPhase.BLOCKED_ON_CHILD)
        try:
            for node_id in self.order:
                last = await ctx.runner.execute(ctx.graph.entry(node_id), env)
        finally:
            await ctx.phase(AgentPhase.RUNNING)
        if self.output is not None:
            return self.output.resolve(env)
        return last


__all__ = [
    "ChildErrorPolicy",
    "FanOut",
    "Gather",
    "Map",
    "Subgraph",
    "run_children",
    "unwrap_group",
]
