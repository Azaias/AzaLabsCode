"""`Workflow`: the builder from spec 6.3, and the validation that makes ids stable.

    wf = Workflow("fusion", input=question, provider=provider)
    outs = wf.fan_out("models", {name: ModelCall(m) for name, m in models})
    joined = wf.gather("join", outs)
    analysis = wf.node("analyze", ModelCall(cfg.analyst_model), input=joined)
    final = wf.node("synthesize", ModelCall(cfg.synth_model), input=(joined, analysis))
    wf.output(final)

**Every builder call takes an explicit `name`** (plan.md delta 20). Ids are
`"/".join(path + [name])` and nothing is inferred from the call site:
`inspect.stack()` breaks under `-O`, in frozen builds, inside decorators and inside
comprehensions, and a node id that changes because the workflow moved into a
comprehension is a session that will not load.

`compile()` is where a workflow stops being editable and starts being a `Graph`. It
runs four checks, all of them before a run starts and therefore before a run spends
anything:

1. ids are unique and well-formed (no `/`, no `@`, non-empty);
2. every reference names a node that exists, and the graph is acyclic (R-W-2);
3. every declared `State` round-trips through JSON with default values -- spec C-3
   stage 1, whose whole purpose is to fail before the money;
4. no `AgentNode`'s agent id collides with a node id, because the two share one
   quiescence map and a collision would have the agent's `exit_agent` delete the
   node's entry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from azalabscode.errors import ConfigurationError
from azalabscode.providers.base import Provider
from azalabscode.tools.dispatcher import ToolDispatcher
from azalabscode.workflows.agent_loop import AgentSpec
from azalabscode.workflows.graph import (
    FanOutRef,
    Graph,
    InputRef,
    NodeEntry,
    NodeRef,
    Ref,
    to_ref,
    topological_order,
    validate_states,
)
from azalabscode.workflows.node import Node
from azalabscode.workflows.nodes.agent import AgentNode
from azalabscode.workflows.nodes.containers import (
    ChildErrorPolicy,
    FanOut,
    Gather,
    Map,
    Subgraph,
)
from azalabscode.workflows.nodes.func import Func, FuncBody
from azalabscode.workflows.nodes.model_call import ModelCall

RESERVED_CHARACTERS = "/@"
"""`/` builds the path; `@` marks the runner's own quiescence key."""


class Workflow:
    """A graph under construction (spec 6.3, R-W-1)."""

    def __init__(
        self,
        name: str,
        *,
        input: Any = None,
        provider: Provider | None = None,
        dispatcher: ToolDispatcher | None = None,
        specs: Mapping[str, AgentSpec] | None = None,
    ) -> None:
        self.name = name
        self.input_value = input
        self.provider = provider
        self.dispatcher = dispatcher
        """The default `ToolDispatcher` for `AgentNode`s that do not bring one."""
        self.specs: dict[str, AgentSpec] = dict(specs or {})
        """Subagent specs, by name, for `ctx.delegate("reviewer", ...)`."""
        self._entries: list[NodeEntry] = []
        self._output: Ref | None = None

    # -- references ---------------------------------------------------------

    @property
    def input(self) -> InputRef:
        """The workflow's input. Inside a `Subgraph`, the subgraph's input."""

        return InputRef()

    def output(self, ref: Any) -> Ref:
        """Declare what the run returns. Without one, the last node's output."""

        self._output = to_ref(ref)
        return self._output

    def ref(self, node_id: str) -> NodeRef:
        """A reference to a node by id, for a graph assembled out of order."""

        return NodeRef(node_id)

    # -- adding nodes -------------------------------------------------------

    def add(
        self,
        name: str,
        node: Node,
        *,
        input: Any = None,
        parent: str | None = None,
    ) -> NodeRef:
        """Register a node under `name` and return a reference to its output.

        The low-level call every other builder method goes through. `parent` is for
        container children and is not part of the public shape -- use `fan_out`.
        """

        node_id = self._register(name, node, input=input, parent=parent)
        return NodeRef(node_id)

    def node(self, name: str, node: Node, *, input: Any = None) -> NodeRef:
        """Spec 6.3's `wf.node(name, Node(...), input=ref)`."""

        return self.add(name, node, input=input)

    def func(
        self,
        name: str,
        fn: FuncBody,
        *,
        input: Any = None,
        output_type: str = "any",
        takes_ctx: bool | None = None,
    ) -> NodeRef:
        """An async Python function as a node (R-W-3)."""

        return self.add(name, Func(fn, output_type=output_type, takes_ctx=takes_ctx), input=input)

    def model_call(self, name: str, model: str, *, input: Any = None, **kwargs: Any) -> NodeRef:
        """One completion, no tools (R-W-3)."""

        return self.add(name, ModelCall(model, **kwargs), input=input)

    def agent(self, name: str, spec: AgentSpec, *, input: Any = None, **kwargs: Any) -> NodeRef:
        """The model-to-tool loop as a node (R-W-3, R-W-8)."""

        return self.add(name, AgentNode(spec, **kwargs), input=input)

    def fan_out(
        self,
        name: str,
        children: Sequence[Node] | Mapping[str, Node],
        *,
        input: Any = None,
        on_child_error: ChildErrorPolicy = "fail",
        max_concurrency: int = 0,
    ) -> FanOutRef:
        """Run N children concurrently on the same input (R-W-3).

        A sequence gets index names (`models/0`); a mapping gets its keys
        (`models/gpt`). Prefer the mapping: an index shifts when a model is added to
        the middle of a list and every saved session below it stops lining up.
        """

        node_id = self._register(
            name,
            FanOut(on_child_error=on_child_error, max_concurrency=max_concurrency),
            input=input,
        )
        items = (
            list(children.items())
            if isinstance(children, Mapping)
            else [(str(index), child) for index, child in enumerate(children)]
        )
        branches: list[tuple[str, NodeRef]] = []
        for child_name, child in items:
            child_id = self._register(child_name, child, input=InputRef(), parent=node_id)
            branches.append((child_name, NodeRef(child_id)))
        return FanOutRef(node_id, branches=tuple(branches))

    def gather(self, name: str, refs: Any, *, input: Any = None) -> NodeRef:
        """Join several references into one list (R-W-3).

        `wf.gather("join", outs)` where `outs` is one reference to a list, or
        `wf.gather("join", [a, b])` where it is several -- both mean the same thing
        downstream, which is why the join is a node and not a syntax.
        """

        source = refs if input is None else input
        return self.add(name, Gather(), input=source)

    def map(
        self,
        name: str,
        template: Node,
        *,
        over: Any,
        on_child_error: ChildErrorPolicy = "fail",
        max_concurrency: int = 0,
    ) -> NodeRef:
        """Run `template` once per element of a runtime list (R-W-3)."""

        return self.add(
            name,
            Map(template, on_child_error=on_child_error, max_concurrency=max_concurrency),
            input=over,
        )

    def subgraph(self, name: str, workflow: Workflow | Graph, *, input: Any = None) -> NodeRef:
        """Nest a workflow as one node (R-W-3).

        The inner nodes are inlined with their ids prefixed, so they keep real
        records and real memos and a resumed run skips the inner nodes that already
        finished. `wf.input` inside the inner workflow becomes this node's input.
        """

        inner = workflow if isinstance(workflow, Graph) else workflow.compile()
        node_id = self._path(name)
        rename = _prefixer(node_id, inner.node_ids())
        self._register(
            name,
            Subgraph(
                order=tuple(rename(item) for item in inner.order),
                output=inner.output.remap(rename) if inner.output is not None else None,
                graph_name=inner.name,
            ),
            input=input,
        )
        for entry in inner.entries:
            self._entries.append(
                NodeEntry(
                    node_id=rename(entry.node_id),
                    node=entry.node,
                    input=entry.input.remap(rename),
                    parent=rename(entry.parent) if entry.parent is not None else node_id,
                    name=entry.name,
                )
            )
        self.specs.update(inner.specs)
        return NodeRef(node_id)

    # -- compilation --------------------------------------------------------

    def compile(self) -> Graph:
        """Validate and freeze. Every failure here happens before the run starts."""

        entries = tuple(self._entries)
        order = topological_order(entries)
        validate_states(entries)
        self._check_agent_ids(entries)
        return Graph(
            name=self.name,
            entries=entries,
            order=order,
            output=self._output,
            input_value=self.input_value,
            provider=self.provider,
            dispatcher=self.dispatcher,
            specs=dict(self.specs),
        )

    def graph_hash(self) -> str:
        """The compiled graph's digest (spec C-2)."""

        return self.compile().graph_hash()

    def node_ids(self) -> frozenset[str]:
        """Every static node id this workflow declares."""

        return frozenset(entry.node_id for entry in self._entries)

    def describe(self) -> str:
        """The graph as text, one node per line."""

        return self.compile().describe()

    # -- internals ----------------------------------------------------------

    def _register(
        self,
        name: str,
        node: Node,
        *,
        input: Any = None,
        parent: str | None = None,
    ) -> str:
        _check_name(name)
        node_id = self._path(name) if parent is None else f"{parent}/{name}"
        if any(entry.node_id == node_id for entry in self._entries):
            raise ConfigurationError(
                f"duplicate node id {node_id!r} in workflow {self.name!r}: every builder "
                "call needs its own name (plan delta 20)"
            )
        self._entries.append(
            NodeEntry(
                node_id=node_id,
                node=node,
                input=to_ref(input) if input is not None else InputRef(),
                parent=parent,
                name=name,
            )
        )
        return node_id

    def _path(self, name: str) -> str:
        return name

    def _check_agent_ids(self, entries: Sequence[NodeEntry]) -> None:
        ids = {entry.node_id for entry in entries}
        for entry in entries:
            node = entry.node
            if isinstance(node, AgentNode) and node.agent_id in ids:
                raise ConfigurationError(
                    f"node {entry.node_id!r} runs an agent whose id {node.agent_id!r} is also "
                    "a node id; they share one quiescence map, so the agent's exit would "
                    "delete the node's entry. Pass AgentNode(..., agent_id=...)."
                )


def _check_name(name: str) -> None:
    """A builder name must be usable as one path segment."""

    if not name:
        raise ConfigurationError("a node name may not be empty (plan delta 20)")
    bad = [character for character in RESERVED_CHARACTERS if character in name]
    if bad:
        raise ConfigurationError(
            f"node name {name!r} contains reserved character(s) {''.join(bad)!r}: "
            "'/' builds the node path and '@' marks the runner's own key"
        )


def _prefixer(prefix: str, inner_ids: frozenset[str]) -> Any:
    """Rename inner node ids under `prefix`, leaving anything foreign alone."""

    def rename(node_id: str) -> str:
        return f"{prefix}/{node_id}" if node_id in inner_ids else node_id

    return rename


__all__ = ["RESERVED_CHARACTERS", "Workflow"]
