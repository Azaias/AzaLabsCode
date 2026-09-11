"""The graph: references between nodes, the compiled DAG, and `graph_hash`.

A `Ref` is a promise about a value that does not exist yet. `wf.input` is one, every
builder call returns one, and `input=(joined, analysis)` is one built out of two
others. Refs are what make the graph a graph: `Ref.deps()` is the edge set, and the
topological order falls out of it.

The compiled `Graph` is **flat**. A fan-out branch and a subgraph's internals are
entries in the same tuple as the top-level nodes, distinguished by `parent`. The
alternative -- containers holding their own sub-graphs -- means every id walk, every
hash and every drift check has to recurse, and each of those is a place to forget a
level. Flat means `node_ids()` is a set comprehension and the runner's outer loop is
`for entry in graph.roots()`.

`graph_hash` covers `(node_id, node_class, state_type, output_type)` for every entry
and **excludes** prompts, models and config (plan.md, "Node ids and drift"): editing
a system prompt must not invalidate a saved session. What it does catch is a node
changing type under a stable id, which is the case where restoring the saved state
would be actively wrong.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from azalabscode.errors import ConfigurationError
from azalabscode.workflows.node import Node

# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


class Ref:
    """A reference to a value the run will produce. The graph's edge type."""

    def deps(self) -> frozenset[str]:
        """Node ids this reference reads. The edge set of the DAG."""

        return frozenset()

    def resolve(self, env: Env) -> Any:
        """The value, given an environment. Raises `KeyError` if it is not ready."""

        raise NotImplementedError

    def describe(self) -> str:
        """One line, for an error message."""

        return repr(self)

    def __getitem__(self, key: int | str) -> Ref:
        """`ref[0]` / `ref["k"]` -- one element of a list or dict output."""

        return ItemRef(self, key)

    def remap(self, rename: Callable[[str], str]) -> Ref:
        """A copy with every node id renamed. What nesting a subgraph needs.

        An `InputRef` is deliberately left alone: inside a subgraph it means the
        subgraph's input, and `Env.child()` is what binds it.
        """

        return self


@dataclass(frozen=True)
class InputRef(Ref):
    """The workflow's own input, or -- inside a container -- the container's input.

    The scoping is the point: a `Subgraph`'s nodes were written against `wf.input`
    of the *inner* workflow, and when that workflow is nested the inner input is
    whatever the `Subgraph` node was given. `Env.child()` rebinds it.
    """

    def deps(self) -> frozenset[str]:
        return frozenset()

    def resolve(self, env: Env) -> Any:
        return env.input

    def describe(self) -> str:
        return "<input>"


@dataclass(frozen=True)
class NodeRef(Ref):
    """The output of one node."""

    node_id: str

    def deps(self) -> frozenset[str]:
        return frozenset({self.node_id})

    def resolve(self, env: Env) -> Any:
        return env.output(self.node_id)

    def describe(self) -> str:
        return self.node_id

    def remap(self, rename: Callable[[str], str]) -> Ref:
        return NodeRef(rename(self.node_id))


@dataclass(frozen=True)
class FanOutRef(NodeRef):
    """A `FanOut`'s list output, with its branches reachable by name.

    `outs["gpt"]` is the branch's own `NodeRef`, not an index into the list, so a
    downstream node depends on that one branch rather than on the whole fan-out.
    """

    branches: tuple[tuple[str, NodeRef], ...] = ()

    def branch(self, name: str) -> NodeRef:
        """One branch's output reference, by the name the builder gave it."""

        for key, ref in self.branches:
            if key == name:
                return ref
        raise KeyError(f"{self.node_id} has no branch {name!r}")

    def __getitem__(self, key: int | str) -> Ref:
        if isinstance(key, str):
            return self.branch(key)
        return self.branches[key][1] if key < len(self.branches) else ItemRef(self, key)

    def __iter__(self) -> Iterator[NodeRef]:
        return (ref for _, ref in self.branches)


@dataclass(frozen=True)
class ItemRef(Ref):
    """One element of another reference's value."""

    source: Ref
    key: int | str

    def deps(self) -> frozenset[str]:
        return self.source.deps()

    def resolve(self, env: Env) -> Any:
        return self.source.resolve(env)[self.key]

    def describe(self) -> str:
        return f"{self.source.describe()}[{self.key!r}]"

    def remap(self, rename: Callable[[str], str]) -> Ref:
        return ItemRef(self.source.remap(rename), self.key)


@dataclass(frozen=True)
class SeqRef(Ref):
    """Several references as one list (or tuple), in order."""

    refs: tuple[Ref, ...]
    as_tuple: bool = False

    def deps(self) -> frozenset[str]:
        return frozenset().union(*(ref.deps() for ref in self.refs)) if self.refs else frozenset()

    def resolve(self, env: Env) -> Any:
        values = [ref.resolve(env) for ref in self.refs]
        return tuple(values) if self.as_tuple else values

    def describe(self) -> str:
        return "(" + ", ".join(ref.describe() for ref in self.refs) + ")"

    def remap(self, rename: Callable[[str], str]) -> Ref:
        return SeqRef(tuple(ref.remap(rename) for ref in self.refs), as_tuple=self.as_tuple)


@dataclass(frozen=True)
class MapRef(Ref):
    """Several references as one dict, keyed."""

    items: tuple[tuple[str, Ref], ...]

    def deps(self) -> frozenset[str]:
        parts = [ref.deps() for _, ref in self.items]
        return frozenset().union(*parts) if parts else frozenset()

    def resolve(self, env: Env) -> Any:
        return {key: ref.resolve(env) for key, ref in self.items}

    def describe(self) -> str:
        return "{" + ", ".join(f"{k}: {v.describe()}" for k, v in self.items) + "}"

    def remap(self, rename: Callable[[str], str]) -> Ref:
        return MapRef(tuple((k, v.remap(rename)) for k, v in self.items))


@dataclass(frozen=True)
class ConstRef(Ref):
    """A literal. What a plain value passed as `input=` becomes."""

    value: Any

    def resolve(self, env: Env) -> Any:
        return self.value

    def describe(self) -> str:
        return repr(self.value)


def to_ref(value: Any) -> Ref:
    """Coerce a builder argument into a `Ref`.

    A tuple stays a tuple and a list stays a list, because spec 6.3's
    `input=(joined, analysis)` is a pair and a node that takes a pair should get one.
    """

    if isinstance(value, Ref):
        return value
    if isinstance(value, tuple):
        return SeqRef(tuple(to_ref(item) for item in value), as_tuple=True)
    if isinstance(value, list):
        return SeqRef(tuple(to_ref(item) for item in value))
    if isinstance(value, Mapping):
        return MapRef(tuple((str(k), to_ref(v)) for k, v in value.items()))
    return ConstRef(value)


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


@dataclass
class Env:
    """Where a `Ref` looks up its value.

    `outputs` is shared by the whole run -- node ids are globally unique, so one map
    is enough -- while `input` is scoped, and rebinding it in `child()` is what makes
    a `Subgraph` see its own input rather than the outer workflow's.
    """

    input: Any = None
    outputs: dict[str, Any] = field(default_factory=dict)

    def output(self, node_id: str) -> Any:
        """One node's output. `KeyError` names the node, not the dict."""

        try:
            return self.outputs[node_id]
        except KeyError:
            raise KeyError(f"node {node_id!r} has no output yet") from None

    def set(self, node_id: str, value: Any) -> None:
        """Record a node's output."""

        self.outputs[node_id] = value

    def child(self, input: Any) -> Env:
        """A view with a different input and the same output map."""

        return Env(input=input, outputs=self.outputs)


# ---------------------------------------------------------------------------
# Entries and the graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeEntry:
    """One node, its id, where its input comes from, and who contains it."""

    node_id: str
    node: Node
    input: Ref = field(default_factory=InputRef)
    parent: str | None = None
    """The container node that runs this one, or `None` for a top-level node."""
    name: str = ""
    """The name the builder was given, without the path prefix."""

    def describe(self) -> str:
        """One line: `id = Class(input)`."""

        return f"{self.node_id} = {self.node.describe()}({self.input.describe()})"


@dataclass(frozen=True)
class Graph:
    """A validated, flat, immutable DAG.

    Produced by `Workflow.compile()`. Everything the runner and the drift check need
    is a lookup on this object; nothing about the builder survives into it.
    """

    name: str
    entries: tuple[NodeEntry, ...]
    order: tuple[str, ...]
    """Top-level node ids in topological order. Container children are not here --
    their container runs them."""
    output: Ref | None = None
    input_value: Any = None
    provider: Any = None
    dispatcher: Any = None
    specs: Mapping[str, Any] = field(default_factory=dict)

    # -- lookups ------------------------------------------------------------

    def entry(self, node_id: str) -> NodeEntry:
        """One entry by id. `KeyError` naming the graph if it is not there."""

        for item in self.entries:
            if item.node_id == node_id:
                return item
        raise KeyError(f"{self.name} has no node {node_id!r}")

    def node_ids(self) -> frozenset[str]:
        """Every static node id, at every level."""

        return frozenset(item.node_id for item in self.entries)

    def children_of(self, node_id: str) -> tuple[NodeEntry, ...]:
        """The entries a container node runs, in declaration order."""

        return tuple(item for item in self.entries if item.parent == node_id)

    def roots(self) -> tuple[NodeEntry, ...]:
        """Top-level entries, in topological order."""

        return tuple(self.entry(node_id) for node_id in self.order)

    # -- identity -----------------------------------------------------------

    def graph_hash(self) -> str:
        """A digest over `(node_id, node_class, state_type, output_type)` (spec C-2)."""

        payload = sorted([item.node_id, *item.node.hash_fields()] for item in self.entries)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def accounts_for(self, node_id: str) -> bool:
        """Whether this graph can explain a node id found in a saved session.

        A static id explains itself. A dynamic one -- `map/3`, `writer/agent/0` --
        is explained by the nearest ancestor that *declares* it makes ids at runtime.
        Without this the drift check would call every `Map` child a missing node and
        refuse to load a session that did nothing wrong (spec C-2 does not consider
        dynamic children; this is the reading most consistent with R-W-6).
        """

        ids = self.node_ids()
        if node_id in ids:
            return True
        parts = node_id.split("/")
        for cut in range(len(parts) - 1, 0, -1):
            prefix = "/".join(parts[:cut])
            if prefix in ids:
                return self.entry(prefix).node.dynamic_children
        return False

    def describe(self) -> str:
        """The whole graph as text, one node per line. For a failure message."""

        return "\n".join(item.describe() for item in self.entries)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def topological_order(entries: Sequence[NodeEntry]) -> tuple[str, ...]:
    """Top-level ids in dependency order. Raises `ConfigurationError` on a cycle.

    Ties are broken by declaration order, so the same graph always produces the same
    sequence and a `by_index` script stays meaningful.
    """

    roots = [item for item in entries if item.parent is None]
    known = {item.node_id for item in entries}
    top = {item.node_id for item in roots}
    container_of = {item.node_id: item.parent for item in entries}

    remaining = list(roots)
    resolved: list[str] = []
    done: set[str] = set()

    while remaining:
        ready = [
            item
            for item in remaining
            if all(_top_level(dep, container_of, top) in done for dep in item.input.deps())
        ]
        if not ready:
            stuck = ", ".join(sorted(item.node_id for item in remaining))
            raise ConfigurationError(
                f"the graph has a cycle or an unreachable dependency among: {stuck}. "
                "Cycles belong inside a node, not between nodes (R-W-2)."
            )
        for item in ready:
            resolved.append(item.node_id)
            done.add(item.node_id)
        remaining = [item for item in remaining if item not in ready]

    missing = {dep for item in entries for dep in item.input.deps() if dep not in known}
    if missing:
        raise ConfigurationError(
            f"the graph references undefined node(s): {', '.join(sorted(missing))}"
        )
    return tuple(resolved)


def _top_level(node_id: str, container_of: Mapping[str, str | None], top: set[str]) -> str:
    """The top-level ancestor of a node id.

    A reference to a fan-out branch is, for ordering purposes, a reference to the
    fan-out: the branch does not run on its own.
    """

    seen: set[str] = set()
    current = node_id
    while current not in top and current in container_of and current not in seen:
        seen.add(current)
        parent = container_of[current]
        if parent is None:
            break
        current = parent
    return current


def validate_states(entries: Iterable[NodeEntry]) -> None:
    """Spec C-3 stage 1: every declared `State` round-trips through JSON at build time.

    Raising here -- before `start()`, before a single token is bought -- is the whole
    point. Stage 2 is the safe point, where the *actual* state is serialized and a
    failure is a `SerializationError` that fails the run (R-W-5).
    """

    for item in entries:
        state_type = item.node.State
        try:
            instance = state_type()
            state_type.model_validate_json(instance.model_dump_json())
        except Exception as error:
            raise ConfigurationError(
                f"node {item.node_id!r} declares State={state_type.__name__}, which does not "
                f"round-trip through JSON with default values: {type(error).__name__}: {error}"
            ) from error


def as_graph(obj: Any) -> Graph:
    """A `Graph`, from a `Graph` or from anything with a `compile()` that returns one.

    Duck-typed on purpose: `runner` must not import the builder, or `Subgraph` --
    which needs the runner -- could not be reached from the builder.
    """

    if isinstance(obj, Graph):
        return obj
    compile_fn = getattr(obj, "compile", None)
    if callable(compile_fn):
        result = compile_fn()
        if isinstance(result, Graph):
            return result
    raise TypeError(f"expected a Workflow or a Graph, got {type(obj).__name__}")


__all__ = [
    "ConstRef",
    "Env",
    "FanOutRef",
    "Graph",
    "InputRef",
    "ItemRef",
    "MapRef",
    "NodeEntry",
    "NodeRef",
    "Ref",
    "SeqRef",
    "as_graph",
    "to_ref",
    "topological_order",
    "validate_states",
]
