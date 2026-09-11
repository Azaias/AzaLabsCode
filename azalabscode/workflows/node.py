"""`Node`: the unit the graph is made of, and the unit a checkpoint is taken at.

Spec R-W-2 puts the cycles *inside* nodes rather than in the graph, and that one
choice is what makes the whole checkpoint story work: a node either produced its
output or it did not, so `(node_id, attempt) -> output` is a complete memo (R-W-6)
and a resumed run has exactly one question to ask per node.

Three declarations carry the weight:

* **`State`** -- a pydantic model the runner stores and restores (R-W-5). The default
  is `EmptyState`, which is the honest answer for a node whose whole state is its
  output. A node that declares one gets it back on resume, populated from the last
  safe point it took.
* **`output_type`** -- a *name*, not a type. It goes into `graph_hash` (spec C-2)
  alongside the node id, the class and the state type, and a name is what survives
  being written into a session file and compared in another process.
* **`dynamic_children`** -- whether this node makes node ids at runtime. `Map` and
  `AgentNode` do (`<parent>/<index>` and `<parent>/agent/<n>`). It is what stops the
  drift check calling a saved `map/3` a missing node when the rebuilt graph has only
  the static `map` (spec C-2 has nothing to say about dynamic children, so the
  decision is here: a dynamic child is accounted for by its declaring parent).

Node outputs must be JSON-native -- `str`, `list`, `dict`, numbers, `None`. The
session spills large ones to `values/` through `json.dumps`, so a pydantic model
would round-trip to its `repr` and a resumed run would hand the next node a string
where it expected an object. Every built-in node returns a JSON-native value and
`Func` is documented to.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from azalabscode.schema import HarnessModel


class EmptyState(HarnessModel):
    """The state of a node that has none. Serializes to `{}`."""


class NodeFailure(HarnessModel):
    """A child node's failure, as the value that stands in for its output (R-W-7).

    Produced only under `on_child_error="continue"`. It is a *value*, so a `Gather`
    downstream sees a list with a hole it can recognise rather than a run that died,
    and JSON-native, so it survives the session like any other output.
    """

    failed: bool = True
    node_id: str = ""
    error: str = ""
    error_type: str = ""

    @classmethod
    def of(cls, node_id: str, error: BaseException) -> NodeFailure:
        """Wrap an exception raised by a child node."""

        return cls(node_id=node_id, error=str(error), error_type=type(error).__name__)


class Node:
    """One step of a workflow: typed input in, JSON-native output out.

    Subclasses override `run`. Everything else is declaration. The base class is
    concrete rather than abstract so that a `Node()` can stand in for a node type in
    a hash test without a subclass ceremony; `run` raises if it is ever called.
    """

    kind: ClassVar[str] = "node"
    """Short name for events and for `graph_hash`. Defaults to the class name."""

    State: ClassVar[type[BaseModel]] = EmptyState
    """The state the runner checkpoints and restores (R-W-5)."""

    output_type: str = "any"
    """The name of what `run` returns. Part of `graph_hash` (spec C-2).

    A plain attribute, not a `ClassVar`: `Func` and `AgentNode` set it per instance,
    because what they return depends on how they were constructed."""

    dynamic_children: ClassVar[bool] = False
    """Whether this node mints node ids at runtime (`Map`, `AgentNode`)."""

    concurrent: ClassVar[bool] = False
    """Whether this node runs children concurrently. Purely informational."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "kind" not in cls.__dict__:
            cls.kind = cls.__name__

    async def run(self, ctx: Any, input: Any) -> Any:
        """Do the work. `ctx` is a `NodeContext`; the annotation avoids a cycle."""

        raise NotImplementedError(f"{type(self).__name__} does not implement run()")

    def describe(self) -> str:
        """One line for an event payload or an error message."""

        return type(self).__name__

    def hash_fields(self) -> tuple[str, str, str]:
        """`(node_class, state_type, output_type)` -- this node's share of `graph_hash`.

        Deliberately excludes prompts, models and config: editing a system prompt
        must not invalidate a saved session (plan.md, "Node ids and drift").
        """

        return (self.kind, self.State.__name__, self.output_type)


__all__ = ["EmptyState", "Node", "NodeFailure"]
