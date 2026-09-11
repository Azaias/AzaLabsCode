"""The builder, the references, the compiled graph and `graph_hash`.

Everything here happens before a run starts, which is the point: spec C-3 stage 1
and plan delta 20 both exist so that a broken graph costs nothing.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from azalabscode import (
    ConfigurationError,
    Env,
    FanOut,
    Func,
    Gather,
    InputRef,
    ModelCall,
    Node,
    NodeRef,
    Workflow,
    to_ref,
)
from azalabscode.workflows.graph import ConstRef, ItemRef, SeqRef


class BadState(BaseModel):
    """A state whose default value cannot be written down (spec C-3 stage 1)."""

    handle: Any = object()


class BadNode(Node):
    State: ClassVar[type[BaseModel]] = BadState


def echo(value: Any) -> Any:
    """A Func body that takes only the input."""

    return value


def with_ctx(ctx: Any, value: Any) -> Any:
    """A Func body that takes the context too."""

    return f"{ctx.node_id}:{value}"


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------


def test_node_ids_come_from_the_explicit_name() -> None:
    """Plan delta 20: never from stack introspection, always from the argument."""

    wf = Workflow("w")
    ref = wf.func("clean", echo)
    assert isinstance(ref, NodeRef)
    assert ref.node_id == "clean"
    assert wf.node_ids() == frozenset({"clean"})


def test_fan_out_children_are_named_under_the_parent() -> None:
    wf = Workflow("w")
    outs = wf.fan_out("models", {"a": ModelCall("m/a"), "b": ModelCall("m/b")})
    assert outs.node_id == "models"
    assert [ref.node_id for ref in outs] == ["models/a", "models/b"]
    assert outs.branch("b").node_id == "models/b"
    by_key = outs["a"]
    assert isinstance(by_key, NodeRef)
    assert by_key.node_id == "models/a"


def test_a_sequence_of_children_gets_index_names() -> None:
    wf = Workflow("w")
    outs = wf.fan_out("models", [ModelCall("m/a"), ModelCall("m/b")])
    assert [ref.node_id for ref in outs] == ["models/0", "models/1"]


def test_a_duplicate_name_is_refused_at_build_time() -> None:
    wf = Workflow("w")
    wf.func("clean", echo)
    with pytest.raises(ConfigurationError, match="duplicate node id"):
        wf.func("clean", echo)


@pytest.mark.parametrize("name", ["", "a/b", "a@b"])
def test_reserved_and_empty_names_are_refused(name: str) -> None:
    """`/` builds the path and `@` is the runner's own key."""

    wf = Workflow("w")
    with pytest.raises(ConfigurationError):
        wf.func(name, echo)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def test_to_ref_keeps_a_tuple_a_tuple_and_a_list_a_list() -> None:
    """Spec 6.3's `input=(joined, analysis)` is a pair; the node should get one."""

    env = Env(input="in", outputs={"a": 1, "b": 2})
    pair = to_ref((NodeRef("a"), NodeRef("b")))
    listed = to_ref([NodeRef("a"), NodeRef("b")])
    assert pair.resolve(env) == (1, 2)
    assert listed.resolve(env) == [1, 2]
    assert pair.deps() == {"a", "b"}


def test_a_plain_value_becomes_a_constant() -> None:
    ref = to_ref("hello")
    assert isinstance(ref, ConstRef)
    assert ref.resolve(Env()) == "hello"
    assert ref.deps() == frozenset()


def test_item_refs_index_into_an_output_and_keep_the_dependency() -> None:
    env = Env(outputs={"a": ["x", "y"]})
    ref = NodeRef("a")[1]
    assert isinstance(ref, ItemRef)
    assert ref.resolve(env) == "y"
    assert ref.deps() == {"a"}


def test_an_input_ref_is_rebound_by_a_child_environment() -> None:
    """What makes a `Subgraph`'s `wf.input` mean the subgraph's input."""

    env = Env(input="outer", outputs={})
    assert InputRef().resolve(env) == "outer"
    assert InputRef().resolve(env.child("inner")) == "inner"
    env.child("inner").set("n", 1)
    assert env.outputs["n"] == 1  # the output map is shared, the input is not


def test_a_missing_output_names_the_node() -> None:
    with pytest.raises(KeyError, match="'a'"):
        NodeRef("a").resolve(Env())


def test_remap_renames_node_ids_and_leaves_the_input_alone() -> None:
    ref = SeqRef((NodeRef("a"), InputRef(), NodeRef("a")[0]))
    renamed = ref.remap(lambda node_id: f"sub/{node_id}")
    assert renamed.deps() == {"sub/a"}
    assert renamed.resolve(Env(input="i", outputs={"sub/a": ["z"]})) == [["z"], "i", "z"]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def test_topological_order_follows_the_references() -> None:
    wf = Workflow("w")
    a = wf.func("a", echo)
    b = wf.func("b", echo, input=a)
    wf.func("c", echo, input=b)
    assert wf.compile().order == ("a", "b", "c")


def test_declaration_order_breaks_ties() -> None:
    """The same graph must always produce the same sequence."""

    wf = Workflow("w")
    wf.func("first", echo)
    wf.func("second", echo)
    assert wf.compile().order == ("first", "second")


def test_a_cycle_is_a_configuration_error() -> None:
    wf = Workflow("w")
    wf.func("a", echo, input=NodeRef("b"))
    wf.func("b", echo, input=NodeRef("a"))
    with pytest.raises(ConfigurationError, match="cycle"):
        wf.compile()


def test_an_undefined_reference_is_a_configuration_error() -> None:
    wf = Workflow("w")
    wf.func("a", echo, input=NodeRef("nope"))
    with pytest.raises(ConfigurationError):
        wf.compile()


def test_a_reference_to_a_fan_out_branch_orders_after_the_fan_out() -> None:
    """A branch does not run on its own, so depending on one means depending on it."""

    wf = Workflow("w")
    outs = wf.fan_out("models", {"a": ModelCall("m")})
    wf.func("after", echo, input=outs.branch("a"))
    assert wf.compile().order == ("models", "after")


def test_a_state_that_cannot_round_trip_fails_at_build_time() -> None:
    """Spec C-3 stage 1: before the run starts, before the money."""

    wf = Workflow("w")
    wf.node("bad", BadNode())
    with pytest.raises(ConfigurationError, match="does not round-trip"):
        wf.compile()


def test_an_agent_id_may_not_collide_with_a_node_id() -> None:
    """They share one quiescence map; the agent's exit would delete the node's entry."""

    from azalabscode import AgentNode, AgentSpec

    wf = Workflow("w")
    wf.node("main", AgentNode(AgentSpec(name="main", model="m")))
    with pytest.raises(ConfigurationError, match="also"):
        wf.compile()


# ---------------------------------------------------------------------------
# graph_hash (spec C-2)
# ---------------------------------------------------------------------------


def _fusion(analyst: str = "m/analyst", prompt: str = "be useful") -> Workflow:
    wf = Workflow("fusion")
    outs = wf.fan_out("models", {"a": ModelCall("m/a", system_prompt=prompt)})
    joined = wf.gather("join", outs)
    wf.node("analyze", ModelCall(analyst), input=joined)
    return wf


def test_graph_hash_is_stable_across_builds() -> None:
    assert _fusion().graph_hash() == _fusion().graph_hash()


def test_graph_hash_ignores_prompts_and_models() -> None:
    """Editing a system prompt must not invalidate a saved session (plan.md)."""

    assert _fusion().graph_hash() == _fusion(analyst="m/other", prompt="different").graph_hash()


def test_graph_hash_changes_when_a_node_changes_type() -> None:
    """The case where restoring the saved state would be actively wrong."""

    other = Workflow("fusion")
    outs = other.fan_out("models", {"a": ModelCall("m/a")})
    joined = other.gather("join", outs)
    other.func("analyze", echo, input=joined)
    assert other.graph_hash() != _fusion().graph_hash()


def test_graph_hash_changes_when_a_node_is_added() -> None:
    grown = _fusion()
    grown.func("extra", echo, input=NodeRef("analyze"))
    assert grown.graph_hash() != _fusion().graph_hash()


def test_func_identity_is_part_of_the_hash() -> None:
    """Swapping the body under a stable node id is drift worth seeing."""

    one, two = Workflow("w"), Workflow("w")
    one.func("f", echo)
    two.func("f", with_ctx)
    assert one.graph_hash() != two.graph_hash()


# ---------------------------------------------------------------------------
# accounts_for -- dynamic children (spec C-2, M5's reading)
# ---------------------------------------------------------------------------


def test_a_static_id_accounts_for_itself() -> None:
    graph = _fusion().compile()
    assert graph.accounts_for("models/a")
    assert not graph.accounts_for("nope")


def test_a_map_accounts_for_its_runtime_children() -> None:
    wf = Workflow("w")
    wf.map("each", Func(echo), over=[1, 2])
    graph = wf.compile()
    assert graph.accounts_for("each/0")
    assert graph.accounts_for("each/17")


def test_a_non_dynamic_node_does_not_account_for_a_child_id() -> None:
    """Otherwise a deleted fan-out branch would read as accounted for."""

    graph = _fusion().compile()
    assert not graph.accounts_for("models/deleted")


def test_an_agent_node_accounts_for_its_subagent_ids() -> None:
    from azalabscode import AgentNode, AgentSpec

    wf = Workflow("w")
    wf.node("writer", AgentNode(AgentSpec(name="scribe", model="m")))
    assert wf.compile().accounts_for("writer/agent/0")


# ---------------------------------------------------------------------------
# Func arity
# ---------------------------------------------------------------------------


def test_func_arity_is_inspected_once() -> None:
    assert Func(echo).takes_ctx is False
    assert Func(with_ctx).takes_ctx is True
    assert Func(echo, takes_ctx=True).takes_ctx is True


def test_gather_and_fan_out_declare_their_output_types() -> None:
    assert Gather().output_type == "list"
    assert FanOut().output_type == "list"
    assert ModelCall("m").output_type == "str"
