"""The built-in node types (R-W-3).

    AgentNode   model-to-tool loop; the cycle lives here so the graph stays a DAG
    ModelCall   one completion, no tools
    FanOut      N named children, concurrently, on one input
    Gather      join several references into a list
    Map         one node per element of a runtime list
    Func        an async Python function
    Subgraph    a nested workflow

A node type is a class with a `run(ctx, input)`; a *node* is an instance of one with
an id the builder assigned. Everything else -- restoring state, memoizing the output,
opening the quiescence entry, taking the safe points -- belongs to the runner, so a
custom node type is a subclass of `Node` and nothing more.
"""

from azalabscode.workflows.nodes.agent import AgentNode, AgentNodeState
from azalabscode.workflows.nodes.containers import (
    ChildErrorPolicy,
    FanOut,
    Gather,
    Map,
    Subgraph,
    run_children,
    unwrap_group,
)
from azalabscode.workflows.nodes.func import Func
from azalabscode.workflows.nodes.model_call import (
    ModelCall,
    ModelCallState,
    render_prompt,
)

__all__ = [
    "AgentNode",
    "AgentNodeState",
    "ChildErrorPolicy",
    "FanOut",
    "Func",
    "Gather",
    "Map",
    "ModelCall",
    "ModelCallState",
    "Subgraph",
    "render_prompt",
    "run_children",
    "unwrap_group",
]
