"""The execution model: the graph, the agent loop, steps, and the transcript rule.

M2 landed three things and the order was not a preference: `step.py` (the
cancellation discipline) and `transcript.py` (the invariant) came first, because
every later scenario depends on both and retrofitting either into a working loop
means rewriting the loop.

M5 added the graph on top of them:

    builder.Workflow    the spec 6.3 builder; ids come from an explicit name
    graph.Graph         the compiled, flat, validated DAG plus `graph_hash`
    runner.Runner       the run body: memo, quiescence, safe points, failures
    context.NodeContext what a node is handed
    nodes/              AgentNode, ModelCall, FanOut, Gather, Map, Func, Subgraph

This layer never imports `control`. It talks to `azalabscode.contracts.RunControl`,
which `Controller` satisfies structurally.
"""

from azalabscode.workflows.agent_loop import AgentLoop, AgentResult, AgentSpec
from azalabscode.workflows.builder import Workflow
from azalabscode.workflows.context import NodeContext
from azalabscode.workflows.graph import (
    ConstRef,
    Env,
    FanOutRef,
    Graph,
    InputRef,
    ItemRef,
    MapRef,
    NodeEntry,
    NodeRef,
    Ref,
    SeqRef,
    as_graph,
    to_ref,
)
from azalabscode.workflows.handle import AgentHandle
from azalabscode.workflows.node import EmptyState, Node, NodeFailure
from azalabscode.workflows.nodes import (
    AgentNode,
    FanOut,
    Func,
    Gather,
    Map,
    ModelCall,
    Subgraph,
    render_prompt,
)
from azalabscode.workflows.runner import Runner
from azalabscode.workflows.state import AgentState
from azalabscode.workflows.step import (
    StepHandle,
    StepResult,
    run_inline_step,
    run_step,
    should_absorb,
)
from azalabscode.workflows.transcript import (
    TurnResults,
    cancelled_fill,
    finalize_turn,
    interrupted_fill,
    not_run_fill,
    repair_transcript,
)

__all__ = [
    "AgentHandle",
    "AgentLoop",
    "AgentNode",
    "AgentResult",
    "AgentSpec",
    "AgentState",
    "ConstRef",
    "EmptyState",
    "Env",
    "FanOut",
    "FanOutRef",
    "Func",
    "Gather",
    "Graph",
    "InputRef",
    "ItemRef",
    "Map",
    "MapRef",
    "ModelCall",
    "Node",
    "NodeContext",
    "NodeEntry",
    "NodeFailure",
    "NodeRef",
    "Ref",
    "Runner",
    "SeqRef",
    "StepHandle",
    "StepResult",
    "Subgraph",
    "TurnResults",
    "Workflow",
    "as_graph",
    "cancelled_fill",
    "finalize_turn",
    "interrupted_fill",
    "not_run_fill",
    "render_prompt",
    "repair_transcript",
    "run_inline_step",
    "run_step",
    "should_absorb",
    "to_ref",
]
