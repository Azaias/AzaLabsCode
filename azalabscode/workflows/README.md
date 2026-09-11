# `azalabscode.workflows` — the execution model

Code-first: you compose Python functions and node objects into a DAG. A single-agent
loop is the simple case. Fan-out, staged pipelines with joins, and agents that spawn or
delegate to other agents are the reason this layer is a graph and not just a loop.

```python
from azalabscode.workflows import AgentNode, AgentSpec, FanOut, Gather, Workflow

wf = Workflow("fusion")
answers = wf.fan_out("answers", over=models, body=lambda m: ModelCall("ask", model=m))
merged = wf.gather("merge", answers)
wf.agent("synthesis", AgentSpec(model="...", tools=[...]), prompt=merged)
graph = wf.compile()
```

Every builder call takes a **required `name`**. Node ids are `"/".join(path + [name])`.
They're never inferred from the call stack, because `inspect.stack()` breaks under `-O`,
in frozen builds, and inside comprehensions. `@` and `/` are reserved in a name.

## Node types

`AgentNode` (a full tool-using loop), `ModelCall` (a single completion), `FanOut` /
`Gather` (concurrency and the join that ends it), `Map` (one body per item), `Subgraph`
(composition), and `Func` (arbitrary async Python). Dynamic children get ids of the form
`<parent>/<index>`, drawn from a checkpointed monotonic counter that is bumped with no
`await` between the read and the write. So concurrent spawns take their ids in call
order, and a child created after a resume can't collide with a sibling from the saved
run.

**Cycles live inside nodes.** The graph itself is acyclic, and that's what makes every
node boundary a checkpoint boundary and every node output memoizable. Node outputs have
to be JSON-native.

## `NodeContext`

This is what a node body is handed, and the whole surface a workflow author programs
against: `ctx.model`, `ctx.tools`, `ctx.emit`, `ctx.checkpoint()`, `ctx.delegate(...)`
(blocking — it runs as the parent's own tool step), and `ctx.spawn(...)` (concurrent —
it returns an `AgentHandle`). Subagents get their own messages and tool results, inherit
the parent's permission mode, and report back into the parent's session, so the whole
tree serializes together.

## The agent loop

`AgentLoop` is the loop from spec §4.3: build a request, stream it, append the assistant
message, dispatch the tool calls in contiguous runs, materialize the results, and repeat
until the model stops calling tools. It also satisfies the `Delegator` protocol, which
is how the `delegate` tool spawns a child without `tools` ever having to import
`AgentSpec`.

`AgentSpec.max_tool_results_chars` (default 200 000) is applied at the top of each
iteration, so eight parallel results that each sit just under the per-tool cap still
can't blow up the next request. Decisions are memoized by `call_id`, which keeps them
byte-stable across a resume.

## Two invariants this layer owns

**The transcript invariant.** For every `ToolCallPart` there is exactly one later
`ToolResultMessage` with the same `call_id`, in call order — never in completion order.
Results are staged in `AgentState.pending_results` and then materialized by
`finalize_turn`, which fills in the holes. So there's never a window where the transcript
is invalid, not even mid-batch. `repair_transcript` is the resume-side half of this.

Take an interrupt with calls r1 done, r2 running, and r3 not yet started: r1's real
result survives, r2 is cancelled and returns `ToolError(kind="cancelled")`, and r3 is
backfilled with the same error. The next model call sees one complete, valid turn.

**The cancellation discipline.** All of it lives in `step.py`, as three rules:

1. Never wrap a step body in `except Exception`. `CancelledError` is a `BaseException`
   in 3.12, and even so it must not be caught there.
2. `cancel_reason is None` means the cancellation didn't come from us — re-raise it
   unconditionally. The absence of a reason is the authoritative signal for TaskGroup
   and shutdown cancellation.
3. Absorb a cancellation only once `current_task().uncancel()` has reconciled the count
   to zero. If an enclosing scope also cancelled us, the count stays above zero, and
   swallowing it would break `TaskGroup.__aexit__` and `asyncio.timeout.__aexit__`.

`should_absorb()` is the one and only implementation of rule 3. Don't write a second
one.

## What this layer must not do

`workflows` never imports `control`, and an import-linter contract enforces that. The
checkpoint seam is the `RunControl` protocol: the runner *declares* safe points, and
control decides whether to write and whether to park. Passing `control=None` gives you a
graph run with no run lifecycle around it — which is exactly what a script wants.
