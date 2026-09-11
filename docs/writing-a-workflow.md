# Writing a workflow

A workflow is a DAG of nodes. You build it with a `Workflow` builder and compile it to a
`Graph`. You write it against the public API — everything exported from
`azalabscode/__init__.py` and from each layer's own `__init__.py` — and an import-linter
contract proves the three reference workflows never reach past that line.

Every example on this page is run by `tests/test_docs_examples.py`, so none of them is
just a sketch.

## 1. A two-stage pipeline

The smallest thing that isn't a single call: draft, then polish. `FakeProvider` stands
in for a real provider, so the example needs no key.

<!-- runnable -->

```python
import asyncio

from azalabscode import Controller, FakeProvider, ModelCall, PermissionMode, ScriptedTurn, Workflow


def build() -> Workflow:
    wf = Workflow("summarize", input="What does this repository do?")
    draft = wf.node(
        "draft",
        ModelCall("anthropic/claude-haiku-4.5", provider=FakeProvider([ScriptedTurn(text="a draft")])),
    )
    wf.node(
        "polish",
        ModelCall(
            "anthropic/claude-haiku-4.5",
            provider=FakeProvider([ScriptedTurn(text="a polished draft")]),
        ),
        input=draft,
    )
    return wf


async def main() -> str:
    controller = Controller(permission_mode=PermissionMode.AUTO)
    controller.bind_workflow(build())
    return str(await controller.run(timeout=10))


assert asyncio.run(main()) == "a polished draft"
```

Three things here are worth calling out.

**Every builder call takes a required `name`.** Node ids are built from it, and they're
what a checkpoint refers to. They're never inferred from the call stack, because
`inspect.stack()` breaks under `-O`, in frozen builds, and inside comprehensions.

**`wf.node(...)` returns a `NodeRef`.** Passing that ref as another node's `input` is how
data flows between nodes. `wf.input` is the workflow's own input, and `wf.ref("draft")`
names a node by id if you didn't keep its ref.

**The run's output is the last node's output,** unless you declare one explicitly with
`wf.output(ref)`.

## 2. An agent with tools

`AgentNode` runs a full tool-using loop. It needs a provider and a `ToolDispatcher`; the
dispatcher needs a `PermissionGate`, and the controller has one.

<!-- runnable -->

```python
import asyncio
import tempfile
from pathlib import Path

from azalabscode import (
    AgentSpec,
    Controller,
    FakeProvider,
    PermissionMode,
    ToolContext,
    ToolDispatcher,
    Workflow,
    default_registry,
)
from azalabscode.providers import tool_call, turn
from azalabscode.workflows import AgentNode


def build(dispatcher: ToolDispatcher) -> Workflow:
    provider = FakeProvider(
        [
            turn(tool_calls=[tool_call("read_file", {"path": "notes.txt"})]),
            turn(text="The note says hello."),
        ]
    )
    wf = Workflow("read-a-file", input="What does notes.txt say?", dispatcher=dispatcher)
    wf.node(
        "loop",
        AgentNode(
            AgentSpec(name="main", model="anthropic/claude-haiku-4.5", tools=["read_file"]),
            provider=provider,
            agent_id="main",
        ),
        input=wf.input,
    )
    return wf


async def main() -> str:
    workspace = Path(tempfile.mkdtemp())
    (workspace / "notes.txt").write_text("hello\n", encoding="utf-8")

    controller = Controller(permission_mode=PermissionMode.AUTO, main_agent="main")
    dispatcher = ToolDispatcher(
        default_registry(),
        gate=controller.permission_gate,
        context=ToolContext(workspace_root=workspace),
    )
    controller.bind_workflow(build(dispatcher))
    return str(await controller.run(timeout=20))


assert asyncio.run(main()) == "The note says hello."
```

`AgentSpec` is where the agent's shape lives: `model`, `system_prompt`, `tools`,
`max_turns`, `parallel_tool_calls`, `allow_delegate`, `subagents`, and
`max_tool_results_chars`. An agent's `agent_id` must not match any node id — `compile()`
refuses it — which is why the node above is `loop` while the agent is `main`.

**Building the dispatcher outside `build()` here is a simplification.** When
`Controller.load()` reloads a session, it rebuilds the graph from `(import_path, config)`
alone, with no caller around to hand it a dispatcher. The reference workflows solve this
with `workflows/tooling.py:ToolingAgentNode`, which builds its own toolset at run time
from `RunControl.permission_gate`. Copy that node if your workflow needs to survive a
reload.

## 3. Making it reloadable

A session records only `(import_path, config)` — no provider, no client, no closure — so
another process can rebuild the graph from a JSON document. That has two consequences:

```python
# workflows/mine/workflow.py

def build(config: MyConfig | dict) -> Workflow:
    cfg = config if isinstance(config, MyConfig) else MyConfig.model_validate(config)
    provider = OpenRouterProvider()          # constructed HERE, not passed in
    ...
```

and, at the call site:

```python
controller.bind_workflow(
    build(cfg),
    import_path="workflows.mine.workflow:build",
    config=cfg.model_dump(mode="json"),
    config_type="MyConfig",
)
```

A provider passed in from outside is a piece of the run that the saved file can't
describe, so the reload will either fail or quietly differ from the original.

`graph_hash` covers `(node_id, node_class, state_type, output_type)` and deliberately
leaves out prompts and config, so editing a system prompt doesn't invalidate a saved
session. A saved node id that's missing from the rebuilt graph is a hard
`GraphMismatchError`; extra ids, or a changed hash, raise a drift *warning* instead,
which you can promote to an error with `strict_graph_hash=True`.

## 4. Concurrency

`fan_out` takes a mapping of name to node and runs them concurrently; `gather` joins them
back up. `map` runs one body per item. `max_concurrency=0` means unbounded.

```python
wf.fan_out("models", {slug: ModelCall(model) for slug, model in branches}, max_concurrency=4)
joined = wf.gather("join", wf.ref("models"))
wf.node("analyze", ModelCall(analyst), input=joined)
```

Dynamic children get ids of the form `<parent>/<index>` from a checkpointed monotonic
counter, so a new child created during a resumed run can't collide with a saved sibling.
Prefer slugs to indices for fan-out branch names: an index shifts the moment a model is
added to the middle of the list, and then every session saved before that stops lining
up.

## 5. Subagents

From inside a node body, `ctx.delegate(spec_name, task)` blocks and returns the child's
result, while `ctx.spawn(spec_name, task)` returns an `AgentHandle` you can await later.
From inside an agent loop, the model calls the `delegate` tool, which resolves specs **by
name** from `Workflow(specs=...)`.

A child gets its own messages and tool results, inherits the parent's permission mode,
and reports back into the parent's session, so the whole tree serializes together. In
`manual` mode a child is limited to tools that need no approval — destructive work stays
with the main agent — and in `auto` that restriction lifts.

## 6. Checkpoints, pause, and interrupt

You get all of these for free. A node body can call `await ctx.checkpoint()` to declare
an extra safe point, but the runner already declares one when it enters and completes
every node, and `AgentLoop` declares one after every model call and every tool batch.

Two rules for a node body that does its own async work:

- **Never wrap a step body in `except Exception`.** `CancelledError` is a `BaseException`
  in 3.12, and catching it there breaks interrupt.
- **Keep node state JSON-native.** The harness serializes what it owns; keeping your own
  custom state serializable is up to you. A `SerializationError` at a safe point fails
  the run rather than letting it carry on unsaved.

## 7. Giving it a TUI

Subclass `HarnessApp` and compose the widgets that your graph's shape calls for:

```python
from azalabscode.tui import HarnessApp, RunStatusBar, SplitPanes, StagePipeline, StreamPane


class FusionApp(HarnessApp):
    def compose(self):
        yield StagePipeline(["models", "analyze", "synthesize"])
        yield SplitPanes([StreamPane(node_id=f"models/{slug}", tail=True) for slug in slugs])
        yield RunStatusBar()
```

The app subscribes to the bus, routes events by `(agent_id, node_id)`, mounts the
approval modal itself, and provides the standard bindings. A widget mounted after
`attach()` has to be registered with `register_consumer`. Panes that stream quickly
should be `tail=True`, since layout dominates the latency tail.

## 8. Running headless

Every reference workflow ships with `--headless` (R-U-7), and the pattern is small:

```python
async with JsonlRecorder(controller.bus, Path("events.jsonl")):
    answer = await controller.run(timeout=None)
```

In `manual` mode, a headless run needs an `ApprovalHandler` that can answer:
`StdinApprovalHandler` for a person, or `QueueApprovalHandler` / `CallbackApprovalHandler`
for a test. A headless `manual` run with no handler will block at the first destructive
call, with nothing on screen to explain why.

## Where to look next

- [Architecture](architecture.md) for the layer map and the five cross-layer seams.
- [Reference workflows](reference-workflows.md) for three complete examples.
- `azalabscode/workflows/README.md` for the invariants this layer owns.
