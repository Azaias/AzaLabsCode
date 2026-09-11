# Architecture

One diagram for the whole system, with the module map underneath it. Intent success
criterion 4 asks that "every layer can be explained in a single diagram, and the diagram
matches the code." The second half of that is why `tests/test_docs_architecture.py`
parses this file: every `azalabscode.*` name written here has to exist, every package in
the tree has to be named here, and the layer order below has to match the one
`import-linter` enforces in `pyproject.toml`. Adding a package to `azalabscode/` breaks
that test until this file is updated in the same commit — which is deliberate. A diagram
that nobody checks is a diagram that drifts.

## The diagram (R-X-5)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ UI          azalabscode.tui                                                │
│   HarnessApp ─ EventRouter ─ azalabscode.tui.widgets ─ TUIApprovalHandler  │
│   consumes: Event stream            calls: Controller                      │
└──────────────┬──────────────────────────────────────┬──────────────────────┘
               │ bus.subscribe()                      │ pause / resume / interrupt
               │ EventSink                            │ send / set_mode / save
               │                                      │ resolve_approval
┌──────────────┴──────────────────────────────────────┴──────────────────────┐
│ CONTROL     azalabscode.control                                            │
│   Controller ─ RunStateMachine ─ QuiescenceTracker ─ RuntimePermissionGate │
│   Session ─ Checkpointer ─ reconcile ─ atomic_write_bytes                  │
│   owns: the Session document        satisfies: RunControl, PermissionGate  │
└──────────────┬─────────────────────────────────────────────────────────────┘
               │ Runner.run(graph, env)      RunControl.safe_point(SafePoint)
┌──────────────┴─────────────────────────────────────────────────────────────┐
│ WORKFLOWS   azalabscode.workflows                                          │
│   Workflow → Graph ─ Runner ─ NodeContext ─ azalabscode.workflows.nodes    │
│   AgentLoop ─ run_step / should_absorb ─ finalize_turn                     │
│   uses: Provider, ToolDispatcher     satisfies: Delegator                  │
└──────────────┬────────────────────────────┬────────────────────────────────┘
               │ Provider.stream()          │ ToolDispatcher.call()
┌──────────────┴───────────────────┐  ┌─────┴──────────────────────────────────┐
│ PROVIDERS  azalabscode.providers │  │ TOOLS   azalabscode.tools              │
│   Provider protocol              │  │   Tool ─ ToolContext ─ ToolDispatcher  │
│   OpenRouterProvider             │  │   azalabscode.tools.builtin: the nine  │
│   FakeProvider ─ ModelsCache     │  │   azalabscode.tools.search_backends    │
│                                  │  │   consumes: PermissionGate, Delegator  │
└──────────────────────────────────┘  └────────────────────────────────────────┘
               │                                      │
┌──────────────┴──────────────────────────────────────┴──────────────────────┐
│ SEAMS       azalabscode.contracts                                          │
│   PermissionGate ─ RunControl ─ Delegator ─ ApprovalHandler ─ EventSink    │
└──────────────┬─────────────────────────────────────────────────────────────┘
┌──────────────┴─────────────────────────────────────────────────────────────┐
│ VOCABULARY  events → messages → permissions → toolio → content → errors    │
│             → cancellation | ids | runstate | schema | sync                │
└────────────────────────────────────────────────────────────────────────────┘
```

Arrows point downward only. `tools` and `providers` never import each other, or anything
above them; `workflows` never imports `control`; `tui` sees events and the `Controller`,
and nothing else. Where the data flow needs to point back up — the tool dispatcher asking
about permissions, the runner writing a checkpoint, the `delegate` tool spawning a
subagent — a protocol in `azalabscode.contracts` inverts the dependency.

**Vocabulary goes down, policy stays up.** Each leaf module holds the shared types; each
layer above it holds the decisions about them. `azalabscode.permissions` knows what an
`ApprovalRequest` *is*; `azalabscode.control` decides what to do with one.

### The layer order, as enforced

Contract 1 in `pyproject.toml` is a `layers` contract with `exhaustive = true`, so every
package in `azalabscode/` must appear in it. The list below is that same list, and the
test asserts the two are equal. Names on one line are siblings, and siblings may not
import each other.

<!-- layers -->

```
tui
control
workflows
providers | tools
contracts
events
messages
permissions
toolio
content
errors
cancellation | ids | runstate | schema | sync
```

## The cross-layer interfaces

Five protocols, all in `azalabscode/contracts.py`. Each one exists because the data has
to flow one way while the imports have to point the other. There is nothing else in that
module: no behavior, and no imports beyond the leaf tier.

| Protocol | Consumed by | Implemented by | Why it exists |
|---|---|---|---|
| `PermissionGate` | `azalabscode.tools.dispatcher` | `azalabscode.control.gate.RuntimePermissionGate` | The dispatcher must ask before running a destructive call. Calling `control` directly would be an upward import. `azalabscode.tools.gates` also ships `AllowAllGate`/`DenyAllGate`, which is what makes the tool layer usable with no controller at all. |
| `RunControl` | `azalabscode.workflows.runner`, `azalabscode.workflows.agent_loop` | `azalabscode.control.controller.Controller` | The runner declares safe points; control decides whether to write and whether to park. Passing `None` gives a graph run with no run lifecycle around it. |
| `Delegator` | `azalabscode.tools.builtin.delegate` | `azalabscode.workflows.agent_loop.AgentLoop` | A subagent is a workflow concept, but `delegate` is a tool. Specs resolve **by name**, so `tools` never learns what an `AgentSpec` is. |
| `ApprovalHandler` | `azalabscode.control.controller` | `azalabscode.tui.approval.TUIApprovalHandler`, `azalabscode.control.approval_handlers` (stdin, queue, callback, deny-all) | The UI seam for `manual` mode. A headless run swaps the implementation, not the control flow. |
| `EventSink` | `azalabscode.events` | `azalabscode.tui.routing.EventRouter`, `azalabscode.events.JsonlRecorder` | The observability seam. Anything that wants the stream subscribes; core never knows who is listening. |

Two more seams are structural rather than protocols.
`azalabscode.workflows.context.NodeContext` is what a node is handed, and it's the only
thing a workflow author programs against. `azalabscode.tools.context.ToolContext` is a
plain dataclass carrying only leaf types, so a remote tool adapter can be wrapped around
it later without touching core.

## The layers, module by module

### `azalabscode.tui` — presentation

`app` (`HarnessApp`), `routing` (`EventRouter`, `EventConsumer`), `bindings` (the
§8.1 key table), `approval` (`TUIApprovalHandler`).

`azalabscode.tui.widgets`: `base`, `transcript`, `stream_pane`, `tool_calls`,
`diff_view`, `agent_tree`, `stage_pipeline`, `split_panes`, `prompt_input`,
`approval_modal`, `status_bar`.

Textual lives here and only here. `import azalabscode` does not pull in Textual, and a
test asserts it. That's what keeps a headless run cheap, and what would make replacing
the TUI framework a swap rather than a rewrite.

### `azalabscode.control` — the run lifecycle

`controller` (`Controller`: pause, resume, interrupt, send, set_mode, save, load),
`state` (`RunStateMachine`, `IllegalTransition`), `quiescence` (`QuiescenceTracker`),
`gate` (`RuntimePermissionGate`), `session` (`Session`, `WorkflowRef`, `AgentState`,
`NodeRecord`, `ValueRef`), `checkpoint` (`Checkpointer`), `resume` (`reconcile`,
`ResumeReport`, `check_graph_drift`), `atomic` (`atomic_write_bytes`,
`sweep_temp_files`), `approval_handlers`.

Two invariants live here and nowhere else. The first: **PAUSED means a pause was
requested and the non-quiescent agent count is zero** — not `parked == active`, which
races on spawn. The second, at a safe point: **write first, park second, and park outside
the checkpoint lock.** A parked agent still holding that lock would deadlock every other
checkpoint, and then PAUSED would never be reached.

### `azalabscode.workflows` — the execution model

`graph` (`Graph`, `Ref` and its variants), `builder` (`Workflow`), `node` (`Node`),
`runner` (`Runner`), `context` (`NodeContext`), `state` (`AgentState`, `EmptyState`),
`agent_loop` (`AgentLoop`), `step` (`run_step`, `should_absorb`, `StepHandle`),
`transcript` (`finalize_turn`, `repair_transcript`), `handle` (`AgentHandle`).

`azalabscode.workflows.nodes`: `agent` (`AgentNode`), `model_call` (`ModelCall`),
`containers` (`FanOut`, `Gather`, `Map`, `Subgraph`), `func` (`Func`).

Cycles live inside nodes, so every node boundary is a checkpoint boundary. The transcript
invariant — for every `ToolCallPart`, exactly one later `ToolResultMessage` with the same
`call_id`, in call order and never completion order — is why results are staged in
`AgentState.pending_results` and materialized by `finalize_turn`, instead of being
appended as they finish. There's never a window in which the transcript is invalid.

Cancellation has three rules, all in `step`: never use `except Exception` around a step
body (`CancelledError` is a `BaseException`); a missing `cancel_reason` means the
cancellation wasn't ours, so re-raise it; and absorb a cancellation only after
`uncancel()` has reconciled the count to zero.

### `azalabscode.providers` — model access

`base` (`Provider`, `ModelRequest`, the `StreamEvent` union, `StreamAccumulator`),
`openrouter` (`OpenRouterProvider`), `testing` (`FakeProvider`, `Script`,
`ScriptedTurn`), `models_cache` (`ModelsCache`, `ModelInfo`).

`complete()` is a helper built on `stream()`, not a second path. Malformed tool-call JSON
becomes a `ToolCallPart` carrying a `parse_error`, never an exception. Retries fire only
before the first byte arrives. `provider_options` passes through verbatim and is opaque
to every other layer — which is how OpenRouter's specifics stay out of the other layers.

### `azalabscode.tools` — what the model can do

`base` (`Tool`, `ToolSet`), `context` (`ToolContext`, `ReadState`, `WorkspaceConfig`),
`dispatcher` (`ToolDispatcher`, `partition_runs`), `registry` (`default_registry`,
`BUILTIN_NAMES`), `budget` (`TurnBudget`, `apply_result_cap`), `gates`
(`AllowAllGate`, `DenyAllGate`, `RecordingGate`), `platform` (`spawn`, `kill_tree`,
`resolve_shell`), `fileio`.

`azalabscode.tools.builtin`: `read_file`, `write_file`, `edit_file`, `glob`, `grep`,
`shell`, `web_fetch`, `web_search`, `delegate`.
`azalabscode.tools.search_backends`: `base` (`SearchBackend`), `serper`.

The dispatcher owns everything a tool isn't trusted to do itself: schema validation,
`validate_params` before the approval prompt, the gate check, the timeout, retries,
result caps, the per-turn budget, contiguous-run batching, and turning any unanticipated
exception into a structured `ToolError`. Defaults fail closed: a tool is neither
concurrency-safe nor read-only unless it says so.

### The vocabulary tier

| Module | Holds |
|---|---|
| `azalabscode.events` | every event model, `EventBus`, `JsonlRecorder` |
| `azalabscode.messages` | `SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolResultMessage`, `Usage` |
| `azalabscode.permissions` | `PermissionMode`, `ApprovalPolicy`, `ApprovalRequest`, `Decision` |
| `azalabscode.toolio` | `ToolResult`, `ToolError`, `ToolErrorKind`, `ToolSchema`, `RetryPolicy` |
| `azalabscode.content` | `TextPart`, `ReasoningPart`, `ToolCallPart`, `ImagePart`, `FilePart` |
| `azalabscode.errors` | `HarnessError` and its subclasses |
| `azalabscode.cancellation` | `CancelReason`, `StepKind`, `StepOutcome` |
| `azalabscode.ids` | ULIDs, the id `NewType`s, `slugify_for_path` |
| `azalabscode.runstate` | `RunState` and the legal transitions |
| `azalabscode.schema` | the JSON-schema helpers a tool's params go through |
| `azalabscode.sync` | `run_sync`, the thin sync entry point |

`azalabscode.messages` sits above `azalabscode.toolio` because a `ToolResultMessage`
carries a `ToolResult`, and `toolio` sits above `content` because a `ToolResult` carries
content parts. Collapsing those three into one module — as the spec's §4.2 does — would
create a cycle; splitting them apart is spec delta 1.

## One turn, end to end

1. `Controller.start()` moves the run to `RUNNING` and drives `Runner.run`.
2. The runner enters a node, takes a `node_entered` safe point through `RunControl`, and
   calls the node body with a `NodeContext`.
3. `AgentLoop` builds a `ModelRequest` and iterates `Provider.stream()`, emitting one
   `ModelDelta` per fragment. The TUI's `StreamPane` batches these on a 33 ms timer.
4. The assistant message is appended, and an `after_model_call` safe point follows.
5. The tool calls are split into the longest possible contiguous runs of
   concurrency-safe calls. Each safe run goes concurrently under a semaphore, each unsafe
   call runs alone, and call order is preserved throughout. For each call, the dispatcher
   validates it and asks the `PermissionGate`; in `manual` mode, for a destructive call,
   the controller raises an `ApprovalRequest` and the agent parks at `waiting_approval`
   until the `ApprovalHandler` answers.
6. The results are staged, and then `finalize_turn` materializes the
   `ToolResultMessage`s in call order. An `after_tool_batch` safe point follows, and the
   loop repeats.
7. When there are no tool calls, the loop returns, the node completes, the runner takes a
   `node_completed` safe point, and the run reaches `COMPLETED`.

Every step in that list emits a typed event (R-X-3). Nothing in core is observable only
through a log line.

## Where each requirement is enforced

| What | Where |
|---|---|
| Downward-only imports, public-API-only reference workflows | six `import-linter` contracts in `pyproject.toml`, run from `tests/test_architecture.py` |
| This diagram matches the tree | `tests/test_docs_architecture.py` |
| Session round-trips | `tests/test_roundtrip.py` |
| Save → kill → load → resume | `tests/test_kill.py` with `tests/kill_child.py` |
| Quiescence and the state machine | `tests/test_quiescence_property.py` |
| The transcript invariant | `tests/test_transcript.py` |
| Streaming latency (R-U-4) | `tests/test_tui_perf.py` |
| Spec §9.3, against a real model | `tests/test_acceptance_real_model.py` over committed event logs |

## Reading further

- `docs/index.md` — what this is and how to start.
- `docs/writing-a-workflow.md` — the workflow author's path through the public API.
- `docs/reference-workflows.md` — what each of the three demonstrates.
- `intent/spec.md` — the governing requirements. `intent/plan.md` — the numbered deltas
  where this implementation deliberately departs from the spec.
