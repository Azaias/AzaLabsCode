# Spec: An owned, extensible agentic harness in Python

**Status:** Draft for engineering review
**Source:** `intent.md` (2026-09-04)
**Package name:** `azalabscode` (decided 2026-09-04; see §13)

This document turns the intent into (a) numbered requirements that can be tested and (b) a design that satisfies them. Where the intent contradicts itself, or where satisfying one requirement forces a weakening of another, the decision taken is recorded in §10 *Areas of concern* and cross-referenced from the requirement. Nothing in §10 is silently resolved elsewhere.

Reading order for a first pass: §1, §2, §4 (architecture), §10. Everything else is reference.

---

## 1. Goals, non-goals, and definitions

### 1.1 Goals (restated from intent)

- G1. A single-process, single-user, async Python 3.12+ harness for running agentic workflows against LLMs, with four independently replaceable layers: **providers**, **workflows**, **control**, **UI**.
- G2. Code-first workflows: Python classes and functions composed into graphs. The single-agent loop is the simple case; fan-out, staged pipelines with joins, and agents spawning agents are first-class.
- G3. Full run-lifecycle control: pause / resume / interrupt-with-injection, a two-mode permission model (`manual` / `auto`) switchable at any time, and complete session serialization to and from disk.
- G4. A composable Textual TUI that can render what the workflow's structure implies, with each workflow shipping its own UI from shared components.
- G5. Essential built-in tools (file read/write/edit, shell, web fetch/search) that are production-grade, not stubs.
- G6. Three reference workflows built purely on the public API, each with its own TUI, serving as the acceptance suite.

### 1.2 Non-goals for v1

RAG / long-term memory; MCP; web UI; broad tool library; tracing dashboards; fine-grained permissions (allowlists, "approve all of kind"); hosted/multi-tenant deployment; backwards-compatibility guarantees; plugin marketplace.

Non-goals are still constraints on design: the tool interface must not preclude MCP; the event model must be sufficient for a future dashboard; the permission gate must be extensible to finer policies without rewriting callers.

### 1.3 Definitions

| Term | Meaning |
|---|---|
| **Run** | One execution of one workflow, identified by `run_id`. Has a lifecycle state (§6.1). |
| **Session** | The serializable document that fully describes a run: workflow reference, config, run state, all agent transcripts, node states, pending approvals. What is saved to and loaded from disk. |
| **Workflow** | A graph definition: nodes, edges, configuration. Importable Python. Stateless as a definition; state lives in the run. |
| **Node** | A unit of execution in a workflow graph. Has an id, an input, an output, and a state. |
| **Agent** | A node (or something inside a node) that runs the model↔tool loop and owns a message list. Identified by `agent_id`, arranged in a tree rooted at `main`. |
| **Step** | One atomic unit inside an agent loop: one model call, or one tool call. The granularity of interrupt. |
| **Safe point** | A moment where the run's state is fully captured in the session document with no in-flight external effect. The granularity of pause and checkpoint. |
| **Checkpoint** | The act of writing the session document (in memory, and optionally to disk). Occurs at every safe point. |
| **Controller** | The control-layer object that owns a run and exposes pause/resume/interrupt/mode/approve. |
| **Event** | An immutable structured record emitted by core describing something that happened. The only channel from core to UI. |

---

## 2. Requirements

Numbered `R-<layer>-<n>`. Each has a verification method: **T** (automated test), **D** (demonstration in a reference workflow), **I** (inspection / review). Requirements marked ⚠ have a corresponding entry in §10.

### 2.1 Cross-cutting

| Id | Requirement | Verify |
|---|---|---|
| R-X-1 | Python ≥ 3.12. Async core; all I/O is `async`. A `azalabscode.sync` module offers `run_sync(...)` for scripts. | T |
| R-X-2 | Layers depend strictly downward: `ui → control → workflows → providers`, with `tools`, `messages`, and `events` as leaf modules importable by all. Enforced by an `import-linter` contract in CI. | T |
| R-X-3 | Every model call, tool call, node transition, agent spawn, run-state transition, approval request/resolution, checkpoint, and injected message emits a typed event (§6.5). No core behavior is observable only through logs. | T |
| R-X-4 | All core data structures are Pydantic v2 models with a `schema_version`. `Session.model_dump_json()` round-trips losslessly (`model_validate_json` produces an equal object). | T |
| R-X-5 | A single architecture diagram (`docs/architecture.md`) names every top-level module and every cross-layer interface. CI checks that every module in the diagram exists and every `azalabscode.*` package appears in the diagram. | T/I |
| R-X-6 | Public API is everything exported from `azalabscode/__init__.py` and the per-layer `__init__.py`s. Reference workflows import only from these. Enforced by import-linter. | T |
| R-X-7 | Structured, human-readable docs: a README per layer, docstrings on every public symbol, and a `docs/` tree built with mkdocs. Not a v1 gate for merge, but a v1 gate for "done." | I |

### 2.2 Providers

| Id | Requirement | Verify |
|---|---|---|
| R-P-1 | `Provider` is a protocol with a single required method: `stream(request: ModelRequest) -> AsyncIterator[StreamEvent]`. Non-streaming is a helper built on it, not a second entry point. | T |
| R-P-2 | `ModelRequest` and `StreamEvent` contain no OpenRouter-specific fields. Provider-specific parameters travel in `ModelRequest.provider_options: dict[str, Any]`, treated as opaque by every other layer. ⚠C-9 | I/T |
| R-P-3 | `OpenRouterProvider` supports: streaming text; streaming reasoning/thinking content where the model emits it; tool-call assembly from argument deltas; final usage (prompt/completion tokens, and cost when the API returns it); model-declared finish reason. | T (recorded fixtures) |
| R-P-4 | Malformed tool-call JSON from the model is surfaced as `ToolCallPart(parse_error=...)`, never raised. The agent loop returns it to the model as a structured tool error. | T |
| R-P-5 | Transport failures before the first byte of response (429, 5xx, connect/read timeout) are retried with exponential backoff and jitter, bounded (default 3 attempts). Failures after the first byte are **not** retried by the provider; the stream ends with `StreamEvent.Error` and the caller decides. ⚠C-8 | T |
| R-P-6 | Cancelling the async iterator closes the HTTP connection promptly (≤ 1 s). Partial usage after cancellation is reported as `usage=None` with `cancelled=True`; the provider does not guess. | T |
| R-P-7 | Model capability metadata (context length, tool support, pricing) is fetched from `/models`, cached on disk with TTL, and exposed as `Provider.model_info(model_id)`. Missing metadata degrades to "unknown," never blocks a call. | T |
| R-P-8 | A `FakeProvider` in `azalabscode.providers.testing` replays scripted `StreamEvent` sequences (including delays and mid-stream errors) for deterministic tests. Shipped as public API because workflow authors need it too. | T |

### 2.3 Tools

| Id | Requirement | Verify |
|---|---|---|
| R-T-1 | `Tool` defines: `name`, `description`, a Pydantic `Params` model (source of the JSON schema), `approval: ApprovalPolicy`, `timeout: float`, `retry: RetryPolicy`, and `async run(params, ctx) -> ToolResult`. | T |
| R-T-2 | `ToolResult` is always returned, never raised, for any failure the tool can anticipate (bad path, non-zero exit, timeout, HTTP error, cancellation). It carries `ok: bool`, `content: list[ContentPart]`, `error: ToolError | None` (with `kind`, `message`, `retryable`), `duration_ms`, and `meta: dict`. Unanticipated exceptions inside `run` are caught by the dispatcher and converted to `ToolError(kind="internal")` with the traceback in `meta`. | T |
| R-T-3 | `ApprovalPolicy` is one of `never`, `always`, or a callable `(params) -> bool` evaluated per call. The tool decides *whether* it is destructive; the control layer decides *what to do about it*. | T |
| R-T-4 | Retries default to **0** for any tool whose `ApprovalPolicy` is not `never`. Non-idempotent tools (`shell`, `write_file`, `edit_file`) may not be configured with retries > 0 without an explicit `unsafe_allow_retry=True`. ⚠C-7 | T |
| R-T-5 | Timeouts are enforced by the dispatcher (not trusted to the tool). On timeout the tool's task is cancelled; `shell` additionally kills the process group. Result is `ToolError(kind="timeout")`. | T |
| R-T-6 | Tool output is bounded. Default cap 50 KB of text per result; overflow keeps head and tail with a marker stating how much was elided and where the full output was spilled on disk (`session_dir/tool_output/<call_id>`). | T |
| R-T-7 | File tools operate relative to a `workspace_root` in `ToolContext`. Paths resolving outside the root are rejected with `ToolError(kind="permission")` unless the workspace is configured `unrestricted=True`. Symlinks are resolved before the check. | T |
| R-T-8 | Built-in tools: `read_file`, `write_file`, `edit_file`, `shell`, `web_fetch`, `web_search`, plus `glob` and `grep`. The last two are added beyond the intent's list; see ⚠C-11. Detailed contracts in §7. | T/D |
| R-T-9 | Every built-in tool description is a versioned string in the tool's module, reviewed like code, and covered by a test that renders the full tool schema and snapshots it (so accidental description edits show up in diffs). | T |
| R-T-10 | The `Tool` interface has no dependency on the provider or workflow layers, and `ToolContext` is a plain dataclass, so an MCP adapter can later wrap remote tools without core changes. | I |

### 2.4 Workflows

| Id | Requirement | Verify |
|---|---|---|
| R-W-1 | A workflow is a Python object created by an importable factory `def build(config: BaseModel) -> Workflow`. The session stores `(import_path, config)`; it never pickles code. ⚠C-2 | T |
| R-W-2 | `Workflow` is a DAG of `Node`s with typed inputs/outputs. Cycles are expressed inside a node (the agent loop is a node), not as graph edges. This keeps the graph checkpointable per node. | T |
| R-W-3 | Built-in node types: `AgentNode` (model↔tool loop), `ModelCall` (one call, no tools), `FanOut` (run N children concurrently with the same input), `Gather` (join), `Map` (fan-out over a list), `Func` (async Python function), `Subgraph`. | T |
| R-W-4 | `AgentNode` supports subagents: `await ctx.delegate(AgentSpec, task) -> AgentResult` and `ctx.spawn(...) -> Handle` for concurrent children. Children get their own message list, a filtered toolset (§6.4), and are registered in the agent tree so the session serializes them. | T/D |
| R-W-5 | Node state is serializable. A node declares `State: type[BaseModel]`; the runner stores/restores it. A node that cannot serialize raises `SerializationError(node_id, field_path, reason)` at the checkpoint, and the run transitions to `FAILED` with that error rather than silently continuing unsaved. ⚠C-3 | T |
| R-W-6 | Node outputs are memoized in the session by `(node_id, attempt)`. On resume, completed nodes are not re-executed; incomplete ones restart from their last checkpoint. ⚠C-1 | T |
| R-W-7 | Concurrency is structured (`asyncio.TaskGroup`). A failing node fails the run unless its parent node declares `on_child_error="continue"`, in which case the failure becomes the child's output (`NodeFailure`). | T |
| R-W-8 | The single-agent loop, expressed via `AgentNode`, is ≤ 20 lines of user code including tool registration and provider setup. | D |

### 2.5 Control

| Id | Requirement | Verify |
|---|---|---|
| R-C-1 | `Controller` exposes: `start()`, `pause()`, `resume()`, `interrupt(message=None, target=None)`, `set_permission_mode(mode)`, `resolve_approval(request_id, decision)`, `save(path)`, and class method `load(path) -> Controller`. All are `async` and idempotent where meaningful. | T |
| R-C-2 | Run states: `CREATED, RUNNING, PAUSING, PAUSED, WAITING_APPROVAL, INTERRUPTING, COMPLETED, FAILED, CANCELLED`. Transitions in §6.1. Every transition emits `RunStateChanged`. | T |
| R-C-3 | `pause()` requests a halt; the run reaches `PAUSED` at the next safe point of **every** active agent. In-flight model calls and tool calls run to completion (bounded by their own timeouts) before the pause takes effect. `pause(hard=True)` cancels in-flight steps instead (semantics of interrupt without injection). ⚠C-1 | T |
| R-C-4 | `interrupt(message, target)` cancels the current step of the target agent (default: `main`), records the cancellation, appends `message` (if given) as a `UserMessage` to that agent's transcript, and resumes that agent at its next model call. Partial model output is kept as an `AssistantMessage(cancelled=True)` when `AgentSpec.keep_cancelled_output` is true (default); any incomplete `ToolCallPart`s in it are dropped so the transcript never contains a tool call without a result. When false, the partial is discarded and only a `ModelCallCancelled` event records it. Cancelled tool calls produce `ToolError(kind="cancelled")` returned to the model. If the run was `PAUSED`, it stays paused after injection. ⚠C-4 | T/D |
| R-C-5 | Permission modes: `manual`, `auto`. Mode is a property of the run, inherited by all agents in the tree. Switchable at any time via `set_permission_mode`. Switching to `auto` resolves all pending approval requests as approved; switching to `manual` has no effect on already-approved in-flight calls. ⚠C-5 | T |
| R-C-6 | In `manual` mode, a tool call whose `ApprovalPolicy` evaluates true for the given params produces an `ApprovalRequest` (tool name, params, agent_id, call_id, a human-readable summary, and for `edit_file`/`write_file` a unified diff). The agent's step blocks until resolved. Resolutions: `approve`, `deny(reason)`. Denial returns `ToolError(kind="denied", message=reason)` to the model. | T/D |
| R-C-7 | Only the `main` agent may raise approval requests. In `manual` mode, subagents receive a toolset filtered to tools whose policy is `never`; a subagent's model attempting an unavailable tool receives a tool error naming the restriction. In `auto` mode subagents receive the full toolset. ⚠C-6 | T |
| R-C-8 | A `Controller` in `manual` mode must have an `ApprovalHandler` registered before `start()`; otherwise `start()` raises `ConfigurationError`. Built-in handlers: `TUIApprovalHandler` (wired by the UI layer), `StdinApprovalHandler`, `DenyAllHandler`, `QueueApprovalHandler` (for tests). ⚠C-5 | T |
| R-C-9 | A pending `ApprovalRequest` is part of the session. Save → kill → load restores the run in `WAITING_APPROVAL` with the same request, and resolution proceeds normally. | T |
| R-C-10 | `save(path)` writes the session atomically (temp file + rename) and is callable in any state. A run in `RUNNING` is checkpointed at its next safe point; the call awaits that. Auto-checkpointing to disk after every safe point is on by default when a `session_dir` is configured. | T |
| R-C-11 | `load(path)` reconstructs the workflow from `(import_path, config)`, restores all state, and returns a `Controller` in `PAUSED` (never auto-starts). `resume()` continues. If the workflow's `import_path` fails to import or its `config` fails validation, `load` raises with a clear message. | T |
| R-C-12 | The **kill test** passes: start a run under `FakeProvider`, pause, save, terminate the interpreter (subprocess test), load in a fresh interpreter, resume, and the run completes with the same final output as an uninterrupted run under the same scripted provider. | T |
| R-C-13 | A tool call in flight at the moment of process death is restored as `ToolError(kind="interrupted", message="process terminated during execution; effect unknown")` and returned to the model on resume. It is never re-executed automatically. ⚠C-1 | T |

### 2.6 UI / TUI

| Id | Requirement | Verify |
|---|---|---|
| R-U-1 | The UI layer consumes only the event stream and the `Controller` API. It has no access to node internals. | T (import-linter) |
| R-U-2 | `HarnessApp(App)` is the Textual base class: it subscribes to events, dispatches them to mounted widgets by `agent_id`/`node_id`, owns the `TUIApprovalHandler`, and provides default key bindings (§8.1). Workflow-specific apps subclass it and compose the layout. | T (Textual pilot) |
| R-U-3 | Shared widgets: `Transcript`, `StreamPane`, `ToolCallList`, `ToolCallDetail`, `ApprovalModal`, `DiffView`, `AgentTree`, `StagePipeline`, `SplitPanes`, `RunStatusBar`, `PromptInput`. Each widget documents which events it consumes. | I/T |
| R-U-4 | Streaming text renders incrementally with ≤ 100 ms latency from event to screen at ≥ 4 concurrent streams (fusion workflow with 4 models). Delta events are coalesced in the UI, not in core. | T (perf test) |
| R-U-5 | Subagent activity is visually distinct from the main agent (separate pane or tree node; different border/colour; agent id shown). | D |
| R-U-6 | An approval prompt can be rendered by any `HarnessApp` subclass without the workflow author writing UI code; the modal is mounted automatically by the base app when `ApprovalRequested` arrives, unless the subclass overrides `on_approval_requested`. | T/D |
| R-U-7 | The TUI is fully optional: every reference workflow also runs headless via a CLI flag (`--headless`, uses `StdinApprovalHandler` or `auto` mode). | T |

### 2.7 Reference workflows (acceptance)

| Id | Requirement | Verify |
|---|---|---|
| R-A-1 | **Coding agent** (`workflows/coding_agent`): interactive session in a working directory; all built-in tools; `delegate` tool for subagents; TUI with transcript, streamed output, diff view on edits, approval modal, subagent tree, status bar; keybindings for pause/resume/interrupt/mode/save. Completes at least three scripted real tasks (see §9.3) using only built-in tools. | D/T |
| R-A-2 | **Model fusion** (`workflows/fusion`): one prompt → N models concurrently → side-by-side panes → analysis stage → synthesis stage, each stage visible as it runs. Pause mid-fan-out, save, load, resume: completed model outputs are retained; only incomplete calls re-run. | D/T |
| R-A-3 | **Tool observability** (`workflows/inspector`): single agent; live table of every tool call with status, duration, and drill-down to params, result, error, and the raw event sequence. | D/T |
| R-A-4 | None of the three imports from a private module, patches core, or subclasses a core class marked `@final`. | T |

---

## 3. Dependencies

| Dependency | Role | Justification |
|---|---|---|
| `pydantic>=2.7` | Data model, JSON schema for tools, serialization | Non-negotiable; every layer's contract |
| `httpx[http2]` | Provider HTTP, `web_fetch` | Async, streaming, timeouts, well-maintained |
| `textual>=0.80` | TUI | Intent default; async-native, shares the loop |
| `anyio` | *Not* used for concurrency (see §4.4), but `anyio.Path`/`to_thread` are convenient. Optional. | Keep the loop plain `asyncio` to match Textual |
| `trafilatura` | HTML → readable text in `web_fetch` | Best-in-class extraction; falls back to `lxml` text if it fails |
| `typer` | CLI entry points | Small, typed |
| `rich` | Diff/syntax rendering (comes with Textual) | — |
| `pytest`, `pytest-asyncio`, `pytest-textual-snapshot`, `respx`, `import-linter`, `ruff`, `pyright` | Dev | Standard |

Search backend: Serper (`google.serper.dev`) by default, configurable via `SearchBackend` (see C-10). Additional adapters may be added later; only Serper ships in v1.

---

## 4. Architecture

### 4.1 The diagram (R-X-5)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ UI          azalabscode.tui                                                 │
│   HarnessApp ─ widgets ─ TUIApprovalHandler                             │
│   consumes: Event stream        calls: Controller API                   │
└──────────────┬──────────────────────────────────────┬───────────────────┘
               │ events (subscribe)                   │ pause/resume/interrupt/
               │                                      │ set_mode/resolve_approval
┌──────────────┴──────────────────────────────────────┴───────────────────┐
│ CONTROL     azalabscode.control                                             │
│   Controller ─ RunState machine ─ PermissionGate ─ Checkpointer         │
│   owns: Session document          drives: Runner                        │
└──────────────┬──────────────────────────────────────────────────────────┘
               │ Runner.run(workflow, session, hooks)
┌──────────────┴──────────────────────────────────────────────────────────┐
│ WORKFLOWS   azalabscode.workflows                                           │
│   Workflow (DAG) ─ Node types ─ AgentLoop ─ NodeContext ─ Runner        │
│   uses: Provider, ToolDispatcher   emits: Event                         │
└──────────────┬────────────────────────────┬─────────────────────────────┘
               │ Provider.stream()          │ ToolDispatcher.call()
┌──────────────┴───────────────┐  ┌─────────┴──────────────────────────────┐
│ PROVIDERS azalabscode.providers  │  │ TOOLS  azalabscode.tools                   │
│   Provider protocol          │  │   Tool, ToolResult, ToolContext        │
│   OpenRouterProvider         │  │   builtins: read/write/edit/glob/grep  │
│   FakeProvider               │  │             shell/web_fetch/web_search │
└──────────────────────────────┘  └────────────────────────────────────────┘
      leaf modules used by all layers:
      azalabscode.messages   azalabscode.events   azalabscode.errors
```

Arrows point downward only. `tools` and `providers` never import each other or anything above them. `events` and `messages` are pure data.

### 4.2 Package layout

```
azalabscode/
  __init__.py            # public API re-exports
  messages.py            # Message types, content parts
  events.py              # Event types, EventBus
  errors.py
  sync.py                # run_sync
  providers/
    __init__.py  base.py  openrouter.py  testing.py  models_cache.py
  tools/
    __init__.py  base.py  dispatcher.py  context.py
    builtin/  read_file.py write_file.py edit_file.py glob.py grep.py
              shell.py web_fetch.py web_search.py
    search_backends/  base.py serper.py
  workflows/
    __init__.py  graph.py  node.py  runner.py  context.py
    nodes/  agent.py model_call.py fanout.py gather.py map.py func.py subgraph.py
    agent_loop.py
  control/
    __init__.py  controller.py  state.py  permissions.py  checkpoint.py  session.py  approval_handlers.py
  tui/
    __init__.py  app.py  bindings.py
    widgets/  transcript.py stream_pane.py tool_calls.py approval.py diff.py
              agent_tree.py stages.py split.py status_bar.py prompt.py
workflows/                 # reference workflows, outside the package
  coding_agent/  fusion/  inspector/
docs/
tests/
```

### 4.3 Data flow for one agent turn

1. `AgentLoop` builds `ModelRequest` from the agent's messages and toolset; emits `ModelCallStarted`.
2. Iterates `Provider.stream()`; each `StreamEvent` becomes a `ModelDelta` event; text/reasoning/tool-call parts accumulate into an `AssistantMessage`.
3. On finish, appends the `AssistantMessage`; emits `ModelCallCompleted`; **safe point → checkpoint**.
4. For each `ToolCallPart` (concurrently if the model emitted several and all tools are marked `parallel_safe`; otherwise sequentially in order):
   a. `ToolDispatcher.call()` asks `PermissionGate.check()`. Gate may block on `ApprovalRequest` (run → `WAITING_APPROVAL`; **safe point**).
   b. On approval, dispatcher runs the tool under timeout; emits `ToolCallStarted/Completed/Failed`.
   c. Appends `ToolResultMessage`; **safe point → checkpoint**.
5. If the model produced no tool calls, the loop ends and the node returns the final `AssistantMessage`. Otherwise go to 1.

Pause is checked at every safe point. Interrupt cancels the task running step 2 or step 4b.

### 4.4 Concurrency model

- One `asyncio` event loop, shared with Textual (`App.run_async()`); the Controller is created inside the app or the app is given an existing Controller.
- The run is a tree of tasks under nested `asyncio.TaskGroup`s mirroring the node tree. Structured concurrency means no orphan tasks after a node completes or fails.
- Each agent step runs in its own task (`asyncio.create_task`, tracked by the agent's `StepHandle`) so it can be cancelled individually by interrupt without cancelling the agent's task.
- User-initiated cancellation is distinguished from shutdown by setting `StepHandle.cancel_reason` **before** calling `task.cancel()`; the step's `except CancelledError` handler inspects the reason. Never rely on the exception alone.
- Pause is a per-run `asyncio.Event` (`_run_gate`) awaited at every safe point. `PAUSING` → `PAUSED` when the count of agents parked at the gate equals the count of active agents.
- Events are published to an `EventBus` with per-subscriber bounded queues (default 10 000). On overflow, `ModelDelta` events are coalesced (adjacent deltas merged) and, if still overflowing, dropped with a `EventsDropped(count)` marker. Lifecycle events are never dropped; publishing blocks instead.

### 4.5 What lives where (ownership of the hard decisions)

| Concern | Layer | Not here |
|---|---|---|
| Whether a tool call is destructive | `tools` (`ApprovalPolicy`) | control |
| Whether to prompt, auto-run, or deny | `control` (`PermissionGate`) | tools, workflows |
| Rendering the prompt | `tui` (`ApprovalModal`) | control |
| Which tools a subagent gets | `control` (`PermissionGate.toolset_for(agent)`) | workflows |
| Retry of a model call | `providers` (before first byte only) | workflows |
| Retry of a tool call | `tools.dispatcher` (per-tool policy) | workflows |
| Where an injected message goes | `control` (target agent id) → `workflows` (appends) | tui |
| Checkpoint timing | `workflows.runner` (declares safe points) | control (writes them) |

---

## 5. Messages and providers (detail)

### 5.1 Message model (`azalabscode.messages`)

```python
class TextPart(BaseModel):        type: Literal["text"]; text: str
class ReasoningPart(BaseModel):   type: Literal["reasoning"]; text: str; signature: str | None
class ToolCallPart(BaseModel):    type: Literal["tool_call"]; call_id: str; name: str
                                  arguments: dict[str, Any] | None; raw_arguments: str
                                  parse_error: str | None
class ImagePart(BaseModel):       type: Literal["image"]; media_type: str; data_b64: str
class FilePart(BaseModel):        type: Literal["file"]; ...   # reserved, not populated in v1

Part = Annotated[TextPart | ReasoningPart | ToolCallPart | ImagePart | FilePart, Field(discriminator="type")]

class SystemMessage(BaseModel):     role: Literal["system"]; content: str
class UserMessage(BaseModel):       role: Literal["user"]; content: list[Part]; injected: bool = False
class AssistantMessage(BaseModel):  role: Literal["assistant"]; content: list[Part]; model: str
                                    usage: Usage | None; finish_reason: str | None; cancelled: bool = False
class ToolResultMessage(BaseModel): role: Literal["tool"]; call_id: str; result: ToolResult

Message = Annotated[..., Field(discriminator="role")]
```

Every message carries `id: str` (ULID) and `ts: datetime`. `AssistantMessage.cancelled=True` is stored when a model call is cancelled by interrupt and the loop is configured `keep_cancelled_output=True` (the default; see R-C-4). The partial message is retained in the transcript, marked, and sent to the model on the next call so it knows what it had produced before being cut off.

### 5.2 Provider protocol

```python
class ModelRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: list[ToolSchema] = []
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] = []
    reasoning: ReasoningConfig | None = None      # effort / budget, provider maps it
    provider_options: dict[str, Any] = {}         # opaque
    metadata: dict[str, str] = {}                 # run_id, agent_id, for logging

class StreamEvent:  # discriminated union
    TextDelta(text)
    ReasoningDelta(text)
    ToolCallStart(index, call_id, name)
    ToolCallDelta(index, arguments_delta)
    ToolCallEnd(index)
    Usage(prompt_tokens, completion_tokens, cached_tokens, cost_usd | None)
    Finish(reason)
    Error(error: ProviderError)   # terminal; kind: rate_limit|auth|server|network|model|unknown

class Provider(Protocol):
    name: str
    def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]: ...
    async def model_info(self, model: str) -> ModelInfo | None: ...
    async def aclose(self) -> None: ...
```

`azalabscode.providers.complete(provider, request) -> AssistantMessage` is the non-streaming helper (R-P-1).

### 5.3 OpenRouter specifics (contained in `openrouter.py`)

- Endpoint `POST /api/v1/chat/completions`, `stream: true`, `stream_options: {include_usage: true}`, `usage: {include: true}`.
- Headers: `Authorization`, `HTTP-Referer`, `X-Title` from config; API key from `OPENROUTER_API_KEY` or config.
- Message conversion: `ReasoningPart` → OpenRouter `reasoning` field on assistant messages where the model supports round-tripping; otherwise dropped with a debug event. `ToolCallPart` ↔ `tool_calls[]`. `ToolResultMessage` → `role: tool`.
- `provider_options` passes through verbatim into the request body (e.g. `{"provider": {"order": [...]}, "transforms": [...]}`).
- Tool-call assembly: buffer `arguments` deltas per `index`; on `ToolCallEnd` or stream finish, `json.loads`; on failure set `parse_error` and keep `raw_arguments`.
- Rate-limit headers are read and surfaced in `ProviderError.retry_after`.

---

## 6. Workflows and control (detail)

### 6.1 Run state machine

```
CREATED ──start()──▶ RUNNING
RUNNING ──pause()──▶ PAUSING ──(all agents at gate)──▶ PAUSED
RUNNING ──approval needed──▶ WAITING_APPROVAL ──resolve──▶ RUNNING
WAITING_APPROVAL ──pause()──▶ PAUSED (request retained)
PAUSED ──resume()──▶ RUNNING  (or WAITING_APPROVAL if a request is pending)
RUNNING|PAUSED|WAITING_APPROVAL ──interrupt()──▶ INTERRUPTING ──▶ previous state
RUNNING ──all nodes done──▶ COMPLETED
any ──unrecoverable error──▶ FAILED
any ──cancel()──▶ CANCELLED
```

`INTERRUPTING` is transient: the target step is cancelled, the injection is applied, and the run returns to whatever state it was in (so interrupting a paused run injects without resuming).

### 6.2 Session document

```python
class Session(BaseModel):
    schema_version: int = 1
    run_id: str
    created_at: datetime
    workflow: WorkflowRef            # import_path: str, config: dict
    run_state: RunState
    permission_mode: PermissionMode
    agents: dict[str, AgentState]    # agent_id -> messages, parent_id, spec, status, current_step
    nodes: dict[str, NodeRecord]     # node_id -> status, attempt, input, output, state(dict), started/finished
    pending_approvals: list[ApprovalRequest]
    inflight: list[InflightStep]     # steps that were running at last checkpoint (see R-C-13)
    event_seq: int                   # last emitted event sequence number
    custom: dict[str, Any] = {}      # workflow-level scratch, must be JSON-serializable
```

Serialized as JSON. Large tool outputs are spilled to `session_dir/tool_output/` and referenced by path; the session file stays small enough to diff.

`WorkflowRef.import_path` is `"package.module:build"`. On `load`, `build(config)` is called with the validated config model. The workflow author must ensure `build` is deterministic given the config — node ids must be stable across processes (§6.3).

### 6.3 Graph and node model

```python
class Node(Protocol[In, Out, S: BaseModel]):
    id: str                       # stable; assigned by the graph builder from the construction path, e.g. "fanout/0/model_call"
    State: type[S]
    async def run(self, ctx: NodeContext[S], input: In) -> Out: ...

class NodeContext:
    run_id, node_id, agent_id | None
    state: S                              # restored on resume
    provider: Provider
    tools: ToolDispatcher
    events: EventEmitter
    async def checkpoint(self) -> None    # declares a safe point; awaits pause gate
    async def delegate(self, spec: AgentSpec, task: str) -> AgentResult
    def spawn(self, spec: AgentSpec, task: str) -> AgentHandle
    permission_mode: PermissionMode        # read-only view
```

`Workflow` is built with a small builder:

```python
wf = Workflow("fusion")
outs = wf.fan_out("models", [ModelCall(m) for m in models], input=wf.input)
joined = wf.gather("join", outs)
analysis = wf.node("analyze", ModelCall(cfg.analyst_model), input=joined)
final = wf.node("synthesize", ModelCall(cfg.synth_model), input=(joined, analysis))
wf.output(final)
```

Node ids are derived from the builder call path, so the same `build(config)` yields the same ids in another process. Dynamic structure (spawning subagents, `Map` over runtime lists) gets ids of the form `<parent>/<index>`; the index is recorded in the session so resume re-associates children correctly.

### 6.4 Agent loop and subagents

`AgentNode(spec: AgentSpec)` where `AgentSpec` = system prompt, model, toolset names, max turns, `keep_cancelled_output: bool = True`, `parallel_tool_calls`. `AgentNode.State` holds `turn: int`, `messages: list[Message]`, `pending_tool_calls: list[call_id]`.

Subagents:
- `delegate(spec, task)` creates a child `AgentNode` with id `<parent_agent>/<n>`, seeds its messages with `spec.system_prompt` + `UserMessage(task)`, runs it to completion in the parent's task group, and returns `AgentResult(final_text, messages_ref, usage)`. The parent's loop is blocked meanwhile (it is the parent's tool step).
- `spawn` returns a handle; the parent may `await handle.result()` later or `handle.cancel()`. Multiple spawns run concurrently.
- The `delegate` **tool** (used by the coding agent) is a thin built-in wrapper that exposes `delegate` to the model with params `(task, tools: list[str] | None, model: str | None)`. It has `ApprovalPolicy = never` (spawning is not destructive; what the child does is governed separately, R-C-7).
- Children inherit the run's permission mode and the parent's `workspace_root`. In `manual` mode `PermissionGate.toolset_for(child)` strips tools whose policy is not `never`; the child's system prompt gets an appended note listing what it cannot do so the model doesn't waste turns.
- Interrupting a parent cancels its current step; if that step is a `delegate`, the child agent is cancelled too (structured concurrency handles this). Interrupting a child directly (`target="main/2"`) does not affect the parent.

### 6.5 Events

All events: `seq: int`, `ts: datetime`, `run_id`, `agent_id: str | None`, `node_id: str | None`.

| Event | Payload |
|---|---|
| `RunStateChanged` | `old, new, reason` |
| `NodeStarted / NodeCompleted / NodeFailed` | `attempt`, `output_summary` / `error` |
| `AgentSpawned / AgentFinished` | `parent_id, spec_summary` / `result_summary, usage` |
| `ModelCallStarted` | `call_id, model, message_count, tool_names` |
| `ModelDelta` | `call_id, part: TextDelta|ReasoningDelta|ToolCallDelta` |
| `ModelCallCompleted` | `call_id, usage, finish_reason, duration_ms` |
| `ModelCallFailed / ModelCallCancelled` | `call_id, error` / `call_id, reason` |
| `ToolCallRequested` | `call_id, tool, params` |
| `ApprovalRequested / ApprovalResolved` | `request` / `request_id, decision, by` |
| `ToolCallStarted / ToolCallCompleted / ToolCallFailed / ToolCallCancelled` | `call_id` + result/error/duration |
| `MessageInjected` | `agent_id, message_id` |
| `PermissionModeChanged` | `old, new, pending_resolved: int` |
| `Checkpoint` | `to_disk: bool, path` |
| `EventsDropped` | `count` |

Events are Pydantic models; the bus also offers `EventBus.record(path)` writing JSONL — the observability hook the intent asks for.

### 6.6 Permission gate

```python
class PermissionGate:
    mode: PermissionMode
    async def check(self, tool: Tool, params: BaseModel, agent_id: str, call_id: str) -> Decision
    def toolset_for(self, agent_id: str, requested: list[str]) -> list[Tool]
    async def set_mode(self, mode: PermissionMode) -> int   # returns number of pending requests auto-resolved
    async def resolve(self, request_id: str, decision: Decision) -> None
```

`check` in `manual` mode for a policy-true call from `main`: creates `ApprovalRequest`, stores it in `session.pending_approvals`, emits `ApprovalRequested`, calls `handler.request(req)` (which may return immediately for queue-based handlers or await the UI), and awaits `resolve`. From a non-main agent it returns `Decision.deny("subagents cannot use approval-gated tools in manual mode")` without prompting — but this path should be unreachable because `toolset_for` already removed the tool; it exists as defense in depth.

---

## 7. Built-in tools (contracts)

Common: all file tools accept absolute paths or paths relative to `workspace_root`; all return `ToolResult` per R-T-2; descriptions are written for the model (what it does, when to use it, when *not* to, common mistakes) and are the versioned artifact in R-T-9.

| Tool | Params | Approval | Timeout | Retry | Notes |
|---|---|---|---|---|---|
| `read_file` | `path`, `offset: int = 1`, `limit: int = 2000` (lines) | never | 10 s | 0 | Line-numbered output (`   12│text`), UTF-8 with fallback to latin-1 and a warning, binary detection → error, 2000-line window with "use offset" hint, image files returned as `ImagePart` if ≤ 5 MB. |
| `write_file` | `path`, `content` | always | 10 s | 0 | Creates parent dirs. Atomic (temp + `os.replace`). Result includes byte count and whether the file existed. Approval summary: full content if new, unified diff if overwriting. |
| `edit_file` | `path`, `old`, `new`, `replace_all: bool = False` | always | 10 s | 0 | `old` must match exactly once unless `replace_all`; 0 or >1 matches → error listing match count and nearest candidates (fuzzy whitespace hint). Result and approval summary carry a unified diff. Atomic write. |
| `glob` | `pattern`, `path: str = "."`, `limit: int = 500` | never | 10 s | 0 | Respects `.gitignore` if present. Results sorted by mtime desc. |
| `grep` | `pattern` (regex), `path = "."`, `glob: str | None`, `context: int = 0`, `limit: int = 200` | never | 30 s | 0 | Uses `rg` if on PATH, else Python fallback. Output `path:line:text`. |
| `shell` | `command`, `cwd: str | None`, `timeout: float = 120`, `env: dict = {}` | always | per-call, cap 600 s | 0 (hard-blocked) | `asyncio.create_subprocess_shell` with `start_new_session=True`; kill process group on timeout/cancel; stdout+stderr interleaved with markers; exit code in `meta`; non-zero exit is `ok=False, kind="exit_status"` so the model treats it as a failure but still sees output. Interactive commands are impossible; the description says so and suggests non-interactive flags. |
| `web_fetch` | `url`, `max_chars: int = 40000`, `raw: bool = False` | never | 30 s | 2 (network/5xx only) | Follows ≤ 5 redirects; blocks `file:`, private IP ranges, and localhost by default (`allow_private=False` in tool config); HTML → markdown-ish text via trafilatura; JSON/text passthrough; PDFs → text via `pypdf` if installed; size cap 10 MB. |
| `web_search` | `query`, `n: int = 10`, `recency: Literal[...] | None` | never | 20 s | 2 | Returns list of `{title, url, snippet, published}`. Default backend Serper; selectable via `ToolConfig.search_backend`. Requires `SERPER_API_KEY`. See C-10. |
| `delegate` | `task`, `tools: list[str] | None`, `model: str | None`, `max_turns: int | None` | never | inherits child | 0 | Only available when the agent's spec sets `allow_delegate=True`. Returns the child's final text and usage. |

All tools set `parallel_safe` (§4.3): `read_file`, `glob`, `grep`, `web_fetch`, `web_search`, `delegate` are `True`; `write_file`, `edit_file`, `shell` are `False`.

---

## 8. TUI (detail)

### 8.1 `HarnessApp` base

- Constructor takes a `Controller` (or a factory) and optional `session_dir`.
- Subscribes to the event bus on mount; routes each event to widgets registered for `(agent_id | "*", node_id | "*")`.
- Default bindings (overridable): `ctrl+p` pause/resume toggle; `escape` interrupt current step, focus `PromptInput` for optional injection (enter with empty input = interrupt without message; `ctrl+c` twice = cancel run); `ctrl+t` toggle permission mode; `ctrl+s` save session; `ctrl+o` open session; `ctrl+l` toggle event log pane; `y`/`n` in `ApprovalModal`.
- Mounts `ApprovalModal` automatically on `ApprovalRequested` (R-U-6). The modal shows tool name, a human summary, params (collapsible), diff for file edits, and the agent id.
- `RunStatusBar` always mounted: run state, permission mode, active agents count, tokens/cost so far, checkpoint indicator.

### 8.2 Widgets

| Widget | Consumes | Notes |
|---|---|---|
| `Transcript` | messages for one agent (`ModelCall*`, `ToolCall*`, `MessageInjected`) | Collapsible tool blocks; reasoning in dim; injected messages highlighted. |
| `StreamPane` | `ModelDelta`, `ModelCallCompleted` for one `call_id` or agent | Minimal; used N-up in fusion. Coalesces deltas at 30 fps. |
| `ToolCallList` / `ToolCallDetail` | `ToolCall*`, `ApprovalRequested/Resolved` | Table with status/duration; detail view shows params, result, error, timing, event trail. |
| `DiffView` | `ToolCallCompleted` for `edit_file`/`write_file`, `ApprovalRequested` | Rich-rendered unified diff with syntax highlight by extension. |
| `AgentTree` | `AgentSpawned/Finished`, per-agent status | Tree; selecting an agent focuses its `Transcript`. |
| `StagePipeline` | `NodeStarted/Completed/Failed` | Horizontal stage chevrons with state; click to focus that stage's pane. |
| `SplitPanes` | — | Layout helper for N-up. |
| `PromptInput` | — | Multi-line input; submits `send(text)` in normal mode, `interrupt(text)` after `escape`. |

### 8.3 Performance

Textual can drop frames under heavy `refresh()` load. `StreamPane` batches deltas via a 33 ms timer and appends to a `RichLog`/`Static` in one write. The perf test (R-U-4) runs 4 `FakeProvider` streams at 200 tokens/s each under `App.run_test()` and asserts frame time.

---

## 9. Reference workflows

### 9.1 Coding agent

- `build(config)`: `AgentNode(AgentSpec(model=config.model, tools=all_builtin + delegate, allow_delegate=True, parallel_tool_calls=True))`; workspace root = cwd.
- TUI: left `Transcript(main)`, right column `AgentTree` over `ToolCallList`; `DiffView` overlays on edits; `PromptInput` bottom; status bar.
- The system prompt is part of the workflow package, not core. It instructs the model to prefer `edit_file` over rewriting, `grep`/`glob` over `shell find`, and to delegate exploration to subagents when the task is broad.
- Subagent specs: `explore` (read-only tools, cheap model), `review` (read-only), `edit` (full tools; only useful in `auto` mode — see ⚠C-6).

### 9.2 Model fusion

- Graph as in §6.3 example. Config: `models: list[str]`, `analyst_model`, `synth_model`.
- TUI: top row `SplitPanes` of `StreamPane` per model; below, `StagePipeline` (models → analyze → synthesize) with the active stage's `StreamPane`.
- Demonstrates R-W-6: pause during fan-out, save, load, resume — panes for completed models are restored from session, incomplete ones re-stream.

### 9.3 Tool observability

- Single `AgentNode` with all tools; TUI is `Transcript` | `ToolCallList` with `ToolCallDetail` drill-down and the raw event JSONL viewer.
- Acceptance tasks for R-A-1/R-A-3 (run under a real model, recorded as fixtures for CI):
  1. "Add a `--verbose` flag to `cli.py` and update the README" in a small sample repo.
  2. "Find every call site of `deprecated_fn` and replace it with `new_fn`, then run the tests."
  3. "Fetch the changelog at `<url>` and summarize the breaking changes into `NOTES.md`."
  Pass = task completes using only built-in tools, with no tool errors of kind `internal`, and no `shell` invocation of `cat`, `sed -i`, `find`, or `grep` where a built-in exists (checked by scanning the event log).

---

## 10. Areas of concern

Each entry: the tension, the decision taken in this spec, and what the decision costs. These are flagged for explicit sign-off; changing any of them changes requirements above.

### C-1. "Resume completes as if uninterrupted" is not literally achievable

**Tension.** Success criterion 2 wants a killed-and-reloaded session to complete "as if uninterrupted." Model streams are non-deterministic and not resumable; a shell command or file write in flight at kill time cannot be checkpointed or safely re-run; a pause "at the next safe point" may arrive after minutes if a tool has a long timeout.

**Decision.** Define the criterion as: *the workflow reaches the same graph-level completion, retains every completed node output and every completed message, and re-issues only steps that had not reached a safe point.* Concretely: partial model output is discarded and the call re-issued (R-W-6); a tool in flight at process death becomes a structured `interrupted` error the model sees (R-C-13); pause waits for in-flight steps unless `hard=True` (R-C-3). The kill test (R-C-12) is run under `FakeProvider` so the "same final output" clause is checkable at all.

**Cost.** With a real model, resumed runs can diverge from what they would have produced. Users should expect "coherent continuation," not replay. This should be stated in user docs.

### C-2. "Code-first Python" vs. "all run state serializable from the first commit"

**Tension.** Closures, lambdas, and ad-hoc classes composed into graphs are not serializable. Pickling code is fragile and unsafe.

**Decision.** Serialize the workflow *by reference* (`import_path` + validated config), never by value (R-W-1). Node ids are derived from the builder call path so a rebuilt graph lines up with stored node state (§6.3). This means a workflow defined inline in a REPL or notebook can run but cannot be saved; `save()` raises `WorkflowNotImportable` for such cases.

**Cost.** Some friction for quick experiments; the fix is to put the workflow in a module. Also, if the workflow code changes between save and load, ids may shift and `load` will fail or misassign; v1 detects the former (missing/extra node ids → error) but cannot detect the latter in general. A `graph_hash` stored in the session and compared on load gives a warning, not a guarantee.

### C-3. Serialization boundary and failure timing

**Tension.** The intent says the harness "raises clearly if custom state cannot be serialized." Raising at checkpoint time — after a model call has already cost money and a tool has already had effects — is late.

**Decision.** Two-stage: (1) at `Workflow.build()` time, every node's declared `State` type is instantiated with defaults and round-tripped through JSON; failure is a `ConfigurationError` before the run starts. (2) At checkpoint time, actual state is serialized; failure is `SerializationError` and the run goes to `FAILED` (R-W-5). Stage 1 catches type-level problems (a `State` field typed `Any` holding a file handle at runtime would still slip through to stage 2).

**Cost.** Stage 2 failures lose the in-memory work since the previous checkpoint. Acceptable for v1 given stage 1 catches the common case.

### C-4. Interrupt semantics: "cancel and continue" vs. "drop to pause"

**Tension.** The intent's assumption is cancel-inject-continue; the alternative is that interrupt always pauses. The two differ in what happens with no injection and in multi-agent runs.

**Decision.** Interrupt does not change the run's coarse state: interrupting a `RUNNING` run cancels the step and continues; interrupting a `PAUSED` run injects and stays paused (R-C-4, §6.1). The TUI's `escape` flow lets the user type a message *before* the run continues, which in practice delivers the "drop to pause" feel without a separate mode. Default target is `main`; other agents by explicit id.

**Cost.** In a fan-out with no `main` agent (the fusion workflow), a targetless interrupt has nothing to cancel; it emits a warning event and does nothing. Fusion's TUI therefore maps `escape` to `cancel()` of the selected pane's node instead. This is workflow-specific behavior the base app cannot infer — the `HarnessApp` subclass must declare an `interrupt_target()` hook.

### C-5. Permission handling "for free" requires a UI, but the harness must also run headless

**Tension.** `manual` mode blocks on a human. With no UI attached, the run would hang silently.

**Decision.** `Controller.start()` refuses to run in `manual` mode without an `ApprovalHandler` (R-C-8). Headless scripts choose `auto`, `StdinApprovalHandler`, or `DenyAllHandler` explicitly. Also, switching to `auto` auto-approves pending requests (R-C-5), because the user's action expresses intent to stop being asked; the count resolved is reported in the event so the UI can show it.

**Cost.** The auto-approve-on-switch rule can execute a queued destructive call the user forgot about. A confirmation in the TUI (`ctrl+t` when requests are pending shows "N pending calls will run — confirm") mitigates it; the core rule stands.

### C-6. Subagent "parallel edit" in the coding agent vs. manual-mode restriction on subagents

**Tension.** The intent lists "a parallel edit" as a subagent use case *and* says that in `manual` mode subagents may only use non-destructive tools, with approvals coming only from `main`. These cannot both hold in `manual` mode.

**Decision.** Honor the restriction (R-C-7). In `manual` mode, an `edit` subagent cannot edit; the coding agent's system prompt tells the model that delegated edits are only possible in `auto` mode and that in `manual` mode it should delegate *exploration and review* and do edits itself. The acceptance tasks in §9.3 are run in both modes; in `manual` mode the "parallel edit" demonstration is out of scope.

**Cost.** The strongest multi-agent demonstration only exists in `auto` mode. The intent already labels the restriction a "v1 simplification"; a follow-up is to route child approval requests through `main`'s handler with the child id shown, which the event/permission design already supports (requests carry `agent_id`).

### C-7. "Timeouts and retries per tool" vs. non-idempotent tools

**Tension.** Retrying `shell` or a file write after a timeout can double-apply effects.

**Decision.** Retries default to 0 for anything approval-gated and are hard-blocked for `shell` (R-T-4). Retries are a property of the *dispatcher* policy and only fire on errors the tool marks `retryable=True`, which built-in destructive tools never do.

**Cost.** Transient failures in destructive tools bubble to the model, which may retry itself. That is the correct place for the decision.

### C-8. Mid-stream provider failure is not retried

**Tension.** A long response that fails at 90% is lost; retrying wastes tokens and the retry may differ.

**Decision.** Provider retries only before the first byte (R-P-5). The agent loop, on `StreamEvent.Error` after partial output, discards the partial and retries the whole call once (configurable `AgentSpec.stream_error_retries=1`), emitting `ModelCallFailed` for the first attempt so the UI shows what happened. This is distinct from a user interrupt: a provider failure is not something the model should be told about as "its own" prior output, so `keep_cancelled_output` does not apply here.

**Cost.** Some wasted spend on flaky connections. Alternative (keep partial output and ask the model to continue) is unreliable across models and is not attempted in v1.

### C-9. "Provider interface must not leak OpenRouter specifics" vs. real-world knobs

**Tension.** Users will want OpenRouter routing preferences, model-specific reasoning parameters, and per-provider tool-calling quirks. A perfectly abstract interface cannot express them; a leaky one defeats the abstraction.

**Decision.** One escape hatch, `provider_options: dict`, opaque to all other layers (R-P-2). Reasoning is abstracted (`ReasoningConfig` with `effort`/`max_tokens`) because it is common across vendors; everything else goes in the bag. Reference workflows may use the bag; the workflow layer never inspects it.

**Cost.** A workflow that uses OpenRouter-specific options is not portable to a second provider without edits — but that is the workflow's choice, and the core stays clean.

### C-10. `web_search` needs a vendor the intent does not budget for

**Tension.** "OpenRouter is the sole provider at launch," but no OpenRouter endpoint returns search results as a tool result. A first-class search tool needs a search API key (Brave, Tavily, Exa) or a self-hosted SearXNG.

**Decision (owner, 2026-09-04).** `web_search` is built against a `SearchBackend` protocol (`async search(query, n, recency) -> list[SearchHit]`). The default and only v1 adapter is **Serper**, keyed by `SERPER_API_KEY`; the backend is selectable in tool config (`search_backend: str = "serper"`, resolved by name so other adapters can be registered without core changes). The tool is registered only if the configured backend has its key; otherwise it is absent from the toolset and the coding agent's prompt omits it.

**Cost.** One more API key. Not avoidable if the success criterion "the built-in tools are the ones the model reaches for" is to include search. Serper returns Google results with a per-query price; the adapter surfaces the `credits` field in `ToolResult.meta` so cost is visible in the inspector.

### C-11. Scope expansion: `glob` and `grep`

**Tension.** The intent's essential toolset omits file search. Without it, a coding agent will use `shell` for `find`/`grep`/`cat`, which (a) violates success criterion 5's spirit ("not workarounds") and (b) forces an approval prompt in `manual` mode for every read-only exploration.

**Decision.** Add `glob` and `grep` as part of the "file" family (R-T-8). They are read-only, small, and make the `manual`-mode coding agent usable.

**Cost.** Two more tools to engineer well. Strongly recommended regardless.

### C-12. Pause fidelity in the UI

**Tension.** The intent wants pause to "halt at the next safe point." From the user's chair, pressing pause during a 5-minute `shell` command and seeing nothing happen looks like a bug.

**Decision.** `PAUSING` is a distinct, displayed state; the status bar shows "pausing — waiting for `shell` (2:14 remaining)" and offers `hard pause` (which is interrupt without injection). Core exposes both; the UI makes the tradeoff visible.

### C-13. Textual as the only presentation layer

**Tension.** Textual's rendering throughput under many concurrent streams is unproven at this scale, and its widget model is opinionated about layout.

**Decision.** Proceed with Textual, but with the perf test in R-U-4 as an early milestone (M4 below) so a switch, if needed, happens before the widget library is large. The UI layer's only inputs are events and the Controller API (R-U-1), so a replacement is a rewrite of `azalabscode.tui` alone.

---

## 11. Testing strategy

- **Unit:** every tool against `tmp_path` workspaces; provider against recorded SSE fixtures via `respx`; message round-trips; gate decisions in both modes; state machine transitions (property-based with `hypothesis` over random command sequences).
- **Control integration:** under `FakeProvider`, scripted scenarios for pause-during-stream, pause-during-tool, interrupt-during-stream, interrupt-during-tool, interrupt-with-injection, mode switch with pending approval, save/load in every state, and the subprocess kill test.
- **Workflow integration:** each reference workflow runs headless under `FakeProvider` to completion; fusion additionally with a mid-fan-out pause/save/load.
- **TUI:** Textual `App.run_test()` + snapshot tests for each widget; the 4-stream perf test.
- **Architecture:** `import-linter` contracts for layer direction and public-API-only imports from `workflows/`; diagram-vs-code check.
- **Acceptance (manual + recorded):** §9.3 tasks against a real model, event logs committed as fixtures, CI replays them through `FakeProvider`.

---

## 12. Milestones

| M | Deliverable | Exit test |
|---|---|---|
| M0 | Repo, layout, import-linter, `messages`, `events`, `FakeProvider`, `OpenRouterProvider` | Provider fixtures pass; a script streams a completion |
| M1 | Tools: all eight built-ins + dispatcher + context | Tool unit tests; schema snapshot |
| M2 | `AgentNode` + `AgentLoop` + `Controller` with pause/resume/interrupt/permissions (no disk) | Control integration scenarios pass in-memory |
| M3 | Session serialization, checkpointing, `save`/`load`, kill test | R-C-12 passes |
| M4 | `HarnessApp` + `Transcript`, `StreamPane`, `ApprovalModal`, `RunStatusBar`; perf test | R-U-4 passes — **go/no-go on Textual** |
| M5 | Graph builder, `FanOut/Gather/Map/Func/Subgraph`, subagents (`delegate`/`spawn`) | Fusion runs headless; subagent tests |
| M6 | Remaining widgets; three reference workflows with TUIs | R-A-1..4 |
| M7 | Docs, diagram check, acceptance fixtures | R-X-5, R-X-7, §9.3 |

Estimated effort is deliberately omitted; ordering is what matters. M4's go/no-go is the only milestone with a designed-in exit ramp.

---

## 13. Decision log

Recorded 2026-09-04 with the owner. These are settled; changing them reopens the referenced sections.

| # | Decision | Effect on spec |
|---|---|---|
| 1 | C-1 through C-13 accepted as written. | No change. |
| 2 | Search backend: **Serper** by default, configurable. | §3, §4.2, §7 `web_search`, C-10 updated. Only the Serper adapter ships in v1; the `SearchBackend` protocol and name-based registry keep it swappable. |
| 3 | Package name: **`azalabscode`**. | All module paths updated. The word "harness" in prose still refers to the system generically. |
| 4 | `keep_cancelled_output` defaults to **`True`**. | R-C-4, §5.1, §6.4 updated. Interrupted model output stays in the transcript, marked `cancelled=True`, with incomplete tool calls stripped, and is sent back to the model on the next turn. Users who want a cleaner context set it to `False` per `AgentSpec`. |

One consequence of decision 4 worth noting for the coding-agent prompt: the system prompt should tell the model that a `cancelled` assistant message is its own prior partial output that the user cut off, and that the injected user message after it takes precedence — otherwise some models resume the cut-off thought instead of following the new instruction.
