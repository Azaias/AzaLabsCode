# Implementation Plan: `azalabscode`

## Context

`intent/intent.md` and `intent/spec.md` describe an owned, extensible async agentic harness in Python — four replaceable layers (providers, workflows, control, UI), production-grade built-in tools, full pause/resume/interrupt/serialize lifecycle, and a Textual TUI expressive enough for non-chat workflows. The repo is currently empty apart from those two documents, a `.gitignore`, an empty Python 3.12.0 venv, and `claude-code-main/` (a TypeScript reference codebase, gitignored).

The motivation is ownership: every layer understood, the Python API shaped by the author, a UI expressive enough that custom workflows are usable rather than merely runnable. `spec.md` is the governing document; this plan implements it and records where it deviates.

### Decisions taken with the owner

- **D1.** Shell execution is cross-platform behind a Windows-first platform strategy layer. Windows is the dev/test platform; the POSIX path is written and smoke-tested.
- **D2.** Delivery is milestone-gated (M0…M7 from spec §12), handed back at each boundary with its exit test passing.
- **D3.** Where the spec is wrong or weaker than it could be, implement the corrected design and record it as a numbered spec delta, later folded into `spec.md` §13.
- **D4.** `OPENROUTER_API_KEY` is available; `SERPER_API_KEY` is not. `web_search` is built against the `SearchBackend` protocol with the Serper adapter written but unregistered until a key exists (spec §7 already makes registration key-conditional). Real-model acceptance runs §9.3 tasks 1 and 2; task 3 is deferred.
- **D5.** If the M4 perf test fails, the fallback is a Rich-based custom render loop, not a relaxed bar. This makes M4 a real branch point, so nothing below `tui/` may reference a Textual type.
- **D6.** The coding agent ships as an installed CLI entry point for daily use. M6 therefore also carries packaging, config and API-key resolution, and a real system prompt — which is what makes success criterion 5 measurable by using it rather than by reading an event log.

### Provenance

`claude-code-main/` is leaked proprietary TypeScript, used only as a design reference for which durable behaviors and edge cases matter. All Python code and all tool description strings here are written fresh; no prompt text or code is transliterated.

### Environment, confirmed

Python 3.12.0 venv (empty) with `WindowsProactorEventLoopPolicy` as default, `pwsh` 7.6.3 Core and Windows PowerShell 5.1 both present, `rg` on PATH, no `uv`, git on branch `dev` with no remote.

---

## Architecture

The spec's §4.1 diagram holds in shape, but its three-module leaf tier is too coarse — it forces two import cycles (see Deltas 1–3). The corrected tiering:

```
tui  →  control  →  workflows  →  providers | tools
                                      ↓
                                  contracts          (inversion protocols)
                                      ↓
     events → messages → permissions → toolio → content → {ids, errors, cancellation}
```

**Rule: vocabulary goes down, policy stays up.** Each leaf holds shared types; each upper layer holds the decisions about them.

### Leaf tier

| Module | Contents |
|---|---|
| `ids` | ULID, `RunId`/`AgentId`/`NodeId`/`CallId` NewTypes, `slugify_for_path()` |
| `errors` | `HarnessError` and subclasses (`ConfigurationError`, `SerializationError`, `WorkflowNotImportable`, `GraphMismatchError`, `CheckpointError`, `SaveTimeout`) |
| `cancellation` | `CancelReason`, `StepKind`, `StepOutcome`, `RECOVERABLE_REASONS` |
| `content` | `TextPart`, `ReasoningPart`, `ToolCallPart`, `ImagePart`, `FilePart` |
| `toolio` | `ToolResult`, `ToolError`, `ToolErrorKind`, `ToolSchema`, `RetryPolicy` |
| `permissions` | `PermissionMode`, `ApprovalPolicy`, `ApprovalRequest`, `Decision` |
| `messages` | `SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolResultMessage`, `Usage` |
| `events` | every event model, `EventBus`, `EventEmitter`, JSONL recorder |
| `contracts` | the four inversion protocols below |

### The four inversion protocols (`azalabscode/contracts.py`)

These are what let the import-linter contract pass while the spec's data flow stays intact.

- **`PermissionGate`** — `check(...) -> Decision`, `visible_tool_names(agent_id, requested)`. Implemented by `control.gate.RuntimePermissionGate`, consumed by `tools.dispatcher`. `tools` also ships `AllowAllGate`/`DenyAllGate` so the tool layer is usable standalone at M1, before `control` exists.
- **`RunControl`** — `safe_point(sp)`, `enter_agent`/`exit_agent`, `phase(agent_id, phase)`, `register_step(handle)`, `permission_mode`. Implemented by `Controller`, consumed by `workflows.runner`. This is the executable form of §4.5's "the runner declares safe points, control writes them."
- **`Delegator`** — `available_specs()`, `delegate(spec_name, task, ...)`. Implemented by `AgentLoop`, consumed by the `delegate` tool. Specs resolve **by name**, so `tools` never imports `AgentSpec`.
- **`ApprovalHandler`** / **`EventSink`** — the UI and observability seams.

`ToolContext` stays a plain dataclass carrying only leaf types (R-T-10), so an MCP adapter can wrap remote tools later without core changes.

### Package layout

Flat layout per spec §4.2, with the leaf tier expanded and `contracts.py` added:

```
azalabscode/
  __init__.py  ids.py  errors.py  cancellation.py  content.py  toolio.py
  permissions.py  messages.py  events.py  contracts.py  sync.py
  providers/  base.py openrouter.py testing.py models_cache.py
  tools/      base.py context.py dispatcher.py registry.py budget.py platform.py
              builtin/  read_file write_file edit_file glob grep shell
                        web_fetch web_search delegate
              search_backends/  base.py serper.py
  workflows/  graph.py builder.py node.py runner.py context.py
              agent_loop.py step.py transcript.py  nodes/
  control/    controller.py state.py gate.py quiescence.py checkpoint.py
              session.py resume.py approval_handlers.py atomic.py
  tui/        app.py bindings.py  widgets/
workflows/    coding_agent/ fusion/ inspector/     (outside the package)
docs/  tests/
```

Single `pyproject.toml` (hatchling) with ruff, pyright, pytest and import-linter config inline. Dependencies per spec §3.

### Enforcement

Five `import-linter` contracts in `pyproject.toml`: a `layers` contract with `exhaustive = true` (which also satisfies half of R-X-5 by forcing every `azalabscode.*` package to be named), a `forbidden` contract stopping `tools`/`providers` from seeing anything above them, a `forbidden` contract stopping `workflows` from importing `control`, an `independence` contract between `providers` and `tools`, and a `forbidden` contract with package-root wildcards proving the reference workflows import only public API (R-X-6, R-A-4). `exclude_type_checking_imports = false` — no `TYPE_CHECKING` escape hatches.

---

## Spec deltas

Implemented as described; each names the requirement it touches.

**Layering**

1. **The leaf tier is nine modules, not three** (R-X-2, §4.2). §4.2's single `messages.py` forces a `messages ↔ tools` cycle: `ToolResultMessage.result: ToolResult` (§5.1) against `ToolResult.content: list[ContentPart]` (R-T-2). The spec does not acknowledge this cycle and it blocks M0. Splitting out `content` and `toolio` below `messages` resolves it.
2. **Permission vocabulary moves to a leaf** (R-C-6, §4.2). `ApprovalRequest`, `Decision`, `PermissionMode`, `ApprovalPolicy` live in `azalabscode.permissions` so `events` can carry them (R-X-3). `control/gate.py` keeps only policy.
3. **`ToolDispatcher` depends on a `PermissionGate` protocol** (§4.3 step 4a), injected by `control`. The spec has `tools` calling `control.PermissionGate.check()` directly — an upward import that fails the contract the spec itself mandates.
4. **`RunControl` formalizes the checkpoint seam** (§4.5). `workflows` never imports `control`; `Controller` satisfies the protocol structurally.
5. **`delegate` resolves specs by name through a `Delegator` protocol** (§7), never importing `AgentSpec`.

**Tools**

6. **Contiguous-run dispatch** (§4.3 step 4). The spec's all-or-nothing rule serializes five reads because one `edit_file` sits among them. Instead, partition the turn's calls into maximal contiguous runs of concurrency-safe calls; each safe run executes concurrently under a semaphore (default 10), each unsafe call runs alone, run order preserved. Write-after-read ordering comes free.
7. **Concurrency-safety is per-call** (§7's `parallel_safe` table). `Tool.is_concurrency_safe(params) -> bool`, defaulting to a class attribute and failing closed on a parse error or a raising predicate.
8. **Per-tool result cap with an opt-out, plus a per-turn budget** (R-T-6). System ceiling stays 50 000 chars, but `Tool.max_result_size_chars` may be lower, or `math.inf` for `read_file` — spilling a file read to disk that the model then re-reads is circular. Overflow becomes a `<persisted-output>` block naming the size, the spill path and a 2 000-byte head preview, which beats a head+tail slice because the model can `read_file` the rest. Separately, `AgentSpec.max_tool_results_chars` (default 200 000) is applied at the top of each loop iteration so eight parallel results just under the per-tool cap cannot blow up the next request; decisions are memoized by `call_id` in the session so they are byte-stable across resume.
9. **`shell` gets an OS strategy layer** (R-T-5, §7). `start_new_session=True` and `killpg` are POSIX-only and the primary platform is Windows. `tools/platform.py` exposes `spawn()`/`kill_tree()`: POSIX uses `start_new_session` + `SIGTERM` then `SIGKILL`; Windows uses `CREATE_NEW_PROCESS_GROUP` + `taskkill /F /T /PID`, default shell `pwsh` falling back to `powershell`. The model-facing description is rendered per platform so the model knows which shell it is talking to.
10. **`shell` merges stdout and stderr into one append-mode file** (§7). Two pipes cannot be interleaved chronologically and need in-process draining. One append-only fd gives correct interleaving, takes the read loop off the hot path, and makes the over-cap case a `seek`. The tail is polled at ~1 Hz for progress events.
11. **`Tool.validate_params` is separate from `run`** (R-T-1). A pure pre-check returning a structured error *before* the approval prompt. Without it, `manual` mode asks the user to approve an edit that was always going to fail. "File not read yet", "string not unique", "path denied" live here.
12. **Three renderings of one result** (R-T-2, §8.2). `ToolResult.content` for the model, `ToolResult.display` (structured, e.g. diff hunks) for widgets, `ToolResult.meta` for telemetry. One run, three consumers, no widget re-parsing model text.
13. **Read-before-write state tracking** (§7). `ToolContext` carries a bounded LRU of `path -> (mtime, offset, limit)` set by `read_file` and cleared by writes. `write_file`/`edit_file` refuse a path never read, or read before an external modification, telling the model to re-read. This is the single highest-value durability behavior in the reference implementation and it prevents silent clobbering.

**Control**

14. **PAUSED is a quiescence count, not `parked == active`** (§4.4). The spec's formulation races on spawn and deadlocks whenever a subagent parks while its parent awaits it. Replace with per-agent phases and one counter (see below).
15. **A cancelled model call drops *all* `ToolCallPart`s** (R-C-4), not only structurally incomplete ones — none were dispatched, and keeping one would need a synthetic result and would tell the model it ran something it did not. If only tool calls were produced, the assistant message is discarded entirely regardless of `keep_cancelled_output`.
16. **An inflight `delegate` is resumed, not errored** (R-C-13). Only leaf tool calls become `ToolError(kind="interrupted")`; a delegate has no external effect of its own, so the child resumes from its own last safe point.
17. **`save()` takes a timeout** (R-C-10), default 120 s, raising `SaveTimeout` naming the blocking step rather than hanging behind a 600 s `shell`. That message is also what C-12's status bar renders.
18. **The kill test kills with `os._exit(9)` from inside the child** (R-C-12) after a synchronization marker — portable, no `SIGKILL` on Windows, no parent-side race. Comparison is on final output plus the node-output map, not the event log, whose sequence numbers and timings legitimately differ.
19. **`Session` and `WorkflowRef` gain fields** (§6.2): `updated_at`, `resume_state`, `rng_seed`, `usage_total`; `config_type`, `config_hash`, `graph_hash`; and per-agent `model_call_seq`, `child_seq`, `pending_results`, `open_call_ids`, `result_budget`.

**Workflows and platform**

20. **Node ids come from an explicit required `name` argument** per builder call (§6.3), never from stack introspection — `inspect.stack()` breaks under `-O`, frozen builds, decorators and comprehensions.
21. **`FakeProvider` gains `match="by_request_hash"`** (R-P-8), and provider construction must happen inside `build(config)` so a script is reconstructible from `(import_path, config)` alone.
22. **`WindowsSelectorEventLoopPolicy` is forbidden project-wide** — it breaks `create_subprocess_shell` silently. A conftest assertion pins the Proactor loop.

---

## Layer designs

### Providers (M0)

`Provider` is a protocol: `stream(ModelRequest) -> AsyncIterator[StreamEvent]`, `model_info(model)`, `aclose()`. `complete()` is a helper over `stream()`, not a second path (R-P-1).

`OpenRouterProvider` uses `httpx.AsyncClient` with HTTP/2, SSE parsing, tool-call deltas buffered per `index`, `usage: {include: true}`. Retries fire only before the first byte with backoff, jitter and `Retry-After` (R-P-5, C-8). Cancelling the iterator closes the response within 1 s (R-P-6). Malformed tool-call JSON becomes `ToolCallPart(parse_error=...)`, never an exception (R-P-4). `provider_options` passes through verbatim and stays opaque everywhere else (R-P-2). Model metadata is cached on disk with a TTL and degrades to "unknown" rather than blocking (R-P-7).

### Tools (M1)

```python
name: str
description: str  # versioned, snapshot-tested (R-T-9)
Params: type[BaseModel]  # single source of the JSON schema
approval: ApprovalPolicy  # never | always | (params) -> bool
timeout: float
retry: RetryPolicy  # 0 whenever approval != never (R-T-4)
max_result_size_chars: int | float


def is_read_only(self, p) -> bool: ...
def is_concurrency_safe(self, p) -> bool: ...  # default False
async def validate_params(self, p, ctx) -> ValidationError | None: ...
async def run(self, p, ctx) -> ToolResult: ...
def approval_summary(self, p, ctx) -> ApprovalSummary: ...  # diff for edits
```

Defaults fail closed. `ToolDispatcher` owns everything the tool is not trusted with: schema validation, `validate_params`, the gate check, the timeout (`asyncio.timeout`, cancelling the task and calling `kill_tree` for `shell`), retries, output caps, the aggregate budget, contiguous-run batching, and conversion of any unanticipated exception to `ToolError(kind="internal")` with the traceback in `meta` (R-T-2).

The eight built-ins per §7, with the durability behaviors that matter:

- **`read_file`** — line-numbered output, 2 000-line window, byte cap that **errors rather than truncates** (a ~100-byte error is far cheaper than 25 K tokens at the cap), BOM strip and CRLF normalisation, latin-1 fallback with a warning, binary detection by extension and null-byte scan, images as `ImagePart` resized to the API limit, directory and device-file guards, "unchanged since last read" short-circuit. `max_result_size_chars = inf`.
- **`write_file`** — atomic, creates parents, requires a prior read of an existing file, re-checks mtime in a critical section with no awaits, writes exactly the line endings the model sent.
- **`edit_file`** — exact match, unique unless `replace_all`, errors that state the match count and the nearest whitespace-normalised candidate, preserves the file's detected line endings, unified diff in both the result and the approval summary.
- **`glob` / `grep`** — `rg` when present with a pure-Python fallback; `grep` gets `output_mode`, context flags, and `head_limit` with an explicit "truncated, paginate with offset" marker; a timeout reports *timed out*, never *no matches*.
- **`shell`** — Deltas 9 and 10; per-call timeout capped at 600 s; non-zero exit is `ok=False, kind="exit_status"` with output attached; retries hard-blocked.
- **`web_fetch`** — manual redirects (`follow_redirects=False`), same-host-modulo-`www` hops only, cross-host redirects returned to the model as an instruction to re-fetch rather than followed; blocks `file:`, loopback and RFC1918; HTML via `trafilatura`; 10 MB cap.
- **`web_search`** — `SearchBackend` protocol, Serper adapter, registered only when its key is present.
- **`delegate`** — thin wrapper over the `Delegator` protocol, `approval=never`.

Every description is a module-level constant covered by a schema snapshot test (R-T-9).

### Control (M2, M3) — the deep design

**Safe points.** `SafePoint` carries `kind` (`turn_start`, `after_model_call`, `after_tool_batch`, `node_entered`, `node_completed`, `approval_park`, `custom`), `node_id`, `agent_id`, `attempt`, a `NodeSnapshot`, and `durable`/`park` flags. `NodeContext.checkpoint()` builds one and hands it to `RunControl.safe_point()`.

Controller side is **write first, park second**, so a PAUSED run is always durable:

```python
async def safe_point(self, sp: SafePoint) -> None:
    async with self._cp_lock:  # serializes concurrent fan-out children
        self._fold(sp)  # mutate Session; may raise SerializationError
        if sp.durable and self._session_dir:
            await self._write_session()  # atomic, in a thread
        self._resolve_pending_saves()  # R-C-10
    if sp.park:
        await self._park(sp.agent_id)  # MUST be outside the lock
```

Parking outside the lock is non-negotiable — a parked agent holding `_cp_lock` deadlocks every other checkpoint and PAUSED is never reached. `_fold` runs the C-3 stage-2 check and raises `SerializationError` uncaught, so the run goes `FAILED` rather than continuing unsaved (R-W-5).

**Quiescence.** Agent phases: `running` and `blocked_io` are non-quiescent; `blocked_on_child`, `parked`, `waiting_approval`, `finished` are quiescent. **PAUSED ⇔ `_pause_requested and _nonquiescent == 0`.** `blocked_on_child` must be quiescent or a subagent parked at the gate deadlocks a parent that will never return — and it is safe, because a delegate step has no external effect of its own and every child effect has its own safe point. The spawn race is eliminated by calling `enter_agent(child)` and flipping the parent to `blocked_on_child` **synchronously inside `spawn()`/`delegate()` before `tg.create_task(...)`**, so there is no instant with the child unregistered and the parent quiescent.

`pause(hard=True)` additionally cancels every step whose agent is `blocked_io`.

**Interrupt.** `StepHandle.request_cancel(reason)` sets `cancel_reason` **before** `task.cancel()`, and the first reason wins (idempotent, R-C-1). Three rules hold everywhere in `workflows`:

1. Never `except Exception` around a step body — `CancelledError` is a `BaseException` in 3.12 and must not be caught there anyway.
2. `cancel_reason is None` ⇒ not ours ⇒ re-raise unconditionally. Absence of a reason is the authoritative signal for TaskGroup/shutdown cancellation.
3. Absorb only when `current_task().uncancel() == 0`. If an enclosing scope also cancelled us the count stays above zero, and swallowing it breaks `TaskGroup.__aexit__` and `asyncio.timeout.__aexit__`, both of which reconcile on the cancel count in 3.12.

A `delegate` step wraps its child in its own `TaskGroup`, so cancelling the step raises either a bare `CancelledError` or a `BaseExceptionGroup`; the handler splits the group and lets a real child failure outrank the cancellation. Interrupting a child directly (`target="main/2"`) leaves the parent untouched (§6.4).

**Transcript invariant.** For every `ToolCallPart` there is exactly one later `ToolResultMessage` with the same `call_id`, in call order, never completion order. This is enforced by not appending results as they complete: each writes into `AgentState.pending_results`, and `_finalize_turn()` materializes the messages in call order, filling holes. There is no window in which the transcript is invalid. `assert_transcript_valid` runs at every safe point in tests and under `-X dev`.

After an interrupt mid-batch with calls r1 done, r2 running, r3 unstarted: r1's real result survives; r2 is cancelled and its dispatcher returns `ToolError(kind="cancelled")` (with `shell` killing its process group first); r3 is backfilled with the same error. The next model call sees a complete, valid turn.

**Session and resume.** `Session` per Delta 19. `ValueRef` is inline or spilled — node outputs over 32 KB go to `session_dir/values/` so `session.json` stays diffable. `inflight` is maintained live via `RunControl.register_step` and snapshotted in `_fold`, so a safe point taken by agent A correctly records agent B mid-`shell`.

`load()` reconciles each `inflight` entry: a `model_call` is dropped and re-issued (nothing was half-appended, C-1); a `tool_call` becomes `ToolError(kind="interrupted", message="process terminated during execution; effect unknown")` and is **never** re-executed (R-C-13); a `delegate` is resumed (Delta 16). Then `repair_transcript` backfills any `open_call_ids` with neither a pending result nor an inflight entry, and `assert_transcript_valid` must hold or `load()` fails hard.

**Atomic save on Windows.** `tempfile.mkstemp` in the **destination directory** (`os.replace` is only atomic within a volume), write + `fsync`, then `os.replace` in a bounded jittered retry loop — on Windows `os.replace` raises `PermissionError` when the destination has an open handle, so the retry is mandatory, not defensive. No directory fsync exists on Windows; the documented limit is that a power loss can lose the newest checkpoint but never corrupt one. The whole thing runs through `asyncio.to_thread` so a 2 MB write does not blow the R-U-4 budget. Any id used in a filename goes through `slugify_for_path()` (reserved device names, `/ \ : * ? " < > |`, length cap plus hash suffix).

**Node ids and drift.** Ids are `"/".join(path_stack + [name])` from an explicit `name` per builder call. Dynamic children (`Map`, `spawn`, `delegate`) get `<parent>/<index>` from a **checkpointed monotonic counter** bumped synchronously with no await between read and write, so concurrent spawns take ids in call order and a post-resume child cannot collide with a saved sibling. Agent ids are a separate namespace (`main`, `main/0`); a delegate child's node id is `<parent_node>/agent/<n>` so the namespaces cannot alias.

`graph_hash` covers `(node_id, node_class, state_type, output_type)` and deliberately **excludes** prompts and config, so editing a system prompt does not invalidate a session. On load: equal hashes are silent; saved ids missing from the rebuilt graph is a hard `GraphMismatchError` (C-2); extra ids or a changed hash is a `GraphDriftWarning` event, promotable to an error with `strict_graph_hash=True`.

**Kill test.** The `FakeProvider` script must survive `save → kill → load` when the workflow is rebuilt from `(import_path, config)` alone. Default matching is `by_request_hash` — the scripted turn is keyed on a hash of `{messages, tools, model}`, so the resumed process re-issues the interrupted call, rebuilds a transcript that hashes to the recorded key, and gets the same response even though the call *index* shifted. `by_index` remains for linear tests, indexed from the checkpointed `model_call_seq` so the provider stays stateless. An unmatched hash raises `ScriptExhausted` — the test fails loudly instead of diverging silently. The script is committed JSON referenced by the workflow's config, which is why the provider must be constructed inside `build(config)`.

### Workflows (M2, M5)

`Workflow` is a DAG whose builder methods each take a required `name`. Node types per R-W-3. Cycles live inside nodes, so every node is a checkpoint boundary. `AgentLoop` implements §4.3 with Delta 6's batching and Delta 8's budget. Subagents via `ctx.delegate` (blocking, the parent's tool step) and `ctx.spawn` (concurrent handle), registered in the agent tree so the whole tree serializes together.

### TUI (M4, M6)

`HarnessApp(App)` subscribes to the bus, routes by `(agent_id, node_id)`, owns `TUIApprovalHandler`, mounts `ApprovalModal` automatically (R-U-6), and provides the §8.1 bindings. Widgets per R-U-3. `StreamPane` batches deltas on a 33 ms timer into one write (R-U-4). Per C-4 the base app exposes an `interrupt_target()` hook, because a fan-out workflow with no `main` agent has nothing for a targetless interrupt to cancel. Per D5, no Textual type leaks below `tui/`.

---

## Build order

Milestone-gated per D2; each ends with its exit test green and is handed back before the next starts.

| M | Deliverable | Exit test |
|---|---|---|
| **M0** | Scaffolding, `pyproject.toml`, ruff/pyright/pytest/import-linter, the full leaf tier, `contracts.py`, `FakeProvider`, `OpenRouterProvider` | Recorded-SSE fixtures pass via `respx`; a script streams a real OpenRouter completion; all five import-linter contracts green; R-X-4 round-trip on every message type |
| **M1** | `Tool` base, `ToolContext`, `ToolDispatcher`, `platform.py`, eight built-ins, search backends | Per-tool unit tests on `tmp_path` workspaces including failure paths; schema snapshot test; contiguous-run batching test; `kill_tree` tested on Windows and POSIX |
| **M2** | Agent loop + `Controller`, in memory | Cancellation suite (absorb vs re-raise vs `uncancel()` count); transcript-invariant property test; quiescence property test over random `pause/resume/interrupt/set_mode/approve` sequences; the eight §11 control scenarios; the delegate deadlock regression |
| **M3** | `Session`, checkpointing, `save`/`load`, resume reconciliation | Atomic-write Windows suite (held-handle retry, no `.tmp` residue over 1000 writes); save from every run state; `SaveTimeout` during a long `shell`; R-C-9 approval survives save→load; R-C-13 interrupted tool never re-runs; R-W-6 zero completed nodes re-executed; **R-C-12 subprocess kill test** |
| **M4** | `HarnessApp`, `Transcript`, `StreamPane`, `ApprovalModal`, `RunStatusBar` | **R-U-4 perf test — go/no-go on Textual (D5)** |
| **M5** | Graph builder, `FanOut`/`Gather`/`Map`/`Func`/`Subgraph`, `delegate`/`spawn` | Fusion runs headless; subagent tree serializes; mid-fan-out pause/save/load/resume retains completed outputs; graph-drift matrix |
| **M6** | Remaining widgets; three reference workflows with TUIs; the coding-agent CLI (D6) | R-A-1…R-A-4; each workflow also runs `--headless` |
| **M7** | Docs, `docs/architecture.md` + diagram check, acceptance fixtures | R-X-5, R-X-7, §9.3 tasks 1 and 2 against a real model with event logs committed |

Within M2, the cancellation discipline (`workflows/step.py`) and the transcript invariant (`workflows/transcript.py`) land **before** the agent loop. Every later scenario test depends on both, and retrofitting either into a working loop means rewriting it.

---

## Verification

- **Layers** — `lint-imports` in CI enforces downward-only dependencies and public-API-only imports from `workflows/` (R-X-2, R-X-6, R-A-4).
- **Round-trip** — `Session.model_validate_json(s.model_dump_json()) == s` as a property test over sessions with every agent phase, spilled values and pending approvals (R-X-4).
- **Kill test** — a pytest case spawning a child interpreter: baseline run, then run/save/`os._exit(9)`/load/resume, asserting equal final output and node-output map, exactly one `ToolError(kind="interrupted")`, a valid transcript for every agent, and the slow tool started exactly once across both processes (R-C-12).
- **State machine** — hypothesis over random command sequences: no illegal transition, PAUSED implies `_nonquiescent == 0`, no deadlock.
- **Tools** — every tool against a real temp workspace including denied paths, timeouts, non-zero exits, binary files and non-unique edits.
- **TUI** — `App.run_test()` snapshots per widget plus the four-stream perf test.
- **Diagram** — a test parsing `docs/architecture.md`, asserting every named module exists and every `azalabscode.*` package is named (R-X-5).
- **Acceptance** — §9.3 tasks 1 and 2 against a real model in a sample repo, with the event log scanned to prove no `ToolError(kind="internal")` and no `shell` invocation of `cat`/`sed -i`/`find`/`grep` where a built-in exists.

### Critical files

`azalabscode/contracts.py` (all of the layering hinges here), `azalabscode/workflows/step.py` (cancellation discipline), `azalabscode/workflows/transcript.py` (the invariant), `azalabscode/control/controller.py` (quiescence, safe points, state machine), `azalabscode/control/session.py` + `resume.py`, `azalabscode/tools/dispatcher.py`, `pyproject.toml` (the executable form of the layer contract).
