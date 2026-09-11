# Progress

## M0 — done

Scaffolding, tooling, the full leaf tier, `contracts.py`, `FakeProvider` and
`OpenRouterProvider`.

### What was built

**Tooling and layout** — `pyproject.toml` (hatchling) carries ruff, pyright,
pytest and five `import-linter` contracts inline. `README.md` documents the layer
graph and the check commands. The package tree from plan.md exists in full;
`tools/`, `workflows/`, `control/` and `tui/` are documented stubs that name the
milestone their contents land at. `workflows/{coding_agent,fusion,inspector}/`
exist as stubs too, so contract 5 has a real source package to check.

**Leaf tier** (`azalabscode/`)

| Module | Contents |
|---|---|
| `schema.py` | `HarnessModel` (`extra="forbid"`, `validate_assignment`), `VersionedModel`, `json_safe()` |
| `ids.py` | ULID (monotonic within a millisecond), typed id aliases, `slugify_for_path()` |
| `errors.py` | `HarnessError` tree, plus `ProviderError` as *data* and `ProviderErrorKind` |
| `cancellation.py` | `CancelReason`, `StepKind`, `StepOutcome`, `RECOVERABLE_REASONS`, `is_recoverable()` |
| `runstate.py` | `RunState` + `LEGAL_TRANSITIONS`, `AgentPhase` + `QUIESCENT_PHASES`, `NodeStatus` |
| `content.py` | `TextPart`/`ReasoningPart`/`ToolCallPart`/`ImagePart`/`FilePart`, `parse_tool_arguments()` |
| `toolio.py` | `ToolResult`/`ToolError`/`ToolDisplay`/`ToolSchema`/`RetryPolicy`, result caps |
| `permissions.py` | `PermissionMode`, `ApprovalPolicy`, `evaluate_policy()`, `Decision`, `ApprovalRequest` |
| `messages.py` | five message types, `Usage`, `assert_transcript_valid()`, `open_call_ids()` |
| `events.py` | 25 event classes, `EventBus` with bounded queues and coalescing, `JsonlRecorder` |
| `contracts.py` | `SafePoint`, `StepHandleLike`, `RunControl`, `PermissionGate`, `ApprovalHandler`, `Delegator`, `EventSink` |
| `sync.py` | `run_sync()` and the Windows event-loop-policy assertion |

**Providers** — `providers/base.py` holds `ModelRequest`, the eight `StreamEvent`
classes, `ModelInfo`, the `Provider` protocol, `StreamAccumulator` and
`complete()`. `providers/openrouter.py` is the only module in the repo that names
OpenRouter. `providers/models_cache.py` is a TTL'd atomic on-disk cache.
`providers/testing.py` is `FakeProvider` with `by_request_hash` and `by_index`
matching and JSON scripts.

**Tests** — 274 offline, 5 network-marked, in `tests/`: `test_roundtrip.py` (79),
`test_openrouter_provider.py` (43), `test_fake_provider.py` (24),
`test_leaf_tier.py` (67), `test_events.py` (24), `test_architecture.py` (13),
`test_contracts.py` (19), `test_openrouter_live.py` (5). Six recorded SSE
fixtures plus a `/models` fixture in `tests/fixtures/`.

**Script** — `scripts/stream_openrouter.py` streams a live completion, with
`--tools` and `--reasoning` flags.

### How the exit test was verified

| Exit criterion | Evidence |
|---|---|
| Recorded-SSE fixtures pass via `respx` | `tests/test_openrouter_provider.py`, 43 passed. Covers text, reasoning (both the `reasoning` field and `reasoning_details[]`), tool-call assembly from split argument deltas, malformed tool JSON, pre-first-byte retry on 500/429/connect-error, no retry after the first byte, `Retry-After`, cancellation closing the connection in under a second, and the models cache. |
| A script streams a real OpenRouter completion | `scripts/stream_openrouter.py` against `anthropic/claude-haiku-4.5`: 39 completion tokens, `finish_reason=stop`, cost `$0.000224`. With `--tools`: `read_file({"path": "src/app.py"})` parsed, `finish_reason=tool_calls`. Also `pytest -m network`, 5 passed. |
| All five import-linter contracts green | `lint-imports` → `Contracts: 5 kept, 0 broken`. Each contract is *also* proven to bite: `test_an_illegal_import_breaks_the_contract_meant_to_catch_it` writes a deliberately illegal import and asserts the matching contract goes BROKEN. |
| R-X-4 round-trip on every message type | `tests/test_roundtrip.py`, 79 passed. Every message type individually and through the `role` discriminator; every content part; every `ToolErrorKind`; every one of the 25 event classes, with `test_event_classes_are_all_covered` asserting the list is exhaustive over `events.__all__`. |

### Exact commands run

```console
.venv/Scripts/python -m pip install "pydantic>=2.7" "httpx[http2]>=0.27" anyio \
  "textual>=0.80" trafilatura typer rich pytest pytest-asyncio \
  pytest-textual-snapshot respx import-linter ruff pyright hypothesis
.venv/Scripts/python -m pytest -q                       # 274 passed
.venv/Scripts/python -m pytest -m network -q            # 5 passed, 269 deselected
.venv/Scripts/lint-imports                              # 5 kept, 0 broken
.venv/Scripts/python -m ruff check .                    # All checks passed
.venv/Scripts/python -m ruff format --check .           # 40 files already formatted
.venv/Scripts/python -m pyright                         # 0 errors, 0 warnings
.venv/Scripts/python scripts/stream_openrouter.py "Explain a ULID in one sentence."
.venv/Scripts/python scripts/stream_openrouter.py --tools "Read the file src/app.py and summarise it."
```

### Decisions

- **D-M0-1. A tenth leaf module, `runstate.py`.** Plan.md lists nine. `events`
  must carry `RunStateChanged(old, new)` and `contracts.RunControl.phase()` must
  take an `AgentPhase`, but `control/state.py` sits three layers above `events`.
  The enums are vocabulary and moved down; the state *machine* stays in
  `control/state.py` at M2. `LEGAL_TRANSITIONS` and `QUIESCENT_PHASES` moved with
  them because both are definitions, not policy.
- **D-M0-2. `ProviderError` is a pydantic model in `errors.py`, not an exception
  in `providers/`.** It rides inside `StreamEvent.Error` and inside
  `ModelCallFailed`; `providers` and `events` are siblings and neither may import
  the other. `ProviderCallError` wraps it for the one place that raises,
  `complete()`.
- **D-M0-3. `schema.py` as an eleventh module.** R-X-4 wants `schema_version` and
  uniform model config; putting the base classes anywhere else would have made
  some leaf import a sibling. It sits in the bottom layer with `ids`,
  `cancellation`, `runstate` and `sync`.
- **D-M0-4. `azalabscode.tui` is not re-exported from the root `__init__`.**
  Importing the harness must not import Textual — a headless run should not pay
  for it, and D5's Textual go/no-go at M4 must stay a swap. `tui` is still public
  API by R-X-6 (it is a layer `__init__`); it is imported directly. Pinned by
  `test_importing_the_harness_does_not_import_textual`.
- **D-M0-5. Contract 5 uses `allow_indirect_imports = true`.** Without it the
  contract is unsatisfiable: `from azalabscode import UserMessage` reaches
  `azalabscode.messages` transitively through the root `__init__`, which is the
  *sanctioned* path. R-X-6 is a rule about what a workflow may *name*, so the
  contract checks direct imports. Both directions are tested.
- **D-M0-6. Ids are `str` in serialized model fields, `NewType` in protocol
  signatures.** `SafePoint.agent_id: AgentId` forces `AgentId("main")` at every
  call site for a type pydantic erases to `str` on the way in and out. The
  NewTypes stay where they are actually checked and mean something.
- **D-M0-7. A concrete provider's `stream()` is annotated
  `AsyncGenerator[StreamEventUnion, None]`; the `Provider` protocol keeps
  `AsyncIterator` per R-P-1.** A caller holding the concrete type needs
  `aclose()` to test R-P-6, and an `AsyncGenerator` is an `AsyncIterator`, so the
  protocol is unaffected.
- **D-M0-8. `reportIncompatibleVariableOverride = false` in pyright.** Every
  discriminated union narrows a base `type: str` to a `Literal` in each subclass.
  Pydantic fields are mutable, so pyright reads all 25 event overrides as
  invariant. This is the pattern pydantic documents; restructuring the hierarchy
  to satisfy the rule would cost more than the rule is worth. No other pyright
  rule is relaxed and the tree is at 0 errors.
- **D-M0-9. Stream events named `UsageReport`, `Finish`, `StreamError`.** Spec
  §5.2 calls them `Usage`, `Finish` and `Error`. `Usage` is already the message
  type and `Error` is too generic to import unqualified.
- **D-M0-10. `run_phases.py` and all Markdown are excluded from ruff.**
  `run_phases.py` is the milestone driver, not part of the deliverable, and it is
  left untracked as it was found. Markdown is excluded because ruff reformats
  Python blocks inside it — it rewrote the `Session` model in `intent/spec.md`
  on the first `ruff format .`. A formatter has no business editing the
  governing document, and `docs/` at M7 will have the same problem.

### Bugs found and fixed while testing

- `slugify_for_path` did not escape backslashes: a heredoc had collapsed `\\` to
  `\` inside the character class, so `\:` matched a colon and the backslash fell
  through. On Windows that turns one path component into two. Caught by a
  hypothesis property test.
- The provider synthesized a `Finish(reason="stop")` after a mid-stream
  `StreamError`, telling the caller a failed call had succeeded. A `StreamError`
  is now terminal.
- The provider emitted `Finish` twice, because OpenRouter repeats `finish_reason`
  on the trailing usage chunk. Found against the live API, pinned by
  `test_exactly_one_finish_event_per_stream`. A second `Finish` would end an agent
  turn that has not ended.
- A retried request left its first attempt's response context manager registered
  on the caller's `AsyncExitStack`. Ownership now transfers only on success.

### Unverified

- **POSIX.** Everything ran on Windows 11 / CPython 3.12.0 only. Nothing in M0 is
  platform-specific — the OS-dependent work is `tools/platform.py` at M1 — but
  `slugify_for_path` and the atomic write in `models_cache.py` are written
  Windows-first and have not been exercised on Linux or macOS.
- **`SERPER_API_KEY`.** Absent, per D4. Nothing in M0 needs it; `web_search` is
  M1.
- **HTTP/2.** `OpenRouterProvider` defaults to `http2=True` and the live tests use
  it, but no test asserts the negotiated protocol. The respx tests run
  `http2=False` because the transport is mocked.
- **`pip install -e .`** was not run; the suite imports from the repo root. The
  editable install path is exercised at M6, where D6 makes packaging a
  deliverable.
- **Event-bus behaviour under real concurrency.** The backpressure tests drive the
  bus directly. Its behaviour under four concurrent streams is R-U-4, at M4.

## M1 — done

`Tool` base, `ToolContext`, `ToolDispatcher`, `platform.py`, the nine built-ins,
and the search backends.

### What was built

**`azalabscode/tools/`** — the policy layer over `azalabscode.toolio`'s vocabulary.

| Module | Contents |
|---|---|
| `platform.py` | `ShellSpec`/`resolve_shell()`, `spawn()`, `kill_tree()`, `KillOutcome`, `describe_platform()`/`shell_hint()`, `split_env()` |
| `context.py` | `ToolContext` (plain dataclass), `ToolConfig`, `WorkspaceConfig`, `resolve_path()`, `ReadState`/`ReadRecord`, `check_read_before_write()` |
| `base.py` | `Tool`, `ToolSet`, `NoParams`, `describe_validation_error()` |
| `gates.py` | `AllowAllGate`, `DenyAllGate`, `RecordingGate` |
| `budget.py` | `apply_result_cap()`, `TurnBudget`, `spill()`, `persisted_output_block()` |
| `fileio.py` | `atomic_write()`, line-ending detection, `unified_diff()`, `nearest_candidate()` |
| `dispatcher.py` | `ToolCall`, `Prepared`, `Batch`, `partition_runs()`, `ToolDispatcher` |
| `registry.py` | `BUILTIN_NAMES`, `default_registry()`, `read_only_registry()` |
| `builtin/` | `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `shell`, `web_fetch`, `web_search`, `delegate` |
| `search_backends/` | `SearchBackend` protocol, `SearchResult`, `SearchError`, `StaticSearchBackend`, `SerperBackend` |

`azalabscode/tools/__init__.py` exports 57 names; the root `__init__` re-exports
the main ones so a reference workflow reaches them through the public path
(contract 5).

**Tests** — 408 new, in ten files:

| File | Tests | Covers |
|---|---|---|
| `test_tool_platform.py` | 26 | shell resolution, `spawn`, `kill_tree` on both branches |
| `test_tool_base.py` | 30 | fail-closed defaults, declaration checks, schema, `ToolSet` |
| `test_tool_context.py` | 35 | containment, `ReadState` LRU, read-before-write |
| `test_tool_dispatcher.py` | 50 | batching, gate, timeouts, retries, caps, budget, events |
| `test_tool_schemas.py` | 55 | the R-T-9 snapshot plus per-tool schema properties |
| `test_builtin_files.py` | 55 | read/write/edit and the `fileio` primitives |
| `test_builtin_search.py` | 59 | glob/grep on both backends, glob translation |
| `test_builtin_shell.py` | 24 | real subprocesses, interleaving, timeout, cancel |
| `test_builtin_web.py` | 59 | address blocking, redirects, Serper adapter |
| `test_builtin_delegate.py` | 15 | the `Delegator` seam |

`tests/fixtures/tool_schemas.json` is the committed schema snapshot;
`tests/conftest.py` gained the `workspace` and `tool_ctx` fixtures.

### How the exit test was verified

| Exit criterion | Evidence |
|---|---|
| Per-tool unit tests on `tmp_path` workspaces including failure paths | 267 tests across the five `test_builtin_*.py` files, every one against a real temp workspace. Failure paths covered: path outside the workspace, relative escape, symlink escape, directory-as-file, missing file, offset past EOF, over-budget read window, binary by extension and by null scan, non-UTF-8 fallback, write without a prior read, write after external modification, edit with 0 matches, edit with more than 1 match, non-zero exit, timeout, cancellation, invalid regex, search timeout, blocked scheme, private address, DNS rebinding, cross-host redirect, redirect loop, 404, connect error, over-cap body, binary content type, missing delegator, unknown spec, failed child. |
| Schema snapshot test | `tests/test_tool_schemas.py::test_every_builtin_schema_matches_the_snapshot` against `tests/fixtures/tool_schemas.json` (375 lines, 8 tools). Regenerate with `python -m tests.test_tool_schemas`. `shell` is excluded from the byte-for-byte snapshot because its description is rendered per platform (delta 9) and gets structural assertions instead. Backed by `test_the_snapshot_covers_every_registered_builtin`, which fails if a new built-in is added without one. |
| Contiguous-run batching test | Tested twice, because the two ways it fails are different. `partition_runs` as a pure function: all-safe, unsafe-in-the-middle, consecutive-unsafe, leading-unsafe, trailing-unsafe, empty, lone-safe, and order preservation. Then at runtime: `test_a_safe_run_actually_overlaps` asserts every start precedes every stop, `test_an_unsafe_call_separates_the_runs_around_it` asserts the write lands between the two read runs, `test_results_come_back_in_call_order_not_completion_order` asserts call-order alignment while the fast call demonstrably finishes first, and `test_the_semaphore_bounds_concurrency_within_a_run` asserts the peak. |
| `kill_tree` tested on Windows and POSIX | Both branches run on every host: `platform`, `killpg`, `getpgid` and `taskkill` are injected seams, so `test_posix_kill_*` (SIGTERM, SIGTERM to SIGKILL escalation, group-not-pid, `getpgid` failure, vanished group) and `test_windows_kill_*` (taskkill, escalation to `proc.kill`, taskkill missing) all execute here. Plus real processes: `test_kill_tree_really_kills_a_real_process_on_this_host` and `test_windows_kill_tree_reaps_a_grandchild_process` spawn a genuine tree under `pwsh` and assert the grandchild never runs. The POSIX real-syscall equivalent is written and `skipif`-guarded; see Unverified. |

### Exact commands run

```console
.venv/Scripts/python -m pytest -q                       # 679 passed, 3 skipped
.venv/Scripts/python -m pytest -m network -q            # 5 passed, 677 deselected
.venv/Scripts/lint-imports                              # 5 kept, 0 broken
.venv/Scripts/python -m ruff check .                    # All checks passed
.venv/Scripts/python -m ruff format .                   # all formatted
.venv/Scripts/python -m pyright                         # 0 errors, 0 warnings
.venv/Scripts/python -m tests.test_tool_schemas         # regenerate the snapshot
```

### Decisions

- **D-M1-1. Two modules beyond plan.md's list: `tools/gates.py` and
  `tools/fileio.py`.** plan.md names `base context dispatcher registry budget
  platform`. The gates were going to live in `dispatcher.py`, but the dispatcher
  is the largest module in the layer and three stub gates are not dispatch logic.
  `fileio.py` exists because `write_file` and `edit_file` share the atomic-write
  critical section, and that is the one place in this layer where a subtle
  difference between two copies would be a data-loss bug.
- **D-M1-2. `validate_params` returns `ToolError | None`, not
  `ValidationError | None`.** plan.md says `ValidationError`, which collides with
  `pydantic.ValidationError` in every module that has both in scope. `ToolError`
  is also what the dispatcher puts into the `ToolResult` anyway, so the return
  value needs no translation.
- **D-M1-3. The dispatcher re-raises `CancelledError` rather than converting it
  to `ToolError(kind="cancelled")`.** plan.md's interrupt narrative says "its
  dispatcher returns `ToolError(kind='cancelled')`". Absorbing it here would take
  the decision away from `workflows/step.py`, which at M2 owns the three rules
  (`cancel_reason is None` means re-raise; absorb only when `uncancel() == 0`) and
  is the only layer that can evaluate them. `ToolDispatcher.cancelled_result()`
  and `.interrupted_result()` are the factories the step layer calls instead. The
  observable outcome plan.md describes is unchanged; only the layer that builds
  the result moves.
- **D-M1-4. `Tool.on_cancel(params, ctx, reason)` was added to the interface.**
  plan.md has the dispatcher "cancelling the task and calling `kill_tree` for
  `shell`", which would require the dispatcher to know which tool is `shell`. A
  hook keeps the dispatcher tool-agnostic and gives every future tool the same
  seam. It runs *outside* the cancelled task -- see the bug below.
- **D-M1-5. `spawn()` execs an explicit shell program rather than using
  `create_subprocess_shell`.** Spec 7 says `create_subprocess_shell`, which
  hard-codes `cmd.exe` on Windows. Delta 9 already requires `pwsh` with a
  `powershell` fallback, and those two requirements cannot both be met through
  `create_subprocess_shell`. `create_subprocess_exec(shell, *args, command)` is
  the same thing with the shell named.
- **D-M1-6. `Tool.timeout_for(params)` was added.** `shell` needs a per-call
  timeout parameter (spec 7) *and* a dispatcher-enforced timeout (R-T-5), and the
  tool's own must fire first or every slow command comes back as a bare dispatcher
  timeout with no output attached. `shell` returns `min(params.timeout, 600) + 5`.
- **D-M1-7. `grep` gained `output_mode`, `case_insensitive` and `offset`.**
  plan.md asks for `output_mode` and `head_limit`; the pagination marker needs a
  matching `offset` to be actionable, and `case_insensitive` exists because the
  alternative is a model writing a character class per letter.
- **D-M1-8. A hand-written glob-to-regex translation replaces `fnmatch`.**
  `fnmatch` is not path-aware and disagrees with ripgrep in both directions --
  `*.py` matched `src/app.py`, and `src/**/*.py` failed to match `src/app.py`.
  The two backends have to give the same answer, so `compile_glob()` implements
  the path-aware semantics both now use.
- **D-M1-9. The snapshot excludes `shell`.** Its description is rendered per
  platform by design (delta 9), so a byte-for-byte snapshot would be
  machine-specific and would fail on the first POSIX run. It is covered by
  structural assertions naming the platform and its idioms instead.
- **D-M1-10. `read_only_registry()` takes a typed `search_backend` rather than
  `**kwargs`.** `**kwargs: object` did not type-check against
  `default_registry`'s signature, and the only keyword it ever needed was that one.
- **D-M1-11. `web_fetch` refuses a scheme it cannot read and a `content-type` it
  cannot read as text, rather than attempting either.** Spec 7 mentions PDFs via
  `pypdf` "if installed"; `pypdf` is not a declared dependency, so a PDF is
  reported as an unsupported type rather than silently returning binary noise.

### Bugs found and fixed while testing

- **A cancelled `shell` survived its own cancellation.** `ToolDispatcher._cleanup`
  awaited `tool.on_cancel` from inside the task that was being cancelled. The
  pending cancellation is delivered at the first `await`, so the cleanup coroutine
  was thrown into *before its body ran* -- `asyncio.shield` does not help, because
  the shielded task is itself thrown into at its first await. The cleanup now runs
  detached in a task of its own (`_cleanup_detached`), with a strong reference held
  in `self._cleanups` because asyncio only weakly references running tasks, and
  `drain_cleanups()` to await them.
- **A race decided whether the process died at all.** `ShellTool.run` popped its
  `_live` entry in a `finally`, which raced `on_cancel` popping the same entry.
  Whichever coroutine the loop resumed first won, so the kill happened only
  sometimes. `run` now leaves the entry in place on the cancellation path and
  `on_cancel` owns it.
- **`on_cancel` aborted before the kill.** It unlinked the orphaned output file
  first, which raises `PermissionError` on Windows while the not-yet-killed process
  still holds the log open -- and the dispatcher swallowed that, so the tree was
  never killed. Deregister and kill first, clean the file up last.
- **`drain_cleanups()` hung the whole test suite.** The first version looped
  `while self._cleanups: await gather(...)`. `gather` over already-finished tasks
  returns without yielding long enough for a `call_soon` done-callback to run, so
  the loop starved the very callbacks it was waiting on and spun forever. It now
  empties the set itself in one pass.
- **A killed-but-unwaited child leaked an asyncio transport**, surfacing as a
  `ResourceWarning` attributed to whatever test happened to be running when the
  collector reached it -- which under `filterwarnings = ["error"]` failed an
  unrelated test. `on_cancel` now reaps the process and closes the transport.
- **`os.killpg` and `signal.SIGKILL` do not exist on Windows**, so importing
  `platform.py` failed outright and the POSIX branch was unreachable from a test.
  Both are now module-level constants with fallbacks (`SIGKILL_NUM = 9`), which is
  what makes the POSIX branch drivable on this host.
- **ripgrep disagreed with the Python fallback twice.** It only honours
  `.gitignore` inside a git repository (fixed with `--no-require-git`), and it
  omits the filename when given a single explicit file, emitting bare `12:text`
  that the parser could not attribute -- so `grep` on one file returned *no
  matches* (fixed with `--with-filename`). It also does not skip `node_modules` or
  `__pycache__` on its own; the skip list is now passed as explicit globs.

### Unverified

- **The POSIX real-syscall path.** Everything ran on Windows 11 / CPython 3.12.0.
  `kill_tree`'s POSIX branch is fully exercised here with injected `killpg` and
  `getpgid`, so its *logic* is tested -- the SIGTERM, the grace window, the SIGKILL
  escalation, the group-not-pid target, and both failure modes. What has not run is
  a real `os.killpg` against a real session:
  `test_posix_kill_tree_reaps_a_grandchild_process` is written and skips on
  Windows. The same applies to `spawn`'s `start_new_session=True`, the symlink
  containment tests in `test_tool_context.py` and `test_builtin_files.py` (symlink
  creation needs privileges here), and `resolve_shell`'s real `bash`/`sh` lookup.
  Three tests skip on this host; all three are either green paths on a POSIX box
  or bugs to find there.
- **`SERPER_API_KEY`.** Still absent, per D4. `SerperBackend` is written and tested
  against a mocked transport (response parsing, the `tbs` recency mapping, HTTP
  errors, a missing key), but has never spoken to `google.serper.dev`.
  `web_search` is therefore absent from `default_registry()` on this machine,
  which is itself asserted.
- **`web_fetch` against the real internet.** Every test uses `respx` and an
  injected resolver. The address-blocking tests are *stronger* that way -- a public
  hostname resolving to `169.254.169.254` cannot be arranged against the real DNS
  -- but no live fetch has happened, so real-world redirect chains, compression and
  `trafilatura`'s behaviour on a real page are untested.
- **Very large outputs.** The cap and spill tests use synthetic strings of a few
  thousand characters, not a 50 MB build log. The `seek`-based over-cap path that
  delta 10 describes for `shell` is implemented as a full read of the output file;
  it has not been profiled against a real large log.
- **Concurrency at the semaphore bound.** `test_the_semaphore_bounds_concurrency_within_a_run`
  runs six calls at a limit of two. Forty parallel `read_file`s against a real
  filesystem -- the case the limit exists for -- has not been run.

## M2 — done

Agent loop, `Controller`, and everything a human can do to a run, in memory.

### What was built

**Ordering constraint honoured.** plan.md requires `workflows/step.py` and
`workflows/transcript.py` before the agent loop. They landed in the first commit
(`80ed645`), with their tests, before a line of the loop existed.

**`azalabscode/workflows/`**

| Module | Contents |
|---|---|
| `step.py` | `StepHandle`, `StepResult`, `should_absorb()`, `run_step()`, `run_inline_step()` |
| `transcript.py` | `TurnResults`, `finalize_turn()`, `repair_transcript()`, the three fills |
| `state.py` | `AgentState` -- transcript, counters, pending results, queued injections |
| `agent_loop.py` | `AgentSpec`, `AgentResult`, `AgentLoop` (which is the `Delegator`) |

`should_absorb()` is the one place the three cancellation rules live. `run_step()`
runs the body in a child task, which is the shape spec 4.4 asks for and the shape
that lets an interrupt cancel a model call without cancelling the agent awaiting
it. `run_inline_step()` runs it in the caller's task, where absorbing means calling
`uncancel()`; both shapes are exercised, which is what makes the "absorb vs
re-raise vs uncancel count" suite mean something.

`TurnResults` buffers results by `call_id` and `finalize_turn` materialises a whole
batch in call order, so there is no window in which the transcript is invalid.

**`azalabscode/control/`**

| Module | Contents |
|---|---|
| `state.py` | `RunStateMachine` over `LEGAL_TRANSITIONS`, `IllegalTransition`, re-export of `AgentState` |
| `quiescence.py` | `QuiescenceTracker`: per-agent phases, the non-quiescent count, the pause gate |
| `gate.py` | `RuntimePermissionGate`, `gated_names()` |
| `approval_handlers.py` | `QueueApprovalHandler`, `StdinApprovalHandler`, `DenyAllHandler`, `CallbackApprovalHandler` |
| `controller.py` | `Controller`, `InterruptResult`, `FoldRecord` |

`Controller` implements `contracts.RunControl` structurally: `safe_point`,
`enter_agent`, `exit_agent`, `phase`, `register_step`, `unregister_step`,
`permission_mode`. The fold happens under `_cp_lock`; the park happens after the
fold and outside the lock.

**Leaf change.** One new event class, `RunWarning(code, message, detail)`, for spec
C-4's "emit a warning event and do nothing" when a targetless interrupt has no
target. `tests/test_roundtrip.py`'s exhaustiveness check was extended with it.

**Tool-layer change.** `ToolDispatcher.dispatch(..., on_result=...)`, a synchronous
callback invoked as each call finishes from inside the concurrency scope. Without
it the interrupt-mid-batch scenario cannot pass: a cancelled batch never returns,
the `TaskGroup` unwinds, and nobody reads the finished tasks, so a `read_file` that
completed a millisecond before the interrupt would be reported to the model as
cancelled.

**Tests** -- 98 new across six files, plus `tests/harness.py`:

| File | Tests | Covers |
|---|---|---|
| `test_step_cancellation.py` | 22 | absorb / re-raise / the uncancel count, exception groups, registration |
| `test_transcript.py` | 13 | the invariant property, `TurnResults`, `repair_transcript` |
| `test_control_scenarios.py` | 32 | the eight spec 11 scenarios, approvals, denials, R-C-8 |
| `test_quiescence_property.py` | 5 | the hypothesis property plus four named sequences |
| `test_delegate_deadlock.py` | 10 | the regression, the agent tree, R-C-7 through the real gate |
| `test_agent_loop.py` | 16 | request construction, malformed JSON, budget, usage, safe points |

`tests/harness.py` builds a real `Controller` + `AgentLoop` over `FakeProvider`
with two recording fake tools. Everything is real except the model and the tools.

### How the exit test was verified

| Exit criterion | Evidence |
|---|---|
| Cancellation suite (absorb vs re-raise vs `uncancel()` count) | `tests/test_step_cancellation.py`, 22 passed. Absorb: each of the three `RECOVERABLE_REASONS`, and the awaiting agent's cancel count comes back untouched so it can take its next step. Re-raise: a cancellation with no reason (`test_a_cancellation_with_no_reason_is_re_raised`) and each of the three unrecoverable reasons. The count: `test_an_inline_step_absorbs_and_reconciles_the_cancel_count` asserts `cancelling()` returns to its entry value, and `test_an_inline_step_re_raises_when_an_enclosing_scope_cancelled_too` asserts that with two cancellations outstanding *neither* is consumed. Plus `test_a_real_child_failure_outranks_the_cancellation` for the delegate's exception group. |
| Transcript-invariant property test | `tests/test_transcript.py::test_the_invariant_holds_for_any_completion_order_and_any_interrupt`, 300 examples over (call count, completion permutation, interrupt point). Asserts the transcript is valid, every call answered exactly once, answers in call order, and completed calls keep their real results. `test_appending_in_completion_order_breaks_it` is the control: the naive implementation fails on the same inputs, so the property is not vacuous. |
| Quiescence property test over random `pause/resume/interrupt/set_mode/approve` sequences | `tests/test_quiescence_property.py`, hypothesis over sequences of up to 8 commands drawn from `pause, pause_hard, resume, interrupt, inject, manual, auto, approve, deny, tick`. After every command: no illegal transition (the machine raises), PAUSED implies `nonquiescent == 0`, and every agent's transcript is valid. At the end: releasing everything must reach a terminal state within 10 s, with no unanswered tool calls. Committed at 40 examples; run once at 400 (23.8 s, all passed) and once at 150 random sequences *with delegation in the mix* -- see Decisions. |
| The eight spec 11 control scenarios | `tests/test_control_scenarios.py::test_scenario_1..8`. 1 pause-during-stream (the call completes, PAUSED follows, no `ModelCallCancelled`). 2 pause-during-tool (`rig.echo.finished == ["slow"]` -- a pause never leaves a half-run effect). 3 interrupt-during-stream (partial kept, marked, tool calls dropped per delta 15). 4 interrupt-during-tool (the plan's r1/r2/r3 example, verbatim). 5 interrupt-with-injection (the injected message sits immediately after the cancelled assistant message and reaches the next request). 6 mode-switch-with-pending-approval (`set_permission_mode` returns 1, the call runs, `Decision.by == "mode_switch"`). 7 and 8 in their in-memory form -- see Decisions. Hard-pause variants of 1 and 2 are separate tests. |
| The delegate deadlock regression | `tests/test_delegate_deadlock.py::test_pausing_during_a_delegate_reaches_paused`: pause while the child is mid-model-call, assert PAUSED is reached with the parent `blocked_on_child` and the child `parked`. Proven non-vacuous by running the same scenario with `BLOCKED_ON_CHILD` removed from `QUIESCENT_PHASES`: it times out at `state=pausing, nonquiescent=1`. `test_the_naive_rule_would_have_deadlocked_here` writes the counts down so the fix cannot be undone quietly. |

### Exact commands run

```console
.venv/Scripts/python -m pytest -q                        # 778 passed, 3 skipped
.venv/Scripts/python -m pytest tests/test_step_cancellation.py -q      # 22
.venv/Scripts/python -m pytest tests/test_transcript.py -q             # 13
.venv/Scripts/python -m pytest tests/test_control_scenarios.py -q      # 32
.venv/Scripts/python -m pytest tests/test_quiescence_property.py -q    # 5
.venv/Scripts/python -m pytest tests/test_delegate_deadlock.py -q      # 10
.venv/Scripts/python -m pytest tests/test_agent_loop.py -q             # 16
.venv/Scripts/lint-imports                               # 5 kept, 0 broken
.venv/Scripts/python -m ruff check .                     # All checks passed
.venv/Scripts/python -m ruff format --check .            # 85 files already formatted
.venv/Scripts/python -m pyright                          # 0 errors, 0 warnings
```

### Decisions

- **D-M2-1. `AgentState` lives in `workflows/state.py`, not `control/`.** The agent
  loop reads and writes it every turn, and contract 3 forbids `workflows` importing
  `control`. Putting it in `control` would have forced the loop to keep a second,
  parallel copy of the transcript for the controller to mirror -- two objects that
  drift. `control` imports it downward and `control/state.py` re-exports it, which
  is the direction the layer graph allows. M3's `Session` embeds it unchanged.
- **D-M2-2. `should_absorb` generalises plan.md's "`uncancel() == 0`" to "the
  awaiting task's cancel count is unchanged".** The literal rule is right only when
  the step body runs in the caller's task. `run_step` puts it in a child, so
  cancelling the step never touches the awaiter's count and `uncancel()` would
  decrement a count some *enclosing* scope owns. The implemented rule samples
  `cancelling()` at entry and absorbs when the count is unchanged, or when it grew
  by exactly one and this handle owns the current task -- the inline case, where
  `uncancel()` is called and must land back on the entry value. Both shapes are
  tested; the observable behaviour plan.md describes is unchanged.
- **D-M2-3. Scenarios 7 and 8 are the in-memory halves of their spec 11 entries.**
  Spec 11's list ends with "save/load in every state" and "the subprocess kill
  test", both of which are M3 deliverables -- there is no `save()` at M2. Scenario 7
  is therefore *fold* in every state: the run is driven through RUNNING, PAUSING,
  PAUSED, WAITING_APPROVAL and an interrupt, and every safe point folds, validates
  the transcript and leaves an `AgentState` that round-trips through JSON. Scenario
  8 is R-C-13's rule driven by an interrupt instead of a process death: once a call
  is answered with a cancellation nothing re-runs it, and the model sees the error.
  M3 replaces both with the disk versions rather than adding to them.
- **D-M2-4. `ToolDispatcher.dispatch` gained an `on_result` callback.** The
  alternative was to reimplement `partition_runs` + `call` inside the agent loop so
  it could record results as they landed, which duplicates the one piece of the tool
  layer that is genuinely subtle. The callback is synchronous by design: an `async`
  one would introduce a suspension point between a call finishing and its result
  being recorded, which is the exact window the callback exists to close.
- **D-M2-5. Injected messages are always queued, never appended directly.** R-C-4
  says `interrupt` "appends `message` to that agent's transcript". Doing that
  literally breaks two things: a user message between an assistant tool-call message
  and its results is rejected by every provider, and an interrupt that cut a model
  call short has not yet written the `cancelled` assistant message that spec
  decision 4 says the injection must follow. The message goes into
  `AgentState.pending_injections` and the loop drains it at the next turn boundary,
  which is where R-C-4 says the agent resumes. `MessageInjected` is emitted at queue
  time either way. The loop also drains at its final turn and on exit, so an
  injection arriving as the agent finishes gives it one more turn rather than being
  dropped.
- **D-M2-6. The agent loop passes `park=True` on every one of its own safe points.**
  `SafePoint.park` defaults to `False`, and the controller returns immediately from
  `_park` when no pause is pending. Reading `park` as "this is a pause point" rather
  than "park right now" is what makes R-C-3 true -- the run reaches PAUSED at the
  next safe point of *every* active agent -- while leaving the approval safe point
  (`park=False`) free to be taken without stalling the approval it is recording.
- **D-M2-7. A delegate child gets its own `ToolDispatcher` with a fresh
  `ReadState`.** Trap 2 in the M1 handoff flagged this as a choice. Sharing is the
  default you get by accident and it is the wrong one: read-before-write exists to
  stop a model writing over contents it has never seen, and a subagent has its own
  context window. The child shares the workspace, the gate, the toolset and the
  emitter, and nothing else.
- **D-M2-8. One new event class, `RunWarning`.** Spec C-4 asks for a warning event
  and there was no class that fit. `GraphDriftWarning` is about graphs;
  `RunStateChanged.reason` would have buried it in a transition. R-X-3 forbids a log
  line as the only channel. It will also carry C-12's "pausing -- waiting for
  `shell`" at M4.
- **D-M2-9. `Controller.set_body()` exists alongside the constructor argument.** An
  agent loop is constructed with the `RunControl` it reports to, so the controller
  has to exist first. Passing the body to `__init__` works only when the body closes
  over nothing.
- **D-M2-10. The quiescence property test does not include delegation.** Adding a
  subagent to the command matrix roughly triples the runtime for a property the
  dedicated regression already pins. Delegation *was* run through the same driver
  (150 random sequences, all passing) as a one-off check; the committed test keeps
  the fast matrix and the delegate deadlock has its own file.
- **D-M2-11. `run_phases.py`, `run_sessions.log`, `STOP` and `.import_linter_cache/`
  are now in `.gitignore`.** They were untracked but not ignored, so a `git add -A`
  swept two of them into a commit. Ignoring them is what makes "do not commit the
  driver" a property of the repo rather than of whoever is typing.

### Bugs found and fixed while testing

- **A run with no agents yet was declared PAUSED.** Quiescence over an empty set is
  vacuously true, and between `create_task(body)` and the loop's first
  `enter_agent` the set *is* empty. `pause()` immediately after `start()` therefore
  reached PAUSED while nothing was parked, and an `interrupt(message=...)` aimed at
  `main` landed on an agent that did not exist yet and was silently dropped. A run
  that has never registered an agent and still has a live body is now treated as
  starting, not paused. This is the same class of bug as the spawn race delta 14
  describes, one level up.
- **An injection arriving after the agent's last turn was never appended.** The loop
  broke out of the turn loop as soon as the model stopped asking for tools, without
  looking at `pending_injections`. Interrupting with a message while the agent was
  finishing dropped the message. The loop now takes another turn when there is
  something queued, which is what "resume that agent at its next model call" has to
  mean when there would otherwise be no next call.
- **`set_permission_mode` would have deadlocked against its own approval hooks.**
  The mode switch held `_command_lock` while auto-resolving pending requests, and
  resolution calls back into `_on_approval_resolved`, which took the same
  non-reentrant lock. Caught by reading rather than by a test, because the scenario
  that triggers it -- switching to `auto` with a request outstanding -- is scenario
  6 and would have hung the suite with no output. The gate hooks no longer take the
  command lock; there is a comment at both call sites saying why.
- **A pre-existing timing flake in M0's `test_rate_limit_honours_retry_after`.** It
  asserted `elapsed >= 0.05` after a 50 ms `Retry-After`. `asyncio.sleep` schedules
  against the loop clock, whose resolution here is ~15.6 ms, so `time.monotonic`
  legitimately measures a few milliseconds short; it failed at 0.047. Now allows one
  Windows timer tick of slack. Not an M2 bug, but it would have failed M3 the same
  way.

### Unverified

- **POSIX.** Everything ran on Windows 11 / CPython 3.12.0. Nothing in M2 is
  platform-specific -- there is no subprocess work in this milestone -- but the
  three tool-layer tests that skip here still skip, and the cancellation timing in
  the scenarios has only been observed under the Proactor loop.
- **Save, load and resume.** M3. `Controller._fold` is the seam: it validates every
  transcript and raises `SerializationError` on a snapshot that will not round-trip,
  but it writes nothing to disk, and `Controller` has no `save`/`load` methods at
  all rather than stubs that raise.
- **`spawn()`.** R-W-4's concurrent-handle half is not implemented; only
  `delegate()` (blocking) is. `AgentState.next_child_id` and the `enter_agent`
  ordering are already the shape `spawn` needs, and the quiescence rule that makes
  `spawn` safe is the one `delegate` exercises. M5.
- **A workflow with no agents.** `_settle_pause` now requires at least one agent to
  have registered before a run can be declared PAUSED. A future workflow made only
  of `Func` nodes would therefore never reach PAUSED. M5's runner must report node
  phases through `RunControl.phase` the same way the agent loop does; this is
  flagged again in handoff.md.
- **Real models.** Every M2 test runs under `FakeProvider`. Nothing here has been
  driven by a real stream, so the interaction between a real provider's timing and
  the interrupt path is untested. Section 9.3's acceptance tasks are M7.
- **The event bus under load.** The scenarios attach one subscriber to a bus with
  the default 10 000-entry queue. Backpressure and delta coalescing under four
  concurrent streams is R-U-4, at M4.

## M3 — done

`Session`, checkpointing, `save`/`load` and resume reconciliation. The four modules
plan.md names for this milestone exist, the seam M2 cut for them needed no
retrofitting, and the exit test is green including the subprocess kill test.

### What was built

**`azalabscode/control/atomic.py`** — `mkstemp` in the *destination* directory
(`os.replace` is atomic only within a volume), write, `fsync`, then `os.replace` in
a bounded jittered retry loop. The retry is mandatory rather than defensive: on
Windows `os.replace` raises `PermissionError` (WinError 5) whenever the destination
has an open handle, confirmed directly before the module was written. 24 attempts
over ~2.5 s, then `CheckpointError` naming the destination. Every exit path unlinks
the temp file. `sweep_temp_files` exists for the one case the write path cannot
control: a death between `mkstemp` and `os.replace`.

**`azalabscode/control/session.py`** — `Session` per spec 6.2 plus delta 19's
`updated_at`, `resume_state`, `rng_seed`, `usage_total`; `WorkflowRef` with
`config_type`, `config_hash`, `graph_hash` and `resolve()`/`validated_config()`;
`ValueRef` (inline or spilled); `NodeRecord`; `InflightStep` carrying `call_ids` and
`child_agent_id`.

**`azalabscode/control/checkpoint.py`** — `Checkpointer`: paths, content-addressed
value spilling over 32 KB into `values/`, and the deliberate split between
`serialize()` (synchronous, called inside `_fold`) and `write_async()` (a thread).

**`azalabscode/control/resume.py`** — `reconcile()` implementing the three
reconciliations, `resume_state_for()`, and `check_graph_drift()` for spec C-2's
three-way answer. `ResumeReport` records what happened, and `load()` puts its
summary in the `RunStateChanged` reason.

**`Controller`** gains `session()` (a projection, not a mirror), `save()`,
`Controller.load()`, the node ledger (`node_started`/`node_finished`/`node_failed`/
`node_completed`/`node_output`), `blocking_description()`, `restored_agent()` and
`permission_gate`. `safe_point()` now serializes inside the lock after the
synchronous fold, writes through a thread still inside it, and parks after — the
ordering plan.md calls non-negotiable, unchanged.

**`AgentLoop`** adopts a restored `AgentState` on the way in and drains
`state.resume_delegates` before its first model call, re-entering the recorded child
id. `delegate()` and the resume path share `_run_delegate`, so a resumed delegate
takes the same registration ordering and the same cancellation handling.

**`RuntimePermissionGate`** gains `restore_pending()` and the carried decision.

**Tests**: `tests/test_atomic_write.py` (12), `tests/test_session.py` (21),
`tests/test_save_load.py` (22), `tests/test_kill.py` (9), plus
`tests/resumable.py` (the workflow module sessions name by import path),
`tests/kill_child.py` (the child driver) and `tests/record_kill_script.py` (the
fixture regenerator). Scenarios 7 and 8 in `tests/test_control_scenarios.py` were
**replaced**, not added to.

### How the exit test was verified

| Exit criterion | Evidence |
|---|---|
| Atomic-write Windows suite: held-handle retry | `tests/test_atomic_write.py::test_a_write_waits_out_a_held_handle_and_still_lands` — a handle is held on the destination and released from a timer thread mid-write; the write lands, and on Windows the elapsed time proves it could not have succeeded before the release. `test_a_replace_over_a_held_handle_fails_without_a_retry` is the control: the bare `os.replace` raising `PermissionError` on the same inputs. `test_a_handle_that_is_never_released_raises_rather_than_hanging` bounds it. |
| Atomic-write Windows suite: no `.tmp` residue over 1000 writes | `test_a_thousand_writes_leave_exactly_one_file` — 1000 sequential writes, then `[p.name for p in tmp_path.iterdir()] == ["session.json"]` and the content is the last one written. `test_a_thousand_concurrent_writes_leave_exactly_one_file` does 200 through `to_thread` at once and asserts the file is one of the payloads rather than a splice. `test_a_reader_never_sees_a_partial_document` interleaves 300 reads with writes of increasing length. |
| Save from every run state | `tests/test_save_load.py::test_save_works_from_every_run_state` — one run driven through CREATED, RUNNING, PAUSING, PAUSED, WAITING_APPROVAL and COMPLETED, saving at each; every document loads, reports the state it was saved in, and has a valid transcript for every agent. The WAITING_APPROVAL snapshot is asserted to carry the pending request. |
| `SaveTimeout` during a long `shell` | `test_a_save_behind_a_long_tool_times_out_and_names_it` — a 30 s tool, `save(timeout=0.15)`, and the exception's `.blocking` names the agent, the step kind and its elapsed time (`main: 1 tool call(s) (0.2s)`). The fake tool stands in for `shell`; the mechanism is the dispatcher step, which is identical. `test_a_timed_out_save_leaves_no_pending_request_behind` checks the abandoned future is not resolved by a later save. |
| R-C-9 approval survives save→load | `test_a_pending_approval_survives_a_save_and_load` (same `request_id`, `tool`, `params` and `summary` after the round trip; the run loads PAUSED per R-C-11 and `resume_state` is WAITING_APPROVAL), `test_a_restored_approval_resolves_and_reaches_the_re_issued_call` ("resolution proceeds normally": the human's answer is applied to the re-issued call and the second handler is never asked), `test_a_carried_denial_is_honoured_too`, and `test_the_blocked_call_comes_back_as_not_started_not_interrupted`. |
| R-C-13 interrupted tool never re-runs | `test_an_inflight_tool_call_reloads_as_interrupted_and_never_re_runs` (in-process, from a mid-batch snapshot), `test_a_result_that_landed_before_the_death_beats_the_interrupted_error` (a completed call keeps its real result), and across a real process death `tests/test_kill.py::test_the_slow_tool_starts_exactly_once_across_both_processes` — the start count is a file, because the process that wrote it is gone — plus `test_the_interrupted_call_is_answered_exactly_once` (exactly one `ToolError(kind="interrupted")`, five messages, valid transcript). |
| R-W-6 zero completed nodes re-executed | `tests/test_kill.py::test_zero_completed_nodes_are_re_executed` — `executions.log` across both processes is `prepare, agent, agent, summarize`: `prepare` and `summarize` once each, `agent` twice because it never completed. `tests/test_save_load.py::test_a_resumed_run_skips_completed_nodes_and_reruns_incomplete_ones` is the same property in one interpreter, so a failure names the mechanism rather than the OS. |
| **R-C-12 subprocess kill test** | `tests/test_kill.py`, 9 tests, real subprocesses. `test_the_kill_test_reaches_the_same_final_output`: baseline, `crash` (exit 9), `resume`, and the final output *and* the node-output map match. `test_pause_save_kill_load_resume_completes` is R-C-12's own wording — start, pause, save, terminate — and reaches the same result with nothing interrupted. `test_the_crashed_process_left_a_loadable_checkpoint` and `test_no_temp_file_survives_the_kill` check what the dead process left behind. |

Every one of the seven regressions was proven non-vacuous by breaking the fix in the
working tree and confirming the test caught it:

| Fix removed | What failed |
|---|---|
| the `os.replace` retry loop | 3 atomic tests, including the concurrent-write one |
| `_needs_safe_point`'s quiescence rule | `test_save_works_from_every_run_state` (`SaveTimeout` in WAITING_APPROVAL) |
| `repair_transcript` in `reconcile` | `test_the_kill_test_reaches_the_same_final_output`, `test_the_interrupted_call_is_answered_exactly_once` |
| the node memo | `test_zero_completed_nodes_are_re_executed`, `test_a_resumed_run_skips_completed_nodes...` |
| the delegate branch in `reconcile` | both delegate-resume tests |
| the carried decision | both R-C-9 resolution tests |
| the terminal checkpoint | `test_every_durable_safe_point_writes_the_session` |

### Exact commands run

```console
.venv/Scripts/python -m pytest -q                          # 840 passed, 3 skipped
.venv/Scripts/python -m pytest tests/test_atomic_write.py -q      # 12
.venv/Scripts/python -m pytest tests/test_session.py -q           # 21
.venv/Scripts/python -m pytest tests/test_save_load.py -q         # 22
.venv/Scripts/python -m pytest tests/test_kill.py -q              # 9
.venv/Scripts/python -m pytest tests/test_control_scenarios.py -q # 30
.venv/Scripts/python -m tests.record_kill_script           # regenerates the fixture
.venv/Scripts/lint-imports                                 # 5 kept, 0 broken
.venv/Scripts/python -m ruff check .                       # All checks passed
.venv/Scripts/python -m ruff format --check .              # 96 files already formatted
.venv/Scripts/python -m pyright                            # 0 errors, 0 warnings
```

### Decisions

- **D-M3-1. `Session` is a projection built by `Controller.session()`, never stored.**
  The alternative is a `Session` the controller mutates alongside its own fields, and
  the handoff flagged the failure mode: two structures that agree until someone adds
  a field to one of them. As a projection, forgetting to add a field produces a
  session that is *missing* it — which the round-trip test catches — rather than one
  that is quietly stale.
- **D-M3-2. A crashed process's checkpoint is whatever autosave last wrote.** The
  kill test's `crash` phase does not call `save()` before dying, because nothing
  real does. What is on disk is the `after_model_call` checkpoint: the tool call is
  on the transcript with no result and `inflight` is empty, because the batch step
  had not been registered yet. Reconciliation therefore has to handle an open call
  with *no* corresponding `inflight` entry, which is the more common case and the one
  a save-then-kill test would never have exercised. R-C-12's literal
  "pause, save, terminate" sequence is covered separately by the `pause_save` phase.
- **D-M3-3. A `save()` waits on quiescence, not on the in-flight set.** The first
  implementation waited whenever a step was registered, which deadlocked a save
  during `WAITING_APPROVAL`: the dispatcher's step is registered while the gate
  blocks on a human, and the next safe point cannot arrive until that human answers.
  A step whose agent is quiescent is not mutating anything, so the run is written
  immediately. Found by the exit test, not by reading.
- **D-M3-4. `_finish` writes a terminal checkpoint.** There is no safe point after
  the body returns — the agent that would declare one has already exited — so
  without this the newest file on disk says RUNNING for a run that completed, and
  `load()` would faithfully resume a finished workflow. The write is direct rather
  than folded, and deliberately appends no `FoldRecord`: the safe-point rhythm is a
  property of the agent loop, and a terminal write is not one of its beats. It also
  releases anything still waiting in `save()`, which would otherwise sit until its
  timeout on a run that will never reach another safe point.
- **D-M3-5. A call blocked at the permission gate reloads as `not started`, not
  `interrupted`.** `interrupted` means "effect unknown, never re-run this". A call
  the gate was still holding provably had no effect, and the model *should* re-issue
  it once the restored approval is resolved. Using `interrupted` there would be false
  and would make R-C-9's "resolution proceeds normally" unachievable. This is the one
  place R-C-9 and R-C-13 point in different directions, and the tiebreaker is which
  statement is true.
- **D-M3-6. R-C-9's resolution is carried to the re-issued call, keyed on
  `(tool, canonical params)`.** Without this, resolving a restored approval is a
  no-op with respect to the work: the call it guarded died with the process and the
  model has to ask again with a *new* call id, so keying on the call id would carry
  nothing. Keyed on the tool and its arguments, the human's answer lands on the same
  call and nothing else — a model that changes the arguments is prompted again, which
  is right, because the human approved the old ones. One-shot, popped on use, and it
  outranks the permission mode: an explicit human decision is more specific than a
  mode.
- **D-M3-7. Delta 16's resumed delegate is implemented, not deferred.** It is not
  named in the exit test, but "resume reconciliation" is the deliverable and plan.md
  lists the delegate as one of its three cases. `AgentState` carries
  `ResumableDelegate` entries, `repair_transcript` gained a `defer` parameter so the
  parent's call is left unanswered, and the loop drains the queue before its next
  model call. Leaving a *trailing* call unanswered is legal —
  `assert_transcript_valid` documents it as a turn in progress — and `defer` refuses
  to leave a non-trailing one open, because that would put the results out of call
  order.
- **D-M3-8. `RunControl` gains `permission_gate`, `restored_agent` and the four
  node-memo methods.** A workflow rebuilt by `load()` constructs its own provider and
  dispatcher from `(import_path, config)` alone (delta 21), so there is no caller to
  hand it a gate or its restored transcript — it has to ask. `permission_gate`
  returns the `PermissionGate` *protocol*, which keeps the direction right;
  `restored_agent` returns `Any`, because `AgentState` sits above `contracts` in the
  layer graph and naming it there would invert the dependency the module exists to
  prevent. The node memo is on the protocol rather than on `Controller` only, because
  M5's `Runner` is what will call it and the test body is written the way that runner
  will be.
- **D-M3-9. `RunStateMachine.restore()` bypasses the transition table.** A load is
  not a transition: the run being adopted was PAUSED in another process and spec
  6.1 has no legal path from CREATED to that — nor should it, because inventing one
  would let a live run take the same shortcut. `_restore` emits a single
  `RunStateChanged(CREATED → PAUSED)` whose reason carries the resume summary, rather
  than the three events a walk through the table would produce.
- **D-M3-10. `resume()` starts the body when a loaded run has none.** `load()`
  returns PAUSED with the gate closed and no task (R-C-11: never auto-starts).
  `start()` on such a controller is also legal and parks at the first safe point;
  `resume()` opens the gate and then starts. `start()`'s guard changed from "state is
  CREATED" to "there is no task yet", which keeps it idempotent for both shapes.
- **D-M3-11. The kill script is generated by a recorder, not written by hand.**
  `match="by_request_hash"` keys each turn on `ModelRequest.fingerprint()`, and the
  resumed process issues a request the baseline never issued (its transcript carries
  an `interrupted` result), so that key cannot be worked out on paper.
  `tests/record_kill_script.py` gets the keys by running *the same subprocess phases
  the test runs*, with a recording wrapper in front of the provider under
  `by_index` — which is valid here precisely because the interrupted step was a
  *tool* call, so `model_call_seq` is exactly where the resumed process should start.
  A drifted key is `ScriptExhausted`, a non-zero child exit and a failed test, never
  a quietly different answer. Same rule as the tool-schema snapshot: read the diff
  before committing it.
- **D-M3-12. The kill rendezvous is a marker file in both directions.** The slow tool
  announces itself by creating `started-one` and then blocks until `release` appears.
  The killer waits for `started-one`; the resumed process creates `release`. No sleep
  is involved anywhere, which is what M2's note about the ~15.6 ms Windows loop clock
  demands, and it means the baseline and the killed run can share one script without
  a timing race deciding which one wins.
- **D-M3-13. Scenarios 7 and 8 were replaced rather than supplemented.** D-M2-3 said
  M3 would, and the handoff said two tests asserting the same thing at different
  fidelities is how a suite rots. The comment where they were names the tests that
  took over. Scenario 4 stayed: an interrupt mid-batch leaves the process alive and
  the fill is `cancelled`, which is the whole difference between R-C-4 and R-C-13.
- **D-M3-14. `ReadState` is still not serialized**, per the M2 handoff's trap 3.
  After a save/kill/load the model is told to re-read every file before writing it.
  That is safe and correct, and the alternative needs one `ReadState` per agent
  (D-M2-7), not one per run.

### Bugs found and fixed while testing

- **A completed run's newest checkpoint said RUNNING.** See D-M3-4. Caught by
  `test_every_durable_safe_point_writes_the_session`, which loaded the file after the
  run finished and found the wrong state. Left alone, `load()` on any normally
  completed run would have resumed a finished workflow.
- **`save()` deadlocked in `WAITING_APPROVAL`.** See D-M3-3. This is the state a user
  is *most* likely to save from — the modal is up, nothing is happening — and it was
  the one state where `save()` would sit until its 120 s timeout.
- **A `gate.check` regression hung the suite instead of failing it.** Found while
  proving the carried decision non-vacuous: with the carry removed, the test blocked
  on a handler nobody was going to answer. The two calls are now wrapped in
  `asyncio.timeout(BOUND)`. This is the project's "every wait must be bounded" rule
  meeting a test that had forgotten it.
- **The scripted turns for the delegate tests were in the wrong order.**
  `FakeProvider` hands out unkeyed turns in order across the whole run, not per
  agent, so the child was getting the parent's answer. Not a harness bug, but it
  produced a failure that looked exactly like one, and the corrected helper says so
  in its docstring.

### Unverified

- **POSIX.** Everything ran on Windows 11 / CPython 3.12.0. The atomic-write module
  is the one piece of M3 with genuinely platform-specific behaviour, and the half
  that matters here — `os.replace` failing on a held destination handle — does not
  happen on POSIX at all. `test_a_replace_over_a_held_handle_fails_without_a_retry`
  is skipped there and `test_a_handle_that_is_never_released_raises_rather_than_hanging`
  skips itself, both by design; the retry loop is a no-op on a platform where the
  first attempt always succeeds. The POSIX gap that is *not* covered anywhere: there
  is no directory `fsync` on either platform in this implementation, so a power loss
  can lose the newest checkpoint on both. It can never corrupt one.
- **`shell` specifically.** The `SaveTimeout` test uses a fake tool that sleeps
  rather than the real `shell`, because a 30 s `pwsh` in the unit suite is 30 s
  nobody gets back. The mechanism under test is the dispatcher's step registration
  and the quiescence count, which are identical for every tool. The real `shell` has
  its own M1 timeout and kill-tree tests.
- **A crash *during* a checkpoint write.** `test_no_temp_file_survives_the_kill`
  passes because `os._exit` happens to land at a moment when no write is in progress.
  A process killed between `mkstemp` and `os.replace` would leave a `.tmp`;
  `sweep_temp_files` exists for it and is tested directly, but nothing calls it
  automatically yet. M6's CLI is where a session directory gets opened by a user and
  is the natural place to sweep.
- **Graph drift against a real graph.** `check_graph_drift` is tested directly
  against hand-built sessions, because there is no graph builder until M5.
  `load()` does not call it yet — there is nothing to compare against — so C-2's
  wiring, as opposed to its logic, is unverified. M5 must call it.
- **Spilled values under concurrency.** `values/` writes are content-addressed and go
  through the same atomic write, but two agents spilling the same large output at the
  same instant has not been driven. The write is idempotent by construction (same
  content, same digest, same filename), so the worst case is a redundant write.
- **`rng_seed` is carried but not consumed.** Delta 19 asks for the field and it
  round-trips; nothing in the harness seeds an RNG from it yet. M5's `Map` and any
  sampling provider are the first callers.
- **Real models.** Every M3 test runs under `FakeProvider`. Nothing here has been
  driven by a real stream, so the interaction between a real provider's timing and a
  mid-stream process death is untested. §9.3's acceptance tasks are M7.

---

## M4 — done

The TUI: `HarnessApp`, `Transcript`, `StreamPane`, `ApprovalModal`, `RunStatusBar`,
and the R-U-4 perf test. **The Textual go/no-go (plan D5, spec C-13) is a go.**
Four concurrent streams at 200 deltas/s each render at **34-47 ms event-to-screen
against a 100 ms budget**, and 62 ms at six streams. The Rich fallback is not built.

### What was built

**`azalabscode/tui/routing.py`** — `EventConsumer` (a runtime-checkable protocol),
`matches()` and `EventRouter`. Deliberately free of any Textual import: this is the
one file the Rich fallback would have kept. Two rules a widget cannot enforce for
itself live here — a raising consumer is recorded and skipped rather than taking the
pump down, and registration order is delivery order.

**`azalabscode/tui/widgets/base.py`** — `EventWidget`, a mixin with no `__init__`
carrying `agent_filter`, `node_filter` and `handle_event`, so it composes with
whatever Textual class a widget wants to be.

**`azalabscode/tui/widgets/stream_pane.py`** — `StreamPane` and `StreamStats`. The
widget R-U-4 is about. Deltas buffer; a 33 ms timer bounds how often the widget asks
to be redrawn; `render()` drains the buffer and closes the latency interval. `tail`
mode fills its container and renders its own last screenful. Statistics live on the
widget, not in the test, because R-U-4 is a property of the shipped widget.

**`azalabscode/tui/widgets/transcript.py`** — `Transcript` and `ToolCallBlock`. One
agent's turns built from events alone (R-U-1): a `StreamPane` per model call, a
collapsible per tool call whose title carries status and duration, a highlighted line
per injected message. Blocks are capped at 400; the session holds the history.

**`azalabscode/tui/widgets/approval_modal.py`** — `ApprovalModal`, showing all five
things spec 8.1 lists, with the diff coloured by line prefix.

**`azalabscode/tui/widgets/status_bar.py`** — `RunStatusBar`. State, mode, agents,
tokens, cost, checkpoint counters, notice. Everything but the checkpoint counters is
a live read off the `Controller`; a bar rebuilt from events has to be right about
every field forever, and one that reads `controller.state` cannot drift.

**`azalabscode/tui/approval.py`** — `TUIApprovalHandler`. Defers every decision
(`request()` returns `None`) and carries the withdrawal path back to the app.

**`azalabscode/tui/bindings.py`** — spec 8.1's table, plus `forwarded_to_app()`.

**`azalabscode/tui/app.py`** — `HarnessApp` and `EventLog`. Layout, routing,
approvals, bindings, attachment.

**`Controller.set_approval_handler()`** and `Controller.approval_handler` are the
only additions below `tui/`. **Import contract 6** is new: `azalabscode.tui` may not
directly import `workflows`, `providers` or `tools`. That is R-U-1 in the form the
spec asks for it (verify: T, import-linter).

### How the exit test was verified

`tests/test_tui_perf.py`, six tests. The exit test is
`test_four_concurrent_streams_render_within_the_latency_budget`: four real
`AgentLoop`s on one real `Controller`, one `FakeProvider` each, four `StreamPane`s
in a `HarnessApp` subclass under `App.run_test(size=(120, 40))`. It asserts, per
pane: the full text arrived in order, 400 deltas were accepted, the bus dropped
none, the UI coalesced (6.6 deltas per write), `render()` was reached, and
`max_latency_ms <= 100`. Then: the event loop was never blocked past 100 ms, and
800 deltas/s actually reached the bus.

Latency is `datetime.now(UTC) - ts` of the oldest delta the frame carried, closed
inside `render()` — the bus, the pump, the batch timer, Textual's frame scheduling
and the layout. The terminal write is not included; headless has no terminal.

The other five tests exist so the exit test cannot be green for the wrong reason:

| Test | What it stops |
|---|---|
| `test_the_load_generator_actually_delivers_the_rate` | passing because no load arrived |
| `test_the_latency_measurement_is_not_vacuous` | a 400 ms flush timer must breach the budget |
| `test_there_is_no_cliff_just_past_the_four_streams_required` | four passing, six collapsing |
| `test_a_single_growing_transcript_pane_meets_the_budget` | the transcript shape being untested |
| `test_asking_for_a_layout_is_what_costs` | `tail` mode outliving its justification |

`tests/test_tui_app.py` is 36 tests over the widgets and the app: routing filters,
the raising-widget rule, delta coalescing, the run reaching the transcript, the
status bar, the modal from five angles (mount, approve, deny, escape, withdrawal by
mode switch), the queue when two agents wait, a subclass overriding
`on_approval_requested`, every binding in spec 8.1, C-4's targetless-interrupt
warning and the `interrupt_target()` hook, `ctrl+s` from four run states, a
`SaveTimeout` rendered as C-12 asks, `ctrl+o` attaching to the loaded controller,
and the handler seam itself.

```console
.venv/Scripts/python -m pytest tests/test_tui_perf.py -q -s
.venv/Scripts/python -m pytest tests/test_tui_app.py -q
.venv/Scripts/python -m pytest -q -s --deselect tests/test_openrouter_live.py
.venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m pyright
.venv/Scripts/python -c "import sys; from importlinter.cli import lint_imports_command as c; sys.exit(c(standalone_mode=False) or 0)"
```

Final: **878 passed, 3 skipped** (POSIX/symlink only), 0 ruff findings, 0 pyright
errors, 6/6 import contracts. Exit-test line from the last full-suite run:

```
R-U-4: 4 streams x 400 tokens in 2.01s (796 deltas/s)
  | model0 max 42.8ms p95 42.0ms x6.6 | model1 max 42.8ms p95 39.1ms x6.6
  | model2 max 38.8ms p95 36.4ms x6.6 | model3 max 33.8ms p95 33.7ms x6.6
  | loop lag max 12.0ms
```

### Decisions

- **D-M4-1. Textual stays (plan D5, spec C-13).** The measured margin is better than
  2x at the configuration R-U-4 names and better than 1.5x at six streams, with the
  event loop never blocked past 13 ms. Nothing in the profile suggests a wall: the
  remaining latency is two timers (a 33 ms batch and a 16.7 ms frame) plus layout,
  and all three are ours to tune. The Rich fallback is not built and the branch is
  closed. What stays from it is cheap and worth keeping — nothing below `tui/`
  references a Textual type, `routing.py` has no Textual import, and contract 6
  makes both statements executable.

- **D-M4-2. The buffer is drained in `render()`, not on the batch timer.** Spec 8.3
  asks for a 33 ms batch timer, and the timer is still what bounds how often the
  widget asks to be redrawn — but moving the content on that timer made the widget's
  delay and Textual's frame delay *add*, because the two timers are independent and
  their phases drift. A delta that arrived just after a flush waited a whole period
  and then a whole frame. Draining at paint time means every delta that has arrived
  by the time a frame is composed is in that frame. Worth about 30 ms of p95 and most
  of the tail. A deviation from a literal reading of 8.3 ("batches deltas via a 33 ms
  timer and appends ... in one write") in favour of what the requirement it serves
  actually asks for.

- **D-M4-3. `StreamPane` has a fixed-height `tail` mode, and an N-up pane uses it.**
  A pane whose height follows its content must call `refresh(layout=True)` on every
  write, which invalidates the screen's arrangement. That cost grows faster than the
  pane count: at 200 deltas/s per stream, four growing panes measure 53 ms against a
  tail pane's 46 ms, and six measure 89 ms against 62 ms with the loop blocked 44 ms
  at a stretch. A tail pane renders the last screenful itself — a bounded slice of the
  text wrapped per frame, so its cost is a function of its size and not of how long
  the run has been going. The default remains `tail=False` because stacking inside a
  `Transcript` needs auto height, and a transcript has one growing pane, not six.
  Both modes are measured; shipping only the cheap one and calling R-U-4 green would
  have been a test of the fixture.

- **D-M4-4. Spec delta 23: `ScriptedTurn.chunk_rate_hz`.** `chunk_delay_s` cannot
  express a rate on this platform. Measured before anything was written:
  `asyncio.sleep(0.005)` across four concurrent streams yields 70 chunks/s, and 160
  with the 1 ms system timer forced — never the 200 spec 8.3 asks for. `chunk_rate_hz`
  sleeps to an absolute deadline on the original grid, so a tick the clock overslept
  is paid back by the next ones instead of compounding. This is also the more faithful
  shape: real SSE arrives in bursts. The alternative — tuning `chunk_delay_s` to
  0.0038 until the number came out right — would have been tuning the test to the
  machine, which is what M3's handoff trap 4 warns against.

- **D-M4-5. The perf rig freezes the test runner's heap.** The exit test failed at
  107 ms inside a full-suite run and passed at 70 ms alone. The cause was measured,
  not guessed: `gc.callbacks` showed generation-2 pauses of 40-93 ms over pytest's
  accumulated heap, and the failing samples lined up with two of them. `gc.freeze()`
  moves what is already live into a permanent generation, so what remains measured is
  the allocation the app itself does — which is the thing under test and is not
  excluded. This narrows the measurement to the harness; it does not make the harness
  faster. **The finding is real for a shipped app too**, and it is in the handoff:
  M6's CLI should call `gc.freeze()` once after startup.

- **D-M4-6. `Controller.set_approval_handler` is new API, and it sets both copies.**
  A TUI is built around a controller — the app needs the run to attach to, the handler
  needs the app to show a modal on — so the handler cannot go through `__init__`
  without an ordering knot. `Controller` keeps the handler twice: `_handler` for the
  R-C-8 check in `start()`, `gate.handler` for the actual call. Setting one and not
  the other is a run that either refuses to start or hangs at the first destructive
  call, so the method sets both and there is no supported way to set one. It refuses
  once a request is pending: swapping the handler would leave the gate parked on a
  future whose only resolver has been discarded.

- **D-M4-7. `ctrl+p` displaces Textual's command palette.** Textual binds the palette
  to `ctrl+p` as a *priority system* binding, which silently wins over the app's own.
  Spec 8.1 gives that key to pause/resume. The palette moves to `ctrl+backslash`
  rather than being switched off — it is genuinely useful once M6 adds commands to it.
  Found by a test, not by reading: `pilot.press("ctrl+p")` left the run RUNNING while
  calling the action directly paused it.

- **D-M4-8. `ApprovalModal` borrows spec 8.1's table through the `app.` namespace.**
  A `ModalScreen` stops binding lookup, so `ctrl+t` and `ctrl+s` did nothing while an
  approval was up. That is the wrong default here for the same reason M3's `save()`
  deadlock was a bug: `WAITING_APPROVAL` is the state a user is *most* likely to
  switch mode or save from. `bindings.forwarded_to_app()` re-points each action at the
  `app.` namespace so the modal borrows the app's actions without redeclaring any;
  `escape` is dropped because the modal needs it to deny.

- **D-M4-9. `escape` interrupts immediately; the message is a second, separate act.**
  Spec 8.1 reads "interrupt current step, focus `PromptInput` for optional injection
  (enter with empty input = interrupt without message)", which can be read either way.
  The reading taken is the one C-4's decision text states — "lets the user type a
  message *before* the run continues" — so `escape` cancels the step at once and then
  focuses whatever `injection_input()` returns; `submit_injection(text)` does the
  injection. Empty text is a no-op because the interrupt already happened.
  `injection_input()` returns `None` at M4: `PromptInput` is an M6 widget, so the base
  app's `escape` is the cancel half only, which is R-C-4 without the optional message.

- **D-M4-10. Import contract 6, direct imports only.** R-U-1's verification method is
  "T (import-linter)", and the layers contract does not satisfy it: `tui` sits on top,
  so it is *permitted* to import anything below. Contract 6 is the other half. It
  allows indirect imports, because importing `Controller` legitimately pulls
  `workflows.state` in two hops down and forbidding that would forbid the seam the UI
  is supposed to use. What it catches is a widget *naming* an `AgentLoop`, a `Tool` or
  a `Provider`. Proven to bite, like the other five.

- **D-M4-11. The status bar is a live read, not a projection of events.** Every field
  it shows also has an event (R-X-3), and rebuilding it from those events would work
  until the first field somebody forgot to update. Reading `controller.state`,
  `.permission_mode`, `.agents`, `.usage()` and `.pending_approvals` cannot drift. The
  events are a hint that something changed, and a 0.25 s timer covers what they miss —
  including the elapsed durations in `blocking_description()` that no event will ever
  fire for.

- **D-M4-12. The base app pushes the modal from the *event*, not from the handler.**
  R-U-6 says the modal is mounted "when `ApprovalRequested` arrives, unless the
  subclass overrides `on_approval_requested`". Doing it from the event rather than
  from `TUIApprovalHandler.request()` is what lets a subclass replace the rendering
  without replacing the handler, and lets a headless run of the same workflow swap the
  handler without touching the app. Withdrawal is the exception and goes through the
  handler, because a gate that resolves a request without the user — a switch to
  `auto`, a cancelled run — emits no event saying so.

### Bugs found and fixed while testing

- **`ctrl+p` was Textual's command palette.** See D-M4-7. The pause binding did
  nothing, and only in the real key path — the action worked when called directly.
- **`ctrl+t` and `ctrl+s` were dead while a modal was up.** See D-M4-8. The state a
  user most wants them in was the one state they did not work in.
- **The exit test was measuring pytest's heap.** See D-M4-5. It would have been very
  easy to read 107 ms as a Textual verdict and build the Rich fallback.
- **The `Transcript` could not match `ApprovalResolved` to a tool block.**
  `ApprovalResolved` carries a `request_id`, not a `call_id`. The first draft scanned
  for any block in "awaiting approval", which is wrong the moment two agents are
  waiting. The transcript now keeps the `request_id -> block` mapping it learns from
  `ApprovalRequested`.
- **A `ctrl+o` test was loading the wrong document.** The session was saved to
  `session_dir/session.json` with autosave on, so the run's own terminal checkpoint
  overwrote the snapshot with a COMPLETED document — and the test still passed three
  of its four assertions, because `load()` reports PAUSED either way. Fixed in the
  test with `autosave=False`, and worth remembering: `load()`'s PAUSED is not evidence
  that the file you meant to write is the file you read.
- **A save-binding test raced its own notice.** The file can appear from the safe
  point `save()` is waiting for, a moment before `save()` returns. Waiting on the file
  and asserting on the notice is a flake; the test waits on the notice instead.

### Unverified

- **A real terminal.** Everything is measured under `App.run_test()`, which is
  headless. The latency figures stop at `render()` — the point the compositor pulls
  content — and do not include the write to a terminal, its own buffering, or an SSH
  hop. R-U-4's budget is an application-side budget and that is what has been
  measured; a slow terminal emulator can add to it and nothing here would see it.
- **Windows only, one machine.** Python 3.12.0, Textual 8.2.8, Windows 11. The
  absolute numbers are machine-specific. What should survive a move is the *shape*:
  the batch timer dominates p50, layout dominates the tail, and gen-2 GC dominates
  anything measured over a large heap.
- **`StreamPane.tail` under a real resize.** The tail slice is recomputed from
  `content_size` on every render, so a resize is picked up on the next frame, but no
  test drives `pilot.resize_terminal`.
- **`ctrl+o` against a session written by another *process*.** The test saves and
  reloads inside one interpreter. M3's `tests/kill_child.py` is the rig for the
  cross-process case and M4 did not need a second interpreter.
- **R-U-5** (subagent activity visually distinct) is verification method **D**, a
  demonstration in a reference workflow. `StreamPane(subagent=True)` and
  `Transcript(subagent=True)` add the `-subagent` class and a distinct border, and
  `AgentTree` is an M6 widget. The requirement is discharged at M6, not here.
- **R-U-7** (`--headless`) is M6's CLI. `StdinApprovalHandler` and `DenyAllHandler`
  exist and are tested; nothing yet has a `--headless` flag to pass.
- **The event-log pane at streaming rates.** `EventLog` excludes `ModelDelta` and is
  capped at 2000 lines, and the perf test runs with it hidden (`display = False`),
  which is the default. A visible log during a four-way fan-out is not measured.
- **Real models.** Every M4 test runs under `FakeProvider`. A real stream's timing —
  bursty, with multi-second gaps — has not been put through the pipeline.

---

## M5 — done

The graph. `Workflow` and the compiled `Graph`, the `Runner`, `NodeContext`, the
seven node types from R-W-3, `spawn`/`delegate` at graph level, and the first
reference workflow. 978 offline tests, 3 skipped (the same POSIX/symlink skips as
M1), 0 ruff findings, 0 pyright errors, 6/6 import contracts.

### What was built, and where

**`azalabscode/workflows/node.py`** — `Node`, `EmptyState`, `NodeFailure`. Four
declarations carry the design: `State` (what the runner checkpoints, R-W-5),
`output_type` (a *name*, part of `graph_hash`), `dynamic_children` (whether the node
mints ids at runtime), and `hash_fields()` (this node's share of the hash,
overridable — `Func` and `Map` both do).

**`azalabscode/workflows/graph.py`** — the `Ref` family (`InputRef`, `NodeRef`,
`FanOutRef`, `ItemRef`, `SeqRef`, `MapRef`, `ConstRef`), `Env`, `NodeEntry`, and the
compiled `Graph` with `graph_hash()`, `accounts_for()` and `topological_order()`.
The graph is **flat**: a fan-out branch and a subgraph's internals are entries in
the same tuple as the top-level nodes, distinguished by `parent`.

**`azalabscode/workflows/builder.py`** — `Workflow`, spec 6.3's builder.
`node`/`func`/`model_call`/`agent`/`fan_out`/`gather`/`map`/`subgraph`/`output`.
`compile()` runs four checks before a run starts: unique well-formed ids, an acyclic
graph with no dangling references, spec C-3 stage 1 (every declared `State`
round-trips through JSON with defaults), and no `AgentNode` agent id colliding with
a node id.

**`azalabscode/workflows/context.py`** — `NodeContext`: state, `checkpoint()`,
`phase()`, `delegate()`, `spawn()`, `gather_handles()`, the graph, the env and the
runner. **`azalabscode/workflows/handle.py`** — `AgentHandle`, a leaf so
`agent_loop` and `context` can both import it.

**`azalabscode/workflows/nodes/`** — `agent.py` (`AgentNode`), `model_call.py`
(`ModelCall`, `render_prompt`), `func.py` (`Func`), `containers.py` (`FanOut`,
`Gather`, `Map`, `Subgraph`, `run_children`, `unwrap_group`).

**`azalabscode/workflows/runner.py`** — `Runner`. The memo (R-W-6), the quiescence
entries, the safe points, the per-node `TaskGroup` that makes `spawn` structured
(R-W-7), and `on_error="continue"`.

**`workflows/fusion/`** — `workflow.py` (`FusionConfig`, `build`, the three
prompts), `headless.py` (`controller_for`, `run_headless`). Spec 6.3's graph behind
an importable `build(config)`, with one `FakeProvider` per branch in fake mode.

Changed in the layers below:

- **`contracts.RunControl`** gained `enter_node`, `exit_node`, `node_state`,
  `node_failed`, `emitter_for`, `rng_seed`, and `node_class` on `node_started`.
- **`Controller`** gained `enter_node`/`exit_node` (a quiescence entry that is *not*
  an agent), `node_state`, `bind_workflow`, `check_graph_drift`, and
  `strict_graph_hash`. `_fold` now stores a safe point's snapshot onto the node's
  record; `safe_point` parks on `agent_id or node_id`; `node_started` keeps the
  previous attempt's state; `enter_agent` fills in a handed-over `AgentState`'s
  missing `parent_id`. `load()` binds a `Workflow` returned by `build(config)` and
  runs the drift check against it.
- **`azalabscode/__init__.py`** and `workflows/__init__.py` export the M5 surface.

Tests: `tests/graphrig.py` (the rig), `tests/test_graph_builder.py` (31),
`tests/test_graph_runner.py` (29), `tests/test_graph_resume.py` (21),
`tests/test_fusion_workflow.py` (14). `tests/test_contracts.py`'s `StubControl`
grew the new protocol methods.

### How the exit test was verified

The exit criterion is four clauses. Each has its own test and each was checked
non-vacuously by removing the mechanism and watching a test fail.

**1. Fusion runs headless.** `test_fusion_runs_headless` calls
`workflows.fusion.headless.run_headless` and asserts the synthesized answer, with no
TUI and no network. `test_every_stage_runs_and_the_branches_stream_under_their_node_ids`
pins the stage order, asserts every branch streams under `agent_id = <node_id>` (so a
`StreamPane` can route it), and asserts `controller.agents == {}` — fusion has no
agents at all, which is the case that made `enter_node` necessary.
`test_the_analyst_sees_every_branch_answer` and
`test_the_synthesizer_sees_the_answers_and_the_analysis` read the *actual prompts*
off the fake providers, because the synthesizer is scripted and a broken `Gather`
would otherwise be invisible in the final answer.

**2. The subagent tree serializes.**
`test_the_subagent_tree_is_registered_and_serializes` runs a `Func` node that spawns
twice and delegates once, then asserts the three children are in `Session.agents`
with `parent_id == "boss"`, each with its own transcript, and that
`Session.model_validate_json(session.model_dump_json()) == session`.
`test_the_subagent_tree_survives_a_save_and_load` reads the tree back off disk.
`test_child_ids_are_allocated_in_call_order` pins `<node>/agent/<n>` in call order;
`test_a_spawn_registers_the_child_before_its_task_exists` pins trap 1 in the form
that survives the fact — `AgentSpawned` for the child precedes its first
`ModelCallStarted`.

**3. Mid-fan-out pause/save/load/resume retains completed outputs.**
`test_mid_fan_out_resume_retains_completed_outputs`: three branches, two of which
finish while the third parks mid-body at its own checkpoint. At PAUSED the session
holds outputs for the two, a partial `state.done` for the third, and no output for
the fan-out itself. A second `build(config)` produces entirely fresh node objects;
after `Controller.load` + `resume()` the two completed branches have `ran == []`
(never executed) and the third has `ran == list(range(stopped_at, 3))` — it restarted
from its checkpoint, not from zero. `NodeStarted` in the second process names only
`branches`, `branches/slow` and `join`.
`test_a_paused_fusion_run_resumes_from_its_file` does the same thing on the real
fusion workflow, resolved from the session's `import_path` with no `build=` passed,
asserting that the completed branches' providers in the *second* process saw zero
requests and the unstarted ones saw exactly one.

**4. The graph-drift matrix.** Nine cases in `test_graph_resume.py`, all through
`Controller.check_graph_drift`:

| saved vs rebuilt | outcome |
|---|---|
| identical | silent |
| prompt or model edited | silent (`graph_hash` excludes both) |
| a node added | `GraphDriftWarning(extra_nodes=["extra"])` |
| a node removed | `GraphMismatchError(missing=["extra"])` |
| a node changed class under a stable id | `GraphDriftWarning`, no extras |
| any of the above two, `strict_graph_hash=True` | `GraphMismatchError` |
| identical, `strict_graph_hash=True` | silent |
| saved `map/0..2`, rebuilt has only `map` | silent (accounted for by the parent) |
| saved `models/gone`, rebuilt has only `models/a` | `GraphMismatchError` |

Plus two end-to-end through `load()`: a grown graph loads, and a graph that lost a
node raises `GraphMismatchError` naming it.

**Non-vacuity.** Each mechanism was removed and the suite re-run:

| removed | fails |
|---|---|
| `Runner._park_point` | `test_a_run_of_memoized_nodes_still_parks` |
| the runner's own quiescence entry | `..._hold_separate_quiescence_entries`, `..._pause_between_two_nodes_does_not_walk_on` |
| the per-node `enter_node` | `..._hold_separate_quiescence_entries`, `..._pause_waits_for_a_node_that_is_still_working` |
| the R-W-6 memo check in `execute` | both resume tests |
| `ctx.adopt(control.node_state(...))` | `test_mid_fan_out_resume_retains_completed_outputs` |
| `_fold`'s snapshot store | `test_node_state_is_checkpointed_onto_the_record` and the mid-fan-out test |

### Commands run

```console
$ .venv/Scripts/python -m pytest -q
978 passed, 3 skipped in 65.15s

$ .venv/Scripts/lint-imports
Contracts: 6 kept, 0 broken.

$ .venv/Scripts/python -m ruff check .
All checks passed!

$ .venv/Scripts/python -m ruff format --check .
123 files already formatted

$ .venv/Scripts/python -m pyright
0 errors, 0 warnings, 0 informations

$ .venv/Scripts/python -m pytest tests/test_fusion_workflow.py tests/test_graph_resume.py \
      tests/test_graph_runner.py tests/test_graph_builder.py -q
95 passed
```

### Decisions

- **D-M5-1. `RunControl` gains `enter_node`/`exit_node` rather than reusing
  `enter_agent`.** Handoff item 3 is real: a graph of `Func` nodes has no agents, and
  `Controller._settle_pause` requires `_agents_seen`. Registering each node through
  `enter_agent` would have worked for quiescence and been wrong for everything else —
  a phantom `AgentState` with no transcript in `Session.agents`, an `AgentSpawned`
  event per node, and an `AgentTree` at M6 drawing nodes as agents. The two new
  methods touch only the quiescence tracker. `phase()` already tolerates a key with
  no `AgentState`, so nothing else had to change.

- **D-M5-2. A node's quiescence key is its node id, and `safe_point` parks on
  `agent_id or node_id`.** `NodeContext.checkpoint()` leaves `agent_id` unset. The
  alternative — putting the node id in `SafePoint.agent_id` — types a node id as an
  agent id everywhere downstream, including in `FoldRecord` and the `Checkpoint`
  event, where a UI would then have to guess which it was looking at.

- **D-M5-3. The runner holds a quiescence entry of its own, keyed `@<graph name>`.**
  Between two nodes nothing else is registered, and quiescence over an empty set is
  vacuously true — a pause landing there would declare PAUSED over a run that is
  still walking. The runner is `running` between nodes and `blocked_on_child` during
  one. `@` is refused in a builder name so the key cannot collide.
  `test_a_run_of_memoized_nodes_still_parks` is the case nothing else covers: a
  memoized node returns without a context and therefore without a safe point of its
  own, so a fully-memoized replay would ignore a pending pause without
  `Runner._park_point`.

- **D-M5-4. Container nodes are `blocked_on_child` while their children run.** Same
  reasoning as spec delta 14's `blocked_on_child` for a delegating agent: the work
  has been handed to something with safe points of its own. `FanOut`, `Map`,
  `Subgraph` and `AgentNode` all do it. A container claiming `running` for the length
  of a fan-out makes PAUSED unreachable.

- **D-M5-5. `AgentNode`'s agent id defaults to `spec.name`, not to the node id.**
  Both are strings in one quiescence map. If they were equal, the agent's
  `enter_agent` would overwrite the node's entry and its `exit_agent` would delete
  it, leaving the node running with nothing registered — a silent hole in the
  pause path. `compile()` refuses the collision outright. The default also gives the
  coding agent `main` for free, which is the id spec C-4's targetless interrupt looks
  for.

- **D-M5-6. `AgentNode`'s output is the final text, not the `AgentResult`.** Node
  outputs go into the session through `json.dumps(value, default=str)`, so a pydantic
  model would round-trip to its `repr` and a resumed run would hand the next node a
  string where it expected an object. The text composes directly with `ModelCall`,
  and nothing is lost: the transcript, the usage and the outcome are already in
  `Session.agents` under the agent's id. `AgentNode(..., output="result")` returns
  `result.model_dump(mode="json")` for a caller that wants the rest. **Every node
  output must be JSON-native**; this is documented on `Node`.

- **D-M5-7. `NodeContext.spawn` is `async`, and spec 6.3 writes it `def`.**
  Registering the child before its task exists is the whole content of delta 14's
  spawn race, and registration emits an event. A synchronous `spawn` could only fire
  the registration off as a task of its own, which reintroduces exactly the instant
  the delta removes. Recorded as **spec delta 24**.

- **D-M5-8. `spawn()` waits for the child to reach its first suspension before
  returning the handle.** Found by a test, not by reading: `handle.cancel()`
  immediately after `spawn()` took the *node* down instead of the child. A task
  cancelled before its first step never enters its own coroutine —
  `coro.throw()` on an unstarted coroutine raises at the top — so `run_inline_step`
  never bound and nothing absorbed. `_run_agent` sets a `running` event on its first
  line and `_start` awaits it. Nothing can cancel the child in between: there is no
  await between `create_task` and that one.

- **D-M5-9. The compiled graph is flat.** Containers hold no sub-graph; every node
  at every level is one entry with a `parent`. Recursion is the alternative, and
  every id walk, every hash and every drift check is then a place to forget a level.
  `Subgraph` is inlined at build time with its ids prefixed and its refs remapped,
  which also means a subgraph's internals get real records and real memos and a
  resumed run skips the inner nodes that already finished.

- **D-M5-10. `Map` child ids are the list index, not a separately checkpointed
  counter.** plan.md asks for a "checkpointed monotonic counter" for dynamic children.
  For `Map` the index in the input list *is* that counter, and it is stable across a
  resume because the list comes from a memoized upstream node. A second counter in
  the session could disagree with the list, and there is no reading of that
  disagreement that is not a bug. `spawn`/`delegate` do use a counter — `child_seq`
  in the snapshot envelope, and `AgentState.child_seq` for the loop's own children.

- **D-M5-11. The node snapshot is an envelope, `{"state": ..., "child_seq": n}`.**
  The subagent counter has to be checkpointed with the state or two resumed spawns
  collide with a saved sibling, and it is not part of the node's declared `State`
  because the node did not declare it. An envelope that no longer validates is
  ignored and the node restarts from defaults, rather than stranding the whole
  session on one node's state having changed shape.

- **D-M5-12. Dynamic children are accounted for by the parent that declares them.**
  Spec C-2 says saved ids missing from the rebuilt graph are fatal. It does not
  consider dynamic children, and taken literally it would make a session containing
  `map/0..3` unloadable, because the rebuilt graph has only the static `map`.
  `Graph.accounts_for(node_id)` walks the id's prefixes and accepts it if the nearest
  static ancestor has `dynamic_children` — `Map` and `AgentNode` only. A deleted
  fan-out branch is still fatal, because `FanOut` does not declare it.
  `Controller.check_graph_drift` unions the accounted ids into the set it hands
  `resume.check_graph_drift`, which is unchanged.

- **D-M5-13. `strict_graph_hash` defaults to `False`.** Handoff item 1 asked for the
  decision. A graph that grew can absorb everything the session knows; refusing to
  load it would make adding a node to a workflow a one-way door for every session
  already on disk. Missing ids are fatal either way, which is the case where there
  is genuinely nowhere to put a saved output.

- **D-M5-14. `build(config)` may return a `Workflow` or a callable body.**
  `Controller.load()` detects a `Workflow`/`Graph` and calls `bind_workflow`, which
  is what makes the drift check possible at all — the controller has to hold the
  graph. A callable body is the M2 shape and still works, which is what keeps
  `tests/resumable.py` and the kill test running untouched.

- **D-M5-15. `graph_hash` is computed from the graph, never passed in.**
  `bind_workflow` derives it, so a `WorkflowRef` on disk cannot claim a hash the
  graph does not have. That is the only condition under which spec C-2's comparison
  means anything.

- **D-M5-16. The runner walks top-level nodes sequentially; concurrency is spelled
  `FanOut` or `Map`.** A runner that ran every ready node concurrently would make the
  graph's concurrency invisible in the workflow file and would make the safe-point
  rhythm depend on the shape of the DAG. Spec 6.3's own example uses `FanOut` for
  exactly this.

- **D-M5-17. `Func` accepts `(input)` or `(ctx, input)`, and accepts a synchronous
  body.** Arity is inspected once at construction. R-W-3 says "async Python
  function" and that is still the contract for anything that does I/O; refusing a
  one-line synchronous transform pushes authors towards `run_until_complete`, which
  is worse. The docstring says a blocking body belongs in `asyncio.to_thread`.

- **D-M5-18. `Runner.run(control=None)` runs a graph with no run around it.** The
  same standalone shape the tool layer has behind `AllowAllGate` and the agent loop
  has with `control=None`. No memo, no safe points, no quiescence.

- **D-M5-19. In fake mode fusion builds one `FakeProvider` per branch.** M5 trap 5.
  A shared script hands out unkeyed turns in order across the whole run, not per
  branch, so a fan-out sharing one would interleave by scheduling order. Per-branch
  providers remove the shared state rather than working around it.

- **D-M5-20. Fusion branch ids come from the model name, not its index.** An index
  shifts when a model is added to the middle of `config.models`, and every session
  saved before that stops lining up. `slugify` drops the vendor prefix (the node id
  is already scoped by `models/`) and duplicates get a `-1` suffix.

- **D-M5-21. Three defects found by the new tests were fixed rather than tested
  around.** (a) A spawned child's `AgentState` did not name its parent, so the tree
  serialized as a flat list — `Controller.enter_agent` now fills in a handed-over
  state's missing `parent_id`, and `NodeContext` passes one. (b) D-M5-8's
  pre-start cancellation window. (c) The mid-fan-out test's own premise was wrong:
  pausing before the fast branches completed parked them mid-body, so there was
  nothing completed for the resume to retain and the test would have passed while
  proving the opposite of what it claimed. The test now waits for them to finish and
  says why in a comment.

### Spec deltas added

- **Delta 24.** `NodeContext.spawn` is `async`. Spec 6.3 writes it `def`; see
  D-M5-7.
- **Delta 25.** A node's declared `State` is checkpointed inside an envelope that
  also carries the node's subagent counter; see D-M5-11.
- **Delta 26.** Dynamic children are accounted for by the parent that declares
  them, rather than being treated as missing nodes; see D-M5-12.

### Unverified

- **The graph has never met a real model.** Every M5 test runs under
  `FakeProvider`. Fusion's real path constructs one `OpenRouterProvider` inside
  `build(config)` and shares it across all five `ModelCall` nodes; the client is
  `httpx.AsyncClient` and is safe to share, but five concurrent streams against a
  real endpoint — rate limits, mid-stream failures, the retry in `ModelCall` — have
  not been exercised. `branch_concurrency` exists for the rate-limit case and has
  only been used to make a test deterministic.
- **No cross-process resume of a graph.** Every save/load here happens in one
  interpreter. M3's `tests/kill_child.py` is the rig for the real thing and it
  drives an `AgentLoop`, not a graph. A graph kill test would be the honest version
  of clause 3 of the exit test and it is not written.
- **A `delegate` in flight from a `Func` node at process death is not resumed.**
  `resume.reconcile` looks the inflight step's `agent_id` up in `session.agents`; for
  a node-level delegate that key is a node id, so the entry is skipped and no
  `ResumableDelegate` is queued. The node itself has not completed, so it re-runs and
  re-delegates from its checkpointed `child_seq` — correct, but it orphans the
  previous child's transcript in `Session.agents` rather than resuming it. Delta 16
  holds for `AgentLoop.delegate`, which is where the `delegate` *tool* runs; it does
  not yet hold for `NodeContext.delegate`.
- **`Map` over a collection large enough to spill.** `ValueRef` spills node outputs
  over 32 KB into `session_dir/values/` and the path is tested directly (M3), but no
  M5 test produces an output that large. Trap 4 from the M4 handoff is still
  undriven.
- **`rng_seed` has a consumer but no user.** `NodeContext.rng` seeds a
  `random.Random` from `(rng_seed, node_id, attempt)` and is deterministic across a
  resume by construction, but no node in the tree samples, so nothing exercises it
  end to end.
- **`on_child_error="continue"` on a `Subgraph`.** The policy is threaded through
  `FanOut` and `Map` and tested on both. `Subgraph` runs its children with the
  default and has no knob; a failing inner node fails the subgraph.
- **Node retries.** `NodeRecord.attempt` is respected everywhere — the memo, the
  state restore, the events — but nothing ever increments it. `Runner(attempt=...)`
  is the seam and no caller uses it.

## M6 — done

The last layer with a hole in it is filled: spec 8.2's widget table is complete, all
three reference workflows exist with their own TUIs and a `--headless` mode, and the
coding agent ships as an installed command (`azc`, plan decision D6).

**1062 offline tests, 3 skipped** (POSIX/symlink only), 0 ruff findings, 0 pyright
errors, 6/6 import contracts. M5 handed over 978 tests; M6 adds 84.

### What was built, and where

**The seven remaining widgets** (`azalabscode/tui/widgets/`, R-U-3):

| file | what |
|---|---|
| `tool_calls.py` | `ToolCallRecord` (the fold), `ToolCallList` (a `DataTable` of every call), `ToolCallDetail` (params, result, error, timing, event trail) |
| `diff_view.py` | `DiffView` — a unified diff with a coloured gutter and a Pygments-highlighted body, from an approval or a completed `edit_file`/`write_file` |
| `agent_tree.py` | `AgentTree` — the live agent tree, delegated vs spawned marks, phases and outcomes; `seed()` for a loaded run |
| `stage_pipeline.py` | `StagePipeline` + `StageChip` — the graph's top-level nodes as chevrons, fan-out branches rolled into their parent |
| `split_panes.py` | `SplitPanes` — the N-up grid layout, wrapping past three columns |
| `prompt_input.py` | `PromptInput` — multi-line, `enter` submits, `alt+enter` is a newline, with the interrupt mode spec 8.2 asks for |

`StreamPane.set_text()` is new (restoring a completed branch from a session, R-A-2),
and `failure_status()` splits a denial and a cancellation out of "failed" across
`ToolCallList`, `ToolCallDetail`, `ToolCallBlock` and the headless summary.

**Two additions to core**, both required by spec 8.2 and neither optional:

- `Controller.send(text, target=None)` — inject a user message **without cancelling**
  the current step. Spec 8.2 gives `PromptInput` two verbs and they differ in exactly
  this. Without it a UI's "send" would have to be an interrupt with nothing to
  interrupt, which cancels a model call the user was happy to let finish.
- `HarnessApp(autostart=...)`, `HarnessApp.send()`, `HarnessApp.send_target()`, and a
  default `injection_input()` that finds a mounted `PromptInput`. A subclass that
  composes a prompt box now gets both verbs with no handler, the same way it gets the
  approval modal (R-U-6).

`Controller`, `EventBus`, `ToolDispatcher` and `Runner` are now `@final`, which is
what turns R-A-4's third clause from a statement about nothing into a check.

**The shared agent node** (`workflows/tooling.py`). `ToolingAgentNode` is what both
tool-using workflows run on. It differs from `AgentNode` in two ways that matter:

1. it builds its own `ToolDispatcher` inside `run()`, from
   `ctx.control.permission_gate` — a graph rebuilt by `Controller.load()` has no
   caller to hand it one (spec delta 21), and the gate does not exist until a
   `Controller` does;
2. with `interactive=True` it runs its loop **once per prompt**, so a session is one
   agent and one transcript across many tasks. Later prompts arrive as pending
   injections, which is the same mechanism `escape` uses (R-C-4).

**The three reference workflows.**

```
workflows/
  cli_support.py       gc.freeze(), sweep_temp_files, .env/API-key and config resolution
  tooling.py           ToolingAgentNode
  coding_agent/        prompts.py workflow.py session.py app.py cli.py __main__.py
  fusion/              workflow.py headless.py app.py cli.py __main__.py
  inspector/           workflow.py headless.py app.py cli.py __main__.py
```

- **`coding_agent` (R-A-1, spec 9.1).** One `ToolingAgentNode` with every built-in
  plus `delegate`, the `explore`/`review`/`edit` subagent specs, and a real system
  prompt in `prompts.py` (never in core). `CodingSession` is the object both the TUI
  and the CLI drive, with `send`/`interrupt`/`close`. `CodingAgentApp` is spec 9.1's
  layout: a transcript per agent switched by the `AgentTree`, `ToolCallList` and its
  drill-down on the right, a `DiffView` overlay that opens on an edit, a
  `PromptInput` docked at the bottom, and the status bar.
- **`fusion` (R-A-2, spec 9.2).** M5 built the graph; M6 adds `FusionApp` —
  `SplitPanes` of `tail=True` `StreamPane`s across the top, the `StagePipeline`
  below, and the active stage's pane under that, following the run until the user
  clicks a chevron. `interrupt_target()` is overridden because fusion has no agents
  at all (spec C-4).
- **`inspector` (R-A-3, spec 9.3).** One agent, every tool it is given, and spec
  9.3's layout: `Transcript` | `ToolCallList` with the `ToolCallDetail` drill-down
  and `ctrl+l` for the raw event sequence. Headless it prints a `CallSummary`, which
  is a second, Textual-free fold of the same events.

**The CLI (D6).** `azc`, `azalabs-fusion` and `azalabs-inspector` are console
scripts in `pyproject.toml`; `workflows` is now packaged alongside `azalabscode`
(still outside it, so import contract 5 keeps proving R-A-4). `workflows/
cli_support.py` carries what D6 asks for: `.env` and environment key resolution that
tolerates `KEY = "value"`, a JSON config file whose values are never overwritten by
a flag that was not typed, and `prepare_process()` — which is where the two items
M4 and M3 left without a caller finally get one, `gc.freeze()` and
`sweep_temp_files()`.

### How the exit test was verified

**R-A-1** — `tests/test_acceptance_coding_agent.py`. Three scripted tasks against a
real sample repository (`tests/acceptance.py:sample_repo`) with the real built-in
tools, the real dispatcher and the real gate; only the model is a `FakeProvider`:

1. §9.3 task 1: add a `--verbose` flag to `cli.py` and update the README. Asserted by
   reading the files afterwards.
2. §9.3 task 2: replace every call site of `deprecated_fn` with `new_fn`, then run
   the tests — the `shell` call really runs `pytest` in the sample repo.
3. A broad question delegated to the `explore` subagent, which searches under its own
   agent id while the parent is blocked on it.

Each run is scanned with `tests/acceptance.py:scan_run`, which is spec 9.3's pass
criterion as a function: only built-in tools, no `ToolError(kind="internal")`, and no
`shell` invocation of a command a built-in covers. The scan is proved non-vacuous
against a `find . -name x` it must reject. Plus the TUI: the transcript per agent, the
tree, the call table and drill-down, the diff overlay opening on an edit, the approval
modal raising and denying two edits in `manual` mode, and the prompt box sending a
second message into a *waiting* session without cancelling anything.

**R-A-2** — `tests/test_acceptance_fusion.py`, on top of M5's
`tests/test_fusion_workflow.py`: one pane per branch fed by its node id, the pipeline
rolling four branches into one chevron with a counter, the visible stage following the
run and then the user's click, a fan-out interrupt that has a target, and — the clause
M6 owed — a mid-fan-out pause/save/load where the completed branches are restored into
their panes from the session and the incomplete one re-streams.

**R-A-3** — `tests/test_acceptance_inspector.py`: every call in the table with its
status and duration, the drill-down showing params, result, error and the raw event
trail, the cursor moving the drill-down, `ctrl+l` showing the event log with the
deltas suppressed, and the headless `CallSummary` producing *the same statuses* as the
widget over the same run.

**R-A-4** — `tests/test_acceptance_public_api.py`: an AST walk over every file in
`workflows/` for private imports, for assignments and `setattr` on anything imported
from core, and for subclassing a `@final` core class; plus import-linter run from
inside pytest so a `pytest` run alone catches a contract break. Each of the three
clauses is also run against a synthetic module that violates it and must be caught.

**`--headless` for each workflow** — driven through the real CLI with
`typer.testing.CliRunner` in all three acceptance files, and by hand through the
installed `azc.exe`.

### Exact commands run

```console
$ .venv/Scripts/python -m pytest -q
1062 passed, 3 skipped in 72.08s

$ .venv/Scripts/lint-imports
Contracts: 6 kept, 0 broken.

$ .venv/Scripts/python -m ruff check .
All checks passed!

$ .venv/Scripts/python -m ruff format --check .
153 files already formatted

$ .venv/Scripts/pyright
0 errors, 0 warnings, 0 informations

$ .venv/Scripts/python -m pip install -e . --no-deps
Successfully installed azalabscode-0.1.0

$ .venv/Scripts/azc.exe --headless --auto --config .tmp_ws/azc_cfg.json "add --verbose"
Done: --verbose added.          # and the file on disk had the flag

$ .venv/Scripts/python -m workflows.fusion --headless --config .tmp_ws/fusion_cfg.json
the final answer

$ .venv/Scripts/python -m workflows.inspector --headless --config .tmp_ws/insp_cfg.json
There are two Python files; a.py defines one().
main       glob       ok      0.05s
main       read_file  ok      ...
```

### Decisions

- **D-M6-1. `Controller.send()` is new core API rather than the workflow poking
  `AgentState.pending_injections`.** Spec 8.2 gives `PromptInput` two verbs, `send`
  and `interrupt`, and only `interrupt` existed. A workflow could have appended to
  `pending_injections` itself — it is a public field — but then the `MessageInjected`
  event (R-X-3) would be missing and every UI would have to re-implement the same
  three lines. Five lines in the controller, one place.

- **D-M6-2. `ToolingAgentNode` rather than spec 9.1's literal `AgentNode`.** Spec 9.1
  writes the coding agent as `AgentNode(AgentSpec(...))`, which cannot work for a
  workflow that must be rebuildable from `(import_path, config)`: `AgentNode` takes a
  `ToolDispatcher`, and building one needs the run's gate, which does not exist when
  `build(config)` is called. The node therefore builds its dispatcher in `run()` off
  `RunControl.permission_gate` — which the protocol already exposes for exactly this
  reason. The second difference, running the loop once per prompt, is what makes it an
  interactive *session* rather than a one-shot task, which is what R-A-1's word
  "interactive" and D6's "daily use" both require.

- **D-M6-3. A prompt after the first is an injection, not a new run.** The alternative
  was one `AgentLoop` per prompt with a fresh transcript, which loses the context that
  makes a session worth having. `AgentLoop.run()` already seeds the transcript only
  when it is empty and already declines to end a turn while injections are pending, so
  the mechanism was there; the node only has to wake up. The visible cost is that
  `AgentSpawned`/`AgentFinished` are re-emitted for `main` once per prompt.
  `AgentTree` folds the repeat.

- **D-M6-4. The interactive node is quiescent while it waits for a prompt.** It stays
  `blocked_on_child` with its agent exited, so an idle session reaches PAUSED and can
  be saved and closed. A prompt taken off the queue goes through
  `ctx.checkpoint(park=True)` before the loop is re-entered, so typing into a paused
  session does not quietly restart it.

- **D-M6-5. `Controller`, `EventBus`, `ToolDispatcher` and `Runner` are `@final`.**
  R-A-4 forbids subclassing "a core class marked `@final`" and nothing in core was
  marked, so the clause was vacuous. These four are compositional — wired together,
  never specialised — and nothing in the repository subclasses any of them. Everything
  a workflow *is* meant to extend (`Node`, `Tool`, `HarnessApp`, the widgets) is
  deliberately not marked.

- **D-M6-6. The §9.3 acceptance tasks run scripted at M6; the real-model versions are
  M7.** The plan's build order puts "§9.3 tasks 1 and 2 against a real model with
  event logs committed" in M7 and gives M6 the workflows, the UI and the CLI. What M6
  owes and delivers is the *check*: `tests/acceptance.py:scan_run` is written against
  the event stream, so M7 runs the identical function over a recorded JSONL log.
  Scripting also makes the file assertions meaningful — a real model's edit is not
  predictable enough to assert `"[--verbose]" in README.md` on.

- **D-M6-7. The §9.3 shell scan covers PowerShell as well as POSIX.** Spec 9.3 names
  `cat`, `sed -i`, `find` and `grep`. On this machine the default shell is `pwsh`, so
  the equivalents a model would actually reach for are `type`, `Get-Content`,
  `Select-String` and `Get-ChildItem`; a check that missed those would pass here for
  the wrong reason. The matcher tokenises rather than substring-matches, so
  `git log --grep=x` and a path ending in `ripgrep` are not false positives.

- **D-M6-8. The headless inspector summary is its own fold, not the widget's.**
  `ToolCallRecord` lives in `azalabscode.tui`, which imports Textual, and import
  contract 5 forbids a reference workflow from reaching into `azalabscode.tui.*`
  anyway. A headless run must not pay for a UI it will not draw. The duplication is
  ~40 lines and is defended by a test asserting the two folds produce the same
  statuses over one run — if they ever diverge, something is being read that is not in
  the event stream, which is what R-X-3 forbids.

- **D-M6-9. `fake` scripts are inline in the config, not a path.** Fusion's `fake`
  field is a `model -> answer` map; the coding agent's and the inspector's `script`
  field is a whole `Script` document as a dict. Inline keeps the config
  self-contained, which is what makes `(import_path, config)` sufficient to rebuild
  the graph in another process (spec delta 21). A path would make a session depend on
  a file outside it.

- **D-M6-10. The `edit` subagent is not in `subagents` by default.** ⚠C-6: a delegated
  agent in `manual` mode has its write tools filtered out by the gate (R-C-7), so an
  `edit` subagent under the default mode is handed a task it cannot do. The spec is
  read literally — the spec exists — and the default list is `["explore", "review"]`.
  `--subagent edit` opts in, and it is only useful with `--auto`.

- **D-M6-11. `azc` defaults to `manual`, and the other two CLIs to `auto`.** The
  coding agent writes files; `permissions.DEFAULT_MODE` is manual and the CLI does not
  override it. Fusion has no tools at all, and the inspector's default toolset is
  read-mostly, so a headless run of either in `manual` with no handler would be a
  `ConfigurationError` (R-C-8) for no benefit. The inspector's *TUI* runs `manual`,
  because there a modal can answer.

- **D-M6-12. The diff overlay opens itself on an edit and is toggled with `ctrl+d`.**
  Spec 9.1 says "DiffView overlays on edits". A docked diff pane would be empty for
  most of a session; an overlay that never appeared on its own would be a feature
  nobody finds. It is an app-level binding, so it does not work while the approval
  modal is up — which is fine, because the modal renders the diff itself.

- **D-M6-13. `StagePipeline` rolls a fan-out branch into its parent chevron.** M5's
  runner emits `NodeStarted` for every node at every level, so a four-model fusion
  would otherwise draw five chevrons. The roll-up is by node-id prefix, which is
  exactly what the runner's ids encode, and anything starting with `@` (the runner's
  own quiescence key) is skipped.

- **D-M6-14. A denial is not a failure.** `failure_status()` maps
  `ToolErrorKind.DENIED` to `denied` and `CANCELLED`/`INTERRUPTED` to `cancelled`
  everywhere a status is shown. The gate denies, and the dispatcher then reports an
  ordinary `ToolCallFailed` carrying `kind="denied"`, so a widget reading the event
  class alone calls every denial a failure. The user needs to know which of the two
  happened; they mean different things about what to do next.

### Bugs found and fixed while testing

- **`AgentTree` re-parented a delegated child to the root.** A delegated agent is
  registered *twice*: once by its parent before the child's task exists (the spawn-race
  ordering, carrying `parent_id` and `delegated=True`) and once by the child's own
  `AgentLoop._enter` (carrying neither). Folding the second event verbatim moved the
  child to the root and changed its mark. `parent_id`, `spec_summary` and `delegated`
  are now sticky. The duplicate `AgentSpawned` itself is pre-existing M2 behaviour and
  is left alone: the ordering that produces it is the thing that removes the spawn
  race.
- **`ToolCallList` dropped every denial.** `ApprovalResolved` carries a `request_id`
  and no `call_id` — deliberately, because a resolution can arrive from a mode switch
  that never saw the call — so the fold silently never reached the row. Both widgets
  now keep a `request_id -> call_id` index built when the request arrives.
- **`AgentTree._nodes` collided with `Widget._nodes`.** Textual keeps a widget's
  children in `self._nodes`; the tree's own `agent_id -> TreeNode` dict shadowed it and
  the app crashed on teardown with `'dict' object has no attribute '_clear'`. Renamed.
- **`ToolCallList` could not be built outside an app.** `DataTable.add_column`
  measures its label against `self.app.console`, so the columns cannot be added in
  `__init__`. They are added in `on_mount`, and events arriving before that are
  buffered rather than dropped.

### Unverified

- **No reference workflow has met a real model.** Every test here runs under
  `FakeProvider`. The real path — `OpenRouterProvider` built inside `build(config)`,
  a real model choosing real tools against a real repository — has been exercised by
  hand only through the scripted path. §9.3 tasks 1 and 2 against a real model with
  committed event logs is M7's exit test and is the honest version of R-A-1's
  "completes at least three scripted real tasks".
- **§9.3 task 3 (`web_fetch` a changelog, summarize into `NOTES.md`) is not run.**
  Plan decision D4 defers it: there is no `SERPER_API_KEY` on this machine and no
  network test in the offline suite. `web_fetch` itself is tested at M1.
- **The TUIs have never been drawn on a real terminal.** Every UI assertion here runs
  under `App.run_test()`, which composes and renders but has no terminal to write to.
  Colours, the diff's Pygments theme against a real background, and the grid at a
  terminal width other than 80 are unverified.
- **`--resume` is untested on the coding agent.** `CodingSession.load()` rebuilds the
  graph from `(import_path, config)` and is the same path M5 tested on fusion, but no
  test drives `azc --resume`, and the known issue below about a node-level delegate at
  process death applies to it.
- **The installed console scripts were verified on Windows only**, and `azc.exe` only
  in `--headless` mode. The TUI entry point is exercised in-process by the acceptance
  tests, not through the installed script.
- **`gc.freeze()` is verified to freeze**, not to help. M4's measurement of a 40–90 ms
  gen-2 collection was taken before it had a caller; nothing re-measures R-U-4 with a
  frozen heap over a long session.

---

## M7 — done

The last milestone, and the only one whose exit test needed the network. v1 is
complete: `docs/architecture.md` exists and is checked against the tree, R-X-7's
per-layer READMEs and mkdocs tree are in place, and spec §9.3 tasks 1 and 2 have been
run against `anthropic/claude-haiku-4.5` with their complete event logs committed as
fixtures.

**1091 offline tests, 3 skipped** (POSIX/symlink only), 0 ruff findings, 0 pyright
errors, 6/6 import contracts, `mkdocs build --strict` clean. M6 handed over 1062
tests; M7 adds 29.

### What was built, and where

**The acceptance run (`scripts/run_acceptance.py`, `tests/fixtures/acceptance/`).**
The runner creates a fresh `sample_repo`, runs `azc --headless --auto --model
anthropic/claude-haiku-4.5 --log ...` against it as a subprocess, and records three
fixtures per task:

| fixture | why it exists |
|---|---|
| `<task>.jsonl` | the complete event log, unfiltered. This is what spec §9.3 is scanned over, and R-X-3's claim is that the log is complete — so nothing is stripped. |
| `<task>.workspace.json` | every file in the workspace afterwards. The log proves *how* the agent worked; only this proves the task was done, because the workspace is a temp directory that CI will never see. |
| `<task>.meta.json` | model, timestamp, exit code, wall time, and the agent's final answer. |

What the two recorded runs did:

| task | wall | events | tools | shell |
|---|---|---|---|---|
| 1 — add a `--verbose` flag to `cli.py`, document it in the README | 9.9 s | 231 | `glob`×2, `read_file`×2, `edit_file`×2 | none |
| 2 — replace every call site of `deprecated_fn`, then run the tests | 15.6 s | 308 | `read_file`×6, `edit_file`×3, `grep`×2, `glob`×1 | `pytest -v` |

Both scans are clean: only built-in tools, no `ToolError(kind="internal")`, and the
one `shell` call is the one thing only a shell can do. Task 2's single shell command
is `pytest -v`, which is the point of the task.

**The assertion over them (`tests/test_acceptance_real_model.py`, 11 tests).**
Properties, never diffs, because a real model does not reproduce a fixture's calls:
the flag is registered with argparse (checked by walking the AST, not by matching
text) and used inside `main`; the README still has its `## Usage` section; no call
site of `deprecated_fn` survives in either caller while its definition does; every
file still parses; the scan is clean; no tool call names a path containing `..`.
Three tests exist to stop the fixture being satisfiable by a cheaper thing than a
real run — every `ModelCallCompleted` must carry non-zero usage totalling over 1 000
tokens, the run must reach `COMPLETED`, and `scan_run` over the *committed* log must
still catch an injected `find . -name '*.py'`.

**`docs/architecture.md` and its check (R-X-5).** One diagram covering all six tiers,
a table of the five cross-layer seams with who consumes and who implements each, a
module-by-module map of every layer, a walk through one turn end to end, and a table
of where each requirement is enforced. `tests/test_docs_architecture.py` (15 tests)
checks it in both directions:

- every `azalabscode.*` dotted name in the document resolves — the longest importable
  prefix is imported and the remainder must be real attributes, so
  `azalabscode.control.gate.RuntimePermissionGate` fails the moment that class is
  renamed;
- every package in the tree and every top-level module is named in the document;
- the layer order in the document equals `import-linter`'s `layers` contract, parsed
  from `pyproject.toml` — the picture and the enforcement cannot disagree;
- all five contract protocols are named;
- every repo path the document cites exists.

**R-X-7.** A `README.md` in each of the five layer packages — `providers`, `tools`,
`workflows`, `control`, `tui` — each covering that layer's API, the behaviors that
are the reason its code is longer than an SDK call, and its traps. `docs/index.md`,
`docs/writing-a-workflow.md` and `docs/reference-workflows.md` are the reader's
entry points; `mkdocs.yml` builds the tree. The docstrings half was already satisfied
at M6 and is now asserted (`test_every_public_symbol_has_a_docstring`, over each
layer's `__all__`, which is R-X-6's definition of public).

**`tests/test_docs_examples.py`.** Every fenced Python block in `docs/` preceded by a
`<!-- runnable -->` marker is compiled and executed, and each carries its own
assertion. The two examples on `docs/writing-a-workflow.md` — a two-stage pipeline
and an agent calling `read_file` through a real dispatcher — are therefore proven to
run, on `FakeProvider`, with no key and no network. A renamed keyword argument now
breaks the docs in the same command as it breaks the code. This is not busywork: the
first draft of the second example was wrong (`ToolContext(workspace=...)`), and this
is what caught it.

### Decisions

- **D-M7-1. mkdocs is an optional extra, not a dev dependency.** R-X-7 asks for "a
  `docs/` tree built with mkdocs" but spec §3's dependency list does not include it,
  and no test imports it. `pip install -e ".[docs]"` gets `mkdocs` and
  `mkdocs-material`; `mkdocs build --strict` is green. The tree and the nav are
  checked by `tests/test_docs_architecture.py` whether or not mkdocs is installed, so
  a machine without the extra still catches a page that was written and never linked.

- **D-M7-2. The acceptance workspace lives outside the repository.** The first
  recorded run put the sample repo under `.tmp_ws/`, inside this project. `pytest`
  then walked up to the root `pyproject.toml`, picked up its `addopts`, and failed
  for a reason that had nothing to do with the task — which sent the model hunting
  for config files with `find . -name "pyproject.toml" ... | head -20`, two genuine
  §9.3 violations. The workspace moved to the system temp directory and the venv's
  `Scripts` went on the front of the child's `PATH` so `python` and `pytest` resolve
  to this project's interpreter. The second run was clean with no change to the
  harness, the tool descriptions or the check. **This is a confounder removed, not a
  test weakened**: a sample repo nested inside another Python project is not the
  "small sample repo" spec §9.3 describes, and the first log is what proved it.

- **D-M7-3. The failing first run was not answered by editing the check or the
  prompt.** The `shell` description already says "glob over `find`". The right
  response to a violation is to ask whether the model was wrong or the environment
  was, and here it was the environment. `BANNED_SHELL_COMMANDS` and `scan_run` are
  untouched from M6.

- **D-M7-4. The docs assert on properties and are executed where they can be.**
  Two mechanisms, because docs rot in two ways: `test_docs_architecture` catches a
  document that names something that no longer exists, and `test_docs_examples`
  catches an example that no longer runs. Blocks without the `<!-- runnable -->`
  marker are illustrative — they may name a `MyConfig` that does not exist — and
  running them would prove nothing.

- **D-M7-5. §9.3 task 3 stays out.** Plan decision D4 defers it and the M7 exit test
  names tasks 1 and 2. It needs a stable public URL to fetch, and a committed fixture
  whose content is somebody else's changelog is a fixture that breaks when they edit
  it. `web_fetch` itself has had unit coverage since M1.

- **D-M7-6. The diagram is checked at package granularity, not module.** R-X-5 asks
  for "every top-level module and every cross-layer interface", and that is what is
  *required* to appear. Every module in the tree is nevertheless named in the
  document's per-layer sections, and the forward check proves each one exists — so a
  deleted module breaks the docs test, while an added leaf module does not. The
  stricter reading would make every new file a docs edit, which buys nothing that
  `exhaustive = true` does not already buy for packages.

### The exit test

> R-X-5, R-X-7, §9.3 tasks 1 and 2 against a real model with event logs committed.

```console
$ .venv/Scripts/python -m pytest -q
1091 passed, 3 skipped

$ .venv/Scripts/python -m pytest tests/test_docs_architecture.py tests/test_docs_examples.py -q
18 passed                                        # R-X-5, and R-X-7's checkable half

$ .venv/Scripts/python -m pytest tests/test_acceptance_real_model.py -q
11 passed                                        # §9.3 tasks 1 and 2, over committed logs

$ .venv/Scripts/lint-imports
Contracts: 6 kept, 0 broken.

$ .venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format --check .
All checks passed!  157 files already formatted

$ .venv/Scripts/python -m pyright
0 errors, 0 warnings, 0 informations

$ .venv/Scripts/python -m pip install -e ".[docs]"
$ .venv/Scripts/python -m mkdocs build --strict
INFO - Documentation built in 1.48 seconds
```

The real-model runs themselves (network, a key, and about 25 s of wall time):

```console
$ .venv/Scripts/python scripts/run_acceptance.py
[task1] exit=0 in 9.9s
[task1] 231 events, scan: clean
[task1] tools: ['edit_file', 'glob', 'read_file']
[task2] exit=0 in 15.6s
[task2] 308 events, scan: clean
[task2] tools: ['edit_file', 'glob', 'grep', 'read_file', 'shell']
```

R-X-7's verification is **I**, inspection, and the spec calls it a gate for "done"
rather than for merge. The mechanical parts — a README per layer, a docstring on
every public symbol, a tree mkdocs builds — are asserted. Whether the prose is any
good is the part inspection is for.

### Unverified

- **One sampling per task.** Each fixture is a single run of one model on one day.
  It is evidence that the built-in tools are what a real model reaches for, not a
  distribution. Re-running `scripts/run_acceptance.py` produces a different log that
  should still pass, and any run that does not is worth reading before it is
  discarded.
- **§9.3 task 3 was not run** (D-M7-5). `web_fetch` against a live URL has still
  never appeared in an acceptance log.
- **Only `anthropic/claude-haiku-4.5`.** A larger model is likelier to pass and a
  smaller one likelier to reach for `shell`; neither was tried. `--model` is a flag
  on the runner precisely so that is one command.
- **The prose was not read by anyone else.** R-X-7 is an inspection requirement and
  the only inspector available was its author.
- **`mkdocs build` was run, `mkdocs serve` was not**, and the built site was never
  opened in a browser. Material for MkDocs also prints a warning that MkDocs 2.0 will
  break its plugin system with no migration path; the docs config here uses no
  plugins, so the exposure is the theme alone.
- **Everything M6 listed as unverified still is** — no reference workflow TUI has been
  drawn on a real terminal, `azc --resume` has no test, and the console scripts were
  verified on Windows only. M7 added no coverage there.
