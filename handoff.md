# Handoff → after M7

**M0–M7 are done. v1 is complete.** There is no next milestone in `intent/plan.md`;
whatever comes next is new work, not a continuation.

**1091 offline tests, 3 skipped** (POSIX/symlink only), 0 ruff findings, 0 pyright
errors, 6/6 import contracts, `mkdocs build --strict` clean. Nothing is failing and
nothing in the tree is a stub.

Every requirement in `intent/spec.md` §2.1 now has an implementation and a test.
Read `docs/architecture.md` first — it is the map, and it is checked against the tree
by `tests/test_docs_architecture.py`, so it cannot be out of date. Then the
`README.md` inside whichever of the five layer packages you are about to touch.

## Where the documentation is

| You want | Read |
|---|---|
| the layer map, the five seams, one turn end to end | `docs/architecture.md` |
| how to start, the three commands | `docs/index.md` |
| how to build a workflow, with runnable examples | `docs/writing-a-workflow.md` |
| what each shipped workflow demonstrates | `docs/reference-workflows.md` |
| one layer in depth, and its traps | `azalabscode/<layer>/README.md` |
| the requirements, and the deltas from them | `intent/spec.md`, `intent/plan.md` |
| what each milestone actually did | `progress.md` |

`mkdocs build --strict` needs `pip install -e ".[docs]"`. mkdocs is deliberately not
in `dev` (decision D-M7-1).

## Architecture as it now stands

```
tui  →  control  →  workflows  →  providers | tools
                                       ↓
                                   contracts
                                       ↓
   events → messages → permissions → toolio → content → errors
                                                          ↓
                              cancellation | ids | runstate | schema | sync
```

Unchanged in shape since M0. The authoritative copy is in `docs/architecture.md`,
and the layer order there is asserted equal to `import-linter`'s `layers` contract in
`pyproject.toml`.

```
azalabscode/
  README.md per layer package                                   (M7)
  tui/       control/  workflows/  providers/  tools/
  contracts.py + the eleven leaf modules

workflows/                                        (outside the package)
  cli_support.py  tooling.py
  coding_agent/  fusion/  inspector/

docs/        architecture.md index.md writing-a-workflow.md
             reference-workflows.md                              (M7)
mkdocs.yml                                                       (M7)
scripts/     stream_openrouter.py  run_acceptance.py             (M7)
tests/       acceptance.py + 50 test modules
  fixtures/  tool_schemas.json  kill_script.json
             acceptance/{task1,task2}.{jsonl,workspace.json,meta.json}   (M7)
```

Installed console scripts: `azc`, `azalabs-fusion`, `azalabs-inspector`.

## What M7 changed

**Nothing in `azalabscode/`.** No source file in the package was edited: M7 added
five `README.md`s beside them and nothing else. `pyproject.toml` gained one optional
dependency group (`docs`). The rest is `docs/`, `scripts/run_acceptance.py`, three
test modules and six fixtures.

That is worth knowing before you go looking for a behavior change to explain a
different test result: there is none.

## The three new test modules, and what they will do to you

**`tests/test_docs_architecture.py`** — `docs/architecture.md` is a test fixture now.

- **Adding a package under `azalabscode/` breaks it** until the package is named in
  the document. Same bargain as `exhaustive = true` in the layers contract, and the
  same fix: edit the diagram in the same commit.
- **Renaming a public class breaks it** if the document names it. The check imports
  the longest importable prefix of each dotted name and then requires the rest to be
  real attributes, so `azalabscode.control.gate.RuntimePermissionGate` is checked as a
  class, not as a string.
- **Reordering the `layers` contract breaks it** unless the fenced block after the
  `<!-- layers -->` marker in the document is reordered to match.
- **Deleting or renaming a test module breaks it** if the document's "where each
  requirement is enforced" table cites it.
- **Adding a `docs/*.md` page breaks it** until the page is in `mkdocs.yml`'s nav.
- It also asserts a docstring on every symbol in each layer's `__all__`. A new public
  export with no docstring fails here, not in review.

**`tests/test_docs_examples.py`** — any fenced ```python block in `docs/` preceded by
`<!-- runnable -->` is compiled and executed, and carries its own `assert`. Marked
examples must stay self-contained: `FakeProvider`, `tempfile`, no key, no network,
and fast. An example that should not run must not carry the marker.

**`tests/test_acceptance_real_model.py`** — assertions over two committed real-model
runs. It never calls a model; it replays `tests/fixtures/acceptance/*.jsonl`. Nothing
here needs a key.

## Regenerating the acceptance fixtures

```console
$ .venv/Scripts/python scripts/run_acceptance.py            # both tasks, ~25 s, needs a key
$ .venv/Scripts/python scripts/run_acceptance.py task2 --model openai/gpt-4o-mini
```

Five things about that runner are deliberate and easy to undo by accident:

1. **`--auto`, always.** A headless `manual` run installs `StdinApprovalHandler` and
   blocks in a worker thread at the first `edit_file`, with nothing on screen to say
   why.
2. **The workspace is under the system temp directory, not `.tmp_ws/`.** A sample repo
   nested inside this project makes `pytest` walk up to the root `pyproject.toml` and
   pick up its `addopts`. That is what made the first recorded run fail §9.3 — the
   model went looking for config files with `find`. See progress D-M7-2.
3. **The venv's `Scripts` goes on the front of the child's `PATH`**, so `python` and
   `pytest` inside the agent's `shell` calls are this project's. Without it the agent
   finds the system Python, discovers pytest is missing, and burns turns on
   `pip install` — an environment problem measured as a tool-choice one.
4. **The log is committed unfiltered.** R-X-3's claim is that the event stream is
   complete; a trimmed log would not test it. 70 KB and 118 KB. Leave them whole.
5. **The workspace snapshot is the only proof the task was done**, because the
   workspace itself is a temp directory. It excludes dot-directories and the log.

If a re-run fails the scan, **read the log before touching anything**. The question is
whether the model was wrong or the environment was; `BANNED_SHELL_COMMANDS` and
`scan_run` in `tests/acceptance.py` are M6 code and have been right both times.

## Known issues

- Nothing failing.
- **A `delegate` in flight from a `Func`-style node at process death is not resumed.**
  `resume.reconcile` looks the inflight step's `agent_id` up in `session.agents`; for
  a node-level delegate that key is a node id, so no `ResumableDelegate` is queued.
  Unchanged since M5. The coding agent delegates through the *tool* (inside
  `AgentLoop`), where spec delta 16 does hold, so it does not bite there.
- **No cross-process resume of a graph.** Every save/load outside `tests/kill_child.py`
  is in one interpreter, and the kill test drives an `AgentLoop`, not a graph.
  `azc --resume` is in the same position: the code path is M5's, tested on fusion, but
  nothing drives it through the CLI.
- **Node retries are a seam with no caller.** `NodeRecord.attempt` is respected by the
  memo, the state restore and the events; nothing increments it.
- **Three tests skip on Windows**: the real POSIX `killpg` tree kill and two symlink
  containment tests. See `progress.md` §M1 Unverified.
- **§9.3 task 3 has never been run** (`web_fetch` a changelog, summarize into
  `NOTES.md`). Plan decision D4 deferred it and the M7 exit test excluded it, so
  `web_fetch` has unit coverage but has never appeared in an acceptance log.
- **The TUIs have never been drawn on a real terminal.** Every UI assertion runs under
  `App.run_test()`, which composes and renders but writes to no terminal.
- **Do not write Python or long Markdown with `bash <<'EOF'` heredocs in this repo.**
  It collapses `\\` and `\n` in the payload; it has corrupted a regex class, a
  docstring, a Markdown append and an f-string across M0–M4 and broke a string
  replacement in M6. Use the Write/Edit tools, or `.venv/Scripts/python - <<'PY'`
  where the escapes live in a normal Python string literal. (M7 appended to
  `progress.md` with a five-line Python script for exactly this reason.)
- `run_phases.py`, `run_sessions.log`, `STOP`, `.import_linter_cache/` and `.tmp_ws/`
  are in `.gitignore`. They are build tooling, not deliverables.

## Traps that have not changed

**Control.** Write first, park second, and park *outside* `_cp_lock`. Register a
child before creating its task. `AgentState` is claimed once —
`RunControl.restored_agent(agent_id)` pops. PAUSED ⇔ pause requested and
`_nonquiescent == 0`.

**Cancellation, the three rules.** Never `except Exception` around a step body
(`CancelledError` is a `BaseException` in 3.12). `cancel_reason is None` ⇒ not ours ⇒
re-raise. Absorb only after reconciling the cancel count — `should_absorb()` in
`workflows/step.py` is the only implementation.

**Graph.** An `AgentNode`/`ToolingAgentNode` agent id must not equal a node id
(`compile()` refuses it): the coding agent's node is `loop` and its agent is `main`,
the inspector's are `agent` and `main`. Node outputs must be JSON-native. `@` and `/`
are reserved in a builder name. The runner's own `node_id` is `@<graph name>` and
appears on `Checkpoint` events — skip anything starting with `@`.

**Providers must be constructed inside `build(config)`** (spec delta 21), or a session
cannot be rebuilt from `(import_path, config)` alone.

**TUI.** Textual system bindings win silently (`ctrl+p` is the command palette, moved
to `ctrl+backslash`; `ctrl+c` needs `priority=True`). A `ModalScreen` stops binding
lookup — app-level bindings that must survive a modal go in `HARNESS_BINDINGS`.
`refresh(layout=True)` is the expensive call; give a streaming widget a stable box.
Do not drain a stream buffer on a timer — `StreamPane` drains in `render()`. Do not
take `_command_lock` from a UI callback. A loaded run has agents and no events for
them — `HarnessApp.seed_from_controller()`, and `Session.event_seq` seeds the new bus.
`ToolResult.display` is the widget-facing rendering; never re-parse
`ToolResult.content`. Do not name a widget attribute after a Textual one (`_nodes`,
`tree`). `DataTable.add_column` needs an active app, so columns go in `on_mount`.
`ApprovalResolved` has no `call_id` — keep a `request_id -> call_id` index. A
delegated agent emits `AgentSpawned` twice and any fold of it must be sticky. A widget
mounted after `attach()` must be registered with `register_consumer`.

## The three committed fixtures

Each fails loudly by design when the thing it snapshots changes. Read the diff before
committing any of them.

```console
$ .venv/Scripts/python -m tests.test_tool_schemas    # tests/fixtures/tool_schemas.json
$ .venv/Scripts/python -m tests.record_kill_script   # tests/fixtures/kill_script.json
$ .venv/Scripts/python scripts/run_acceptance.py     # tests/fixtures/acceptance/*
```

`shell` is deliberately absent from the schema snapshot: its description is rendered
per platform. The kill script's keys are `ModelRequest.fingerprint()` hashes, so
changing the system prompt, the task or the toolset in `tests/resumable.py`
invalidates them.

## Testing

Four rigs.

`tests/harness.py` — the **agent** rig: `build_rig(turns, workspace=..., mode=...,
handler=..., spec=..., specs=..., extra_tools=..., max_parallel=..., session_dir=...,
autosave=..., workflow=...)` returns a `Rig` with a real `Controller`, a real
`AgentLoop`, a `FakeProvider` and two recording fake tools.

`tests/graphrig.py` — the **graph** rig: `build_graph_rig(workflow, session_dir=...,
mode=..., import_path=..., config=..., strict_graph_hash=..., bus=...)`. Ships
`SteppingNode`, the only way to stop a node *mid-body* deterministically.

`tests/acceptance.py` — the **acceptance** rig: `sample_repo(root)` writes the
six-file sample project the §9.3 tasks operate on; `scan_run(events,
allowed_tools=...)` applies §9.3's pass criteria and returns a `RunScan` whose
`.report()` is the assertion message. `shell_heads(command)` is the tokeniser
underneath, covering POSIX and PowerShell command names. Pass `ALL_TOOLS`, never a
registry built on this machine — `web_search` drops out without a `SERPER_API_KEY`.

`tests/resumable.py` + `tests/kill_child.py` — the out-of-process rig.

Habits worth keeping:

- **Every wait in `tests/**` carries its own bound.** There is still no
  `pytest-timeout`. `tests/**` has an `ASYNC109` per-file ignore.
- **Prove a regression test non-vacuous by removing the fix and watching it fail.**
  M7 did it for the acceptance scan by injecting a `find` into the committed log.
- **Check what a test is actually asserting on.** M6's fusion resume test asserts on
  the *second* controller's panes; asserting on the first would prove nothing.
- **A modal nobody answers is a hang, not a failure.** A TUI test that triggers an
  approval must answer every modal the script will raise.
- **Assert on properties for anything a real model produced.** Exact text is a record
  of one sampling.

## Environment

Python 3.12.0 in `.venv`, project installed editable (`pip install -e . --no-deps`),
which is what puts `azc`, `azalabs-fusion` and `azalabs-inspector` on the path.
Installed: pydantic 2.13.5, httpx 0.28.1 + h2, anyio, textual 8.2.8, rich 15.0.0,
trafilatura, typer 0.27.2, pytest 8.4.2 + pytest-asyncio 1.4.0 +
pytest-textual-snapshot 1.1.0, respx, hypothesis, import-linter 2.15, ruff, pyright,
and now mkdocs 1.6.1 + mkdocs-material. No `pytest-timeout`. `OPENROUTER_API_KEY` is
in `.env` (gitignored) as `KEY = "value"` — note the spaces around `=`, which
`workflows/cli_support.py:read_dotenv` and `tests/conftest.py:load_dotenv` both
tolerate. No `SERPER_API_KEY`. `rg` is on PATH; `pwsh` 7.6.3 and Windows PowerShell
5.1 are both present. `WindowsProactorEventLoopPolicy` is pinned by a conftest
assertion (delta 22).

The Windows loop clock resolves to about 15.6 ms. Anything that needs a *rate* must
sleep to an absolute deadline, not for a fixed interval — that is what
`ScriptedTurn.chunk_rate_hz` does (delta 23).

Live calls use `anthropic/claude-haiku-4.5`. `anthropic/claude-3.5-haiku` does not
resolve on this account — check `/models` before hard-coding a model id.

## Performance, unchanged from M4

R-U-4 on Windows 11 / Python 3.12.0 / Textual 8.2.8, four `FakeProvider` streams at
200 deltas/s through four real `AgentLoop`s:

| configuration | max latency | p95 | loop lag (max) |
|---|---|---|---|
| 4 tail panes (the fusion layout, R-U-4's own) | 34–47 ms | 34–42 ms | 12 ms |
| 6 tail panes | 58–63 ms | 44–48 ms | 12 ms |
| 4 growing panes | 53 ms | 52 ms | 13 ms |
| 6 growing panes | 87–90 ms | 70–87 ms | 44 ms |
| 1 growing pane (the transcript shape) | well inside | — | 13 ms |

Layout dominates the tail; garbage collection dominates anything measured over a
large heap. Fusion's panes are `tail=True` for the first reason, and every CLI calls
`prepare_process()` — `gc.freeze()` plus the session-directory sweep — for the second.

## If you are picking up new work

The obvious candidates, in the order the known issues above justify them:

1. **Cross-process resume through a CLI.** `azc --resume` and
   `azalabs-fusion --resume` are untested end to end, and the node-level delegate gap
   is in the same area.
2. **§9.3 task 3**, with a URL you control so the fixture does not rot.
3. **MCP tools.** Out of scope for v1, and `ToolContext` was kept to leaf types so an
   adapter can wrap remote tools without touching core (R-T-10). Nothing blocks it.
4. **A second provider.** The interface has never been implemented twice, which is
   the only real test of whether it leaks OpenRouter.

None of these is planned. `intent/plan.md` ends at M7.
