# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The interpreter is the project venv. Always call it explicitly.

```console
$ .venv/Scripts/python -m pytest                    # offline suite (~1090 tests)
$ .venv/Scripts/python -m pytest -m network         # adds the live OpenRouter test
$ .venv/Scripts/python -m pytest tests/test_session.py::test_name    # one test
$ .venv/Scripts/python -m pytest -k quiescence      # by name
$ .venv/Scripts/lint-imports                        # the six layer contracts
$ .venv/Scripts/python -m ruff check .
$ .venv/Scripts/python -m ruff format --check .
$ .venv/Scripts/python -m pyright
$ .venv/Scripts/python -m mkdocs build --strict     # needs pip install -e ".[docs]"
```

`filterwarnings = ["error"]` and `--strict-markers` are on: a warning fails the run.
`asyncio_mode = "auto"`, so async tests need no decorator.

Regenerate the built-in tool schema snapshot after an intentional description or
params change, then read the diff before committing:

```console
$ .venv/Scripts/python -m tests.test_tool_schemas
```

Record real-model acceptance fixtures (spends OpenRouter credit):

```console
$ .venv/Scripts/python scripts/run_acceptance.py [task1|task2] [--model ...]
```

The three console scripts (`azc`, `azalabs-fusion`, `azalabs-inspector`) are the
reference workflows; each runs a TUI by default and takes `--headless`.

## Architecture

Read `docs/architecture.md` first — it is the map, and `tests/test_docs_architecture.py`
keeps it matching the tree. Then the `README.md` inside the layer package you are
touching. `intent/spec.md` is the governing requirements document; `intent/plan.md`
records the numbered deltas where this implementation deliberately differs from it.

```
tui  ->  control  ->  workflows  ->  providers | tools
                                          |
                                      contracts
                                          |
    events -> messages -> permissions -> toolio -> content -> errors
                                                         -> cancellation | ids | runstate | schema | sync
```

**Vocabulary goes down, policy stays up.** Each leaf module holds shared types; each
upper layer holds the decisions about them. `permissions` knows what an
`ApprovalRequest` is; `control` decides what to do with one.

Where data must flow upward, a protocol in `azalabscode/contracts.py` inverts the
import: `PermissionGate` (dispatcher asks control), `RunControl` (runner declares
safe points, control decides), `Delegator` (the `delegate` tool spawns a subagent
without `tools` learning what an `AgentSpec` is), `ApprovalHandler`, `EventSink`.
Each of those is optional — passing `control=None` or `AllowAllGate()` runs the
layer standalone, and tests rely on that.

`workflows/` at the repo root (the three reference workflows) is packaged but sits
*outside* `azalabscode/` so import-linter contract 5 proves a workflow reaches the
harness only through the public API: `azalabscode/__init__.py` and each layer's own
`__init__.py`. Naming a private module from `workflows/` fails the contract.

## Invariants that a plausible-looking edit will break

These are enforced by tests, but the failure is usually far from the edit.

- **Layer changes are three-file changes.** Adding a package under `azalabscode/`
  requires editing the `layers` list in `pyproject.toml` (contract 1 is
  `exhaustive = true`) *and* the diagram plus layer block in `docs/architecture.md`,
  in the same commit. Every `azalabscode.*` name written in that doc must exist.
- **Control: write first, park second, and park outside the checkpoint lock.** A
  parked agent holding the lock deadlocks every other checkpoint, so PAUSED is never
  reached. PAUSED means "a pause was requested and the non-quiescent agent count is
  zero" — not `parked == active`. Child registration and the parent's flip to
  `blocked_on_child` happen synchronously *before* `tg.create_task`; do not tidy that
  ordering.
- **Transcript:** for every `ToolCallPart`, exactly one later `ToolResultMessage` with
  the same `call_id`, in call order, never completion order. Results stage in
  `AgentState.pending_results` and are materialized by `finalize_turn`; never append
  as they finish.
- **Cancellation (`workflows/step.py`):** never `except Exception` around a step body
  (`CancelledError` is a `BaseException`); a missing `cancel_reason` means the
  cancellation was not ours, so re-raise; absorb only after `uncancel()` reconciles
  to zero.
- **Textual imports live under `azalabscode/tui/` and nowhere else.** A test asserts
  `import azalabscode` does not pull in Textual. Contract 6 also forbids `tui` from
  naming `workflows`, `providers` or `tools` directly — the UI sees events and the
  `Controller`.
- **Tool defaults fail closed:** not read-only, not concurrency-safe, approval
  required. `retry` must be 0 whenever `approval != never`. `validate_params` is a
  pure pre-check that runs *before* the approval prompt, deliberately separate from
  `run`.
- **Widgets render `ToolResult.display`.** Never re-parse `ToolResult.content` — that
  string is written for a model.
- **Node ids come from required `name` arguments**, never from `inspect.stack()`.
  `@` and `/` are reserved in a name. Node outputs must be JSON-native, and cycles
  live inside nodes so every node boundary is a checkpoint boundary.
- **Docs examples run.** A fenced Python block in `docs/` preceded by `<!-- runnable -->`
  is exec'd by `tests/test_docs_examples.py` and carries its own assertions.

## Platform

Windows is the dev and test platform; the POSIX path is written and smoke-tested.
`tests/conftest.py` fails collection if the event-loop policy cannot spawn
subprocesses (`WindowsSelectorEventLoopPolicy` breaks the `shell` tool). OS-specific
behavior belongs in `azalabscode/tools/platform.py`, not in a tool.

`claude-code-main/` is a gitignored TypeScript reference codebase, excluded from
ruff and pyright. It is a design reference only — nothing is transliterated from it.

## Status

M0–M7 are done; v1 is complete and there is no next milestone in `intent/plan.md`.
`progress.md` logs what each milestone built; `handoff.md` is the state at the end of
M7. `run_phases.py` is the milestone driver, not part of the deliverable.
