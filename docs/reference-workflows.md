# The reference workflows

Three workflows ship with the harness. They aren't just demos — they're the acceptance
suite for intent success criterion 1. Each is built **entirely on the public API, with
no changes to core** (an import-linter contract proves it), and each has its own TUI and
a `--headless` mode.

They live in `workflows/`, outside the `azalabscode` package, which is what keeps that
proof honest.

```
workflows/
  cli_support.py   startup, config resolution, API-key resolution
  tooling.py       ToolingAgentNode -- the node both tool-using workflows run on
  coding_agent/    prompts.py workflow.py session.py app.py cli.py
  fusion/          workflow.py headless.py app.py cli.py
  inspector/       workflow.py headless.py app.py cli.py
```

Each one has a `build(config)` — the importable factory that a session records — along
with a Pydantic config model, a Typer CLI, and an app. Each accepts `--config`,
`--session-dir`, `--log`, and `--headless`.

## `azc` — the coding agent

An interactive session in a working directory. The agent reads, edits, and runs code
with the built-in tools. Destructive calls stop at an approval modal that shows a diff.
It can delegate to subagents, and the TUI shows their activity separately from the main
agent's.

```console
$ azc                                     # a session in this directory
$ azc "add a --verbose flag to cli.py"    # ... starting from a task
$ azc --headless --auto "fix the failing test"
$ azc --resume .azalabscode/session.json
```

**Layout.** A `Transcript` on the left; an `AgentTree`, a `ToolCallList`, and a
`ToolCallDetail` on the right; a `DiffView`, a `PromptInput`, a hidden `EventLog`, and a
`RunStatusBar` below. `ctrl+t` toggles the permission mode, `ctrl+p` pauses, `escape`
interrupts and lets you type a correction, `ctrl+s` saves, and `ctrl+o` reopens.

**Graph.** A single node — `loop`, a `ToolingAgentNode` — whose agent is `main`. It has
three subagent specs: `explore` and `review` are read-only and on by default; `edit` is
opt-in, because a subagent that writes is exactly the case the v1 permission model is
weakest on.

**What it proves.** This is the workflow that pushes the essential tools and the
multi-agent model the hardest, and it's the one that makes success criterion 5 — "the
built-in tools are the ones the model reaches for" — measurable, by actually *using* the
harness. `tests/test_acceptance_real_model.py` holds two real-model runs of it, scanned
against spec §9.3.

**`manual` is the default, and it stays the default.** A mistaken prompt costs you a
keystroke; a mistaken `shell` can cost you a repository. `--auto` is how a caller says
otherwise.

## `azalabs-fusion` — model fusion

The same question sent to N models at once, side by side, followed by an analysis stage
and then a synthesis stage.

```console
$ azalabs-fusion "why is my build slow?" -m openai/gpt-5 -m google/gemini-3
$ azalabs-fusion --headless "..." -m anthropic/claude-haiku-4.5
```

**Graph.** `models` (a fan-out, one `ModelCall` per branch) → `join` → `analyze` →
`synthesize`. The branch node ids are `models/<slug>`, slugged from the model name
rather than its index — because an index shifts the moment a model is added to the middle
of the list, and then every session saved before that stops lining up with the graph.

**Layout.** `SplitPanes` with one `StreamPane` per branch across the top, a
`StagePipeline` beneath them, and a stage pane for the analysis and synthesis output.
The branch panes are `tail=True`, because layout dominates the streaming latency tail —
this is the exact layout R-U-4 is measured against.

**What it proves.** The concurrency model, the join, and the fact that a non-chat UI
falls right out of the graph's shape. It's also the workflow that the
pause/save/load/resume-mid-fan-out test drives: completed branch outputs survive the
reload, and a `StreamPane` is refilled from the memoized node output, since a loaded run
has no events to rebuild it from.

**Each branch gets its own provider in fake mode.** A single `FakeProvider` hands out
unkeyed turns in order across the whole run, so four branches sharing one script would
interleave by scheduling order. Giving each branch its own provider removes that shared
state, and that's what makes the headless run reproducible enough to assert on.

## `azalabs-inspector` — tool observability

One agent working on a task, with a live inspector that shows every tool call's inputs,
outputs, timing, and errors.

```console
$ azalabs-inspector "where is the retry logic?" --workspace ../repo
$ azalabs-inspector --headless "list every TODO" -t grep -t read_file
```

**Graph.** A single node — `agent`, a `ToolingAgentNode` — whose agent is `main`.
`--tool/-t` restricts the toolset.

**Layout.** A `Transcript` on the left; `ToolCallList` over `ToolCallDetail` on the
right; a `PromptInput`, a togglable raw-JSONL `EventLog`, and a `RunStatusBar`.

**What it proves.** R-X-3's claim that every model call, tool call, node transition,
agent spawn, state change, approval, and checkpoint is a typed event — the event-log
viewer is that claim, on screen. It also proves `ToolResult.display`: the detail pane
renders the structure the tool produced, and never re-parses the text written for the
model.

## The shared parts

`workflows/cli_support.py` holds what all three do at startup: `prepare_process()` (a
`gc.freeze()` — a generation-2 collection costs 40–90 ms, most of R-U-4's budget — plus a
sweep of `.tmp` files left behind by a killed process), config-file and flag merging, and
resolving the API key from the environment and then from the nearest `.env`.

`workflows/tooling.py` holds `ToolingAgentNode`, the node that the coding agent and the
inspector share. It builds its own `ToolDispatcher` at run time from
`RunControl.permission_gate`, which is what lets a reloaded session rebuild a tool-using
graph from `(import_path, config)` alone. It runs its loop once per prompt, so an
interactive session stays one agent and one transcript across many tasks.

## Copying one

Start from `workflows/inspector/` — it's the smallest complete example: a config model, a
`build(config)`, a headless entry point, an app, and a CLI. Then read
[Writing a workflow](writing-a-workflow.md) for the rules that are easy to get wrong —
especially constructing providers inside `build()`, and keeping an agent id distinct from
every node id.
