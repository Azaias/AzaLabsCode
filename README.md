# azalabscode

An async agentic harness in Python that you own and can extend. It has four layers,
and you can replace any one of them without touching the others.

```
tui  ->  control  ->  workflows  ->  providers | tools
                                          |
                                      contracts          (inversion protocols)
                                          |
    events -> messages -> permissions -> toolio -> content -> {ids, errors,
                                                    cancellation, runstate, schema}
```

The imports only ever point downward. A lower module defines a shared type; the layer
above it decides what to do with that type. When data genuinely needs to travel back
up — the tool dispatcher asking permission, the runner writing a checkpoint — a
protocol in `contracts.py` inverts the dependency so the import can still point down.
There are four such protocols: `PermissionGate`, `RunControl`, `Delegator`, and
`ApprovalHandler`/`EventSink`.

The requirements, the design, and the open questions are in `intent/spec.md`. The
implementation plan, and the places where this code deliberately departs from the
spec, are in `intent/plan.md`.

Start with [`docs/index.md`](docs/index.md). The layer map and the seams between
layers are in [`docs/architecture.md`](docs/architecture.md), and every layer has its
own `README.md` under `azalabscode/<layer>/`.

## Status

v1 is complete — milestones M0 through M7 are all done. Here is what exists today:

- the full leaf tier and `contracts.py`
- **providers:** the `Provider` protocol, `OpenRouterProvider`, and `FakeProvider`
- **tools:** `Tool`, `ToolContext`, and `ToolDispatcher`; the OS-specific code in
  `tools/platform.py`; the nine built-in tools; and the `SearchBackend` protocol with
  its Serper adapter
- **workflows:** the `Workflow` builder and the `Graph` it compiles to; the `Runner`,
  `NodeContext`, and the seven node types (`AgentNode`, `ModelCall`, `FanOut`,
  `Gather`, `Map`, `Func`, `Subgraph`); the `AgentLoop`; the cancellation rules in
  `workflows/step.py`; and the transcript invariant in `workflows/transcript.py`
- **control:** the `Controller`, with pause, resume, interrupt, permission modes, and
  approvals; the run state machine; quiescence tracking; and durable sessions —
  `save`, `load`, resume reconciliation, and the graph-drift check from spec C-2
- **the TUI:** `HarnessApp`, the event router, and the full widget set from spec 8.2:
  `Transcript`, `StreamPane`, `ToolCallList`, `ToolCallDetail`, `DiffView`,
  `AgentTree`, `StagePipeline`, `SplitPanes`, `PromptInput`, `ApprovalModal`, and
  `RunStatusBar`
- **three reference workflows,** each with its own TUI and a `--headless` mode:
  `workflows/coding_agent` (R-A-1), `workflows/fusion` (R-A-2), and
  `workflows/inspector` (R-A-3)
- **`pyproject.toml`,** wiring up ruff, pyright, pytest, the six import-linter
  contracts, and the three console scripts
- **the docs:** the `docs/` tree, a README per layer, the check that keeps the
  architecture diagram matching the actual module tree, and two real-model runs of the
  spec 9.3 acceptance tasks, with their full event logs committed as fixtures

Every layer also runs on its own. The tool layer runs behind `AllowAllGate` with no
controller; the agent loop runs with `control=None` when you just want an agent and no
run around it; and a `Runner` walks a graph with `control=None` when you want no run at
all.

## Setup

```console
$ .venv/Scripts/python -m pip install -e ".[dev]"
$ export OPENROUTER_API_KEY=sk-or-...      # or put it in .env
```

## Checks

```console
$ .venv/Scripts/python -m pytest            # unit tests, no network
$ .venv/Scripts/python -m pytest -m network # adds the live OpenRouter test
$ .venv/Scripts/lint-imports                # the six layer contracts
$ .venv/Scripts/python -m ruff check .
$ .venv/Scripts/python -m ruff format --check .
$ .venv/Scripts/python -m pyright
$ .venv/Scripts/python -m mkdocs build --strict   # needs `pip install -e ".[docs]"`
```

## The reference workflows

Installing the package adds three commands to your path. Each one runs a TUI by
default and accepts `--headless` (R-U-7). Each also takes `--config` (a JSON file),
`--session-dir`, `--resume`, and `--log`.

```console
$ azc                                    # a coding session in this directory
$ azc "add a --verbose flag to cli.py"   # ... with a task to start from
$ azc --headless --auto "fix the failing test"

$ azalabs-fusion "why is my build slow?" -m openai/gpt-5 -m google/gemini-3
$ azalabs-inspector "where is the retry logic?" --workspace ../repo
```

`azc` starts in `manual` mode, where every destructive tool call stops at a modal that
shows you a diff. The keys: `ctrl+t` switches to `auto`, `ctrl+p` pauses, `escape`
interrupts the current step so you can type a correction, `ctrl+s` saves, and `ctrl+o`
reopens a session.

## Streaming a real completion

```console
$ .venv/Scripts/python scripts/stream_openrouter.py "Explain a ULID in one sentence."
```

This prints tokens as they arrive, then the finish reason, the usage, and the cost.
