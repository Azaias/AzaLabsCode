# azalabscode

An async agentic harness in Python that you own and can extend. It has four layers —
providers, workflows, control, and the UI — and you can replace any one of them without
touching the others. It also ships nine built-in tools, chosen and tuned to be the ones
a model actually reaches for.

It exists because the frameworks that already do this get in your way: they impose
their own abstractions, hide the agent loop, and ship a UI that assumes one agent in
one chat transcript. Here, every layer is meant to be read and understood, the Python
API is shaped on purpose, and the TUI can render whatever structure a workflow happens
to have.

## Install

```console
$ .venv/Scripts/python -m pip install -e ".[dev]"
$ export OPENROUTER_API_KEY=sk-or-...      # or put it in a .env file
```

Python 3.12 or newer. OpenRouter is the only provider today, and it sits behind a
protocol that keeps its details from leaking into the rest of the system.

## Three commands

```console
$ azc                                    # a coding session in this directory
$ azc "add a --verbose flag to cli.py"   # ... starting from a task
$ azc --headless --auto "fix the failing test"

$ azalabs-fusion "why is my build slow?" -m openai/gpt-5 -m google/gemini-3
$ azalabs-inspector "where is the retry logic?" --workspace ../repo
```

`azc` starts in `manual` mode, where every destructive tool call stops at a modal that
shows you a diff. `ctrl+t` switches to `auto`, `ctrl+p` pauses, `escape` interrupts the
current step so you can type a correction, `ctrl+s` saves, and `ctrl+o` reopens a
session.

## The smallest useful program

```python
import asyncio

from azalabscode import PermissionMode, run_sync
from azalabscode.control import Controller
from azalabscode.providers import OpenRouterProvider
from azalabscode.tools import default_registry
from azalabscode.workflows import AgentSpec, Workflow

wf = Workflow("ask")
wf.agent(
    "loop",
    AgentSpec(agent_id="main", model="anthropic/claude-haiku-4.5", tools=["read_file", "grep"]),
    prompt="Summarize what this repository does.",
)

controller = Controller(wf.compile(), env, mode=PermissionMode.AUTO)
print(run_sync(controller.start()))
```

Each layer also runs on its own: the tool layer dispatches behind `AllowAllGate` with
no controller, the agent loop runs with `control=None`, and a `Runner` walks a graph
with no run around it at all.

## What to read next

| If you want | Read |
|---|---|
| The shape of the system — one diagram and a module map | [Architecture](architecture.md) |
| To build your own workflow on the public API | [Writing a workflow](writing-a-workflow.md) |
| To see what the three shipped workflows demonstrate | [Reference workflows](reference-workflows.md) |
| The detail of one layer | the `README.md` inside `azalabscode/<layer>/` |
| The governing requirements, and where this code departs from them | `intent/spec.md` and `intent/plan.md` |

## What v1 is not

No RAG and no long-term memory. No MCP tools yet, though nothing in the tool interface
rules them out. No web UI. No tracing dashboard — the structured events are in core,
but the thing that consumes them is not. No per-tool permission allowlists beyond the
two modes. It is single-process, single-user, and local.
