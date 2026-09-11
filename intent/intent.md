# Intent: An owned, extensible agentic harness in Python

## Problem

Existing agent frameworks (LangGraph, pydantic-ai, smolagents, the vendor SDKs) each impose their own abstractions, hide their loop internals, and ship UIs that assume a single agent in a chat transcript. Building a non-trivial workflow on top of them — several models running in parallel, a staged analysis-and-synthesis pipeline, deep inspection of tool calls — means fighting the framework's opinions and living with whatever its UI can show.

The core motivation is ownership. I want a harness where I understand every layer, where the Python API is mine to shape, and where the UI is expressive enough that custom workflows are actually usable rather than merely runnable.

## Who this is for

Me. A single developer, comfortable in Python, building workflows for my own use.

That does not lower the bar. The system should be stable, well-documented, and clean enough that I would be comfortable handing it to someone else. "Solo user" means no need for backwards-compatibility guarantees or a plugin marketplace; it does not mean fragile internals or undocumented behavior.

## What we are building

An async Python harness with four separable layers, each replaceable without touching the others:

1. **Providers** — model access. OpenRouter is the only provider on day one, but it sits behind a provider abstraction (streaming, tool-call parsing, usage accounting) so others can be added later without changing the workflow or UI layers.
2. **Workflows** — the execution model. Code-first: Python classes and functions composed into graphs. Must support the single-agent loop as the simple case, and beyond it: fan-out to multiple models running concurrently, staged pipelines with joins, and agents that spawn or delegate to other agents. Multi-agent orchestration is in scope for v1.
3. **Control** — the run lifecycle. Every workflow supports pause (halt at the next safe point and wait), resume, and interrupt (cancel in-flight model or tool calls, optionally inject a message, continue). Tool execution is governed by a permission mode: `manual`, where destructive tool calls (file writes, shell commands, anything a tool declares as needing approval) are surfaced to the user for confirmation before running, and `auto`, where they execute without prompting. The mode can be switched at any point during a session. Permission handling lives here, not in individual workflows, so any workflow gets it for free and any UI can render the approval prompt — though in practice the coding-agent workflow is its main consumer. All run state is serializable from the first commit; save-to-disk and load-from-disk of a full session is a v1 deliverable.
4. **UI / TUI** — a composable presentation layer. Not a chat window with a spinner. It must be able to render whatever the workflow's structure implies: parallel panes for concurrent model outputs, stage-by-stage views for pipelines, drill-down inspection of tool calls with inputs, outputs, timing, and errors. Each workflow can ship its own UI built from shared components.

Alongside the layers, a small set of **essential built-in tools**: file read/write/edit, shell execution, and web fetch/search. These are first-class, not reference stubs — durable (timeouts, retries, structured failure), fast, and with heavily engineered descriptions and schemas so models use them well. The goal is that a model running on this harness has what it needs to do real work without the user writing tools first.

## Constraints

- Python 3.12+.
- Async core throughout. A thin sync entry point may exist for scripts.
- Dependencies are acceptable when they earn their place; no aversion to pulling in good libraries.
- OpenRouter is the sole provider at launch. The provider interface must not leak OpenRouter specifics.
- Textual is the default TUI framework unless it proves limiting.
- Runs locally, single process, single user.

## Out of scope for v1

- RAG and long-term memory.
- MCP tool support (the tool interface should not preclude it later).
- A web UI. TUI only.
- A broad tool library beyond the essentials above.
- Tracing and evaluation dashboards. Observability *hooks* — structured events for every model call, tool call, state transition — are in core; the dashboard that consumes them is not.
- Fine-grained permission controls — per-tool allowlists, "approve all of this kind" — beyond the two modes.
- Hosted, multi-user, or multi-tenant deployment.

## Success criteria

v1 is done when all of the following hold:

1. **Three reference workflows exist, built entirely on the public API without modifying the core**, each with its own TUI:
   - A coding-agent workflow in the style of Claude Code: an interactive session in a working directory where the agent reads, edits, and runs code using the built-in file and shell tools, with a permission model for destructive actions, diff display for edits, and streamed output. The agent can delegate to subagents — spawning a scoped child agent for a subtask (exploration, a parallel edit, a review pass) and receiving its result — and the TUI shows subagent activity distinctly from the main agent's. This is the workflow that exercises the essential tools and the multi-agent model hardest, and is the primary proof that they are good enough for real work.
   - A model-fusion workflow: the same prompt sent to several models concurrently, outputs shown side by side, followed by an analysis stage and a synthesis stage.
   - A tool-observability workflow: a single agent working on a task, with a live inspector showing every tool call's inputs, outputs, timing, and errors.
2. **A session can be paused, serialized to disk, the process killed, the session reloaded, and resumed** — and the workflow completes as if uninterrupted.
3. **Interrupt works mid-flight**: an in-progress model stream or tool call can be cancelled, a message injected, and the run continued.
4. **Every layer can be explained in a single diagram**, and the diagram matches the code.
5. **The built-in tools are the ones the model reaches for**, not workarounds — measured by whether the reference workflows complete real tasks using only the essential toolset.

## Open questions and assumptions

These were not settled during scoping. Each is written as the assumption I am proceeding with; strike or revise as needed.

- **Interrupt semantics.** Assumed: cancel current model/tool call, optionally inject a user message, resume at the workflow's next decision point. Alternative: interrupt always drops to a pause state and waits.
- **Serialization boundary.** Assumed: the workflow author is responsible for keeping custom state serializable; the harness serializes what it owns (messages, tool results, run position, provider metadata) and raises clearly if custom state cannot be serialized.
- **Multi-agent model.** Assumed: agents are workflow nodes that can themselves contain workflows, so orchestration composes rather than needing a separate concept. Subagents get their own context (messages, tool results) but inherit the parent's permission mode and report back into the parent's session so the whole tree serializes together. Approval prompts come only from the main agent: in `manual` mode, subagents are limited to tools that do not require approval (read, search, explore), and destructive work stays with the main agent; in `auto` mode the restriction lifts. This is a v1 simplification, not a permanent design choice.
- **Tool durability.** Assumed: timeouts and retries are configured per tool with sane defaults, and failures are returned to the model as structured errors rather than raised.
- **Where "prompt engineering" for tools lives.** Assumed: in the tool definitions themselves (descriptions, schemas, usage guidance), versioned with the code, not in a separate prompt store.
