# `azalabscode.tui` — the presentation layer

This is not a chat window with a spinner. The widgets exist so that a fan-out of four
models, a staged pipeline, and a tool-call drill-down can each be rendered from the same
event stream.

```python
from azalabscode.tui import HarnessApp, StreamPane, Transcript

class MyApp(HarnessApp):
    def compose(self):
        yield Transcript(agent_id="main")
        yield StreamPane(node_id="draft", tail=True)

await MyApp(session, autostart=True).run_async()
```

`HarnessApp` subscribes to the bus, routes each event by `(agent_id, node_id)`, owns the
`TUIApprovalHandler`, mounts an `ApprovalModal` on its own, and provides the standard key
table. A subclass composes the widgets it wants and, usually, writes no handlers at all.

## The widgets

`Transcript` (the messages for one agent), `StreamPane` (live model output),
`ToolCallList` + `ToolCallDetail` (inputs, outputs, timing, errors), `DiffView` (edit
approvals and edit results), `AgentTree` (the subagent tree, live), `StagePipeline` (a
pipeline's stages and their states), `SplitPanes` (n panes for n concurrent nodes),
`PromptInput`, `ApprovalModal`, `RunStatusBar`, and `EventLog` (the raw JSONL viewer).

`ToolResult.display` is the widget-facing rendering. Never re-parse
`ToolResult.content` — that string is written for a model, and a widget that parses it
will break the first time a tool's description changes.

## Textual lives here and only here

No module below `tui/` may name a Textual type. `import azalabscode` does not pull in
Textual — a test asserts this in a subprocess — which is what keeps a headless run cheap,
and what would make swapping out the UI framework a swap rather than a rewrite. Import
contract 6 also forbids `tui` from naming a `Tool`, an `AgentLoop`, a `Provider`, a
`Workflow`, a `Runner`, or a `Node`: the UI sees events and the `Controller`, and nothing
else.

## Performance

`StreamPane` batches deltas on a 33 ms timer into a single write, and drains its buffer
inside `render()` rather than on a timer. `refresh(layout=True)` is the expensive call,
so a streaming widget is given a stable box. Measured on Windows 11 / Textual 8.2.8, with
four `FakeProvider` streams at 200 deltas/s through four real agent loops: 34–47 ms max
latency for the four-pane fusion layout, and 12 ms max loop lag. Layout dominates the
tail, and garbage collection dominates anything measured over a large heap — which is why
every CLI calls `prepare_process()` (a `gc.freeze()` plus a session-directory sweep).

## Traps

- **Textual's own bindings win silently.** `ctrl+p` is the command palette (we moved it
  to `ctrl+backslash`); `ctrl+c` needs `priority=True` to be overridden.
- **A `ModalScreen` stops binding lookup.** App-level bindings that have to survive a
  modal go in `HARNESS_BINDINGS`, not in a subclass's `BINDINGS`.
- **Don't name a widget attribute after a Textual one.** `_nodes` and `tree` are already
  taken by `Widget` and `DOMNode`; shadowing either one crashes the app on teardown.
- **`DataTable.add_column` needs an active app,** so columns go in `on_mount`, and
  anything that arrives earlier is buffered.
- **`ApprovalResolved` carries no `call_id`.** Keep a `request_id -> call_id` index from
  `ApprovalRequested`, or a denial will silently miss its row.
- **A loaded run has agents but no events for them.** `seed_from_controller()` builds the
  initial widget state, `StreamPane.set_text()` fills a pane from a memoized node output,
  and `Session.event_seq` seeds the new bus.
- **A widget mounted after `attach()` has to be registered by hand** with
  `register_consumer` — `refresh_consumers()` only walks the tree that existed at attach
  time.
- **Never take the controller's command lock from a UI callback.**
