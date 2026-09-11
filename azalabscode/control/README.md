# `azalabscode.control` — the run lifecycle

Pause, resume, interrupt, permission modes, approvals, and durable sessions. Every
workflow gets all of it, without having to know it exists.

```python
from azalabscode.control import Controller

controller = Controller(graph, env, session_dir=Path(".azalabscode"), mode=PermissionMode.MANUAL)
task = asyncio.create_task(controller.start())
await controller.pause()                 # halts at the next safe point, durably
await controller.save()                  # ... or just wait: a durable safe point wrote it
await controller.interrupt("actually, use the other file")
await controller.set_mode(PermissionMode.AUTO)
```

## Safe points

A `SafePoint` carries its `kind` (`turn_start`, `after_model_call`, `after_tool_batch`,
`node_entered`, `node_completed`, `approval_park`, or `custom`), the node and agent it
belongs to, an attempt number, a `NodeSnapshot`, and its `durable` / `park` flags. The
workflow layer declares safe points through the `RunControl` protocol; this layer
decides what they mean.

**Write first, park second — and park outside the lock:**

```python
async with self._cp_lock:
    self._fold(sp)                    # mutate Session; may raise SerializationError
    if sp.durable and self._session_dir:
        await self._write_session()   # atomic, in a thread
    self._resolve_pending_saves()
if sp.park:
    await self._park(sp.agent_id)     # MUST be outside the lock
```

Parking while still holding the lock would deadlock every other checkpoint, and then
PAUSED would never be reached. `_fold` lets a `SerializationError` escape uncaught, so a
run whose custom state can't be saved goes `FAILED` rather than quietly continuing
unsaved.

## Quiescence

An agent's phase is either quiescent or not. `running` and `blocked_io` are
non-quiescent; `blocked_on_child`, `parked`, `waiting_approval`, and `finished` are
quiescent.

**PAUSED means: a pause was requested *and* the non-quiescent count is zero.** It does
*not* mean `parked == active` — that definition races on spawn, and it deadlocks
whenever a subagent parks while its parent is awaiting it. `blocked_on_child` has to
count as quiescent, or a child parked at the approval gate would block a parent that
never returns. And that's safe to do, because a delegate step has no external effect of
its own, and every effect a child does have gets its own safe point.

The spawn race is closed by registering the child and flipping the parent to
`blocked_on_child` **synchronously, before `tg.create_task(...)`**. There's no instant
where the child is unregistered and the parent is already quiescent. Don't "tidy" that
ordering — it's load-bearing.

`pause(hard=True)` does one extra thing: it cancels every step whose agent is
`blocked_io`.

## Interrupt

`StepHandle.request_cancel(reason)` sets the reason **before** it calls `task.cancel()`,
and the first reason wins, so a double interrupt is idempotent. A cancelled model call
drops *all* of its `ToolCallPart`s: none of them were dispatched, and keeping one would
force a synthetic result telling the model it ran something it never ran.

Interrupting a child directly (`target="main/2"`) leaves the parent alone. An in-flight
`delegate` is **resumed**, not errored — only leaf tool calls turn into
`ToolError(kind="interrupted")`.

`send(text)` is the non-cancelling counterpart: it injects a user message and lets the
current step run to completion.

## Session and resume

`Session` is the entire run as one document: the agents with their phases and pending
results, the node records, the workflow reference, usage, the RNG seed, and the
`inflight` steps, which are kept up to date live so that a safe point taken by agent A
correctly records that agent B is mid-`shell`. Node outputs larger than 32 KB spill to
`session_dir/values/`, which keeps `session.json` diffable.

`reconcile()` runs on load and handles each interrupted step by kind: a `model_call` is
dropped and re-issued (nothing was half-appended); a `tool_call` becomes
`ToolError(kind="interrupted", message="process terminated during execution; effect
unknown")` and is **never** re-executed; a `delegate` is resumed. Then
`repair_transcript` backfills anything still missing, and the transcript invariant has
to hold or `load()` fails hard.

`check_graph_drift` compares a `graph_hash` taken over `(node_id, node_class,
state_type, output_type)`. It deliberately *excludes* prompts and config, so editing a
system prompt doesn't invalidate a session. A saved id that's missing from the rebuilt
graph is a hard error; extra ids, or a changed hash, are a warning event — which you can
promote to an error with `strict_graph_hash=True`.

## Atomic writes on Windows

`atomic_write_bytes` creates its temp file in the **destination directory** (`os.replace`
is only atomic within a single volume), writes it, `fsync`s it, and then `os.replace`s
it in a bounded, jittered retry loop. On Windows that replace raises `PermissionError`
whenever the destination has an open handle, so the retry is mandatory, not just
defensive. There's no directory fsync on Windows; the documented limit is that a power
loss can lose the newest checkpoint, but it can never corrupt one. `sweep_temp_files`
clears the `.tmp` files left behind by a process that was killed between `mkstemp` and
`os.replace`, and every CLI calls it at startup.

`save()` takes a timeout (default 120 s) and raises `SaveTimeout` naming the step that
blocked, rather than hanging behind a 600 s `shell`.

## Approvals

`RuntimePermissionGate` is the policy; the vocabulary lives in
`azalabscode.permissions`. In `manual` mode, a destructive call parks its agent at
`waiting_approval` and raises an `ApprovalRequest` to whichever `ApprovalHandler` is
installed — `TUIApprovalHandler` in a TUI, `StdinApprovalHandler` headless,
`QueueApprovalHandler` or `CallbackApprovalHandler` in tests, and `DenyAllHandler` when
nothing is allowed to run. In `manual` mode subagents are limited to tools that need no
approval; in `auto` that restriction lifts. A pending approval survives save and load.
