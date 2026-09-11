# `azalabscode.tools` — what the model can do

Nine built-in tools, a dispatcher that handles everything a tool isn't trusted to
handle itself, and an OS layer that makes `shell` behave the same on Windows and POSIX.

```python
from azalabscode.tools import AllowAllGate, ToolContext, ToolDispatcher, default_registry

tools = default_registry()                      # web_search drops out with no SERPER_API_KEY
dispatcher = ToolDispatcher(tools, gate=AllowAllGate())
context = ToolContext(workspace=Path.cwd(), ...)
results = await dispatcher.run_batch(calls, context)
```

The layer runs standalone. With `AllowAllGate` and `DenyAllGate` you can dispatch tool
calls without a controller, a run, or a workflow around them.

## Writing a tool

```python
name: str
description: str                    # versioned with the code, snapshot-tested
Params: type[BaseModel]             # the single source of the JSON schema
approval: ApprovalPolicy            # never | always | (params) -> bool
timeout: float
retry: RetryPolicy                  # must be 0 whenever approval != never
max_result_size_chars: int | float

def is_read_only(self, p) -> bool: ...
def is_concurrency_safe(self, p) -> bool: ...            # default False
async def validate_params(self, p, ctx) -> ValidationError | None: ...
async def run(self, p, ctx) -> ToolResult: ...
def approval_summary(self, p, ctx) -> ApprovalSummary: ...
```

Every default fails closed: a tool is not read-only, not concurrency-safe, and needs
approval unless it says otherwise.

**`validate_params` is deliberately separate from `run`.** It's a pure pre-check that
returns a structured error *before* the approval prompt. Without it, `manual` mode would
ask you to approve an edit that was always going to fail. Checks like "file not read
yet", "string not unique", and "path outside the workspace" all belong here.

**A result has three renderings.** `ToolResult.content` is for the model,
`ToolResult.display` is structured for widgets (diff hunks, match lists), and
`ToolResult.meta` is for telemetry. One run, three consumers — and no widget ever
re-parses the text that was written for the model.

## The dispatcher

`ToolDispatcher` owns schema validation, `validate_params`, the `PermissionGate` check,
the timeout (which cancels the task and calls `kill_tree` for `shell`), retries, the
per-tool result cap, the per-turn budget, batching, and turning any unanticipated
exception into a `ToolError(kind="internal")` with the traceback in `meta`. A tool
author writes `run()` and gets all of that for free.

**Batching goes by contiguous run.** The dispatcher splits a turn's calls into the
longest possible runs of concurrency-safe calls. Each safe run executes concurrently
under a semaphore, each unsafe call runs alone, and the original call order is always
preserved. So five reads don't get serialized just because one `edit_file` sits among
them — and write-after-read ordering comes for free.

## The nine built-ins

| Tool | The durability that matters |
|---|---|
| `read_file` | line-numbered, a 2 000-line window, a byte cap that **errors rather than truncates**, BOM strip, CRLF normalisation, latin-1 fallback, binary detection, images as `ImagePart`. The result size is uncapped on purpose: spilling a file read to disk for the model to re-read would be circular. |
| `write_file` | atomic, creates parent directories, refuses a path that was never read, re-checks mtime in a critical section with no awaits, and writes back exactly the line endings it was given. |
| `edit_file` | exact match, unique unless `replace_all`, errors that report the match count and the nearest whitespace-normalised candidate, and a unified diff in both the result and the approval summary. |
| `glob` / `grep` | `rg` when it's installed, a pure-Python fallback otherwise. `grep` has `output_mode`, context flags, and `head_limit` with an explicit "truncated, paginate" marker. A timeout reports *timed out*, never *no matches*. |
| `shell` | stdout and stderr merged into one append-mode file, so the interleaving stays chronological and the read loop stays off the hot path. The per-call timeout is capped at 600 s, the whole process tree is killed, and retries are hard-blocked. |
| `web_fetch` | manual redirects, same-host-modulo-`www` hops only, `file:`/loopback/RFC1918 blocked, HTML run through `trafilatura`, a 10 MB cap. |
| `web_search` | sits behind the `SearchBackend` protocol; the Serper adapter only registers when its key is present. |
| `delegate` | a thin wrapper over the `Delegator` protocol. `approval=never`, while the child's own tool calls are gated normally. |

**Read-before-write is the highest-value behavior here.** `ToolContext` keeps a bounded
LRU of `path -> (mtime, offset, limit)`, which `read_file` sets and writes clear. A
write to a path that was never read — or one that was read before an external change —
is refused, with instructions to re-read first. This is what stops a model from
silently clobbering a file.

## The platform layer

`tools/platform.py` exposes `spawn()`, `kill_tree()`, and `resolve_shell()`. On POSIX
it uses `start_new_session` plus `SIGTERM` then `SIGKILL`; on Windows it uses
`CREATE_NEW_PROCESS_GROUP` plus `taskkill /F /T /PID`, preferring `pwsh` and falling
back to `powershell`. The `shell` tool's model-facing description is rendered per
platform, so the model knows which shell it's talking to — a model told nothing will
write `ls | head` on a Windows box and get a confusing failure.

## Descriptions are code

Every tool description is a module-level constant, covered by a schema snapshot test
(`tests/fixtures/tool_schemas.json`). The prompt engineering for a tool lives with the
tool and is versioned alongside it, not kept in a separate store. `shell` is
deliberately left out of that snapshot, because its description depends on the platform.
