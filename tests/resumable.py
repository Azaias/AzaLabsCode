"""A real workflow module, importable by path, for the M3 save/load and kill tests.

`WorkflowRef.import_path` is `"tests.resumable:build"`, so a session written by one
process is reconstructible by another from `(import_path, config)` alone -- which is
the whole claim R-C-11 and R-C-12 make. Nothing about the writing process leaks into
the file: the provider, the tools and the dispatcher are all constructed inside
`build(config)` (spec delta 21).

Three properties make this usable as a kill-test fixture, and each one exists to
remove a race rather than to make the test shorter:

* **The slow tool blocks on a file, not a clock.** It waits for a release marker to
  appear, and announces itself by creating a start marker first. The killer waits
  for the start marker. `asyncio.sleep` is not a stopwatch on Windows (~15.6 ms
  resolution) and a sleep-based rendezvous is how a kill test becomes flaky.

* **Every effect is a line in a file.** `starts.log` records each time the slow tool
  begins, `executions.log` records each time a node body actually runs. Both survive
  the process that wrote them, which is the only way to assert "started exactly once
  across both processes" (R-C-12) and "zero completed nodes re-executed" (R-W-6).

* **The node loop is hand-written.** M5's `Runner` will do this properly; here it is
  eight lines that ask `node_completed`, take the memo if there is one, and run the
  body otherwise. That is the behaviour R-W-6 specifies, and writing it out makes
  the test assert the mechanism rather than a framework.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from azalabscode.ids import MAIN_AGENT
from azalabscode.permissions import ApprovalPolicy, PermissionMode
from azalabscode.providers.testing import FakeProvider, MatchMode
from azalabscode.toolio import NO_RETRY, RetryPolicy, ToolResult
from azalabscode.tools.base import Tool
from azalabscode.tools.context import ToolContext
from azalabscode.tools.dispatcher import ToolDispatcher
from azalabscode.workflows.agent_loop import AgentLoop, AgentSpec
from azalabscode.workflows.state import AgentState

RECORD_ENV = "AZALABSCODE_RECORD_KEYS"
"""When set to a path, every request fingerprint is appended there as JSONL.

This is how `tests/fixtures/kill_script.json` gets its `by_request_hash` keys: the
recorder runs the *same subprocess phases* the test runs, so a key can never drift
from the request that will actually be issued. A missing key is `ScriptExhausted`,
which fails loudly instead of diverging silently.
"""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class SlowParams(BaseModel):
    """Parameters for `slow`."""

    model_config = {"extra": "forbid"}

    label: str = Field(description="Which invocation this is, for the marker files.")


class SlowTool(Tool):
    """Announces itself, then blocks until released. The thing a kill happens during.

    `run` appends to `starts.log` **before** it blocks, so the count of starts is
    accurate even for the invocation that never finished. That count is what R-C-12
    asserts: exactly one across the two processes.
    """

    name: ClassVar[str] = "slow"
    description: ClassVar[str] = "Block until released. Test double."
    Params: ClassVar[type[BaseModel]] = SlowParams

    approval: ApprovalPolicy = "never"
    timeout: float = 120.0
    retry: RetryPolicy = NO_RETRY
    concurrency_safe: ClassVar[bool] = True
    read_only: ClassVar[bool] = False

    def __init__(self, markers: Path, timeout_s: float = 60.0) -> None:
        super().__init__()
        self.markers = markers
        self.timeout_s = timeout_s

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Record the start, wait for the release marker, then answer."""

        assert isinstance(params, SlowParams)
        self.markers.mkdir(parents=True, exist_ok=True)
        with (self.markers / "starts.log").open("a", encoding="utf-8") as log:
            log.write(f"{params.label}\n")
        (self.markers / f"started-{params.label}").write_text("1", encoding="utf-8")

        release = self.markers / "release"
        deadline = time.monotonic() + self.timeout_s
        while not release.exists():
            if time.monotonic() > deadline:  # pragma: no cover - the test always releases
                return ToolResult.ok_text(f"slow: {params.label} (timed out waiting)")
            await asyncio.sleep(0.01)
        return ToolResult.ok_text(f"slow: {params.label}")


class NoteParams(BaseModel):
    """Parameters for `note`."""

    model_config = {"extra": "forbid"}

    text: str = Field(description="What to append.")


class NoteTool(Tool):
    """Appends a line to a file. Approval-gated, so R-C-9 has something to gate.

    Not concurrency-safe and not read-only: it is the stand-in for `write_file` in a
    test that must not depend on the real filesystem tools.
    """

    name: ClassVar[str] = "note"
    description: ClassVar[str] = "Append a note to the run's notes file. Test double."
    Params: ClassVar[type[BaseModel]] = NoteParams

    approval: ApprovalPolicy = "always"
    timeout: float = 30.0
    retry: RetryPolicy = NO_RETRY
    concurrency_safe: ClassVar[bool] = False
    read_only: ClassVar[bool] = False

    def __init__(self, markers: Path) -> None:
        super().__init__()
        self.markers = markers

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Append and confirm."""

        assert isinstance(params, NoteParams)
        self.markers.mkdir(parents=True, exist_ok=True)
        with (self.markers / "notes.log").open("a", encoding="utf-8") as log:
            log.write(f"{params.text}\n")
        return ToolResult.ok_text(f"noted: {params.text}")


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """Everything `build` needs. Serialized into the session verbatim."""

    model_config = {"extra": "forbid"}

    script_path: str
    workspace: str
    markers: str
    task: str = "run the slow thing and report"
    model: str = "fake/model"
    match: MatchMode = "by_request_hash"
    mode: PermissionMode = PermissionMode.AUTO
    slow_timeout_s: float = 60.0


NODES = ("prepare", "agent", "summarize")
"""The three nodes. `agent` is the one that is allowed to run twice."""


def build(config: dict[str, Any] | Config) -> Any:
    """Return the run body for `config` (spec 6.2's `import_path` contract).

    The provider is constructed here rather than injected, which is what makes a
    session reconstructible from `(import_path, config)` alone (delta 21). The
    `by_index` branch seeds itself from the checkpointed `model_call_seq` so even the
    indexed mode survives a reload without the provider holding resume state.
    """

    cfg = config if isinstance(config, Config) else Config.model_validate(config)
    markers = Path(cfg.markers)
    workspace = Path(cfg.workspace)
    record_path = os.environ.get(RECORD_ENV) or ""

    async def body(control: Any) -> dict[str, Any]:
        outputs: dict[str, Any] = {}

        outputs["prepare"] = await _node(control, markers, "prepare", lambda: "prepared")

        if control.node_completed("agent"):
            outputs["agent"] = control.node_output("agent")
        else:
            _record_execution(markers, "agent")
            outputs["agent"] = await _run_agent(control, cfg, markers, workspace, record_path)
            await control.node_finished("agent", outputs["agent"])

        outputs["summarize"] = await _node(
            control, markers, "summarize", lambda: f"summary of {outputs['agent']}"
        )
        return {"final": outputs["summarize"], "nodes": dict(outputs)}

    return body


async def _node(control: Any, markers: Path, name: str, run: Any) -> Any:
    """Run a node body, or take its memoized output (R-W-6).

    The memo check comes first and it is the whole point: a completed node's body is
    never entered again, so `executions.log` records it exactly once no matter how
    many processes the run spans.
    """

    if control.node_completed(name):
        return control.node_output(name)
    await control.node_started(name)
    _record_execution(markers, name)
    value = run()
    await control.node_finished(name, value)
    return value


def _record_execution(markers: Path, name: str) -> None:
    """Append one line per node body actually executed. R-W-6's evidence."""

    markers.mkdir(parents=True, exist_ok=True)
    with (markers / "executions.log").open("a", encoding="utf-8") as log:
        log.write(f"{name}\n")


async def _run_agent(
    control: Any, cfg: Config, markers: Path, workspace: Path, record_path: str
) -> str:
    """The agent node: one `AgentLoop` over a scripted provider and two fake tools."""

    # Claimed here rather than left to the loop so the provider can be seeded from
    # the checkpointed counter -- `by_index` has to start where the last process
    # stopped or every scripted turn after the resume is off by one.
    restored = control.restored_agent(MAIN_AGENT)
    state = restored if isinstance(restored, AgentState) else None
    provider: Any = FakeProvider(
        script_path=cfg.script_path,
        match=cfg.match,
        call_index=state.model_call_seq if state is not None else 0,
    )
    if record_path:
        provider = _RecordingProvider(provider, Path(record_path))

    workspace.mkdir(parents=True, exist_ok=True)
    tools = [SlowTool(markers, timeout_s=cfg.slow_timeout_s), NoteTool(markers)]
    dispatcher = ToolDispatcher(
        tools,
        context=ToolContext(workspace_root=workspace),
        gate=control.permission_gate,
        emitter=control.emitter_for(MAIN_AGENT),
    )
    loop = AgentLoop(
        AgentSpec(name="main", model=cfg.model, system_prompt="be terse"),
        provider=provider,
        dispatcher=dispatcher,
        control=control,
        emitter=control.emitter_for(MAIN_AGENT),
        agent_id=MAIN_AGENT,
        state=state,
    )
    result = await loop.run(cfg.task)
    return result.final_text


class _RecordingProvider:
    """Wraps a `FakeProvider` and writes every request fingerprint to a file.

    Used only by the script recorder. It sits in front of the real provider rather
    than replacing it, so the run being recorded is byte-for-byte the run the test
    will later replay.
    """

    def __init__(self, inner: FakeProvider, path: Path) -> None:
        self.inner = inner
        self.path = path

    @property
    def name(self) -> str:
        """Delegate the provider's name."""

        return self.inner.name

    async def stream(self, request: Any) -> Any:
        """Record the fingerprint, then replay."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {"key": request.fingerprint(), "index": self.inner.call_index},
                )
                + "\n"
            )
        async for event in self.inner.stream(request):
            yield event

    async def model_info(self, model: str) -> Any:
        """Delegate."""

        return await self.inner.model_info(model)

    async def aclose(self) -> None:
        """Delegate."""

        await self.inner.aclose()


__all__ = [
    "NODES",
    "RECORD_ENV",
    "Config",
    "NoteTool",
    "SlowTool",
    "build",
]
