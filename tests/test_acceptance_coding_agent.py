"""R-A-1: the coding agent, its TUI, and three scripted tasks in a real repository.

The tasks are spec 9.3's, minus the one plan decision D4 defers (task 3 needs a
`SERPER_API_KEY`/network); the third here delegates to a subagent instead, which is
what R-A-1 asks the coding agent to demonstrate and what §9.3's list does not cover.

**Everything except the model is real.** Real files in a temp workspace, the real
built-in tools, the real dispatcher, the real gate, the real controller. The model is
a `FakeProvider` with a `by_index` script, so "did the agent change the file" is a
question about the file rather than about a mock: every assertion below reads the
workspace after the run.

The plan puts the *real-model* version of these tasks at M7 ("§9.3 tasks 1 and 2
against a real model with event logs committed"). What M6 owes is the workflow, the
UI and the check; `tests/acceptance.py:scan_run` is that check, and M7 runs the same
function over a recorded log.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from azalabscode import (
    AgentSpawned,
    Event,
    EventBus,
    MessageInjected,
    PermissionMode,
    RunState,
    Subscription,
    ToolCallRequested,
)
from tests.acceptance import sample_repo, scan_run
from workflows.coding_agent.app import CodingAgentApp
from workflows.coding_agent.cli import app as coding_cli
from workflows.coding_agent.session import CodingSession
from workflows.coding_agent.workflow import ALL_TOOLS, CodingAgentConfig

BOUND = 20.0

TOOLS = ["read_file", "write_file", "edit_file", "glob", "grep", "shell", "delegate"]
"""The built-ins these tasks use. `web_fetch`/`web_search` are the deferred task's."""


# ---------------------------------------------------------------------------
# Scripts. One turn per model call, in order (`by_index`).
# ---------------------------------------------------------------------------


def call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    """One scripted tool call."""

    return {"call_id": call_id, "name": name, "arguments": arguments}


def turn(*calls: dict[str, Any], text: str = "") -> dict[str, Any]:
    """One scripted model turn: tool calls, or a final answer."""

    if calls:
        return {"text": text, "tool_calls": list(calls), "finish_reason": "tool_calls"}
    return {"text": text}


def script(*turns: dict[str, Any]) -> dict[str, Any]:
    """A `Script` document, matched by index."""

    return {"match": "by_index", "turns": list(turns)}


VERBOSE_FLAG = '    parser.add_argument("--verbose", action="store_true")'


def task_one_script() -> dict[str, Any]:
    """§9.3 task 1: add a `--verbose` flag to `cli.py` and update the README."""

    return script(
        turn(call("t1", "glob", pattern="*.py")),
        turn(call("t2", "read_file", path="cli.py"), call("t3", "read_file", path="README.md")),
        turn(
            call(
                "t4",
                "edit_file",
                path="cli.py",
                old='    parser.add_argument("path")',
                new=f'    parser.add_argument("path")\n{VERBOSE_FLAG}',
            )
        ),
        turn(
            call(
                "t5",
                "edit_file",
                path="README.md",
                old="    python cli.py PATH",
                new="    python cli.py [--verbose] PATH",
            )
        ),
        turn(text="Added a --verbose flag to cli.py and documented it in the README."),
    )


def task_two_script() -> dict[str, Any]:
    """§9.3 task 2: replace every call site of `deprecated_fn`, then run the tests."""

    command = f'"{sys.executable}" -m pytest -q test_lib.py'
    return script(
        turn(call("t1", "grep", pattern="deprecated_fn", output_mode="content")),
        turn(call("t2", "read_file", path="use_a.py"), call("t3", "read_file", path="use_b.py")),
        turn(
            call(
                "t4",
                "edit_file",
                path="use_a.py",
                old="deprecated_fn",
                new="new_fn",
                replace_all=True,
            )
        ),
        turn(
            call(
                "t5",
                "edit_file",
                path="use_b.py",
                old="lib.deprecated_fn",
                new="lib.new_fn",
            )
        ),
        turn(call("t6", "shell", command=command)),
        turn(text="Replaced both call sites with new_fn and ran the tests; they pass."),
    )


def task_three_script() -> dict[str, Any]:
    """A broad question, delegated to the `explore` subagent (R-A-1, R-U-5).

    The child's model calls come out of the same script, so its turns sit between the
    parent's: parent delegates, child answers, parent concludes.
    """

    return script(
        turn(call("t1", "delegate", spec="explore", task="where is deprecated_fn defined?")),
        # -- the child's turns --
        turn(call("c1", "grep", pattern="def deprecated_fn", output_mode="content")),
        turn(text="lib.py defines deprecated_fn at line 4; use_a.py and use_b.py call it."),
        # -- back in the parent --
        turn(text="deprecated_fn lives in lib.py and has two call sites."),
    )


def config(workspace: Path, script_doc: dict[str, Any], **overrides: Any) -> CodingAgentConfig:
    """A coding-agent config that runs offline against `workspace`."""

    payload: dict[str, Any] = {
        "task": "do the thing",
        "model": "fake/model",
        "subagent_model": "fake/model",
        "workspace": str(workspace),
        "tools": list(TOOLS),
        "interactive": False,
        "script": script_doc,
    }
    payload.update(overrides)
    return CodingAgentConfig.model_validate(payload)


class Recorder:
    """Collects the run's events so the §9.3 scan has something to scan."""

    def __init__(self, bus: EventBus) -> None:
        self.events: list[Event] = []
        self.sub: Subscription = bus.subscribe(name="acceptance")

    async def pump(self) -> None:
        """Drain until the bus closes."""

        async for event in self.sub:
            self.events.append(event)


async def run_task(
    workspace: Path,
    script_doc: dict[str, Any],
    *,
    task: str,
    mode: PermissionMode = PermissionMode.AUTO,
    **overrides: Any,
) -> tuple[str, list[Event]]:
    """Run one scripted task to completion and return `(answer, events)`."""

    cfg = config(workspace, script_doc, task=task, **overrides)
    bus = EventBus()
    session = CodingSession.create(cfg, bus=bus, mode=mode, autosave=False)
    recorder = Recorder(bus)
    pump = asyncio.create_task(recorder.pump())
    try:
        answer = await session.run_once(timeout=BOUND)
    finally:
        recorder.sub.unsubscribe()
        await pump
        bus.close()
    return answer, recorder.events


# ---------------------------------------------------------------------------
# The three tasks (R-A-1: "completes at least three scripted real tasks")
# ---------------------------------------------------------------------------


async def test_task_one_adds_a_flag_and_updates_the_readme(tmp_path: Path) -> None:
    """§9.3 task 1, against real files."""

    repo = sample_repo(tmp_path / "repo")
    answer, events = await run_task(
        repo, task_one_script(), task="Add a --verbose flag to cli.py and update the README"
    )

    assert "--verbose" in (repo / "cli.py").read_text(encoding="utf-8")
    assert "[--verbose]" in (repo / "README.md").read_text(encoding="utf-8")
    assert "verbose" in answer

    scan = scan_run(events, allowed_tools=ALL_TOOLS)
    assert scan.ok, scan.report()
    assert set(scan.tools_used) == {"glob", "read_file", "edit_file"}


async def test_task_two_replaces_every_call_site_and_runs_the_tests(tmp_path: Path) -> None:
    """§9.3 task 2. The `shell` call really runs pytest in the sample repo."""

    repo = sample_repo(tmp_path / "repo")
    answer, events = await run_task(
        repo,
        task_two_script(),
        task="Replace every call site of deprecated_fn with new_fn, then run the tests",
    )

    assert "deprecated_fn" not in (repo / "use_a.py").read_text(encoding="utf-8")
    assert "deprecated_fn" not in (repo / "use_b.py").read_text(encoding="utf-8")
    assert "new_fn" in (repo / "use_a.py").read_text(encoding="utf-8")
    # The definition is untouched: the task was about call sites.
    assert "def deprecated_fn" in (repo / "lib.py").read_text(encoding="utf-8")
    assert "pass" in answer

    scan = scan_run(events, allowed_tools=ALL_TOOLS)
    assert scan.ok, scan.report()
    assert scan.shell_commands and "pytest" in scan.shell_commands[0]


async def test_task_three_delegates_the_broad_question(tmp_path: Path) -> None:
    """R-A-1's `delegate`: the subagent searches, the parent answers (R-U-5)."""

    repo = sample_repo(tmp_path / "repo")
    answer, events = await run_task(
        repo,
        task_three_script(),
        task="Where is deprecated_fn defined and who calls it?",
        subagents=["explore", "review"],
    )

    assert "lib.py" in answer
    spawned = [event for event in events if isinstance(event, AgentSpawned)]
    # `main/0` is registered twice: once by the parent before the child's task exists
    # (the spawn-race ordering), and once by the child's own `_enter`. Only the first
    # carries `parent_id` and `delegated`, which is why `AgentTree` folds them stickily.
    assert dict.fromkeys(event.agent_id for event in spawned) == {"main": None, "main/0": None}
    child = next(event for event in spawned if event.agent_id == "main/0")
    assert child.parent_id == "main" and child.delegated is True
    # The child did the searching, under its own agent id.
    child_tools = [
        event.tool
        for event in events
        if isinstance(event, ToolCallRequested) and event.agent_id == "main/0"
    ]
    assert child_tools == ["grep"]

    scan = scan_run(events, allowed_tools=ALL_TOOLS)
    assert scan.ok, scan.report()


def test_the_shell_scan_would_catch_a_builtin_done_the_hard_way() -> None:
    """The §9.3 check is non-vacuous: prove it fails on the thing it forbids."""

    from tests.acceptance import shell_heads

    assert shell_heads("grep -rn foo .") == ["grep"]
    assert shell_heads("git log --grep=foo") == ["git"]
    assert shell_heads('type "cli.py" | Select-String foo') == ["type", "select-string"]
    assert shell_heads(r"C:\tools\rg.exe foo") == ["rg"]

    events = [ToolCallRequested(call_id="c1", tool="shell", params={"command": "find . -name x"})]
    scan = scan_run(events)
    assert not scan.ok
    assert "glob" in scan.report()


# ---------------------------------------------------------------------------
# The TUI (R-A-1's widget list)
# ---------------------------------------------------------------------------


async def test_the_tui_shows_the_transcript_the_calls_the_diff_and_the_tree(
    tmp_path: Path,
) -> None:
    """Spec 9.1's list: transcript, streamed output, diff on edits, tree, status bar."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo, task_three_script(), task="who calls deprecated_fn?")
    session = CodingSession.create(cfg, bus=EventBus(), mode=PermissionMode.AUTO, autosave=False)
    tui = CodingAgentApp(session)
    async with tui.run_test() as pilot:
        await session.controller.run(timeout=BOUND)
        await pilot.pause()

        # A transcript per agent, and the subagent has its own (R-U-5).
        assert set(tui.transcripts) == {"main", "main/0"}
        for transcript in tui.transcripts.values():
            for pane in transcript.panes.values():
                pane.flush_now()
        assert "lib.py" in tui.transcripts["main"].text()
        assert tui.transcripts["main/0"].subagent is True

        # The tree drew both, nested.
        assert tui.agent_tree.children_of("main") == ["main/0"]

        # Every tool call is in the table, and the drill-down follows the cursor.
        assert {record.tool for record in tui.calls.records.values()} == {"delegate", "grep"}
        record = next(iter(tui.calls.records.values()))
        tui.detail.show(record)
        assert record.tool in tui.detail.render_record().plain

        # The status bar is mounted and live (spec 8.1).
        bar = tui.status_bar
        assert bar is not None
        assert "COMPLETED" in " ".join(text for text, _style in bar.segments())
    session.controller.bus.close()


async def test_an_edit_opens_the_diff_view(tmp_path: Path) -> None:
    """Spec 9.1: "DiffView overlays on edits". It appears when there is a diff."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo, task_one_script(), task="add a --verbose flag")
    session = CodingSession.create(cfg, bus=EventBus(), mode=PermissionMode.AUTO, autosave=False)
    tui = CodingAgentApp(session)
    async with tui.run_test() as pilot:
        assert tui.diff.display is False
        await session.controller.run(timeout=BOUND)
        await pilot.pause()

        assert tui.diff.display is True
        assert len(tui.diff.history) == 2  # cli.py and README.md
        assert "--verbose" in tui.diff.render_diff().plain
        tui.action_toggle_diff()
        assert tui.diff.display is False
    session.controller.bus.close()


async def test_a_destructive_call_raises_the_modal_and_can_be_denied(tmp_path: Path) -> None:
    """R-U-6: `manual` mode, no UI code written by the workflow, and `n` denies."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo, task_one_script(), task="add a --verbose flag")
    session = CodingSession.create(cfg, bus=EventBus(), mode=PermissionMode.MANUAL, autosave=False)
    tui = CodingAgentApp(session)
    async with tui.run_test() as pilot:
        await session.controller.start()
        await _wait(lambda: tui._modal is not None)
        await pilot.pause()
        modal = tui._modal
        assert modal is not None
        assert "edit_file" in modal.request.summary.title
        assert modal.request.summary.diff  # the edit is reviewable, not a leap of faith

        # Deny this one and every one after it: the script asks for two edits, and a
        # modal nobody answers is a hang rather than a failure.
        await _answer_every_modal(tui, pilot, "n")
        await session.controller.wait(timeout=BOUND)

        # Denied: the files are untouched and the model was told why.
        assert "--verbose" not in (repo / "cli.py").read_text(encoding="utf-8")
        assert "[--verbose]" not in (repo / "README.md").read_text(encoding="utf-8")
        denied = [
            block
            for transcript in tui.transcripts.values()
            for block in transcript.tool_blocks.values()
            if block.status == "denied"
        ]
        assert len(denied) == 2
    session.controller.bus.close()


async def _answer_every_modal(tui: CodingAgentApp, pilot: Any, key: str) -> None:
    """Press `key` at every approval modal until the run reaches a terminal state."""

    terminal = {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}
    async with asyncio.timeout(BOUND):
        while tui.controller.state not in terminal:
            if tui._modal is not None:
                await pilot.press(key)
            await pilot.pause()
            await asyncio.sleep(0.01)


async def test_the_prompt_box_sends_without_cancelling(tmp_path: Path) -> None:
    """Spec 8.2's normal mode: `send(text)`, not `interrupt(text)`.

    The session is interactive, so it waits after its first answer; the typed message
    is what starts it again, and it arrives as an injection rather than as a new run.
    """

    repo = sample_repo(tmp_path / "repo")
    second = script(
        turn(text="First answer."),
        turn(text="Second answer."),
    )
    cfg = config(repo, second, task="say something", interactive=True)
    session = CodingSession.create(cfg, bus=EventBus(), mode=PermissionMode.AUTO, autosave=False)
    tui = CodingAgentApp(session)
    recorder = Recorder(session.controller.bus)
    pump = asyncio.create_task(recorder.pump())
    async with tui.run_test() as pilot:
        await session.controller.start()
        await _wait(lambda: "First answer." in _all_text(tui))
        await pilot.pause()
        # The agent answered and the session is idle, not finished: it is waiting.
        assert session.controller.state is not RunState.COMPLETED

        tui.prompt.insert("and again")
        await pilot.press("enter")
        await pilot.pause()
        await _wait(lambda: "Second answer." in _all_text(tui))

        injected = [event.text for event in recorder.events if isinstance(event, MessageInjected)]
        assert injected == ["and again"]
        # `send` does not cancel: nothing was interrupted to deliver it.
        assert not [event for event in recorder.events if event.type == "model_call_cancelled"]

        session.close()
        await session.controller.wait(timeout=BOUND)
        assert session.controller.state is RunState.COMPLETED
    recorder.sub.unsubscribe()
    await pump
    session.controller.bus.close()


def _all_text(tui: CodingAgentApp) -> str:
    for transcript in tui.transcripts.values():
        for pane in transcript.panes.values():
            pane.flush_now()
    return "\n".join(transcript.text() for transcript in tui.transcripts.values())


async def _wait(predicate: Any, *, timeout: float = BOUND) -> None:
    """Poll until `predicate()` holds. Always bounded."""

    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# --headless and the CLI (R-U-7, D6)
# ---------------------------------------------------------------------------


def test_the_coding_agent_runs_headless_from_the_cli(tmp_path: Path) -> None:
    """R-U-7: the same session, the same tools, no UI."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo, task_one_script(), task="add a --verbose flag")
    config_path = tmp_path / "azc.json"
    config_path.write_text(json.dumps(cfg.model_dump(mode="json")), encoding="utf-8")

    result = CliRunner().invoke(
        coding_cli,
        [
            "--headless",
            "--auto",
            "--config",
            str(config_path),
            "--session-dir",
            str(tmp_path / "sessions"),
            "add a --verbose flag",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--verbose" in (repo / "cli.py").read_text(encoding="utf-8")
    assert (tmp_path / "sessions" / "session.json").exists()


def test_headless_without_a_task_is_a_message_not_a_traceback(tmp_path: Path) -> None:
    """A CLI that tracebacks at a user is a CLI nobody uses twice."""

    result = CliRunner().invoke(coding_cli, ["--headless"])
    assert result.exit_code == 2
    assert "needs a task" in result.output


@pytest.mark.parametrize("flag", ["--help"])
def test_the_cli_documents_itself(flag: str) -> None:
    """`azc --help` names the session, the modes and `--headless`."""

    result = CliRunner().invoke(coding_cli, [flag])
    assert result.exit_code == 0
    assert "--headless" in result.output
    assert "--auto" in result.output
    assert "--resume" in result.output
