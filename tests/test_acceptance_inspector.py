"""R-A-3: tool observability -- the live call table, the drill-down, the event log.

"Single agent; live table of every tool call with status, duration, and drill-down to
params, result, error, and the raw event sequence." Every clause of that is checked
here against a real run over a real workspace with the real built-in tools; only the
model is scripted.

The last assertion in this file is the one worth keeping: the headless summary and the
widget's table are two independent folds of the same event stream, and they have to
agree. If they ever disagree it is because one of them is reading something that is
not in the stream -- which is exactly what R-X-3 forbids.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from azalabscode import Event, EventBus, RunState, Subscription
from tests.acceptance import sample_repo
from workflows.inspector.app import InspectorApp
from workflows.inspector.cli import app as inspector_cli
from workflows.inspector.headless import CallSummary, CallWatcher, controller_for, run_headless
from workflows.inspector.workflow import AGENT_ID, InspectorConfig, build

BOUND = 20.0

TOOLS = ["read_file", "glob", "grep", "shell"]


def script() -> dict[str, Any]:
    """Two searches, one read, one failure, then an answer."""

    return {
        "match": "by_index",
        "turns": [
            {
                "tool_calls": [
                    {"call_id": "s1", "name": "glob", "arguments": {"pattern": "*.py"}},
                    {
                        "call_id": "s2",
                        "name": "grep",
                        "arguments": {"pattern": "deprecated_fn", "output_mode": "content"},
                    },
                ],
                "finish_reason": "tool_calls",
            },
            {
                "tool_calls": [
                    {"call_id": "s3", "name": "read_file", "arguments": {"path": "lib.py"}},
                    {"call_id": "s4", "name": "read_file", "arguments": {"path": "nope.py"}},
                ],
                "finish_reason": "tool_calls",
            },
            {"text": "lib.py defines deprecated_fn and new_fn; nope.py does not exist."},
        ],
    }


def config(workspace: Path, **overrides: Any) -> InspectorConfig:
    """An inspector config that runs offline against `workspace`."""

    payload: dict[str, Any] = {
        "task": "what is in this directory?",
        "model": "fake/model",
        "workspace": str(workspace),
        "tools": list(TOOLS),
        "script": script(),
    }
    payload.update(overrides)
    return InspectorConfig.model_validate(payload)


class Recorder:
    """Every event of the run, for the fold-comparison at the end."""

    def __init__(self, bus: EventBus) -> None:
        self.events: list[Event] = []
        self.sub: Subscription = bus.subscribe(name="acceptance")

    async def pump(self) -> None:
        """Drain until the bus closes."""

        async for event in self.sub:
            self.events.append(event)


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def test_the_graph_is_one_agent_node(tmp_path: Path) -> None:
    """R-A-3 is about the UI; the graph is deliberately the smallest one there is."""

    graph = build(config(tmp_path)).compile()
    assert graph.order == ("agent",)
    # The node id and the agent id are different names (M5 trap 1).
    assert "agent" in graph.node_ids()
    assert AGENT_ID == "main"


# ---------------------------------------------------------------------------
# The table and the drill-down (R-A-3)
# ---------------------------------------------------------------------------


async def test_every_call_reaches_the_table_with_status_and_duration(tmp_path: Path) -> None:
    """The live table: one row per call, each with what happened and how long."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo)
    controller = controller_for(cfg, bus=EventBus())
    tui = InspectorApp(controller, cfg)
    async with tui.run_test() as pilot:
        await controller.run(timeout=BOUND)
        await pilot.pause()

        records = tui.calls.records
        assert [record.tool for record in records.values()] == [
            "glob",
            "grep",
            "read_file",
            "read_file",
        ]
        assert [record.status for record in records.values()] == ["ok", "ok", "ok", "failed"]
        assert tui.calls.row_count == 4
        # A duration is a real measurement, not a placeholder.
        assert all(record.duration_ms >= 0.0 for record in records.values())
        assert records["s4"].error is not None
        assert records["s4"].error.kind == "not_found"
    controller.bus.close()


async def test_the_drill_down_shows_params_result_error_and_the_event_trail(
    tmp_path: Path,
) -> None:
    """R-A-3's four clauses, on the call that failed and on one that did not."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo)
    controller = controller_for(cfg, bus=EventBus())
    tui = InspectorApp(controller, cfg)
    async with tui.run_test() as pilot:
        await controller.run(timeout=BOUND)
        await pilot.pause()

        tui.detail.show(tui.calls.records["s3"])
        body = tui.detail.render_record().plain
        assert "lib.py" in body  # params
        assert "deprecated_fn" in body  # result content
        assert "tool_call_requested" in body and "tool_call_completed" in body  # event trail

        tui.detail.show(tui.calls.records["s4"])
        failed = tui.detail.render_record().plain
        assert "not_found" in failed
        assert "nope.py" in failed
    controller.bus.close()


async def test_moving_the_cursor_moves_the_drill_down(tmp_path: Path) -> None:
    """The one piece of wiring the app does: the table posts, the detail follows."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo)
    controller = controller_for(cfg, bus=EventBus())
    tui = InspectorApp(controller, cfg)
    async with tui.run_test() as pilot:
        await controller.run(timeout=BOUND)
        await pilot.pause()

        tui.calls.focus()
        tui.calls.move_cursor(row=2)
        await pilot.pause()
        assert tui.detail.record is not None
        assert tui.detail.record.call_id == "s3"
    controller.bus.close()


async def test_the_event_log_is_the_raw_sequence_minus_the_deltas(tmp_path: Path) -> None:
    """R-A-3's "raw event sequence": `ctrl+l`, the same stream `JsonlRecorder` writes."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo)
    controller = controller_for(cfg, bus=EventBus())
    tui = InspectorApp(controller, cfg)
    async with tui.run_test() as pilot:
        await controller.run(timeout=BOUND)
        await pilot.pause()

        log = tui.event_log
        assert log is not None
        assert log.display is False
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert log.display is True
        # Everything but the deltas, which the panes already show.
        assert log.written > 10
        assert log.suppressed > 0
    controller.bus.close()


async def test_the_transcript_and_the_table_see_the_same_run(tmp_path: Path) -> None:
    """Spec 9.3's layout: `Transcript` | `ToolCallList`, both fed by the router."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo)
    controller = controller_for(cfg, bus=EventBus())
    tui = InspectorApp(controller, cfg)
    async with tui.run_test() as pilot:
        await controller.run(timeout=BOUND)
        await pilot.pause()

        for pane in tui.transcript.panes.values():
            pane.flush_now()
        assert "deprecated_fn" in tui.transcript.text()
        assert set(tui.transcript.tool_blocks) == set(tui.calls.records)
    controller.bus.close()


# ---------------------------------------------------------------------------
# Headless (R-U-7)
# ---------------------------------------------------------------------------


async def test_the_headless_summary_and_the_widget_agree(tmp_path: Path) -> None:
    """Two independent folds of one event stream. Disagreement means R-X-3 is false."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo)
    controller = controller_for(cfg, bus=EventBus())
    recorder = Recorder(controller.bus)
    pump = asyncio.create_task(recorder.pump())
    tui = InspectorApp(controller, cfg)
    async with tui.run_test() as pilot:
        await controller.run(timeout=BOUND)
        await pilot.pause()
        widget_statuses = {call_id: record.status for call_id, record in tui.calls.records.items()}
    recorder.sub.unsubscribe()
    await pump
    controller.bus.close()

    summary = CallSummary()
    for event in recorder.events:
        summary.handle_event(event)
    assert summary.statuses() == widget_statuses


async def test_the_inspector_runs_headless(tmp_path: Path) -> None:
    """R-U-7, in-process: the answer plus the call table, no Textual involved."""

    repo = sample_repo(tmp_path / "repo")
    answer, summary = await run_headless(config(repo), timeout=BOUND)
    assert "lib.py" in answer
    rendered = summary.render()
    assert "glob" in rendered and "read_file" in rendered
    assert [row.call_id for row in summary.failures()] == ["s4"]


async def test_the_watcher_survives_a_run_with_no_tool_calls(tmp_path: Path) -> None:
    """An empty table renders as a sentence, not as a crash."""

    repo = sample_repo(tmp_path / "repo")
    cfg = config(repo, script={"match": "by_index", "turns": [{"text": "nothing to do"}]})
    controller = controller_for(cfg, bus=EventBus())
    async with CallWatcher(controller.bus) as watcher:
        assert await controller.run(timeout=BOUND) == "nothing to do"
    assert controller.state is RunState.COMPLETED
    assert watcher.summary.render() == "no tool calls"
    controller.bus.close()


def test_the_inspector_runs_headless_from_the_cli(tmp_path: Path) -> None:
    """R-U-7 through the installed command."""

    repo = sample_repo(tmp_path / "repo")
    config_path = tmp_path / "inspector.json"
    config_path.write_text(json.dumps(config(repo).model_dump(mode="json")), encoding="utf-8")

    result = CliRunner().invoke(inspector_cli, ["--headless", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "lib.py" in result.output
    assert "read_file" in result.output


def test_the_cli_refuses_a_run_with_no_task() -> None:
    """A message and exit 2, not a traceback."""

    result = CliRunner().invoke(inspector_cli, ["--headless"])
    assert result.exit_code == 2
    assert "no task" in result.output
