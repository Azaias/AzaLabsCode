"""The M6 widgets (R-U-3): the seven spec 8.2 names that M4 did not build.

Each widget is driven the way the app drives it -- `handle_event` with real event
objects -- rather than through a private setter, because that is the contract the
router relies on and the only one a workflow author can depend on (R-U-1).

The widgets are mounted inside a real `App.run_test()` even where the assertion is
about folding rather than rendering. A `DataTable` that works detached and raises
once it has a screen is a widget that works in this file and nowhere else.
"""

from __future__ import annotations

from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widget import Widget

from azalabscode.events import (
    AgentFinished,
    AgentPhaseChanged,
    AgentSpawned,
    ApprovalRequested,
    ApprovalResolved,
    Checkpoint,
    Event,
    ModelDelta,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallProgress,
    ToolCallRequested,
    ToolCallStarted,
)
from azalabscode.messages import Usage
from azalabscode.permissions import ApprovalRequest, ApprovalSummary, Decision
from azalabscode.runstate import AgentPhase
from azalabscode.toolio import ToolDisplay, ToolError, ToolErrorKind, ToolResult
from azalabscode.tui import (
    AgentTree,
    DiffView,
    PromptInput,
    SplitPanes,
    StagePipeline,
    StreamPane,
    ToolCallDetail,
    ToolCallList,
)
from azalabscode.tui.widgets.diff_view import highlight_diff, syntax_for
from azalabscode.tui.widgets.stage_pipeline import StageChip
from azalabscode.tui.widgets.tool_calls import summarize

DIFF = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1,2 +1,2 @@\n-old = 1\n+new = 2\n unchanged\n"


class WidgetApp(App[None]):
    """Mounts whatever a test hands it. No controller: these widgets need none."""

    def __init__(self, *widgets: Widget) -> None:
        super().__init__()
        self._widgets = widgets

    def compose(self) -> ComposeResult:
        yield from self._widgets


def feed(widget: Any, *events: Event) -> None:
    """Deliver events the way `EventRouter` does."""

    for index, event in enumerate(events, start=1):
        event.seq = index
        widget.handle_event(event)


def approval(**overrides: Any) -> ApprovalRequest:
    """A pending `edit_file` request with a diff on it."""

    fields: dict[str, Any] = {
        "run_id": "run_test",
        "agent_id": "main",
        "call_id": "call_1",
        "tool": "edit_file",
        "params": {"path": "src/app.py", "old": "old = 1", "new": "new = 2"},
        "summary": ApprovalSummary(title="edit_file src/app.py", detail="1 replacement", diff=DIFF),
    }
    fields.update(overrides)
    return ApprovalRequest(**fields)


def edit_result() -> ToolResult:
    """What `edit_file` returns: text for the model, a diff for the widgets."""

    return ToolResult.ok_text(
        "Edited src/app.py: 1 replacement(s).",
        display=ToolDisplay(
            kind="diff",
            data={"path": "src/app.py", "diff": DIFF, "replacements": 1},
        ),
    )


# ---------------------------------------------------------------------------
# ToolCallList / ToolCallDetail (R-U-3, R-A-3)
# ---------------------------------------------------------------------------


async def test_a_call_folds_from_requested_through_completed() -> None:
    """The table's row is a fold of the event stream and nothing else (R-U-1)."""

    table = ToolCallList()
    app = WidgetApp(table)
    request = approval(call_id="c1")
    async with app.run_test():
        feed(
            table,
            ToolCallRequested(
                call_id="c1", tool="edit_file", params={"path": "a.py"}, agent_id="main"
            ),
            ApprovalRequested(request=request, agent_id="main"),
            ApprovalResolved(
                request_id=request.request_id,
                decision=Decision.approve(by="user"),
                agent_id="main",
            ),
            ToolCallStarted(call_id="c1", tool="edit_file", agent_id="main"),
            ToolCallCompleted(
                call_id="c1", tool="edit_file", result=edit_result(), duration_ms=1234.0
            ),
        )
        record = table.records["c1"]
        assert record.tool == "edit_file"
        assert record.status == "ok"
        assert record.duration_text() == "1.23s"
        assert record.result is not None and record.result.display is not None
        assert table.row_count == 1
        # Every event is on the trail, which is what the drill-down renders.
        assert [event.type for event in record.trail] == [
            "tool_call_requested",
            "approval_requested",
            "approval_resolved",
            "tool_call_started",
            "tool_call_completed",
        ]


async def test_a_denied_call_reads_as_denied_not_failed() -> None:
    """A denial is a decision, not a fault, and the table says so."""

    table = ToolCallList()
    request = approval(call_id="c1", tool="shell")
    async with WidgetApp(table).run_test():
        feed(
            table,
            ToolCallRequested(call_id="c1", tool="shell", params={"command": "rm -rf /"}),
            ApprovalRequested(request=request),
            ApprovalResolved(
                request_id=request.request_id, decision=Decision.deny("no", by="user")
            ),
        )
        assert table.records["c1"].status == "denied"
        assert table.records["c1"].detail == "no"


async def test_the_table_caps_its_rows() -> None:
    """A table is a view, not a store: the oldest row goes and its record with it."""

    table = ToolCallList(max_rows=3)
    async with WidgetApp(table).run_test():
        for index in range(6):
            feed(table, ToolCallRequested(call_id=f"c{index}", tool="read_file", params={}))
        assert table.row_count == 3
        assert set(table.records) == {"c3", "c4", "c5"}


async def test_an_unrelated_event_does_not_make_a_row() -> None:
    """`ModelDelta` and `Checkpoint` have no `call_id` to fold and are ignored."""

    table = ToolCallList()
    async with WidgetApp(table).run_test():
        feed(table, ModelDelta(call_id="m1", text="hi"), Checkpoint(kind="turn_start"))
        assert table.row_count == 0


async def test_the_detail_view_shows_params_result_and_trail() -> None:
    """R-A-3's drill-down: params, result, error, timing and the raw events."""

    table = ToolCallList()
    detail = ToolCallDetail()
    async with WidgetApp(table, detail).run_test():
        feed(
            table,
            ToolCallRequested(call_id="c1", tool="grep", params={"pattern": "deprecated_fn"}),
            ToolCallStarted(call_id="c1", tool="grep"),
            ToolCallCompleted(
                call_id="c1",
                tool="grep",
                result=ToolResult.ok_text("3 matches"),
                duration_ms=42.0,
            ),
        )
        detail.show(table.records["c1"])
        body = detail.render_record().plain
        assert "grep" in body
        assert "deprecated_fn" in body
        assert "3 matches" in body
        assert "0.04s" in body
        assert "tool_call_completed" in body


async def test_the_detail_view_follows_the_call_it_is_showing() -> None:
    """A drill-down opened on a running call updates instead of freezing."""

    table = ToolCallList()
    detail = ToolCallDetail()
    async with WidgetApp(table, detail).run_test():
        feed(table, ToolCallRequested(call_id="c1", tool="shell", params={"command": "pytest"}))
        detail.show(table.records["c1"])
        assert "progress" not in detail.render_record().plain

        progress = ToolCallProgress(call_id="c1", tool="shell", text="collected 42 items")
        table.handle_event(progress)
        detail.handle_event(progress)
        assert "collected 42 items" in detail.render_record().plain


async def test_the_detail_view_renders_an_error() -> None:
    """A failure shows the kind, the message and the structured details."""

    table = ToolCallList()
    detail = ToolCallDetail()
    async with WidgetApp(table, detail).run_test():
        feed(
            table,
            ToolCallRequested(call_id="c1", tool="read_file", params={"path": "missing.py"}),
            ToolCallFailed(
                call_id="c1",
                tool="read_file",
                error=ToolError(
                    kind=ToolErrorKind.NOT_FOUND,
                    message="no such file",
                    details={"path": "missing.py"},
                ),
                duration_ms=3.0,
            ),
        )
        detail.show(table.records["c1"])
        body = detail.render_record().plain
        assert "not_found" in body
        assert "no such file" in body
        assert "missing.py" in body


def test_params_are_summarized_in_declaration_order() -> None:
    """The table's params column is one line, truncated, never reordered."""

    assert summarize({"path": "a.py", "limit": 10}) == "path=a.py, limit=10"
    assert summarize({}) == ""
    assert summarize({"command": "x" * 200}).endswith("…")


# ---------------------------------------------------------------------------
# DiffView (R-U-3, R-A-1)
# ---------------------------------------------------------------------------


async def test_a_pending_edit_and_a_completed_one_are_both_shown() -> None:
    """Spec 8.2 gives `DiffView` both events; they mean different things."""

    view = DiffView()
    async with WidgetApp(view).run_test():
        assert view.empty
        view.handle_event(ApprovalRequested(request=approval()))
        assert view.current == ("src/app.py", DIFF, True)
        assert "pending" in str(view.border_title)

        view.handle_event(
            ToolCallCompleted(call_id="c1", tool="edit_file", result=edit_result(), duration_ms=1.0)
        )
        assert view.current == ("src/app.py", DIFF, False)
        assert len(view.history) == 2
        view.previous()
        assert view.current is not None and view.current[2] is True


async def test_a_result_with_no_diff_is_not_shown() -> None:
    """`read_file` completes with a `file` display; a diff view has nothing to say."""

    view = DiffView()
    async with WidgetApp(view).run_test():
        view.handle_event(
            ToolCallCompleted(
                call_id="c1",
                tool="read_file",
                result=ToolResult.ok_text("...", display=ToolDisplay(kind="file", data={})),
            )
        )
        assert view.empty


def test_a_diff_is_highlighted_by_extension_and_survives_one_that_is_not() -> None:
    """Syntax highlighting is by path; an unknown extension is not an error."""

    assert syntax_for("src/app.py") is not None
    assert syntax_for("LICENSE") is None
    assert syntax_for(None) is None

    rendered = highlight_diff(DIFF, "src/app.py")
    assert "-old = 1" in rendered.plain
    assert "+new = 2" in rendered.plain
    # The gutter is coloured by the diff's grammar, not by the lexer.
    assert any(span.style == "bold green" for span in rendered.spans)

    plain = highlight_diff(DIFF, "notes.unknownext")
    assert plain.plain == rendered.plain


# ---------------------------------------------------------------------------
# AgentTree (R-U-3, R-U-5)
# ---------------------------------------------------------------------------


async def test_the_tree_nests_children_under_their_parent() -> None:
    """R-U-5: a subagent is a child node, marked and labelled with its own id."""

    tree = AgentTree()
    async with WidgetApp(tree).run_test():
        feed(
            tree,
            AgentSpawned(agent_id="main", spec_summary="main (m)"),
            AgentSpawned(
                agent_id="main/0", parent_id="main", spec_summary="explore (m)", delegated=True
            ),
            AgentPhaseChanged(
                agent_id="main", old=AgentPhase.RUNNING, new=AgentPhase.BLOCKED_ON_CHILD
            ),
            AgentFinished(
                agent_id="main/0",
                outcome="end_turn",
                usage=Usage(prompt_tokens=10, completion_tokens=5),
            ),
        )
        assert tree.children_of("main") == ["main/0"]
        assert tree.agents["main"].phase == "blocked_on_child"
        child = tree.agents["main/0"]
        assert child.delegated and child.finished and child.tokens == 15
        label = child.label(root=False).plain
        assert "main/0" in label and "↳" in label and "end_turn" in label


async def test_a_child_that_arrives_before_its_parent_is_reparented() -> None:
    """A tree that drops an out-of-order spawn is quietly missing a branch."""

    tree = AgentTree()
    async with WidgetApp(tree).run_test():
        feed(tree, AgentSpawned(agent_id="main/0", parent_id="main", spec_summary="explore"))
        assert tree.children_of("main") == ["main/0"]
        feed(tree, AgentSpawned(agent_id="main", spec_summary="main"))
        node = tree._rows["main/0"]
        assert node.parent is tree._rows["main"]


async def test_the_tree_can_be_seeded_from_a_loaded_run() -> None:
    """A loaded run has agents and no events for them (M4's `seed_from_controller`)."""

    tree = AgentTree()

    class FakeState:
        def __init__(self, agent_id: str, parent_id: str | None) -> None:
            self.agent_id = agent_id
            self.parent_id = parent_id
            self.spec_name = "restored"
            self.phase = AgentPhase.PARKED
            self.outcome = None
            self.usage = Usage()

    async with WidgetApp(tree).run_test():
        tree.seed({"main": FakeState("main", None), "main/0": FakeState("main/0", "main")})
        assert tree.children_of("main") == ["main/0"]
        assert tree.agents["main"].phase == "parked"


# ---------------------------------------------------------------------------
# StagePipeline (R-U-3, R-A-2)
# ---------------------------------------------------------------------------


async def test_fan_out_branches_roll_up_into_one_stage() -> None:
    """`models/gpt` is a branch of the `models` stage, not a stage of its own."""

    pipeline = StagePipeline([("models", "models"), ("analyze", "analyze")])
    async with WidgetApp(pipeline).run_test():
        feed(
            pipeline,
            NodeStarted(node_id="models", node_class="FanOut"),
            NodeStarted(node_id="models/a", node_class="ModelCall"),
            NodeStarted(node_id="models/b", node_class="ModelCall"),
            NodeCompleted(node_id="models/a", duration_ms=10.0),
        )
        assert pipeline.state_of("models") == "running"
        assert "1/2" in pipeline.render_line_text()
        feed(
            pipeline,
            NodeCompleted(node_id="models/b"),
            NodeCompleted(node_id="models", duration_ms=20.0),
        )
        assert pipeline.state_of("models") == "ok"
        assert pipeline.state_of("analyze") == "pending"


async def test_the_runners_own_node_id_is_not_a_stage() -> None:
    """`@fusion` is the runner's quiescence key, not a graph node (M5 trap 7)."""

    pipeline = StagePipeline()
    async with WidgetApp(pipeline).run_test():
        feed(pipeline, NodeStarted(node_id="@fusion"), NodeStarted(node_id="join"))
        assert set(pipeline.stages) == {"join"}


async def test_a_failed_child_fails_its_stage() -> None:
    """A fan-out whose branch failed did not succeed, whatever the container says."""

    pipeline = StagePipeline(["models"])
    async with WidgetApp(pipeline).run_test():
        feed(
            pipeline,
            NodeStarted(node_id="models"),
            NodeFailed(node_id="models/a", error="boom"),
            NodeCompleted(node_id="models"),
        )
        assert pipeline.state_of("models") == "failed"


async def test_declared_stage_order_beats_arrival_order() -> None:
    """A pipeline that reordered itself on a late event would be unreadable."""

    pipeline = StagePipeline(["models", "analyze", "synthesize"])
    async with WidgetApp(pipeline).run_test():
        feed(pipeline, NodeStarted(node_id="synthesize"), NodeStarted(node_id="models"))
        rendered = pipeline.render_line_text()
        assert rendered.index("models") < rendered.index("synthesize")


async def test_a_discovered_stage_gets_a_chip() -> None:
    """A graph the app did not describe still draws, in arrival order."""

    pipeline = StagePipeline()
    async with WidgetApp(pipeline).run_test() as pilot:
        feed(pipeline, NodeStarted(node_id="one"), NodeStarted(node_id="two"))
        await pilot.pause()
        assert [chip.stage.node_id for chip in pipeline.query(StageChip)] == ["one", "two"]


# ---------------------------------------------------------------------------
# SplitPanes (R-U-3, R-A-2)
# ---------------------------------------------------------------------------


async def test_the_grid_wraps_past_max_columns() -> None:
    """Four 20-column panes on an 80-column terminal is four unreadable columns."""

    panes = SplitPanes(*[StreamPane(tail=True) for _ in range(4)])
    async with WidgetApp(panes).run_test():
        assert panes.columns_for(2) == 2
        assert panes.columns_for(4) == 3
        assert panes.styles.grid_size_columns == 3
        assert panes.styles.grid_size_rows == 2


async def test_panes_can_be_added_after_mount() -> None:
    """A fusion layout built from a loaded session adds its panes on attach."""

    panes = SplitPanes(columns=2)
    async with WidgetApp(panes).run_test():
        await panes.add_pane(StreamPane(tail=True))
        await panes.add_pane(StreamPane(tail=True))
        assert len(panes.panes) == 2
        assert panes.styles.grid_size_columns == 2


# ---------------------------------------------------------------------------
# PromptInput (R-U-3, spec 8.1)
# ---------------------------------------------------------------------------


async def test_enter_submits_and_alt_enter_is_a_newline() -> None:
    """A prompt box that needs a chord to send is a box that gets abandoned."""

    prompt = PromptInput()
    submitted: list[PromptInput.Submitted] = []

    class PromptApp(WidgetApp):
        def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
            submitted.append(message)

    async with PromptApp(prompt).run_test() as pilot:
        prompt.focus()
        await pilot.press("h", "i")
        await pilot.press("alt+enter")
        await pilot.press("!")
        assert prompt.text == "hi\n!"
        await pilot.press("enter")
        await pilot.pause()
        assert prompt.text == ""
        assert [message.text for message in submitted] == ["hi\n!"]
        assert submitted[0].interrupt is False


async def test_interrupt_mode_is_visible_and_one_shot() -> None:
    """Spec 8.2: `send(text)` normally, `interrupt(text)` after `escape`."""

    prompt = PromptInput()
    submitted: list[PromptInput.Submitted] = []

    class PromptApp(WidgetApp):
        def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
            submitted.append(message)

    async with PromptApp(prompt).run_test() as pilot:
        prompt.focus()
        prompt.set_interrupt_mode(True)
        assert prompt.has_class("-interrupt")
        await pilot.press("x")
        await pilot.press("enter")
        await pilot.pause()
        assert submitted[0].interrupt is True
        # One-shot: the next message is an ordinary send.
        assert not prompt.has_class("-interrupt")
        await pilot.press("y")
        await pilot.press("enter")
        await pilot.pause()
        assert submitted[1].interrupt is False


async def test_escape_leaves_the_box() -> None:
    """A `TextArea` swallows most keys; without this the only way out is the mouse."""

    prompt = PromptInput()
    escaped: list[PromptInput.Escaped] = []

    class PromptApp(WidgetApp):
        def on_prompt_input_escaped(self, message: PromptInput.Escaped) -> None:
            escaped.append(message)

    async with PromptApp(prompt).run_test() as pilot:
        prompt.focus()
        prompt.set_interrupt_mode(True)
        await pilot.press("escape")
        await pilot.pause()
        assert escaped
        assert not prompt.has_class("-interrupt")
        assert not prompt.has_focus


@pytest.mark.parametrize("text", ["", "   "])
async def test_an_empty_submission_still_posts(text: str) -> None:
    """Spec 8.1 gives it a meaning: interrupt without a message."""

    prompt = PromptInput()
    submitted: list[PromptInput.Submitted] = []

    class PromptApp(WidgetApp):
        def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
            submitted.append(message)

    async with PromptApp(prompt).run_test() as pilot:
        prompt.focus()
        prompt.insert(text)
        await pilot.press("enter")
        await pilot.pause()
        assert len(submitted) == 1
        assert submitted[0].text == text
