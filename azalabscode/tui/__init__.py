"""The Textual presentation layer.

`HarnessApp` is the base class a workflow's TUI subclasses (R-U-2); the widgets in
`tui.widgets` are the shared components it composes (R-U-3). Nothing here reads
anything but the event stream and the `Controller` API (R-U-1), which is what makes
the whole layer replaceable.

**M4 settled the Textual go/no-go (plan decision D5): go.** The R-U-4 perf test --
four concurrent streams at 200 deltas/s each, event-to-screen latency measured at
the paint -- lands at 34-47 ms against a 100 ms budget, and at 62 ms with six
streams. Textual stays and the Rich fallback is not built.

Two things in `StreamPane` are what bought that margin, and undoing either costs
most of it: the buffer is drained in `render()` rather than on the batch timer, so
the timer's delay and the frame's delay overlap instead of adding; and an N-up pane
is fixed-height and renders its own tail, so it never asks Textual to re-arrange
the screen. The property that made the branch cheap is worth keeping regardless --
nothing below `tui/` references a Textual type, and import contract 6 enforces it.

M6 completes spec 8.2's widget table and adds the three reference workflows' apps
on top of it. `HarnessApp` grew two hooks for them: `injection_input()` now has a
`PromptInput` to return, and `send()` is the normal-mode counterpart to
`submit_injection()`.

This package is deliberately absent from `azalabscode/__init__.py`. Importing the
harness must not import Textual: a headless run should not pay for a UI it will not
draw, and a test asserts it does not.
"""

from azalabscode.tui.app import EventLog, HarnessApp
from azalabscode.tui.approval import TUIApprovalHandler
from azalabscode.tui.bindings import HARNESS_BINDINGS
from azalabscode.tui.routing import ANY, EventConsumer, EventRouter
from azalabscode.tui.widgets import (
    AgentInfo,
    AgentTree,
    ApprovalModal,
    DiffView,
    EventWidget,
    PromptInput,
    RunStatusBar,
    SplitPanes,
    StageChip,
    StagePipeline,
    StageState,
    StreamPane,
    StreamStats,
    ToolCallBlock,
    ToolCallDetail,
    ToolCallList,
    ToolCallRecord,
    Transcript,
    highlight_diff,
    render_diff,
)

__all__ = [
    "ANY",
    "HARNESS_BINDINGS",
    "AgentInfo",
    "AgentTree",
    "ApprovalModal",
    "DiffView",
    "EventConsumer",
    "EventLog",
    "EventRouter",
    "EventWidget",
    "HarnessApp",
    "PromptInput",
    "RunStatusBar",
    "SplitPanes",
    "StageChip",
    "StagePipeline",
    "StageState",
    "StreamPane",
    "StreamStats",
    "TUIApprovalHandler",
    "ToolCallBlock",
    "ToolCallDetail",
    "ToolCallList",
    "ToolCallRecord",
    "Transcript",
    "highlight_diff",
    "render_diff",
]
