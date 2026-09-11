"""Shared widgets (R-U-3).

M4 landed the four the perf test needs and the base app mounts: `Transcript`,
`StreamPane`, `ApprovalModal` and `RunStatusBar`. M6 completes spec 8.2's table with
`ToolCallList`, `ToolCallDetail`, `DiffView`, `AgentTree`, `StagePipeline`,
`SplitPanes` and `PromptInput` -- the seven the three reference workflows need.

Every widget here consumes the event stream and nothing else (R-U-1). Each one names
the events it consumes in its class docstring, and each one that can be written to at
streaming rates batches on a timer rather than refreshing per event -- `StreamPane`
is the reference for that and the others reuse it.

`SplitPanes` is the exception that proves the rule: it consumes nothing, because it
is a layout and the panes inside it are the consumers.
"""

from azalabscode.tui.widgets.agent_tree import AgentInfo, AgentTree
from azalabscode.tui.widgets.approval_modal import ApprovalModal, render_diff
from azalabscode.tui.widgets.base import EventWidget
from azalabscode.tui.widgets.diff_view import DiffView, highlight_diff
from azalabscode.tui.widgets.prompt_input import PromptInput
from azalabscode.tui.widgets.split_panes import SplitPanes
from azalabscode.tui.widgets.stage_pipeline import StageChip, StagePipeline, StageState
from azalabscode.tui.widgets.status_bar import RunStatusBar
from azalabscode.tui.widgets.stream_pane import FLUSH_INTERVAL_S, StreamPane, StreamStats
from azalabscode.tui.widgets.tool_calls import (
    ToolCallDetail,
    ToolCallList,
    ToolCallRecord,
    failure_status,
)
from azalabscode.tui.widgets.transcript import ToolCallBlock, Transcript

__all__ = [
    "FLUSH_INTERVAL_S",
    "AgentInfo",
    "AgentTree",
    "ApprovalModal",
    "DiffView",
    "EventWidget",
    "PromptInput",
    "RunStatusBar",
    "SplitPanes",
    "StageChip",
    "StagePipeline",
    "StageState",
    "StreamPane",
    "StreamStats",
    "ToolCallBlock",
    "ToolCallDetail",
    "ToolCallList",
    "ToolCallRecord",
    "Transcript",
    "failure_status",
    "highlight_diff",
    "render_diff",
]
