"""Tool definitions, the dispatcher, and the built-in tools.

The vocabulary a tool *produces* -- `ToolResult`, `ToolError`, `ToolSchema`,
`RetryPolicy` -- lives in `azalabscode.toolio`, below this layer, so that `messages`
can carry a tool result without importing `tools` (spec delta 1). This layer holds
the policy: what a tool is, how one is run, and the eight built-ins.

The layering seam is `contracts.PermissionGate`: the dispatcher depends on the
protocol, `control` injects the implementation (spec delta 3). `AllowAllGate`,
`DenyAllGate` and `RecordingGate` ship here so the tool layer is usable and testable
standalone.

Everything exported here is public API (R-X-6).
"""

from azalabscode.tools.base import (
    NoParams,
    Tool,
    ToolSet,
    describe_validation_error,
)
from azalabscode.tools.budget import (
    TurnBudget,
    apply_result_cap,
    result_text_size,
)
from azalabscode.tools.builtin.delegate import DelegateParams, DelegateTool
from azalabscode.tools.builtin.edit_file import EditFileParams, EditFileTool
from azalabscode.tools.builtin.glob import GlobParams, GlobTool
from azalabscode.tools.builtin.grep import GrepParams, GrepTool
from azalabscode.tools.builtin.read_file import ReadFileParams, ReadFileTool
from azalabscode.tools.builtin.shell import ShellParams, ShellTool
from azalabscode.tools.builtin.web_fetch import WebFetchParams, WebFetchTool
from azalabscode.tools.builtin.web_search import WebSearchParams, WebSearchTool
from azalabscode.tools.builtin.write_file import WriteFileParams, WriteFileTool
from azalabscode.tools.context import (
    ReadRecord,
    ReadState,
    ToolConfig,
    ToolContext,
    ToolPathError,
    WorkspaceConfig,
)
from azalabscode.tools.dispatcher import (
    DEFAULT_MAX_PARALLEL,
    Batch,
    Prepared,
    ToolCall,
    ToolDispatcher,
    partition_runs,
)
from azalabscode.tools.gates import AllowAllGate, DenyAllGate, GateCheck, RecordingGate
from azalabscode.tools.platform import (
    CURRENT_PLATFORM,
    KillOutcome,
    ShellSpec,
    describe_platform,
    kill_tree,
    resolve_shell,
    spawn,
)
from azalabscode.tools.registry import BUILTIN_NAMES, default_registry, read_only_registry
from azalabscode.tools.search_backends.base import (
    SearchBackend,
    SearchError,
    SearchResult,
    StaticSearchBackend,
)
from azalabscode.tools.search_backends.serper import SerperBackend, serper_from_env

__all__ = [
    "BUILTIN_NAMES",
    "CURRENT_PLATFORM",
    "DEFAULT_MAX_PARALLEL",
    "AllowAllGate",
    "Batch",
    "DelegateParams",
    "DelegateTool",
    "DenyAllGate",
    "EditFileParams",
    "EditFileTool",
    "GateCheck",
    "GlobParams",
    "GlobTool",
    "GrepParams",
    "GrepTool",
    "KillOutcome",
    "NoParams",
    "Prepared",
    "ReadFileParams",
    "ReadFileTool",
    "ReadRecord",
    "ReadState",
    "RecordingGate",
    "SearchBackend",
    "SearchError",
    "SearchResult",
    "SerperBackend",
    "ShellParams",
    "ShellSpec",
    "ShellTool",
    "StaticSearchBackend",
    "Tool",
    "ToolCall",
    "ToolConfig",
    "ToolContext",
    "ToolDispatcher",
    "ToolPathError",
    "ToolSet",
    "TurnBudget",
    "WebFetchParams",
    "WebFetchTool",
    "WebSearchParams",
    "WebSearchTool",
    "WorkspaceConfig",
    "WriteFileParams",
    "WriteFileTool",
    "apply_result_cap",
    "default_registry",
    "describe_platform",
    "describe_validation_error",
    "kill_tree",
    "partition_runs",
    "read_only_registry",
    "resolve_shell",
    "result_text_size",
    "serper_from_env",
    "spawn",
]
