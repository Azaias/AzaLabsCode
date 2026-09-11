"""Tool-observability reference workflow (spec 9.3, R-A-3).

One agent, every tool it was given, and a UI that shows every call: a live table with
status and duration, a drill-down into params, result, error and the raw event trail,
and the JSONL event log behind it all.
"""

from workflows.inspector.workflow import (
    AGENT_ID,
    CONFIG_TYPE,
    DEFAULT_TOOLS,
    IMPORT_PATH,
    NODE_ID,
    SYSTEM_PROMPT,
    InspectorConfig,
    build,
)

__all__ = [
    "AGENT_ID",
    "CONFIG_TYPE",
    "DEFAULT_TOOLS",
    "IMPORT_PATH",
    "NODE_ID",
    "SYSTEM_PROMPT",
    "InspectorConfig",
    "build",
]
