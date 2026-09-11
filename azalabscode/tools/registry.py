"""Building the built-in toolset.

`default_registry()` is the one call a workflow makes. It exists so the
key-conditional registration of `web_search` (D4, spec 7) happens in exactly one
place: a `web_search` with no backend is omitted rather than shipped broken, because
a tool the model can see and that always fails costs turns.
"""

from __future__ import annotations

from collections.abc import Sequence

from azalabscode.tools.base import Tool, ToolSet
from azalabscode.tools.builtin.delegate import DelegateTool
from azalabscode.tools.builtin.edit_file import EditFileTool
from azalabscode.tools.builtin.glob import GlobTool
from azalabscode.tools.builtin.grep import GrepTool
from azalabscode.tools.builtin.read_file import ReadFileTool
from azalabscode.tools.builtin.shell import ShellTool
from azalabscode.tools.builtin.web_fetch import WebFetchTool
from azalabscode.tools.builtin.web_search import WebSearchTool
from azalabscode.tools.builtin.write_file import WriteFileTool
from azalabscode.tools.search_backends.base import SearchBackend
from azalabscode.tools.search_backends.serper import serper_from_env

BUILTIN_NAMES = (
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "shell",
    "web_fetch",
    "web_search",
    "delegate",
)
"""Every built-in, in the order a model should think about them: read before write,
local before network, delegation last."""


def default_registry(
    *,
    search_backend: SearchBackend | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> ToolSet:
    """Build the built-in toolset.

    `web_search` is registered only when a backend exists -- one passed in, or a
    Serper backend built from `SERPER_API_KEY`. `delegate` is always registered; it
    refuses at call time when the context carries no `Delegator`, which is the only
    place that fact is known.
    """

    wanted = set(include) if include is not None else set(BUILTIN_NAMES)
    wanted -= set(exclude)

    tools: list[Tool] = []
    if "read_file" in wanted:
        tools.append(ReadFileTool())
    if "write_file" in wanted:
        tools.append(WriteFileTool())
    if "edit_file" in wanted:
        tools.append(EditFileTool())
    if "glob" in wanted:
        tools.append(GlobTool())
    if "grep" in wanted:
        tools.append(GrepTool())
    if "shell" in wanted:
        tools.append(ShellTool())
    if "web_fetch" in wanted:
        tools.append(WebFetchTool())
    if "web_search" in wanted:
        backend = search_backend or serper_from_env()
        if backend is not None:
            tools.append(WebSearchTool(backend))
    if "delegate" in wanted:
        tools.append(DelegateTool())

    return ToolSet(tools)


def read_only_registry(*, search_backend: SearchBackend | None = None) -> ToolSet:
    """The built-ins that never change anything.

    What a subagent gets in `manual` mode (R-C-7) when the gate filters by policy,
    and a reasonable default for an inspection workflow.
    """

    return default_registry(
        search_backend=search_backend,
        include=("read_file", "glob", "grep", "web_fetch", "web_search"),
    )


__all__ = ["BUILTIN_NAMES", "default_registry", "read_only_registry"]
