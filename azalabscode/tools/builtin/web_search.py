"""`web_search`: a thin tool over the `SearchBackend` protocol (spec C-10).

The tool holds no vendor knowledge. It validates parameters, calls the backend,
turns a `SearchError` into the right `ToolErrorKind`, and renders results in a shape
that is useful to both the model and a widget.

Registration is key-conditional: `default_registry` omits this tool when no backend
can be built. A tool the model can see but that always fails is worse than an absent
one -- the model spends turns retrying it.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from azalabscode.permissions import ApprovalPolicy
from azalabscode.toolio import (
    NETWORK_RETRY,
    RetryPolicy,
    ToolDisplay,
    ToolError,
    ToolErrorKind,
    ToolResult,
)
from azalabscode.tools.base import Tool
from azalabscode.tools.context import ToolContext
from azalabscode.tools.search_backends.base import SearchBackend, SearchError, SearchResult

DESCRIPTION = """\
Search the web and get back a ranked list of titles, URLs and snippets.

Use this to find pages. It does not return page contents -- pick the promising URLs \
out of the results and web_fetch them.

Write the query the way you would type it into a search box: keywords, not a \
sentence, and no boolean operators. If the first search is unhelpful, change the \
words rather than raising `n`; the tenth result is rarely the answer when the first \
three are wrong.

`recency` restricts results by age when the answer is time-sensitive (a release, an \
incident, a version number). Leave it unset otherwise -- it discards good older \
sources.

Snippets are search-engine summaries, not quotations. Do not cite them as if you had \
read the page.\
"""

type Recency = Literal["hour", "day", "week", "month", "year"]

_ERROR_KINDS = {
    "timeout": ToolErrorKind.TIMEOUT,
    "http": ToolErrorKind.HTTP,
    "network": ToolErrorKind.NETWORK,
    "configuration": ToolErrorKind.UNAVAILABLE,
}


class WebSearchParams(BaseModel):
    """Parameters for `web_search`."""

    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, description="Search keywords.")
    n: int = Field(default=10, ge=1, le=20, description="Maximum results to return.")
    recency: Recency | None = Field(
        default=None, description="Only results from the last hour/day/week/month/year."
    )


class WebSearchTool(Tool):
    """Query a search backend."""

    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = DESCRIPTION
    Params: ClassVar[type[BaseModel]] = WebSearchParams

    approval: ApprovalPolicy = "never"
    timeout: float = 20.0
    retry: RetryPolicy = NETWORK_RETRY
    concurrency_safe: ClassVar[bool] = True
    read_only: ClassVar[bool] = True

    def __init__(self, backend: SearchBackend) -> None:
        self.backend = backend
        super().__init__()

    async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
        """Reject a blank query without spending a request."""

        assert isinstance(params, WebSearchParams)
        if not params.query.strip():
            return ToolError(kind=ToolErrorKind.INVALID_PARAMS, message="query must not be blank")
        return None

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Search."""

        assert isinstance(params, WebSearchParams)
        try:
            results = await self.backend.search(params.query, n=params.n, recency=params.recency)
        except SearchError as exc:
            return ToolResult.failure(
                _ERROR_KINDS.get(exc.kind, ToolErrorKind.NETWORK),
                f"search failed: {exc}",
                meta={"backend": getattr(self.backend, "name", "unknown")},
            )

        if not results:
            return ToolResult.ok_text(
                f"No results for {params.query!r}. Try different keywords; if you set "
                f"recency, try without it.",
                display=ToolDisplay(kind="search", data={"query": params.query, "results": []}),
                meta={"backend": getattr(self.backend, "name", "unknown"), "count": 0},
            )

        return ToolResult.ok_text(
            _render(params.query, results),
            display=ToolDisplay(
                kind="search",
                data={
                    "query": params.query,
                    "results": [r.model_dump(mode="json") for r in results],
                },
            ),
            meta={
                "backend": getattr(self.backend, "name", "unknown"),
                "count": len(results),
                "query": params.query,
            },
        )

    async def aclose(self) -> None:
        """Close the backend."""

        await self.backend.aclose()


def _render(query: str, results: list[SearchResult]) -> str:
    """Numbered results. The URL goes on its own line so it survives wrapping."""

    lines = [f"{len(results)} result(s) for {query!r}:", ""]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   {r.url}")
        if r.published:
            lines.append(f"   published: {r.published}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = ["DESCRIPTION", "Recency", "WebSearchParams", "WebSearchTool"]
