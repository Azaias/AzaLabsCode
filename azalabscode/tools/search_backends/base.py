"""The `SearchBackend` protocol and its result type (spec C-10).

`web_search` needs a vendor the intent does not budget for. The protocol is the
answer: Serper ships in v1, the tool depends on this shape, and swapping in Brave or
Tavily later is a new adapter rather than a change to the tool.

`SearchResult` is a leaf-shaped Pydantic model so a result can be serialized into a
`ToolResult.display` and rendered by a widget without re-parsing model-facing text.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from azalabscode.schema import HarnessModel


class SearchResult(HarnessModel):
    """One search hit, normalised across backends."""

    title: str
    url: str
    snippet: str = ""
    published: str | None = None
    """Whatever date string the backend supplied. Not parsed: backends disagree about
    format and a wrong date is worse than an opaque one."""


class SearchError(Exception):
    """A backend could not answer.

    Raised rather than returned because a backend is not a tool: `web_search`
    converts it into the `ToolResult` the model sees, with the right error kind.
    """

    def __init__(self, message: str, *, kind: str = "network", status: int | None = None) -> None:
        self.kind = kind
        self.status = status
        super().__init__(message)


@runtime_checkable
class SearchBackend(Protocol):
    """What `web_search` asks of a search vendor."""

    name: str

    async def search(
        self, query: str, *, n: int = 10, recency: str | None = None
    ) -> list[SearchResult]:
        """Return up to `n` results. Raise `SearchError` on failure."""
        ...

    async def aclose(self) -> None:
        """Release any client the backend holds."""
        ...


class StaticSearchBackend:
    """A backend that returns a fixed list. For tests and offline demos.

    Shipped rather than confined to the test tree because `SERPER_API_KEY` is not
    always present, and a `web_search` that cannot be exercised at all is a
    `web_search` nobody notices is broken.
    """

    name = "static"

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results or []
        self.queries: list[str] = []

    async def search(
        self, query: str, *, n: int = 10, recency: str | None = None
    ) -> list[SearchResult]:
        """Record the query and return the configured results."""

        self.queries.append(query)
        return self.results[:n]

    async def aclose(self) -> None:
        """Nothing to close."""

        return None


__all__ = ["SearchBackend", "SearchError", "SearchResult", "StaticSearchBackend"]
