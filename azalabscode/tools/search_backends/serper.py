"""Serper (`google.serper.dev`) adapter for `SearchBackend`.

Written but **not registered unless `SERPER_API_KEY` is present** (D4, spec 7). A
tool the model can see but that always fails is worse than a tool that is absent:
the model spends turns retrying it.

No key exists on this machine, so this adapter is exercised against a mocked
transport only. Its shape is taken from Serper's documented `/search` response.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from azalabscode.tools.search_backends.base import SearchError, SearchResult

ENDPOINT = "https://google.serper.dev/search"
API_KEY_ENV = "SERPER_API_KEY"

_RECENCY_TO_TBS = {
    "hour": "qdr:h",
    "day": "qdr:d",
    "week": "qdr:w",
    "month": "qdr:m",
    "year": "qdr:y",
}


class SerperBackend:
    """Search via Serper's Google-backed API."""

    name = "serper"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
        endpoint: str = ENDPOINT,
    ) -> None:
        self.api_key = api_key or os.environ.get(API_KEY_ENV, "")
        if not self.api_key:
            raise SearchError(
                f"no Serper API key; set {API_KEY_ENV} or pass api_key",
                kind="configuration",
            )
        self.endpoint = endpoint
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def search(
        self, query: str, *, n: int = 10, recency: str | None = None
    ) -> list[SearchResult]:
        """One `/search` call, normalised into `SearchResult`s."""

        payload: dict[str, Any] = {"q": query, "num": max(1, min(n, 20))}
        if recency and recency in _RECENCY_TO_TBS:
            payload["tbs"] = _RECENCY_TO_TBS[recency]

        try:
            response = await self._get_client().post(
                self.endpoint,
                json=payload,
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise SearchError(f"search timed out: {exc}", kind="timeout") from exc
        except httpx.HTTPError as exc:
            raise SearchError(f"search request failed: {exc}", kind="network") from exc

        if response.status_code >= 400:
            raise SearchError(
                f"search backend returned HTTP {response.status_code}",
                kind="http",
                status=response.status_code,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise SearchError("search backend returned malformed JSON", kind="http") from exc

        return _parse(body, n)

    async def aclose(self) -> None:
        """Close the client if this backend created it."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def _parse(body: dict[str, Any], n: int) -> list[SearchResult]:
    """Pull `organic` results out of a Serper response, ignoring the rest.

    Knowledge panels and "people also ask" blocks are dropped: they have a different
    shape per query and a model given three shapes for one field will guess.
    """

    out: list[SearchResult] = []
    for item in body.get("organic", [])[:n]:
        if not isinstance(item, dict):
            continue
        url = item.get("link") or item.get("url")
        if not url:
            continue
        out.append(
            SearchResult(
                title=str(item.get("title") or url),
                url=str(url),
                snippet=str(item.get("snippet") or ""),
                published=str(item["date"]) if item.get("date") else None,
            )
        )
    return out


def serper_from_env(**kwargs: Any) -> SerperBackend | None:
    """A configured backend, or `None` when no key is set.

    The `None` is the registration switch: `default_registry` omits `web_search`
    entirely rather than shipping a tool that cannot work.
    """

    if not os.environ.get(API_KEY_ENV):
        return None
    try:
        return SerperBackend(**kwargs)
    except SearchError:  # pragma: no cover - key present but rejected
        return None


__all__ = ["API_KEY_ENV", "ENDPOINT", "SerperBackend", "serper_from_env"]
