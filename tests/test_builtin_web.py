"""`web_fetch` and `web_search` against a mocked transport.

No test here touches the network. `respx` supplies the responses and the DNS
resolver is injected, which is what makes the address-blocking tests meaningful:
the interesting case is a *public* hostname resolving to a private address, and that
cannot be arranged against the real internet.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from azalabscode.toolio import ToolErrorKind
from azalabscode.tools import StaticSearchBackend, ToolContext
from azalabscode.tools.builtin.web_fetch import (
    BlockedURL,
    WebFetchParams,
    WebFetchTool,
    _check_url,
    _same_site,
    normalise_url,
)
from azalabscode.tools.builtin.web_search import WebSearchParams, WebSearchTool
from azalabscode.tools.search_backends.base import SearchError, SearchResult
from azalabscode.tools.search_backends.serper import SerperBackend, serper_from_env


def public_resolver(host: str) -> list[str]:
    """Every hostname resolves to a public address."""

    return ["93.184.216.34"]


def private_resolver(host: str) -> list[str]:
    """Every hostname resolves to a loopback address -- the DNS-rebinding shape."""

    return ["127.0.0.1"]


def make_tool(**kwargs: Any) -> WebFetchTool:
    return WebFetchTool(resolver=kwargs.pop("resolver", public_resolver), **kwargs)


async def fetch(tool: WebFetchTool, ctx: ToolContext, url: str, **kwargs: object):
    params = WebFetchParams(url=url, **kwargs)  # type: ignore[arg-type]
    error = await tool.validate_params(params, ctx)
    if error is not None:
        from azalabscode.toolio import ToolResult

        return ToolResult.failure(error.kind, error.message)
    return await tool.run(params, ctx)


# ---------------------------------------------------------------------------
# Address and scheme checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "javascript:alert(1)",
        "data:text/html,hi",
    ],
)
def test_only_http_and_https_are_fetchable(url: str) -> None:
    with pytest.raises(BlockedURL, match="only http and https"):
        _check_url(url, allow_private=False, resolve=public_resolver)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://localhost/x",
        "http://10.0.0.5/x",
        "http://192.168.1.1/x",
        "http://172.16.0.1/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/x",
        "http://0.0.0.0/x",
    ],
)
def test_private_and_local_addresses_are_refused(url: str) -> None:
    """`169.254.169.254` is the case that matters most: an agent that can reach it
    can read cloud instance credentials."""

    with pytest.raises(BlockedURL, match="private or local"):
        _check_url(url, allow_private=False, resolve=private_resolver)


def test_a_public_hostname_resolving_to_a_private_address_is_refused() -> None:
    """Blocking `localhost` by string does nothing about DNS pointing elsewhere. The
    check has to happen after resolution, which is why the resolver is a seam."""

    with pytest.raises(BlockedURL, match="private or local"):
        _check_url("https://totally-public.example/", allow_private=False, resolve=private_resolver)


def test_a_public_address_passes() -> None:
    _check_url("https://example.com/page", allow_private=False, resolve=public_resolver)


def test_a_hostname_that_does_not_resolve_is_refused_not_attempted() -> None:
    def failing(host: str) -> list[str]:
        raise OSError("Name or service not known")

    with pytest.raises(BlockedURL, match="could not resolve"):
        _check_url("https://nope.invalid/", allow_private=False, resolve=failing)


def test_allow_private_turns_the_check_off_entirely() -> None:
    """The escape hatch exists for a local dev server; it is off by default."""

    _check_url("http://127.0.0.1:8000/", allow_private=True, resolve=private_resolver)


def test_a_url_with_no_host_is_refused() -> None:
    with pytest.raises(BlockedURL, match="no host"):
        _check_url("http:///path", allow_private=False, resolve=public_resolver)


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("https://example.com/a", "https://example.com/b", True),
        ("https://example.com/a", "https://www.example.com/b", True),
        ("https://www.example.com/a", "https://example.com/b", True),
        ("http://example.com/a", "https://example.com/b", True),
        ("https://example.com/a", "http://example.com/b", False),
        ("https://example.com/a", "https://evil.com/b", False),
        ("https://example.com/a", "https://sub.example.com/b", False),
        ("https://example.com/a", "file:///etc/passwd", False),
    ],
)
def test_same_site_allows_www_and_upgrades_but_nothing_else(a: str, b: str, same: bool) -> None:
    """A downgrade from https to http is a redirect worth refusing, and a subdomain
    is a different host as far as this check is concerned."""

    assert _same_site(a, b) is same


def test_the_fragment_is_dropped() -> None:
    assert normalise_url("https://x.test/a#frag") == "https://x.test/a"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


@respx.mock
async def test_html_is_extracted_to_readable_text(tool_ctx: ToolContext) -> None:
    html = (
        "<html><head><title>T</title><script>var x=1;</script></head>"
        "<body><nav>menu menu menu</nav><article><p>"
        "The important sentence that the page is actually about, repeated enough "
        "that an extractor recognises it as the body content of this document."
        "</p></article></body></html>"
    )
    respx.get("https://example.com/page").mock(
        return_value=httpx.Response(200, html=html, headers={"content-type": "text/html"})
    )

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://example.com/page")

    assert result.ok is True
    assert "important sentence" in result.text
    assert "var x=1" not in result.text


@respx.mock
async def test_raw_returns_the_body_unprocessed(tool_ctx: ToolContext) -> None:
    respx.get("https://example.com/page").mock(
        return_value=httpx.Response(
            200, html="<html><body><p>hi</p></body></html>", headers={"content-type": "text/html"}
        )
    )

    async with httpx.AsyncClient() as client:
        result = await fetch(
            make_tool(client=client), tool_ctx, "https://example.com/page", raw=True
        )

    assert "<p>hi</p>" in result.text
    assert result.meta["mode"] == "raw"


@respx.mock
async def test_json_passes_through(tool_ctx: ToolContext) -> None:
    respx.get("https://api.example.com/v1").mock(
        return_value=httpx.Response(200, json={"ok": True, "items": [1, 2]})
    )

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://api.example.com/v1")

    assert '"items"' in result.text


@respx.mock
async def test_a_same_site_redirect_is_followed(tool_ctx: ToolContext) -> None:
    respx.get("https://example.com/old").mock(
        return_value=httpx.Response(302, headers={"location": "https://www.example.com/new"})
    )
    respx.get("https://www.example.com/new").mock(
        return_value=httpx.Response(200, text="arrived", headers={"content-type": "text/plain"})
    )

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://example.com/old")

    assert result.ok is True
    assert "arrived" in result.text
    assert result.meta["redirects"] == ["https://www.example.com/new"]


@respx.mock
async def test_a_cross_host_redirect_is_returned_rather_than_followed(
    tool_ctx: ToolContext,
) -> None:
    """The model makes a fresh, separately-checked call. That keeps every host
    actually reached visible in the transcript."""

    respx.get("https://example.com/out").mock(
        return_value=httpx.Response(302, headers={"location": "https://elsewhere.test/landing"})
    )

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://example.com/out")

    assert result.ok is True
    assert "https://elsewhere.test/landing" in result.text
    assert "not followed automatically" in result.text
    assert result.display is not None
    assert result.display.kind == "redirect"


@respx.mock
async def test_a_redirect_to_a_private_address_is_refused_on_the_second_hop(
    tool_ctx: ToolContext,
) -> None:
    """The hop is re-checked. A check that only looked at the URL the model supplied
    would walk straight past this."""

    calls = {"n": 0}

    def resolver(host: str) -> list[str]:
        calls["n"] += 1
        return ["93.184.216.34"] if calls["n"] == 1 else ["169.254.169.254"]

    respx.get("https://example.com/a").mock(
        return_value=httpx.Response(302, headers={"location": "https://example.com/b"})
    )

    async with httpx.AsyncClient() as client:
        tool = WebFetchTool(client=client, resolver=resolver)
        result = await fetch(tool, tool_ctx, "https://example.com/a")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.PERMISSION


@respx.mock
async def test_a_redirect_loop_is_bounded(tool_ctx: ToolContext) -> None:
    respx.get("https://example.com/loop").mock(
        return_value=httpx.Response(302, headers={"location": "https://example.com/loop"})
    )

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://example.com/loop")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.HTTP
    assert "redirects" in result.error.message


@respx.mock
async def test_a_redirect_with_no_location_is_an_http_error(tool_ctx: ToolContext) -> None:
    respx.get("https://example.com/x").mock(return_value=httpx.Response(302))

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://example.com/x")

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.HTTP


@respx.mock
async def test_a_4xx_is_reported_with_its_status(tool_ctx: ToolContext) -> None:
    respx.get("https://example.com/missing").mock(return_value=httpx.Response(404))

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://example.com/missing")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.HTTP
    assert "404" in result.error.message


@respx.mock
async def test_a_network_failure_is_a_network_error(tool_ctx: ToolContext) -> None:
    respx.get("https://example.com/x").mock(side_effect=httpx.ConnectError("refused"))

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://example.com/x")

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.NETWORK


@respx.mock
async def test_a_timeout_is_a_timeout_error(tool_ctx: ToolContext) -> None:
    respx.get("https://example.com/x").mock(side_effect=httpx.ReadTimeout("slow"))

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://example.com/x")

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.TIMEOUT


@respx.mock
async def test_a_body_over_the_size_cap_is_refused(tool_ctx: ToolContext) -> None:
    tool_ctx.config.max_fetch_bytes = 100
    respx.get("https://example.com/big").mock(
        return_value=httpx.Response(200, text="x" * 500, headers={"content-type": "text/plain"})
    )

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://example.com/big")

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.UNSUPPORTED


@respx.mock
async def test_a_binary_content_type_is_refused(tool_ctx: ToolContext) -> None:
    respx.get("https://example.com/blob").mock(
        return_value=httpx.Response(
            200, content=b"\x00\x01", headers={"content-type": "application/octet-stream"}
        )
    )

    async with httpx.AsyncClient() as client:
        result = await fetch(make_tool(client=client), tool_ctx, "https://example.com/blob")

    assert result.error is not None
    assert result.error.kind is ToolErrorKind.UNSUPPORTED


@respx.mock
async def test_output_is_truncated_at_max_chars_with_a_marker(tool_ctx: ToolContext) -> None:
    respx.get("https://example.com/long").mock(
        return_value=httpx.Response(200, text="y" * 5000, headers={"content-type": "text/plain"})
    )

    async with httpx.AsyncClient() as client:
        result = await fetch(
            make_tool(client=client), tool_ctx, "https://example.com/long", max_chars=200
        )

    assert "truncated at 200 characters of 5000" in result.text
    assert result.display is not None
    assert result.display.data["truncated"] is True


def test_web_fetch_retries_only_on_network_and_http() -> None:
    """R-T-4: a GET is idempotent, so retries are legitimate here -- and only here."""

    tool = make_tool()
    assert tool.approval == "never"
    assert tool.retry.attempts == 2
    assert ToolErrorKind.PERMISSION not in tool.retry.retry_on
    assert ToolErrorKind.NETWORK in tool.retry.retry_on


def test_web_fetch_is_concurrency_safe() -> None:
    assert make_tool().concurrency_safe is True


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


async def test_search_renders_numbered_results(tool_ctx: ToolContext) -> None:
    backend = StaticSearchBackend(
        [
            SearchResult(title="First", url="https://a.test/1", snippet="about a"),
            SearchResult(title="Second", url="https://b.test/2", published="2026-01-01"),
        ]
    )
    tool = WebSearchTool(backend)
    result = await tool.run(WebSearchParams(query="thing"), tool_ctx)

    assert result.ok is True
    assert "1. First" in result.text
    assert "https://a.test/1" in result.text
    assert "published: 2026-01-01" in result.text
    assert backend.queries == ["thing"]


async def test_search_results_are_structured_for_a_widget(tool_ctx: ToolContext) -> None:
    """Spec delta 12 again: the UI must not parse the model-facing rendering."""

    tool = WebSearchTool(StaticSearchBackend([SearchResult(title="T", url="https://x.test/")]))
    result = await tool.run(WebSearchParams(query="q"), tool_ctx)

    assert result.display is not None
    assert result.display.data["results"][0]["url"] == "https://x.test/"


async def test_no_results_suggests_what_to_change(tool_ctx: ToolContext) -> None:
    tool = WebSearchTool(StaticSearchBackend([]))
    result = await tool.run(WebSearchParams(query="q", recency="day"), tool_ctx)

    assert result.ok is True
    assert "No results" in result.text
    assert "recency" in result.text


async def test_n_bounds_the_result_count(tool_ctx: ToolContext) -> None:
    backend = StaticSearchBackend(
        [SearchResult(title=str(i), url=f"https://x.test/{i}") for i in range(10)]
    )
    tool = WebSearchTool(backend)
    result = await tool.run(WebSearchParams(query="q", n=3), tool_ctx)

    assert result.meta["count"] == 3


async def test_a_blank_query_is_rejected_without_spending_a_request(
    tool_ctx: ToolContext,
) -> None:
    backend = StaticSearchBackend([])
    tool = WebSearchTool(backend)
    error = await tool.validate_params(WebSearchParams(query="   "), tool_ctx)

    assert error is not None
    assert error.kind is ToolErrorKind.INVALID_PARAMS
    assert backend.queries == []


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("timeout", ToolErrorKind.TIMEOUT),
        ("http", ToolErrorKind.HTTP),
        ("network", ToolErrorKind.NETWORK),
        ("configuration", ToolErrorKind.UNAVAILABLE),
    ],
)
async def test_a_backend_failure_maps_to_the_right_error_kind(
    tool_ctx: ToolContext, kind: str, expected: ToolErrorKind
) -> None:
    class Failing:
        name = "failing"

        async def search(self, query: str, *, n: int = 10, recency: str | None = None):
            raise SearchError("backend is down", kind=kind)

        async def aclose(self) -> None:
            return None

    tool = WebSearchTool(Failing())
    result = await tool.run(WebSearchParams(query="q"), tool_ctx)

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is expected


# ---------------------------------------------------------------------------
# Serper adapter
# ---------------------------------------------------------------------------


@respx.mock
async def test_serper_parses_organic_results() -> None:
    respx.post("https://google.serper.dev/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Result",
                        "link": "https://x.test/1",
                        "snippet": "text",
                        "date": "2 days ago",
                    },
                    {"title": "No link"},
                ],
                "knowledgeGraph": {"title": "ignored"},
            },
        )
    )

    async with httpx.AsyncClient() as client:
        backend = SerperBackend("key", client=client)
        results = await backend.search("query")

    assert len(results) == 1
    assert results[0].url == "https://x.test/1"
    assert results[0].published == "2 days ago"


@respx.mock
async def test_serper_maps_recency_to_a_time_filter() -> None:
    route = respx.post("https://google.serper.dev/search").mock(
        return_value=httpx.Response(200, json={"organic": []})
    )

    async with httpx.AsyncClient() as client:
        await SerperBackend("key", client=client).search("q", recency="week")

    import json

    assert json.loads(route.calls[0].request.content)["tbs"] == "qdr:w"


@respx.mock
async def test_a_serper_http_error_becomes_a_search_error() -> None:
    respx.post("https://google.serper.dev/search").mock(return_value=httpx.Response(403))

    async with httpx.AsyncClient() as client:
        backend = SerperBackend("key", client=client)
        with pytest.raises(SearchError) as exc:
            await backend.search("q")

    assert exc.value.kind == "http"
    assert exc.value.status == 403


def test_serper_without_a_key_refuses_to_construct() -> None:
    with pytest.raises(SearchError, match="no Serper API key"):
        SerperBackend("")


def test_serper_from_env_returns_none_when_no_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registration switch: `default_registry` omits `web_search` rather than
    shipping a tool that always fails (D4)."""

    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    assert serper_from_env() is None


def test_web_search_is_absent_from_the_registry_without_a_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azalabscode.tools import default_registry

    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    assert "web_search" not in default_registry().names()


def test_web_search_is_registered_when_a_backend_is_supplied() -> None:
    from azalabscode.tools import default_registry

    registry = default_registry(search_backend=StaticSearchBackend([]))
    assert "web_search" in registry.names()


def test_the_static_backend_satisfies_the_protocol() -> None:
    from azalabscode.tools.search_backends.base import SearchBackend

    assert isinstance(StaticSearchBackend([]), SearchBackend)
