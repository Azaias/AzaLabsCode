"""`web_fetch`: retrieve a URL, with the network treated as hostile.

Three rules, each closing a real hole:

- **Manual redirects.** `follow_redirects=False`, and each hop is re-checked. A
  server that 302s to `http://169.254.169.254/latest/meta-data/` would otherwise walk
  straight past a check that only looked at the URL the model supplied.
- **Same host, modulo `www`.** A cross-host redirect is *returned to the model* as an
  instruction to fetch the new URL, not followed. The model then makes a fresh,
  separately-checked call, and the redirect target is visible in the transcript.
- **Address checks resolve DNS first.** Blocking `localhost` by string does nothing
  about `http://spoof.example/` resolving to `127.0.0.1`. Every resolved address is
  checked against the private, loopback, link-local and reserved ranges.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any, ClassVar
from urllib.parse import urlparse, urlunparse

import httpx
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

DESCRIPTION = """\
Fetch a URL and return its content as text.

HTML is converted to readable text with the navigation, ads and boilerplate removed. \
JSON and plain text come back as they are. Pass raw=True to get the unprocessed body \
instead, which you want for an API response you intend to parse.

Only http and https. Requests to localhost, private networks and link-local \
addresses are refused, and the address is checked after DNS resolution, so a public \
hostname pointing at an internal address is refused too.

Redirects within the same site are followed. A redirect to a different host is not \
-- you get told the new URL and can fetch it yourself if you want it. This keeps \
every host you actually reach visible in the transcript.

Content is truncated to `max_chars`. If a page matters more than that, fetch it and \
then narrow with a second, more specific request.

This tool reads pages. It cannot log in, submit forms, or run JavaScript, so a page \
that renders client-side will come back nearly empty.\
"""

DEFAULT_MAX_CHARS = 40_000
_TEXT_TYPES = ("text/", "application/json", "application/xml", "+json", "+xml", "javascript")


class WebFetchParams(BaseModel):
    """Parameters for `web_fetch`."""

    model_config = {"extra": "forbid"}

    url: str = Field(description="Absolute http(s) URL to fetch.")
    max_chars: int = Field(
        default=DEFAULT_MAX_CHARS, ge=100, le=500_000, description="Maximum characters to return."
    )
    raw: bool = Field(
        default=False,
        description="Return the body unprocessed instead of extracting readable text.",
    )


class BlockedURL(Exception):
    """A URL failed the address or scheme check."""


class WebFetchTool(Tool):
    """Fetch a URL and extract readable text."""

    name: ClassVar[str] = "web_fetch"
    description: ClassVar[str] = DESCRIPTION
    Params: ClassVar[type[BaseModel]] = WebFetchParams

    approval: ApprovalPolicy = "never"
    timeout: float = 30.0
    retry: RetryPolicy = NETWORK_RETRY
    concurrency_safe: ClassVar[bool] = True
    read_only: ClassVar[bool] = True

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        resolver: Any = None,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._resolve = resolver or _resolve_host
        super().__init__()

    async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
        """Scheme, shape and address checks, before a single byte goes out."""

        assert isinstance(params, WebFetchParams)
        try:
            _check_url(
                params.url, allow_private=ctx.config.allow_private_network, resolve=self._resolve
            )
        except BlockedURL as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION, message=str(exc))
        return None

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Fetch, following same-host redirects only."""

        assert isinstance(params, WebFetchParams)
        client = self._get_client(ctx)
        url = params.url
        hops: list[str] = []

        for _ in range(ctx.config.max_redirects + 1):
            try:
                _check_url(
                    url, allow_private=ctx.config.allow_private_network, resolve=self._resolve
                )
            except BlockedURL as exc:
                return ToolResult.failure(ToolErrorKind.PERMISSION, str(exc))

            try:
                response = await client.get(
                    url,
                    follow_redirects=False,
                    headers={"User-Agent": ctx.config.user_agent, "Accept": "*/*"},
                )
            except httpx.TimeoutException as exc:
                return ToolResult.failure(ToolErrorKind.TIMEOUT, f"fetching {url} timed out: {exc}")
            except httpx.HTTPError as exc:
                return ToolResult.failure(ToolErrorKind.NETWORK, f"could not fetch {url}: {exc}")

            if response.is_redirect:
                target = response.headers.get("location", "")
                if not target:
                    return ToolResult.failure(
                        ToolErrorKind.HTTP,
                        f"{url} returned {response.status_code} with no Location header",
                    )
                absolute = str(httpx.URL(url).join(target))
                if not _same_site(url, absolute):
                    return ToolResult.ok_text(
                        f"{url} redirects to a different host: {absolute}\n\n"
                        f"Cross-host redirects are not followed automatically. Call "
                        f"web_fetch again with that URL if you want it.",
                        display=ToolDisplay(kind="redirect", data={"from": url, "to": absolute}),
                        meta={"redirect": absolute, "status": response.status_code},
                    )
                hops.append(absolute)
                url = absolute
                continue

            return await self._render(params, response, url, hops, ctx)

        return ToolResult.failure(
            ToolErrorKind.HTTP,
            f"more than {ctx.config.max_redirects} redirects starting from {params.url}",
        )

    def _get_client(self, ctx: ToolContext) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=False)
        return self._client

    async def _render(
        self,
        params: WebFetchParams,
        response: httpx.Response,
        url: str,
        hops: list[str],
        ctx: ToolContext,
    ) -> ToolResult:
        if response.status_code >= 400:
            return ToolResult.failure(
                ToolErrorKind.HTTP,
                f"{url} returned HTTP {response.status_code} {response.reason_phrase}",
                meta={"status": response.status_code, "url": url},
            )

        body = response.content
        if len(body) > ctx.config.max_fetch_bytes:
            return ToolResult.failure(
                ToolErrorKind.UNSUPPORTED,
                f"{url} is {len(body)} bytes, over the {ctx.config.max_fetch_bytes}-byte limit",
            )

        content_type = response.headers.get("content-type", "").lower()
        if not any(t in content_type for t in _TEXT_TYPES) and content_type:
            return ToolResult.failure(
                ToolErrorKind.UNSUPPORTED,
                f"{url} is {content_type or 'an unknown type'}, which web_fetch cannot "
                f"read as text",
            )

        text = body.decode(response.encoding or "utf-8", errors="replace")
        extracted = text
        mode = "raw"
        if not params.raw and "html" in content_type:
            extracted = _extract_html(text) or text
            mode = "extracted"

        truncated = len(extracted) > params.max_chars
        shown = extracted[: params.max_chars]
        note = (
            f"\n\n[truncated at {params.max_chars} characters of {len(extracted)}]"
            if truncated
            else ""
        )
        header = f"{url} ({response.status_code}, {content_type or 'unknown type'})"
        if hops:
            header += f"\nfollowed {len(hops)} same-site redirect(s)"

        return ToolResult.ok_text(
            f"{header}\n\n{shown}{note}",
            display=ToolDisplay(
                kind="web",
                data={
                    "url": url,
                    "status": response.status_code,
                    "content_type": content_type,
                    "chars": len(extracted),
                    "truncated": truncated,
                    "mode": mode,
                },
            ),
            meta={
                "url": url,
                "status": response.status_code,
                "bytes": len(body),
                "redirects": hops,
                "mode": mode,
            },
        )

    async def aclose(self) -> None:
        """Close the HTTP client if this tool created it."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def _extract_html(html: str) -> str:
    """HTML to readable text via trafilatura, falling back to a tag strip.

    Imported lazily: trafilatura pulls in lxml and a good deal else, and importing
    the harness should not pay for a tool that may never be called.
    """

    try:
        import trafilatura
    except ImportError:  # pragma: no cover - dependency is declared
        return _strip_tags(html)
    try:
        out = trafilatura.extract(html, include_links=False, include_comments=False)
    except Exception:
        return _strip_tags(html)
    return out or _strip_tags(html)


def _strip_tags(html: str) -> str:
    """Last-resort tag strip, so a page always yields something."""

    import re

    without_scripts = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", text).strip()


def _resolve_host(host: str) -> list[str]:
    """Every address `host` resolves to. Raises `OSError` when it does not resolve."""

    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({str(info[4][0]) for info in infos})


def _check_url(url: str, *, allow_private: bool, resolve: Any) -> None:
    """Scheme and address checks. Raises `BlockedURL`."""

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise BlockedURL(
            f"only http and https URLs may be fetched; {parsed.scheme or 'that'} is refused"
        )
    host = parsed.hostname
    if not host:
        raise BlockedURL(f"{url!r} has no host")

    if allow_private:
        return

    addresses: list[str]
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        try:
            addresses = resolve(host)
        except OSError as exc:
            raise BlockedURL(f"could not resolve host {host!r}: {exc}") from exc

    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:  # pragma: no cover
            raise BlockedURL(f"host {host!r} resolved to an unusable address {raw!r}") from None
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise BlockedURL(
                f"{host!r} resolves to {address}, which is a private or local address. "
                f"web_fetch only reaches public hosts."
            )


def _same_site(current: str, target: str) -> bool:
    """Same host modulo a leading `www.`, and the scheme may only get stronger."""

    a, b = urlparse(current), urlparse(target)
    if b.scheme not in ("http", "https"):
        return False
    if a.scheme == "https" and b.scheme == "http":
        return False
    return _canonical_host(a.hostname) == _canonical_host(b.hostname)


def _canonical_host(host: str | None) -> str:
    if not host:
        return ""
    lowered = host.lower()
    return lowered[4:] if lowered.startswith("www.") else lowered


def normalise_url(url: str) -> str:
    """Drop the fragment; it never reaches the server anyway."""

    parsed = urlparse(url)
    return urlunparse(parsed._replace(fragment=""))


__all__ = ["DESCRIPTION", "BlockedURL", "WebFetchParams", "WebFetchTool", "normalise_url"]
