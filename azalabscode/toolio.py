"""Tool input/output vocabulary, shared by `tools`, `workflows`, `control` and `tui`.

The types a tool *produces* live here rather than in `tools/` so that a
`ToolResultMessage` can carry a `ToolResult` without `messages` importing the tool
layer (spec delta 1). `tools/` keeps the `Tool` base class and the dispatcher --
the policy -- and imports this vocabulary downward like everyone else.
"""

from __future__ import annotations

import math
import random
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from azalabscode.content import Part, TextPart, text_of
from azalabscode.schema import HarnessModel, VersionedModel

DEFAULT_MAX_RESULT_CHARS = 50_000
"""System ceiling on a single tool result (R-T-6).

A tool may set a *lower* cap, or `math.inf` to opt out entirely -- `read_file` does,
because spilling a file read to disk that the model then has to re-read is circular
(spec delta 8).
"""

DEFAULT_TURN_RESULT_BUDGET = 200_000
"""Ceiling on the combined tool results folded into one model request.

Eight parallel results each just under the per-tool cap would otherwise blow up the
next request. Applied at the top of each loop iteration and memoized by call id so
the decision is byte-stable across a resume.
"""


class ToolErrorKind(StrEnum):
    """Why a tool call did not succeed.

    The model sees the kind, so these read as things a model can act on. `internal`
    is the one it cannot: it means the harness has a bug, and the traceback goes to
    `ToolResult.meta` for the human.
    """

    INVALID_PARAMS = "invalid_params"
    """Schema validation or `validate_params` rejected the call before it ran."""
    NOT_FOUND = "not_found"
    PERMISSION = "permission"
    """Outside `workspace_root`, or a blocked URL or address range (R-T-7)."""
    DENIED = "denied"
    """A human said no in `manual` mode; `message` carries their reason (R-C-6)."""
    UNAVAILABLE = "unavailable"
    """The tool is not in this agent's toolset -- e.g. a subagent in `manual` mode."""
    EXIT_STATUS = "exit_status"
    """A subprocess exited non-zero. Output is still attached."""
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    """Cancelled in-process by an interrupt. The effect, if any, is known."""
    INTERRUPTED = "interrupted"
    """The process died mid-call. The effect is *not* known, and it is never re-run."""
    NETWORK = "network"
    HTTP = "http"
    BUDGET = "budget"
    """Elided to keep the turn under the per-turn result budget (spec delta 8)."""
    UNSUPPORTED = "unsupported"
    INTERNAL = "internal"


RETRYABLE_TOOL_KINDS: frozenset[ToolErrorKind] = frozenset(
    {ToolErrorKind.NETWORK, ToolErrorKind.HTTP, ToolErrorKind.TIMEOUT}
)


class ToolError(HarnessModel):
    """A tool failure as data. Returned, never raised (R-T-2)."""

    kind: ToolErrorKind
    message: str
    retryable: bool | None = None
    """`None` means "derive from the kind". Set explicitly to override."""
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_retryable(self) -> bool:
        """Whether the dispatcher may retry, before per-tool policy is consulted."""

        if self.retryable is not None:
            return self.retryable
        return self.kind in RETRYABLE_TOOL_KINDS


class ToolDisplay(HarnessModel):
    """The widget-facing rendering of a result (spec delta 12).

    One tool run, three consumers: `ToolResult.content` for the model, this for the
    UI, `ToolResult.meta` for telemetry. Structured so a widget never has to
    re-parse text that was written for a model. `kind` names the renderer
    (diff, file, matches, shell) and `data` is its payload.
    """

    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


class ToolResult(VersionedModel):
    """The single return type of every tool call.

    Always returned. Any exception a tool did not anticipate is caught by the
    dispatcher and converted into `ok=False` with an internal-kind error and the
    traceback in `meta` (R-T-2).
    """

    ok: bool = True
    content: list[Part] = Field(default_factory=list)
    display: ToolDisplay | None = None
    error: ToolError | None = None
    duration_ms: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_error_agrees_with_ok(self) -> ToolResult:
        if self.ok and self.error is not None:
            raise ValueError("ToolResult.ok is True but an error is attached")
        if not self.ok and self.error is None:
            raise ValueError("ToolResult.ok is False but no error is attached")
        return self

    @property
    def text(self) -> str:
        """The model-facing text of this result."""

        return text_of(self.content)

    @classmethod
    def ok_text(cls, text: str, **kwargs: Any) -> ToolResult:
        """A successful text result."""

        return cls(ok=True, content=[TextPart(text=text)], **kwargs)

    @classmethod
    def failure(
        cls,
        kind: ToolErrorKind,
        message: str,
        *,
        content: list[Part] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """A failed result.

        The message is also placed in `content` when the caller supplies none,
        because the model only ever reads `content` -- an error the model cannot see
        is an error it cannot recover from.
        """

        parts: list[Part] = content if content is not None else [TextPart(text=message)]
        return cls(
            ok=False,
            content=parts,
            error=ToolError(kind=kind, message=message),
            **kwargs,
        )


class ToolSchema(HarnessModel):
    """A tool as the model sees it: name, description, JSON-Schema parameters.

    Rendered from the tool's Pydantic `Params` model, which is the single source of
    truth for the schema (R-T-1). Snapshot-tested so an accidental description edit
    shows up in a diff (R-T-9).
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @field_validator("parameters")
    @classmethod
    def _must_be_an_object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("tool parameter schema must be a JSON-Schema object")
        return value


class RetryPolicy(HarnessModel):
    """Per-tool retry configuration.

    Defaults to no retries. `unsafe_allow_retry` is the explicit acknowledgement
    R-T-4 demands before a non-idempotent tool may retry at all; `shell` blocks even
    that, since a partially-applied command is not something a retry can undo.
    """

    attempts: int = 0
    """Retries *after* the first try. 0 means the tool runs exactly once."""
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 8.0
    jitter: float = 0.25
    """Fraction of the delay drawn at random, to decorrelate concurrent retries."""
    retry_on: frozenset[ToolErrorKind] = RETRYABLE_TOOL_KINDS
    unsafe_allow_retry: bool = False

    @field_validator("attempts")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("attempts must be >= 0")
        return value

    def delay_for(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Backoff before retry number `attempt` (1-based), honouring `Retry-After`."""

        if retry_after is not None:
            return min(retry_after, self.max_backoff_s)
        base = min(self.initial_backoff_s * (2 ** (attempt - 1)), self.max_backoff_s)
        return base * (1.0 - self.jitter * random.random())

    def should_retry(self, error: ToolError, attempt: int) -> bool:
        """Whether a failure of this kind gets another try."""

        return attempt <= self.attempts and error.kind in self.retry_on


NO_RETRY = RetryPolicy()
"""The default for anything that changes the world."""

NETWORK_RETRY = RetryPolicy(attempts=2)
"""Two retries on network and HTTP failures; used by `web_fetch` and `web_search`."""


def is_unbounded(cap: int | float) -> bool:
    """True when a result-size cap means "no cap"."""

    return cap == math.inf


__all__ = [
    "DEFAULT_MAX_RESULT_CHARS",
    "DEFAULT_TURN_RESULT_BUDGET",
    "NETWORK_RETRY",
    "NO_RETRY",
    "RETRYABLE_TOOL_KINDS",
    "RetryPolicy",
    "ToolDisplay",
    "ToolError",
    "ToolErrorKind",
    "ToolResult",
    "ToolSchema",
    "is_unbounded",
]
