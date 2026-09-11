"""Exception hierarchy, plus the one serializable error model shared across layers.

Two kinds of thing live here and the split is deliberate:

- **Exceptions** are raised at the boundary where a caller can still do something
  about it -- bad config, an unimportable workflow, a checkpoint that will not
  serialize. They never cross a process boundary.
- **`ProviderError`** is *data*. It travels inside `StreamEvent.Error` and inside
  `ModelCallFailed` events, both of which are serialized. `providers` and `events`
  are siblings in the layer graph and neither may import the other, so the shared
  vocabulary sits below both.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from azalabscode.schema import HarnessModel


class HarnessError(Exception):
    """Base for every exception the harness raises deliberately."""


class ConfigurationError(HarnessError):
    """A run cannot start as configured.

    Raised before any spend: a `manual`-mode controller with no `ApprovalHandler`
    (R-C-8), a node whose declared `State` type will not round-trip through JSON
    (spec C-3 stage one), a provider with no API key.
    """


class SerializationError(HarnessError):
    """State that must be checkpointed cannot be serialized (R-W-5).

    Carries the node and field path so the fix is obvious. Raised at a safe point
    and deliberately *not* caught by the runner: the run transitions to FAILED
    rather than continuing to do work that can never be saved.
    """

    def __init__(self, node_id: str, field_path: str, reason: str) -> None:
        self.node_id = node_id
        self.field_path = field_path
        self.reason = reason
        super().__init__(f"{node_id}: cannot serialize {field_path}: {reason}")


class WorkflowNotImportable(HarnessError):
    """`WorkflowRef.import_path` did not resolve to a `build(config)` callable (R-C-11)."""

    def __init__(self, import_path: str, reason: str) -> None:
        self.import_path = import_path
        self.reason = reason
        super().__init__(f"cannot import workflow {import_path!r}: {reason}")


class GraphMismatchError(HarnessError):
    """A saved session names nodes the rebuilt graph does not have (spec C-2).

    Extra nodes or a changed `graph_hash` are a warning; *missing* nodes are fatal,
    because the session holds outputs with nowhere to go.
    """

    def __init__(self, missing: list[str], graph_hash: str, saved_hash: str) -> None:
        self.missing = missing
        self.graph_hash = graph_hash
        self.saved_hash = saved_hash
        super().__init__(
            f"saved session references {len(missing)} node(s) absent from the rebuilt "
            f"graph: {', '.join(sorted(missing)[:5])}"
            f"{' ...' if len(missing) > 5 else ''} "
            f"(saved graph_hash={saved_hash}, rebuilt={graph_hash})"
        )


class CheckpointError(HarnessError):
    """A checkpoint could not be written to disk."""


class SaveTimeout(HarnessError):
    """`save()` waited past its timeout for a safe point (spec delta 17).

    Names the step that was blocking, because "save timed out" alone tells the user
    nothing and this message is what the status bar renders.
    """

    def __init__(self, timeout_s: float, blocking: str) -> None:
        self.timeout_s = timeout_s
        self.blocking = blocking
        super().__init__(
            f"save did not reach a safe point within {timeout_s:g}s; blocked on {blocking}"
        )


class ProviderErrorKind(StrEnum):
    """Coarse classification of a provider failure, used to decide retry policy."""

    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    SERVER = "server"
    NETWORK = "network"
    MODEL = "model"
    """The request was well-formed but the model or route rejected it."""
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


RETRYABLE_PROVIDER_KINDS: frozenset[ProviderErrorKind] = frozenset(
    {
        ProviderErrorKind.RATE_LIMIT,
        ProviderErrorKind.SERVER,
        ProviderErrorKind.NETWORK,
    }
)
"""Kinds worth retrying -- and only before the first byte of the response (R-P-5)."""


class ProviderError(HarnessModel):
    """A provider failure as data, so it can ride an event or a session document."""

    kind: ProviderErrorKind = ProviderErrorKind.UNKNOWN
    message: str
    status_code: int | None = None
    retry_after: float | None = None
    """Seconds, parsed from a `Retry-After` header when the provider sends one."""
    provider: str | None = None
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def retryable(self) -> bool:
        """Whether a *pre-first-byte* retry is worth attempting."""

        return self.kind in RETRYABLE_PROVIDER_KINDS


class ProviderCallError(HarnessError):
    """Raised by the non-streaming `complete()` helper when a stream ends in error.

    The streaming path never raises: it yields a terminal `StreamError` event and
    lets the caller decide (R-P-5, spec C-8).
    """

    def __init__(self, error: ProviderError) -> None:
        self.error = error
        super().__init__(f"{error.kind}: {error.message}")


class ScriptExhausted(HarnessError):
    """`FakeProvider` was asked for a turn its script does not contain.

    Loud on purpose. A silently-diverging fake is worse than no fake at all: the
    kill test would pass while proving nothing.
    """


__all__ = [
    "RETRYABLE_PROVIDER_KINDS",
    "CheckpointError",
    "ConfigurationError",
    "GraphMismatchError",
    "HarnessError",
    "ProviderCallError",
    "ProviderError",
    "ProviderErrorKind",
    "SaveTimeout",
    "ScriptExhausted",
    "SerializationError",
    "WorkflowNotImportable",
]
