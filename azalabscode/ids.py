"""Identifier vocabulary: ULIDs, typed id aliases, and filesystem-safe slugs.

ULIDs rather than UUID4 because every id in this system ends up sorted at some
point -- messages in a transcript, events in a JSONL log, checkpoints in a
directory listing -- and a lexicographically sortable id makes that free. The
implementation is short enough that it does not earn a dependency.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from typing import NewType

RunId = NewType("RunId", str)
AgentId = NewType("AgentId", str)
NodeId = NewType("NodeId", str)
CallId = NewType("CallId", str)
StepId = NewType("StepId", str)
MessageId = NewType("MessageId", str)
RequestId = NewType("RequestId", str)

MAIN_AGENT: AgentId = AgentId("main")
"""The root agent of a run. Approval requests may come only from this agent (R-C-7)."""

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")

_lock = threading.Lock()
_last_ms = -1
_last_rand = 0


def _encode(value: int, length: int) -> str:
    out: list[str] = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def new_ulid(*, now_ms: int | None = None) -> str:
    """Return a fresh 26-character Crockford-base32 ULID.

    Monotonic within a millisecond: two ULIDs minted in the same millisecond sort
    in creation order, because the random component is incremented rather than
    redrawn. Ids minted in the same millisecond in *different* processes can still
    interleave; nothing in the harness relies on a cross-process ordering.
    """

    global _last_ms, _last_rand

    ms = int(time.time() * 1000) if now_ms is None else now_ms
    with _lock:
        if ms == _last_ms:
            _last_rand += 1
            if _last_rand >= (1 << 80):  # pragma: no cover - 2^80 ids in one ms
                ms += 1
                _last_ms = ms
                _last_rand = int.from_bytes(os.urandom(10), "big")
        else:
            _last_ms = ms
            _last_rand = int.from_bytes(os.urandom(10), "big")
        rand = _last_rand
    return _encode(ms, 10) + _encode(rand, 16)


def is_ulid(value: str) -> bool:
    """True when `value` is a well-formed ULID string."""

    return bool(_ULID_RE.match(value))


def ulid_time_ms(value: str) -> int:
    """Milliseconds since the epoch encoded in a ULID's timestamp prefix."""

    if not is_ulid(value):
        raise ValueError(f"not a ULID: {value!r}")
    out = 0
    for ch in value[:10]:
        out = (out << 5) | _CROCKFORD.index(ch)
    return out


def new_run_id() -> RunId:
    """Mint a run id."""

    return RunId(new_ulid())


def new_call_id() -> CallId:
    """Mint a call id, used when a provider omits one from a tool call."""

    return CallId(f"call_{new_ulid()}")


def new_message_id() -> MessageId:
    """Mint a message id."""

    return MessageId(new_ulid())


def new_request_id() -> RequestId:
    """Mint an approval-request id."""

    return RequestId(f"req_{new_ulid()}")


def new_step_id() -> StepId:
    """Mint a step id."""

    return StepId(f"step_{new_ulid()}")


# Windows refuses these as filenames regardless of extension, and refuses the
# listed characters anywhere in a path component. Agent and node ids contain "/"
# by construction, so every id that reaches a filename goes through the slugifier.
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_UNSAFE_CHARS = re.compile("[" + re.escape('/\\:*?"<>|') + "\\x00-\\x1f]")

MAX_SLUG_LENGTH = 64


def slugify_for_path(value: str, *, max_length: int = MAX_SLUG_LENGTH) -> str:
    """Turn an arbitrary id into a safe single path component.

    Handles the three ways a path component goes wrong on Windows: reserved device
    names, forbidden characters, and length. Over-long values are truncated and
    given an 8-character digest suffix so two long ids sharing a prefix do not
    collide.
    """

    if not value:
        return "_empty"

    slug = _UNSAFE_CHARS.sub("_", value).rstrip(" .")
    if not slug:
        slug = "_"

    if slug.split(".")[0].upper() in _RESERVED_NAMES:
        slug = f"_{slug}"

    if len(slug) > max_length:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[: max_length - 9]}-{digest}"

    return slug


__all__ = [
    "MAIN_AGENT",
    "MAX_SLUG_LENGTH",
    "AgentId",
    "CallId",
    "MessageId",
    "NodeId",
    "RequestId",
    "RunId",
    "StepId",
    "is_ulid",
    "new_call_id",
    "new_message_id",
    "new_request_id",
    "new_run_id",
    "new_step_id",
    "new_ulid",
    "slugify_for_path",
    "ulid_time_ms",
]
