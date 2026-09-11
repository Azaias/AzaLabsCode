"""Permission vocabulary: modes, policies, approval requests and decisions.

This is vocabulary, not policy (spec delta 2). `ApprovalPolicy` states whether a
tool call is destructive -- the tool's call to make (R-T-3). What to *do* about a
destructive call is `control.gate.RuntimePermissionGate`'s call, and how to render
the prompt is the TUI's. Those live in their own layers; only the shared types are
here, so that `events` can carry an `ApprovalRequest` without importing upward.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from azalabscode.ids import new_request_id
from azalabscode.schema import HarnessModel, VersionedModel


class PermissionMode(StrEnum):
    """How the run treats tool calls that declare themselves destructive.

    A property of the *run*, inherited by every agent in the tree, switchable at any
    time (R-C-5). There are exactly two; finer-grained control is out of scope for
    v1 by explicit non-goal.
    """

    MANUAL = "manual"
    """Destructive calls surface an `ApprovalRequest` and block until resolved."""

    AUTO = "auto"
    """Destructive calls run without prompting."""


type ApprovalPolicyLiteral = Literal["never", "always"]
type ApprovalPredicate = Callable[[Any], bool]
type ApprovalPolicy = ApprovalPolicyLiteral | ApprovalPredicate
"""`"never"`, `"always"`, or a `(params) -> bool` predicate evaluated per call.

A predicate lets one tool be destructive only sometimes -- a `shell` that reads
versus one that writes -- without splitting it into two tools. It is a callable on
the tool class, so it is never serialized.
"""


def evaluate_policy(policy: ApprovalPolicy, params: Any) -> bool:
    """Resolve an `ApprovalPolicy` against concrete params.

    Fails closed: a predicate that raises is treated as "needs approval". A tool
    whose own risk assessment crashed is not a tool to run unattended.
    """

    if policy == "never":
        return False
    if policy == "always":
        return True
    if callable(policy):
        try:
            return bool(policy(params))
        except Exception:
            return True
    raise ValueError(f"not an approval policy: {policy!r}")


class DecisionKind(StrEnum):
    """The two ways an approval request can be resolved."""

    APPROVE = "approve"
    DENY = "deny"


class Decision(HarnessModel):
    """The outcome of a permission check.

    Returned by the gate for *every* call, approval-gated or not: in `auto` mode and
    for `never`-policy tools it is an immediate approval, so the dispatcher has one
    code path rather than two.
    """

    kind: DecisionKind
    reason: str | None = None
    """Required in spirit for a denial: it is returned to the model verbatim."""
    by: str | None = None
    """Who decided -- a user id, `"auto"`, `"mode_switch"`, `"policy"`."""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def approved(self) -> bool:
        """True when the call may proceed."""

        return self.kind is DecisionKind.APPROVE

    @classmethod
    def approve(cls, *, by: str | None = None, reason: str | None = None) -> Decision:
        """An approval."""

        return cls(kind=DecisionKind.APPROVE, by=by, reason=reason)

    @classmethod
    def deny(cls, reason: str, *, by: str | None = None) -> Decision:
        """A denial. `reason` reaches the model as a denied-kind tool error."""

        return cls(kind=DecisionKind.DENY, reason=reason, by=by)


class ApprovalSummary(HarnessModel):
    """What a human needs to see to decide, built by the tool (`approval_summary`).

    Kept structured rather than pre-rendered so the TUI, a stdin prompt and a log
    line can each present it in their own idiom. The `diff` field is what makes an
    `edit_file` approval reviewable rather than a leap of faith (R-C-6).
    """

    title: str
    """One line: the action, e.g. `edit_file src/app.py`."""
    detail: str = ""
    """Free text: the command, the target, the size."""
    diff: str | None = None
    """Unified diff for `write_file` and `edit_file`."""
    danger: bool = False
    """True for the irreversible subset, so a UI can style it differently."""


class ApprovalRequest(VersionedModel):
    """A pending request for a human decision.

    Part of the session document, so a save/kill/load cycle restores the run in
    WAITING_APPROVAL with the same request and resolution proceeds normally
    (R-C-9). `params` is the already-validated parameter model dumped to JSON --
    the request must survive a process restart, so it cannot hold a live object.
    """

    request_id: str = Field(default_factory=new_request_id)
    run_id: str
    agent_id: str
    node_id: str | None = None
    call_id: str
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    summary: ApprovalSummary
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


DEFAULT_MODE = PermissionMode.MANUAL
"""Default for a new run. The safe direction: a mistaken prompt costs a keystroke,
a mistaken execution costs a file."""


__all__ = [
    "DEFAULT_MODE",
    "ApprovalPolicy",
    "ApprovalPolicyLiteral",
    "ApprovalPredicate",
    "ApprovalRequest",
    "ApprovalSummary",
    "Decision",
    "DecisionKind",
    "PermissionMode",
    "evaluate_policy",
]
