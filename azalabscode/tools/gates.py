"""Standalone `PermissionGate` implementations, so `tools` is usable before `control`.

The real gate is `control.gate.RuntimePermissionGate` at M2, which prompts a human
and owns the pending-approval set. These three satisfy the same protocol with no
policy at all, which is what makes the dispatcher testable at M1 and what a script
that does not want a controller can use.

`RecordingGate` is the third: it approves and keeps the checks, so a test can assert
that the dispatcher asked at all -- the failure mode where a tool runs without the
gate ever being consulted is invisible to a gate that only says yes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from azalabscode.ids import AgentId, CallId, NodeId
from azalabscode.permissions import ApprovalSummary, Decision, PermissionMode


@dataclass
class GateCheck:
    """One recorded call to `check`."""

    tool_name: str
    needs_approval: bool
    summary: ApprovalSummary
    params: dict[str, Any]
    agent_id: str
    call_id: str
    node_id: str | None = None


class AllowAllGate:
    """Approves everything. The `auto`-mode shape, with no run behind it."""

    def __init__(self, mode: PermissionMode = PermissionMode.AUTO) -> None:
        self._mode = mode

    @property
    def mode(self) -> PermissionMode:
        """The mode this gate reports."""

        return self._mode

    async def check(
        self,
        *,
        tool_name: str,
        needs_approval: bool,
        summary: ApprovalSummary,
        params: dict[str, Any],
        agent_id: AgentId,
        call_id: CallId,
        node_id: NodeId | None = None,
    ) -> Decision:
        """Approve."""

        return Decision.approve(by="allow_all")

    def visible_tool_names(self, agent_id: AgentId, requested: Sequence[str]) -> list[str]:
        """Everything requested."""

        return list(requested)


class DenyAllGate:
    """Denies everything. Useful as a default for an untrusted subagent."""

    def __init__(self, reason: str = "all tool calls are denied by policy") -> None:
        self.reason = reason

    @property
    def mode(self) -> PermissionMode:
        """Reports `manual`: something is refusing on a human's behalf."""

        return PermissionMode.MANUAL

    async def check(
        self,
        *,
        tool_name: str,
        needs_approval: bool,
        summary: ApprovalSummary,
        params: dict[str, Any],
        agent_id: AgentId,
        call_id: CallId,
        node_id: NodeId | None = None,
    ) -> Decision:
        """Deny."""

        return Decision.deny(self.reason, by="deny_all")

    def visible_tool_names(self, agent_id: AgentId, requested: Sequence[str]) -> list[str]:
        """Nothing."""

        return []


@dataclass
class RecordingGate:
    """Approves (or denies) and records every check, for tests.

    `decisions` maps a tool name to the decision to return; anything absent gets
    `default`. `checks` is the audit trail.
    """

    default: Decision = field(default_factory=lambda: Decision.approve(by="recording"))
    decisions: dict[str, Decision] = field(default_factory=dict)
    checks: list[GateCheck] = field(default_factory=list)
    hidden: set[str] = field(default_factory=set)
    permission_mode: PermissionMode = PermissionMode.AUTO

    @property
    def mode(self) -> PermissionMode:
        """The configured mode."""

        return self.permission_mode

    async def check(
        self,
        *,
        tool_name: str,
        needs_approval: bool,
        summary: ApprovalSummary,
        params: dict[str, Any],
        agent_id: AgentId,
        call_id: CallId,
        node_id: NodeId | None = None,
    ) -> Decision:
        """Record the check and return the configured decision."""

        self.checks.append(
            GateCheck(
                tool_name=tool_name,
                needs_approval=needs_approval,
                summary=summary,
                params=params,
                agent_id=str(agent_id),
                call_id=str(call_id),
                node_id=str(node_id) if node_id is not None else None,
            )
        )
        return self.decisions.get(tool_name, self.default)

    def visible_tool_names(self, agent_id: AgentId, requested: Sequence[str]) -> list[str]:
        """Everything requested, minus `hidden`."""

        return [n for n in requested if n not in self.hidden]


__all__ = ["AllowAllGate", "DenyAllGate", "GateCheck", "RecordingGate"]
