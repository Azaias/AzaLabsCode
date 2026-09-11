"""`RuntimePermissionGate`: what to do about a call the tool declared destructive.

Spec 4.5 splits the decision in two and this module owns the second half. The tool
says *whether* a call is destructive (`ApprovalPolicy`, R-T-3); the gate says whether
to run it, prompt for it, or refuse it. The TUI renders the prompt and neither of
the other two layers knows how.

The gate is injected into `ToolDispatcher` as a `contracts.PermissionGate`, which is
what keeps `tools` from importing `control` (spec delta 3). It is consulted for
*every* call, gated or not, so the dispatcher has a single code path -- a
`read_file` gets an immediate approval rather than a special case.

Two rules from the spec are enforced here and nowhere else:

* **R-C-5.** Switching to `auto` resolves every pending request as approved. The
  user's action expresses intent to stop being asked; the count is reported in the
  event so a UI can show what it just let through.
* **R-C-7.** Only `main` may raise an approval request. In `manual` mode a subagent
  sees a toolset filtered to `never`-policy tools, and the `check` path denies
  anyway as defence in depth -- `visible_tool_names` should have made it
  unreachable, but a mode switch between building the request and issuing the call
  makes "should" the wrong word.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

from azalabscode.contracts import ApprovalHandler
from azalabscode.events import ApprovalRequested, ApprovalResolved, EventEmitter
from azalabscode.ids import MAIN_AGENT, AgentId, CallId, NodeId, RunId
from azalabscode.permissions import (
    DEFAULT_MODE,
    ApprovalRequest,
    ApprovalSummary,
    Decision,
    PermissionMode,
)

SUBAGENT_DENIAL = (
    "subagents cannot use approval-gated tools in manual mode; "
    "report what you found and let the main agent make the change"
)

type ToolGatePredicate = Callable[[str], bool]
"""`tool name -> whether the tool can ever need approval`."""


def gated_names(tools: Any) -> frozenset[str]:
    """Names of tools whose `ApprovalPolicy` is not `never`.

    Takes anything iterable of tools with `name` and `approval` attributes, so the
    gate can be built from a `ToolSet` without this module importing one.
    """

    return frozenset(t.name for t in tools if getattr(t, "approval", "always") != "never")


def _carry_key(tool_name: str, params: dict[str, Any]) -> tuple[str, str]:
    """How a decision made after a reload is matched to the re-issued call.

    Not the call id: the model mints a new one every time it asks, so keying on it
    would carry nothing. `(tool, canonical params)` matches the *same* call and
    nothing else, so a model that changes its mind about the arguments is prompted
    again -- which is right, because the human approved the old ones.
    """

    return tool_name, json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def _predicate(spec: ToolGatePredicate | Collection[str]) -> ToolGatePredicate:
    if callable(spec):
        return spec
    names = frozenset(spec)
    return lambda name: name in names


@dataclass
class _Pending:
    """A request waiting on a human, and the future its resolution completes."""

    request: ApprovalRequest
    future: asyncio.Future[Decision]


class RuntimePermissionGate:
    """The `PermissionGate` a live run uses.

    `on_request` and `on_resolved` are the controller's hooks: they move the run into
    and out of `WAITING_APPROVAL` and take the approval safe point that makes R-C-9
    possible. They are optional so the gate is testable on its own.
    """

    def __init__(
        self,
        *,
        run_id: RunId | str = "",
        mode: PermissionMode = DEFAULT_MODE,
        handler: ApprovalHandler | None = None,
        gated_tools: ToolGatePredicate | Collection[str] = (),
        main_agent: AgentId | str = MAIN_AGENT,
        emitter: EventEmitter | None = None,
        on_request: Callable[[ApprovalRequest], Awaitable[None]] | None = None,
        on_resolved: Callable[[ApprovalRequest, Decision], Awaitable[None]] | None = None,
    ) -> None:
        self.run_id = str(run_id)
        self._mode = mode
        self.handler = handler
        self._gated = _predicate(gated_tools)
        self.main_agent = str(main_agent)
        self.emitter = emitter
        self.on_request = on_request
        self.on_resolved = on_resolved
        self._pending: dict[str, _Pending] = {}
        self._restored_ids: set[str] = set()
        self._carried: dict[tuple[str, str], Decision] = {}

    # -- mode ---------------------------------------------------------------

    @property
    def mode(self) -> PermissionMode:
        """The run's current permission mode."""

        return self._mode

    async def set_mode(self, mode: PermissionMode) -> int:
        """Switch mode, returning how many pending requests were auto-approved.

        Switching to `manual` deliberately does nothing to calls already approved
        and in flight (R-C-5): the approval was given, the effect may already have
        happened, and retracting it would mean lying about what ran.
        """

        if mode is self._mode:
            return 0
        self._mode = mode
        if mode is not PermissionMode.AUTO:
            return 0

        resolved = 0
        for request_id in list(self._pending):
            await self.resolve(
                request_id,
                Decision.approve(by="mode_switch", reason="permission mode switched to auto"),
            )
            resolved += 1
        return resolved

    # -- the protocol -------------------------------------------------------

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
        """Decide one call, blocking on a human if the mode says to.

        The agent is quiescent while this blocks -- the controller's `on_request`
        hook sets `waiting_approval` -- so a pause can still land, and a save can
        still be taken, while a request sits unanswered.
        """

        if not needs_approval:
            return Decision.approve(by="policy", reason="the tool declares this call safe")

        carried = self._carried.pop(_carry_key(tool_name, params), None)
        if carried is not None:
            # A decision a human made about a request restored from a checkpoint
            # (R-C-9). The call it answered was never executed -- it was blocked at
            # this gate -- so the model re-issued it, with a new call id. Honouring
            # the decision here is what makes "resolution proceeds normally" mean
            # something across a process death instead of prompting twice.
            return carried

        if self._mode is PermissionMode.AUTO:
            return Decision.approve(by="auto")
        if str(agent_id) != self.main_agent:
            return Decision.deny(SUBAGENT_DENIAL, by="policy")
        if self.handler is None:  # pragma: no cover - R-C-8 rejects this at start()
            return Decision.deny(
                "no approval handler is registered, so this call cannot be approved",
                by="policy",
            )

        request = ApprovalRequest(
            run_id=self.run_id,
            agent_id=str(agent_id),
            node_id=str(node_id) if node_id is not None else None,
            call_id=str(call_id),
            tool=tool_name,
            params=params,
            summary=summary,
        )
        future: asyncio.Future[Decision] = asyncio.get_running_loop().create_future()
        self._pending[request.request_id] = _Pending(request=request, future=future)

        if self.on_request is not None:
            await self.on_request(request)
        if self.emitter is not None:
            await self.emitter.emit(ApprovalRequested(request=request, agent_id=str(agent_id)))

        try:
            immediate = await self.handler.request(request)
            if immediate is not None and not future.done():
                await self.resolve(request.request_id, immediate)
            return await future
        except asyncio.CancelledError:
            # The step was interrupted while the request sat unanswered. Withdraw it
            # rather than leaving a modal on screen for a call that no longer exists.
            self._pending.pop(request.request_id, None)
            await self._withdraw(request, "the tool call was cancelled")
            raise

    def visible_tool_names(self, agent_id: AgentId, requested: Sequence[str]) -> list[str]:
        """Filter a toolset for one agent (R-C-7).

        In `manual` mode a subagent gets only the tools that never need approval, so
        the model never sees a tool it would be refused. In `auto` mode the
        restriction lifts, which is the only mode in which the intent's "parallel
        edit" subagent is possible (spec C-6).
        """

        if self._mode is PermissionMode.AUTO or str(agent_id) == self.main_agent:
            return list(requested)
        return [name for name in requested if not self._gated(name)]

    # -- resolution ---------------------------------------------------------

    @property
    def pending(self) -> list[ApprovalRequest]:
        """Requests waiting on a human, oldest first. Part of the session (R-C-9)."""

        return [p.request for p in self._pending.values()]

    def pending_request(self, request_id: str) -> ApprovalRequest | None:
        """One pending request by id."""

        entry = self._pending.get(request_id)
        return entry.request if entry is not None else None

    def restore_pending(self, requests: Sequence[ApprovalRequest]) -> int:
        """Re-register requests that came back from a checkpoint (R-C-9).

        The future nobody awaits is the point. In a live run the future is what the
        dispatcher is blocked on; here the dispatcher died with the process, so the
        future exists only so that `resolve()` has one code path. What carries the
        human's answer forward is `_carried`: the tool call the request was blocking
        never ran, so the model re-issues it, and the decision made now is applied to
        that re-issued call rather than prompting a second time.
        """

        loop = asyncio.get_running_loop()
        for request in requests:
            if request.request_id in self._pending:
                continue
            self._pending[request.request_id] = _Pending(
                request=request, future=loop.create_future()
            )
            self._restored_ids.add(request.request_id)
        return len(requests)

    @property
    def carried_decisions(self) -> int:
        """How many restored decisions are waiting for their call to be re-issued."""

        return len(self._carried)

    async def resolve(self, request_id: str, decision: Decision) -> bool:
        """Resolve a pending request. Returns False if there was nothing to resolve.

        Idempotent: resolving twice is what happens when a human answers a modal at
        the same moment the mode switches to `auto`, and it must not raise.
        """

        entry = self._pending.pop(request_id, None)
        if entry is None:
            return False
        if not entry.future.done():
            entry.future.set_result(decision)
        if request_id in self._restored_ids:
            # Nothing is awaiting this future: the call it guarded died with the last
            # process. Carry the decision so the re-issued call inherits it (R-C-9).
            self._restored_ids.discard(request_id)
            self._carried[_carry_key(entry.request.tool, entry.request.params)] = decision
        if self.emitter is not None:
            await self.emitter.emit(
                ApprovalResolved(
                    request_id=request_id,
                    decision=decision,
                    by=decision.by,
                    agent_id=entry.request.agent_id,
                )
            )
        if self.on_resolved is not None:
            await self.on_resolved(entry.request, decision)
        return True

    async def cancel_all(self, reason: str) -> int:
        """Withdraw every pending request. Used when a run is cancelled."""

        count = 0
        for request_id in list(self._pending):
            entry = self._pending.pop(request_id)
            if not entry.future.done():
                entry.future.set_result(Decision.deny(reason, by="run_cancelled"))
            await self._withdraw(entry.request, reason)
            count += 1
        return count

    async def _withdraw(self, request: ApprovalRequest, reason: str) -> None:
        if self.handler is None:
            return
        try:
            await self.handler.cancel(request.request_id, reason)
        except Exception:  # pragma: no cover - a handler that fails to withdraw
            return


__all__ = ["SUBAGENT_DENIAL", "RuntimePermissionGate", "ToolGatePredicate", "gated_names"]
