"""`TUIApprovalHandler`: the seam between the permission gate and a modal.

It defers every decision. `request()` returns `None`, which parks the gate on its
future and lets the run reach `WAITING_APPROVAL` while the modal is up; the answer
comes back later through `Controller.resolve_approval`. That is the whole reason
`ApprovalHandler.request` is allowed to return `None` (R-U-6): a handler that had
to answer inside the call would have to block the event loop the modal needs to
draw on.

The handler does not push the screen. `HarnessApp` does that from the
`ApprovalRequested` *event*, so a subclass that overrides `on_approval_requested`
replaces the rendering without replacing the handler, and a headless run of the
same workflow swaps the handler without touching the app.

Withdrawal is the other direction and it has no event behind it: when the gate
resolves a request without the user -- a switch to `auto` mode, a cancelled run --
it calls `cancel()` and nothing else says so. `on_withdraw` is how that reaches the
app, and without it the modal stays up asking a question nobody is listening to.
"""

from __future__ import annotations

from collections.abc import Callable

from azalabscode.permissions import ApprovalRequest, Decision

type WithdrawCallback = Callable[[str, str], None]
"""`(request_id, reason) -> None`. Called from the gate's task; must not block."""


class TUIApprovalHandler:
    """Defers every request to the UI and never decides anything itself."""

    def __init__(self, on_withdraw: WithdrawCallback | None = None) -> None:
        self.on_withdraw = on_withdraw
        self.pending: dict[str, ApprovalRequest] = {}
        self.seen: list[ApprovalRequest] = []
        self.withdrawn: list[tuple[str, str]] = []

    async def request(self, request: ApprovalRequest) -> Decision | None:
        """Record the request and defer. Always returns `None` (R-U-6)."""

        self.pending[request.request_id] = request
        self.seen.append(request)
        return None

    async def cancel(self, request_id: str, reason: str) -> None:
        """Withdraw a request the gate no longer needs an answer for."""

        self.pending.pop(request_id, None)
        self.withdrawn.append((request_id, reason))
        if self.on_withdraw is not None:
            self.on_withdraw(request_id, reason)

    def resolved(self, request_id: str) -> None:
        """Forget a request the app has answered."""

        self.pending.pop(request_id, None)


__all__ = ["TUIApprovalHandler", "WithdrawCallback"]
