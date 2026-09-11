"""Where a pending approval meets a human (R-C-8).

A `manual`-mode run with no handler would block on the first destructive call and
look like a hang, so `Controller.start()` refuses to start one (spec C-5). These are
the ways to satisfy it without a UI; `TUIApprovalHandler` arrives with the TUI at
M4.

A handler returns a `Decision` to resolve immediately, or `None` to resolve later
out of band through `Controller.resolve_approval`. The TUI returns `None` -- the
modal outlives the call -- and so does `QueueApprovalHandler`, which is what lets a
test drive the request and the resolution from opposite ends.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable

from azalabscode.permissions import ApprovalRequest, Decision


class DenyAllHandler:
    """Refuses everything, with a reason the model sees.

    The right default for an unattended run that must not be allowed to change
    anything: it fails visibly and immediately rather than hanging.
    """

    def __init__(self, reason: str = "running unattended; destructive calls are refused") -> None:
        self.reason = reason
        self.seen: list[ApprovalRequest] = []

    async def request(self, request: ApprovalRequest) -> Decision | None:
        """Deny."""

        self.seen.append(request)
        return Decision.deny(self.reason, by="deny_all")

    async def cancel(self, request_id: str, reason: str) -> None:
        """Nothing to withdraw."""

        return None


class CallbackApprovalHandler:
    """Answers with a function of the request. The simplest useful policy.

    Deliberately synchronous: a policy that has to do I/O to decide is an
    `ApprovalHandler` of its own, not a callback.
    """

    def __init__(self, decide: Callable[[ApprovalRequest], Decision]) -> None:
        self._decide = decide
        self.seen: list[ApprovalRequest] = []

    async def request(self, request: ApprovalRequest) -> Decision | None:
        """Apply the callback."""

        self.seen.append(request)
        return self._decide(request)

    async def cancel(self, request_id: str, reason: str) -> None:
        """Nothing to withdraw."""

        return None


class QueueApprovalHandler:
    """Publishes requests to a queue and defers the decision (tests, and any UI).

    `request()` returns `None`, so the gate parks on its future and the run reaches
    `WAITING_APPROVAL`. The test then takes the request off the queue and calls
    `Controller.resolve_approval`, which is exactly the path a modal takes.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[ApprovalRequest] = asyncio.Queue()
        self.seen: list[ApprovalRequest] = []
        self.withdrawn: list[tuple[str, str]] = []

    async def request(self, request: ApprovalRequest) -> Decision | None:
        """Enqueue and defer."""

        self.seen.append(request)
        self.queue.put_nowait(request)
        return None

    async def cancel(self, request_id: str, reason: str) -> None:
        """Record the withdrawal so a test can assert the modal was dismissed."""

        self.withdrawn.append((request_id, reason))

    async def next_request(self, *, timeout: float = 5.0) -> ApprovalRequest:  # noqa: ASYNC109 - the bound is the point; there is no ambient deadline
        """The next request, or `TimeoutError`.

        Bounded because there is no `pytest-timeout` here: an unbounded wait for a
        request that never arrives takes the whole suite down with no output.
        """

        async with asyncio.timeout(timeout):
            return await self.queue.get()


class StdinApprovalHandler:
    """Prompts on the terminal. The headless answer to R-C-8 (spec C-5).

    Reading stdin happens in a worker thread so the event loop keeps running -- the
    run is not paused while the prompt is up, other agents continue, and events keep
    flowing to any recorder that is attached.
    """

    def __init__(self, *, stream: object | None = None, default_deny: bool = True) -> None:
        self._stream = stream
        self.default_deny = default_deny

    async def request(self, request: ApprovalRequest) -> Decision | None:
        """Render the request and read a decision."""

        prompt = self.render(request)
        answer = await asyncio.to_thread(self._read, prompt)
        if answer in {"y", "yes"}:
            return Decision.approve(by="stdin")
        if answer in {"n", "no", ""} and self.default_deny:
            return Decision.deny("denied at the terminal", by="stdin")
        return Decision.deny(f"denied at the terminal ({answer!r})", by="stdin")

    async def cancel(self, request_id: str, reason: str) -> None:
        """The prompt is already gone; there is nothing to take back."""

        return None

    @staticmethod
    def render(request: ApprovalRequest) -> str:
        """The text of the prompt. Separated out so a test can read it."""

        lines = [
            "",
            f"[{request.agent_id}] {request.summary.title}",
        ]
        if request.summary.detail:
            lines.append(f"  {request.summary.detail}")
        if request.summary.diff:
            lines.append(request.summary.diff)
        lines.append("Allow? [y/N] ")
        return "\n".join(lines)

    def _read(self, prompt: str) -> str:
        stream = self._stream
        if stream is None:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            return sys.stdin.readline().strip().lower()
        line = stream.readline()  # type: ignore[attr-defined]
        return str(line).strip().lower()


__all__ = [
    "CallbackApprovalHandler",
    "DenyAllHandler",
    "QueueApprovalHandler",
    "StdinApprovalHandler",
]
