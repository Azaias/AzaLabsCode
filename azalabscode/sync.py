"""A thin synchronous entry point for scripts (R-X-1).

The core is async throughout. This exists so a one-off script does not have to
write `asyncio.run` boilerplate, and so the Windows event-loop policy is pinned in
exactly one place.

That pinning matters more than it looks: `WindowsSelectorEventLoopPolicy` makes
`asyncio.create_subprocess_shell` fail with `NotImplementedError`, which takes the
`shell` tool out entirely. The Proactor policy is the 3.12 default on Windows, but
libraries do reset it, so `run_sync` asserts rather than assumes, and a conftest
assertion pins it for the test suite.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any


def assert_subprocess_capable_loop_policy() -> None:
    """Raise if the current Windows event-loop policy cannot spawn subprocesses.

    Project-wide rule (spec delta 22): `WindowsSelectorEventLoopPolicy` is
    forbidden. It breaks `create_subprocess_shell` silently enough that the failure
    reads as a tool bug rather than a policy bug.
    """

    if sys.platform != "win32":
        return
    policy = asyncio.get_event_loop_policy()
    selector = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector is not None and isinstance(policy, selector):
        raise RuntimeError(
            "WindowsSelectorEventLoopPolicy is active; asyncio subprocesses do not "
            "work under it and the shell tool would fail with NotImplementedError. "
            "Use the default WindowsProactorEventLoopPolicy."
        )


def run_sync[T](coro: Coroutine[Any, Any, T], *, debug: bool | None = None) -> T:
    """Run a coroutine to completion from synchronous code.

    Refuses to run inside an existing event loop rather than nesting one: a nested
    loop deadlocks the Textual app that is almost certainly the caller's real
    context, and the error message is more useful than the hang.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        coro.close()
        raise RuntimeError(
            "run_sync() was called from inside a running event loop; await the "
            "coroutine directly instead"
        )

    assert_subprocess_capable_loop_policy()
    return asyncio.run(coro, debug=debug)


__all__ = ["assert_subprocess_capable_loop_policy", "run_sync"]
