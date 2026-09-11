"""`Func`: an async Python function as a graph node (R-W-3).

The escape hatch, and the node type a workflow author writes most often. Everything
between two model calls -- reformatting, filtering, picking a winner, building a
prompt -- is a `Func`, and each one is a checkpoint boundary for free.

Two conveniences, both decided rather than discovered:

* **The function may take `(input)` or `(ctx, input)`.** Arity is inspected once, at
  construction, not per call. A function that wants a safe point mid-way or a
  subagent needs the context; most do not, and making every one of them accept an
  argument it ignores is noise in the workflow file.
* **A synchronous function is accepted.** R-W-3 says "async Python function" and that
  is the contract, but a one-line `lambda`-shaped transform that blocks for a
  microsecond is not worth an `async def`, and refusing it pushes authors towards
  `asyncio.get_event_loop().run_until_complete`, which is worse. Anything genuinely
  blocking still belongs in a thread and the docstring says so.

The return value must be JSON-native: it goes into the session as this node's
memoized output (R-W-6).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, ClassVar

from azalabscode.workflows.node import Node

FuncBody = Callable[..., Any]


class Func(Node):
    """Run a Python function on the node's input."""

    kind: ClassVar[str] = "Func"

    def __init__(
        self,
        fn: FuncBody,
        *,
        output_type: str = "any",
        takes_ctx: bool | None = None,
    ) -> None:
        self.fn = fn
        self.output_type = output_type
        self.takes_ctx = _takes_ctx(fn) if takes_ctx is None else takes_ctx

    async def run(self, ctx: Any, input: Any) -> Any:
        """Call the function. A blocking body belongs in `asyncio.to_thread`."""

        result = self.fn(ctx, input) if self.takes_ctx else self.fn(input)
        if inspect.isawaitable(result):
            return await result
        return result

    def describe(self) -> str:
        name = getattr(self.fn, "__name__", type(self.fn).__name__)
        return f"Func({name})"

    def hash_fields(self) -> tuple[str, str, str]:
        """Includes the function's name: swapping the body under a stable node id is
        exactly the drift `graph_hash` exists to make visible (spec C-2)."""

        base = super().hash_fields()
        return (f"Func<{getattr(self.fn, '__name__', '?')}>", base[1], base[2])


def _takes_ctx(fn: FuncBody) -> bool:
    """Whether `fn` wants the context as its first argument.

    Counts positional parameters, ignoring `self` on a bound method (already bound)
    and anything keyword-only. A `*args` function is treated as taking only the
    input: guessing that it wants a context would silently change what it receives.
    """

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # a builtin or a C callable
        return False
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


__all__ = ["Func"]
