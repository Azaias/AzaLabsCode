"""The `Tool` base class: what a tool declares, and what it is *not* trusted with.

R-T-1 fixes the shape. The division of labour matters more than the shape: a tool
declares its parameters, whether it is destructive, whether two of it may run at
once, and how to do the work. Everything else -- schema validation, the permission
check, the timeout, retries, output caps, exception containment -- belongs to
`ToolDispatcher`, because a tool that enforced its own timeout would be a tool that
could forget to.

Every default fails closed. An unmodified subclass is `approval="always"`, not
concurrency-safe, not read-only, and retries zero times.

Three spec deltas shape the interface:

- **Delta 7.** Concurrency safety is per *call*: a `shell` running `ls` is safe and
  one running `rm` is not. `is_concurrency_safe(params)` defaults to the class
  attribute and fails closed if it raises.
- **Delta 11.** `validate_params` is a pure pre-check separate from `run`, so a
  `manual`-mode run does not ask a human to approve an edit that was always going
  to fail.
- **Delta 12.** One run, three renderings: `content` for the model, `display` for
  widgets, `meta` for telemetry.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ValidationError

from azalabscode.errors import ConfigurationError
from azalabscode.permissions import ApprovalPolicy, ApprovalSummary, evaluate_policy
from azalabscode.toolio import (
    DEFAULT_MAX_RESULT_CHARS,
    NO_RETRY,
    RetryPolicy,
    ToolError,
    ToolErrorKind,
    ToolResult,
    ToolSchema,
)
from azalabscode.tools.context import ToolContext


class NoParams(BaseModel):
    """The parameter model for a tool that takes nothing."""

    model_config = {"extra": "forbid"}


class Tool:
    """Base class for every tool, built-in or workflow-supplied.

    Subclasses set the class attributes and override `run`. A subclass that sets
    `retry.attempts > 0` while being approval-gated raises `ConfigurationError` at
    construction (R-T-4): the check is at build time, not call time, because a
    misconfigured non-idempotent retry is a bug to find before the run costs money.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    Params: ClassVar[type[BaseModel]] = NoParams

    approval: ApprovalPolicy = "always"
    """Whether this call is destructive. The tool's call; what to do about it is the
    gate's (R-T-3, spec 4.5)."""

    timeout: float = 30.0
    retry: RetryPolicy = NO_RETRY
    max_result_size_chars: int | float = DEFAULT_MAX_RESULT_CHARS
    """May be lower than the system ceiling, or `math.inf` to opt out (delta 8)."""

    concurrency_safe: ClassVar[bool] = False
    """Class-level default behind `is_concurrency_safe`. False fails closed."""

    read_only: ClassVar[bool] = False
    hard_block_retry: ClassVar[bool] = False
    """`shell` sets this: a partially-applied command is not something a retry undoes."""

    def __init__(self) -> None:
        self._check_declaration()

    # -- declaration checks -------------------------------------------------

    def _check_declaration(self) -> None:
        """Enforce R-T-4 and the minimum a subclass must declare."""

        if not self.name:
            raise ConfigurationError(f"{type(self).__name__} does not set a tool name")
        if not self.description:
            raise ConfigurationError(f"tool {self.name!r} has no description (R-T-9)")

        if self.retry.attempts > 0:
            if self.hard_block_retry:
                raise ConfigurationError(
                    f"tool {self.name!r} hard-blocks retries: a partially-applied "
                    f"effect cannot be undone by running the call again"
                )
            if self.approval != "never" and not self.retry.unsafe_allow_retry:
                raise ConfigurationError(
                    f"tool {self.name!r} is approval-gated and therefore not idempotent, "
                    f"but declares retry.attempts={self.retry.attempts}; set "
                    f"retry.unsafe_allow_retry=True to override (R-T-4)"
                )

        if isinstance(self.max_result_size_chars, int | float) and (
            self.max_result_size_chars <= 0
        ):
            raise ConfigurationError(f"tool {self.name!r} has a non-positive max_result_size_chars")

    # -- schema -------------------------------------------------------------

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """The JSON Schema for this tool's parameters.

        Rendered from the Pydantic model, which is the single source of truth
        (R-T-1). `$defs` are left in place; providers accept them and inlining
        them would lose enum names.
        """

        schema = cls.Params.model_json_schema(mode="validation")
        schema.pop("title", None)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema

    @classmethod
    def schema(cls) -> ToolSchema:
        """The tool as the model sees it. Snapshot-tested (R-T-9)."""

        return ToolSchema(
            name=cls.name,
            description=cls.description,
            parameters=cls.json_schema(),
        )

    def parse_params(self, raw: dict[str, Any] | BaseModel) -> BaseModel:
        """Validate raw model-supplied arguments into the `Params` model.

        Raises `pydantic.ValidationError`; the dispatcher converts it into
        `ToolError(kind="invalid_params")`. `Params` forbids extra keys, so a model
        inventing an argument is told about it rather than having it dropped.
        """

        if isinstance(raw, self.Params):
            return raw
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        return self.Params.model_validate(raw)

    # -- per-call predicates ------------------------------------------------

    def is_read_only(self, params: BaseModel) -> bool:
        """Whether this call only observes. Defaults to the class attribute."""

        return self.read_only

    def is_concurrency_safe(self, params: BaseModel) -> bool:
        """Whether this call may run alongside others (delta 7).

        Defaults to the class attribute. The dispatcher calls this through
        `concurrency_safe_for`, which fails closed on an exception -- a tool whose
        own safety assessment crashed does not get to run in parallel.
        """

        return self.concurrency_safe

    def concurrency_safe_for(self, params: BaseModel | None) -> bool:
        """`is_concurrency_safe` with the fail-closed wrapper the dispatcher uses."""

        if params is None:
            return False
        try:
            return bool(self.is_concurrency_safe(params))
        except Exception:
            return False

    def needs_approval(self, params: BaseModel) -> bool:
        """Resolve this tool's `ApprovalPolicy` against concrete params (R-T-3).

        `evaluate_policy` fails closed: a predicate that raises means yes.
        """

        return evaluate_policy(self.approval, params)

    def timeout_for(self, params: BaseModel) -> float:
        """The dispatcher's timeout for this call.

        Overridden by `shell`, whose per-call `timeout` parameter is the real bound
        and whose class `timeout` is only the ceiling.
        """

        return self.timeout

    def result_cap_for(self, params: BaseModel) -> int | float:
        """The per-call output cap. `math.inf` opts out (delta 8)."""

        return self.max_result_size_chars

    # -- the two halves of doing the work -----------------------------------

    async def validate_params(self, params: BaseModel, ctx: ToolContext) -> ToolError | None:
        """A pure pre-check, run *before* the approval prompt (delta 11).

        "File not read yet", "string not unique", "path outside the workspace" all
        belong here. Returning a `ToolError` aborts the call without prompting.
        Must not mutate anything: it may run on a call that is then denied.
        """

        return None

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Do the work. Return a `ToolResult`, never raise for an anticipated failure.

        Anything that does escape is caught by the dispatcher and converted to
        `ToolError(kind="internal")` with the traceback in `meta` (R-T-2). That is a
        safety net, not a supported path.
        """

        raise NotImplementedError

    async def on_cancel(self, params: BaseModel, ctx: ToolContext, reason: str) -> None:
        """Clean up after a timeout or a cancellation. Default: nothing to do.

        Run by the dispatcher *outside* the cancelled task, so it can still await.
        `shell` overrides it to kill the process tree -- doing that inside a `finally`
        in `run` would have to await while the task is being cancelled, which is
        exactly where cleanup gets skipped.
        """

        return None

    def approval_summary(self, params: BaseModel, ctx: ToolContext) -> ApprovalSummary:
        """What a human needs to see to decide (R-C-6).

        The default is the tool name and a compact parameter rendering. File tools
        override it to attach a unified diff, which is the difference between an
        informed approval and a leap of faith.
        """

        detail = ", ".join(
            f"{k}={_short(v)}" for k, v in params.model_dump(exclude_none=True).items()
        )
        return ApprovalSummary(
            title=self.name,
            detail=detail,
            danger=not self.is_read_only(params),
        )

    # -- helpers for subclasses --------------------------------------------

    @classmethod
    def failure(cls, kind: ToolErrorKind, message: str, **kwargs: Any) -> ToolResult:
        """Shorthand for a failed result; the message also reaches the model."""

        return ToolResult.failure(kind, message, **kwargs)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


def _short(value: Any, limit: int = 80) -> str:
    """Render a parameter value for a one-line summary."""

    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text)} chars)"


def describe_validation_error(exc: ValidationError, tool_name: str) -> str:
    """Turn a Pydantic error into something a model can act on.

    Pydantic's default rendering includes a docs URL and an input echo that can be
    the whole file the model tried to write. This keeps the location and the reason
    and drops the rest.
    """

    lines: list[str] = []
    for err in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"  {loc}: {err['msg']}")
    body = "\n".join(lines) if lines else "  (no detail)"
    return f"invalid parameters for {tool_name}:\n{body}"


def is_unbounded_cap(cap: int | float) -> bool:
    """True when a cap means "no cap"."""

    return cap == math.inf


async def maybe_await(value: Any) -> Any:
    """Await `value` if it is awaitable. Lets a hook be sync or async."""

    if inspect.isawaitable(value):
        return await value
    return value


class ToolSet:
    """An ordered, name-keyed collection of tool instances.

    Thin on purpose: `ToolRegistry` in `registry.py` is the thing with the building
    logic. This is what a dispatcher holds.
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> Self:
        """Register a tool, replacing any tool of the same name."""

        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool | None:
        """A tool by name, or `None`."""

        return self._tools.get(name)

    def names(self) -> list[str]:
        """Every registered tool name, in registration order."""

        return list(self._tools)

    def schemas(self, names: list[str] | None = None) -> list[ToolSchema]:
        """Schemas for `names`, or for everything, skipping names that do not exist."""

        wanted = names if names is not None else self.names()
        return [self._tools[n].schema() for n in wanted if n in self._tools]

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Any:
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools


__all__ = [
    "NoParams",
    "Tool",
    "ToolSet",
    "describe_validation_error",
    "is_unbounded_cap",
    "maybe_await",
]
