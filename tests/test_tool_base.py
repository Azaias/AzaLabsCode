"""The `Tool` base class: declaration checks, defaults, and `ToolSet`.

The theme is that **every default fails closed**. An unmodified subclass needs
approval, is not concurrency-safe, is not read-only, and retries zero times. A tool
author who forgets something gets the cautious behaviour, not the dangerous one.
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest
from pydantic import BaseModel, Field, ValidationError

from azalabscode.errors import ConfigurationError
from azalabscode.permissions import ApprovalPolicy
from azalabscode.toolio import DEFAULT_MAX_RESULT_CHARS, RetryPolicy, ToolResult
from azalabscode.tools import NoParams, Tool, ToolContext, ToolSet, describe_validation_error


class Params(BaseModel):
    model_config = {"extra": "forbid"}

    value: int = Field(default=1, ge=0, description="A number.")


class Minimal(Tool):
    """The smallest legal tool."""

    name: ClassVar[str] = "minimal"
    description: ClassVar[str] = "Does the minimum."
    Params: ClassVar[type[BaseModel]] = Params

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok_text("done")


# ---------------------------------------------------------------------------
# Defaults fail closed
# ---------------------------------------------------------------------------


def test_an_unmodified_tool_needs_approval() -> None:
    assert Minimal().approval == "always"
    assert Minimal().needs_approval(Params()) is True


def test_an_unmodified_tool_is_not_concurrency_safe() -> None:
    assert Minimal().concurrency_safe is False
    assert Minimal().is_concurrency_safe(Params()) is False


def test_an_unmodified_tool_is_not_read_only() -> None:
    assert Minimal().read_only is False
    assert Minimal().is_read_only(Params()) is False


def test_an_unmodified_tool_does_not_retry() -> None:
    assert Minimal().retry.attempts == 0
    assert Minimal().retry.unsafe_allow_retry is False


def test_an_unmodified_tool_uses_the_system_result_ceiling() -> None:
    assert Minimal().max_result_size_chars == DEFAULT_MAX_RESULT_CHARS


def test_an_unmodified_tool_validates_nothing_and_cleans_up_nothing(
    tool_ctx: ToolContext,
) -> None:
    """The base implementations must be usable, not abstract: a tool with nothing to
    pre-check should not have to write an empty override."""

    import asyncio

    tool = Minimal()
    assert asyncio.run(tool.validate_params(Params(), tool_ctx)) is None
    assert asyncio.run(tool.on_cancel(Params(), tool_ctx, "timeout")) is None


def test_the_base_run_is_not_implemented() -> None:
    import asyncio

    class Bare(Minimal):
        pass

    with pytest.raises(NotImplementedError):
        asyncio.run(Tool.run(Bare(), Params(), None))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Declaration checks
# ---------------------------------------------------------------------------


def test_a_tool_without_a_name_is_rejected() -> None:
    class Nameless(Minimal):
        name: ClassVar[str] = ""

    with pytest.raises(ConfigurationError, match="does not set a tool name"):
        Nameless()


def test_a_tool_without_a_description_is_rejected() -> None:
    """R-T-9 makes the description a reviewed artefact; an empty one is a bug."""

    class Silent(Minimal):
        description: ClassVar[str] = ""

    with pytest.raises(ConfigurationError, match="no description"):
        Silent()


def test_a_non_positive_result_cap_is_rejected() -> None:
    class Zero(Minimal):
        max_result_size_chars: int | float = 0

    with pytest.raises(ConfigurationError, match="non-positive"):
        Zero()


def test_an_infinite_cap_is_legal() -> None:
    """`read_file` opts out entirely (spec delta 8)."""

    class Unbounded(Minimal):
        max_result_size_chars: int | float = math.inf

    assert Unbounded().max_result_size_chars == math.inf


def test_a_never_gated_tool_may_retry() -> None:
    class Fetcher(Minimal):
        approval: ApprovalPolicy = "never"
        retry: RetryPolicy = RetryPolicy(attempts=2)

    assert Fetcher().retry.attempts == 2


# ---------------------------------------------------------------------------
# Approval policy (R-T-3)
# ---------------------------------------------------------------------------


def test_a_predicate_policy_is_evaluated_per_call() -> None:
    """One tool destructive only sometimes, rather than two tools."""

    class Conditional(Minimal):
        approval: ApprovalPolicy = staticmethod(lambda p: p.value > 10)

    tool = Conditional()
    assert tool.needs_approval(Params(value=1)) is False
    assert tool.needs_approval(Params(value=99)) is True


def test_a_predicate_that_raises_means_yes() -> None:
    """A tool whose own risk assessment crashed is not one to run unattended."""

    def boom(params: BaseModel) -> bool:
        raise RuntimeError("cannot decide")

    class Cranky(Minimal):
        approval: ApprovalPolicy = staticmethod(boom)

    assert Cranky().needs_approval(Params()) is True


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_the_schema_comes_from_the_params_model() -> None:
    """R-T-1: one source of truth, so the schema cannot drift from validation."""

    schema = Minimal.schema()
    assert schema.name == "minimal"
    assert schema.parameters["properties"]["value"]["description"] == "A number."


def test_the_model_title_is_stripped_from_the_schema() -> None:
    """`"title": "Params"` is a Pydantic artefact; it is noise in a tool schema."""

    assert "title" not in Minimal.json_schema()


def test_a_tool_with_no_parameters_still_renders_an_object_schema() -> None:
    class Empty(Minimal):
        Params: ClassVar[type[BaseModel]] = NoParams

    schema = Empty.schema()
    assert schema.parameters["type"] == "object"
    assert schema.parameters["properties"] == {}


def test_parse_params_accepts_a_dict_and_an_instance() -> None:
    tool = Minimal()
    assert tool.parse_params({"value": 5}).value == 5  # type: ignore[attr-defined]
    existing = Params(value=7)
    assert tool.parse_params(existing) is existing


def test_parse_params_rejects_an_unknown_key() -> None:
    with pytest.raises(ValidationError):
        Minimal().parse_params({"value": 1, "surprise": 2})


def test_the_validation_message_is_compact_and_names_the_field() -> None:
    """Pydantic's default rendering echoes the input, which for a `write_file` is the
    whole file the model tried to write."""

    try:
        Minimal().parse_params({"value": -1})
    except ValidationError as exc:
        message = describe_validation_error(exc, "minimal")
    else:  # pragma: no cover
        pytest.fail("expected a ValidationError")

    assert "minimal" in message
    assert "value" in message
    assert "https://" not in message
    assert len(message) < 300


# ---------------------------------------------------------------------------
# Approval summary
# ---------------------------------------------------------------------------


def test_the_default_summary_renders_the_parameters(tool_ctx: ToolContext) -> None:
    summary = Minimal().approval_summary(Params(value=3), tool_ctx)
    assert summary.title == "minimal"
    assert "value=3" in summary.detail
    assert summary.danger is True


def test_a_long_parameter_is_shortened_in_the_summary(tool_ctx: ToolContext) -> None:
    """A modal is not the place for a 40 000-character value."""

    class Wordy(BaseModel):
        model_config = {"extra": "forbid"}

        text: str = ""

    class WordyTool(Minimal):
        Params: ClassVar[type[BaseModel]] = Wordy

    summary = WordyTool().approval_summary(Wordy(text="x" * 5000), tool_ctx)
    assert "5000 chars" in summary.detail
    assert len(summary.detail) < 300


def test_a_read_only_tool_is_not_marked_dangerous(tool_ctx: ToolContext) -> None:
    class Reader(Minimal):
        read_only: ClassVar[bool] = True

    assert Reader().approval_summary(Params(), tool_ctx).danger is False


# ---------------------------------------------------------------------------
# ToolSet
# ---------------------------------------------------------------------------


def test_a_toolset_keeps_registration_order() -> None:
    class A(Minimal):
        name: ClassVar[str] = "a"

    class B(Minimal):
        name: ClassVar[str] = "b"

    assert ToolSet([B(), A()]).names() == ["b", "a"]


def test_registering_the_same_name_replaces_it() -> None:
    class A(Minimal):
        name: ClassVar[str] = "a"

    first, second = A(), A()
    toolset = ToolSet([first])
    toolset.add(second)

    assert len(toolset) == 1
    assert toolset.get("a") is second


def test_an_unknown_name_returns_none() -> None:
    assert ToolSet([Minimal()]).get("nope") is None
    assert "nope" not in ToolSet([Minimal()])


def test_schemas_skip_names_that_do_not_exist() -> None:
    """The gate's filtered list can name a tool this set does not have; that is a
    filter, not an error."""

    toolset = ToolSet([Minimal()])
    assert [s.name for s in toolset.schemas(["minimal", "ghost"])] == ["minimal"]


def test_schemas_defaults_to_everything() -> None:
    assert [s.name for s in ToolSet([Minimal()]).schemas()] == ["minimal"]


def test_a_toolset_is_iterable() -> None:
    tool = Minimal()
    assert list(ToolSet([tool])) == [tool]


def test_the_repr_names_the_tool() -> None:
    assert "minimal" in repr(Minimal())
