"""The leaf tier: ids, slugs, the transcript invariant, permissions, cancellation.

These modules are vocabulary, so most of them are cheap to test and cheap to get
subtly wrong. The two that carry real weight are `slugify_for_path` (every id that
reaches a Windows filename goes through it) and `assert_transcript_valid` (the
invariant the whole control layer is built to preserve).
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from azalabscode.cancellation import (
    RECOVERABLE_REASONS,
    CancelReason,
    StepKind,
    StepOutcome,
    is_recoverable,
)
from azalabscode.content import (
    TextPart,
    ToolCallPart,
    parse_tool_arguments,
    text_of,
    tool_calls_of,
)
from azalabscode.errors import (
    ConfigurationError,
    GraphMismatchError,
    HarnessError,
    SaveTimeout,
    SerializationError,
)
from azalabscode.ids import (
    MAX_SLUG_LENGTH,
    is_ulid,
    new_ulid,
    slugify_for_path,
    ulid_time_ms,
)
from azalabscode.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    TranscriptError,
    Usage,
    UserMessage,
    assert_transcript_valid,
    open_call_ids,
    usage_total,
)
from azalabscode.permissions import (
    ApprovalRequest,
    ApprovalSummary,
    Decision,
    PermissionMode,
    evaluate_policy,
)
from azalabscode.runstate import (
    LEGAL_TRANSITIONS,
    QUIESCENT_PHASES,
    AgentPhase,
    RunState,
    is_legal_transition,
    is_quiescent,
)
from azalabscode.schema import json_safe
from azalabscode.toolio import (
    DEFAULT_MAX_RESULT_CHARS,
    RetryPolicy,
    ToolError,
    ToolErrorKind,
    ToolResult,
    ToolSchema,
    is_unbounded,
)

# ---------------------------------------------------------------------------
# ids
# ---------------------------------------------------------------------------


def test_ulids_are_well_formed_and_unique() -> None:
    ids = [new_ulid() for _ in range(2000)]
    assert all(is_ulid(u) for u in ids)
    assert len(set(ids)) == len(ids)


def test_ulids_minted_in_the_same_millisecond_sort_in_creation_order() -> None:
    """Monotonicity within a millisecond is what makes sorted ids useful at all."""

    ids = [new_ulid(now_ms=1_757_000_000_000) for _ in range(500)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_ulids_sort_by_time_across_milliseconds() -> None:
    early = new_ulid(now_ms=1_000_000_000_000)
    late = new_ulid(now_ms=2_000_000_000_000)
    assert early < late
    assert ulid_time_ms(early) == 1_000_000_000_000
    assert ulid_time_ms(late) == 2_000_000_000_000


def test_ulid_time_rejects_a_non_ulid() -> None:
    assert not is_ulid("not-a-ulid")
    assert not is_ulid("I" * 26), "Crockford base32 excludes I, L, O and U"
    with pytest.raises(ValueError, match="not a ULID"):
        ulid_time_ms("nope")


# ---------------------------------------------------------------------------
# slugify_for_path -- every id that reaches a filename goes through this
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("main", "main"),
        ("main/0", "main_0"),
        ("root/agent/2", "root_agent_2"),
        ("a:b*c?d", "a_b_c_d"),
        ('quote"lt<gt>pipe|', "quote_lt_gt_pipe_"),
        ("back\\slash", "back_slash"),
        ("", "_empty"),
    ],
)
def test_slugify_handles_the_characters_windows_refuses(value: str, expected: str) -> None:
    assert slugify_for_path(value) == expected


@pytest.mark.parametrize("name", ["CON", "con", "PRN", "aux", "NUL", "COM1", "LPT9"])
def test_slugify_escapes_reserved_device_names(name: str) -> None:
    """Windows refuses these as filenames whatever the extension."""

    slug = slugify_for_path(name)
    assert slug.startswith("_")
    assert slug.split(".")[0].upper() not in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT9"}


def test_slugify_escapes_a_reserved_name_with_an_extension() -> None:
    assert slugify_for_path("nul.json") == "_nul.json"


def test_slugify_strips_trailing_dots_and_spaces() -> None:
    """A trailing dot or space is silently stripped by Windows, which is worse than
    being rejected: the path you asked for is not the path you got."""

    assert slugify_for_path("name. ") == "name"
    assert slugify_for_path("name...") == "name"


def test_slugify_truncates_with_a_digest_so_long_ids_do_not_collide() -> None:
    a = "x" * 200 + "-alpha"
    b = "x" * 200 + "-beta"
    assert len(slugify_for_path(a)) <= MAX_SLUG_LENGTH
    assert slugify_for_path(a) != slugify_for_path(b)


@given(st.text(max_size=300))
@settings(max_examples=300)
def test_slugify_always_produces_a_usable_path_component(value: str) -> None:
    slug = slugify_for_path(value)
    assert slug
    assert len(slug) <= MAX_SLUG_LENGTH
    assert not re.search(r'[/\\:*?"<>|]', slug)
    assert not slug.endswith((" ", "."))
    assert slug.split(".")[0].upper() not in {"CON", "PRN", "AUX", "NUL"}


# ---------------------------------------------------------------------------
# content
# ---------------------------------------------------------------------------


def test_tool_argument_parsing_never_raises() -> None:
    assert parse_tool_arguments('{"a": 1}') == ({"a": 1}, None)
    assert parse_tool_arguments("") == ({}, None), "several models emit '' for no args"
    assert parse_tool_arguments("   ") == ({}, None)

    value, error = parse_tool_arguments('{"a": ')
    assert value is None and error is not None

    value, error = parse_tool_arguments("[1, 2]")
    assert value is None
    assert error is not None
    assert "object" in error


def test_text_of_ignores_reasoning_and_tool_calls() -> None:
    from azalabscode.content import ReasoningPart

    parts = [
        ReasoningPart(text="thinking"),
        TextPart(text="the "),
        ToolCallPart(call_id="c", name="t"),
        TextPart(text="answer"),
    ]
    assert text_of(parts) == "the answer"
    assert [c.call_id for c in tool_calls_of(parts)] == ["c"]


def test_tool_call_part_ok_reflects_parse_state() -> None:
    good = ToolCallPart(call_id="c", name="t", arguments={}, raw_arguments="{}")
    bad = ToolCallPart(call_id="c", name="t", raw_arguments="{", parse_error="boom")
    assert good.ok is True
    assert bad.ok is False


# ---------------------------------------------------------------------------
# The transcript invariant
# ---------------------------------------------------------------------------


def assistant_with(*call_ids: str) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolCallPart(call_id=cid, name="t", arguments={}, raw_arguments="{}")
            for cid in call_ids
        ],
        model="x/y",
    )


def result_for(call_id: str) -> ToolResultMessage:
    return ToolResultMessage(call_id=call_id, name="t", result=ToolResult.ok_text("ok"))


def test_a_valid_transcript_passes() -> None:
    assert_transcript_valid(
        [
            SystemMessage(content="sys"),
            UserMessage.of("go"),
            assistant_with("a", "b"),
            result_for("a"),
            result_for("b"),
            AssistantMessage(content=[TextPart(text="done")], model="x/y"),
        ]
    )


def test_results_must_appear_in_call_order_not_completion_order() -> None:
    """The invariant is defined in call order. This is the case a naive
    append-as-they-complete loop gets wrong every time two tools run concurrently."""

    with pytest.raises(TranscriptError, match="out of call order"):
        assert_transcript_valid([assistant_with("a", "b"), result_for("b"), result_for("a")])


def test_a_result_for_an_unknown_call_is_rejected() -> None:
    with pytest.raises(TranscriptError, match="unknown call"):
        assert_transcript_valid([result_for("ghost")])


def test_a_duplicated_result_is_rejected() -> None:
    with pytest.raises(TranscriptError, match="duplicate result"):
        assert_transcript_valid([assistant_with("a"), result_for("a"), result_for("a")])


def test_a_duplicated_call_id_is_rejected() -> None:
    with pytest.raises(TranscriptError, match="duplicate tool call id"):
        assert_transcript_valid([assistant_with("a"), result_for("a"), assistant_with("a")])


def test_a_turn_in_progress_is_not_an_error() -> None:
    """Trailing unanswered calls mean a turn is running, not that state is corrupt.

    It becomes an error only where the next model request is built, which is why the
    loop calls `open_call_ids` there instead.
    """

    assert_transcript_valid([assistant_with("a", "b"), result_for("a")])
    assert open_call_ids([assistant_with("a", "b"), result_for("a")]) == ["b"]


def test_open_call_ids_preserves_call_order() -> None:
    messages = [assistant_with("a", "b", "c"), result_for("a"), result_for("b")]
    assert open_call_ids(messages) == ["c"]
    assert open_call_ids([assistant_with("x", "y", "z")]) == ["x", "y", "z"]


def test_usage_total_sums_assistant_messages_only() -> None:
    messages = [
        UserMessage.of("go"),
        AssistantMessage(content=[], model="x", usage=Usage(prompt_tokens=5, cost_usd=0.1)),
        AssistantMessage(content=[], model="x", usage=Usage(completion_tokens=7)),
        AssistantMessage(content=[], model="x", usage=None),
    ]
    total = usage_total(messages)
    assert total.prompt_tokens == 5
    assert total.completion_tokens == 7
    assert total.cost_usd == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# permissions
# ---------------------------------------------------------------------------


def test_approval_policy_evaluation() -> None:
    assert evaluate_policy("never", None) is False
    assert evaluate_policy("always", None) is True
    assert evaluate_policy(lambda p: p["danger"], {"danger": True}) is True
    assert evaluate_policy(lambda p: p["danger"], {"danger": False}) is False


def test_a_policy_predicate_that_raises_fails_closed() -> None:
    """A tool whose own risk assessment crashed is not one to run unattended."""

    def boom(params: object) -> bool:
        raise RuntimeError("bad predicate")

    assert evaluate_policy(boom, {}) is True


def test_an_invalid_policy_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="not an approval policy"):
        evaluate_policy("sometimes", {})  # type: ignore[arg-type]


def test_decision_constructors() -> None:
    approve = Decision.approve(by="user")
    deny = Decision.deny("wrong file", by="user")
    assert approve.approved is True
    assert deny.approved is False
    assert deny.reason == "wrong file"


def test_approval_request_mints_its_own_id() -> None:
    request = ApprovalRequest(
        run_id="r",
        agent_id="main",
        call_id="c",
        tool="shell",
        summary=ApprovalSummary(title="shell rm -rf", danger=True),
    )
    assert request.request_id.startswith("req_")


def test_permission_modes_are_exactly_two() -> None:
    """Finer-grained control is an explicit v1 non-goal."""

    assert {m.value for m in PermissionMode} == {"manual", "auto"}


# ---------------------------------------------------------------------------
# cancellation and run state
# ---------------------------------------------------------------------------


def test_a_cancellation_with_no_reason_is_never_recoverable() -> None:
    """Absence of a reason is the authoritative signal that it was not ours."""

    assert is_recoverable(None) is False


@pytest.mark.parametrize("reason", list(CancelReason))
def test_recoverability_matches_the_frozen_set(reason: CancelReason) -> None:
    assert is_recoverable(reason) is (reason in RECOVERABLE_REASONS)


def test_shutdown_and_run_cancellation_are_not_recoverable() -> None:
    assert not is_recoverable(CancelReason.SHUTDOWN)
    assert not is_recoverable(CancelReason.RUN_CANCELLED)
    assert not is_recoverable(CancelReason.PARENT_FAILED)


def test_step_kinds_and_outcomes_are_stable_strings() -> None:
    assert StepKind.TOOL_CALL == "tool_call"
    assert StepOutcome.INTERRUPTED == "interrupted"


def test_every_run_state_has_a_transition_entry() -> None:
    assert set(LEGAL_TRANSITIONS) == set(RunState)


def test_terminal_states_have_no_outgoing_transitions() -> None:
    for state in (RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED):
        assert LEGAL_TRANSITIONS[state] == frozenset()


def test_self_transitions_are_always_legal() -> None:
    """`Controller` methods are idempotent, so `pause()` on a paused run is a no-op."""

    for state in RunState:
        assert is_legal_transition(state, state)


def test_the_spec_transitions_are_permitted() -> None:
    assert is_legal_transition(RunState.CREATED, RunState.RUNNING)
    assert is_legal_transition(RunState.RUNNING, RunState.PAUSING)
    assert is_legal_transition(RunState.PAUSING, RunState.PAUSED)
    assert is_legal_transition(RunState.PAUSED, RunState.RUNNING)
    assert is_legal_transition(RunState.WAITING_APPROVAL, RunState.PAUSED)
    assert is_legal_transition(RunState.INTERRUPTING, RunState.PAUSED)
    assert not is_legal_transition(RunState.CREATED, RunState.PAUSED)
    assert not is_legal_transition(RunState.COMPLETED, RunState.RUNNING)


def test_blocked_on_child_is_quiescent() -> None:
    """Spec delta 14. If it were not, a subagent parked at the gate would deadlock a
    parent that is never going to return."""

    assert is_quiescent(AgentPhase.BLOCKED_ON_CHILD)
    assert AgentPhase.BLOCKED_ON_CHILD in QUIESCENT_PHASES


def test_running_and_blocked_io_are_not_quiescent() -> None:
    assert not is_quiescent(AgentPhase.RUNNING)
    assert not is_quiescent(AgentPhase.BLOCKED_IO)


# ---------------------------------------------------------------------------
# toolio
# ---------------------------------------------------------------------------


def test_tool_result_failure_puts_the_message_where_the_model_can_see_it() -> None:
    """An error the model cannot read is an error it cannot recover from."""

    result = ToolResult.failure(ToolErrorKind.NOT_FOUND, "no such file: a.py")
    assert result.ok is False
    assert result.text == "no such file: a.py"
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.NOT_FOUND


def test_retryability_defaults_by_kind_and_can_be_overridden() -> None:
    assert ToolError(kind=ToolErrorKind.NETWORK, message="x").is_retryable is True
    assert ToolError(kind=ToolErrorKind.DENIED, message="x").is_retryable is False
    assert ToolError(kind=ToolErrorKind.DENIED, message="x", retryable=True).is_retryable is True


def test_retry_policy_defaults_to_no_retries() -> None:
    """R-T-4: anything that changes the world runs exactly once by default."""

    policy = RetryPolicy()
    assert policy.attempts == 0
    assert policy.unsafe_allow_retry is False
    assert not policy.should_retry(ToolError(kind=ToolErrorKind.NETWORK, message="x"), 1)


def test_retry_policy_respects_kind_and_attempt_count() -> None:
    policy = RetryPolicy(attempts=2)
    network = ToolError(kind=ToolErrorKind.NETWORK, message="x")
    denied = ToolError(kind=ToolErrorKind.DENIED, message="x")
    assert policy.should_retry(network, 1) is True
    assert policy.should_retry(network, 2) is True
    assert policy.should_retry(network, 3) is False
    assert policy.should_retry(denied, 1) is False


def test_retry_backoff_is_bounded_jittered_and_honours_retry_after() -> None:
    policy = RetryPolicy(attempts=5, initial_backoff_s=1.0, max_backoff_s=4.0)
    delays = [policy.delay_for(n) for n in range(1, 6)]
    assert all(0 < d <= 4.0 for d in delays)
    assert policy.delay_for(1, retry_after=2.0) == 2.0
    assert policy.delay_for(1, retry_after=99.0) == 4.0, "capped"


def test_negative_retry_attempts_are_rejected() -> None:
    with pytest.raises(ValueError, match="attempts must be"):
        RetryPolicy(attempts=-1)


def test_tool_schema_requires_an_object_schema() -> None:
    with pytest.raises(ValueError, match="JSON-Schema object"):
        ToolSchema(name="t", description="d", parameters={"type": "string"})


def test_result_caps() -> None:
    import math

    assert DEFAULT_MAX_RESULT_CHARS == 50_000
    assert is_unbounded(math.inf) is True
    assert is_unbounded(DEFAULT_MAX_RESULT_CHARS) is False


# ---------------------------------------------------------------------------
# errors and schema
# ---------------------------------------------------------------------------


def test_every_harness_error_shares_a_base() -> None:
    for cls in (
        ConfigurationError,
        SerializationError,
        GraphMismatchError,
        SaveTimeout,
    ):
        assert issubclass(cls, HarnessError)


def test_serialization_error_names_the_node_and_field() -> None:
    error = SerializationError("root/agent", "state.handle", "not JSON serializable")
    assert "root/agent" in str(error)
    assert "state.handle" in str(error)
    assert error.node_id == "root/agent"


def test_save_timeout_names_the_blocking_step() -> None:
    """ "save timed out" alone tells the user nothing, and this is what the status
    bar renders."""

    error = SaveTimeout(120.0, "shell(npm run build) running for 300s")
    assert "120" in str(error)
    assert "npm run build" in str(error)


def test_graph_mismatch_lists_the_missing_nodes() -> None:
    error = GraphMismatchError(["a", "b"], "new", "old")
    assert "2 node(s)" in str(error)
    assert "old" in str(error) and "new" in str(error)


def test_json_safe_detects_unserializable_values() -> None:
    assert json_safe({"a": [1, "x", None, True]}) is True
    assert json_safe({"a": object()}) is False
    assert json_safe({(1, 2): "tuple key"}) is False
