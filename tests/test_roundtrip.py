"""R-X-4: every core model round-trips through JSON losslessly.

`model_validate_json(model_dump_json())` must produce an equal object. This is the
property the whole save/kill/load story rests on, so it is tested exhaustively --
every message type, every content part, every tool-error kind, every event class --
rather than on a representative sample.

Two things make this a real test rather than a tautology:

- `extra="forbid"` on `HarnessModel` means a field that fails to serialize surfaces
  as a validation error on the way back in, not as a silently dropped key.
- Hypothesis fills the leaf models with values a hand-written case would not think
  of: empty strings, surrogate-adjacent text, huge ints, NaN-free floats.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from azalabscode.content import (
    FilePart,
    ImagePart,
    ReasoningPart,
    TextPart,
    ToolCallPart,
)
from azalabscode.contracts import DelegateOutcome, SafePoint, SafePointKind
from azalabscode.errors import ProviderError, ProviderErrorKind
from azalabscode.events import (
    EVENT_ADAPTER,
    AgentFinished,
    AgentPhaseChanged,
    AgentSpawned,
    ApprovalRequested,
    ApprovalResolved,
    Checkpoint,
    Event,
    EventsDropped,
    GraphDriftWarning,
    MessageInjected,
    ModelCallCancelled,
    ModelCallCompleted,
    ModelCallFailed,
    ModelCallStarted,
    ModelDelta,
    NodeCompleted,
    NodeFailed,
    NodeStarted,
    PermissionModeChanged,
    RunStateChanged,
    RunWarning,
    ToolCallCancelled,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallProgress,
    ToolCallRequested,
    ToolCallStarted,
)
from azalabscode.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from azalabscode.permissions import (
    ApprovalRequest,
    ApprovalSummary,
    Decision,
    PermissionMode,
)
from azalabscode.providers.base import (
    Finish,
    ModelInfo,
    ModelPricing,
    ModelRequest,
    ReasoningConfig,
    ReasoningDelta,
    StreamError,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageReport,
)
from azalabscode.providers.testing import Script, ScriptedToolCall, ScriptedTurn
from azalabscode.runstate import AgentPhase, RunState
from azalabscode.toolio import (
    RetryPolicy,
    ToolDisplay,
    ToolError,
    ToolErrorKind,
    ToolResult,
    ToolSchema,
)

TS = datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)


def roundtrip(model):
    """Serialize and re-parse, asserting equality. Returns the reparsed object."""

    again = type(model).model_validate_json(model.model_dump_json())
    assert again == model, f"{type(model).__name__} did not round-trip"
    return again


# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------


def all_parts() -> list:
    """One instance of every content part, including the reserved `FilePart`."""

    return [
        TextPart(text="hello\nworld\té\U0001f600"),
        TextPart(text=""),
        ReasoningPart(text="thinking", signature="sig-1"),
        ReasoningPart(text="", signature=None, redacted=True),
        ToolCallPart(
            call_id="call_1",
            name="read_file",
            arguments={"path": "a.py", "offset": 1},
            raw_arguments='{"path": "a.py", "offset": 1}',
        ),
        ToolCallPart(
            call_id="call_2",
            name="write_file",
            arguments=None,
            raw_arguments='{"path": ',
            parse_error="invalid JSON: Expecting value",
        ),
        ImagePart(media_type="image/png", data_b64="aGVsbG8=", detail="high"),
        FilePart(media_type="application/pdf", filename="spec.pdf", uri="file:///spec.pdf"),
    ]


@pytest.mark.parametrize("part", all_parts(), ids=lambda p: f"{p.type}-{id(p) % 1000}")
def test_content_parts_roundtrip(part) -> None:
    roundtrip(part)


# ---------------------------------------------------------------------------
# Messages -- the requirement names these specifically
# ---------------------------------------------------------------------------


def sample_tool_result() -> ToolResult:
    return ToolResult(
        ok=True,
        content=[
            TextPart(text="   1│import os"),
            ImagePart(media_type="image/png", data_b64="eA=="),
        ],
        display=ToolDisplay(kind="file", data={"path": "a.py", "lines": 1}),
        duration_ms=12.5,
        meta={"bytes": 9, "encoding": "utf-8"},
    )


def all_messages() -> list:
    """One instance of every message type, exercising every optional field."""

    return [
        SystemMessage(content="You are a careful engineer."),
        UserMessage(content=[TextPart(text="hi")]),
        UserMessage(
            content=[
                TextPart(text="look at this"),
                ImagePart(media_type="image/jpeg", data_b64="/9j/"),
            ],
            injected=True,
        ),
        AssistantMessage(
            content=[TextPart(text="Reading.")],
            model="anthropic/claude-sonnet-4",
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=3,
                cached_tokens=2,
                reasoning_tokens=1,
                cost_usd=0.0001,
            ),
            finish_reason="stop",
        ),
        AssistantMessage(
            content=[
                ReasoningPart(text="plan", signature="s"),
                TextPart(text="partial answ"),
            ],
            model="x/y",
            usage=None,
            finish_reason=None,
            cancelled=True,
        ),
        AssistantMessage(
            content=[
                ToolCallPart(
                    call_id="call_9",
                    name="grep",
                    arguments={"pattern": "TODO"},
                    raw_arguments='{"pattern":"TODO"}',
                )
            ],
            model="x/y",
        ),
        ToolResultMessage(call_id="call_9", name="grep", result=sample_tool_result()),
        ToolResultMessage(
            call_id="call_10",
            name="shell",
            result=ToolResult.failure(ToolErrorKind.EXIT_STATUS, "exit 1"),
        ),
    ]


@pytest.mark.parametrize("message", all_messages(), ids=lambda m: m.role)
def test_every_message_type_roundtrips(message) -> None:
    """R-X-4 on every message type."""

    roundtrip(message)


def test_message_union_roundtrips_through_the_discriminator() -> None:
    """A heterogeneous transcript survives a list-level round-trip.

    The per-message test above uses the concrete class, which cannot catch a broken
    `role` discriminator. This one goes through the union, which is how a session
    actually stores a transcript.
    """

    from pydantic import TypeAdapter

    from azalabscode.messages import Message

    adapter: TypeAdapter = TypeAdapter(list[Message])
    transcript = all_messages()
    again = adapter.validate_json(adapter.dump_json(transcript))
    assert again == transcript
    assert [m.role for m in again] == [m.role for m in transcript]


def test_tool_result_ok_and_error_stay_consistent() -> None:
    """`ok` and `error` must agree; the validator runs on the way back in too."""

    with pytest.raises(ValueError, match="no error is attached"):
        ToolResult(ok=False)
    with pytest.raises(ValueError, match="but an error is attached"):
        ToolResult(ok=True, error=ToolError(kind=ToolErrorKind.TIMEOUT, message="x"))


@pytest.mark.parametrize("kind", list(ToolErrorKind))
def test_every_tool_error_kind_roundtrips(kind: ToolErrorKind) -> None:
    roundtrip(ToolResult.failure(kind, f"failed: {kind}"))


# ---------------------------------------------------------------------------
# Permissions, contracts, provider types
# ---------------------------------------------------------------------------


def test_permission_models_roundtrip() -> None:
    request = ApprovalRequest(
        run_id="run1",
        agent_id="main",
        node_id="root/agent",
        call_id="call_1",
        tool="edit_file",
        params={"path": "a.py", "old": "x", "new": "y"},
        summary=ApprovalSummary(
            title="edit_file a.py",
            detail="1 replacement",
            diff="--- a.py\n+++ a.py\n@@\n-x\n+y\n",
            danger=True,
        ),
        created_at=TS,
    )
    roundtrip(request)
    roundtrip(Decision.approve(by="user"))
    roundtrip(Decision.deny("not that file", by="user"))


def test_contract_models_roundtrip() -> None:
    roundtrip(
        SafePoint(
            kind=SafePointKind.AFTER_TOOL_BATCH,
            node_id="root/agent",
            agent_id="main",
            attempt=2,
            snapshot={"turn": 3, "messages": []},
            durable=True,
            park=False,
        )
    )
    roundtrip(DelegateOutcome(agent_id="main/0", final_text="done", turns=4, meta={"cost": 0.1}))


def test_provider_models_roundtrip() -> None:
    roundtrip(
        ModelRequest(
            model="anthropic/claude-sonnet-4",
            messages=all_messages(),
            tools=[
                ToolSchema(
                    name="read_file",
                    description="Read a file.",
                    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                )
            ],
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.2,
            top_p=0.9,
            stop=["\n\n"],
            seed=7,
            reasoning=ReasoningConfig(effort="high", max_tokens=2000, exclude=False),
            parallel_tool_calls=True,
            provider_options={"provider": {"order": ["anthropic"]}, "transforms": []},
            metadata={"run_id": "r", "agent_id": "main"},
        )
    )
    roundtrip(
        ModelInfo(
            id="x/y",
            name="X Y",
            context_length=128000,
            max_output_tokens=8192,
            supports_tools=True,
            supports_reasoning=False,
            supports_images=None,
            pricing=ModelPricing(prompt=1e-6, completion=None),
            fetched_at=1_757_000_000.0,
        )
    )
    roundtrip(RetryPolicy(attempts=2, retry_on=frozenset({ToolErrorKind.NETWORK})))
    roundtrip(
        ProviderError(
            kind=ProviderErrorKind.RATE_LIMIT,
            message="slow down",
            status_code=429,
            retry_after=3.5,
            provider="openrouter",
            request_id="req-1",
            details={"header": "x"},
        )
    )


@pytest.mark.parametrize(
    "event",
    [
        TextDelta(text="hi"),
        ReasoningDelta(text="thinking", signature="s"),
        ToolCallStart(index=0, call_id="c", name="read_file"),
        ToolCallDelta(index=0, arguments_delta='{"a":'),
        ToolCallEnd(index=0),
        UsageReport(usage=Usage(prompt_tokens=1, completion_tokens=2, cost_usd=0.5)),
        Finish(reason="tool_calls"),
        StreamError(error=ProviderError(kind=ProviderErrorKind.SERVER, message="boom")),
    ],
    ids=lambda e: e.type,
)
def test_stream_events_roundtrip(event) -> None:
    roundtrip(event)


def test_scripts_roundtrip() -> None:
    script = Script(
        match="by_request_hash",
        turns=[
            ScriptedTurn(key="abc", text="hello", usage=Usage(prompt_tokens=1)),
            ScriptedTurn(
                tool_calls=[
                    ScriptedToolCall(call_id="c1", name="read_file", arguments={"path": "a"}),
                    ScriptedToolCall(
                        call_id="c2", name="grep", arguments_chunks=['{"pat', 'tern":"x"}']
                    ),
                ],
                finish_reason="tool_calls",
                delay_s=0.01,
            ),
            ScriptedTurn(
                error=ProviderError(kind=ProviderErrorKind.NETWORK, message="cut"),
                error_after_chunks=2,
            ),
        ],
        model_info={"x/y": ModelInfo(id="x/y", context_length=1000)},
    )
    roundtrip(script)


# ---------------------------------------------------------------------------
# Events -- R-X-3 says every one is typed, so every one must round-trip
# ---------------------------------------------------------------------------


def all_events() -> list[Event]:
    """One instance of every concrete event class."""

    return [
        RunStateChanged(old=RunState.RUNNING, new=RunState.PAUSING, reason="user"),
        PermissionModeChanged(
            old=PermissionMode.MANUAL, new=PermissionMode.AUTO, pending_resolved=2
        ),
        Checkpoint(kind="after_model_call", to_disk=True, path="s/session.json", duration_ms=4.0),
        RunWarning(
            code="interrupt_no_target",
            message="no active step for target 'main'",
            detail={"target": "main"},
        ),
        GraphDriftWarning(saved_hash="a", rebuilt_hash="b", extra_nodes=["root/x"]),
        NodeStarted(attempt=1, node_class="AgentNode"),
        NodeCompleted(attempt=1, duration_ms=9.0, output_summary="ok"),
        NodeFailed(attempt=1, error="boom", error_type="ValueError"),
        AgentSpawned(parent_id="main", spec_summary="explorer", delegated=True),
        AgentFinished(result_summary="found it", usage=Usage(prompt_tokens=3), outcome="completed"),
        AgentPhaseChanged(old=AgentPhase.RUNNING, new=AgentPhase.PARKED, nonquiescent=0),
        ModelCallStarted(call_id="c", model="x/y", message_count=4, tool_names=["grep"]),
        ModelDelta(call_id="c", text="hi"),
        ModelDelta(call_id="c", reasoning="hmm"),
        ModelDelta(call_id="c", tool_call_index=1, tool_call_delta='{"a"'),
        ModelCallCompleted(call_id="c", usage=Usage(completion_tokens=5), finish_reason="stop"),
        ModelCallFailed(
            call_id="c",
            error=ProviderError(kind=ProviderErrorKind.AUTH, message="nope"),
            attempt=1,
            will_retry=False,
        ),
        ModelCallCancelled(call_id="c", reason="user_interrupt", kept_partial=True),
        ToolCallRequested(call_id="c", tool="grep", params={"pattern": "x"}),
        ApprovalRequested(
            request=ApprovalRequest(
                run_id="r",
                agent_id="main",
                call_id="c",
                tool="shell",
                summary=ApprovalSummary(title="shell"),
                created_at=TS,
            )
        ),
        ApprovalResolved(request_id="req_1", decision=Decision.approve(by="user"), by="user"),
        ToolCallStarted(call_id="c", tool="grep"),
        ToolCallProgress(call_id="c", tool="shell", text="building...", elapsed_ms=1000.0),
        ToolCallCompleted(call_id="c", tool="grep", result=sample_tool_result(), duration_ms=3.0),
        ToolCallFailed(
            call_id="c",
            tool="shell",
            error=ToolError(kind=ToolErrorKind.TIMEOUT, message="600s"),
            duration_ms=600_000.0,
        ),
        ToolCallCancelled(call_id="c", tool="shell", reason="user_interrupt"),
        MessageInjected(message_id="m1", text="stop, do the other thing"),
        EventsDropped(count=17),
    ]


@pytest.mark.parametrize("event", all_events(), ids=lambda e: e.type)
def test_every_event_roundtrips(event: Event) -> None:
    roundtrip(event)


def test_event_union_roundtrips_through_the_discriminator() -> None:
    """Every event re-parses to its own class through the `type` discriminator."""

    for event in all_events():
        again = EVENT_ADAPTER.validate_json(event.model_dump_json())
        assert type(again) is type(event)
        assert again == event


def test_event_classes_are_all_covered() -> None:
    """The list above is exhaustive over the exported event classes.

    Without this, adding an event class and forgetting to cover it would leave a
    silent gap in the R-X-4 evidence.
    """

    import azalabscode.events as events_module

    exported = {
        name
        for name in events_module.__all__
        if isinstance(getattr(events_module, name), type)
        and issubclass(getattr(events_module, name), Event)
        and getattr(events_module, name) is not Event
    }
    covered = {type(e).__name__ for e in all_events()}
    assert exported == covered, f"uncovered event classes: {sorted(exported - covered)}"


# ---------------------------------------------------------------------------
# Property-based coverage of the leaf models
# ---------------------------------------------------------------------------

text_strategy = st.text(max_size=200)


@given(
    prompt=st.integers(min_value=0, max_value=10**9),
    completion=st.integers(min_value=0, max_value=10**9),
    cached=st.integers(min_value=0, max_value=10**9),
    reasoning=st.integers(min_value=0, max_value=10**9),
    cost=st.one_of(st.none(), st.floats(min_value=0, max_value=1e6, allow_nan=False)),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_usage_roundtrips(prompt, completion, cached, reasoning, cost) -> None:
    roundtrip(
        Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            reasoning_tokens=reasoning,
            cost_usd=cost,
        )
    )


@given(text=text_strategy, sig=st.one_of(st.none(), text_strategy))
@settings(max_examples=50)
def test_reasoning_part_roundtrips(text: str, sig: str | None) -> None:
    roundtrip(ReasoningPart(text=text, signature=sig))


@given(
    call_id=st.text(min_size=1, max_size=40),
    name=st.text(min_size=1, max_size=40),
    raw=text_strategy,
)
@settings(max_examples=50)
def test_tool_call_part_roundtrips(call_id: str, name: str, raw: str) -> None:
    roundtrip(ToolCallPart(call_id=call_id, name=name, raw_arguments=raw))


@given(text=text_strategy)
@settings(max_examples=50)
def test_assistant_message_roundtrips(text: str) -> None:
    roundtrip(AssistantMessage(content=[TextPart(text=text)], model="x/y"))


def test_usage_addition_keeps_cost_absent_when_neither_side_reported_one() -> None:
    """A missing cost and a zero cost are different claims; only the provider knows."""

    assert (Usage() + Usage()).cost_usd is None
    assert (Usage(cost_usd=0.5) + Usage()).cost_usd == 0.5
    assert (Usage(cost_usd=0.5) + Usage(cost_usd=0.25)).cost_usd == 0.75
    total = Usage(prompt_tokens=3, cached_tokens=1) + Usage(completion_tokens=4)
    assert (total.prompt_tokens, total.completion_tokens, total.total_tokens) == (3, 4, 7)


def test_extra_keys_are_rejected_rather_than_dropped() -> None:
    """`extra="forbid"` is what makes the round-trip assertion meaningful."""

    with pytest.raises(ValueError):
        TextPart.model_validate({"type": "text", "text": "x", "surprise": 1})
