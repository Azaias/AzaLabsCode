"""The transcript invariant, as a property over completion orders and interrupts.

The invariant: every `ToolCallPart` has exactly one later `ToolResultMessage` with
the same `call_id`, in *call* order.

The property test is the point of this file. Any single hand-written case passes by
accident; what has to hold is that no permutation of completion order, and no cut
point in the middle of a batch, can produce a transcript that
`assert_transcript_valid` rejects. `test_appending_in_completion_order_breaks_it`
is the control: it proves the property is not vacuous by showing the obvious
implementation failing on the same inputs.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from azalabscode.content import TextPart, ToolCallPart
from azalabscode.messages import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    TranscriptError,
    UserMessage,
    assert_transcript_valid,
    open_call_ids,
)
from azalabscode.toolio import ToolErrorKind, ToolResult
from azalabscode.workflows.transcript import (
    TurnResults,
    cancelled_fill,
    finalize_turn,
    interrupted_fill,
    not_run_fill,
    repair_transcript,
)


def call(index: int, name: str = "read_file") -> ToolCallPart:
    """A tool call with a predictable id."""

    return ToolCallPart(
        call_id=f"c{index}",
        name=name,
        arguments={"path": f"f{index}.py"},
        raw_arguments=f'{{"path": "f{index}.py"}}',
    )


def turn_of(n: int) -> tuple[list[Message], list[ToolCallPart]]:
    """A transcript ending in an assistant message that requested `n` calls."""

    calls = [call(i) for i in range(n)]
    messages: list[Message] = [
        UserMessage.of("do the thing"),
        AssistantMessage(content=[TextPart(text="working"), *calls], model="m"),
    ]
    return messages, calls


def result_for(index: int) -> ToolResult:
    """A successful result, distinguishable from a filled hole."""

    return ToolResult.ok_text(f"contents of f{index}.py")


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@st.composite
def batches(draw: st.DrawFn) -> tuple[int, list[int], int]:
    """`(call count, completion order, how many completed before the interrupt)`."""

    n = draw(st.integers(min_value=1, max_value=8))
    order = draw(st.permutations(list(range(n))))
    cut = draw(st.integers(min_value=0, max_value=n))
    return n, list(order), cut


@settings(max_examples=300, deadline=None)
@given(batches())
def test_the_invariant_holds_for_any_completion_order_and_any_interrupt(
    case: tuple[int, list[int], int],
) -> None:
    """The property the whole control layer is built to preserve.

    Whatever order the tools finished in, and wherever the interrupt landed, the
    transcript is valid, every call is answered exactly once, and the answers are in
    call order.
    """

    n, order, cut = case
    messages, calls = turn_of(n)
    turn = TurnResults.for_calls(calls)

    for index in order[:cut]:
        turn.record(calls[index].call_id, result_for(index))

    written = finalize_turn(messages, turn, fill=cancelled_fill, agent_id="main")

    assert_transcript_valid(messages, agent_id="main")
    assert [m.call_id for m in written] == [c.call_id for c in calls]
    assert open_call_ids(messages) == []

    completed = set(order[:cut])
    for i, message in enumerate(written):
        if i in completed:
            assert message.result.ok
            assert message.result.text == f"contents of f{i}.py"
        else:
            assert not message.result.ok
            assert message.result.error is not None
            assert message.result.error.kind is ToolErrorKind.CANCELLED


@settings(max_examples=200, deadline=None)
@given(batches())
def test_appending_in_completion_order_breaks_it(case: tuple[int, list[int], int]) -> None:
    """The control. Without this the property test could be passing vacuously.

    This is the obvious implementation -- append each result as it arrives -- and it
    produces an invalid transcript whenever completion order differs from call
    order, or whenever a batch is cut short.
    """

    n, order, cut = case
    messages, calls = turn_of(n)
    naive: list[Message] = list(messages)
    for index in order[:cut]:
        naive.append(
            ToolResultMessage(
                call_id=calls[index].call_id, name=calls[index].name, result=result_for(index)
            )
        )

    # The only completion order the naive append survives is the one where the
    # calls happened to finish in call order with no gaps.
    in_call_order = order[:cut] == list(range(cut))
    if not in_call_order:
        with pytest.raises(TranscriptError):
            assert_transcript_valid(naive)
    elif cut < n:
        # In call order but incomplete: valid as a turn in progress, and *not* valid
        # as something to send to a model, which is what `open_call_ids` reports.
        assert_transcript_valid(naive)
        assert open_call_ids(naive) == [c.call_id for c in calls[cut:]]
    else:
        assert_transcript_valid(naive)


# ---------------------------------------------------------------------------
# TurnResults
# ---------------------------------------------------------------------------


def test_a_completed_call_keeps_its_real_result_when_the_batch_is_cut() -> None:
    """The plan's worked example: r1 done, r2 running, r3 unstarted."""

    messages, calls = turn_of(3)
    turn = TurnResults.for_calls(calls)
    turn.record("c0", result_for(0))

    written = finalize_turn(messages, turn, agent_id="main")

    assert written[0].result.ok
    assert written[1].result.error is not None
    assert written[2].result.error is not None
    assert [m.call_id for m in written] == ["c0", "c1", "c2"]


def test_the_first_result_for_a_call_wins() -> None:
    """A late-arriving duplicate must not overwrite what was already recorded."""

    _, calls = turn_of(1)
    turn = TurnResults.for_calls(calls)
    turn.record("c0", ToolResult.ok_text("first"))
    turn.record("c0", ToolResult.ok_text("second"))

    assert turn.results["c0"].text == "first"


def test_missing_and_complete_track_the_holes() -> None:
    """What the loop consults to decide whether a batch finished."""

    _, calls = turn_of(3)
    turn = TurnResults.for_calls(calls)
    assert not turn.complete
    assert [c.call_id for c in turn.missing()] == ["c0", "c1", "c2"]

    for c in calls:
        turn.record(c.call_id, ToolResult.ok_text("x"))
    assert turn.complete
    assert turn.missing() == []
    assert turn.call_ids == ["c0", "c1", "c2"]


@pytest.mark.parametrize(
    ("fill", "kind"),
    [
        (cancelled_fill, ToolErrorKind.CANCELLED),
        (not_run_fill, ToolErrorKind.CANCELLED),
        (interrupted_fill, ToolErrorKind.INTERRUPTED),
    ],
)
def test_each_fill_produces_the_kind_the_model_is_meant_to_see(fill, kind) -> None:
    """`cancelled` means the effect is known; `interrupted` means it is not."""

    result = fill(call(0))
    assert result.error is not None
    assert result.error.kind is kind
    assert result.text  # the model only ever reads `content`
    assert result.meta["call_id"] == "c0"


def test_finalize_rejects_a_turn_the_transcript_never_requested() -> None:
    """The validation is about the caller, not about `TurnResults`."""

    messages, _ = turn_of(1)
    stray = TurnResults.for_calls([call(9)])
    stray.record("c9", ToolResult.ok_text("x"))

    with pytest.raises(TranscriptError):
        finalize_turn(messages, stray, agent_id="main")


# ---------------------------------------------------------------------------
# repair_transcript (what load() calls at M3)
# ---------------------------------------------------------------------------


def test_repair_backfills_every_open_call() -> None:
    """R-C-13: a call in flight at process death is answered, never re-run."""

    messages, _calls = turn_of(3)
    messages.append(ToolResultMessage(call_id="c0", name="read_file", result=result_for(0)))

    written = repair_transcript(messages, agent_id="main")

    assert [m.call_id for m in written] == ["c1", "c2"]
    assert all(m.result.error is not None for m in written)
    assert written[0].result.error is not None
    assert written[0].result.error.kind is ToolErrorKind.INTERRUPTED
    assert_transcript_valid(messages, agent_id="main")
    assert open_call_ids(messages) == []


def test_repair_prefers_a_pending_result_over_the_interrupted_error() -> None:
    """A result that completed but was never materialised is not lost."""

    messages, _ = turn_of(2)
    written = repair_transcript(
        messages,
        pending={"c1": ToolResult.ok_text("real")},
        agent_id="main",
    )

    assert written[0].result.error is not None
    assert written[1].result.ok
    assert written[1].result.text == "real"


def test_repair_is_a_no_op_on_a_complete_transcript() -> None:
    """Idempotent: `load()` may call it on a transcript that needs nothing."""

    messages, calls = turn_of(2)
    turn = TurnResults.for_calls(calls)
    for i, c in enumerate(calls):
        turn.record(c.call_id, result_for(i))
    finalize_turn(messages, turn)

    assert repair_transcript(messages) == []
    assert repair_transcript(messages) == []
    assert_transcript_valid(messages)


def test_repair_answers_a_call_whose_part_is_gone() -> None:
    """Defensive: a transcript edited by hand still comes back valid."""

    messages: list[Message] = [
        AssistantMessage(content=[ToolCallPart(call_id="x1", name="", raw_arguments="")], model="m")
    ]
    written = repair_transcript(messages)
    assert len(written) == 1
    assert_transcript_valid(messages)
