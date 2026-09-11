"""The session document itself: round-trip, workflow resolution, spilling, drift.

R-X-4 wants `Session.model_validate_json(s.model_dump_json()) == s` for a session
carrying every agent phase, spilled values and pending approvals. That is the
property test at the bottom of this file; the rest is the vocabulary it needs to
hold: `WorkflowRef` resolving an import path, `ValueRef` deciding inline versus
spilled, and `check_graph_drift`'s three-way answer (spec C-2).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from azalabscode.cancellation import StepKind
from azalabscode.content import TextPart, ToolCallPart
from azalabscode.control import Checkpointer, check_graph_drift
from azalabscode.control.session import (
    SPILL_THRESHOLD_BYTES,
    InflightStep,
    NodeRecord,
    Session,
    ValueRef,
    WorkflowRef,
    hash_config,
)
from azalabscode.errors import (
    ConfigurationError,
    GraphMismatchError,
    WorkflowNotImportable,
)
from azalabscode.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from azalabscode.permissions import ApprovalRequest, ApprovalSummary, PermissionMode
from azalabscode.runstate import AgentPhase, NodeStatus, RunState
from azalabscode.toolio import ToolErrorKind, ToolResult
from azalabscode.workflows.state import AgentState, ResumableDelegate

# ---------------------------------------------------------------------------
# WorkflowRef
# ---------------------------------------------------------------------------


def test_a_workflow_ref_resolves_its_build_callable() -> None:
    """The whole reconstruction recipe is `(import_path, config)` (R-C-11)."""

    ref = WorkflowRef.of("tests.resumable:build", {"script_path": "x"})
    assert callable(ref.resolve())
    assert ref.config_hash == hash_config({"script_path": "x"})


@pytest.mark.parametrize(
    ("path", "fragment"),
    [
        ("tests.resumable", "expected 'package.module:build'"),
        ("tests.does_not_exist:build", "ModuleNotFoundError"),
        ("tests.resumable:nope", "has no attribute 'nope'"),
    ],
    ids=["no-colon", "no-module", "no-attribute"],
)
def test_an_unresolvable_workflow_says_why(path: str, fragment: str) -> None:
    """R-C-11: "raises with a clear message". The message names the path and the cause."""

    with pytest.raises(WorkflowNotImportable) as caught:
        WorkflowRef(import_path=path).resolve()
    assert path in str(caught.value)
    assert fragment in str(caught.value)


def test_a_config_that_no_longer_validates_fails_at_load_not_in_build() -> None:
    """R-C-11's second half. Failing here beats a `TypeError` five lines into `build`."""

    ref = WorkflowRef.of(
        "tests.resumable:build",
        {"script_path": "s", "workspace": "w"},  # `markers` is required and missing
        config_type="tests.resumable:Config",
    )
    with pytest.raises(ConfigurationError) as caught:
        ref.validated_config()
    assert "tests.resumable:Config" in str(caught.value)


def test_a_config_with_no_declared_type_is_passed_through() -> None:
    """Not every workflow declares a config model; the dict is then the contract."""

    ref = WorkflowRef.of("tests.resumable:build", {"anything": 1})
    assert ref.validated_config() == {"anything": 1}


def test_the_graph_hash_is_not_the_config_hash() -> None:
    """Editing a prompt must not invalidate a session, so the two are separate fields."""

    a = WorkflowRef.of("m:build", {"prompt": "be terse"}, graph_hash="g1")
    b = WorkflowRef.of("m:build", {"prompt": "be verbose"}, graph_hash="g1")
    assert a.config_hash != b.config_hash
    assert a.graph_hash == b.graph_hash


# ---------------------------------------------------------------------------
# ValueRef and spilling
# ---------------------------------------------------------------------------


def test_a_small_value_stays_inline(tmp_path: Path) -> None:
    """`session.json` holds it, so the checkpoint is still one readable file."""

    ref = Checkpointer(session_dir=tmp_path).value_ref({"answer": 42})
    assert not ref.spilled
    assert ref.resolve(tmp_path) == {"answer": 42}
    assert not (tmp_path / "values").exists()


def test_a_large_value_spills_to_the_values_directory(tmp_path: Path) -> None:
    """A 40 MB node output in the document would make it undiffable and unreadable."""

    payload = {"text": "x" * (SPILL_THRESHOLD_BYTES + 1)}
    ref = Checkpointer(session_dir=tmp_path).value_ref(payload)

    assert ref.spilled
    assert ref.path is not None and ref.path.startswith("values/")
    assert (tmp_path / ref.path).exists()
    assert ref.resolve(tmp_path) == payload
    assert ref.size > SPILL_THRESHOLD_BYTES


def test_two_identical_large_values_share_one_file(tmp_path: Path) -> None:
    """Content-addressed, so a node re-run with the same output does not duplicate it."""

    payload = {"text": "y" * (SPILL_THRESHOLD_BYTES + 1)}
    cp = Checkpointer(session_dir=tmp_path)
    first, second = cp.value_ref(payload), cp.value_ref(payload)

    assert first.path == second.path
    assert len(list((tmp_path / "values").iterdir())) == 1


def test_an_in_memory_run_keeps_everything_inline() -> None:
    """No directory means nothing is being written, so nothing needs to be kept small."""

    ref = Checkpointer().value_ref({"text": "z" * (SPILL_THRESHOLD_BYTES + 1)})
    assert not ref.spilled


def test_resolving_a_spilled_ref_without_a_directory_is_an_error(tmp_path: Path) -> None:
    """A caller that lost track of the session directory should hear about it."""

    ref = Checkpointer(session_dir=tmp_path).value_ref({"t": "q" * (SPILL_THRESHOLD_BYTES + 1)})
    with pytest.raises(ValueError, match="spilled"):
        ref.resolve(None)


# ---------------------------------------------------------------------------
# Graph drift (spec C-2)
# ---------------------------------------------------------------------------


def _session_with_nodes(*names: str, graph_hash: str = "g1") -> Session:
    return Session(
        run_id="r",
        workflow=WorkflowRef.of("m:build", {}, graph_hash=graph_hash),
        nodes={name: NodeRecord(node_id=name, status=NodeStatus.COMPLETED) for name in names},
    )


def test_an_unchanged_graph_is_silent() -> None:
    """Equal hashes and equal ids: nothing to say."""

    session = _session_with_nodes("a", "b")
    assert check_graph_drift(session, node_ids={"a", "b"}, graph_hash="g1") is None


def test_a_missing_node_is_fatal() -> None:
    """The session holds an output with nowhere to put it. There is no honest resume."""

    session = _session_with_nodes("a", "b")
    with pytest.raises(GraphMismatchError) as caught:
        check_graph_drift(session, node_ids={"a"}, graph_hash="g2")
    assert "b" in str(caught.value)


def test_an_extra_node_is_a_warning_not_an_error() -> None:
    """A graph that grew can still absorb everything the session knows."""

    session = _session_with_nodes("a")
    warning = check_graph_drift(session, node_ids={"a", "c"}, graph_hash="g2")
    assert warning is not None
    assert warning.extra_nodes == ["c"]
    assert (warning.saved_hash, warning.rebuilt_hash) == ("g1", "g2")


def test_strict_mode_promotes_drift_to_an_error() -> None:
    """`strict_graph_hash=True` for a run that must be byte-identical or not run."""

    session = _session_with_nodes("a")
    with pytest.raises(GraphMismatchError):
        check_graph_drift(session, node_ids={"a", "c"}, graph_hash="g2", strict=True)


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def rich_session() -> Session:
    """A session with one of everything the round-trip is supposed to survive."""

    call = ToolCallPart(call_id="c1", name="note", arguments={"text": "hi"}, raw_arguments='{"t"}')
    return Session(
        run_id="run_1",
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        updated_at=datetime(2026, 9, 2, tzinfo=UTC),
        workflow=WorkflowRef.of("tests.resumable:build", {"n": 1}, graph_hash="g"),
        run_state=RunState.WAITING_APPROVAL,
        resume_state=RunState.WAITING_APPROVAL,
        permission_mode=PermissionMode.MANUAL,
        gated_tools=["note"],
        agents={
            "main": AgentState(
                agent_id="main",
                phase=AgentPhase.WAITING_APPROVAL,
                messages=[
                    SystemMessage(content="be terse"),
                    UserMessage(content=[TextPart(text="do it")]),
                    AssistantMessage(content=[call], usage=Usage(prompt_tokens=10)),
                ],
                pending_results={"c1": ToolResult.ok_text("done")},
                open_call_ids=["c1"],
                pending_injections=[UserMessage(content=[TextPart(text="also this")])],
                resume_delegates=[
                    ResumableDelegate(call_id="c9", child_agent_id="main/0", task="look")
                ],
                result_budget={"c1": 100},
                model_call_seq=3,
                child_seq=1,
                usage=Usage(prompt_tokens=10, completion_tokens=2),
            ),
            "main/0": AgentState(
                agent_id="main/0",
                parent_id="main",
                phase=AgentPhase.FINISHED,
                messages=[
                    ToolResultMessage(
                        call_id="c2",
                        name="slow",
                        result=ToolResult.failure(ToolErrorKind.INTERRUPTED, "gone"),
                    )
                ],
            ),
        },
        nodes={
            "prepare": NodeRecord(
                node_id="prepare",
                status=NodeStatus.COMPLETED,
                output=ValueRef(inline="prepared", size=9),
                started_at=datetime(2026, 9, 1, tzinfo=UTC),
            ),
            "agent": NodeRecord(
                node_id="agent",
                status=NodeStatus.RUNNING,
                output=ValueRef(path="values/abc.json", size=99_999),
            ),
        },
        pending_approvals=[
            ApprovalRequest(
                run_id="run_1",
                agent_id="main",
                call_id="c1",
                tool="note",
                params={"text": "hi"},
                summary=ApprovalSummary(title="note", detail="hi", danger=True),
            )
        ],
        inflight=[
            InflightStep(
                step_id="s1",
                kind=StepKind.TOOL_CALL,
                agent_id="main",
                call_ids=["c1"],
                description="1 tool call(s)",
                duration_ms=12.5,
            ),
            InflightStep(
                step_id="s2",
                kind=StepKind.DELEGATE,
                agent_id="main",
                child_agent_id="main/0",
                description="delegate to explorer",
            ),
        ],
        event_seq=42,
        usage_total=Usage(prompt_tokens=10, completion_tokens=2),
        rng_seed=7,
        custom={"scratch": [1, 2, 3]},
    )


def test_a_full_session_round_trips_losslessly() -> None:
    """R-X-4, on a document with every field populated."""

    session = rich_session()
    assert Session.loads(session.model_dump_json()) == session


def test_a_session_round_trips_through_a_file(tmp_path: Path) -> None:
    """The path the controller actually uses, including the indented bytes."""

    session = rich_session()
    target = tmp_path / "session.json"
    target.write_bytes(session.dumps())

    assert Session.load(target) == session
    assert json.loads(target.read_text(encoding="utf-8"))["run_id"] == "run_1"
    assert b"\n  " in target.read_bytes(), "a checkpoint is meant to be readable"


def test_an_unknown_key_is_rejected_rather_than_dropped() -> None:
    """`extra="forbid"` is what makes round-trip fidelity checkable at all."""

    payload = json.loads(rich_session().model_dump_json())
    payload["invented_by_a_future_version"] = True
    with pytest.raises(ValueError, match="invented_by_a_future_version"):
        Session.loads(json.dumps(payload))


def test_the_convenience_accessors_agree_with_the_fields() -> None:
    """`completed_nodes` and `inflight_for` are what `load()` and the tests read."""

    session = rich_session()
    assert session.completed_nodes() == ["prepare"]
    assert [s.kind for s in session.inflight_for("main")] == [
        StepKind.TOOL_CALL,
        StepKind.DELEGATE,
    ]
    assert session.agent("main/0") is not None
    assert session.agent("nope") is None


# ---------------------------------------------------------------------------
# R-X-4 as a property
# ---------------------------------------------------------------------------

phases = st.sampled_from(list(AgentPhase))
states = st.sampled_from(list(RunState))
statuses = st.sampled_from(list(NodeStatus))


@st.composite
def agent_states(draw: st.DrawFn, agent_id: str) -> AgentState:
    """An agent in an arbitrary phase, with an arbitrary amount in flight."""

    call_ids = draw(st.lists(st.text(min_size=1, max_size=6), max_size=3, unique=True))
    messages: list[object] = [SystemMessage(content=draw(st.text(max_size=20)))]
    if call_ids:
        messages.append(
            AssistantMessage(
                content=[ToolCallPart(call_id=c, name="note", raw_arguments="{}") for c in call_ids]
            )
        )
    return AgentState(
        agent_id=agent_id,
        phase=draw(phases),
        messages=messages,  # type: ignore[arg-type]
        pending_results={
            c: ToolResult.ok_text("ok") for c in draw(st.lists(st.sampled_from(call_ids or ["x"])))
        }
        if call_ids
        else {},
        open_call_ids=call_ids,
        model_call_seq=draw(st.integers(0, 50)),
        child_seq=draw(st.integers(0, 5)),
        turn=draw(st.integers(0, 40)),
        usage=Usage(prompt_tokens=draw(st.integers(0, 10_000))),
    )


@st.composite
def sessions(draw: st.DrawFn) -> Session:
    """A whole session: several agents, some nodes, maybe a pending approval."""

    agent_ids = ["main", *[f"main/{i}" for i in range(draw(st.integers(0, 3)))]]
    approvals = [
        ApprovalRequest(
            run_id="r",
            agent_id="main",
            call_id=draw(st.text(min_size=1, max_size=6)),
            tool="note",
            params={"text": draw(st.text(max_size=20))},
            summary=ApprovalSummary(title="note"),
        )
        for _ in range(draw(st.integers(0, 2)))
    ]
    node_ids = draw(st.lists(st.text(min_size=1, max_size=8), max_size=4, unique=True))
    return Session(
        run_id=draw(st.text(min_size=1, max_size=10)),
        run_state=draw(states),
        resume_state=draw(states),
        permission_mode=draw(st.sampled_from(list(PermissionMode))),
        agents={a: draw(agent_states(a)) for a in agent_ids},
        nodes={
            n: NodeRecord(
                node_id=n,
                status=draw(statuses),
                attempt=draw(st.integers(0, 3)),
                output=ValueRef(inline=draw(st.one_of(st.none(), st.text(max_size=30)))),
            )
            for n in node_ids
        },
        pending_approvals=approvals,
        event_seq=draw(st.integers(0, 100_000)),
        rng_seed=draw(st.one_of(st.none(), st.integers())),
        custom={"k": draw(st.lists(st.integers(), max_size=3))},
    )


@given(sessions())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_every_session_round_trips(session: Session) -> None:
    """R-X-4 over random sessions with every agent phase and every run state."""

    assert Session.loads(session.model_dump_json()) == session
