"""Resume reconciliation: what a loaded session means, step by step.

A checkpoint is taken at a safe point, but a process does not die at one. What comes
back from disk is a run that stopped somewhere arbitrary, and the entries in
`Session.inflight` are the record of where. Each kind is reconciled differently, and
the differences are the whole point (plan.md, "Session and resume"):

* **`model_call`** -- dropped and re-issued. Nothing was half-appended: the agent
  loop appends the assistant message only after the stream finishes, so a killed
  stream leaves no trace on the transcript (spec C-1). `model_call_seq` was not
  incremented either, so `FakeProvider(match="by_index")` lands on the same turn.

* **`tool_call`** -- every call it covered that has no result becomes
  `ToolError(kind="interrupted", message="process terminated during execution;
  effect unknown")` and is **never** re-executed (R-C-13). A result that had already
  landed in `pending_results` wins over the error, which is what stops a `shell` that
  finished a millisecond before the kill from being reported as interrupted.

* **`delegate`** -- resumed, not errored (delta 16). The call is left unanswered and
  a `ResumableDelegate` is queued on the parent's state; the agent loop re-runs it
  against the child's restored transcript before its next model call.

Then `repair_transcript` answers everything still open -- calls that never made it
into an `inflight` entry at all, because the process died between the safe point and
the dispatch -- and `assert_transcript_valid` must hold, or `load()` fails hard.
There is exactly one reconciliation path and it is this one: `repair_transcript`
already prefers a pending result over an interrupted error, so nothing here
re-implements that decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from azalabscode.cancellation import StepKind
from azalabscode.content import ToolCallPart
from azalabscode.control.session import Session
from azalabscode.errors import GraphMismatchError
from azalabscode.events import GraphDriftWarning
from azalabscode.messages import AssistantMessage, assert_transcript_valid, open_call_ids
from azalabscode.runstate import AgentPhase, RunState
from azalabscode.workflows.state import AgentState, ResumableDelegate
from azalabscode.workflows.transcript import (
    interrupted_fill,
    not_run_fill,
    repair_transcript,
)

DELEGATE_TOOL = "delegate"
"""The one tool whose in-flight call is resumed rather than errored (delta 16)."""

_TRANSIENT_STATES = frozenset({RunState.PAUSING, RunState.INTERRUPTING, RunState.PAUSED})


def resume_state_for(state: RunState) -> RunState:
    """What a `resume()` should return to, given the state at save time.

    `PAUSING` and `INTERRUPTING` are transient and collapse to `RUNNING`.
    `WAITING_APPROVAL` survives: the request the run was blocked on is in the
    session, so the reloaded run is still waiting for it (R-C-9).
    """

    if state is RunState.WAITING_APPROVAL:
        return RunState.WAITING_APPROVAL
    if state in _TRANSIENT_STATES or state is RunState.RUNNING:
        return RunState.RUNNING
    return RunState.RUNNING


@dataclass
class ResumeReport:
    """What reconciliation did, so `load()` can say it and a test can assert it."""

    dropped_model_calls: list[str] = field(default_factory=list)
    """Call ids of streams that were cut off. Re-issued, never recovered."""
    interrupted_calls: list[str] = field(default_factory=list)
    """Tool calls answered with `interrupted`. Never re-executed (R-C-13)."""
    recovered_results: list[str] = field(default_factory=list)
    """Calls whose real result was in `pending_results` and survived."""
    resumed_delegates: list[str] = field(default_factory=list)
    """Call ids of delegates left open for the loop to re-run (delta 16)."""
    not_run_calls: list[str] = field(default_factory=list)
    """Calls blocked at the permission gate. Never started, so safe to re-issue."""
    repaired: dict[str, int] = field(default_factory=dict)
    """Per agent, how many results were backfilled."""

    @property
    def total_repaired(self) -> int:
        """How many holes were filled across every agent."""

        return sum(self.repaired.values())

    def summary(self) -> str:
        """One line for a log or an event payload."""

        return (
            f"{len(self.dropped_model_calls)} model call(s) re-issued, "
            f"{len(self.interrupted_calls)} tool call(s) interrupted, "
            f"{len(self.recovered_results)} result(s) recovered, "
            f"{len(self.not_run_calls)} never started, "
            f"{len(self.resumed_delegates)} delegate(s) resumed"
        )


def reconcile(session: Session) -> ResumeReport:
    """Bring every agent's transcript back to a valid, resumable state.

    Mutates the `AgentState`s in `session` in place: this is called by `load()`
    before the states are handed to the controller, and the object the loop later
    adopts is the object reconciled here (D-M2-1 -- one state, not two).

    Raises `TranscriptError` if a transcript cannot be made valid. That is the hard
    failure plan.md asks for: a session that reloads into an invalid transcript would
    be rejected by the provider on the first request, and finding out here is
    cheaper.
    """

    report = ResumeReport()
    deferred: dict[str, set[str]] = {}

    for step in session.inflight:
        state = session.agents.get(step.agent_id)
        if state is None:
            continue
        if step.kind is StepKind.MODEL_CALL:
            report.dropped_model_calls.append(step.call_id or step.step_id)
        elif step.kind is StepKind.DELEGATE:
            entry = _resumable_delegate(state, step.child_agent_id)
            if entry is not None:
                state.resume_delegates = [*state.resume_delegates, entry]
                deferred.setdefault(step.agent_id, set()).add(entry.call_id)
                report.resumed_delegates.append(entry.call_id)

    for agent_id, state in session.agents.items():
        open_before = set(open_call_ids(state.messages))
        recovered = sorted(open_before & set(state.pending_results))
        report.recovered_results.extend(recovered)

        # A call blocked at the permission gate had *definitely* not run: the
        # dispatcher was waiting for a human, not for the tool. Telling the model its
        # effect is unknown would be false, and would stop it re-issuing a call it is
        # entitled to re-issue once the restored approval is resolved (R-C-9).
        answers = dict(state.pending_results)
        parts = {call.call_id: call for call in _tool_calls(state.messages)}
        for call_id in _approval_blocked(session, agent_id) & open_before:
            part = parts.get(call_id)
            if part is not None and call_id not in answers:
                answers[call_id] = not_run_fill(part)
                report.not_run_calls.append(call_id)

        written = repair_transcript(
            state.messages,
            fill=interrupted_fill,
            pending=answers,
            defer=deferred.get(agent_id, set()),
            agent_id=agent_id,
        )
        report.repaired[agent_id] = len(written)
        answered_elsewhere = set(recovered) | set(report.not_run_calls)
        report.interrupted_calls.extend(
            message.call_id
            for message in written
            if message.result.error is not None and message.call_id not in answered_elsewhere
        )

        # The batch these belonged to is now materialised on the transcript, so the
        # buffer must be empty: a leftover entry would be re-applied by the *next*
        # reconciliation and answer a call twice.
        state.pending_results = {}
        state.open_call_ids = open_call_ids(state.messages)
        if state.phase is not AgentPhase.FINISHED:
            state.phase = AgentPhase.PARKED
        assert_transcript_valid(state.messages, agent_id=agent_id)

    return report


def _approval_blocked(session: Session, agent_id: str) -> set[str]:
    """Call ids this agent had sitting at the permission gate when the process died."""

    return {
        request.call_id
        for request in session.pending_approvals
        if request.agent_id == agent_id and request.call_id
    }


def _resumable_delegate(state: AgentState, child_agent_id: str | None) -> ResumableDelegate | None:
    """Rebuild the `delegate` call that was in flight, from the transcript itself.

    The arguments the model sent are on the `ToolCallPart`, which is the only copy
    that survives -- the parsed params live in the dispatcher and die with it. A call
    whose arguments no longer parse is not resumable, and falls through to the
    `interrupted` error like any other tool call.
    """

    if not child_agent_id:
        return None
    open_ids = set(open_call_ids(state.messages))
    for call in reversed(_tool_calls(state.messages)):
        if call.call_id not in open_ids or call.name != DELEGATE_TOOL:
            continue
        arguments = _arguments(call)
        if arguments is None:
            return None
        task = arguments.get("task")
        if not isinstance(task, str) or not task:
            return None
        tools = arguments.get("tools")
        model = arguments.get("model")
        max_turns = arguments.get("max_turns")
        return ResumableDelegate(
            call_id=call.call_id,
            child_agent_id=child_agent_id,
            spec_name=str(arguments.get("spec") or ""),
            task=task,
            tools=list(tools) if isinstance(tools, list) else None,
            model=str(model) if isinstance(model, str) else None,
            max_turns=max_turns if isinstance(max_turns, int) else None,
        )
    return None


def _tool_calls(messages: list[Any]) -> list[ToolCallPart]:
    """Every tool call in a transcript, in call order."""

    out: list[ToolCallPart] = []
    for message in messages:
        if isinstance(message, AssistantMessage):
            out.extend(message.tool_calls)
    return out


def _arguments(call: ToolCallPart) -> dict[str, Any] | None:
    """The call's arguments as a dict, or `None` if they never parsed."""

    if call.arguments is not None:
        return dict(call.arguments)
    if not call.raw_arguments:
        return None
    try:
        parsed = json.loads(call.raw_arguments)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def check_graph_drift(
    session: Session,
    *,
    node_ids: set[str],
    graph_hash: str,
    strict: bool = False,
) -> GraphDriftWarning | None:
    """Compare a saved session against a rebuilt graph (spec C-2).

    Three outcomes, and the asymmetry is deliberate:

    * equal hashes and no missing ids -- silent.
    * **missing** ids -- `GraphMismatchError`. The session holds outputs for nodes
      the rebuilt graph does not have, so there is nowhere to put them and no honest
      way to continue.
    * extra ids or a changed hash -- a `GraphDriftWarning` event, because a graph
      that grew can still absorb everything the session knows. `strict=True`
      promotes it to the same hard error.
    """

    saved_ids = set(session.nodes)
    missing = sorted(saved_ids - node_ids)
    saved_hash = session.workflow.graph_hash
    if missing:
        raise GraphMismatchError(missing, graph_hash, saved_hash)

    extra = sorted(node_ids - saved_ids)
    if not extra and (not saved_hash or saved_hash == graph_hash):
        return None
    if strict:
        raise GraphMismatchError([], graph_hash, saved_hash)
    return GraphDriftWarning(saved_hash=saved_hash, rebuilt_hash=graph_hash, extra_nodes=extra)


__all__ = [
    "DELEGATE_TOOL",
    "ResumeReport",
    "check_graph_drift",
    "reconcile",
    "resume_state_for",
]
