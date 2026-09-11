"""`AgentState`: everything one agent owns, and everything a resume needs to restore.

It lives in `workflows` rather than in `control` because the agent loop is what reads
and writes it, and `workflows` may not import `control` (import-linter contract 3).
`control` imports it downward and embeds it in the session document (spec 6.2,
delta 19), which is the direction the layer graph allows.

Four fields are not obvious, and each exists because of a specific failure:

* **`pending_results`** -- results that completed before the turn was finalised. A
  turn's results are never appended one at a time (see `workflows.transcript`), so
  between the first completion and the end of the batch they live here. After a
  process death these are what stops a `shell` that had already finished from being
  reported to the model as interrupted.
* **`open_call_ids`** -- calls with no result yet. With `pending_results` this is
  what `repair_transcript` reconciles against on load.
* **`pending_injections`** -- messages from `interrupt(message=...)` that arrived
  mid-turn. They are appended at the next turn boundary, because a user message
  between an assistant tool-call message and its results is rejected by every
  provider, and because an interrupted model call has not yet written the
  `cancelled` assistant message the injection is meant to follow.
* **`result_budget`** -- `TurnBudget.decisions`, memoized by call id so a resumed
  turn elides exactly the result the interrupted one did (spec delta 8).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from azalabscode.messages import Message, Usage, UserMessage
from azalabscode.runstate import AgentPhase
from azalabscode.schema import HarnessModel, VersionedModel
from azalabscode.toolio import ToolResult


class ResumableDelegate(HarnessModel):
    """A `delegate` call that was in flight at process death (spec delta 16).

    A leaf tool call becomes `ToolError(kind="interrupted")` and is never re-run,
    because its effect is unknown. A delegate has no external effect of its own --
    every effect belongs to the child, and the child has its own safe points -- so it
    is *resumed* instead: the same child id, the same task, the child's own
    transcript picked up where it stopped.

    The call is left unanswered on the parent's transcript until the resumed child
    finishes, which is legal: an unanswered trailing call is a turn in progress, and
    `assert_transcript_valid` says so explicitly.
    """

    call_id: str
    child_agent_id: str
    spec_name: str = ""
    task: str = ""
    tools: list[str] | None = None
    model: str | None = None
    max_turns: int | None = None


class AgentState(VersionedModel):
    """One agent's transcript, counters and in-flight bookkeeping."""

    agent_id: str
    parent_id: str | None = None
    spec_name: str = ""
    phase: AgentPhase = AgentPhase.RUNNING
    messages: list[Message] = Field(default_factory=list)
    pending_results: dict[str, ToolResult] = Field(default_factory=dict)
    open_call_ids: list[str] = Field(default_factory=list)
    pending_injections: list[UserMessage] = Field(default_factory=list)
    resume_delegates: list[ResumableDelegate] = Field(default_factory=list)
    """Delegate calls a `load()` reconciled as resumable. Drained by the agent loop
    before its next model call, which is what makes delta 16 true rather than
    documented."""
    result_budget: dict[str, int] = Field(default_factory=dict)
    model_call_seq: int = 0
    """Model calls issued. Seeds `FakeProvider(match="by_index")` across a resume."""
    child_seq: int = 0
    """Children spawned. Bumped with no await between the read and the write, so two
    concurrent spawns cannot take the same id (spec 6.3)."""
    turn: int = 0
    usage: Usage = Field(default_factory=Usage)
    final_text: str = ""
    outcome: str | None = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def next_child_id(self) -> str:
        """Allocate `<agent_id>/<n>` for a delegate or a spawn.

        Synchronous by design: an `await` between reading and writing the counter is
        exactly how two concurrent spawns end up with the same id, and a resumed run
        would then collide with a saved sibling.
        """

        index = self.child_seq
        self.child_seq = index + 1
        return f"{self.agent_id}/{index}"

    def take_injections(self) -> list[UserMessage]:
        """Drain queued injected messages onto the transcript. Returns what moved.

        Called at a turn boundary, which is the only place a user message may be
        inserted without breaking the tool-call/tool-result pairing.
        """

        if not self.pending_injections:
            return []
        drained = list(self.pending_injections)
        self.pending_injections = []
        self.messages = [*self.messages, *drained]
        return drained


__all__ = ["AgentState", "ResumableDelegate"]
