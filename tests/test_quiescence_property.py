"""The quiescence property, over random command sequences.

Three things have to hold for *every* interleaving of `pause`, `pause(hard=True)`,
`resume`, `interrupt`, `set_permission_mode` and `approve`/`deny`:

1. **No illegal transition.** `RunStateMachine.transition` raises
   `IllegalTransition` rather than silently accepting one, so any sequence that
   drives the controller into a state the spec 6.1 table forbids fails here.
2. **PAUSED implies nothing is in flight.** `nonquiescent == 0` at every moment the
   run reports PAUSED. This is delta 14's definition, and the whole reason PAUSED is
   trustworthy enough to save from at M3.
3. **No deadlock.** After the sequence, releasing everything must let the run reach a
   terminal state inside a bound. This is the assertion that costs the most and
   catches the most: the delegate deadlock, a parked agent holding the checkpoint
   lock, and a gate waiting on an approval nobody will answer all show up here as a
   timeout rather than as a wrong answer.

The transcript invariant is checked after every command too, because a command
sequence is exactly how you find the interleaving that breaks it.

Hypothesis drives a synchronous test that owns its own event loop rather than an
async one: the interaction between `@given` and an async fixture is one more thing
that can fail for reasons unrelated to the property.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from azalabscode.control import QueueApprovalHandler
from azalabscode.errors import ScriptExhausted
from azalabscode.messages import assert_transcript_valid, open_call_ids
from azalabscode.permissions import PermissionMode
from azalabscode.runstate import TERMINAL_STATES, RunState
from azalabscode.workflows import AgentSpec
from tests.harness import Rig, build_rig, calls, echo_call, says, touch_call

COMMANDS = (
    "pause",
    "pause_hard",
    "resume",
    "interrupt",
    "inject",
    "manual",
    "auto",
    "approve",
    "deny",
    "tick",
)

SPEC = AgentSpec(name="main", model="fake/model", system_prompt="be useful", max_turns=4)


def script() -> list:
    """A run with enough shape to be interesting: slow tools, a gated tool, text.

    Padded with terminal turns so that a sequence full of interrupts -- each of
    which consumes a scripted turn -- still has an answer waiting for it.
    """

    return [
        calls(echo_call("a", sleep=0.02), touch_call("b"), chunk_delay_s=0.005),
        calls(echo_call("c", sleep=0.02), chunk_delay_s=0.005),
        calls(touch_call("d"), chunk_delay_s=0.005),
        says("finished", chunk_delay_s=0.005),
        says("finished"),
        says("finished"),
        says("finished"),
        says("finished"),
    ]


async def apply(rig: Rig, command: str) -> None:
    """Run one command against the controller."""

    control = rig.controller
    if command == "pause":
        await control.pause()
    elif command == "pause_hard":
        await control.pause(hard=True)
    elif command == "resume":
        await control.resume()
    elif command == "interrupt":
        await control.interrupt()
    elif command == "inject":
        await control.interrupt(message="do the other thing instead")
    elif command == "manual":
        await control.set_permission_mode(PermissionMode.MANUAL)
    elif command == "auto":
        await control.set_permission_mode(PermissionMode.AUTO)
    elif command == "approve":
        pending = control.pending_approvals
        if pending:
            await control.approve(pending[0].request_id)
    elif command == "deny":
        pending = control.pending_approvals
        if pending:
            await control.deny(pending[0].request_id, "not this time")
    else:
        await asyncio.sleep(0.01)


def check_invariants(rig: Rig) -> None:
    """Everything that must be true between any two commands."""

    control = rig.controller
    if control.state is RunState.PAUSED:
        assert control.nonquiescent == 0, (
            f"PAUSED with {control.nonquiescent} non-quiescent agent(s): "
            f"{control.blocking_agents()}"
        )
    for agent_id, state in control.agents.items():
        assert_transcript_valid(state.messages, agent_id=agent_id)


async def drive(commands: list[str], workspace: Path) -> None:
    """Start a run, apply the commands, then release everything and finish."""

    handler = QueueApprovalHandler()
    rig = build_rig(
        script(),
        workspace=workspace,
        mode=PermissionMode.MANUAL,
        handler=handler,
        spec=SPEC,
    )
    try:
        await rig.controller.start()
        check_invariants(rig)

        for command in commands:
            await apply(rig, command)
            check_invariants(rig)

        # Release everything the sequence may have left holding the run up, then
        # require it to finish. A deadlock shows up here as a TimeoutError.
        await rig.controller.set_permission_mode(PermissionMode.AUTO)
        await rig.controller.resume()
        # The script is padded, so this should never fire; if it does, the
        # sequence consumed more turns than the script has and the run failing is
        # the script's fault, not the controller's.
        with contextlib.suppress(ScriptExhausted):
            await rig.controller.wait(timeout=10.0)

        assert rig.controller.state in TERMINAL_STATES
        assert rig.controller.nonquiescent == 0
        for agent_id, state in rig.controller.agents.items():
            assert_transcript_valid(state.messages, agent_id=agent_id)
            assert open_call_ids(state.messages) == [], (
                f"agent {agent_id} finished with unanswered tool calls"
            )
    finally:
        if rig.controller.state not in TERMINAL_STATES:
            await rig.controller.cancel("property teardown")
        await rig.aclose()


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(st.sampled_from(COMMANDS), min_size=1, max_size=8))
def test_no_command_sequence_deadlocks_or_breaks_an_invariant(commands: list[str]) -> None:
    """The property. Owns its own loop, so nothing here depends on plugin ordering."""

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(drive(commands, Path(tmp)))


# ---------------------------------------------------------------------------
# The specific sequences worth naming
# ---------------------------------------------------------------------------


async def _run(commands: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        await drive(commands, Path(tmp))


async def test_pause_then_interrupt_then_resume() -> None:
    """The sequence a user actually performs: stop, redirect, continue."""

    await _run(["pause", "tick", "inject", "resume"])


async def test_a_mode_switch_while_paused_with_a_pending_approval() -> None:
    """Two features that both touch WAITING_APPROVAL, in the order that races."""

    await _run(["tick", "tick", "pause", "auto", "resume"])


async def test_repeated_pauses_and_resumes() -> None:
    """Idempotence under repetition, which is what a held-down key produces."""

    await _run(["pause", "pause", "resume", "resume", "pause", "resume"])


async def test_hard_pause_then_deny_then_resume() -> None:
    """A cancelled step and a refused approval in the same run."""

    await _run(["tick", "pause_hard", "deny", "resume", "tick"])
