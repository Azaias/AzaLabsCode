"""R-C-12: the subprocess kill test. The one that decides whether M3 is real.

Every other M3 test runs inside one interpreter, where `save` and `load` share a
heap and a bug can hide behind a live object neither of them should be able to see.
This one does not: a child process runs the workflow, dies with `os._exit(9)`, and a
*second* child rebuilds the whole run from `(import_path, config)` and a JSON file.

Spec delta 18 fixes three things about how it is done, and each removes a way for the
test to lie:

* **The kill is `os._exit(9)` from inside the child**, after a synchronization
  marker. Portable -- there is no `SIGKILL` on Windows -- and there is no
  parent-side race about whether the kill landed before or after the tool started.
* **The rendezvous is a file, not a sleep.** The slow tool creates `started-one` and
  then blocks on a `release` marker. `asyncio.sleep` is not a stopwatch on Windows.
* **The comparison is the final output and the node-output map, not the event log.**
  Sequence numbers and timings differ legitimately between one process and two.

Two kill shapes are covered. `crash` dies mid-tool, which is what R-C-13 is about:
the checkpoint on disk is whatever autosave wrote at the last safe point, exactly
what a real crash leaves. `pause_save` is R-C-12's own wording -- start, pause, save,
terminate -- and exercises the explicit `save()` path instead.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.kill_child import CRASH_EXIT_CODE, inspect_session, paths
from tests.record_kill_script import FINAL_TEXT

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "kill_script.json"
CHILD_TIMEOUT_S = 120.0

pytestmark = pytest.mark.slow


def run_phase(phase: str, directory: Path, *, expect: int = 0) -> subprocess.CompletedProcess[str]:
    """Run one child phase and assert its exit code.

    The exit code is an assertion, not bookkeeping: a `crash` phase that exited 0
    finished normally, and every downstream assertion about it would be vacuous.
    """

    result = subprocess.run(
        [sys.executable, "-m", "tests.kill_child", phase, str(directory)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=CHILD_TIMEOUT_S,
    )
    assert result.returncode == expect, (
        f"phase {phase!r} exited {result.returncode}, expected {expect}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


@pytest.fixture
def kill_dir(tmp_path: Path) -> Path:
    """A directory with the committed script in it, ready for a phase to run."""

    directory = tmp_path / "kill"
    directory.mkdir()
    shutil.copyfile(FIXTURE, paths(directory)["script"])
    return directory


def read(path: Path) -> dict:
    """One phase's JSON output."""

    return json.loads(path.read_text(encoding="utf-8"))


def lines(path: Path) -> list[str]:
    """A marker log's lines, or an empty list if the file was never created."""

    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# The baseline
# ---------------------------------------------------------------------------


def test_the_uninterrupted_run_is_what_everything_is_compared_against(kill_dir: Path) -> None:
    """A baseline that does not hold means every comparison below proves nothing."""

    run_phase("baseline", kill_dir)
    p = paths(kill_dir)
    baseline = read(p["baseline"])

    assert baseline["result"]["final"] == f"summary of {FINAL_TEXT}"
    assert baseline["result"]["nodes"]["agent"] == FINAL_TEXT
    assert baseline["state"] == "completed"
    assert lines(p["markers"] / "starts.log") == ["one"]
    assert lines(p["markers"] / "executions.log") == ["prepare", "agent", "summarize"]


# ---------------------------------------------------------------------------
# R-C-12: crash mid-tool, then resume
# ---------------------------------------------------------------------------


def test_the_kill_test_reaches_the_same_final_output(tmp_path: Path) -> None:
    """R-C-12 whole: baseline, crash, resume, and the two runs agree.

    Separate directories, because the baseline creates the `release` marker the slow
    tool waits on -- a shared directory would let the crash phase's tool finish
    before the kill and the test would prove nothing.
    """

    base = tmp_path / "baseline"
    killed = tmp_path / "killed"
    for directory in (base, killed):
        directory.mkdir()
        shutil.copyfile(FIXTURE, paths(directory)["script"])

    run_phase("baseline", base)
    run_phase("crash", killed, expect=CRASH_EXIT_CODE)
    run_phase("resume", killed)

    baseline = read(paths(base)["baseline"])
    resumed = read(paths(killed)["resumed"])

    assert resumed["state"] == "completed"
    assert resumed["result"]["final"] == baseline["result"]["final"]
    assert resumed["result"]["nodes"] == baseline["result"]["nodes"]


def test_the_slow_tool_starts_exactly_once_across_both_processes(kill_dir: Path) -> None:
    """R-C-13 across a real process death: never re-executed, not even once.

    The count is in a file rather than in memory precisely because the process that
    started the tool is gone. One line means the first process started it and the
    second one did not.
    """

    run_phase("crash", kill_dir, expect=CRASH_EXIT_CODE)
    assert lines(paths(kill_dir)["markers"] / "starts.log") == ["one"]

    run_phase("resume", kill_dir)
    assert lines(paths(kill_dir)["markers"] / "starts.log") == ["one"]


def test_the_interrupted_call_is_answered_exactly_once(kill_dir: Path) -> None:
    """Exactly one `ToolError(kind="interrupted")`, and a valid transcript (R-C-13)."""

    run_phase("crash", kill_dir, expect=CRASH_EXIT_CODE)
    run_phase("resume", kill_dir)

    resumed = read(paths(kill_dir)["resumed"])
    assert resumed["tool_error_kinds"] == ["interrupted"]
    assert len(resumed["interrupted_calls"]) == 1
    # system, user, assistant(tool call), tool result, assistant(final)
    assert resumed["message_roles"] == ["system", "user", "assistant", "tool", "assistant"]


def test_zero_completed_nodes_are_re_executed(kill_dir: Path) -> None:
    """R-W-6, across the process boundary.

    `prepare` completed before the crash and `summarize` after the resume, so each
    runs exactly once across two processes. `agent` runs twice, and must: it never
    completed, so it is the incomplete node that restarts from its checkpoint.
    """

    run_phase("crash", kill_dir, expect=CRASH_EXIT_CODE)
    after_crash = lines(paths(kill_dir)["markers"] / "executions.log")
    assert after_crash == ["prepare", "agent"]

    saved = inspect_session(kill_dir)
    assert saved["completed_nodes"] == ["prepare"]

    run_phase("resume", kill_dir)
    executed = lines(paths(kill_dir)["markers"] / "executions.log")
    assert executed.count("prepare") == 1, "a completed node was executed again"
    assert executed.count("summarize") == 1
    assert executed.count("agent") == 2, "the incomplete node must restart"


def test_the_crashed_process_left_a_loadable_checkpoint(kill_dir: Path) -> None:
    """What autosave wrote at the last safe point is a complete, valid session."""

    run_phase("crash", kill_dir, expect=CRASH_EXIT_CODE)
    saved = inspect_session(kill_dir)

    assert saved["agents"] == ["main"]
    assert saved["run_state"] == "running"
    assert saved["resume_state"] == "running"
    # The safe point that wrote this was `after_model_call`: the tool call is on the
    # transcript with no result, and the batch step had not been registered yet.
    assert saved["open_call_ids"]["main"] == ["call_slow_one"]


def test_no_temp_file_survives_the_kill(kill_dir: Path) -> None:
    """A process killed mid-write must not leave a `.tmp` beside the session.

    Not guaranteed by the retry loop -- a process can die between `mkstemp` and
    `os.replace` -- but it is guaranteed here, because `os._exit` lands at a moment
    when no write is in progress. If this ever fails it means a checkpoint write is
    running when it should not be.
    """

    run_phase("crash", kill_dir, expect=CRASH_EXIT_CODE)
    session_dir = paths(kill_dir)["session_dir"]
    assert [p.name for p in session_dir.iterdir()] == ["session.json"]


# ---------------------------------------------------------------------------
# R-C-12 verbatim: pause, save, terminate
# ---------------------------------------------------------------------------


def test_pause_save_kill_load_resume_completes(tmp_path: Path) -> None:
    """Spec R-C-12's own sequence, and the same final output.

    Different from the crash case in one way that matters: nothing was in flight, so
    the resumed run re-issues the *first* model call and the slow tool runs for the
    first time in the second process. The final output is still the baseline's.
    """

    base = tmp_path / "baseline"
    paused = tmp_path / "paused"
    for directory in (base, paused):
        directory.mkdir()
        shutil.copyfile(FIXTURE, paths(directory)["script"])

    run_phase("baseline", base)
    run_phase("pause_save", paused, expect=CRASH_EXIT_CODE)

    saved = inspect_session(paused)
    assert saved["run_state"] == "paused"
    assert saved["open_call_ids"]["main"] == []

    run_phase("resume", paused)
    baseline = read(paths(base)["baseline"])
    resumed = read(paths(paused)["resumed"])

    assert resumed["result"] == baseline["result"]
    assert resumed["tool_error_kinds"] == [], "nothing was in flight, so nothing was interrupted"
    assert lines(paths(paused)["markers"] / "starts.log") == ["one"]


def test_the_script_fixture_matches_what_the_run_actually_requests(kill_dir: Path) -> None:
    """A key that drifted makes `FakeProvider` raise `ScriptExhausted`, loudly.

    This is the guard on the recorder: `tests/record_kill_script.py` generates the
    fixture by running these same phases, so a change to the system prompt, the task
    or the toolset invalidates the keys. The failure is a non-zero exit from the
    child with `ScriptExhausted` in its stderr, not a quietly different answer.
    """

    result = run_phase("baseline", kill_dir)
    assert "ScriptExhausted" not in result.stderr
    script = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert script["match"] == "by_request_hash"
    assert all(turn["key"] for turn in script["turns"]), "every turn must be keyed"
