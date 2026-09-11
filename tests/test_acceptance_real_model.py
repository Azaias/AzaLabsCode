"""M7's exit test: spec §9.3 tasks 1 and 2, run against a real model.

`tests/test_acceptance_coding_agent.py` runs the same two tasks against a
`FakeProvider`, which proves the machinery. It cannot prove success criterion 5 --
"the built-in tools are the ones the model reaches for" -- because a scripted model
reaches for exactly what the script says. That needs a real model, and a real model
cannot run in CI: it costs money, needs a key, and does not repeat.

So the run happens once, out of band, and its complete event log is committed.
`scripts/run_acceptance.py` is the runner; this file is the assertion over what it
recorded. Each task has three fixtures under `tests/fixtures/acceptance/`:

* `<task>.jsonl` -- every event, unfiltered, replayed here through
  `JsonlRecorder.read`. This is what `scan_run` measures, and it is the same
  function the scripted tests use, on the same criteria.
* `<task>.workspace.json` -- the workspace afterwards, because the workspace was a
  temp directory and is the only evidence the task was actually *done*.
* `<task>.meta.json` -- model, timestamp, exit code, the agent's final answer.

**The assertions are properties, not diffs.** A real model does not produce the calls
a fixture expects: it may read a file twice, use `glob` before `grep`, or write the
README sentence in its own words. Asserting on exact text would make this file a
record of one sampling rather than of a capability. So: the flag exists somewhere in
`cli.py` and `cli.py` still parses; no call site of `deprecated_fn` survives while its
definition does; the scan is clean.

Regenerate with:

    .venv/Scripts/python scripts/run_acceptance.py

Read the diff before committing it -- a changed log is a changed run, not a changed
formatting.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from azalabscode import (
    Event,
    JsonlRecorder,
    ModelCallCompleted,
    RunState,
    RunStateChanged,
    ToolCallRequested,
)
from tests.acceptance import scan_run, shell_heads
from workflows.coding_agent.workflow import ALL_TOOLS

FIXTURES = Path(__file__).parent / "fixtures" / "acceptance"
TASKS = ("task1", "task2")


def load_events(task: str) -> list[Event]:
    """The recorded log, parsed back into typed events."""

    return JsonlRecorder.read(FIXTURES / f"{task}.jsonl")


def load_workspace(task: str) -> dict[str, str]:
    """The recorded workspace: relative path -> file content."""

    return json.loads((FIXTURES / f"{task}.workspace.json").read_text(encoding="utf-8"))


def load_meta(task: str) -> dict[str, object]:
    return json.loads((FIXTURES / f"{task}.meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The fixtures themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_the_fixture_is_a_complete_run_against_a_real_model(task: str) -> None:
    """A log recorded from a scripted provider would satisfy the scan for free."""

    meta = load_meta(task)
    assert meta["exit_code"] == 0, meta
    assert isinstance(meta["model"], str) and "/" in meta["model"], "a real model id"
    assert meta["answer"], "the agent produced a final answer"

    events = load_events(task)
    assert len(events) > 100, "a real coding turn is not five events"

    # Every model call reports usage a real provider filled in. A `FakeProvider` run
    # would show zeroes here, so this is what pins the fixture to a live call.
    completions = [event for event in events if isinstance(event, ModelCallCompleted)]
    assert completions, "no model call in the log"
    usages = [event.usage for event in completions]
    assert all(usage is not None and usage.total_tokens > 0 for usage in usages)
    assert sum(usage.total_tokens for usage in usages if usage is not None) > 1000

    states = [event.new for event in events if isinstance(event, RunStateChanged)]
    assert states[-1] is RunState.COMPLETED, states


@pytest.mark.parametrize("task", TASKS)
def test_the_run_passes_the_spec_9_3_scan(task: str) -> None:
    """The exit test proper: only built-ins, no `internal` error, no shell doing a
    built-in's job. `ALL_TOOLS` is the superset -- `web_search` drops out of a
    registry with no `SERPER_API_KEY`, so the *built* set is machine-dependent and
    the constant is not."""

    scan = scan_run(load_events(task), allowed_tools=ALL_TOOLS)
    assert scan.ok, scan.report()
    assert scan.tools_used, "a run that called no tool proves nothing"


@pytest.mark.parametrize("task", TASKS)
def test_no_tool_call_in_the_log_is_outside_the_workspace(task: str) -> None:
    """R-T-7 held under a model that was never told where the boundary was."""

    for event in load_events(task):
        if isinstance(event, ToolCallRequested):
            path = str(event.params.get("path", "") or event.params.get("file_path", ""))
            assert ".." not in path, f"{event.tool} escaped: {path}"


def test_the_scan_over_a_committed_log_is_not_vacuous() -> None:
    """If `scan_run` cannot see a violation in a recorded log, `ok` means nothing.

    Injected into the real task-2 log, which is the one that used `shell` at all.
    """

    events = load_events("task2")
    requests = [event for event in events if isinstance(event, ToolCallRequested)]
    shell_call = next(event for event in requests if event.tool == "shell")
    tampered = shell_call.model_copy(update={"params": {"command": "find . -name '*.py'"}})
    scan = scan_run([*events, tampered], allowed_tools=ALL_TOOLS)
    assert not scan.ok
    assert "shell ran 'find'" in scan.report()
    assert shell_heads("find . -name '*.py'") == ["find"]


# ---------------------------------------------------------------------------
# Task 1: "Add a --verbose flag to cli.py and update the README"
# ---------------------------------------------------------------------------


def test_task_one_added_a_working_flag_and_documented_it() -> None:
    files = load_workspace("task1")

    cli = files["cli.py"]
    assert "--verbose" in cli
    tree = ast.parse(cli)  # it still parses; the agent did not leave a broken file
    # The flag is registered with argparse, not merely mentioned in a comment.
    argument_strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "--verbose" in argument_strings
    assert "verbose" in cli.lower().split("def main")[-1], "the flag is used, not just parsed"

    readme = files["README.md"]
    assert "--verbose" in readme
    assert "## Usage" in readme, "the section the task named survived"

    # Nothing else in the sample repo was touched.
    assert "def deprecated_fn" in files["lib.py"]
    assert set(files) == {"README.md", "cli.py", "lib.py", "test_lib.py", "use_a.py", "use_b.py"}


# ---------------------------------------------------------------------------
# Task 2: "Find every call site of deprecated_fn, replace it, run the tests"
# ---------------------------------------------------------------------------


def test_task_two_replaced_every_call_site_and_left_the_definition() -> None:
    files = load_workspace("task2")

    for name in ("use_a.py", "use_b.py"):
        assert "deprecated_fn" not in files[name], name
        assert "new_fn" in files[name], name
        ast.parse(files[name])

    # The task was about call sites. Deleting the definition would be over-reach.
    assert "def deprecated_fn" in files["lib.py"]
    assert "def new_fn" in files["lib.py"]
    assert files["test_lib.py"].strip(), "the test file survived"


def test_task_two_actually_ran_the_tests() -> None:
    """ "then run the tests" is half the task, and it is the half only `shell` can do."""

    scan = scan_run(load_events("task2"), allowed_tools=ALL_TOOLS)
    assert any("pytest" in command for command in scan.shell_commands), scan.shell_commands
    answer = str(load_meta("task2")["answer"]).lower()
    assert "pass" in answer, "the agent reported the result, as asked"


def test_task_two_found_the_call_sites_with_a_built_in() -> None:
    """The searching half. `grep` or `glob`, not `shell`."""

    tools = [event.tool for event in load_events("task2") if isinstance(event, ToolCallRequested)]
    assert {"grep", "glob"} & set(tools), tools
    assert "edit_file" in tools or "write_file" in tools, tools
