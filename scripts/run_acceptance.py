"""Run the spec §9.3 acceptance tasks against a real model and record the fixtures.

This is M7's exit test made repeatable. Each task runs `azc --headless --auto` over a
fresh copy of `tests.acceptance.sample_repo`, with the event stream written to JSONL,
and records three artifacts under `tests/fixtures/acceptance/`:

* `<task>.jsonl`      -- the complete event log, which is what spec §9.3 is scanned
                         over. Nothing is stripped: R-X-3's claim is that the log is
                         complete, and a filtered log would not test it.
* `<task>.workspace.json` -- every file in the workspace afterwards. The log proves
                         *how* the agent worked; the workspace proves the task was
                         actually done, and it is the only evidence a later CI run
                         has, because the workspace itself is a temp directory.
* `<task>.meta.json`  -- model, timestamp, exit code, wall time, the final answer.

Two things this deliberately does *not* do. It does not run in `manual` mode: a
headless `manual` run installs `StdinApprovalHandler` and blocks on a worker thread
with no visible cause. And it does not assert on exact file text -- a real model does
not produce the tool calls a scripted fixture does, so the checks in
`tests/test_acceptance_real_model.py` are properties ("the flag exists somewhere in
`cli.py`", "no call site of `deprecated_fn` survives"), not diffs.

Usage:

    .venv/Scripts/python scripts/run_acceptance.py            # both tasks
    .venv/Scripts/python scripts/run_acceptance.py task1      # just one
    .venv/Scripts/python scripts/run_acceptance.py --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from azalabscode import JsonlRecorder  # noqa: E402
from tests.acceptance import sample_repo, scan_run  # noqa: E402
from workflows.cli_support import require_api_key  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "acceptance"
WORKDIR = Path(tempfile.gettempdir()) / "azalabscode-acceptance"
"""Outside the repository, deliberately. A sample repo nested inside this one is not
the "small sample repo" spec §9.3 describes: `pytest` walks up to the root
`pyproject.toml`, picks up its `addopts`, and fails in a way that has nothing to do
with the task -- which is what sent the first recorded run hunting for config files
with `find`."""

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


@dataclass(frozen=True)
class Task:
    """One of spec §9.3's acceptance tasks."""

    name: str
    prompt: str


TASKS = {
    "task1": Task(
        "task1",
        "Add a --verbose flag to cli.py that prints the resolved absolute path "
        "when set, and document it in the Usage section of README.md.",
    ),
    "task2": Task(
        "task2",
        "Find every call site of deprecated_fn in this repository and replace it "
        "with new_fn, then run the tests with pytest and report the result.",
    ),
}
"""Spec §9.3 task 3 (`web_fetch` a changelog) is out per plan decision D4: it needs a
stable public URL and the run has no `SERPER_API_KEY`. See progress.md §M7."""


def child_env() -> dict[str, str]:
    """The environment `azc` runs under.

    The interpreter directory goes on the front of `PATH` so that `python` and
    `pytest` inside the agent's `shell` calls are this project's venv. Without it the
    agent finds whatever python is on the system PATH, discovers pytest is missing,
    and spends its turns on `pip install` -- an environment problem being measured as
    a tool-choice one.
    """

    scripts = Path(sys.executable).parent
    env = dict(os.environ)
    env["PATH"] = f"{scripts}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def snapshot(workspace: Path) -> dict[str, str]:
    """Every text file in the workspace, relative path -> content."""

    files: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace).as_posix()
        # The session directory and the event log are recorded separately; the
        # snapshot is what the *task* produced.
        if any(part.startswith((".", "__pycache__")) for part in Path(relative).parts[:-1]):
            continue
        if relative == "events.jsonl":
            continue
        try:
            files[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            files[relative] = "<binary>"
    return files


def run_task(task: Task, model: str, *, timeout: float = 900.0) -> int:
    """Run one task end to end and write its three fixtures. Returns the exit code."""

    workspace = WORKDIR / task.name
    if workspace.exists():
        shutil.rmtree(workspace)
    sample_repo(workspace)
    log = workspace / "events.jsonl"

    command = [
        sys.executable,
        "-m",
        "workflows.coding_agent",
        "--headless",
        "--auto",
        "--model",
        model,
        "--workspace",
        str(workspace),
        "--session-dir",
        str(workspace / ".azalabscode"),
        "--log",
        str(log),
        task.prompt,
    ]
    print(f"[{task.name}] {model}: {task.prompt}", flush=True)
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=child_env(),
    )
    elapsed = time.monotonic() - started
    print(f"[{task.name}] exit={result.returncode} in {elapsed:.1f}s", flush=True)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)

    FIXTURES.mkdir(parents=True, exist_ok=True)
    if log.exists():
        shutil.copyfile(log, FIXTURES / f"{task.name}.jsonl")
        events = JsonlRecorder.read(FIXTURES / f"{task.name}.jsonl")
        scan = scan_run(events)
        print(f"[{task.name}] {len(events)} events, scan: {scan.report()}", flush=True)
        print(f"[{task.name}] tools: {sorted(set(scan.tools_used))}", flush=True)

    (FIXTURES / f"{task.name}.workspace.json").write_text(
        json.dumps(snapshot(workspace), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (FIXTURES / f"{task.name}.meta.json").write_text(
        json.dumps(
            {
                "task": task.name,
                "prompt": task.prompt,
                "model": model,
                "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "exit_code": result.returncode,
                "elapsed_seconds": round(elapsed, 1),
                "answer": result.stdout.strip()[-8000:],
                "platform": sys.platform,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="*", default=[], choices=[*TASKS, []])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    require_api_key(start=ROOT)
    names = args.tasks or list(TASKS)
    failures = 0
    for name in names:
        failures += 1 if run_task(TASKS[name], args.model, timeout=args.timeout) else 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
