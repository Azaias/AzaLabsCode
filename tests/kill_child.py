"""The child interpreter for the R-C-12 kill test. Never imported by the test itself.

Four phases, each a separate process, each driving the same `tests.resumable`
workflow through the same public `Controller` API. Run as:

    python -m tests.kill_child <phase> <directory>

* **`baseline`** -- run to completion, uninterrupted. Writes `baseline.json`.
* **`crash`** -- run until the slow tool announces itself, then `os._exit(9)` from
  inside the child (spec delta 18: portable, no `SIGKILL` on Windows, and no
  parent-side race about *when* the kill lands). What is left on disk is whatever
  autosave wrote at the last safe point, which is exactly what a real crash leaves.
* **`pause_save`** -- run, `pause()`, `save()`, then `os._exit(9)`. This is R-C-12's
  own wording: "start a run under `FakeProvider`, pause, save, terminate the
  interpreter".
* **`resume`** -- `Controller.load()` the session, `resume()`, run to completion.
  Writes `resumed.json`.

The comparison the test makes is on the final output and the node-output map, never
on the event log: sequence numbers and timings legitimately differ between a run that
happened once and a run that happened in two processes (delta 18).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from azalabscode.control import Controller, WorkflowRef
from azalabscode.control.session import Session
from azalabscode.ids import MAIN_AGENT
from azalabscode.permissions import PermissionMode
from azalabscode.providers.testing import MatchMode
from azalabscode.runstate import RunState
from azalabscode.sync import assert_subprocess_capable_loop_policy
from tests.resumable import Config, build

IMPORT_PATH = "tests.resumable:build"
CONFIG_TYPE = "tests.resumable:Config"
CRASH_EXIT_CODE = 9
"""What `os._exit(9)` leaves behind. The test asserts on it, so a phase that
exited cleanly cannot be mistaken for one that was killed."""

WATCH_INTERVAL_S = 0.005


def paths(directory: Path) -> dict[str, Path]:
    """Every path the phases share, derived from one directory."""

    return {
        "root": directory,
        "session": directory / "session" / "session.json",
        "session_dir": directory / "session",
        "markers": directory / "markers",
        "workspace": directory / "ws",
        "script": directory / "kill_script.json",
        "baseline": directory / "baseline.json",
        "resumed": directory / "resumed.json",
    }


def make_config(directory: Path, *, match: MatchMode = "by_request_hash") -> Config:
    """The workflow config, identical in every phase.

    Identical is the point: a resumed process must rebuild the same workflow from the
    config in the file, and a config that differed between phases would hide a bug in
    exactly the mechanism the test exists to check.
    """

    p = paths(directory)
    return Config(
        script_path=str(p["script"]),
        workspace=str(p["workspace"]),
        markers=str(p["markers"]),
        match=match,
        mode=PermissionMode.AUTO,
    )


def _controller(directory: Path, config: Config) -> Controller:
    """A fresh controller wired to the session directory, autosaving every safe point."""

    p = paths(directory)
    return Controller(
        build(config),
        permission_mode=config.mode,
        session_dir=p["session_dir"],
        workflow=WorkflowRef.of(
            IMPORT_PATH,
            config.model_dump(mode="json"),
            config_type=CONFIG_TYPE,
        ),
    )


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


async def _watch_for(marker: Path, *, timeout_s: float = 30.0) -> bool:
    """Wait for a file to appear. A marker, never a sleep -- delta 18 and M2's clock.

    `asyncio.sleep` is not a stopwatch on Windows (~15.6 ms loop resolution), so a
    rendezvous built on one is a flaky test waiting to happen. A file either exists
    or it does not.
    """

    deadline = asyncio.get_running_loop().time() + timeout_s
    while not marker.exists():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(WATCH_INTERVAL_S)
    return True


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


async def run_baseline(directory: Path, *, match: MatchMode = "by_request_hash") -> None:
    """The uninterrupted run R-C-12 compares against."""

    p = paths(directory)
    config = make_config(directory, match=match)
    p["markers"].mkdir(parents=True, exist_ok=True)
    # Released up front: the slow tool is the kill rendezvous, not a delay, and the
    # baseline has nothing to rendezvous with.
    (p["markers"] / "release").write_text("1", encoding="utf-8")

    controller = _controller(directory, config)
    result = await controller.run(timeout=60.0)
    _write(p["baseline"], {"result": result, "state": str(controller.state)})


async def run_crash(directory: Path, *, match: MatchMode = "by_request_hash") -> None:
    """Die mid-tool. The checkpoint on disk is whatever autosave last wrote."""

    p = paths(directory)
    config = make_config(directory, match=match)
    controller = _controller(directory, config)

    await controller.start()
    if not await _watch_for(p["markers"] / "started-one"):  # pragma: no cover - a hung child
        print("the slow tool never started", file=sys.stderr)
        os._exit(2)
    # Flush what the process has already written, then stop existing. No unwinding,
    # no `finally`, no chance for the harness to tidy up after itself -- which is the
    # only honest simulation of a process death.
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(CRASH_EXIT_CODE)


async def run_pause_save(directory: Path, *, match: MatchMode = "by_request_hash") -> None:
    """R-C-12 verbatim: start, pause, save, terminate the interpreter."""

    p = paths(directory)
    config = make_config(directory, match=match)
    controller = _controller(directory, config)

    await controller.start()
    await controller.pause()
    await controller.wait_for_state(RunState.PAUSED, timeout=30.0)
    await controller.save(p["session"], timeout=30.0)
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(CRASH_EXIT_CODE)


async def run_resume(directory: Path) -> None:
    """Load the session, resume, and run to completion in a fresh interpreter."""

    p = paths(directory)
    p["markers"].mkdir(parents=True, exist_ok=True)
    # Whatever the previous process was waiting for, this one is allowed to finish.
    (p["markers"] / "release").write_text("1", encoding="utf-8")

    controller = await Controller.load(p["session"])
    report = controller.resume_report
    await controller.resume()
    result = await controller.wait(timeout=60.0)

    state = controller.agent(MAIN_AGENT)
    _write(
        p["resumed"],
        {
            "result": result,
            "state": str(controller.state),
            "interrupted_calls": sorted(report.interrupted_calls) if report else [],
            "resumed_delegates": sorted(report.resumed_delegates) if report else [],
            "dropped_model_calls": len(report.dropped_model_calls) if report else 0,
            "message_roles": [m.role for m in state.messages] if state else [],
            "tool_error_kinds": _error_kinds(state),
        },
    )


def _error_kinds(state: Any) -> list[str]:
    """Every tool-result error kind on the transcript, in order.

    R-C-12 asks for "exactly one `ToolError(kind='interrupted')`", and this is what
    the parent counts.
    """

    if state is None:
        return []
    out: list[str] = []
    for message in state.messages:
        result = getattr(message, "result", None)
        if result is not None and result.error is not None:
            out.append(str(result.error.kind))
    return out


def inspect_session(directory: Path) -> dict[str, Any]:
    """What the crashed process left on disk, for the parent to assert against."""

    session = Session.load(paths(directory)["session"])
    return {
        "run_state": str(session.run_state),
        "resume_state": str(session.resume_state),
        "agents": sorted(session.agents),
        "completed_nodes": session.completed_nodes(),
        "open_call_ids": {a: list(s.open_call_ids) for a, s in session.agents.items()},
        "inflight": [s.kind for s in session.inflight],
    }


PHASES = {
    "baseline": run_baseline,
    "crash": run_crash,
    "pause_save": run_pause_save,
    "resume": run_resume,
}


def main(argv: list[str]) -> int:
    """Dispatch one phase. Returns the exit code for phases that return at all."""

    assert_subprocess_capable_loop_policy()
    if len(argv) < 3 or argv[1] not in PHASES:
        print(
            f"usage: python -m tests.kill_child [{'|'.join(PHASES)}] <directory>", file=sys.stderr
        )
        return 2
    phase = PHASES[argv[1]]
    directory = Path(argv[2])
    kwargs: dict[str, Any] = {}
    if len(argv) > 3 and argv[1] != "resume":
        kwargs["match"] = argv[3]
    asyncio.run(phase(directory, **kwargs))
    return 0


if __name__ == "__main__":  # pragma: no cover - the child's entry point
    raise SystemExit(main(sys.argv))
