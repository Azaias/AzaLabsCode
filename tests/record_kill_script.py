"""Regenerate `tests/fixtures/kill_script.json`, the kill test's scripted provider.

    .venv/Scripts/python -m tests.record_kill_script

Then **read the diff** before committing it, the same rule as the tool-schema
snapshot. A script whose keys silently changed is a kill test that stopped testing
what it claims to.

Why a recorder rather than a hand-written file: `match="by_request_hash"` keys each
turn on `ModelRequest.fingerprint()`, and there is no way to write that down by hand.
The resumed process rebuilds a transcript containing an `interrupted` tool result the
baseline never had, so it issues a request the baseline never issued, and that
request needs its own key. Working the hash out on paper is not a thing anyone should
do twice.

The recorder gets the keys by **running the real subprocess phases** with a recording
wrapper in front of the provider (`tests.resumable.RECORD_ENV`), under
`match="by_index"` -- which works precisely because the interrupted step was a *tool*
call, so `model_call_seq` is exactly where the resumed process should start. The
committed script then runs by hash. If the two ever disagree, the test raises
`ScriptExhausted` and fails loudly rather than diverging quietly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from azalabscode.providers.testing import Script, ScriptedToolCall, ScriptedTurn
from tests.kill_child import CRASH_EXIT_CODE, make_config, paths
from tests.resumable import RECORD_ENV

FIXTURE = Path(__file__).parent / "fixtures" / "kill_script.json"

SLOW_CALL_ID = "call_slow_one"
FINAL_TEXT = "the slow thing reported in"
"""What the run's final output is, in both the uninterrupted and the resumed case.

R-C-12's "same final output" is only checkable if the scripted answer after an
`interrupted` tool result is the same as the answer after a successful one. That is a
property of this script, and it is deliberate: the requirement is about the harness
reaching the same completion, not about a model being indifferent to its inputs.
"""


def bootstrap_turns() -> list[ScriptedTurn]:
    """The two turns, unkeyed, for the recording pass under `by_index`."""

    return [
        ScriptedTurn(
            tool_calls=[
                ScriptedToolCall(call_id=SLOW_CALL_ID, name="slow", arguments={"label": "one"})
            ],
            finish_reason="tool_calls",
        ),
        ScriptedTurn(text=FINAL_TEXT, finish_reason="stop"),
    ]


def _run(phase: str, directory: Path, record: Path, *, expect: int = 0) -> None:
    """Run one child phase, recording its request fingerprints."""

    result = subprocess.run(
        [sys.executable, "-m", "tests.kill_child", phase, str(directory), "by_index"],
        cwd=Path(__file__).resolve().parent.parent,
        env={**_env(), RECORD_ENV: str(record)},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != expect:
        raise SystemExit(
            f"phase {phase} exited {result.returncode}, expected {expect}\n"
            f"{result.stdout}\n{result.stderr}"
        )


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _keys(record: Path) -> list[tuple[str, int]]:
    """`(fingerprint, turn index)` for every request the phase issued."""

    if not record.exists():
        return []
    out: list[tuple[str, int]] = []
    for line in record.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        out.append((entry["key"], entry["index"]))
    return out


def record() -> Script:
    """Drive every phase and assemble the keyed script."""

    turns = bootstrap_turns()
    collected: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="killrec-") as raw:
        root = Path(raw)
        for name, phases in (
            ("baseline", [("baseline", 0)]),
            ("crash", [("crash", CRASH_EXIT_CODE), ("resume", 0)]),
            ("pause", [("pause_save", CRASH_EXIT_CODE), ("resume", 0)]),
        ):
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            Script(match="by_index", turns=turns).save(paths(directory)["script"])
            make_config(directory)  # side-effect free; asserts the config still validates
            record_path = directory / "keys.jsonl"
            for phase, expect in phases:
                _run(phase, directory, record_path, expect=expect)
            for key, index in _keys(record_path):
                collected[key] = index

    keyed: list[ScriptedTurn] = []
    for key, index in sorted(collected.items(), key=lambda item: (item[1], item[0])):
        turn = turns[index].model_copy(deep=True)
        turn.key = key
        keyed.append(turn)
    return Script(match="by_request_hash", turns=keyed)


def main() -> int:
    """Write the fixture and say what went into it."""

    script = record()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    script.save(FIXTURE)
    print(f"wrote {FIXTURE} with {len(script.turns)} keyed turn(s):")
    for turn in script.turns:
        shape = f"{len(turn.tool_calls)} tool call(s)" if turn.tool_calls else f"text {turn.text!r}"
        print(f"  {turn.key}  ->  {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
