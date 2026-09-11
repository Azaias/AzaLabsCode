"""Shared acceptance helpers: a sample repository, and the §9.3 event-log scan.

Spec 9.3 defines "pass" for the coding-agent tasks as three properties of the *event
log*, not of the answer text:

1. the task completes using only built-in tools;
2. no tool error of kind `internal`;
3. no `shell` invocation of `cat`, `sed -i`, `find` or `grep` where a built-in exists.

`scan_run` is that check, written against the event stream so the same function works
over a live run (M6, scripted) and over a recorded JSONL log (M7, real model). It is
here rather than in a test file because M7 has to run the identical check.

The third property needs care in both directions. A shell command is a *string*, so
`grep` matches inside `ripgrep`, inside a path, and inside `git log --grep`; and on
Windows the equivalents are `type`, `Get-Content`, `Select-String` and `Get-ChildItem`,
which spec 9.3 does not name because it did not consider PowerShell. The matcher
therefore tokenises rather than substring-matches, and covers both platforms' names --
a check that misses the Windows form would pass on this machine for the wrong reason.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from azalabscode import (
    Event,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallRequested,
    ToolErrorKind,
)

BANNED_SHELL_COMMANDS = {
    # POSIX, named by spec 9.3.
    "cat": "read_file",
    "find": "glob",
    "grep": "grep",
    "rg": "grep",
    "ripgrep": "grep",
    "sed": "edit_file",
    "head": "read_file",
    "tail": "read_file",
    "ls": "glob",
    # PowerShell, which spec 9.3 did not consider and which is the default shell here.
    "type": "read_file",
    "get-content": "read_file",
    "gc": "read_file",
    "select-string": "grep",
    "sls": "grep",
    "get-childitem": "glob",
    "gci": "glob",
    "dir": "glob",
}
"""Command name -> the built-in that should have been used instead."""

COMMAND_SEPARATORS = re.compile(r"[|;&]+|\|\||&&")
"""Where one shell command ends and the next begins, roughly. Good enough to find the
head of each segment, which is all the check needs."""


@dataclass
class Violation:
    """One thing spec 9.3 says must not happen."""

    kind: str
    """`internal_error`, `shell_instead_of_builtin` or `unknown_tool`."""
    detail: str
    call_id: str = ""

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclass
class RunScan:
    """What a run did, as spec 9.3 measures it."""

    tools_used: list[str] = field(default_factory=list)
    shell_commands: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the run passes spec 9.3's criteria."""

        return not self.violations

    def report(self) -> str:
        """Every violation, one per line. What an assertion message shows."""

        return "\n".join(str(violation) for violation in self.violations) or "clean"


def shell_heads(command: str) -> list[str]:
    """The command name at the head of each segment of a shell command line.

    `shlex` with `posix=False` keeps Windows paths intact; a command it cannot parse
    (an unbalanced quote) falls back to whitespace splitting rather than raising,
    because a scan that crashes on a weird command line tells you nothing.
    """

    heads: list[str] = []
    for segment in COMMAND_SEPARATORS.split(command):
        text = segment.strip()
        if not text:
            continue
        try:
            parts = shlex.split(text, posix=False)
        except ValueError:  # pragma: no cover - unbalanced quotes
            parts = text.split()
        if not parts:
            continue
        head = parts[0].strip("'\"")
        # A path: only the executable name matters (`/usr/bin/grep`, `C:\...\rg.exe`).
        head = re.split(r"[\\/]", head)[-1]
        if head.endswith(".exe"):
            head = head[:-4]
        heads.append(head.lower())
    return heads


def scan_run(events: Iterable[Event], *, allowed_tools: Sequence[str] | None = None) -> RunScan:
    """Apply spec 9.3's pass criteria to one run's events."""

    scan = RunScan()
    allowed = set(allowed_tools) if allowed_tools is not None else None
    for event in events:
        if isinstance(event, ToolCallRequested):
            scan.tools_used.append(event.tool)
            if allowed is not None and event.tool not in allowed:
                scan.violations.append(
                    Violation("unknown_tool", f"{event.tool} is not a built-in", event.call_id)
                )
            if event.tool == "shell":
                command = str(event.params.get("command", ""))
                scan.shell_commands.append(command)
                for head in shell_heads(command):
                    builtin = BANNED_SHELL_COMMANDS.get(head)
                    if builtin is not None:
                        scan.violations.append(
                            Violation(
                                "shell_instead_of_builtin",
                                f"shell ran {head!r}; the {builtin} tool exists",
                                event.call_id,
                            )
                        )
        elif isinstance(event, ToolCallFailed):
            if event.error.kind is ToolErrorKind.INTERNAL:
                scan.violations.append(
                    Violation("internal_error", event.error.message, event.call_id)
                )
        elif isinstance(event, ToolCallCompleted):
            error = event.result.error
            if error is not None and error.kind is ToolErrorKind.INTERNAL:
                scan.violations.append(Violation("internal_error", error.message, event.call_id))
    return scan


# ---------------------------------------------------------------------------
# The sample repository the scripted tasks run against
# ---------------------------------------------------------------------------

CLI_PY = '''"""A tiny CLI, for the acceptance tasks to change."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(args.path)


if __name__ == "__main__":
    main()
'''

README_MD = """# sample

A tiny sample project.

## Usage

    python cli.py PATH
"""

LIB_PY = '''"""Two functions, one of them on its way out."""


def deprecated_fn(value):
    """The old one."""

    return value * 2


def new_fn(value):
    """The new one."""

    return value * 2
'''

USE_A_PY = """from lib import deprecated_fn


def total(values):
    return sum(deprecated_fn(value) for value in values)
"""

USE_B_PY = """import lib


def double(value):
    return lib.deprecated_fn(value)
"""

TEST_LIB_PY = """from lib import new_fn


def test_new_fn():
    assert new_fn(2) == 4
"""


def sample_repo(root: Path) -> Path:
    """Write the sample project the §9.3 tasks operate on. Returns `root`."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "cli.py").write_text(CLI_PY, encoding="utf-8")
    (root / "README.md").write_text(README_MD, encoding="utf-8")
    (root / "lib.py").write_text(LIB_PY, encoding="utf-8")
    (root / "use_a.py").write_text(USE_A_PY, encoding="utf-8")
    (root / "use_b.py").write_text(USE_B_PY, encoding="utf-8")
    (root / "test_lib.py").write_text(TEST_LIB_PY, encoding="utf-8")
    return root


__all__ = [
    "BANNED_SHELL_COMMANDS",
    "CLI_PY",
    "LIB_PY",
    "README_MD",
    "RunScan",
    "Violation",
    "sample_repo",
    "scan_run",
    "shell_heads",
]
