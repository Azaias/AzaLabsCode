"""Schema snapshot for every built-in tool (R-T-9).

The point is the *diff*. Every built-in's name, description and JSON-Schema
parameters are rendered and compared to a committed snapshot, so an accidental edit
to a description -- the thing that most directly steers a model, and the thing least
likely to be caught by any other test -- shows up in review as a deliberate change.

Regenerate with `python -m tests.test_tool_schemas` after an intentional edit, then
read the diff before committing it.

`shell` is excluded from the byte-for-byte snapshot: its description is rendered per
platform (spec delta 9), so the snapshot would be machine-specific. It gets its own
structural assertions instead, here and in `test_builtin_shell.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from azalabscode.toolio import ToolSchema
from azalabscode.tools import BUILTIN_NAMES, StaticSearchBackend, Tool, default_registry

SNAPSHOT = Path(__file__).parent / "fixtures" / "tool_schemas.json"

PLATFORM_DEPENDENT = {"shell"}
"""Tools whose description is rendered per platform and cannot be snapshotted."""


def all_builtins() -> list[Tool]:
    """Every built-in, with `web_search` forced on so the snapshot is complete."""

    registry = default_registry(search_backend=StaticSearchBackend([]))
    return list(registry)


def render() -> dict[str, Any]:
    """The snapshot document: every tool's full schema, keyed by name."""

    return {
        tool.name: tool.schema().model_dump(mode="json")
        for tool in all_builtins()
        if tool.name not in PLATFORM_DEPENDENT
    }


def write_snapshot() -> None:
    """Regenerate the committed snapshot."""

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(
        json.dumps(render(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


# ---------------------------------------------------------------------------


def test_the_snapshot_exists() -> None:
    assert SNAPSHOT.exists(), (
        f"missing schema snapshot at {SNAPSHOT}; regenerate with "
        f"`python -m tests.test_tool_schemas`"
    )


def test_every_builtin_schema_matches_the_snapshot() -> None:
    """R-T-9. A failure here is not necessarily a bug -- it means a description or a
    parameter changed, and the diff is the review."""

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    actual = render()

    assert sorted(actual) == sorted(expected), (
        "the set of snapshotted tools changed; regenerate with `python -m tests.test_tool_schemas`"
    )
    for name in sorted(expected):
        assert actual[name] == expected[name], (
            f"the schema for {name!r} changed. If that was deliberate, regenerate "
            f"with `python -m tests.test_tool_schemas` and review the diff."
        )


def test_the_snapshot_covers_every_registered_builtin() -> None:
    snapshotted = set(json.loads(SNAPSHOT.read_text(encoding="utf-8"))) | PLATFORM_DEPENDENT
    assert snapshotted == set(BUILTIN_NAMES)


# ---------------------------------------------------------------------------
# Properties every schema must have, snapshot or not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", all_builtins(), ids=lambda t: t.name)
def test_the_schema_is_a_valid_object_schema(tool: Tool) -> None:
    schema = tool.schema()
    assert isinstance(schema, ToolSchema)
    assert schema.parameters["type"] == "object"
    assert "properties" in schema.parameters


@pytest.mark.parametrize("tool", all_builtins(), ids=lambda t: t.name)
def test_every_parameter_is_documented(tool: Tool) -> None:
    """An undescribed parameter is one the model has to guess at."""

    for name, spec in tool.schema().parameters["properties"].items():
        assert spec.get("description") or spec.get("anyOf") or spec.get("$ref"), (
            f"{tool.name}.{name} has no description"
        )


@pytest.mark.parametrize("tool", all_builtins(), ids=lambda t: t.name)
def test_the_description_is_substantial(tool: Tool) -> None:
    """Spec 7: descriptions say what the tool does, when to use it, when *not* to,
    and the common mistakes. A one-liner does none of that."""

    assert len(tool.description) > 200, f"{tool.name} has a stub description"
    assert tool.description.strip() == tool.description.rstrip(), (
        f"{tool.name}'s description has leading whitespace"
    )


@pytest.mark.parametrize("tool", all_builtins(), ids=lambda t: t.name)
def test_the_schema_is_json_serialisable(tool: Tool) -> None:
    """It goes into an HTTP body verbatim."""

    json.dumps(tool.schema().model_dump(mode="json"))


@pytest.mark.parametrize("tool", all_builtins(), ids=lambda t: t.name)
def test_extra_parameters_are_forbidden(tool: Tool) -> None:
    """A model inventing an argument must be told, not silently ignored."""

    assert tool.Params.model_config.get("extra") == "forbid"


def test_the_registry_builds_every_builtin_in_the_documented_order() -> None:
    names = [t.name for t in all_builtins()]
    assert names == list(BUILTIN_NAMES)


def test_a_narrowed_registry_only_builds_what_was_asked_for() -> None:
    registry = default_registry(include=("read_file", "grep"))
    assert registry.names() == ["read_file", "grep"]


def test_exclusion_removes_a_tool() -> None:
    registry = default_registry(exclude=("shell", "write_file", "edit_file"))
    assert "shell" not in registry.names()
    assert "read_file" in registry.names()


def test_the_read_only_registry_contains_nothing_that_changes_anything() -> None:
    """What a subagent gets in `manual` mode (R-C-7)."""

    from azalabscode.tools import read_only_registry

    for tool in read_only_registry(search_backend=StaticSearchBackend([])):
        assert tool.read_only is True, f"{tool.name} is not read-only"
        assert tool.approval == "never", f"{tool.name} would need approval"


def test_the_parallel_safe_table_matches_spec_7() -> None:
    """Spec 7 fixes which tools are parallel-safe. Delta 7 makes it per-call, but the
    class-level default must still match the table."""

    expected = {
        "read_file": True,
        "glob": True,
        "grep": True,
        "web_fetch": True,
        "web_search": True,
        "delegate": True,
        "write_file": False,
        "edit_file": False,
        "shell": False,
    }
    actual = {tool.name: tool.concurrency_safe for tool in all_builtins()}
    assert actual == expected


def test_the_approval_table_matches_spec_7() -> None:
    expected = {
        "read_file": "never",
        "glob": "never",
        "grep": "never",
        "web_fetch": "never",
        "web_search": "never",
        "delegate": "never",
        "write_file": "always",
        "edit_file": "always",
        "shell": "always",
    }
    actual = {tool.name: tool.approval for tool in all_builtins()}
    assert actual == expected


def test_no_approval_gated_tool_retries() -> None:
    """R-T-4, checked across the whole built-in set rather than one tool at a time."""

    for tool in all_builtins():
        if tool.approval != "never":
            assert tool.retry.attempts == 0, f"{tool.name} retries a gated call"


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    write_snapshot()
    print(f"wrote {SNAPSHOT}", file=sys.stderr)
