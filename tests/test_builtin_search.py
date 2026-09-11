"""`glob` and `grep` against real temp workspaces, on both backends.

Every search test runs twice where it matters: once with `rg` as found on PATH, and
once with `rg` forced absent so the pure-Python fallback is exercised. The two
backends have to agree on the answer, and only running the one this machine happens
to have would let the other rot.

The behaviour worth naming: **a timeout reports "timed out", never "no matches."** A
model told the second concludes the code it is looking for does not exist, and acts
on that.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from azalabscode.toolio import ToolErrorKind
from azalabscode.tools import GlobTool, GrepTool, ToolContext
from azalabscode.tools.builtin.glob import GlobParams
from azalabscode.tools.builtin.grep import GrepParams, _parse_rg_line

GLOB = GlobTool()
GREP = GrepTool()


@pytest.fixture(params=["rg", "python"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the test on ripgrep and on the pure-Python fallback.

    Forcing `shutil.which` to miss `rg` is what makes the fallback a tested path
    rather than dead code on any machine that has ripgrep installed.
    """

    if request.param == "python":
        import shutil

        real = shutil.which
        monkeypatch.setattr(
            shutil, "which", lambda name, *a, **k: None if name == "rg" else real(name, *a, **k)
        )
    return request.param


@pytest.fixture
def tree(workspace: Path) -> Path:
    """A small source tree with something to find in it."""

    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text(
        "import os\n\n\ndef main():\n    return os.getcwd()\n", encoding="utf-8"
    )
    (workspace / "src" / "util.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_app.py").write_text(
        "def test_main():\n    assert main()\n", encoding="utf-8"
    )
    (workspace / "README.md").write_text("# Project\n\nSee src/app.py.\n", encoding="utf-8")
    return workspace


async def run_glob(ctx: ToolContext, **kwargs: object):
    params = GlobParams(**kwargs)  # type: ignore[arg-type]
    error = await GLOB.validate_params(params, ctx)
    if error is not None:
        from azalabscode.toolio import ToolResult

        return ToolResult.failure(error.kind, error.message)
    return await GLOB.run(params, ctx)


async def run_grep(ctx: ToolContext, **kwargs: object):
    params = GrepParams(**kwargs)  # type: ignore[arg-type]
    error = await GREP.validate_params(params, ctx)
    if error is not None:
        from azalabscode.toolio import ToolResult

        return ToolResult.failure(error.kind, error.message)
    return await GREP.run(params, ctx)


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


async def test_glob_finds_files_by_recursive_pattern(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    result = await run_glob(tool_ctx, pattern="**/*.py")

    assert result.ok is True
    assert "src/app.py" in result.text
    assert "src/util.py" in result.text
    assert "tests/test_app.py" in result.text
    assert "README.md" not in result.text


async def test_a_single_star_does_not_cross_directories(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    result = await run_glob(tool_ctx, pattern="*.md")
    assert "README.md" in result.text
    assert "app.py" not in result.text


async def test_a_scoped_pattern_restricts_the_subtree(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    result = await run_glob(tool_ctx, pattern="src/**/*.py")
    assert "src/app.py" in result.text
    assert "tests/test_app.py" not in result.text


async def test_glob_never_returns_directories(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    result = await run_glob(tool_ctx, pattern="*")
    for line in result.display.data["paths"]:  # type: ignore[union-attr]
        assert not (tree / line).is_dir()


async def test_no_matches_explains_the_pattern_rules(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    """ "No files match" plus a hint beats a bare empty list: the single-star mistake
    is the most common one and the fix is one character."""

    result = await run_glob(tool_ctx, pattern="**/*.rs")
    assert result.ok is True
    assert "No files match" in result.text
    assert "'**/*.py'" in result.text


async def test_results_are_newest_first(tool_ctx: ToolContext, tree: Path, backend: str) -> None:
    import os
    import time

    newest = tree / "src" / "util.py"
    os.utime(newest, (time.time() + 100, time.time() + 100))

    result = await run_glob(tool_ctx, pattern="**/*.py")
    assert result.display is not None
    assert result.display.data["paths"][0] == "src/util.py"


async def test_the_limit_is_explicit_about_truncation(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    result = await run_glob(tool_ctx, pattern="**/*.py", limit=1)
    assert "3 files matched" in result.text
    assert result.display is not None
    assert result.display.data["truncated"] is True


async def test_gitignored_files_are_skipped(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    (tree / ".gitignore").write_text("secret.py\nbuild/\n", encoding="utf-8")
    (tree / "secret.py").write_text("x = 1\n", encoding="utf-8")
    (tree / "build").mkdir()
    (tree / "build" / "out.py").write_text("y = 1\n", encoding="utf-8")

    result = await run_glob(tool_ctx, pattern="**/*.py")

    assert "secret.py" not in result.text
    assert "build/out.py" not in result.text
    assert "src/app.py" in result.text


async def test_noisy_directories_are_never_walked(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    (tree / "node_modules").mkdir()
    (tree / "node_modules" / "junk.py").write_text("x\n", encoding="utf-8")
    (tree / "__pycache__").mkdir()
    (tree / "__pycache__" / "cached.py").write_text("x\n", encoding="utf-8")

    result = await run_glob(tool_ctx, pattern="**/*.py")
    assert "node_modules" not in result.text
    assert "__pycache__" not in result.text


async def test_glob_of_a_missing_directory_is_not_found(tool_ctx: ToolContext) -> None:
    result = await run_glob(tool_ctx, pattern="*", path="nope")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.NOT_FOUND


async def test_glob_of_a_file_is_an_invalid_param(tool_ctx: ToolContext, tree: Path) -> None:
    result = await run_glob(tool_ctx, pattern="*", path="README.md")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS


async def test_glob_outside_the_workspace_is_refused(tool_ctx: ToolContext, tmp_path: Path) -> None:
    result = await run_glob(tool_ctx, pattern="*", path=str(tmp_path))
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.PERMISSION


async def test_a_windows_style_backslash_pattern_is_accepted(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    """A model on Windows will write `src\\*.py` sooner or later."""

    result = await run_glob(tool_ctx, pattern="src\\*.py")
    assert "src/app.py" in result.text


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


async def test_grep_returns_path_line_text(tool_ctx: ToolContext, tree: Path, backend: str) -> None:
    result = await run_grep(tool_ctx, pattern="def main")

    assert result.ok is True
    assert re.search(r"src/app\.py:4: *def main", result.text)


async def test_grep_reports_no_matches_as_a_completed_search(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    result = await run_grep(tool_ctx, pattern="nonexistent_symbol_xyz")

    assert result.ok is True
    assert "No matches" in result.text
    assert "The search completed" in result.text


async def test_a_timeout_is_reported_as_a_timeout_not_as_no_matches(
    tool_ctx: ToolContext, tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinction the whole design turns on: a model told "no matches" after a
    timeout concludes the code does not exist."""

    import azalabscode.tools.builtin.grep as grep_mod

    async def slow(*args: object, **kwargs: object):
        raise TimeoutError("too slow")

    monkeypatch.setattr(grep_mod, "_rg_search", slow)
    monkeypatch.setattr(grep_mod, "_python_search", slow)
    monkeypatch.setattr(grep_mod.shutil, "which", lambda name: "rg" if name == "rg" else None)

    result = await run_grep(tool_ctx, pattern="anything")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.TIMEOUT
    assert "not the same as finding nothing" in result.error.message


async def test_files_with_matches_returns_paths_only(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    """ "Which files mention this" is a different question from "show me the lines",
    and answering the first with the second costs thousands of tokens."""

    result = await run_grep(tool_ctx, pattern="def ", output_mode="files_with_matches")

    assert "src/app.py" in result.text
    assert ":" not in result.text.split("\n")[0]
    assert result.display is not None
    assert result.display.data["mode"] == "files_with_matches"


async def test_count_mode_reports_matches_per_file(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    result = await run_grep(tool_ctx, pattern="def ", output_mode="count")
    assert re.search(r"src/app\.py: \d", result.text)


async def test_a_glob_filter_narrows_the_search(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    result = await run_grep(tool_ctx, pattern="src", glob="**/*.md")

    # Assert on the *searched* path, not on the line text: README.md's content
    # mentions src/app.py, and a substring check would pass on the wrong evidence.
    assert result.display is not None
    searched = {row.split(":")[0] for row in result.display.data["rows"]}
    assert searched == {"README.md"}


async def test_case_insensitive_matching(tool_ctx: ToolContext, tree: Path, backend: str) -> None:
    result = await run_grep(tool_ctx, pattern="DEF MAIN", case_insensitive=True)
    assert "app.py" in result.text

    result = await run_grep(tool_ctx, pattern="DEF MAIN")
    assert "No matches" in result.text


async def test_context_lines_are_returned(tool_ctx: ToolContext, tree: Path, backend: str) -> None:
    result = await run_grep(tool_ctx, pattern="return os", context=1, path="src/app.py")
    assert "def main" in result.text


async def test_truncation_names_the_offset_to_continue_from(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    """Silent truncation makes a model believe it has seen every call site."""

    result = await run_grep(tool_ctx, pattern="def ", limit=1)

    assert "truncated" in result.text
    assert "offset=1" in result.text
    assert result.display is not None
    assert result.display.data["truncated"] is True


async def test_pagination_with_offset_returns_the_next_page(
    tool_ctx: ToolContext, tree: Path, backend: str
) -> None:
    first = await run_grep(tool_ctx, pattern="def ", limit=1, offset=0)
    second = await run_grep(tool_ctx, pattern="def ", limit=1, offset=1)
    assert first.display is not None and second.display is not None
    assert first.display.data["rows"] != second.display.data["rows"]


async def test_an_invalid_regex_is_caught_before_the_search_runs(
    tool_ctx: ToolContext, tree: Path
) -> None:
    """Caught inside `rg` it is an exit code and a stderr string the model has to
    interpret; caught here it is one line naming the problem."""

    result = await run_grep(tool_ctx, pattern="(unclosed")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.INVALID_PARAMS
    assert "invalid regular expression" in result.error.message


async def test_grep_of_a_missing_path_is_not_found(tool_ctx: ToolContext) -> None:
    result = await run_grep(tool_ctx, pattern="x", path="nope")
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.NOT_FOUND


async def test_grep_outside_the_workspace_is_refused(tool_ctx: ToolContext, tmp_path: Path) -> None:
    result = await run_grep(tool_ctx, pattern="x", path=str(tmp_path))
    assert result.error is not None
    assert result.error.kind is ToolErrorKind.PERMISSION


async def test_grep_skips_binary_files_in_the_python_fallback(
    tool_ctx: ToolContext, tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    (tree / "blob.dat").write_bytes(b"\x00\x01def main\x00")

    result = await run_grep(tool_ctx, pattern="def main")
    assert "blob.dat" not in result.text


async def test_grep_of_a_single_file_works(tool_ctx: ToolContext, tree: Path, backend: str) -> None:
    result = await run_grep(tool_ctx, pattern="helper", path="src/util.py")
    assert "util.py" in result.text
    assert "app.py" not in result.text


async def test_both_backends_agree_on_the_same_query(
    tool_ctx: ToolContext, tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback that quietly disagrees with `rg` is a fallback nobody notices is
    wrong until the day ripgrep is missing."""

    import shutil

    with_rg = await run_grep(tool_ctx, pattern="def ", output_mode="files_with_matches")

    real = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda name, *a, **k: None if name == "rg" else real(name, *a, **k)
    )
    without_rg = await run_grep(tool_ctx, pattern="def ", output_mode="files_with_matches")

    assert sorted(with_rg.display.data["rows"]) == sorted(  # type: ignore[union-attr]
        without_rg.display.data["rows"]  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------
# ripgrep output parsing
# ---------------------------------------------------------------------------


def test_a_windows_drive_letter_is_not_mistaken_for_the_line_separator() -> None:
    """`C:\\src\\app.py:12:text` has three colons and only the last two are separators."""

    match = _parse_rg_line("C:\\src\\app.py:12:    def main():")
    assert match is not None
    assert match.path == "C:\\src\\app.py"
    assert match.line == 12
    assert match.text == "    def main():"


def test_a_posix_path_parses() -> None:
    match = _parse_rg_line("/home/x/app.py:7:body")
    assert match is not None
    assert match.path == "/home/x/app.py"
    assert match.line == 7


def test_an_unparseable_line_is_dropped_rather_than_crashing() -> None:
    assert _parse_rg_line("--") is None
    assert _parse_rg_line("") is None


# ---------------------------------------------------------------------------
# Glob translation
# ---------------------------------------------------------------------------


def test_a_single_star_stops_at_a_separator() -> None:
    """`fnmatch` gets this wrong: it turns `*` into `.*`, so `*.py` would match
    `src/app.py` and every top-level glob would silently be recursive."""

    from azalabscode.tools.builtin.glob import compile_glob

    assert compile_glob("*.py").match("app.py")
    assert not compile_glob("*.py").match("src/app.py")


def test_double_star_slash_matches_zero_directories() -> None:
    """The other half of the fnmatch bug: `src/**/*.py` must cover `src/app.py`, not
    insist on a directory in between."""

    from azalabscode.tools.builtin.glob import compile_glob

    pattern = compile_glob("src/**/*.py")
    assert pattern.match("src/app.py")
    assert pattern.match("src/a/b/app.py")
    assert not pattern.match("tests/app.py")


def test_a_bare_double_star_crosses_separators() -> None:
    from azalabscode.tools.builtin.glob import compile_glob

    assert compile_glob("**/*.py").match("a/b/c.py")
    assert compile_glob("**/*.py").match("c.py")


def test_a_question_mark_matches_one_non_separator_character() -> None:
    from azalabscode.tools.builtin.glob import compile_glob

    assert compile_glob("a?.py").match("ab.py")
    assert not compile_glob("a?.py").match("a/b.py")


def test_a_character_class_is_passed_through() -> None:
    from azalabscode.tools.builtin.glob import compile_glob

    assert compile_glob("test_[ab].py").match("test_a.py")
    assert not compile_glob("test_[ab].py").match("test_c.py")


def test_a_dot_is_a_literal_not_a_wildcard() -> None:
    from azalabscode.tools.builtin.glob import compile_glob

    assert not compile_glob("a.py").match("axpy")


def test_the_pattern_must_match_the_whole_path() -> None:
    """A prefix match would make `*.py` match `app.pyc`."""

    from azalabscode.tools.builtin.glob import compile_glob

    assert not compile_glob("*.py").match("app.pyc")
