"""What the three CLIs share: startup, config merging, and API-key resolution (D6).

Small surface, and every one of these is a thing that fails silently if it is wrong: a
flag that was not typed overwriting a configured value, a `.env` the parser cannot
read because it has spaces around the `=`, a temp-file sweep that never runs.

The `.env` shape matters here specifically. The key in this repository is written
`OPENROUTER_API_KEY = "sk-..."`, with spaces and quotes, and a strict `KEY=value`
parser finds nothing and reports "no API key" while the key is sitting in the file.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest

from workflows.cli_support import (
    API_KEY_ENV,
    CONFIG_ENV,
    CliError,
    dotenv_candidates,
    load_config_file,
    merge_config,
    prepare_process,
    read_dotenv,
    require_api_key,
    resolve_api_key,
)

# ---------------------------------------------------------------------------
# .env and the key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("KEY=value", "value"),
        ("KEY = value", "value"),
        ('KEY = "value"', "value"),
        ("KEY='value'", "value"),
        ("  KEY =  value  ", "value"),
    ],
)
def test_dotenv_tolerates_the_shapes_a_hand_edited_file_has(
    tmp_path: Path, line: str, expected: str
) -> None:
    """The file in this repo has spaces around the `=`; a strict parser finds nothing."""

    env = tmp_path / ".env"
    env.write_text(f"# a comment\n\n{line}", encoding="utf-8")  # no trailing newline
    assert read_dotenv(env) == {"KEY": expected}


def test_a_missing_dotenv_is_empty_not_an_error(tmp_path: Path) -> None:
    assert read_dotenv(tmp_path / "nope.env") == {}


def test_the_environment_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key exported for one command must not be shadowed by a stale file."""

    (tmp_path / ".env").write_text(f'{API_KEY_ENV} = "from-file"', encoding="utf-8")
    monkeypatch.setenv(API_KEY_ENV, "from-env")
    assert resolve_api_key(start=tmp_path) == "from-env"


def test_the_key_is_found_in_a_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`azc` is run from a subdirectory of the repository as often as from its root."""

    monkeypatch.delenv(API_KEY_ENV, raising=False)
    (tmp_path / ".env").write_text(f'{API_KEY_ENV} = "from-parent"', encoding="utf-8")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert resolve_api_key(start=deep) == "from-parent"
    # Written back into the environment, because that is where `OpenRouterProvider`
    # looks -- passing it down through `build(config)` would put a secret in the
    # session document. `monkeypatch` restores the variable at teardown.
    assert os.environ[API_KEY_ENV] == "from-parent"


def test_the_candidate_list_walks_up_to_the_root(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    candidates = dotenv_candidates(deep)
    assert candidates[0] == deep / ".env"
    assert tmp_path / ".env" in candidates


def test_a_missing_key_names_the_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The error a user sees has to say what to do about it."""

    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(CliError) as caught:
        require_api_key(start=tmp_path)
    assert API_KEY_ENV in str(caught.value)
    assert ".env" in str(caught.value)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_an_unset_flag_never_overwrites_a_configured_value() -> None:
    """Why every CLI option defaults to `None` rather than to its real default."""

    base = {"model": "from-file", "tools": ["grep"]}
    assert merge_config(base, {"model": None, "tools": ()}) == base
    assert merge_config(base, {"model": "typed"})["model"] == "typed"
    # A tuple from a repeated option becomes a list, which is what pydantic wants.
    assert merge_config(base, {"tools": ("a", "b")})["tools"] == ["a", "b"]


def test_a_config_file_can_come_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cfg.json"
    path.write_text('{"model": "x"}', encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV, str(path))
    assert load_config_file(None) == {"model": "x"}


def test_no_config_at_all_is_an_empty_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    assert load_config_file(None) == {}


@pytest.mark.parametrize(
    ("content", "fragment"),
    [("{not json", "not valid JSON"), ("[1, 2]", "must contain a JSON object")],
)
def test_a_broken_config_is_a_message_not_a_traceback(
    tmp_path: Path, content: str, fragment: str
) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CliError) as caught:
        load_config_file(str(path))
    assert fragment in str(caught.value)


def test_a_missing_config_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(CliError) as caught:
        load_config_file(str(tmp_path / "nope.json"))
    assert "nope.json" in str(caught.value)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def test_startup_freezes_the_heap_and_sweeps_orphaned_temp_files(tmp_path: Path) -> None:
    """M4's largest latency risk and M3's uncalled sweep, both handled in one call.

    `gc.freeze()` moves everything already allocated into a permanent generation, so
    the count of frozen objects is how the freeze is observable at all.
    """

    session_dir = tmp_path / "run"
    session_dir.mkdir()
    (session_dir / ".session.json.a1b2.tmp").write_text("orphan", encoding="utf-8")
    (session_dir / "session.json").write_text("{}", encoding="utf-8")

    gc.unfreeze()
    assert gc.get_freeze_count() == 0
    try:
        swept = prepare_process(session_dir)
        assert swept == 1
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()

    assert not list(session_dir.glob("*.tmp"))
    assert (session_dir / "session.json").exists()


def test_startup_with_no_session_directory_is_a_no_op(tmp_path: Path) -> None:
    """A run with nowhere to checkpoint still freezes, and sweeps nothing."""

    try:
        assert prepare_process(None) == 0
    finally:
        gc.unfreeze()
