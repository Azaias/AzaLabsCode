"""What the three reference CLIs share: startup, config resolution, and the API key.

Plan decision D6 puts packaging, config and API-key resolution in M6 because that is
what makes success criterion 5 measurable by *using* the harness rather than by
reading an event log. All three of those are here rather than in the coding agent, so
that fusion and the inspector start the same way and a fix lands once.

Two of the startup steps are not obvious and both come from earlier milestones:

* **`gc.freeze()` before the run, after the imports.** M4 measured a generation-2
  collection at 40--90 ms, which is most of R-U-4's 100 ms budget in one hit, and a
  long coding session is exactly the heap that makes it expensive. Freezing moves
  everything already allocated -- the interpreter, the imports, the widget classes --
  into a permanent generation the collector stops walking. One line, and it is the
  largest single latency risk in the UI.
* **Sweeping the session directory.** `sweep_temp_files` deletes `.tmp` files left by
  a process killed between `mkstemp` and `os.replace`. It has been written and tested
  since M3 with no caller; a CLI that is about to write into that directory is the
  natural one.

`.env` parsing is deliberately tolerant of the shapes a hand-edited file has --
spaces around the `=`, quotes, comments, no trailing newline -- because the file in
this repo has spaces around the `=` and a strict parser would silently find no key.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from azalabscode import HarnessError, sweep_temp_files

API_KEY_ENV = "OPENROUTER_API_KEY"
SEARCH_KEY_ENV = "SERPER_API_KEY"
ENV_FILE = ".env"
CONFIG_ENV = "AZALABSCODE_CONFIG"
"""A JSON file of defaults, so daily use does not mean retyping four flags."""


class CliError(HarnessError):
    """A message for the user, not a traceback. The CLI prints it and exits 2."""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def read_dotenv(path: Path | None = None) -> dict[str, str]:
    """Parse a `.env` file into a dict. Missing file is an empty dict, not an error."""

    env_path = path or Path.cwd() / ENV_FILE
    values: dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def dotenv_candidates(start: Path | None = None) -> list[Path]:
    """`.env` files to try: the working directory, then each parent up to the root."""

    here = (start or Path.cwd()).resolve()
    return [parent / ENV_FILE for parent in [here, *here.parents]]


def resolve_api_key(name: str = API_KEY_ENV, *, start: Path | None = None) -> str:
    """Find an API key in the environment, then in the nearest `.env`.

    The environment wins: a key exported for one command must not be shadowed by a
    stale file. The found value is written back into `os.environ` because that is
    where `OpenRouterProvider` and `serper_from_env` look, and passing it down by
    hand through `build(config)` would put a secret in the session document.
    """

    existing = os.environ.get(name)
    if existing:
        return existing
    for candidate in dotenv_candidates(start):
        value = read_dotenv(candidate).get(name)
        if value:
            os.environ[name] = value
            return value
    return ""


def require_api_key(name: str = API_KEY_ENV, *, start: Path | None = None) -> str:
    """`resolve_api_key`, but a `CliError` naming the fix when there is none."""

    key = resolve_api_key(name, start=start)
    if not key:
        raise CliError(
            f"no {name}. Set it in the environment, or put `{name}=sk-...` in a "
            f"{ENV_FILE} file in this directory or any parent."
        )
    return key


def prepare_process(session_dir: Path | None = None) -> int:
    """Freeze the heap and sweep orphaned temp files. Returns how many were swept.

    Call once, at the top of a CLI command, after the imports and before the run.
    """

    gc.collect()
    gc.freeze()
    return sweep_temp_files(session_dir) if session_dir is not None else 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config_file(path: str | Path | None) -> dict[str, Any]:
    """Read a JSON config file. `None` falls back to `$AZALABSCODE_CONFIG`."""

    chosen = path if path is not None else os.environ.get(CONFIG_ENV)
    if not chosen:
        return {}
    config_path = Path(chosen)
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(f"cannot read config {config_path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise CliError(f"{config_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CliError(f"{config_path} must contain a JSON object")
    return data


def merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay command-line values on file values, dropping the ones not given.

    `None` means "not given on the command line", which is why every CLI option that
    can also live in the config file defaults to `None` rather than to its real
    default. A flag defaulting to its real value would silently overwrite the file.
    """

    merged = dict(base)
    for key, value in overrides.items():
        if value is None or value == ():
            continue
        merged[key] = list(value) if isinstance(value, tuple) else value
    return merged


__all__ = [
    "API_KEY_ENV",
    "CONFIG_ENV",
    "ENV_FILE",
    "SEARCH_KEY_ENV",
    "CliError",
    "dotenv_candidates",
    "load_config_file",
    "merge_config",
    "prepare_process",
    "read_dotenv",
    "require_api_key",
    "resolve_api_key",
]
