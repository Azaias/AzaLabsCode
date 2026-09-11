"""Shared fixtures and the project-wide event-loop assertion.

The loop-policy check is not defensive noise. `WindowsSelectorEventLoopPolicy`
makes `asyncio.create_subprocess_shell` raise `NotImplementedError`, which would
take out the `shell` tool at M1 and read as a tool bug rather than a policy bug
(spec delta 22). Pinning it here means the whole suite fails loudly and once, at
collection time, rather than in one subprocess test much later.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from azalabscode.sync import assert_subprocess_capable_loop_policy

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    """Fail the run immediately if the event-loop policy cannot spawn subprocesses."""

    assert_subprocess_capable_loop_policy()


@pytest.fixture
def fixtures() -> Path:
    """The recorded-fixture directory."""

    return FIXTURES


@pytest.fixture
def sse() -> object:
    """Read a recorded SSE body by name."""

    def _read(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return _read


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the models cache at a temp directory so tests never touch the real one."""

    cache = tmp_path / "cache"
    monkeypatch.setenv("AZALABSCODE_CACHE_DIR", str(cache))
    return cache


@pytest.fixture
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any ambient OpenRouter key so a test cannot accidentally hit the API."""

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read a `.env` file into a dict without adding a dependency.

    Tolerates the shapes a hand-edited file actually has: spaces around the `=`,
    quoted values, comments, a missing trailing newline.
    """

    env_path = path or Path(__file__).resolve().parent.parent / ".env"
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


@pytest.fixture(scope="session")
def openrouter_api_key() -> str:
    """The real API key from the environment or `.env`, or skip the test."""

    key = os.environ.get("OPENROUTER_API_KEY") or load_dotenv().get("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("no OPENROUTER_API_KEY in the environment or .env")
    return key


# ---------------------------------------------------------------------------
# Tool-layer fixtures (M1)
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty workspace root for a tool test.

    Resolved, because `ToolContext` resolves its root and on Windows `tmp_path` can
    be a short-name path that does not compare equal to its resolved form. A
    containment check against the unresolved root would then reject every path
    inside it.
    """

    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def tool_ctx(workspace: Path):
    """A `ToolContext` rooted at the temp workspace."""

    from azalabscode.tools import ToolContext

    return ToolContext(workspace_root=workspace)
