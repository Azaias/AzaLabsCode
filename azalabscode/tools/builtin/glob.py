"""`glob`: find files by name pattern, newest first.

`rg --files` is used when ripgrep is on PATH, because it already implements
`.gitignore` semantics correctly -- nested ignore files, negation, `.git/info/exclude`
-- and reimplementing that is a project, not a function. The pure-Python fallback
covers the common case: the repository-root `.gitignore` plus a small set of
directories nobody wants to walk.

Results are sorted by mtime descending. In a codebase the file you just touched is
almost always the file you are asking about.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from azalabscode.permissions import ApprovalPolicy
from azalabscode.toolio import (
    NO_RETRY,
    RetryPolicy,
    ToolDisplay,
    ToolError,
    ToolErrorKind,
    ToolResult,
)
from azalabscode.tools.base import Tool
from azalabscode.tools.context import ToolContext, ToolPathError

DESCRIPTION = """\
Find files whose path matches a glob pattern, most recently modified first.

Patterns are matched against the path relative to the search root, using `/` as the \
separator on every platform. `**` crosses directories, `*` does not.

  *.py                 Python files at the top of the search root
  **/*.py              Python files anywhere below it
  src/**/test_*.py     test files anywhere under src/

Files ignored by .gitignore are skipped, as are .git, node_modules, __pycache__ and \
virtualenvs. Directories are never returned, only files.

Use this to locate files by name. To search inside files, use grep. If you know the \
exact path already, just read_file it -- a glob to confirm a file exists is a wasted \
turn.

At most `limit` paths come back; if there were more you are told so, and a narrower \
pattern is a better answer than a bigger limit.\
"""

DEFAULT_LIMIT = 500
RG_TIMEOUT_S = 8.0

_ALWAYS_SKIP = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".eggs",
        ".next",
        "target",
        ".gradle",
        ".import_linter_cache",
        ".hypothesis",
    }
)


class GlobParams(BaseModel):
    """Parameters for `glob`."""

    model_config = {"extra": "forbid"}

    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'src/**/test_*.py'.")
    path: str = Field(
        default=".", description="Directory to search under, relative to the workspace root."
    )
    limit: int = Field(
        default=DEFAULT_LIMIT, ge=1, le=10_000, description="Maximum paths to return."
    )


class GlobTool(Tool):
    """List files matching a glob pattern."""

    name: ClassVar[str] = "glob"
    description: ClassVar[str] = DESCRIPTION
    Params: ClassVar[type[BaseModel]] = GlobParams

    approval: ApprovalPolicy = "never"
    timeout: float = 10.0
    retry: RetryPolicy = NO_RETRY
    concurrency_safe: ClassVar[bool] = True
    read_only: ClassVar[bool] = True

    async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
        """Check the search root exists and is a directory."""

        assert isinstance(params, GlobParams)
        try:
            root = ctx.resolve_path(params.path)
        except ToolPathError as exc:
            return exc.as_tool_error()
        if not root.exists():
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND, message=f"no such directory: {params.path}"
            )
        if not root.is_dir():
            return ToolError(
                kind=ToolErrorKind.INVALID_PARAMS,
                message=f"{params.path} is a file, not a directory",
            )
        return None

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Walk the tree and match."""

        assert isinstance(params, GlobParams)
        root = ctx.resolve_path(params.path)
        started = time.monotonic()

        paths, backend = await _list_files(root)
        pattern = params.pattern.replace("\\", "/")
        matched = [p for p in paths if _matches(p, root, pattern)]

        def mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        matched.sort(key=mtime, reverse=True)
        total = len(matched)
        shown = matched[: params.limit]
        rendered = [ctx.display_path(p) for p in shown]

        if not rendered:
            body = (
                f"No files match {params.pattern!r} under {ctx.display_path(root)}. "
                f"Check the pattern -- '*.py' only matches the top level, '**/*.py' "
                f"matches every directory."
            )
        else:
            body = "\n".join(rendered)
            if total > len(shown):
                body += (
                    f"\n\n[{total} files matched, showing the {len(shown)} most recently "
                    f"modified. Narrow the pattern to see the rest.]"
                )

        return ToolResult.ok_text(
            body,
            display=ToolDisplay(
                kind="paths",
                data={"paths": rendered, "total": total, "truncated": total > len(shown)},
            ),
            meta={
                "backend": backend,
                "total": total,
                "returned": len(shown),
                "elapsed_ms": (time.monotonic() - started) * 1000.0,
            },
        )


@lru_cache(maxsize=256)
def compile_glob(pattern: str) -> re.Pattern[str]:
    """Translate a path glob into a regex with real `**` semantics.

    `fnmatch` is not path-aware: it turns both `*` and `**` into `.*`, so `*.py`
    wrongly matches `src/app.py`, and `src/**/*.py` wrongly *fails* to match
    `src/app.py` because it insists on a directory between them. Neither is what a
    model means, and ripgrep -- which we defer to when it is installed -- implements
    the path-aware version. The two backends have to agree.

    - `**/` matches zero or more directories, so `src/**/*.py` covers `src/app.py`.
    - `**` on its own matches anything, separators included.
    - `*` and `?` stop at a separator.
    - `[...]` is passed through as a character class.
    """

    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif ch == "*":
            out.append("[^/]*")
            i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        elif ch == "[":
            close = pattern.find("]", i + 1)
            if close == -1:
                out.append(re.escape(ch))
                i += 1
            else:
                body = pattern[i + 1 : close]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = close + 1
        else:
            out.append(re.escape(ch))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _matches(path: Path, root: Path, pattern: str) -> bool:
    """Match a path against a glob pattern relative to the search root.

    A pattern without a separator also matches the basename anywhere in the tree,
    which is what `glob("*.py")` is almost always meant to do.
    """

    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - path outside root
        return False

    compiled = compile_glob(pattern)
    if compiled.match(rel):
        return True
    if "/" not in pattern:
        return bool(compiled.match(path.name))
    return False


async def _list_files(root: Path) -> tuple[list[Path], str]:
    """Every non-ignored file under `root`, using `rg --files` when available."""

    if shutil.which("rg"):
        try:
            paths = await _rg_files(root)
            return paths, "rg"
        except (OSError, TimeoutError, subprocess.SubprocessError):
            pass
    return list(_walk(root)), "python"


def rg_common_args() -> list[str]:
    """Flags that make ripgrep agree with the pure-Python fallback.

    - `--no-require-git`: by default ripgrep only honours `.gitignore` *inside* a git
      repository. A workspace that is not a repo would then see ignored files from
      one backend and not the other.
    - `--hidden` plus an explicit `.git` exclusion: dotfiles are usually the point
      (`.github/`, `.env.example`), but `.git` itself never is.
    - The skip list, so `node_modules` and `__pycache__` are excluded even when no
      `.gitignore` mentions them. `rg` does not skip them on its own.
    """

    args = ["--hidden", "--no-require-git", "--glob", "!.git/**"]
    for name in sorted(_ALWAYS_SKIP):
        args.extend(["--glob", f"!**/{name}/**"])
    return args


async def _rg_files(root: Path) -> list[Path]:
    """`rg --files`: ripgrep's own gitignore handling, which is the correct one."""

    proc = await asyncio.create_subprocess_exec(
        shutil.which("rg") or "rg",
        "--files",
        *rg_common_args(),
        str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), RG_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    text = out.decode("utf-8", "replace")
    return [Path(line) for line in text.splitlines() if line.strip()]


def _walk(root: Path) -> Iterator[Path]:
    """Pure-Python fallback walk, honouring the root `.gitignore` and the skip list."""

    ignores = _load_gitignore(root)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _ALWAYS_SKIP and not _ignored(here / d, root, ignores, is_dir=True)
        ]
        for name in filenames:
            candidate = here / name
            if not _ignored(candidate, root, ignores, is_dir=False):
                yield candidate


def _load_gitignore(root: Path) -> list[str]:
    """Patterns from the root `.gitignore`, comments and negations dropped.

    Negation (`!pattern`) is deliberately not implemented: getting it half right is
    worse than not having it, and `rg` -- which does implement it -- is the path
    taken whenever it is installed.
    """

    path = root / ".gitignore"
    if not path.exists():
        return []
    out: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            out.append(line.rstrip("/"))
    except OSError:  # pragma: no cover
        return []
    return out


def _ignored(path: Path, root: Path, patterns: list[str], *, is_dir: bool) -> bool:
    if path.name in _ALWAYS_SKIP:
        return True
    if not patterns:
        return False
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover
        return False
    for pattern in patterns:
        compiled = compile_glob(pattern)
        if compiled.match(rel) or compiled.match(path.name):
            return True
        # A gitignore entry also covers everything beneath it: `build/` hides
        # `build/out.py`, which is what most entries in a real .gitignore mean.
        if compile_glob(f"{pattern}/**").match(rel) or rel.startswith(f"{pattern}/"):
            return True
    return False


__all__ = ["DESCRIPTION", "GlobParams", "GlobTool", "compile_glob", "rg_common_args"]
