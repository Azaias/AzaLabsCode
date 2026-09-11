"""`grep`: regex search across files, via `rg` when it exists and Python when it does not.

Three behaviours are worth stating because getting them wrong is expensive:

- **A timeout reports "timed out", never "no matches."** A search that ran out of
  time and a search that found nothing are opposite facts, and a model told the
  second will conclude the code it is looking for does not exist.
- **Truncation is explicit** and names the offset to continue from. Silent truncation
  makes a model believe it has seen every call site.
- **`output_mode`** (spec delta from `content`/`files_with_matches`/`count`) exists
  because "which files mention this" is a different question from "show me the lines",
  and answering the first with the second costs thousands of tokens.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

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
from azalabscode.tools.builtin.glob import _list_files, _matches, rg_common_args
from azalabscode.tools.context import ToolContext, ToolPathError

DESCRIPTION = """\
Search file contents with a regular expression.

Syntax is Rust regex when ripgrep is installed and Python `re` otherwise; they agree \
on everything except lookaround and backreferences, which ripgrep does not support. \
Escape regex metacharacters if you want a literal match: `foo\\(bar\\)`.

output_mode:
  content            matching lines as `path:line: text` (the default)
  files_with_matches just the paths -- use this first when you want to know where \
something lives, then read the interesting files
  count              matches per file

Narrow the search with `glob` ('*.py', '**/test_*.py') and `path`. Use `context` to \
get surrounding lines, and `case_insensitive` rather than building `[Ff][Oo][Oo]`.

Results stop at `limit` and you are told when there were more, with the offset to \
continue from. A search that times out says so -- it does not report zero matches.

Files ignored by .gitignore are skipped.\
"""

DEFAULT_LIMIT = 200
MAX_MATCH_LINE_CHARS = 500
RG_EXIT_NO_MATCH = 1

type OutputMode = Literal["content", "files_with_matches", "count"]


class GrepParams(BaseModel):
    """Parameters for `grep`."""

    model_config = {"extra": "forbid"}

    pattern: str = Field(description="Regular expression to search for.")
    path: str = Field(
        default=".", description="File or directory to search, relative to the workspace root."
    )
    glob: str | None = Field(
        default=None, description="Only search files matching this glob, e.g. '**/*.py'."
    )
    output_mode: OutputMode = Field(
        default="content", description="'content', 'files_with_matches', or 'count'."
    )
    context: int = Field(default=0, ge=0, le=20, description="Lines of context around each match.")
    case_insensitive: bool = Field(default=False, description="Match without regard to case.")
    limit: int = Field(
        default=DEFAULT_LIMIT, ge=1, le=5_000, description="Maximum results to return."
    )
    offset: int = Field(
        default=0, ge=0, description="Skip this many results; use with limit to paginate."
    )


class GrepTool(Tool):
    """Regex search over file contents."""

    name: ClassVar[str] = "grep"
    description: ClassVar[str] = DESCRIPTION
    Params: ClassVar[type[BaseModel]] = GrepParams

    approval: ApprovalPolicy = "never"
    timeout: float = 30.0
    retry: RetryPolicy = NO_RETRY
    concurrency_safe: ClassVar[bool] = True
    read_only: ClassVar[bool] = True

    async def validate_params(self, params: BaseModel, ctx: ToolContext) -> Any:
        """Compile the pattern and check the search root, before anything runs.

        A bad regex caught here is a one-line error; caught inside `rg` it is an
        exit code and a stderr string the model has to interpret.
        """

        assert isinstance(params, GrepParams)
        try:
            re.compile(params.pattern)
        except re.error as exc:
            return ToolError(
                kind=ToolErrorKind.INVALID_PARAMS,
                message=f"invalid regular expression {params.pattern!r}: {exc}",
            )
        try:
            root = ctx.resolve_path(params.path)
        except ToolPathError as exc:
            return exc.as_tool_error()
        if not root.exists():
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND, message=f"no such file or directory: {params.path}"
            )
        return None

    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Search."""

        assert isinstance(params, GrepParams)
        root = ctx.resolve_path(params.path)
        started = time.monotonic()
        budget = max(1.0, self.timeout - 2.0)

        matches: list[Match]
        backend = "python"
        if shutil.which("rg"):
            try:
                matches = await _rg_search(params, root, limit_s=budget)
                backend = "rg"
            except TimeoutError:
                return self._timeout_result(params, budget, "rg")
            except (OSError, subprocess.SubprocessError, ValueError):
                matches = await _python_search(params, root, deadline=started + budget)
        else:
            try:
                matches = await _python_search(params, root, deadline=started + budget)
            except TimeoutError:
                return self._timeout_result(params, budget, "python")

        return _render(params, matches, ctx, backend, (time.monotonic() - started) * 1000.0)

    def _timeout_result(self, params: GrepParams, budget: float, backend: str) -> ToolResult:
        """A timeout is a distinct outcome from "no matches" and must read as one."""

        return ToolResult.failure(
            ToolErrorKind.TIMEOUT,
            f"the search for {params.pattern!r} did not finish within {budget:g}s. "
            f"This is not the same as finding nothing: narrow it with `path` or "
            f"`glob` and try again.",
            meta={"backend": backend, "timed_out": True},
        )


class Match:
    """One matching line."""

    __slots__ = ("line", "path", "text")

    def __init__(self, path: str, line: int, text: str) -> None:
        self.path = path
        self.line = line
        self.text = text


async def _rg_search(params: GrepParams, root: Path, *, limit_s: float) -> list[Match]:
    """Run ripgrep in `--vimgrep`-ish mode and parse `path:line:text`."""

    argv = [
        shutil.which("rg") or "rg",
        "--line-number",
        "--no-heading",
        "--color=never",
        # Given a single explicit file, ripgrep omits the filename and emits bare
        # `12:text`, which the parser cannot attribute to anything. Forcing it on
        # makes the output shape independent of how many paths were passed.
        "--with-filename",
        *rg_common_args(),
    ]
    if params.case_insensitive:
        argv.append("--ignore-case")
    if params.context:
        argv.extend(["--context", str(params.context)])
    if params.glob:
        argv.extend(["--glob", params.glob])
    argv.extend(["--regexp", params.pattern, str(root)])

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), limit_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise

    if proc.returncode not in (0, RG_EXIT_NO_MATCH):
        raise ValueError(err.decode("utf-8", "replace").strip() or "ripgrep failed")

    matches: list[Match] = []
    for line in out.decode("utf-8", "replace").splitlines():
        parsed = _parse_rg_line(line)
        if parsed is not None:
            matches.append(parsed)
    return matches


_RG_LINE = re.compile(r"^(?P<path>.+?)[:\-](?P<line>\d+)[:\-](?P<text>.*)$")


def _parse_rg_line(line: str) -> Match | None:
    """Parse one ripgrep output line.

    Windows paths start `C:\\`, so the first colon is not the separator. The regex is
    non-greedy from the left and the drive-letter case is repaired explicitly.
    """

    m = _RG_LINE.match(line)
    if m is None:
        return None
    path = m.group("path")
    rest_line = m.group("line")
    text = m.group("text")
    if len(path) == 1 and line[1:3] == ":\\":
        # 'C' + ':\path\to\file:12:text' -- re-split after the drive letter.
        m2 = _RG_LINE.match(line[2:])
        if m2 is None:
            return None
        path = line[:2] + m2.group("path")
        rest_line = m2.group("line")
        text = m2.group("text")
    try:
        number = int(rest_line)
    except ValueError:  # pragma: no cover
        return None
    return Match(path=path, line=number, text=text)


async def _python_search(params: GrepParams, root: Path, *, deadline: float) -> list[Match]:
    """Pure-Python fallback. Reads files in a thread so the loop stays responsive."""

    flags = re.IGNORECASE if params.case_insensitive else 0
    pattern = re.compile(params.pattern, flags)

    if root.is_file():  # noqa: ASYNC240 - one stat on a path already resolved by validate_params
        candidates = [root]
        base = root.parent
    else:
        found, _ = await _list_files(root)
        base = root
        candidates = found
        if params.glob:
            wanted = params.glob.replace("\\", "/")
            candidates = [p for p in candidates if _matches(p, base, wanted)]

    return await asyncio.to_thread(_scan_files, candidates, pattern, params.context, deadline)


def _scan_files(
    paths: list[Path], pattern: re.Pattern[str], context: int, deadline: float
) -> list[Match]:
    matches: list[Match] = []
    for path in paths:
        if time.monotonic() > deadline:
            raise TimeoutError("grep exceeded its budget")
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        lines = text.replace("\r\n", "\n").split("\n")
        for i, line in enumerate(lines):
            if pattern.search(line):
                lo = max(0, i - context)
                hi = min(len(lines), i + context + 1)
                for j in range(lo, hi):
                    matches.append(Match(path=str(path), line=j + 1, text=lines[j]))
    return matches


def _render(
    params: GrepParams,
    matches: list[Match],
    ctx: ToolContext,
    backend: str,
    elapsed_ms: float,
) -> ToolResult:
    """Turn matches into the requested output mode, with explicit truncation."""

    if params.output_mode == "files_with_matches":
        seen: list[str] = []
        for m in matches:
            rel = ctx.display_path(Path(m.path))
            if rel not in seen:
                seen.append(rel)
        return _paginate(params, seen, ctx, backend, elapsed_ms, unit="file")

    if params.output_mode == "count":
        counts: dict[str, int] = {}
        for m in matches:
            rel = ctx.display_path(Path(m.path))
            counts[rel] = counts.get(rel, 0) + 1
        rows = [f"{path}: {n}" for path, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        return _paginate(params, rows, ctx, backend, elapsed_ms, unit="file")

    rows = []
    for m in matches:
        text = m.text
        if len(text) > MAX_MATCH_LINE_CHARS:
            text = f"{text[:MAX_MATCH_LINE_CHARS]}... [clipped]"
        rows.append(f"{ctx.display_path(Path(m.path))}:{m.line}: {text}")
    return _paginate(params, rows, ctx, backend, elapsed_ms, unit="match")


def _paginate(
    params: GrepParams,
    rows: list[str],
    ctx: ToolContext,
    backend: str,
    elapsed_ms: float,
    *,
    unit: str,
) -> ToolResult:
    total = len(rows)
    window = rows[params.offset : params.offset + params.limit]

    if total == 0:
        body = (
            f"No matches for {params.pattern!r} in {params.path}. "
            f"The search completed; there is nothing there. Try a looser pattern, "
            f"case_insensitive=True, or a wider path."
        )
    else:
        body = "\n".join(window)
        end = params.offset + len(window)
        if end < total:
            body += (
                f"\n\n[showing {unit}s {params.offset + 1}-{end} of {total}; "
                f"truncated. Paginate with offset={end}.]"
            )

    return ToolResult.ok_text(
        body,
        display=ToolDisplay(
            kind="matches",
            data={
                "rows": window,
                "total": total,
                "offset": params.offset,
                "truncated": params.offset + len(window) < total,
                "mode": params.output_mode,
            },
        ),
        meta={
            "backend": backend,
            "total": total,
            "returned": len(window),
            "elapsed_ms": elapsed_ms,
        },
    )


__all__ = ["DESCRIPTION", "GrepParams", "GrepTool", "OutputMode"]
