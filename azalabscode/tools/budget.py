"""Output caps and the per-turn result budget (R-T-6, spec delta 8).

Two separate limits, and conflating them is how a turn blows up:

- **The per-result cap** bounds one tool result. Overflow is spilled to disk and
  replaced with a `<persisted-output>` block naming the size, the spill path and a
  head preview. A head-and-tail slice, which is what spec R-T-6 asks for, throws
  away the middle permanently; a spill plus a path lets the model `read_file` the
  rest if it turns out to matter.
- **The per-turn budget** bounds the *sum* folded into the next model request.
  Eight parallel results each just under the per-result cap is 400 000 characters,
  which the per-result cap alone does nothing about.

Budget decisions are memoized by `call_id`, so replaying a resumed turn produces
byte-identical results. Without that, a resume that re-orders completions would
elide a different result and the transcript would differ from the saved one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from azalabscode.content import Part, TextPart
from azalabscode.ids import slugify_for_path
from azalabscode.toolio import (
    DEFAULT_TURN_RESULT_BUDGET,
    ToolError,
    ToolErrorKind,
    ToolResult,
    is_unbounded,
)

PREVIEW_CHARS = 2_000
"""How much of an over-cap result is inlined. Enough to see what it is, cheap enough
that the model is not paying for the whole thing twice."""


def result_text_size(result: ToolResult) -> int:
    """Model-facing character count of a result.

    Only `TextPart`s count. An `ImagePart` is bounded by the tool that produced it
    and is not measured in characters.
    """

    return sum(len(p.text) for p in result.content if isinstance(p, TextPart))


def spill_dir_for(session_dir: Path | None, fallback: Path | None = None) -> Path | None:
    """Where over-cap output goes: `session_dir/tool_output`, or `fallback`."""

    if session_dir is not None:
        return Path(session_dir) / "tool_output"
    return fallback


def spill(text: str, *, call_id: str, directory: Path, suffix: str = ".txt") -> Path:
    """Write `text` to a file named after the call. Returns the path.

    The call id goes through `slugify_for_path`: ids are model- and provider-shaped
    strings and reach a filename here.
    """

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slugify_for_path(call_id)}{suffix}"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def persisted_output_block(
    *,
    size: int,
    path: Path | None,
    preview: str,
    reason: str = "output exceeded the per-result limit",
) -> str:
    """The replacement text for an over-cap result.

    Names the size and the location first, because that is what the model needs to
    decide whether to go and get the rest.
    """

    where = f"Full output saved to: {path}" if path is not None else "Full output was discarded."
    return (
        f"<persisted-output>\n"
        f"{reason} ({size} characters).\n"
        f"{where}\n\n"
        f"First {min(len(preview), PREVIEW_CHARS)} characters:\n"
        f"{preview}\n"
        f"</persisted-output>"
    )


def apply_result_cap(
    result: ToolResult,
    *,
    cap: int | float,
    call_id: str,
    spill_directory: Path | None,
) -> ToolResult:
    """Bound one result to `cap` characters, spilling the overflow (R-T-6).

    A result already under the cap is returned unchanged -- same object, so the
    common path costs nothing. `math.inf` opts out entirely: `read_file` does,
    because spilling a file read to disk that the model then re-reads is circular.

    Non-text parts survive. An `ImagePart` is not what blew the cap.
    """

    if is_unbounded(cap):
        return result

    size = result_text_size(result)
    if size <= cap:
        return result

    full_text = "".join(p.text for p in result.content if isinstance(p, TextPart))
    path: Path | None = None
    if spill_directory is not None:
        try:
            path = spill(full_text, call_id=call_id, directory=spill_directory)
        except OSError:
            path = None

    block = persisted_output_block(size=size, path=path, preview=full_text[:PREVIEW_CHARS])
    kept: list[Part] = [p for p in result.content if not isinstance(p, TextPart)]
    meta = dict(result.meta)
    meta["output_truncated"] = True
    meta["output_chars"] = size
    if path is not None:
        meta["output_path"] = str(path)

    return result.model_copy(update={"content": [TextPart(text=block), *kept], "meta": meta})


@dataclass
class TurnBudget:
    """Ceiling on the combined tool results in one turn (delta 8).

    `charge` is idempotent per `call_id`: the second call with the same id returns
    the same verdict, so a resumed turn folds the same bytes into the transcript as
    the run that was interrupted. That is what makes the saved transcript and the
    resumed one comparable.
    """

    limit: int | float = DEFAULT_TURN_RESULT_BUDGET
    spent: int = 0
    decisions: dict[str, int] = field(default_factory=dict)
    """`call_id -> charged size`. A charge of -1 marks a call that was elided."""

    def remaining(self) -> int | float:
        """Characters still available this turn."""

        if is_unbounded(self.limit):
            return math.inf
        return max(0, int(self.limit) - self.spent)

    def charge(self, result: ToolResult, *, call_id: str) -> ToolResult:
        """Fold one result into the budget, eliding it if it does not fit.

        An elided result becomes `ToolError(kind="budget")` naming what was dropped,
        not silence: a model told nothing came back will call the tool again.
        """

        if is_unbounded(self.limit):
            return result

        prior = self.decisions.get(call_id)
        if prior is not None:
            return result if prior >= 0 else self._elide(result, call_id)

        size = result_text_size(result)
        if size <= self.remaining():
            self.decisions[call_id] = size
            self.spent += size
            return result

        self.decisions[call_id] = -1
        return self._elide(result, call_id)

    def _elide(self, result: ToolResult, call_id: str) -> ToolResult:
        size = result_text_size(result)
        message = (
            f"result omitted: it is {size} characters and the combined tool output for "
            f"this turn has reached its {self.limit}-character limit. Re-run this call "
            f"on its own, or narrow it (fewer lines, a tighter pattern)."
        )
        meta = dict(result.meta)
        meta["budget_elided"] = True
        meta["output_chars"] = size
        return ToolResult(
            ok=False,
            content=[TextPart(text=message)],
            error=ToolError(
                kind=ToolErrorKind.BUDGET,
                message=message,
                details={"call_id": call_id, "chars": size},
            ),
            display=result.display,
            duration_ms=result.duration_ms,
            meta=meta,
        )

    def reset(self) -> None:
        """Start a fresh turn."""

        self.spent = 0
        self.decisions.clear()


__all__ = [
    "PREVIEW_CHARS",
    "TurnBudget",
    "apply_result_cap",
    "persisted_output_block",
    "result_text_size",
    "spill",
    "spill_dir_for",
]
