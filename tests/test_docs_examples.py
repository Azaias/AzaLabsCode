"""Every example in `docs/` marked `<!-- runnable -->` is executed here.

Documentation that does not run is documentation that is wrong within a milestone,
and the examples on `docs/writing-a-workflow.md` are the first thing a workflow
author copies. Extracting and running them costs one test and removes a whole class
of rot: a renamed keyword argument breaks the docs in the same command as it breaks
the code.

The examples carry their own assertions -- each ends in an `assert` on what the run
returned -- so this file only has to find them and exec them. They are self-contained
by construction: `FakeProvider` for the model, `tempfile` for the workspace, no
network and no key.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"

RUNNABLE = re.compile(r"<!--\s*runnable\s*-->\s*\n+```python\n(.*?)```", re.DOTALL)
"""A fenced Python block preceded by the marker. An unmarked block is illustrative:
it may name a `MyConfig` that does not exist, and running it would prove nothing."""


def examples() -> list[tuple[str, str]]:
    """`(id, source)` for every runnable example in the docs tree."""

    found: list[tuple[str, str]] = []
    for page in sorted(DOCS.glob("*.md")):
        for index, source in enumerate(RUNNABLE.findall(page.read_text(encoding="utf-8"))):
            found.append((f"{page.stem}-{index}", source))
    return found


CASES = examples()


def test_the_docs_contain_runnable_examples() -> None:
    """If the marker is renamed, every case below silently disappears."""

    assert len(CASES) >= 2, "docs/writing-a-workflow.md marks its examples <!-- runnable -->"


@pytest.mark.parametrize("source", [source for _, source in CASES], ids=[name for name, _ in CASES])
def test_the_example_runs(source: str) -> None:
    """Each example asserts on its own result; a failure here is a failure there."""

    exec(compile(source, "<docs example>", "exec"), {"__name__": "__docs__"})
