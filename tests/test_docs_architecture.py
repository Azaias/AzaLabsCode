"""R-X-5: the architecture diagram must match the code.

Intent success criterion 4 is "every layer can be explained in a single diagram, and
the diagram matches the code". A diagram nobody checks drifts within one milestone,
so `docs/architecture.md` is a test fixture: this file is the check R-X-5 asks CI to
run, in both directions.

* **Every name the document writes must exist.** Not only modules -- the longest
  dotted prefix that imports is resolved, and the rest must be real attributes of it.
  So `azalabscode.control.gate.RuntimePermissionGate` fails the moment that class is
  renamed, which is the failure mode a prose diagram actually has.
* **Every `azalabscode.*` package and every top-level module must be named.** Adding a
  package to the tree breaks this test until the diagram is edited in the same commit.
  That is the same bargain `exhaustive = true` makes in the layers contract.
* **The layer order in the document must equal the one `import-linter` enforces.**
  The contract in `pyproject.toml` is the executable form of the diagram; if the two
  disagree, the picture is wrong and the code is right.
* **Every repo path the document cites must exist**, because a docs tree that points
  at a renamed test is worse than one that points at nothing.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
ARCHITECTURE = DOCS / "architecture.md"
PACKAGE = ROOT / "azalabscode"

DOTTED = re.compile(r"\bazalabscode(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
"""A dotted name rooted at the package. Trailing punctuation is excluded by the
character class, so `azalabscode.tools.builtin:` yields the module, not the colon."""

CITED_PATH = re.compile(r"`([A-Za-z0-9_./-]+\.(?:md|py|toml|json|jsonl))`")
"""A repo path in backticks. Markdown link targets are covered by the same shape."""


def document() -> str:
    return ARCHITECTURE.read_text(encoding="utf-8")


def packages_on_disk() -> set[str]:
    """Every `azalabscode.*` package, by dotted name."""

    return {".".join(path.parent.relative_to(ROOT).parts) for path in PACKAGE.rglob("__init__.py")}


def top_level_modules() -> set[str]:
    """Every module directly under `azalabscode/`. R-X-5 names these explicitly."""

    return {f"azalabscode.{path.stem}" for path in PACKAGE.glob("*.py") if path.stem != "__init__"}


def layer_order_from_document() -> list[str]:
    """The fenced block after the `<!-- layers -->` marker, one layer per line."""

    _, _, after = document().partition("<!-- layers -->")
    assert after, "the document has no <!-- layers --> marker"
    block = after.split("```")[1]
    return [line.strip() for line in block.strip().splitlines() if line.strip()]


def layer_order_from_contract() -> list[str]:
    """The `layers` list from import-linter contract 1."""

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contracts = config["tool"]["importlinter"]["contracts"]
    layers = next(c for c in contracts if c["type"] == "layers")
    return [str(layer).strip() for layer in layers["layers"]]


# ---------------------------------------------------------------------------


def test_the_architecture_document_exists() -> None:
    """R-X-5 names the path. It is not a suggestion."""

    assert ARCHITECTURE.is_file()
    assert len(document()) > 2000, "a diagram with no explanation is not the deliverable"


def test_every_name_the_diagram_uses_exists() -> None:
    """The forward direction: nothing in the document is stale."""

    missing: list[str] = []
    for name in sorted(set(DOTTED.findall(document()))):
        parts = name.split(".")
        module = None
        for stop in range(len(parts), 0, -1):
            try:
                module = importlib.import_module(".".join(parts[:stop]))
            except ImportError:
                continue
            rest = parts[stop:]
            break
        else:  # pragma: no cover - `azalabscode` itself always imports
            missing.append(f"{name}: no importable prefix")
            continue
        target = module
        for attribute in rest:
            if not hasattr(target, attribute):
                missing.append(f"{name}: {module.__name__} has no {attribute!r}")
                break
            target = getattr(target, attribute)
    assert not missing, "\n".join(missing)


def test_every_package_appears_in_the_diagram() -> None:
    """The reverse direction, and the half R-X-5 states outright."""

    text = document()
    missing = [name for name in sorted(packages_on_disk()) if name not in text]
    assert not missing, f"add these to docs/architecture.md: {missing}"


def test_every_top_level_module_appears_in_the_diagram() -> None:
    """ "names every top-level module" -- the leaf tier is where they all are."""

    text = document()
    missing = [name for name in sorted(top_level_modules()) if name not in text]
    assert not missing, f"add these to docs/architecture.md: {missing}"


def test_the_diagram_layer_order_is_the_contract_layer_order() -> None:
    """The diagram and `lint-imports` cannot disagree about which way is down."""

    assert layer_order_from_document() == layer_order_from_contract()


def test_the_diagram_is_not_vacuous() -> None:
    """A document that named nothing would pass both directions above only if the
    tree were empty. Pin the shape instead of trusting that."""

    text = document()
    assert len(set(DOTTED.findall(text))) >= 30
    assert "```" in text, "the diagram is a fenced block"


@pytest.mark.parametrize(
    "protocol",
    ["PermissionGate", "RunControl", "Delegator", "ApprovalHandler", "EventSink"],
)
def test_every_cross_layer_interface_is_named(protocol: str) -> None:
    """R-X-5: "every cross-layer interface". They are exactly `contracts.__all__`'s
    protocols, and each is what keeps one upward data flow from becoming an upward
    import."""

    import azalabscode.contracts as contracts

    assert hasattr(contracts, protocol)
    assert protocol in document(), f"{protocol} is a seam and belongs in the diagram"


def test_every_repo_path_the_document_cites_exists() -> None:
    """Docs that point at a renamed file are worse than docs that point at nothing."""

    missing = [
        path
        for path in sorted(set(CITED_PATH.findall(document())))
        if "/" in path and not (ROOT / path).exists()
    ]
    assert not missing, f"docs/architecture.md cites files that do not exist: {missing}"


def test_the_docs_tree_has_its_entry_points() -> None:
    """R-X-7's `docs/` tree. `mkdocs.yml` must name every page, or a page is written
    and never built."""

    pages = {"index.md", "architecture.md", "writing-a-workflow.md", "reference-workflows.md"}
    on_disk = {path.name for path in DOCS.glob("*.md")}
    assert pages <= on_disk, f"missing docs pages: {sorted(pages - on_disk)}"

    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for page in sorted(on_disk):
        assert page in config, f"{page} is not in the mkdocs nav"


def test_every_layer_has_a_readme() -> None:
    """R-X-7: "a README per layer". The layers are the five packages a caller
    programs against; the leaf tier is documented in `docs/architecture.md`."""

    for layer in ("providers", "tools", "workflows", "control", "tui"):
        readme = PACKAGE / layer / "README.md"
        assert readme.is_file(), f"azalabscode/{layer} has no README.md"
        assert len(readme.read_text(encoding="utf-8")) > 400, f"{layer}: a stub is not a README"


def test_every_public_symbol_has_a_docstring() -> None:
    """R-X-7: "docstrings on every public symbol". Checked over what each layer's
    `__all__` actually exports, which is the definition R-X-6 gives for public."""

    undocumented: list[str] = []
    for name in (
        "azalabscode",
        *[f"azalabscode.{n}" for n in ("providers", "tools", "workflows", "control", "tui")],
    ):
        module = importlib.import_module(name)
        assert module.__doc__, f"{name} has no module docstring"
        for symbol in getattr(module, "__all__", ()):
            value = getattr(module, symbol)
            if (isinstance(value, type) or callable(value)) and not getattr(value, "__doc__", None):
                undocumented.append(f"{name}.{symbol}")
    assert not undocumented, undocumented
