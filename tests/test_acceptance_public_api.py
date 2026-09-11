"""R-A-4: none of the three workflows imports a private module, patches core, or
subclasses a `@final` core class.

Import-linter contract 5 already enforces the first clause in CI, and this file runs
the same contract from pytest so a `pytest` run alone catches a violation -- but the
other two clauses have no linter, so they are checked here by reading the source.

The check is an AST walk rather than a runtime probe on purpose. Monkey-patching is
something a module *does at import time*, so a runtime check would have to import the
workflow first, at which point the patch has already happened and the damage is done.
The AST is also what a reviewer reads, which is the right level for a rule whose point
is "this code stays a consumer of the public API".

**Every clause is proved non-vacuous.** A test that says "no violations" over source
that could not violate it is worth nothing, so each check is also run against a small
synthetic module that does the forbidden thing, and must find it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import azalabscode

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"

PUBLIC_MODULES = {
    "azalabscode",
    "azalabscode.control",
    "azalabscode.providers",
    "azalabscode.tools",
    "azalabscode.workflows",
    "azalabscode.tui",
}
"""R-X-6: the package root and the per-layer `__init__`s. Nothing else is public."""


def workflow_sources() -> list[Path]:
    """Every Python file in the three reference workflows."""

    return sorted(path for path in WORKFLOWS.rglob("*.py") if "__pycache__" not in path.parts)


def parse(path: Path) -> ast.Module:
    """Parse one file."""

    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ---------------------------------------------------------------------------
# Clause 1: no private imports
# ---------------------------------------------------------------------------


def imported_modules(tree: ast.Module) -> set[str]:
    """Every `azalabscode...` module named by an import in `tree`."""

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.startswith("azalabscode"))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("azalabscode"):
            found.add(str(node.module))
    return found


def test_the_workflows_import_only_public_modules() -> None:
    """R-A-4, R-X-6. The same rule import-linter contract 5 enforces."""

    offenders: dict[str, set[str]] = {}
    for path in workflow_sources():
        private = {
            module for module in imported_modules(parse(path)) if module not in PUBLIC_MODULES
        }
        if private:
            offenders[str(path.relative_to(WORKFLOWS.parent))] = private
    assert not offenders, f"private imports: {offenders}"


def test_the_import_contract_itself_is_green() -> None:
    """Run import-linter from pytest, so a `pytest` run alone catches a violation."""

    # `python -m importlinter.cli` is a click *group* and exits 0 having done nothing,
    # which would make this test green whatever the contracts said. Calling the
    # function is the only invocation that actually lints.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from importlinter.cli import lint_imports; sys.exit(lint_imports())",
        ],
        cwd=WORKFLOWS.parent,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 kept, 0 broken" in result.stdout, result.stdout


def test_a_private_import_would_be_caught(tmp_path: Path) -> None:
    """Non-vacuous: the same walk over a module that does the forbidden thing."""

    offender = tmp_path / "bad.py"
    offender.write_text("from azalabscode.control.gate import RuntimePermissionGate\n", "utf-8")
    private = {
        module for module in imported_modules(parse(offender)) if module not in PUBLIC_MODULES
    }
    assert private == {"azalabscode.control.gate"}


# ---------------------------------------------------------------------------
# Clause 2: no patching of core
# ---------------------------------------------------------------------------


@dataclass
class Patch:
    """An assignment or `setattr` that would mutate something imported from core."""

    file: str
    line: int
    target: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line} patches {self.target}"


def core_names(tree: ast.Module) -> set[str]:
    """Local names bound to something imported from `azalabscode`."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("azalabscode"):
                    names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("azalabscode"):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def patches(path: Path) -> list[Patch]:
    """Assignments to an attribute of an imported core name, and `setattr` on one."""

    tree = parse(path)
    names = core_names(tree)
    found: list[Patch] = []
    relative = _relative(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"setattr", "delattr"} and node.args:
                root = _root_name(node.args[0])
                if root in names:
                    found.append(Patch(relative, node.lineno, f"{root} via {node.func.id}()"))
            continue
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Attribute):
                root = _root_name(target)
                if root in names:
                    found.append(Patch(relative, node.lineno, f"{root}.{target.attr}"))
    return found


def _relative(path: Path) -> str:
    """`path` relative to the repository root, or its name if it is outside it."""

    try:
        return str(path.relative_to(WORKFLOWS.parent))
    except ValueError:
        return path.name


def _root_name(node: ast.expr) -> str | None:
    """The leftmost name of an attribute chain: `a.b.c` -> `a`."""

    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def test_the_workflows_do_not_patch_core() -> None:
    """R-A-4: a workflow that patched core would not be a demonstration of anything."""

    found = [patch for path in workflow_sources() for patch in patches(path)]
    assert not found, "\n".join(str(patch) for patch in found)


def test_a_patch_would_be_caught(tmp_path: Path) -> None:
    """Non-vacuous, both forms."""

    offender = tmp_path / "bad.py"
    offender.write_text(
        "from azalabscode import Controller\n"
        "Controller.save = None\n"
        "setattr(Controller, 'load', None)\n",
        "utf-8",
    )
    found = patches(offender)
    assert [patch.target for patch in found] == ["Controller.save", "Controller via setattr()"]


# ---------------------------------------------------------------------------
# Clause 3: no subclassing of a `@final` core class
# ---------------------------------------------------------------------------


def final_core_classes() -> set[str]:
    """Every class in `azalabscode` decorated `@final`.

    Read off the runtime objects rather than the source: `typing.final` sets
    `__final__` on the class, so this finds them wherever they are defined and
    whatever they are re-exported as.
    """

    return {
        name
        for name in azalabscode.__all__
        if isinstance(getattr(azalabscode, name, None), type)
        and getattr(getattr(azalabscode, name), "__final__", False)
    }


def base_names(tree: ast.Module) -> set[str]:
    """Every name used as a base class anywhere in `tree`."""

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                root = base
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    found.add(root.id)
    return found


def test_core_marks_its_uninheritable_classes_final() -> None:
    """The clause only means something if something is actually marked.

    `Controller`, `EventBus`, `ToolDispatcher` and `Runner` are compositional: they
    are wired together, never specialised. Marking them is what turns R-A-4's third
    clause from a statement about nothing into a check.
    """

    assert {"Controller", "EventBus", "ToolDispatcher", "Runner"} <= final_core_classes()


def test_the_workflows_subclass_nothing_final() -> None:
    """R-A-4's third clause, over the three reference workflows."""

    final = final_core_classes()
    offenders: dict[str, set[str]] = {}
    for path in workflow_sources():
        used = base_names(parse(path)) & final
        if used:
            offenders[str(path.relative_to(WORKFLOWS.parent))] = used
    assert not offenders, f"subclasses a @final core class: {offenders}"


def test_subclassing_a_final_class_would_be_caught(tmp_path: Path) -> None:
    """Non-vacuous, and it is what the type checker would say too."""

    offender = tmp_path / "bad.py"
    offender.write_text(
        "from azalabscode import Controller\n\n\nclass Mine(Controller):\n    pass\n", "utf-8"
    )
    assert base_names(parse(offender)) & final_core_classes() == {"Controller"}


@pytest.mark.parametrize("name", ["Controller", "EventBus", "ToolDispatcher", "Runner"])
def test_a_final_class_still_works_normally(name: str) -> None:
    """`@final` is a type-checker annotation; it must not change runtime behaviour."""

    cls: Any = getattr(azalabscode, name)
    assert isinstance(cls, type)
    assert cls.__final__ is True
