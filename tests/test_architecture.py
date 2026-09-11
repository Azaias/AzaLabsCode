"""Architecture enforcement: the six import-linter contracts and the public API.

R-X-2, R-X-6 and R-U-1 are checked by `lint-imports` reading `pyproject.toml`. Running it
from pytest means a broken layer shows up in the same command as a broken test,
rather than only in a CI step someone can forget.

The contracts are also checked *negatively* -- a deliberately illegal import must
break the contract that exists to catch it. A contract that passes because its
source package happens to be empty is not evidence of anything.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_CONTRACTS = [
    "1. Layers depend strictly downward",
    "2. Leaf-facing layers never import upward",
    "3. workflows never imports control",
    "4. providers and tools are independent",
    "5. Reference workflows use the public API only",
    "6. The UI sees events and the Controller, nothing else",
]


def lint_imports(cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    """Run `lint-imports` in `cwd` and return the completed process."""

    # `importlinter.cli` has no `__main__`; the console script calls this function
    # directly. Invoking it the same way keeps the test and CI on one code path.
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from importlinter.cli import lint_imports_command as c; "
            "sys.exit(c(standalone_mode=False) or 0)",
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_all_import_contracts_are_kept() -> None:
    """R-X-2, R-X-6, R-U-1. The exit test for M0 names the first two explicitly."""

    result = lint_imports()
    assert result.returncode == 0, result.stdout + result.stderr
    for name in EXPECTED_CONTRACTS:
        assert f"{name} KEPT" in result.stdout, result.stdout
    assert "Contracts: 6 kept, 0 broken." in result.stdout


def test_the_layers_contract_is_exhaustive() -> None:
    """`exhaustive = true` also satisfies half of R-X-5: every `azalabscode.*`
    package must be named, so a new one cannot be added without being placed."""

    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "exhaustive = true" in config
    assert "exclude_type_checking_imports = false" in config, "no TYPE_CHECKING escape hatch"


def test_every_package_in_the_tree_is_named_in_the_layers_contract() -> None:
    """A module added without a layer breaks the contract, but only once someone
    runs it. This asserts the mapping directly."""

    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    layers_block = config.split('name = "1. Layers depend strictly downward"')[1]
    layers_block = layers_block.split("[[tool.importlinter.contracts]]")[0]

    on_disk = {
        path.stem if path.suffix == ".py" else path.name
        for path in (ROOT / "azalabscode").iterdir()
        if (path.suffix == ".py" and path.stem != "__init__")
        or (path.is_dir() and (path / "__init__.py").exists())
    }
    for module in on_disk:
        assert module in layers_block, f"{module} is not placed in the layers contract"


@pytest.mark.parametrize(
    ("target", "bad_import", "contract"),
    [
        (
            "azalabscode/tools/_probe.py",
            "from azalabscode.control import *  # noqa: F403",
            "2. Leaf-facing layers never import upward",
        ),
        (
            "azalabscode/workflows/_probe.py",
            "from azalabscode.control import *  # noqa: F403",
            "3. workflows never imports control",
        ),
        (
            "azalabscode/providers/_probe.py",
            "from azalabscode.tools import *  # noqa: F403",
            "4. providers and tools are independent",
        ),
        (
            "workflows/fusion/_probe.py",
            "from azalabscode.messages import UserMessage  # noqa: F401",
            "5. Reference workflows use the public API only",
        ),
        (
            "azalabscode/tui/_probe.py",
            "from azalabscode.workflows import *  # noqa: F403",
            "6. The UI sees events and the Controller, nothing else",
        ),
    ],
    ids=[
        "tools-imports-control",
        "workflows-imports-control",
        "providers-imports-tools",
        "refwf-private",
        "tui-imports-workflows",
    ],
)
def test_an_illegal_import_breaks_the_contract_meant_to_catch_it(
    target: str, bad_import: str, contract: str
) -> None:
    """Each contract is proven to bite, not merely to pass."""

    path = ROOT / target
    assert not path.exists()
    path.write_text(f"{bad_import}\n", encoding="utf-8")
    try:
        result = lint_imports()
        assert f"{contract} BROKEN" in result.stdout, result.stdout
        assert result.returncode != 0
    finally:
        path.unlink()

    assert lint_imports().returncode == 0, "the tree must be clean again afterwards"


def test_a_reference_workflow_may_import_the_public_api() -> None:
    """The complement of the case above: contract 5 must not forbid the sanctioned
    path, or it would be unsatisfiable at M6."""

    path = ROOT / "workflows" / "fusion" / "_probe.py"
    path.write_text(
        textwrap.dedent(
            """
            from azalabscode import UserMessage  # noqa: F401
            from azalabscode.providers import FakeProvider  # noqa: F401
            """
        ).lstrip(),
        encoding="utf-8",
    )
    try:
        result = lint_imports()
        assert result.returncode == 0, result.stdout
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# The public API surface (R-X-6)
# ---------------------------------------------------------------------------


def test_the_root_package_exports_everything_it_names() -> None:
    import azalabscode

    for name in azalabscode.__all__:
        assert hasattr(azalabscode, name), f"__all__ names {name} but it is not exported"


def test_the_provider_layer_exports_everything_it_names() -> None:
    import azalabscode.providers as providers

    for name in providers.__all__:
        assert hasattr(providers, name)


def test_importing_the_harness_does_not_import_textual() -> None:
    """Keeping `tui` out of the root re-export is what makes a headless run cheap,
    and what keeps M4's Textual go/no-go a swap rather than a rewrite."""

    code = "import azalabscode, sys; print('textual' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_every_layer_package_is_importable_and_documented() -> None:
    """The stub layers carry the M-number their contents land at, so a later session
    does not have to reconstruct the plan from the file tree."""

    import importlib

    for name in ("tools", "workflows", "control", "tui", "providers"):
        module = importlib.import_module(f"azalabscode.{name}")
        assert module.__doc__, f"azalabscode.{name} has no module docstring"
        assert hasattr(module, "__all__")


def test_the_contracts_module_holds_only_seams() -> None:
    """`contracts.py` is where the layering hinges. It must not grow behavior."""

    import azalabscode.contracts as contracts

    source = Path(contracts.__file__).read_text(encoding="utf-8")
    assert "import httpx" not in source
    assert "asyncio" not in source, "a protocol module has nothing to schedule"
