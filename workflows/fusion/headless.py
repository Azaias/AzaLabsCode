"""Running fusion without a TUI (spec 9.2's `--headless`).

The TUI is M6. What is here is the smaller half that M5's exit test needs and that
every workflow wants anyway: build the graph, bind it to a `Controller` with the
import path and config that let a session rebuild it, and run.

`controller_for` and `run_headless` are separate on purpose. A test that wants to
pause mid-fan-out needs the controller *before* the run starts, and a caller that
just wants an answer should not have to know that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from azalabscode import Controller, EventBus, PermissionMode
from workflows.fusion.workflow import CONFIG_TYPE, IMPORT_PATH, FusionConfig, build


def controller_for(
    config: FusionConfig | dict[str, Any],
    *,
    session_dir: str | Path | None = None,
    bus: EventBus | None = None,
    mode: PermissionMode = PermissionMode.AUTO,
    autosave: bool = True,
    strict_graph_hash: bool = False,
    run_id: str | None = None,
) -> Controller:
    """A controller with the fusion graph bound and its rebuild recipe recorded.

    `auto` mode by default: fusion runs `ModelCall` nodes and no tools, so there is
    nothing for an approval gate to be asked about, and a headless run with no
    handler in `manual` mode would refuse to start (R-C-8).
    """

    cfg = config if isinstance(config, FusionConfig) else FusionConfig.model_validate(config)
    controller = Controller(
        run_id=run_id,
        bus=bus,
        permission_mode=mode,
        session_dir=session_dir,
        autosave=autosave,
        strict_graph_hash=strict_graph_hash,
    )
    controller.bind_workflow(
        build(cfg),
        import_path=IMPORT_PATH,
        config=cfg.model_dump(mode="json"),
        config_type=CONFIG_TYPE,
    )
    return controller


async def run_headless(
    config: FusionConfig | dict[str, Any],
    *,
    session_dir: str | Path | None = None,
    timeout: float | None = None,  # noqa: ASYNC109 - the bound is the caller's, and Controller.run takes it
) -> str:
    """Build, run and return the synthesized answer."""

    controller = controller_for(config, session_dir=session_dir)
    result = await controller.run(timeout=timeout)
    return str(result or "")


__all__ = ["controller_for", "run_headless"]
