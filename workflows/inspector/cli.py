"""`inspector` on the command line: a TUI by default, `--headless` when asked (R-U-7).

    python -m workflows.inspector "where is the retry logic?" --workspace ../repo
    python -m workflows.inspector --headless --config inspector.json "..."

Headless prints the answer and then the call table, which is the part R-A-3 is about.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from azalabscode import EventBus, JsonlRecorder, PermissionMode, run_sync
from workflows.cli_support import (
    CliError,
    load_config_file,
    merge_config,
    prepare_process,
    require_api_key,
)
from workflows.inspector.headless import CallWatcher, controller_for
from workflows.inspector.workflow import InspectorConfig

app = typer.Typer(add_completion=False, help="Watch one agent's tool calls, live.")


def build_config(
    task: str,
    *,
    model: str | None,
    workspace: str | None,
    tools: tuple[str, ...],
    config_path: str | None,
) -> InspectorConfig:
    """File defaults, then command-line overrides."""

    data = merge_config(
        load_config_file(config_path),
        {
            "task": task or None,
            "model": model,
            "workspace": workspace,
            "tools": tools,
        },
    )
    try:
        cfg = InspectorConfig.model_validate(data)
    except ValueError as exc:
        raise CliError(f"invalid inspector config: {exc}") from exc
    if not cfg.task:
        raise CliError("no task: pass one as the argument or set it in the config file")
    return cfg


async def run_headless(
    cfg: InspectorConfig, *, session_dir: Path | None, log: Path | None
) -> tuple[str, str]:
    """Run with no UI. Returns `(answer, call table)` (R-U-7)."""

    controller = controller_for(cfg, session_dir=session_dir, mode=PermissionMode.AUTO)
    async with CallWatcher(controller.bus) as watcher:
        if log is None:
            answer = await controller.run()
        else:
            async with JsonlRecorder(controller.bus, log):
                answer = await controller.run()
    return str(answer or ""), watcher.summary.render()


async def run_tui(cfg: InspectorConfig, *, session_dir: Path | None, log: Path | None) -> None:
    """Run under `InspectorApp` (spec 9.3). Imported here so `--headless` stays cheap."""

    from azalabscode.tui import TUIApprovalHandler
    from workflows.inspector.app import InspectorApp

    controller = controller_for(
        cfg,
        session_dir=session_dir,
        bus=EventBus(),
        mode=PermissionMode.MANUAL,
        approval_handler=TUIApprovalHandler(),
    )
    tui = InspectorApp(controller, cfg, session_dir=session_dir, autostart=True)
    if log is not None:
        async with JsonlRecorder(controller.bus, log):
            await tui.run_async()
    else:
        await tui.run_async()


@app.command()
def main(
    task: Annotated[str, typer.Argument(help="What to ask the agent.")] = "",
    model: Annotated[str | None, typer.Option(help="Model id.")] = None,
    workspace: Annotated[str | None, typer.Option(help="Directory the tools may touch.")] = None,
    tools: Annotated[
        list[str] | None, typer.Option("--tool", "-t", help="A tool name. Repeat to add more.")
    ] = None,
    config: Annotated[str | None, typer.Option(help="JSON config file.")] = None,
    session_dir: Annotated[Path | None, typer.Option(help="Where to checkpoint.")] = None,
    log: Annotated[Path | None, typer.Option(help="Write the event stream as JSONL.")] = None,
    headless: Annotated[bool, typer.Option("--headless", help="No TUI (R-U-7).")] = False,
) -> None:
    """Run the tool-observability workflow."""

    try:
        cfg = build_config(
            task,
            model=model,
            workspace=workspace,
            tools=tuple(tools or ()),
            config_path=config,
        )
        if cfg.script is None:
            require_api_key()
        prepare_process(session_dir)
        if headless:
            answer, table = run_sync(run_headless(cfg, session_dir=session_dir, log=log))
        else:
            run_sync(run_tui(cfg, session_dir=session_dir, log=log))
            return
    except CliError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(2) from exc
    typer.echo(answer)
    typer.echo("")
    typer.echo(table)


def entrypoint() -> None:
    """Console-script target."""

    app()


if __name__ == "__main__":  # pragma: no cover - exercised through `python -m`
    entrypoint()


__all__ = ["app", "build_config", "entrypoint", "main", "run_headless", "run_tui"]
