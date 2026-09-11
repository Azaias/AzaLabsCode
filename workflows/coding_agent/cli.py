"""`azc`: the coding agent as an installed command (plan decision D6).

    azc                                  # interactive session in the working directory
    azc "add a --verbose flag to cli.py"  # start with a task, then keep talking
    azc --headless "fix the failing test" # one task, no UI, approvals on stdin
    azc --resume .azalabscode/session.json
    azc --auto "..."                      # no approval prompts (spec C-5, C-6)

D6 is what makes success criterion 5 -- "I use this instead of the thing I was using"
-- measurable by using it rather than by reading an event log, so this file carries
the parts that make daily use bearable: config and API-key resolution, a session
directory that defaults to somewhere sensible, and the two startup steps earlier
milestones left without a caller (`gc.freeze()`, `sweep_temp_files`).

**`manual` is the default mode.** `permissions.DEFAULT_MODE` is manual and this does
not override it: a mistaken prompt costs a keystroke, a mistaken `shell` costs a
repository. `--auto` is how a caller says otherwise, and R-C-8 makes a headless
`manual` run legal only because there is a `StdinApprovalHandler` to answer it.

**The workspace defaults to the working directory** (spec 9.1), and it is the root
every file tool is confined to (R-T-7). `--workspace` moves it; nothing widens it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from azalabscode import (
    EventBus,
    JsonlRecorder,
    PermissionMode,
    StdinApprovalHandler,
    run_sync,
)
from workflows.cli_support import (
    CliError,
    load_config_file,
    merge_config,
    prepare_process,
    require_api_key,
)
from workflows.coding_agent.session import CodingSession
from workflows.coding_agent.workflow import CodingAgentConfig

SESSION_DIRNAME = ".azalabscode"
"""Where a session goes when the caller does not say. Inside the workspace, so a
resume from the same directory finds it and two repositories do not share one."""

app = typer.Typer(add_completion=False, help="An agentic coding session in this directory.")


def build_config(
    task: str,
    *,
    model: str | None,
    workspace: str | None,
    tools: tuple[str, ...],
    subagents: tuple[str, ...],
    config_path: str | None,
    interactive: bool,
) -> CodingAgentConfig:
    """File defaults, then command-line overrides. Unset flags never overwrite."""

    data = merge_config(
        load_config_file(config_path),
        {
            "task": task or None,
            "model": model,
            "workspace": workspace,
            "tools": tools,
            "subagents": subagents,
            "interactive": interactive,
        },
    )
    try:
        return CodingAgentConfig.model_validate(data)
    except ValueError as exc:
        raise CliError(f"invalid coding-agent config: {exc}") from exc


def session_directory(cfg: CodingAgentConfig, given: Path | None) -> Path:
    """Where checkpoints go: what was asked for, or `<workspace>/.azalabscode`."""

    return given if given is not None else Path(cfg.workspace_path()) / SESSION_DIRNAME


async def run_headless(
    cfg: CodingAgentConfig,
    *,
    session_dir: Path,
    log: Path | None,
    resume: Path | None,
    mode: PermissionMode,
) -> str:
    """One task, no UI (R-U-7). Approvals go to stdin unless the mode is `auto`."""

    handler = None if mode is PermissionMode.AUTO else StdinApprovalHandler()
    if resume is not None:
        session = await CodingSession.load(resume, approval_handler=handler)
    else:
        session = CodingSession.create(
            cfg, session_dir=session_dir, mode=mode, approval_handler=handler
        )
    if log is None:
        return await session.run_once()
    async with JsonlRecorder(session.controller.bus, log):
        return await session.run_once()


async def run_tui(
    cfg: CodingAgentConfig,
    *,
    session_dir: Path,
    log: Path | None,
    resume: Path | None,
    mode: PermissionMode,
) -> None:
    """The interactive session (R-A-1). Imported here so `--headless` stays cheap."""

    from workflows.coding_agent.app import CodingAgentApp

    if resume is not None:
        session = await CodingSession.load(resume, bus=EventBus())
    else:
        session = CodingSession.create(cfg, session_dir=session_dir, mode=mode)
    tui = CodingAgentApp(session, session_dir=session_dir, autostart=True)
    if log is not None:
        async with JsonlRecorder(session.controller.bus, log):
            await tui.run_async()
    else:
        await tui.run_async()


@app.command()
def main(
    task: Annotated[str, typer.Argument(help="What to do. Optional in a TUI session.")] = "",
    model: Annotated[str | None, typer.Option(help="Model id.")] = None,
    workspace: Annotated[
        str | None, typer.Option(help="Root the tools may touch. Defaults to the cwd.")
    ] = None,
    tools: Annotated[
        list[str] | None, typer.Option("--tool", "-t", help="Restrict the toolset.")
    ] = None,
    subagents: Annotated[
        list[str] | None, typer.Option("--subagent", help="Specs `delegate` may name.")
    ] = None,
    config: Annotated[str | None, typer.Option(help="JSON config file.")] = None,
    session_dir: Annotated[Path | None, typer.Option(help="Where to checkpoint.")] = None,
    resume: Annotated[Path | None, typer.Option(help="Reopen a saved session.json.")] = None,
    log: Annotated[Path | None, typer.Option(help="Write the event stream as JSONL.")] = None,
    auto: Annotated[bool, typer.Option("--auto", help="Approve every tool call (C-5).")] = False,
    headless: Annotated[bool, typer.Option("--headless", help="No TUI (R-U-7).")] = False,
) -> None:
    """Run a coding session."""

    try:
        cfg = build_config(
            task,
            model=model,
            workspace=workspace,
            tools=tuple(tools or ()),
            subagents=tuple(subagents or ()),
            config_path=config,
            interactive=not headless,
        )
        if headless and not cfg.task and resume is None:
            raise CliError("--headless needs a task: pass one as the argument")
        if cfg.script is None:
            require_api_key()
        directory = session_directory(cfg, session_dir)
        prepare_process(directory)
        mode = PermissionMode.AUTO if auto else PermissionMode.MANUAL
        if headless:
            answer = run_sync(
                run_headless(cfg, session_dir=directory, log=log, resume=resume, mode=mode)
            )
        else:
            run_sync(run_tui(cfg, session_dir=directory, log=log, resume=resume, mode=mode))
            return
    except CliError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(2) from exc
    typer.echo(answer)


def entrypoint() -> None:
    """Console-script target: `azc`."""

    app()


if __name__ == "__main__":  # pragma: no cover - exercised through `python -m`
    entrypoint()


__all__ = [
    "SESSION_DIRNAME",
    "app",
    "build_config",
    "entrypoint",
    "main",
    "run_headless",
    "run_tui",
    "session_directory",
]
