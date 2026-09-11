"""`fusion` on the command line: a TUI by default, `--headless` when asked (R-U-7).

    python -m workflows.fusion "why is my build slow?" -m openai/gpt-5 -m google/gemini-3
    python -m workflows.fusion --headless --config fusion.json "..."
    python -m workflows.fusion --resume runs/fusion/session.json

The two modes run the *same* controller with the same graph bound to it. That is what
makes R-U-7 more than a flag: the headless path is not a second implementation, it is
the same run with nothing subscribed to it but a recorder.

Every option that can also live in the config file defaults to `None`, so a flag that
was not typed cannot overwrite a value that was configured (`merge_config`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from azalabscode import Controller, EventBus, JsonlRecorder, PermissionMode, run_sync
from workflows.cli_support import (
    CliError,
    load_config_file,
    merge_config,
    prepare_process,
    require_api_key,
)
from workflows.fusion.headless import controller_for
from workflows.fusion.workflow import FusionConfig

DEFAULT_MODELS = ["anthropic/claude-haiku-4.5", "openai/gpt-4o-mini", "google/gemini-2.0-flash"]
DEFAULT_ANALYST = "anthropic/claude-haiku-4.5"

app = typer.Typer(add_completion=False, help="Ask N models one question and synthesize one answer.")


def build_config(
    question: str,
    *,
    models: tuple[str, ...],
    analyst_model: str | None,
    synth_model: str | None,
    config_path: str | None,
) -> FusionConfig:
    """File defaults, then command-line overrides, then the built-in defaults."""

    data = merge_config(
        load_config_file(config_path),
        {
            "question": question or None,
            "models": models,
            "analyst_model": analyst_model,
            "synth_model": synth_model,
        },
    )
    data.setdefault("models", DEFAULT_MODELS)
    data.setdefault("analyst_model", DEFAULT_ANALYST)
    data.setdefault("synth_model", data["analyst_model"])
    try:
        cfg = FusionConfig.model_validate(data)
    except ValueError as exc:
        raise CliError(f"invalid fusion config: {exc}") from exc
    if not cfg.question:
        raise CliError("no question: pass one as the argument or set it in the config file")
    if not cfg.models:
        raise CliError("no models: pass -m at least once or set `models` in the config file")
    return cfg


async def run_headless(
    cfg: FusionConfig,
    *,
    session_dir: Path | None,
    log: Path | None,
    resume: Path | None,
) -> str:
    """Run with no UI and return the synthesized answer (R-U-7)."""

    controller = await _controller(cfg, session_dir=session_dir, resume=resume)
    if log is None:
        return str(await controller.run() or "")
    async with JsonlRecorder(controller.bus, log):
        return str(await controller.run() or "")


async def run_tui(
    cfg: FusionConfig,
    *,
    session_dir: Path | None,
    log: Path | None,
    resume: Path | None,
) -> str:
    """Run under `FusionApp` (spec 9.2). Imported here so `--headless` stays cheap."""

    from workflows.fusion.app import FusionApp

    controller = await _controller(cfg, session_dir=session_dir, resume=resume)
    tui = FusionApp(controller, cfg, session_dir=session_dir, autostart=True)
    if log is not None:
        async with JsonlRecorder(controller.bus, log):
            await _run_app(tui)
    else:
        await _run_app(tui)
    return str(controller.result or "")


async def _run_app(tui: Any) -> None:
    """Hand the terminal to Textual. The app starts the run once it is subscribed."""

    await tui.run_async()


async def _controller(
    cfg: FusionConfig, *, session_dir: Path | None, resume: Path | None
) -> Controller:
    if resume is not None:
        return await Controller.load(resume, bus=EventBus())
    return controller_for(cfg, session_dir=session_dir, mode=PermissionMode.AUTO)


@app.command()
def main(
    question: Annotated[str, typer.Argument(help="The question to ask every model.")] = "",
    models: Annotated[
        list[str] | None, typer.Option("--model", "-m", help="A model id. Repeat per branch.")
    ] = None,
    analyst_model: Annotated[str | None, typer.Option(help="Model for the analysis stage.")] = None,
    synth_model: Annotated[str | None, typer.Option(help="Model for the synthesis stage.")] = None,
    config: Annotated[str | None, typer.Option(help="JSON config file.")] = None,
    session_dir: Annotated[Path | None, typer.Option(help="Where to checkpoint the run.")] = None,
    resume: Annotated[Path | None, typer.Option(help="Resume a saved session.json.")] = None,
    log: Annotated[Path | None, typer.Option(help="Write the event stream as JSONL.")] = None,
    headless: Annotated[bool, typer.Option("--headless", help="No TUI (R-U-7).")] = False,
) -> None:
    """Run the fusion workflow."""

    try:
        cfg = build_config(
            question,
            models=tuple(models or ()),
            analyst_model=analyst_model,
            synth_model=synth_model,
            config_path=config,
        )
        if cfg.fake is None:
            require_api_key()
        prepare_process(session_dir)
        runner = run_headless if headless else run_tui
        answer = run_sync(runner(cfg, session_dir=session_dir, log=log, resume=resume))
    except CliError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(2) from exc
    if headless:
        typer.echo(answer)


def entrypoint() -> None:
    """Console-script target."""

    app()


if __name__ == "__main__":  # pragma: no cover - exercised through `python -m`
    entrypoint()


__all__ = ["DEFAULT_ANALYST", "DEFAULT_MODELS", "app", "build_config", "entrypoint", "main"]
