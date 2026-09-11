"""R-A-2: model fusion, end to end, with its TUI.

M5 proved the graph half of R-A-2 in `test_fusion_workflow.py` -- fan-out, join,
analyse, synthesize, and a mid-fan-out pause/save/load/resume that re-runs only the
incomplete branch. This file is the half M6 owes: the panes, the stage pipeline, the
restore of completed branches after a load, the interrupt target a fan-out needs
(spec C-4), and `--headless`.

Every model is a `FakeProvider` with its own script -- one per branch, because a
shared script hands out turns in whichever order the branches are scheduled (M5 trap
5) and the assertion here is per pane.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from azalabscode import Controller, ModelCallStarted, RunState, Session
from workflows.fusion.app import FusionApp
from workflows.fusion.cli import app as fusion_cli
from workflows.fusion.headless import controller_for
from workflows.fusion.workflow import FusionConfig

BOUND = 15.0

MODELS = ["openai/gpt-4o", "anthropic/claude-haiku-4.5"]
FAKE = {
    "openai/gpt-4o": "Rayleigh scattering.",
    "anthropic/claude-haiku-4.5": "Short wavelengths scatter more.",
    "x/analyst": "Both agree on scattering.",
    "x/synth": "Sunlight scatters off air molecules.",
}


def config(**overrides: Any) -> FusionConfig:
    """The offline fusion config every test here starts from."""

    payload: dict[str, Any] = {
        "question": "why is the sky blue?",
        "models": list(MODELS),
        "analyst_model": "x/analyst",
        "synth_model": "x/synth",
        "fake": dict(FAKE),
    }
    payload.update(overrides)
    return FusionConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# The layout (spec 9.2)
# ---------------------------------------------------------------------------


async def test_every_branch_gets_its_own_pane_and_every_stage_a_chevron() -> None:
    """One pane per model, side by side, each fed by its own node id (R-U-3)."""

    cfg = config()
    controller = controller_for(cfg)
    tui = FusionApp(controller, cfg)
    async with tui.run_test() as pilot:
        assert await controller.run(timeout=BOUND) == FAKE["x/synth"]
        await pilot.pause()

        panes = tui.panes_text()
        assert panes["models/gpt-4o"].strip() == FAKE["openai/gpt-4o"]
        assert panes["models/claude-haiku-4-5"].strip() == FAKE["anthropic/claude-haiku-4.5"]
        assert panes["analyze"].strip() == FAKE["x/analyst"]
        assert panes["synthesize"].strip() == FAKE["x/synth"]

        pipeline = tui.pipeline
        assert pipeline.state_of("models") == "ok"
        assert pipeline.state_of("analyze") == "ok"
        assert pipeline.state_of("synthesize") == "ok"
        # The fan-out is one chevron with a counter, not one chevron per branch.
        assert "2/2" in pipeline.render_line_text()
    controller.bus.close()


async def test_the_visible_stage_follows_the_run_until_a_stage_is_picked() -> None:
    """Spec 9.2: "each stage visible as it runs", and then the user's choice wins."""

    cfg = config()
    controller = controller_for(cfg)
    tui = FusionApp(controller, cfg)
    async with tui.run_test() as pilot:
        await controller.run(timeout=BOUND)
        await pilot.pause()
        # Synthesize started last, so it is what is on screen.
        assert tui.stage_panes["synthesize"].display is True
        assert tui.stage_panes["analyze"].display is False

        tui.pipeline.select("analyze")
        await pilot.pause()
        assert tui.pinned_stage == "analyze"
        assert tui.stage_panes["analyze"].display is True
    controller.bus.close()


async def test_a_fan_out_interrupt_has_a_target(tmp_path: Path) -> None:
    """Spec C-4: a targetless `escape` in a fan-out cancels nothing.

    Fusion has no agents at all, so the target is a *node* id -- which works because
    a `ModelCall`'s step is registered under its node id.
    """

    cfg = config(fake_chunk_delay_s=0.05, branch_concurrency=1)
    controller = controller_for(cfg, session_dir=tmp_path / "run")
    tui = FusionApp(controller, cfg)
    async with tui.run_test() as pilot:
        assert tui.interrupt_target() == "models/gpt-4o"  # nothing finished yet
        await controller.start()
        await controller.wait(timeout=BOUND)
        await pilot.pause()
        # Everything finished: there is no branch left to interrupt.
        assert tui.interrupt_target() is None
    controller.bus.close()


# ---------------------------------------------------------------------------
# Save, load, and the panes that survive it (R-A-2)
# ---------------------------------------------------------------------------


async def test_completed_branches_are_restored_into_their_panes(tmp_path: Path) -> None:
    """R-A-2: "panes for completed models are restored from session".

    There are no events to rebuild them from after a load -- the deltas belong to the
    process that wrote the file -- so the pane is filled from the memoized node
    output, and the branch that had not finished is left empty for the resume to
    stream into.
    """

    session_dir = tmp_path / "run"
    cfg = config(fake_chunk_delay_s=0.02, branch_concurrency=1)
    controller = controller_for(cfg, session_dir=session_dir)
    await controller.start()
    await _wait_for(controller, ModelCallStarted)
    await controller.pause()
    assert await controller.wait_for_state(RunState.PAUSED, timeout=BOUND) is RunState.PAUSED
    saved = await controller.save(timeout=BOUND)
    controller.bus.close()

    session = Session.load(saved)
    done = [node for node in session.completed_nodes() if node.startswith("models/")]
    missing = [f"models/{slug}" for slug, _ in cfg.branches() if f"models/{slug}" not in done]
    assert done, "the pause landed before any branch completed"
    assert missing, "the pause did not land mid-fan-out"

    loaded = await Controller.load(saved)
    tui = FusionApp(loaded, cfg)
    async with tui.run_test() as pilot:
        await pilot.pause()
        for node_id in done:
            assert tui.branch_panes[node_id].text.strip()
        for node_id in missing:
            assert tui.branch_panes[node_id].text == ""

        await loaded.resume()
        assert await loaded.wait(timeout=BOUND) == FAKE["x/synth"]
        await pilot.pause()
        panes = tui.panes_text()
        for node_id in [*done, *missing]:
            assert panes[node_id].strip(), f"{node_id} is empty after the resume"
    loaded.bus.close()


async def _wait_for(controller: Controller, kind: type) -> None:
    """Wait for one event class on a fresh subscription. Bounded."""

    sub = controller.bus.subscribe(name="wait")
    try:
        async with asyncio.timeout(BOUND):
            async for event in sub:
                if isinstance(event, kind):
                    return
    finally:
        sub.unsubscribe()


# ---------------------------------------------------------------------------
# --headless (R-U-7)
# ---------------------------------------------------------------------------


def test_fusion_runs_headless_from_the_cli(tmp_path: Path) -> None:
    """R-U-7: the same controller, the same graph, nothing subscribed to it."""

    config_path = tmp_path / "fusion.json"
    config_path.write_text(json.dumps(config().model_dump(mode="json")), encoding="utf-8")
    log = tmp_path / "events.jsonl"

    result = CliRunner().invoke(
        fusion_cli,
        ["--headless", "--config", str(config_path), "--log", str(log)],
    )
    assert result.exit_code == 0, result.output
    assert FAKE["x/synth"] in result.output
    assert log.exists() and log.read_text(encoding="utf-8").count("\n") > 10


def test_the_cli_refuses_a_run_it_cannot_configure(tmp_path: Path) -> None:
    """A missing question is a message and exit 2, not a traceback."""

    result = CliRunner().invoke(fusion_cli, ["--headless", "-m", "a/b"])
    assert result.exit_code == 2
    assert "no question" in result.output
