"""The fusion reference workflow, headless (spec 9.2, M5 exit test).

The other three exit criteria are checked on purpose-built graphs in
`test_graph_resume.py`, where a node can be stopped mid-body. This file checks the
one that has to be the *real* workflow: fusion, built by its own `build(config)`,
resolved from the session by import path, run with no TUI attached.

Every model is a `FakeProvider` with its own script. Spec C-1: "the same final
output" is only a checkable clause under a fake, and a fan-out sharing one script
would interleave (M5 trap 5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from azalabscode import (
    Controller,
    ModelCallStarted,
    NodeStarted,
    RunState,
    Session,
    WorkflowNotImportable,
)
from tests.graphrig import attach, build_graph_rig
from workflows.fusion.headless import controller_for, run_headless
from workflows.fusion.workflow import (
    CONFIG_TYPE,
    IMPORT_PATH,
    FusionConfig,
    build,
)

MODELS = ["openai/gpt-4o", "anthropic/claude-haiku-4.5", "google/gemini-2.5-flash"]
FAKE = {
    "openai/gpt-4o": "Rayleigh scattering.",
    "anthropic/claude-haiku-4.5": "Short wavelengths scatter more.",
    "google/gemini-2.5-flash": "The atmosphere scatters blue light.",
    "x/analyst": "All three agree on scattering; only one names Rayleigh.",
    "x/synth": "Sunlight scatters off air molecules, and short blue wavelengths scatter most.",
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
# The graph
# ---------------------------------------------------------------------------


def test_the_graph_is_spec_6_3s_example() -> None:
    graph = build(config()).compile()
    assert graph.order == ("models", "join", "analyze", "synthesize")
    assert sorted(graph.node_ids()) == [
        "analyze",
        "join",
        "models",
        "models/claude-haiku-4-5",
        "models/gemini-2-5-flash",
        "models/gpt-4o",
        "synthesize",
    ]


def test_branch_ids_come_from_the_model_name_not_its_index() -> None:
    """An index shifts when a model is added to the middle of the list."""

    reordered = config(models=list(reversed(MODELS)))
    assert build(reordered).compile().node_ids() == build(config()).compile().node_ids()


def test_duplicate_model_names_are_disambiguated() -> None:
    duplicated = config(models=["a/same", "b/same"])
    assert [slug for slug, _ in duplicated.branches()] == ["same", "same-1"]


def test_the_graph_hash_survives_a_prompt_edit() -> None:
    edited = config(branch_prompt="something else entirely")
    assert build(edited).graph_hash() == build(config()).graph_hash()


def test_the_graph_hash_changes_when_a_model_is_added() -> None:
    wider = config(models=[*MODELS, "meta/llama-4"])
    assert build(wider).graph_hash() != build(config()).graph_hash()


def test_build_accepts_a_plain_dict() -> None:
    """`load()` hands `build` a validated config; a dict must work too (R-W-1)."""

    assert build(config().model_dump(mode="json")).compile().order[0] == "models"


def test_an_unknown_config_key_is_refused() -> None:
    with pytest.raises(ValueError):
        FusionConfig.model_validate({"question": "q", "nonsense": 1})


# ---------------------------------------------------------------------------
# Headless
# ---------------------------------------------------------------------------


async def test_fusion_runs_headless() -> None:
    """The M5 exit criterion. No TUI, no network, one answer out."""

    answer = await run_headless(config(), timeout=15)
    assert answer == FAKE["x/synth"]


async def test_every_stage_runs_and_the_branches_stream_under_their_node_ids() -> None:
    rig = build_graph_rig(
        build(config()), import_path=IMPORT_PATH, config=config().model_dump(mode="json")
    )
    try:
        assert await rig.controller.run(timeout=15) == FAKE["x/synth"]
        assert rig.controller.state is RunState.COMPLETED

        started = [e.node_id for e in rig.of_type(NodeStarted) if e.node_class]
        assert started == [
            "models",
            "models/gpt-4o",
            "models/claude-haiku-4-5",
            "models/gemini-2-5-flash",
            "join",
            "analyze",
            "synthesize",
        ]
        # A branch is routable by `agent_id` without being an agent (R-U-3).
        calls = {e.agent_id for e in rig.of_type(ModelCallStarted)}
        assert calls == {
            "models/gpt-4o",
            "models/claude-haiku-4-5",
            "models/gemini-2-5-flash",
            "analyze",
            "synthesize",
        }
        assert rig.controller.agents == {}  # fusion has no agents at all
    finally:
        await rig.aclose()


async def test_the_analyst_sees_every_branch_answer() -> None:
    """The join is what the fan-out is for; a wrong `Gather` would be invisible
    in the final answer, because the synthesizer is scripted."""

    graph = build(config()).compile()
    rig = build_graph_rig(graph)
    try:
        await rig.controller.run(timeout=15)
        analyst = graph.entry("analyze").node
        prompt = analyst.provider.requests[0].messages[-1].text  # type: ignore[attr-defined]
        for text in (FAKE[model] for model in MODELS):
            assert text in prompt
    finally:
        await rig.aclose()


async def test_the_synthesizer_sees_the_answers_and_the_analysis() -> None:
    graph = build(config()).compile()
    rig = build_graph_rig(graph)
    try:
        await rig.controller.run(timeout=15)
        synth = graph.entry("synthesize").node
        prompt = synth.provider.requests[0].messages[-1].text  # type: ignore[attr-defined]
        assert FAKE["x/analyst"] in prompt
        assert FAKE["openai/gpt-4o"] in prompt
    finally:
        await rig.aclose()


# ---------------------------------------------------------------------------
# Session, import path, resume
# ---------------------------------------------------------------------------


async def test_the_session_can_rebuild_fusion_from_its_import_path(tmp_path: Path) -> None:
    """R-W-1 end to end: the file names the factory, and the factory resolves."""

    controller = controller_for(config(), session_dir=tmp_path / "run")
    try:
        await controller.run(timeout=15)
    finally:
        controller.bus.close()

    saved = tmp_path / "run" / "session.json"
    session = Session.load(saved)
    assert session.workflow.import_path == IMPORT_PATH
    assert session.workflow.config_type == CONFIG_TYPE
    assert session.workflow.graph_hash == build(config()).graph_hash()

    rebuilt = session.workflow.resolve()(session.workflow.validated_config())
    assert rebuilt.graph_hash() == session.workflow.graph_hash


async def test_a_paused_fusion_run_resumes_from_its_file(tmp_path: Path) -> None:
    """Spec 9.2's own demonstration of R-W-6, on the workflow it describes.

    Branch concurrency is 1 so the pause lands deterministically: whenever it
    arrives, at most the branch in flight can still finish, and the ones behind it
    on the semaphore have not started. Concurrency is what fusion is *for*; it is
    turned down here so the assertion is about the memo and not about the clock.
    """

    session_dir = tmp_path / "run"
    cfg = config(fake_chunk_delay_s=0.02, branch_concurrency=1)
    rig = build_graph_rig(
        build(cfg),
        session_dir=session_dir,
        import_path=IMPORT_PATH,
        config=cfg.model_dump(mode="json"),
    )
    try:
        await rig.controller.start()
        await rig.wait_type(ModelCallStarted)
        await rig.controller.pause()
        assert await rig.wait_state(RunState.PAUSED) is RunState.PAUSED

        saved = await rig.controller.save(timeout=10)
        session = Session.load(saved)
        done = [n for n in session.completed_nodes() if n.startswith("models/")]
        missing = [
            node_id
            for node_id in build(cfg).compile().node_ids()
            if node_id.startswith("models/") and node_id not in session.nodes
        ]
        assert len(done) >= 1, "nothing completed: the pause landed before any branch"
        assert len(missing) >= 1, "everything ran: the pause did not land mid-fan-out"
    finally:
        await rig.aclose()

    loaded = await Controller.load(saved)
    events, sub, pump = attach(loaded)
    try:
        assert loaded.graph is not None
        await loaded.resume()
        assert await loaded.wait(timeout=15) == FAKE["x/synth"]

        # The branches the session already had were never asked for again: their
        # providers in *this* process have seen no requests at all (R-W-6).
        for node_id in done:
            node = loaded.graph.entry(node_id).node
            assert node.provider.requests == []  # type: ignore[attr-defined]
        for node_id in missing:
            node = loaded.graph.entry(node_id).node
            assert len(node.provider.requests) == 1  # type: ignore[attr-defined]
        assert [e.node_id for e in events if isinstance(e, NodeStarted)].count("analyze") == 1
    finally:
        sub.unsubscribe()
        await pump
        loaded.bus.close()


async def test_a_session_naming_a_missing_module_fails_at_load(tmp_path: Path) -> None:
    """R-C-11: with the path and the reason, not five lines into `build`."""

    controller = controller_for(config(), session_dir=tmp_path / "run")
    try:
        await controller.run(timeout=15)
    finally:
        controller.bus.close()

    saved = tmp_path / "run" / "session.json"
    session = Session.load(saved)
    session.workflow.import_path = "workflows.fusion.nowhere:build"
    saved.write_bytes(session.dumps())

    with pytest.raises(WorkflowNotImportable, match="nowhere"):
        await Controller.load(saved)
