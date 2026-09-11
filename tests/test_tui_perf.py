"""R-U-4: the perf test that decides whether the TUI stays Textual (plan D5).

Spec 8.3 states the shape: "runs 4 `FakeProvider` streams at 200 tokens/s each
under `App.run_test()` and asserts frame time". R-U-4 states the budget: streaming
text renders incrementally with **<= 100 ms latency from event to screen at >= 4
concurrent streams**, with delta events coalesced in the UI, not in core.

Everything below the app is real: a real `Controller`, four real `AgentLoop`s, a
real `EventBus`, four real `FakeProvider`s. The only fixture is the script.

**What "event to screen" means here.** `StreamPane` stamps the `ts` of the oldest
delta it is holding and closes the interval in `render()`, which is where the
compositor pulls the widget's content. That covers the queue, the pump, the 33 ms
batch timer, Textual's own frame scheduling and the layout -- everything this
process controls. It does not cover the write to a terminal, because
`App.run_test()` is headless and there is no terminal to write to.

**Why the rate is driven by `chunk_rate_hz` and not `chunk_delay_s`** (spec delta
23): on Windows the loop clock resolves to about 15.6 ms, so a 5 ms sleep per
chunk yields roughly 70 chunks/s, not 200. Asking for a rate and sleeping to an
absolute deadline gets the requested average out of a clock that cannot hit the
interval -- and `test_the_load_generator_actually_delivers_the_rate` asserts that
it did, so a green run can never be one where the load never arrived.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll

from azalabscode.control import Controller
from azalabscode.events import EventBus, ModelDelta
from azalabscode.ids import AgentId
from azalabscode.permissions import PermissionMode
from azalabscode.providers.testing import FakeProvider, ScriptedTurn
from azalabscode.tools.context import ToolContext
from azalabscode.tools.dispatcher import ToolDispatcher
from azalabscode.tui import HarnessApp, RunStatusBar, StreamPane
from azalabscode.workflows.agent_loop import AgentLoop, AgentSpec

STREAMS = 4
"""R-U-4's ">= 4 concurrent streams" -- the fusion workflow's four models."""

RATE_HZ = 200.0
"""Spec 8.3's 200 tokens/s, per stream. 800 deltas/s across the app."""

TOKENS = 400
"""Two seconds of stream per pane. Long enough that a backlog would show up in the
tail latencies; short enough that the exit test is not a coffee break."""

LATENCY_BUDGET_MS = 100.0
"""R-U-4's budget, verbatim."""

LOOP_LAG_BUDGET_MS = 100.0
"""How long the event loop may be blocked at a stretch. This is spec 8.3's "frame
time" in the form that is actually measurable: if rendering blocks the loop, a
50 ms sampler overshoots its deadline by however long the block lasted."""

TIMEOUT_S = 60.0
"""Bound on the whole run. There is no `pytest-timeout` in this project, so a
deadlock has to be caught here or it takes the suite down with no output."""


def token_chunks(count: int) -> list[str]:
    """`count` distinct tokens, so a lost or reordered one is visible in a diff."""

    return [f"t{index:04d} " for index in range(count)]


def stream_script(count: int, rate_hz: float) -> list[ScriptedTurn]:
    """One turn that streams `count` chunks at `rate_hz` and stops."""

    return [
        ScriptedTurn(
            text_chunks=token_chunks(count),
            chunk_rate_hz=rate_hz,
            finish_reason="stop",
        )
    ]


class FusionApp(HarnessApp):
    """Four panes side by side: the layout R-U-4 names, at the width it names."""

    CSS = """
    .pane {
        width: 1fr;
        border: solid $primary;
    }
    """

    def __init__(
        self,
        controller: Controller,
        agent_ids: list[str],
        *,
        flush_interval_s: float = 1 / 30,
        tail: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(controller, **kwargs)  # type: ignore[arg-type]
        self.agent_ids = agent_ids
        self.flush_interval_s = flush_interval_s
        self.tail = tail

    def compose(self) -> ComposeResult:
        """N-up stream panes over a status bar.

        `tail=True` is the fusion layout: a fixed pane per model showing that
        model's live output. `tail=False` puts each pane in a scroll container and
        lets it grow, which is the stacked shape a `Transcript` uses and the
        expensive one -- both are measured, because shipping only the cheap
        configuration and calling R-U-4 green would be a test of the fixture.
        """

        with Horizontal():
            for agent_id in self.agent_ids:
                pane = StreamPane(
                    agent_id=agent_id,
                    title=agent_id,
                    tail=self.tail,
                    flush_interval_s=self.flush_interval_s,
                )
                if self.tail:
                    pane.add_class("pane")
                    yield pane
                else:
                    with VerticalScroll(classes="pane"):
                        yield pane
        yield RunStatusBar(self.controller, id="status-bar")

    @property
    def panes(self) -> list[StreamPane]:
        """The four panes, in layout order."""

        return list(self.query(StreamPane))


def build_fanout(
    workspace: Path,
    *,
    streams: int = STREAMS,
    tokens: int = TOKENS,
    rate_hz: float = RATE_HZ,
) -> tuple[Controller, list[str], list[FakeProvider]]:
    """A controller whose body runs `streams` sibling agent loops concurrently.

    Each stream gets its own `FakeProvider`. Sharing one would be a bug the M3
    handoff already names: unkeyed turns are handed out in order across the whole
    run, not per agent, so four streams sharing a script would interleave.
    """

    agent_ids = [f"model{index}" for index in range(streams)]
    providers = [FakeProvider(stream_script(tokens, rate_hz)) for _ in agent_ids]
    bus = EventBus()
    controller = Controller(
        run_id="run_perf",
        bus=bus,
        permission_mode=PermissionMode.AUTO,
    )
    dispatcher = ToolDispatcher(
        [],
        context=ToolContext(workspace_root=workspace),
        gate=controller.gate,
        emitter=bus.emitter(),
    )

    async def body(control: object) -> list[str]:
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(
                    AgentLoop(
                        AgentSpec(name=agent_id, model="fake/model", system_prompt="stream"),
                        provider=provider,
                        dispatcher=dispatcher,
                        control=control,  # type: ignore[arg-type]
                        emitter=control.emitter_for(agent_id),  # type: ignore[attr-defined]
                        agent_id=AgentId(agent_id),
                    ).run("stream"),
                    name=f"agent:{agent_id}",
                )
                for agent_id, provider in zip(agent_ids, providers, strict=True)
            ]
        return [task.result().final_text for task in tasks]

    controller.set_body(body)
    return controller, agent_ids, providers


async def sample_loop_lag(stop: asyncio.Event, out: list[float], interval: float = 0.05) -> None:
    """Record how far each 50 ms sleep overshot its deadline.

    A loop that is busy rendering cannot service this task, so the overshoot is a
    direct measure of the longest stretch the loop was blocked -- which is what
    "frame time" is asking about from the user's side.

    The clock is `loop.time()`, not `perf_counter()`. They are different clocks and
    the skew between them is enough to make an unloaded run report a *negative*
    mean overshoot, which is nonsense and would make any comparison against it
    vacuous.
    """

    clock = asyncio.get_running_loop().time
    while not stop.is_set():
        due = clock() + interval
        await asyncio.sleep(interval)
        out.append((clock() - due) * 1000.0)


async def run_fanout(
    workspace: Path,
    *,
    flush_interval_s: float = 1 / 30,
    tokens: int = TOKENS,
    rate_hz: float = RATE_HZ,
    streams: int = STREAMS,
    tail: bool = True,
) -> tuple[FusionApp, list[StreamPane], list[float], float, list[object]]:
    """Drive the whole thing and hand back what the assertions need."""

    with frozen_heap():
        return await _run_fanout(
            workspace,
            flush_interval_s=flush_interval_s,
            tokens=tokens,
            rate_hz=rate_hz,
            streams=streams,
            tail=tail,
        )


@contextlib.contextmanager
def frozen_heap() -> Iterator[None]:
    """Keep the *test runner's* heap out of the measurement.

    A full suite run leaves several hundred thousand live objects behind, and a
    generation-2 collection over that heap takes 40-90 ms on this machine -- more
    than half of R-U-4's entire budget, in a single stop-the-world pause that has
    nothing to do with the TUI. Measured: the same code reports 65-85 ms in a fresh
    process and 107-139 ms after 800 tests, with the difference sitting in two
    outlier samples that line up exactly with two gen-2 collections.

    `gc.freeze()` moves everything currently live into a permanent generation that
    is never rescanned, so what remains is the allocation the app itself does --
    which is the thing under test and is *not* excluded. This narrows the
    measurement to the harness; it does not make the harness faster.

    The finding is real for a shipped app too: a long coding session accumulates a
    heap the same way, and M6's CLI should call `gc.freeze()` once after startup.
    """

    gc.collect()
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


async def _run_fanout(
    workspace: Path,
    *,
    flush_interval_s: float,
    tokens: int,
    rate_hz: float,
    streams: int,
    tail: bool,
) -> tuple[FusionApp, list[StreamPane], list[float], float, list[object]]:
    """`run_fanout` without the heap isolation. Not called directly."""

    controller, agent_ids, _ = build_fanout(
        workspace, streams=streams, tokens=tokens, rate_hz=rate_hz
    )
    app = FusionApp(controller, agent_ids, flush_interval_s=flush_interval_s, tail=tail)
    events: list[object] = []
    recorder = controller.bus.subscribe(name="perf-recorder")

    async def record() -> None:
        async for event in recorder:
            events.append(event)

    lag: list[float] = []
    stop = asyncio.Event()

    async with app.run_test(size=(120, 40)) as pilot:
        pump = asyncio.create_task(record())
        sampler = asyncio.create_task(sample_loop_lag(stop, lag))
        started = time.perf_counter()
        await controller.start()
        await controller.wait(timeout=TIMEOUT_S)
        elapsed = time.perf_counter() - started
        stop.set()
        await sampler
        # One more frame so the last flush is painted before anything is read.
        await pilot.pause()
        panes = app.panes
        recorder.unsubscribe()
        await pump

    return app, panes, lag, elapsed, events


# ---------------------------------------------------------------------------
# The exit test
# ---------------------------------------------------------------------------


async def test_four_concurrent_streams_render_within_the_latency_budget(
    workspace: Path,
) -> None:
    """R-U-4. The go/no-go on Textual (plan D5, spec C-13)."""

    app, panes, lag, elapsed, events = await run_fanout(workspace)

    assert len(panes) == STREAMS
    expected = "".join(token_chunks(TOKENS))

    for pane in panes:
        # Nothing lost, nothing reordered: the pane holds the whole stream.
        assert pane.text == expected, f"{pane.agent_filter} lost or reordered text"
        assert pane.stats.deltas == TOKENS
        assert pane.stats.dropped == 0, "core dropped deltas; R-U-4 says the UI coalesces"
        assert pane.stats.paints > 0, "the widget never reached the compositor"

        # Coalescing happened in the UI. 200 deltas/s against a 30 fps timer is
        # about 6-7 per write; anything near 1.0 means the batching is not working.
        assert pane.stats.flushes < pane.stats.deltas
        assert pane.stats.coalescing_ratio >= 3.0, pane.stats.coalescing_ratio

        # The budget.
        assert pane.stats.max_latency_ms <= LATENCY_BUDGET_MS, (
            f"{pane.agent_filter}: max {pane.stats.max_latency_ms:.1f} ms, "
            f"p95 {pane.stats.percentile(0.95):.1f} ms over {pane.stats.paints} paints"
        )

    # The loop was never blocked long enough for a keystroke to feel late.
    assert max(lag) <= LOOP_LAG_BUDGET_MS, f"worst loop lag {max(lag):.1f} ms"

    # And the load was real: 4 streams x 200 deltas/s actually reached the bus.
    deltas = sum(1 for event in events if isinstance(event, ModelDelta))
    assert deltas == STREAMS * TOKENS
    assert deltas / elapsed >= 0.85 * RATE_HZ * STREAMS, (
        f"only {deltas / elapsed:.0f} deltas/s were generated; the test did not load anything"
    )

    print(
        f"\nR-U-4: {STREAMS} streams x {TOKENS} tokens in {elapsed:.2f}s "
        f"({deltas / elapsed:.0f} deltas/s) | "
        + " | ".join(
            f"{p.agent_filter} max {p.stats.max_latency_ms:.1f}ms "
            f"p95 {p.stats.percentile(0.95):.1f}ms x{p.stats.coalescing_ratio:.1f}"
            for p in panes
        )
        + f" | loop lag max {max(lag):.1f}ms"
    )
    assert app.events_seen > 0


async def test_the_load_generator_actually_delivers_the_rate(workspace: Path) -> None:
    """A green exit test must not be a test that never applied any load.

    `chunk_delay_s` cannot express 200/s on this platform (spec delta 23); this
    asserts the replacement does, end to end through the provider and the loop.
    """

    _, panes, _, elapsed, events = await run_fanout(workspace, tokens=200)

    deltas = sum(1 for event in events if isinstance(event, ModelDelta))
    per_stream = deltas / STREAMS / elapsed
    assert per_stream >= 0.85 * RATE_HZ, f"{per_stream:.0f} deltas/s per stream"
    assert all(pane.stats.deltas == 200 for pane in panes)


async def test_the_latency_measurement_is_not_vacuous(workspace: Path) -> None:
    """Break the batching and the budget must fail.

    A perf assertion that cannot fail is a decoration. Widening the flush interval
    from 33 ms to 400 ms leaves everything else identical -- same load, same
    measurement, same panes -- so a breach here is proof the number in the exit
    test is measuring the pipeline and not a constant.
    """

    _, panes, _, _, _ = await run_fanout(workspace, flush_interval_s=0.4, tokens=200)

    worst = max(pane.stats.max_latency_ms for pane in panes)
    assert worst > LATENCY_BUDGET_MS, (
        f"a 400 ms batch timer produced {worst:.1f} ms, which means the measurement "
        "is not sensitive to the thing it exists to measure"
    )


@pytest.mark.parametrize("streams", [6])
async def test_there_is_no_cliff_just_past_the_four_streams_required(
    workspace: Path, streams: int
) -> None:
    """R-U-4 says ">= 4". Four passing and six collapsing would be a cliff worth
    knowing about before M6 builds a fusion TUI on top of it."""

    _, panes, lag, _, _ = await run_fanout(workspace, streams=streams, tokens=200)

    assert len(panes) == streams
    worst = max(pane.stats.max_latency_ms for pane in panes)
    assert worst <= LATENCY_BUDGET_MS, f"{streams} streams: max {worst:.1f} ms"
    assert max(lag) <= LOOP_LAG_BUDGET_MS


# ---------------------------------------------------------------------------
# The expensive mode: panes whose height follows their content
# ---------------------------------------------------------------------------


async def test_a_single_growing_transcript_pane_meets_the_budget(workspace: Path) -> None:
    """The coding agent's shape: one auto-height pane streaming inside a scroll.

    This is the configuration `Transcript` actually produces: one pane growing at a
    time, inside a scroll container, asking Textual for a layout on every write.
    """

    _, panes, lag, _, _ = await run_fanout(workspace, streams=1, tokens=400, tail=False)

    pane = panes[0]
    assert pane.stats.deltas == 400
    assert pane.stats.max_latency_ms <= LATENCY_BUDGET_MS, (
        f"one growing pane: max {pane.stats.max_latency_ms:.1f} ms, "
        f"p95 {pane.stats.percentile(0.95):.1f} ms"
    )
    assert max(lag) <= LOOP_LAG_BUDGET_MS


async def test_asking_for_a_layout_is_what_costs(workspace: Path) -> None:
    """Measure the cost `StreamPane.tail` exists to avoid.

    Kept as a test rather than a note because the number is the justification for
    the mode, and a justification nobody re-runs is a number that quietly stops
    being true. If this fails, `tail` has stopped earning its complexity.

    Six panes, not four. At four the two modes measure about 46 ms and 53 ms --
    both comfortable, and too close to tell apart reliably. The layout cost grows
    faster than the pane count, so six is where it shows: about 62 ms against
    89 ms, and a mean event-loop lag of about -3 ms against about +11 ms.
    """

    _, tail_panes, tail_lag, _, _ = await run_fanout(workspace, streams=6, tokens=TOKENS, tail=True)
    _, grow_panes, grow_lag, _, _ = await run_fanout(
        workspace, streams=6, tokens=TOKENS, tail=False
    )

    tail_worst = max(pane.stats.max_latency_ms for pane in tail_panes)
    grow_worst = max(pane.stats.max_latency_ms for pane in grow_panes)
    tail_mean_lag = sum(tail_lag) / len(tail_lag)
    grow_mean_lag = sum(grow_lag) / len(grow_lag)

    assert tail_worst <= LATENCY_BUDGET_MS
    assert grow_worst > tail_worst, "the growing mode is no longer the expensive one"
    assert grow_mean_lag - tail_mean_lag > 3.0, (
        f"growing {grow_mean_lag:.1f} ms vs tail {tail_mean_lag:.1f} ms mean loop lag: "
        "the layout is no longer blocking the loop, so StreamPane.tail has stopped "
        "earning its complexity and should go"
    )
    print(
        f"\nlayout cost at 6 panes: tail max {tail_worst:.1f} ms, mean loop lag "
        f"{tail_mean_lag:.1f} ms | growing max {grow_worst:.1f} ms, mean loop lag "
        f"{grow_mean_lag:.1f} ms"
    )
