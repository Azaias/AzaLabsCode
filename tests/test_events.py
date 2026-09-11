"""The event bus: fan-out, backpressure, coalescing, and the JSONL recorder.

Spec 4.4 sets the policy this file tests: bounded per-subscriber queues; under
pressure `ModelDelta` coalesces into the tail and is otherwise dropped and counted;
lifecycle events are never dropped, publishing blocks instead. A slow UI may lose
streaming text. It may not lose the fact that a tool ran.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from azalabscode.events import (
    EventBus,
    EventsDropped,
    JsonlRecorder,
    ModelCallStarted,
    ModelDelta,
    NodeStarted,
    RunStateChanged,
    Subscription,
    ToolCallStarted,
    delta_from_part,
)
from azalabscode.runstate import RunState


async def drain(sub: Subscription, limit: int = 1000) -> list[Any]:
    """Pull everything currently queued without waiting for more."""

    out: list[Any] = []
    while len(sub) and len(out) < limit:
        event = await sub.get()
        if event is None:
            break
        out.append(event)
    return out


async def next_event(sub: Subscription) -> Any:
    """The next event, asserting one arrives. Keeps the type checker out of the tests."""

    event = await sub.get()
    assert event is not None
    return event


# ---------------------------------------------------------------------------
# Fan-out and sequencing
# ---------------------------------------------------------------------------


async def test_every_subscriber_sees_every_event() -> None:
    bus = EventBus(run_id="r1")
    a = bus.subscribe(name="a")
    b = bus.subscribe(name="b")

    await bus.publish(NodeStarted(node_id="n1"))
    await bus.publish(ToolCallStarted(call_id="c", tool="grep"))

    assert [e.type for e in await drain(a)] == ["node_started", "tool_call_started"]
    assert [e.type for e in await drain(b)] == ["node_started", "tool_call_started"]


async def test_sequence_numbers_are_monotonic_and_run_id_is_stamped() -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe()
    for _ in range(5):
        await bus.publish(NodeStarted())
    events = await drain(sub)
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    assert all(e.run_id == "r1" for e in events)


async def test_seeding_continues_a_reloaded_runs_numbering() -> None:
    """On resume the bus is seeded from `Session.event_seq`, so a reloaded run's log
    continues rather than restarting at zero and colliding with what is on disk."""

    bus = EventBus(run_id="r1")
    bus.seed_seq(400)
    sub = bus.subscribe()
    await bus.publish(NodeStarted())
    assert (await next_event(sub)).seq == 401


async def test_an_emitter_stamps_agent_and_node_ids() -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe()
    emitter = bus.emitter(agent_id="main", node_id="root/agent")

    await emitter.emit(ModelCallStarted(call_id="c", model="x/y"))
    child = emitter.bind(agent_id="main/0")
    await child.emit(ModelCallStarted(call_id="d", model="x/y"))

    events = await drain(sub)
    assert (events[0].agent_id, events[0].node_id) == ("main", "root/agent")
    assert (events[1].agent_id, events[1].node_id) == ("main/0", "root/agent")


async def test_an_explicit_id_on_the_event_wins_over_the_emitters() -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe()
    await bus.emitter(agent_id="main").emit(
        ModelCallStarted(call_id="c", model="x", agent_id="other")
    )
    assert (await next_event(sub)).agent_id == "other"


async def test_unsubscribing_stops_delivery() -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe()
    sub.unsubscribe()
    await bus.publish(NodeStarted())
    assert await sub.get() is None


async def test_closing_the_bus_ends_iteration_after_the_backlog_drains() -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe()
    await bus.publish(NodeStarted())
    await bus.publish(NodeStarted())
    bus.close()

    seen = [event async for event in sub]
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


async def test_deltas_coalesce_into_the_tail_when_a_queue_is_full() -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe(maxsize=2)

    await bus.publish(NodeStarted())
    await bus.publish(ModelDelta(call_id="c", text="a"))
    assert sub.full
    await bus.publish(ModelDelta(call_id="c", text="b"))
    await bus.publish(ModelDelta(call_id="c", text="c"))

    events = await drain(sub)
    assert events[0].type == "node_started"
    assert events[1].text == "abc", "adjacent deltas merged rather than dropped"
    assert sub.dropped_total == 0


async def test_deltas_for_different_calls_do_not_coalesce() -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe(maxsize=1)

    await bus.publish(ModelDelta(call_id="c1", text="a"))
    await bus.publish(ModelDelta(call_id="c2", text="b"))

    assert sub.dropped_total == 1, "a different call is not an adjacent delta"


async def test_text_and_reasoning_deltas_do_not_coalesce_into_each_other() -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe(maxsize=1)

    await bus.publish(ModelDelta(call_id="c", text="visible"))
    await bus.publish(ModelDelta(call_id="c", reasoning="hidden"))

    assert sub.dropped_total == 1
    first = await next_event(sub)
    assert first.text == "visible"
    assert first.reasoning is None


async def test_tool_call_argument_deltas_coalesce_per_index() -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe(maxsize=1)

    await bus.publish(ModelDelta(call_id="c", tool_call_index=0, tool_call_delta='{"a"'))
    await bus.publish(ModelDelta(call_id="c", tool_call_index=0, tool_call_delta=":1}"))
    await bus.publish(ModelDelta(call_id="c", tool_call_index=1, tool_call_delta="{}"))

    assert sub.dropped_total == 1, "a different slot is not an adjacent delta"
    assert (await next_event(sub)).tool_call_delta == '{"a":1}'


async def test_a_dropped_delta_is_reported_once_the_subscriber_drains() -> None:
    """The consumer must learn it lost data; that is what `EventsDropped` is for."""

    bus = EventBus(run_id="r1")
    sub = bus.subscribe(maxsize=1)

    await bus.publish(ModelDelta(call_id="c", text="kept"))
    await bus.publish(ModelDelta(call_id="c2", text="lost"))
    await bus.publish(ModelDelta(call_id="c3", text="also lost"))
    assert sub.dropped_total == 2

    first = await next_event(sub)
    assert first.text == "kept"
    marker = await next_event(sub)
    assert isinstance(marker, EventsDropped)
    assert marker.count == 2


async def test_a_lifecycle_event_blocks_rather_than_being_dropped() -> None:
    """A slow UI may lose streaming text; it may not lose the fact that a tool ran."""

    bus = EventBus(run_id="r1")
    sub = bus.subscribe(maxsize=1)
    await bus.publish(ModelDelta(call_id="c", text="filling the queue"))
    assert sub.full

    publish = asyncio.create_task(bus.publish(ToolCallStarted(call_id="c", tool="shell")))
    await asyncio.sleep(0.02)
    assert not publish.done(), "publish should be waiting for room"

    await sub.get()
    await asyncio.wait_for(publish, timeout=1.0)
    assert (await next_event(sub)).type == "tool_call_started"
    assert sub.dropped_total == 0


async def test_a_blocked_publish_unblocks_when_the_subscriber_goes_away() -> None:
    """Otherwise one abandoned subscriber wedges the whole run."""

    bus = EventBus(run_id="r1")
    sub = bus.subscribe(maxsize=1)
    await bus.publish(ModelDelta(call_id="c", text="x"))

    publish = asyncio.create_task(bus.publish(NodeStarted()))
    await asyncio.sleep(0.02)
    sub.unsubscribe()
    await asyncio.wait_for(publish, timeout=1.0)


async def test_a_slow_subscriber_does_not_stall_a_fast_one() -> None:
    bus = EventBus(run_id="r1")
    fast = bus.subscribe(maxsize=100)
    slow = bus.subscribe(maxsize=2)

    for i in range(20):
        await bus.publish(ModelDelta(call_id="c", text=str(i)))

    assert len(await drain(fast)) == 20
    assert len(slow) <= 2


def test_publish_nowait_never_blocks() -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe(maxsize=1)
    bus.publish_nowait(ModelDelta(call_id="c", text="a"))
    bus.publish_nowait(NodeStarted())
    assert sub.dropped_total == 1


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------


async def test_the_jsonl_recorder_writes_and_reads_back_typed_events(tmp_path: Path) -> None:
    bus = EventBus(run_id="r1")
    path = tmp_path / "logs" / "events.jsonl"

    async with bus.record(path) as recorder:
        await bus.publish(RunStateChanged(old=RunState.CREATED, new=RunState.RUNNING))
        await bus.publish(ModelDelta(call_id="c", text="hi"))
        await bus.publish(ToolCallStarted(call_id="c", tool="grep"))
        await asyncio.sleep(0)

    assert recorder.count == 3
    events = JsonlRecorder.read(path)
    assert [e.type for e in events] == ["run_state_changed", "model_delta", "tool_call_started"]
    assert events[0].new is RunState.RUNNING
    assert [e.seq for e in events] == [1, 2, 3]


async def test_the_recorder_appends_rather_than_truncating(tmp_path: Path) -> None:
    """A resumed run continues its own log; a truncating recorder would erase it."""

    path = tmp_path / "events.jsonl"
    for _ in range(2):
        bus = EventBus(run_id="r1")
        async with bus.record(path):
            await bus.publish(NodeStarted())
            await asyncio.sleep(0)

    assert len(JsonlRecorder.read(path)) == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_delta_from_part_maps_each_part_to_its_field() -> None:
    from azalabscode.content import ReasoningPart, TextPart, ToolCallPart

    assert delta_from_part("c", TextPart(text="a")).text == "a"
    assert delta_from_part("c", ReasoningPart(text="b")).reasoning == "b"
    delta = delta_from_part("c", ToolCallPart(call_id="x", name="t", raw_arguments="{}"), index=2)
    assert delta.tool_call_index == 2
    assert delta.tool_call_delta == "{}"


def test_coalesce_key_is_none_for_an_empty_delta() -> None:
    assert ModelDelta(call_id="c").coalesce_key is None


async def test_events_are_never_dropped_when_nobody_is_behind() -> None:
    bus = EventBus(run_id="r1", default_queue_size=10_000)
    sub = bus.subscribe()
    for i in range(5000):
        await bus.publish(ModelDelta(call_id="c", text=str(i)))
    assert sub.dropped_total == 0
    assert len(sub) == 5000


@pytest.mark.parametrize("maxsize", [1, 2, 8])
async def test_the_queue_never_exceeds_its_bound(maxsize: int) -> None:
    bus = EventBus(run_id="r1")
    sub = bus.subscribe(maxsize=maxsize)
    for i in range(200):
        await bus.publish(ModelDelta(call_id=f"c{i}", text="x"))
    assert len(sub) <= maxsize
