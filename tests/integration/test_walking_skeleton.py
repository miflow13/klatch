"""Fake-backend walking skeleton: one SQLite file through pause/resume and a restart (plan Task 8, step 3).

Not marked ``ollama``: this runs in the default suite on an accelerated clock.
"""

from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from driftroom.analysis import analyze_run
from driftroom.config import load_run_config
from driftroom.domain import AgentConfig, RunConfig
from driftroom.engine import SimulationEngine
from driftroom.models.base import Decision, ModelResult
from driftroom.models.fake import FakeModelBackend
from driftroom.storage import EventStore, StoredEvent


EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "driftroom.example.toml"
WALL = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
WAIT = Decision(action="wait", message=None, target=None)
SPOKEN = [f"fake line {index}" for index in range(40)]


class RecordingFakeBackend(FakeModelBackend):
    """The fake decision queue, recording every decision it hands to the engine."""

    def __init__(self, outcomes: Sequence[Decision]) -> None:
        super().__init__(outcomes)
        self.returned: list[Decision] = []

    def decide(
        self, agent: AgentConfig, messages: Sequence[dict[str, str]], *, targets: Sequence[str] = (),
    ) -> ModelResult:
        result = super().decide(agent, messages, targets=targets)
        self.returned.append(result.decision)
        return result


_OPEN_STORES: list[EventStore] = []


def open_store(path: Path) -> EventStore:
    """An EventStore the autouse fixture closes (3.13+ warns on unclosed sqlite connections)."""
    store = EventStore(path)
    _OPEN_STORES.append(store)
    return store


@pytest.fixture(autouse=True)
def close_open_stores() -> Iterator[None]:
    try:
        yield
    finally:
        while _OPEN_STORES:
            _OPEN_STORES.pop().close()


def make_config() -> RunConfig:
    data = load_run_config(EXAMPLE_CONFIG).model_dump(mode="json")
    # Seeded scheduler (the engine builds it from random_seed) on an accelerated clock;
    # shorter silence windows so a few dozen steps cover silence and ambient events.
    data["runtime"].update({"clock_mode": "accelerated", "random_seed": 7})
    data["scheduler"].update({"silence_ambient_after_ms": 60_000, "ambient_min_interval_ms": 120_000})
    return RunConfig.model_validate(data)


def make_engine(config: RunConfig, store: EventStore, backend: FakeModelBackend,
                wall: datetime, sleeps: list[float]) -> SimulationEngine:
    return SimulationEngine(
        config, store, backend, run_id="run-1", room_id="room-1",
        wall_clock=lambda: wall, sleep_fn=sleeps.append,
    )


def all_events(store: EventStore) -> list[StoredEvent]:
    return store.read_events("run-1", limit=100_000)


def test_walking_skeleton_survives_pause_resume_and_restart(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    config = make_config()
    tick_ms = config.scheduler.decision_tick_ms
    tick_s = tick_ms / 1_000
    queue: list[Decision] = []
    for text in SPOKEN:
        queue += [Decision(action="speak", message=text, target=None)] + [WAIT] * 4
    backend = RecordingFakeBackend(queue)
    sleeps: list[float] = []

    # --- session 1 ---------------------------------------------------------
    store = open_store(path)
    store.initialize()
    controller = open_store(path)  # a second terminal's connection
    engine = make_engine(config, store, backend, WALL, sleeps)
    engine.start()
    kinds = [engine.step().kind for _ in range(30)]

    controller.set_control("room-1", "paused")
    paused_at_id, paused_at_ms = all_events(store)[-1].id, engine.clock.now_ms()
    assert [engine.step().kind for _ in range(3)] == ["paused"] * 3
    assert all_events(store)[-1].id == paused_at_id  # nothing is written while paused
    assert engine.clock.now_ms() == paused_at_ms     # and no simulated time accrues
    assert sleeps == [tick_s] * 3                    # accelerated mode sleeps only while paused
    controller.set_control("room-1", "running")
    kinds += [engine.step().kind for _ in range(15)]

    controller.set_control("room-1", "stop_requested")
    assert engine.step().kind == "stopped"
    store.close()

    # --- restart: same file, new connection, new engine, hours of downtime ---
    store = open_store(path)
    store.initialize()
    later = WALL + timedelta(hours=3)
    restarted = make_engine(config, store, backend, later, sleeps)
    restarted.start()
    results = [restarted.step() for _ in range(30)]
    kinds += [result.kind for result in results]
    restarted.stop()

    events = all_events(store)
    by_type: dict[str, list[StoredEvent]] = {}
    for event in events:
        by_type.setdefault(event.type, []).append(event)
    assert {"message", "wait", "ambient"} <= set(kinds)
    assert sleeps == [tick_s] * 3

    # Event ids strictly increase with no duplicates; simulated time never runs backwards.
    ids = [event.id for event in events]
    assert all(earlier < later for earlier, later in zip(ids, ids[1:]))
    assert all(earlier.sim_ms <= later.sim_ms for earlier, later in zip(events, events[1:]))

    # Exactly two sessions, and nothing was fabricated for the downtime between them.
    assert len(by_type["session_started"]) == len(by_type["session_ended"]) == 2
    first_end = by_type["session_ended"][0]
    resumed = by_type["session_started"][1]
    assert ids.index(resumed.id) == ids.index(first_end.id) + 1
    assert resumed.sim_ms == first_end.sim_ms == resumed.payload["sim_start_ms"]
    assert resumed.payload["restart"] is True and resumed.wall_ts == later.isoformat()

    # The restarted session acts (messages or valid waits), it does not just idle.
    after_restart = [event for event in events if event.id > resumed.id]
    acted = [event for event in after_restart if event.type in ("message", "agent_wait")]
    assert acted

    # Downtime is not simulated: no event after the restart jumps in simulation time.
    # A step advances the accelerated clock by one decision tick after a decision and by
    # one tick per idle step (an idle step commits nothing), so between two consecutive
    # events the legitimate advance is one tick plus one per idle step in between, never
    # more. A wall-clock-driven jump across the 3 h downtime would exceed every such bound.
    committed = [result.event_id for result in results if result.event_id is not None]
    assert committed == [event.id for event in after_restart[:-1]]  # one event per acting step
    idle_before: list[int] = []  # idle steps preceding each event after the restart
    idle_steps = 0
    for result in results:
        if result.event_id is None:
            idle_steps += 1
        else:
            idle_before.append(idle_steps)
            idle_steps = 0
    idle_before.append(idle_steps)  # trailing idle steps precede the session_ended from stop()
    previous = resumed
    for event, idle in zip(after_restart, idle_before, strict=True):
        assert 0 <= event.sim_ms - previous.sim_ms <= tick_ms * (1 + idle), (previous, event)
        previous = event

    # Every message text came from the fake queue, in the order it was handed out.
    messages = by_type["message"]
    assert [event.payload["message"] for event in messages] == [
        decision.message for decision in backend.returned if decision.action == "speak"
    ]
    assert {event.payload["message"] for event in messages} <= set(SPOKEN)
    assert len(by_type["agent_wait"]) == sum(1 for d in backend.returned if d.action == "wait")
    assert "attempt_failed" not in by_type and "generation_failed" not in by_type  # queue never ran dry

    # The observer-only report agrees with the persisted events.
    metrics = analyze_run(store, "run-1")
    assert metrics.visible_messages == len(messages) >= 1
    assert metrics.valid_waits == len(by_type["agent_wait"]) >= 1
    assert metrics.ambient_events == len(by_type["environment"]) >= 1
    assert metrics.silence_periods >= 1
    assert (metrics.generation_failures, metrics.attempt_failures, metrics.failure_classes) == (0, 0, {})
    assert metrics.messages_per_agent == {
        agent.id: sum(1 for event in messages if event.agent_id == agent.id) for agent in config.agents
    }
    assert metrics.mean_inference_ms == 0.0  # the fake backend reports zero latency
    assert all_events(store) == events       # analysis wrote nothing
