"""The walking-skeleton engine composes store, clock, scheduler, prompts, and model."""

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import pytest

from driftroom.clock import VirtualClock
from driftroom.config import ENGINE_VERSION, load_run_config, run_config_hash, run_fingerprint
from driftroom.domain import AgentConfig, RunConfig
from driftroom.engine import EngineStepResult, SimulationEngine
from driftroom.models.base import (
    BackendTimeoutError, Decision, DecisionValidationError, EmptyModelContentError,
    ModelBackend, ModelResult, decision_schema,
)
from driftroom.models.fake import FakeModelBackend
from driftroom.models.ollama_backend import OllamaBackend
from driftroom.prompting import TRAIT_RENDERER_VERSION, TURN_FORMAT_VERSION, prompt_template_hash
from driftroom.storage import EventRecord, EventStore, StoredEvent


EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "driftroom.example.toml"
WALL = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
WAIT = Decision(action="wait", message=None, target=None)
# Every agent scores exactly base_bias, so the first configured agent is always selected.
FORCED = dict(
    base_bias=1.0, talkativeness_weight=0, direct_mention_bonus=0, topic_overlap_weight=0,
    elapsed_weight=0, relationship_weight=0, recent_speaker_penalty=0, cooldown_penalty=0,
    random_jitter=0,
)
UNREACHABLE = {**FORCED, "base_bias": -1.0}


def speak(text: str, target: str | None = None) -> Decision:
    return Decision(action="speak", message=text, target=target)


def make_config(scheduler: dict[str, Any] | None = None, **runtime: Any) -> RunConfig:
    data = load_run_config(EXAMPLE_CONFIG).model_dump(mode="json")
    data["runtime"].update({"clock_mode": "accelerated", "random_seed": 7, **runtime})
    data["scheduler"].update(scheduler or {})
    return RunConfig.model_validate(data)


def forbidden_sleep(seconds: float) -> None:
    raise AssertionError(f"engine slept {seconds}s in accelerated mode")


class RecordingBackend(ModelBackend):
    """FakeModelBackend outcomes, plus call capture and a reentrancy guard."""

    def __init__(
        self, outcomes: Sequence[Decision | Exception],
        on_call: Callable[[int], None] | None = None,
    ) -> None:
        self._fake = FakeModelBackend(outcomes)
        self._on_call = on_call
        self.calls: list[tuple[AgentConfig, list[dict[str, str]]]] = []
        self.busy = False

    @property
    def targets(self) -> list[tuple[str, ...]]:
        return self._fake.targets

    def decide(
        self, agent: AgentConfig, messages: Sequence[dict[str, str]], *, targets: Sequence[str] = (),
    ) -> ModelResult:
        assert not self.busy, "two model calls overlapped"
        self.busy = True
        try:
            self.calls.append((agent, list(messages)))
            if self._on_call is not None:
                self._on_call(len(self.calls))
            return self._fake.decide(agent, messages, targets=targets)
        finally:
            self.busy = False


_OPEN_STORES: list[EventStore] = []


def open_store(path: Path) -> EventStore:
    """An EventStore that the autouse fixture below closes after the test (3.13+ warns on unclosed sqlite connections)."""
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


def accelerated_clock() -> VirtualClock:
    return VirtualClock("accelerated", WALL, monotonic_fn=lambda: 0.0)


def make_engine(
    tmp_path: Path, config: RunConfig, backend: ModelBackend, **kwargs: Any,
) -> tuple[SimulationEngine, EventStore, VirtualClock]:
    store = open_store(tmp_path / "room.sqlite3")
    store.initialize()
    clock = kwargs.pop("clock", accelerated_clock())
    kwargs.setdefault("sleep_fn", forbidden_sleep)
    engine = SimulationEngine(
        config, store, backend, run_id="run-1", room_id="room-1", clock=clock,
        wall_clock=lambda: WALL, **kwargs,
    )
    return engine, store, clock


def events(store: EventStore) -> list[StoredEvent]:
    return store.read_events("run-1", limit=100_000)


def types(store: EventStore) -> list[str]:
    return [event.type for event in events(store)]


def history(messages: list[dict[str, str]]) -> str:
    return messages[1]["content"]


# --- lifecycle -------------------------------------------------------------

def test_start_records_full_regime_and_stop_is_idempotent(tmp_path) -> None:
    config = make_config()
    engine, store, _ = make_engine(tmp_path, config, RecordingBackend([]))
    with pytest.raises(RuntimeError):
        engine.step()

    started_id = engine.start()

    [started] = events(store)
    assert (started.id, started.type, started.agent_id) == (started_id, "session_started", None)
    assert started.wall_ts == WALL.isoformat()
    assert store.get_run("run-1").config == config
    assert started.payload == {
        "run_fingerprint": run_fingerprint(
            config, prompt_hash=prompt_template_hash(), model_digests={"qwen3:4b": None}
        ),
        "config_hash": run_config_hash(config),
        "prompt_hash": prompt_template_hash(),
        "trait_renderer_version": TRAIT_RENDERER_VERSION,
        "turn_format_version": TURN_FORMAT_VERSION,
        "engine_version": ENGINE_VERSION,
        "model_names": ["qwen3:4b"],
        "model_digests": {"qwen3:4b": None},
        "random_seed": 7,
        "clock_mode": "accelerated",
        "clock_speed": 1.0,
        "startup_mode": "blank",
        "runtime_mode": "balanced",
        "sim_start_ms": 0,
        "restart": False,
    }
    assert re.fullmatch(r"[0-9a-f]{64}", started.payload["run_fingerprint"])

    ended_id = engine.stop()
    assert engine.stop() is None
    assert types(store) == ["session_started", "session_ended"]
    ended = events(store)[-1]
    assert (ended.id, ended.payload) == (ended_id, {"sim_end_ms": 0})
    assert engine.step() == EngineStepResult("stopped", None)
    assert types(store) == ["session_started", "session_ended"]


def test_stop_stays_idempotent_when_the_observer_raises(tmp_path) -> None:
    def observer(event: StoredEvent) -> None:
        if event.type == "session_ended":
            raise RuntimeError("observer broke")

    engine, store, _ = make_engine(tmp_path, make_config(), RecordingBackend([]), observer=observer)
    engine.start()

    with pytest.raises(RuntimeError, match="observer broke"):
        engine.stop()
    assert engine.stop() is None
    assert engine.step() == EngineStepResult("stopped", None)
    assert types(store) == ["session_started", "session_ended"]


def test_supplied_model_digests_enter_the_fingerprint(tmp_path) -> None:
    config = make_config()
    digests = {"qwen3:4b": "sha256:abc"}
    engine, store, _ = make_engine(tmp_path, config, RecordingBackend([]), model_digests=digests)
    engine.start()
    payload = events(store)[0].payload
    assert payload["model_digests"] == digests
    assert payload["run_fingerprint"] == run_fingerprint(
        config, prompt_hash=prompt_template_hash(), model_digests=digests
    )
    assert payload["run_fingerprint"] != run_fingerprint(
        config, prompt_hash=prompt_template_hash(), model_digests={"qwen3:4b": None}
    )


# --- decisions -------------------------------------------------------------

def test_waits_and_speech_commit_before_observers_and_feed_next_context(tmp_path) -> None:
    reader = open_store(tmp_path / "room.sqlite3")
    seen: list[StoredEvent] = []

    def observer(event: StoredEvent) -> None:
        # A second connection can already read the event: it was committed first.
        assert reader.read_events("run-1", after_id=event.id - 1, limit=1) == [event]
        seen.append(event)

    backend = RecordingBackend([WAIT, speak("hey"), WAIT])
    engine, store, _ = make_engine(tmp_path, make_config(FORCED), backend, observer=observer)
    engine.start()

    results = [engine.step() for _ in range(3)]

    assert [result.kind for result in results] == ["wait", "message", "wait"]
    committed = events(store)
    assert [event.type for event in committed] == ["session_started", "agent_wait", "message", "agent_wait"]
    assert [result.event_id for result in results] == [event.id for event in committed[1:]]
    speaker = backend.calls[1][0]
    message = committed[2]
    assert message.agent_id == speaker.id
    assert message.payload == {
        "speaker": speaker.name, "message": "hey", "target": None,
        "scheduler_score": 1.0, "reasons": [], "latency_ms": 0.0,
        "prompt_tokens": None, "output_tokens": None, "model": "qwen3:4b", "attempt": 1,
    }
    wait = committed[1]
    assert wait.agent_id == backend.calls[0][0].id
    assert wait.payload["attempt"] == 1 and "message" not in wait.payload
    assert seen == committed
    assert f"{speaker.name}: hey" not in history(backend.calls[1][1])
    assert f"{speaker.name}: hey" in history(backend.calls[2][1])


def test_model_calls_are_strictly_serial(tmp_path) -> None:
    backend = RecordingBackend([WAIT, speak("one"), WAIT, speak("two"), WAIT] * 10)
    engine, store, _ = make_engine(tmp_path, make_config(FORCED), backend)
    engine.start()

    engine.run(max_steps=50)

    assert len(backend.calls) == 50
    assert types(store).count("agent_wait") + types(store).count("message") == 50
    assert "session_ended" not in types(store)


def test_failed_attempts_are_infrastructure_events_never_waits(tmp_path) -> None:
    backend = RecordingBackend([
        BackendTimeoutError("slow"), DecisionValidationError("bad envelope"),
        EmptyModelContentError("empty"), WAIT,
    ])
    engine, store, _ = make_engine(tmp_path, make_config(FORCED, retry_count=1), backend)
    engine.start()

    exhausted = engine.step()
    recovered = engine.step()

    committed = events(store)
    assert [event.type for event in committed] == [
        "session_started", "attempt_failed", "attempt_failed", "generation_failed",
        "attempt_failed", "agent_wait",
    ]
    assert len(backend.calls) == 4
    agent_id = backend.calls[0][0].id
    assert all(event.agent_id == agent_id for event in committed[1:])
    assert committed[1].payload == {
        "attempt": 1, "error_class": "BackendTimeoutError", "error": "slow",
        "raw_excerpt": None, "detail": None, "scheduler_score": 1.0, "reasons": [],
    }
    assert committed[2].payload["attempt"] == 2
    assert committed[2].payload["error_class"] == "DecisionValidationError"
    assert committed[3].payload == {"attempts": 2, "last_error_class": "DecisionValidationError"}
    assert exhausted == EngineStepResult("generation_failed", committed[3].id)
    assert committed[4].payload["error_class"] == "EmptyModelContentError"
    assert recovered == EngineStepResult("wait", committed[5].id)
    assert committed[5].payload["attempt"] == 2


# Raw envelopes real qwen3:4b returned under the old nullable-only schema (2026-10-03).
REAL_SAMPLES = [
    '{"action": "wait", "message": "wait", "target": "me"}',
    '{"action": "wait", "message": "i\'m waiting for someone to talk first. maybe we can start with a joke? \U0001f602", '
    '"target": "nobody"}',
    '{"action": "wait", "message": "i\'m here waiting for someone to talk to me though i\'ll be the first one to say '
    'i don\'t know if i\'m even real lol", "target": "you"}',
]


class ScriptedOllamaClient:
    """A fake ``ollama.Client``: replays raw chat contents and records each request."""

    def __init__(self, contents: Sequence[str]) -> None:
        self._contents = deque(contents)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"model": "qwen3:4b", "message": {"role": "assistant", "content": self._contents.popleft()},
                "done": True, "done_reason": "stop"}


def test_real_failing_samples_record_raw_excerpt_and_detail_on_attempt_failed(tmp_path) -> None:
    config = make_config(FORCED, retry_count=2)
    client = ScriptedOllamaClient(REAL_SAMPLES)
    engine, store, _ = make_engine(tmp_path, config, OllamaBackend(config.runtime, client=client))
    engine.start()

    failed = engine.step()

    committed = events(store)
    assert [event.type for event in committed] == [
        "session_started", "attempt_failed", "attempt_failed", "attempt_failed", "generation_failed",
    ]
    for attempt, (event, sample) in enumerate(zip(committed[1:4], REAL_SAMPLES), start=1):
        assert event.payload == {
            "attempt": attempt, "error_class": "DecisionValidationError",
            "error": "Ollama decision violates schema", "raw_excerpt": sample,
            "detail": "wait requires a null message", "scheduler_score": 1.0, "reasons": [],
        }
    assert failed == EngineStepResult("generation_failed", committed[4].id)
    assert [call["format"] for call in client.calls] == [decision_schema(["Milo", "Ada"])] * 3


def test_engine_passes_the_other_participants_names_as_targets(tmp_path) -> None:
    config = make_config(FORCED)
    backend = RecordingBackend([WAIT])
    engine, _, _ = make_engine(tmp_path, config, backend)
    engine.start()

    engine.step()

    [(agent, _)] = backend.calls
    assert agent.id == "june"
    assert backend.targets == [("Milo", "Ada")]


@pytest.mark.parametrize("decision", [
    speak("hello?", target="nobody"), speak("talking to myself", target="June"),
    Decision(action="wait", message=None, target="you"),
])
def test_a_non_participant_target_is_an_attempt_failed_never_a_message(tmp_path, decision) -> None:
    backend = RecordingBackend([decision, WAIT])
    engine, store, _ = make_engine(tmp_path, make_config(FORCED, retry_count=1), backend)
    engine.start()

    result = engine.step()

    committed = events(store)
    assert [event.type for event in committed] == ["session_started", "attempt_failed", "agent_wait"]
    failed = committed[1].payload
    assert (failed["attempt"], failed["error_class"], failed["error"]) == (
        1, "DecisionValidationError", "target is not a participant",
    )
    assert failed["raw_excerpt"] is None
    assert repr(decision.target) in failed["detail"]
    assert result == EngineStepResult("wait", committed[2].id)


def test_a_participant_target_is_recorded_on_the_message(tmp_path) -> None:
    backend = RecordingBackend([speak("hi ada", target="Ada")])
    engine, store, _ = make_engine(tmp_path, make_config(FORCED), backend)
    engine.start()

    assert engine.step().kind == "message"
    assert events(store)[-1].payload["target"] == "Ada"


def test_non_backend_errors_propagate_without_events(tmp_path) -> None:
    backend = RecordingBackend([RuntimeError("engine bug")])
    engine, store, _ = make_engine(tmp_path, make_config(FORCED), backend)
    engine.start()
    with pytest.raises(RuntimeError, match="engine bug"):
        engine.step()
    assert types(store) == ["session_started"]


def test_prompt_context_is_bounded_to_recent_events(tmp_path) -> None:
    backend = RecordingBackend([WAIT])
    engine, store, _ = make_engine(tmp_path, make_config(FORCED, recent_context_events=2), backend)
    engine.start()
    for index in range(5):
        store.commit_event(EventRecord(
            "run-1", WALL.isoformat(), 0, "message", "milo",
            {"speaker": "Milo", "message": f"line {index}", "target": None},
        ))

    engine.step()

    transcript = history(backend.calls[0][1])
    assert "Milo: line 3" in transcript and "Milo: line 4" in transcript
    assert not any(f"line {index}" in transcript for index in range(3))


def test_invisible_events_never_push_room_history_out_of_the_context_window(tmp_path) -> None:
    # A quiet room fills up with agent_wait events; the window is bounded over
    # visible events, so the transcript and scheduler signals survive.
    backend = RecordingBackend([speak("hey")] + [WAIT] * 25)
    engine, store, _ = make_engine(tmp_path, make_config(FORCED), backend)
    engine.start()

    kinds = [engine.step().kind for _ in range(26)]

    assert kinds == ["message"] + ["wait"] * 25
    assert types(store).count("agent_wait") == 25 > make_config().runtime.recent_context_events
    speaker = backend.calls[0][0]
    assert f"{speaker.name}: hey" in history(backend.calls[25][1])


def test_keyboard_interrupt_during_run_ends_session_and_reraises(tmp_path) -> None:
    def interrupt(call: int) -> None:
        if call == 2:
            raise KeyboardInterrupt

    backend = RecordingBackend([WAIT, WAIT], on_call=interrupt)
    engine, store, _ = make_engine(tmp_path, make_config(FORCED), backend)
    engine.start()
    with pytest.raises(KeyboardInterrupt):
        engine.run()
    assert types(store) == ["session_started", "agent_wait", "session_ended"]


def test_run_reports_every_step_to_on_step_including_the_final_stopped(tmp_path) -> None:
    engine, store, _ = make_engine(tmp_path, make_config(UNREACHABLE), RecordingBackend([]))
    engine.start()
    seen: list[EngineStepResult] = []

    def on_step(result: EngineStepResult) -> None:
        seen.append(result)
        if len(seen) == 3:
            store.set_control("room-1", "stop_requested")

    engine.run(on_step=on_step)

    assert [result.kind for result in seen] == ["idle", "idle", "idle", "stopped"]
    assert seen[-1].event_id == events(store)[-1].id


def test_run_calls_on_step_once_per_step_up_to_max_steps(tmp_path) -> None:
    engine, _, _ = make_engine(tmp_path, make_config(UNREACHABLE), RecordingBackend([]))
    engine.start()
    seen: list[EngineStepResult] = []

    engine.run(5, on_step=seen.append)

    assert [result.kind for result in seen] == ["idle"] * 5


# --- control, time, ambient, restart -----------------------------------------

def test_pause_during_inference_lets_generation_commit_then_pauses(tmp_path) -> None:
    controller = open_store(tmp_path / "room.sqlite3")

    def pause_on_first_call(call: int) -> None:
        if call == 1:
            controller.set_control("room-1", "paused")

    sleeps: list[float] = []
    backend = RecordingBackend([speak("hello"), WAIT], on_call=pause_on_first_call)
    engine, store, clock = make_engine(tmp_path, make_config(FORCED), backend, sleep_fn=sleeps.append)
    engine.start()

    assert engine.step().kind == "message"
    assert types(store) == ["session_started", "message"]
    before = clock.now_ms()
    assert engine.step() == EngineStepResult("paused", None)
    assert engine.step() == EngineStepResult("paused", None)
    assert len(backend.calls) == 1
    assert clock.now_ms() == before  # a paused accelerated room accrues no simulated time
    assert sleeps == [5.0, 5.0]
    assert types(store) == ["session_started", "message"]

    controller.set_control("room-1", "running")
    assert engine.step().kind == "wait"
    assert len(backend.calls) == 2

    controller.set_control("room-1", "stop_requested")
    stopped = engine.step()
    assert stopped.kind == "stopped"
    assert types(store) == ["session_started", "message", "agent_wait", "session_ended"]
    assert stopped.event_id == events(store)[-1].id
    assert engine.step().kind == "stopped"
    assert len(backend.calls) == 2 and types(store).count("session_ended") == 1


@pytest.mark.parametrize(("control", "next_kind"), [("paused", "paused"), ("stop_requested", "stopped")])
def test_control_change_during_a_failed_attempt_stops_the_retries(tmp_path, control, next_kind) -> None:
    controller = open_store(tmp_path / "room.sqlite3")

    def pause_on_first_call(call: int) -> None:
        if call == 1:
            controller.set_control("room-1", control)

    sleeps: list[float] = []
    backend = RecordingBackend([BackendTimeoutError("slow"), WAIT, WAIT], on_call=pause_on_first_call)
    engine, store, _ = make_engine(
        tmp_path, make_config(FORCED, retry_count=2), backend, sleep_fn=sleeps.append,
    )
    engine.start()

    failed = engine.step()

    committed = events(store)
    assert [event.type for event in committed] == ["session_started", "attempt_failed", "generation_failed"]
    assert len(backend.calls) == 1
    assert committed[2].payload == {
        "attempts": 1, "last_error_class": "BackendTimeoutError", "interrupted_by_control": control,
    }
    assert failed == EngineStepResult("generation_failed", committed[2].id)
    assert engine.step().kind == next_kind
    assert len(backend.calls) == 1


def test_ambient_silence_event_is_neutral_throttled_and_never_calls_model(tmp_path) -> None:
    config = make_config({
        **UNREACHABLE, "decision_tick_ms": 1_000, "silence_ambient_after_ms": 5_000,
        "ambient_min_interval_ms": 12_000,
    })
    backend = RecordingBackend([])
    engine, store, clock = make_engine(tmp_path, config, backend)
    engine.start()

    results = [engine.step() for _ in range(40)]

    ambient = [event for event in events(store) if event.type == "environment"]
    assert [event.sim_ms for event in ambient] == [5_000, 17_000, 29_000]
    assert all(event.payload == {"text": "the room has been quiet for a while"} for event in ambient)
    assert all(event.agent_id is None for event in ambient)
    assert [result.event_id for result in results if result.kind == "ambient"] == [event.id for event in ambient]
    assert {result.kind for result in results} == {"idle", "ambient"}
    assert backend.calls == []
    assert clock.now_ms() == 37_000  # 37 idle ticks; ambient steps do not advance time


def test_ambient_interval_holds_when_waits_crowd_the_context_window(tmp_path) -> None:
    # Root-cause regression: non-visible agent_wait events must not push the last
    # ambient event out of the bounded window; the interval must still hold.
    elapsed = [0.0]

    def one_second_per_call(call: int) -> None:
        elapsed[0] += 1.0

    def realtime_sleep(seconds: float) -> None:
        elapsed[0] += seconds

    config = make_config(
        {**FORCED, "silence_ambient_after_ms": 5_000, "ambient_min_interval_ms": 12_000},
        clock_mode="realtime", recent_context_events=2,
    )
    clock = VirtualClock("realtime", WALL, monotonic_fn=lambda: elapsed[0])
    backend = RecordingBackend([WAIT] * 30, on_call=one_second_per_call)
    engine, store, _ = make_engine(tmp_path, config, backend, clock=clock, sleep_fn=realtime_sleep)
    engine.start()

    for _ in range(25):
        engine.step()

    # Each decision takes 1 s of inference plus one 5 s tick: ambient every 12 s.
    ambient_ms = [event.sim_ms for event in events(store) if event.type == "environment"]
    assert ambient_ms == [6_000 + 12_000 * k for k in range(8)]
    assert types(store).count("agent_wait") == 17


def test_quiet_room_burn_never_spins_or_fabricates_speech(tmp_path) -> None:
    config = make_config({"silence_ambient_after_ms": 60_000, "ambient_min_interval_ms": 120_000})
    tick = config.scheduler.decision_tick_ms
    backend = RecordingBackend([WAIT] * 500)
    engine, store, clock = make_engine(tmp_path, config, backend)  # sleep_fn raises
    engine.start()

    kinds: list[str] = []
    for _ in range(500):
        before = clock.now_ms()
        kind = engine.step().kind
        kinds.append(kind)
        # Every step that reaches the scheduler consumes one decision tick, so
        # decision pacing follows simulated time, not inference latency (§13A).
        assert clock.now_ms() - before == (0 if kind == "ambient" else tick)

    assert set(kinds) == {"wait", "idle", "ambient"}
    assert len(backend.calls) == kinds.count("wait") == types(store).count("agent_wait")
    assert "message" not in types(store)
    ambient_ms = [event.sim_ms for event in events(store) if event.type == "environment"]
    assert len(ambient_ms) == kinds.count("ambient") >= 2
    assert all(later - earlier >= 120_000 for earlier, later in zip(ambient_ms, ambient_ms[1:]))
    assert clock.now_ms() == (len(kinds) - kinds.count("ambient")) * tick > 0


def test_every_decision_consumes_one_tick_of_simulated_time(tmp_path) -> None:
    config = make_config({**FORCED, "decision_tick_ms": 2_000}, retry_count=0)
    backend = RecordingBackend([speak("hi"), WAIT, BackendTimeoutError("slow")])
    engine, store, clock = make_engine(tmp_path, config, backend)  # sleep_fn raises
    engine.start()

    advanced = []
    for _ in range(3):
        before = clock.now_ms()
        kind = engine.step().kind
        advanced.append((kind, clock.now_ms() - before))

    assert advanced == [("message", 2_000), ("wait", 2_000), ("generation_failed", 2_000)]
    # The decision is stamped with the time it was made, before the tick.
    assert [event.sim_ms for event in events(store)[1:]] == [0, 2_000, 4_000, 4_000]


def test_realtime_decisions_wait_one_tick_without_touching_the_clock(tmp_path) -> None:
    sleeps: list[float] = []
    config = make_config({**FORCED, "decision_tick_ms": 2_000}, clock_mode="realtime")
    clock = VirtualClock("realtime", WALL, monotonic_fn=lambda: 0.0)
    engine, _, _ = make_engine(
        tmp_path, config, RecordingBackend([WAIT, speak("hi")]), clock=clock, sleep_fn=sleeps.append,
    )
    engine.start()

    assert [engine.step().kind for _ in range(2)] == ["wait", "message"]
    assert sleeps == [2.0, 2.0]


def decision_gaps(store: EventStore) -> list[int]:
    last: dict[str, int] = {}
    gaps = []
    for event in events(store):
        if event.type in ("agent_wait", "message"):
            if event.agent_id in last:
                gaps.append(event.sim_ms - last[event.agent_id])
            last[event.agent_id] = event.sim_ms
    return gaps


def test_agents_are_not_reasked_within_the_cooldown_of_their_last_decision(tmp_path) -> None:
    config = make_config()
    backend = RecordingBackend([WAIT] * 200)
    engine, store, _ = make_engine(tmp_path, config, backend)
    engine.start()

    kinds = [engine.step().kind for _ in range(200)]

    assert kinds.count("wait") == len(backend.calls) >= 20
    assert len({agent.id for agent, _ in backend.calls}) == 3
    gaps = decision_gaps(store)
    assert gaps and min(gaps) >= config.scheduler.speaker_cooldown_ms


def test_restart_restores_decision_cooldowns_from_persisted_events(tmp_path) -> None:
    # Forced regime: every agent scores 1.0 unless cooling down (score 0, below
    # threshold), so selection runs june, milo, ada, idle, june, ... in config order.
    # A 20 s wait cooldown keeps that cycle short enough to span three sessions.
    config = make_config({**FORCED, "cooldown_penalty": 1.0, "wait_cooldown_ms": 20_000})
    store = open_store(tmp_path / "room.sqlite3")
    store.initialize()
    asked: list[str] = []
    for _ in range(3):
        backend = RecordingBackend([WAIT] * 2)
        engine = SimulationEngine(
            config, store, backend, run_id="run-1", room_id="room-1",
            wall_clock=lambda: WALL, sleep_fn=forbidden_sleep,
        )
        engine.start()
        engine.run(max_steps=2)
        engine.stop()
        asked += [agent.id for agent, _ in backend.calls]

    # Without restored cooldowns the second session would re-ask june at 10 s.
    assert asked == ["june", "milo", "ada", "june", "milo"]
    assert min(decision_gaps(store)) >= config.scheduler.speaker_cooldown_ms


def test_an_all_wait_room_is_asked_at_most_once_per_wait_cooldown(tmp_path) -> None:
    config = make_config()
    wait_cooldown = config.scheduler.wait_cooldown_ms
    backend = RecordingBackend([WAIT] * 2_000)
    engine, store, clock = make_engine(tmp_path, config, backend)
    engine.start()

    steps = 0
    while clock.now_ms() < 7_200_000:
        engine.step()
        steps += 1

    assert steps >= 7_200_000 // config.scheduler.decision_tick_ms
    assert 1 <= len(backend.calls) <= 3 * (7_200_000 // wait_cooldown) + 3
    assert len(backend.calls) == types(store).count("agent_wait")
    gaps = decision_gaps(store)
    assert gaps and min(gaps) >= wait_cooldown


def test_restart_restores_wait_cooldowns_from_persisted_events(tmp_path) -> None:
    # Forced regime: june (first in config order) always wins unless cooling down.
    config = make_config({**FORCED, "cooldown_penalty": 1.0})
    wait_cooldown = config.scheduler.wait_cooldown_ms
    first, store, _ = make_engine(tmp_path, config, RecordingBackend([WAIT]))
    first.start()
    assert first.step().kind == "wait"
    first.stop()
    store.close()

    reopened = open_store(tmp_path / "room.sqlite3")
    backend = RecordingBackend([WAIT] * 10)
    engine = SimulationEngine(
        config, reopened, backend, run_id="run-1", room_id="room-1",
        wall_clock=lambda: WALL, sleep_fn=forbidden_sleep,
    )
    engine.start()
    while engine.clock.now_ms() < wait_cooldown:
        engine.step()

    june_waits = [e.sim_ms for e in events(reopened) if e.type == "agent_wait" and e.agent_id == "june"]
    assert june_waits == [0]
    assert [agent.id for agent, _ in backend.calls] == ["milo", "ada"]


def test_restart_resumes_from_last_persisted_time_without_offline_events(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    config = make_config(UNREACHABLE)
    store = open_store(path)
    store.initialize()
    first = SimulationEngine(
        config, store, RecordingBackend([]), run_id="run-1", room_id="room-1",
        wall_clock=lambda: WALL, sleep_fn=forbidden_sleep,
    )
    first.start()
    for _ in range(4):
        assert first.step().kind == "idle"
    ended_id = first.stop()
    store.close()

    reopened = open_store(path)
    reopened.initialize()
    later = datetime(2026, 10, 3, 9, tzinfo=timezone.utc)
    second = SimulationEngine(
        config, reopened, RecordingBackend([]), run_id="run-1", room_id="room-1",
        wall_clock=lambda: later, sleep_fn=forbidden_sleep,
    )
    started_id = second.start()

    committed = events(reopened)
    assert [event.type for event in committed] == ["session_started", "session_ended", "session_started"]
    assert started_id == ended_id + 1
    assert committed[1].sim_ms == 20_000
    assert committed[0].payload["restart"] is False
    assert committed[2].payload["restart"] is True
    assert committed[2].payload["sim_start_ms"] == committed[2].sim_ms == 20_000
    assert reopened.get_run("run-1").started_at == WALL.isoformat()
    assert second.step().kind == "idle"
    second.stop()
    assert events(reopened)[-1].payload == {"sim_end_ms": 25_000}


def restart_after_control(tmp_path: Path, control: str) -> tuple[SimulationEngine, EventStore]:
    config = make_config(UNREACHABLE)
    store = open_store(tmp_path / "room.sqlite3")
    store.initialize()
    controller = open_store(tmp_path / "room.sqlite3")
    first = SimulationEngine(
        config, store, RecordingBackend([]), run_id="run-1", room_id="room-1",
        wall_clock=lambda: WALL, sleep_fn=lambda seconds: None,
    )
    first.start()
    assert first.step().kind == "idle"
    controller.set_control("room-1", control)
    if control == "stop_requested":
        assert first.step().kind == "stopped"  # the room was stopped through the control table
    second = SimulationEngine(
        config, store, RecordingBackend([]), run_id="run-1", room_id="room-1",
        wall_clock=lambda: WALL, sleep_fn=lambda seconds: None,
    )
    second.start()
    return second, store


def test_start_clears_a_stale_stop_request(tmp_path) -> None:
    engine, store = restart_after_control(tmp_path, "stop_requested")

    assert store.get_control("room-1").desired_state == "running"
    assert engine.step().kind == "idle"
    assert types(store).count("session_ended") == 1


def test_start_leaves_a_paused_room_paused(tmp_path) -> None:
    engine, store = restart_after_control(tmp_path, "paused")

    assert store.get_control("room-1").desired_state == "paused"
    assert engine.step() == EngineStepResult("paused", None)


class FailingAgentBackend(ModelBackend):
    """June's generations always fail; everyone else waits."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def decide(
        self, agent: AgentConfig, messages: Sequence[dict[str, str]], *, targets: Sequence[str] = (),
    ) -> ModelResult:
        self.asked.append(agent.id)
        if agent.id == "june":
            raise BackendTimeoutError("slow")
        return FakeModelBackend([WAIT]).decide(agent, messages)


def test_a_failing_agent_is_not_reasked_within_the_decision_cooldown(tmp_path) -> None:
    # Forced regime: june (first in config order) always wins unless cooling down.
    config = make_config({**FORCED, "cooldown_penalty": 1.0}, retry_count=0)
    backend = FailingAgentBackend()
    engine, store, _ = make_engine(tmp_path, config, backend)
    engine.start()

    for _ in range(40):
        engine.step()

    failures = [event.sim_ms for event in events(store) if event.type == "generation_failed"]
    assert len(failures) >= 2
    assert min(b - a for a, b in zip(failures, failures[1:])) >= config.scheduler.speaker_cooldown_ms
    assert {"milo", "ada"} <= set(backend.asked)


def test_a_realtime_pause_accrues_no_simulated_time_and_fabricates_no_silence(tmp_path) -> None:
    wall = [0.0]

    def sleep(seconds: float) -> None:
        wall[0] += seconds

    config = make_config(FORCED, clock_mode="realtime")
    clock = VirtualClock("realtime", WALL, monotonic_fn=lambda: wall[0])
    backend = RecordingBackend([speak("hey"), WAIT])
    engine, store, _ = make_engine(tmp_path, config, backend, clock=clock, sleep_fn=sleep)
    engine.start()
    assert engine.step().kind == "message"

    store.set_control("room-1", "paused")
    assert {engine.step().kind for _ in range(120)} == {"paused"}  # ten minutes of wall time
    assert wall[0] >= 600.0
    store.set_control("room-1", "running")

    assert engine.step().kind == "wait"
    assert [event.type for event in events(store)] == ["session_started", "message", "agent_wait"]
    assert events(store)[-1].sim_ms == 5_000
    assert "later]" not in history(backend.calls[-1][1])


def test_prompt_template_edits_during_a_run_stop_the_engine_before_any_model_call(tmp_path, monkeypatch) -> None:
    import driftroom.engine

    backend = RecordingBackend([WAIT])
    engine, store, _ = make_engine(tmp_path, make_config(FORCED), backend)
    engine.start()
    monkeypatch.setattr(driftroom.engine, "prompt_template_hash", lambda: "edited")

    with pytest.raises(RuntimeError, match="prompt templates changed during run"):
        engine.step()

    assert backend.calls == []
    assert types(store) == ["session_started"]
