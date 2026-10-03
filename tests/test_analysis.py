"""Observer-only run analysis over persisted events (spec §31, §32)."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from driftroom import analysis
from driftroom.analysis import RunMetrics, analyze_run
from driftroom.config import load_run_config
from driftroom.domain import RunConfig
from driftroom.storage import EventRecord, EventStore, RunRecord, StoredEvent


EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "driftroom.example.toml"
WALL = "2026-10-01T12:00:00+00:00"
SILENCE_MS = 60_000


def make_config() -> RunConfig:
    data = load_run_config(EXAMPLE_CONFIG).model_dump(mode="json")
    data["scheduler"]["silence_ambient_after_ms"] = SILENCE_MS
    return RunConfig.model_validate(data)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[EventStore]:
    """A seeded run store, always closed (Python 3.13+ warns on unclosed sqlite connections)."""
    store = EventStore(tmp_path / "room.sqlite3")
    try:
        store.initialize()
        store.create_run(RunRecord("run-1", "room-1", WALL, make_config()))
        yield store
    finally:
        store.close()


def add(store: EventStore, event_type: str, sim_ms: int, payload: dict[str, Any] | None = None,
        agent_id: str | None = None, run_id: str = "run-1") -> int:
    return store.commit_event(EventRecord(run_id, WALL, sim_ms, event_type, agent_id, payload or {}))


def say(store: EventStore, agent_id: str, sim_ms: int, text: str, latency_ms: float = 100.0) -> int:
    return add(store, "message", sim_ms, {
        "speaker": agent_id.title(), "message": text, "target": None,
        "scheduler_score": 0.5, "reasons": [], "latency_ms": latency_ms,
        "prompt_tokens": None, "output_tokens": None, "model": "qwen3:4b", "attempt": 1,
    }, agent_id)


def wait(store: EventStore, agent_id: str, sim_ms: int, latency_ms: float = 100.0) -> int:
    return add(store, "agent_wait", sim_ms, {
        "scheduler_score": 0.5, "reasons": [], "latency_ms": latency_ms,
        "prompt_tokens": None, "output_tokens": None, "model": "qwen3:4b", "attempt": 1,
    }, agent_id)


def fail(store: EventStore, agent_id: str, sim_ms: int, error_class: str) -> int:
    return add(store, "attempt_failed", sim_ms, {
        "attempt": 1, "error_class": error_class, "error": "boom", "scheduler_score": 0.5, "reasons": [],
    }, agent_id)


class ReadOnlyStore:
    """Exposes only the store's read API, so any write attempt fails loudly."""

    READS = frozenset({"read_events", "get_run"})

    def __init__(self, store: EventStore) -> None:
        self._store = store
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name not in self.READS:
            raise AssertionError(f"analysis touched EventStore.{name}")
        self.calls.append(name)
        return getattr(self._store, name)


def all_events(store: EventStore) -> list[StoredEvent]:
    return store.read_events("run-1", limit=100_000)


def test_analyze_run_reports_every_metric_for_a_synthetic_run(store) -> None:
    add(store, "session_started", 0)
    say(store, "june", 0, "hey", latency_ms=100.0)                      # 1 word
    wait(store, "milo", 5_000, latency_ms=300.0)
    say(store, "milo", 10_000, "That's a great point, as an AI I agree", latency_ms=200.0)  # 9 words
    fail(store, "ada", 15_000, "BackendTimeoutError")
    fail(store, "ada", 15_000, "DecisionValidationError")
    add(store, "generation_failed", 15_000, {"attempts": 2, "last_error_class": "DecisionValidationError"}, "ada")
    fail(store, "ada", 20_000, "BackendTimeoutError")
    wait(store, "ada", 20_000, latency_ms=400.0)
    add(store, "environment", 70_000, {"text": "the room has been quiet for a while"})  # 60 s gap: silence
    say(store, "june", 75_000, "in conclusion idk lol", latency_ms=500.0)  # 4 words
    add(store, "session_ended", 75_000, {"sim_end_ms": 75_000})
    # Another run in the same file must not leak into this run's metrics.
    store.create_run(RunRecord("run-2", "room-2", WALL, make_config()))
    add(store, "message", 0, {"speaker": "June", "message": "As an AI, other run", "target": None,
                              "latency_ms": 9_999.0}, "june", run_id="run-2")

    reader = ReadOnlyStore(store)
    metrics = analyze_run(reader, "run-1")

    assert metrics == RunMetrics(
        visible_messages=3,
        valid_waits=2,
        generation_failures=1,
        attempt_failures=3,
        messages_per_agent={"june": 2, "milo": 1, "ada": 0},
        mean_message_words=(1 + 9 + 4) / 3,
        median_message_words=4.0,
        assistant_phrase_hits=3,
        as_an_ai_hits=1,
        silence_periods=1,
        ambient_events=1,
        mean_inference_ms=(100.0 + 300.0 + 200.0 + 400.0 + 500.0) / 5,
        failure_classes={"BackendTimeoutError": 2, "DecisionValidationError": 1},
        decisions_per_sim_minute=(3 + 2 + 1) / 1.25,  # 75 s of simulated time
        wait_ratio=2 / (3 + 2),
        median_inference_ms=300.0,
        repeated_message_ratio=0.0,
        distinct_token_ratio=1.0,  # 15 tokens over three messages, none repeated
        self_repetition_ratio=0.0,
    )
    assert set(reader.calls) == {"read_events", "get_run"}


def test_empty_run_has_zero_counts_and_no_inference_mean(store) -> None:
    add(store, "session_started", 0)

    metrics = analyze_run(store, "run-1")

    assert metrics == RunMetrics(
        visible_messages=0, valid_waits=0, generation_failures=0, attempt_failures=0,
        messages_per_agent={"june": 0, "milo": 0, "ada": 0},
        mean_message_words=0.0, median_message_words=0.0,
        assistant_phrase_hits=0, as_an_ai_hits=0, silence_periods=0, ambient_events=0,
        mean_inference_ms=None, failure_classes={},
        decisions_per_sim_minute=0.0, wait_ratio=0.0, median_inference_ms=None,
        repeated_message_ratio=0.0, distinct_token_ratio=0.0, self_repetition_ratio=0.0,
    )


def test_the_startup_line_is_not_counted_as_an_ambient_event(store) -> None:
    add(store, "session_started", 0)
    add(store, "environment", 0, {"text": "June, Milo and Ada are in the room", "kind": "startup"})
    add(store, "environment", SILENCE_MS, {"text": "the room has been quiet for a while"})

    assert analyze_run(store, "run-1").ambient_events == 1


def test_unknown_run_is_an_error(store) -> None:
    with pytest.raises(KeyError, match="run-x"):
        analyze_run(store, "run-x")


@pytest.mark.parametrize("phrase", [
    "That's a great point", "I completely agree", "Building on what you said",
    "It's important to note", "In conclusion", "As an AI",
])
def test_each_spec_assistant_phrase_is_detected_case_insensitively(store, phrase) -> None:
    say(store, "june", 0, f"ok {phrase.upper()}. and again: {phrase.lower()}!")

    metrics = analyze_run(store, "run-1")

    assert metrics.assistant_phrase_hits == 2
    assert metrics.as_an_ai_hits == (2 if phrase == "As an AI" else 0)


def test_phrase_scan_matches_whole_phrases_only_and_ignores_environment_text(store) -> None:
    say(store, "june", 0, "she worked as an aide; in conclusionary terms, whatever")
    add(store, "environment", 1_000, {"text": "As an AI, in conclusion"})

    metrics = analyze_run(store, "run-1")

    assert (metrics.assistant_phrase_hits, metrics.as_an_ai_hits) == (0, 0)


def test_analysis_never_alters_persisted_text(store) -> None:
    say(store, "june", 0, "  As an AI,   I completely agree\n\nIn Conclusion  ")
    before = all_events(store)

    analyze_run(store, "run-1")

    assert all_events(store) == before


def test_silence_periods_count_visible_gaps_at_or_over_the_threshold(store) -> None:
    add(store, "session_started", 0)
    say(store, "june", 0, "a")
    wait(store, "milo", 30_000)                        # invisible: does not break the silence
    say(store, "milo", SILENCE_MS, "b")                # gap == threshold: silence
    add(store, "environment", 2 * SILENCE_MS - 1, {"text": "quiet"})  # gap just under: not silence
    say(store, "ada", 4 * SILENCE_MS, "c")             # long gap: still one period
    wait(store, "june", 5 * SILENCE_MS)                # trailing gap to the last event: silence

    assert analyze_run(store, "run-1").silence_periods == 3


def test_trailing_gap_under_threshold_is_not_silence(store) -> None:
    say(store, "june", 0, "a")
    add(store, "session_ended", SILENCE_MS - 1, {"sim_end_ms": SILENCE_MS - 1})

    assert analyze_run(store, "run-1").silence_periods == 0


def test_analysis_pages_through_every_event(store, monkeypatch) -> None:
    monkeypatch.setattr(analysis, "PAGE_SIZE", 2)
    for index in range(7):
        say(store, "june", index * 1_000, f"line {index}")

    metrics = analyze_run(store, "run-1")

    assert metrics.visible_messages == 7
    assert metrics.messages_per_agent["june"] == 7


@pytest.mark.parametrize("text", [
    "That's a great point and It's important to note it",
    "That’s a great point and It’s important to note it",
    "That’s a great point and It's important to note it",
])
def test_apostrophe_phrases_match_straight_and_curly_apostrophes(store, text) -> None:
    say(store, "june", 0, text)
    before = all_events(store)

    metrics = analyze_run(store, "run-1")

    assert (metrics.assistant_phrase_hits, metrics.as_an_ai_hits) == (2, 0)
    assert all_events(store) == before  # the stored text keeps its original apostrophes


def test_message_from_an_agent_outside_the_run_config_is_counted_under_its_id(store) -> None:
    say(store, "june", 0, "hi")
    say(store, "stranger", 1_000, "who am I")
    say(store, "stranger", 2_000, "still here")

    metrics = analyze_run(store, "run-1")

    assert metrics.messages_per_agent == {"june": 1, "milo": 0, "ada": 0, "stranger": 2}
    assert metrics.visible_messages == 3


def test_message_without_an_agent_id_is_a_value_error_not_an_assertion(store) -> None:
    event_id = add(store, "message", 0, {"speaker": "?", "message": "hi", "target": None, "latency_ms": 1.0})

    with pytest.raises(ValueError, match=f"message event {event_id} has no agent"):
        analyze_run(store, "run-1")


def test_silence_from_the_first_session_start_to_the_first_visible_event_counts(store) -> None:
    add(store, "session_started", 0)
    wait(store, "milo", 30_000)                        # invisible: the room is still silent
    say(store, "june", SILENCE_MS, "a")                # opening gap == threshold: silence
    add(store, "session_ended", SILENCE_MS, {"sim_end_ms": SILENCE_MS})
    add(store, "session_started", SILENCE_MS, {})      # a restart's session start is not the run's start
    say(store, "milo", 2 * SILENCE_MS - 1, "b")

    assert analyze_run(store, "run-1").silence_periods == 1


def test_opening_gap_under_threshold_is_not_silence(store) -> None:
    add(store, "session_started", 0)
    say(store, "june", SILENCE_MS - 1, "a")

    assert analyze_run(store, "run-1").silence_periods == 0


def test_cadence_counts_every_decision_over_the_simulated_span(store) -> None:
    add(store, "session_started", 60_000)
    wait(store, "june", 60_000)
    wait(store, "milo", 90_000)
    add(store, "generation_failed", 120_000, {"attempts": 2, "last_error_class": "BackendTimeoutError"}, "ada")
    say(store, "june", 150_000, "hi")
    add(store, "session_ended", 180_000, {"sim_end_ms": 180_000})

    metrics = analyze_run(store, "run-1")

    assert metrics.decisions_per_sim_minute == 4 / 2  # four decisions over two simulated minutes
    assert metrics.wait_ratio == 2 / 3
    assert metrics.median_inference_ms == 100.0


def test_identical_messages_are_fully_repeated_with_a_low_distinct_token_ratio(store) -> None:
    add(store, "session_started", 0)
    for sim_ms, speaker in ((0, "june"), (5_000, "milo"), (10_000, "ada")):
        say(store, speaker, sim_ms, "the harbor lights glow")

    metrics = analyze_run(store, "run-1")

    assert metrics.repeated_message_ratio == 1.0
    assert metrics.distinct_token_ratio == pytest.approx(4 / 12)


def test_disjoint_messages_are_never_repeated_and_fully_distinct(store) -> None:
    add(store, "session_started", 0)
    say(store, "june", 0, "alpha bravo")
    say(store, "milo", 5_000, "charlie delta")
    say(store, "ada", 10_000, "echo foxtrot")

    metrics = analyze_run(store, "run-1")

    assert metrics.repeated_message_ratio == 0.0
    assert metrics.distinct_token_ratio == 1.0


def test_repeated_ratio_counts_messages_from_the_second_onward_at_the_threshold(store) -> None:
    # Default threshold 0.6: "a b c" vs "a b c d e" is 3/5 = 0.6 (counts); then a miss.
    add(store, "session_started", 0)
    say(store, "june", 0, "alpha bravo charlie")
    say(store, "milo", 5_000, "alpha bravo charlie delta echo")
    add(store, "environment", 6_000, {"text": "alpha bravo charlie delta echo"})
    say(store, "ada", 10_000, "zulu yankee")

    # Two pairs from the second message onward: one repeat (the environment line is skipped).
    assert analyze_run(store, "run-1").repeated_message_ratio == 0.5


def test_fewer_than_two_messages_or_no_tokens_give_zero_ratios(store) -> None:
    add(store, "session_started", 0)
    assert analyze_run(store, "run-1").repeated_message_ratio == 0.0
    assert analyze_run(store, "run-1").distinct_token_ratio == 0.0
    say(store, "june", 0, "...")
    assert analyze_run(store, "run-1").repeated_message_ratio == 0.0
    assert analyze_run(store, "run-1").distinct_token_ratio == 0.0
    say(store, "milo", 5_000, "...")  # two token-less messages are not a repeat either
    metrics = analyze_run(store, "run-1")
    assert metrics.repeated_message_ratio == 0.0
    assert metrics.distinct_token_ratio == 0.0


def test_repeated_ratio_uses_the_run_configs_threshold(store) -> None:
    other = make_config()
    other.scheduler.repetition_similarity_threshold = 1.0
    store.create_run(RunRecord("run-strict", "room-s", WALL, other))
    for sim_ms, text in ((0, "alpha bravo charlie"), (5_000, "alpha bravo charlie delta")):
        add(store, "message", sim_ms, {"speaker": "June", "message": text, "target": None,
                                       "latency_ms": 1.0}, "june", run_id="run-strict")
        add(store, "message", sim_ms, {"speaker": "June", "message": text, "target": None,
                                       "latency_ms": 1.0}, "june", run_id="run-1")

    assert analyze_run(store, "run-strict").repeated_message_ratio == 0.0  # 0.75 < 1.0
    assert analyze_run(store, "run-1").repeated_message_ratio == 1.0  # 0.75 >= 0.6


def _alternating_loop(store: EventStore, run_id: str = "run-1") -> None:
    # June: A then A'; Milo: B then B'; A !~ B.
    for sim_ms, speaker, text in (
        (0, "june", "the harbor lights glow tonight"),
        (5_000, "milo", "completely unrelated words about breakfast"),
        (10_000, "june", "the harbor lights glow tonight friend"),
        (15_000, "milo", "completely unrelated words about breakfast again"),
    ):
        add(store, "message", sim_ms, {"speaker": speaker.title(), "message": text, "target": None,
                                       "latency_ms": 1.0}, speaker, run_id=run_id)


def test_an_alternating_loop_is_repeated_over_the_window_and_every_repeat_is_a_self_copy(store) -> None:
    _alternating_loop(store)

    metrics = analyze_run(store, "run-1")

    assert metrics.repeated_message_ratio == pytest.approx(2 / 3)  # window 5: A' and B' of 3 pairs
    assert metrics.self_repetition_ratio == 1.0  # A' and B' both copy their speaker's last line


def test_repeated_ratio_with_window_one_is_the_consecutive_pair_definition(store) -> None:
    other = make_config()
    other.scheduler.repetition_window = 1
    store.create_run(RunRecord("run-narrow", "room-n", WALL, other))
    _alternating_loop(store, "run-narrow")

    metrics = analyze_run(store, "run-narrow")

    assert metrics.repeated_message_ratio == 0.0
    assert metrics.self_repetition_ratio == 1.0  # the self metric does not depend on the window


def test_disjoint_messages_have_no_self_repetition(store) -> None:
    add(store, "session_started", 0)
    say(store, "june", 0, "alpha bravo")
    say(store, "milo", 5_000, "charlie delta")
    say(store, "ada", 10_000, "echo foxtrot")

    assert analyze_run(store, "run-1").self_repetition_ratio == 0.0


def test_self_repetition_counts_only_messages_with_an_earlier_message_by_the_same_speaker(store) -> None:
    add(store, "session_started", 0)
    say(store, "june", 0, "alpha bravo charlie")
    say(store, "milo", 5_000, "alpha bravo charlie")  # copies June, but Milo has no earlier message
    say(store, "june", 10_000, "zulu yankee xray")  # June, no repeat
    say(store, "milo", 15_000, "alpha bravo charlie")  # Milo copies his own previous line

    assert analyze_run(store, "run-1").self_repetition_ratio == 0.5  # 1 of 2 eligible messages
