"""Cheap, deterministic conversational opportunity scoring."""

from datetime import datetime, timezone
from pathlib import Path
from random import Random

import pytest
from pydantic import ValidationError

from driftroom.config import load_run_config
from driftroom.domain import (
    VISIBLE_EVENT_TYPES, AgentConfig, AgentTraits, SchedulerConfig,
)
from driftroom.clock import VirtualClock
from driftroom.scheduler import CandidateScore, Scheduler
from driftroom.storage import StoredEvent


EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "driftroom.example.toml"


def agent(agent_id: str, name: str, *, reserved: float = 0.5) -> AgentConfig:
    return AgentConfig(
        id=agent_id, name=name, model="fake", traits=AgentTraits(reserved=reserved)
    )


def message(event_id: int, sim_ms: int, speaker: str, text: str) -> StoredEvent:
    return StoredEvent(
        event_id, "run-1", "2026-10-01T12:00:00Z", sim_ms,
        "message", speaker, {"message": text},
    )


def neutral_config(**overrides: float | int) -> SchedulerConfig:
    values = dict(
        base_bias=0, talkativeness_weight=0, direct_mention_bonus=0,
        topic_overlap_weight=0, elapsed_weight=0, relationship_weight=0,
        recent_speaker_penalty=0, cooldown_penalty=0, random_jitter=0,
    )
    values.update(overrides)
    return SchedulerConfig(**values)


def test_scheduler_rejects_nonpositive_time_intervals() -> None:
    for field in ("decision_tick_ms", "silence_ambient_after_ms"):
        with pytest.raises(ValidationError):
            SchedulerConfig(**{field: 0})


def test_experimental_v1_scheduler_defaults_are_recorded() -> None:
    config = SchedulerConfig()
    assert config.model_dump() == {
        "base_bias": -0.35,
        "talkativeness_weight": 0.45,
        "direct_mention_bonus": 0.55,
        "topic_overlap_weight": 0.25,
        "elapsed_weight": 0.30,
        "relationship_weight": 0.10,
        "recent_speaker_penalty": 0.45,
        "cooldown_penalty": 1.00,
        "random_jitter": 0.15,
        "candidate_threshold": 0.20,
        "decision_tick_ms": 5_000,
        "silence_ambient_after_ms": 300_000,
        "ambient_min_interval_ms": 900_000,
        "speaker_cooldown_ms": 20_000,
    }


def test_default_regime_lets_every_agent_qualify_and_room_recover() -> None:
    config = load_run_config(EXAMPLE_CONFIG)
    scheduler = Scheduler(config.scheduler, rng=Random(1))
    agents = config.agents

    blank = [scheduler.select_candidate(agents, [], 10_000_000) for _ in range(2_000)]
    after_milo = [message(1, 10_000_000, "milo", "the light in here is kind of strange")]
    recovering = [
        scheduler.select_candidate(agents, after_milo, 10_060_000) for _ in range(2_000)
    ]

    assert {c.agent_id for c in blank if c is not None} == {a.id for a in agents}
    assert None in blank
    assert {c.agent_id for c in recovering if c is not None} - {"milo"}

def test_reasons_name_only_nonzero_score_contributions() -> None:
    june = agent("june", "June")
    scheduler = Scheduler(neutral_config(), rng=Random(3))

    score = scheduler.score_agents([june], [message(1, 0, "june", "June")], 50_000)[0]

    assert score.reasons == ()


def test_direct_mention_adds_exact_configured_bonus() -> None:
    june = agent("june", "June")
    config = neutral_config(direct_mention_bonus=0.55)
    scheduler = Scheduler(config, rng=Random(3))

    baseline = scheduler.score_agents([june], [message(1, 0, "atlas", "hello")], 50_000)[0]
    mentioned = scheduler.score_agents([june], [message(1, 0, "atlas", "hello June")], 50_000)[0]

    assert mentioned.score - baseline.score == pytest.approx(0.55)
    assert "direct_mention" in mentioned.reasons


def test_cooldown_penalty_expires_at_configured_boundary() -> None:
    june = agent("june", "June")
    scheduler = Scheduler(neutral_config(cooldown_penalty=1.0), rng=Random(3))
    history = [message(1, 10_000, "june", "hi")]

    inside = scheduler.score_agents([june], history, 29_999)[0]
    expired = scheduler.score_agents([june], history, 30_000)[0]

    assert expired.score - inside.score == pytest.approx(1.0)
    assert "cooldown" in inside.reasons


def test_recent_decision_applies_cooldown_until_configured_boundary() -> None:
    june = agent("june", "June")
    scheduler = Scheduler(
        neutral_config(cooldown_penalty=1.0, speaker_cooldown_ms=20_000), rng=Random(3)
    )

    fresh = scheduler.score_agents([june], [], 50_000)[0]
    inside = scheduler.score_agents([june], [], 50_000, last_decision_ms={"june": 40_000})[0]
    expired = scheduler.score_agents([june], [], 50_000, last_decision_ms={"june": 30_000})[0]
    other = scheduler.score_agents([june], [], 50_000, last_decision_ms={"milo": 40_000})[0]

    assert fresh.score - inside.score == pytest.approx(1.0)
    assert inside.reasons == ("decision_cooldown",)
    assert expired.score == fresh.score and "decision_cooldown" not in expired.reasons
    assert other == fresh


def test_speaker_and_decision_cooldowns_apply_at_most_one_penalty() -> None:
    june = agent("june", "June")
    scheduler = Scheduler(
        neutral_config(cooldown_penalty=1.0, speaker_cooldown_ms=20_000), rng=Random(3)
    )
    spoke = [message(1, 40_000, "june", "hi")]

    speaking = scheduler.score_agents([june], spoke, 50_000)[0]
    both = scheduler.score_agents([june], spoke, 50_000, last_decision_ms={"june": 40_000})[0]
    selected = scheduler.select_candidate([june], spoke, 50_000, last_decision_ms={"june": 40_000})

    assert both == speaking
    assert both.score == pytest.approx(-1.0)
    assert both.reasons == ("cooldown",)
    assert selected is None


def test_latest_speaker_gets_recent_participation_penalty() -> None:
    june = agent("june", "June")
    scheduler = Scheduler(neutral_config(recent_speaker_penalty=0.45), rng=Random(3))

    recent = scheduler.score_agents([june], [message(1, 0, "june", "hi")], 50_000)[0]
    older = scheduler.score_agents(
        [june], [message(1, 0, "june", "hi"), message(2, 25_000, "atlas", "hey")], 50_000
    )[0]

    assert older.score - recent.score == pytest.approx(0.45)
    assert "recent_speaker" in recent.reasons


def test_topic_overlap_uses_only_last_three_own_visible_messages() -> None:
    june = agent("june", "June")
    scheduler = Scheduler(neutral_config(topic_overlap_weight=0.25), rng=Random(3))
    history = [
        message(1, 0, "june", "pumpkin"),
        message(2, 1_000, "june", "maple"),
        message(3, 2_000, "june", "cedar"),
        message(4, 3_000, "june", "birch"),
        message(5, 4_000, "atlas", "PUMPKIN, cedar?"),
    ]

    score = scheduler.score_agents([june], history, 50_000)[0]
    no_history = scheduler.score_agents([june], history[-1:], 50_000)[0]

    assert score.score == pytest.approx(0.125)
    assert no_history.score == 0
    assert "topic_overlap" in score.reasons


def test_elapsed_urge_uses_simulated_milliseconds() -> None:
    june = agent("june", "June")
    scheduler = Scheduler(neutral_config(elapsed_weight=0.30), rng=Random(3))
    history = [message(1, 0, "atlas", "hello")]

    initial = scheduler.score_agents([june], history, 0)[0]
    later = scheduler.score_agents([june], history, 300_000)[0]

    assert later.score - initial.score == pytest.approx(0.30)
    assert "elapsed" in later.reasons


def environment(event_id: int, sim_ms: int, text: str) -> StoredEvent:
    return StoredEvent(
        event_id, "run-1", "2026-10-01T12:00:00Z", sim_ms,
        "environment", None, {"text": text},
    )


def test_environment_events_are_visible_but_never_mentions() -> None:
    milo = agent("milo", "Milo")
    june = agent("june", "June")
    scheduler = Scheduler(
        neutral_config(
            recent_speaker_penalty=0.45, direct_mention_bonus=0.55,
            topic_overlap_weight=0.25,
        ),
        rng=Random(3),
    )
    history = [
        message(1, 0, "milo", "has june returned"),
        environment(2, 300_000, "June has returned"),
    ]

    scores = {
        score.agent_id: score
        for score in scheduler.score_agents([milo, june], history, 300_000)
    }

    assert VISIBLE_EVENT_TYPES == frozenset({"message", "environment"})
    assert "recent_speaker" not in scores["milo"].reasons
    assert "topic_overlap" not in scores["milo"].reasons
    assert "direct_mention" not in scores["june"].reasons

def test_traits_and_optional_state_scale_urge() -> None:
    june = agent("june", "June", reserved=0.2)
    scheduler = Scheduler(
        neutral_config(talkativeness_weight=0.45, relationship_weight=0.10),
        rng=Random(3),
    )
    history = [message(1, 0, "atlas", "hello")]

    score = scheduler.score_agents(
        [june], history, 0,
        energy={"june": 0.5}, attention={"june": 0.5},
        relationship_affinity={"june": 0.4},
    )[0]

    assert score.score == pytest.approx(0.13)
    assert "talkativeness" in score.reasons
    assert "relationship" in score.reasons


def test_fixed_seed_reproduces_jitter_and_candidate_order() -> None:
    agents = [agent("june", "June"), agent("atlas", "Atlas"), agent("sora", "Sora")]
    config = neutral_config(random_jitter=0.15)

    first = Scheduler(config, rng=Random(7)).score_agents(agents, [], 0)
    second = Scheduler(config, rng=Random(7)).score_agents(agents, [], 0)

    assert first == second
    assert [candidate.agent_id for candidate in first] == ["sora", "june", "atlas"]
    assert any(candidate.score != 0 for candidate in first)


class ZeroUniform(Random):
    def uniform(self, a: float, b: float) -> float:
        return 0.0


def test_zero_jitter_draw_is_not_reported_as_a_reason() -> None:
    june = agent("june", "June")
    scheduler = Scheduler(neutral_config(random_jitter=0.15), rng=ZeroUniform(3))

    score = scheduler.score_agents([june], [], 0)[0]

    assert score.score == 0
    assert "jitter" not in score.reasons

def test_selection_returns_none_below_threshold_and_accepts_boundary() -> None:
    june = agent("june", "June")
    below = Scheduler(
        neutral_config(base_bias=0.349, candidate_threshold=0.35), rng=Random(3)
    )
    boundary = Scheduler(
        neutral_config(base_bias=0.35, candidate_threshold=0.35), rng=Random(3)
    )

    assert below.select_candidate([june], [], 0) is None
    assert boundary.select_candidate([june], [], 0).agent_id == "june"


def test_select_from_reuses_one_scored_list_without_drawing_rng() -> None:
    agents = [agent("june", "June"), agent("atlas", "Atlas"), agent("sora", "Sora")]
    config = neutral_config(base_bias=0.40, random_jitter=0.15, candidate_threshold=0.20)
    scheduler = Scheduler(config, rng=Random(7))

    scores = scheduler.score_agents(agents, [], 0)
    state_after_scoring = scheduler.rng.getstate()
    first = scheduler.select_from(scores)
    second = scheduler.select_from(scores)

    assert first is scores[0]
    assert second is first
    assert scheduler.rng.getstate() == state_after_scoring
    assert scheduler.select_from([]) is None
    assert scheduler.select_from([CandidateScore("june", 0.199, ())]) is None


def test_select_candidate_draws_rng_like_one_scoring_pass() -> None:
    agents = [agent("june", "June"), agent("atlas", "Atlas"), agent("sora", "Sora")]
    config = neutral_config(random_jitter=0.15)
    scored = Scheduler(config, rng=Random(11))
    selected = Scheduler(config, rng=Random(11))

    scored.score_agents(agents, [], 0)
    selected.select_candidate(agents, [], 0)

    assert selected.rng.getstate() == scored.rng.getstate()

def test_quiet_realtime_room_yields_one_decision_tick() -> None:
    scheduler = Scheduler(neutral_config(), rng=Random(3))
    monotonic = [10.0]
    clock = VirtualClock(
        "realtime", datetime(2026, 10, 1, tzinfo=timezone.utc),
        monotonic_fn=lambda: monotonic[0],
    )
    sleeps: list[float] = []

    def sleep_fn(seconds: float) -> None:
        sleeps.append(seconds)
        monotonic[0] += seconds

    scheduler.wait_for_next_tick(clock, sleep_fn=sleep_fn)

    assert sleeps == [5.0]
    assert clock.now_ms() == 5_000


def test_thousand_quiet_accelerated_ticks_advance_without_model_calls() -> None:
    scheduler = Scheduler(neutral_config(), rng=Random(3))
    clock = VirtualClock("accelerated", datetime(2026, 10, 1, tzinfo=timezone.utc))
    june = agent("june", "June")
    model_calls = 0

    def forbidden_sleep(_seconds: float) -> None:
        raise AssertionError("accelerated time must not sleep")

    for _ in range(1_000):
        candidate = scheduler.select_candidate([june], [], clock.now_ms())
        if candidate is not None:
            model_calls += 1
        else:
            scheduler.wait_for_next_tick(clock, sleep_fn=forbidden_sleep)

    assert model_calls == 0
    assert clock.now_ms() == 5_000_000
