"""Cheap, deterministic conversational opportunity scoring."""

import asyncio
from datetime import datetime, timezone
from random import Random

import pytest
from pydantic import ValidationError

from driftroom.domain import AgentConfig, AgentTraits, SchedulerConfig
from driftroom.clock import VirtualClock
from driftroom.scheduler import Scheduler
from driftroom.storage import StoredEvent


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
        "candidate_threshold": 0.35,
        "decision_tick_ms": 5_000,
        "silence_ambient_after_ms": 300_000,
        "ambient_min_interval_ms": 900_000,
        "speaker_cooldown_ms": 20_000,
    }


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


def test_selection_returns_none_below_threshold_and_accepts_boundary() -> None:
    june = agent("june", "June")
    below = Scheduler(neutral_config(base_bias=0.349), rng=Random(3))
    boundary = Scheduler(neutral_config(base_bias=0.35), rng=Random(3))

    assert below.select_candidate([june], [], 0) is None
    assert boundary.select_candidate([june], [], 0).agent_id == "june"


def test_quiet_realtime_room_yields_one_decision_tick() -> None:
    scheduler = Scheduler(neutral_config(), rng=Random(3))
    monotonic = [10.0]
    clock = VirtualClock(
        "realtime", datetime(2026, 10, 1, tzinfo=timezone.utc),
        monotonic_fn=lambda: monotonic[0],
    )
    sleeps: list[float] = []

    async def sleep_fn(seconds: float) -> None:
        sleeps.append(seconds)
        monotonic[0] += seconds

    asyncio.run(scheduler.wait_for_next_tick(clock, sleep_fn=sleep_fn))

    assert sleeps == [5.0]
    assert clock.now_ms() == 5_000


def test_thousand_quiet_accelerated_ticks_advance_without_model_calls() -> None:
    scheduler = Scheduler(neutral_config(), rng=Random(3))
    clock = VirtualClock("accelerated", datetime(2026, 10, 1, tzinfo=timezone.utc))
    june = agent("june", "June")
    model_calls = 0

    async def forbidden_sleep(_seconds: float) -> None:
        raise AssertionError("accelerated time must not sleep")

    async def run_quiet_room() -> None:
        nonlocal model_calls
        for _ in range(1_000):
            candidate = scheduler.select_candidate([june], [], clock.now_ms())
            if candidate is not None:
                model_calls += 1
            else:
                await scheduler.wait_for_next_tick(clock, sleep_fn=forbidden_sleep)

    asyncio.run(run_quiet_room())

    assert model_calls == 0
    assert clock.now_ms() == 5_000_000
