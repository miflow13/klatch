import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from driftroom.config import (
    AgentConfig,
    AgentTraits,
    RunConfig,
    RuntimeConfig,
    SamplingConfig,
    SchedulerConfig,
    canonical_config_json,
    load_run_config,
    run_config_hash,
)


def test_configuration_defaults() -> None:
    assert SamplingConfig().temperature == 0.8
    assert SamplingConfig().top_p == 0.9
    assert SamplingConfig().top_k == 40
    assert SamplingConfig().repeat_penalty == 1.08
    assert RuntimeConfig().startup_mode == "blank"
    assert RuntimeConfig().runtime_mode == "balanced"
    assert RuntimeConfig().model_thinking is False


def test_example_has_three_agents_on_shared_model() -> None:
    config = load_run_config(Path("driftroom.example.toml"))

    assert len(config.agents) == 3
    assert {agent.model for agent in config.agents} == {"qwen3:4b"}


def test_duplicate_agent_ids_fail_validation() -> None:
    traits = AgentTraits()
    agent = AgentConfig(id="same", name="June", model="qwen3:4b", traits=traits)

    with pytest.raises(ValidationError, match="unique"):
        RunConfig(agents=[agent, agent, agent])


def test_run_config_accepts_more_than_three_agents() -> None:
    agents = [
        AgentConfig(id=f"agent-{index}", name=f"Agent {index}", model="qwen3:4b")
        for index in range(4)
    ]

    assert len(RunConfig(agents=agents).agents) == 4


def test_traits_are_bounded_and_runtime_has_required_defaults() -> None:
    assert AgentTraits().reserved == 0.5
    with pytest.raises(ValidationError):
        AgentTraits(curiosity=1.1)
    runtime = RuntimeConfig()
    assert runtime.recent_context_events == 20
    assert runtime.max_output_tokens == 256
    assert runtime.inference_timeout_seconds == 120
    assert runtime.retry_count == 1


@pytest.mark.parametrize(
    "values",
    [
        {"temperature": -0.1},
        {"top_p": 1.01},
        {"top_k": -1},
        {"repeat_penalty": 0},
        {"temperature": float("nan")},
        {"repeat_penalty": float("inf")},
    ],
)
def test_sampling_rejects_out_of_range_and_nonfinite_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SamplingConfig(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"recent_context_events": -1},
        {"max_output_tokens": 0},
        {"inference_timeout_seconds": 0},
        {"inference_timeout_seconds": float("inf")},
        {"retry_count": -1},
    ],
)
def test_runtime_rejects_invalid_operational_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(**values)


def test_canonical_json_is_stable_and_hash_tracks_behavior_parameters() -> None:
    config = load_run_config(Path("driftroom.example.toml"))
    canonical = canonical_config_json(config)

    # Key order and separators: re-serializing the parsed output must be a no-op.
    assert canonical == json.dumps(
        json.loads(canonical), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )

    sampling_changed = config.model_copy(deep=True)
    sampling_changed.agents[0].sampling.temperature += 0.01
    scheduler_changed = config.model_copy(deep=True)
    scheduler_changed.scheduler.decision_tick_ms += 1

    original_hash = run_config_hash(config)
    assert run_config_hash(sampling_changed) != original_hash
    assert run_config_hash(scheduler_changed) != original_hash
    assert len(original_hash) == 64


def test_canonical_json_matches_independently_written_expected_string() -> None:
    config = RunConfig(
        agents=[AgentConfig(id=ident, name=ident.upper(), model="m") for ident in "abc"]
    )

    def agent(ident: str) -> str:
        return (
            '{"id":"' + ident + '","model":"m","name":"' + ident.upper() + '",'
            '"sampling":{"repeat_penalty":1.08,"temperature":0.8,"top_k":40,"top_p":0.9},'
            '"traits":{"curiosity":0.5,"formality":0.5,"humor":0.5,'
            '"impulsiveness":0.5,"reserved":0.5}}'
        )

    expected = (
        '{"agents":[' + ",".join(agent(ident) for ident in "abc") + "],"
        '"runtime":{"inference_timeout_seconds":120.0,"max_output_tokens":256,'
        '"model_thinking":false,"recent_context_events":20,"retry_count":1,'
        '"runtime_mode":"balanced","startup_mode":"blank"},'
        '"scheduler":{"ambient_min_interval_ms":900000,"base_bias":-0.35,'
        '"candidate_threshold":0.35,"cooldown_penalty":1.0,"decision_tick_ms":5000,'
        '"direct_mention_bonus":0.55,"elapsed_weight":0.3,"random_jitter":0.15,'
        '"recent_speaker_penalty":0.45,"relationship_weight":0.1,'
        '"silence_ambient_after_ms":300000,"speaker_cooldown_ms":20000,'
        '"talkativeness_weight":0.45,"topic_overlap_weight":0.25}}'
    )

    assert canonical_config_json(config) == expected


def test_toml_and_constructor_configs_hash_identically() -> None:
    loaded = load_run_config(Path("driftroom.example.toml"))
    built = RunConfig(
        agents=[
            AgentConfig(
                model="qwen3:4b",
                name=name,
                id=ident,
                traits=AgentTraits(
                    formality=formality,
                    impulsiveness=impulsiveness,
                    humor=humor,
                    curiosity=curiosity,
                    reserved=reserved,
                ),
            )
            for ident, name, reserved, curiosity, humor, impulsiveness, formality in [
                ("june", "June", 0.7, 0.8, 0.5, 0.25, 0.2),
                ("milo", "Milo", 0.3, 0.6, 0.8, 0.65, 0.2),
                ("ada", "Ada", 0.5, 0.9, 0.4, 0.35, 0.6),
            ]
        ],
        runtime=RuntimeConfig(inference_timeout_seconds=120),
    )

    assert run_config_hash(built) == run_config_hash(loaded)
