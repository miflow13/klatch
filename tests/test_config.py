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


def test_canonical_json_is_stable_and_hash_tracks_behavior_parameters() -> None:
    config = load_run_config(Path("driftroom.example.toml"))
    assert canonical_config_json(config) == canonical_config_json(config)

    sampling_changed = config.model_copy(deep=True)
    sampling_changed.agents[0].sampling.temperature += 0.01
    scheduler_changed = config.model_copy(deep=True)
    scheduler_changed.scheduler.decision_tick_ms += 1

    original_hash = run_config_hash(config)
    assert run_config_hash(sampling_changed) != original_hash
    assert run_config_hash(scheduler_changed) != original_hash
    assert len(original_hash) == 64
