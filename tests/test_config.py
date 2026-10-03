import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from driftroom.config import (
    ENGINE_VERSION,
    AgentConfig,
    AgentTraits,
    RunConfig,
    RuntimeConfig,
    SamplingConfig,
    SchedulerConfig,
    canonical_config_json,
    load_run_config,
    run_config_hash,
    run_fingerprint,
)


def test_configuration_defaults() -> None:
    assert SamplingConfig().temperature == 0.8
    assert SamplingConfig().top_p == 0.9
    assert SamplingConfig().top_k == 40
    assert SamplingConfig().repeat_penalty == 1.08
    assert SamplingConfig().repeat_last_n == 1024
    assert RuntimeConfig().startup_mode == "blank"
    assert RuntimeConfig().topic is None
    assert RuntimeConfig().runtime_mode == "balanced"
    assert RuntimeConfig().model_thinking is False
    assert RuntimeConfig().random_seed is None
    assert RuntimeConfig().clock_mode == "realtime"
    assert RuntimeConfig().clock_speed == 1.0
    assert RuntimeConfig().max_context_tokens == 8192


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


def test_repeat_last_n_accepts_the_ollama_boundary_values() -> None:
    assert SamplingConfig(repeat_last_n=-1).repeat_last_n == -1
    assert SamplingConfig(repeat_last_n=0).repeat_last_n == 0


@pytest.mark.parametrize(
    "values",
    [
        {"temperature": -0.1},
        {"top_p": 1.01},
        {"top_k": -1},
        {"repeat_penalty": 0},
        {"repeat_last_n": -2},
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
        {"clock_speed": 0},
        {"clock_speed": -1.0},
        {"clock_speed": float("inf")},
        {"clock_mode": "paused"},
        {"max_context_tokens": 0},
        {"random_seed": 1.5},
    ],
)
def test_runtime_rejects_invalid_operational_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(**values)


@pytest.mark.parametrize("values", [{"repetition_window": 0}, {"repetition_window": -1}])
def test_scheduler_rejects_a_repetition_window_below_one(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SchedulerConfig(**values)


def test_runtime_rejects_model_thinking_in_v0_1() -> None:
    with pytest.raises(ValidationError, match="v0.1 requires model_thinking=false"):
        RuntimeConfig(model_thinking=True)


def test_rejected_model_thinking_assignment_leaves_value_unchanged() -> None:
    runtime = RuntimeConfig()
    with pytest.raises(ValidationError, match="v0.1 requires model_thinking=false"):
        runtime.model_thinking = True
    assert runtime.model_thinking is False

    config = _fingerprint_config()
    with pytest.raises(ValidationError, match="v0.1 requires model_thinking=false"):
        config.runtime.model_thinking = True
    assert config.runtime.model_thinking is False


def test_topic_mode_requires_a_nonblank_topic() -> None:
    assert RuntimeConfig(startup_mode="topic", topic="rain on the roof").topic == "rain on the roof"
    for missing in ({}, {"topic": ""}, {"topic": "  \n\t"}):
        with pytest.raises(ValidationError, match="topic startup mode requires a nonblank topic"):
            RuntimeConfig(startup_mode="topic", **missing)


@pytest.mark.parametrize("mode", ["blank", "environment"])
def test_a_topic_is_rejected_outside_topic_mode(mode: str) -> None:
    assert RuntimeConfig(startup_mode=mode).topic is None
    with pytest.raises(ValidationError, match=f"topic is only allowed with startup_mode='topic', not {mode!r}"):
        RuntimeConfig(startup_mode=mode, topic="rain on the roof")


def test_custom_startup_mode_is_reserved() -> None:
    with pytest.raises(ValidationError, match="custom startup mode is reserved and not available in v0.1"):
        RuntimeConfig(startup_mode="custom")


def test_rejected_startup_assignments_leave_the_runtime_unchanged() -> None:
    runtime = RuntimeConfig()
    with pytest.raises(ValidationError, match="requires a nonblank topic"):
        runtime.startup_mode = "topic"
    with pytest.raises(ValidationError, match="only allowed with startup_mode='topic'"):
        runtime.topic = "rain"
    with pytest.raises(ValidationError, match="reserved"):
        runtime.startup_mode = "custom"
    assert (runtime.startup_mode, runtime.topic) == ("blank", None)
    assert runtime == RuntimeConfig()

    topical = RuntimeConfig(startup_mode="topic", topic="rain")
    with pytest.raises(ValidationError, match="requires a nonblank topic"):
        topical.topic = None
    assert (topical.startup_mode, topical.topic) == ("topic", "rain")


def test_canonical_json_is_canonically_ordered_and_hash_tracks_behavior_parameters() -> None:
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
            '"sampling":{"repeat_last_n":1024,"repeat_penalty":1.08,"temperature":0.8,'
            '"top_k":40,"top_p":0.9},'
            '"traits":{"curiosity":0.5,"formality":0.5,"humor":0.5,'
            '"impulsiveness":0.5,"reserved":0.5}}'
        )

    expected = (
        '{"agents":[' + ",".join(agent(ident) for ident in "abc") + "],"
        '"runtime":{"clock_mode":"realtime","clock_speed":1.0,'
        '"inference_timeout_seconds":120.0,"max_context_tokens":8192,'
        '"max_output_tokens":256,"model_thinking":false,"random_seed":null,'
        '"recent_context_events":20,"retry_count":1,'
        '"runtime_mode":"balanced","startup_mode":"blank","topic":null},'
        '"scheduler":{"ambient_min_interval_ms":900000,"base_bias":-0.35,'
        '"candidate_threshold":0.2,"cooldown_penalty":1.0,"decision_tick_ms":5000,'
        '"direct_mention_bonus":0.55,"elapsed_weight":0.3,"random_jitter":0.15,'
        '"recent_speaker_penalty":0.45,"relationship_weight":0.1,'
        '"repetition_damping":0.5,"repetition_similarity_threshold":0.6,'
        '"repetition_window":5,"silence_ambient_after_ms":300000,"speaker_cooldown_ms":20000,'
        '"talkativeness_weight":0.45,"topic_overlap_weight":0.25,'
        '"wait_cooldown_ms":90000}}'
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
    )

    assert run_config_hash(built) == run_config_hash(loaded)


def _fingerprint_config() -> RunConfig:
    return RunConfig(
        agents=[AgentConfig(id=ident, name=ident.upper(), model="m") for ident in "abc"]
    )


_BASE_DIGESTS = {"m": "sha256:aaa", "n": None}


def _baseline_fingerprint(config: RunConfig | None = None, **overrides: object) -> str:
    kwargs: dict[str, object] = {
        "prompt_hash": "p" * 64,
        "model_digests": dict(_BASE_DIGESTS),
        "engine_version": "driftroom-engine-test",
    }
    kwargs.update(overrides)
    return run_fingerprint(config or _fingerprint_config(), **kwargs)  # type: ignore[arg-type]


def test_run_fingerprint_is_sha256_hex_and_equal_for_equal_inputs() -> None:
    first = _baseline_fingerprint(_fingerprint_config())
    second = _baseline_fingerprint(_fingerprint_config())

    assert len(first) == 64
    assert all(char in "0123456789abcdef" for char in first)
    assert first == second


def test_run_fingerprint_defaults_engine_version_constant() -> None:
    config = _fingerprint_config()
    digests = {"m": None}
    explicit = run_fingerprint(
        config, prompt_hash="p", model_digests=digests, engine_version=ENGINE_VERSION
    )

    assert ENGINE_VERSION == "driftroom-engine-0.1.5"
    assert run_fingerprint(config, prompt_hash="p", model_digests=digests) == explicit


def test_run_fingerprint_requires_a_digest_entry_for_every_agent_model() -> None:
    agents = [
        AgentConfig(id="a", name="A", model="m"),
        AgentConfig(id="b", name="B", model="other"),
        AgentConfig(id="c", name="C", model="third"),
    ]
    config = RunConfig(agents=agents)

    with pytest.raises(ValueError, match="other.*third"):
        run_fingerprint(config, prompt_hash="p", model_digests={"m": "sha256:aaa"})

    # A None digest is still a provided entry.
    run_fingerprint(
        config,
        prompt_hash="p",
        model_digests={"m": None, "other": None, "third": None},
    )


def _with(mutate: Callable[[RunConfig], None]) -> RunConfig:
    config = _fingerprint_config()
    mutate(config)
    return config


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            lambda: _baseline_fingerprint(
                _with(lambda c: setattr(c.agents[0].sampling, "temperature", 0.81))
            ),
            id="sampling",
        ),
        pytest.param(
            lambda: _baseline_fingerprint(
                _with(lambda c: setattr(c.scheduler, "decision_tick_ms", 5001))
            ),
            id="scheduler",
        ),
        pytest.param(
            lambda: _baseline_fingerprint(
                _with(lambda c: setattr(c.runtime, "random_seed", 7))
            ),
            id="random_seed",
        ),
        pytest.param(
            lambda: _baseline_fingerprint(
                _with(lambda c: setattr(c.runtime, "clock_mode", "accelerated"))
            ),
            id="clock_mode",
        ),
        pytest.param(
            lambda: _baseline_fingerprint(
                _with(lambda c: setattr(c.runtime, "clock_speed", 2.0))
            ),
            id="clock_speed",
        ),
        pytest.param(
            lambda: _baseline_fingerprint(
                _with(lambda c: setattr(c.runtime, "max_context_tokens", 4096))
            ),
            id="max_context_tokens",
        ),
        pytest.param(
            lambda: _baseline_fingerprint(
                _with(lambda c: setattr(c, "runtime", c.runtime.model_copy(
                    update={"startup_mode": "topic", "topic": "rain"}
                )))
            ),
            id="topic",
        ),
        pytest.param(
            lambda: _baseline_fingerprint(
                _with(lambda c: setattr(c.runtime, "startup_mode", "environment"))
            ),
            id="startup_mode",
        ),
        pytest.param(lambda: _baseline_fingerprint(prompt_hash="q" * 64), id="prompt_hash"),
        pytest.param(
            lambda: _baseline_fingerprint(model_digests={"m": "sha256:bbb", "n": None}),
            id="model_digest_changed",
        ),
        pytest.param(
            lambda: _baseline_fingerprint(model_digests={"m": "sha256:aaa", "n": "sha256:ccc"}),
            id="model_digest_none_to_value",
        ),
        pytest.param(
            lambda: _baseline_fingerprint(engine_version="driftroom-engine-other"),
            id="engine_version",
        ),
    ],
)
def test_run_fingerprint_changes_when_any_single_input_changes(
    variant: Callable[[], str],
) -> None:
    assert variant() != _baseline_fingerprint()
