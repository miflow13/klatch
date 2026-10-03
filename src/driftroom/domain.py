"""Validated domain configuration models for a Driftroom run."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import (
    BaseModel, ConfigDict, Field, ModelWrapValidatorHandler, field_validator, model_validator,
)


# Event types that participants can see; "environment" events are neutral,
# unattributed room descriptions. An ambient silence event (§14) has payload
# {"text": ...}; the startup line of the environment and topic modes (§9) has
# {"text": ..., "kind": "startup"}. Both are visible; only payloads without a
# "kind" key are ambient events for ambient throttling and the ambient metric.
VISIBLE_EVENT_TYPES: frozenset[str] = frozenset({"message", "environment"})


def is_ambient_event(event_type: str, payload: Mapping[str, object]) -> bool:
    """True for an ambient silence event, false for a startup line or any other event."""
    return event_type == "environment" and "kind" not in payload


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, allow_inf_nan=False
    )


class SamplingConfig(StrictModel):
    temperature: float = Field(default=0.8, ge=0)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=40, ge=0)
    repeat_penalty: float = Field(default=1.08, gt=0)
    # Window the repeat penalty looks back over (Ollama: 0 disables, -1 = num_ctx).
    # Provisional default; large enough to cover the room history in the prompt.
    repeat_last_n: int = Field(default=1024, ge=-1)


class AgentTraits(StrictModel):
    reserved: float = Field(default=0.5, ge=0, le=1)
    curiosity: float = Field(default=0.5, ge=0, le=1)
    humor: float = Field(default=0.5, ge=0, le=1)
    impulsiveness: float = Field(default=0.5, ge=0, le=1)
    formality: float = Field(default=0.5, ge=0, le=1)


class AgentConfig(StrictModel):
    id: str
    name: str
    model: str
    traits: AgentTraits = Field(default_factory=AgentTraits)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)


class SchedulerConfig(StrictModel):
    base_bias: float = -0.35
    talkativeness_weight: float = 0.45
    direct_mention_bonus: float = 0.55
    topic_overlap_weight: float = 0.25
    elapsed_weight: float = 0.30
    relationship_weight: float = 0.10
    recent_speaker_penalty: float = 0.45
    cooldown_penalty: float = 1.00
    random_jitter: float = 0.15
    candidate_threshold: float = 0.20
    decision_tick_ms: int = Field(default=5_000, gt=0)
    silence_ambient_after_ms: int = Field(default=300_000, gt=0)
    ambient_min_interval_ms: int = Field(default=900_000, gt=0)
    speaker_cooldown_ms: int = Field(default=20_000, ge=0)
    wait_cooldown_ms: int = Field(default=90_000, ge=0)
    # Two roles: the Jaccard threshold for the repetition rules (a message at least this
    # alike to one of the preceding ``repetition_window`` messages) and the cap above which
    # the topic-overlap term contributes nothing. 0 disables topic overlap entirely.
    repetition_similarity_threshold: float = Field(default=0.6, ge=0, le=1)
    repetition_damping: float = Field(default=0.5, ge=0)
    # How many preceding messages (since the last environment event) the repetition rules
    # compare against; 1 reproduces the pairwise rule.
    repetition_window: int = Field(default=5, ge=1)


class RuntimeConfig(StrictModel):
    # "custom" stays in the Literal so choosing it gets a specific error (spec §9).
    startup_mode: Literal["blank", "topic", "environment", "custom"] = "blank"
    # Required by, and only allowed with, startup_mode="topic".
    topic: str | None = None
    runtime_mode: Literal["eco", "balanced", "fast"] = "balanced"
    model_thinking: bool = False
    recent_context_events: int = Field(default=20, ge=0)
    max_output_tokens: int = Field(default=256, gt=0)
    inference_timeout_seconds: float = Field(default=120, gt=0)
    retry_count: int = Field(default=1, ge=0)
    random_seed: int | None = None
    clock_mode: Literal["realtime", "accelerated"] = "realtime"
    clock_speed: float = Field(default=1.0, gt=0)
    max_context_tokens: int = Field(default=8192, gt=0)

    @field_validator("model_thinking")
    @classmethod
    def thinking_is_disabled_in_v0_1(cls, value: bool) -> bool:
        # A field validator runs before validate_assignment commits the value,
        # so a rejected assignment leaves the object unchanged.
        if value:
            raise ValueError("v0.1 requires model_thinking=false")
        return value

    @field_validator("startup_mode")
    @classmethod
    def custom_startup_mode_is_reserved(cls, value: str) -> str:
        if value == "custom":
            raise ValueError("custom startup mode is reserved and not available in v0.1")
        return value

    @model_validator(mode="wrap")
    @classmethod
    def topic_matches_startup_mode(
        cls, data: Any, handler: ModelWrapValidatorHandler["RuntimeConfig"]
    ) -> "RuntimeConfig":
        # On assignment ``data`` is the instance, already updated in place by the
        # handler; restore it when the combination is rejected, so a rejected
        # assignment leaves the object unchanged (as the field validators do).
        snapshot = (
            (dict(data.__dict__), set(data.__pydantic_fields_set__))
            if isinstance(data, RuntimeConfig) else None
        )
        result = handler(data)
        try:
            if result.startup_mode == "topic":
                if result.topic is None or not result.topic.strip():
                    raise ValueError("topic startup mode requires a nonblank topic")
            elif result.topic is not None:
                raise ValueError(
                    f"topic is only allowed with startup_mode='topic', not {result.startup_mode!r}"
                )
        except ValueError:
            if snapshot is not None:
                data.__dict__.clear()
                data.__dict__.update(snapshot[0])
                data.__pydantic_fields_set__.clear()
                data.__pydantic_fields_set__.update(snapshot[1])
            raise
        return result


class RunConfig(StrictModel):
    agents: list[AgentConfig] = Field(min_length=3)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def agent_ids_are_unique(self) -> "RunConfig":
        ids = [agent.id for agent in self.agents]
        if len(ids) != len(set(ids)):
            raise ValueError("agent IDs must be unique")
        return self
