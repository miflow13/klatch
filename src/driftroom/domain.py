"""Validated domain configuration models for a Driftroom run."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, allow_inf_nan=False
    )


class SamplingConfig(StrictModel):
    temperature: float = Field(default=0.8, ge=0)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=40, ge=0)
    repeat_penalty: float = Field(default=1.08, gt=0)


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
    candidate_threshold: float = 0.35
    decision_tick_ms: int = 5_000
    silence_ambient_after_ms: int = 300_000
    ambient_min_interval_ms: int = 900_000
    speaker_cooldown_ms: int = 20_000


class RuntimeConfig(StrictModel):
    startup_mode: Literal["blank", "topic", "environment", "custom"] = "blank"
    runtime_mode: Literal["eco", "balanced", "fast"] = "balanced"
    model_thinking: bool = False
    recent_context_events: int = Field(default=20, ge=0)
    max_output_tokens: int = Field(default=256, gt=0)
    inference_timeout_seconds: float = Field(default=120, gt=0)
    retry_count: int = Field(default=1, ge=0)


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
