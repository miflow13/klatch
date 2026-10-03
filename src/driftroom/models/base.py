"""Shared result and decision types for local model inference."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from driftroom.domain import AgentConfig


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["speak", "wait"]
    message: str | None
    target: str | None

    @model_validator(mode="after")
    def action_matches_message(self) -> "Decision":
        if self.action == "speak" and not (self.message and self.message.strip()):
            raise ValueError("speak requires a nonblank message")
        if self.action == "wait" and self.message is not None:
            raise ValueError("wait requires a null message")
        return self


@dataclass(frozen=True)
class ModelResult:
    decision: Decision
    latency_ms: float
    prompt_tokens: int | None
    output_tokens: int | None
    model: str
    model_digest: str | None = None


@dataclass(frozen=True)
class ModelInfo:
    """A locally installed model and its content digest, when Ollama lists it."""

    name: str
    digest: str | None


class BackendError(Exception):
    """The local model backend did not yield a usable response."""


class DecisionValidationError(BackendError):
    """The model returned content that violates the decision schema."""


class EmptyModelContentError(BackendError):
    """The model response contained no decision content."""


class BackendTimeoutError(BackendError):
    """The model call exceeded its configured timeout."""


class ModelBackend(ABC):
    @abstractmethod
    def decide(
        self, agent: AgentConfig, messages: Sequence[dict[str, str]]
    ) -> ModelResult:
        """Make one constrained decision, or raise an infrastructure error."""
