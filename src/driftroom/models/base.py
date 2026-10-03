"""Shared result and decision types for local model inference."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from driftroom.domain import AgentConfig
from driftroom.observer import _escape_controls


RAW_EXCERPT_CHARS = 300


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


def decision_schema(targets: Sequence[str]) -> dict[str, Any]:
    """The constrained-output schema for one agent's decision (spec §11, §16).

    Two closed branches let the grammar enforce what ``Decision`` validates
    after the fact: speak with a non-empty message and an optional participant
    target (``targets`` = the other agents' display names), or wait with nulls.
    ``Decision`` stays the backstop for backends without grammar support.
    """
    names = list(targets)
    target = {"anyOf": [{"enum": names}, {"type": "null"}]} if names else {"type": "null"}
    return {"type": "object", "anyOf": [
        {"properties": {"action": {"const": "speak"}, "message": {"type": "string", "minLength": 1},
                        "target": target},
         "required": ["action", "message", "target"], "additionalProperties": False},
        {"properties": {"action": {"const": "wait"}, "message": {"type": "null"}, "target": {"type": "null"}},
         "required": ["action", "message", "target"], "additionalProperties": False},
    ]}


def raw_excerpt(content: str) -> str:
    """The first characters of raw model content, safe to log and print.

    Bounded and control-escaped for the research log only; the content itself,
    and anything persisted or prompted from it, is never altered.
    """
    return _escape_controls(content[:RAW_EXCERPT_CHARS])


def validation_detail(exc: ValidationError) -> str:
    """The first validation problem as one line, e.g. ``wait requires a null message``.

    A location can echo a model-chosen key, so the line is bounded and escaped too.
    """
    error = exc.errors()[0]
    message = str(error["msg"]).removeprefix("Value error, ")
    location = ".".join(str(part) for part in error["loc"])
    line = f"{location}: {message}" if location else message
    return raw_excerpt(line.split("\n", 1)[0])


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
    """The local model backend did not yield a usable response.

    ``raw_excerpt`` (bounded, control-escaped raw model content) and ``detail``
    (the one-line reason) are set when the model produced content that was
    rejected, and are None for infrastructure failures such as timeouts.
    """

    def __init__(self, *args: object, raw_excerpt: str | None = None, detail: str | None = None) -> None:
        super().__init__(*args)
        self.raw_excerpt = raw_excerpt
        self.detail = detail


class DecisionValidationError(BackendError):
    """The model returned content that violates the decision schema."""


class TruncatedGenerationError(DecisionValidationError):
    """Generation hit the configured output-token limit (``num_predict``)
    before completing the decision envelope."""


class EmptyModelContentError(BackendError):
    """The model response contained no decision content."""


class BackendTimeoutError(BackendError):
    """The model call exceeded its configured timeout."""


class ModelBackend(ABC):
    @abstractmethod
    def decide(
        self,
        agent: AgentConfig,
        messages: Sequence[dict[str, str]],
        *,
        targets: Sequence[str] = (),
    ) -> ModelResult:
        """Make one constrained decision, or raise an infrastructure error.

        ``targets`` are the display names ``agent`` may address (the other
        participants); a backend with constrained output builds its schema from them.
        """
