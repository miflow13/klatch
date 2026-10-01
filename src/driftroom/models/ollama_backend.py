"""Single-call, schema-constrained inference through local Ollama."""

from collections.abc import Mapping, Sequence
import json
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from driftroom.domain import AgentConfig

from .base import (
    BackendError,
    BackendTimeoutError,
    Decision,
    DecisionValidationError,
    EmptyModelContentError,
    ModelBackend,
    ModelResult,
)


_LOCAL_CHAT_URL = "http://127.0.0.1:11434/api/chat"


def _field(value: object, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


class _LocalOllamaClient:
    def __init__(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def chat(self, **kwargs: object) -> object:
        request = Request(
            _LOCAL_CHAT_URL,
            data=json.dumps(kwargs).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.load(response)


class OllamaBackend(ModelBackend):
    def __init__(self, client: object | None = None, *, timeout_seconds: float = 120) -> None:
        self._client = client if client is not None else _LocalOllamaClient(timeout_seconds)

    def decide(
        self, agent: AgentConfig, messages: Sequence[dict[str, str]]
    ) -> ModelResult:
        started = time.monotonic()
        try:
            response = self._client.chat(
                model=agent.model,
                messages=list(messages),
                think=False,
                stream=False,
                format=Decision.model_json_schema(),
                options={
                    "temperature": agent.sampling.temperature,
                    "top_p": agent.sampling.top_p,
                    "top_k": agent.sampling.top_k,
                    "repeat_penalty": agent.sampling.repeat_penalty,
                },
            )
        except TimeoutError as exc:
            raise BackendTimeoutError("Ollama inference timed out") from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise BackendTimeoutError("Ollama inference timed out") from exc
            raise BackendError("Ollama request failed") from exc
        except Exception as exc:
            raise BackendError("Ollama request failed") from exc

        content = _field(_field(response, "message"), "content")
        if not isinstance(content, str) or not content.strip():
            raise EmptyModelContentError("Ollama returned empty decision content")
        try:
            decision = Decision.model_validate_json(content)
        except ValidationError as exc:
            raise DecisionValidationError("Ollama decision violates schema") from exc

        total_duration = _field(response, "total_duration")
        latency_ms = (
            total_duration / 1_000_000
            if isinstance(total_duration, (int, float))
            else (time.monotonic() - started) * 1_000
        )
        return ModelResult(
            decision=decision,
            latency_ms=latency_ms,
            prompt_tokens=_field(response, "prompt_eval_count"),
            output_tokens=_field(response, "eval_count"),
            model=_field(response, "model") or agent.model,
            model_digest=_field(response, "digest") or _field(response, "model_digest"),
        )
