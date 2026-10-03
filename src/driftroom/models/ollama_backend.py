"""Single-call, schema-constrained inference through local Ollama."""

from collections.abc import Callable, Mapping, Sequence
import time
from typing import Any, TypeVar

import httpx
import ollama
from pydantic import ValidationError

from driftroom.domain import AgentConfig, RuntimeConfig

from .base import (
    BackendError,
    BackendTimeoutError,
    Decision,
    DecisionValidationError,
    EmptyModelContentError,
    ModelBackend,
    ModelInfo,
    ModelResult,
    TruncatedGenerationError,
    decision_schema,
    raw_excerpt,
    validation_detail,
)


_T = TypeVar("_T")


def _field(value: object, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _call(operation: str, request: Callable[[], _T]) -> _T:
    """Run one client request, mapping transport failures to backend errors."""
    try:
        return request()
    except (TimeoutError, httpx.ReadTimeout) as exc:
        raise BackendTimeoutError(f"Ollama {operation} timed out") from exc
    except ollama.ResponseError as exc:
        raise BackendError(
            f"Ollama {operation} failed with HTTP {exc.status_code}: {exc.error}"
        ) from exc
    except Exception as exc:
        raise BackendError(f"Ollama {operation} failed: {exc}") from exc


class OllamaBackend(ModelBackend):
    def __init__(
        self,
        runtime: RuntimeConfig,
        *,
        client: object | None = None,
        host: str | None = None,
    ) -> None:
        self._runtime = runtime
        self._client = (
            client
            if client is not None
            else ollama.Client(host=host, timeout=runtime.inference_timeout_seconds)
        )

    def decide(
        self,
        agent: AgentConfig,
        messages: Sequence[dict[str, str]],
        *,
        targets: Sequence[str] = (),
    ) -> ModelResult:
        started = time.monotonic()
        response = _call(
            "chat request",
            lambda: self._client.chat(
                model=agent.model,
                messages=list(messages),
                # Thinking is never configurable in v0.1 (spec §16).
                think=False,
                stream=False,
                # Per agent: the grammar enforces the speak/wait contract (§11).
                format=decision_schema(targets),
                options={
                    "temperature": agent.sampling.temperature,
                    "top_p": agent.sampling.top_p,
                    "top_k": agent.sampling.top_k,
                    "repeat_penalty": agent.sampling.repeat_penalty,
                    "num_predict": self._runtime.max_output_tokens,
                    "num_ctx": self._runtime.max_context_tokens,
                },
            ),
        )

        content = _field(_field(response, "message"), "content")
        excerpt = raw_excerpt(content if isinstance(content, str) else ("" if content is None else repr(content)))

        # A length-cut envelope is not a trustworthy decision, even if it parses.
        if _field(response, "done_reason") == "length":
            raise TruncatedGenerationError(
                "Ollama hit the output-token limit (max_output_tokens="
                f"{self._runtime.max_output_tokens}) before completing the decision",
                raw_excerpt=excerpt, detail="done_reason=length",
            )

        if not isinstance(content, str) or not content.strip():
            raise EmptyModelContentError(
                "Ollama returned empty decision content", raw_excerpt=excerpt, detail="empty content"
            )
        try:
            decision = Decision.model_validate_json(content)
        except ValidationError as exc:
            raise DecisionValidationError(
                "Ollama decision violates schema", raw_excerpt=excerpt, detail=validation_detail(exc)
            ) from exc

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
            # Chat responses carry no digest; the engine records it via model_info().
            model_digest=None,
        )

    def model_info(self, model: str) -> ModelInfo:
        """Look up a locally installed model's digest by exact name."""
        listing = _call("model list request", self._client.list)
        for entry in _field(listing, "models") or ():
            if model in (_field(entry, "model"), _field(entry, "name")):
                return ModelInfo(name=model, digest=_field(entry, "digest"))
        return ModelInfo(name=model, digest=None)
