"""Contracts for constrained decisions and model backend outcomes."""

import pytest
from pydantic import ValidationError

from driftroom.domain import AgentConfig, SamplingConfig
from driftroom.models.base import (
    BackendError,
    BackendTimeoutError,
    Decision,
    DecisionValidationError,
    EmptyModelContentError,
)
from driftroom.models.fake import FakeModelBackend
from driftroom.models.ollama_backend import OllamaBackend


AGENT = AgentConfig(
    id="june",
    name="June",
    model="qwen3:4b",
    sampling=SamplingConfig(
        temperature=0.67, top_p=0.88, top_k=27, repeat_penalty=1.12
    ),
)
MESSAGES = [{"role": "system", "content": "You are June."}]


class RecordingClient:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def response(content: str) -> dict[str, object]:
    return {
        "model": "qwen3:4b",
        "digest": "sha256:abc",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "total_duration": 25_000_000,
        "prompt_eval_count": 12,
        "eval_count": 4,
    }


@pytest.mark.parametrize("message", [None, "", " \t\n"])
def test_speak_requires_nonblank_message(message: str | None) -> None:
    with pytest.raises(ValidationError):
        Decision(action="speak", message=message)


def test_speak_preserves_message_and_string_target() -> None:
    decision = Decision(action="speak", message="hi June", target="june")
    assert decision.message == "hi June"
    assert decision.target == "june"


def test_wait_requires_null_message() -> None:
    with pytest.raises(ValidationError):
        Decision(action="wait", message="quiet")


def test_wait_accepts_null_message_and_target() -> None:
    assert Decision(action="wait", message=None, target=None).target is None


def test_target_rejects_numeric_event_id() -> None:
    with pytest.raises(ValidationError):
        Decision(action="speak", message="hi", target=123)


def test_unknown_action_fails_validation() -> None:
    with pytest.raises(ValidationError):
        Decision(action="leave", message=None)


def test_ollama_sends_exact_constrained_request_and_retains_metadata() -> None:
    client = RecordingClient(response('{"action":"speak","message":"hello","target":"kai"}'))
    result = OllamaBackend(client=client).decide(AGENT, MESSAGES)

    assert client.calls == [
        {
            "model": "qwen3:4b",
            "messages": MESSAGES,
            "think": False,
            "stream": False,
            "format": Decision.model_json_schema(),
            "options": {
                "temperature": 0.67,
                "top_p": 0.88,
                "top_k": 27,
                "repeat_penalty": 1.12,
            },
        }
    ]
    assert result.decision == Decision(action="speak", message="hello", target="kai")
    assert (result.model, result.model_digest) == ("qwen3:4b", "sha256:abc")
    assert (result.latency_ms, result.prompt_tokens, result.output_tokens) == (
        25.0,
        12,
        4,
    )


def test_valid_wait_remains_a_model_result() -> None:
    client = RecordingClient(response('{"action":"wait","message":null,"target":null}'))
    result = OllamaBackend(client=client).decide(AGENT, MESSAGES)
    assert result.decision == Decision(action="wait", message=None, target=None)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "content",
    [
        '{"action":"speak","message":"  ","target":null}',
        '{"action":"wait","message":"no","target":null}',
        '{"action":"speak","message":"hi","target":42}',
        "not json",
    ],
)
def test_invalid_model_decision_raises_validation_error(content: str) -> None:
    client = RecordingClient(response(content))
    with pytest.raises(DecisionValidationError):
        OllamaBackend(client=client).decide(AGENT, MESSAGES)
    assert len(client.calls) == 1


def test_empty_content_raises_distinct_error() -> None:
    client = RecordingClient(response(""))
    with pytest.raises(EmptyModelContentError):
        OllamaBackend(client=client).decide(AGENT, MESSAGES)


def test_backend_exception_preserves_cause() -> None:
    client = RecordingClient(error=RuntimeError("service unavailable"))
    with pytest.raises(BackendError) as caught:
        OllamaBackend(client=client).decide(AGENT, MESSAGES)
    assert type(caught.value) is BackendError
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert len(client.calls) == 1


def test_timeout_raises_distinct_error() -> None:
    client = RecordingClient(error=TimeoutError("timed out"))
    with pytest.raises(BackendTimeoutError):
        OllamaBackend(client=client).decide(AGENT, MESSAGES)
    assert len(client.calls) == 1


def test_fake_backend_consumes_decisions_and_exceptions_in_order() -> None:
    decision = Decision(action="wait", message=None)
    backend = FakeModelBackend([decision, BackendTimeoutError("late")])
    result = backend.decide(AGENT, MESSAGES)
    assert result.decision is decision
    assert result.model == "qwen3:4b"
    with pytest.raises(BackendTimeoutError):
        backend.decide(AGENT, MESSAGES)
    with pytest.raises(BackendError):
        backend.decide(AGENT, MESSAGES)
