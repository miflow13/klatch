"""Contracts for constrained decisions and model backend outcomes."""

import httpx
import ollama
import pytest
from pydantic import ValidationError

from driftroom.domain import AgentConfig, RuntimeConfig, SamplingConfig
from driftroom.models import ollama_backend
from driftroom.models.base import (
    BackendError,
    BackendTimeoutError,
    Decision,
    DecisionValidationError,
    EmptyModelContentError,
    ModelInfo,
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
RUNTIME = RuntimeConfig(
    max_output_tokens=300, max_context_tokens=4096, inference_timeout_seconds=7.5
)


class RecordingClient:
    def __init__(
        self,
        response: object = None,
        error: Exception | None = None,
        list_response: object = None,
    ) -> None:
        self.response = response
        self.error = error
        self.list_response = list_response
        self.calls: list[dict[str, object]] = []
        self.list_calls = 0

    def chat(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    def list(self) -> object:
        self.list_calls += 1
        if self.error is not None:
            raise self.error
        return self.list_response


def backend(client: RecordingClient) -> OllamaBackend:
    return OllamaBackend(RUNTIME, client=client)


def response(content: str) -> dict[str, object]:
    # "digest" is not part of Ollama's chat response; it is planted here to
    # prove decide() does not invent a model digest from the chat payload.
    return {
        "model": "qwen3:4b",
        "digest": "sha256:planted",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "total_duration": 25_000_000,
        "prompt_eval_count": 12,
        "eval_count": 4,
    }


@pytest.mark.parametrize("message", [None, "", " \t\n"])
def test_speak_requires_nonblank_message(message: str | None) -> None:
    with pytest.raises(ValidationError):
        Decision(action="speak", message=message, target=None)


def test_speak_preserves_message_and_string_target() -> None:
    decision = Decision(action="speak", message="hi June", target="june")
    assert decision.message == "hi June"
    assert decision.target == "june"


def test_wait_requires_null_message() -> None:
    with pytest.raises(ValidationError):
        Decision(action="wait", message="quiet", target=None)


def test_wait_accepts_null_message_and_target() -> None:
    assert Decision(action="wait", message=None, target=None).target is None


def test_target_rejects_numeric_event_id() -> None:
    with pytest.raises(ValidationError):
        Decision(action="speak", message="hi", target=123)


def test_schema_requires_explicit_message_and_target_keys() -> None:
    assert Decision.model_json_schema()["required"] == ["action", "message", "target"]


@pytest.mark.parametrize("missing", ["message", "target"])
def test_decision_rejects_omitted_nullable_key(missing: str) -> None:
    payload = {"action": "wait", "message": None, "target": None}
    del payload[missing]
    with pytest.raises(ValidationError):
        Decision.model_validate(payload)


def test_unknown_action_fails_validation() -> None:
    with pytest.raises(ValidationError):
        Decision(action="leave", message=None, target=None)


def test_ollama_sends_exact_constrained_request_and_retains_metadata() -> None:
    client = RecordingClient(response('{"action":"speak","message":"hello","target":"kai"}'))
    result = backend(client).decide(AGENT, MESSAGES)

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
                "num_predict": 300,
                "num_ctx": 4096,
            },
        }
    ]
    assert client.calls[0]["think"] is False
    assert result.decision == Decision(action="speak", message="hello", target="kai")
    assert (result.model, result.model_digest) == ("qwen3:4b", None)
    assert (result.latency_ms, result.prompt_tokens, result.output_tokens) == (
        25.0,
        12,
        4,
    )


def test_valid_wait_remains_a_model_result() -> None:
    client = RecordingClient(response('{"action":"wait","message":null,"target":null}'))
    result = backend(client).decide(AGENT, MESSAGES)
    assert result.decision == Decision(action="wait", message=None, target=None)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "content",
    [
        '{"action":"speak","message":"  ","target":null}',
        '{"action":"wait","message":"no","target":null}',
        '{"action":"speak","message":"hi","target":42}',
        '{"action":"wait"}',
        "not json",
    ],
)
def test_invalid_model_decision_raises_validation_error(content: str) -> None:
    client = RecordingClient(response(content))
    with pytest.raises(DecisionValidationError):
        backend(client).decide(AGENT, MESSAGES)
    assert len(client.calls) == 1


def test_empty_content_raises_distinct_error() -> None:
    client = RecordingClient(response(""))
    with pytest.raises(EmptyModelContentError):
        backend(client).decide(AGENT, MESSAGES)


def test_backend_exception_preserves_cause() -> None:
    client = RecordingClient(error=RuntimeError("service unavailable"))
    with pytest.raises(BackendError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert type(caught.value) is BackendError
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert len(client.calls) == 1


def test_timeout_raises_distinct_error() -> None:
    client = RecordingClient(error=TimeoutError("timed out"))
    with pytest.raises(BackendTimeoutError):
        backend(client).decide(AGENT, MESSAGES)
    assert len(client.calls) == 1


def test_ollama_response_error_reports_status_and_server_text() -> None:
    error = ollama.ResponseError("model 'x' not found", 404)
    client = RecordingClient(error=error)
    with pytest.raises(BackendError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert type(caught.value) is BackendError
    assert "404" in str(caught.value)
    assert "not found" in str(caught.value)
    assert caught.value.__cause__ is error
    assert len(client.calls) == 1


def test_httpx_timeout_raises_distinct_error() -> None:
    error = httpx.ReadTimeout("slow")
    client = RecordingClient(error=error)
    with pytest.raises(BackendTimeoutError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert caught.value.__cause__ is error
    assert len(client.calls) == 1


def test_httpx_connect_error_raises_backend_error() -> None:
    error = httpx.ConnectError("refused")
    client = RecordingClient(error=error)
    with pytest.raises(BackendError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert type(caught.value) is BackendError
    assert caught.value.__cause__ is error
    assert len(client.calls) == 1


def test_default_client_uses_host_and_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, object]] = []

    def fake_client(**kwargs: object) -> RecordingClient:
        built.append(kwargs)
        return RecordingClient(response('{"action":"wait","message":null,"target":null}'))

    monkeypatch.setattr(ollama_backend.ollama, "Client", fake_client)
    OllamaBackend(RUNTIME, host="http://127.0.0.1:11999").decide(AGENT, MESSAGES)
    assert built == [{"host": "http://127.0.0.1:11999", "timeout": 7.5}]


def listing(*entries: dict[str, object]) -> ollama.ListResponse:
    return ollama.ListResponse(models=[ollama.ListResponse.Model(**e) for e in entries])


def test_model_info_returns_digest_of_listed_model() -> None:
    client = RecordingClient(
        list_response=listing(
            {"model": "llama3:8b", "digest": "sha256:other"},
            {"model": "qwen3:4b", "digest": "sha256:qwen"},
        )
    )
    assert backend(client).model_info("qwen3:4b") == ModelInfo("qwen3:4b", "sha256:qwen")
    assert client.list_calls == 1
    assert client.calls == []


def test_model_info_accepts_mapping_entries_keyed_by_name() -> None:
    client = RecordingClient(
        list_response={"models": [{"name": "qwen3:4b", "digest": "sha256:qwen"}]}
    )
    assert backend(client).model_info("qwen3:4b").digest == "sha256:qwen"


def test_model_info_returns_none_digest_for_unlisted_model() -> None:
    client = RecordingClient(
        list_response=listing({"model": "qwen3:4b-q8_0", "digest": "sha256:q8"})
    )
    assert backend(client).model_info("qwen3:4b") == ModelInfo("qwen3:4b", None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ReadTimeout("slow"), BackendTimeoutError),
        (ollama.ResponseError("boom", 500), BackendError),
        (httpx.ConnectError("refused"), BackendError),
    ],
)
def test_model_info_maps_transport_failures(
    error: Exception, expected: type[BackendError]
) -> None:
    client = RecordingClient(error=error)
    with pytest.raises(expected) as caught:
        backend(client).model_info("qwen3:4b")
    assert type(caught.value) is expected
    assert caught.value.__cause__ is error


def test_fake_backend_consumes_decisions_and_exceptions_in_order() -> None:
    decision = Decision(action="wait", message=None, target=None)
    backend = FakeModelBackend([decision, BackendTimeoutError("late")])
    result = backend.decide(AGENT, MESSAGES)
    assert result.decision is decision
    assert result.model == "qwen3:4b"
    with pytest.raises(BackendTimeoutError):
        backend.decide(AGENT, MESSAGES)
    with pytest.raises(BackendError):
        backend.decide(AGENT, MESSAGES)
