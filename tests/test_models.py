"""Contracts for constrained decisions and model backend outcomes."""

import json

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
    ENVELOPE_SPEAK,
    ENVELOPE_WAIT,
    TruncatedGenerationError,
    decision_schema,
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


def response(content: str, done_reason: str | None = None) -> dict[str, object]:
    # "digest" is not part of Ollama's chat response; it is planted here to
    # prove decide() does not invent a model digest from the chat payload.
    payload: dict[str, object] = {
        "model": "qwen3:4b",
        "digest": "sha256:planted",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "total_duration": 25_000_000,
        "prompt_eval_count": 12,
        "eval_count": 4,
    }
    if done_reason is not None:
        payload["done_reason"] = done_reason
    return payload


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


# Raw envelopes real qwen3:4b returned under the old nullable-only schema (2026-10-03).
REAL_SAMPLES = [
    '{"action": "wait", "message": "wait", "target": "me"}',
    '{"action": "wait", "message": "i\'m waiting for someone to talk first. maybe we can start with a joke? \U0001f602", '
    '"target": "nobody"}',
    '{"action": "wait", "message": "i\'m here waiting for someone to talk to me though i\'ll be the first one to say '
    'i don\'t know if i\'m even real lol", "target": "you"}',
]


def conforms(schema: dict[str, object], value: object) -> bool:
    """A tiny structural checker for exactly the keywords decision_schema uses (no jsonschema dependency)."""
    if "anyOf" in schema and not any(conforms(branch, value) for branch in schema["anyOf"]):
        return False
    kind = schema.get("type")
    if kind == "object" and not isinstance(value, dict):
        return False
    if kind == "string" and not isinstance(value, str):
        return False
    if kind == "null" and value is not None:
        return False
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "minLength" in schema and len(value) < schema["minLength"]:
        return False
    properties = schema.get("properties", {})
    if properties:
        if any(key not in value for key in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return False
        if not all(conforms(sub, value[key]) for key, sub in properties.items() if key in value):
            return False
    return True


def test_envelope_literals_are_say_and_quiet() -> None:
    # "wait" collided with Qwen3's reasoning interjection (logprob evidence, 2026-10-03).
    assert (ENVELOPE_SPEAK, ENVELOPE_WAIT) == ("say", "quiet")


def test_decision_schema_is_a_two_branch_speak_wait_contract() -> None:
    assert decision_schema(["June", "Ada"]) == {
        "type": "object",
        "anyOf": [
            {
                "properties": {
                    "action": {"const": "say"},
                    "message": {"type": "string", "minLength": 1},
                    "target": {"anyOf": [{"enum": ["June", "Ada"]}, {"type": "null"}]},
                },
                "required": ["action", "message", "target"],
                "additionalProperties": False,
            },
            {
                "properties": {
                    "action": {"const": "quiet"},
                    "message": {"type": "null"},
                    "target": {"type": "null"},
                },
                "required": ["action", "message", "target"],
                "additionalProperties": False,
            },
        ],
    }


def test_decision_schema_without_participants_only_allows_a_null_target() -> None:
    schema = decision_schema([])
    assert schema["anyOf"][0]["properties"]["target"] == {"type": "null"}
    assert schema["anyOf"][1] == decision_schema(["June"])["anyOf"][1]
    assert decision_schema(()) == schema


def test_decision_schema_accepts_the_contract_and_nothing_else() -> None:
    schema = decision_schema(["June", "Ada"])
    assert conforms(schema, {"action": "say", "message": "hi", "target": "Ada"})
    assert conforms(schema, {"action": "say", "message": "hi", "target": None})
    assert conforms(schema, {"action": "quiet", "message": None, "target": None})
    assert not conforms(schema, {"action": "say", "message": "", "target": None})
    assert not conforms(schema, {"action": "say", "message": "hi", "target": "nobody"})
    assert not conforms(schema, {"action": "quiet", "message": None, "target": "Ada"})
    assert not conforms(schema, {"action": "quiet", "message": None})
    assert not conforms(schema, {"action": "quiet", "message": None, "target": None, "extra": 1})
    assert not conforms(decision_schema([]), {"action": "say", "message": "hi", "target": "Ada"})
    # The design's action names are no longer envelope literals.
    assert not conforms(schema, {"action": "speak", "message": "hi", "target": None})
    assert not conforms(schema, {"action": "wait", "message": None, "target": None})


@pytest.mark.parametrize("sample", REAL_SAMPLES)
def test_real_failing_samples_violate_both_the_model_and_the_schema(sample: str) -> None:
    with pytest.raises(ValidationError, match="wait requires a null message"):
        Decision.model_validate_json(sample)
    assert not conforms(decision_schema(["June", "Ada"]), json.loads(sample))


@pytest.mark.parametrize("sample", REAL_SAMPLES)
def test_validation_error_carries_raw_excerpt_and_detail(sample: str) -> None:
    client = RecordingClient(response(sample))
    with pytest.raises(DecisionValidationError) as caught:
        backend(client).decide(AGENT, MESSAGES, targets=["Milo", "Ada"])
    assert caught.value.raw_excerpt == sample
    # The grammar can no longer emit "wait"; the envelope check rejects it first.
    assert caught.value.detail == "unknown envelope action"


def test_validation_detail_names_the_field_for_field_errors() -> None:
    client = RecordingClient(response('{"action":"say","message":"hi","target":42}'))
    with pytest.raises(DecisionValidationError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert caught.value.detail == "target: Input should be a valid string"


def test_validation_detail_escapes_model_chosen_keys() -> None:
    client = RecordingClient(response('{"action":"quiet","message":null,"target":null,"\\u001b[2J":1}'))
    with pytest.raises(DecisionValidationError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert caught.value.detail == "\\x1b[2J: Extra inputs are not permitted"


def test_raw_excerpt_is_bounded_and_escaped_without_touching_the_content() -> None:
    content = "\x1b[2J" + "x" * 400 + "\x07"
    client = RecordingClient(response(content))
    with pytest.raises(DecisionValidationError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert caught.value.raw_excerpt == "\\x1b[2J" + "x" * 296
    assert caught.value.detail.startswith("Invalid JSON")
    assert "\n" not in caught.value.detail
    assert client.response["message"]["content"] == content


def test_truncated_and_empty_errors_carry_raw_excerpt_and_detail() -> None:
    client = RecordingClient(response('{"action":"say","message":"cut', done_reason="length"))
    with pytest.raises(TruncatedGenerationError) as truncated:
        backend(client).decide(AGENT, MESSAGES)
    assert truncated.value.raw_excerpt == '{"action":"say","message":"cut'
    assert truncated.value.detail == "done_reason=length"

    client = RecordingClient(response("  \n "))
    with pytest.raises(EmptyModelContentError) as empty:
        backend(client).decide(AGENT, MESSAGES)
    assert empty.value.raw_excerpt == "  \n "
    assert empty.value.detail == "empty content"


def test_infrastructure_errors_carry_no_raw_content() -> None:
    error = BackendTimeoutError("slow")
    assert (error.raw_excerpt, error.detail) == (None, None)


def test_ollama_sends_the_per_agent_schema_for_the_given_targets() -> None:
    client = RecordingClient(response('{"action":"quiet","message":null,"target":null}'))
    backend(client).decide(AGENT, MESSAGES, targets=["Milo", "Ada"])
    backend(client).decide(AGENT, MESSAGES)
    assert client.calls[0]["format"] == decision_schema(["Milo", "Ada"])
    assert client.calls[1]["format"] == decision_schema([])
    assert len(client.calls) == 2


def test_ollama_sends_exact_constrained_request_and_retains_metadata() -> None:
    client = RecordingClient(response('{"action":"say","message":"hello","target":"Kai"}'))
    result = backend(client).decide(AGENT, MESSAGES, targets=["Kai"])

    assert client.calls == [
        {
            "model": "qwen3:4b",
            "messages": MESSAGES,
            "think": False,
            "stream": False,
            "format": decision_schema(["Kai"]),
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
    assert result.decision == Decision(action="speak", message="hello", target="Kai")
    assert (result.model, result.model_digest) == ("qwen3:4b", None)
    assert (result.latency_ms, result.prompt_tokens, result.output_tokens) == (
        25.0,
        12,
        4,
    )


def test_valid_wait_remains_a_model_result() -> None:
    client = RecordingClient(response('{"action":"quiet","message":null,"target":null}'))
    result = backend(client).decide(AGENT, MESSAGES)
    assert result.decision == Decision(action="wait", message=None, target=None)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "content",
    [
        '{"action":"say","message":"  ","target":null}',
        '{"action":"quiet","message":"no","target":null}',
        '{"action":"say","message":"hi","target":42}',
        '{"action":"quiet"}',
        "not json",
    ],
)
def test_invalid_model_decision_raises_validation_error(content: str) -> None:
    client = RecordingClient(response(content))
    with pytest.raises(DecisionValidationError):
        backend(client).decide(AGENT, MESSAGES)
    assert len(client.calls) == 1


def test_say_envelope_maps_to_a_speak_decision() -> None:
    client = RecordingClient(response('{"action":"say","message":"hi","target":"Ada"}'))
    result = backend(client).decide(AGENT, MESSAGES, targets=["Ada"])
    assert result.decision == Decision(action="speak", message="hi", target="Ada")
    assert len(client.calls) == 1


def test_quiet_envelope_maps_to_a_wait_decision() -> None:
    client = RecordingClient(response('{"action":"quiet","message":null,"target":null}'))
    result = backend(client).decide(AGENT, MESSAGES)
    assert result.decision == Decision(action="wait", message=None, target=None)


@pytest.mark.parametrize(
    "content",
    [
        # Legacy literals: the grammar can no longer produce them.
        '{"action":"wait","message":null,"target":null}',
        '{"action":"speak","message":"hi","target":null}',
        '{"action":"SAY","message":"hi","target":null}',
        '{"action":42,"message":null,"target":null}',
        '{"action":["say"],"message":"hi","target":null}',
        '{"message":null,"target":null}',
    ],
)
def test_unknown_envelope_action_raises_validation_error(content: str) -> None:
    client = RecordingClient(response(content))
    with pytest.raises(DecisionValidationError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert caught.value.detail == "unknown envelope action"
    assert caught.value.raw_excerpt == content
    assert len(client.calls) == 1


def test_mapped_envelope_is_still_validated_by_decision() -> None:
    client = RecordingClient(response('{"action":"quiet","message":"psst","target":null}'))
    with pytest.raises(DecisionValidationError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert caught.value.detail == "wait requires a null message"
    assert caught.value.raw_excerpt == '{"action":"quiet","message":"psst","target":null}'


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


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("refused"), ConnectionError("refused")]
)
def test_connection_refused_raises_backend_error_not_timeout(
    error: Exception,
) -> None:
    # httpx.ConnectError is what raw httpx raises; the real ollama.Client
    # converts it to the builtin ConnectionError.
    client = RecordingClient(error=error)
    with pytest.raises(BackendError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert type(caught.value) is BackendError
    assert not isinstance(caught.value, BackendTimeoutError)
    assert caught.value.__cause__ is error
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectTimeout("no route"),
        httpx.PoolTimeout("pool exhausted"),
        httpx.WriteTimeout("stalled upload"),
    ],
)
def test_non_read_httpx_timeouts_are_network_errors(error: Exception) -> None:
    client = RecordingClient(error=error)
    with pytest.raises(BackendError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert type(caught.value) is BackendError
    assert not isinstance(caught.value, BackendTimeoutError)
    assert caught.value.__cause__ is error
    assert len(client.calls) == 1


def test_length_cutoff_with_partial_json_raises_truncated_error() -> None:
    client = RecordingClient(response('{"action":"say","mess', done_reason="length"))
    with pytest.raises(TruncatedGenerationError) as caught:
        backend(client).decide(AGENT, MESSAGES)
    assert isinstance(caught.value, DecisionValidationError)
    assert isinstance(caught.value, BackendError)
    assert "max_output_tokens" in str(caught.value)
    assert len(client.calls) == 1


def test_length_cutoff_with_complete_json_is_still_truncated() -> None:
    content = '{"action":"quiet","message":null,"target":null}'
    client = RecordingClient(response(content, done_reason="length"))
    with pytest.raises(TruncatedGenerationError):
        backend(client).decide(AGENT, MESSAGES)
    assert len(client.calls) == 1


def test_length_cutoff_with_empty_content_is_truncated_not_empty() -> None:
    client = RecordingClient(response("", done_reason="length"))
    with pytest.raises(TruncatedGenerationError):
        backend(client).decide(AGENT, MESSAGES)


def test_stop_reason_with_valid_json_succeeds() -> None:
    content = '{"action":"quiet","message":null,"target":null}'
    client = RecordingClient(response(content, done_reason="stop"))
    result = backend(client).decide(AGENT, MESSAGES)
    assert result.decision.action == "wait"


def test_default_client_uses_host_and_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, object]] = []

    def fake_client(**kwargs: object) -> RecordingClient:
        built.append(kwargs)
        return RecordingClient(response('{"action":"quiet","message":null,"target":null}'))

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
    fake = FakeModelBackend([decision, BackendTimeoutError("late")])
    result = fake.decide(AGENT, MESSAGES)
    assert result.decision is decision
    assert result.model == "qwen3:4b"
    with pytest.raises(BackendTimeoutError):
        fake.decide(AGENT, MESSAGES)
    with pytest.raises(BackendError):
        fake.decide(AGENT, MESSAGES)


def test_fake_backend_records_the_targets_it_was_given() -> None:
    fake = FakeModelBackend([Decision(action="wait", message=None, target=None)] * 2)
    fake.decide(AGENT, MESSAGES, targets=["Milo", "Ada"])
    fake.decide(AGENT, MESSAGES)
    assert fake.targets == [("Milo", "Ada"), ()]
