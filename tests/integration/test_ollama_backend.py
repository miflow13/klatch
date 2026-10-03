"""Opt-in smoke test against an already installed local qwen3:4b model."""

import httpx
import ollama
import pytest

from driftroom.domain import AgentConfig, RuntimeConfig
from driftroom.models.ollama_backend import OllamaBackend


@pytest.mark.ollama
def test_local_qwen3_returns_a_constrained_decision() -> None:
    try:
        installed = ollama.Client(timeout=2).list()
    except (httpx.HTTPError, ollama.ResponseError, ConnectionError, ValueError) as exc:
        pytest.skip(f"local Ollama unavailable: {exc}")

    names = {model.model for model in installed.models}
    if "qwen3:4b" not in names:
        pytest.skip("local qwen3:4b is not installed")

    agent = AgentConfig(id="june", name="June", model="qwen3:4b")
    messages = [
        {
            "role": "system",
            "content": (
                "You are June in a text room with Ada. Return one JSON decision: "
                "speak with a nonblank message, or wait with null message and target."
            ),
        },
        {"role": "user", "content": "Room history: Ada said hello."},
    ]
    backend = OllamaBackend(RuntimeConfig())
    result = backend.decide(agent, messages, targets=["Ada"])
    decision = result.decision
    if decision.action == "wait":
        assert (decision.message, decision.target) == (None, None)
    else:
        assert decision.action == "speak"
        assert decision.message is not None and decision.message.strip()
        assert decision.target in {None, "Ada"}
    assert result.model == "qwen3:4b"

    digest = backend.model_info("qwen3:4b").digest
    assert isinstance(digest, str) and digest
