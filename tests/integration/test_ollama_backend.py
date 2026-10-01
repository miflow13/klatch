"""Opt-in smoke test against an already installed local qwen3:4b model."""

import json
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from driftroom.domain import AgentConfig
from driftroom.models.ollama_backend import OllamaBackend


@pytest.mark.ollama
def test_local_qwen3_returns_a_constrained_decision() -> None:
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            installed = json.load(response)
    except (OSError, URLError, TimeoutError, ValueError) as exc:
        pytest.skip(f"local Ollama unavailable: {exc}")

    names = {
        model.get("name") or model.get("model")
        for model in installed.get("models", [])
        if isinstance(model, dict)
    }
    if "qwen3:4b" not in names:
        pytest.skip("local qwen3:4b is not installed")

    agent = AgentConfig(id="june", name="June", model="qwen3:4b")
    messages = [
        {
            "role": "system",
            "content": (
                "You are June in a text room. Return one JSON decision: "
                "speak with a nonblank message, or wait with null message."
            ),
        },
        {"role": "user", "content": "Room history: Kai said hello."},
    ]
    result = OllamaBackend(timeout_seconds=120).decide(agent, messages)
    assert result.decision.action in {"speak", "wait"}
    assert result.model == "qwen3:4b"
