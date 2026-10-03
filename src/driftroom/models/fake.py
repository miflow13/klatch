"""Deterministic model outcomes for simulation tests."""

from collections import deque
from collections.abc import Iterable, Sequence

from driftroom.domain import AgentConfig

from .base import BackendError, Decision, ModelBackend, ModelResult


class FakeModelBackend(ModelBackend):
    def __init__(self, outcomes: Iterable[Decision | Exception]) -> None:
        self._outcomes = deque(outcomes)
        # The targets of every call, for tests; the fake's outcomes ignore them.
        self.targets: list[tuple[str, ...]] = []

    def decide(
        self,
        agent: AgentConfig,
        messages: Sequence[dict[str, str]],
        *,
        targets: Sequence[str] = (),
    ) -> ModelResult:
        self.targets.append(tuple(targets))
        if not self._outcomes:
            raise BackendError("fake model outcomes exhausted")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return ModelResult(
            decision=outcome,
            latency_ms=0.0,
            prompt_tokens=None,
            output_tokens=None,
            model=agent.model,
        )
