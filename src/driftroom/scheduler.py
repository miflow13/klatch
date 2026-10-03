"""Cheap urge scoring before any model inference is considered."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import re
from random import Random
import time

from .clock import VirtualClock
from .domain import VISIBLE_EVENT_TYPES, AgentConfig, SchedulerConfig
from .storage import StoredEvent


@dataclass(frozen=True)
class CandidateScore:
    agent_id: str
    score: float
    reasons: tuple[str, ...]


class Scheduler:
    """Score visible history using configured weights and one injected RNG stream.

    Baseline talkativeness is ``1 - reserved`` and is scaled by energy and
    attention. Missing state is neutral (1 for both, 0 for affinity).
    """

    def __init__(self, config: SchedulerConfig, *, rng: Random | None = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else Random()

    def score_agents(
        self,
        agents: Sequence[AgentConfig],
        events: Sequence[StoredEvent],
        now_ms: int,
        *,
        energy: Mapping[str, float] | None = None,
        attention: Mapping[str, float] | None = None,
        relationship_affinity: Mapping[str, float] | None = None,
    ) -> list[CandidateScore]:
        visible = [event for event in events if event.type in VISIBLE_EVENT_TYPES]
        messages = [event for event in visible if event.type == "message"]
        # Only a message can name or echo anyone; environment text is never matched.
        latest_message = (
            visible[-1] if visible and visible[-1].type == "message" else None
        )
        latest_text = (
            str(latest_message.payload.get("message", ""))
            if latest_message is not None else ""
        )
        latest_tokens = set(re.findall(r"\b\w+\b", latest_text.casefold()))
        scores = []
        for agent in agents:
            score = self.config.base_bias
            reasons: list[str] = []
            talkativeness = (1 - agent.traits.reserved) * (
                energy.get(agent.id, 1.0) if energy is not None else 1.0
            ) * (attention.get(agent.id, 1.0) if attention is not None else 1.0)
            if talkativeness and self.config.talkativeness_weight:
                score += self.config.talkativeness_weight * talkativeness
                reasons.append("talkativeness")
            affinity = (
                relationship_affinity.get(agent.id, 0.0)
                if relationship_affinity is not None else 0.0
            )
            if affinity and self.config.relationship_weight:
                score += self.config.relationship_weight * affinity
                reasons.append("relationship")
            if self.config.direct_mention_bonus and re.search(
                rf"(?<!\w){re.escape(agent.name)}(?!\w)", latest_text, re.I
            ):
                score += self.config.direct_mention_bonus
                reasons.append("direct_mention")
            own_messages = [event for event in messages if event.agent_id == agent.id]
            last_own_ms = own_messages[-1].sim_ms if own_messages else 0
            elapsed_fraction = min(
                1.0, max(0, now_ms - last_own_ms) / self.config.silence_ambient_after_ms
            )
            if elapsed_fraction and self.config.elapsed_weight:
                score += self.config.elapsed_weight * elapsed_fraction
                reasons.append("elapsed")
            own_text = " ".join(
                str(event.payload.get("message", "")) for event in own_messages[-3:]
            )
            own_tokens = set(re.findall(r"\b\w+\b", own_text.casefold()))
            if latest_tokens and own_tokens:
                overlap = len(latest_tokens & own_tokens) / len(latest_tokens)
                if overlap and self.config.topic_overlap_weight:
                    score += self.config.topic_overlap_weight * overlap
                    reasons.append("topic_overlap")
            if (
                self.config.cooldown_penalty
                and own_messages
                and now_ms - own_messages[-1].sim_ms < self.config.speaker_cooldown_ms
            ):
                score -= self.config.cooldown_penalty
                reasons.append("cooldown")
            if (
                self.config.recent_speaker_penalty
                and latest_message is not None
                and latest_message.agent_id == agent.id
            ):
                score -= self.config.recent_speaker_penalty
                reasons.append("recent_speaker")
            if self.config.random_jitter:
                jitter = self.rng.uniform(
                    -self.config.random_jitter, self.config.random_jitter
                )
                if jitter:
                    score += jitter
                    reasons.append("jitter")
            scores.append(CandidateScore(agent.id, score, tuple(reasons)))
        return sorted(scores, key=lambda candidate: candidate.score, reverse=True)

    def select_candidate(
        self,
        agents: Sequence[AgentConfig],
        events: Sequence[StoredEvent],
        now_ms: int,
        *,
        energy: Mapping[str, float] | None = None,
        attention: Mapping[str, float] | None = None,
        relationship_affinity: Mapping[str, float] | None = None,
    ) -> CandidateScore | None:
        return self.select_from(
            self.score_agents(
                agents,
                events,
                now_ms,
                energy=energy,
                attention=attention,
                relationship_affinity=relationship_affinity,
            )
        )

    def select_from(self, scores: Sequence[CandidateScore]) -> CandidateScore | None:
        """Pick the top of an already scored list without drawing new jitter."""
        if scores and scores[0].score >= self.config.candidate_threshold:
            return scores[0]
        return None

    def wait_for_next_tick(
        self,
        clock: VirtualClock,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if clock.mode == "realtime":
            sleep_fn(self.config.decision_tick_ms / 1_000)
        else:
            clock.advance_ms(self.config.decision_tick_ms)
