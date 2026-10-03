"""Single-threaded loop: control, ambient, scheduling, one inference, commit, notify."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from random import Random
import time
from typing import Literal

from .clock import VirtualClock
from .config import ENGINE_VERSION, run_config_hash, run_fingerprint
from .domain import VISIBLE_EVENT_TYPES, RunConfig
from .models.base import BackendError, ModelBackend
from .prompting import (
    TRAIT_RENDERER_VERSION, TURN_FORMAT_VERSION, TurnContext, build_turn_messages, prompt_template_hash,
)
from .scheduler import Scheduler
from .storage import EventRecord, EventStore, RunRecord, StoredEvent


# Neutral and descriptive, never directive: it opens an opportunity, not a turn (§14).
AMBIENT_SILENCE_TEXT = "the room has been quiet for a while"


@dataclass(frozen=True)
class EngineStepResult:
    kind: Literal["message", "wait", "generation_failed", "ambient", "idle", "paused", "stopped"]
    event_id: int | None


def _latest(*sim_ms: int | None) -> int | None:
    present = [value for value in sim_ms if value is not None]
    return max(present) if present else None


class SimulationEngine:
    def __init__(
        self,
        config: RunConfig,
        store: EventStore,
        backend: ModelBackend,
        *,
        run_id: str,
        room_id: str,
        clock: VirtualClock | None = None,
        scheduler: Scheduler | None = None,
        observer: Callable[[StoredEvent], None] | None = None,
        model_digests: Mapping[str, str | None] | None = None,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.store = store
        self.backend = backend
        self.run_id = run_id
        self.room_id = room_id
        self.clock = clock
        self.scheduler = scheduler
        self.observer = observer
        self.model_digests = (
            dict(model_digests) if model_digests is not None
            else {agent.model: None for agent in config.agents}
        )
        self.wall_clock = wall_clock
        self.sleep_fn = sleep_fn
        self._sim_start_ms: int | None = None
        self._ended = False
        self._last_visible_ms: int | None = None
        self._last_ambient_ms: int | None = None

    def start(self) -> int:
        runtime = self.config.runtime
        restart = self.store.get_run(self.run_id) is not None
        if not restart:
            self.store.create_run(
                RunRecord(self.run_id, self.room_id, self.wall_clock().isoformat(), self.config)
            )
        # Resume from the last persisted moment; downtime produces no events.
        last = self.store.read_recent_events(self.run_id, 1)
        if self.clock is None:
            self.clock = VirtualClock(
                runtime.clock_mode, self.wall_clock(), runtime.clock_speed,
                initial_ms=last[0].sim_ms if last else 0,
            )
        if self.scheduler is None:
            self.scheduler = Scheduler(self.config.scheduler, rng=Random(runtime.random_seed))
        prompt_hash = prompt_template_hash()
        self._sim_start_ms = self.clock.now_ms()
        return self._commit("session_started", None, self._sim_start_ms, {
            "run_fingerprint": run_fingerprint(
                self.config, prompt_hash=prompt_hash, model_digests=self.model_digests
            ),
            "config_hash": run_config_hash(self.config),
            "prompt_hash": prompt_hash,
            "trait_renderer_version": TRAIT_RENDERER_VERSION,
            "turn_format_version": TURN_FORMAT_VERSION,
            "engine_version": ENGINE_VERSION,
            "model_names": sorted({agent.model for agent in self.config.agents}),
            "model_digests": dict(self.model_digests),
            "random_seed": runtime.random_seed,
            "clock_mode": runtime.clock_mode,
            "clock_speed": runtime.clock_speed,
            "startup_mode": runtime.startup_mode,
            "runtime_mode": runtime.runtime_mode,
            "sim_start_ms": self._sim_start_ms,
            "restart": restart,
        })

    def step(self) -> EngineStepResult:
        if self._sim_start_ms is None:
            raise RuntimeError("start() must be called before step()")
        if self._ended:
            return EngineStepResult("stopped", None)
        tick_ms = self.config.scheduler.decision_tick_ms
        # Polled between inferences: a request made mid-generation lets that
        # generation commit, then takes effect here before another call (§24A).
        desired = self.store.get_control(self.room_id).desired_state
        if desired == "stop_requested":
            return EngineStepResult("stopped", self.stop())
        if desired == "paused":
            self.sleep_fn(tick_ms / 1_000)  # never advance_ms: a paused room accrues no simulated time
            return EngineStepResult("paused", None)
        recent = self.store.read_recent_events(self.run_id, self.config.runtime.recent_context_events)
        now = self.clock.now_ms()
        if self._ambient_due(recent, now):
            return EngineStepResult("ambient", self._commit("environment", None, now, {"text": AMBIENT_SILENCE_TEXT}))
        candidate = self.scheduler.select_from(
            self.scheduler.score_agents(self.config.agents, recent, now)
        )
        if candidate is None:
            self.scheduler.wait_for_next_tick(self.clock, sleep_fn=self.sleep_fn)
            return EngineStepResult("idle", None)
        agent = next(agent for agent in self.config.agents if agent.id == candidate.agent_id)
        messages = build_turn_messages(TurnContext(agent, recent, now))
        selection = {"scheduler_score": candidate.score, "reasons": list(candidate.reasons)}
        attempts = 1 + self.config.runtime.retry_count
        for attempt in range(1, attempts + 1):
            try:
                result = self.backend.decide(agent, messages)
            except BackendError as exc:
                last_error = exc
                self._commit("attempt_failed", agent.id, now, {
                    "attempt": attempt, "error_class": type(exc).__name__, "error": str(exc), **selection,
                })
                continue
            metadata = {
                **selection, "latency_ms": result.latency_ms, "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens, "model": result.model, "attempt": attempt,
            }
            decision = result.decision
            if decision.action == "wait":
                return EngineStepResult("wait", self._commit("agent_wait", agent.id, now, metadata))
            return EngineStepResult("message", self._commit("message", agent.id, now, {
                "speaker": agent.name, "message": decision.message, "target": decision.target, **metadata,
            }))
        # Exhausted retries are infrastructure failures, never social silence (spec §11, §35).
        return EngineStepResult("generation_failed", self._commit("generation_failed", agent.id, now, {
            "attempts": attempts, "last_error_class": type(last_error).__name__,
        }))

    def _ambient_due(self, recent: Sequence[StoredEvent], now: int) -> bool:
        # The bounded window alone is not enough: in a quiet room agent_wait events
        # crowd visible and ambient events out of it. So the engine also remembers
        # what it committed this session. Events older than both the window and this
        # session (e.g. before a restart) are treated as absent (accepted for v0.1).
        scheduler = self.config.scheduler
        visible = [event.sim_ms for event in recent if event.type in VISIBLE_EVENT_TYPES]
        ambient = [event.sim_ms for event in recent if event.type == "environment"]
        last_visible_ms = _latest(self._sim_start_ms, *visible[-1:], self._last_visible_ms)
        last_ambient_ms = _latest(*ambient[-1:], self._last_ambient_ms)
        assert last_visible_ms is not None
        return now - last_visible_ms >= scheduler.silence_ambient_after_ms and (
            last_ambient_ms is None or now - last_ambient_ms >= scheduler.ambient_min_interval_ms
        )

    def run(self, max_steps: int | None = None) -> None:
        steps = 0
        try:
            while max_steps is None or steps < max_steps:
                if self.step().kind == "stopped":
                    return
                steps += 1
        except KeyboardInterrupt:
            self.stop()
            raise

    def stop(self) -> int | None:
        if self._sim_start_ms is None:
            raise RuntimeError("start() must be called before stop()")
        if self._ended:
            return None
        now = self.clock.now_ms()
        event_id = self._commit("session_ended", None, now, {"sim_end_ms": now})
        self._ended = True
        return event_id

    def _commit(
        self, event_type: str, agent_id: str | None, sim_ms: int, payload: dict[str, object]
    ) -> int:
        """Persist first; observers only ever see committed events (spec §37)."""
        record = EventRecord(self.run_id, self.wall_clock().isoformat(), sim_ms, event_type, agent_id, payload)
        event_id = self.store.commit_event(record)
        if event_type in VISIBLE_EVENT_TYPES:
            self._last_visible_ms = sim_ms
        if event_type == "environment":
            self._last_ambient_ms = sim_ms
        if self.observer is not None:
            self.observer(StoredEvent(
                event_id, record.run_id, record.wall_ts, record.sim_ms,
                record.type, record.agent_id, dict(payload),
            ))
        return event_id
