"""Observer-only behavioral metrics computed from a run's persisted events (spec §31, §32).

Analysis reads events and computes numbers. It never writes events, memories,
relationship state, scheduler values or control state, and nothing it returns
may feed prompts, memory retrieval, scheduling, relationships or ambient events.
Transcript text is scanned, never modified.
"""

from collections import Counter
from dataclasses import dataclass
import re
from statistics import fmean, median

from .domain import VISIBLE_EVENT_TYPES
from .storage import EventStore, StoredEvent


PAGE_SIZE = 1_000

# Spec §10: repeated assistant discourse that signals degrading behavior.
ASSISTANT_PHRASES: tuple[str, ...] = (
    "That's a great point",
    "I completely agree",
    "Building on what you said",
    "It's important to note",
    "In conclusion",
    "As an AI",
)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    # Whole phrase, case-insensitive: "as an aide" is not "as an AI". A straight
    # apostrophe in the phrase also matches a curly one; the scanned text is never changed.
    body = "['’]".join(re.escape(part) for part in phrase.split("'"))
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


_ASSISTANT_PATTERNS = tuple(_phrase_pattern(phrase) for phrase in ASSISTANT_PHRASES)
_AS_AN_AI = _phrase_pattern("As an AI")


@dataclass(frozen=True)
class RunMetrics:
    visible_messages: int
    valid_waits: int
    generation_failures: int
    attempt_failures: int
    messages_per_agent: dict[str, int]
    mean_message_words: float
    median_message_words: float
    assistant_phrase_hits: int
    as_an_ai_hits: int
    silence_periods: int
    ambient_events: int
    mean_inference_ms: float | None
    failure_classes: dict[str, int]


def _read_all(store: EventStore, run_id: str) -> list[StoredEvent]:
    events: list[StoredEvent] = []
    while page := store.read_events(run_id, events[-1].id if events else 0, PAGE_SIZE):
        events.extend(page)
    return events


def _silence_periods(events: list[StoredEvent], threshold_ms: int) -> int:
    """Gaps of at least ``threshold_ms`` between consecutive visible events,
    plus the trailing gap from the last visible event to the last event of the run."""
    times = [event.sim_ms for event in events if event.type in VISIBLE_EVENT_TYPES]
    if not times:
        return 0
    times.append(events[-1].sim_ms)
    return sum(1 for earlier, later in zip(times, times[1:]) if later - earlier >= threshold_ms)


def analyze_run(store: EventStore, run_id: str) -> RunMetrics:
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(f"unknown run: {run_id}")
    events = _read_all(store, run_id)
    by_type: dict[str, list[StoredEvent]] = {}
    for event in events:
        by_type.setdefault(event.type, []).append(event)
    messages = by_type.get("message", [])
    texts = [str(event.payload["message"]) for event in messages]
    words = [len(text.split()) for text in texts]
    per_agent = {agent.id: 0 for agent in run.config.agents}
    for event in messages:
        if event.agent_id is None:
            raise ValueError(f"message event {event.id} has no agent")
        per_agent[event.agent_id] = per_agent.get(event.agent_id, 0) + 1
    latencies = [
        float(event.payload["latency_ms"])
        for event in events if event.type in ("message", "agent_wait")
    ]
    attempt_failures = by_type.get("attempt_failed", [])
    return RunMetrics(
        visible_messages=len(messages),
        valid_waits=len(by_type.get("agent_wait", [])),
        generation_failures=len(by_type.get("generation_failed", [])),
        attempt_failures=len(attempt_failures),
        messages_per_agent=per_agent,
        mean_message_words=fmean(words) if words else 0.0,
        median_message_words=float(median(words)) if words else 0.0,
        assistant_phrase_hits=sum(len(p.findall(text)) for p in _ASSISTANT_PATTERNS for text in texts),
        as_an_ai_hits=sum(len(_AS_AN_AI.findall(text)) for text in texts),
        silence_periods=_silence_periods(events, run.config.scheduler.silence_ambient_after_ms),
        ambient_events=len(by_type.get("environment", [])),
        mean_inference_ms=fmean(latencies) if latencies else None,
        failure_classes=dict(Counter(str(event.payload["error_class"]) for event in attempt_failures)),
    )
