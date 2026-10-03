"""Versioned, plain-language prompt rendering for room participants."""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path

from .domain import VISIBLE_EVENT_TYPES, AgentConfig
from .storage import StoredEvent


TRAIT_RENDERER_VERSION = "traits-v1"
# v1 was frozen at the first recorded run. v2 renders an empty room as
# "(no messages yet)" instead of a blank ROOM HISTORY section. v2 frozen from the
# first accepted gate run: any further change to the transcript format must bump
# this version. v3 stamps lines from a recorded clock_display_start instead of
# always from 00:00.
TURN_FORMAT_VERSION = "room-transcript-v3"
EMPTY_ROOM_HISTORY = "(no messages yet)"
_PROMPTS = Path(__file__).resolve().parent / "prompts"

_TRAIT_PHRASES = {
    "reserved": (
        "You tend to speak readily when you have something to add.",
        "You are somewhat reserved and choose your moments to speak.",
        "You tend to be fairly reserved and usually speak when something genuinely interests you.",
    ),
    "curiosity": (
        "You are not especially curious.",
        "You are somewhat curious.",
        "You are very curious.",
    ),
    "humor": (
        "You rarely use humor.",
        "Your humor is occasional and fairly dry.",
        "You often notice chances for humor.",
    ),
    "impulsiveness": (
        "You are not especially impulsive.",
        "You sometimes respond on impulse.",
        "You often respond on impulse.",
    ),
    "formality": (
        "You usually write casually rather than formally.",
        "Your writing is neither especially casual nor formal.",
        "You usually write more formally.",
    ),
}


def _band(value: float) -> int:
    if value < 0.34:
        return 0
    if value < 0.67:
        return 1
    return 2


def render_traits(agent: AgentConfig) -> str:
    """Turn bounded numeric traits into stable language for the model."""
    return "\n".join(
        phrases[_band(getattr(agent.traits, trait))]
        for trait, phrases in _TRAIT_PHRASES.items()
    )


def clock_display_offset_ms(start: str) -> int:
    """Milliseconds from midnight for a validated ``HH:MM`` clock_display_start."""
    hours, minutes = start.split(":")
    return (int(hours) * 60 + int(minutes)) * 60_000


def render_room_history(
    events: Sequence[StoredEvent], now_sim_ms: int, *, display_offset_ms: int = 0,
) -> str:
    """Show committed speech and neutral environment events as a room transcript.

    Environment events are unattributed descriptions rendered in parentheses;
    coarse silence cues measure gaps between any visible events. Continuation
    lines of a text are indented under the stamp, so text can never start a
    transcript line and pass as another speaker; the stored text is unchanged.
    A room with no visible events renders as an explicit "(no messages yet)",
    never as an empty section. Stamps read as a 24-hour clock that starts
    ``display_offset_ms`` after midnight and wraps; silence gaps are differences,
    so the offset does not touch them.
    """
    lines: list[str] = []
    last_visible_ms: int | None = None
    for event in events:
        if event.type not in VISIBLE_EVENT_TYPES:
            continue
        if last_visible_ms is not None and event.sim_ms - last_visible_ms >= 300_000:
            minutes = round((event.sim_ms - last_visible_ms) / 60_000)
            lines.append(f"[about {minutes} minutes later]")
        minutes = (event.sim_ms + display_offset_ms) // 60_000 % 1440
        stamp = f"[{minutes // 60:02d}:{minutes % 60:02d}]"
        continuation = "\n" + " " * len(stamp + " ")
        if event.type == "environment":
            text = continuation.join(str(event.payload["text"]).splitlines())
            lines.append(f"{stamp} ({text})")
        else:
            speaker = str(event.payload.get("speaker") or event.agent_id or "Room")
            message = continuation.join(str(event.payload["message"]).splitlines())
            lines.append(f"{stamp} {speaker}: {message}")
        last_visible_ms = event.sim_ms
    if last_visible_ms is not None and now_sim_ms - last_visible_ms >= 300_000:
        minutes = round((now_sim_ms - last_visible_ms) / 60_000)
        lines.append(f"[about {minutes} minutes later]")
    return "\n".join(lines) if lines else EMPTY_ROOM_HISTORY


@dataclass(frozen=True)
class TurnContext:
    agent: AgentConfig
    events: Sequence[StoredEvent]
    now_sim_ms: int
    current_state: str = ""
    relationships: str = ""
    memories: str = ""
    display_offset_ms: int = 0


def build_turn_messages(context: TurnContext) -> list[dict[str, str]]:
    """Build one instruction and one observed-state message for a turn."""
    system_template = (_PROMPTS / "agent_system_v1.txt").read_text(encoding="utf-8")
    turn_template = (_PROMPTS / "turn_context_v1.txt").read_text(encoding="utf-8")
    return [
        {"role": "system", "content": system_template.format(agent_name=context.agent.name)},
        {
            "role": "user",
            "content": turn_template.format(
                personality=render_traits(context.agent),
                current_state=context.current_state,
                relationships=context.relationships,
                memories=context.memories,
                room_history=render_room_history(
                    context.events, context.now_sim_ms, display_offset_ms=context.display_offset_ms,
                ),
            ),
        },
    ]


def prompt_template_hash() -> str:
    """Fingerprint prompt bytes and the two rendering contracts."""
    digest = hashlib.sha256()
    parts = (
        (_PROMPTS / "agent_system_v1.txt").read_bytes(),
        (_PROMPTS / "turn_context_v1.txt").read_bytes(),
        TRAIT_RENDERER_VERSION.encode("utf-8"),
        TURN_FORMAT_VERSION.encode("utf-8"),
    )
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()
