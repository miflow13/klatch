"""Terminal rendering of committed events (spec §29).

Pure functions over stored events: no I/O and no engine access. Callers print.
"""

import json

from .domain import VISIBLE_EVENT_TYPES
from .storage import StoredEvent


INDENT = " " * 7  # aligns the message under the speaker, past "hh:mm  "


def format_sim_time(sim_ms: int, *, seconds: bool = False) -> str:
    """Render simulation time as hh:mm (as prompts show it), or hh:mm:ss."""
    total_seconds = sim_ms // 1_000
    hours, minutes = divmod(total_seconds // 60, 60)
    stamp = f"{hours:02d}:{minutes:02d}"
    return f"{stamp}:{total_seconds % 60:02d}" if seconds else stamp


def render_event(event: StoredEvent, *, debug: bool = False) -> str | None:
    """Render one event, or None for infrastructure events outside debug mode."""
    stamp = format_sim_time(event.sim_ms)
    if event.type == "message":
        speaker = str(event.agent_id or event.payload.get("speaker") or "room").lower()
        return f"{stamp}  {speaker}\n{INDENT}{event.payload['message']}"
    if event.type == "environment":
        return f"{stamp}  ({event.payload['text']})"
    if event.type in VISIBLE_EVENT_TYPES:
        raise ValueError(f"visible event type has no renderer: {event.type}")
    if not debug:
        return None
    payload = json.dumps(event.payload, sort_keys=True, ensure_ascii=False)
    return f"{stamp}  [{event.type}] {event.agent_id or '-'} {payload}"
