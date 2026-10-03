"""Terminal rendering of committed events (spec §29).

Pure functions over stored events: no I/O and no engine access. Callers print.
Model and environment text is neutralized for the terminal here, at render time
only; persisted events are never altered.
"""

import json
import re

from .domain import VISIBLE_EVENT_TYPES
from .storage import StoredEvent


INDENT = " " * 7  # aligns the message under the speaker, past "hh:mm  "

# Every C0 control except newline, plus DEL and the C1 range (ESC/CSI sequences included).
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")


def _escape_controls(text: str) -> str:
    """Show control characters as visible ``\\xNN`` escapes so they cannot drive the terminal."""
    return _CONTROL_CHARACTERS.sub(lambda match: f"\\x{ord(match.group()):02x}", text)


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
        speaker = _escape_controls(str(event.agent_id or event.payload.get("speaker") or "room").lower())
        lines = _escape_controls(str(event.payload["message"])).split("\n")
        return f"{stamp}  {speaker}\n" + "\n".join(INDENT + line for line in lines)
    if event.type == "environment":
        return f"{stamp}  ({_escape_controls(str(event.payload['text']))})"
    if event.type in VISIBLE_EVENT_TYPES:
        raise ValueError(f"visible event type has no renderer: {event.type}")
    if not debug:
        return None
    payload = json.dumps(event.payload, sort_keys=True, ensure_ascii=False)
    return _escape_controls(f"{stamp}  [{event.type}] {event.agent_id or '-'} {payload}")
