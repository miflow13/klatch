"""Terminal observer rendering and the operational CLI (spec §24A, §29, §30)."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest
from typer.testing import CliRunner

from driftroom.config import load_run_config
from driftroom.domain import RunConfig
from driftroom.storage import EventRecord, EventStore, RunRecord, StoredEvent


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "driftroom.example.toml"
COMMANDS = ("start", "watch", "pause", "resume", "stop", "status", "export")
# Every agent always clears the threshold, so each step asks exactly one agent.
FORCED = dict(
    base_bias=1.0, talkativeness_weight=0, direct_mention_bonus=0, topic_overlap_weight=0,
    elapsed_weight=0, relationship_weight=0, recent_speaker_penalty=0, cooldown_penalty=0,
    random_jitter=0,
)

runner = CliRunner()


def cli_app():
    from driftroom.cli import app
    return app


def config() -> RunConfig:
    return load_run_config(EXAMPLE_CONFIG)


def event(event_type: str, sim_ms: int, payload: dict[str, object], agent_id: str | None = None) -> StoredEvent:
    return StoredEvent(1, "run-1", "2026-10-01T12:00:00+00:00", sim_ms, event_type, agent_id, payload)


@contextmanager
def opened(db: Path) -> Iterator[EventStore]:
    """An EventStore that is always closed (Python 3.13+ warns on unclosed sqlite connections)."""
    store = EventStore(db)
    try:
        yield store
    finally:
        store.close()


@contextmanager
def seeded_store(db: Path, run_id: str = "run-1", room_id: str = "room-1") -> Iterator[EventStore]:
    with opened(db) as store:
        store.initialize()
        store.create_run(RunRecord(run_id, room_id, "2026-10-01T12:00:00+00:00", config()))
        yield store


def write_config(path: Path, *, seed: int | None = None, forced: bool = False) -> Path:
    """Copy the example config with an accelerated clock, an optional seed, and optionally FORCED scheduling."""
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    replacements = {"clock_mode": '"accelerated"'}
    if seed is not None:
        text = re.sub(r"^# random_seed = .*$", "random_seed = 0", text, flags=re.M)
        replacements["random_seed"] = str(seed)
    if forced:
        replacements.update({key: str(value) for key, value in FORCED.items()})
    for key, value in replacements.items():
        text, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", text, flags=re.M)
        assert count == 1, key
    path.write_text(text, encoding="utf-8")
    return path


def read_types(db: Path, run_id: str) -> list[str]:
    with opened(db) as store:
        return [stored.type for stored in store.read_events(run_id)]


def control_of(db: Path, room_id: str) -> str:
    with opened(db) as store:
        return store.get_control(room_id).desired_state


def append(store: EventStore, event_type: str, sim_ms: int, payload: dict[str, object],
           agent_id: str | None = None, run_id: str = "run-1") -> int:
    return store.commit_event(EventRecord(run_id, "2026-10-01T12:00:00+00:00", sim_ms, event_type, agent_id, payload))


def seed_history(store: EventStore) -> None:
    """A run with every event type the engine writes, in engine order."""
    append(store, "session_started", 0, {"model_names": ["qwen3:4b"]})
    append(store, "message", 33_240_000, {"speaker": "June", "message": "wait do either of you remember joining this room",
                                          "target": None}, "june")
    append(store, "agent_wait", 33_245_000, {"latency_ms": 1.0}, "milo")
    append(store, "attempt_failed", 33_250_000, {"attempt": 1, "error_class": "BackendTimeoutError",
                                                 "error": "timed out"}, "ada")
    append(store, "attempt_failed", 33_250_000, {"attempt": 2, "error_class": "BackendTimeoutError",
                                                 "error": "timed out"}, "ada")
    append(store, "generation_failed", 33_250_000, {"attempts": 2, "last_error_class": "BackendTimeoutError"}, "ada")
    append(store, "agent_wait", 33_255_000, {"latency_ms": 1.0}, "june")
    append(store, "environment", 33_300_000, {"text": "the room has been quiet for a while"})
    append(store, "message", 33_301_000, {"speaker": "Milo", "message": "not really?", "target": "june"}, "milo")
    append(store, "session_ended", 33_302_500, {"sim_end_ms": 33_302_500})


# 7. Observer renderer -------------------------------------------------------------------

def test_render_message_uses_lowercase_agent_id_and_indented_text() -> None:
    from driftroom.observer import render_event

    rendered = render_event(event("message", 33_240_000, {
        "speaker": "June", "message": "wait do either of you remember joining this room", "target": None,
    }, "June"))

    assert rendered == "09:14  june\n       wait do either of you remember joining this room"


def test_render_environment_is_one_parenthesized_line() -> None:
    from driftroom.observer import render_event

    rendered = render_event(event("environment", 33_300_000, {"text": "the room has been quiet for a while"}))

    assert rendered == "09:15  (the room has been quiet for a while)"


@pytest.mark.parametrize("event_type", [
    "session_started", "session_ended", "agent_wait", "attempt_failed", "generation_failed",
])
def test_infrastructure_events_are_hidden_by_default(event_type: str) -> None:
    from driftroom.observer import render_event

    assert render_event(event(event_type, 0, {"attempts": 2}, "ada")) is None


def test_debug_renders_infrastructure_events_with_type_agent_and_payload() -> None:
    from driftroom.observer import render_event

    rendered = render_event(
        event("attempt_failed", 33_250_000, {"attempt": 1, "error_class": "BackendTimeoutError"}, "ada"),
        debug=True,
    )

    assert rendered is not None
    assert rendered.startswith("09:14  [attempt_failed] ada")
    assert "BackendTimeoutError" in rendered


def test_debug_does_not_change_visible_rendering() -> None:
    from driftroom.observer import render_event

    message = event("message", 60_000, {"speaker": "Ada", "message": "hi", "target": None}, "ada")

    assert render_event(message, debug=True) == render_event(message) == "00:01  ada\n       hi"


# 1. Help ---------------------------------------------------------------------------------

def test_help_lists_the_seven_commands() -> None:
    result = runner.invoke(cli_app(), ["--help"])

    assert result.exit_code == 0, result.output
    for command in COMMANDS:
        assert command in result.output


# 2. Controls -----------------------------------------------------------------------------

@pytest.mark.parametrize(("command", "state"), [
    ("pause", "paused"), ("resume", "running"), ("stop", "stop_requested"),
])
def test_control_commands_update_room_controls(tmp_path: Path, command: str, state: str) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        store.set_control("room-1", "paused" if state == "running" else "running")

    result = runner.invoke(cli_app(), [command, "--db", str(db), "--room", "room-1"])

    assert result.exit_code == 0, result.output
    assert control_of(db, "room-1") == state


def test_control_command_rejects_unknown_room(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db):
        pass

    result = runner.invoke(cli_app(), ["pause", "--db", str(db), "--room", "room-typo"])

    assert result.exit_code != 0
    assert "room-typo" in result.output
    with pytest.raises(KeyError):
        control_of(db, "room-typo")


def test_commands_reject_a_missing_database(tmp_path: Path) -> None:
    db = tmp_path / "missing.db"

    result = runner.invoke(cli_app(), ["status", "--db", str(db), "--run", "run-1"])

    assert result.exit_code != 0
    assert not db.exists()


def test_control_commands_never_import_the_model_backend(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db):
        pass
    script = textwrap.dedent(f"""
        import sys
        from typer.testing import CliRunner
        from driftroom.cli import app
        for command in ("pause", "resume", "stop"):
            result = CliRunner().invoke(app, [command, "--db", {str(db)!r}, "--room", "room-1"])
            assert result.exit_code == 0, result.output
        print(sorted(name for name in sys.modules if name == "ollama" or name.startswith(("ollama.", "driftroom.models"))))
    """)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"
    assert control_of(db, "room-1") == "stop_requested"


# 3. Start on the fake backend -----------------------------------------------------------

def test_start_fake_runs_the_engine_end_to_end(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "run.toml", seed=3, forced=True)
    db = tmp_path / "room.db"

    result = runner.invoke(cli_app(), [
        "start", "--config", str(config_path), "--db", str(db), "--run", "run-fake", "--fake", "--max-steps", "6",
    ])

    assert result.exit_code == 0, result.output
    assert "run-fake" in result.output
    with opened(db) as store:
        events = store.read_events("run-fake")
        assert store.get_control("room-1").desired_state == "running"
    types = [stored.type for stored in events]
    assert types[0] == "session_started"
    assert types[-1] == "session_ended"
    assert "message" in types and "agent_wait" in types
    first_message = next(stored for stored in events if stored.type == "message")
    assert f"  {first_message.agent_id}\n       {first_message.payload['message']}" in result.output


def test_start_fake_defaults_run_id_and_room(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "run.toml")
    db = tmp_path / "room.db"

    result = runner.invoke(cli_app(), ["start", "--config", str(config_path), "--db", str(db), "--fake", "--max-steps", "1"])

    assert result.exit_code == 0, result.output
    run_id = re.search(r"^run id: (\S+)$", result.output, flags=re.M).group(1)
    assert run_id.startswith("run-")
    with opened(db) as store:
        record = store.get_run(run_id)
    assert record is not None and record.room_id == "room-1"


def test_start_stops_the_engine_on_keyboard_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from driftroom.engine import SimulationEngine

    def interrupted_step(self: SimulationEngine) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(SimulationEngine, "step", interrupted_step)
    config_path = write_config(tmp_path / "run.toml")
    db = tmp_path / "room.db"

    result = runner.invoke(cli_app(), ["start", "--config", str(config_path), "--db", str(db), "--run", "run-x", "--fake"])

    assert result.exit_code == 0, result.output
    assert read_types(db, "run-x") == ["session_started", "session_ended"]



# 4. Status -------------------------------------------------------------------------------

def test_status_reports_run_control_time_and_counts(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        seed_history(store)
        store.set_control("room-1", "paused")

    result = runner.invoke(cli_app(), ["status", "--db", str(db), "--run", "run-1"])

    assert result.exit_code == 0, result.output
    lines = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip() for line in result.output.splitlines()}
    assert lines["run"] == "run-1"
    assert lines["room"] == "room-1"
    assert lines["control"] == "paused"
    assert lines["sim time"] == "09:15:02"
    assert lines["events"] == "10"
    assert lines["messages"] == "2"
    assert lines["waits"] == "2"
    assert lines["attempt failures"] == "2"
    assert lines["generation failures"] == "1"
    assert lines["ambient events"] == "1"
    assert lines["models"] == "qwen3:4b"


def test_status_rejects_unknown_run(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db):
        pass

    result = runner.invoke(cli_app(), ["status", "--db", str(db), "--run", "run-nope"])

    assert result.exit_code != 0
    assert "run-nope" in result.output


# 5. Watch --------------------------------------------------------------------------------

def test_watch_once_prints_visible_history_and_hides_infrastructure(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        seed_history(store)

    result = runner.invoke(cli_app(), ["watch", "--db", str(db), "--run", "run-1", "--once"])

    assert result.exit_code == 0, result.output
    assert "09:14  june\n       wait do either of you remember joining this room" in result.output
    assert "09:15  (the room has been quiet for a while)" in result.output
    assert "09:15  milo\n       not really?" in result.output
    assert "attempt_failed" not in result.output
    assert "generation_failed" not in result.output
    assert "agent_wait" not in result.output


def test_watch_once_debug_also_prints_infrastructure(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        seed_history(store)

    result = runner.invoke(cli_app(), ["watch", "--db", str(db), "--run", "run-1", "--once", "--debug"])

    assert result.exit_code == 0, result.output
    assert "[attempt_failed] ada" in result.output
    assert "[generation_failed] ada" in result.output
    assert "09:15  milo\n       not really?" in result.output


def test_watch_once_reads_past_one_page_of_events(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        for index in range(501):
            append(store, "message", index * 1_000, {"speaker": "Ada", "message": f"line {index}", "target": None}, "ada")

    result = runner.invoke(cli_app(), ["watch", "--db", str(db), "--run", "run-1", "--once"])

    assert result.exit_code == 0, result.output
    assert "line 0\n" in result.output and "line 500\n" in result.output


# 6. Export -------------------------------------------------------------------------------

def test_export_jsonl_contains_every_event_including_failures(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        seed_history(store)
    output = tmp_path / "run.jsonl"

    result = runner.invoke(cli_app(), [
        "export", "--db", str(db), "--run", "run-1", "--format", "jsonl", "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == read_types(db, "run-1")
    assert set(rows[0]) == {"id", "run_id", "wall_ts", "sim_ms", "type", "agent_id", "payload"}
    failure = next(row for row in rows if row["type"] == "generation_failed")
    assert failure["agent_id"] == "ada"
    assert failure["payload"] == {"attempts": 2, "last_error_class": "BackendTimeoutError"}


def test_export_text_contains_only_visible_room_history(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        seed_history(store)
    output = tmp_path / "run.txt"

    result = runner.invoke(cli_app(), [
        "export", "--db", str(db), "--run", "run-1", "--format", "text", "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8") == (
        "09:14  june\n"
        "       wait do either of you remember joining this room\n"
        "\n"
        "09:15  (the room has been quiet for a while)\n"
        "\n"
        "09:15  milo\n"
        "       not really?\n"
    )


def test_export_rejects_unknown_format(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        seed_history(store)
    output = tmp_path / "run.out"

    result = runner.invoke(cli_app(), [
        "export", "--db", str(db), "--run", "run-1", "--format", "csv", "--output", str(output),
    ])

    assert result.exit_code != 0
    assert not output.exists()


def test_python_dash_m_driftroom_runs_the_cli() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    completed = subprocess.run([sys.executable, "-m", "driftroom", "--help"], capture_output=True, text=True,
                               env=env, check=False)

    assert completed.returncode == 0, completed.stderr
    assert "status" in completed.stdout


# Review fixes: observer text neutralization ---------------------------------------------

def test_render_message_indents_continuation_lines_and_escapes_control_characters() -> None:
    from driftroom.observer import render_event

    message = event("message", 60_000, {"speaker": "Ada", "message": "line one\nline two \x1b[2J", "target": None}, "ada")

    assert render_event(message) == "00:01  ada\n       line one\n       line two \\x1b[2J"


def test_render_escapes_every_c0_and_c1_control_character_except_newline() -> None:
    from driftroom.observer import render_event

    controls = "".join(chr(code) for code in (*range(0x00, 0x0a), *range(0x0b, 0x20), *range(0x7f, 0xa0)))
    rendered = render_event(event("message", 0, {"speaker": "Ada", "message": f"a{controls}b", "target": None}, "ada"))

    assert rendered is not None
    assert not any(ord(char) < 0x20 and char != "\n" or 0x7f <= ord(char) < 0xa0 for char in rendered)
    assert "\\x00" in rendered and "\\x0d" in rendered and "\\x7f" in rendered and "\\x9b" in rendered


def test_render_environment_escapes_control_characters() -> None:
    from driftroom.observer import render_event

    rendered = render_event(event("environment", 60_000, {"text": "lights \x1b]0;pwned\x07flicker"}))

    assert rendered == "00:01  (lights \\x1b]0;pwned\\x07flicker)"


def test_debug_rendering_escapes_c1_control_characters() -> None:
    from driftroom.observer import render_event

    rendered = render_event(event("attempt_failed", 0, {"error": "bad \x9b2J"}, "ada"), debug=True)

    assert rendered is not None and "\x9b" not in rendered and "\\x9b" in rendered


def test_rendering_never_alters_the_persisted_event() -> None:
    from driftroom.observer import render_event

    payload = {"speaker": "Ada", "message": "a\nb\x1b", "target": None}
    stored = event("message", 0, payload, "ada")
    render_event(stored)

    assert stored.payload["message"] == "a\nb\x1b"


# Review fixes: restart from the recorded room and regime -------------------------------

def start_args(db: Path, run_id: str, *extra: str) -> list[str]:
    return ["start", "--db", str(db), "--run", run_id, "--max-steps", "1", *extra]


def test_restart_uses_the_recorded_room(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    config_path = write_config(tmp_path / "run.toml")
    first = runner.invoke(cli_app(), start_args(db, "run-lab", "--config", str(config_path), "--room", "lab", "--fake"))
    assert first.exit_code == 0, first.output

    restarted = runner.invoke(cli_app(), start_args(db, "run-lab", "--config", str(config_path), "--fake"))

    assert restarted.exit_code == 0, restarted.output
    assert "room id: lab" in restarted.output
    assert read_types(db, "run-lab").count("session_started") == 2
    with pytest.raises(KeyError):
        control_of(db, "room-1")


def test_restart_without_config_uses_the_recorded_config(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    config_path = write_config(tmp_path / "run.toml", seed=3)
    first = runner.invoke(cli_app(), start_args(db, "run-a", "--config", str(config_path), "--room", "lab", "--fake"))
    assert first.exit_code == 0, first.output

    restarted = runner.invoke(cli_app(), start_args(db, "run-a", "--fake"))

    assert restarted.exit_code == 0, restarted.output
    with opened(db) as store:
        sessions = [stored for stored in store.read_events("run-a") if stored.type == "session_started"]
    assert len(sessions) == 2
    assert sessions[1].payload["config_hash"] == sessions[0].payload["config_hash"]
    assert sessions[1].payload["random_seed"] == 3
    assert sessions[1].payload["restart"] is True


def test_restart_rejects_a_conflicting_room(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    config_path = write_config(tmp_path / "run.toml")
    assert runner.invoke(cli_app(), start_args(db, "run-lab", "--config", str(config_path), "--room", "lab",
                                               "--fake")).exit_code == 0
    before = read_types(db, "run-lab")

    result = runner.invoke(cli_app(), start_args(db, "run-lab", "--room", "other", "--fake"))

    assert result.exit_code == 1
    assert "lab" in result.output and "other" in result.output
    assert read_types(db, "run-lab") == before


def test_restart_rejects_a_conflicting_config(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    original = write_config(tmp_path / "a.toml", seed=3)
    changed = write_config(tmp_path / "b.toml", seed=4)
    assert runner.invoke(cli_app(), start_args(db, "run-a", "--config", str(original), "--fake")).exit_code == 0
    before = read_types(db, "run-a")

    result = runner.invoke(cli_app(), start_args(db, "run-a", "--config", str(changed), "--fake"))

    assert result.exit_code == 1
    assert "recorded" in result.output
    assert read_types(db, "run-a") == before


def test_new_run_requires_a_config(tmp_path: Path) -> None:
    db = tmp_path / "room.db"

    result = runner.invoke(cli_app(), start_args(db, "run-new", "--fake"))

    assert result.exit_code == 1
    assert "--config" in result.output
    if db.exists():
        with opened(db) as store:
            assert store.get_run("run-new") is None


def test_start_reports_engine_start_errors_without_a_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from driftroom.engine import SimulationEngine

    def broken_start(self: SimulationEngine) -> int:
        raise KeyError("unknown room: lab")

    monkeypatch.setattr(SimulationEngine, "start", broken_start)
    db = tmp_path / "room.db"

    result = runner.invoke(cli_app(), start_args(db, "run-a", "--config", str(write_config(tmp_path / "r.toml")),
                                                 "--fake"))

    assert result.exit_code == 1
    assert "unknown room: lab" in result.output
    assert isinstance(result.exception, SystemExit)


# Review fixes: fence fake runs ----------------------------------------------------------

class RealStubBackend:
    """Stands in for OllamaBackend so a refusal cannot be confused with an unreachable server."""

    def __init__(self, runtime: object, *, host: str | None = None) -> None:
        pass

    def model_info(self, model: str) -> object:
        from driftroom.models.base import ModelInfo
        return ModelInfo(model, "sha256:real")

    def decide(self, agent: object, messages: object) -> object:
        from driftroom.models.base import Decision
        from driftroom.models.fake import FakeModelBackend
        return FakeModelBackend([Decision(action="wait", message=None, target=None)]).decide(agent, messages)


def test_fake_option_is_hidden_from_help() -> None:
    result = runner.invoke(cli_app(), ["start", "--help"])

    assert result.exit_code == 0, result.output
    assert "--fake" not in result.output


def test_fake_runs_record_fake_model_digests(tmp_path: Path) -> None:
    db = tmp_path / "room.db"

    result = runner.invoke(cli_app(), start_args(db, "run-f", "--config", str(write_config(tmp_path / "r.toml")),
                                                 "--fake"))

    assert result.exit_code == 0, result.output
    with opened(db) as store:
        started = store.read_events("run-f")[0]
    assert started.type == "session_started"
    assert started.payload["model_digests"] == {"qwen3:4b": "fake"}


def test_fake_start_refuses_a_run_recorded_with_real_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import driftroom.models.ollama_backend as ollama_backend

    monkeypatch.setattr(ollama_backend, "OllamaBackend", RealStubBackend)
    db = tmp_path / "room.db"
    config_path = write_config(tmp_path / "r.toml")
    real = runner.invoke(cli_app(), start_args(db, "run-real", "--config", str(config_path)))
    assert real.exit_code == 0, real.output
    before = read_types(db, "run-real")

    result = runner.invoke(cli_app(), start_args(db, "run-real", "--fake"))

    assert result.exit_code == 1
    assert "fake" in result.output
    assert read_types(db, "run-real") == before


def test_real_start_refuses_a_fake_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import driftroom.models.ollama_backend as ollama_backend

    monkeypatch.setattr(ollama_backend, "OllamaBackend", RealStubBackend)
    db = tmp_path / "room.db"
    config_path = write_config(tmp_path / "r.toml")
    assert runner.invoke(cli_app(), start_args(db, "run-f", "--config", str(config_path), "--fake")).exit_code == 0
    before = read_types(db, "run-f")

    result = runner.invoke(cli_app(), start_args(db, "run-f"))

    assert result.exit_code == 1
    assert "fake" in result.output
    assert read_types(db, "run-f") == before


# Review fixes: readers never write to foreign databases ---------------------------------

@pytest.mark.parametrize("kind", ["text", "empty-sqlite", "foreign-sqlite"])
def test_status_rejects_a_non_driftroom_database_without_modifying_it(tmp_path: Path, kind: str) -> None:
    import sqlite3

    db = tmp_path / "other.db"
    if kind == "text":
        db.write_text("just some notes\n", encoding="utf-8")
    else:
        connection = sqlite3.connect(db)
        if kind == "foreign-sqlite":
            connection.execute("CREATE TABLE notes (body TEXT)")
            connection.commit()
        connection.close()
    before = db.read_bytes()

    result = runner.invoke(cli_app(), ["status", "--db", str(db), "--run", "run-1"])

    assert result.exit_code == 1
    assert "not a Driftroom database" in result.output
    assert isinstance(result.exception, SystemExit)
    assert db.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["other.db"]


@pytest.mark.parametrize("command", ["watch", "export", "pause"])
def test_readers_and_controls_reject_a_foreign_sqlite_database(tmp_path: Path, command: str) -> None:
    import sqlite3

    db = tmp_path / "other.db"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE notes (body TEXT)")
    connection.commit()
    connection.close()
    before = db.read_bytes()
    args = {
        "watch": ["watch", "--db", str(db), "--run", "run-1", "--once"],
        "export": ["export", "--db", str(db), "--run", "run-1", "--format", "text", "--output", str(tmp_path / "o.txt")],
        "pause": ["pause", "--db", str(db), "--room", "room-1"],
    }[command]

    result = runner.invoke(cli_app(), args)

    assert result.exit_code == 1
    assert "not a Driftroom database" in result.output
    assert isinstance(result.exception, SystemExit)
    assert db.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["other.db"]


# Review fixes: atomic export ------------------------------------------------------------

def test_export_into_a_missing_directory_fails_cleanly(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        seed_history(store)
    output = tmp_path / "missing" / "run.txt"

    result = runner.invoke(cli_app(), [
        "export", "--db", str(db), "--run", "run-1", "--format", "text", "--output", str(output),
    ])

    assert result.exit_code == 1
    assert f"output directory does not exist: {output.parent}" in result.output
    assert isinstance(result.exception, SystemExit)


def test_export_replaces_the_output_atomically_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        seed_history(store)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    output = out_dir / "run.txt"
    output.write_text("previous export\n", encoding="utf-8")

    result = runner.invoke(cli_app(), [
        "export", "--db", str(db), "--run", "run-1", "--format", "text", "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8").startswith("09:14  june\n")
    assert [path.name for path in out_dir.iterdir()] == ["run.txt"]


def test_failed_export_keeps_the_previous_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import driftroom.cli as cli

    db = tmp_path / "room.db"
    with seeded_store(db) as store:
        seed_history(store)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    output = out_dir / "run.txt"
    output.write_text("previous export\n", encoding="utf-8")

    def exploding_render(stored: StoredEvent, *, debug: bool = False) -> str | None:
        if stored.type == "environment":
            raise RuntimeError("boom")
        return f"{stored.type}"

    monkeypatch.setattr(cli, "render_event", exploding_render)

    result = runner.invoke(cli.app, [
        "export", "--db", str(db), "--run", "run-1", "--format", "text", "--output", str(output),
    ])

    assert result.exit_code != 0
    assert output.read_text(encoding="utf-8") == "previous export\n"
    assert [path.name for path in out_dir.iterdir()] == ["run.txt"]


# Final review fixes: one engine per database, a recorded digest, a fixed regime ---------

@contextmanager
def held_engine_lock(db: Path) -> Iterator[None]:
    import fcntl

    with open(f"{db}.engine.lock", "a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def test_start_refuses_a_database_another_engine_holds(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    config_path = write_config(tmp_path / "r.toml")
    first = runner.invoke(cli_app(), start_args(db, "run-a", "--config", str(config_path), "--fake"))
    assert first.exit_code == 0, first.output
    before = read_types(db, "run-a")

    with held_engine_lock(db):
        result = runner.invoke(cli_app(), start_args(db, "run-a", "--fake"))

    assert result.exit_code == 1
    assert f"another driftroom engine is already running on {db}" in result.output
    assert read_types(db, "run-a") == before


def test_a_finished_start_releases_the_engine_lock(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    config_path = write_config(tmp_path / "r.toml")

    result = runner.invoke(cli_app(), start_args(db, "run-a", "--config", str(config_path), "--fake"))

    assert result.exit_code == 0, result.output
    with held_engine_lock(db):  # would raise BlockingIOError if start still held it
        pass


class NoDigestBackend(RealStubBackend):
    def model_info(self, model: str) -> object:
        from driftroom.models.base import ModelInfo
        return ModelInfo(model, None)


def test_real_start_refuses_a_model_without_a_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import driftroom.models.ollama_backend as ollama_backend

    monkeypatch.setattr(ollama_backend, "OllamaBackend", NoDigestBackend)
    db = tmp_path / "room.db"

    result = runner.invoke(cli_app(), start_args(db, "run-real", "--config", str(write_config(tmp_path / "r.toml"))))

    assert result.exit_code == 1
    assert "qwen3:4b" in result.output and "ollama pull qwen3:4b" in result.output
    with opened(db) as store:
        assert store.read_events("run-real") == []


def changed_prompt_hash(monkeypatch: pytest.MonkeyPatch) -> str:
    import driftroom.engine
    import driftroom.prompting

    changed = "0" * 64
    monkeypatch.setattr(driftroom.prompting, "prompt_template_hash", lambda: changed)
    monkeypatch.setattr(driftroom.engine, "prompt_template_hash", lambda: changed)
    return changed


def test_restart_refuses_a_changed_regime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "room.db"
    assert runner.invoke(cli_app(), start_args(db, "run-a", "--config", str(write_config(tmp_path / "r.toml")),
                                               "--fake")).exit_code == 0
    before = read_types(db, "run-a")
    changed_prompt_hash(monkeypatch)

    result = runner.invoke(cli_app(), start_args(db, "run-a", "--fake"))

    assert result.exit_code == 1
    assert "regime" in result.output and "--allow-regime-change" in result.output
    assert "prompt" in result.output
    assert read_types(db, "run-a") == before


def test_allow_regime_change_restarts_under_the_new_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from driftroom.config import run_fingerprint

    db = tmp_path / "room.db"
    assert runner.invoke(cli_app(), start_args(db, "run-a", "--config", str(write_config(tmp_path / "r.toml")),
                                               "--fake")).exit_code == 0
    changed = changed_prompt_hash(monkeypatch)

    result = runner.invoke(cli_app(), start_args(db, "run-a", "--fake", "--allow-regime-change"))

    assert result.exit_code == 0, result.output
    with opened(db) as store:
        sessions = [stored for stored in store.read_events("run-a") if stored.type == "session_started"]
        recorded = store.get_run("run-a").config
    assert len(sessions) == 2
    expected = run_fingerprint(recorded, prompt_hash=changed, model_digests={"qwen3:4b": "fake"})
    assert sessions[1].payload["run_fingerprint"] == expected != sessions[0].payload["run_fingerprint"]


def test_allow_regime_change_is_documented_in_help() -> None:
    result = runner.invoke(cli_app(), ["start", "--help"])

    assert result.exit_code == 0, result.output
    assert "--allow-regime-change" in result.output
