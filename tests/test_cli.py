"""Terminal observer rendering and the operational CLI (spec §24A, §29, §30)."""

import json
import os
from pathlib import Path
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


def seeded_store(db: Path, run_id: str = "run-1", room_id: str = "room-1") -> EventStore:
    store = EventStore(db)
    store.initialize()
    store.create_run(RunRecord(run_id, room_id, "2026-10-01T12:00:00+00:00", config()))
    return store


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
    seeded_store(db).set_control("room-1", "paused" if state == "running" else "running")

    result = runner.invoke(cli_app(), [command, "--db", str(db), "--room", "room-1"])

    assert result.exit_code == 0, result.output
    assert EventStore(db).get_control("room-1").desired_state == state


def test_control_command_rejects_unknown_room(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    seeded_store(db)

    result = runner.invoke(cli_app(), ["pause", "--db", str(db), "--room", "room-typo"])

    assert result.exit_code != 0
    assert "room-typo" in result.output
    with pytest.raises(KeyError):
        EventStore(db).get_control("room-typo")


def test_commands_reject_a_missing_database(tmp_path: Path) -> None:
    db = tmp_path / "missing.db"

    result = runner.invoke(cli_app(), ["status", "--db", str(db), "--run", "run-1"])

    assert result.exit_code != 0
    assert not db.exists()


def test_control_commands_never_import_the_model_backend(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    seeded_store(db).close()
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
    assert EventStore(db).get_control("room-1").desired_state == "stop_requested"


# 3. Start on the fake backend -----------------------------------------------------------

def test_start_fake_runs_the_engine_end_to_end(tmp_path: Path) -> None:
    data = config().model_dump(mode="json")
    data["runtime"].update({"clock_mode": "accelerated", "random_seed": 3})
    data["scheduler"].update(FORCED)
    config_path = tmp_path / "run.toml"
    config_path.write_text(_toml(data), encoding="utf-8")
    db = tmp_path / "room.db"

    result = runner.invoke(cli_app(), [
        "start", "--config", str(config_path), "--db", str(db), "--run", "run-fake", "--fake", "--max-steps", "6",
    ])

    assert result.exit_code == 0, result.output
    assert "run-fake" in result.output
    store = EventStore(db)
    types = [stored.type for stored in store.read_events("run-fake")]
    assert types[0] == "session_started"
    assert types[-1] == "session_ended"
    assert "message" in types and "agent_wait" in types
    first_message = next(stored for stored in store.read_events("run-fake") if stored.type == "message")
    assert f"  {first_message.agent_id}\n       {first_message.payload['message']}" in result.output
    assert store.get_control("room-1").desired_state == "running"


def test_start_fake_defaults_run_id_and_room(tmp_path: Path) -> None:
    data = config().model_dump(mode="json")
    data["runtime"].update({"clock_mode": "accelerated"})
    config_path = tmp_path / "run.toml"
    config_path.write_text(_toml(data), encoding="utf-8")
    db = tmp_path / "room.db"

    result = runner.invoke(cli_app(), ["start", "--config", str(config_path), "--db", str(db), "--fake", "--max-steps", "1"])

    assert result.exit_code == 0, result.output
    run_id = EventStore(db)._db.execute("SELECT run_id, room_id FROM runs").fetchone()
    assert run_id[0].startswith("run-") and run_id[0] in result.output
    assert run_id[1] == "room-1"


def test_start_stops_the_engine_on_keyboard_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from driftroom.engine import SimulationEngine

    def interrupted_step(self: SimulationEngine) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(SimulationEngine, "step", interrupted_step)
    data = config().model_dump(mode="json")
    data["runtime"].update({"clock_mode": "accelerated"})
    config_path = tmp_path / "run.toml"
    config_path.write_text(_toml(data), encoding="utf-8")
    db = tmp_path / "room.db"

    result = runner.invoke(cli_app(), ["start", "--config", str(config_path), "--db", str(db), "--run", "run-x", "--fake"])

    assert result.exit_code == 0, result.output
    assert [stored.type for stored in EventStore(db).read_events("run-x")] == ["session_started", "session_ended"]


def _toml(data: dict[str, object]) -> str:
    def value(item: object) -> str:
        return json.dumps(item)

    lines: list[str] = []
    for section in ("scheduler", "runtime"):
        lines.append(f"[{section}]")
        lines += [f"{key} = {value(item)}" for key, item in data[section].items() if item is not None]  # type: ignore[union-attr]
    for agent in data["agents"]:  # type: ignore[union-attr]
        lines.append("[[agents]]")
        lines += [f"{key} = {value(agent[key])}" for key in ("id", "name", "model")]
        for table in ("traits", "sampling"):
            lines.append(f"[agents.{table}]")
            lines += [f"{key} = {value(item)}" for key, item in agent[table].items()]
    return "\n".join(lines) + "\n"


# 4. Status -------------------------------------------------------------------------------

def test_status_reports_run_control_time_and_counts(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    store = seeded_store(db)
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
    seeded_store(db)

    result = runner.invoke(cli_app(), ["status", "--db", str(db), "--run", "run-nope"])

    assert result.exit_code != 0
    assert "run-nope" in result.output


# 5. Watch --------------------------------------------------------------------------------

def test_watch_once_prints_visible_history_and_hides_infrastructure(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    seed_history(seeded_store(db))

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
    seed_history(seeded_store(db))

    result = runner.invoke(cli_app(), ["watch", "--db", str(db), "--run", "run-1", "--once", "--debug"])

    assert result.exit_code == 0, result.output
    assert "[attempt_failed] ada" in result.output
    assert "[generation_failed] ada" in result.output
    assert "09:15  milo\n       not really?" in result.output


def test_watch_once_reads_past_one_page_of_events(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    store = seeded_store(db)
    for index in range(501):
        append(store, "message", index * 1_000, {"speaker": "Ada", "message": f"line {index}", "target": None}, "ada")

    result = runner.invoke(cli_app(), ["watch", "--db", str(db), "--run", "run-1", "--once"])

    assert result.exit_code == 0, result.output
    assert "line 0\n" in result.output and "line 500\n" in result.output


# 6. Export -------------------------------------------------------------------------------

def test_export_jsonl_contains_every_event_including_failures(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    seed_history(seeded_store(db))
    output = tmp_path / "run.jsonl"

    result = runner.invoke(cli_app(), [
        "export", "--db", str(db), "--run", "run-1", "--format", "jsonl", "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == [stored.type for stored in EventStore(db).read_events("run-1")]
    assert set(rows[0]) == {"id", "run_id", "wall_ts", "sim_ms", "type", "agent_id", "payload"}
    failure = next(row for row in rows if row["type"] == "generation_failed")
    assert failure["agent_id"] == "ada"
    assert failure["payload"] == {"attempts": 2, "last_error_class": "BackendTimeoutError"}


def test_export_text_contains_only_visible_room_history(tmp_path: Path) -> None:
    db = tmp_path / "room.db"
    seed_history(seeded_store(db))
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
    seed_history(seeded_store(db))
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
