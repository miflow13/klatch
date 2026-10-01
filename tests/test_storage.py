"""Durability and control behavior at the EventStore boundary."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from driftroom.config import load_run_config
from driftroom.storage import EventRecord, EventStore, RunRecord, StateUpdate


def run_record() -> RunRecord:
    return RunRecord(
        run_id="run-1",
        room_id="room-1",
        started_at="2026-10-01T12:00:00+00:00",
        config=load_run_config(Path("driftroom.example.toml")),
    )


def message(sim_ms: int) -> EventRecord:
    return EventRecord(
        run_id="run-1",
        wall_ts=datetime(2026, 10, 1, tzinfo=timezone.utc).isoformat(),
        sim_ms=sim_ms,
        type="message",
        agent_id="june",
        payload={"text": f"message {sim_ms}"},
    )


def test_wal_mode_and_versioned_schema(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    store = EventStore(path)
    store.initialize()
    store.close()

    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"runs", "agents", "events", "room_controls", "agent_state", "relationships", "memories"} <= tables


def test_event_ids_persist_and_cursor_returns_only_later_events(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    store = EventStore(path)
    store.initialize()
    run = run_record()
    store.create_run(run)
    first = store.append_event(message(100))
    second = store.append_event(message(200))
    assert isinstance(first, int) and second > first
    store.close()

    reopened = EventStore(path)
    reopened.initialize()
    assert reopened.get_run(run.run_id) == run
    assert [(event.id, event.payload["text"]) for event in reopened.read_events(run.run_id)] == [
        (first, "message 100"),
        (second, "message 200"),
    ]
    assert [event.id for event in reopened.read_events(run.run_id, after_id=first)] == [second]
    reopened.close()


def test_controller_write_is_visible_to_separate_engine_connection(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    engine = EventStore(path)
    controller = EventStore(path)
    engine.initialize()
    controller.initialize()
    engine.create_run(run_record())
    controller.set_control("room-1", "paused")
    assert engine.get_control("room-1").desired_state == "paused"
    assert engine.read_events("run-1") == []
    engine.close()
    controller.close()


def test_event_and_derived_state_commit_together(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    store = EventStore(path)
    store.initialize()
    store.create_run(run_record())
    event_id = store.commit_event(
        message(100),
        [StateUpdate(table="agent_state", values={"run_id": "run-1", "agent_id": "june", "energy": 0.62, "attention": 0.7, "mood": "neutral", "updated_sim_ms": 100})],
    )
    assert [event.id for event in store.read_events("run-1")] == [event_id]
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT energy FROM agent_state WHERE run_id='run-1' AND agent_id='june'").fetchone() == (0.62,)

    with pytest.raises(sqlite3.IntegrityError):
        store.commit_event(
            message(200),
            [StateUpdate(table="agent_state", values={"run_id": "missing", "agent_id": "june", "energy": 0.5, "attention": 0.7, "mood": "neutral", "updated_sim_ms": 200})],
        )
    assert [event.id for event in store.read_events("run-1")] == [event_id]
    store.close()


def test_pause_and_append_race_keeps_one_event_and_control(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    engine = EventStore(path)
    controller = EventStore(path)
    engine.initialize()
    controller.initialize()
    engine.create_run(run_record())
    with ThreadPoolExecutor(max_workers=2) as pool:
        append = pool.submit(engine.append_event, message(100))
        pause = pool.submit(controller.set_control, "room-1", "paused")
        event_id = append.result()
        pause.result()
    assert [event.id for event in controller.read_events("run-1")] == [event_id]
    assert engine.get_control("room-1").desired_state == "paused"
    engine.close()
    controller.close()
