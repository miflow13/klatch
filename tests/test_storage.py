"""Durability and control behavior at the EventStore boundary."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
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
        payload={"speaker": "June", "message": f"message {sim_ms}"},
    )


@contextmanager
def opened(path: Path, *, initialize: bool = True) -> Iterator[EventStore]:
    """An EventStore that is always closed (Python 3.13+ warns on unclosed sqlite connections)."""
    store = EventStore(path)
    try:
        if initialize:
            store.initialize()
        yield store
    finally:
        store.close()


def test_wal_mode_and_versioned_schema(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    with opened(path):
        pass

    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"runs", "agents", "events", "room_controls", "agent_state", "relationships", "memories"} <= tables


def test_event_ids_persist_and_cursor_returns_only_later_events(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    run = run_record()
    with opened(path) as store:
        store.create_run(run)
        first = store.append_event(message(100))
        second = store.append_event(message(200))
    assert isinstance(first, int) and second > first

    with opened(path) as reopened:
        assert reopened.get_run(run.run_id) == run
        assert [(event.id, event.payload["message"]) for event in reopened.read_events(run.run_id)] == [
            (first, "message 100"),
            (second, "message 200"),
        ]
        assert [event.id for event in reopened.read_events(run.run_id, after_id=first)] == [second]


def test_controller_write_is_visible_to_separate_engine_connection(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    with opened(path) as engine, opened(path) as controller:
        engine.create_run(run_record())
        controller.set_control("room-1", "paused")
        assert engine.get_control("room-1").desired_state == "paused"
        assert engine.read_events("run-1") == []


def test_event_and_derived_state_commit_together(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    with opened(path) as store:
        store.create_run(run_record())
        event_id = store.commit_event(
            message(100),
            [StateUpdate(table="agent_state", values={"run_id": "run-1", "agent_id": "june", "energy": 0.62, "attention": 0.7, "mood": "neutral", "updated_sim_ms": 100})],
        )
        assert [event.id for event in store.read_events("run-1")] == [event_id]
        with closing(sqlite3.connect(path)) as db:
            assert db.execute("SELECT energy FROM agent_state WHERE run_id='run-1' AND agent_id='june'").fetchone() == (0.62,)

        with pytest.raises(sqlite3.IntegrityError):
            store.commit_event(
                message(200),
                [StateUpdate(table="agent_state", values={"run_id": "run-1", "agent_id": "missing", "energy": 0.5, "attention": 0.7, "mood": "neutral", "updated_sim_ms": 200})],
            )
        assert [event.id for event in store.read_events("run-1")] == [event_id]


@pytest.mark.parametrize("table", ["agent_state", "relationships", "memories"])
def test_state_update_cannot_cross_event_run(tmp_path, table: str) -> None:
    path = tmp_path / "room.sqlite3"
    with opened(path) as store:
        store.create_run(run_record())
        store.create_run(replace(run_record(), run_id="run-2", room_id="room-2"))
        source_id = store.append_event(replace(message(10), run_id="run-2"))
        values_by_table = {
            "agent_state": {"run_id": "run-2", "agent_id": "june", "energy": 0.5, "attention": 0.7, "mood": "neutral", "updated_sim_ms": 100},
            "relationships": {"run_id": "run-2", "source_agent_id": "june", "target_agent_id": "milo", "familiarity": 0.2, "affinity": 0.0, "tension": 0.0, "updated_sim_ms": 100},
            "memories": {"run_id": "run-2", "agent_id": "june", "source_event_id": source_id, "salience": 0.3, "times_recalled": 0, "last_recalled_ms": 100},
        }

        with pytest.raises(ValueError, match="run"):
            store.commit_event(message(100), [StateUpdate(table=table, values=values_by_table[table])])

        assert store.read_events("run-1") == []
        assert [event.id for event in store.read_events("run-2")] == [source_id]
        with closing(sqlite3.connect(path)) as db:
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


def test_key_only_state_update_is_rejected_before_sql(tmp_path) -> None:
    with opened(tmp_path / "room.sqlite3") as store:
        store.create_run(run_record())

        with pytest.raises(ValueError, match="non-key"):
            store.commit_event(
                message(100),
                [StateUpdate(table="agent_state", values={"run_id": "run-1", "agent_id": "june"})],
            )

        assert store.read_events("run-1") == []


def test_memory_source_event_must_belong_to_update_run(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    with opened(path) as store:
        store.create_run(run_record())
        store.create_run(replace(run_record(), run_id="run-2", room_id="room-2"))
        other_run_source = store.append_event(replace(message(10), run_id="run-2"))
        memory = StateUpdate(
            table="memories",
            values={"run_id": "run-1", "agent_id": "june", "source_event_id": other_run_source, "salience": 0.3, "times_recalled": 0, "last_recalled_ms": 100},
        )

        with pytest.raises(ValueError, match="source_event_id"):
            store.commit_event(message(100), [memory])

        assert store.read_events("run-1") == []
        with closing(sqlite3.connect(path)) as db:
            assert db.execute("SELECT COUNT(*) FROM memories").fetchone() == (0,)


def test_pause_and_append_race_keeps_one_event_and_control(tmp_path) -> None:
    path = tmp_path / "room.sqlite3"
    with opened(path) as engine, opened(path) as controller:
        engine.create_run(run_record())
        with ThreadPoolExecutor(max_workers=2) as pool:
            append = pool.submit(engine.append_event, message(100))
            pause = pool.submit(controller.set_control, "room-1", "paused")
            event_id = append.result()
            pause.result()
        assert [event.id for event in controller.read_events("run-1")] == [event_id]
        assert engine.get_control("room-1").desired_state == "paused"


def test_read_recent_events_returns_last_n_in_ascending_order(tmp_path) -> None:
    with opened(tmp_path / "room.sqlite3") as store:
        store.create_run(run_record())
        ids = [store.append_event(message(sim_ms)) for sim_ms in (100, 200, 300, 400, 500)]

        recent = store.read_recent_events("run-1", 2)
        assert [event.id for event in recent] == ids[-2:]
        assert [event.payload["message"] for event in recent] == ["message 400", "message 500"]
        assert [event.id for event in store.read_recent_events("run-1", 50)] == ids
        assert store.read_recent_events("run-1", 0) == []
        assert store.read_recent_events("other-run", 5) == []
        with pytest.raises(ValueError):
            store.read_recent_events("run-1", -1)


def test_read_recent_events_can_bound_the_window_over_selected_types(tmp_path) -> None:
    with opened(tmp_path / "room.sqlite3") as store:
        store.create_run(run_record())
        first = store.append_event(message(100))
        for sim_ms in (200, 300, 400):
            store.append_event(replace(message(sim_ms), type="agent_wait", payload={}))
        environment = store.append_event(
            replace(message(500), type="environment", agent_id=None, payload={"text": "quiet"})
        )
        store.append_event(replace(message(600), type="attempt_failed", payload={}))

        visible = store.read_recent_events("run-1", 2, types={"message", "environment"})
        assert [event.id for event in visible] == [first, environment]
        assert [event.id for event in store.read_recent_events("run-1", 1, types=["message"])] == [first]
        assert [event.type for event in store.read_recent_events("run-1", 2)] == ["environment", "attempt_failed"]
        assert store.read_recent_events("run-1", 5, types=()) == []
        # Type names are bound parameters, never interpolated into the SQL.
        assert store.read_recent_events("run-1", 5, types=["message' OR '1'='1"]) == []


def test_has_schema_detects_initialized_stores_without_creating_tables(tmp_path) -> None:
    foreign = tmp_path / "foreign.db"
    with closing(sqlite3.connect(foreign)) as connection:
        connection.execute("CREATE TABLE notes (body TEXT)")
        connection.commit()
    with opened(foreign, initialize=False) as store:
        assert store.has_schema() is False
        assert store.has_schema() is False
    with closing(sqlite3.connect(foreign)) as connection:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert tables == ["notes"]

    with opened(tmp_path / "driftroom.db") as store:
        assert store.has_schema() is True
