"""SQLite event history and the small cross-process control plane."""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Iterator, Literal

from .config import canonical_config_json, run_config_hash
from .domain import RunConfig


DesiredState = Literal["running", "paused", "stop_requested"]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    room_id: str
    started_at: str
    config: RunConfig


@dataclass(frozen=True)
class EventRecord:
    run_id: str
    wall_ts: str
    sim_ms: int
    type: str
    agent_id: str | None
    payload: Mapping[str, object]


@dataclass(frozen=True)
class StoredEvent:
    id: int
    run_id: str
    wall_ts: str
    sim_ms: int
    type: str
    agent_id: str | None
    payload: dict[str, object]


@dataclass(frozen=True)
class ControlState:
    room_id: str
    desired_state: DesiredState


@dataclass(frozen=True)
class StateUpdate:
    """One derived-state row to upsert with an event."""

    table: Literal["agent_state", "relationships", "memories"]
    values: Mapping[str, object]


_STATE_KEYS = {
    "agent_state": ("run_id", "agent_id"),
    "relationships": ("run_id", "source_agent_id", "target_agent_id"),
    "memories": ("run_id", "agent_id", "source_event_id"),
}

_STATE_COLUMNS = {
    "agent_state": frozenset((*_STATE_KEYS["agent_state"], "energy", "attention", "mood", "updated_sim_ms")),
    "relationships": frozenset((*_STATE_KEYS["relationships"], "familiarity", "affinity", "tension", "updated_sim_ms")),
    "memories": frozenset((*_STATE_KEYS["memories"], "salience", "times_recalled", "last_recalled_ms")),
}


class EventStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._db = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA foreign_keys=ON")

    def initialize(self) -> None:
        self._db.execute("PRAGMA journal_mode=WAL")
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version != 0 and version != 1:
            raise RuntimeError(f"unsupported Driftroom schema version: {version}")
        try:
            self._db.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    room_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    config_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agents (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    agent_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    PRIMARY KEY (run_id, agent_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    wall_ts TEXT NOT NULL,
                    sim_ms INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    agent_id TEXT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_run_id_id ON events(run_id, id);
                CREATE TABLE IF NOT EXISTS room_controls (
                    room_id TEXT PRIMARY KEY,
                    desired_state TEXT NOT NULL CHECK (desired_state IN ('running', 'paused', 'stop_requested'))
                );
                CREATE TABLE IF NOT EXISTS agent_state (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    agent_id TEXT NOT NULL,
                    energy REAL NOT NULL,
                    attention REAL NOT NULL,
                    mood TEXT NOT NULL,
                    updated_sim_ms INTEGER NOT NULL,
                    PRIMARY KEY (run_id, agent_id),
                    FOREIGN KEY (run_id, agent_id) REFERENCES agents(run_id, agent_id)
                );
                CREATE TABLE IF NOT EXISTS relationships (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    source_agent_id TEXT NOT NULL,
                    target_agent_id TEXT NOT NULL,
                    familiarity REAL NOT NULL,
                    affinity REAL NOT NULL,
                    tension REAL NOT NULL,
                    updated_sim_ms INTEGER NOT NULL,
                    PRIMARY KEY (run_id, source_agent_id, target_agent_id)
                );
                CREATE TABLE IF NOT EXISTS memories (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    agent_id TEXT NOT NULL,
                    source_event_id INTEGER NOT NULL REFERENCES events(id),
                    salience REAL NOT NULL,
                    times_recalled INTEGER NOT NULL,
                    last_recalled_ms INTEGER NOT NULL,
                    PRIMARY KEY (run_id, agent_id, source_event_id)
                );
                PRAGMA user_version=1;
                COMMIT;
                """
            )
        except BaseException:
            self._db.rollback()
            raise

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._db.rollback()
            raise
        else:
            self._db.commit()

    def create_run(self, run: RunRecord) -> None:
        with self._transaction():
            self._db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                (run.run_id, run.room_id, run.started_at, canonical_config_json(run.config), run_config_hash(run.config)),
            )
            self._db.executemany(
                "INSERT INTO agents VALUES (?, ?, ?, ?)",
                [(run.run_id, agent.id, agent.name, agent.model) for agent in run.config.agents],
            )
            self._db.execute(
                "INSERT OR IGNORE INTO room_controls VALUES (?, 'running')", (run.room_id,)
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._db.execute(
            "SELECT run_id, room_id, started_at, config_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return RunRecord(row["run_id"], row["room_id"], row["started_at"], RunConfig.model_validate_json(row["config_json"]))

    def append_event(self, event: EventRecord) -> int:
        return self.commit_event(event)

    def commit_event(self, event: EventRecord, state_updates: Sequence[StateUpdate] = ()) -> int:
        with self._transaction():
            cursor = self._db.execute(
                "INSERT INTO events (run_id, wall_ts, sim_ms, type, agent_id, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                (event.run_id, event.wall_ts, event.sim_ms, event.type, event.agent_id,
                 json.dumps(event.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)),
            )
            event_id = cursor.lastrowid
            assert event_id is not None
            for update in state_updates:
                self._apply_state_update(update, event.run_id)
        return event_id

    def _apply_state_update(self, update: StateUpdate, event_run_id: str) -> None:
        table = update.table
        if table not in _STATE_KEYS:
            raise ValueError(f"unsupported state table: {table}")
        values = update.values
        if not values or not set(values) <= _STATE_COLUMNS[table] or not set(_STATE_KEYS[table]) <= set(values):
            raise ValueError(f"invalid columns for {table}")
        if values["run_id"] != event_run_id:
            raise ValueError("state update run_id must match event run_id")
        if table == "memories":
            source = self._db.execute(
                "SELECT 1 FROM events WHERE id=? AND run_id=?",
                (values["source_event_id"], event_run_id),
            ).fetchone()
            if source is None:
                raise ValueError("memory source_event_id must belong to the event run")
        columns = tuple(values)
        names = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        keys = _STATE_KEYS[table]
        changes = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in keys)
        sql = f"INSERT INTO {table} ({names}) VALUES ({placeholders}) ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {changes}"
        self._db.execute(sql, tuple(values.values()))

    def read_events(self, run_id: str, after_id: int = 0, limit: int = 500) -> list[StoredEvent]:
        rows = self._db.execute(
            "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT ?",
            (run_id, after_id, limit),
        ).fetchall()
        return [
            StoredEvent(row["id"], row["run_id"], row["wall_ts"], row["sim_ms"],
                        row["type"], row["agent_id"], json.loads(row["payload_json"]))
            for row in rows
        ]

    def set_control(self, room_id: str, desired_state: DesiredState) -> None:
        if desired_state not in ("running", "paused", "stop_requested"):
            raise ValueError(f"invalid control state: {desired_state}")
        with self._transaction():
            self._db.execute(
                "INSERT INTO room_controls VALUES (?, ?) ON CONFLICT(room_id) DO UPDATE SET desired_state=excluded.desired_state",
                (room_id, desired_state),
            )

    def get_control(self, room_id: str) -> ControlState:
        row = self._db.execute(
            "SELECT desired_state FROM room_controls WHERE room_id=?", (room_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown room: {room_id}")
        return ControlState(room_id, row["desired_state"])

    def close(self) -> None:
        self._db.close()
