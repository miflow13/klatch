"""Operational CLI (spec §24A, §30): run the engine, observe it, and control it.

Every command except ``start`` only touches SQLite through ``EventStore``; the
model backend and engine are imported inside ``start`` so control commands never
load them. The CLI never writes dialogue: only the engine commits events.

Only ``start`` initializes a database. The other commands open it as it is and
refuse anything without the Driftroom tables, so they never add schema to a
foreign SQLite file. A restart takes the room and config recorded for the run
(spec §25, §33), so the regime of a run cannot change underneath it.
"""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
from random import Random
import secrets
import sqlite3
import time
from typing import Annotated, NoReturn

from rich.console import Console
import typer

from .config import ENGINE_VERSION, load_run_config, run_config_hash, run_fingerprint
from .domain import VISIBLE_EVENT_TYPES
from .observer import format_sim_time, render_event
from .storage import DesiredState, EventStore, RunRecord, StoredEvent


app = typer.Typer(
    name="driftroom", help="Run, observe, and control a Driftroom chatroom.",
    no_args_is_help=True, add_completion=False,
)

DbOption = Annotated[Path, typer.Option("--db", help="SQLite database path.")]
RunOption = Annotated[str, typer.Option("--run", help="Run ID.")]
RoomOption = Annotated[str, typer.Option("--room", help="Room ID.")]

DEFAULT_ROOM = "room-1"
# ``start --fake`` records this digest for every model, so session_started and the run
# fingerprint mark the session as canned test chatter rather than model output.
FAKE_DIGEST = "fake"

# Canned lines for ``start --fake``: test tooling for end-to-end checks, not product behavior.
_FAKE_LINES = ("hey", "anyone around", "lol same", "huh", "not sure tbh", "ok fair", "wait what", "mm")


class ExportFormat(str, Enum):
    text = "text"
    jsonl = "jsonl"


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _open_store(db: Path, *, create: bool = False) -> EventStore:
    """Open the store. Only ``create`` (used by ``start``) initializes the schema; every
    other command opens the file as it is and refuses one without Driftroom tables."""
    if not create and not db.exists():
        _fail(f"database not found: {db}")
    store: EventStore | None = None
    try:
        store = EventStore(db)
        if create:
            store.initialize()
        elif not store.has_schema():
            store.close()
            _fail(f"not a Driftroom database: {db}")
    except sqlite3.DatabaseError as exc:
        if store is not None:
            store.close()
        _fail(f"not a Driftroom database: {db} ({exc})")
    return store


def _require_run(store: EventStore, run_id: str) -> RunRecord:
    record = store.get_run(run_id)
    if record is None:
        _fail(f"unknown run: {run_id}")
    return record


def _all_events(store: EventStore, run_id: str) -> Iterator[StoredEvent]:
    last_id = 0
    while batch := store.read_events(run_id, after_id=last_id):
        yield from batch
        last_id = batch[-1].id


def _first_session(store: EventStore, run_id: str) -> StoredEvent | None:
    """The run's first ``session_started`` event: the regime the run was recorded under."""
    return next((event for event in _all_events(store, run_id) if event.type == "session_started"), None)


def _recorded_digests(store: EventStore, run_id: str) -> dict[str, object] | None:
    """Model digests from the run's first ``session_started`` event, if it has one."""
    first = _first_session(store, run_id)
    if first is None:
        return None
    digests = first.payload.get("model_digests")
    return dict(digests) if isinstance(digests, dict) else {}


def _is_fake(digests: dict[str, object]) -> bool:
    return bool(digests) and all(digest == FAKE_DIGEST for digest in digests.values())


def _print_event(console: Console, event: StoredEvent, *, debug: bool) -> None:
    rendered = render_event(event, debug=debug)
    if rendered is None:
        return
    style = None if event.type in VISIBLE_EVENT_TYPES else "dim"
    console.print(rendered, style=style, markup=False, emoji=False, highlight=False, soft_wrap=True)
    console.print()


def _fake_backend(seed: int):  # noqa: ANN202 - the class is local to keep model imports out of control commands
    from .models.base import Decision, ModelBackend, ModelResult
    from .models.fake import FakeModelBackend

    def decisions() -> Iterator[Decision]:
        rng = Random(seed)
        while True:
            yield Decision(action="speak", message=rng.choice(_FAKE_LINES), target=None)
            yield Decision(action="wait", message=None, target=None)

    class FakeChatterBackend(ModelBackend):
        """An endless FakeModelBackend: alternates speak and wait."""

        def __init__(self) -> None:
            self._decisions = decisions()

        def decide(self, agent, messages) -> ModelResult:  # noqa: ANN001
            return FakeModelBackend([next(self._decisions)]).decide(agent, messages)

    return FakeChatterBackend()


@contextmanager
def _engine_lock(db: Path) -> Iterator[None]:
    """Hold ``<db>.engine.lock`` exclusively: one engine per database (spec §17, §24A).

    The lock is advisory and dies with the process, so a crashed engine never
    leaves a stale lock behind.
    """
    import fcntl

    lock_path = Path(f"{db}.engine.lock")
    try:
        handle = lock_path.open("a+b")
    except OSError as exc:
        _fail(f"could not open the engine lock {lock_path}: {exc}")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail(f"another driftroom engine is already running on {db}")
        yield
    finally:
        handle.close()  # closing the descriptor releases the lock


def _regime_changes(first: StoredEvent, current: dict[str, object]) -> list[str]:
    labels = {
        "config_hash": "config", "prompt_hash": "prompt templates",
        "model_digests": "model digests", "engine_version": "engine version",
    }
    return [label for key, label in labels.items() if first.payload.get(key) != current[key]]


@app.command()
def start(
    db: DbOption,
    config: Annotated[Path | None, typer.Option(
        "--config", exists=True, dir_okay=False,
        help="Run config TOML. Required for a new run; a restart uses the recorded config.",
    )] = None,
    run: Annotated[str | None, typer.Option("--run", help="Run ID (default: run-<UTC timestamp>).")] = None,
    room: Annotated[str | None, typer.Option(
        "--room", help=f"Room ID for a new run (default: {DEFAULT_ROOM}); a restart uses the recorded room.",
    )] = None,
    fake: Annotated[bool, typer.Option(
        "--fake", hidden=True, help="Test tooling: canned fake decisions instead of Ollama.",
    )] = False,
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1, help="Stop after N engine steps.")] = None,
    host: Annotated[str | None, typer.Option("--host", help="Ollama host URL.")] = None,
    allow_regime_change: Annotated[bool, typer.Option(
        "--allow-regime-change",
        help="Restart a run even though its regime (config, prompt templates, model digests, engine "
             "version) differs from the one its first session recorded.",
    )] = False,
) -> None:
    """Run the simulation engine in this terminal (a new run, or a restart of a recorded one)."""
    from .engine import SimulationEngine
    from .models.base import BackendError

    given_config = None
    if config is not None:
        try:
            given_config = load_run_config(config)
        except (OSError, ValueError) as exc:
            _fail(f"invalid config {config}: {exc}")
    run_id = run or f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    # `start` is the only command that creates a database, so it creates the directory too.
    try:
        db.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _fail(f"cannot create directory {db.parent}: {exc}")
    with _engine_lock(db):
        store = _open_store(db, create=True)
        try:
            record = store.get_run(run_id)
            if record is None:
                if given_config is None:
                    _fail(f"--config is required to start a new run ({run_id})")
                run_config, room_id = given_config, room or DEFAULT_ROOM
            else:
                # A restart continues the recorded run (spec §25, §33): same room, same regime.
                if room is not None and room != record.room_id:
                    _fail(f"run {run_id} belongs to room {record.room_id}, not {room}; omit --room to restart it")
                if given_config is not None and run_config_hash(given_config) != run_config_hash(record.config):
                    _fail(f"{config} differs from the config recorded for run {run_id}; a restart keeps the "
                          "recorded regime (omit --config, or start a new run)")
                run_config, room_id = record.config, record.room_id
                recorded = _recorded_digests(store, run_id)
                if recorded is not None and fake and not _is_fake(recorded):
                    _fail(f"run {run_id} was recorded with real models; --fake cannot continue it")
                if recorded is not None and not fake and _is_fake(recorded):
                    _fail(f"run {run_id} is a fake-backend test run; it can only be restarted with --fake")
            models = sorted({agent.model for agent in run_config.agents})
            if fake:
                seed = run_config.runtime.random_seed
                backend = _fake_backend(0 if seed is None else seed)
                model_digests: dict[str, str | None] = {model: FAKE_DIGEST for model in models}
            else:
                from .models.ollama_backend import OllamaBackend

                backend = OllamaBackend(run_config.runtime, host=host)
                try:
                    model_digests = {model: backend.model_info(model).digest for model in models}
                except BackendError as exc:
                    _fail(str(exc))
                for model, digest in model_digests.items():
                    if digest is None:
                        _fail(f"Ollama reports no digest for model {model}, so this run could not record "
                              f"exactly which model ran; install it with `ollama pull {model}`")
            first = _first_session(store, run_id) if record is not None else None
            if first is not None:
                from .prompting import prompt_template_hash

                # The same inputs the engine records in session_started (spec §33, §34).
                prompt_hash = prompt_template_hash()
                fingerprint = run_fingerprint(run_config, prompt_hash=prompt_hash, model_digests=model_digests)
                if first.payload.get("run_fingerprint") != fingerprint and not allow_regime_change:
                    changed = _regime_changes(first, {
                        "config_hash": run_config_hash(run_config), "prompt_hash": prompt_hash,
                        "model_digests": model_digests, "engine_version": ENGINE_VERSION,
                    })
                    _fail(f"restarting run {run_id} would change its recorded regime "
                          f"({', '.join(changed) or 'run fingerprint'} changed since its first session). "
                          "Everything observed in a run is read as one regime, so continuing would mix two "
                          "experiments under one run id. Start a new run instead, or pass "
                          "--allow-regime-change to continue this one; the new session_started then "
                          "records the new fingerprint.")
            console = Console()
            typer.echo(f"run id: {run_id}")
            typer.echo(f"room id: {room_id}")
            try:
                engine = SimulationEngine(
                    run_config, store, backend, run_id=run_id, room_id=room_id,
                    observer=lambda event: _print_event(console, event, debug=False),
                    model_digests=model_digests,
                )
                engine.start()
            except KeyError as exc:
                _fail(str(exc.args[0]) if exc.args else repr(exc))
            except (ValueError, BackendError) as exc:
                _fail(str(exc))
            # Read after engine.start(), which clears a stale stop request; a paused room stays paused.
            typer.echo(f"control: {store.get_control(room_id).desired_state}")
            try:
                engine.run(max_steps)
            except KeyboardInterrupt:
                pass  # run() already ended the session; stop() below is idempotent
            engine.stop()
            typer.echo(f"session ended: {run_id}")
        finally:
            store.close()


@app.command()
def watch(
    db: DbOption,
    run: RunOption,
    debug: Annotated[bool, typer.Option("--debug", help="Also show infrastructure events, dimmed.")] = False,
    poll_seconds: Annotated[float, typer.Option("--poll-seconds", min=0.01, help="Polling interval.")] = 1.0,
    once: Annotated[bool, typer.Option("--once", help="Print committed events and exit.")] = False,
) -> None:
    """Follow a run's committed events from SQLite."""
    store = _open_store(db)
    try:
        _require_run(store, run)
        console = Console()
        last_id = 0
        while True:
            batch = store.read_events(run, after_id=last_id)
            for event in batch:
                _print_event(console, event, debug=debug)
                last_id = event.id
            if batch:
                continue
            if once:
                return
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        return
    finally:
        store.close()


def _set_control(db: Path, room: str, state: DesiredState) -> None:
    store = _open_store(db)
    try:
        try:
            store.get_control(room)
        except KeyError:
            _fail(f"unknown room: {room}")
        store.set_control(room, state)
        typer.echo(f"{room}: {state}")
    finally:
        store.close()


@app.command()
def pause(db: DbOption, room: RoomOption) -> None:
    """Ask the engine to pause between inferences."""
    _set_control(db, room, "paused")


@app.command()
def resume(db: DbOption, room: RoomOption) -> None:
    """Ask a paused engine to continue."""
    _set_control(db, room, "running")


@app.command()
def stop(db: DbOption, room: RoomOption) -> None:
    """Ask the engine to end its session."""
    _set_control(db, room, "stop_requested")


@app.command()
def status(db: DbOption, run: RunOption) -> None:
    """Summarize a run from its persisted events."""
    store = _open_store(db)
    try:
        record = _require_run(store, run)
        counts: Counter[str] = Counter()
        latest_sim_ms = 0
        for event in _all_events(store, run):
            counts[event.type] += 1
            latest_sim_ms = event.sim_ms
        rows = (
            ("run", run),
            ("room", record.room_id),
            ("control", store.get_control(record.room_id).desired_state),
            ("sim time", format_sim_time(latest_sim_ms, seconds=True)),
            ("events", counts.total()),
            ("messages", counts["message"]),
            ("waits", counts["agent_wait"]),
            ("attempt failures", counts["attempt_failed"]),
            ("generation failures", counts["generation_failed"]),
            ("ambient events", counts["environment"]),
            ("models", ", ".join(sorted({agent.model for agent in record.config.agents}))),
        )
        for label, value in rows:
            typer.echo(f"{label}: {value}")
    finally:
        store.close()


@app.command()
def export(
    db: DbOption,
    run: RunOption,
    export_format: Annotated[ExportFormat, typer.Option("--format", help="text: room history; jsonl: every event.")],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, help="File to write.")],
) -> None:
    """Write a run's room transcript (text) or full event log (jsonl)."""
    store = _open_store(db)
    try:
        _require_run(store, run)
        if not output.parent.is_dir():
            _fail(f"output directory does not exist: {output.parent}")
        # Write beside the target and swap it in at the end, so a failed export never
        # leaves a truncated file where the previous one was.
        partial = output.with_name(f".{output.name}.{secrets.token_hex(4)}.partial")
        written = 0
        try:
            with partial.open("x", encoding="utf-8", newline="\n") as handle:
                for event in _all_events(store, run):
                    if export_format is ExportFormat.jsonl:
                        handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
                    elif (rendered := render_event(event)) is not None:
                        handle.write(("\n" if written else "") + rendered + "\n")
                    else:
                        continue
                    written += 1
            os.replace(partial, output)
        except OSError as exc:
            _fail(f"could not write {output}: {exc}")
        finally:
            partial.unlink(missing_ok=True)
        typer.echo(f"exported {written} events to {output}")
    finally:
        store.close()
