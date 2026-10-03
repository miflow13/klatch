"""Operational CLI (spec §24A, §30): run the engine, observe it, and control it.

Every command except ``start`` only touches SQLite through ``EventStore``; the
model backend and engine are imported inside ``start`` so control commands never
load them. The CLI never writes dialogue: only the engine commits events.
"""

from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from random import Random
import time
from typing import Annotated, NoReturn

from rich.console import Console
import typer

from .config import load_run_config
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

# Canned lines for ``start --fake``: test tooling for end-to-end checks, not product behavior.
_FAKE_LINES = ("hey", "anyone around", "lol same", "huh", "not sure tbh", "ok fair", "wait what", "mm")


class ExportFormat(str, Enum):
    text = "text"
    jsonl = "jsonl"


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _open_store(db: Path, *, create: bool = False) -> EventStore:
    if not create and not db.exists():
        _fail(f"database not found: {db}")
    store = EventStore(db)
    store.initialize()
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


@app.command()
def start(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False, help="Run config TOML.")],
    db: DbOption,
    run: Annotated[str | None, typer.Option("--run", help="Run ID (default: run-<UTC timestamp>).")] = None,
    room: RoomOption = "room-1",
    fake: Annotated[bool, typer.Option("--fake", help="Use canned fake decisions instead of Ollama.")] = False,
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1, help="Stop after N engine steps.")] = None,
    host: Annotated[str | None, typer.Option("--host", help="Ollama host URL.")] = None,
) -> None:
    """Run the simulation engine in this terminal."""
    from .engine import SimulationEngine

    try:
        run_config = load_run_config(config)
    except (OSError, ValueError) as exc:
        _fail(f"invalid config {config}: {exc}")
    run_id = run or f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    store = _open_store(db, create=True)
    try:
        model_digests: dict[str, str | None] | None = None
        if fake:
            seed = run_config.runtime.random_seed
            backend = _fake_backend(0 if seed is None else seed)
        else:
            from .models.base import BackendError
            from .models.ollama_backend import OllamaBackend

            backend = OllamaBackend(run_config.runtime, host=host)
            try:
                model_digests = {
                    model: backend.model_info(model).digest
                    for model in sorted({agent.model for agent in run_config.agents})
                }
            except BackendError as exc:
                _fail(str(exc))
        console = Console()
        typer.echo(f"run id: {run_id}")
        typer.echo(f"room id: {room}")
        engine = SimulationEngine(
            run_config, store, backend, run_id=run_id, room_id=room,
            observer=lambda event: _print_event(console, event, debug=False),
            model_digests=model_digests,
        )
        engine.start()
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
        written = 0
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for event in _all_events(store, run):
                if export_format is ExportFormat.jsonl:
                    handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
                elif (rendered := render_event(event)) is not None:
                    handle.write(("\n" if written else "") + rendered + "\n")
                else:
                    continue
                written += 1
        typer.echo(f"exported {written} events to {output}")
    finally:
        store.close()
