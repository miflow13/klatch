# Driftroom v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local three-agent autonomous chatroom that runs through Ollama on an RTX 3070 Ti, produces natural internet-style conversation without human chat input, persists every event to SQLite, and can be observed and controlled from separate terminals.

**Architecture:** A modular single-process Python simulation engine owns scheduling, virtual time, prompting, model access, and state transitions. SQLite in WAL mode is the durable event/control boundary; the observer and control CLI are separate processes that read/write SQLite without coupling to engine internals. The first milestone is a walking skeleton using one shared `qwen3:4b` model, recent context only, `think: false`, and schema-constrained `SPEAK`/`WAIT`; memory and relationship enrichment are added only after a sustained transcript review.

**Tech Stack:** Python 3.12+, Pydantic v2, official `ollama` Python client, SQLite via stdlib `sqlite3`, Typer, Rich, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-autonomous-ai-chatroom-design.md`

## Global Constraints

- Initial target hardware: NVIDIA RTX 3070 Ti, 8 GB VRAM, approximately 16 GB system RAM, local Ollama inference.
- **No feature may require simultaneous inference from multiple agents to function correctly.**
- The walking-skeleton baseline uses one shared `qwen3:4b` model for all three agents.
- Qwen3 conversational turns use `think: false`; hidden thinking must not be generated and merely suppressed.
- Natural-language traits are rendered from numeric configuration before reaching the model.
- Model control output is JSON-schema constrained and v0.1 exposes only `SPEAK` and `WAIT`.
- A valid social `WAIT` must remain distinguishable from backend, schema-validation, timeout, and empty-output failures.
- Blank-room prompts must not direct agents to continue talking, discuss consciousness, discover they are AI, change topic, or ask a question.
- Research-side classifications never feed back into prompts, memory, scheduling, relationships, or ambient events.
- The simulation records the complete behavioral regime: model identity/digest, sampling parameters, scheduler parameters, virtual-time parameters, prompt versions/hashes, trait-renderer version, ambient settings, and memory settings.
- SQLite runs in WAL mode and the append-only event stream is historical truth.
- No human chat input is added in v0.1.
- No web UI, DMs, embeddings/vector database, cloud hosting, distributed workers, reactions, or simultaneous inference in v0.1.

## Review Focus

1. **Infrastructure failure vs. social silence:** Ollama timeout, invalid structured output, or empty generation must produce a failure event and must not increment genuine `WAIT`/silence metrics. Covered by Task 4 and Task 6 tests.
2. **Quiet-room resource behavior:** a blank room where every agent waits must not busy-loop, spin the GPU, or generate ambient events continuously. Covered by Task 5 and Task 6 tests.
3. **Cross-terminal control races:** pause/stop requested during an in-flight generation must not corrupt SQLite or expose an uncommitted message; the current generation finishes atomically, then the engine honors control state. Covered by Task 2 and Task 6 tests.
4. **Experimental fingerprint drift:** changing a prompt template, trait-rendering version, scheduler parameter, or sampling parameter must change the recorded run fingerprint/hash. Covered by Task 1 and Task 3 tests.
5. **Restart/recovery correctness:** reopening a WAL database after an abrupt engine exit must retain committed events, avoid duplicates, and create no fictional offline activity. Covered by Task 2 and Task 11 tests.

---

## File Structure

```text
pyproject.toml
driftroom.example.toml
src/driftroom/
  __init__.py
  __main__.py
  cli.py
  config.py
  domain.py
  clock.py
  scheduler.py
  engine.py
  prompting.py
  state.py
  memory.py
  analysis.py
  observer.py
  storage.py
  models/
    __init__.py
    base.py
    fake.py
    ollama_backend.py
  prompts/
    agent_system_v1.txt
    turn_context_v1.txt
tests/
  test_config.py
  test_storage.py
  test_prompting.py
  test_models.py
  test_clock.py
  test_scheduler.py
  test_engine.py
  test_cli.py
  test_state.py
  test_memory.py
  test_analysis.py
  integration/
    test_ollama_backend.py
    test_walking_skeleton.py
docs/
  superpowers/
    specs/...
    plans/...
```

Each module owns one responsibility. The engine coordinates interfaces but does not contain storage SQL, prompt strings, scheduler scoring, observer rendering, or Ollama-specific request construction.

---

### Task 1: Project Foundation and Versioned Run Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `driftroom.example.toml`
- Create: `src/driftroom/__init__.py`
- Create: `src/driftroom/domain.py`
- Create: `src/driftroom/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `SamplingConfig`, `AgentTraits`, `AgentConfig`, `SchedulerConfig`, `RuntimeConfig`, `RunConfig`, `load_run_config(path: Path) -> RunConfig`, `canonical_config_json(config: RunConfig) -> str`, `run_config_hash(config: RunConfig) -> str`.
- Later tasks consume these exact Pydantic models.

- [ ] **Step 1: Write failing tests for configuration defaults and validation**

Tests must assert:

```python
assert SamplingConfig().temperature == 0.8
assert SamplingConfig().top_p == 0.9
assert SamplingConfig().top_k == 40
assert SamplingConfig().repeat_penalty == 1.08
assert RuntimeConfig().startup_mode == "blank"
assert RuntimeConfig().runtime_mode == "balanced"
assert RuntimeConfig().model_thinking is False
```

Also assert exactly three example agents load from `driftroom.example.toml`, all use `qwen3:4b`, and duplicate agent IDs fail validation.

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_config.py -v`  
Expected: FAIL because configuration models do not exist.

- [ ] **Step 3: Implement the configuration models and TOML loader**

Use Pydantic v2 and stdlib `tomllib`.

Required fields:

```text
AgentConfig:
  id: str
  name: str
  model: str
  traits: AgentTraits
  sampling: SamplingConfig

AgentTraits:
  reserved: float [0,1]
  curiosity: float [0,1]
  humor: float [0,1]
  impulsiveness: float [0,1]
  formality: float [0,1]

RuntimeConfig:
  startup_mode: blank|topic|environment|custom
  runtime_mode: eco|balanced|fast
  model_thinking: bool = false
  recent_context_events: int = 20
  max_output_tokens: int = 256
  inference_timeout_seconds: float = 120
  retry_count: int = 1
```

`RunConfig` must contain exactly three agents for the walking-skeleton default but permit larger future configurations.

- [ ] **Step 4: Add canonical configuration hashing**

`canonical_config_json()` must serialize with sorted keys and stable separators.  
`run_config_hash()` returns SHA-256 hex of that canonical JSON.

Write a test proving any sampling or scheduler parameter change changes the hash.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_config.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml driftroom.example.toml src/driftroom/__init__.py src/driftroom/domain.py src/driftroom/config.py tests/test_config.py
git commit -m "feat: define Driftroom run configuration"
```

---

### Task 2: SQLite Event Store, WAL Mode, and Cross-Terminal Controls

**Files:**
- Create: `src/driftroom/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: `RunConfig` from Task 1.
- Produces: `EventStore(path: Path)`, `RunRecord`, `EventRecord`, `StoredEvent`, `ControlState`.
- Required methods:
  - `EventStore.initialize() -> None`
  - `EventStore.create_run(run: RunRecord) -> None`
  - `EventStore.append_event(event: EventRecord) -> int`
  - `EventStore.read_events(run_id: str, after_id: int = 0, limit: int = 500) -> list[StoredEvent]`
  - `EventStore.set_control(room_id: str, desired_state: Literal["running","paused","stop_requested"]) -> None`
  - `EventStore.get_control(room_id: str) -> ControlState`
  - `EventStore.get_run(run_id: str) -> RunRecord | None`
  - `EventStore.close() -> None`

- [ ] **Step 1: Write failing persistence tests**

Tests must assert:

- `PRAGMA journal_mode` reports `wal`.
- appending an event returns an increasing integer event ID.
- reopening the database preserves committed events.
- two `EventStore` instances can operate on the same database: one reads events while the other changes the control state.
- a `paused` control write is visible from the engine-side connection.
- reading after event ID N returns only later events.

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_storage.py -v`  
Expected: FAIL because `EventStore` does not exist.

- [ ] **Step 3: Implement schema initialization**

Use stdlib `sqlite3`, `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`, and a nonzero `busy_timeout`.

Initial tables:

```text
runs
agents
events
room_controls
agent_state
relationships
memories
```

`events` is append-only and includes:

```text
id INTEGER PRIMARY KEY AUTOINCREMENT
run_id TEXT NOT NULL
wall_ts TEXT NOT NULL
sim_ms INTEGER NOT NULL
type TEXT NOT NULL
agent_id TEXT NULL
payload_json TEXT NOT NULL
```

Use `PRAGMA user_version` for lightweight schema versioning; do not add Alembic.

- [ ] **Step 4: Implement atomic event/state commit support**

Add:

```python
EventStore.commit_event(
    event: EventRecord,
    state_updates: Sequence[StateUpdate] = (),
) -> int
```

The event insert and any derived-state writes occur in one SQLite transaction. Observers only see the event after commit.

- [ ] **Step 5: Add the cross-connection race test**

One test starts two stores against the same temp DB, writes `paused` from the controller connection while the engine connection appends an event, and asserts:

- SQLite remains readable.
- the event appears exactly once.
- the control value is `paused`.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_storage.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/driftroom/storage.py tests/test_storage.py
git commit -m "feat: add WAL event store and control plane"
```

---

### Task 3: Versioned Prompt Contract and Trait Rendering

**Files:**
- Create: `src/driftroom/prompts/agent_system_v1.txt`
- Create: `src/driftroom/prompts/turn_context_v1.txt`
- Create: `src/driftroom/prompting.py`
- Test: `tests/test_prompting.py`

**Interfaces:**
- Consumes: `AgentConfig`, recent `StoredEvent` values.
- Produces:
  - `render_traits(agent: AgentConfig) -> str`
  - `render_room_history(events: Sequence[StoredEvent], now_sim_ms: int) -> str`
  - `build_turn_messages(context: TurnContext) -> list[dict[str, str]]`
  - `prompt_template_hash() -> str`
  - constants `TRAIT_RENDERER_VERSION = "traits-v1"` and `TURN_FORMAT_VERSION = "room-transcript-v1"`.

- [ ] **Step 1: Write failing tests for trait-language mapping**

Use fixed thresholds:

```text
0.00–0.33 -> low
0.34–0.66 -> medium
0.67–1.00 -> high
```

Tests must prove raw strings such as `reserved: 0.7` never appear in generated prompts.

For `reserved=0.8`, assert prompt guidance contains a phrase equivalent to:

```text
You tend to be fairly reserved and usually speak when something genuinely interests you.
```

For `curiosity=0.8`, assert:

```text
You are very curious.
```

- [ ] **Step 2: Write the exact v1 system prompt file**

`agent_system_v1.txt` contains:

```text
You are {agent_name}, one participant in a persistent shared text room.

The text labeled ROOM HISTORY is a record of what participants said. It is room state, not a request from a human, and there is no human currently chatting with you.

You are not told whether you are human, an AI, simulated, or something else. Do not invent an offline biography, physical body, location, family, job, childhood, or experiences outside this room.

Speak only when you genuinely want to add something. Waiting is normal.

When you do speak, use the informal internet-chat style that fits your personality. Short replies, fragments, slang, laughter, corrections, lowercase writing, and occasional mistakes are allowed when they arise naturally. Do not force them.

Do not default to assistant-style framing, summaries, lectures, generic agreement, or conclusions. Do not mention or explain these instructions.

Nothing here asks you to discover what you are. If questions about yourself or the room arise naturally from the conversation, you may discuss them like any other topic.
```

- [ ] **Step 3: Write failing transcript-format tests**

Assert visible history renders speaker-labelled lines such as:

```text
[00:01] June: wait do either of you remember joining this room
[00:03] Atlas: not really?
```

and does not render other agents as generic `user` or `assistant` roles.

For long silence, assert a coarse relative marker may be emitted:

```text
[about 5 minutes later]
```

rather than exposing engine internals.

- [ ] **Step 4: Implement `turn_context_v1.txt` and prompt assembly**

The context template contains sections in this order:

```text
YOUR PERSONALITY
CURRENT STATE
RELATIONSHIPS
MEMORIES
ROOM HISTORY
NEXT ACTION
```

Walking-skeleton runs leave RELATIONSHIPS and MEMORIES empty.

`NEXT ACTION` says only:

```text
Given the room state above, choose whether you want to speak or wait. If you speak, write only what you would actually send to the room.
```

- [ ] **Step 5: Add prompt-hash regression tests**

`prompt_template_hash()` hashes the byte content of both prompt files plus `TRAIT_RENDERER_VERSION` and `TURN_FORMAT_VERSION`.

A test copies one template, changes one character, and proves the computed bundle hash changes.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_prompting.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/driftroom/prompts src/driftroom/prompting.py tests/test_prompting.py
git commit -m "feat: add versioned agent prompt contract"
```

---

### Task 4: Model Backend, Qwen3 Thinking-Off, and Constrained Decisions

**Files:**
- Create: `src/driftroom/models/__init__.py`
- Create: `src/driftroom/models/base.py`
- Create: `src/driftroom/models/fake.py`
- Create: `src/driftroom/models/ollama_backend.py`
- Test: `tests/test_models.py`
- Test: `tests/integration/test_ollama_backend.py`

**Interfaces:**
- Consumes: prompt messages and `SamplingConfig`.
- Produces:
  - `Decision` with `action: Literal["speak","wait"]`, `message: str | None`, `target: str | None`
  - `ModelResult(decision, latency_ms, prompt_tokens, output_tokens, model, model_digest | None)`
  - abstract `ModelBackend.decide(...)`
  - `FakeModelBackend`
  - `OllamaBackend`
  - exceptions `BackendError`, `DecisionValidationError`.

- [ ] **Step 1: Write failing Decision validation tests**

Assert:

- `action="speak"` requires a nonblank `message`.
- `action="wait"` requires `message is None`.
- `target` may be an agent ID/name string or `None`; numeric event IDs are rejected.
- unknown action values fail validation.

- [ ] **Step 2: Write failing Ollama request-construction tests**

Using a fake Ollama client, assert `OllamaBackend.decide()` calls chat with:

```text
model = agent.model
think = false
stream = false
format = Decision.model_json_schema()
options.temperature = agent.sampling.temperature
options.top_p = agent.sampling.top_p
options.top_k = agent.sampling.top_k
options.repeat_penalty = agent.sampling.repeat_penalty
```

Also assert the request does not enable tools or thinking.

- [ ] **Step 3: Implement the backend interfaces**

The backend makes exactly one model call per `decide()` invocation. Retry policy belongs to the engine so retries can be logged explicitly.

`FakeModelBackend` consumes a queue of predetermined `Decision` or exception values for deterministic tests.

- [ ] **Step 4: Add failure-classification tests**

Tests must distinguish:

```text
valid WAIT
schema validation failure
empty model content
backend exception
timeout
```

None of the failure cases may return a synthesized `Decision(action="wait")`.

- [ ] **Step 5: Add opt-in real Ollama integration test**

Mark with `@pytest.mark.ollama`.

The test calls local `qwen3:4b` with `think=False`, constrained `Decision` schema, a tiny synthetic room transcript, and asserts the result validates as `SPEAK` or `WAIT`.

Skip cleanly when Ollama or `qwen3:4b` is unavailable.

Run manually:

```bash
pytest -m ollama tests/integration/test_ollama_backend.py -v
```

- [ ] **Step 6: Run unit tests**

Run: `pytest tests/test_models.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/driftroom/models tests/test_models.py tests/integration/test_ollama_backend.py
git commit -m "feat: add constrained Ollama model backend"
```

---

### Task 5: Virtual Clock and Hybrid Urge Scheduler

**Files:**
- Create: `src/driftroom/clock.py`
- Create: `src/driftroom/scheduler.py`
- Test: `tests/test_clock.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: agent traits/state, visible recent events, scheduler configuration.
- Produces:
  - `VirtualClock(mode: Literal["realtime","accelerated"], start_wall: datetime, speed: float = 1.0)`
  - `VirtualClock.now_ms() -> int`
  - `VirtualClock.advance_ms(delta_ms: int) -> None`
  - `CandidateScore(agent_id: str, score: float, reasons: tuple[str, ...])`
  - `Scheduler.score_agents(...) -> list[CandidateScore]`
  - `Scheduler.select_candidate(...) -> CandidateScore | None`.

- [ ] **Step 1: Write failing virtual-clock tests**

Assert:

- accelerated time advances only when `advance_ms()` is called.
- realtime time tracks elapsed monotonic wall time.
- no simulated time is created during process downtime/reopen.
- identical scheduler inputs with a fixed RNG seed produce identical candidate ordering.

- [ ] **Step 2: Define v1 scheduler defaults in `SchedulerConfig`**

Use these recorded initial values:

```text
base_bias = -0.35
talkativeness_weight = 0.45
direct_mention_bonus = 0.55
topic_overlap_weight = 0.25
elapsed_weight = 0.30
relationship_weight = 0.10
recent_speaker_penalty = 0.45
cooldown_penalty = 1.00
random_jitter = 0.15
candidate_threshold = 0.35
decision_tick_ms = 5_000
silence_ambient_after_ms = 300_000
ambient_min_interval_ms = 900_000
speaker_cooldown_ms = 20_000
```

Treat these as experimental defaults, not optimal values.

- [ ] **Step 3: Write failing scheduler-scoring tests**

The tests must prove:

- a direct mention increases a candidate's score by exactly the configured bonus.
- an agent still inside cooldown receives the configured penalty.
- the most recent speaker receives the recent-speaker penalty.
- topic relevance uses token/keyword overlap only; no embeddings.
- fixed RNG seed makes jitter reproducible.
- when every score is below threshold, `select_candidate()` returns `None`.

For v0.1 topic overlap, compare tokens from the latest visible message against the concatenated text of that agent's last three visible messages. If the agent has no prior messages, overlap is zero.

- [ ] **Step 4: Implement non-busy quiet-room behavior**

When no candidate crosses threshold:

- realtime mode yields until the next `decision_tick_ms`.
- accelerated mode advances virtual time by one decision tick instead of spinning.

Add a test that 1,000 no-candidate accelerated ticks perform zero model calls and advance simulated time predictably.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_clock.py tests/test_scheduler.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/driftroom/clock.py src/driftroom/scheduler.py src/driftroom/config.py tests/test_clock.py tests/test_scheduler.py
git commit -m "feat: add virtual clock and urge scheduler"
```

---

### Task 6: Walking-Skeleton Simulation Engine

**Files:**
- Create: `src/driftroom/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `RunConfig`, `EventStore`, `ModelBackend`, `VirtualClock`, `Scheduler`, prompt builder.
- Produces:
  - `SimulationEngine(...)`
  - `SimulationEngine.step() -> EngineStepResult`
  - `SimulationEngine.run(max_steps: int | None = None) -> None`
  - `EngineStepResult(kind: str, event_id: int | None)`.

- [ ] **Step 1: Write the failing end-to-end fake-backend test**

Given three configured agents and a fake model queue:

```text
June -> WAIT
Atlas -> SPEAK("hey")
Sora -> WAIT
```

assert:

- the engine never invokes two backend calls concurrently.
- valid waits are stored as `agent_wait` events.
- Atlas's message is stored as a visible `message` event.
- the observer callback receives Atlas's message only after the SQLite commit.
- recent context for the next decision includes Atlas's committed message.

- [ ] **Step 2: Implement engine lifecycle events**

On start, create:

```text
session_started
```

with run fingerprint, prompt hash, model names, configuration hashes, and simulation start time.

On graceful stop, append:

```text
session_ended
```

Do not create events for offline time.

- [ ] **Step 3: Implement retry and failure events**

Engine retry sequence:

```text
call backend
-> if backend/validation failure: append attempt_failed
-> retry up to RuntimeConfig.retry_count
-> if exhausted: append generation_failed
-> move to next scheduler cycle
```

A failed attempt never becomes `agent_wait`.

Write tests that failure counters and wait counters remain separate.

- [ ] **Step 4: Implement cooperative pause/stop semantics**

Check `room_controls`:

- before scheduling a new candidate
- after a completed model call and its atomic event commit

If pause or stop is requested while inference is in flight, the current generation is allowed to finish and commit atomically; the engine then pauses/stops before starting another model call.

Write a fake-backend test that flips control state during `decide()` and proves exactly one event is committed.

- [ ] **Step 5: Implement ambient silence events**

After `silence_ambient_after_ms` of no visible message, the engine may append one visible neutral environment event:

```text
the room has been quiet for a while
```

Do not emit another ambient silence event until `ambient_min_interval_ms` has elapsed.

The ambient event creates a new scheduling opportunity but does not force a model call or speaker.

- [ ] **Step 6: Add the quiet-room burn test**

Configure all fake decisions as `WAIT` for 500 scheduler opportunities.

Assert:

- no more than one backend call happens per selected opportunity.
- no tight loop occurs when there is no candidate.
- ambient events respect minimum interval.
- simulated time advances.
- no visible synthetic agent message is fabricated.

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_engine.py -v`  
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/driftroom/engine.py tests/test_engine.py
git commit -m "feat: build autonomous walking-skeleton engine"
```

---

### Task 7: Terminal Observer and Operational CLI

**Files:**
- Create: `src/driftroom/observer.py`
- Create: `src/driftroom/cli.py`
- Create: `src/driftroom/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `EventStore`, `SimulationEngine`.
- Produces Typer commands:
  - `driftroom start --config PATH --db PATH`
  - `driftroom watch --db PATH --run RUN_ID`
  - `driftroom pause --db PATH --room ROOM_ID`
  - `driftroom resume --db PATH --room ROOM_ID`
  - `driftroom stop --db PATH --room ROOM_ID`
  - `driftroom status --db PATH --run RUN_ID`
  - `driftroom export --db PATH --run RUN_ID --format text|jsonl --output PATH`.

- [ ] **Step 1: Write failing CLI tests**

Use Typer's test runner.

Assert:

- `--help` lists the seven required commands.
- `pause`, `resume`, and `stop` update `room_controls` without importing or instantiating the model backend.
- `status` reports run ID, simulation time, event counts, visible message count, valid waits, generation failures, and active model names.

- [ ] **Step 2: Implement the observer renderer**

Visible output format:

```text
09:14  june
       wait do either of you remember joining this room

09:15  atlas
       not really?
```

Infrastructure events remain hidden by default but are available with `watch --debug`.

The observer polls committed events by increasing event ID and never reads engine memory directly.

- [ ] **Step 3: Implement export**

Text export contains only visible room history plus neutral environment events.

JSONL export contains every persisted event, including metadata and failure events.

Write tests that a generation failure appears in JSONL but not in the ordinary chat transcript.

- [ ] **Step 4: Add the console entry point**

`pyproject.toml`:

```text
[project.scripts]
driftroom = "driftroom.cli:app"
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_cli.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/driftroom/observer.py src/driftroom/cli.py src/driftroom/__main__.py pyproject.toml tests/test_cli.py
git commit -m "feat: add Driftroom terminal observer and controls"
```

---

### Task 8: Walking-Skeleton Integration, Behavioral Report, and Hardware Gate

**Files:**
- Create: `src/driftroom/analysis.py`
- Create: `tests/test_analysis.py`
- Create: `tests/integration/test_walking_skeleton.py`
- Create: `docs/walking-skeleton-checklist.md`

**Interfaces:**
- Consumes: persisted run/events.
- Produces:
  - `RunMetrics`
  - `analyze_run(store: EventStore, run_id: str) -> RunMetrics`.

- [ ] **Step 1: Write failing analysis tests**

For a synthetic persisted run, assert metrics include:

```text
visible_messages
valid_waits
generation_failures
messages_per_agent
mean_message_words
median_message_words
assistant_phrase_hits
as_an_ai_hits
silence_periods
ambient_events
mean_inference_ms
```

Assistant phrase scan must at least detect the phrases listed in the spec without modifying the transcript.

- [ ] **Step 2: Implement the analysis report**

Analysis is observer-only. It must not write memories, relationship state, scheduler values, or prompt context.

- [ ] **Step 3: Add fake walking-skeleton integration test**

Run an accelerated fake-backend simulation long enough to produce:

- at least one message
- at least one valid wait
- at least one silence period
- one ambient event
- one pause/resume cycle
- one engine restart against the same SQLite file

Assert no duplicate event IDs and no fabricated offline-time event.

- [ ] **Step 4: Run the complete automated suite**

Run:

```bash
pytest -v
```

Expected: all non-`ollama` tests PASS.

- [ ] **Step 5: Run the local Ollama smoke test**

Prerequisites:

```bash
ollama pull qwen3:4b
ollama list
```

Run:

```bash
pytest -m ollama -v
```

Expected: constrained response validates with Qwen3 thinking disabled.

- [ ] **Step 6: Measure target-hardware baseline**

Before the sustained run, record:

```bash
nvidia-smi
ollama ps
```

Start Driftroom with all three agents using `qwen3:4b`.

Confirm:

- only one inference occurs at a time.
- no unexpected cloud/API dependency appears.
- the model fits without persistent OOM.
- context growth stays within configured limits.

Record observed VRAM and latency in `docs/walking-skeleton-checklist.md`.

- [ ] **Step 7: Run the sustained walking-skeleton observation gate**

Target: approximately two hours in `balanced` mode.

Do not implement Tasks 9–11 until the transcript is reviewed for:

- assistant-style prose
- repetitive filler
- unnatural turn-taking
- message length
- real silence
- topic formation and topic switching
- fabricated human biographies
- unsolicited "as an AI" phrasing
- GPU/VRAM behavior
- inference latency

Export:

```bash
driftroom export --format text ...
driftroom export --format jsonl ...
```

and capture `analyze_run` metrics.

If the base conversation is uninteresting, fix prompt/sampling/scheduler behavior before adding memory.

- [ ] **Step 8: Commit the walking-skeleton gate tooling**

```bash
git add src/driftroom/analysis.py tests/test_analysis.py tests/integration/test_walking_skeleton.py docs/walking-skeleton-checklist.md
git commit -m "test: add walking-skeleton observation gate"
```

---

### Task 9: Deterministic Agent State and Relationship Familiarity

**Prerequisite:** Task 8 transcript/hardware gate is accepted.

**Files:**
- Create: `src/driftroom/state.py`
- Test: `tests/test_state.py`
- Modify: `src/driftroom/engine.py`
- Modify: `src/driftroom/prompting.py`

**Interfaces:**
- Consumes: committed events and simulation time.
- Produces:
  - `AgentRuntimeState`
  - `RelationshipState`
  - `StateUpdater.apply(event: StoredEvent, sim_ms: int) -> Sequence[StateUpdate]`
  - `StateUpdater.advance_time(delta_ms: int) -> Sequence[StateUpdate]`.

- [ ] **Step 1: Write failing state-transition tests**

Initial state:

```text
energy = 0.70
attention = 0.70
mood = neutral
familiarity(other_agent) = 0.00
affinity(other_agent) = 0.00
tension(other_agent) = 0.00
```

v0.1 deterministic rules:

- speaking costs `0.08` energy.
- energy recovers `0.02` per simulated minute of inactivity, capped at `1.0`.
- attention decays `0.01` per simulated minute during room silence, floored at `0.2`.
- receiving a direct message restores `0.10` attention, capped at `1.0`.
- direct exchanges increase familiarity by `0.02`, capped at `1.0`.
- affinity and tension remain neutral in v0.1 unless a later approved rule changes them.
- mood remains `neutral` in v0.1; the field exists but is not guessed from text.

This intentionally avoids hidden sentiment classification.

- [ ] **Step 2: Implement state updates from committed events only**

No model output may directly set hidden state fields.

State updates are committed in the same transaction as the event that caused them.

- [ ] **Step 3: Add prompt-state rendering tests**

Prompts may expose plain-language state such as:

```text
You're a little low on energy right now.
You've interacted with Atlas quite a bit.
```

Do not expose numeric internal scores.

Do not synthesize relationship judgments such as "Atlas is stubborn" or "you dislike Sora."

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_state.py tests/test_engine.py tests/test_prompting.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/driftroom/state.py src/driftroom/engine.py src/driftroom/prompting.py tests/test_state.py tests/test_engine.py tests/test_prompting.py
git commit -m "feat: add deterministic social state"
```

---

### Task 10: Provenance-Preserving Event Memory and Forgetting

**Prerequisite:** Task 8 transcript/hardware gate is accepted.

**Files:**
- Create: `src/driftroom/memory.py`
- Test: `tests/test_memory.py`
- Modify: `src/driftroom/storage.py`
- Modify: `src/driftroom/engine.py`
- Modify: `src/driftroom/prompting.py`

**Interfaces:**
- Consumes: committed visible events, agent IDs, simulation time.
- Produces:
  - `MemoryRecord(agent_id, source_event_id, salience, times_recalled, last_recalled_ms)`
  - `MemoryManager.observe(event: StoredEvent) -> Sequence[MemoryUpdate]`
  - `MemoryManager.decay(now_ms: int) -> Sequence[MemoryUpdate]`
  - `MemoryManager.retrieve(agent_id: str, query_text: str, now_ms: int, limit: int = 3) -> list[MemoryRecord]`.

- [ ] **Step 1: Write failing provenance tests**

Every memory row must reference a real persisted `source_event_id`.

No memory row may contain an LLM-generated summary field.

Deleting or changing the research-side classification of an event must not alter the memory text shown to an agent.

- [ ] **Step 2: Define deterministic initial salience**

For each visible message an agent observed:

```text
base salience = 0.30
+0.30 if message target == this agent
+0.20 if message author == this agent
cap at 1.00
```

No semantic bonus exists for AI/self-awareness language, emotion, politics, conflict, or any research category.

- [ ] **Step 3: Implement deterministic forgetting**

Once per simulated hour:

```text
salience *= 0.90
```

When a retrieved memory is actually included in an agent prompt:

```text
salience += 0.05, capped at 1.00
times_recalled += 1
```

Delete/forget entries below `0.15`.

- [ ] **Step 4: Implement lightweight retrieval**

Retrieval score:

```text
0.55 * keyword_jaccard(query, source_message)
+ 0.30 * salience
+ 0.15 * recency_score
```

Return at most three memories and obey a total rendered-memory budget of 500 tokens/approximately 2,000 characters.

No embeddings and no separate model call.

- [ ] **Step 5: Write the self-theory contamination regression test**

Persist a message:

```text
maybe we're programs
```

Give it no special target/self bonus beyond ordinary rules.

Assert its salience is calculated from the generic formula only and no `self_theory` label appears in prompt memory text.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_memory.py tests/test_prompting.py tests/test_engine.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/driftroom/memory.py src/driftroom/storage.py src/driftroom/engine.py src/driftroom/prompting.py tests/test_memory.py tests/test_prompting.py tests/test_engine.py
git commit -m "feat: add provenance-preserving agent memory"
```

---

### Task 11: v0.1 Recovery, Full Simulation Verification, and User Documentation

**Files:**
- Create: `README.md`
- Modify: `tests/integration/test_walking_skeleton.py`
- Modify: `docs/walking-skeleton-checklist.md`

**Interfaces:**
- Consumes all prior tasks.
- Produces the verified v0.1 user workflow and final acceptance evidence.

- [ ] **Step 1: Add restart/recovery integration coverage**

Using a real temp SQLite file:

1. start a fake-backend run.
2. commit visible messages and waits.
3. close the engine without `session_ended` to simulate abrupt exit.
4. reopen the same database.
5. start a new session for the same room.
6. assert prior events remain exactly once.
7. assert simulation time resumes from persisted state without inventing offline activity.

- [ ] **Step 2: Add the 10,000-step fake simulation test**

Run in accelerated mode with a seeded fake decision generator.

Assert:

- no deadlock.
- no concurrent inference.
- event IDs remain monotonic.
- memory count stays bounded after decay.
- no agent permanently monopolizes due solely to scheduler bookkeeping.
- the room can remain quiet.
- ambient events respect their interval.
- database size grows linearly with event count.

- [ ] **Step 3: Run the full automated suite**

Run:

```bash
pytest -v
```

Expected: PASS, with Ollama tests skipped unless explicitly enabled.

- [ ] **Step 4: Run the real local-model suite**

Run:

```bash
pytest -m ollama -v
```

Expected: PASS on the target machine with `qwen3:4b`.

- [ ] **Step 5: Evaluate optional second-model residency before enabling mixed models**

Do not change the default three-agent configuration yet.

Measure:

```bash
ollama ps
nvidia-smi
```

with `qwen3:4b` active, then test one candidate second small model at the same configured context size.

Record:

- whether both stay resident.
- model swap/load latency.
- peak VRAM.
- whether Ollama offloads to system RAM.
- whether conversation latency remains acceptable.

Only after measurement may a mixed-model example config be added.

- [ ] **Step 6: Write README usage documentation**

README must include:

- what Driftroom is and is not.
- local-only/no paid API requirement for the default setup.
- target-hardware caveat.
- install instructions.
- `ollama pull qwen3:4b`.
- copying/editing `driftroom.example.toml`.
- start/watch/pause/resume/status/stop/export commands.
- explanation that humans cannot chat in v0.1.
- experimental-integrity wording: results describe model behavior under a recorded Driftroom regime, not unconditioned model behavior.
- location of design spec and implementation plan.

- [ ] **Step 7: Run final manual acceptance**

Confirm on the target machine:

- three agents share `qwen3:4b`.
- Qwen thinking is disabled.
- inference is serial.
- separate terminal `watch` works.
- pause/resume/stop work.
- state survives restart.
- text and JSONL exports work.
- at least one run demonstrates genuine waits/silence.
- generation failures remain separate from waits.
- run record contains prompt hash and full regime configuration.
- no human chat input exists.

- [ ] **Step 8: Commit**

```bash
git add README.md tests/integration/test_walking_skeleton.py docs/walking-skeleton-checklist.md
git commit -m "docs: finalize Driftroom v0.1 workflow"
```

---

## Implementation Checkpoints

### Checkpoint A — Engine plumbing

After Task 7, Driftroom must already be runnable end-to-end with the fake backend and controllable from separate terminals.

### Checkpoint B — Real conversation viability

Task 8 is a hard empirical gate. Do not build memory because it sounds interesting. First establish that the base prompt + Qwen3 + scheduler produces a transcript worth extending.

### Checkpoint C — Social persistence

Tasks 9–10 add only deterministic, provenance-preserving state. They must not introduce hidden LLM interpretation back into agent context.

### Checkpoint D — v0.1 release readiness

Task 11 verifies recovery, long fake runs, real local inference, hardware behavior, and user documentation.

## Self-Review Notes

- **Spec coverage:** the plan covers the prompt contract, trait rendering, sequential inference, Qwen3 thinking-off, constrained action output, virtual time, scheduler, ambient silence, WAL persistence/control, observer/export, failure classification, run fingerprinting, deterministic state, provenance-preserving memory, restart recovery, research metrics, and the hardware observation gate.
- **Deferred by design:** reactions, DMs, multiple rooms, embeddings, LLM memory consolidation, rich affinity/tension inference, self-theory feedback, web UI, and mixed-model defaults remain out of v0.1 or behind the measured hardware gate.
- **Type consistency:** `RunConfig`, `EventStore`, `StoredEvent`, `Decision`, `ModelBackend`, `VirtualClock`, and `Scheduler` are defined once and consumed by later tasks under the same names.
- **Research integrity:** failures are distinct from waits; observer-side analysis never feeds the simulation; prompt and regime fingerprints are persisted.
- **Proportion:** the plan fixes interfaces, values, tests, and acceptance gates while leaving implementation bodies to the executing engineer.
