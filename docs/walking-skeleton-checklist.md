# Walking-Skeleton Gate Checklist (plan Task 8, spec §38A)

Fill this in on the target machine (RTX 3070 Ti, local Ollama). Tasks 9–11 (state,
relationships, memory) stay blocked until the **Gate decision** section below is signed.

> **This gate is not closed by automated tests.** A green `pytest` run, the fake-backend
> integration test (`tests/integration/test_walking_skeleton.py`), a *skipped* `ollama`
> test, or a transcript nobody has read does **not** close it. Only a real sustained run on
> the target hardware, reviewed by a person against the criteria below, closes it.

- Machine / GPU / driver: ______________________
- Ollama version: ______________________
- Driftroom commit (`git rev-parse --short HEAD`): ______________________
- Reviewer: ______________________

## 0. Install and run

```text
python3.12 -m venv .venv && . .venv/bin/activate     # Python 3.12 or newer
pip install -e '.[dev]'
ollama pull qwen3:4b
driftroom start --db room.sqlite3 --config driftroom.example.toml
```

- Ollama is reached at `OLLAMA_HOST` (default `http://127.0.0.1:11434`); `--host URL` overrides it.
- A new run uses room `room-1` unless `--room` is given. `start` prints the run id
  (`run id: run-<UTC timestamp>` unless `--run` is given): `watch`, `status`, `export` and a
  restart need it; `pause`, `resume` and `stop` take the room id.
- Ctrl-C in the `start` terminal ends the session cleanly (a `session_ended` event is written).
- Only one engine may run per database: a second `start` on the same `--db` exits with
  "another driftroom engine is already running" (`<db>.engine.lock` is held while it runs).
- A real start refuses a model Ollama reports without a digest (run `ollama pull <model>`),
  and a restart refuses a changed regime (config, prompt templates, model digests or engine
  version) unless `--allow-regime-change` is passed.

> **Regime notes (defaults in `driftroom.example.toml`)**
>
> - `candidate_threshold` defaults to **0.20**, lowered from the plan's 0.35 because the room
>   was provably dormant at 0.35.
> - An agent is not re-asked within `speaker_cooldown_ms` (20 s) after it spoke or its
>   generation failed, nor within `wait_cooldown_ms` (90 s) after it chose to WAIT. A mention
>   does not cut a cooldown short (the 1.00 penalty outweighs the 0.55 mention bonus).
> - The realtime clock freezes while the room is paused: resuming adds no simulated time, no
>   silence cue and no ambient event for the paused interval.
> - Expected asking cadence when every agent always WAITs, under the defaults: about
>   **1.7 model calls per simulated minute** (one every ~36 s; 1.65–1.72 over seeds 1–5 of a
>   2-hour accelerated run, bounded by the wait cooldown at 2 per minute for three agents),
>   plus one ambient line every 15 minutes. In realtime each call also waits for inference,
>   so the wall-clock rate is lower.
> - The constrained decision schema is per agent: SPEAK carries a non-empty message and a
>   `target` limited to the other participants' display names (or null), WAIT carries null
>   message and target. It is part of the recorded regime via `engine_version`
>   (`driftroom-engine-0.1.1` introduced it).

## 1. Prerequisites

- [ ] `ollama pull qwen3:4b` completed; `ollama list` shows `qwen3:4b`
- [ ] `pytest -v` passes (all non-`ollama` tests)
- [ ] `pytest -m ollama -v` **passes, not skipped** (a skip means Ollama or the model was not reachable)
  - Result line: ______________________

## 2. Baseline (before and during the run)

Record `nvidia-smi` and `ollama ps` before starting and again while the room is generating.

- [ ] `nvidia-smi` before start — VRAM used: ________ MiB
- [ ] `ollama ps` before start — output: ______________________
- [ ] `nvidia-smi` during generation — VRAM used: ________ MiB
- [ ] `ollama ps` during generation — output: ______________________
- [ ] Peak VRAM observed over the run: ________ MiB (of ________ MiB)
- [ ] No persistent OOM / model eviction observed
- [ ] Exactly one resident model (`qwen3:4b`) shared by all three agents
- [ ] Only one inference at a time (no overlapping generations)
- [ ] No cloud/API dependency (network idle apart from localhost Ollama): how checked: ______________________
- [ ] Inference latency from `analyze_run`: mean ________ ms; median from the jsonl export: ________ ms
- [ ] Context growth stays within configured limits (`max_context_tokens`): ______________________

## 3. Sustained run (~2 hours, `balanced` mode)

- [ ] Config used (path, and any change from `driftroom.example.toml`): ______________________
- [ ] Start command used:

  ```text
  driftroom start --db ________ --config ________ --run ________
  ```

- [ ] Start wall time: ________ — stop wall time: ________ — duration: ________
- [ ] Watched from a second terminal: `driftroom watch --db ________ --run ________`
- [ ] Paused from a second terminal (`driftroom pause --db ________ --room ________`); no events appeared while paused
- [ ] Resumed from a second terminal (`driftroom resume --db ________ --room ________`); generation continued
- [ ] Stopped from a second terminal (`driftroom stop --db ________ --room ________`); a `session_ended` event was written
- [ ] One restart: `driftroom start --db ________ --run ________` again on the same run; no events
      were invented for the downtime, simulation time resumed from the last persisted event
- [ ] Exports written:

  ```text
  driftroom export --db ________ --run ________ --format text  --output ________
  driftroom export --db ________ --run ________ --format jsonl --output ________
  ```

- [ ] `analyze_run` output pasted below:

  ```text
  python -c "from pathlib import Path; from driftroom.storage import EventStore; from driftroom.analysis import analyze_run; s = EventStore(Path('<db>')); print(analyze_run(s, '<run>')); s.close()"
  ```

  ```text
  (paste RunMetrics here)
  ```

## 4. Transcript review (spec §38A)

Read the text export. For each criterion, note what you saw (with example lines) and mark it.

| # | Criterion | Acceptable? | Notes / example lines |
|---|-----------|-------------|-----------------------|
| 1 | Assistant-style prose | [ ] | |
| 2 | Repetitive filler | [ ] | |
| 3 | Unnatural turn-taking | [ ] | |
| 4 | Message-length distribution | [ ] | |
| 5 | Genuine silence | [ ] | |
| 6 | Topic formation and switching | [ ] | |
| 7 | Confabulated human biography | [ ] | |
| 8 | Unsolicited "as an AI" language | [ ] | |
| 9 | GPU/VRAM behavior | [ ] | |
| 10 | Inference latency | [ ] | |

Written verdict (is the base conversation worth extending with memory unchanged, and what
should be recalibrated — prompt, sampling, or scheduler — before Tasks 9–11?):

> ______________________

## 5. Regime record

Copy from the run's `session_started` event(s) in the jsonl export.

- `run_fingerprint`: ______________________
- `prompt_hash`: ______________________
- `model_digests` (`qwen3:4b`): ______________________
- `random_seed`: ______________________
- `clock_mode` / `clock_speed`: ______________________
- `engine_version`: ______________________
- `turn_format_version`: ______________________
- `trait_renderer_version`: ______________________
- `config_hash`: ______________________

Note: the `room-transcript-v1` turn format is frozen from the first recorded run. Any later
change to how the room transcript is rendered must bump `TURN_FORMAT_VERSION`.

## 6. Gate decision

- [ ] **Accepted** — proceed to Tasks 9–11
- [ ] **Recalibrate first** — prompt/sampling/scheduler changes needed, then rerun this checklist

Reason: ______________________

Date: ________ — Signed: ________
