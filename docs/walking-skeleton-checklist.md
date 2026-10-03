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

- The wheel-asset test is skipped unless `pip` and `setuptools` are importable in the venv
  (`pip install setuptools` to exercise it); it does not affect running Driftroom.
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
> - The envelope's JSON `action` literals are `say` and `quiet`, mapped back to SPEAK and
>   WAIT by the backend; the design's actions and events are unchanged. Under the old
>   `speak`/`wait` literals, qwen3:4b's reasoning interjection "Wait, ..." (it deliberates in
>   its answer even with thinking disabled) made the grammar pick `wait` with probability
>   ~1.0, even when the agent was asked a direct question (Ollama log-probs, 2026-10-03).
>   The literals and the reworded NEXT ACTION sentence are part of the recorded regime
>   (`driftroom-engine-0.1.3`); the gate runs `gate-2` and `gate-3` were recorded under the
>   old `speak`/`wait` literals.
> - `runtime.startup_mode` selects what a **fresh** start writes, right after
>   `session_started` and at the same simulated time; a restart writes nothing extra:
>   - `blank` (the spec default): nothing. The first prompt's ROOM HISTORY reads
>     `(no messages yet)`.
>   - `environment`: one environment line naming who is in the room, in config order, e.g.
>     `[00:00] (June, Milo and Ada are in the room)`.
>   - `topic`: one environment line `the room was opened with the topic: <topic>`, with
>     `runtime.topic` set (it is required in this mode and rejected in the others).
>   - `custom` is reserved and rejected in v0.1.
>
>   The startup line is descriptive, never an instruction, and is stored with payload
>   `{"text": ..., "kind": "startup"}`. It is visible room activity but not an ambient
>   event: it does not delay the first "quiet for a while" line and is not counted in
>   `ambient_events` by `analyze_run` (the `status` command's "ambient events" line still
>   counts every environment event). An empty transcript renders as `(no messages yet)`
>   (`room-transcript-v2`, `driftroom-engine-0.1.2`).
> - Recommendation: run this gate once in `blank` and once in `environment` mode and record
>   both verdicts. Both are legitimate regimes; the comparison between them is the finding
>   (on the first real runs, an empty `blank` room produced only WAITs).

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

Findings (carried from runs):

- say/quiet: neither model chose quiet in 36 constrained samples with conversation present; all
  silence in gate-6 came from the scheduler. (Probe 2026-10-03, §6.)

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

- `repeat_last_n`: 1024 (provisional; confirm with the offline probe). It is the window the
  repeat penalty looks back over (Ollama default 64 tokens covers only the NEXT ACTION text,
  so copying the previous message was never penalised). Recorded in each agent's `sampling`.
- Repetition rules (`driftroom-engine-0.1.5`, spec §36), over `repetition_window` (5) messages since
  the last environment event: (1) when the latest message has Jaccard token similarity
  >= `repetition_similarity_threshold` (0.6) to any of the preceding window messages, every agent's
  score loses `repetition_damping` (0.5) with reason `repetition`; (2) an agent whose own last message
  is that similar to any of the window messages before it loses `repetition_damping` with reason
  `self_repetition` (both can stack), but only while that message is itself among the last
  `repetition_window` messages, so a one-time echoer recovers after that many fresh messages by
  others. An environment event (ambient or startup line) restarts both rules, so a looping room
  goes quiet and recovers on the next ambient event. Window 1 is the pairwise ROOM rule (the
  copier still gets `self_repetition`). The topic-overlap term is capped (contributes nothing
  above that threshold; 0 disables it) and is unchanged.
- `analyze_run` reports `repeated_message_ratio` (now windowed: similarity to any of the previous
  `repetition_window` messages), `distinct_token_ratio`, and `same_speaker_repetition_ratio` (messages similar
  to their speaker's previous message, among messages with an earlier message by that speaker).
  Baselines:
  - gate-4 (engine 0.1.3): consecutive-pair `repeated_message_ratio` = 0.69 ; `distinct_token_ratio` = 0.09
  - gate-5 (first 13 min): windowed `repeated_message_ratio` = 7/22 ~ 0.32 ; self-copies 6/22 ;
    `distinct_token_ratio` = 0.28

- `repeat_penalty`: 1.15 (default; was 1.08). Reason: in the offline probe of 2026-10-03 (§6) it cut
  qwen3:4b similarity to the previous 5 messages from 0.65 to 0.42, and moved qwen3:8b-q4_K_M
  from 0.18 to 0.17. Recorded in each agent's `sampling` and in the config hash.
- qwen3:8b-q4_K_M at `num_ctx` 8192 splits 13%/87% CPU/GPU on the RTX 3070 Ti (6.6 GB per
  `ollama ps`); `max_context_tokens` 6144 is recommended for full GPU residency.

Note: the `room-transcript-v2` turn format is frozen from the first accepted gate run. Any
later change to how the room transcript is rendered must bump `TURN_FORMAT_VERSION`.

## 6. Gate decision

- [ ] **Accepted** — proceed to Tasks 9–11
- [ ] **Recalibrate first** — prompt/sampling/scheduler changes needed, then rerun this checklist

Reason: ______________________

Per-run decisions:

- gate-4 (engine 0.1.3): recalibrate first — copy-the-last-message loop from minute 13; R17
  applied; gate-5 pending
- gate-5 (first 13 min): recalibrate first — alternating two-thread loop from minute 7; R18 applied;
  gate-6 pending
- gate-6 (engine 0.1.5, qwen3:4b, environment mode, ~12.5 sim-min, stopped by the user when lines
  began repeating a template): recalibrate first — template repetition; the say/quiet decision was
  never "quiet" once conversation existed. Metrics: visible_messages 19, valid_waits 0,
  generation_failures 0, silence_periods 0, ambient_events 0, messages_per_agent june 2 / milo 10 /
  ada 7, mean_message_words 26.2 (median 22), decisions_per_sim_minute 1.52, wait_ratio 0.0,
  median_inference_ms 763 (mean 917), repeated_message_ratio 0.11, same_speaker_repetition_ratio 0.0,
  distinct_token_ratio 0.34. Applied: `repeat_penalty` default 1.15 (R19); gate-7 runs on
  qwen3:8b-q4_K_M via the user's env.toml (example model unchanged).

Probe 2026-10-03 (gate-6 context replayed, 20 visible events, 6 samples per cell,
grammar-constrained, think=false; similarity = mean over samples of the max Jaccard word-set
similarity to the previous 5 messages; RTX 3070 Ti):

| model | setting | quiet | similarity | mean s/call |
|-------|---------|-------|------------|-------------|
| qwen3:4b | default (rp 1.08, temp 0.8) | 0/6 | 0.65 | — |
| qwen3:4b | rp 1.15 | 0/6 | 0.42 | — |
| qwen3:4b | rp 1.15 + temp 1.0 | 0/6 | 0.49 | — |
| qwen3:8b-q4_K_M | default | 0/6 | 0.18 | 3.0 (incl. load) |
| qwen3:8b-q4_K_M | rp 1.15 | 0/6 | 0.17 | 1.0 |
| qwen3:8b-q4_K_M | rp 1.15 + temp 1.0 | 0/6 | 0.15 | 0.9 |

Date: ________ — Signed: ________
