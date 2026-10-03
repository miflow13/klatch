"""Virtual time is explicit and independent of process downtime."""

from datetime import datetime, timezone

import pytest

from driftroom.clock import VirtualClock


START = datetime(2026, 10, 1, tzinfo=timezone.utc)


def test_accelerated_time_changes_only_on_advance() -> None:
    monotonic = [100.0]
    clock = VirtualClock("accelerated", START, monotonic_fn=lambda: monotonic[0])

    assert clock.now_ms() == 0
    monotonic[0] += 100.0
    assert clock.now_ms() == 0
    clock.advance_ms(5_000)
    assert clock.now_ms() == 5_000


def test_realtime_tracks_monotonic_elapsed_time() -> None:
    monotonic = [200.0]
    clock = VirtualClock("realtime", START, monotonic_fn=lambda: monotonic[0])

    monotonic[0] = 202.5
    assert clock.now_ms() == 2_500


def test_realtime_clock_cannot_be_advanced_manually() -> None:
    monotonic = [200.0]
    clock = VirtualClock(
        "realtime", START, initial_ms=1_200, monotonic_fn=lambda: monotonic[0]
    )

    with pytest.raises(ValueError, match="accelerated"):
        clock.advance_ms(5_000)
    monotonic[0] = 202.0
    assert clock.now_ms() == 3_200


def test_reopen_restores_simulated_time_without_offline_catchup() -> None:
    old_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    clock = VirtualClock(
        "realtime", old_start, initial_ms=12_000, monotonic_fn=lambda: 900.0
    )

    assert clock.now_ms() == 12_000


def test_clock_rejects_backwards_time_and_invalid_speed() -> None:
    with pytest.raises(ValueError):
        VirtualClock("realtime", START, speed=0)
    clock = VirtualClock("accelerated", START)
    with pytest.raises(ValueError):
        clock.advance_ms(-1)


def test_paused_realtime_clock_freezes_and_resumes_without_the_paused_time() -> None:
    monotonic = [100.0]
    clock = VirtualClock("realtime", START, initial_ms=1_000, monotonic_fn=lambda: monotonic[0])
    monotonic[0] = 102.0
    clock.pause()
    clock.pause()  # idempotent
    assert clock.now_ms() == 3_000

    monotonic[0] += 600.0
    assert clock.now_ms() == 3_000
    clock.resume()
    clock.resume()  # idempotent
    assert clock.now_ms() == 3_000

    monotonic[0] += 1.5
    assert clock.now_ms() == 4_500


def test_pause_and_resume_are_noops_for_an_accelerated_clock() -> None:
    clock = VirtualClock("accelerated", START, monotonic_fn=lambda: 0.0)
    clock.resume()
    clock.pause()
    clock.advance_ms(5_000)
    clock.resume()
    assert clock.now_ms() == 5_000
