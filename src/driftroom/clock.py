"""An explicit simulation clock anchored to a run, not to process uptime."""

from datetime import datetime
import math
import time
from typing import Callable, Literal


class VirtualClock:
    def __init__(
        self,
        mode: Literal["realtime", "accelerated"],
        start_wall: datetime,
        speed: float = 1.0,
        *,
        initial_ms: int = 0,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in ("realtime", "accelerated"):
            raise ValueError("unknown virtual clock mode")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("clock speed must be positive and finite")
        if initial_ms < 0:
            raise ValueError("initial simulation time cannot be negative")
        self.mode = mode
        self.start_wall = start_wall
        self.speed = speed
        self._monotonic_fn = monotonic_fn
        self._started_monotonic = monotonic_fn()
        self._sim_ms = initial_ms
        self._paused_monotonic: float | None = None

    def now_ms(self) -> int:
        if self.mode == "realtime":
            current = self._monotonic_fn() if self._paused_monotonic is None else self._paused_monotonic
            return self._sim_ms + int((current - self._started_monotonic) * self.speed * 1_000)
        return self._sim_ms

    def pause(self) -> None:
        """Freeze a realtime clock: no wall time counts until ``resume()``.

        Idempotent. An accelerated clock only moves on ``advance_ms`` and is unaffected.
        """
        if self.mode == "realtime" and self._paused_monotonic is None:
            self._paused_monotonic = self._monotonic_fn()

    def resume(self) -> None:
        """Continue a paused realtime clock from where it froze. Idempotent."""
        if self.mode == "realtime" and self._paused_monotonic is not None:
            self._started_monotonic += self._monotonic_fn() - self._paused_monotonic
            self._paused_monotonic = None

    def advance_ms(self, delta_ms: int) -> None:
        if self.mode != "accelerated":
            raise ValueError("advance_ms is only available in accelerated mode")
        if delta_ms < 0:
            raise ValueError("simulation time cannot move backwards")
        self._sim_ms += delta_ms
