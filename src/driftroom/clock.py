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

    def now_ms(self) -> int:
        if self.mode == "realtime":
            return self._sim_ms + int(
                (self._monotonic_fn() - self._started_monotonic) * self.speed * 1_000
            )
        return self._sim_ms

    def advance_ms(self, delta_ms: int) -> None:
        if self.mode != "accelerated":
            raise ValueError("advance_ms is only available in accelerated mode")
        if delta_ms < 0:
            raise ValueError("simulation time cannot move backwards")
        self._sim_ms += delta_ms
