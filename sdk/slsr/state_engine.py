"""StateEngine \u2014 tracks the agent's ``(Q, M, T)`` state vector.

Composite score::

    S = Q * M * T

All components live in ``[0.0, 1.0]`` so the product also lies in ``[0, 1]``.
Multiplication encodes *joint necessity*: an agent that is high-quality but
has no memory (M=0) or is a newborn (T~=0) is not yet ready.

Responsibilities:
    * Maintain the current (Q, M, T) and a rolling trajectory.
    * Provide derivatives dQ/dt, dM/dt, dT/dt for stagnation detection.
    * Emit ``StateAlarm`` flags when any component falls below a floor.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

from slsr.models import StateConfig, StateVector

logger = logging.getLogger(__name__)


@dataclass
class StateAlarm:
    """Emitted when a state component collapses below its configured floor."""

    kind: str  # "quality", "memory", or "time"
    value: float
    floor: float


def _clamp01(x: float) -> float:
    """Clamp a float into ``[0.0, 1.0]``."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class StateEngine:
    """Tracks and exposes an agent's `StateVector` over its lifetime."""

    def __init__(
        self,
        config: StateConfig | None = None,
        *,
        genesis_ts: float | None = None,
        time_fn=time.time,
    ) -> None:
        self._cfg = config or StateConfig()
        self._time = time_fn
        self._genesis_ts: float = genesis_ts if genesis_ts is not None else self._time()

        self._q: float = 0.0
        self._m: float = 0.0
        self._t: float = 0.0
        self._ticks: int = 0

        self._trajectory: deque[tuple[float, StateVector]] = deque(
            maxlen=self._cfg.trajectory_window
        )
        self._alarms: list[StateAlarm] = []
        # Record the initial vector
        self._trajectory.append((self._time(), self.snapshot()))

    # ------------------------------------------------------------------ #
    # Mutators
    # ------------------------------------------------------------------ #

    def update_quality(self, q: float) -> None:
        """Overwrite ``Q`` with a new fitness score (clamped to ``[0,1]``)."""
        self._q = _clamp01(q)
        self._check_alarms()
        self._record()

    def update_memory(self, m: float) -> None:
        """Overwrite ``M`` with a new memory-richness value."""
        self._m = _clamp01(m)
        self._check_alarms()
        self._record()

    def add_memory(self, delta: float) -> None:
        """Accumulate experience into ``M`` (saturates at 1.0)."""
        self.update_memory(self._m + delta)

    def tick_time(self) -> None:
        """Advance ``T`` based on wall-clock elapsed vs. ``maturation_horizon_sec``.

        ``T`` saturates at 1.0 once the agent has aged past the maturation
        horizon.
        """
        self._ticks += 1
        elapsed = max(0.0, self._time() - self._genesis_ts)
        horizon = max(self._cfg.maturation_horizon_sec, 1e-9)
        self._t = _clamp01(elapsed / horizon)
        self._check_alarms()
        self._record()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def snapshot(self) -> StateVector:
        """Return a frozen snapshot of the current ``(Q, M, T)`` vector."""
        return StateVector(Q=self._q, M=self._m, T=self._t)

    def trajectory(self, n: int = 100) -> list[StateVector]:
        """Return the most recent ``n`` state vectors (oldest first)."""
        n = max(1, min(n, len(self._trajectory)))
        return [v for _, v in list(self._trajectory)[-n:]]

    def derivatives(self) -> dict[str, float]:
        """Compute finite-difference derivatives of Q, M, T vs. wall-clock.

        Returns zeros when the trajectory has fewer than two points.
        """
        if len(self._trajectory) < 2:
            return {"dQ/dt": 0.0, "dM/dt": 0.0, "dT/dt": 0.0}
        (t0, v0), (t1, v1) = self._trajectory[0], self._trajectory[-1]
        dt = max(1e-9, t1 - t0)
        return {
            "dQ/dt": (v1.Q - v0.Q) / dt,
            "dM/dt": (v1.M - v0.M) / dt,
            "dT/dt": (v1.T - v0.T) / dt,
        }

    def alarms(self) -> list[StateAlarm]:
        """Return (and clear) the queue of alarms raised since last call."""
        out = list(self._alarms)
        self._alarms.clear()
        return out

    @property
    def genesis_ts(self) -> float:
        return self._genesis_ts

    @property
    def ticks(self) -> int:
        return self._ticks

    @property
    def config(self) -> StateConfig:
        return self._cfg

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _record(self) -> None:
        self._trajectory.append((self._time(), self.snapshot()))

    def _check_alarms(self) -> None:
        if 0.0 < self._q < self._cfg.quality_floor:
            self._alarms.append(StateAlarm("quality", self._q, self._cfg.quality_floor))
        if 0.0 < self._m < self._cfg.memory_floor:
            self._alarms.append(StateAlarm("memory", self._m, self._cfg.memory_floor))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        v = self.snapshot()
        return f"StateEngine(Q={v.Q:.3f}, M={v.M:.3f}, T={v.T:.3f}, S={v.S:.3f})"


__all__ = ["StateAlarm", "StateEngine"]
