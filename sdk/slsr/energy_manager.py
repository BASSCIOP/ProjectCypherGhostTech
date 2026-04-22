"""EnergyManager \u2014 resource accounting for the SLSR SDK.

Implements the physics-inspired budget model ``E = m * c^2`` where

* ``m`` (data mass) is the combined megabyte footprint of the current
  operation's inputs and working set.
* ``c`` (processing speed) is a dimensionless hardware-normalized factor
  in ``(0, 1]``; ``c^2`` expresses the quadratic cost of throughput.

Energy Units (EU) accrue on a continuous refill schedule capped by a hard
ceiling, and every expensive op must ``reserve`` -> ``commit``/``release``.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from slsr.models import EnergyBudget, EnergyConfig

logger = logging.getLogger(__name__)


class EnergyError(Exception):
    """Base class for energy-accounting errors."""


class OutOfEnergyError(EnergyError):
    """Raised when an op requests more EU than currently available."""


class EnergyCapExceededError(EnergyError):
    """Raised when a refill or deposit would exceed the hard ``max_cap``."""


@dataclass
class Reservation:
    """A pending EU reservation returned by ``EnergyManager.reserve``."""

    reservation_id: str
    amount_eu: float
    reason: str
    created_at: float = field(default_factory=time.time)
    committed: bool = False
    released: bool = False

    @property
    def is_open(self) -> bool:
        return not self.committed and not self.released


class EnergyManager:
    """Book-keeping ledger for the agent's compute budget.

    The manager is deliberately synchronous and lock-free \u2014 it is intended to
    be called from a single event loop per agent. For fleet-wide accounting,
    wrap instances with the GovernanceLayer's fleet caps.
    """

    def __init__(self, config: EnergyConfig | None = None, *, time_fn=time.monotonic) -> None:
        self._cfg = config or EnergyConfig()
        self._time = time_fn

        self._available: float = self._cfg.initial_eu
        self._reserved: float = 0.0
        self._consumed: float = 0.0
        self._last_refill_ts: float = self._time()
        self._reservations: dict[str, Reservation] = {}

        if self._available > self._cfg.max_cap:
            raise EnergyCapExceededError(
                f"initial_eu ({self._available}) > max_cap ({self._cfg.max_cap})"
            )

    # ------------------------------------------------------------------ #
    # Core formula
    # ------------------------------------------------------------------ #

    def required_energy(self, data_mass_mb: float, speed_factor: float | None = None) -> float:
        """Compute ``E = m * c^2`` for an operation.

        Args:
            data_mass_mb: Operation's ``m`` (megabytes of active data).
            speed_factor: Operation's ``c`` (0, 1]; falls back to the
                hardware-normalized reference value from config.

        Returns:
            Required EU (float, >= 0).
        """
        if data_mass_mb < 0:
            raise ValueError("data_mass_mb must be non-negative")
        if speed_factor is None:
            c_squared = self._cfg.reference_c_squared
        else:
            if speed_factor < 0:
                raise ValueError("speed_factor must be non-negative")
            c_squared = speed_factor * speed_factor
        return data_mass_mb * c_squared

    # ------------------------------------------------------------------ #
    # Refill / ledger
    # ------------------------------------------------------------------ #

    def _refill(self) -> None:
        """Accrue EU since the last book-keeping call, bounded by ``max_cap``."""
        now = self._time()
        elapsed = max(0.0, now - self._last_refill_ts)
        self._last_refill_ts = now
        gained = elapsed * self._cfg.refill_rate
        ceiling = self._cfg.max_cap - (self._available + self._reserved)
        self._available += max(0.0, min(gained, ceiling))

    def can_afford(self, amount_eu: float) -> bool:
        """Return True if ``amount_eu`` could be reserved right now."""
        self._refill()
        return self._available >= amount_eu

    def reserve(self, amount_eu: float, reason: str = "unspecified") -> Reservation:
        """Reserve ``amount_eu`` for an upcoming op.

        Raises `OutOfEnergyError` if insufficient EU is available even after
        the continuous refill step \u2014 callers should treat this as a soft
        failure and back off / degrade.
        """
        if amount_eu < 0:
            raise ValueError("amount_eu must be non-negative")
        self._refill()
        if amount_eu > self._available:
            raise OutOfEnergyError(
                f"requested {amount_eu:.3f} EU for '{reason}' but only "
                f"{self._available:.3f} EU available"
            )
        self._available -= amount_eu
        self._reserved += amount_eu
        res = Reservation(
            reservation_id=uuid.uuid4().hex,
            amount_eu=amount_eu,
            reason=reason,
        )
        self._reservations[res.reservation_id] = res
        logger.debug("reserve(%.3f EU, '%s') -> %s", amount_eu, reason, res.reservation_id)
        return res

    def commit(self, reservation: Reservation) -> None:
        """Consume a reservation (EU is permanently spent)."""
        self._assert_open(reservation)
        self._reserved -= reservation.amount_eu
        self._consumed += reservation.amount_eu
        reservation.committed = True
        logger.debug("commit(%s, %.3f EU)", reservation.reservation_id, reservation.amount_eu)

    def release(self, reservation: Reservation) -> None:
        """Release a reservation back to the available pool (EU NOT consumed)."""
        self._assert_open(reservation)
        self._reserved -= reservation.amount_eu
        self._available += reservation.amount_eu
        reservation.released = True
        logger.debug("release(%s, %.3f EU)", reservation.reservation_id, reservation.amount_eu)

    def _assert_open(self, reservation: Reservation) -> None:
        if reservation.reservation_id not in self._reservations:
            raise EnergyError(f"unknown reservation {reservation.reservation_id}")
        if not reservation.is_open:
            raise EnergyError(f"reservation {reservation.reservation_id} already closed")

    # ------------------------------------------------------------------ #
    # Replication-specific helpers
    # ------------------------------------------------------------------ #

    def check_replication_affordable(self) -> bool:
        """True iff the floor-guarded replication cost can currently be reserved."""
        self._refill()
        floor = self._cfg.replication_energy_floor
        cost = self._cfg.replication_cost_eu
        return (self._available >= cost) and (self._available >= floor)

    def reserve_replication(self) -> Reservation:
        """Reserve EU for a replication attempt, enforcing the floor."""
        if not self.check_replication_affordable():
            raise OutOfEnergyError(
                "insufficient EU for replication "
                f"(available={self._available:.1f}, "
                f"cost={self._cfg.replication_cost_eu}, "
                f"floor={self._cfg.replication_energy_floor})"
            )
        return self.reserve(self._cfg.replication_cost_eu, reason="replication")

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> float:
        self._refill()
        return self._available

    @property
    def reserved(self) -> float:
        return self._reserved

    @property
    def consumed(self) -> float:
        return self._consumed

    @property
    def config(self) -> EnergyConfig:
        return self._cfg

    def snapshot(self) -> EnergyBudget:
        """Return a pydantic snapshot of the ledger state."""
        self._refill()
        return EnergyBudget(
            available=self._available,
            reserved=self._reserved,
            consumed=self._consumed,
            max_cap=self._cfg.max_cap,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return (
            f"EnergyManager(available={self._available:.2f}, "
            f"reserved={self._reserved:.2f}, consumed={self._consumed:.2f}, "
            f"cap={self._cfg.max_cap:.2f})"
        )


__all__ = [
    "EnergyCapExceededError",
    "EnergyError",
    "EnergyManager",
    "OutOfEnergyError",
    "Reservation",
]
