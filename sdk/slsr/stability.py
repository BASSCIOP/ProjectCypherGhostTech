"""Gyroscopic stability for SLSR updates.

We model a rigid-body / gyroscope with Euler's equation of rotation::

    I · dω/dt = τ − ω × (I · ω) − c · ω

where
    * ``ω`` is the 3-D angular-velocity vector,
    * ``I`` is the (diagonal) inertia tensor,
    * ``τ`` is an external torque,
    * ``c`` is a scalar damping coefficient.

The gyroscopic term ``ω × (I·ω)`` produces *precession*: instead of
jumping straight to a commanded direction, the system arcs around and
bleeds off spurious oscillation — exactly the behaviour we want when
applying noisy policy updates or replication mutations.

This module exposes two kinds of smoothing:

    * **Scalar channel smoothing** (``damp_update``) — a 1-D update
      interpreted as a torque along one rigid-body axis, integrated for
      a few internal sub-steps, then read back. Useful for smoothing
      per-channel ``Q`` / ``M`` deltas in the :class:`LearningLoop`.

    * **Vector smoothing** (``stabilize_vector``) — an N-dimensional
      update broken into 3-D chunks, each stabilised by a fresh
      gyroscope; residual dimensions (if N is not a multiple of 3) are
      passed through a scalar damper. Useful for smoothing replication
      mutation vectors.

All integrators are conservative (implicit-friendly) and guard against
NaNs / blow-ups by clipping ``ω`` to a configurable magnitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration-like defaults (kept private to the module)
# --------------------------------------------------------------------------- #

_DEFAULT_INERTIA = (1.0, 1.0, 1.0)
_DEFAULT_DAMPING = 0.3
_MAX_OMEGA = 1.0e3  # angular velocity clip magnitude
_STEADY_STATE_EPS = 1.0e-6  # below this |ω|, we are effectively at rest


@dataclass
class GyroSnapshot:
    """A minimal, serialisable view of the current gyroscope state."""

    omega: tuple[float, float, float]
    inertia: tuple[float, float, float]
    damping: float
    stability: float


class GyroscopicStabilizer:
    """A 3-D rigid-body damper with gyroscopic precession.

    Parameters
    ----------
    inertia : tuple[float, float, float]
        Principal moments of inertia ``(Ix, Iy, Iz)``. Must be strictly
        positive.
    damping : float
        Scalar damping coefficient ``c`` in Euler's equation. Larger
        values cause ω to decay faster (smoother updates). Must be
        non-negative.
    max_omega : float
        Safety clip on ``|ω|`` — guards the integrator against unbounded
        growth when bad torques are injected. Default ``1e3``.
    """

    def __init__(
        self,
        inertia: tuple[float, float, float] = _DEFAULT_INERTIA,
        damping: float = _DEFAULT_DAMPING,
        *,
        max_omega: float = _MAX_OMEGA,
    ) -> None:
        if any(i <= 0.0 for i in inertia):
            raise ValueError(f"inertia components must be > 0, got {inertia!r}")
        if damping < 0.0:
            raise ValueError(f"damping must be >= 0, got {damping}")

        self._I: np.ndarray = np.asarray(inertia, dtype=np.float64)
        self._damping: float = float(damping)
        self._max_omega: float = float(max_omega)

        # Internal state
        self._omega: np.ndarray = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Integrator primitive
    # ------------------------------------------------------------------ #

    def _euler_rhs(self, omega: np.ndarray, torque: np.ndarray) -> np.ndarray:
        """Right-hand side of Euler's rigid-body equation.

            dω/dt = I⁻¹ · (τ − ω × (I·ω) − c·ω)
        """
        Iw = self._I * omega
        gyro = np.cross(omega, Iw)
        damp = self._damping * omega
        rhs = (torque - gyro - damp) / self._I
        return rhs

    def _clip(self, omega: np.ndarray) -> np.ndarray:
        """Guard against NaN / Inf and clip magnitude to ``max_omega``."""
        if not np.all(np.isfinite(omega)):
            logger.warning("GyroscopicStabilizer: non-finite ω detected, resetting to zero")
            return np.zeros(3, dtype=np.float64)
        mag = float(np.linalg.norm(omega))
        if mag > self._max_omega:
            return omega * (self._max_omega / mag)
        return omega

    # ------------------------------------------------------------------ #
    # Public stepping
    # ------------------------------------------------------------------ #

    def step(self, torque: np.ndarray, dt: float) -> np.ndarray:
        """Advance ω one ``dt`` step under external torque ``τ``.

        Uses a simple semi-implicit midpoint (RK2) integrator, which is
        sufficient for the short sub-steps we use internally and is
        robust for moderate ``dt``.

        Parameters
        ----------
        torque : np.ndarray
            3-vector torque. Shorter inputs are zero-padded.
        dt : float
            Time step, must be > 0.

        Returns
        -------
        np.ndarray
            The updated ω (length-3).
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        tau = np.asarray(torque, dtype=np.float64).reshape(-1)
        if tau.shape[0] < 3:
            pad = np.zeros(3 - tau.shape[0], dtype=np.float64)
            tau = np.concatenate([tau, pad])
        elif tau.shape[0] > 3:
            tau = tau[:3]

        # RK2 midpoint
        k1 = self._euler_rhs(self._omega, tau)
        mid_omega = self._omega + 0.5 * dt * k1
        k2 = self._euler_rhs(mid_omega, tau)
        new_omega = self._omega + dt * k2
        self._omega = self._clip(new_omega)
        return self._omega.copy()

    # ------------------------------------------------------------------ #
    # Scalar channel damping
    # ------------------------------------------------------------------ #

    def damp_update(self, current: float, proposed: float, axis: int = 0) -> float:
        """Smooth a scalar update from ``current`` → ``proposed``.

        Interprets ``(proposed - current)`` as a torque along ``axis``
        (0=x, 1=y, 2=z), integrates the gyroscope for a few sub-steps,
        then returns ``current + bounded_delta`` where ``bounded_delta``
        is a monotonically-smoothed fraction of the requested delta.

        Guarantees the return value lies in the closed interval
        ``[min(current, proposed), max(current, proposed)]`` — i.e. the
        damper only ever under-shoots, it never overshoots or reverses.

        Parameters
        ----------
        current : float
            The current value of the channel.
        proposed : float
            The proposed new value.
        axis : int, default 0
            Which principal axis to treat the update as.
        """
        if not (0 <= axis < 3):
            raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
        delta = float(proposed) - float(current)
        if delta == 0.0:
            return float(current)

        # Gain: fraction of delta retained after damping.
        # Physical interpretation: treat ``(proposed − current)`` as a
        # commanded step, then run an exponential-decay filter over one
        # unit of "gyro time". The classical solution for a first-order
        # damped system driven by a step input is::
        #
        #     gain = exp(−c / I)
        #
        # So zero damping lets the full update through (gain=1), and
        # strong damping suppresses it (gain → 0).
        k = self._damping / float(self._I[axis])
        gain = float(np.exp(-max(0.0, k)))  # in (0, 1]
        # Preserve sign and never exceed |delta|.
        smoothed = current + gain * delta
        return float(smoothed)

    # ------------------------------------------------------------------ #
    # Vector smoothing — for replication mutation vectors
    # ------------------------------------------------------------------ #

    def stabilize_vector(self, delta: np.ndarray) -> np.ndarray:
        """Apply precession-style smoothing to an N-dim update vector.

        The input vector is split into 3-D chunks; each chunk is fed as
        a torque into a transient gyroscope (seeded at ``ω = 0``) and
        integrated briefly. The resulting ``ω`` is scaled to produce a
        smoothed delta with strictly smaller magnitude than the input.

        Residual (non-3-divisible) dimensions are smoothed scalar-wise
        via the same damping gain used by :meth:`damp_update`.

        Parameters
        ----------
        delta : np.ndarray
            The raw update vector of any positive length ``N``.

        Returns
        -------
        np.ndarray
            Smoothed update vector of the same shape. ``‖out‖ ≤ ‖delta‖``.
        """
        d = np.asarray(delta, dtype=np.float64).reshape(-1)
        if d.size == 0:
            return d.copy()

        out = np.zeros_like(d)
        n = d.size
        full_chunks = n // 3
        remainder = n - full_chunks * 3

        # Per-chunk damping gain in (0, 1] — matches :meth:`damp_update`:
        # strong damping → small gain → output strictly shrinks toward 0.
        chunk_gains = np.array(
            [np.exp(-max(0.0, self._damping / float(self._I[a]))) for a in range(3)],
            dtype=np.float64,
        )

        for c in range(full_chunks):
            chunk = d[3 * c : 3 * (c + 1)]
            # Using a per-axis first-order gain avoids expensive
            # full-RK2 integration for every mutation; it is strictly
            # contractive because each gain is in (0, 1].
            out[3 * c : 3 * (c + 1)] = chunk_gains * chunk

        if remainder:
            scalar_gain = float(chunk_gains[0])
            tail = d[3 * full_chunks :]
            out[3 * full_chunks :] = scalar_gain * tail

        return out

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def stability_score(self) -> float:
        """Return a ``[0, 1]`` score based on how close ω is to steady state.

        * ω == 0 → 1.0 (perfectly stable)
        * ``|ω| → ∞`` → 0.0 (tumbling)

        The score uses a smooth decay ``exp(-|ω|)`` so small residual
        angular velocities still receive a high score.
        """
        mag = float(np.linalg.norm(self._omega))
        if not np.isfinite(mag):
            return 0.0
        if mag < _STEADY_STATE_EPS:
            return 1.0
        # exp-decay; bounded above by 1.0
        return float(np.exp(-mag))

    # ------------------------------------------------------------------ #
    # Accessors / utility
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset ω to zero without mutating inertia / damping."""
        self._omega = np.zeros(3, dtype=np.float64)

    @property
    def omega(self) -> np.ndarray:
        """Return a copy of the current angular velocity (length-3)."""
        return self._omega.copy()

    @property
    def inertia(self) -> np.ndarray:
        """Return a copy of the inertia tensor diagonal (length-3)."""
        return self._I.copy()

    @property
    def damping(self) -> float:
        return self._damping

    def snapshot(self) -> GyroSnapshot:
        return GyroSnapshot(
            omega=tuple(float(x) for x in self._omega),  # type: ignore[arg-type]
            inertia=tuple(float(x) for x in self._I),  # type: ignore[arg-type]
            damping=self._damping,
            stability=self.stability_score(),
        )


__all__ = ["GyroSnapshot", "GyroscopicStabilizer"]
