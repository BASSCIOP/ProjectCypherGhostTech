"""ConvergenceOracle \u2014 readiness / convergence gating.

Canonical threshold (Invariant I9 \u2014 cannot be lowered)::

    CONV_THRESHOLD = |pi * cos(sqrt(e))| normalized to [0, 1]
                    = (pi * cos(sqrt(e)) + pi) / (2 * pi)
                    ~ 0.5792

Three named gates are exposed:
    * ``is_ready_to_act``          \u2014 routine action emission
    * ``is_ready_to_learn_commit`` \u2014 durable policy-state write
    * ``is_ready_to_replicate``    \u2014 strictest; uses
                                    ``max(CONV_THRESHOLD, replication_floor)``
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

from slsr import CONV_THRESHOLD
from slsr.models import ConvergenceDecision, OracleConfig, StateVector
from slsr.protocols import FitnessFunction
from slsr.state_engine import StateEngine

logger = logging.getLogger(__name__)


# A callback returning the agent's current 8-dimensional embedding, or
# ``None`` if no embedding is available yet. Kept as a loose ``Callable``
# rather than a ``Protocol`` to avoid depending on numpy at import time.
StateVectorProvider = Callable[[], Any]


class ConfigError(ValueError):
    """Raised when an oracle is misconfigured (e.g. threshold override)."""


class DefaultFitness:
    """Default fitness function: returns the raw composite score ``S = Q*M*T``."""

    def score(  # noqa: D401
        self, state: StateVector, context: dict[str, Any] | None = None
    ) -> float:
        return state.S


class ConvergenceOracle:
    """Readiness gate against the canonical threshold.

    The canonical threshold is imported from `slsr.CONV_THRESHOLD` and is
    **never** configurable. Operators may only layer *additional* higher
    floors on top (e.g. the replication floor).
    """

    CANONICAL_THRESHOLD: float = CONV_THRESHOLD

    def __init__(
        self,
        state_engine: StateEngine,
        fitness_fn: FitnessFunction | None = None,
        config: OracleConfig | None = None,
        *,
        rng: random.Random | None = None,
        use_e8_geometry: bool = False,
        e8_symmetry_min: float = 0.4,
        state_vector_provider: StateVectorProvider | None = None,
    ) -> None:
        self._engine = state_engine
        self._fitness: FitnessFunction = fitness_fn or DefaultFitness()
        self._cfg = config or OracleConfig()
        self._rng = rng or random.Random()

        # Invariant I9 \u2014 replication floor must sit ABOVE canonical threshold.
        if self._cfg.replication_floor < self.CANONICAL_THRESHOLD:
            raise ConfigError(
                f"replication_floor ({self._cfg.replication_floor}) must be >= canonical "
                f"threshold ({self.CANONICAL_THRESHOLD:.4f}). The canonical threshold is a "
                "hard-coded invariant and cannot be lowered."
            )

        # --- Optional E8 geometric gate -------------------------------- #
        self._use_e8: bool = bool(use_e8_geometry)
        if not (0.0 <= float(e8_symmetry_min) <= 1.0):
            raise ConfigError(
                f"e8_symmetry_min must be in [0, 1], got {e8_symmetry_min!r}"
            )
        self._e8_min: float = float(e8_symmetry_min)
        self._state_vector_provider: StateVectorProvider | None = state_vector_provider
        self._e8_system = None  # lazy init on first use
        self._last_e8_score: float | None = None

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def current_score(self, context: dict[str, Any] | None = None) -> float:
        """Compute the fitness score for the current agent state."""
        vec = self._engine.snapshot()
        score = float(self._fitness.score(vec, context or {}))
        return max(0.0, min(1.0, score))

    def _apply_stochastic_jitter(self, score: float) -> float:
        if not self._cfg.stochastic_gate or self._cfg.jitter_sigma <= 0:
            return score
        eps = self._rng.gauss(0.0, self._cfg.jitter_sigma)
        return max(0.0, min(1.0, score + eps))

    # --- E8 geometric gate --------------------------------------------- #

    def _e8_symmetry_score(self) -> float | None:
        """Return the E8 symmetry score for the current embedding, or None.

        Returns ``None`` when the E8 gate is disabled or when no
        ``state_vector_provider`` was supplied (graceful fall-back to the
        scalar-only gate).
        """
        if not self._use_e8 or self._state_vector_provider is None:
            return None
        try:
            embedding = self._state_vector_provider()
        except Exception as exc:  # pragma: no cover - user callback
            logger.warning("state_vector_provider raised: %s \u2014 skipping E8 check", exc)
            return None
        if embedding is None:
            return None

        # Lazy-initialise the E8 system on first use so that users who
        # never enable the flag do not pay the (small) construction cost.
        if self._e8_system is None:
            from slsr.geometry import E8RootSystem  # local import: optional dep

            self._e8_system = E8RootSystem()

        import numpy as np  # local import for the same reason

        arr = np.asarray(embedding, dtype=np.float64)
        # symmetry_score handles both (8,) and (N, 8)
        score = float(self._e8_system.symmetry_score(arr))
        self._last_e8_score = score
        return score

    def _gate(self, threshold: float, purpose: str) -> ConvergenceDecision:
        raw = self.current_score()
        jittered = self._apply_stochastic_jitter(raw)
        margin = jittered - threshold
        scalar_ready = margin >= 0.0

        # Extra geometric gate (opt-in)
        e8_score = self._e8_symmetry_score()
        e8_rationale = ""
        if e8_score is None:
            geometric_ready = True
        else:
            geometric_ready = e8_score >= self._e8_min
            e8_rationale = (
                f" e8={e8_score:.4f} (min={self._e8_min:.2f}, "
                f"{'pass' if geometric_ready else 'fail'})"
            )

        is_ready = scalar_ready and geometric_ready
        rationale = (
            f"[{purpose}] score={jittered:.4f} threshold={threshold:.4f} "
            f"margin={margin:+.4f} (raw={raw:.4f}){e8_rationale}"
        )
        logger.debug(rationale)
        return ConvergenceDecision(
            is_ready=is_ready,
            score=jittered,
            threshold=threshold,
            margin=margin,
            rationale=rationale,
        )

    # ------------------------------------------------------------------ #
    # Public gates
    # ------------------------------------------------------------------ #

    def is_ready_to_act(self) -> ConvergenceDecision:
        """Gate for routine action emission \u2014 canonical threshold only."""
        return self._gate(self.CANONICAL_THRESHOLD, "act")

    def is_ready_to_learn_commit(self) -> ConvergenceDecision:
        """Gate for persisting a learned update to durable state."""
        return self._gate(self.CANONICAL_THRESHOLD, "learn_commit")

    def is_ready_to_replicate(self) -> ConvergenceDecision:
        """Strictest gate: ``max(CONV_THRESHOLD, replication_floor)``."""
        threshold = max(self.CANONICAL_THRESHOLD, self._cfg.replication_floor)
        return self._gate(threshold, "replicate")

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def replication_floor(self) -> float:
        return self._cfg.replication_floor

    @property
    def config(self) -> OracleConfig:
        return self._cfg

    @property
    def use_e8_geometry(self) -> bool:
        return self._use_e8

    @property
    def e8_symmetry_min(self) -> float:
        return self._e8_min

    @property
    def last_e8_score(self) -> float | None:
        """Most recent E8 symmetry score, or None if the gate hasn't fired."""
        return self._last_e8_score

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return (
            f"ConvergenceOracle(canonical={self.CANONICAL_THRESHOLD:.4f}, "
            f"replication_floor={self._cfg.replication_floor:.4f})"
        )


__all__ = [
    "ConfigError",
    "ConvergenceOracle",
    "DefaultFitness",
    "StateVectorProvider",
]
