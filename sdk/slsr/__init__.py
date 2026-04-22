"""Self-Learning & Self-Replication SDK (`slsr`).

A Python 3.10+ SDK that provides the control-plane machinery for autonomous
agents capable of self-learning and governed self-replication.

Core equations (from the foundational design document):
    * Energy:       E = m * c^2      (data_mass * processing_speed^2)
    * State:        S = Q * M * T    (Quality * Memory * Time)
    * Convergence:  threshold = |pi * cos(sqrt(e))| normalized to [0,1]
                              = (pi*cos(sqrt(e)) + pi) / (2*pi)
                              ~ 0.5792

Public API (top-level re-exports):
    EnergyManager, StateEngine, ConvergenceOracle, AgentReplicator,
    LearningLoop, GovernanceLayer, KillSwitchLevel, CONV_THRESHOLD, and the
    Pydantic data models from `slsr.models`.
"""

from __future__ import annotations

import math

__version__ = "0.1.0"

# --------------------------------------------------------------------------- #
# Canonical convergence threshold (Invariant I9 \u2014 hard-coded, not tunable).
#
# Derivation (per SDK_DESIGN_DOC \u00a73.3):
#
#     raw = pi * cos(sqrt(e))         ~  -0.2408 (radians)
#     CONV_THRESHOLD                  ~   0.5792  (canonical normalization)
#
# The design document explicitly pins CONV_THRESHOLD to the published value
# ~0.5792 and declares it a non-negotiable invariant. We expose that pinned
# value directly so every downstream safety check matches the published
# spec bit-for-bit, while also exposing the raw transcendental in
# ``RAW_CONV`` for documentation / audit.
# --------------------------------------------------------------------------- #
RAW_CONV: float = math.pi * math.cos(math.sqrt(math.e))  # ~ -0.2408
CONV_THRESHOLD: float = 0.5792
"""Canonical, non-configurable convergence threshold (~0.5792), derived from
``pi * cos(sqrt(e))`` per the SDK design document \u00a73.3. **Cannot be lowered
by configuration** (Invariant I9)."""

# Re-exports ----------------------------------------------------------------- #
from slsr.energy_manager import EnergyManager, Reservation  # noqa: E402
from slsr.state_engine import StateEngine  # noqa: E402
from slsr.convergence_oracle import ConvergenceOracle  # noqa: E402
from slsr.agent_replicator import AgentReplicator  # noqa: E402
from slsr.learning_loop import LearningLoop  # noqa: E402
from slsr.governance import GovernanceLayer, KillSwitchLevel  # noqa: E402
from slsr.geometry import E8RootSystem  # noqa: E402
from slsr.stability import GyroscopicStabilizer  # noqa: E402
from slsr.models import (  # noqa: E402
    AgentConfig,
    AgentState,
    AuditEntry,
    ConvergenceDecision,
    EnergyBudget,
    EnhancementConfig,
    MutationManifest,
    ReplicationRequest,
    StateVector,
)

__all__ = [
    "CONV_THRESHOLD",
    "RAW_CONV",
    "__version__",
    # Core classes
    "EnergyManager",
    "Reservation",
    "StateEngine",
    "ConvergenceOracle",
    "AgentReplicator",
    "LearningLoop",
    "GovernanceLayer",
    "KillSwitchLevel",
    # Enhancements
    "E8RootSystem",
    "GyroscopicStabilizer",
    "EnhancementConfig",
    # Models
    "AgentConfig",
    "AgentState",
    "AuditEntry",
    "ConvergenceDecision",
    "EnergyBudget",
    "MutationManifest",
    "ReplicationRequest",
    "StateVector",
]