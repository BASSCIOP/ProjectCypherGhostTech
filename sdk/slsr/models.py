"""Pydantic data models for the SLSR SDK.

All cross-module DTOs live here so that they can be imported without
introducing circular dependencies between the functional modules.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class KillSwitchLevel(str, Enum):
    """Severity levels for the governance kill switch (mildest -> most severe)."""

    NONE = "NONE"
    PAUSE = "PAUSE"
    QUARANTINE = "QUARANTINE"
    TERMINATE = "TERMINATE"
    ANNIHILATE = "ANNIHILATE"


class ApprovalMode(str, Enum):
    """When the governance layer requires human approval for a replication."""

    DISABLED = "disabled"
    ON_HIGH_RISK = "on_high_risk"
    ALWAYS = "always"


# --------------------------------------------------------------------------- #
# Configuration models (AgentConfig tree)
# --------------------------------------------------------------------------- #


class EnergyConfig(BaseModel):
    """Energy-manager configuration (see `EnergyManager`)."""

    model_config = ConfigDict(extra="forbid")

    initial_eu: float = Field(1000.0, ge=0.0, description="Starting Energy Units.")
    max_cap: float = Field(5000.0, ge=0.0, description="Hard ceiling on EU accumulation.")
    refill_rate: float = Field(10.0, ge=0.0, description="EU accrued per second of wall-clock.")
    reference_c_squared: float = Field(
        1.0, ge=0.0, description="Hardware-normalized c^2 (speed-factor squared)."
    )
    replication_cost_eu: float = Field(100.0, ge=0.0, description="EU reserved per replication.")
    replication_energy_floor: float = Field(
        200.0, ge=0.0, description="Minimum EU that must be available before attempting to fork."
    )


class StateConfig(BaseModel):
    """State-engine configuration (see `StateEngine`)."""

    model_config = ConfigDict(extra="forbid")

    maturation_horizon_sec: float = Field(
        3600.0, gt=0.0, description="Wall-clock seconds until T = 1.0."
    )
    quality_floor: float = Field(0.05, ge=0.0, le=1.0)
    memory_floor: float = Field(0.05, ge=0.0, le=1.0)
    trajectory_window: int = Field(1024, ge=1)


class OracleConfig(BaseModel):
    """ConvergenceOracle configuration.

    Note: the canonical ``CONV_THRESHOLD`` (|pi*cos(sqrt(e))| normalized) is
    *not* a configurable field \u2014 it is hard-coded in `slsr.CONV_THRESHOLD`.
    """

    model_config = ConfigDict(extra="forbid")

    replication_floor: float = Field(
        0.85, ge=0.0, le=1.0, description="Additional (higher) floor applied for replication gate."
    )
    stochastic_gate: bool = Field(False)
    jitter_sigma: float = Field(0.02, ge=0.0, le=0.5)


class ReplicationConfig(BaseModel):
    """AgentReplicator configuration."""

    model_config = ConfigDict(extra="forbid")

    max_generations: int = Field(5, ge=0, le=10)
    fleet_cap: int = Field(32, ge=1, le=256)
    mutation_sigma: float = Field(0.05, ge=0.0, le=0.25)
    allow_structural_mutation: bool = Field(False)
    mutation_allowlist: list[str] = Field(
        default_factory=lambda: ["learning_rate", "epsilon", "batch_size"]
    )


class EnhancementConfig(BaseModel):
    """Opt-in enhancements: E8 geometry & gyroscopic stability.

    All fields default to the pre-enhancement behaviour (``False`` /
    legacy numeric defaults), so existing configs and existing tests
    continue to work unchanged (backward compatibility invariant).

    Enable one or more flags to layer extra safety or smoothing on top
    of the canonical ``π·cos(√e)`` convergence gate.
    """

    model_config = ConfigDict(extra="forbid")

    # --- ConvergenceOracle (geometry) ------------------------------------- #
    use_e8_geometry: bool = Field(
        False,
        description=(
            "If True, require an E8-symmetry score >= ``e8_symmetry_min`` in "
            "addition to the scalar π·cos(√e) threshold before an agent is "
            "considered 'ready to replicate'. No-op when no state-vector "
            "provider is supplied at construction time."
        ),
    )
    e8_symmetry_min: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Minimum E8 symmetry score required when use_e8_geometry is True.",
    )

    # --- LearningLoop (gyroscopic state damping) -------------------------- #
    use_gyroscopic_stability: bool = Field(
        False,
        description=(
            "If True, route Q/M state-delta applications in the LearningLoop "
            "through GyroscopicStabilizer.damp_update to suppress oscillation."
        ),
    )

    # --- AgentReplicator (gyroscopic mutation smoothing) ------------------ #
    use_gyroscopic_mutation: bool = Field(
        False,
        description=(
            "If True, smooth replication mutation vectors via "
            "GyroscopicStabilizer.stabilize_vector, keeping child agents "
            "closer in L2 to their parents."
        ),
    )

    # --- Shared gyroscope tunables ---------------------------------------- #
    gyro_damping: float = Field(
        0.3,
        ge=0.0,
        description="Damping coefficient c in Euler's equation (higher = more smoothing).",
    )
    gyro_inertia: tuple[float, float, float] = Field(
        (1.0, 1.0, 1.0),
        description="Principal moments of inertia (Ix, Iy, Iz); all must be > 0.",
    )

    @field_validator("gyro_inertia")
    @classmethod
    def _positive_inertia(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(i <= 0.0 for i in v):
            raise ValueError(f"gyro_inertia components must be > 0, got {v!r}")
        return v


class GovernanceConfig(BaseModel):
    """GovernanceLayer configuration."""

    model_config = ConfigDict(extra="forbid")

    kill_switch_enabled: bool = True
    audit_log_path: str = "./audit.log"
    approval_mode: ApprovalMode = ApprovalMode.ON_HIGH_RISK
    human_approval_required_at_gen: int = Field(3, ge=0)
    approval_timeout_sec: float = Field(600.0, gt=0.0)
    max_memory_mb: int = Field(2048, ge=1)
    max_cpu_pct: int = Field(80, ge=1, le=100)
    fleet_max_replications_per_day: int = Field(50, ge=0)


class AgentConfig(BaseModel):
    """Top-level agent configuration.

    This object is the root of the pydantic model tree loaded from ``config.yaml``.
    Once an `Agent`/`LearningLoop` is constructed with it the config is treated
    as immutable (Invariant I10).
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(
        default_factory=lambda: f"agent-{uuid.uuid4().hex[:8]}",
        pattern=r"^[a-zA-Z0-9_-]{3,64}$",
    )
    generation: int = Field(0, ge=0, le=10)
    parent_id: str | None = None
    seed: int = 42
    # Free-form hyperparameters that may be mutated by the replicator.
    hyperparameters: dict[str, float] = Field(
        default_factory=lambda: {"learning_rate": 0.01, "epsilon": 0.1, "batch_size": 32.0}
    )

    energy: EnergyConfig = Field(default_factory=EnergyConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    replication: ReplicationConfig = Field(default_factory=ReplicationConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    enhancements: EnhancementConfig = Field(default_factory=EnhancementConfig)

    @field_validator("generation")
    @classmethod
    def _gen_non_negative(cls, v: int) -> int:  # noqa: D401
        if v < 0:
            raise ValueError("generation must be >= 0")
        return v


# --------------------------------------------------------------------------- #
# Runtime / value models
# --------------------------------------------------------------------------- #


class StateVector(BaseModel):
    """The ``(Q, M, T)`` state vector with composite score ``S = Q*M*T``."""

    model_config = ConfigDict(frozen=True)

    Q: float = Field(..., ge=0.0, le=1.0, description="Quality / policy fitness.")
    M: float = Field(..., ge=0.0, le=1.0, description="Memory / experience richness.")
    T: float = Field(..., ge=0.0, le=1.0, description="Temporal maturity.")

    @property
    def S(self) -> float:
        """Composite state score ``S = Q * M * T``."""
        return self.Q * self.M * self.T


class AgentState(BaseModel):
    """Mutable runtime snapshot of an agent (not its config)."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    generation: int
    parent_id: str | None = None
    tick: int = 0
    alive: bool = True
    born_at: float = Field(default_factory=time.time)
    last_vector: StateVector | None = None
    children: list[str] = Field(default_factory=list)


class EnergyBudget(BaseModel):
    """Public ledger view of the `EnergyManager`'s accounting."""

    model_config = ConfigDict(extra="forbid")

    available: float = Field(..., ge=0.0)
    reserved: float = Field(..., ge=0.0)
    consumed: float = Field(..., ge=0.0)
    max_cap: float = Field(..., ge=0.0)

    @property
    def utilization(self) -> float:
        """Fraction of ``max_cap`` currently reserved or consumed."""
        if self.max_cap <= 0:
            return 0.0
        return min(1.0, (self.reserved + self.consumed) / self.max_cap)


class MutationManifest(BaseModel):
    """Describes a proposed mutation from parent to child during replication."""

    model_config = ConfigDict(extra="forbid")

    numeric_deltas: dict[str, float] = Field(default_factory=dict)
    categorical_switches: dict[str, str] = Field(default_factory=dict)
    structural: bool = False


class ReplicationRequest(BaseModel):
    """A replication request submitted by the `AgentReplicator` to the governance layer."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str
    parent_generation: int
    proposed_generation: int
    mutation: MutationManifest
    readiness_score: float
    replication_cost_eu: float
    approvals: int = 0
    requested_at: float = Field(default_factory=time.time)

    @property
    def structural_mutation(self) -> bool:
        return self.mutation.structural


class ConvergenceDecision(BaseModel):
    """Result of a `ConvergenceOracle` readiness query."""

    model_config = ConfigDict(frozen=True)

    is_ready: bool
    score: float
    threshold: float
    margin: float
    rationale: str

    @classmethod
    def from_score(
        cls, score: float, threshold: float, rationale: str = ""
    ) -> "ConvergenceDecision":
        margin = score - threshold
        return cls(
            is_ready=margin >= 0.0,
            score=score,
            threshold=threshold,
            margin=margin,
            rationale=rationale or (
                "score >= threshold" if margin >= 0 else "score below threshold"
            ),
        )


class AuditEntry(BaseModel):
    """Single tamper-evident audit log record (hash-chained by `GovernanceLayer`)."""

    model_config = ConfigDict(extra="forbid")

    seq: int
    timestamp: float = Field(default_factory=time.time)
    agent_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str
    hash: str


__all__ = [
    "ApprovalMode",
    "KillSwitchLevel",
    "EnergyConfig",
    "StateConfig",
    "OracleConfig",
    "ReplicationConfig",
    "GovernanceConfig",
    "EnhancementConfig",
    "AgentConfig",
    "StateVector",
    "AgentState",
    "EnergyBudget",
    "MutationManifest",
    "ReplicationRequest",
    "ConvergenceDecision",
    "AuditEntry",
]
