"""AgentReplicator \u2014 governed self-replication with lineage tracking.

Replication pipeline (strict order, Invariants I3, I4, I7):

    1. ConvergenceOracle.is_ready_to_replicate()      [gate 1: fitness]
    2. EnergyManager.reserve_replication()            [gate 2: resource]
    3. GovernanceLayer.evaluate_replication_request() [gate 3: policy]
    4. (Optional) HumanApprovalGate.await_approval()  [gate 4: human]
    5. Apply mutation (bounded by mutation_sigma)
    6. InfrastructureAdapter.spawn(child)
    7. GovernanceLayer.register_child(...)
    8. AuditLog.append

Mutations are bounded per-field to ``mutation_sigma * parent_value`` for
numerics; categorical switches must be present in the ``mutation_allowlist``.
Structural mutations are forbidden unless explicitly enabled in config.
"""

from __future__ import annotations

import copy
import logging
import random
import uuid
from dataclasses import dataclass, field

from slsr.convergence_oracle import ConvergenceOracle
from slsr.energy_manager import EnergyManager, OutOfEnergyError
from slsr.governance import (
    GovernanceLayer,
    GovernanceError,
    PolicyDecision,
)
from slsr.models import (
    AgentConfig,
    MutationManifest,
    ReplicationConfig,
    ReplicationRequest,
)
from slsr.protocols import InfrastructureAdapter, ReplicationPolicy
from slsr.stability import GyroscopicStabilizer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ReplicationError(Exception):
    """Base class for replication failures."""


class MutationOutOfBoundsError(ReplicationError):
    """Raised when a proposed mutation violates ``mutation_sigma`` bounds."""


class GenerationLimitExceededError(ReplicationError):
    """Raised when replication would exceed ``max_generations``."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class LineageNode:
    """Tree node describing one agent's place in the replication lineage."""

    agent_id: str
    generation: int
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    mutation: MutationManifest | None = None


@dataclass
class ReplicationOutcome:
    """Returned by `AgentReplicator.replicate`."""

    success: bool
    child_config: AgentConfig | None = None
    mutation: MutationManifest | None = None
    request: ReplicationRequest | None = None
    decision: PolicyDecision | None = None
    reason: str = ""


# --------------------------------------------------------------------------- #
# Default strategies
# --------------------------------------------------------------------------- #


class GaussianNumericMutator:
    """Default `ReplicationPolicy`: samples bounded Gaussian deltas for numerics."""

    def __init__(self, sigma: float, allowlist: list[str], rng: random.Random | None = None):
        self._sigma = sigma
        self._allowlist = allowlist
        self._rng = rng or random.Random()

    def propose_mutation(self, parent_config: AgentConfig) -> MutationManifest:
        numeric: dict[str, float] = {}
        for field_name in self._allowlist:
            if field_name not in parent_config.hyperparameters:
                continue
            parent_val = parent_config.hyperparameters[field_name]
            # delta is sampled as a *relative* fraction of the parent value,
            # bounded in [-sigma, +sigma] via truncation.
            raw = self._rng.gauss(0.0, self._sigma)
            bounded = max(-self._sigma, min(self._sigma, raw))
            numeric[field_name] = bounded * parent_val
        return MutationManifest(
            numeric_deltas=numeric,
            categorical_switches={},
            structural=False,
        )


class NullInfrastructureAdapter:
    """In-memory adapter that just returns a deterministic handle.

    Useful for tests and the demo where we don't actually fork processes.
    """

    def __init__(self) -> None:
        self._spawned: list[str] = []

    def spawn(self, blueprint: AgentConfig):  # type: ignore[override]
        from slsr.protocols import ChildHandle

        self._spawned.append(blueprint.agent_id)
        return ChildHandle(agent_id=blueprint.agent_id, runtime_ref="in-memory")

    def terminate(self, handle) -> None:  # type: ignore[override]
        if handle.agent_id in self._spawned:
            self._spawned.remove(handle.agent_id)

    @property
    def spawned(self) -> list[str]:
        return list(self._spawned)


# --------------------------------------------------------------------------- #
# AgentReplicator
# --------------------------------------------------------------------------- #


class AgentReplicator:
    """Governed replication pipeline + lineage registry."""

    def __init__(
        self,
        parent_config: AgentConfig,
        *,
        oracle: ConvergenceOracle,
        energy: EnergyManager,
        governance: GovernanceLayer,
        config: ReplicationConfig | None = None,
        adapter: InfrastructureAdapter | None = None,
        policy: ReplicationPolicy | None = None,
        rng: random.Random | None = None,
        use_gyroscopic_mutation: bool = False,
        stabilizer: GyroscopicStabilizer | None = None,
    ) -> None:
        self._parent_cfg = parent_config
        self._oracle = oracle
        self._energy = energy
        self._governance = governance
        self._cfg = config or ReplicationConfig()
        self._adapter: InfrastructureAdapter = adapter or NullInfrastructureAdapter()
        self._rng = rng or random.Random(parent_config.seed)
        self._policy: ReplicationPolicy = policy or GaussianNumericMutator(
            sigma=self._cfg.mutation_sigma,
            allowlist=self._cfg.mutation_allowlist,
            rng=self._rng,
        )

        # --- Optional gyroscopic mutation smoother -------------------- #
        self._use_gyro_mut: bool = bool(use_gyroscopic_mutation)
        if self._use_gyro_mut:
            if stabilizer is not None:
                self._stabilizer: GyroscopicStabilizer | None = stabilizer
            else:
                enh = getattr(parent_config, "enhancements", None)
                damping = float(enh.gyro_damping) if enh is not None else 0.3
                inertia = tuple(enh.gyro_inertia) if enh is not None else (1.0, 1.0, 1.0)
                self._stabilizer = GyroscopicStabilizer(
                    inertia=inertia, damping=damping  # type: ignore[arg-type]
                )
        else:
            self._stabilizer = stabilizer

        # Lineage registry keyed by agent_id. Root is the parent we were
        # constructed with \u2014 register it lazily.
        self._lineage: dict[str, LineageNode] = {
            parent_config.agent_id: LineageNode(
                agent_id=parent_config.agent_id,
                generation=parent_config.generation,
                parent_id=parent_config.parent_id,
            )
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def can_replicate(self) -> tuple[bool, str]:
        """Quick pre-flight check (no side effects, no audit)."""
        if self._parent_cfg.generation >= self._cfg.max_generations:
            return False, (
                f"parent generation {self._parent_cfg.generation} "
                f">= max_generations={self._cfg.max_generations}"
            )
        if not self._oracle.is_ready_to_replicate().is_ready:
            return False, "convergence oracle: not ready to replicate"
        if not self._energy.check_replication_affordable():
            return False, "energy manager: insufficient EU for replication"
        return True, "preflight OK"

    def replicate(self) -> ReplicationOutcome:
        """Attempt one full replication cycle.

        Returns a `ReplicationOutcome` describing what happened, *not*
        raising on expected failure modes (denied, out-of-energy, etc.).
        Unexpected errors propagate.
        """
        parent = self._parent_cfg
        logger.info("replication attempt for parent=%s (gen=%d)", parent.agent_id, parent.generation)

        # Generation cap (enforced both here and in governance \u2014 defence in depth)
        if parent.generation >= self._cfg.max_generations:
            reason = (
                f"generation limit: parent at gen {parent.generation}, "
                f"max={self._cfg.max_generations}"
            )
            self._governance.audit(
                parent.agent_id, "replication_aborted", reason=reason
            )
            return ReplicationOutcome(success=False, reason=reason)

        # Gate 1: oracle
        decision = self._oracle.is_ready_to_replicate()
        if not decision.is_ready:
            self._governance.audit(
                parent.agent_id,
                "replication_aborted",
                reason="convergence not ready",
                score=decision.score,
                threshold=decision.threshold,
            )
            return ReplicationOutcome(
                success=False,
                reason=f"not ready: {decision.rationale}",
            )

        # Gate 2: energy reservation
        try:
            reservation = self._energy.reserve_replication()
        except OutOfEnergyError as exc:
            self._governance.audit(
                parent.agent_id, "replication_aborted", reason="out of energy"
            )
            return ReplicationOutcome(success=False, reason=str(exc))

        try:
            # Build mutation + request
            mutation = self._policy.propose_mutation(parent)

            # Optional: smooth the numeric-delta vector with the
            # gyroscopic stabilizer so children don't jump far from the
            # parent in parameter space.
            if self._use_gyro_mut and self._stabilizer is not None and mutation.numeric_deltas:
                mutation = self._apply_gyro_smoothing(mutation)

            self._validate_mutation(mutation, parent)
            request = ReplicationRequest(
                parent_id=parent.agent_id,
                parent_generation=parent.generation,
                proposed_generation=parent.generation + 1,
                mutation=mutation,
                readiness_score=decision.score,
                replication_cost_eu=reservation.amount_eu,
            )

            # Gate 3: governance policy
            pol = self._governance.evaluate_replication_request(
                request,
                max_generations=self._cfg.max_generations,
                fleet_cap=self._cfg.fleet_cap,
                allow_structural_mutation=self._cfg.allow_structural_mutation,
            )
            if not pol.allowed:
                self._energy.release(reservation)
                return ReplicationOutcome(
                    success=False, decision=pol, request=request, reason=pol.reason
                )

            # Apply mutation -> child config
            child_cfg = self._apply_mutation(parent, mutation)

            # Spawn via infrastructure adapter
            handle = self._adapter.spawn(child_cfg)
            logger.info("spawned child=%s via %s", handle.agent_id, type(self._adapter).__name__)

            # Governance registration + lineage
            self._governance.register_child(
                child_id=child_cfg.agent_id,
                parent_id=parent.agent_id,
                generation=child_cfg.generation,
                fleet_cap=self._cfg.fleet_cap,
            )
            self._lineage[child_cfg.agent_id] = LineageNode(
                agent_id=child_cfg.agent_id,
                generation=child_cfg.generation,
                parent_id=parent.agent_id,
                mutation=mutation,
            )
            self._lineage[parent.agent_id].children.append(child_cfg.agent_id)

            # Commit energy now that the op has fully succeeded
            self._energy.commit(reservation)

            self._governance.audit(
                parent.agent_id,
                "replication_succeeded",
                request_id=request.request_id,
                child_id=child_cfg.agent_id,
                generation=child_cfg.generation,
            )
            return ReplicationOutcome(
                success=True,
                child_config=child_cfg,
                mutation=mutation,
                request=request,
                decision=pol,
                reason="ok",
            )
        except GovernanceError as gerr:
            self._energy.release(reservation)
            return ReplicationOutcome(success=False, reason=f"governance: {gerr}")
        except ReplicationError as rerr:
            self._energy.release(reservation)
            return ReplicationOutcome(success=False, reason=f"replication: {rerr}")
        except Exception:
            # Unknown error \u2014 always refund and re-raise
            self._energy.release(reservation)
            raise

    # ------------------------------------------------------------------ #
    # Mutation mechanics
    # ------------------------------------------------------------------ #

    def _apply_gyro_smoothing(self, mutation: MutationManifest) -> MutationManifest:
        """Return a new `MutationManifest` with numeric deltas gyro-smoothed.

        The smoothing is strictly contractive (output L2 <= input L2), so it
        cannot cause a validated mutation to become *more* extreme. We
        therefore apply it **before** :meth:`_validate_mutation`.
        """
        assert self._stabilizer is not None  # guard; callers check
        import numpy as np  # local import

        keys = list(mutation.numeric_deltas.keys())
        values = np.asarray(
            [mutation.numeric_deltas[k] for k in keys], dtype=np.float64
        )
        smoothed = self._stabilizer.stabilize_vector(values)
        new_deltas = {k: float(smoothed[i]) for i, k in enumerate(keys)}
        return MutationManifest(
            numeric_deltas=new_deltas,
            categorical_switches=dict(mutation.categorical_switches),
            structural=mutation.structural,
        )

    def _validate_mutation(self, mutation: MutationManifest, parent: AgentConfig) -> None:
        sigma = self._cfg.mutation_sigma
        allow = set(self._cfg.mutation_allowlist)

        for field_name, delta in mutation.numeric_deltas.items():
            if field_name not in allow:
                raise MutationOutOfBoundsError(
                    f"numeric field '{field_name}' not in mutation_allowlist"
                )
            parent_val = parent.hyperparameters.get(field_name, 0.0)
            bound = abs(sigma * parent_val) + 1e-9
            if abs(delta) > bound:
                raise MutationOutOfBoundsError(
                    f"delta for '{field_name}' = {delta:+.4f} exceeds bound "
                    f"{bound:.4f} (sigma={sigma})"
                )

        for field_name in mutation.categorical_switches:
            if field_name not in allow:
                raise MutationOutOfBoundsError(
                    f"categorical field '{field_name}' not in mutation_allowlist"
                )

        if mutation.structural and not self._cfg.allow_structural_mutation:
            raise MutationOutOfBoundsError("structural mutation disabled by config")

    def _apply_mutation(
        self, parent: AgentConfig, mutation: MutationManifest
    ) -> AgentConfig:
        """Return a deep-copied child `AgentConfig` with the mutation applied."""
        child_dict = copy.deepcopy(parent.model_dump())

        hp = child_dict.setdefault("hyperparameters", {})
        for k, delta in mutation.numeric_deltas.items():
            hp[k] = hp.get(k, 0.0) + delta
        for k, v in mutation.categorical_switches.items():
            hp[k] = v

        child_dict["agent_id"] = self._new_child_id(parent.agent_id)
        child_dict["parent_id"] = parent.agent_id
        child_dict["generation"] = parent.generation + 1
        # seed derivation: deterministic but unique
        child_dict["seed"] = (parent.seed * 31 + parent.generation + 7) & 0x7FFFFFFF

        return AgentConfig.model_validate(child_dict)

    @staticmethod
    def _new_child_id(parent_id: str) -> str:
        return f"{parent_id}-c{uuid.uuid4().hex[:6]}"

    # ------------------------------------------------------------------ #
    # Lineage introspection
    # ------------------------------------------------------------------ #

    def lineage(self) -> dict[str, LineageNode]:
        """Return the current lineage registry (defensive shallow copy)."""
        return dict(self._lineage)

    def descendants(self, agent_id: str) -> list[str]:
        """Return all descendants of ``agent_id`` in the lineage tree."""
        out: list[str] = []
        stack = [agent_id]
        while stack:
            cur = stack.pop()
            node = self._lineage.get(cur)
            if node is None:
                continue
            for child in node.children:
                out.append(child)
                stack.append(child)
        return out

    @property
    def adapter(self) -> InfrastructureAdapter:
        return self._adapter

    @property
    def config(self) -> ReplicationConfig:
        return self._cfg

    @property
    def use_gyroscopic_mutation(self) -> bool:
        return self._use_gyro_mut

    @property
    def stabilizer(self) -> GyroscopicStabilizer | None:
        return self._stabilizer


__all__ = [
    "AgentReplicator",
    "GaussianNumericMutator",
    "GenerationLimitExceededError",
    "LineageNode",
    "MutationOutOfBoundsError",
    "NullInfrastructureAdapter",
    "ReplicationError",
    "ReplicationOutcome",
]
