"""Protocol interfaces for pluggable SLSR SDK extension points.

The SDK is built around PEP-544 `Protocol`s so that users can subclass or
supply light-weight duck-typed implementations without inheriting from
concrete base classes. See ``SDK_DESIGN_DOC.md`` \u00a77.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from slsr.models import AgentConfig, MutationManifest, StateVector


# --------------------------------------------------------------------------- #
# Protocol: LearningStrategy
# --------------------------------------------------------------------------- #


@dataclass
class LearningResult:
    """Structured output of a `LearningStrategy.update()` call."""

    loss: float
    data_mass_mb: float
    speed_factor: float  # c (not c^2); EnergyManager squares it internally
    info: dict[str, Any] | None = None


@runtime_checkable
class LearningStrategy(Protocol):
    """How the agent turns a batch of observations into a policy update."""

    def update(self, batch: Any, energy_budget: float) -> LearningResult:  # noqa: D401
        """Apply one learning step and return resource + loss metadata."""
        ...


# --------------------------------------------------------------------------- #
# Protocol: FitnessFunction
# --------------------------------------------------------------------------- #


@runtime_checkable
class FitnessFunction(Protocol):
    """Maps a `StateVector` (+ context) to a scalar in ``[0, 1]``."""

    def score(self, state: StateVector, context: dict[str, Any] | None = None) -> float: ...


# --------------------------------------------------------------------------- #
# Protocol: ReplicationPolicy
# --------------------------------------------------------------------------- #


@runtime_checkable
class ReplicationPolicy(Protocol):
    """Decides *what* to mutate and by how much when forking a child."""

    def propose_mutation(self, parent_config: AgentConfig) -> MutationManifest: ...


# --------------------------------------------------------------------------- #
# Protocol: EnvironmentAdapter
# --------------------------------------------------------------------------- #


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Thin shim over whatever environment the agent is acting in."""

    def observe(self) -> Any: ...

    def act(self, action: Any) -> Any: ...


# --------------------------------------------------------------------------- #
# Protocol: InfrastructureAdapter
# --------------------------------------------------------------------------- #


@dataclass
class ChildHandle:
    """Opaque handle returned after a successful child spawn."""

    agent_id: str
    runtime_ref: Any = None


@runtime_checkable
class InfrastructureAdapter(Protocol):
    """Actually creates child processes/containers/pods."""

    def spawn(self, blueprint: AgentConfig) -> ChildHandle: ...

    def terminate(self, handle: ChildHandle) -> None: ...


# --------------------------------------------------------------------------- #
# Protocol: HumanApprovalGate
# --------------------------------------------------------------------------- #


@runtime_checkable
class HumanApprovalGate(Protocol):
    """Routes high-risk replication requests to a human approver."""

    def request_approval(self, request_payload: dict[str, Any], timeout_sec: float) -> bool: ...


__all__ = [
    "ChildHandle",
    "EnvironmentAdapter",
    "FitnessFunction",
    "HumanApprovalGate",
    "InfrastructureAdapter",
    "LearningResult",
    "LearningStrategy",
    "ReplicationPolicy",
]
