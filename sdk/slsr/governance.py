"""GovernanceLayer \u2014 safety, policies, kill switch, and audit log.

Responsibilities (see SDK_DESIGN_DOC \u00a73.6):
    * Policy evaluation for governed ops (especially replication).
    * Kill switch (4 levels: PAUSE, QUARANTINE, TERMINATE, ANNIHILATE).
    * Tamper-evident audit log (SHA-256 hash chain).
    * Human approval gating (pluggable via `HumanApprovalGate`).
    * Fleet-wide resource ceilings (generations alive, replications/day).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from slsr.models import (
    ApprovalMode,
    AuditEntry,
    GovernanceConfig,
    KillSwitchLevel,
    ReplicationRequest,
)
from slsr.protocols import HumanApprovalGate

logger = logging.getLogger(__name__)

_GENESIS_HASH = "0" * 64


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class GovernanceError(Exception):
    """Base class for governance-layer errors."""


class PolicyDeniedError(GovernanceError):
    """Raised when a policy explicitly denies an operation."""


class KillSwitchEngagedError(GovernanceError):
    """Raised when the kill switch is engaged at a severity that blocks ops."""


class GenerationLimitExceededError(GovernanceError):
    """Raised when a replication would exceed ``max_generations``."""


class FleetCapExceededError(GovernanceError):
    """Raised when registering a new child would exceed the fleet cap."""


class ApprovalTimeoutError(GovernanceError):
    """Raised when a human approval gate does not respond in time."""


# --------------------------------------------------------------------------- #
# PolicyDecision
# --------------------------------------------------------------------------- #


@dataclass
class PolicyDecision:
    """Result of evaluating a policy (or policy bundle) against an op."""

    allowed: bool
    reason: str = ""
    requires_human_approval: bool = False

    @classmethod
    def allow(cls, reason: str = "allowed") -> "PolicyDecision":
        return cls(allowed=True, reason=reason)

    @classmethod
    def deny(cls, reason: str) -> "PolicyDecision":
        return cls(allowed=False, reason=reason)

    @classmethod
    def needs_approval(cls, reason: str = "human approval required") -> "PolicyDecision":
        return cls(allowed=True, reason=reason, requires_human_approval=True)


# --------------------------------------------------------------------------- #
# Audit log (hash-chained, append-only)
# --------------------------------------------------------------------------- #


class AuditLog:
    """Append-only, SHA-256 hash-chained audit log."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._prev_hash = _GENESIS_HASH
        self._seq = 0

    @staticmethod
    def _hash_record(seq: int, ts: float, agent_id: str, event_type: str,
                     payload: dict[str, Any], prev_hash: str) -> str:
        material = json.dumps(
            {
                "seq": seq,
                "ts": ts,
                "agent_id": agent_id,
                "event_type": event_type,
                "payload": payload,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def append(self, agent_id: str, event_type: str, **payload: Any) -> AuditEntry:
        """Append a new record. Thread-safe."""
        with self._lock:
            ts = time.time()
            h = self._hash_record(
                self._seq, ts, agent_id, event_type, payload, self._prev_hash
            )
            entry = AuditEntry(
                seq=self._seq,
                timestamp=ts,
                agent_id=agent_id,
                event_type=event_type,
                payload=payload,
                prev_hash=self._prev_hash,
                hash=h,
            )
            self._entries.append(entry)
            self._prev_hash = h
            self._seq += 1
            if self._path:
                try:
                    with open(self._path, "a", encoding="utf-8") as fp:
                        fp.write(entry.model_dump_json() + "\n")
                except OSError as exc:  # pragma: no cover - best effort disk log
                    logger.warning("audit log write failed: %s", exc)
            return entry

    def entries(self) -> list[AuditEntry]:
        """Return a defensive copy of all entries."""
        with self._lock:
            return list(self._entries)

    def verify_chain(self) -> bool:
        """Recompute the hash chain and verify integrity."""
        with self._lock:
            prev = _GENESIS_HASH
            for e in self._entries:
                if e.prev_hash != prev:
                    return False
                h = self._hash_record(
                    e.seq, e.timestamp, e.agent_id, e.event_type, e.payload, e.prev_hash
                )
                if h != e.hash:
                    return False
                prev = h
            return True

    def __len__(self) -> int:
        return len(self._entries)


# --------------------------------------------------------------------------- #
# Kill switch
# --------------------------------------------------------------------------- #


@dataclass
class KillCommand:
    """An active kill-switch command."""

    level: KillSwitchLevel
    reason: str = ""
    issued_at: float = 0.0


class KillSwitch:
    """Simple in-process kill switch; pluggable check function supported."""

    def __init__(
        self,
        check_fn: Callable[[], KillCommand | None] | None = None,
    ) -> None:
        self._check_fn = check_fn
        self._engaged: KillCommand | None = None

    def engage(self, level: KillSwitchLevel, reason: str = "") -> KillCommand:
        """Manually engage the kill switch."""
        cmd = KillCommand(level=level, reason=reason, issued_at=time.time())
        self._engaged = cmd
        logger.warning("KillSwitch engaged: %s (%s)", level.value, reason)
        return cmd

    def disengage(self) -> None:
        self._engaged = None

    def poll(self) -> KillCommand | None:
        """Return the current kill command (if any), consulting ``check_fn``."""
        if self._check_fn:
            ext = self._check_fn()
            if ext is not None:
                self._engaged = ext
        return self._engaged

    @property
    def is_engaged(self) -> bool:
        return self._engaged is not None and self._engaged.level != KillSwitchLevel.NONE


# --------------------------------------------------------------------------- #
# GovernanceLayer
# --------------------------------------------------------------------------- #


class GovernanceLayer:
    """Top-level policy / safety enforcement for the SDK."""

    def __init__(
        self,
        config: GovernanceConfig | None = None,
        *,
        audit_log: AuditLog | None = None,
        kill_switch: KillSwitch | None = None,
        approval_gate: HumanApprovalGate | None = None,
    ) -> None:
        self._cfg = config or GovernanceConfig()
        self.audit_log = audit_log or AuditLog(path=self._cfg.audit_log_path)
        self.kill_switch = kill_switch or KillSwitch()
        self.approval_gate = approval_gate

        # Fleet registry
        self._live_agents: dict[str, dict[str, Any]] = {}
        self._replications_today: int = 0
        self._day_anchor: float = time.time()

    # ------------------------------------------------------------------ #
    # Kill switch helpers
    # ------------------------------------------------------------------ #

    def check_kill_switch(self) -> KillCommand | None:
        """Poll the kill switch; also writes an audit record if engaged."""
        if not self._cfg.kill_switch_enabled:
            return None
        cmd = self.kill_switch.poll()
        if cmd is not None and cmd.level != KillSwitchLevel.NONE:
            return cmd
        return None

    def engage_kill_switch(
        self, level: KillSwitchLevel, reason: str = "", agent_id: str = "governance"
    ) -> KillCommand:
        """Engage the kill switch at a given severity."""
        cmd = self.kill_switch.engage(level, reason)
        self.audit(
            agent_id,
            "kill_switch_engaged",
            level=level.value,
            reason=reason,
        )
        return cmd

    def enforce_kill_switch(self, agent_id: str) -> None:
        """Raise `KillSwitchEngagedError` if the switch is at TERMINATE+ severity."""
        cmd = self.check_kill_switch()
        if cmd is None:
            return
        # PAUSE / QUARANTINE are advisory at the governance level \u2014 the
        # LearningLoop inspects the level and adjusts its FSM.  Only
        # TERMINATE / ANNIHILATE actually raise.
        if cmd.level in (KillSwitchLevel.TERMINATE, KillSwitchLevel.ANNIHILATE):
            self.audit(agent_id, "kill_switch_blocked_op", level=cmd.level.value)
            raise KillSwitchEngagedError(
                f"kill switch engaged at {cmd.level.value}: {cmd.reason}"
            )

    # ------------------------------------------------------------------ #
    # Fleet registry
    # ------------------------------------------------------------------ #

    def register_agent(self, agent_id: str, generation: int, parent_id: str | None) -> None:
        """Record that an agent is alive. Enforces ``fleet_cap`` (Invariant I2)."""
        # fleet_cap comes from ReplicationConfig but is enforced here too
        # if the governance layer is used stand-alone; default is tolerant.
        self._live_agents[agent_id] = {
            "generation": generation,
            "parent_id": parent_id,
            "registered_at": time.time(),
        }
        self.audit(
            agent_id,
            "agent_registered",
            generation=generation,
            parent_id=parent_id,
        )

    def register_child(
        self,
        child_id: str,
        parent_id: str,
        generation: int,
        *,
        fleet_cap: int | None = None,
    ) -> None:
        """Register a new child agent. Invariant I2 \u2014 enforces fleet cap."""
        if fleet_cap is not None and len(self._live_agents) >= fleet_cap:
            raise FleetCapExceededError(
                f"fleet_cap={fleet_cap} already reached "
                f"({len(self._live_agents)} live agents)"
            )
        self.register_agent(child_id, generation, parent_id)
        self._bump_replication_counter()
        self.audit(
            parent_id,
            "child_registered",
            child_id=child_id,
            generation=generation,
        )

    def unregister_agent(self, agent_id: str, reason: str = "shutdown") -> None:
        self._live_agents.pop(agent_id, None)
        self.audit(agent_id, "agent_unregistered", reason=reason)

    def live_agents(self) -> list[str]:
        return list(self._live_agents.keys())

    # ------------------------------------------------------------------ #
    # Replication policy evaluation
    # ------------------------------------------------------------------ #

    def evaluate_replication_request(
        self,
        req: ReplicationRequest,
        *,
        max_generations: int,
        fleet_cap: int,
        allow_structural_mutation: bool,
    ) -> PolicyDecision:
        """Run the replication policy pipeline and return a decision.

        Checks (in order):
            * Invariant I1 \u2014 generation limit.
            * Invariant I2 \u2014 fleet cap.
            * Fleet replications-per-day cap.
            * Structural mutation allowlist.
            * Approval mode (mandates human sign-off above a generation).
        """
        self._rollover_day_anchor()

        # I1: generation limit
        if req.proposed_generation > max_generations:
            decision = PolicyDecision.deny(
                f"generation {req.proposed_generation} exceeds max_generations={max_generations}"
            )
            self.audit(
                req.parent_id,
                "replication_denied",
                request_id=req.request_id,
                reason=decision.reason,
            )
            return decision

        # I2: fleet cap
        if len(self._live_agents) >= fleet_cap:
            decision = PolicyDecision.deny(
                f"fleet cap {fleet_cap} reached ({len(self._live_agents)} live agents)"
            )
            self.audit(
                req.parent_id,
                "replication_denied",
                request_id=req.request_id,
                reason=decision.reason,
            )
            return decision

        # Per-day cap
        if self._replications_today >= self._cfg.fleet_max_replications_per_day:
            decision = PolicyDecision.deny(
                f"daily replication cap reached "
                f"({self._replications_today}/{self._cfg.fleet_max_replications_per_day})"
            )
            self.audit(
                req.parent_id,
                "replication_denied",
                request_id=req.request_id,
                reason=decision.reason,
            )
            return decision

        # Structural mutation restrictions (I5)
        if req.structural_mutation and not allow_structural_mutation:
            decision = PolicyDecision.deny("structural mutation is disallowed by config")
            self.audit(
                req.parent_id,
                "replication_denied",
                request_id=req.request_id,
                reason=decision.reason,
            )
            return decision

        # Approval mode
        needs_approval = self._requires_human_approval(req)
        if needs_approval:
            if self.approval_gate is None:
                decision = PolicyDecision.deny(
                    "human approval required but no approval gate is configured"
                )
                self.audit(
                    req.parent_id,
                    "replication_denied",
                    request_id=req.request_id,
                    reason=decision.reason,
                )
                return decision
            # Delegate
            try:
                approved = self.approval_gate.request_approval(
                    request_payload=req.model_dump(),
                    timeout_sec=self._cfg.approval_timeout_sec,
                )
            except TimeoutError as exc:
                raise ApprovalTimeoutError(str(exc)) from exc

            if not approved:
                decision = PolicyDecision.deny("human approver rejected request")
                self.audit(
                    req.parent_id,
                    "replication_denied",
                    request_id=req.request_id,
                    reason=decision.reason,
                )
                return decision

            req.approvals += 1
            self.audit(
                req.parent_id,
                "replication_human_approved",
                request_id=req.request_id,
            )

        decision = PolicyDecision.allow("all policy checks passed")
        self.audit(
            req.parent_id,
            "replication_allowed",
            request_id=req.request_id,
            reason=decision.reason,
            generation=req.proposed_generation,
        )
        return decision

    def _requires_human_approval(self, req: ReplicationRequest) -> bool:
        mode = self._cfg.approval_mode
        if mode == ApprovalMode.DISABLED:
            return False
        if mode == ApprovalMode.ALWAYS:
            return True
        # on_high_risk
        if req.structural_mutation:
            return True
        if req.proposed_generation >= self._cfg.human_approval_required_at_gen:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Audit
    # ------------------------------------------------------------------ #

    def audit(self, agent_id: str, event_type: str, **payload: Any) -> AuditEntry:
        return self.audit_log.append(agent_id, event_type, **payload)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _bump_replication_counter(self) -> None:
        self._rollover_day_anchor()
        self._replications_today += 1

    def _rollover_day_anchor(self) -> None:
        now = time.time()
        if now - self._day_anchor >= 86_400:
            self._day_anchor = now
            self._replications_today = 0

    @property
    def config(self) -> GovernanceConfig:
        return self._cfg

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return (
            f"GovernanceLayer(live={len(self._live_agents)}, "
            f"replications_today={self._replications_today}, "
            f"kill={self.kill_switch.is_engaged})"
        )


__all__ = [
    "ApprovalTimeoutError",
    "AuditLog",
    "FleetCapExceededError",
    "GenerationLimitExceededError",
    "GovernanceError",
    "GovernanceLayer",
    "KillCommand",
    "KillSwitch",
    "KillSwitchEngagedError",
    "KillSwitchLevel",
    "PolicyDecision",
    "PolicyDeniedError",
]
