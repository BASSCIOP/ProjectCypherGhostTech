"""LearningLoop \u2014 the agent's FSM orchestrator.

States::

    INIT -> OBSERVE -> LEARN -> EVALUATE -> DECIDE -> ACT -> [REPLICATE?] -> OBSERVE ...

This reference implementation is deliberately synchronous (``run_cycles``)
AND async (``arun_cycles``) so it can be driven from scripts, tests, or
a larger asyncio-based orchestrator.

Backpressure: when EnergyManager signals low_energy (available < 2 *
tick_cost), the loop enters ``DORMANT`` mode \u2014 it skips LEARN, keeps
OBSERVE, and waits for refill.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from slsr.agent_replicator import AgentReplicator, ReplicationOutcome
from slsr.convergence_oracle import ConvergenceOracle
from slsr.energy_manager import EnergyManager, OutOfEnergyError
from slsr.governance import (
    GovernanceLayer,
    KillSwitchEngagedError,
)
from slsr.models import (
    AgentConfig,
    AgentState,
    ConvergenceDecision,
    KillSwitchLevel,
    StateVector,
)
from slsr.state_engine import StateEngine
from slsr.stability import GyroscopicStabilizer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helper dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class TickResult:
    """Summary of a single LearningLoop tick."""

    tick: int
    phase: str
    state_vector: StateVector
    act_decision: ConvergenceDecision | None = None
    replicate_decision: ConvergenceDecision | None = None
    replication_outcome: ReplicationOutcome | None = None
    observation: Any = None
    action: Any = None
    energy_available: float = 0.0
    dormant: bool = False
    notes: str = ""
    # --- enhancement audit metadata (populated when the flags are on) --- #
    gyro_damping_meta: dict[str, Any] | None = None
    gyro_stability: float | None = None
    e8_symmetry: float | None = None


@dataclass
class RunSummary:
    """Aggregate result of a full `run_cycles` / `arun_cycles`."""

    agent_id: str
    ticks_run: int
    final_state: StateVector
    children_spawned: list[str] = field(default_factory=list)
    energy_consumed: float = 0.0
    halted_reason: str = ""


# --------------------------------------------------------------------------- #
# Default strategies (self-contained so the demo runs without external deps)
# --------------------------------------------------------------------------- #


class SimpleReward:
    """A toy "environment" \u2014 returns (obs, reward) tuples.

    The reward drifts upward over time to simulate a learning agent
    progressing toward convergence.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._step = 0

    def observe(self) -> dict[str, Any]:
        self._step += 1
        base = 1.0 - math.exp(-self._step / 20.0)
        noise = self._rng.gauss(0.0, 0.05)
        reward = max(0.0, min(1.0, base + noise))
        return {"step": self._step, "reward": reward, "obs": [self._rng.random() for _ in range(4)]}

    def act(self, action: Any) -> None:
        # A real env would update; this one is a no-op sink.
        logger.debug("env.act(%r)", action)


class ExponentialMovingAverageLearner:
    """Trivial "learner" that tracks an EMA of rewards as a quality proxy."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._q: float = 0.0

    def update(self, batch: dict[str, Any]) -> float:
        reward = float(batch.get("reward", 0.0))
        self._q = (1 - self._alpha) * self._q + self._alpha * reward
        return self._q

    @property
    def q(self) -> float:
        return self._q


# --------------------------------------------------------------------------- #
# LearningLoop
# --------------------------------------------------------------------------- #


HookCallback = Callable[["LearningLoop", TickResult], None]


class LearningLoop:
    """Orchestrates the full OBSERVE -> LEARN -> EVALUATE -> DECIDE -> ACT cycle."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        energy: EnergyManager,
        state: StateEngine,
        oracle: ConvergenceOracle,
        governance: GovernanceLayer,
        replicator: AgentReplicator | None = None,
        environment: Any | None = None,
        learner: Any | None = None,
        tick_energy_cost: float = 5.0,
        memory_growth_rate: float = 0.02,
        use_gyroscopic_stability: bool = False,
        stabilizer: "GyroscopicStabilizer | None" = None,
    ) -> None:
        self._cfg = config
        self._energy = energy
        self._state = state
        self._oracle = oracle
        self._governance = governance
        self._replicator = replicator
        self._env = environment or SimpleReward(seed=config.seed)
        self._learner = learner or ExponentialMovingAverageLearner()
        self._tick_energy_cost = tick_energy_cost
        self._memory_growth_rate = memory_growth_rate

        # --- Optional gyroscopic state-delta smoother ------------------ #
        self._use_gyro: bool = bool(use_gyroscopic_stability)
        if self._use_gyro:
            if stabilizer is not None:
                self._stabilizer: GyroscopicStabilizer | None = stabilizer
            else:
                enh = getattr(config, "enhancements", None)
                damping = float(enh.gyro_damping) if enh is not None else 0.3
                inertia = tuple(enh.gyro_inertia) if enh is not None else (1.0, 1.0, 1.0)
                self._stabilizer = GyroscopicStabilizer(
                    inertia=inertia, damping=damping  # type: ignore[arg-type]
                )
        else:
            self._stabilizer = stabilizer  # allow external observation even when off

        self._tick: int = 0
        self._children: list[str] = []
        self._status = AgentState(
            agent_id=config.agent_id,
            generation=config.generation,
            parent_id=config.parent_id,
        )

        self._hooks: dict[str, list[HookCallback]] = {
            "on_tick": [],
            "before_learn": [],
            "after_evaluate": [],
            "before_replicate": [],
            "on_shutdown": [],
        }

        # Self-register with governance
        self._governance.register_agent(
            config.agent_id, config.generation, config.parent_id
        )
        self._governance.audit(config.agent_id, "agent_born", generation=config.generation)

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #

    def on_tick(self, cb: HookCallback) -> None:
        self._hooks["on_tick"].append(cb)

    def before_learn(self, cb: HookCallback) -> None:
        self._hooks["before_learn"].append(cb)

    def after_evaluate(self, cb: HookCallback) -> None:
        self._hooks["after_evaluate"].append(cb)

    def before_replicate(self, cb: HookCallback) -> None:
        self._hooks["before_replicate"].append(cb)

    def on_shutdown(self, cb: HookCallback) -> None:
        self._hooks["on_shutdown"].append(cb)

    def _fire(self, name: str, result: TickResult) -> None:
        for cb in self._hooks.get(name, []):
            try:
                cb(self, result)
            except Exception as exc:  # pragma: no cover - hooks are user code
                logger.warning("hook %s raised: %s", name, exc)

    # ------------------------------------------------------------------ #
    # FSM primitive: one tick
    # ------------------------------------------------------------------ #

    def tick(self, *, attempt_replication: bool = True) -> TickResult:
        """Execute exactly one FSM tick (sync-friendly).

        If the energy manager is low, switches to a dormant tick (OBSERVE only).
        """
        self._tick += 1
        self._status.tick = self._tick

        # Kill switch gate at every FSM transition (Invariant I6)
        try:
            self._governance.enforce_kill_switch(self._cfg.agent_id)
        except KillSwitchEngagedError:
            self._status.alive = False
            raise

        kill_cmd = self._governance.check_kill_switch()
        pause_mode = kill_cmd is not None and kill_cmd.level == KillSwitchLevel.PAUSE

        # --- OBSERVE ---
        observation = self._env.observe()

        # Decide if we can afford a full tick
        dormant = pause_mode or (self._energy.available < 2 * self._tick_energy_cost)

        action = None
        act_decision: ConvergenceDecision | None = None
        replicate_decision: ConvergenceDecision | None = None
        replication_outcome: ReplicationOutcome | None = None
        notes: list[str] = []
        gyro_meta: dict[str, Any] | None = None

        if dormant:
            notes.append("dormant: low energy or PAUSE kill-switch" if dormant else "")
            # Still advance T so the agent ages even in dormancy
            self._state.tick_time()
        else:
            # --- LEARN ---
            result_stub = TickResult(
                tick=self._tick,
                phase="LEARN",
                state_vector=self._state.snapshot(),
                observation=observation,
                energy_available=self._energy.available,
            )
            self._fire("before_learn", result_stub)

            try:
                res = self._energy.reserve(self._tick_energy_cost, reason="learn")
            except OutOfEnergyError:
                notes.append("learn reservation failed \u2014 entering dormancy")
                dormant = True
                self._state.tick_time()
            else:
                q_hat = self._learner.update(observation)
                self._energy.commit(res)

                # --- EVALUATE ---
                pre_vec = self._state.snapshot()
                proposed_q = float(q_hat)
                proposed_m = float(pre_vec.M + self._memory_growth_rate)

                if self._use_gyro and self._stabilizer is not None:
                    # Route the scalar channel deltas through the
                    # gyroscopic damper (axis 0 = Q, axis 1 = M). This
                    # is a pure *smoothing* wrapper: the return value is
                    # always between the current and proposed values.
                    smoothed_q = self._stabilizer.damp_update(
                        pre_vec.Q, proposed_q, axis=0
                    )
                    smoothed_m = self._stabilizer.damp_update(
                        pre_vec.M, proposed_m, axis=1
                    )
                    # Run a token step so the stabilizer's ω tracks the
                    # current activity — this keeps stability_score
                    # meaningful across cycles.
                    self._stabilizer.step(
                        torque=[
                            proposed_q - pre_vec.Q,
                            proposed_m - pre_vec.M,
                            0.0,
                        ],
                        dt=1.0,
                    )
                    gyro_meta = {
                        "Q_pre": pre_vec.Q,
                        "Q_proposed": proposed_q,
                        "Q_applied": smoothed_q,
                        "M_pre": pre_vec.M,
                        "M_proposed": proposed_m,
                        "M_applied": smoothed_m,
                    }
                    self._state.update_quality(smoothed_q)
                    self._state.update_memory(smoothed_m)
                else:
                    gyro_meta = None
                    self._state.update_quality(proposed_q)
                    self._state.update_memory(proposed_m)

                self._state.tick_time()

                vector = self._state.snapshot()
                eval_result = TickResult(
                    tick=self._tick,
                    phase="EVALUATE",
                    state_vector=vector,
                    observation=observation,
                    energy_available=self._energy.available,
                )
                self._fire("after_evaluate", eval_result)

                # --- DECIDE ---
                act_decision = self._oracle.is_ready_to_act()

                # --- ACT ---
                if act_decision.is_ready:
                    action = {"choice": "a", "score": act_decision.score}
                    self._env.act(action)

                # --- REPLICATE? ---
                if attempt_replication and self._replicator is not None:
                    replicate_decision = self._oracle.is_ready_to_replicate()
                    if replicate_decision.is_ready:
                        result_stub.phase = "REPLICATE"
                        self._fire("before_replicate", result_stub)
                        replication_outcome = self._replicator.replicate()
                        if replication_outcome.success and replication_outcome.child_config:
                            child_id = replication_outcome.child_config.agent_id
                            self._children.append(child_id)
                            self._status.children.append(child_id)

        vector = self._state.snapshot()
        self._status.last_vector = vector
        gyro_stab = (
            self._stabilizer.stability_score()
            if (self._use_gyro and self._stabilizer is not None)
            else None
        )
        e8_sym = self._oracle.last_e8_score if self._oracle.use_e8_geometry else None
        result = TickResult(
            tick=self._tick,
            phase="DORMANT" if dormant else "ACT",
            state_vector=vector,
            act_decision=act_decision,
            replicate_decision=replicate_decision,
            replication_outcome=replication_outcome,
            observation=observation,
            action=action,
            energy_available=self._energy.available,
            dormant=dormant,
            notes=" | ".join(n for n in notes if n),
            gyro_damping_meta=gyro_meta,
            gyro_stability=gyro_stab,
            e8_symmetry=e8_sym,
        )
        self._fire("on_tick", result)
        return result

    # ------------------------------------------------------------------ #
    # Bulk execution
    # ------------------------------------------------------------------ #

    def run_cycles(
        self,
        n: int,
        *,
        attempt_replication: bool = True,
        stop_on_child: bool = False,
    ) -> RunSummary:
        """Run ``n`` ticks synchronously. Returns a summary."""
        halted = ""
        for _ in range(n):
            try:
                tr = self.tick(attempt_replication=attempt_replication)
            except KillSwitchEngagedError as exc:
                halted = f"kill_switch:{exc}"
                break
            if stop_on_child and tr.replication_outcome and tr.replication_outcome.success:
                halted = "first replication reached"
                break
        else:
            halted = "max_cycles_reached"

        return self._summary(halted)

    async def arun_cycles(
        self,
        n: int,
        *,
        attempt_replication: bool = True,
        stop_on_child: bool = False,
        tick_delay_sec: float = 0.0,
    ) -> RunSummary:
        """Async version of `run_cycles` \u2014 yields control between ticks."""
        halted = ""
        for _ in range(n):
            try:
                tr = self.tick(attempt_replication=attempt_replication)
            except KillSwitchEngagedError as exc:
                halted = f"kill_switch:{exc}"
                break
            if stop_on_child and tr.replication_outcome and tr.replication_outcome.success:
                halted = "first replication reached"
                break
            if tick_delay_sec > 0:
                await asyncio.sleep(tick_delay_sec)
        else:
            halted = "max_cycles_reached"
        return self._summary(halted)

    def shutdown(self, reason: str = "normal") -> None:
        """Graceful shutdown \u2014 fires on_shutdown hooks and writes final audit."""
        self._status.alive = False
        final = TickResult(
            tick=self._tick,
            phase="SHUTDOWN",
            state_vector=self._state.snapshot(),
            energy_available=self._energy.available,
            notes=reason,
        )
        self._fire("on_shutdown", final)
        self._governance.unregister_agent(self._cfg.agent_id, reason=reason)

    def _summary(self, halted: str) -> RunSummary:
        return RunSummary(
            agent_id=self._cfg.agent_id,
            ticks_run=self._tick,
            final_state=self._state.snapshot(),
            children_spawned=list(self._children),
            energy_consumed=self._energy.consumed,
            halted_reason=halted,
        )

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def status(self) -> AgentState:
        return self._status

    @property
    def state_engine(self) -> StateEngine:
        return self._state

    @property
    def energy_manager(self) -> EnergyManager:
        return self._energy

    @property
    def oracle(self) -> ConvergenceOracle:
        return self._oracle

    @property
    def governance(self) -> GovernanceLayer:
        return self._governance

    @property
    def replicator(self) -> AgentReplicator | None:
        return self._replicator

    @property
    def stabilizer(self) -> "GyroscopicStabilizer | None":
        """The `GyroscopicStabilizer` in use, if any (read-only)."""
        return self._stabilizer

    @property
    def use_gyroscopic_stability(self) -> bool:
        return self._use_gyro

    @property
    def tick_count(self) -> int:
        return self._tick


__all__ = [
    "ExponentialMovingAverageLearner",
    "HookCallback",
    "LearningLoop",
    "RunSummary",
    "SimpleReward",
    "TickResult",
]
