"""Unit tests for the SLSR SDK core modules.

Run with:
    pytest -q
"""

from __future__ import annotations

import math
import time

import pytest

import slsr
from slsr.agent_replicator import (
    AgentReplicator,
    GaussianNumericMutator,
    MutationOutOfBoundsError,
    NullInfrastructureAdapter,
)
from slsr.convergence_oracle import ConfigError, ConvergenceOracle, DefaultFitness
from slsr.energy_manager import (
    EnergyCapExceededError,
    EnergyManager,
    OutOfEnergyError,
)
from slsr.governance import (
    AuditLog,
    GovernanceLayer,
    KillSwitchEngagedError,
)
from slsr.learning_loop import (
    ExponentialMovingAverageLearner,
    LearningLoop,
    SimpleReward,
)
from slsr.models import (
    AgentConfig,
    EnergyConfig,
    GovernanceConfig,
    KillSwitchLevel,
    MutationManifest,
    OracleConfig,
    ReplicationConfig,
    ReplicationRequest,
    StateConfig,
    StateVector,
)
from slsr.state_engine import StateEngine


# --------------------------------------------------------------------------- #
# CONV_THRESHOLD constant
# --------------------------------------------------------------------------- #


class TestConvThreshold:
    def test_threshold_value(self):
        # Per SDK design doc, pinned at ~0.5792.
        assert 0.57 < slsr.CONV_THRESHOLD < 0.59
        assert slsr.CONV_THRESHOLD == pytest.approx(0.5792, abs=1e-4)

    def test_raw_transcendental_exposed(self):
        # The raw pi*cos(sqrt(e)) value is exposed for audit/documentation.
        expected_raw = math.pi * math.cos(math.sqrt(math.e))
        assert slsr.RAW_CONV == pytest.approx(expected_raw, abs=1e-9)
        assert slsr.RAW_CONV < 0  # it's negative in radians

    def test_oracle_class_exposes_same_constant(self):
        assert ConvergenceOracle.CANONICAL_THRESHOLD == slsr.CONV_THRESHOLD


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #


class TestModels:
    def test_state_vector_product(self):
        v = StateVector(Q=0.5, M=0.4, T=0.25)
        assert v.S == pytest.approx(0.5 * 0.4 * 0.25)

    def test_state_vector_bounds(self):
        with pytest.raises(Exception):
            StateVector(Q=1.1, M=0.0, T=0.0)

    def test_agent_config_defaults(self):
        cfg = AgentConfig()
        assert cfg.generation == 0
        assert cfg.parent_id is None
        assert "learning_rate" in cfg.hyperparameters

    def test_replication_request_structural_flag(self):
        req = ReplicationRequest(
            parent_id="p",
            parent_generation=0,
            proposed_generation=1,
            mutation=MutationManifest(structural=True),
            readiness_score=0.9,
            replication_cost_eu=10.0,
        )
        assert req.structural_mutation is True


# --------------------------------------------------------------------------- #
# EnergyManager
# --------------------------------------------------------------------------- #


class TestEnergyManager:
    def test_required_energy_formula(self):
        em = EnergyManager(EnergyConfig(reference_c_squared=4.0))
        # default: speed_factor None -> uses reference_c_squared directly
        assert em.required_energy(10.0) == pytest.approx(40.0)
        # explicit speed_factor
        assert em.required_energy(10.0, speed_factor=0.5) == pytest.approx(10.0 * 0.25)

    def test_reserve_commit_release(self):
        em = EnergyManager(EnergyConfig(initial_eu=100, max_cap=200, refill_rate=0))
        res = em.reserve(20, "test")
        assert em.available == pytest.approx(80)
        assert em.reserved == pytest.approx(20)
        em.commit(res)
        assert em.consumed == pytest.approx(20)
        assert em.reserved == 0

        res2 = em.reserve(30, "test2")
        em.release(res2)
        assert em.consumed == pytest.approx(20)  # unchanged
        assert em.available == pytest.approx(80)

    def test_out_of_energy(self):
        em = EnergyManager(EnergyConfig(initial_eu=5, max_cap=5, refill_rate=0))
        with pytest.raises(OutOfEnergyError):
            em.reserve(10, "too big")

    def test_initial_above_cap_rejected(self):
        with pytest.raises(EnergyCapExceededError):
            EnergyManager(EnergyConfig(initial_eu=10, max_cap=5))

    def test_replication_affordability(self):
        em = EnergyManager(
            EnergyConfig(
                initial_eu=500,
                max_cap=1000,
                refill_rate=0,
                replication_cost_eu=100,
                replication_energy_floor=200,
            )
        )
        assert em.check_replication_affordable() is True
        em.reserve(350, "burn")
        # now only 150 available; below floor of 200
        assert em.check_replication_affordable() is False
        with pytest.raises(OutOfEnergyError):
            em.reserve_replication()

    def test_refill_accrues_capped(self):
        clock = [0.0]
        em = EnergyManager(
            EnergyConfig(initial_eu=0, max_cap=10, refill_rate=100),
            time_fn=lambda: clock[0],
        )
        clock[0] += 1.0
        assert em.available == pytest.approx(10.0)  # capped at max_cap


# --------------------------------------------------------------------------- #
# StateEngine
# --------------------------------------------------------------------------- #


class TestStateEngine:
    def test_product_equals_s(self):
        clock = [0.0]
        se = StateEngine(
            StateConfig(maturation_horizon_sec=10),
            genesis_ts=0.0,
            time_fn=lambda: clock[0],
        )
        se.update_quality(0.8)
        se.update_memory(0.5)
        clock[0] = 5.0
        se.tick_time()
        v = se.snapshot()
        assert v.S == pytest.approx(v.Q * v.M * v.T)
        assert v.T == pytest.approx(0.5)

    def test_t_saturates_at_one(self):
        clock = [0.0]
        se = StateEngine(
            StateConfig(maturation_horizon_sec=1),
            genesis_ts=0.0,
            time_fn=lambda: clock[0],
        )
        clock[0] = 100.0
        se.tick_time()
        assert se.snapshot().T == 1.0

    def test_clamping(self):
        se = StateEngine(StateConfig(maturation_horizon_sec=10))
        se.update_quality(1.5)
        se.update_memory(-3.0)
        v = se.snapshot()
        assert v.Q == 1.0
        assert v.M == 0.0

    def test_trajectory(self):
        se = StateEngine(StateConfig(maturation_horizon_sec=10, trajectory_window=5))
        for i in range(10):
            se.update_quality(i / 10.0)
        traj = se.trajectory(5)
        # length is capped by window
        assert 1 <= len(traj) <= 5


# --------------------------------------------------------------------------- #
# ConvergenceOracle
# --------------------------------------------------------------------------- #


class TestConvergenceOracle:
    def _make(self, Q, M, T, replication_floor=0.85, stochastic=False):
        clock = [0.0]
        se = StateEngine(
            StateConfig(maturation_horizon_sec=1),
            genesis_ts=0.0,
            time_fn=lambda: clock[0],
        )
        se.update_quality(Q)
        se.update_memory(M)
        clock[0] = T  # T = clock / maturation_horizon_sec
        se.tick_time()
        return ConvergenceOracle(
            se,
            config=OracleConfig(
                replication_floor=replication_floor, stochastic_gate=stochastic
            ),
        )

    def test_below_canonical_threshold_is_not_ready(self):
        o = self._make(0.1, 0.1, 0.1)  # S = 0.001
        d = o.is_ready_to_act()
        assert d.is_ready is False
        assert d.threshold == pytest.approx(slsr.CONV_THRESHOLD)

    def test_above_canonical_but_below_replication_floor(self):
        # Find (Q, M, T) with S between threshold (~0.58) and 0.85
        o = self._make(0.95, 0.95, 0.75)  # S = 0.677
        act = o.is_ready_to_act()
        rep = o.is_ready_to_replicate()
        assert act.is_ready is True
        assert rep.is_ready is False

    def test_above_replication_floor(self):
        o = self._make(0.98, 0.95, 0.95)  # S = 0.884
        rep = o.is_ready_to_replicate()
        assert rep.is_ready is True

    def test_replication_floor_below_canonical_rejected(self):
        se = StateEngine(StateConfig())
        with pytest.raises(ConfigError):
            ConvergenceOracle(se, config=OracleConfig(replication_floor=0.1))

    def test_default_fitness_returns_s(self):
        v = StateVector(Q=0.5, M=0.5, T=0.5)
        assert DefaultFitness().score(v) == pytest.approx(0.125)


# --------------------------------------------------------------------------- #
# GovernanceLayer / Audit log
# --------------------------------------------------------------------------- #


class TestGovernance:
    def test_audit_chain_integrity(self):
        log = AuditLog()
        log.append("a1", "born")
        log.append("a1", "learn")
        log.append("a1", "act", score=0.9)
        assert len(log) == 3
        assert log.verify_chain() is True

    def test_audit_chain_tamper_detection(self):
        log = AuditLog()
        log.append("a1", "born")
        e = log.append("a1", "act")
        # tamper: mutate payload after the fact
        e.payload["injected"] = True  # type: ignore[index]
        assert log.verify_chain() is False

    def test_kill_switch_terminate_blocks(self):
        gov = GovernanceLayer(GovernanceConfig(audit_log_path=""))
        gov.engage_kill_switch(KillSwitchLevel.TERMINATE, "drill")
        with pytest.raises(KillSwitchEngagedError):
            gov.enforce_kill_switch("a1")

    def test_kill_switch_pause_does_not_block(self):
        gov = GovernanceLayer(GovernanceConfig(audit_log_path=""))
        gov.engage_kill_switch(KillSwitchLevel.PAUSE, "advisory")
        # PAUSE is advisory \u2014 enforce does not raise
        gov.enforce_kill_switch("a1")

    def test_replication_policy_blocks_generation_over_cap(self):
        gov = GovernanceLayer(GovernanceConfig(audit_log_path=""))
        gov.register_agent("p", 5, None)
        req = ReplicationRequest(
            parent_id="p",
            parent_generation=5,
            proposed_generation=6,
            mutation=MutationManifest(),
            readiness_score=0.9,
            replication_cost_eu=10.0,
        )
        decision = gov.evaluate_replication_request(
            req,
            max_generations=5,
            fleet_cap=32,
            allow_structural_mutation=False,
        )
        assert decision.allowed is False
        assert "max_generations" in decision.reason

    def test_structural_mutation_blocked_without_allow(self):
        gov = GovernanceLayer(
            GovernanceConfig(audit_log_path="", approval_mode="disabled")
        )
        gov.register_agent("p", 0, None)
        req = ReplicationRequest(
            parent_id="p",
            parent_generation=0,
            proposed_generation=1,
            mutation=MutationManifest(structural=True),
            readiness_score=0.9,
            replication_cost_eu=10.0,
        )
        decision = gov.evaluate_replication_request(
            req, max_generations=5, fleet_cap=32, allow_structural_mutation=False
        )
        assert decision.allowed is False


# --------------------------------------------------------------------------- #
# AgentReplicator
# --------------------------------------------------------------------------- #


def _make_replicator_setup(parent_gen: int = 0, replication_floor: float = 0.85):
    """Helper to build a fully-wired replicator + oracle + energy for tests."""
    clock = [0.0]
    cfg = AgentConfig(
        agent_id="parent-001",
        generation=parent_gen,
        hyperparameters={"learning_rate": 0.01, "epsilon": 0.1, "batch_size": 32.0},
    )
    se = StateEngine(
        StateConfig(maturation_horizon_sec=1),
        genesis_ts=0.0,
        time_fn=lambda: clock[0],
    )
    # push state above replication floor
    se.update_quality(0.98)
    se.update_memory(0.95)
    clock[0] = 0.95
    se.tick_time()
    oracle = ConvergenceOracle(
        se, config=OracleConfig(replication_floor=replication_floor)
    )
    energy = EnergyManager(
        EnergyConfig(
            initial_eu=1000,
            max_cap=2000,
            refill_rate=0,
            replication_cost_eu=100,
            replication_energy_floor=200,
        )
    )
    gov = GovernanceLayer(
        GovernanceConfig(audit_log_path="", approval_mode="disabled")
    )
    gov.register_agent(cfg.agent_id, cfg.generation, cfg.parent_id)
    replicator = AgentReplicator(
        cfg,
        oracle=oracle,
        energy=energy,
        governance=gov,
        config=ReplicationConfig(max_generations=3, mutation_sigma=0.1),
        adapter=NullInfrastructureAdapter(),
    )
    return replicator, cfg, oracle, energy, gov


class TestAgentReplicator:
    def test_successful_replication(self):
        rep, cfg, *_ = _make_replicator_setup()
        outcome = rep.replicate()
        assert outcome.success is True
        assert outcome.child_config is not None
        assert outcome.child_config.generation == cfg.generation + 1
        assert outcome.child_config.parent_id == cfg.agent_id

    def test_mutation_bounded(self):
        rep, cfg, *_ = _make_replicator_setup()
        outcome = rep.replicate()
        assert outcome.success
        mutation = outcome.mutation
        for k, delta in mutation.numeric_deltas.items():
            parent_val = cfg.hyperparameters[k]
            assert abs(delta) <= abs(rep.config.mutation_sigma * parent_val) + 1e-9

    def test_generation_cap_blocks_replication(self):
        rep, *_ = _make_replicator_setup(parent_gen=3)
        outcome = rep.replicate()
        # parent already at max_generations=3
        assert outcome.success is False
        assert "generation" in outcome.reason.lower()

    def test_lineage_tracking(self):
        rep, cfg, *_ = _make_replicator_setup()
        outcome = rep.replicate()
        assert outcome.success
        lineage = rep.lineage()
        assert cfg.agent_id in lineage
        assert outcome.child_config.agent_id in lineage
        assert outcome.child_config.agent_id in lineage[cfg.agent_id].children

    def test_out_of_bounds_mutation_rejected(self):
        rep, cfg, *_ = _make_replicator_setup()
        bad = MutationManifest(numeric_deltas={"learning_rate": 1.0})  # way > sigma
        with pytest.raises(MutationOutOfBoundsError):
            rep._validate_mutation(bad, cfg)

    def test_structural_mutation_rejected_by_default(self):
        rep, cfg, *_ = _make_replicator_setup()
        bad = MutationManifest(structural=True)
        with pytest.raises(MutationOutOfBoundsError):
            rep._validate_mutation(bad, cfg)

    def test_unaffordable_energy_refuses_replication(self):
        rep, cfg, oracle, energy, gov = _make_replicator_setup()
        # burn energy below the floor
        energy.reserve(900, "burn")
        outcome = rep.replicate()
        assert outcome.success is False

    def test_mutator_respects_allowlist(self):
        import random as _r

        mutator = GaussianNumericMutator(
            sigma=0.05, allowlist=["learning_rate"], rng=_r.Random(0)
        )
        cfg = AgentConfig(hyperparameters={"learning_rate": 0.1, "forbidden": 5.0})
        m = mutator.propose_mutation(cfg)
        assert set(m.numeric_deltas.keys()) == {"learning_rate"}


# --------------------------------------------------------------------------- #
# LearningLoop
# --------------------------------------------------------------------------- #


class TestLearningLoop:
    def _build_loop(self, **overrides):
        cfg = AgentConfig()
        cfg.state.maturation_horizon_sec = 0.1  # ramp T quickly
        cfg.governance.audit_log_path = ""
        cfg.governance.approval_mode = "disabled"
        energy = EnergyManager(
            EnergyConfig(initial_eu=500, max_cap=2000, refill_rate=0)
        )
        state = StateEngine(cfg.state)
        oracle = ConvergenceOracle(state, config=cfg.oracle)
        gov = GovernanceLayer(cfg.governance)
        loop = LearningLoop(
            cfg,
            energy=energy,
            state=state,
            oracle=oracle,
            governance=gov,
            replicator=None,
            environment=SimpleReward(seed=1),
            learner=ExponentialMovingAverageLearner(alpha=0.5),
            tick_energy_cost=5.0,
            memory_growth_rate=0.2,
        )
        return loop

    def test_runs_and_updates_state(self):
        loop = self._build_loop()
        summary = loop.run_cycles(10, attempt_replication=False)
        assert summary.ticks_run == 10
        # After several cycles Q should be > 0
        assert summary.final_state.Q > 0.0
        assert summary.final_state.M > 0.0
        assert summary.final_state.T > 0.0

    def test_dormancy_on_low_energy(self):
        loop = self._build_loop()
        # Drain most energy first
        res = loop.energy_manager.reserve(485, "drain")
        loop.energy_manager.commit(res)
        summary = loop.run_cycles(3, attempt_replication=False)
        # Should still complete all 3 ticks even when dormant
        assert summary.ticks_run == 3

    def test_kill_switch_halts_run(self):
        loop = self._build_loop()
        loop.governance.engage_kill_switch(KillSwitchLevel.TERMINATE, "test")
        summary = loop.run_cycles(5, attempt_replication=False)
        assert "kill_switch" in summary.halted_reason

    def test_hooks_called(self):
        loop = self._build_loop()
        calls = {"tick": 0, "eval": 0}

        def _tick(_l, _r):
            calls["tick"] += 1

        def _eval(_l, _r):
            calls["eval"] += 1

        loop.on_tick(_tick)
        loop.after_evaluate(_eval)
        loop.run_cycles(3, attempt_replication=False)
        assert calls["tick"] == 3
        assert calls["eval"] >= 1
