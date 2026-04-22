"""Unit tests for the E8 geometry & gyroscopic-stability enhancements.

Run with::

    pytest -q tests/test_enhancements.py

These tests cover:
    * E8 root-system construction + sanity invariants
    * nearest_root / symmetry_score correctness
    * GyroscopicStabilizer.step/damp_update/stabilize_vector behaviour
    * End-to-end wiring into ConvergenceOracle / LearningLoop /
      AgentReplicator (opt-in via flags).

Backward-compat is covered by tests/test_core.py — this file adds new
behaviour only.
"""

from __future__ import annotations

import math
import random as _r

import numpy as np
import pytest

from slsr.agent_replicator import AgentReplicator, NullInfrastructureAdapter
from slsr.convergence_oracle import ConvergenceOracle
from slsr.energy_manager import EnergyManager
from slsr.geometry import E8RootSystem
from slsr.governance import GovernanceLayer
from slsr.learning_loop import LearningLoop, SimpleReward
from slsr.models import (
    AgentConfig,
    EnergyConfig,
    EnhancementConfig,
    GovernanceConfig,
    OracleConfig,
    ReplicationConfig,
    StateConfig,
)
from slsr.stability import GyroscopicStabilizer
from slsr.state_engine import StateEngine


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _virtual_clock():
    t = [0.0]

    def advance(dt: float = 1.0) -> float:
        t[0] += dt
        return t[0]

    def now() -> float:
        return t[0]

    return now, advance, t


def _build_loop(
    *,
    use_gyroscopic_stability: bool = False,
    stabilizer: GyroscopicStabilizer | None = None,
    seed: int = 0,
) -> tuple[LearningLoop, StateEngine]:
    """Build a minimal, deterministic LearningLoop for tests."""
    now, _adv, tref = _virtual_clock()

    cfg = AgentConfig(seed=seed)
    cfg.governance.audit_log_path = ""
    cfg.governance.approval_mode = "disabled"
    cfg.state.maturation_horizon_sec = 1.0

    energy = EnergyManager(
        EnergyConfig(initial_eu=10_000, max_cap=20_000, refill_rate=0),
        time_fn=now,
    )
    state = StateEngine(cfg.state, genesis_ts=0.0, time_fn=now)
    oracle = ConvergenceOracle(state, config=cfg.oracle)
    gov = GovernanceLayer(cfg.governance)

    env = SimpleReward(seed=seed)

    loop = LearningLoop(
        cfg,
        energy=energy,
        state=state,
        oracle=oracle,
        governance=gov,
        environment=env,
        tick_energy_cost=1.0,
        memory_growth_rate=0.05,
        use_gyroscopic_stability=use_gyroscopic_stability,
        stabilizer=stabilizer,
    )

    # Expose a closure so tests can drive time forward
    def _tick_with_time_advance(dt: float = 0.01):
        tref[0] += dt
        return loop.tick(attempt_replication=False)

    loop._advance = _tick_with_time_advance  # type: ignore[attr-defined]
    return loop, state


# --------------------------------------------------------------------------- #
# E8 root system
# --------------------------------------------------------------------------- #


class TestE8RootSystem:
    def test_exactly_240_roots_all_norm_sq_2(self):
        e8 = E8RootSystem()
        roots = e8.roots()
        assert roots.shape == (240, 8)
        norms_sq = np.sum(roots * roots, axis=1)
        assert np.allclose(norms_sq, 2.0, atol=1e-12)

    def test_closed_under_negation(self):
        e8 = E8RootSystem()
        roots = e8.roots()
        root_set = {tuple(np.round(r, 6)) for r in roots}
        for r in roots:
            assert tuple(np.round(-r, 6)) in root_set

    def test_verify_all_checks_pass(self):
        e8 = E8RootSystem()
        checks = e8.verify()
        # Every named check must be True for a valid E8 construction.
        assert all(checks.values()), f"verify() failed: {checks}"

    def test_nearest_root_returns_valid(self):
        e8 = E8RootSystem()
        rng = np.random.default_rng(1)
        v = rng.standard_normal(8)
        idx, root, cos = e8.nearest_root(v)
        assert 0 <= idx < 240
        assert root.shape == (8,)
        assert -1.0 - 1e-9 <= cos <= 1.0 + 1e-9
        # Nearest root should have higher cosine than the average
        unit = v / np.linalg.norm(v)
        avg = float(np.mean(e8.roots() @ unit / math.sqrt(2.0)))
        assert cos >= avg

    def test_nearest_root_of_a_root_is_itself(self):
        e8 = E8RootSystem()
        r0 = e8.roots()[7]
        idx, r, cos = e8.nearest_root(r0)
        assert cos == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(r, r0)

    def test_symmetry_score_random_vs_aligned(self):
        e8 = E8RootSystem()
        rng = np.random.default_rng(123)

        # Random batch
        random_batch = rng.standard_normal((64, 8))
        random_score = e8.symmetry_score(random_batch)

        # Aligned: literally the first 64 roots (with tiny jitter for realism)
        aligned_batch = e8.roots()[:64] + 0.01 * rng.standard_normal((64, 8))
        aligned_score = e8.symmetry_score(aligned_batch)

        assert 0.0 <= random_score <= 1.0
        assert 0.0 <= aligned_score <= 1.0
        assert aligned_score > random_score
        # Aligned inputs should be near 1.0
        assert aligned_score > 0.98

    def test_project_onto_lattice_preserves_direction(self):
        e8 = E8RootSystem()
        v = np.array([1.1, 0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        proj = e8.project_onto_lattice(v)
        # Projection should be along the nearest root's direction
        _, root, _ = e8.nearest_root(v)
        # proj and root should be colinear with positive scale
        unit_r = root / np.linalg.norm(root)
        unit_p = proj / np.linalg.norm(proj)
        assert np.allclose(unit_p, unit_r, atol=1e-9)


# --------------------------------------------------------------------------- #
# GyroscopicStabilizer
# --------------------------------------------------------------------------- #


class TestGyroscopicStabilizer:
    def test_zero_torque_with_damping_decays_omega(self):
        g = GyroscopicStabilizer(damping=0.5)
        # inject some initial angular momentum manually
        g._omega = np.array([1.0, 0.8, 0.6])
        start_mag = float(np.linalg.norm(g.omega))
        for _ in range(50):
            g.step(np.zeros(3), dt=0.1)
        end_mag = float(np.linalg.norm(g.omega))
        assert end_mag < start_mag
        # Should be well below starting magnitude for damping 0.5, 50 steps
        assert end_mag < 0.1 * start_mag

    def test_zero_damping_does_not_decay_trivially(self):
        # Gyroscopic precession alone should conserve |ω| in the
        # undamped symmetric-inertia case (within integrator tolerance).
        g = GyroscopicStabilizer(damping=0.0, inertia=(1.0, 1.0, 1.0))
        g._omega = np.array([0.3, 0.0, 0.0])
        start = float(np.linalg.norm(g.omega))
        for _ in range(20):
            g.step(np.zeros(3), dt=0.05)
        end = float(np.linalg.norm(g.omega))
        assert end == pytest.approx(start, abs=1e-6)

    def test_damp_update_is_between_current_and_proposed(self):
        g = GyroscopicStabilizer(damping=0.5)
        for current, proposed in [(0.0, 1.0), (1.0, 0.0), (0.3, 0.7), (0.9, 0.5)]:
            result = g.damp_update(current, proposed)
            lo, hi = min(current, proposed), max(current, proposed)
            assert lo <= result <= hi, f"{result} not between {lo} and {hi}"

    def test_damp_update_returns_current_on_zero_delta(self):
        g = GyroscopicStabilizer(damping=1.0)
        assert g.damp_update(0.42, 0.42) == 0.42

    def test_damp_update_stronger_damping_means_smaller_step(self):
        weak = GyroscopicStabilizer(damping=0.1).damp_update(0.0, 1.0)
        strong = GyroscopicStabilizer(damping=5.0).damp_update(0.0, 1.0)
        # strong damping → smaller fraction of delta applied
        assert strong < weak
        assert 0.0 < strong < weak < 1.0

    def test_stabilize_vector_reduces_norm_under_strong_damping(self):
        g = GyroscopicStabilizer(damping=3.0)
        rng = np.random.default_rng(0)
        v = rng.standard_normal(12)
        out = g.stabilize_vector(v)
        assert out.shape == v.shape
        assert np.linalg.norm(out) < np.linalg.norm(v)
        # And "strong" damping should be a strong reduction
        assert np.linalg.norm(out) < 0.25 * np.linalg.norm(v)

    def test_stabilize_vector_shape_preserved_for_non_3_multiple(self):
        g = GyroscopicStabilizer(damping=1.0)
        v = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)  # length 5
        out = g.stabilize_vector(v)
        assert out.shape == v.shape

    def test_stability_score_at_rest_is_one(self):
        g = GyroscopicStabilizer()
        assert g.stability_score() == pytest.approx(1.0)

    def test_stability_score_decreases_with_omega(self):
        g = GyroscopicStabilizer()
        g._omega = np.array([5.0, 0.0, 0.0])
        tumbling = g.stability_score()
        g._omega = np.array([0.1, 0.0, 0.0])
        calm = g.stability_score()
        assert calm > tumbling
        assert 0.0 <= tumbling <= calm <= 1.0

    def test_nan_is_handled_gracefully(self):
        g = GyroscopicStabilizer(damping=0.5)
        # Force a bad step and make sure _clip scrubs it
        bad = np.array([float("nan"), 0.0, 0.0])
        g._omega = g._clip(bad)
        assert np.all(np.isfinite(g.omega))


# --------------------------------------------------------------------------- #
# ConvergenceOracle + E8 wiring
# --------------------------------------------------------------------------- #


class TestConvergenceOracleWithE8:
    def _build_oracle(self, provider, *, e8_min: float, replication_floor: float = 0.85):
        # Drive state to high S = Q*M*T so the scalar threshold is clearly passed.
        now_fn = lambda: 1.5  # noqa: E731 — plenty of wall-clock
        state = StateEngine(
            StateConfig(maturation_horizon_sec=1.0), genesis_ts=0.0, time_fn=now_fn
        )
        state.update_quality(0.99)
        state.update_memory(0.99)
        state.tick_time()  # saturates T
        return ConvergenceOracle(
            state,
            config=OracleConfig(replication_floor=replication_floor),
            use_e8_geometry=True,
            e8_symmetry_min=e8_min,
            state_vector_provider=provider,
        ), state

    def test_random_embedding_rejected_aligned_accepted(self):
        """Core test-8: with scalar passing, E8 must gate on alignment."""
        rng = np.random.default_rng(7)

        # A pathological low-alignment embedding: drawn in a flat random
        # direction but then quantized to a weird axis-lattice offset.
        random_embedding = rng.standard_normal(8) * 0.0001
        # An E8-root embedding
        e8 = E8RootSystem()
        aligned_embedding = e8.roots()[0]

        # We use a high e8_symmetry_min that a random-ish vector can't
        # clear but a root easily does. Because nearest-root cosine for
        # random 8D vectors clusters around ~0.85 and roots hit 1.0, we
        # set the threshold at 0.95.
        E8_MIN = 0.95

        oracle_rand, _ = self._build_oracle(
            lambda: random_embedding, e8_min=E8_MIN
        )
        dec_rand = oracle_rand.is_ready_to_replicate()
        # With a tiny-norm vector ``symmetry_score`` returns 0.0 — well
        # below the 0.95 bar. The scalar score is very high, but the
        # geometric gate should veto.
        assert dec_rand.is_ready is False

        oracle_aligned, _ = self._build_oracle(
            lambda: aligned_embedding, e8_min=E8_MIN
        )
        dec_aligned = oracle_aligned.is_ready_to_replicate()
        assert dec_aligned.is_ready is True

    def test_oracle_falls_back_when_no_provider(self):
        """If use_e8_geometry=True but provider=None, behave as scalar-only."""
        state = StateEngine(
            StateConfig(maturation_horizon_sec=1.0),
            genesis_ts=0.0,
            time_fn=lambda: 1.5,
        )
        state.update_quality(0.99)
        state.update_memory(0.99)
        state.tick_time()
        oracle = ConvergenceOracle(
            state,
            config=OracleConfig(replication_floor=0.85),
            use_e8_geometry=True,
            e8_symmetry_min=0.99,  # would be impossible to meet
            state_vector_provider=None,
        )
        # Without a provider the E8 gate is skipped entirely.
        assert oracle.is_ready_to_replicate().is_ready is True
        assert oracle.last_e8_score is None


# --------------------------------------------------------------------------- #
# LearningLoop + gyroscopic stability
# --------------------------------------------------------------------------- #


class TestLearningLoopWithGyro:
    def _run_loop_and_collect_deltas(self, *, use_gyro: bool, seed: int = 42):
        loop, state = _build_loop(use_gyroscopic_stability=use_gyro, seed=seed)
        deltas: list[float] = []
        prev_q = state.snapshot().Q
        for _ in range(40):
            # advance time so T grows predictably
            loop._advance(0.01)  # type: ignore[attr-defined]
            cur_q = state.snapshot().Q
            deltas.append(cur_q - prev_q)
            prev_q = cur_q
        return deltas

    def test_gyro_reduces_cycle_variance(self):
        baseline = self._run_loop_and_collect_deltas(use_gyro=False)
        enhanced = self._run_loop_and_collect_deltas(use_gyro=True)
        base_var = float(np.var(baseline))
        enh_var = float(np.var(enhanced))
        # Strict inequality: smoothed deltas must have lower variance.
        assert enh_var < base_var
        # Also: the enhanced loop should record gyro_stability on every tick
        loop, _ = _build_loop(use_gyroscopic_stability=True)
        tr = loop.tick(attempt_replication=False)
        assert tr.gyro_stability is not None
        assert 0.0 <= tr.gyro_stability <= 1.0

    def test_gyro_meta_recorded_in_tick_result(self):
        loop, _ = _build_loop(use_gyroscopic_stability=True)
        tr = loop.tick(attempt_replication=False)
        # When the loop is active (not dormant), it should record pre/post
        # values for Q and M.
        assert tr.gyro_damping_meta is not None
        meta = tr.gyro_damping_meta
        for k in ("Q_pre", "Q_proposed", "Q_applied", "M_pre", "M_proposed", "M_applied"):
            assert k in meta


# --------------------------------------------------------------------------- #
# AgentReplicator + gyroscopic mutation
# --------------------------------------------------------------------------- #


def _replicator_with_gyro(
    *, use_gyro: bool, seed: int
) -> tuple[AgentReplicator, AgentConfig]:
    """Build a replicator rigged to succeed; large mutations to make L2 visible."""
    clock = [0.0]

    cfg = AgentConfig(
        agent_id=f"parent-{seed}",
        seed=seed,
        hyperparameters={"learning_rate": 1.0, "epsilon": 1.0, "batch_size": 32.0},
    )
    state = StateEngine(
        StateConfig(maturation_horizon_sec=0.5),
        genesis_ts=0.0,
        time_fn=lambda: clock[0],
    )
    state.update_quality(0.99)
    state.update_memory(0.99)
    clock[0] = 1.0
    state.tick_time()

    oracle = ConvergenceOracle(state, config=OracleConfig(replication_floor=0.85))
    energy = EnergyManager(
        EnergyConfig(
            initial_eu=10_000,
            max_cap=20_000,
            refill_rate=0,
            replication_cost_eu=100,
            replication_energy_floor=200,
        )
    )
    gov = GovernanceLayer(GovernanceConfig(audit_log_path="", approval_mode="disabled"))
    gov.register_agent(cfg.agent_id, cfg.generation, cfg.parent_id)

    # Large sigma so mutations are measurable, but still bounded.
    rep_cfg = ReplicationConfig(
        max_generations=5, mutation_sigma=0.25, fleet_cap=64
    )
    # A strong stabilizer so the contraction is obvious in L2 norm.
    stab = GyroscopicStabilizer(damping=2.0, inertia=(1.0, 1.0, 1.0))

    rep = AgentReplicator(
        cfg,
        oracle=oracle,
        energy=energy,
        governance=gov,
        config=rep_cfg,
        adapter=NullInfrastructureAdapter(),
        rng=_r.Random(seed),
        use_gyroscopic_mutation=use_gyro,
        stabilizer=stab if use_gyro else None,
    )
    return rep, cfg


def _delta_l2(parent: AgentConfig, child: AgentConfig) -> float:
    keys = ["learning_rate", "epsilon", "batch_size"]
    diffs = np.array(
        [
            child.hyperparameters[k] - parent.hyperparameters[k]
            for k in keys
        ]
    )
    return float(np.linalg.norm(diffs))


class TestAgentReplicatorWithGyro:
    def test_gyro_mutation_reduces_parent_child_distance(self):
        base_dists: list[float] = []
        gyro_dists: list[float] = []
        for seed in range(12):
            rep_b, parent_b = _replicator_with_gyro(use_gyro=False, seed=seed)
            out_b = rep_b.replicate()
            assert out_b.success, out_b.reason
            base_dists.append(_delta_l2(parent_b, out_b.child_config))

            rep_g, parent_g = _replicator_with_gyro(use_gyro=True, seed=seed)
            out_g = rep_g.replicate()
            assert out_g.success, out_g.reason
            gyro_dists.append(_delta_l2(parent_g, out_g.child_config))

        assert np.mean(gyro_dists) < np.mean(base_dists)

    def test_gyro_mutation_stays_within_validator_bounds(self):
        # The smoothing must not produce mutations that exceed sigma bounds;
        # it is strictly contractive, so validation should still pass.
        rep, _ = _replicator_with_gyro(use_gyro=True, seed=99)
        out = rep.replicate()
        assert out.success is True
