"""Demo: SLSR SDK with the E8 geometry & gyroscopic stability enhancements.

What this script demonstrates
-----------------------------
1. Instantiates an agent with **all three enhancements** enabled:
       * ``use_e8_geometry=True``       (ConvergenceOracle)
       * ``use_gyroscopic_stability=True`` (LearningLoop)
       * ``use_gyroscopic_mutation=True``  (AgentReplicator)
2. Runs the learning loop for ~30 cycles, printing E8 symmetry and
   gyroscopic-stability scores each cycle.
3. Attempts replication and shows the enhanced convergence check
   (scalar ``π·cos(√e)`` threshold *and* E8 geometric score).
4. Runs a paired baseline (enhancements OFF) with the same seed and
   prints a comparison table for:
       (a) per-cycle Q-delta variance,
       (b) average parent→child L2 mutation magnitude,
       (c) replication decisions (ready / not-ready).

Run it from the SDK root with::

    python examples/demo_enhanced.py
"""

from __future__ import annotations

import math
import random as _r
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slsr.agent_replicator import AgentReplicator, NullInfrastructureAdapter  # noqa: E402
from slsr.convergence_oracle import ConvergenceOracle  # noqa: E402
from slsr.energy_manager import EnergyManager  # noqa: E402
from slsr.geometry import E8RootSystem  # noqa: E402
from slsr.governance import GovernanceLayer  # noqa: E402
from slsr.learning_loop import LearningLoop  # noqa: E402
from slsr.models import (  # noqa: E402
    AgentConfig,
    ApprovalMode,
    EnergyConfig,
    GovernanceConfig,
    OracleConfig,
    ReplicationConfig,
    StateConfig,
)
from slsr.stability import GyroscopicStabilizer  # noqa: E402
from slsr.state_engine import StateEngine  # noqa: E402

# --------------------------------------------------------------------------- #
# Demo infrastructure
# --------------------------------------------------------------------------- #


class FastRewardEnv:
    """Demo env where reward saturates quickly so the agent reliably
    crosses the 0.85 replication floor within ~20 ticks."""

    def __init__(self, seed: int = 0):
        self._rng = _r.Random(seed)
        self._step = 0

    def observe(self):
        self._step += 1
        base = 1.0 - math.exp(-self._step / 6.0)
        noise = self._rng.gauss(0.0, 0.08)  # a little noise → oscillation
        reward = max(0.0, min(1.0, base + noise))
        return {"step": self._step, "reward": reward}

    def act(self, action):
        return None


class E8EmbeddingProvider:
    """Produces an 8-D embedding from the agent's current state.

    The embedding is a linear combination of a fixed E8 root direction
    and a small random perturbation — so an agent that has "learned"
    stays mostly aligned with the lattice, while early-stage agents look
    random.
    """

    def __init__(self, state_engine: StateEngine, anchor_idx: int = 0, seed: int = 1):
        self._engine = state_engine
        self._anchor = E8RootSystem().roots()[anchor_idx]
        self._rng = np.random.default_rng(seed)

    def __call__(self) -> np.ndarray:
        vec = self._engine.snapshot()
        # Alignment grows with Q*M; noise shrinks correspondingly.
        alignment = float(vec.Q * vec.M)
        noise = (1.0 - alignment) * self._rng.standard_normal(8) * 0.8
        return alignment * self._anchor + noise


# --------------------------------------------------------------------------- #
# Pretty-printing
# --------------------------------------------------------------------------- #


def hr(title: str = "") -> None:
    bar = "=" * 78
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def fmt_score(x: float | None) -> str:
    return "  —  " if x is None else f"{x:.4f}"


# --------------------------------------------------------------------------- #
# Build a stack of modules (enhanced or baseline)
# --------------------------------------------------------------------------- #


def build_stack(
    *,
    enhanced: bool,
    seed: int = 7,
    virtual_dt: float = 0.05,
) -> dict:
    """Wire up a complete LearningLoop + replicator, with a virtual clock."""
    clock = [0.0]

    def now() -> float:
        return clock[0]

    def advance(dt: float = virtual_dt) -> None:
        clock[0] += dt

    cfg = AgentConfig(
        agent_id=f"agent-{'enh' if enhanced else 'base'}-{seed}",
        seed=seed,
        hyperparameters={"learning_rate": 0.05, "epsilon": 0.1, "batch_size": 32.0},
    )
    cfg.state.maturation_horizon_sec = 1.0
    cfg.governance.audit_log_path = ""
    cfg.governance.approval_mode = ApprovalMode.DISABLED
    cfg.enhancements.use_e8_geometry = enhanced
    cfg.enhancements.use_gyroscopic_stability = enhanced
    cfg.enhancements.use_gyroscopic_mutation = enhanced
    cfg.enhancements.e8_symmetry_min = 0.85
    cfg.enhancements.gyro_damping = 0.3

    energy = EnergyManager(
        EnergyConfig(
            initial_eu=10_000,
            max_cap=20_000,
            refill_rate=1.0,
            replication_cost_eu=100,
            replication_energy_floor=200,
        ),
        time_fn=now,
    )
    state = StateEngine(cfg.state, genesis_ts=0.0, time_fn=now)

    embedder = E8EmbeddingProvider(state, anchor_idx=3, seed=seed)

    oracle = ConvergenceOracle(
        state,
        config=OracleConfig(replication_floor=0.85),
        use_e8_geometry=enhanced,
        e8_symmetry_min=cfg.enhancements.e8_symmetry_min,
        state_vector_provider=embedder if enhanced else None,
    )
    governance = GovernanceLayer(cfg.governance)

    # Shared stabilizer, used by both loop and replicator when enabled.
    stabilizer = GyroscopicStabilizer(
        inertia=cfg.enhancements.gyro_inertia,
        damping=cfg.enhancements.gyro_damping,
    )

    replicator = AgentReplicator(
        cfg,
        oracle=oracle,
        energy=energy,
        governance=governance,
        config=ReplicationConfig(
            max_generations=5, mutation_sigma=0.25, fleet_cap=32
        ),
        adapter=NullInfrastructureAdapter(),
        rng=_r.Random(seed + 1),
        use_gyroscopic_mutation=enhanced,
        stabilizer=stabilizer if enhanced else None,
    )

    loop = LearningLoop(
        cfg,
        energy=energy,
        state=state,
        oracle=oracle,
        governance=governance,
        replicator=replicator,
        environment=FastRewardEnv(seed=seed),
        tick_energy_cost=2.0,
        memory_growth_rate=0.05,
        use_gyroscopic_stability=enhanced,
        stabilizer=stabilizer if enhanced else None,
    )

    return {
        "cfg": cfg,
        "loop": loop,
        "state": state,
        "oracle": oracle,
        "replicator": replicator,
        "clock": clock,
        "advance": advance,
    }


# --------------------------------------------------------------------------- #
# Run one experiment
# --------------------------------------------------------------------------- #


def run_cycles(stack: dict, n_cycles: int, *, print_per_cycle: bool = False) -> dict:
    loop: LearningLoop = stack["loop"]
    state: StateEngine = stack["state"]
    oracle: ConvergenceOracle = stack["oracle"]

    q_deltas: list[float] = []
    e8_scores: list[float] = []
    stab_scores: list[float] = []
    replication_decisions: list[bool] = []

    prev_q = state.snapshot().Q

    for cycle in range(1, n_cycles + 1):
        stack["advance"]()
        tr = loop.tick(attempt_replication=False)
        cur_q = state.snapshot().Q
        q_deltas.append(cur_q - prev_q)
        prev_q = cur_q

        if tr.e8_symmetry is not None:
            e8_scores.append(tr.e8_symmetry)
        if tr.gyro_stability is not None:
            stab_scores.append(tr.gyro_stability)

        # Evaluate replication-readiness each cycle but do NOT actually
        # replicate (we'll attempt that explicitly below).
        dec = oracle.is_ready_to_replicate()
        replication_decisions.append(dec.is_ready)

        if print_per_cycle:
            print(
                f"  cycle {cycle:02d}: "
                f"Q={state.snapshot().Q:.3f} "
                f"M={state.snapshot().M:.3f} "
                f"T={state.snapshot().T:.3f} "
                f"S={state.snapshot().S:.3f} "
                f"| E8={fmt_score(tr.e8_symmetry)} "
                f"stab={fmt_score(tr.gyro_stability)} "
                f"| ready_repl={dec.is_ready}"
            )

    return {
        "q_deltas": q_deltas,
        "e8_scores": e8_scores,
        "stab_scores": stab_scores,
        "replication_decisions": replication_decisions,
    }


def attempt_replications(stack: dict, n: int = 3) -> list[tuple[bool, float, str]]:
    """Trigger ``n`` replication attempts and return per-attempt stats."""
    rep: AgentReplicator = stack["replicator"]
    cfg: AgentConfig = stack["cfg"]
    results: list[tuple[bool, float, str]] = []

    for _ in range(n):
        outcome = rep.replicate()
        if outcome.success and outcome.child_config is not None:
            parent_hp = cfg.hyperparameters
            child_hp = outcome.child_config.hyperparameters
            keys = sorted(parent_hp.keys() & child_hp.keys())
            l2 = float(
                np.linalg.norm(
                    [child_hp[k] - parent_hp[k] for k in keys]
                )
            )
            results.append((True, l2, outcome.reason))
        else:
            results.append((False, 0.0, outcome.reason))

    return results


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    hr("SLSR SDK — Enhanced Demo (E8 geometry + gyroscopic stability)")
    e8 = E8RootSystem()
    checks = e8.verify()
    print("E8 self-check:", checks)
    print(f"  #roots = {e8.roots().shape[0]} in R^{e8.DIM}")

    hr("1) Enhanced agent: 30-cycle learning run")
    enh_stack = build_stack(enhanced=True, seed=13)
    enh_metrics = run_cycles(enh_stack, n_cycles=30, print_per_cycle=True)

    hr("2) Baseline agent (same seed, all enhancements OFF)")
    base_stack = build_stack(enhanced=False, seed=13)
    base_metrics = run_cycles(base_stack, n_cycles=30, print_per_cycle=False)

    hr("3) Replication attempts — enhanced vs baseline")
    print("  [enhanced]")
    enh_reps = attempt_replications(enh_stack, n=3)
    for ok, l2, reason in enh_reps:
        print(f"    success={ok} parent→child L2={l2:.4f}  ({reason})")

    print("  [baseline]")
    base_reps = attempt_replications(base_stack, n=3)
    for ok, l2, reason in base_reps:
        print(f"    success={ok} parent→child L2={l2:.4f}  ({reason})")

    # ------------------------------------------------------------------ #
    # Comparison table
    # ------------------------------------------------------------------ #

    hr("4) Baseline vs Enhanced — summary table")

    def _var(xs: list[float]) -> float:
        return float(np.var(xs)) if xs else float("nan")

    def _mean_l2(rows: list[tuple[bool, float, str]]) -> float:
        succ = [l2 for ok, l2, _ in rows if ok]
        return float(np.mean(succ)) if succ else float("nan")

    def _ready_rate(xs: list[bool]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    rows = [
        ("metric", "baseline", "enhanced"),
        (
            "(a) Q-delta variance over 30 cycles",
            f"{_var(base_metrics['q_deltas']):.6f}",
            f"{_var(enh_metrics['q_deltas']):.6f}",
        ),
        (
            "(b) mean parent→child L2 on successful reps",
            f"{_mean_l2(base_reps):.4f}",
            f"{_mean_l2(enh_reps):.4f}",
        ),
        (
            "(c) fraction of cycles 'ready_to_replicate'",
            f"{_ready_rate(base_metrics['replication_decisions']):.3f}",
            f"{_ready_rate(enh_metrics['replication_decisions']):.3f}",
        ),
        (
            "(d) successful replications (of 3 attempts)",
            f"{sum(1 for ok, *_ in base_reps if ok)}",
            f"{sum(1 for ok, *_ in enh_reps if ok)}",
        ),
    ]

    col_widths = [max(len(r[i]) for r in rows) + 2 for i in range(3)]
    for i, row in enumerate(rows):
        line = "".join(cell.ljust(col_widths[j]) for j, cell in enumerate(row))
        print("  " + line)
        if i == 0:
            print("  " + "-" * (sum(col_widths)))

    hr("5) Final interpretive notes")
    print(
        "  * Variance (a): gyroscopic damping on Q/M updates suppresses "
        "single-cycle spikes, so the enhanced agent's Q-delta variance is "
        "typically strictly lower."
    )
    print(
        "  * L2 distance (b): stabilize_vector is strictly contractive "
        "(‖out‖ ≤ ‖in‖), so mutated children sit closer to parents in "
        "hyperparameter space — catastrophic jumps are damped out."
    )
    print(
        "  * Readiness rate (c): the E8 gate rejects cycles whose 8-D "
        "embedding is noisy, even when the scalar S-score has already "
        "passed. Expect enhanced < baseline early on, then they converge "
        "as Q·M grows and the embedding snaps to a lattice direction."
    )

    hr("Demo complete.")


if __name__ == "__main__":
    main()
