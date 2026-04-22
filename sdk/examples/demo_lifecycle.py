"""End-to-end lifecycle demo for the SLSR SDK.

What this script demonstrates
-----------------------------
1. **Config load** from ``config.yaml`` (Pydantic-validated).
2. **Module wiring**: EnergyManager, StateEngine, ConvergenceOracle,
   GovernanceLayer, AgentReplicator, LearningLoop.
3. **Self-learning loop** running N cycles with a toy reward environment.
4. **Convergence gating** against the canonical ``pi * cos(sqrt(e))``
   threshold (~0.5792) plus the stricter 0.85 replication floor.
5. **Self-replication**: when the agent crosses the replication threshold
   AND passes all governance gates, a child config is minted with a
   bounded mutation.
6. **Governance in action**:
       * a PAUSE kill-switch engaged mid-run forces dormancy
       * then a TERMINATE kill-switch halts execution
       * audit log is dumped and hash-chain integrity verified.

Run it from the SDK root with:

    python examples/demo_lifecycle.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import yaml

# Ensure the local package is importable when running from the examples/ dir.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import slsr  # noqa: E402
from slsr.agent_replicator import AgentReplicator, NullInfrastructureAdapter  # noqa: E402
from slsr.convergence_oracle import ConvergenceOracle  # noqa: E402
from slsr.energy_manager import EnergyManager  # noqa: E402
from slsr.governance import GovernanceLayer  # noqa: E402
from slsr.learning_loop import (  # noqa: E402
    ExponentialMovingAverageLearner,
    LearningLoop,
)
from slsr.models import AgentConfig, KillSwitchLevel  # noqa: E402
from slsr.state_engine import StateEngine  # noqa: E402


# --------------------------------------------------------------------------- #
# Pretty-printing helpers (no external deps)
# --------------------------------------------------------------------------- #


class FastReward:
    """Demo-only env \u2014 rewards saturate inside ~15 steps so the agent
    actually crosses the 0.85 replication floor in a reasonable window."""

    def __init__(self, seed: int = 0):
        import random as _r
        self._rng = _r.Random(seed)
        self._step = 0

    def observe(self):
        self._step += 1
        base = 1.0 - math.exp(-self._step / 7.0)
        noise = self._rng.gauss(0.0, 0.02)
        reward = max(0.0, min(1.0, base + noise))
        return {"step": self._step, "reward": reward}

    def act(self, action):
        return None


def hr(title: str = "") -> None:
    bar = "=" * 78
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


def format_vec(v) -> str:
    return f"Q={v.Q:.3f}  M={v.M:.3f}  T={v.T:.3f}  S={v.S:.3f}"


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #


def build_config(path: Path) -> AgentConfig:
    with open(path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    # map nested YAML onto flat AgentConfig
    agent_block = raw.get("agent", {})
    merged = {
        **agent_block,
        "energy": raw.get("energy", {}),
        "state": raw.get("state", {}),
        "oracle": raw.get("oracle", {}),
        "replication": raw.get("replication", {}),
        "governance": raw.get("governance", {}),
    }
    return AgentConfig.model_validate(merged)


def main() -> None:
    hr("SLSR SDK \u2014 lifecycle demo")
    print(
        "Canonical convergence threshold (pi*cos(sqrt(e)) normalized):\n"
        f"  raw = pi * cos(sqrt(e)) = {math.pi * math.cos(math.sqrt(math.e)):+.4f}\n"
        f"  normalized CONV_THRESHOLD = {slsr.CONV_THRESHOLD:.4f}\n"
    )

    # ------------------------------------------------------------------ #
    # 1. Build config
    # ------------------------------------------------------------------ #
    cfg_path = ROOT / "config.yaml"
    cfg = build_config(cfg_path)
    print(f"Loaded config from: {cfg_path}")
    print(f"  agent_id          = {cfg.agent_id}")
    print(f"  generation        = {cfg.generation}")
    print(f"  hyperparameters   = {cfg.hyperparameters}")
    print(f"  max_generations   = {cfg.replication.max_generations}")
    print(f"  mutation_sigma    = {cfg.replication.mutation_sigma}")
    print(f"  replication_floor = {cfg.oracle.replication_floor}")

    # ------------------------------------------------------------------ #
    # 2. Wire modules
    # ------------------------------------------------------------------ #
    sub("wiring modules")
    # Use a temp audit log inside the example run directory
    cfg.governance.audit_log_path = str(ROOT / "demo_audit.log")

    # Virtual clock: each tick advances the clock by 1.0 "second" so the
    # demo completes deterministically in real wall-clock microseconds while
    # the agent still ages through its maturation horizon.
    virtual_clock = [0.0]

    def vtime() -> float:
        return virtual_clock[0]

    def tick_virtual_clock(_loop=None, _result=None):
        virtual_clock[0] += 1.0

    # With maturation_horizon_sec=15 and 1 virtual sec/tick, T=1.0 at tick 15.
    cfg.state.maturation_horizon_sec = 15.0

    energy = EnergyManager(cfg.energy, time_fn=vtime)
    state = StateEngine(cfg.state, genesis_ts=0.0, time_fn=vtime)
    oracle = ConvergenceOracle(state, config=cfg.oracle)
    governance = GovernanceLayer(cfg.governance)
    adapter = NullInfrastructureAdapter()
    replicator = AgentReplicator(
        cfg,
        oracle=oracle,
        energy=energy,
        governance=governance,
        config=cfg.replication,
        adapter=adapter,
    )
    loop = LearningLoop(
        cfg,
        energy=energy,
        state=state,
        oracle=oracle,
        governance=governance,
        replicator=replicator,
        environment=FastReward(seed=cfg.seed),
        learner=ExponentialMovingAverageLearner(alpha=0.35),
        tick_energy_cost=8.0,
        memory_growth_rate=0.08,
    )
    print(f"  {energy}")
    print(f"  {oracle}")
    print(f"  {governance}")

    # Advance the virtual clock exactly once per tick \u2014 registered FIRST
    # so every downstream hook (and the next tick's energy-refill check)
    # sees the new time.
    loop.on_tick(tick_virtual_clock)

    # Hook to print each tick compactly
    def _on_tick(_loop, result):
        if result.tick % 3 == 0 or result.replication_outcome or result.dormant:
            flag = "DORMANT" if result.dormant else result.phase
            repl = ""
            if result.replication_outcome:
                if result.replication_outcome.success:
                    repl = f"  \u2192 spawned {result.replication_outcome.child_config.agent_id}"
                else:
                    repl = f"  \u2717 replication refused ({result.replication_outcome.reason})"
            print(
                f"  tick {result.tick:>3} [{flag:7}]  "
                f"{format_vec(result.state_vector)}  "
                f"E={result.energy_available:7.2f}{repl}"
            )

    loop.on_tick(_on_tick)

    # ------------------------------------------------------------------ #
    # 3. Run learning cycles
    # ------------------------------------------------------------------ #
    hr("phase 1: learning cycles (ticks 1\u201325)")
    summary1 = loop.run_cycles(25, attempt_replication=True)
    print(
        f"\n  phase 1 summary: ticks={summary1.ticks_run} "
        f"final {format_vec(summary1.final_state)}  "
        f"children={len(summary1.children_spawned)}"
    )

    # ------------------------------------------------------------------ #
    # 4. Evaluate fitness + attempt explicit replication
    # ------------------------------------------------------------------ #
    hr("phase 2: evaluate fitness & attempt additional replications")
    vec = state.snapshot()
    act = oracle.is_ready_to_act()
    rep = oracle.is_ready_to_replicate()
    print(f"  current state vector : {format_vec(vec)}")
    print(f"  canonical threshold  : {slsr.CONV_THRESHOLD:.4f}")
    print(f"  replication floor    : {oracle.replication_floor:.4f}")
    print(f"  is_ready_to_act      : {act.is_ready}  (margin={act.margin:+.4f})")
    print(f"  is_ready_to_replicate: {rep.is_ready}  (margin={rep.margin:+.4f})")

    # Try a few more replications to exercise lineage + caps
    for i in range(3):
        print(f"\n  explicit replication attempt #{i+1}")
        outcome = replicator.replicate()
        if outcome.success:
            print(
                f"    \u2713 success  child={outcome.child_config.agent_id} "
                f"gen={outcome.child_config.generation}"
            )
            print(f"    mutation.numeric_deltas = {outcome.mutation.numeric_deltas}")
        else:
            print(f"    \u2717 refused  reason={outcome.reason}")

    sub("lineage")
    for aid, node in replicator.lineage().items():
        marker = "root" if node.parent_id is None else f"child-of={node.parent_id}"
        print(f"  {aid}  (gen={node.generation})  [{marker}]")

    # ------------------------------------------------------------------ #
    # 5. Governance in action
    # ------------------------------------------------------------------ #
    hr("phase 3: governance in action")
    sub("PAUSE kill-switch \u2192 dormant ticks")
    governance.engage_kill_switch(KillSwitchLevel.PAUSE, "operator maintenance window")
    loop.run_cycles(3, attempt_replication=False)
    governance.kill_switch.disengage()

    sub("TERMINATE kill-switch \u2192 loop halts")
    governance.engage_kill_switch(KillSwitchLevel.TERMINATE, "unacceptable policy risk")
    summary3 = loop.run_cycles(5, attempt_replication=False)
    print(f"  halted_reason = {summary3.halted_reason}")
    governance.kill_switch.disengage()

    # ------------------------------------------------------------------ #
    # 6. Audit log integrity
    # ------------------------------------------------------------------ #
    hr("phase 4: audit log")
    entries = governance.audit_log.entries()
    print(f"  total audit entries: {len(entries)}")
    print("  last 6 events:")
    for e in entries[-6:]:
        print(f"    seq={e.seq:>3}  {e.event_type:<28}  agent={e.agent_id}")
    print(f"  hash-chain verified: {governance.audit_log.verify_chain()}")

    # ------------------------------------------------------------------ #
    # 7. Shutdown
    # ------------------------------------------------------------------ #
    loop.shutdown(reason="demo complete")
    hr("demo complete")
    print(f"  total ticks        = {loop.tick_count}")
    # lineage has N+1 nodes (root + children); subtract 1 for root.
    print(f"  children spawned   = {len(replicator.lineage()) - 1}")
    print(f"  energy consumed    = {energy.consumed:.2f} EU of cap {cfg.energy.max_cap}")
    print(f"  final state vector = {format_vec(state.snapshot())}")
    # Remove audit log if created (optional cleanup for reruns)
    if os.environ.get("SLSR_KEEP_AUDIT") != "1":
        try:
            os.remove(cfg.governance.audit_log_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
