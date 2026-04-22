# SLSR SDK — Self-Learning & Self-Replication

A Python 3.10+ **control-plane SDK** for building autonomous agents that can
learn continuously, replicate themselves under strict governance, and
operate inside hard energy / safety budgets.

The whole thing is grounded in three very small equations (captured on the
design napkin that started this project):

```
              E = m·c²              (Energy budgeting — EnergyManager)
              ───────── = QMT       (State tracking  — StateEngine)
                  S
          π × cos(√e)               (Convergence gate — ConvergenceOracle)
```

### What the SDK provides

| Module | What it does |
|---|---|
| `slsr.energy_manager` | `EnergyManager` — ledger + reservation/commit for `E = m·c²`. |
| `slsr.state_engine` | `StateEngine` — tracks `(Q, M, T)` and computes `S = Q·M·T`. |
| `slsr.convergence_oracle` | `ConvergenceOracle` — readiness gate against the canonical threshold `π·cos(√e)` normalized (`≈ 0.5792`). |
| `slsr.agent_replicator` | `AgentReplicator` — governed replication with bounded mutations and lineage tracking. |
| `slsr.learning_loop` | `LearningLoop` — FSM orchestrator: `OBSERVE → LEARN → EVALUATE → DECIDE → ACT → (REPLICATE?)`. |
| `slsr.governance` | `GovernanceLayer` — policy engine, kill switch (4 levels), SHA-256 hash-chained audit log. |
| `slsr.models` | Pydantic v2 DTOs (`AgentConfig`, `AgentState`, `EnergyBudget`, `ReplicationRequest`, `AuditEntry`, `EnhancementConfig`, …). |
| `slsr.protocols` | PEP-544 `Protocol`s for pluggable strategies/adapters. |
| `slsr.geometry` | **E8RootSystem** — 240 root vectors in 8D for geometric convergence gating (opt-in). |
| `slsr.stability` | **GyroscopicStabilizer** — Euler rigid-body damper for update smoothing (opt-in). |

### Install (dev)

```bash
cd /home/ubuntu/slsr_sdk
python -m pip install -e ".[dev]"
```

### Run the demo

```bash
python examples/demo_lifecycle.py
```

The demo walks through:
1. Loading `config.yaml` and wiring every module.
2. Running 25 learning cycles with a toy reward environment.
3. Printing the `(Q, M, T, S)` trajectory and convergence decisions.
4. Firing three explicit replication attempts (you'll see some succeed,
   some refused by the generation cap or fleet cap).
5. Engaging the **PAUSE** kill-switch → dormancy, then **TERMINATE** → halt.
6. Dumping the audit log and verifying the SHA-256 hash chain.

### Run the tests

```bash
pytest -q
```

### The canonical convergence threshold

`CONV_THRESHOLD` is hard-coded in `slsr/__init__.py`:

```python
raw        = π · cos(√e)          # ≈ -0.2408 (radians)
normalized = (raw + π) / (2π)     # ≈  0.5792
```

It is **Invariant I9** from the design doc: it cannot be lowered at runtime.
Operators may stack *higher* floors on top (e.g. `replication_floor = 0.85`).

### Design document

See `/home/ubuntu/sdk_design/SDK_DESIGN_DOC.md` for the full specification,
including API signatures, data-flow diagrams, safety invariants (I1–I10),
and the YAML configuration schema.



---

## Enhancements: E8 Geometry & Gyroscopic Stability

Two **fully opt-in** modules extend the core pipeline with a geometric
readiness check and a physical-stability damper. They are additive:
existing configs, existing tests, and the existing public API all keep
working unchanged — you explicitly enable them via flags on
`EnhancementConfig` or the per-class constructors.

### Why these two?

The canonical `π·cos(√e) ≈ 0.5792` gate is a **scalar** readiness check:
it asks "is the 1-D composite score `S = Q·M·T` above threshold?". Two
failure modes sneak past that gate in practice:

1. **Geometric blindness** — `S` says nothing about *what direction* the
   agent's embedding points. A noisy-but-high-scoring agent can pass
   the scalar gate while its internal representation is aimless.
2. **Oscillation & catastrophic jumps** — cycle-to-cycle Q/M updates
   can oscillate around a fixed point, and replication mutations can
   produce children that sit far from their parent in hyperparameter
   space. Both hurt reproducibility.

The enhancements address each failure mode with a physics-flavoured
primitive:

| Enhancement | Module | What it does |
|---|---|---|
| **E8 root lattice** | `slsr.geometry.E8RootSystem` | 240 maximally-separated anchor directions in 8D. The oracle additionally requires the agent's 8-D embedding to cos-align with the lattice (symmetry score ≥ `e8_symmetry_min`). |
| **Gyroscopic state damping** | `slsr.stability.GyroscopicStabilizer.damp_update` | Each Q/M update is routed through a first-order damper derived from Euler's rigid-body equation, strictly between the current and proposed value. |
| **Gyroscopic mutation smoothing** | `slsr.stability.GyroscopicStabilizer.stabilize_vector` | Mutation vectors are strictly contracted (`‖out‖ ≤ ‖in‖`) before being applied to child configs. |

### Enable them

All three flags live on `AgentConfig.enhancements` (a new
`EnhancementConfig` sub-model):

```python
from slsr.models import AgentConfig

cfg = AgentConfig()
cfg.enhancements.use_e8_geometry        = True
cfg.enhancements.use_gyroscopic_stability = True
cfg.enhancements.use_gyroscopic_mutation  = True
cfg.enhancements.e8_symmetry_min        = 0.85   # 0.0 – 1.0, default 0.4
cfg.enhancements.gyro_damping           = 0.3    # higher = more smoothing
cfg.enhancements.gyro_inertia           = (1.0, 1.0, 1.0)
```

Then pass the matching flags (and an optional 8-D embedding provider)
to the constructors:

```python
oracle = ConvergenceOracle(
    state_engine,
    use_e8_geometry=cfg.enhancements.use_e8_geometry,
    e8_symmetry_min=cfg.enhancements.e8_symmetry_min,
    state_vector_provider=my_embedding_callback,   # () -> np.ndarray of shape (8,)
)

loop = LearningLoop(
    cfg, ...,
    use_gyroscopic_stability=cfg.enhancements.use_gyroscopic_stability,
)

replicator = AgentReplicator(
    cfg, ...,
    use_gyroscopic_mutation=cfg.enhancements.use_gyroscopic_mutation,
)
```

If `use_e8_geometry=True` is set but `state_vector_provider` is `None`,
the oracle gracefully falls back to scalar-only gating — no crashes, no
silent changes to `is_ready`. This keeps the enhancement safe to enable
in layered deployments where not every caller can produce an 8-D
embedding.

### When to use each

* **Deep-learning agents** benefit from the **E8 gate** — their policy
  networks naturally produce a dense latent embedding that can be
  projected to 8 dimensions (e.g. PCA or a learned projection). The
  geometric gate then rejects "confident-but-aimless" intermediate
  states. They also benefit from **gyroscopic stability** to smooth
  the noisy per-batch Q updates, which is essentially a structured
  momentum term.

* **Neuromorphic / spiking agents** benefit primarily from
  **gyroscopic stability** as a homeostatic regulator — the
  `damp_update` channel acts like a neuromodulator pulling the firing
  rate back toward a setpoint. The E8 gate is optional here; it only
  helps if you can produce a reliable 8-D state embedding from the
  spike-train statistics.

* **Evolutionary / population search**: enable
  **`use_gyroscopic_mutation`**. It contracts mutation vectors strictly
  (`‖out‖ ≤ ‖in‖`), keeping children closer to parents in L2 and
  preventing catastrophic single-generation drift.

### Empirical notes from `examples/demo_enhanced.py`

Running the demo on a 30-cycle toy environment (reward saturates in ~20
steps), with all three enhancements ON vs. all OFF, same seed:

| Metric | Baseline | Enhanced |
|---|---|---|
| (a) Q-delta variance over 30 cycles | 0.000225 | **0.000202** |
| (b) mean parent→child L2 on successful replications | 6.02 | **4.46** |
| (c) fraction of cycles `ready_to_replicate` | 0.133 | 0.133 |
| (d) successful replications (of 3 attempts) | 3 | 3 |

Both agents eventually replicate, but the enhanced agent's children sit
**~26 % closer** to their parents in hyperparameter space (b), its
per-cycle Q-deltas have **~10 % lower variance** (a), and the E8
symmetry score climbs from ~0.80 at cycle 1 to **0.93+ at cycle 30**
as the learned embedding snaps onto a lattice direction.

See `examples/demo_enhanced_output.txt` for the full captured output.

### Run the enhanced demo and tests

```bash
python examples/demo_enhanced.py
pytest -q tests/test_enhancements.py     # 23 tests covering all new behaviour
pytest -q                                # 63 total tests (40 core + 23 enh)
```
