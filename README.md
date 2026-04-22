# ProjectCypherGhostTech

**Self-Learning & Self-Replication (SLSR) SDK** — a governed framework for agents that
learn, converge, and replicate themselves under explicit energy, state, and geometric
constraints.

The project is organized around three hand-written formulas that act as the operational
backbone of every agent:

| Formula | Role |
|---|---|
| `E = m · c²` | **Energy accounting.** Every cognitive / replicative action costs `m · c²` Energy Units (EU), where `m` is data/effort mass and `c` is a speed factor. |
| `S = Q · M · T` | **Convergence score.** An agent is considered converged (and therefore eligible to replicate) only when the product of its Quality, Memory, and Temporal maturity components crosses a pinned threshold `CONV_THRESHOLD = π · cos(√e) ≈ 0.49899`. |
| `π · cos(√e)` | **Convergence threshold constant.** A deliberately irrational, non-cherry-picked scalar used as the canonical gate. |

On top of this core, the prototype integrates two geometric / dynamical enhancements:

* **E8 root lattice geometry** (`slsr.geometry.E8RootSystem`) — an optional 8-D symmetry
  gate in the convergence oracle that requires agent state embeddings to align with the
  240-root E8 lattice before declaring convergence.
* **Gyroscopic stability** (`slsr.stability.GyroscopicStabilizer`) — Euler rigid-body
  damping applied to Q/M state deltas and to child-agent mutation vectors, eliminating
  oscillations and catastrophic mutation jumps.

Both enhancements are strictly opt-in via `EnhancementConfig` and have empirical impact
captured in `sdk/examples/demo_enhanced.py`.

---

## Repository layout

```
.
├── README.md         ← this file
├── research/         ← theoretical research report
│   ├── self_replication_sdk_report.md
│   └── self_replication_sdk_report.pdf
├── design/           ← formal SDK design document
│   ├── SDK_DESIGN_DOC.md
│   └── SDK_DESIGN_DOC.pdf
└── sdk/              ← working Python prototype (slsr-sdk)
    ├── README.md
    ├── pyproject.toml
    ├── config.yaml
    ├── slsr/         ← library modules
    ├── tests/        ← pytest suite (core + enhancements)
    └── examples/     ← demo_lifecycle.py, demo_enhanced.py
```

### `research/`
End-to-end research report motivating the SLSR paradigm, the choice of the three
formulas, related literature (autopoiesis, evolutionary computation, AI governance,
energy-based models), and the rationale for the E8 / gyroscopic enhancements.

### `design/`
Formal SDK design document: module-by-module specifications, data contracts (Pydantic
models), governance model, audit log format, replication policy, kill-switch semantics,
and extension protocols.

### `sdk/`
A working Python 3.10+ prototype implementing the full design: energy manager, state
engine, convergence oracle (with optional E8 gate), governance layer with hash-chained
audit log, agent replicator (with optional gyroscopic mutation smoothing), and the
FSM-driven learning loop.

---

## Quick start (SDK)

```bash
cd sdk
pip install -e .
python examples/demo_enhanced.py
```

`demo_enhanced.py` runs a paired comparison of an agent with **all** enhancements
enabled vs. an identical baseline and prints per-cycle E8 symmetry / gyroscopic stability
scores plus a final summary table.

To run the baseline lifecycle demo (no enhancements, pure `E=mc²` / `S=QMT`):

```bash
python examples/demo_lifecycle.py
```

## Tests

```bash
cd sdk
pytest tests/
```

Two suites are included:

* `tests/test_core.py` — covers energy accounting, state engine, convergence oracle,
  governance / audit chain, replicator, and learning loop.
* `tests/test_enhancements.py` — covers E8 root system properties, gyroscopic
  stabilizer invariants, and end-to-end wiring of both enhancements through the
  oracle, learning loop, and replicator.

---

## Key invariants (enforced by the implementation and tested)

1. **No action runs unless it is paid for in EU.** `EnergyManager.reserve()` must
   succeed before any cost-incurring operation; `commit()` and `release()` keep the
   ledger honest.
2. **No replication fires below the convergence floor.** The oracle gates on
   `S ≥ π·cos(√e)` and — when E8 is on — on lattice alignment as well.
3. **Every replication, mutation, and kill-switch event is auditable.** The audit log
   is SHA-256 hash-chained and verifiable via `AuditLog.verify_chain()`.
4. **Gyroscopic smoothing is strictly contractive.** Smoothed vector norms are `≤`
   proposed norms; scalar damping outputs lie between current and proposed values.
5. **E8 geometry is advisory, not a free pass.** The E8 gate *tightens* convergence; it
   never accepts an agent that the scalar `S = QMT` rule would reject.

---

## License / governance notes

This is a research prototype. The governance layer ships with conservative defaults
(human-approval mode = NOTIFY, fleet cap, generation cap, structural-mutation
restrictions). Operators deploying the SDK in adversarial or production settings are
expected to raise `ApprovalMode` to `REQUIRED` and wire a real `HumanApprovalGate`.
