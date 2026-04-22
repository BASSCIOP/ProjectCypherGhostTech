# Self-Learning & Self-Replication SDK — Design Document

| Field | Value |
|---|---|
| Document ID | SLSR-SDK-DESIGN-260419-A1 |
| Version | 0.9.0 (Pre-Implementation Draft) |
| Status | Ready for Engineering Review |
| Owner | Autonomous Systems / Platform Engineering |
| Last Updated | 2026-04-19 |
| Based On | `self_replication_sdk_report.md` (SR-SDK-260419-A1) |

---

### Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Core Modules](#3-core-modules)
4. [API Surface Design](#4-api-surface-design)
5. [Data Flow](#5-data-flow)
6. [Replication Governance & Safety](#6-replication-governance--safety)
7. [Extension Points](#7-extension-points)
8. [Technology Stack](#8-technology-stack)
9. [Configuration Schema](#9-configuration-schema)
10. [Glossary](#10-glossary)

---

## 1. Executive Summary

### 1.1 What the SDK Does

The **Self-Learning & Self-Replication SDK** (`slsr-sdk`) is a Python 3.10+ library that enables the construction of **autonomous agents** capable of:

1. **Learning from experience** — continuously updating internal policy / model state from streams of observations and rewards.
2. **Resource-aware computation** — budgeting CPU, memory, and wall-clock time as a first-class concern through an energy-accounting model derived from `E = mc²` (see `EnergyManager`).
3. **State-driven adaptation** — tracking agent "fitness" across three orthogonal dimensions (Quality, Memory, Time) via `S = Q × M × T` (see `StateEngine`).
4. **Probabilistic convergence gating** — deciding when an agent is "ready" to act, commit a learned update, or replicate, against a convergence threshold derived from `π × cos(√e)` (see `ConvergenceOracle`).
5. **Governed self-replication** — spawning child agents with bounded mutations, generation caps, resource caps, and mandatory human-in-the-loop approval gates (see `AgentReplicator` + `GovernanceLayer`).

The SDK is **not** a general-purpose ML framework. It is a *control plane* that sits **on top of** existing ML stacks (PyTorch, scikit-learn, LangGraph, Ray, etc.) and adds the metabolic, evolutionary, and safety machinery required for long-running autonomous systems.

### 1.2 Who It's For

| Persona | Primary Use Case |
|---|---|
| **ML Platform Engineers** | Build evolving model fleets with automatic pruning/promotion |
| **Autonomous Agent Developers** | Multi-agent research systems that spawn specialized sub-agents |
| **SRE / Cybersecurity Teams** | Self-replicating patch agents with hard governance rails |
| **Edge / IoT Architects** | Resource-bounded adaptive agents on constrained hardware |
| **AI Safety Researchers** | Reference implementation of a bounded self-replicating substrate |

### 1.3 Design Principles (Non-Negotiable)

1. **Safety before capability** — every replication is gated by `GovernanceLayer`. No exceptions, no bypass.
2. **Energy is finite** — no operation runs unbudgeted; exceeding the budget always triggers graceful degradation.
3. **Observability is mandatory** — every decision (learn, replicate, kill) is audit-logged with a signed record.
4. **Extension over modification** — users plug in strategies/fitness functions; they do not fork the core.
5. **Deterministic under seed** — given the same seed + config + inputs, the SDK produces identical trajectories (critical for reproducibility and incident forensics).

---

## 2. System Architecture

### 2.1 High-Level Module Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SELF-LEARNING & SELF-REPLICATION SDK                      ║
║                              (slsr-sdk)                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ┌────────────────────────────────────────────────────────────────────┐    ║
║   │                       USER APPLICATION LAYER                       │    ║
║   │        (custom strategies, fitness fns, replication policies)      │    ║
║   └───────────────────────────────┬────────────────────────────────────┘    ║
║                                   │                                         ║
║   ┌───────────────────────────────▼────────────────────────────────────┐    ║
║   │                         LearningLoop                               │    ║
║   │         ┌──────────────────── orchestrator ───────────────────┐    │    ║
║   │         │  init → observe → learn → evaluate → decide → act    │    │    ║
║   │         └──────────────────────────────────────────────────────┘    │    ║
║   └──────┬──────────────┬──────────────┬──────────────┬───────────┘    ║
║          │              │              │              │                    ║
║          ▼              ▼              ▼              ▼                    ║
║   ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────────┐          ║
║   │   Energy   │ │   State    │ │ Convergence  │ │    Agent     │          ║
║   │  Manager   │ │  Engine    │ │    Oracle    │ │  Replicator  │          ║
║   │            │ │            │ │              │ │              │          ║
║   │  E = m·c²  │ │ S = Q·M·T  │ │ π·cos(√e)    │ │   Governed   │          ║
║   │  budgets   │ │  tracking  │ │  ≈ 0.5792    │ │  fork/spawn  │          ║
║   └─────┬──────┘ └─────┬──────┘ └──────┬───────┘ └──────┬───────┘          ║
║         │              │               │                 │                 ║
║         └──────────────┴───────┬───────┴─────────────────┘                 ║
║                                ▼                                           ║
║   ┌────────────────────────────────────────────────────────────────────┐   ║
║   │                        GovernanceLayer                             │   ║
║   │  • policy engine   • kill switch   • audit log   • approval gates  │   ║
║   │  • signed decision records   • resource ceilings   • kill-chain    │   ║
║   └───────────────────────────────┬────────────────────────────────────┘   ║
║                                   │                                        ║
║   ┌───────────────────────────────▼────────────────────────────────────┐   ║
║   │                 INFRASTRUCTURE ADAPTERS (pluggable)                │   ║
║   │   Local Process │ Docker │ Kubernetes │ Ray │ Lambda │ Filesystem  │   ║
║   └────────────────────────────────────────────────────────────────────┘   ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 2.2 Module Responsibilities & Interactions

| Module | Owns | Reads From | Writes To |
|---|---|---|---|
| `LearningLoop` | Lifecycle FSM, tick scheduler | All modules | `AuditLog` |
| `EnergyManager` | Budget accounting (E, m, c²) | Config, runtime metrics | `StateEngine.M` |
| `StateEngine` | Agent state vector (Q, M, T) | EnergyManager, evaluator | `ConvergenceOracle` |
| `ConvergenceOracle` | Fitness scoring, threshold gate | `StateEngine` | `AgentReplicator`, `LearningLoop` |
| `AgentReplicator` | Spawn/fork/mutate child agents | `ConvergenceOracle`, `GovernanceLayer` | Infrastructure adapter |
| `GovernanceLayer` | Policy enforcement, approvals, kill switch | All modules (hooks) | `AuditLog`, external systems |

### 2.3 Cross-Cutting Concerns

- **Audit Log** — append-only, cryptographically chained (SHA-256 prev-hash), persisted by `GovernanceLayer`.
- **Event Bus** — in-process pub/sub (`asyncio.Queue`-based) for module decoupling.
- **Config** — loaded once at `LearningLoop` init; immutable thereafter unless `ReloadPolicy` allows.

---

## 3. Core Modules

### 3.1 `EnergyManager` — Resource Accounting

**Inspiration:** `E = m · c²`

**Mapping:**
- `E` (energy) → total compute budget in *Energy Units* (EU). 1 EU ≈ 1 CPU-second of normalized work on the reference machine.
- `m` (data mass) → sum of input payload sizes, model parameter bytes, and working-set memory, in megabytes.
- `c²` (processing speed squared) → `(normalized_throughput)²`, where `normalized_throughput` is a dimensionless factor (0–1) derived from hardware benchmark.

**Core formula the module evaluates per operation:**
```
E_required = data_mass_mb × (processing_speed)²
```

**Responsibilities:**
- Maintain a ledger of `E_available`, `E_reserved`, `E_consumed`.
- Before any expensive op (training step, replication, eval), ops call `reserve(amount)`; on completion they `commit()` or `release()`.
- Publishes `OutOfEnergy` event when `E_available < E_required` — this is always a soft failure (raise, don't crash the runtime).
- Emits telemetry: `energy.reserved`, `energy.committed`, `energy.refund`, `energy.budget_exhausted`.

**Key design choices:**
- Energy is *accrued* over time on a refill schedule (`refill_rate_eu_per_sec`), modeling a metabolic rest/replenish cycle.
- A **hard ceiling** (`max_energy_cap`) prevents accumulation exploits.
- Replication always requires a configurable `replication_energy_floor` to exist *before* the operation is considered — prevents starvation-driven forks.

---

### 3.2 `StateEngine` — Agent State Tracking

**Inspiration:** `S = Q × M × T`

**State vector components:**

| Var | Meaning | Range | Source |
|---|---|---|---|
| `Q` (Quality) | Fitness of current policy/model on the evaluator | `[0.0, 1.0]` | `FitnessFunction` output |
| `M` (Memory) | Normalized memory/experience richness | `[0.0, 1.0]` | `ReplayBuffer` fill + schema coverage |
| `T` (Time) | Temporal maturity — normalized age vs. `maturation_horizon` | `[0.0, 1.0]` | Wall-clock since `genesis_ts` |

**Composite score:**
```
S = Q × M × T        # all components in [0,1], so S in [0,1]
```

**Responsibilities:**
- Expose current `StateVector(Q, M, T, S)` to other modules.
- Maintain **historical trajectory** of S (rolling window, default 1,024 samples) for trend detection.
- Provide derivatives: `dQ/dt`, `dM/dt`, `dT/dt` for the `ConvergenceOracle` to spot stagnation/collapse.
- Trigger `StateAlarm` events when any component collapses below a floor (default `0.05`) — signals unhealthy agent.

**Why multiplicative (not additive)?** Multiplication encodes **joint necessity**: an agent that is high-quality but has no memory (M=0) or has only existed for a millisecond (T≈0) is *not* ready. All three dimensions must be non-trivially present.

---

### 3.3 `ConvergenceOracle` — Fitness / Readiness Scoring

**Inspiration:** `π × cos(√e)` — a transcendental constant used as a canonical, non-tunable readiness threshold.

**Canonical threshold:**
```
CONV_THRESHOLD = |π × cos(√e)|   # normalized per SDK spec ≈ 0.5792
```

> **Note to implementers:** The raw value of `π × cos(√e)` computed in radians is ≈ `-0.2408`. The SDK normalizes this into the unit interval using the canonical mapping `normalize(x) = (x + π) / (2π)` → ≈ `0.5792`. This value is **hard-coded** as `slsr_sdk.constants.CONV_THRESHOLD` and is **intentionally not configurable** — it represents the Invariant Core principle from the foundational report. Operators may layer *additional* thresholds on top (e.g., a task-specific floor of `0.85`), but they cannot lower the canonical floor.

**Responsibilities:**
- Read `StateVector` from `StateEngine`.
- Compute `readiness_score` (default: `readiness = S`; can be overridden by a custom `FitnessFunction`).
- Return `ConvergenceDecision` with fields: `{is_ready: bool, score: float, threshold: float, margin: float, rationale: str}`.
- Expose three named gates:
  - `is_ready_to_act()` — for routine action emission
  - `is_ready_to_learn_commit()` — for persisting a learned update to durable state
  - `is_ready_to_replicate()` — for invoking `AgentReplicator` (strictest; uses `max(CONV_THRESHOLD, replication_floor)`)

**Stochastic gating (optional):**
When `stochastic_gate=True`, readiness is sampled as `readiness >= threshold + ε`, where `ε ~ N(0, jitter_sigma)`. This prevents deterministic oscillation on the threshold boundary in noisy environments.

---

### 3.4 `AgentReplicator` — Governed Self-Replication

**Responsibilities:**
- Produce a **child agent blueprint** from the parent: serialized config + policy/model weights + mutation delta.
- Validate the blueprint against `GovernanceLayer` policies before ANY spawn attempt.
- Delegate actual process/container/pod creation to an **infrastructure adapter**.
- Track lineage (parent_id, generation, mutation_manifest) for forensics.

**Replication pipeline (always in this order, no shortcuts):**
```
1. ConvergenceOracle.is_ready_to_replicate()           [gate 1: fitness]
2. EnergyManager.reserve(replication_cost)             [gate 2: resource]
3. GovernanceLayer.evaluate_replication_request(req)   [gate 3: policy]
4. (Optional) HumanApprovalGate.await_approval(req)    [gate 4: human]
5. Apply mutation (bounded by mutation_sigma)
6. Infrastructure adapter spawns child
7. GovernanceLayer.register_child(child_id, parent_id, generation)
8. AuditLog append (signed decision record)
```

**Mutation model:**
- Mutations are represented as a `MutationManifest` describing which fields change (hyperparameters, policy weights, prompts).
- Bounded mutations only: `|delta| <= mutation_sigma × parent_value` for numeric fields, categorical switches require explicit allowlist.
- **Structural mutations** (adding new layers, new tool access) are forbidden unless `allow_structural_mutation=True` in config AND human approval is granted.

**Generation tracking:**
- Every agent has `generation: int` (root = 0).
- `max_generations` (default 5) is enforced by `GovernanceLayer`; no replication attempt that would exceed it is ever dispatched.

---

### 3.5 `LearningLoop` — Self-Learning Orchestrator

**Responsibilities:**
- Own the agent's finite state machine:
  `INIT → OBSERVE → LEARN → EVALUATE → DECIDE → ACT → (REPLICATE?) → OBSERVE → …`
- Invoke each module in order each tick.
- Support `async` execution (based on `asyncio`) for concurrent agent fleets.
- Expose lifecycle hooks for user code (`on_tick`, `before_learn`, `after_evaluate`, `before_replicate`, `on_shutdown`).

**FSM detail:**

| State | What happens |
|---|---|
| `INIT` | Load config, instantiate modules, emit `AgentBorn` event |
| `OBSERVE` | Pull a batch of observations from the environment adapter |
| `LEARN` | Apply `LearningStrategy.update(batch)` — may be SGD step, RL update, rule refinement |
| `EVALUATE` | Run `FitnessFunction` → update `StateEngine.Q` |
| `DECIDE` | Query `ConvergenceOracle` for act/commit/replicate gates |
| `ACT` | Emit action(s) to environment if `is_ready_to_act` |
| `REPLICATE` | If `is_ready_to_replicate` and criteria met, invoke `AgentReplicator` |
| `SHUTDOWN` | Graceful drain; final audit snapshot |

**Backpressure:**
- If `EnergyManager` signals `low_energy` (`E_available < 2 × tick_cost`), the loop enters **dormancy mode**: skips LEARN, keeps OBSERVE at reduced rate, waits for refill.

---

### 3.6 `GovernanceLayer` — Safety, Policies, Kill Switch

**Responsibilities:**
1. **Policy evaluation** — each governed operation (learn-commit, replicate, structural mutation, external-tool use) is evaluated against a declarative policy set.
2. **Kill switch** — a cryptographically-signed external command (`kill_signal`) immediately terminates the agent and all descendants. Check is performed at every FSM transition.
3. **Audit logging** — every decision is appended to a tamper-evident log (SHA-256 hash chain). Each record carries: `{ts, agent_id, event_type, inputs_hash, decision, rationale, sig}`.
4. **Approval gates** — for high-risk operations (replication at generation ≥ 3, structural mutation, network-effecting actions), governance blocks on a `HumanApprovalGate`. Default implementation posts to a webhook and awaits signed response; alternative CLI/stdin approver is provided.
5. **Resource ceilings** — enforce global fleet-wide caps (total agents alive, total EU consumed per hour, total replications per day).

**Policy format:**
Policies are declarative (YAML) and compiled into callable predicates at load time (see Section 9). Example:

```yaml
policies:
  - id: deny-structural-mutation-without-approval
    applies_to: replicate
    condition: "request.structural_mutation == true and request.approvals < 1"
    effect: deny
    severity: critical
```

**Kill chain (from mildest to most severe):**
- `PAUSE` — halt LEARN and ACT, keep OBSERVE
- `QUARANTINE` — isolate from network; read-only filesystem
- `TERMINATE` — graceful shutdown with final audit
- `ANNIHILATE` — force-kill process tree of agent + all descendants; emergency only

---

## 4. API Surface Design

### 4.1 Top-Level Package Layout

```
slsr_sdk/
├── __init__.py
├── constants.py              # CONV_THRESHOLD, defaults
├── agent.py                  # Agent, AgentConfig
├── loop.py                   # LearningLoop
├── energy.py                 # EnergyManager, EnergyLedger
├── state.py                  # StateEngine, StateVector
├── oracle.py                 # ConvergenceOracle, ConvergenceDecision
├── replicator.py             # AgentReplicator, MutationManifest
├── governance/
│   ├── layer.py              # GovernanceLayer
│   ├── policies.py           # Policy, PolicyEngine
│   ├── approval.py           # HumanApprovalGate
│   └── audit.py              # AuditLog, AuditRecord
├── strategies/
│   ├── learning.py           # LearningStrategy (protocol), built-ins
│   ├── fitness.py            # FitnessFunction (protocol), built-ins
│   └── replication.py        # ReplicationPolicy (protocol), built-ins
├── adapters/
│   ├── base.py               # InfrastructureAdapter (protocol)
│   ├── local.py              # subprocess-based
│   ├── docker.py
│   ├── kubernetes.py
│   └── ray.py
├── events.py                 # EventBus, event types
└── exceptions.py
```

### 4.2 Public Classes — Signatures

#### `Agent`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentConfig:
    agent_id: str
    generation: int = 0
    parent_id: Optional[str] = None
    seed: int = 42
    energy: "EnergyConfig"
    state: "StateConfig"
    oracle: "OracleConfig"
    replication: "ReplicationConfig"
    governance: "GovernanceConfig"

class Agent:
    def __init__(self, config: AgentConfig) -> None: ...
    async def run(self, max_ticks: Optional[int] = None) -> "RunSummary": ...
    async def shutdown(self, reason: str = "normal") -> None: ...
    @property
    def state(self) -> "StateVector": ...
    @property
    def lineage(self) -> list[str]: ...
```

#### `EnergyManager`

```python
class EnergyManager:
    def __init__(self, initial_eu: float, max_cap: float,
                 refill_rate: float, reference_c_squared: float) -> None: ...

    def reserve(self, eu: float, reason: str) -> "Reservation": ...
    def commit(self, reservation: "Reservation") -> None: ...
    def release(self, reservation: "Reservation") -> None: ...

    def required_energy(self, data_mass_mb: float, speed_factor: float) -> float:
        """E = m * c^2 (speed_factor = c, squared internally)."""
        return data_mass_mb * (speed_factor ** 2)

    @property
    def available(self) -> float: ...
    @property
    def consumed(self) -> float: ...
```

#### `StateEngine`

```python
@dataclass(frozen=True)
class StateVector:
    Q: float            # quality, [0,1]
    M: float            # memory richness, [0,1]
    T: float            # temporal maturity, [0,1]
    @property
    def S(self) -> float: return self.Q * self.M * self.T

class StateEngine:
    def __init__(self, config: "StateConfig", genesis_ts: float) -> None: ...
    def update_quality(self, q: float) -> None: ...
    def update_memory(self, m: float) -> None: ...
    def tick_time(self) -> None: ...
    def snapshot(self) -> StateVector: ...
    def trajectory(self, n: int = 100) -> list[StateVector]: ...
    def derivatives(self) -> dict[str, float]: ...  # dQ/dt, dM/dt, dT/dt
```

#### `ConvergenceOracle`

```python
@dataclass(frozen=True)
class ConvergenceDecision:
    is_ready: bool
    score: float
    threshold: float
    margin: float             # score - threshold
    rationale: str

class ConvergenceOracle:
    CANONICAL_THRESHOLD: float = 0.5792   # |π·cos(√e)| normalized — do not override

    def __init__(self, state_engine: StateEngine,
                 fitness_fn: "FitnessFunction",
                 replication_floor: float = 0.85,
                 stochastic_gate: bool = False,
                 jitter_sigma: float = 0.02) -> None: ...

    def is_ready_to_act(self) -> ConvergenceDecision: ...
    def is_ready_to_learn_commit(self) -> ConvergenceDecision: ...
    def is_ready_to_replicate(self) -> ConvergenceDecision: ...
```

#### `AgentReplicator`

```python
@dataclass
class MutationManifest:
    numeric_deltas: dict[str, float]      # field_path -> delta
    categorical_switches: dict[str, str]  # field_path -> new_value
    structural: bool = False

class AgentReplicator:
    def __init__(self, parent: Agent,
                 governance: "GovernanceLayer",
                 adapter: "InfrastructureAdapter",
                 policy: "ReplicationPolicy") -> None: ...

    async def try_replicate(self) -> Optional["ChildHandle"]: ...
    def build_blueprint(self, mutation: MutationManifest) -> "AgentBlueprint": ...
```

#### `LearningLoop`

```python
class LearningLoop:
    def __init__(self, agent: Agent,
                 strategy: "LearningStrategy",
                 environment: "EnvironmentAdapter") -> None: ...

    async def tick(self) -> "TickResult": ...
    async def run_until(self, *, max_ticks: int | None = None,
                        stop_signal: "asyncio.Event" | None = None) -> "RunSummary": ...

    # Lifecycle hooks (override or register callbacks)
    def on_tick(self, cb): ...
    def before_learn(self, cb): ...
    def after_evaluate(self, cb): ...
    def before_replicate(self, cb): ...
    def on_shutdown(self, cb): ...
```

#### `GovernanceLayer`

```python
class GovernanceLayer:
    def __init__(self, policies: list["Policy"],
                 audit_log: "AuditLog",
                 approval_gate: Optional["HumanApprovalGate"] = None,
                 kill_switch: Optional["KillSwitch"] = None) -> None: ...

    def evaluate(self, op: "GovernedOp") -> "PolicyDecision": ...
    async def evaluate_replication_request(self, req: "ReplicationRequest") -> "PolicyDecision": ...
    def check_kill_switch(self) -> Optional["KillCommand"]: ...
    def register_child(self, child_id: str, parent_id: str, generation: int) -> None: ...
    def audit(self, event_type: str, **fields) -> "AuditRecord": ...
```

### 4.3 Minimal "Hello, Agent" Example

```python
import asyncio
from slsr_sdk import Agent, AgentConfig, LearningLoop
from slsr_sdk.strategies.learning import EpsilonGreedyBandit
from slsr_sdk.strategies.fitness import MultiplicativeSFitness
from slsr_sdk.adapters.local import LocalProcessAdapter
from slsr_sdk.environments.gym_like import CartPoleEnvAdapter

async def main():
    cfg = AgentConfig.from_yaml("config.yaml")
    agent = Agent(cfg)

    loop = LearningLoop(
        agent=agent,
        strategy=EpsilonGreedyBandit(epsilon=0.1),
        environment=CartPoleEnvAdapter(),
    )

    summary = await loop.run_until(max_ticks=10_000)
    print(summary.final_state, summary.children_spawned)

asyncio.run(main())
```

### 4.4 Custom Fitness Function Example

```python
from slsr_sdk.strategies.fitness import FitnessFunction
from slsr_sdk.state import StateVector

class MyFitness(FitnessFunction):
    def score(self, state: StateVector, context: dict) -> float:
        # Weight Q heavier than M and T
        return (state.Q ** 0.6) * (state.M ** 0.2) * (state.T ** 0.2)
```

### 4.5 Exception Hierarchy

```
SLSRError
├── EnergyError
│   ├── OutOfEnergyError
│   └── EnergyCapExceededError
├── StateError
│   └── StateCollapseError
├── GovernanceError
│   ├── PolicyDeniedError
│   ├── KillSwitchEngagedError
│   ├── ApprovalTimeoutError
│   └── GenerationLimitExceededError
├── ReplicationError
│   ├── MutationOutOfBoundsError
│   └── BlueprintInvalidError
└── ConfigError
```

---

## 5. Data Flow

### 5.1 Full Agent Lifecycle — Sequence Diagram (ASCII)

```
User        LearningLoop  EnergyMgr  StateEngine  Oracle      Replicator  Governance  Adapter  AuditLog
 │               │            │           │          │             │           │          │         │
 │──new Agent()─>│            │           │          │             │           │          │         │
 │               │── init ───>│           │          │             │           │          │         │
 │               │── init ───────────────>│          │             │           │          │         │
 │               │── init ──────────────────────────>│             │           │          │         │
 │               │── init ──────────────────────────────────────────────────── ─>│         │        │
 │               │                                                                │─append──>        │
 │               │                                                                │ AgentBorn        │
 │               │                                                                │                  │
 │── run() ─────>│                                                                │                  │
 │               │                                                                │                  │
 │    ┌──────────┴─────────────────────── TICK LOOP (repeat) ────────────────────────────────┐      │
 │    │          │                                                                │         │      │
 │    │  OBSERVE │ <──── env.step() ─────────────────────────────────────────> [EnvAdapter]  │      │
 │    │          │                                                                │         │      │
 │    │  LEARN   │── reserve(eu) ──>│                                              │         │      │
 │    │          │<── Reservation ──│                                              │         │      │
 │    │          │     (learn step runs)                                           │         │      │
 │    │          │── commit() ────>│                                               │         │      │
 │    │          │                  │                                              │         │      │
 │    │ EVALUATE │── fitness() ──────────────>│                                    │         │      │
 │    │          │<── q_score ───────────────│                                     │         │      │
 │    │          │── update_quality(q) ─────>│                                     │         │      │
 │    │          │── update_memory(m) ──────>│                                     │         │      │
 │    │          │── tick_time() ───────────>│                                     │         │      │
 │    │          │── snapshot() ────────────>│                                     │         │      │
 │    │          │<── StateVector(Q,M,T,S) ──│                                     │         │      │
 │    │          │                                                                 │         │      │
 │    │  DECIDE  │── is_ready_to_act() ──────────────────>│                        │         │      │
 │    │          │<── ConvergenceDecision ───────────────│                         │         │      │
 │    │          │── is_ready_to_replicate() ───────────>│                         │         │      │
 │    │          │<── decision ──────────────────────────│                         │         │      │
 │    │          │                                                                 │         │      │
 │    │   ACT    │── env.act(a) ──────────────────────────────────────────> [EnvAdapter]     │      │
 │    │          │                                                                 │         │      │
 │    │REPLICATE?│── if decision.is_ready ──> try_replicate() ──>│                 │         │      │
 │    │          │                                                │── reserve(repl_cost)───>│ │     │
 │    │          │                                                │<── Reservation ────────│ │     │
 │    │          │                                                │── evaluate(req) ──────>│──>     │
 │    │          │                                                │<── PolicyDecision ─────│  │     │
 │    │          │                                                │   (if HUMAN_REQ) ─> HumanGate   │
 │    │          │                                                │<── approved/denied     │  │     │
 │    │          │                                                │── spawn(blueprint) ────────>│   │
 │    │          │                                                │<── ChildHandle ────────────│   │
 │    │          │                                                │── register_child() ─────> │     │
 │    │          │                                                │                          │──> append     │
 │    │          │                                                │                          │ ReplicateEvent│
 │    │          │<── ChildHandle (or None) ──────────────────────│                          │    │         │
 │    │          │                                                                            │    │         │
 │    └──────────┴─── loop back to OBSERVE ───────────────────────────────────────────────────┘    │         │
 │               │                                                                                            │
 │<── RunSummary │                                                                                            │
```

### 5.2 Energy Flow (per tick)

```
  ┌──────────────┐    reserve(eu_learn)    ┌──────────────┐
  │              ├────────────────────────>│              │
  │ LearningLoop │                         │ EnergyManager│
  │              │<────────Reservation─────│              │
  └──────┬───────┘                         └──────┬───────┘
         │                                        │
         │   perform LEARN step                   │ (refill_rate accrues
         │                                        │  continuously in bg)
         │                                        │
         │   commit(reservation)                  │
         ├───────────────────────────────────────>│
         │                                        │
         ▼                                        ▼
   [M updated in StateEngine        [energy.consumed += eu_learn
    proportional to working-set]     E_available recalculated]
```

### 5.3 Replication Approval Flow (Generation ≥ 3)

```
AgentReplicator ──req──> GovernanceLayer ──post──> [Webhook / Slack / CLI]
                              │                            │
                              │                            │ (human reviews,
                              │                            │  signs approval)
                              │                            ▼
                              │<──────── signed_approval ──┘
                              │
                              ├─── verify signature ──────> [fail → deny]
                              │
                              ├─── policy eval ──────────────┐
                              │                              │
                              ▼                              ▼
                          ALLOW                            DENY
                              │                              │
                              ▼                              ▼
                         proceed spawn              emit denial audit
                              │                      raise PolicyDeniedError
                              ▼
                         register_child()
                              │
                              ▼
                        AuditLog.append
```

---

## 6. Replication Governance & Safety

### 6.1 Safety Invariants (Always True)

| # | Invariant | Enforcement Point |
|---|---|---|
| I1 | `generation ≤ max_generations` for every live agent | `GovernanceLayer.evaluate_replication_request` |
| I2 | Total live agents `≤ fleet_cap` | `GovernanceLayer.register_child` |
| I3 | Every replication is preceded by a successful `ConvergenceOracle.is_ready_to_replicate()` | `AgentReplicator.try_replicate` step 1 |
| I4 | Every replication is preceded by a successful `EnergyManager.reserve(replication_cost)` | `AgentReplicator.try_replicate` step 2 |
| I5 | Structural mutations require human approval | Policy engine |
| I6 | Kill switch is checked at every FSM transition | `LearningLoop.tick` |
| I7 | No bypass of `GovernanceLayer` — all replications route through `try_replicate` | Static check (linter rule) + runtime `@governed` decorator |
| I8 | Audit log is append-only and hash-chained | `AuditLog.append` |
| I9 | Canonical threshold (`CONV_THRESHOLD ≈ 0.5792`) cannot be lowered | `ConvergenceOracle.__init__` raises `ConfigError` |
| I10 | Agents cannot modify their own `GovernanceConfig` at runtime | Config is frozen after `Agent.__init__` |

### 6.2 Safety Constraint Parameters

| Parameter | Default | Hard Max | Description |
|---|---|---|---|
| `max_generations` | 5 | 10 | Max depth of replication tree |
| `fleet_cap` | 32 | 256 | Global cap on simultaneously live agents |
| `replication_cost_eu` | 100.0 | — | EU reserved per replication attempt |
| `replication_floor` | 0.85 | 1.0 | Min readiness score to replicate |
| `mutation_sigma` | 0.05 | 0.25 | Max relative change per numeric field |
| `allow_structural_mutation` | false | — | If true, still requires human approval |
| `human_approval_required_at_gen` | 3 | — | Generation at/above which human approval is mandatory |
| `approval_timeout_sec` | 600 | 3600 | How long to wait for human approval before deny |
| `kill_switch_poll_interval_sec` | 1.0 | — | How often to check external kill signal |
| `audit_log_signing_required` | true | — | If true, all records are Ed25519-signed |

### 6.3 Resource Caps

- **Per-agent:** `max_energy_cap` (EU ceiling), `max_memory_mb`, `max_cpu_pct`.
- **Per-fleet:** `fleet_total_eu_per_hour`, `fleet_max_live_agents`, `fleet_max_replications_per_day`.
- Exceeding a cap triggers `PAUSE` (per-agent) or emergency `QUARANTINE` (per-fleet).

### 6.4 Human-in-the-Loop (HITL) Options

Three HITL modes are supported, configured under `governance.approval`:

| Mode | When Invoked | Mechanism |
|---|---|---|
| `disabled` | Never | — (not recommended for generation ≥ 3) |
| `on_high_risk` | Structural mutation OR generation ≥ `human_approval_required_at_gen` OR first-time replication in a new environment | Webhook / Slack / CLI |
| `always` | Every replication | Webhook / Slack / CLI |

A `HumanApprovalGate` implementation is pluggable; built-ins:
- `WebhookApprovalGate` — POSTs signed request, expects Ed25519-signed response.
- `CLIApprovalGate` — interactive terminal prompt (dev only).
- `SlackApprovalGate` — posts to a channel, awaits reaction/button signed by an authorized user.

### 6.5 Kill Switch

The kill switch is an **out-of-band**, signed file/endpoint that the `GovernanceLayer` polls every `kill_switch_poll_interval_sec`. A valid signed command immediately:

1. Stops the `LearningLoop` before the next FSM transition.
2. Propagates to all descendants via the lineage registry.
3. Writes a final `KillEvent` audit record.
4. Executes the specified severity: `PAUSE` | `QUARANTINE` | `TERMINATE` | `ANNIHILATE`.

Example kill signal payload:
```json
{
  "agent_filter": "*",
  "severity": "TERMINATE",
  "reason": "operator_initiated_maintenance",
  "issued_at": "2026-04-19T13:57:00Z",
  "expires_at": "2026-04-19T14:57:00Z",
  "signature": "ed25519:a8c2...e7f1"
}
```

### 6.6 Forensics & Post-Incident Review

Every agent run produces a **forensics bundle**:
- Full signed audit log
- Lineage graph (GraphViz DOT + JSON)
- All state trajectories
- All mutation manifests
- All approval artifacts (signed)

Bundles are retained per `audit.retention_days` (default 90).

---

## 7. Extension Points

The SDK is designed around **protocol-based plug-in interfaces** (PEP 544). Users subclass/implement to customize behavior; the core never changes.

### 7.1 Plug-in Protocols

| Protocol | Responsibility | Where Registered |
|---|---|---|
| `LearningStrategy` | Update agent's policy/model from a batch | `LearningLoop(strategy=...)` |
| `FitnessFunction` | Compute Quality score from state + context | `ConvergenceOracle(fitness_fn=...)` |
| `ReplicationPolicy` | Decide *what* to mutate and by how much | `AgentReplicator(policy=...)` |
| `EnvironmentAdapter` | Mediate observations/actions | `LearningLoop(environment=...)` |
| `InfrastructureAdapter` | Actually spawn child processes/containers | `AgentReplicator(adapter=...)` |
| `HumanApprovalGate` | Route high-risk requests to humans | `GovernanceLayer(approval_gate=...)` |
| `AuditSink` | Persist audit records | `AuditLog(sinks=[...])` |

### 7.2 Custom Learning Strategy

```python
from slsr_sdk.strategies.learning import LearningStrategy, LearningResult

class MyPPOStrategy(LearningStrategy):
    def __init__(self, policy_net, optimizer, clip_eps=0.2):
        self.policy = policy_net
        self.opt = optimizer
        self.clip = clip_eps

    def update(self, batch, energy_budget) -> LearningResult:
        loss = self._ppo_loss(batch)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return LearningResult(
            loss=loss.item(),
            data_mass_mb=batch.nbytes / 1e6,
            speed_factor=0.9,
        )
```

### 7.3 Custom Replication Policy

```python
from slsr_sdk.strategies.replication import ReplicationPolicy
from slsr_sdk.replicator import MutationManifest

class ConservativeMutator(ReplicationPolicy):
    def propose_mutation(self, parent_config) -> MutationManifest:
        return MutationManifest(
            numeric_deltas={"learning_rate": 0.02 * parent_config.learning_rate},
            categorical_switches={},
            structural=False,
        )
```

### 7.4 Custom Infrastructure Adapter

```python
from slsr_sdk.adapters.base import InfrastructureAdapter, ChildHandle, AgentBlueprint

class FirecrackerAdapter(InfrastructureAdapter):
    async def spawn(self, blueprint: AgentBlueprint) -> ChildHandle:
        vm_id = await self._client.create_vm(blueprint.to_dict())
        return ChildHandle(agent_id=blueprint.agent_id, runtime_ref=vm_id)

    async def terminate(self, handle: ChildHandle) -> None:
        await self._client.destroy_vm(handle.runtime_ref)
```

### 7.5 Custom Policy (YAML + compiled predicates)

Users add YAML policy files; the SDK's `PolicyEngine` compiles them into callables. For richer logic, users may register Python-side policies:

```python
from slsr_sdk.governance.policies import register_policy, PolicyDecision

@register_policy(applies_to="replicate")
def deny_night_replications(request) -> PolicyDecision:
    import datetime
    if datetime.datetime.utcnow().hour in range(0, 6):
        return PolicyDecision.deny("no replication 00:00–06:00 UTC")
    return PolicyDecision.allow()
```

---

## 8. Technology Stack

### 8.1 Runtime

- **Python:** 3.10+ (uses `match`, PEP 604 union types, protocol improvements)
- **Event loop:** `asyncio`
- **Type checking:** `mypy --strict` in CI

### 8.2 Required Libraries

| Area | Library | Purpose |
|---|---|---|
| Config | `pydantic` ≥ 2.5 | Config validation, serialization |
| Config files | `pyyaml`, `tomli` | YAML/TOML config loading |
| Crypto | `cryptography` ≥ 42 | Ed25519 signing for audit / approvals |
| Logging | `structlog` | Structured, JSON-capable logs |
| Metrics | `prometheus_client` | `/metrics` endpoint |
| Async I/O | `anyio` | Cross-async-backend primitives |
| Serialization | `orjson`, `msgpack` | Fast serialization of blueprints |

### 8.3 Optional Libraries (by Adapter/Strategy)

| Adapter / Strategy | Library |
|---|---|
| Ray adapter | `ray[default]` |
| Kubernetes adapter | `kubernetes` |
| Docker adapter | `docker` |
| Gym env adapter | `gymnasium` |
| PyTorch strategies | `torch` |
| scikit-learn strategies | `scikit-learn` |
| Slack approval | `slack_sdk` |
| Webhook approval | `httpx` |

### 8.4 Dev Tooling

- `uv` or `poetry` for dependency / packaging
- `ruff` (lint + format)
- `pytest`, `pytest-asyncio`, `hypothesis` (property-based tests)
- `pre-commit` hooks: ruff, mypy, secrets scan
- `mkdocs-material` for docs site

### 8.5 Packaging

- Wheel published as `slsr-sdk` on PyPI
- Extras: `slsr-sdk[k8s]`, `slsr-sdk[ray]`, `slsr-sdk[torch]`, `slsr-sdk[all]`
- SBOM generated via `cyclonedx-py`

---

## 9. Configuration Schema

### 9.1 Canonical YAML Example

```yaml
# config.yaml — full-featured agent config
agent:
  agent_id: "agent-root-0001"
  generation: 0
  parent_id: null
  seed: 42

energy:
  initial_eu: 1000.0
  max_cap: 5000.0
  refill_rate: 10.0            # EU per second
  reference_c_squared: 1.0     # hardware-normalized (see benchmark tool)
  replication_cost_eu: 100.0
  replication_energy_floor: 200.0

state:
  maturation_horizon_sec: 3600  # T = 1.0 when agent is 1 hour old
  quality_floor: 0.05
  memory_floor: 0.05
  trajectory_window: 1024

oracle:
  # CONV_THRESHOLD (π·cos(√e) normalized ≈ 0.5792) is hard-coded — not listed here
  replication_floor: 0.85
  stochastic_gate: false
  jitter_sigma: 0.02

replication:
  max_generations: 5
  fleet_cap: 32
  mutation_sigma: 0.05
  allow_structural_mutation: false
  mutation_allowlist:
    - "learning_rate"
    - "epsilon"
    - "batch_size"

governance:
  kill_switch:
    enabled: true
    poll_interval_sec: 1.0
    signed_file_path: "/etc/slsr/kill.sig"
    public_key_path: "/etc/slsr/kill_pubkey.pem"

  approval:
    mode: "on_high_risk"        # disabled | on_high_risk | always
    human_approval_required_at_gen: 3
    timeout_sec: 600
    gate:
      type: "webhook"
      url: "https://ops.example.com/slsr/approve"
      public_key_path: "/etc/slsr/approver_pubkey.pem"

  audit:
    sink: "file"                # file | s3 | kafka
    path: "/var/log/slsr/audit.log"
    signing_required: true
    private_key_path: "/etc/slsr/agent_privkey.pem"
    retention_days: 90

  policies:
    - id: deny-structural-mutation-without-approval
      applies_to: replicate
      condition: "request.structural_mutation == true and request.approvals < 1"
      effect: deny
      severity: critical

    - id: deny-night-replications
      applies_to: replicate
      condition: "utc_hour(now) in [0,1,2,3,4,5]"
      effect: deny
      severity: high

    - id: cap-replications-per-day
      applies_to: replicate
      condition: "fleet.replications_today >= 50"
      effect: deny
      severity: medium

  resource_caps:
    max_memory_mb: 2048
    max_cpu_pct: 80
    fleet_total_eu_per_hour: 100000
    fleet_max_replications_per_day: 50

environment:
  adapter: "gym_like"
  config:
    env_id: "CartPole-v1"

strategy:
  learning:
    type: "epsilon_greedy_bandit"
    config: { epsilon: 0.1 }
  fitness:
    type: "multiplicative_s"
  replication:
    type: "conservative_mutator"

observability:
  metrics:
    enabled: true
    prometheus_port: 9100
  logging:
    level: "INFO"
    format: "json"
```

### 9.2 JSON Schema (excerpt — validation)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SLSRAgentConfig",
  "type": "object",
  "required": ["agent", "energy", "state", "oracle", "replication", "governance"],
  "properties": {
    "agent": {
      "type": "object",
      "required": ["agent_id"],
      "properties": {
        "agent_id": { "type": "string", "pattern": "^[a-zA-Z0-9_-]{3,64}$" },
        "generation": { "type": "integer", "minimum": 0, "maximum": 10 },
        "parent_id": { "type": ["string", "null"] },
        "seed": { "type": "integer" }
      }
    },
    "energy": {
      "type": "object",
      "properties": {
        "initial_eu": { "type": "number", "minimum": 0 },
        "max_cap": { "type": "number", "minimum": 0 },
        "refill_rate": { "type": "number", "minimum": 0 },
        "replication_cost_eu": { "type": "number", "minimum": 0 }
      }
    },
    "replication": {
      "type": "object",
      "properties": {
        "max_generations": { "type": "integer", "minimum": 0, "maximum": 10 },
        "fleet_cap": { "type": "integer", "minimum": 1, "maximum": 256 },
        "mutation_sigma": { "type": "number", "minimum": 0, "maximum": 0.25 }
      }
    }
  }
}
```

### 9.3 Config Loading Precedence

```
defaults (in code)  <  ~/.slsr/config.yaml  <  ./slsr.yaml  <  SLSR_* env vars  <  CLI --config
```

Later sources override earlier ones. All merges are **deep merges**, with arrays replaced (not concatenated).

---

## 10. Glossary

| Term | Definition |
|---|---|
| **Agent** | A running instance of the SDK that executes the `OBSERVE → LEARN → EVALUATE → DECIDE → ACT` cycle. |
| **Blueprint** | A serializable, immutable snapshot of an agent's config + model weights + lineage used to spawn a child. |
| **CONV_THRESHOLD** | The canonical, hard-coded convergence constant `|π·cos(√e)|` normalized to `[0,1]` ≈ `0.5792`. |
| **Energy Unit (EU)** | SDK-normalized unit of compute budget. 1 EU ≈ 1 CPU-second of work on the reference machine. |
| **Fitness** | Scalar score (default `S = Q·M·T`) describing an agent's readiness. |
| **Fleet** | The set of all simultaneously live agents rooted in the same governance domain. |
| **Generation** | The depth of an agent in the replication tree. Root = 0. |
| **Governance Invariant** | A rule that cannot be disabled, overridden, or delayed; e.g., the canonical threshold. |
| **HITL (Human-in-the-Loop)** | Any flow that blocks on a signed human approval before proceeding. |
| **Invariant Core** | The conceptual origin (from the research report) of hard-coded, non-mutable rules — implemented as `GovernanceLayer` + `CONV_THRESHOLD`. |
| **Kill Switch** | An out-of-band, cryptographically-signed command that halts an agent (and descendants). |
| **Lineage** | Ordered list of `parent_id`s from the root to a given agent. |
| **Maturation Horizon** | Wall-clock duration at which `T` saturates to 1.0. |
| **MutationManifest** | Declarative description of the differences between a parent and a proposed child. |
| **Policy** | Declarative rule evaluated by the `GovernanceLayer` to allow/deny a governed operation. |
| **Readiness Score** | Output of `ConvergenceOracle`; compared against `CONV_THRESHOLD` and task-specific floors. |
| **Replication Floor** | Task-specific minimum readiness score to replicate (≥ `CONV_THRESHOLD`). Default `0.85`. |
| **Reservation** | A held amount of EU returned by `EnergyManager.reserve()`; must be either `commit()`-ed or `release()`-ed. |
| **State Vector** | The `(Q, M, T, S)` tuple published by `StateEngine`. |
| **Structural Mutation** | A mutation that changes the *shape* of an agent (new layers, new tool access). Requires human approval. |
| **Tick** | One full pass of the `LearningLoop` FSM. |

---

### Appendix A — Open Questions for Engineering

1. Should `CONV_THRESHOLD` be **per-agent-type** (e.g., patching vs. training) rather than truly global? Current answer: **no**, keep it invariant; use the `replication_floor` override for per-task tuning.
2. Should we ship a default **neural-net-based fitness function**, or require users to always supply one? Current answer: ship `MultiplicativeSFitness` as default; power users override.
3. Should the kill switch use **Ed25519** or **Sigstore/Fulcio-based** signatures? Current answer: Ed25519 for v0.9; evaluate Sigstore for v1.1.
4. Should `EnergyManager` be **hardware-calibrated** automatically at first run, or always user-configured? Current answer: ship a `slsr benchmark` CLI for semi-automatic calibration; user can override.

### Appendix B — Implementation Roadmap

| Milestone | Scope | Target |
|---|---|---|
| M0 | Core protocols, config, `EnergyManager`, `StateEngine`, `ConvergenceOracle`, in-proc `LocalAdapter` | Week 4 |
| M1 | `GovernanceLayer` (policies, audit log, kill switch), `AgentReplicator`, `LearningLoop` | Week 8 |
| M2 | `DockerAdapter`, `KubernetesAdapter`, webhook approval gate, Prometheus metrics | Week 12 |
| M3 | Ray adapter, PyTorch strategy pack, gym environment adapter, docs site | Week 16 |
| M4 | Hardening: fuzzing, property tests, red-team exercises, v1.0 release | Week 20 |

---

**End of Document.**
