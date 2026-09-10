# Future Reinforcement Learning (RL) Integration Architecture

**Building in Public.** OpenGrad lays down rigorous architectural boundaries in advance so future reinforcement learning (RL) algorithms integrate cleanly without restructuring the repository.

---

## 1. Architectural Extension Boundary

In OpenGrad, all model training algorithms implement the `TrainerBackend` protocol defined in `src/opengrad/training/protocol.py`:

```python
class TrainerBackend(Protocol):
    name: str

    def train(
        self,
        experiment_id: str,
        config: dict[str, Any],
        output_dir: Path,
        *,
        dry_run: bool = False,
    ) -> TrainingRunResult: ...
```

When RL algorithms (e.g., GRPO, PPO, RLOO, or REINFORCE variants) are introduced, they will be added as self-contained implementations of `TrainerBackend`:

```python
class GRPOTrainerBackend(TrainerBackend):
    name = "grpo"

    def __init__(self, rollout_provider: RolloutProvider, reward_functions: list[RewardFunction]):
        self.rollout_provider = rollout_provider
        self.reward_functions = reward_functions

    def train(self, experiment_id: str, config: dict[str, Any], output_dir: Path, *, dry_run: bool = False) -> TrainingRunResult:
        # 1. Sample prompt batches
        # 2. Generate N rollouts per prompt using RolloutProvider
        # 3. Compute group relative rewards across rollouts
        # 4. Compute policy gradient update with KL penalty
        # 5. Save checkpoint and register in CheckpointRegistry
        ...
```

---

## 2. Reusable Subsystems for RL

The following components designed for SFT, DPO, and On-Policy Distillation will be reused directly by the RL engine:

1. **`RolloutProvider` (`src/opengrad/training/distillation.py`)**:
   - Decouples policy generation from parameter updates.
   - Generates multi-sample candidate trajectories from the active policy checkpoint with temperature sampling and seed tracking.
   - Emits versioned `RolloutRecord` objects with lineage metadata.
2. **`TeacherProvider` & Verifier Evaluators (`src/opengrad/training/teacher.py`, `src/opengrad/benchmarks/`)**:
   - Evaluator adapters (e.g., BFCL AST syntax checkers, IFEval deterministic rules, OpenWeights tool argument validators) will serve as rule-based ground-truth reward functions.
3. **`CheckpointRegistry` (`src/opengrad/checkpoints/registry.py`)**:
   - Manages intermediate policy checkpoints, global steps, and promotion states.
4. **`ExperimentStore` & Event Ledger (`src/opengrad/experiments/store.py`, `ledger.py`)**:
   - Records chronological training milestones, policy gradient loss, KL divergence, and reward statistics.

---

## 3. Supported RL Algorithms & Frameworks (Future Scope)

When compute and experimental milestones permit, the following frameworks can plug into OpenGrad's `TrainerBackend`:
- **GRPO (Group Relative Policy Optimization)**: Evaluates group-relative advantage estimates across parallel rollouts without training an auxiliary critic model.
- **PPO (Proximal Policy Optimization)**: Classic actor-critic reinforcement learning with value model baseline.
- **verl / Slime / Atropos**: Distributed RL backends capable of orchestrating multi-GPU rollout generation and training.

Because the experiment lifecycle, dataset manifests, and benchmark suites are completely backend-agnostic, adding GRPO will require **zero changes** to OpenGrad's evaluation, regression detection, or promotion systems.
