# OpenGrad Experiment Foundation Audit

**Motto: Building in Public.** Every hypothesis, checkpoint, and failure is documented with empirical rigor.

---

## 1. Executive Summary

This audit assesses the state of the OpenGrad repository as of Phase 0.5/Phase 1.0 transition. OpenGrad currently provides validated CPU-safe research infrastructure, an immutable canonical tool-use dataset release (`arrochi112/OpenGrad-ToolPolicy-Canonical-v1`), a frozen baseline evaluation runner, and a post-training benchmark suite (covering Tiers A through E, including speculative decoding and OpenWeights mobile tool-calling evaluation).

To turn OpenGrad into an **agent-operable post-training experiment operating system**, the repository must transition from static script execution into an **experiment-first lifecycle**:
```text
Hypothesis → Immutable Config → Dataset Snapshot → Preflight Validation → Training (SFT/DPO/Distill) → Checkpoint Registration → Versioned Benchmarks → Failure Analysis → Regression Detection → Promotion Policy → Reproducible Artifacts
```

---

## 2. Existing Architecture & Reusable Components

| Component | Path | Status | Reusability |
| :--- | :--- | :--- | :--- |
| **Canonical Tool Schema** | `src/opengrad/data/canonical.py` | Working | Reusable for SFT trajectories and evaluation records. |
| **Renderer Engine** | `src/opengrad/data/renderers.py` | Working | Reusable for Qwen3.5-2B chat template rendering & tokenization. |
| **Dataset Materialization** | `src/opengrad/data/materialize.py` | Working | Sharded Parquet streaming, resumable checksums. |
| **Contamination Scanner** | `src/opengrad/contamination/scanner.py` & `benchmarks/contamination/` | Working | Reusable for 5-level contamination screening. |
| **Environment Capture** | `src/opengrad/env_capture.py` | Working | Reusable for hardware, OS, git, and python dependency capture. |
| **Benchmark Subsystem** | `src/opengrad/benchmarks/` | Working | Complete adapter suite (BFCL, tau3, ACEBench, IFEval, LiveBench, MMLU-Pro, OpenWeights, Microsuite, etc.). |
| **Speculative Engine** | `src/opengrad/benchmarks/speculative/` | Working | Speedup accounting, MTP per-depth telemetry, Pareto analysis. |
| **Failure Taxonomy** | `src/opengrad/benchmarks/taxonomy.py` | Working | Reusable canonical 27-code failure taxonomy. |

---

## 3. Missing & Weak Components

### Missing Components:
1. **Experiment Identity & Ledger**: No centralized `ExperimentStore` or append-only event ledger tracking lifecycle state transitions (`CREATED`, `PREFLIGHT`, `TRAINING`, `TRAINED`, `EVALUATED`, `PROMOTED`, `REJECTED`).
2. **Unified Training Backend Protocol**: SFT, DPO, and On-Policy Distillation exist as disconnected scripts or planned YAML files rather than implementations of a typed `TrainerBackend` protocol.
3. **RolloutProvider & TeacherProvider**: On-policy distillation requires clean abstraction for generating student rollouts and acquiring teacher supervision/scores without hardcoding external APIs.
4. **Checkpoint Registry**: Checkpoints are stored as ad-hoc paths without lifecycle tracking, hash fingerprinting, or formal promotion/rejection states.
5. **Automated Regression Engine & Promotion Policy**: No automated evaluator comparing a candidate against an explicit baseline checkpoint to enforce tolerance gates.
6. **Template & Masking Inspector**: Missing a diagnostic CLI tool (`opengrad inspect-template`) rendering token IDs and labels side-by-side to catch assistant loss masking bugs.
7. **Failure Clustering**: Missing automated grouping of benchmark task failures across failure categories.
8. **Future RL Architecture Boundary**: Clean interface boundary for future GRPO/PPO/verl integration without rewriting the codebase.

### Weak Components:
1. **CLI Experience**: Subcommands currently split between `cli.py` and `benchmarks/cli.py`. Need a unified agent-safe CLI with `--json` support and stable error codes.
2. **Experiment Diff**: Need a command (`opengrad experiment diff`) highlighting multi-variable divergences between runs to support causal reasoning.

---

## 4. Architectural Risks & Mitigations

1. **Risk: Training / Evaluation Leakage**
   * *Mitigation:* Enforce strict dataset manifest checks during preflight; reject runs where evaluation splits or contaminated datasets appear in training manifests.
2. **Risk: Prompt & Tokenizer Drift**
   * *Mitigation:* Require explicit prompt template fingerprinting and tokenizer checksums in resolved experiment configurations.
3. **Risk: Silent Failure or Fallback to Incompatible Defaults**
   * *Mitigation:* Strict schema validation rejecting unknown keys and raising explicit errors on missing research parameters.
4. **Risk: Untracked Multi-Variable Changes**
   * *Mitigation:* `opengrad experiment diff` warns whenever multiple hyperparameters/datasets change simultaneously between baseline and candidate.

---

## 5. Implementation Roadmap

- **Phase A**: Audit + lifecycle design (`docs/EXPERIMENT_FOUNDATION_AUDIT.md`).
- **Phase B**: Canonical experiment schema, store, and append-only event ledger (`src/opengrad/experiments/`).
- **Phase C**: Immutable dataset manifests and preflight validation gates.
- **Phase D**: Tokenizer, template, and assistant loss-mask inspection (`opengrad inspect-template`).
- **Phase E**: Unified `TrainerBackend` implementations (`SFTTrainerBackend`, `DPOTrainerBackend`, `OnPolicyDistillationTrainerBackend`, `RolloutProvider`, `TeacherProvider`).
- **Phase F**: `CheckpointRegistry` with promotion lifecycles (`CANDIDATE`, `EVALUATING`, `PROMOTED`, `REJECTED`, `ARCHIVED`).
- **Phase G & H**: Regression detection engine (`RegressionEngine`) and deterministic promotion policy (`PromotionPolicy`).
- **Phase I**: Failure analysis and clustering engine (`FailureAnalyzer`).
- **Phase J**: Experiment diff and cost/resource telemetry.
- **Phase K**: Agent-safe CLI (`opengrad train`, `evaluate`, `compare`, `failures`, `checkpoint`, `experiment`, `promote`, `reject`, `doctor`).
- **Phase L & M**: Comprehensive CPU-safe tests, golden fixtures, and documentation (`FUTURE_RL_INTEGRATION.md`, `AGENT_INTEGRATION.md`, etc.).
