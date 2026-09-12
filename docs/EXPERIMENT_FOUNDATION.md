# OpenGrad Experiment Foundation

**Building in Public.** Every hypothesis is stated in advance, every intervention is controlled, and every regression is published.

---

## 1. Core Principle

In OpenGrad, **model training is not the experiment**. The experiment is the closed, reproducible scientific loop:

```text
Hypothesis
   ↓
Immutable Configuration (ExperimentConfig)
   ↓
Dataset Snapshot & Manifests
   ↓
Preflight Validation Gate
   ↓
Training (SFT / DPO / On-Policy Distillation)
   ↓
Checkpoint Registration (CANDIDATE)
   ↓
Versioned Multi-Tier Evaluation
   ↓
Failure Analysis & Clustering
   ↓
Automated Regression Detection vs. Baseline
   ↓
Deterministic Promotion Policy (PROMOTE / REVIEW / REJECT)
   ↓
Reproducible Artifacts & Append-Only Event Ledger
```

---

## 2. Canonical Artifact Contract

Every experiment run produces an immutable, machine-readable directory under `runs/<experiment-id>/`:

```text
runs/<experiment-id>/
├── resolved_config.yaml     # Fully frozen configuration with no implicit defaults
├── experiment.json          # Machine-readable experiment metadata and status
├── environment.json         # Python, PyTorch, CUDA, driver, and hardware environment
├── ledger.jsonl             # Append-only chronological event ledger
├── dataset_manifests/       # Manifests and deterministic fingerprints of input datasets
├── logs/                    # Training and evaluation logs
├── metrics/                 # Time-series training loss and evaluation scores
├── checkpoints/             # Saved model checkpoints and step metadata
├── eval/                    # Benchmark run results and predictions
├── failures/                # Normalized failure records and cluster summaries
├── regression/              # Regression reports against the designated baseline
└── promotion/               # Formal promotion verdict and rule evaluations
```

---

## 3. Experiment States

An experiment progresses strictly through defined states:
- `CREATED`: Configuration instantiated, awaiting preflight.
- `PREFLIGHT`: Running automated validation gates.
- `TRAINING`: Active model weight optimization.
- `TRAINED`: Training concluded, candidate checkpoint registered.
- `EVALUATING`: Executing benchmark evaluation suites.
- `EVALUATED`: Benchmark results collected and normalized.
- `REVIEW`: Candidate under regression and safety evaluation.
- `PROMOTED`: Checkpoint meets all non-regression gates and is accepted.
- `REJECTED`: Checkpoint violated regression or quality thresholds.
- `ARCHIVED`: Superseded or historical checkpoint.
- `FAILED`: Execution terminated with fatal error.
- `INVALID`: Preserved as evidence, but not a valid training result — for example a mock-provider
  infrastructure pass that produced no model artifact. A record must not carry `INVALID` together
  with an effective trained or promoted claim in its metadata.

---

## 4. Illustrative Future Record

This is a schema-shaped example only; it is not a run and contains no result:

```yaml
experiment_id: example-only
status: EXAMPLE

model:
  family: qwen
  revision: <immutable revision>

intervention:
  type: sft

data:
  mixture: <versioned config>

evaluation:
  benchmark_revision: <commit>

environment:
  hardware: <captured>
  software: <captured>

results:
  capability: <not-run>
  regressions: <not-run>
  efficiency: <not-run>
```

Use the real [experiment schema](../registry/experiments.schema.json) and
[experiment report template](../hf/EXPERIMENT_REPORT_TEMPLATE.md) for actual records.
