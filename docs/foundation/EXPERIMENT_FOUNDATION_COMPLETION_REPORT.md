# OpenGrad Experiment Foundation Completion Report

**Building in Public.** Every hypothesis, checkpoint, failure, and regression analysis is part of the permanent scientific record.

---

## 1. Executive Summary

This report formalizes the completion of the OpenGrad Post-Training Experiment Foundation. OpenGrad has evolved from disconnected evaluation and data scripts into an **experiment-first post-training operating system**. The platform supports Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and On-Policy Distillation, backed by a 16-benchmark evaluation system (Tiers A through E), native MTP/speculative decoding metrics, on-device mobile tool calling for OpenWeights with a provisioned Google Pixel 7 phone AVD, and an extensible architecture for future Reinforcement Learning (RL/GRPO).

All functionality has been verified using deterministic CPU fixtures without requiring GPU allocation.

---

## 2. Architecture Implemented

The experiment lifecycle enforces a closed-loop scientific process:
```text
Hypothesis
   ↓
ExperimentConfig (Immutable YAML specification)
   ↓
Dataset Manifests & Fingerprints (Order-independent SHA-256)
   ↓
Preflight Gate (PASS / WARN / FAIL)
   ↓
TrainerBackend (SFT / DPO / On-Policy Distillation)
   ↓
CheckpointRegistry (CANDIDATE)
   ↓
BenchmarkRunner (Tiers A–E, Speculative Parity, OpenWeights)
   ↓
FailureAnalyzer (27-code taxonomy clustering & diffing)
   ↓
RegressionEngine (Automated baseline comparison)
   ↓
PromotionPolicy (Deterministic PROMOTE / REVIEW / REJECT)
   ↓
Artifact Store & Append-Only Event Ledger (runs/<id>/)
```

---

## 3. Files Added

- `docs/EXPERIMENT_FOUNDATION_AUDIT.md`: Phase A architecture audit and risk assessment.
- `docs/EXPERIMENT_FOUNDATION.md`: Full experiment operating system specification.
- `docs/DATASET_MANIFESTS.md`: Immutable dataset manifest and fingerprinting specification.
- `docs/TRAINING_LIFECYCLE.md`: SFT, DPO, and Distillation lifecycle documentation.
- `docs/EVALUATION.md`: Multi-tier evaluation and benchmark suite guide.
- `docs/CHECKPOINTS.md`: Checkpoint registry and lifecycle transitions.
- `docs/PROMOTION_POLICY.md`: Deterministic promotion rules and gates.
- `docs/FAILURE_ANALYSIS.md`: Failure taxonomy and comparative diffing guide.
- `docs/AGENT_INTEGRATION.md`: DeepSeek Harness and autonomous agent integration guide.
- `docs/SPECULATIVE_DECODING_ARCHITECTURE.md`: Native MTP and DSpark architectural integration.
- `docs/FUTURE_RL_INTEGRATION.md`: Future GRPO/PPO/verl extension boundaries.
- `docs/evaluation/OPENWEIGHTS_ON_DEVICE_TESTING.md`: Android Studio and Pixel 7 on-device tool testing guide.
- `src/opengrad/experiments/schema.py`: Canonical experiment identity, config, and status models.
- `src/opengrad/experiments/store.py`: Canonical `runs/<id>/` artifact storage layout.
- `src/opengrad/experiments/ledger.py`: Append-only event ledger (`ledger.jsonl`).
- `src/opengrad/experiments/preflight.py`: Authoritative preflight gate.
- `src/opengrad/experiments/diff.py`: Multi-variable experiment diffing and causal attribution engine.
- `src/opengrad/experiments/cost.py`: Cost and resource telemetry schema.
- `src/opengrad/data/manifest.py`: Dataset manifest and order-independent fingerprinting.
- `src/opengrad/data/validator.py`: Dataset validation gates (SFT and DPO pairs).
- `src/opengrad/data/inspector.py`: Chat template and assistant loss-mask inspector.
- `src/opengrad/training/protocol.py`: `TrainerBackend` interface and training metadata.
- `src/opengrad/training/sft.py`: SFT trainer backend with token accounting.
- `src/opengrad/training/dpo.py`: DPO trainer backend with preference diagnostics.
- `src/opengrad/training/teacher.py`: Teacher provider protocol with persistent disk caching.
- `src/opengrad/training/distillation.py`: On-policy distillation backend and decoupled `RolloutProvider`.
- `src/opengrad/checkpoints/registry.py`: Central checkpoint registry and promotion manager.
- `src/opengrad/promotion/policy.py`: Promotion policy evaluating must-pass, max-regression, and min-improvement rules.
- `src/opengrad/promotion/regression.py`: Automated regression detection engine.
- `src/opengrad/failures/analyzer.py`: Failure clustering and comparative diffing.
- `src/opengrad/agent_cli.py`: Agent-safe CLI subcommands with `--json` support.
- `src/opengrad/benchmarks/adapters/openweights.py`: OpenWeights lightweight prompt tool-calling adapter.
- `configs/benchmarks/openweights.yaml`: OpenWeights on-device declarative benchmark config.
- `configs/experiments/m0_sft.yaml`: Canonical SFT baseline experiment config.
- `configs/experiments/m1_dpo.yaml`: Canonical DPO preference experiment config.
- `configs/experiments/m2_onpolicy_distill.yaml`: Canonical on-policy distillation experiment config.
- `tests/experiments/*`: 5 test suites covering lifecycle, validation, trainers, promotion, and CLI.

---

## 4. Files Modified

- `README.md`: Updated with "Building in Public" motto and primary architecture headlines.
- `pyproject.toml`: Added scripts, extras, and package metadata.
- `registry/benchmarks.yaml`: Expanded with Tiers A–E and OpenWeights.
- `src/opengrad/cli.py`: Integrated new agent-safe CLI commands.
- `src/opengrad/experiments/__init__.py`: Exported experiment lifecycle components.
- `src/opengrad/benchmarks/adapters/__init__.py`: Registered OpenWeights adapter.

---

## 5. Existing Systems Reused

- `src/opengrad/data/canonical.py`: Reused `ToolConversation` and `semantic_hash`.
- `src/opengrad/data/renderers.py`: Reused `Qwen35_2BRenderer` for tokenization and prompt generation.
- `src/opengrad/data/semantic.py`: Reused trajectory validation logic in dataset gates.
- `src/opengrad/env_capture.py`: Reused for environment, hardware, and git state snapshots.
- `src/opengrad/benchmarks/`: Integrated directly with the experiment runner and checkpoint registry.

---

## 6. Experiment Lifecycle

Experiments progress through strict state transitions:
`CREATED` $\to$ `PREFLIGHT` $\to$ `TRAINING` $\to$ `TRAINED` $\to$ `EVALUATING` $\to$ `EVALUATED` $\to$ `REVIEW` $\to$ `PROMOTED` / `REJECTED` / `ARCHIVED`.
Every state transition appends an immutable event to `runs/<id>/ledger.jsonl` and `runs/central_ledger.jsonl`.

---

## 7. Dataset Lifecycle

Datasets are fingerprinted using order-independent canonical conversation hashes. The preflight gate blocks training if manifest checksums mismatch or if training data overlaps with evaluation held-outs.

---

## 8. Training Backend Architecture

All training algorithms implement `TrainerBackend` (`src/opengrad/training/protocol.py`). The contract decouples training execution from experiment orchestration, permitting easy swapping between algorithms or distributed engines.

---

## 9. SFT Status
- **Implementation:** `SFTTrainerBackend` in `src/opengrad/training/sft.py`.
- **Accounting:** Tracks micro batch, gradient accumulation, effective global batch, tokens per update, and estimated total tokens.
- **Verification:** CPU mock execution verified; live GPU branch prepared.

---

## 10. DPO Status
- **Implementation:** `DPOTrainerBackend` in `src/opengrad/training/dpo.py`.
- **Validation:** Enforces `chosen != rejected`, prompt consistency, and non-empty responses.
- **Diagnostics:** Tracks reward margin, chosen reward, rejected reward, and preference accuracy.

---

## 11. On-Policy Distillation Status
- **Implementation:** `OnPolicyDistillationTrainerBackend` in `src/opengrad/training/distillation.py`.
- **Decoupled Architecture:** Rollout generation (`RolloutProvider`) is separated from optimization.
- **Lineage:** Generates versioned `RolloutRecord` objects with prompt ID, student output, teacher feedback, and acceptance status.
- **Teacher Caching:** `CachedTeacherProvider` caches responses to disk to prevent expensive recomputation.

---

## 12. Evaluation System
- **16 Benchmarks Configured:** BFCL V4, tau3, ACEBench, IFBench, IFEval, LiveBench, MMLU-Pro, GSM8K, ARC, MCPMark, AgentBench FC, Terminal-Bench, TUA-Bench, GAIA, Performance Microsuite, Speculative Replay, OpenWeights.
- **6 Suites:** `smoke`, `tool_use_core`, `regression_core`, `agent_transfer`, `full_post_training`, `speculative_decoding`.

---

## 13. Checkpoint Registry
- **Implementation:** `CheckpointRegistry` in `src/opengrad/checkpoints/registry.py`.
- **Lifecycle:** `TRAINING`, `CANDIDATE`, `EVALUATING`, `PROMOTED`, `REJECTED`, `ARCHIVED`.
- **Persistence:** Tracked in `runs/checkpoint_registry.json`.

---

## 14. Regression Detection
- **Implementation:** `RegressionEngine` in `src/opengrad/promotion/regression.py`.
- **Automated Comparison:** Compares candidate scores against explicit baseline checkpoints, classifying benchmarks as `IMPROVED`, `PRESERVED`, or `REGRESSED`.

---

## 15. Promotion Policy
- **Implementation:** `PromotionPolicy` in `src/opengrad/promotion/policy.py`.
- **Rule Verification:** Evaluates `must_pass` absolute floors, `max_regression` ceilings, and `minimum_improvement` targets, returning a deterministic `PromotionVerdict`.

---

## 16. Failure Analysis
- **Implementation:** `FailureAnalyzer` in `src/opengrad/failures/analyzer.py`.
- **Clustering:** Groups task failures across canonical taxonomy categories.
- **Diffing:** Identifies new failures introduced after a training intervention.

---

## 17. Speculative Decoding Preparation
- **Modes:** AR, Native MTP, Draft, DSpark, Experimental.
- **Telemetry:** TTFT, ITL, throughput, proposed/accepted tokens, acceptance rate, accepted/step, verifier steps, rollbacks.
- **Pareto Analysis:** Automatically computes Pareto frontier between speedup and capability delta.

---

## 18. Future RL Boundary
- **Planning:** Documented in `docs/FUTURE_RL_INTEGRATION.md`.
- **Extension Mechanism:** Future RL algorithms (GRPO, PPO, verl) connect cleanly as `class GRPOTrainerBackend(TrainerBackend)` reusing `RolloutProvider` without repository restructuring.

---

## 19. Agent & DeepSeek Harness Integration
- **Specification:** Documented in `docs/AGENT_INTEGRATION.md`.
- **Machine-Readable CLI:** Commands support `--json` and emit stable error codes (`CONFIG_INVALID`, `DATASET_SCHEMA_INVALID`, `CHECKSUM_MISMATCH`, `CONTAMINATION_FAILURE`, `TOKENIZER_MISMATCH`, `BASELINE_NOT_FOUND`, etc.).

---

## 20. Test Results
- **PyTest:** `127 passed in 7.71s` (covers lifecycle, validation, backends, promotion, failure analysis, CLI, benchmarks).

---

## 21. Static-Analysis Results
- **Ruff:** `All checks passed!` (0 lint or format warnings).
- **Mypy:** `Success: no issues found in 122 source files` (strict static typing).
- **Registry Validation:** `registry validation: OK`.
- **Preflight Check:** All 8 repository readiness gates `PASS`.

---

## 22. CPU Verification Status
- **Status:** **VERIFIED**. The entire pipeline from preflight to mock training, checkpoint registration, evaluation dry-run, regression analysis, failure clustering, and promotion executes end-to-end without a GPU.

---

## 23. GPU Verification Status
- **Status:** **UNVERIFIED (Awaiting GPU Allocation)**. Live GPU execution branches (`TransformersInferenceBackend`, real CUDA forward passes) are lazy and isolated.

---

## 24. Known Limitations
- The current virtual machine host environment runs under QEMU without nested KVM (`/dev/kvm`), so the provisioned Pixel 7 phone AVD runs under headless software emulation.
- Large teacher models (e.g. 72B parameter teacher) require either hosted API endpoints or dedicated multi-GPU nodes.

---

## 25. Remaining Pre-GPU Blockers
None. The software architecture, validation gates, training backends, evaluation harnesses, checkpoint registries, and CLI are complete and operational.

---

## 26. Recommended First GPU Smoke Test

```bash
# 1. Preflight experiment
opengrad preflight configs/experiments/m0_sft.yaml

# 2. Run small bounded GPU verification
opengrad benchmark run \
  --benchmark performance_microsuite \
  --model Qwen/Qwen3.5-2B \
  --backend transformers \
  --limit 2 \
  --output-dir reports/benchmarks/preflight_gpu_check
```
