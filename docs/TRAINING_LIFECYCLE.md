# Training Lifecycle Specification

**Building in Public.** Every training run in OpenGrad is governed by an immutable configuration and an append-only event ledger.

---

## 1. Supported Training Algorithms

OpenGrad standardizes training underneath the `TrainerBackend` interface:

| Algorithm | Identifier | Implementation | Purpose |
| :--- | :--- | :--- | :--- |
| **SFT** | `sft` | `SFTTrainerBackend` | Supervised Fine-Tuning for initial tool policy and format adherence. |
| **DPO** | `dpo` | `DPOTrainerBackend` | Direct Preference Optimization to penalize over-calling and syntax violations. |
| **On-Policy Distill** | `on_policy_distillation` | `OnPolicyDistillationTrainerBackend` | Rollout generation supervised by teacher model feedback. |

---

## 2. Token & Batch Accounting

Every training launch computes and records exact token accounting:
- `micro_batch_size`
- `gradient_accumulation_steps`
- `world_size`
- `effective_global_batch_size = micro_batch_size * gradient_accumulation_steps * world_size`
- `tokens_per_update = effective_global_batch_size * max_seq_length`
- `estimated_total_tokens = tokens_per_update * max_steps`

Saved to `runs/<experiment-id>/training_metadata.json`.

---

## 3. Training Execution

To execute training in dry-run mode (CPU safe, verifies all plumbing):
```bash
opengrad train configs/experiments/m0_sft.yaml --dry-run
```

To execute on real GPU hardware:
```bash
opengrad train configs/experiments/m0_sft.yaml
```
Outputs final checkpoints to `runs/<experiment-id>/checkpoints/` and automatically registers them in the `CheckpointRegistry`.
