---
base_model: Qwen/Qwen3.5-2B
base_model_relation: finetune
library_name: transformers
license: other
license_name: composite-per-source
tags:
  - tool-calling
  - function-calling
  - dpo
  - calibration
  - qwen3.5
  - opengrad
datasets:
  - arrochi112/OpenGrad-ToolPolicy-Canonical-v2-minus-xlam
---

# OpenGrad — M1 DPO calibration on selected M0-final-v2

Direct Preference Optimization applied to the healthy selected M0-final-v2 checkpoint, not to the
base model or a historical DPO checkpoint. Produced by [OpenGrad](https://github.com/arjhinety/OpenGrad).

**Research artifact, not a production model.**

## Results (pre-registered internal confirmatory partition, 1,277 examples)

| run | `call_f1` | precision | recall | over_call | clarification | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| M0-final-v2 @1800 | 0.7470 | 0.7350 | 0.7594 | 0.1505 | 0.7682 | 0.5430 |
| **M1-v2 @30** | **0.7548** | 0.7358 | **0.7748** | 0.1529 | 0.7655 | 0.5386 |

M1 preserves the M0 calibrated frontier and makes a small improvement in call F1 and recall. It
does not materially reduce over-calling; this is calibration retention/slight improvement, not a
large frontier movement.

## Evaluation and promotion

Checkpoint selection used the frozen DEV partition (2,373 examples, fingerprint `88a56821…`).
Checkpoints 30/60 were within the pre-registered 0.01 macro tolerance, so the earlier checkpoint
30 was selected. The confirmatory partition (fingerprint `d6d1e394…`) was then scored exactly once
on checkpoint 30.

The prospective `tool_use_promotion_v4` policy passed: precision/recall/macro floors, over-call
ceiling, clarification/unsupported floors, parse validity, and M0-relative regression checks.
Checkpoint 30 is **PROMOTED**. This policy does not compare recall to B0's degenerate always-call
recall.

The confirmatory partition is **pre-registered internal evidence, not an untouched external test**.
The evaluation population has no ANSWER examples, so `no_call_accuracy` is NA. Tool-selection
accuracy, argument validity, and schema validity are not computed by the current evaluator and are
not treated as satisfied.

## Frozen lineage

- Parent experiment: `m0_sft_canonical_v2_final`
- Parent checkpoint: `checkpoint-1800`
- Parent model hash: `7144579aeecec8b4de25f193ab63085efdf8d9d76b85ed915352291b0152277a`
- Preference dataset: 481 local calibration pairs, hash `d39168948d09fc3c355cd83f9f0857f310086322b0968fd2e7d78125150faef4`
- DPO: beta 0.05, learning rate 5e-7, cosine schedule, 120 steps, seed 42, bfloat16

The preference set combines deterministic base/M0 disagreements on Canonical-v2 training prompts
with a bounded curated When2Call training slice. Frozen behavioral evaluation IDs were excluded;
no paid external API was used. All pair origins and input hashes are recorded in the repository.

## Checkpoints

All retained checkpoints are published here: `dpo-checkpoint-30` (selected), 60, 90 and 120.
The first M1 identity is preserved in the repository as a failed 119/120-step run; it was not
silently overwritten.

## Intended use

Research artifact. Not safety-tuned, not aligned, and not intended for autonomous tool use. It
inherits the limitations of its public training sources and its evaluator's unmeasured dimensions.
See the repository reports:

- `reports/M1_DPO_EXECUTION_REPORT.md`
- `reports/M1_DPO_EVALUATION.md`
- `reports/M2_DECISION.md`
