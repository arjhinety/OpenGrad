# M1 DPO execution report

**Status:** `COMPLETED / PROMOTED` under `tool_use_promotion_v4`, which was introduced after M0 was
evaluated; M0 also clears v4 and M1-v2 fails the v3 gate that rejected M0 (see the evaluation report)
**Experiment:** `m1_dpo_canonical_v2_final_v2`
**Parent:** selected `m0_sft_canonical_v2_final::checkpoint-1800`
**Parent weight hash:** `7144579aeecec8b4de25f193ab63085e…`
**Evaluation:** [`M1_DPO_EVALUATION.md`](M1_DPO_EVALUATION.md)

## Design

M1 tests whether preference optimization can calibrate the selected M0-final-v2 policy without
returning to the base model or destroying recovered call capability. The student and frozen
reference both start from the selected M0 checkpoint. This is not the historical base-starting
DPO identity and does not overwrite it.

The preference artifact is `data/processed/m1_calibration_preference_pairs_v1.jsonl`, 481 pairs,
sha256 `d39168948d09fc3c355cd83f9f0857f310086322b0968fd2e7d78125150faef4`:

- 241 pairs from deterministic base/M0 disagreements: 141 expected `CALL` and 100 expected `ANSWER`;
- 240 pairs from the bounded curated When2Call training slice: 40 `ANSWER`, 100 `CLARIFY`,
  100 `UNSUPPORTED` (only 73 curated `ANSWER` pairs were available);
- frozen DEV/confirmatory IDs excluded before local generation;
- no paid external API;
- five credential-like source rows excluded from the new artifact and recorded in the dataset report;
- all pair origins and input hashes retained.

## Frozen recipe

| | |
|---|---|
| Base/student initialization | M0-final-v2 checkpoint 1800 |
| Reference | independent copy of the same M0 checkpoint |
| Beta | 0.05 |
| Learning rate | 5e-7 |
| Scheduler | cosine |
| Batch / accumulation | 2 / 2 |
| Horizon | 120 steps |
| Warmup | 12 steps |
| Checkpoints | 30, 60, 90, 120 |
| Precision | bfloat16 |
| Seed | 42 |
| Preference prompt format | rendered; no second rendering |
| Policy | prospective `tool_use_promotion_v4` |

The first identity, `m1_dpo_canonical_v2_final`, is preserved as `FAILED`: it completed only
119/120 steps because a final partial accumulation was dropped after five pairs were skipped by
the length contract. M1-v2 fixes the loop to continue deterministic epochs until the exact
horizon; no scientific parameter changed.

## Training telemetry

- 120/120 optimizer steps completed.
- Final reward margin: 0.0351; preference accuracy: 1.0.
- DPO loss: 0.6931 first, 0.6757 final, minimum 0.0192, mean 0.5980.
- Peak VRAM: 35.1 GiB.
- No OOM, NaN/Inf, interruption or restart.
- Optimizer/scheduler/RNG state recorded on the newest checkpoint; all four model checkpoints retained.

## Lineage

The experiment record (`runs/m1_dpo_canonical_v2_final_v2/experiment.json`) records:

- parent checkpoint ID `m0_sft_canonical_v2_final::checkpoint-1800`;
- parent model hash;
- preference dataset ID/hash;
- clean launch commit and exact configuration identity.

The per-checkpoint entries in `runs/checkpoint_registry.json` do not: `parent_checkpoint` is `null`
for all four M1-v2 checkpoints.

M1 is ready for its evaluation conclusion; it is not evidence that DPO improves unmeasured tool
selection, argument validity or schema validity.
