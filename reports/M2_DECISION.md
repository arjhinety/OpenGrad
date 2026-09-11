# M2 distillation decision

**Decision:** `NOT RUN / NOT JUSTIFIED`
**Date:** 2026-09-11
**Preceding result:** M1-v2 DPO, promoted checkpoint 30

M1-v2 preserved the selected M0 frontier and made a small favorable movement: confirmatory
`call_f1` 0.7470 → 0.7548, recall 0.7594 → 0.7748, precision 0.7350 → 0.7358, over-call
0.1505 → 0.1529, clarification 0.7682 → 0.7655, unsupported 0.5430 → 0.5386. It passes every
measurable requirement of prospective `tool_use_promotion_v4` and is authoritatively promoted.
The result does not expose a measured calibration failure that requires teacher-guided refinement.

The existing M2 configuration and implementation cannot answer a valid additional question:

- `src/opengrad/training/distillation.py` defaults to mock rollout and mock teacher providers;
- its live path raises `NotImplementedError`;
- accepted rows never update student weights or save a real model checkpoint;
- the configured prompt dataset is not materialized or registry-backed;
- teacher identity, revision, tokenizer and scoring contract are not pinned;
- the existing M2 run has no valid checkpoints, evaluation artifacts or promotion evidence.

Launching it would consume GPU while producing synthetic/mock infrastructure output, not
on-policy teacher-guided distillation. Replacing that path now would be a new implementation and
scientific design, not a continuation of the frozen M2 experiment. No M2 run is therefore launched,
and no GPU cycles are fabricated merely to maintain utilization.

## What remains justified

M2 can be reconsidered after a separate design freeze that supplies a real teacher identity and
revision, materialized prompt states excluding frozen evaluation IDs, student and teacher
checkpoint lineage, actual on-policy student rollouts, a defined teacher-guided loss/update,
complete provenance, real checkpoints with optimizer state, and the same DEV/confirmatory discipline.
The open limitations that could motivate that future design are evaluator gaps in tool selection,
argument validity, schema validity, and the single-seed uncertainty — not evidence that the current
mock M2 path is ready.
