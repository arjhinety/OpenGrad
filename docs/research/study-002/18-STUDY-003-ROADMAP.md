# 18 — Study 003 roadmap: on-policy distillation

## Why this is a separate study

`docs/research/STUDIES.md:59-83` scoped Study 002 as three subjects — relabelling, on-policy distillation
and speculative decoding. This design set **splits that scope**, and the split is recorded rather than
silent ([README](README.md)):

1. Study 002 = the refusal/direct-answer mechanism and its correction.
2. Study 003 = on-policy distillation (M2 / RQ4).
3. Study 004 = speculative decoding integration (RQ5, ROADMAP step 14).

Two reasons, both of which are lessons from Study 001:

- **The subjects have no live implementation.** `reports/M2_DECISION.md` records that
  `src/opengrad/training/distillation.py` *"defaults to mock rollout and mock teacher providers; its live
  path raises `NotImplementedError`"*, that accepted rows never update student weights, that the configured
  prompt dataset is not materialized or registry-backed, that teacher identity and tokenizer are not pinned,
  and that the existing M2 run *"has no valid checkpoints, evaluation artifacts or promotion evidence."*
  Pre-registering thresholds for something that does not exist reproduces the `#24`/`#26` defect class:
  a number or a plan attached to an artifact that does not exist.
- **They answer different questions.** The refusal mechanism is a *data contract* question; distillation is
  an *optimization* question; speculative decoding is a *runtime* question. Sharing one preregistration
  across three questions means one amendment invalidates all three.

## Where Study 003 starts

Not from zero. `reports/M2_DECISION.md:30-38` already specifies what a justifiable M2 design must supply,
and that list is Study 003's requirement set verbatim:

| Requirement (from the M2 decision) | Status |
|---|---|
| a real teacher identity and revision | to be chosen and pinned |
| materialized prompt states **excluding frozen evaluation IDs** | to be built |
| student and teacher checkpoint lineage | to be recorded |
| actual on-policy student rollouts | to be implemented |
| a defined teacher-guided loss/update | to be specified |
| complete provenance | as [15](15-PROVENANCE-VALIDATORS.md) |
| real checkpoints with optimizer state | to be produced |
| the same DEV/confirmatory discipline | as [06](06-SPLIT-SPEC.md) |

The decision's own reasoning also fixes Study 003's ethics: *"Launching it would consume GPU while producing
synthetic/mock infrastructure output, not on-policy teacher-guided distillation... No M2 run is therefore
launched, and no GPU cycles are fabricated merely to maintain utilization."* Study 003 inherits that as a
rule: no arm runs to fill a schedule.

## Study 003's research questions (draft, to be pre-registered)

- **RQ3.1** Does on-policy distillation from a stronger teacher improve tool-policy behaviour on a fixed
  compute budget, against an SFT-only control matched on supervised tokens?
- **RQ3.2** Does distillation repair the refusal/answer decision, or does it inherit whatever contract the
  rollout data carries? This is the direct link back to Study 002: if the behaviour lives in the label
  contract, distillation from a teacher that answers directly should move it — and that is a *second,
  independent* mechanism test of H1.
- **RQ3.3** What does the teacher's own refusal behaviour do to the student's? A teacher that refuses is a
  label source, and Study 002's finding predicts its behaviour transfers.
- **RQ3.4** Is the tokenizer and chat-template contract between teacher and student a confound? `#34`
  already showed 3 of 21 engine flips were tokenizer-caused, and a teacher/student pair with different
  templates is the same hazard at a larger scale.

RQ3.2 is the reason Study 003 is sequenced *after* Study 002 rather than beside it: it is a replication
opportunity for the mechanism, and it only becomes interpretable once the mechanism's own study has
established a vocabulary.

## Inherited rules

Study 003 adopts, unchanged: the arm-matrix single-factor discipline ([04](04-ARM-MATRIX.md)), the seed
policy ([05](05-SEED-AND-REPRODUCIBILITY-POLICY.md)), the split and resolvability rules
([06](06-SPLIT-SPEC.md)), the metric spec ([07](07-METRIC-SPEC.md)), the statistics plan
([10](10-STATISTICS-PLAN.md)), the gate structure ([11](11-THRESHOLDS.md)), retention
([12](12-ARTIFACT-RETENTION.md)), hardware agnosticism ([13](13-HARDWARE-AGNOSTIC-EXECUTION.md)) and the
validators ([15](15-PROVENANCE-VALIDATORS.md)).

It must also adopt the **`ANSWER`-mode population** from Study 002. A distillation study evaluated on a
three-mode partition would be unable to see the behaviour it may inherit, which is L1 repeated.

## What Study 003 may not do

- It may not launch on a mock or partially-implemented path. The M2 decision already refused that, and the
  audit found the consequence of describing such a path as working (`#67`: `TRAINED` was corrected to
  `INVALID` for a run that *"executed only the mock distillation path"*).
- It may not treat the teacher's outputs as ground truth without measuring the teacher's own behaviour on
  the same partitions. A teacher that refuses is a label source with a measurable refusal rate.
- It may not reuse Study 002's `P-CONF` for selection. The one-shot and disjointness rules apply across
  studies, not only within them.
- It may not report a distillation win without an SFT-only control matched on **measured** supervised
  tokens — the `#6` defect, which is one study old and already written down.

## Sequencing

Study 003 is unblocked by Study 002 and blocked by its own implementation work. Its first deliverable is not
a run; it is a pre-registration satisfying `reports/M2_DECISION.md:30-38`, with each row marked
`LIVE`/`NOT_IMPLEMENTED` exactly as [04](04-ARM-MATRIX.md) does. If more than a stated few rows remain
`NOT_IMPLEMENTED` at the pre-GPU gate, Study 003 does not launch.