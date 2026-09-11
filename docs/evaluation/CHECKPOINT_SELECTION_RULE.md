# Checkpoint selection rule (frozen before the M0-final run)

**Status:** `FROZEN` before training started. This document is the rule, written and committed
before any checkpoint from `m0_sft_canonical_v2_final` was evaluated, so the rule cannot be
shaped by the results it will judge.

## Why it is frozen in advance

The partial-v2 experiment selected its checkpoint by taking the maximum `call_f1` over four
checkpoints scored on the same examples it then reported. That is a maximum over several draws,
not an unbiased measurement, and it is also the wrong quantity to maximise: B0 scores 0.6191 on
`call_f1` while calling a tool on 64.3% of examples whose correct answer is not a call. A rule
keyed on `call_f1` prefers a degenerate policy.

Writing the rule down first is what makes the selection auditable. It is not a formality: the
rule below is stated in terms that could have rejected every checkpoint.

## The rule

Applied to every scheduled checkpoint, evaluated on the **DEV** partition
(`reports/evaluation/behavioral-heldout-v2-partition.json`, fingerprint `88a56821…`).

### Step 1 — eligibility

A checkpoint is eligible only if all of the following hold on DEV:

| Constraint | Threshold | Rationale |
|---|---|---|
| `parse_valid_rate` | ≥ 0.99 | Below this the other numbers are not interpretable |
| `over_call_rate` | ≤ 0.20 | Calling on requests whose gold answer is not a call |

### Step 2 — ranking

Eligible checkpoints are ranked by **`macro_behaviour_score`**, the mean per-class recall over the
classes the DEV partition can actually measure (`must_call_accuracy`, `clarification_accuracy`,
`unsupported_accuracy`). `no_call_accuracy` is excluded because the population contains zero
`ANSWER` examples, so it is 0.0 for every model and would drag every score down by the same
constant — see `src/opengrad/promotion/tool_use_policy.py`, policy
`tool_use_promotion_v3`.

Higher is better. This is the balanced metric the README's "measurement, not leaderboard chasing"
principle refers to, and it is the metric on which the corrected partial-v2 model is ahead of B0
by 0.28 while `call_f1` is very slightly behind.

### Step 3 — tie-break

Two checkpoints are **tied** if their macro scores differ by less than `0.01`. The DEV partition
has 2,373 examples, which is too small to resolve a difference of a few points — a point estimate
alone would be a false ranking. On a tie, the **earlier** (smaller-step) checkpoint is selected,
because it is the more training-efficient of two effectively equal models and the later one gives
no measurable evidence of being better.

### Step 4 — record

The verdict records the selected checkpoint, its macro score, every ineligible checkpoint with the
constraint it failed, and whether the tie-break decided it. A run in which no checkpoint is
eligible selects nothing and reports that.

## What is recorded but is not the selection metric

Reported for every checkpoint, and used for the scientific comparison rather than the choice:

`call_precision`, `call_recall`, `call_f1`, `over_call_rate`, `under_call_rate`,
`clarification_accuracy`, `unsupported_accuracy`, `must_call_accuracy`, and the confusion matrix.

Their absence from the ranking is deliberate: a single headline metric is what produced the
degenerate B0 policy the project exists to correct.

## Promotion is separate from selection

Selecting a checkpoint on DEV does **not** promote it. Promotion applies
`tool_use_promotion_v3` against B0, which additionally requires `call_f1` retention, absolute
behavioural floors, and per-dimension non-regression.

A checkpoint can therefore be **selected and not promoted**, and that outcome is expected rather
than a failure: the non-regression constraint on `call_recall` is a real measurement, and B0's
recall of 0.9722 is itself a property of a degenerate always-call policy, so any model that fixes
over-calling necessarily loses raw recall. The recorded partial-v2 checkpoint 1200 fails exactly
that constraint today, at −0.467 against a 0.10 allowance. That verdict is left as it is; the
constraint is not relaxed to make a run look better.

If the selected checkpoint fails promotion, the run is reported as `COMPLETED / NOT PROMOTED`.

## Confirmatory evaluation

The **CONFIRMATORY** partition (fingerprint `d6d1e394…`, 1,277 examples) is evaluated **exactly
once**, on the selected checkpoint, after selection is complete. It is never used to choose a
checkpoint, change steps, change the learning rate, change thresholds, change the source mixture,
or select generation settings.

It is a **pre-registered internal confirmatory partition**, not an untouched external benchmark:
the wider upstream population has already influenced earlier OpenGrad work, and saying otherwise
would overstate the evidence.
