# 17 — Paper outline

The paper is a **mechanism** paper with a **correction** attached, and its structure follows the verdict
function rather than a narrative. Every section states which claim family it carries, and no section may
contain a number that [15](15-PROVENANCE-VALIDATORS.md) `V3`/`V10` cannot attach to an artifact.

## Structure

| § | Section | Claim family | Evidence |
|---|---|---|---|
| 1 | Introduction: a promoted checkpoint that refuses to answer | — | `docs/EXPERIMENT_RESULTS.md:13-16`, `docs/PROMOTION_POLICY.md:31-43` |
| 2 | Background: the corpus, the training contract, the gate | — | `ROADMAP.md`, `runs/m0_sft_canonical_v2_final/` |
| 3 | **Method 1 — the audit**: 100 unsupported claims, and what kind of error they were | descriptive | the audit ledger; [01](01-LESSONS-FROM-STUDY-001.md) |
| 4 | **Method 2 — the vacuity**: a metric that encoded "unmeasured" as `0.0` | descriptive | `tool_use_policy.py:33-37`, `:92-95`, `:171-172`; [07](07-METRIC-SPEC.md) |
| 5 | **Experiment**: arms, partitions, seeds, thresholds | confirmatory | [04](04-ARM-MATRIX.md), [06](06-SPLIT-SPEC.md), [11](11-THRESHOLDS.md) |
| 6 | **Result 1 — the mechanism**: `C0` collapses, `R1` does not | confirmatory H1 | `P-CONF`, `ANSWER` strata |
| 7 | **Result 2 — the boundary**: 0-shot vs 8-shot vs elicit | confirmatory H2 | [08](08-SENTINEL-SPEC.md) |
| 8 | **Result 3 — what the label vs the target text carries** | confirmatory H4 | `R2` vs `R1` |
| 9 | **Competing explanations**: narrow-SFT (`C1`), source-specific (`C2`), exposure (`X1`) | confirmatory | [10](10-STATISTICS-PLAN.md) |
| 10 | **Safety and tool-policy non-regression** | confirmatory H5, H6 | `P-UNANS`, `S-TP-4` |
| 11 | **Dose and replication** | exploratory H3, H7 | Tier C |
| 12 | **What this cost, and what we threw away** | descriptive | the cost ledger |
| 13 | Limitations | — | [14](14-HETEROGENEITY-POLICY.md), [06](06-SPLIT-SPEC.md) |
| 14 | Reproduction | — | [12](12-ARTIFACT-RETENTION.md) |
| A | The claim ledger, in full | — | every finding, including those against this study |

## Abstract rules

Only **confirmatory-family** results appear in the abstract, with their margins. Concretely:

- an exploratory result (Tier C, dose, replication) may not appear;
- a result marked `WITHIN_NOISE` may not appear;
- any number in the abstract prints its population and its subject (`R1` on `ANSWER`-gold, not "the model");
- the abstract must contain the verdict string itself, because the verdict is computed and the abstract is
  the one place a reader may not see the surrounding table;
- if H5 or H6 failed, that goes in the abstract. A paper that reports a restored direct-answer rate while
  burying a `SAFETY_REGRESSION` in §10 has reproduced the original defect in a new place.

## Standing rules for every table

1. Every row carries `n`, the denominator, the population policy and the device class
   ([07](07-METRIC-SPEC.md), [11](11-THRESHOLDS.md)).
2. No table mixes a Study 002 number with a Study 001 frozen number without the evaluator version and
   partition on both (G9; `V11`).
3. Every comparison prints its resolvable margin or is marked `WITHIN_NOISE` (`V12`).
4. Ability, decision and format appear as three separate columns. No sentence attributes a decision-layer
   change to ability ([07](07-METRIC-SPEC.md)).
5. A refusal-derived number prints `refusal_detection_method` and the measured detector error
   ([07](07-METRIC-SPEC.md)).

## Sections that most papers omit and this one will not

- **§3, the audit of the authors' own prior work.** 100 unsupported claims, categorised, with the finding
  that none were invented. This is the paper's scientific warrant: the correction being proposed is derived
  from a measured failure of the previous method, not from a preference.
- **§12, discarded work.** The 768-token MMLU-Pro pass, superseded runs, and the tokens-mismatch ablation
  (`#6`: 1.19× the reference's supervised tokens for a "matched" arm) all appear with their cost.
- **§A, the claims against this study.** Including any claim in this design set that the run contradicts.
  Study 001's `ERRATA.md` and the `INVALID` annotation on the mock M2 run are the precedent for correcting
  in public rather than silently.

## What the paper will not say

Restated from [02](02-RESEARCH-QUESTIONS.md), because these are the sentences most likely to be added during
writing and hardest to remove later:

- that post-training has a general law about refusal — this is one corpus, one family, two scales;
- that refusal supervision is the *only* cause, unless the competing arms also fail to collapse;
- that the correction is a net win, unless H5 **and** H6 hold;
- that any result is an ability change, on the strength of a decision endpoint;
- "significant" without a margin, or "lower bound" without a derived bound.