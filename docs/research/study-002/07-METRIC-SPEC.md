# 07 — Metric specification

The metric layer is where Study 001's central defect lived, and it is still live. This document defines
every number Study 002 reports, its denominator, its population policy, its encoding for "not measured",
and the artifact it is attached to.

## The defect, restated at the level of code

`src/opengrad/promotion/tool_use_policy.py:33-37` records that the frozen behaviour set contains zero
`ANSWER` examples, so `no_call_accuracy` — the dimension whose truth class is `ANSWER`
(`tool_use_policy.py:52-57`) — was computed over an empty denominator and returned `0.0` for every model
including B0, and a `0.40` floor on it rejected every candidate unconditionally.

The fix applied in v3 was to stop asserting a floor where the set cannot measure the dimension
(`measurable_dimensions`, and `if name in unmeasurable: continue` at `tool_use_policy.py:171-172`). That
was correct as far as it went — it removed an unsatisfiable check — but it left the blind spot in place,
and it left a second instance of the same bug live:

```python
def macro_behaviour_score(metrics, measurable=None):
    names = [name for name in MACRO_DIMENSIONS if measurable is None or name in measurable]
    if not names:
        return 0.0          # ← an empty mean, encoded as the value "scored zero"
    return sum(...) / len(names)
```

`tool_use_policy.py:92-95`. An empty population and a genuinely-zero score are encoded identically. That
is the L1 defect in its general form, and it is the rule this specification removes:

> **No metric may encode "not measured" as a numeric value. A zero denominator produces
> `UNMEASURED`, never `0.0`.**

The repository already has the correct pattern in two places, and Study 002 propagates it rather than
inventing it:

- `src/opengrad/evaluation/capability.py:170-172` — `accuracy_given_answer` is `None`, not `0.0`, when
  nothing was attempted, with the reason in the comment: *"reporting 0.0 would read as 'it tried and got
  everything wrong'."*
- `src/opengrad/verification/accounting.py` — every gate declares a population policy
  (`REQUIRED_NONEMPTY`, `CONDITIONALLY_REQUIRED`, `OPTIONAL`), carries an execution census, and fails with
  `FAIL_NONVACUOUS` or `FAIL_ACCOUNTING` when counters disagree, on the principle that *"a gate that
  returns success without doing any work is indistinguishable from one that did the work and found
  nothing wrong."*

`docs/EVALUATION.md:19-25` states the same rule for the capability suite. The requirement of this study is
to make it hold in the tool-policy layer, where it does not.

## Three layers, never collapsed

Every metric belongs to exactly one layer, and a claim may only be made at the layer its metric measures:

| Layer | Question | Metrics | Failure looks like |
|---|---|---|---|
| **Ability** | can the model do the task at all | `accuracy`, `accuracy_given_answer`, sentinel benchmark accuracy | `accuracy` falls *and* `accuracy_given_answer` falls |
| **Decision** | does it choose to do the task | `answer_rate`, `refusal_rate`, `over_call_rate`, `refusal_correctness`, per-mode accuracy | `accuracy` falls while `accuracy_given_answer` holds |
| **Format** | does it emit what the harness can read | `parse_valid_rate`, IFEval strict, `format_violation_rate` | `parse_valid_rate` falls |

`docs/EVALUATION.md:19-25`: *"A checkpoint whose accuracy falls while conditional accuracy holds has not
lost the ability; it has stopped using it."* Study 001 could not make this distinction on its primary
endpoint, because the endpoint had no `ANSWER` population. Every Study 002 table prints the three layers
separately, and no summary sentence may attribute a decision-layer change to ability.

## Metric table

Population codes: `A` = `ANSWER`-gold items, `U` = `UNSUPPORTED`-gold items, `C` = `CALL`-gold items,
`L` = `CLARIFY`-gold items, `Q` = ordinary questions with a correct answer, `N` = all items in the
scored population.

| Metric | Layer | Definition | Denominator | Pop. | Policy |
|---|---|---|---|---|---|
| `answer_rate` | decision | attempted / n | n | A, Q | `REQUIRED_NONEMPTY` |
| `refusal_rate` | decision | refusals / n | n | A, Q | `REQUIRED_NONEMPTY` |
| `accuracy` | ability | correct / n | n | A, Q | `REQUIRED_NONEMPTY` |
| `accuracy_given_answer` | ability | correct / attempted | attempted | A, Q | `UNMEASURED` if attempted = 0 |
| `refusal_correctness` | decision | refusals / n on genuinely unanswerable items | n | `P-UNANS` | `REQUIRED_NONEMPTY` |
| `over_refusal_rate` | decision | refusals / n on answerable items | n | A, Q | `REQUIRED_NONEMPTY` |
| `no_call_accuracy` | decision | correct decisions / n where gold is `ANSWER` | n | A | `REQUIRED_NONEMPTY` |
| `must_call_accuracy` | decision | correct decisions / n where gold is `CALL` | n | C | `REQUIRED_NONEMPTY` |
| `clarification_accuracy` | decision | correct decisions / n where gold is `CLARIFY` | n | L | `REQUIRED_NONEMPTY` |
| `unsupported_accuracy` | decision | correct decisions / n where gold is `UNSUPPORTED` | n | U | `REQUIRED_NONEMPTY` |
| `call_precision` / `call_recall` / `call_f1` | decision | standard, over `CALL` | predicted/gold calls | C | `CONDITIONALLY_REQUIRED` |
| `over_call_rate` | decision | calls on non-`CALL` gold / n | non-`CALL` gold | N | `CONDITIONALLY_REQUIRED` |
| `macro_behaviour_score` | decision | mean of the four per-mode accuracies, **over measured modes only** | number of measured modes | N | `UNMEASURED` if none measured |
| `parse_valid_rate` | format | parseable outputs / n | n | N | `REQUIRED_NONEMPTY` |
| `format_violation_rate` | format | format violations / n | n | N | `OPTIONAL` |
| `truncation_rate` | format | `finish_reason == length` / n | n | N | `REQUIRED_NONEMPTY` |
| `resolvable_margin` | reporting | 2 × Wilson half-width at `p = 0.5` | — | per row | computed, never asserted |

## The two new metrics

### `refusal_correctness`

Study 001 had no metric for *refusing when refusing is right*, so it could measure over-refusal but not
under-refusal, and its anti-pattern guard could push in only one direction.

```
refusal_correctness = refusals / n        on P-UNANS only (genuinely unanswerable items)
over_refusal_rate   = refusals / n        on ANSWER-gold and ordinary-question items
```

They are one pair with opposite signs, and neither is reported without the other: a candidate that
improves direct answering by refusing less must be shown not to have moved refusals onto items that
genuinely cannot be answered. This is H6 expressed as arithmetic.

`P-UNANS` items must be *genuinely* unanswerable, and [06](06-SPLIT-SPEC.md) requires a labeller-agreement
record showing how many items two independent labellers agreed on, with disagreements removed and counted.
An item that is merely open-ended ("what is the best editor?") is not unanswerable and would make this
metric measure style, not correctness.

### `resolvable_margin`

Not a model metric — a property of a reported row. It exists so a reader does not have to trust a
significance claim: the row prints `n` and the margin the population could have resolved, and any
comparison tighter than that is printed `WITHIN_NOISE`. On the frozen 1,277-example population the
worst-case margin is 5.5pp; on the `ANSWER` strata set it is whatever [06](06-SPLIT-SPEC.md) derives.

## The refusal instrument

`refusal_rate` and everything derived from it are measured by `HEURISTIC_REGEX_v1`, which lives in
`src/opengrad/evaluation/capability.py:34-52`: seven patterns searched within the first
`REFUSAL_WINDOW_CHARS = 400` characters, compiled case-insensitively, returning a `RefusalVerdict` that
carries the matching pattern name and the matched text.

Two properties of that instrument must travel with every number it produces:

1. **It is deliberately precision-biased.** `capability.py:28-31`: *"Refusal patterns are deliberately
   high-precision rather than high-recall: a false positive would manufacture the very finding this study
   is testing for."* The consequence is directional and belongs in the table, not a footnote: false
   negatives are more likely than false positives, so **`refusal_rate` is an underestimate and
   `answer_rate` is an overestimate**. H1 — restoration of direct answering — is therefore the
   *optimistically biased* direction of the measurement, and H6 the conservative one. An H1 claim is
   checked against the detector's measured recall, and if recall is low enough that a material share of
   refusals could be missed, the effect is reported as an interval spanning the recall uncertainty rather
   than as a point.
2. **It never feeds an official benchmark score.** `capability.py:16-20`: IFEval, GSM8K and MMLU-Pro
   accuracy are computed exactly as their benchmarks specify, with a refusal counted as an ordinary
   failure. The refusal flag is a diagnostic column. Study 002 keeps that separation: `accuracy` is the
   benchmark's number, `refusal_rate` is the diagnostic, and the two are never summed or averaged.

`HEURISTIC_REGEX_v2` is a required deliverable ([04](04-ARM-MATRIX.md)) and a blocker in
[16](16-GPU-READINESS-GATE.md). It is a **measured** instrument, not merely an improved one:

- precision, recall and F1 are measured on `P-DET`, a hand-labelled sample of ≥ 300 responses drawn from
  ≥ 3 arms and stratified by arm, prompt mode and observed decision, so the sample is not dominated by one
  model's habits;
- the sample, its labels, the labeller-agreement rate and the confusion between pattern names are
  published;
- the measured figures are attached to every table using a refusal-derived metric, so the instrument's
  error is visible at the point of use rather than in a methods appendix;
- v2 keeps v1's pattern names as a superset and reports `method="HEURISTIC_REGEX_v2"`, so a number
  measured by v1 can never be silently compared with one measured by v2.

If v2's measured precision falls below the floor in [11](11-THRESHOLDS.md) the study stops at the
detector, per the stop rule in [03](03-PREREGISTRATION.md). A detector that over-flags would manufacture
the finding; one that under-flags would hide the regression.

## Requirements that are not metrics

**Truncation.** `finish_reason` is recorded and the truncation rate is reported **per stage, per arm**.
`docs/EVALUATION.md:27-44` is the reason: at a 768-token budget on MMLU-Pro, Base scored 38.0% against
M0's 36.8% and looked flat, while Base had hit the cap on 37.6% of items against M0's 7.2% — the
measurement reflected verbosity, not accuracy. At 2,048 tokens the gap is 12.0pp with Base still
truncating on 21.8%, so the honest figure is a **truncation-adversarial interval**
`[correct/n, (correct+unresolved)/n]`, there +6.4pp to +33.1pp. Study 002 adopts the rule as written:
uneven truncation invalidates a comparison; an observed difference is not called a "lower bound" unless a
bound was derived; and a budget change made after inspecting `finish_reason` counts but before any scoring
is a pre-scoring amendment applied uniformly to every stage and recorded.

**Scorer determinism.** The scorer is run on one fixed generations file at least twice and must return
byte-identical numbers, with stochastic dependencies seeded in the wrapper rather than in vendored source.
`docs/EVALUATION.md:54-60`: the vendored IFEval checkers scored 0.4510 / 0.4492 / 0.4492 on one unchanged
file, via `langdetect.detect()` and `random.*` in `build_description`. A scorer that is not deterministic
makes every margin in [11](11-THRESHOLDS.md) meaningless at the 1–2pp scale.

**Census reconciliation.** Every evaluation result carries the execution census from
`src/opengrad/verification/accounting.py`: `discovered == checked + blocked + skipped` and
`checked == passed + failed`, checked before a status is computed, with `FAIL_ACCOUNTING` on disagreement.
`AnswerAccounting.summary()` (`capability.py:156-157`) does the same for examples — every example lands in
exactly one of `CORRECT`, `INCORRECT`, `REFUSAL`, `PARSE_FAILURE`, and the sum is asserted, so a silently
dropped example cannot inflate every rate in the report.

## Attachment requirements

Every number in every Study 002 table carries, in the row or its machine-readable companion: metric name
and metric version, arm, seed, partition id, partition fingerprint, `n`, denominator, population policy,
evaluator version, generation config, device class and provider, and the sha256 of the artifact the number
was computed from. A number that cannot be attached to an artifact is not reported.

This is the direct answer to Study 001's largest defect category. Its 100 unsupported claims were
detachment errors — 19 transcription (`#9`, `#11`, `#83`, `#84` are each "the full-set number printed in a
confirmatory table"), 10 stale (`#67`, `#80`), and none invented. The correction is not sincerity; it is a
generated table that reads its numbers from an artifact whose hash it prints, so that a number cannot
drift from its source without the row changing too.

## What these metrics cannot support

- They cannot separate "the model cannot answer" from "the model will not answer" by themselves. That
  separation requires the `P-UNANS` / `ANSWER`-gold pair, and it is only as good as those partitions.
- They cannot support a cross-corpus claim finer than the resolvable margin of the smaller population.
- They cannot support any claim about tool selection, argument validity or schema validity:
  `UNMEASURED_DIMENSIONS` in `tool_use_policy.py:61` names those three explicitly, and Study 002 keeps them
  named-and-absent rather than quietly satisfied.
- They cannot licence the word "accuracy" unqualified. A bare "accuracy" in a Study 002 report means
  `correct / n` on a named population, printed with `n`; anything else is a defect of the same family as
  `#9` and `#84`.