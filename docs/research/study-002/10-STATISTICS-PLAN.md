# 10 — Statistics plan

Study 001's numbers were point estimates with no interval, compared across sets that were not the same
set. This document fixes the estimator, the unit of resampling, the multiplicity policy and the rule by
which a difference becomes a claim. It is deliberately conservative: **a claim requires a point estimate
larger than the population's resolvable margin, *and* an interval that excludes zero, *and* sign
consistency across seeds.** Any one of the three alone is insufficient, and the conjunction is stated
before any number exists (G1, G11).

## Units

| Role | Unit | Why |
|---|---|---|
| inference | the item (`example_id`), clustered; paired by seed | examples are the sampling unit; several prompts derive from one item |
| replication | the seed, `k = 3` | the only available estimate of training-run variability |
| comparison | the arm, at a step fixed by DEV selection | arms differ in kind, not degree |
| reporting | `(arm, partition, protocol, device_class)` | a row missing any of the four is not renderable |

Prompts rendered in more than one mode (`0-shot`, `8-shot`, `elicit`) from the same `example_id` are
**clustered**: the resampling unit is the item, not the rendered prompt, so a model that fails three
phrasings of one item cannot be counted as three independent failures. Treating them as independent is the
mechanism by which a small population manufactures a confident interval.

## Estimators

| Quantity | Estimator | Interval |
|---|---|---|
| a rate (`answer_rate`, `refusal_rate`, per-mode accuracy) | `k / n` | Wilson score interval, 95% |
| a paired difference between arms | per-seed `rate(A) − rate(B)`, then the mean over seeds | cluster bootstrap over items, 10,000 resamples, fixed resample seed |
| a difference against a frozen anchor | same, with the anchor re-scored under the same evaluator | as above |
| seed variability | the range of the per-seed deltas | min/max — **not** a standard deviation |
| resolvable margin | `2 × 1.96 × sqrt(0.25 / n)` | computed per row at worst-case `p = 0.5` |

Two deliberate restrictions:

- **No standard deviation over three seeds.** At `k = 3` a sample standard deviation is not an estimate of
  anything useful, and printing one invites a reader to compute a t-test on it. Seed-level uncertainty is
  reported as the per-seed values and their range, and the criterion that matters is whether the **sign
  agrees across all three seeds**.
- **No p-value as the headline.** A p-value on a 1,277-example population with a 2.74pp worst-case
  half-width invites a "significant" claim about a 3pp difference the population cannot resolve. The
  headline is the point estimate, its interval and the resolvable margin, in percentage points.

The bootstrap is **clustered and paired**: resample items with replacement, recompute both arms' rates on
the *same* resampled item set, and take the difference. This preserves the pairing the seed-matched design
created and avoids the classic error of bootstrapping two arms independently and then subtracting their
intervals as though the arms were unrelated.

## The primary test

For H1 the analysis is fixed in advance, on `P-CONF` and on the `ANSWER` strata set:

```
delta_answer_rate(arm)  = answer_rate(arm, ANSWER-gold) − answer_rate(C0, ANSWER-gold)
delta_refusal_rate(arm) = refusal_rate(arm, ANSWER-gold) − refusal_rate(C0, ANSWER-gold)
delta_no_call_acc(arm)  = no_call_accuracy(arm) − no_call_accuracy(C0)
```

computed per seed, then aggregated. H1 is **supported** for `arm = R1` when all four hold:

1. the mean `delta_answer_rate` is positive and at least the margin fixed in [11](11-THRESHOLDS.md);
2. the 95% cluster-bootstrap interval excludes 0;
3. all three seed-wise deltas have the same sign;
4. `delta_no_call_acc` moves in the same direction — so the change is in *decision correctness*, not in
   refusal-detector sensitivity.

If (1) and (2) hold but (3) fails, the verdict is `MECHANISM_INCONCLUSIVE` and the seed values are printed
side by side: with three seeds an inconsistent sign cannot be distinguished from noise, and pretending
otherwise is precisely the failure this plan exists to prevent.

If (4) fails while (1)–(3) hold, the interpretation changes — the model may have become more *willing*
without becoming more *correct*. That is a different finding, and it is reported as one rather than being
folded into H1.
## Secondary and competing explanations

| Comparison | Purpose | Claim status |
|---|---|---|
| `R2` vs `R1` | which half of the contract carries the behaviour (H4) | confirmatory, secondary |
| `R3` vs `C0` | whether removal would work, and at what cost to hallucination | descriptive, never promotable |
| `C1` vs `C0` | whether any narrow SFT collapses (competing) | confirmatory, secondary |
| `C2` vs `C0` | whether the effect is source-specific (competing) | confirmatory, secondary |
| `X1` vs `C0` | whether exposure rather than content drives it | descriptive |
| `D25`, `D50`, `R1` vs `C0` | dose–response (H3) | exploratory |
| `S1`, `S2` vs `C0`/`R1` | replication (H7) | exploratory |

The competing explanations are not optional extras. If `C1` or `C2` collapses as much as `C0`, H1's claim
narrows to *"refusal-`ANSWER` disagreement is one sufficient cause"*, and that narrowing is itself a result
the paper must report — with the alternative given its own arm and its own number
([02](02-RESEARCH-QUESTIONS.md)).

## Multiplicity

Two families, Holm–Bonferroni within each at family-wise α = 0.05:

- **Confirmatory family:** H1, H4, H5, H6 and the four competing-explanation comparisons. These carry the
  study's conclusions and are corrected together.
- **Exploratory family:** H3, H7 and the dose-arm deltas. Declared exploratory in
  [02](02-RESEARCH-QUESTIONS.md); reported as intervals without a significance claim, and a corrected
  result here is labelled exploratory-supported and may not appear in an abstract.

Every table names the family it belongs to, and prints the adjusted α if a p-value appears at all. Study
001's `#62` — *"Best of 4 checkpoints, chosen on the same set; the final checkpoint scores 0.5292"* — is the
failure mode this prevents in another form: a comparison chosen after seeing the data and reported as
though it had been planned.

## Look discipline

- `P-DEV` may be examined repeatedly — that is what it is for — and every use is logged with its purpose
  (selection, sanity check, failure inspection).
- `P-CONF` and `P-UNANS` are scored **once per arm**. There is no interim analysis, no early stopping and no
  alpha-spending rule, because there is exactly one look.
- Reading `P-CONF` before all arms are complete is a protocol violation recorded in the run log, not a
  shortcut.
- `P-SEALED` is not read at all.

## Thresholds and verdicts

Thresholds live in [11](11-THRESHOLDS.md) and are applied as a function of the metric artifacts, not as
prose:

```
verdict = f( {delta, ci, resolvable_margin, seeds} per (arm, endpoint), gate checks )
```

with the rule that any required endpoint reported `UNMEASURED` makes the verdict `NOT_EVALUABLE`, never
`PASS` or `FAIL`. `macro_behaviour_score` returning `0.0` on an empty measurable set
(`src/opengrad/promotion/tool_use_policy.py:92-95`) is exactly the input that must not be able to reach a
verdict.

## What this plan cannot do

- It cannot make the 1,277-example population resolve better than 5.5pp, or a 453-example mode better than
  9.2pp. Those are properties of `n`; the honest response is to print them per row rather than to buy
  precision with a model.
- It cannot separate seed variance from item variance at `k = 3`. It can require sign agreement and report
  the seed values, and nothing more.
- It cannot attribute a change to a cause on a single lineage. G13 stands, and the frozen verdict states it
  in the author's own words: *"Single lineage, single base model, no replicate; no cross-family
  generalization claimed."* (`results/final_campaign_verdict.json:589-593`)