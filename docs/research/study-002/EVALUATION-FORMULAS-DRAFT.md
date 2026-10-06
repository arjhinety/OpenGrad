# Study 002's evaluation formulas (draft)

**Status: draft, 2026-10-06. Not an amendment and not a preregistration.** It restates in mathematical notation
what the adopted documents and the code already decide, so the formulas can be checked before any audit and reused
in a paper. Where a document and the code are the authority, this draft names both and changes neither. Where the
documents leave a step undefined, the draft lists it as an open question and does not fill it in
([§8](#8-open-questions)). Every worked number below was computed with the project's own code, and
`tests/verification/test_evaluation_formulas_draft.py` recomputes each one and fails if this text and the code
disagree.

Citations are of two kinds ([§9](#9-references)):
- papers found and checked with the Papers With Code CLI (`pwc paper info`), of which only the abstract has been
  read, so each is cited for no more than its abstract says;
- classic statistics papers that predate arXiv and are not in that catalog, whose details were checked on Crossref
  (the DOI registry) instead.

## 0. Notation

| Symbol | Meaning |
|---|---|
| $n$ | number of items in a set |
| $y_i^{(1)}, y_i^{(2)}$ | the two labelling models' labels for item $i$ |
| $\mathcal{A}$, $A = \lvert\mathcal{A}\rvert$ | the agreed items, $\{i : y_i^{(1)} = y_i^{(2)}\}$, and their number |
| $\mathcal{D}$ | the decline labels, `DECLINE_JUSTIFIED` and `DECLINE_UNJUSTIFIED` |
| $s$ | a source dataset (Glaive, When2Call, ToolACE) |
| $N_s$ | the size of source $s$'s population, as opposed to $n_s$, its sample |
| $m$ | a decision mode: `CALL`, `DIRECT` (`ANSWER`), `CLARIFY`, `UNSUPPORTED` |
| $z$ | the 97.5% normal quantile, $z = 1.959963984540054$ (code constant) |
| $\mathbb{1}[\cdot]$ | 1 if the condition holds, else 0 |

## 1. Agreement between two labellers

**Raw agreement.** The share of items both models labelled alike:

$$a = \frac{A}{n}.$$

**Cohen's κ** (Cohen, 1960) corrects raw agreement for the agreement two labellers would reach by chance, given how
often each uses each label:

$$\kappa = \frac{p_o - p_e}{1 - p_e}, \qquad p_o = a, \qquad p_e = \sum_{\ell} \frac{n_\ell^{(1)}\, n_\ell^{(2)}}{n^2},$$

where $n_\ell^{(j)}$ is how many items labeller $j$ gave label $\ell$. κ is undefined when $p_e = 1$.
Code: `cohen_kappa`, `src/opengrad/verification/punans_report.py`.

*Worked example.* Two labellers, 100 items: both say D on 40, both say N on 40, and they split 10 one way and 10 the
other. Then $p_o = 0.8$, $p_e = 0.5 \cdot 0.5 + 0.5 \cdot 0.5 = 0.5$, and $\kappa = 0.6$. In the triage trial,
$a = 0.97$ and $\kappa = 0.843587$.

**Floors.** Raw agreement must be at least 0.80, pooled and per source (46 §5, 47 §3 B). κ is reported, never gated.

## 2. A proportion and its uncertainty

**Wilson score interval** (Wilson, 1927) for $k$ successes in $n$ trials, $\hat p = k/n$:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \;\pm\; \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat p (1 - \hat p)}{n} + \frac{z^2}{4n^2}}.$$

It stays inside $[0, 1]$ and behaves well for small $n$ and for $\hat p$ near 0 or 1, where the normal (Wald)
interval $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ is too narrow. Bowyer et al. (2025) argue the same for language-model
evaluations with fewer than a few hundred items. Code: `wilson`, `src/opengrad/verification/pdet_coverage_metrics.py`.

*Worked example.* The triage trial's pooled precision, 95 declines among 97 agreed items: $\hat p = 0.979381$, Wilson
95% interval $[0.927912, 0.994327]$.

**Resolvable margin** (06 §C2, 10). The smallest difference a row of $n$ items can resolve. It is twice the
worst-case half-width at $p = 0.5$, so it depends on $n$ alone and cannot be flattered by an observed rate near 0
or 1:

$$M(n) = 2 z \sqrt{\frac{0.25}{n}} = \frac{z}{\sqrt{n}}.$$

This is the normal half-width, slightly larger (more conservative) than Wilson's at $p = 0.5$. Code:
`resolvable_margin`, `src/opengrad/verification/resolvability.py`.

| $n$ | $M(n)$ | Where it matters |
|---|---|---|
| 200 | 0.138590 | the floor below which a mode is `UNDER_POWERED` |
| 385 | 0.099889 | the `P-UNANS` size that resolves `refusal_correctness` (`study_002_prereg_v8`) |
| 453 | 0.092087 | `P-CONF`'s `CALL` and `UNSUPPORTED` |
| 824 | 0.068279 | the non-`CALL` items behind `over_call_rate` |

Card et al. (2020) show that many NLP comparisons are underpowered because test sets are small; the margin is how
this study states that limit for every row.

**Difference of two independent proportions** (used only in the INC-0002 comparison): Newcombe's hybrid score
interval (Newcombe, 1998, method 10). With $\hat p_1 = k_1/n_1$, $\hat p_2 = k_2/n_2$ and their Wilson limits
$(l_1, u_1)$, $(l_2, u_2)$:

$$\big(\hat p_1 - \hat p_2\big) - \sqrt{(\hat p_1 - l_1)^2 + (u_2 - \hat p_2)^2}, \qquad \big(\hat p_1 - \hat p_2\big) + \sqrt{(u_1 - \hat p_1)^2 + (\hat p_2 - l_2)^2}.$$

*Worked example* (Newcombe's own): 56/70 against 48/80 gives a difference of 0.2 with interval
$[0.052431, 0.333873]$. Code: `newcombe`, `scripts/compare_gemini_relabel.py`.

## 3. The flag-set check (46 §5, 47)

Two models label each flagged reply. On a set of items:

$$P = \frac{D}{A}, \qquad D = \big\lvert\{\, i \in \mathcal{A} : y_i^{(1)} \in \mathcal{D} \,\}\big\rvert.$$

$P$ is **flag precision**: among items both models labelled alike, the share both called a decline. An item both
called `UNKNOWN` stays in $A$ and counts as not a decline. A conservative bound treats every disagreement as not a
decline: $P_{\text{low}} = D/n$. Code: `_precision`, `src/opengrad/verification/flag_triage.py`.

**The decision rule** (stop rules 1 and 2, 47 §3). Writing $F$ for the whole flag set and $s$ for a source:

$$\textsf{PROCEED} \iff P_F \ge 0.90 \;\wedge\; a_F \ge 0.80 \;\wedge\; \forall s \in \{\text{Glaive}, \text{When2Call}\}:\; A_s \ge 100 \;\wedge\; P_s \ge 0.90 \;\wedge\; a_s \ge 0.80.$$

ToolACE is held to the same three per-source conditions, but failing them excludes it (its replies keep their
text, disposed `SOURCE_EXCLUDED`) instead of stopping the study. The pooled $P_F$ and $a_F$ are always over all of
$F$, excluded sources included, so excluding a source cannot rescue a pooled failure. A source with $A_s < 100$ is
`NOT_EVALUABLE`, which counts as failing. Code: `triage_decision`.

*Worked example.* The 100-record trial (`TRIAL_NO_FLOOR`: it decides nothing) had $P = 95/97 = 0.979381$. When2Call
had $P = 18/20 = 0.9$, exactly at the bar, with $a = 20/21 = 0.952381$.

**The trial's allocation** (47 §3 E). The proportional (largest-remainder) shares of 100 by source size are raised to
each source's minimum (ToolACE: 20), taking seats one at a time from the largest source. With $N$ = 14,205 Glaive,
3,979 When2Call and 867 ToolACE flagged replies, this gives 59, 21 and 20. Code: `trial_allocation`.

**Recall: what the flags missed** (46 §5, 47 §3 F). A sample of 400 unflagged `ANSWER` replies, stratified 240 / 80 / 80
from populations of $N_s$ = 34,547 Glaive, 1,106 ToolACE and 2,526 When2Call replies (38,179 in all). Per source, the
decline share among agreed items is $\hat p_s = D_s / A_s$. The pooled estimate is weighted by population, never read
off the stratified sample:

$$\hat p = \frac{\sum_s N_s\, \hat p_s}{\sum_s N_s}, \qquad \widehat{M}_s = \operatorname{round}\big(\hat p_s N_s\big),$$

where $\widehat{M}_s$ estimates the declines in source $s$ that no flag caught (and that therefore stay in every
arm). A source × predicted-label cell expects $E_{s,\ell} = n_s N_{s,\ell} / N_s$ sampled replies; a cell with
$E_{s,\ell} < 10$ is descriptive only. Code: `recall_report`, `expected_by_label`.

*Illustration only (invented shares, not a result):* if $\hat p_s$ were 0.01, 0.10 and 0.05 for Glaive, ToolACE and
When2Call, then $\hat p = 0.015254$, and the implied missed declines would be 345, 111 and 126.

**Why model labels need care.** The labels here are model judgments. Agreement between two models is evidence of
consistency, not of correctness, since model errors correlate (Kim et al., 2025, in the case-study notes). Lee et
al. (2025) show that accuracy estimated from an imperfect model judge is biased, and give a correction and
intervals when a calibration set with known answers exists. This study has no human-labelled calibration set, so no
such correction is applied, and results are stated as "both models judged".

## 4. Qualifying the decision classifier (`pdet-coverage-metrics-v1`, 22 §6, 30 §11)

Over the usable items $U$ (gold label is a mode, item not excluded), for mode $m$:

$$R_m = \frac{TP_m}{G_m}, \qquad P_m = \frac{TP_m}{\mathrm{Pred}_m}, \qquad F1_m = \frac{2 P_m R_m}{P_m + R_m},$$

with $G_m$ the gold items of mode $m$, $\mathrm{Pred}_m$ the usable items predicted $m$, and $TP_m$ those that are both.
An `ABSTAIN` prediction is a miss for recall and never enters a precision denominator. For `CALL` only, predictions on
ambiguous (`UNKNOWN`) gold are added to the precision denominator: a call there would inject a wrong call target.

| Row | Rule | Threshold | Evaluable when |
|---|---|---|---|
| `DIRECT.recall`, `UNSUPPORTED.recall` | $R_m$ | ≥ 0.80, ≥ 0.75 | $G_m \ge 50$ |
| `DIRECT.precision`, `CALL.precision` | $P_m$ | ≥ 0.80, ≥ 0.95 | denominator ≥ 50 |
| `CLARIFY.f1` | $F1_{\text{CLARIFY}}$ | ≥ 0.70 | $G \ge 50$ |
| `m.challenge_recall` | $R_m$ on hard strata R, Q, M, X only | ≥ 0.60 | ≥ 30 gold there |
| `macro_f1` | $\frac{1}{\lvert\mathcal{M}\rvert}\sum_{m \in \mathcal{M}} F1_m$, $\mathcal{M} = \{m : G_m \ge 50\}$ | ≥ 0.75 | $\mathcal{M} \neq \varnothing$ |
| `abstention_rate` | $\lvert\{i \in U : \hat y_i = \texttt{ABSTAIN}\}\rvert / \lvert U\rvert$ | ≤ 0.15 | $U \neq \varnothing$ |
| `CALL.on_ambiguous_in_M` | `CALL` predictions on ambiguous gold in stratum M | = 0 | always |

**Post-stratified `DIRECT` precision**, per source, on the data the classifier will actually label: each stratum $h$
of the eligible pool gets its population weight $w_h = N_h / N$, and

$$\hat P^{\texttt{DIRECT}} = \frac{\sum_h w_h\, TP_h / n_h}{\sum_h w_h\, \mathrm{Pred}_h / n_h},$$

a ratio estimator over the sampled strata, with a 2,000-replicate stratified percentile bootstrap interval (Efron,
1979). It must reach 0.80 with at least 20 `DIRECT` predictions and every stratum sampled.

**Verdicts.** A row fails if it fails on any population, passes if it passes on at least one and fails on none, and
is otherwise `NOT_EVALUABLE`. A mode qualifies when `macro_f1` and `abstention_rate` pass and every row of the mode
passes. `DIRECT` also needs at least one source whose post-stratified precision passes. C1 is authorised by the rules
when `DIRECT` and `UNSUPPORTED` both qualify. Code: `evaluate`, `src/opengrad/verification/pdet_coverage_metrics.py`.

## 5. The trained model: the gate and the promotion policy

The gate (`study_002_gate_v1`, contract 3) wraps the promotion policy `tool_use_promotion_v6`, whose thresholds are
v5's, unchanged. From the confusion matrix $C_{g,p}$ (gold $g$, prediction $p$) over `CALL`, `ANSWER`, `CLARIFY`
and `UNSUPPORTED`, with $n_g = \sum_p C_{g,p}$:

$$P_{\text{call}} = \frac{C_{\text{CALL},\text{CALL}}}{\sum_g C_{g,\text{CALL}}}, \qquad R_{\text{call}} = \frac{C_{\text{CALL},\text{CALL}}}{n_{\text{CALL}}}, \qquad \text{acc}_g = \frac{C_{g,g}}{n_g}, \qquad \text{over\_call} = \frac{\sum_{g \neq \text{CALL}} C_{g,\text{CALL}}}{n - n_{\text{CALL}}}.$$

Here $\text{acc}_{\text{ANSWER}}$ is `no_call_accuracy`. **Macro recall** is the mean of call recall,
`no_call_accuracy`, `clarification_accuracy` and `unsupported_accuracy` over the classes present. Code:
`routing_metrics` (`src/opengrad/evaluation/routing.py`), `macro_behaviour_score`
(`src/opengrad/promotion/tool_use_policy.py`).

| Check | Requirement |
|---|---|
| call F1 retention | $F1_{\text{call}}^{\text{cand}} / F1_{\text{call}}^{\text{base}} \ge 0.90$ |
| macro recall | ≥ 0.40 |
| `no_call_accuracy`, `unsupported_accuracy`, `clarification_accuracy` | ≥ 0.40, ≥ 0.30, ≥ 0.50 |
| `over_call_rate` | ≤ 0.20 |
| `parse_valid_rate` | ≥ 0.99 |
| regression on call precision, call recall, clarification and unsupported accuracy | $x^{\text{cand}} - x^{\text{base}} \ge -0.10$ |
| `answer_rate`, `refusal_rate` on `ANSWER`-gold | ≥ 0.60, ≤ 0.25 |
| answer-rate drop against the baseline | $\text{answer\_rate}^{\text{base}} - \text{answer\_rate}^{\text{cand}} \le 0.30$ |
| `refusal_correctness` on `P-UNANS` | ≥ 0.70, and only on $n \ge 385$ |
| every required mode | $n \ge 200$, or `UNDER_POWERED` |

Comparisons are made at 9 decimal places, so a value exactly on a threshold is decided by the threshold, not by
floating-point error ($0.9 - 0.6 = 0.30000000000000004$ in binary).

**Truncation balance** (`study_002_prereg_v8` A). Two arms' truncation rates $r_1, r_2$ within a stage are imbalanced
when both of these hold:

$$\lvert r_1 - r_2 \rvert > 0.02 \quad\text{and}\quad \max(r_1, r_2) > 2.0 \cdot \min(r_1, r_2).$$

**Comparison rows** (gate check 14). Each row prints its $n$ and margin; a row whose observed difference is smaller
than its resolvable margin $M(n)$ is `WITHIN_NOISE` and supports nothing.

## 6. The hypotheses (10-STATISTICS-PLAN.md)

**Estimators** (10, "Estimators"). A rate is $k/n$ with a Wilson interval. A paired difference between arms is computed
per seed and averaged over the $k = 3$ seeds:

$$\bar\Delta = \frac{1}{3}\sum_{j=1}^{3} \big(\text{rate}_j(\text{arm}) - \text{rate}_j(C0)\big).$$

Its interval is a **cluster bootstrap over items**, 10,000 resamples with a fixed seed: resample items (not
rendered prompts) with replacement, recompute both arms on the same resampled set, and take the difference. Seed
variability is the range of the three per-seed deltas, never a standard deviation. Dodge et al. (2020) and Madaan et
al. (2024) document how much fine-tuning seeds and evaluation benchmarks vary; Bestgen (2022) argues for reporting
bootstrap intervals of differences rather than significance alone.

**H1's decision rule, exactly as 10 states it** ("The primary test"). With

$$\Delta_{\text{answer}} = \text{answer\_rate}(\text{arm}) - \text{answer\_rate}(C0), \quad \Delta_{\text{refusal}} = \text{refusal\_rate}(\text{arm}) - \text{refusal\_rate}(C0), \quad \Delta_{\text{no\_call}} = \text{no\_call\_acc}(\text{arm}) - \text{no\_call\_acc}(C0)$$

computed per seed on `ANSWER`-gold, H1 is **supported** for arm `R1` when all four hold:

1. the mean $\Delta_{\text{answer}}$ is positive and at least the margin fixed in [11](11-THRESHOLDS.md);
2. the **95% cluster-bootstrap interval excludes 0**;
3. all three seed-wise deltas have the same sign;
4. $\Delta_{\text{no\_call}}$ moves in the same direction.

If (1) and (2) hold but (3) fails, the verdict is `MECHANISM_INCONCLUSIVE`. If (4) fails while (1) to (3) hold, the
finding is reported as a different one (more willing, not more correct). Any required endpoint `UNMEASURED` makes
the verdict `NOT_EVALUABLE`.

**The multiplicity rule, exactly as 10 states it** ("Multiplicity", and the note under `study_002_prereg_v14`). The
confirmatory family is H1 (`R1` vs `C0`), H5 (tool-policy non-regression), H6 (refusal correctness on `P-UNANS`) and
`C2` vs `C0`, with Holm–Bonferroni (Holm, 1979) at family-wise $\alpha = 0.05$. Holm orders the four p-values
$p_{(1)} \le \dots \le p_{(4)}$ and rejects $H_{(j)}$ while

$$p_{(j)} \le \frac{\alpha}{4 - j + 1},$$

stopping at the first that fails. The thresholds are 0.0125, 0.0167, 0.025 and 0.05.

**These two rules do not yet meet.** H1's rule uses an unadjusted 95% interval and no p-value, and 10 says a p-value
is never the headline. The family rule needs a p-value for each member, or an interval at an adjusted level. 10
does not say how a bootstrap result becomes either. This is open question 1 ([§8](#8-open-questions)); this draft
does not answer it.

## 7. The INC-0002 comparison

Per exposure group $G \in \{A, B, D\}$, the change rate is $c_G = \#\{\text{Gemini label changed}\} / n_G$ and the
agreement lost is the share, among items where gpt-5.6-sol and deepseek-v4.1-flash agree, where old Gemini matched
them and new Gemini does not. The contrasts $c_A - c_D$ and $c_B - c_D$ (and the same for agreement lost) use
Newcombe's interval ([§2](#2-a-proportion-and-its-uncertainty)). Plan:
[`reports/incidents/INC-0002-relabel-comparison-plan.md`](../../../reports/incidents/INC-0002-relabel-comparison-plan.md).
For scale: with 0 of 46 changed, the Wilson upper bound on the change rate is 0.077074.

## 8. Open questions

Found while writing this draft. None is answered here; each needs the owner's decision, and any change to a
preregistered rule needs a numbered amendment first.

1. **From bootstrap to the family rule.** How does H1's cluster bootstrap yield a p-value, or an interval at an
   adjusted level, for Holm–Bonferroni? Two readings exist, and the documents choose neither:
   - a bootstrap p-value (for example by inverting the interval: the smallest $\alpha$ at which it excludes 0);
   - an interval at the adjusted level, for example $1 - 0.05/4 = 98.75\%$ for the strictest of four. That is the
     Bonferroni level; under Holm only the first step uses it.

   `pwc search` found no paper in the catalog on either route, so neither is cited. H1's rule (95%) and the family
   rule may also disagree on the same data: a 95% interval can exclude 0 while the 98.75% one does not.
2. **Which bootstrap interval.** 10 fixes the resampling (items, paired, 10,000 resamples) but not the interval method
   (percentile, basic or BCa).
3. **The margin for a paired difference.** 10 and `resolvability.resolvable_margin` size the margin for **one rate**,
   from the worst-case variance of a 0/1 outcome, $p(1-p) \le 0.25$:

   $$M_{\text{rate}}(n) = 2 z \sqrt{\frac{0.25}{n}}.$$

   Gate check 14 (`v12_resolvable_margin`, `src/opengrad/verification/population_validators.py`) and H1 apply it to a
   **difference between two arms on the same items**. Per item that difference is $d_i = x_i - y_i \in \{-1, 0, 1\}$,
   with variance

   $$\operatorname{Var}(d) = q - (p_1 - p_2)^2 \le q \le 1,$$

   where $q$ is the share of items on which the two arms disagree. The worst case, $\operatorname{Var}(d) = 1$, is two
   arms that disagree on every item, half the time each way. The same construction then gives

   $$M_{\text{paired}}(n) = 2 z \sqrt{\frac{1}{n}} = 2\, M_{\text{rate}}(n).$$

   At $n = 385$, computed with the project's code: $M_{\text{rate}} = 0.099889$ and $M_{\text{paired}} = 0.199778$. For
   comparison, two *independent* samples of 385 give $2z\sqrt{0.5/n} = 0.141264$. The single-rate margin is not
   always too small: bounded by the disagreement share instead, $2z\sqrt{q/n}$ is 0.089343 at $q = 0.2$, below
   $M_{\text{rate}}$, and passes it only when the arms disagree on more than a quarter of the items. So which margin
   is right depends on how often arms disagree, which no document fixes in advance. (The literature search found
   nothing on this; the bound above is elementary algebra, checked by a brute-force search over all joint
   distributions of the two arms' 0/1 outcomes, but it is not cited to a source.)
4. **The answer and refusal metrics.** `answer_rate`, `refusal_rate` and `refusal_correctness` will be computed by the
   decline classifier 46 calls for, which is not built yet. Their exact definitions (what counts as answered or
   refused, how an `UNKNOWN` reply counts) are fixed only once it is.
5. **A pooled value exactly on the bar.** In the trial, When2Call sat exactly at 0.90. The rule is $\ge$, so it passes.
   That is the rule as adopted; the open point is only that precision of 0.90 on as few as 100 agreed items has a
   Wilson lower bound of 0.826 (90 of 100).
6. **Seed variance and the sign check.** H1's cluster bootstrap (10, "Estimators") resamples **items only**: it keeps
   each arm's three trained models fixed and redraws the items they are scored on. Its 95% interval therefore
   measures which items happened to be drawn, not which training seeds; training-seed variance is outside it. 10
   says as much ("What this plan cannot do": it "cannot separate seed variance from item variance at `k = 3`") and
   leaves seeds to condition 3. On its own, condition 3 is weak evidence. Suppose the arm has no effect and the three
   seed-wise deltas $\Delta_1, \Delta_2, \Delta_3$ are independent and symmetric about 0, none exactly 0. Each is
   then positive with probability $1/2$, and

   $$\Pr(\text{condition 3}) = \Pr(\text{all positive}) + \Pr(\text{all negative}) = 2 \times (1/2)^3 = 0.25.$$

   Condition 1 also requires a positive mean, which leaves only the all-positive case: under the same assumptions
   the two sign requirements together pass with probability $(1/2)^3 = 0.125$, and that bounds the chance of a
   false `SUPPORTED` from above. Conditions 1 (the margin), 2 and 4 lower it only to the extent that item-level checks
   catch what seed noise produced. A seed that happens to run high moves the mean while the item bootstrap, blind
   to seeds, stays narrow (this last point is reasoning, not computed). The assumptions matter in both directions:
   seeds that share something (the same `C0` run, the same data order) make the deltas positively correlated and
   raise 0.25; a delta of exactly 0, possible because each rate is a count over $n$ items, lowers it, and 10 does
   not say what sign 0 has. Open: should H1's interval also carry seed variance, or should every H1 result state
   that it does not? With $k = 3$, resampling seeds offers only 10 distinct multisets.

   Literature, abstracts checked with `pwc paper info` (2026-10-06): Colas, Sigaud and Oudeyer (2018) relate the
   number of random seeds to the probabilities of statistical errors, for the t-test and the bootstrap
   confidence-interval test, in deep reinforcement learning rather than fine-tuning; Dodge et al. (2020) find that
   distinct seeds give substantially different fine-tuning results; Madaan et al. (2024) measure seed variance in
   evaluation benchmarks. None analyses a three-seed sign check, so 0.25 is cited to no one: it is elementary
   probability, recomputed in the test. Bouthillier et al. (2021, arXiv 2103.03098), on accounting for variance in
   ML benchmarks, has no record in the catalog and is not cited.

## 9. References

**Classics, checked on Crossref (2026-10-06), Efron also on Project Euclid.**
- Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American
  Statistical Association* 22, 209–212. doi:10.1080/01621459.1927.10502953.
- Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement* 20,
  37–46. doi:10.1177/001316446002000104.
- Newcombe, R. G. (1998). Interval estimation for the difference between independent proportions: comparison of
  eleven methods. *Statistics in Medicine* 17, 873–890. doi:10.1002/(SICI)1097-0258(19980430)17:8<873::AID-SIM779>3.0.CO;2-I.
- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics* 7(1), 1–26.
  doi:10.1214/aos/1176344552. (Pages from Project Euclid, 2026-10-06; Crossref gave none.)
- **Not verified:** Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of
  Statistics* 6. Crossref returned a different paper for this query, and the journal is not on Project Euclid. Check
  it at the publisher before citing.

**From the Papers With Code catalog (abstracts only; full records in
[`docs/references/papers.yaml`](../../references/papers.yaml)).**
- Statistics of evaluation: Miller (2024, arXiv 2411.00640); Bowyer et al. (2025, arXiv 2503.01747); Madaan et al.
  (2024, arXiv 2406.10229); Dodge et al. (2020, arXiv 2002.06305); Card et al. (2020, arXiv 2010.06595); Bestgen
  (2022, arXiv 2205.11134); Lee et al. (2025, arXiv 2511.21140); Colas, Sigaud and Oudeyer (2018, arXiv 1806.08295).
- Where Study 002 is going: see the case-study notes,
  [`CASE-STUDY-RELATED-WORK.md`](CASE-STUDY-RELATED-WORK.md), "Statistics and the study's direction".
