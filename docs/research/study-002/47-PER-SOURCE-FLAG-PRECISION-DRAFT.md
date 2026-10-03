# 47 — Per-source flag precision and agreement for the triage (`study_002_prereg_v15`)

> **Status: ADOPTED 2026-10-03, as drafted.** The owner chose the proposal on every §6 item, including the
> withholding of the figure without `ABSTAIN` that the draft added. It is recorded in [03](03-PREREGISTRATION.md)
> and `reports/ERRATA.md` §32 in the same commit, before any triage label exists. The rules are code in
> `src/opengrad/verification/flag_triage.py`. The text below is kept as drafted, as 46's was.
>
> **As drafted:** *Status: DRAFT, not adopted. It is recorded in neither 03 nor `reports/ERRATA.md` unless the
> owner adopts it, and it must be decided before any triage label exists.*
>
> - **What the owner asked on 2026-10-03,** before the triage task is built:
>   - report flag precision separately for Glaive, When2Call and ToolACE, not only pooled;
>   - stratify the 100-record trial so ToolACE is represented in proportion or more;
>   - draft a per-source precision floor, or the exclusion of a source that fails it;
>   - also draft a per-source agreement floor, and report per source how many records drop out as
>     disagreements or `UNKNOWN`;
>   - make the consequence of failing depend on the source: Glaive or When2Call failing stops the corpus
>     intervention, and only ToolACE can be excluded;
>   - keep the adopted pooled floors over all of `F`, ToolACE included, even when ToolACE is excluded;
>   - stratify 46 §5's 400-reply recall sample by source, with minimums for ToolACE and When2Call, and
>     estimate the pooled share of declines with population weights;
>   - give the recall population per source by predicted label, report recall per source and predicted label,
>     and give the implied missed declines a second time without `ABSTAIN`-predicted replies;
>   - give the expected sampled replies per source and predicted label under each allocation, and mark cells
>     expected below 10 as descriptive only.
> - **What this document adds:** those rules, as proposals. §6 lists the choices.
> - **Numbers:** every count, floor and sample size below is pinned by `tests/verification/test_draft_47.py` to
>   the artifact or code constant it comes from. Figures quoted from papers are pinned as quoted.

## 1. Why a pooled figure is not enough

46 §5 (adopted, `study_002_prereg_v14`) puts two floors on the triage of the flag set `F`:
- stop rule 2, raw agreement ≥ 0.80 between the two labelling models;
- stop rule 1, flag precision ≥ 0.90, meaning the share labelled a decline among items both models label alike.

Both are pooled over `F`, and `F` is dominated by one source. The table gives two kinds of count:
- **flags**, records classifier v2 flags
  ([`flag-set.manifest.json`](../../../reports/study-002/flag-set/flag-set.manifest.json));
- **prior flags**, single exchanges the v1 regex calls a refusal, 03's prior
  ([`sft_refusal_supervision_audit_canonical_v2.json`](../../../results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json)).

| Source | Flags | Prior flags |
|---|---|---|
| Glaive | 14,205 | 14,066 |
| When2Call | 3,979 | 4,038 |
| ToolACE | 867 | 10 |
| All | 19,051 | 18,114 |

Neither is a count of corrections. A flag is a prediction. How many records `R1` corrects is unknown until the
triage labels them and the base model answers them (46 §6).

Three things make the smaller sources the risky ones:
- **ToolACE is where the two detectors disagree most.** Classifier v2 flags 867 of its records; the v1 regex,
  built for high precision, flags 10. At least one of them misreads ToolACE, and the triage is where we find out which.
- **When2Call was never in classifier v2's test set.** Its one test (P-DET-COVERAGE-v2) held Glaive and ToolACE
  replies only, as did every set drawn for v2 (`dev-v2` and `v2-devcheck-1` to `-5`). The labelled When2Call
  items are all in sets v2 was developed on: P-DET-v1, and classifier v1's development and check sets, which
  37 kept as v2 development data.
- **The pooled floors can be met while a source fails both.** Glaive is 74.6% of `F`. Suppose ToolACE's
  precision were 0.5, Glaive's and When2Call's 0.95, and agreed items fell in proportion to the flags. The pool
  would be 0.930 and pass. And precision is computed on agreed items only: a source whose hard cases split the
  models would show a clean precision on its easy remainder, while Glaive's volume carries pooled agreement
  over 0.80.

Machine learning studies this as performance on subpopulations. Yang et al. (2023, *Change is Hard*, arXiv
2302.12254) start from the observation that models often do poorly on subgroups underrepresented in training.
They find a trade-off between worst-group accuracy and other metrics, so the metric reported has to be chosen
with care. De Jong et al. (2025, arXiv 2501.18055) find that classifiers built on pathology foundation models
make errors attributable to the medical center: they confuse an image with images of other classes from the
same center. Source plays the same role here.

## 2. What changes without adoption

Reporting is measurement, not a rule, so it is in the code now (`src/opengrad/verification/flag_triage.py`,
`flag_precision`). For each source and pooled, every triage report holds:
- items, agreed items, raw agreement and κ;
- what keeps a record from any correction: disagreements, and items both models called `UNKNOWN`, each counted
  and summed (46 §5: "unchanged, counted");
- items either model called `UNKNOWN`, which overlaps both counts above;
- precision on agreed items, the measure the adopted rule uses, with its 95% Wilson interval;
- a **lower bound** that counts every disagreement as not a decline (declines over all items).

The recall sample's report (`recall_report`) holds, per source:
- the share of declines among agreed items, with its Wilson interval;
- the share if every disagreement were a decline;
- agreed items and declines per classifier v2 predicted label;
- the declines the share implies in the source's population, and the same with `ABSTAIN`-predicted replies
  left out.

Its pooled share is weighted by population (§3 F).

The adopted rule is unchanged. Until the owner adopts something else, stop rules 1 and 2 are decided by:
- the pooled precision floor of 0.90;
- the pooled agreement floor of 0.80.

The per-source figures carry no status unless a floor below is adopted.

## 3. Proposals

**A. A per-source precision floor.** Each source's precision on agreed items must be ≥ 0.90, the same value as
the pooled floor.

**B. A per-source agreement floor.** Each source's raw agreement must be ≥ 0.80, the same value as stop rule 2.

**C. What a failing source means depends on the source** *(proposed)*:

| Source fails A or B | Consequence |
|---|---|
| Glaive | the corpus intervention stops (stop rule 1 for A, stop rule 2 for B) |
| When2Call | the corpus intervention stops (stop rule 1 for A, stop rule 2 for B) |
| ToolACE | the source is excluded |

An excluded source's flagged records stay unchanged in every arm, with the disposition `SOURCE_EXCLUDED`. They
are counted and reported. If the pooled floors still pass, the study goes ahead on Glaive and When2Call.

**The pooled floors stay over all of `F`.** Stop rules 1 and 2 keep their adopted object: every flagged record,
ToolACE included, even when ToolACE is excluded. Exclusion changes dispositions only, and never rescues a
pooled failure. Take Glaive and When2Call at 0.92 and ToolACE at 0.40, with agreed items in proportion to the
flags. The pool is 0.896, below 0.90, so stop rule 1 stops the study, although C alone would only exclude
ToolACE. ToolACE's exclusion therefore matters only when the pool still passes.

The reason is that the triage labels must not choose which records a floor is judged on. Recomputing the pool
after dropping the source that failed would do exactly that. Dwork et al. (2014, arXiv 1411.2664) point out
that the theory of statistical inference assumes the hypotheses and analyses are fixed before the data are
gathered, while in practice new analyses are chosen from the outcomes of earlier ones.

**Why only ToolACE can be excluded.** `R1` vs `C0` tests H1: a corpus whose flagged labels and targets are
corrected does not collapse (02, H1). H1 is falsified if `R1` still collapses. An excluded source keeps its
declines in `R1`. A collapse in `R1` could then come from those declines, so it would falsify H1 only for the
sources corrected. The more flags a source holds, the less `R1` says about H1 as written:
- **Glaive** holds 14,205 of the 19,051 flags. Without them, `R1` could correct at most 4,846 flagged records
  and would leave 74.6% of the flags in place.
- **When2Call** holds 3,979 of the 19,051 flags, and 4,038 of the 18,114 prior flags. 62.1% of its 6,505
  records are prior flags. It is also the source Study 001's collapse is associated with (02, quoting
  `ROADMAP.md`). `C2` (Canonical-v2 without When2Call) exists to test that source as the competing explanation.
  If `R1` kept When2Call's declines, a collapse in `R1` could come from those declines (H1) or from the source,
  and `R1` and `C2` would stop separating them.
- **ToolACE** holds 867 of the 19,051 flags, and 10 of the 18,114 prior flags. Without them, `R1` still covers
  the sources holding 18,184 of the flags and 18,104 of the prior flags. The declines that remain make `R1`
  more like `C0`. That biases `R1` towards a null result, which here means towards wrongly falsifying H1. 46 §5
  already states that direction for missed declines.

These are flag counts. The share of corrections each source would hold is unknown until the triage.

The exclusion is decided from triage labels, before any arm is trained or scored, so no outcome steers which
sources stay. Choosing a subgroup after seeing outcomes is a harder problem with its own methods. Cheng et al.
(2025, *Chiseling*, arXiv 2509.19490) note that existing ways of finding a subgroup on which a treatment effect
exceeds a threshold either lack inferential guarantees, heavily restrict the search, or lose efficiency to
naive data splitting.

*Alternatives:*
- **Always stop:** every failing source stops the intervention, ToolACE included. ToolACE, which held 10 of
  the 18,114 prior flags, could then halt the study.
- **Always exclude:** every failing source is excluded, Glaive and When2Call included. If no source passes, the
  intervention stops. `R1` could then test H1 on one source, even ToolACE alone.

(These alternatives have no letters, so they are not confused with the arms `C1` and `C2`.)

**D. Small sources.** A source with fewer than 100 agreed items is `NOT_EVALUABLE` for A and B, and is treated
as failing, never passed for lack of data. Under C, a `NOT_EVALUABLE` Glaive or When2Call stops the intervention
and a `NOT_EVALUABLE` ToolACE is excluded.

**E. The trial.** The 100-record trial is drawn per source, in ascending order of
`sha256("opengrad-flag-triage-trial-v14" | record id)`. The seed string is new here; it names v14, the procedure
the trial serves.
- **proposed:** proportional shares, except that ToolACE gets at least 20, taken from Glaive. That gives Glaive
  59, When2Call 21, ToolACE 20;
- **alternative:** strictly proportional, Glaive 75, When2Call 21, ToolACE 4.

With only 4 items the trial would say almost nothing about ToolACE. And 46 §5 allows its one revision of the
procedure after the trial, so the trial is the only look before that revision.

Trial records are part of `F`. They are labelled again in the full triage, under the procedure as it stands
after the trial, and only those labels count toward the floors.

**F. The recall sample.** 46 §5 draws 400 `ANSWER`-labelled first replies outside `F`, labels them the same
way, and reports the share of declines among them, with no floor. A decline found there is one the flag set
missed, and it stays in every arm. The population is every eligible `ANSWER` first reply classifier v2 did not
flag, from the manifest's `precision_reporting` (eligible replies by source and label, minus `UNSUPPORTED`).
Eligible means the first reply is non-empty prose; the one empty reply is left out.

| Source | Replies outside `F` | Random draw, expected | Proposed |
|---|---|---|---|
| Glaive | 34,547 | 361.95 | 240 |
| When2Call | 2,526 | 26.46 | 80 |
| ToolACE | 1,106 | 11.59 | 80 |
| All | 38,179 | 400 | 400 |

Outside Glaive, most of these replies are not predicted `DIRECT`. By classifier v2's predicted label:

| Source | `DIRECT` | `CLARIFY` | `ABSTAIN` | `CALL` | All |
|---|---|---|---|---|---|
| Glaive | 33,829 | 636 | 82 | 0 | 34,547 |
| When2Call | 80 | 2,446 | 0 | 0 | 2,526 |
| ToolACE | 73 | 655 | 366 | 12 | 1,106 |
| All | 33,982 | 3,737 | 448 | 12 | 38,179 |

A share of declines over a whole source can hide a part of it that behaves differently. Oakden-Rayner et al.
(2019, arXiv 1909.12475) call this hidden stratification: a model's overall performance can be high while it
consistently fails on an important subset. Here, a source's share of missed declines could come mostly from
its `CLARIFY` or `ABSTAIN` replies. Although 46 §5 leaves abstentions unflagged on purpose, a `DECLINE_*`
label on an `ABSTAIN`-predicted reply counts here as a missed decline. Two proposals follow, both reported,
not gated:
- **Per predicted label, always.** `recall_report` gives each source's agreed items and declines per predicted
  label.
- **Missed declines a second time, without `ABSTAIN`.** Beside a source's implied missed declines (its share of
  declines times its replies outside `F`), the report gives the same figure with `ABSTAIN`-predicted replies
  left out of both: the share from the other agreed items, times the replies outside `F` that are not
  `ABSTAIN`. For ToolACE that population is 740 of 1,106. The two figures are separate estimates; their
  difference is not the number of declines among `ABSTAIN` replies.

Both are already in the code as reporting, ahead of the owner's decision, as §2's other reports are. If the
owner refuses either, it comes out of the code.

**Expected replies per cell.** Each source's draw is uniform within the source, so a source-and-label cell
expects the source's sample size times the cell's share of the source's population. The expected sampled
replies under each allocation (`expected_by_label`; † marks a cell expected below 10; — marks a cell with no
population):

| Allocation | Source | `DIRECT` | `CLARIFY` | `ABSTAIN` | `CALL` |
|---|---|---|---|---|---|
| 240 / 80 / 80 | Glaive | 235.0 | 4.4 † | 0.6 † | — |
| 240 / 80 / 80 | When2Call | 2.5 † | 77.5 | — | — |
| 240 / 80 / 80 | ToolACE | 5.3 † | 47.4 | 26.5 | 0.9 † |
| 354 / 26 / 20 | Glaive | 346.6 | 6.5 † | 0.8 † | — |
| 354 / 26 / 20 | When2Call | 0.8 † | 25.2 | — | — |
| 354 / 26 / 20 | ToolACE | 1.3 † | 11.8 | 6.6 † | 0.2 † |
| Random | Glaive | 354.4 | 6.7 † | 0.9 † | — |
| Random | When2Call | 0.8 † | 25.6 | — | — |
| Random | ToolACE | 0.8 † | 6.9 † | 3.8 † | 0.1 † |

The random draw's cells are expectations of a draw of 400 over all 38,179 replies, so its per-source sizes are
fractional too. The owner asked for these to come from `trial_allocation`, which would give 362, 26 and 12, a
proportional stratified draw rather than a random one. The two differ by less than 0.5 in any cell (When2Call
`CLARIFY` 25.6 against 25.2), and no cell crosses 10 either way.

**A cell expected below 10 is descriptive only** *(proposed)*. Its agreed items and declines are reported,
and no share, interval or estimate is drawn from the cell on its own. Its items still count in the source's
share and in the pooled share; leaving them out would bias both. A cell reaching 10 gets no share either: the
report gives per-cell counts only, so the rule decides which cells may be read as more than counts. The
classification uses expected counts, fixed before the draw, and is not revisited on the counts that arrive.

*This draft adds* one more use of the same cut-off: the figure without `ABSTAIN` is drawn only where the
source's expected sample outside `ABSTAIN` reaches 10. Under a random draw ToolACE's would be 7.8, so its
figure without `ABSTAIN` would be descriptive only. The cut-off of 10 is a convention chosen here; no paper we
found fixes it. Čiginas (2022, arXiv 2202.13085) notes that direct estimates are not
efficient for survey domains with small samples.

A random draw of 400 would give ToolACE about 12 replies and When2Call about 26. Suppose about one in ten were
declines (1 of 12, or 8 of 80). With 1 decline in 12, the 95% Wilson interval runs from 1.5% to 35.4%; with 8
in 80, from 5.2% to 18.5%. Card et al. (2020, arXiv 2010.06595) find underpowered experiments common in NLP. On
several GLUE tasks, small test sets leave most comparisons underpowered, and noise is then hard to tell from a
real difference.
- **proposed:** drawn per source in ascending order of `sha256("opengrad-flag-triage-recall-v14" | record id)`.
  ToolACE and When2Call get at least 80 each, one fifth of the sample, the share the trial gives ToolACE (20 of
  100). The rest goes to Glaive, which keeps 240;
- **alternative:** at least 20 each, the trial's count: Glaive 354, When2Call 26, ToolACE 20;
- **alternative:** a plain random draw: Glaive about 362, When2Call about 26, ToolACE about 12.

**The pooled share is weighted by population, under every choice.** A sample with minimums holds far more
ToolACE and When2Call than the corpus does, so its raw share of declines is not an estimate of the corpus's.
The pooled estimate is each source's share of declines times its replies outside `F`, summed, divided by the
38,179. The unweighted share of the sample is reported beside it, labelled as not an estimate.

Weighting removes the tilt; it is the minimums that move the pooled estimate's precision. If every source
misses declines at a similar rate, the minimums widen the pooled interval compared with a proportional draw.
If ToolACE and When2Call miss many more than Glaive, they can narrow it. Chen et al. (2025, arXiv 2501.18121)
choose each group's sample size to minimise the variance of a pooled mean, in a differential-privacy setting:
the best sizes depend on the groups, which is why no single draw is best in every case.

## 4. A caution about `UNKNOWN`

A labelling model can lean on one label. Gao et al. (2026, arXiv 2609.38827) report that on ANLI one
direct-decision model put 38.8% of its predictions and 51.3% of its errors on "Neutral". That held despite an
accuracy of 74.95% and nearly balanced gold labels. The paper studies ordinal scales and finds the effect much
weaker on nominal labels, which ours are. So it is an analogy, not evidence about our labellers.

A labeller that sends a source's hard cases to `UNKNOWN` changes that source's precision on agreed items:
- an `UNKNOWN` the other model disagrees with drops out. Precision rises if the item would have been an agreed
  non-decline, and falls if it would have been an agreed decline;
- an `UNKNOWN` both models give counts as not a decline, and lowers precision.

So the reports hold, per source, the counts of `UNKNOWN` and the lower bound. They are reported, not gated.

## 5. Not changed

- every threshold value, including 0.90 and 0.80, which this draft reuses rather than moves;
- the pooled floors' object: all of `F`, whatever C excludes;
- 46's flag-set definition, labels and procedure, and the other v14 decisions;
- every population, the arms and the arms' seeds.

**Candidates already scored:** none. **Triage labels:** none exist; the triage task is not built.

## 6. For the owner

1. **Per-source precision floor (A).** *Proposed:* yes, 0.90.
2. **Per-source agreement floor (B).** *Proposed:* yes, 0.80.
3. **If When2Call fails.** *Proposed:* the intervention stops.
   - *Stop:* `R1` keeps testing H1 as written. But When2Call, on which classifier v2 was never tested, can
     halt the study.
   - *Exclude:* the study goes ahead, but `R1` would keep When2Call's declines, so it no longer separates H1
     from the source explanation `C2` tests.
4. **If Glaive fails.** *Proposed:* the intervention stops.
   - *Stop:* the same reasoning as for When2Call.
   - *Exclude:* `R1` would leave most of the flags in place, more than excluding When2Call would.
5. **If ToolACE fails.** *Proposed:* it is excluded. The pooled floors still count it, so a ToolACE bad enough
   to pull the pool below a floor stops the study.
   - *Exclude:* `R1` loses the smallest share of flags. The bias is towards a null result, that is, towards
     wrongly falsifying H1.
   - *Stop:* the study would halt over 867 of 19,051 flags, a source that held 10 of the prior's 18,114.
6. **Small sources (D).** *Proposed:* at least 100 agreed items, else treated as failing under items 3 to 5.
7. **The trial (E).** *Proposed:* ToolACE at least 20. *Alternative:* proportional (4). Either way, trial
   records are labelled again in the full triage, and only those labels count.
8. **The recall sample (F).** *Proposed:* ToolACE and When2Call at least 80 each, Glaive 240.
   *Alternatives:* at least 20 each (Glaive 354, When2Call 26, ToolACE 20), or a plain random draw. Under every
   choice the pooled share is each source's share times its replies outside `F`, summed, over the 38,179; the
   sample's own share is reported only for comparison.
   - *Stratify:* each source gets a usable estimate of what the flag set missed. The pooled interval widens
     if the sources miss declines at similar rates, and can narrow if ToolACE and When2Call miss many more.
   - *Random draw:* ToolACE's and When2Call's shares rest on about 12 and 26 replies.

   Separately, per predicted label (`DIRECT`, `CLARIFY`, `ABSTAIN`, `CALL`), the population outside `F` is:
   Glaive 33,829, 636, 82 and 0; When2Call 80, 2,446, 0 and 0; ToolACE 73, 655, 366 and 12.
   *Proposed:* keep both reports, under every choice: each source's agreed items and declines per predicted
   label, and each source's implied missed declines a second time with `ABSTAIN`-predicted replies left out.
   *Alternative:* drop either; it then comes out of the code.

   Expected sampled replies per cell (the table in §3 F gives every cell). The cells that reach 10:
   - *240 / 80 / 80:* Glaive `DIRECT` 235.0, When2Call `CLARIFY` 77.5, ToolACE `CLARIFY` 47.4 and `ABSTAIN` 26.5;
   - *354 / 26 / 20:* Glaive `DIRECT` 346.6, When2Call `CLARIFY` 25.2, ToolACE `CLARIFY` 11.8;
   - *random draw:* Glaive `DIRECT` 354.4 and When2Call `CLARIFY` 25.6. No ToolACE cell reaches 10.

   *Proposed:* every other cell is descriptive only, with no share, interval or estimate drawn from it on its
   own; its items still count in the source's and the pooled share.

   *This draft adds, as its own choice:* the figure without `ABSTAIN` is withheld where the source's expected
   sample outside `ABSTAIN` is below 10 (ToolACE under a random draw, 7.8). *Proposed:* yes. *Alternative:*
   report it whatever the expected sample, as the per-source figures are.

## If adopted

In the adoption commit:
- this status line changes;
- the amendment is recorded in 03 and `reports/ERRATA.md`;
- the floors, the minimum of agreed items, the source-dependent consequence, the trial minimum, the recall
  minimums, the recall seed and the descriptive-only cut-off go into `flag_triage.py` as adopted constants, and
  `recall_report` withholds the figure without `ABSTAIN` where the rule above says so;
- 46 §5, 11 and 16 gain dated notes (16 because check 7's disposition map gains `SOURCE_EXCLUDED`);
- the status surfaces, including the public site, record it.

All of this happens before the triage task exists.
