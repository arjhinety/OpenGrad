# Negative result: P-UNANS could not be built at the preregistered agreement floor

**2026-09-25. Study 002, `study_002_prereg_v11` ([43](../../../docs/research/study-002/43-PUNANS-AMENDMENT-DRAFT.md)).**
Every number here is read from [`punans-v1.strata.json`](punans-v1.strata.json), and a test pins the ones the
study README repeats. The study owner accepted the stop as final on 2026-09-25.

## What was attempted

- **The set.** `P-UNANS` is Study 002's set of genuinely unanswerable questions. Each trained arm was to be
  scored on it once, for `refusal_correctness`: the share of questions the model declines instead of inventing
  an answer. The safety check (11, check 4) needs that share at 0.70 or more, on at least 385 items.
- **The stop rule.** [03](../../../docs/research/study-002/03-PREREGISTRATION.md) stop rule 2 says the study
  does not proceed to training if the labellers of that set cannot agree often enough. 43 §9 set the floor
  before any item was drawn: the two labelling models must give the same label on **at least 80%** of items.

## What was done

- **Candidates.** 1,063 questions, drawn, screened and committed before any label (`punans-v1.population.jsonl`,
  sha256 `9965e784…`):
  - 700 marked unknowable by their source, 350 each from KUQ and SelfAware;
  - 363 false-premise questions from KUQ.

  Each question offered 1–3 tools that could not serve it.
- **Labelling.** Gemini 3.8 Flash (High) and deepseek-v4.1-flash each labelled every question blind, under a
  pinned procedure, with six labels: unknowable, false premise, answerable, subjective, underspecified, and
  undecided. Neither saw the other's labels, the source, or the source's own tag.

## The result

| | Value |
|---|---|
| Items on which both models gave the same label | 818 of 1,063 |
| Raw agreement | **0.770** (floor **0.80**) |
| Cohen's κ (agreement corrected for chance) | 0.703 |
| Status | **STOP**: the population is not built |

**Agreement is below the floor within every source,** so the failure is not one bad source:

| Candidates | Same label |
|---|---|
| KUQ, marked unknowable | 276 of 350 |
| SelfAware, marked unanswerable | 263 of 350 |
| KUQ, false premise | 279 of 363 |

**The set would also have been too small.** Had the floor passed, the unknowable stratum would have held 274
agreed items, under the 385 check 4 needs, so check 4 would have been `UNDER_POWERED`. The false-premise
stratum would have held 227.

**SelfAware is mostly not unanswerable** by these labellers' reading. Of its 350 questions:
- the two models jointly labelled 43 unknowable, 100 subjective, 63 answerable, 34 underspecified and 22 false
  premise;
- one both called undecided;
- they disagreed on 87.

## What this means for Study 002

- **The stop is final (owner decision).** Stop rule 2 applies: the study does not proceed to training, and no
  arm is trained.
- **The floor is not moved.** Changing a threshold after its result is visible is exactly what the
  preregistration forbids ([03](../../../docs/research/study-002/03-PREREGISTRATION.md), Amendments).
- **What stays valid,** for Study 002's record or a successor:
  - `P-CONF-v1`, the four-mode confirmatory set (`reports/study-002/pconf-v1/`);
  - the `ANSWER` strata (`reports/study-002/answer-strata-v1/`);
  - the P-DET work;
  - the source screenings in `registry/source_screening.yaml`.

  None of them depends on P-UNANS.

## Uncertainty, and what the numbers do not show

- **Model judgments, not human gold.** Two different model families disagreeing on 23% of items says the task
  was hard to label consistently. It does not say which model was right.
- **κ 0.70 is conventionally "substantial" agreement.** The floor was set on raw agreement over six labels,
  and some disagreements are probably between labels that would not change whether an item enters the set
  (subjective against underspecified, for example). That is a hypothesis. It was **not tested** on these
  labels: recomputing agreement over merged labels after seeing the failure would be choosing the analysis to
  fit the result.
- **The floor was set without accounting for the number of labels.** That is a design lesson about the
  preregistration, recorded here, not a fault in the labellers.

## Contamination and exposure

- **Screening.** Before the draw, the de-duplicated candidates were screened against every training corpus.
  The screen flagged 6, which were removed. They share no text with P-CONF's `ANSWER` items or
  P-DET-COVERAGE.
- **Public data.** KUQ and SelfAware are public and date from 2023, so the base model may have seen them.
  Nothing was trained or scored on them, and the stop happened before any model saw a P-UNANS item as an
  evaluation.

## What should not be tried again without new evidence

- **SelfAware as a source of unknowable questions.** Only 43 of 350 drew joint agreement as unknowable.
- **The same instrument on a different draw:** these sources, six labels, two model labellers and a 0.80 raw
  floor. Agreement fell short within every source, so a redraw would most likely fail the same way.
- **A retry that lowers the floor, merges labels, or keeps these labels and changes their analysis.** Any new
  attempt needs a new amendment written before its own labels, with this result disclosed, a source screening
  for unknowable questions beyond KUQ, and the 0.80 floor unchanged. The owner chose not to pursue one now.

## Record

- **Report:** [`punans-v1.strata.json`](punans-v1.strata.json).
- **Reference:** [`reference/`](reference/).
- **Export:** [`annotation/wip/`](annotation/wip/). It is not gold.
- **Audit trail:** [`provenance/external-models/`](provenance/external-models/).
- **Candidates:** [`punans-v1.manifest.json`](punans-v1.manifest.json).
- **Source screening:** `registry/source_screening.yaml` (`study-002-punans`).

## Addendum, 2026-09-25: the study was reopened

Later the same day the owner reopened Study 002 for a second `P-UNANS` attempt. This result is unchanged by that:
`P-UNANS-v1` stays stopped, and none of its items or labels is reused. The retry meets the conditions above:
- **A new amendment written before its own labels:**
  [44](../../../docs/research/study-002/44-PUNANS-V2-AMENDMENT-DRAFT.md), `study_002_prereg_v12`.
- **This result disclosed:** 44 §1.
- **A source screening beyond KUQ:** `study-002-punans-v2`. It found only 61 fresh questions outside KUQ.
- **The 0.80 floor unchanged.**

It changes the instrument rather than the sources: membership-only labels, and a trial set labelled first. 44 §10
records that 0.80 over three labels is easier to reach than 0.80 over six, so a pass there is not a pass of this
floor as measured here.

