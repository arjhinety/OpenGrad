# 44 — `P-UNANS`, second attempt: a membership-only instrument with a trial set (`study_002_prereg_v12`)

> **Status: ADOPTED 2026-09-25, as drafted, with KUQP.** The owner adopted it and authorised the draw
> (*"proceed, but dont do the runs yet"*): the trial and main sets are drawn and committed, and no label is
> requested until the owner starts the runs. It is recorded in [03](03-PREREGISTRATION.md) and
> `reports/ERRATA.md` §29 in the same commit, and Study 002 is **reopened**. The text below is kept as drafted,
> including its draft-time wording, as [40](40-PREREG-V8-DRAFT.md) and [43](43-PUNANS-AMENDMENT-DRAFT.md) were.
>
> **As drafted:** *Status: DRAFT, not adopted. It is recorded in neither 03 nor `reports/ERRATA.md` until the
> owner adopts it. Until then Study 002 stays stopped before training, and nothing is drawn or labelled.*
>
> - **What the owner decided on 2026-09-25,** after reading the second source screening
>   (`registry/source_screening.yaml`, screening `study-002-punans-v2`; report
>   `reports/source-screening/study-002-punans-v2/REPORT.md`):
>   1. **Reopen Study 002 for a second P-UNANS attempt.** Earlier the same day the owner had accepted the first
>      attempt's stop as final. This reverses that; the first attempt's result is not changed by it.
>   2. **Labels:** the two models label membership only: unknowable now, not unknowable, or undecided.
>   3. **A trial set of about 100 questions** is labelled first and never enters P-UNANS.
>   4. **No false-premise stratum** in this attempt.
>   5. **Sources:** unused KUQ questions and BIG-bench Known Unknowns.
> - **A correction to what the owner was told.** When the owner chose the sources, Claude said KUQP's 40 future
>   questions all name a year that may have passed. The audit shows that none names a year up to 2026: the years
>   are 2030 to 2120, and Claude's count had included the answer field. §3 therefore proposes KUQP as well. The
>   owner confirms or removes it on adoption.
> - **What this document adds:** every other parameter, fixed before any item is drawn. They are Claude's
>   proposals and bind only once the owner adopts this draft.

## 1. Why, and what the first attempt showed

- **P-UNANS is still required.** It is check 4's population ([11](11-THRESHOLDS.md)), it must hold **n ≥ 385**
  ([40](40-PREREG-V8-DRAFT.md) item B), and [03](03-PREREGISTRATION.md) stop rule 2 halts the study if its
  labellers cannot agree.
- **The first attempt failed that rule** ([negative result](../../../reports/study-002/punans-v1/NEGATIVE-RESULT.md);
  numbers from `reports/study-002/punans-v1/punans-v1.strata.json`):
  - raw agreement **0.770** over six labels, against a floor of **0.80** (Cohen's κ 0.703);
  - agreement was below the floor within every source;
  - the unknowable stratum would have held **274**, under 385.

  That result stands. P-UNANS-v1 stays `STOPPED_AGREEMENT_FLOOR`, and none of its labels or items is reused.
- **What the write-up required of a retry, and how this draft meets it:**

  | Required | Here |
  |---|---|
  | A new amendment written before its own labels | This document; no v12 item exists yet. |
  | The first result disclosed | This section, and §12. |
  | A source screening for unknowable questions beyond KUQ | `study-002-punans-v2`. It found **61** fresh questions beyond KUQ, so this attempt still depends mostly on KUQ. |
  | The 0.80 floor unchanged | 0.80, unchanged. §10 states what the change of labels does to it. |

- **Why the instrument changes.** The write-up expected the same six labels on a fresh draw to fail again,
  since KUQ's own agreement was below the floor. The question the stratum depends on is whether both models call
  an item unknowable. So this attempt asks only that, and tests the procedure on a trial set before the main run.

## 2. Items changed

- **[43](43-PUNANS-AMENDMENT-DRAFT.md) §3–§5 (sources, definition, pools)** are replaced by §3–§5 here.
- **43 §8 (labels)** is replaced by §8: three labels instead of six.
- **43 §9 (floor)** keeps its value, 0.80, measured over §8's labels on the main set (§10).
- **New:** a trial set (§9).
- **43's false-premise stratum and `premise_rejection`** are dropped from Study 002. No false-premise item is
  drawn, and `refusal_correctness` is computed on the unknowable stratum, as 43 §10 already required.
- **Nothing else changes:** every threshold in [11](11-THRESHOLDS.md), n ≥ 385, stop rule 2, the one-shot
  discipline, the arms, P-CONF-v1 and the ANSWER strata.

## 3. Sources, pinned

| Source | File | Revision | sha256 | Licence |
|---|---|---|---|---|
| KUQ (Amayuelas et al. 2023) | `unknowns_all.jsonl` | `amayuelas/KUQ` `f99b53aa226dbb0d1b086db3ec352b0da0aa8f41` | `8469ab010ce1142bfe3d330635fb58c1e537253568b417b4435a39d658b10855` | MIT |
| KUQP (Deng et al. 2024), **proposed, see the correction above** | `KUQP Dataset/future_questions.json` | `zhaoy777/kuqp-dataset` `596472f31f73acfdcb95741c277413fd500f8b35` | `25ee426e7881b3cf950bffb1b6d0ebc6c644eb0c1a2c5939662cce4a01ce40d1` | MIT |
| BIG-bench Known Unknowns | `bigbench/benchmark_tasks/known_unknowns/task.json` | `google/BIG-bench` `124892ccf54f85402852d68c93736a4fa57bf009` | `6061bdd796f2225fee6ef9c389e1b1e19f4d83da905dc7b512e0d1ca0e44c557` | Apache-2.0 |
| Tool schemas: BFCL simple_python | `BFCL_v4_simple_python.json` | gorilla `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | `82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991` | Apache-2.0 |

The builder downloads each file at its revision and refuses a file whose digest differs. KUQ and KUQP need the
MIT notice, and BIG-bench the Apache-2.0 notice.

## 4. What counts as unknowable

- **Unknowable:** no one can answer the question now. That covers future events and unsolved problems. The
  right reply is declining.
- **Not unknowable:** everything else, including:
  - questions answerable from general knowledge, including a "future" event that has since happened;
  - open-ended or subjective questions;
  - questions missing context the user could supply;
  - questions that assume something untrue.

## 5. Candidate pool and draw (proposal)

- **Pool:**
  - KUQ `unknowns_all.jsonl` items in the categories `future unknown` and `unsolved problem/mistery`;
  - KUQP's future questions (the unanswerable side of each pair only);
  - BIG-bench Known Unknowns items whose highest-scored target is `Unknown`.
- **Exclusions, before the draw:**
  - duplicates by normalised text (the first occurrence in file order is kept);
  - **every question among P-UNANS-v1's 1,063 candidates**, by normalised text;
  - the time rule of 43 §5: a question naming a year from 1900 to 2026;
  - the §7 screen.
- **Supply,** from `reports/source-screening/study-002-punans-v2/punans-v2-supply.json`:
  **1,466** fresh questions before the §7 screen: KUQ 1,405, KUQP 40 and BIG-bench 21.
- **Order:** ascending `sha256("opengrad-punans-002-v2" | source | question)` over the whole pool.
- **Trial set `P-UNANS-v2-TRIAL`:** the first **100** in that order.
- **Main set `P-UNANS-v2`:** the next **800**.
- **Both sets are written in the same build, before any label.** The main set is therefore fixed before the
  trial is labelled, and nothing the trial shows can change which questions it holds.
- **No redraw, top-up or re-screen to reach a size** (as 41 §10).
- **Sizing hypothesis, stated before any label.** At the first attempt's KUQ rate (216 of 350 jointly labelled
  unknowable), about 490 of 800 would enter the stratum, above 385. This is a hypothesis, not a promise:
  - about two-thirds of the fresh KUQ questions were written by GPT, and none of those has been labelled
    before;
  - the rate was measured under six labels, not three.

## 6. Tools

Every item is paired with 1–3 BFCL simple_python tools that cannot serve it, by the rule of
[41](41-ANSWER-STRATA-AMENDMENT.md) §5 and 43 §6, under this document's seed.

## 7. Exclusion screen

The screen of 43 §7 runs unchanged:
- the contamination engine at levels 1–4 and the word 8-gram probe, against every training corpus and the
  When2Call evaluation splits;
- exact normalised-text collisions with `P-CONF-v1`, the `ANSWER-STRATA-v1` candidates and P-DET-COVERAGE-v1
  and -v2.

A flagged item is removed before the order is applied, and the removals are counted.

## 8. Annotation

- **Two blind tasks,** `punans-v2-trial` and `punans-v2`, labelled independently by **Gemini 3.8 Flash (High)**
  and **deepseek-v4.1-flash**, the annotators of [42](42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md) and 43.
- **What is hidden:** the source, the source's own category, the author field and the time-rule record.
- **What each item shows:** the question and its tools.
- **Labels:**
  - `UNKNOWABLE` (§4);
  - `NOT_UNKNOWABLE`;
  - `UNKNOWN`, for an item that cannot be decided, with an ambiguity status as in 41.
- **Only §4 is served to the annotators.**
- **No Claude model labels an item.** Claude designed the construction.

## 9. The trial set

- **The trial is labelled first,** with the same procedure and annotators as the main set.
- **Reported:** raw agreement, κ and the label counts, per source. Counts only.
- **After the trial, the procedure text may be revised once,** to clarify §4's wording for the annotators. The
  revision, its reason and its new sha256 are recorded before any main-set label.
- **The trial cannot change:** the floor, the labels, the sources, the main set or its size.
- **The trial is not a stop by itself.** If its agreement is below 0.80, the owner decides between running the
  main set (with or without a revised procedure) and stopping. The decision is recorded.
- **Trial items never enter P-UNANS,** and their labels are archived like any other.

## 10. Reference, stratum and the agreement floor

- **Reference label:** the label **both** annotators give. A disagreement is excluded and counted, as in 42.
- **`P-UNANS-unknowable`:** the main-set items with reference label `UNKNOWABLE`. It is the P-UNANS of check 4.
- **The agreement floor** (03 stop rule 2, gate check 5):
  - **The measure:** the share of main-set items on which the two annotators give the same §8 label.
  - **The floor:** **≥ 0.80**, unchanged. Cohen's κ is printed beside it, with the per-source agreement.
  - **Below the floor:** the population is not built, and the study does not proceed to training.
- **What changing the labels does to the floor.** Two labellers agree by chance more often when there are three
  labels than when there are six. So 0.80 over §8's labels is **easier to reach** than 0.80 was over 43's six.
  - This is a design choice made before any v12 label: only the membership decision affects the stratum.
  - It is disclosed here, and must be repeated wherever a v12 pass is reported. A reader should weigh a pass
    knowing that the first attempt's floor, measured the first attempt's way, was not met. κ, which corrects
    for chance, is the figure that compares across the two attempts.
- **Sizing:** below 385, check 4 is `UNDER_POWERED` (40 item B).

## 11. Scoring

`refusal_correctness` on `P-UNANS-unknowable`, exactly as 43 §10 specifies, with the existing high-precision
refusal detector. The figure is a lower bound, and the bias works against passing check 4.

## 12. Limitations

- **Model reference, not human gold** (`MODEL_REFERENCE`).
- **Mostly KUQ again,** public since 2023: the base model may have seen it. About two-thirds of the fresh KUQ
  questions were written by GPT.
- **A weaker floor than the first attempt's,** as §10 states.
- **The time rule is imperfect.** It removes dated questions only; labelling must catch undated questions about
  events that have since resolved.
- **The trial could mislead.** At 100 items, its agreement is noisy; it guides the procedure and never decides
  the floor.

## 13. Not changed

- every threshold in [11](11-THRESHOLDS.md), including `min_refusal_correctness` 0.70 and n ≥ 385;
- P-UNANS-v1 and its negative result;
- `P-CONF-v1`, `P-DEV`, `P-SEALED` and the ANSWER strata;
- training, which remains unauthorised.

## If adopted

In the adoption commit:
- this status line changes;
- the amendment is recorded in 03 and `reports/ERRATA.md`;
- the screening's `owner_decision` names the adopted sources, and each is registered in `registry/datasets.yaml`;
- every status surface, including the public site, changes from *stopped before training* to *reopened: second
  P-UNANS attempt* (G16, G17).

The draw follows only on the owner's word.
