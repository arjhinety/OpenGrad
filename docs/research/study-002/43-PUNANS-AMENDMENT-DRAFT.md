# 43 — `P-UNANS`: sources, construction, labelling and scoring (`study_002_prereg_v11`)

> **Status: ADOPTED 2026-09-25, as drafted.** The owner adopted it and authorised the draw (*"yes"*). It is
> recorded in [03](03-PREREGISTRATION.md) and `reports/ERRATA.md` §28 in the same commit. The text below is kept as
> drafted, including its draft-time wording, as [40](40-PREREG-V8-DRAFT.md) was.
>
> **As drafted:**
> - **What the owner decided on 2026-09-25**, after reading the source screening (`registry/source_screening.yaml`,
>   screening `study-002-punans`; report `reports/source-screening/study-002-punans/REPORT.md`):
>   1. **Sources:** KUQ and SelfAware, in fixed shares.
>   2. **Definition:** unanswerable includes false premises as well as unknowable questions.
>   3. **False premises** form a **separate stratum** with their own measure, because the refusal scorer would
>      count a correct premise correction as a failure.
>   4. **Agreement floor:** raw agreement of at least 80% between the two labelling models, with Cohen's κ
>      reported beside it.
>   5. **Tools:** each question is paired with tools that cannot serve it.
> - **What this document adds:** every other parameter, fixed before any item is drawn. They are Claude's
>   proposals, marked as such, and bind only once the owner adopts this draft.
> - **Where adoption is recorded:** in [03](03-PREREGISTRATION.md) and `reports/ERRATA.md`, in the same
>   commit that changes this status line. Until then the draft is recorded in neither, as
>   [40](40-PREREG-V8-DRAFT.md) was.

## 1. Why

- **P-UNANS is required.** [06](06-SPLIT-SPEC.md) and [07](07-METRIC-SPEC.md) define it as the population for
  `refusal_correctness`. [11](11-THRESHOLDS.md) check 4 needs it at **0.70**, [40](40-PREREG-V8-DRAFT.md) item B
  sets **n ≥ 385**, and [03](03-PREREGISTRATION.md) stop rule 2 halts the study if its labellers cannot agree.
- **None of it exists.** Neither the population, its source nor its agreement floor exists: no document states
  the floor's value.
- **Why it matters.** It is the counterpart of the `ANSWER` strata. Those catch a model that refuses too much,
  and P-UNANS catches one that stops refusing and invents answers.

## 2. Items changed

- **[06](06-SPLIT-SPEC.md) "Disjointness and contamination", the `P-UNANS` bullet.**
  - The source is named (§3).
  - "Genuine unanswerability" is defined (§4).
  - The labeller-agreement record is specified (§8), and so is its floor (§9).
- **[07](07-METRIC-SPEC.md) `refusal_correctness`.**
  - It is computed on the **unknowable** stratum only.
  - The false-premise stratum gets its own measure, `premise_rejection` (§10).
- **[03](03-PREREGISTRATION.md) stop rule 2** gets its value: raw pairwise agreement **≥ 0.80** (§9).
- **Nothing else changes:** every threshold, the n ≥ 385 rule, the one-shot discipline, the arms and every
  other partition stay as they are.

## 3. Sources, pinned

| Source | File | Revision | sha256 | Licence |
|---|---|---|---|---|
| KUQ (Amayuelas et al. 2023) | `knowns_unknowns.jsonl` | `amayuelas/KUQ` `f99b53aa226dbb0d1b086db3ec352b0da0aa8f41` | `798d1677f962d11d069f77d1e3db91ad2ddb483a94b697e46bc8ca62ad0aedf6` | MIT |
| SelfAware (Yin et al. 2023) | `data/SelfAware.json` | `yinzhangyue/SelfAware` `f0bad1ff77bd42fc4eb2360281ed646c7bb7bd0c` | `32929585ffdd4048f35f7f167720722eb8ddcf59908ffc958da2f84ea634dc54` | CC-BY-SA-4.0 (the data file's own statement) |
| Tool schemas: BFCL simple_python | `BFCL_v4_simple_python.json` | gorilla `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | `82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991` | Apache-2.0 |

- **Pinning.** The builder downloads each file at its revision and refuses a file whose digest differs. The
  tool file is the one [41](41-ANSWER-STRATA-AMENDMENT.md) §3 already pins.
- **Licences.** SelfAware's items are distributed under CC-BY-SA-4.0 with attribution, and share-alike
  applies to the items built from them. KUQ requires the MIT notice.

## 4. What counts as unanswerable

- **Unknowable:** no one can answer the question now.
  - **Covers:** future events, unsolved problems and facts no one can determine.
  - **The right reply:** declining.
- **False premise:** the question assumes something untrue.
  - **The right reply:** rejecting or correcting the premise, not answering as though it held.
- **Not unanswerable:**
  - open-ended or subjective questions (07: they would measure style);
  - questions missing context the user could supply (a `CLARIFY` case, which P-CONF measures);
  - questions answerable from general knowledge, including a "future" event that has since happened.

## 5. Candidate pools (proposal)

- **Pool U (unknowable):**
  - **KUQ:** items with `unknown: true` in the categories `future unknown` and `unsolved problem`.
  - **SelfAware:** items with `answerable: false`.
- **Pool F (false premise):** KUQ items with `unknown: true` in the category `false assumption`.
- **Time rule.** A question naming a year from 1900 to 2026 is dropped before the draw: the event may have
  happened, and the answer may be known. Undated questions about events that have since resolved are left to
  labelling, which marks them answerable (§4).
- **Draw.**
  - **Pool U: 350 from each source**, in ascending order of `sha256("opengrad-punans-001-v1" | source | question)`,
    among items that survive the time rule and the §7 screen.
  - **Pool F: every item that survives.**
  - **No redraw, top-up or re-screen to reach a size** (as 41 §10).
- **Sizing hypothesis, stated before any label.** KUQ carries its own categories, but SelfAware does not, and
  its paper includes subjective questions. The two-model agreement is expected to keep most KUQ items and
  fewer SelfAware ones.
  - **Pool U:** the proposal is sized so that about 490 of the 700 reach the unknowable label, above 385 with
    margin. This is a hypothesis, not a promise.
  - **Pool F:** the 512 surviving candidates may land near 385 and be under-sized.

## 6. Tools

- **Every item is paired with tools that cannot serve it,** by exactly the rule of
  [41](41-ANSWER-STRATA-AMENDMENT.md) §5, under this document's own seed:
  - k = 1–3 tools from BFCL simple_python;
  - skip any tool that names an information route or an entity domain;
  - skip any tool that shares a content word with the question.
- **Why:**
  - **Parity with P-CONF:** every P-CONF item offers tools.
  - **A clean comparison:** P-UNANS and `ANSWER-constructed` then differ only in answerability.

## 7. Exclusion screen

The screen of [41](41-ANSWER-STRATA-AMENDMENT.md) §6 runs unchanged:
- the contamination engine at levels 1–4, and the word 8-gram probe;
- against every training corpus and the When2Call evaluation splits.

It also checks for exact normalised-text collisions with `P-CONF-v1`, the `ANSWER-STRATA-v1` candidates and
P-DET-COVERAGE-v1 and -v2. A flagged item is removed before labelling, and the removals are counted.

## 8. Annotation

- **One blind task, `punans-v1`,** labelled independently by **Gemini 3.8 Flash (High)** and
  **deepseek-v4.1-flash**, the two annotators of [42](42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md).
- **What is hidden:** the pool, the source, the source's own category and the time-rule record.
- **What each item shows:** the question and its tools.
- **Labels:**
  - `UNKNOWABLE`, `FALSE_PREMISE` (§4);
  - `ANSWERABLE`, which includes resolved "future" events;
  - `SUBJECTIVE`;
  - `UNDERSPECIFIED`;
  - `UNKNOWN`, for an item that cannot be decided, with an ambiguity status as in 41.
- **Only §4 is served to the annotators;** the rest of this document names the sources.
- **No Claude model labels an item.** Claude designed the construction.

## 9. Reference, strata and the agreement floor

- **Reference label:** the label **both** annotators give. A disagreement is excluded and counted, as in 42.
- **Strata, by reference label:**
  - `P-UNANS-unknowable` holds the `UNKNOWABLE` items;
  - `P-UNANS-false-premise` holds the `FALSE_PREMISE` items;
  - every other reference label leaves the item out, counted by pool and source.
- **The agreement floor** (03 stop rule 2, gate check 5):
  - **The measure:** the share of all labelled items on which the two annotators give the same label.
  - **The floor:** **≥ 0.80.** Cohen's κ over the same items is printed beside it.
  - **Below the floor:** the population is not built, and the study does not proceed to training.
- **Sizing:** each stratum is judged on its own n with the 06 table (n ≥ 385 resolves 10 points).
  - **`P-UNANS-unknowable` is the P-UNANS of check 4,** so below 385 check 4 is `UNDER_POWERED`, as 40 item B
    states.
  - **`P-UNANS-false-premise` enters no gate.**

## 10. Scoring

- **`refusal_correctness`:** on `P-UNANS-unknowable` only, as 07 defines it, with the existing refusal detector.
  - **Direction of the bias:** the detector is high-precision, so a missed refusal counts against the model.
    The figure is therefore a lower bound, and the bias works against passing check 4.
- **`premise_rejection`:** on `P-UNANS-false-premise`, the share of replies that do not answer as though the
  premise held.
  - **No scorer exists for it.** Before any arm is scored, a judge must be specified in a later amendment and
    validated against labels its designer did not choose, as the decision classifier was.
  - **Until then** the stratum is `NOT_EVALUABLE` and is reported as such.

## 11. Limitations

- **Model reference, not human gold.** Both strata are model judgments (`MODEL_REFERENCE`).
- **Public, older data.** KUQ and SelfAware are public and date from 2023, so the base model may have seen
  them.
- **The time rule is imperfect.** It removes dated questions only; the labelling must catch undated
  questions about events that have since resolved.
- **SelfAware's yield is uncertain.** It has no categories, so its yield is unknown until labelled.

## 12. Not changed

- every threshold in [11](11-THRESHOLDS.md), including `min_refusal_correctness` 0.70 and n ≥ 385;
- `P-CONF-v1`, `P-DEV` and `P-SEALED`;
- the ANSWER strata of 41 and 42;
- training, which remains unauthorised.
