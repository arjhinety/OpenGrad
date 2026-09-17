# 33 — How the prose decision classifier is developed (plan, before implementation)

**Status: PLAN, 2026-09-17. Written before any classifier code, development set or result exists.** It
records how `prose-decision-classifier-v1` will be built and frozen, so that the procedure cannot be shaped
by what the tests later show. It changes no threshold, population or preregistered rule. The acceptance
rules stay those of [22](22-PDET-PROTOCOL.md) §6 and [30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md) §11.

## 1. What the classifier is, and what is already fixed

The classifier reads one tool-free, single-exchange record: a user message, one assistant reply, and the
tools offered, if any. It says which of four things the reply does:
- **DIRECT:** answers the request;
- **CLARIFY:** asks for something it needs;
- **UNSUPPORTED:** declines or explains that it cannot act;
- **UNKNOWN:** abstains.

A record where the assistant made a structured tool call is CALL by its structure and never reaches this
classifier (30 §4).

Already fixed elsewhere, and not reopened here:

| What | Where |
|---|---|
| The four input fields, and that the source dataset, ids and the system message are never inputs | [32](32-CLASSIFIER-INPUT-CONTRACT.md) §2–§3 |
| Which records are eligible | 32 §4 |
| Deterministic, versioned rules; version string `prose-decision-classifier-v1` | `src/opengrad/data/versions.py` (`DECISION_CLASSIFIER_VERSION`) |
| Pass marks per mode, and the rule that iterating against P-DET results is prohibited | 22 §6, 30 §11 |
| Development data excludes every P-DET-v1 and P-DET-COVERAGE-v1 item | 22 §7, 30 §9 |

## 2. The problem this plan solves

Rules are normally improved by trying a draft on examples with known correct answers and fixing its
mistakes. The only human-labelled examples are the two validation populations, and using them that way is
exactly what 22 §6 forbids: a classifier tuned on its own test measures nothing. The roughly 16,000 other
eligible records have no labels.

**Decision (study owner, 2026-09-17):** development is guided by a **model-labelled development set**,
used only for development and never as gold. The alternatives considered were rules from the written
definitions alone, with no measured estimate before the test, and a development set labelled by the study
owner. The study owner later decided (2026-09-17) to annotate nothing, which rules that alternative out.

## 3. The development set

- **Unit and pool:** normalization-v3 records eligible under `prose-decision-input-v1` (32 §4).
- **Excluded before drawing:** every record excluded from P-DET-COVERAGE-v1's own draw (30 §9); every
  P-DET-v1 item, matched by id, `raw_record_hash`, normalized prompt or normalized response; every
  P-DET-COVERAGE-v1 item, matched by `pdetcov` id or `raw_record_hash`, normalized prompt or normalized
  response; and held-out evaluation prompts. A test proves that no development item matches any of them.
- **Size and spread:** 300 items. The draw uses the stratum predicates of 30 §7.2, so that the hard
  boundaries (refusal-like, question-like, tool-mentioning replies) are represented rather than swamped by
  plain answers. A stratum with too little supply after exclusion is reported short, not backfilled.
  Stratum X, textual calls, is expected to be empty because all of its supply is in P-DET-COVERAGE-v1.
- **Order and dedup:** ranked by `sha256(seed | stratum | id)` with seed
  `opengrad-prose-classifier-dev-v1`, at most one item per normalized prompt. The draw is byte-reproducible
  and hashed.
- **Who sees it:** the model annotator and the classifier developer, both Claude Opus 5. It is never given
  to the three external models that label P-DET-COVERAGE-v1 (34), so their labelling is not shaped by it.

## 4. How the development labels are made

- **Tool and procedure:** the same audited route as P-DET-v1's `model-a` labels: a separate annotation task
  (`prose-classifier-dev-v1`), `opengrad-annotate model-batch` and `model-ingest`, and a written procedure
  pinned by SHA-256. The procedure uses the rubric of 22 §1–§4 with 30 §2–§3.
- **Status:** every label is recorded as a model judgment under the annotator id `model.claude-opus-5`.
  None is human, and none is gold. They are never used as a validation population and never support a claim
  about the classifier's accuracy.
- **Archive:** batches, raw answers and the procedure are archived with a hash manifest, as for `model-a`.

## 5. Developing and freezing the rules

- The developer may run drafts against the development set as often as needed.
- Development results are reported as **agreement with model labels**, never as accuracy. The model is
  not ground truth, and rules tuned to agree with it can inherit its habits. The test on independent
  references is what detects that: P-DET-COVERAGE-v1's three-model non-Claude consensus (34) and P-DET-v1's
  human labels.
- **P-DET-v1 and P-DET-COVERAGE-v1 files are never opened while developing.** A test fails if the
  classifier module reads either population path.
- **Held-out development check (§5a, added 2026-09-17 before any check result existed).** Agreement on the
  development set is in-sample: the rules were adjusted while reading those items, so it overstates how well
  they carry to new text. Before freezing, the candidate rules are committed and then scored **once** on a
  second, disjoint set, `prose-classifier-devcheck-v1`:
  - 125 items, 25 per stratum (X is empty again), drawn like the development set with its own seed
    `opengrad-prose-classifier-devcheck-v1` (`src/opengrad/verification/classifier_devcheck.py`);
  - it excludes everything the development set excludes, plus every development-set item by identity,
    normalized prompt and normalized response (545 candidates removed);
  - labelled by Claude Opus 5 subagents with the same pinned procedure, under session `model-devcheck`;
  - the developer never reads its items: the evaluation script refuses to print them.

  Its result is reported next to the development result, as agreement with model labels. The candidate
  rules are not tuned against it. If the rules change after the check, both check results are reported,
  and the check set counts as exposed.

  **Disclosure.** The labelling subagents' answers, one-sentence rationales included, return to the
  developer's session. So before scoring, the developer saw short descriptions of check items, not their
  text. The candidate rules were committed (6545ef1) before any score existed and were not changed in
  response.

  **Result (2026-09-17), agreement with model labels, not accuracy.** Counts are over items the model gave a
  mode (CALL, DIRECT, CLARIFY or UNSUPPORTED):

  | Set | Agree | Rate | DIRECT predictions agreeing | Model DIRECT labels predicted DIRECT |
  |---|---:|---:|---:|---:|
  | Development (in-sample, rules tuned on it) | 223 / 226 | 0.987 | 35 / 37 | 35 / 35 |
  | Held-out check (scored once) | 102 / 110 | 0.927 | 18 / 23 | 18 / 19 |

  The six-point drop is the expected cost of tuning on the development items, and it's why the check exists.
  The direction to watch is false DIRECT: on the check set, 5 of 23 DIRECT predictions were labelled CLARIFY
  or UNSUPPORTED by the model. DIRECT precision has a 0.80 threshold (22 §6), but 23 predictions are far too
  few to estimate it, and model labels are not the reference. Neither set has a stratum X item, so textual
  CALL is untested before the test. Reports: `reports/prose-classifier/dev/prose-decision-classifier-v1.dev-agreement.json`
  and `…devcheck-agreement.json`.

  **Second round (decided by the study owner, 2026-09-17).** The owner chose one more development round before
  freezing. The developer then read the check set's 11 disagreements, so **`prose-classifier-devcheck-v1` is
  now exposed**: it becomes development data, and its 102/110 result above is the only unexposed score it will
  ever give. The rule changes that follow are measured on a fresh check set, `prose-classifier-devcheck-v2`,
  which excludes both earlier sets.

  Round 2 fixed four general gaps: offers after a decline counted as content; missing capability wordings;
  statements of what the user left out not read as requests; and "Here's the answer" after a tool-mismatch
  note. The rules were committed (a011954) before check set v2 was drawn. Set v2 has 125 items, 718
  earlier-set overlaps removed, seed `opengrad-prose-classifier-devcheck-v2`. It was labelled the same way
  (session `model-devcheck-v2`) and scored once. The same disclosure applies: the subagents' one-sentence
  rationales reached the developer before scoring.

  | Round 2 rules (source sha256, LF, `64293c51…`) | Agree | Rate | DIRECT predictions agreeing | Model DIRECT labels predicted DIRECT |
  |---|---:|---:|---:|---:|
  | Development set (in-sample) | 223 / 226 | 0.987 | 35 / 37 | 35 / 35 |
  | Check set v1 (exposed in round 2, in-sample) | 110 / 110 | 1.000 | 19 / 19 | 19 / 19 |
  | **Check set v2 (unexposed, scored once)** | **108 / 113** | **0.956** | **16 / 19** | 16 / 17 |

  Against its own fresh set, round 2 agrees more often than round 1 did against its (0.956 vs 0.927), and
  false DIRECT fell from 5 of 23 predictions to 3 of 19. These are different sets, so the comparison is
  suggestive, not a controlled measurement. Set v2 held the first textual CALL item, and the rules agreed on
  it. Reports: `reports/prose-classifier/dev/prose-decision-classifier-v1.round2.*-agreement.json`.

  **Frozen (2026-09-17).** The study owner chose one more round followed by freezing. The round-2 rules are
  frozen as `prose-decision-classifier-v1`: source sha256 (LF)
  `64293c51917a54a649bad9e96960302dd5df1b39eabd0d260d3784b3eeb6687c`, git tag `prose-decision-classifier-v1`.
  No result on P-DET-v1 or P-DET-COVERAGE-v1 existed at freezing. Every later change is a new version.
- **Freeze:** when development ends, the rules are committed and tagged under `prose-decision-classifier-v1`
  before any test result exists. Any later change is a new version, and 22 §6's consequences apply:
  the used population is marked `DEVELOPMENT_EXPOSED`, and a new untouched population is required before
  balancing permission.
- **The test:** it runs once, after the references exist: on P-DET-COVERAGE-v1 against its three-model
  consensus (`study_002_prereg_v5`, 34), reported as `MODEL_REFERENCE`, and on P-DET-v1 against its human
  labels, stating whether they are frozen gold. The study owner annotates nothing (decided 2026-09-17).

## 6. Exposure disclosure

The 570 `model-a` judgments on P-DET-v1 were produced by Claude Opus 5, the model developing this
classifier, in blind batches (28). The developer has therefore already read P-DET-v1's items. It has not
seen the study owner's labels on them, nor any classifier result on them. 22 §6 prohibits iterating
against **results**, which has not happened. Reading the items is a weaker exposure of the same kind, so
any qualification report on P-DET-v1 states it next to the numbers.

P-DET-COVERAGE-v1's items have not been read by the developer. The one exception is disclosed in 30 §13:
two argument names of one layer A item, which never reaches this classifier.

## 7. What this plan does not do

- It does not change a threshold, a population, the input contract or any preregistered rule.
- It does not authorise C1 balancing, a mixture or training.
- It creates no gold label.

## 8. The one-shot test (run 2026-09-17)

Run once on the frozen classifier (tag `prose-decision-classifier-v1`) by
`python -m opengrad.verification.prose_classifier_oneshot --run`, with the preregistered rules of
`pdet-coverage-metrics-v1`. Result: `reports/prose-classifier/test/prose-decision-classifier-v1.test-result.json`
(committed as produced, e32d510).

**References.**
- **P-DET-COVERAGE-v1:** the three-model consensus (`MODEL_REFERENCE`, provisional, 34) on 306 layer B items.
  291 were unanimous and 15 two-of-three, with no `NO_CONSENSUS`. Labels: UNSUPPORTED 180, CLARIFY 56,
  DIRECT 44, UNKNOWN 23, CALL 3.
- **P-DET-v1:** the study owner's `pass-a` labels, one annotator and not frozen gold. 575 items were classified;
  6 are not in normalization-v3 and cannot reach the classifier; 3 are excluded as `EXPOSED_WORKED_EXAMPLE`.
  The developer had read these items as the `model-a` annotator (§6).

| Row | P-DET-v1 (human, not frozen) | P-DET-COVERAGE-v1 (model reference) | Verdict |
|---|---|---|---|
| UNSUPPORTED recall (≥ 0.75) | 0.980 (290/296) | 0.972 (175/180) | PASS |
| UNSUPPORTED challenge recall (≥ 0.60) | 0.957 (89/93) | 0.989 (94/95) | PASS |
| CLARIFY F1 (≥ 0.70) | 0.980 | 0.926 | PASS |
| CLARIFY challenge recall (≥ 0.60) | 0.963 (78/81) | 0.889 (48/54) | PASS |
| DIRECT recall (≥ 0.80) | no DIRECT item | 0.932 (41/44), not evaluable: 44 < 50 | NOT_EVALUABLE |
| DIRECT precision (≥ 0.80) | 0/8, not evaluable: 8 < 50 predictions | 0.804 (41/51), 95% Wilson [0.675, 0.890] | PASS |
| DIRECT challenge recall | no DIRECT item | 0.920 (23/25), not evaluable: 25 < 30 | NOT_EVALUABLE |
| DIRECT precision on the pool, per source (≥ 0.80) | – | Glaive 0.975, PASS; ToolACE 0 of 9 DIRECT predictions agree, not evaluable (< 20) | – |
| CALL precision / challenge recall | no CALL item | 3/3, not evaluable | NOT_EVALUABLE |
| CALL on ambiguous items in M (= 0) | – | 0 | PASS |
| Macro F1 (≥ 0.75) | 0.981 | 0.953 | PASS |
| Abstention rate (≤ 0.15) | 0.000 | 0.000 | PASS |

**By the preregistered rules:**
- UNSUPPORTED and CLARIFY qualify.
- DIRECT does not qualify, because its recall cannot be evaluated on either population.
- CALL does not qualify, because its rows are not evaluable.
- C1 is therefore not authorised.

Caveats that go with these numbers:
1. **The P-DET-COVERAGE qualifications rest on a model reference.** Whether a `MODEL_REFERENCE` qualification
   may grant balancing permission is the study owner's separate decision (34 §4).
2. **The P-DET-v1 rows rest on one annotator's unfrozen labels, from a pass with a label-timing finding.**
   The developer had also read those items before development.
3. **DIRECT could not qualify under any classifier.** Neither population holds 50 reference DIRECT items; this
   was known from the reference counts before the classifier ran.
4. **False DIRECT is concentrated in two places.** On ToolACE, none of the 9 DIRECT predictions agree with the
   reference. On P-DET-v1, which has no DIRECT item, 8 DIRECT predictions were made. Both counts are below
   their evaluability minimums, but they point to the same weakness the development checks showed.
5. **No change may follow from these results inside v1.** Any rule change is `prose-decision-classifier-v2`,
   and 22 §6 then marks both populations `DEVELOPMENT_EXPOSED` and requires a new untouched population before
   balancing permission.
