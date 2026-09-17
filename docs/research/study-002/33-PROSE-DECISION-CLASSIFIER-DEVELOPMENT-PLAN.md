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
owner, which costs annotation time that P-DET-COVERAGE-v1 needs.

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
- **Who sees it:** the model annotator and the classifier developer. It is not shown to the study owner,
  who annotates P-DET-COVERAGE-v1, a population drawn from the same kind of records.

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
  not ground truth, and rules tuned to agree with it can inherit its habits. The test on human labels is
  what detects that.
- **P-DET-v1 and P-DET-COVERAGE-v1 files are never opened while developing.** A test fails if the
  classifier module reads either population path.
- **Freeze:** when development ends, the rules are committed and tagged under `prose-decision-classifier-v1`
  before any test result exists. Any later change is a new version, and 22 §6's consequences apply:
  the used population is marked `DEVELOPMENT_EXPOSED`, and a new untouched population is required before
  balancing permission.
- **The test:** it runs once, on human labels only, after the study owner has annotated. On P-DET-v1 it
  uses whatever labels exist at that time, and its report states whether they are frozen gold.

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
