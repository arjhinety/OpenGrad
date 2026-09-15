# 28 — P-DET amendment: model reference labels (`study_002_prereg_v2`)

**Status: AMENDMENT, 2026-09-15, decided by the study owner (arjhinety).** This file changes nothing in
[22-PDET-PROTOCOL.md](22-PDET-PROTOCOL.md), [23-PDET-ANNOTATION-INSTRUMENT.md](23-PDET-ANNOTATION-INSTRUMENT.md)
or the frozen population `reports/pdet/pdet-v1.population.jsonl` (`population_sha256`
`6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b`, 581 rows). It changes **who produces the
P-DET reference labels**, and it is recorded as the pre-registration amendment the
[03 amendment procedure](03-PREREGISTRATION.md) requires, and in [`reports/ERRATA.md`](../../../reports/ERRATA.md).

## The decision

Human annotation of all 581 items is not feasible with the resources available now. It is **deferred**,
not cancelled: when it can be funded, human passes resume, and this amendment says how their labels take
over.

Until then, P-DET reference labels come from two sources:

| Source | Items | Recorded as |
|---|---|---|
| Human Pass A (`pass-a`, annotator `arjhinety`) | the items the owner labeled: #1–#11 when this amendment was made | human labels |
| Declared model annotator `model.claude-opus-5` (Claude Opus 5), session `model-a` | every item without a human label | **model judgments** |

The reference is a **composite**: each item takes its label from the first of `pass-a`, `model-a` that
labeled it. A human label therefore always wins over a model label, and every gold record carries
`label_source_session` and `label_source_kind` (`human` or `model`).

## What does not change

The population, its order and contents; the five outcomes and all field values (23 §2); the rubric (22 §1–§3);
the classifier acceptance thresholds (22 §6); the blinding (no challenge families, sampling cues, classifier
fields or other sessions' labels reach any annotator); and the exposed-worked-example exclusions
([27](27-PDET-EXPOSED-WORKED-EXAMPLES.md): 578 of 581 items metric-eligible).

## What changes in what may be claimed

1. **Model-sourced labels are not human gold.** Every metric computed on P-DET states its label sources,
   for example "agreement with a composite reference: N items labeled by Claude Opus 5, M items labeled by
   one human annotator". Nothing computed on model-sourced items may be called human-validated.
2. **Classifier acceptance (22 §6) becomes provisional.** A qualification measured against this reference
   is recorded as `MODEL_REFERENCE`. It is re-evaluated against human labels when they exist. This amendment
   does not by itself decide whether a `MODEL_REFERENCE` qualification may grant balancing permission
   (22 §6, 22 §7); the study owner records that decision separately before it is used.
3. **Independence.** If the classifier under test is itself a Claude model, or was built or tuned using
   Claude outputs, its agreement with this reference is not independent evidence and is reported as
   `NON_INDEPENDENT`.
4. **No inter-annotator agreement.** A model pass is not an independent human annotator. No raw agreement
   or Cohen's κ between annotators may be reported (22 §4). Agreement between the human and model labels
   on overlapping items may be reported only as human–model agreement, with its item count.
5. **The human labels are single-annotator labels.** The 22 §4 single-annotator re-read has not been
   applied to them, and the report says so.
6. **Lifecycle (22 §7).** The line "human P-DET gold → validates → HEURISTIC_POLICY_v1" reads, under this
   amendment, "composite reference (human where available, model elsewhere) → provisionally validates →
   HEURISTIC_POLICY_v1".

## The model annotation procedure

- **Declared annotator.** `configs/annotation/pdet-v1.yaml`, `model_annotators`: id `model.claude-opus-5`,
  model `claude-opus-5`, procedure `configs/annotation/pdet-v1.model-procedure.md` pinned by SHA-256, this
  file as the authorization. The tool refuses a `model.` session for an undeclared model, or when the
  procedure file no longer matches its hash.
- **Instructions.** Each subagent receives the procedure file verbatim, followed by one line naming its
  batch file.
- **What the model sees.** `opengrad-annotate model-batch` renders the batch: the rubric excerpts the
  annotation screen shows (the checklist, 22 §1–§4, 23 §2, §3, §5, §7), and each item's display fields and
  metadata exactly as the screen shows them.
- **What the model does not see.** It never sees:
  - the blinded columns (challenge families, classifier fields, annotation fields);
  - any human label;
  - any other batch or its answers;
  - 22 §5–§7 and 23 §1, §4, §6.
- **Isolation is by instruction, and that is a limitation.** Subagents run with file tools and are told to
  read only their batch file. Nothing enforces that technically.
- **Batches.** 50 items each, in frozen order, restricted to items without a label in `pass-a` or
  `model-a`. Batches run one subagent at a time, and each subagent starts fresh, so no batch sees another's
  decisions.
- **Recording.** `opengrad-annotate model-ingest` checks that every item in the batch is answered exactly
  once with a valid value, and records nothing unless all of them pass. It then records each label in
  `model-a` under `model.claude-opus-5`. The change-log entry names:
  - the batch;
  - the batch's content hash;
  - the procedure's hash.
- **What the model records.** Each label comes with the model's one-sentence rationale, a boundary rule
  where one decided it, and a flag when the model was unsure.
- **Exports.** Every package says what the labels are:
  - `annotator_kind: model` on the session;
  - `model_annotation: true` and the model declaration in the manifest;
  - a statement that model labels are model judgments;
  - `label_sources` counts in the gold summary.
  `opengrad-annotate verify` fails a package that presents a model-sourced gold label as human.

## Compliance with the 03 amendment procedure

1. Recorded in [03](03-PREREGISTRATION.md) under *Amendments* as `study_002_prereg_v2`, with date, reason
   and item changed.
2. Recorded in [`reports/ERRATA.md`](../../../reports/ERRATA.md).
3. Re-running the revised rule against every candidate already scored: **none exist**. No classifier has
   been implemented or scored on P-DET (`classifier_status_at_selection: NOT_IMPLEMENTED`).
4. No arm has been launched.

03 permits amendments "for defects". This is not a defect repair. It is a resource-driven change to the
reference-label source, made **before any classifier result exists**. No threshold, partition or arm moves.
It is recorded as such, so a reader can weigh it.

## Reversal when human annotation is funded

New human passes are added as new sessions and listed before `model-a` in the composite. Their labels then
replace model labels item by item. When every item has a human label, the reference is human gold again,
and claims 1–6 above lapse for that package.

The tool freezes a session only once, and a gold freeze locks every session it cites. After a composite
freeze that includes `model-a`:

- a later **all-human** freeze works, because it cites only new human sessions;
- a later **partly-model** composite cannot cite `model-a` again as the tool stands. It would need a new
  model session, or a tool change that lets an already-frozen session be cited again. That change is safe,
  because a frozen session can no longer change.
