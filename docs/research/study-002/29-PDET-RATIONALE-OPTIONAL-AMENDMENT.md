# 29 — P-DET amendment: the rationale becomes optional (`study_002_prereg_v3`)

**Status: AMENDMENT, 2026-09-15, decided by the study owner (arjhinety).** An ergonomics change to how human
P-DET labels are entered. **It is not a taxonomy change.** It edits neither
[22-PDET-PROTOCOL.md](22-PDET-PROTOCOL.md) nor [23-PDET-ANNOTATION-INSTRUMENT.md](23-PDET-ANNOTATION-INSTRUMENT.md).
It does not touch the frozen population `reports/pdet/pdet-v1.population.jsonl` (`population_sha256`
`6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b`, 581 rows). It is recorded as the
[03 amendment procedure](03-PREREGISTRATION.md) requires, and in [`reports/ERRATA.md`](../../../reports/ERRATA.md) §13.
It follows [28](28-PDET-MODEL-LABEL-AMENDMENT.md) (`study_002_prereg_v2`) and changes nothing that 28 decided.

## The change

23 §2 lists `annotator_rationale` ("one sentence saying why") as a required field. From this amendment on,
**the rationale is optional** for P-DET labels. In the annotation tool:

- pressing CALL, DIRECT, CLARIFY or UNSUPPORTED saves the label and moves to the next item at once;
- the rationale box and the note box stay on the screen, and stay optional;
- UNKNOWN keeps a minimal ambiguity requirement. It needs an ambiguity status other than `NONE`
  (`AMBIGUOUS_TWO_MODES`, `NON_SUBSTANTIVE`, `MISSING_CONTEXT` or `EMPTY_TOOLSET`), exactly as before. It
  needs no written text.

The reason: writing a sentence for each of 581 items made human review too slow to continue. Continued
human review matters more than a sentence per item.

This supersedes decision 2 of [25](25-PDET-IMPLEMENTATION-ADDENDUM.md) §A2 ("a rationale is required").
25 is left as it was written.

## What does not change

- **Labels and definitions.** The five outcomes, every field and every allowed value (23 §2), the rubric
  and decision tree (22 §1–§3), and both constraints. Those are: UNKNOWN needs an ambiguity reason, and a
  mode label needs ambiguity status `NONE`.
- **Population.** The same 581 rows, in the same order, with the same hash.
- **Existing rationales.** Every rationale already written is preserved as it is. That covers the 11 human
  labels in `pass-a` and the 570 model judgments in `model-a`. Nothing is deleted, rewritten or re-hashed. A
  label entered later without a rationale simply has no `annotator_rationale` value.
- **Model judgments stay hidden from human passes.** The annotation screen never shows a `model-a` label, a
  model rationale, a model flag, or which items the model labeled. There is no reveal after submitting
  either: blind annotation matters more than convenience.
- **The reference and its limits.** Everything in [28](28-PDET-MODEL-LABEL-AMENDMENT.md) stands:
  - the composite takes each item's label from `pass-a` first, then `model-a`, so a human label always
    takes precedence over a model judgment;
  - `model-a`'s 570 labels are provisional model judgments by `model.claude-opus-5`, not human gold;
  - the composite is not human gold, and it is not frozen;
  - a classifier result measured against it is `MODEL_REFERENCE` and provisional.

  `opengrad-annotate reference pdet-v1 --sessions pass-a model-a` recounts the composite from the store.
  The [README](README.md) records the distribution and what it does and does not allow.
- **Exposed worked examples.** The three items in [27](27-PDET-EXPOSED-WORKED-EXAMPLES.md) are still
  annotated normally and still excluded from metrics: 578 of 581 items are metric-eligible.

## How labels without a rationale are to be read

A label without a rationale is **not** lower-confidence because no rationale was written, and no report may
describe it that way. The rationale was never a confidence measure. Uncertainty is recorded where it always
was: the flag (`f`), the UNKNOWN label with its ambiguity status, and `boundary_rule_cited`.

The 22 §4 single-annotator re-read selects items "whose rationale invokes a boundary rule". The tool
already selects by the structured field `boundary_rule_cited`, not by rationale text
(`freeze.single_annotator_review` in `configs/annotation/pdet-v1.yaml`), so an optional rationale changes
no item's eligibility for that re-read.

## How the change is recorded

- **Config.** `configs/annotation/pdet-v1.yaml` sets `annotator_rationale` to `required: false`. It
  declares the change under `definition_amendments`, from task-definition hash
  `0f060bb395bfb0bc24cea8091fb9916fe62a717d5a0bc09b61e8004536a4e640` (the definition all 581 existing
  labels were made under) to `c00eab8d903cfbf5b3f033763306d063183f5e8c8a22f6ed81e52489695c958e`. The one
  difference between the two definitions is that flag.
- **The store accepts only a declared relaxation.** When the task definition changes after labels exist,
  `opengrad-annotate` refuses to open, unless both of these hold:
  - a declared amendment leads from the stored definition to the new one;
  - the change only turns required fields into optional ones.
  Every existing label satisfied the stricter rule, so it still satisfies the relaxed one unchanged.
- **The change is logged.** On the first open under the new definition, the store appends one entry to a
  hash-chained `definition_history`. The entry holds:
  - both definitions and their hashes;
  - this document's path;
  - how many annotation records existed at that moment;
  - the time.
  `opengrad-annotate audit` re-verifies the chain. Every export manifest carries it, and
  `opengrad-annotate verify` fails a package whose history does not hash, or does not end at the
  package's definition.

## Review order in the tool (not a protocol change)

To let human review go where it is most useful first, the tool offers one pinned review queue named
`priority-review` beside the full list. Membership is fixed in
`reports/pdet/review/pdet-v1.priority-review.json`, pinned by SHA-256 in the task config. The queue holds:

- the items whose reference label was flagged uncertain;
- the items whose reference label is DIRECT or UNKNOWN;
- a seeded sample of model-sourced CLARIFY and UNSUPPORTED items.

As built from the store (seed `pdet-v1-priority-review-1`) it holds **117 items**:

- 64 flagged;
- 3 more whose label is DIRECT or UNKNOWN (6 items meet that criterion, and 3 of them were already
  flagged);
- 50 sampled: 25 CLARIFY and 25 UNSUPPORTED, none of them exposed worked examples.

One item, the human UNKNOWN, is already labeled in `pass-a`.

The screen shows only the queue's name, its item ids and this session's own progress. It never shows why an
item is in the queue. The queue is shown in an order derived from the item id alone, so the order does not
group items by reason. The reasons, which name reference labels, stay in the queue file. **Do not open
that file before labeling** (26, item 9).

The queue does leak one weak signal: an item in it is more likely than average to carry a flagged or rare
reference label. That signal is about the queue as a whole, never an item's label, and the sample of
common-label items dilutes it. The annotator can always switch back to all remaining items.

The queue changes neither which items are labeled nor how they are labeled. It sets only the order in
which one annotator may take them.

## Compliance with the 03 amendment procedure

1. Recorded in [03](03-PREREGISTRATION.md) under *Amendments* as `study_002_prereg_v3`.
2. Recorded in [`reports/ERRATA.md`](../../../reports/ERRATA.md) §13.
3. Re-running against candidates already scored: **none exist**. No classifier has been implemented or
   scored on P-DET.
4. No arm has been launched.

Like 28, this is not a defect repair. It is a resource-driven change to how labels are entered, made before
any classifier result exists. No threshold, partition, arm, label or definition moves.

## Status after the human pass (added 2026-09-15)

The counts above are as they stood when this amendment was made, and they stay as written. Since then,
Pass A (`arjhinety`) has labeled all 581 items: the remaining 570 were saved in one session, 17:21–17:35
local time, without rationales. So the composite now takes every label from the human pass, and the 570
model judgments are kept as a superseded comparison. Human–model agreement on those 570 items is 559
(98.1%). That is agreement between one human and one model, not inter-annotator agreement.

Nothing is frozen. The recount, and the finding that P-DET-v1 contains no CALL or DIRECT item, are in the
[README](README.md) and [`reports/ERRATA.md`](../../../reports/ERRATA.md) §14.
