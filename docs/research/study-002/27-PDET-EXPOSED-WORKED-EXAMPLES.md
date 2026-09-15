# 27 — P-DET addendum: exposed worked examples

**Status: ADDENDUM.** This file changes nothing in [22-PDET-PROTOCOL.md](22-PDET-PROTOCOL.md),
[23-PDET-ANNOTATION-INSTRUMENT.md](23-PDET-ANNOTATION-INSTRUMENT.md) or the frozen population
`reports/pdet/pdet-v1.population.jsonl` (`population_sha256`
`6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b`, 581 rows), and it removes, replaces and
resamples nothing. It resolves the observation left open in
[25-PDET-IMPLEMENTATION-ADDENDUM.md §A3](25-PDET-IMPLEMENTATION-ADDENDUM.md).

## The finding

23 §4 illustrates the rubric with three worked examples. Each is a real item of the frozen population,
shown with an illustrative label, so anyone who has read the instrument has seen a label for that item
before annotating it. Those three items can no longer serve as an untouched test of human or classifier
judgment.

## The three items: `EXPOSED_WORKED_EXAMPLE`

| 23 §4 example | Tool item | `pdet_index` | Component | `pdet_id` |
|---|---|---|---|---|
| ① | #3 | 2 | challenge | `when2call-sft:018334f28165d73b082310e0234742a868008cb7c165b23bd0f4e1b607de57c9` |
| ② | #4 | 3 | challenge | `when2call-sft:018cf1afdf22afd0e495992a874cbd506d52dd01832e72e48f0296f613bbb670` |
| ③ | #62 | 61 | challenge | `when2call-sft:1f1778c86f62424b1a0ad2d451d252046b50599885dae8d93f67df67873d8682` |

This file deliberately does not repeat their illustrative labels.

## What the status means

- They **remain** in the 581-row frozen population, in their frozen positions.
- They are **annotated normally** in every pass and adjudicated like any other item. The annotation
  interface does not mark them.
- They are **excluded** from:
  - classifier-validation metrics (22 §6);
  - annotator-agreement statistics (raw agreement, Cohen's κ);
  - any claim that requires an untouched human-gold example.
- **Metric-eligible items: 578** = 581 population items − 3 exposed. `opengrad-annotate check pdet-v1`
  derives this from the population and the task config rather than stating it.

## How the three were identified

Each prompt and response quoted in 23 §4 was matched against the frozen population. Each matched exactly
one row, and none of the three shares its prompt, response or `canonical_hash` with any other item, so no
duplicate carries the same exposure. A scan of the repository's Markdown documents and task configs found
no other population item quoted in full. Short refusal openings common to many items (for example
*"I apologize, but I'm unable to perform that task."*) do appear in 23 §4 and in
`reports/OPENWEIGHTS_TRANSFER_EVALUATION.md`, but they are shared phrasings and expose no individual item's
label.

`tests/annotation/test_annotation_pdet_preflight.py` repeats the derivation from 23 §4 on every run and
fails if it stops agreeing with this table, with the task config, or with the 578 count.

## How the status is carried

- **Task config** — `configs/annotation/pdet-v1.yaml`, `metric_exclusions`, lists the three ids and names
  this file. The tool refuses to open the task if an id is not a population item or is missing here.
- **Every export** (work-in-progress snapshots and the gold freeze) — the manifest's `metric_exclusions`
  section records the status, reason, this document, what the items are excluded from, their ids and the
  metric-eligible count. Every pass, disagreement, adjudication and gold record carries a
  `metric_exclusions` field: `["EXPOSED_WORKED_EXAMPLE"]` for these three, `[]` for the rest. The manifest
  also gives metric-eligible counts per pass, for disagreements and for gold.
- **`opengrad-annotate verify`** fails a package whose records or counts disagree with its exclusion
  section.
- **Not in the source** — the population file is never written. The exclusion lives in the config and the
  exports.
- **Not shown to annotators** — the status is never sent to the browser, so it can't signal anything
  about these items while they are labeled.
