# 25 — P-DET implementation addendum (annotation tooling)

**Status: ADDENDUM.** This file does not modify [22-PDET-PROTOCOL.md](22-PDET-PROTOCOL.md) or
[23-PDET-ANNOTATION-INSTRUMENT.md](23-PDET-ANNOTATION-INSTRUMENT.md); both stay frozen byte-for-byte. It
records how the annotation tool (`opengrad-annotate`, [`docs/ANNOTATION_TOOL.md`](../../ANNOTATION_TOOL.md),
task config [`configs/annotation/pdet-v1.yaml`](../../../configs/annotation/pdet-v1.yaml)) implements the
instrument, and the one place where it differs — in file naming only.

## A1. Session-qualified pass filenames — a naming-only deviation

| | 23 §2 / §5.3 | Tool |
|---|---|---|
| Pass file | `reports/pdet/pdet-v1.annotations.<annotator_id>.jsonl` | `reports/pdet/annotation/pdet-v1.annotations.<annotator_id>.<session_id>.jsonl` |
| Adjudication file | `reports/pdet/pdet-v1.adjudicated.jsonl` | `reports/pdet/annotation/pdet-v1.adjudicated.jsonl` |

**Why.** A filename keyed on the annotator id alone lets a second pass by the same annotator overwrite the
first. Qualifying it with the session (pass) id makes every pass a separate file, which is what keeps passes
independent and a repeated pass from destroying an earlier one. The files live in the `annotation/`
subdirectory beside — never over — the three frozen P-DET files, which the task config lists as protected.

**What is unchanged.** Population membership (the 581 frozen items, `population_sha256`
`6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b`), their order and contents; the labels
and allowed values (23 §2); the record field names (`pdet_id`, `gold_policy_label`, `ambiguity_status`,
`annotator_rationale`, `boundary_rule_cited`, `annotator_id`, `annotation_version`); the annotation
procedure (22 §3–§4); pass isolation; the adjudication rules (22 §4, 23 §5); and every P-DET semantic,
threshold and acceptance criterion (22 §1–§2, §5–§7). The tool adds files (disagreements, gold, change logs,
a manifest); it replaces none.

## A2. Decisions in the tool's P-DET configuration

1. **Challenge families are hidden while labeling.** `challenge_families` (and the annotation fields
   themselves, and `classifier_version_at_selection`) are `blind_fields`: never displayed, filtered on or
   sent to the browser. They were the observable cue predicates used to draw the challenge component, and a
   name such as `refusal_plain` shown beside an item would act as a label suggestion. They are
   preserved in the population, which is never written, and every exported record carries its
   `source_row_hash`, so the link from a label to the full frozen row is never lost.
2. **A rationale is required** for every P-DET label, because 23 §2 marks `annotator_rationale` as
   required. This is a property of the P-DET task configuration, not of the tool: the generic note field
   stays optional, and other tasks need not require a rationale.
3. **The rubric panel shows excerpts, not whole files.** Beside each item the annotator sees 22 §1–§4
   (modes, boundary discriminations, decision tree, procedure) and 23 §2, §3, §5, §7 (fields, how to decide,
   passes, what not to do), plus a short checklist
   ([26-PDET-ANNOTATOR-CHECKLIST.md](26-PDET-ANNOTATOR-CHECKLIST.md)). It omits 22 §5–§7 and 23 §1, §4, §6,
   which carry sampling cues (the strata, including the refusal-signal stratum, and challenge-family names),
   the classifier acceptance criteria, per-mode coverage counting that invites balancing, and illustrative
   labels for population items. The omitted text never reaches the browser, and the server refuses to start
   if a served excerpt contains any term listed in `instruction_forbidden_terms`. The full frozen texts are
   unchanged in the repository.
4. **UNKNOWN / AMBIGUOUS** is recorded as 23 §2 prescribes: `gold_policy_label = UNKNOWN` together with an
   `ambiguity_status` other than `NONE`. The tool enforces both directions: UNKNOWN requires a reason, and a
   mode label requires `NONE`.
5. **Adjudication** records both original values, the adjudicated `gold_policy_label` and
   `ambiguity_status`, the deciding decision-tree step (22 §3) as `decision_tree_step`, a rationale and the
   adjudicator id (23 §5.3). The adjudicator writes an adjudication rationale instead of the per-pass
   `annotator_rationale`.

## A3. Observation — recorded, not resolved

The three worked examples in 23 §4 are real items of the frozen population, all in the challenge component:
`pdet_index` 2, 3 and 61 (items #3, #4 and #62 in the tool), `pdet_id`s
`when2call-sft:018334f2…57c9`, `when2call-sft:018cf1af…b670` and `when2call-sft:1f1778c8…8682`. The
instrument shows illustrative labels for them. The tool does not display 23 §4, but an annotator who has
read the instrument has seen those labels. The protocol does not say how such items are to be treated, and
this addendum does not decide it; the study owner should record a decision (for example, reporting them
separately) before metrics are computed.

## A4. Checks before and after annotation

```bash
python -m opengrad.verification.pdet --verify     # the frozen population and manifest (23 §1)
opengrad-annotate check pdet-v1                    # the task config, source pin, 581 items, blinding, rubric
```
