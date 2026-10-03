# INC-0002: how the Gemini re-label will be compared

**Written 2026-10-04, before any comparison is computed.** The re-label (session `model-gemini-r2`, under the
tool gate) is running; no re-labelled label has been read, and nothing below has been computed. Incident:
[`docs/INCIDENT_LOG.md`](../../docs/INCIDENT_LOG.md) INC-0002. Owner decision of 2026-10-03: re-label every
P-DET-COVERAGE-v1 and v2 Gemini batch and rebuild the references as new versions.

## Why the comparison needs a baseline

A model does not label the same item the same way twice, and the re-label also runs on a newer agy (1.2.16) and
possibly a changed hosted model. Labels will therefore change even where Gemini saw nothing it should not have.
A change is evidence of the exposure only in excess of the change seen in comparable batches without it.

## Groups

Each old Gemini batch that kept its labels is put in one group, from agy's conversation store: tool names and the
location class of their path arguments only, never content. The group of an item is the group of its old batch.

| Group | Rule | v1 batches (items) | routing | v2 batches (items) |
|---|---|---|---|---|
| A | read an answer file of gpt-5.6-sol or deepseek-v4.1-flash | 3 (46) | 0 | 3 (60) |
| B | read inside the repository, but no other model's answers | 4 (80) | 0 | 8 (160) |
| D | browsed only outside the repository | 9 (180) | 3 (30) | 10 (200) |

Web searches fall in A (v2, 1 batch) and B (v2, 1 batch), and are reported but not a group of their own. No batch
of these tasks is clean: every one browsed somewhere. So D is the baseline for the repository exposure only. It
cannot measure pure run-to-run variation, and that limit is stated with every result.

## What is computed, per task and group

1. **Change rate:** the share of items whose re-labelled Gemini label differs from the old one, with a Wilson 95%
   interval. Ambiguity-status changes are counted separately.
2. **Copying check:** among items where gpt-5.6-sol and deepseek-v4.1-flash give the same label, the share where
   old Gemini matches them and the share where new Gemini does. Copying would show as old agreement above new
   agreement, more in A than in D.
3. **Contrasts:** A minus D and B minus D for the change rate and for the agreement drop (old minus new), each
   with a Newcombe 95% interval for the difference of two proportions.
4. **The references:** the number of items whose consensus label or consensus kind changes between the frozen
   and the re-labelled reference, by group.
5. **What rests on them:** the saved predictions of classifier v1's test (33 §8) and v2's test (37 §8) re-scored
   against the re-labelled references with the same metric code (`pdet_coverage_metrics`), beside the frozen
   numbers, with each qualification the same rules would give. The classifiers are not run again.

## How it will be read

- An interval for A minus D or B minus D that excludes zero is evidence that the exposure changed Gemini's labels.
- An interval that includes zero means no difference detectable at these sizes (A has 106 items, B 240), not no
  effect.
- Whatever the contrasts show, the re-labelled references are the ones without the exposure. Whether results
  built on the frozen references stand, and whether 38's permissions stand, are the owner's decisions; this plan
  only measures.
