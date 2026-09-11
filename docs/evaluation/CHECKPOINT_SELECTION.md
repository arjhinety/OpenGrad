# Checkpoint selection and held-out evaluation

This note records which evaluation set a checkpoint may be chosen on, and why the distinction
matters for the Canonical-v2 experiments.

## The problem the distinction solves

The partial-v2 M0 run reported its best `call_f1` at **checkpoint 1200**, and that number was
obtained by evaluating four checkpoints on `behavioral-heldout-v2` and taking the best. The
selection therefore used the same examples it reported, which makes the reported value an
optimistic estimate: it is the maximum over four draws, not an unbiased measurement of one model.

That does not invalidate the partial-v2 finding. Its claim — that the v1 corpus could not teach
tool calling — rests on a difference of two orders of magnitude (`call_recall` 0.0031 vs 0.5050)
and a monotone collapse, neither of which a four-way selection can manufacture. But it does mean
the same set cannot be reused as clean test evidence for a *close* comparison.

## Roles

| Set | Manifest | Role |
|---|---|---|
| `behavioral-heldout-v2` | `reports/evaluation/behavioral-heldout-v2.manifest.json` | **Development / checkpoint selection.** Shared with B0 and the partial-v2 run, whose checkpoint 1200 was selected on it. Disjoint from every training corpus by construction (the training releases exclude it). |
| Held-out final evaluation | not yet materialized | **Confirmatory.** Must be disjoint from checkpoint selection, from the training corpus, and from any set used to tune anything. |

`3,650` distinct examples (3,952 before the two quarantined items were removed) is small. It is
adequate for detecting a collapse of the kind v1 exhibited and not adequate for resolving a
difference of a few points, so a close call between checkpoints on this set should be treated as
a tie rather than as a ranking.

## What each experiment may do

* **Checkpoint selection** may use `behavioral-heldout-v2`. The selected checkpoint is reported
  as *selected on the development set*, with the number of checkpoints compared stated alongside,
  so a reader can discount for it.
* **Confirmatory claims** require the held-out final set, evaluated **once**, on a checkpoint
  chosen in advance. Evaluating many checkpoints on it and reporting the best repeats the
  selection problem on the set that exists to avoid it.
* **Cross-corpus comparisons** (v1 vs partial-v2 vs final-v2) should prefer metrics that are
  insensitive to a four-way maximum: per-class behaviour at a fixed step, or the shape of the
  trajectory across steps, rather than a peak value.

## Status for the definitive M0

The final held-out evaluation set has **not** been materialized. `m0_sft_canonical_v2_final`
therefore names `behavioral-heldout-v2` explicitly as
`evaluation.checkpoint_selection_manifest`, and its promotion decision is provisional: a
checkpoint promoted on development evidence requires confirmation on a disjoint set before the
result is treated as final.

This is recorded as a known limitation rather than a satisfied requirement. Materializing a
disjoint confirmatory set is the next data task after the first SFT, not a prerequisite for
running it, because the run's *between-corpus* question does not depend on it.
