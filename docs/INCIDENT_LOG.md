# Incident log

This project publishes its mistakes along with its results. An incident that changed what we
can claim is part of the research record, not an embarrassment to be quietly cleaned up. If a
number in a report has weaker support than it appears to, that has to be visible from the
report, and this file is where the reason lives.

Each entry states what happened, what was lost, what survived, what it does to any published
claim, and what changed as a result. Entries are never edited to look better; a later entry can
supersede an earlier one, and both stay.

| ID | Date | Summary | State |
|---|---|---|---|
| INC-0001 | 2026-09-10 | Checkpoint weights deleted before upload; partly unrecoverable | Open — weights unrecoverable |

---

## INC-0001 — Checkpoint weights deleted before upload

**Date:** 2026-09-10
**Severity:** High for provenance, low for the headline conclusions
**Cause:** operational error — local checkpoint weights deleted to reclaim disk before every
checkpoint had been uploaded
**State:** the deleted weights are unrecoverable. The affected *numbers* remain verifiable. One
claim in the M1 DPO report does not survive a repetition of the run.

### What happened

Training runs save checkpoint weights under `runs/<experiment_id>/checkpoints/`. Disk on this
machine is a single 96 GB volume that was around 60% full, and a full-parameter 2B checkpoint
with optimizer state is ~11 GiB, so the run directories were treated as reclaimable space.

Local checkpoint weights were deleted after the runs finished. That was correct for the runs
whose checkpoints had already been uploaded — and wrong for the runs whose checkpoints had not.
The deletion was not conditional on the upload having happened, and nothing in the pipeline
required it to be. Two runs lost weights that exist nowhere else.

There was no attempt to conceal this, and no automatic process did it: it was a manual storage
clean-up with a wrong ordering assumption in it. The assumption was "the upload step covers the
checkpoints", which was true for M0 on corpus v2 and false for the other two runs.

### What was lost, and what survived

| Run | Checkpoints saved | Uploaded | On disk now | Status |
|---|---:|---:|---:|---|
| `qwen35_2b_m0_sft_full_v3` (M0, corpus v1) | 6 (400–2400) | **0** | 0 | **all 6 weights lost** |
| `qwen35_2b_m1_dpo_v1` (M1 DPO) | 3 (100, 200, 300) | 1 (step 300) | 0 | **steps 100 and 200 lost** |
| `qwen35_2b_m0_sft_micro` | 3 (10, 20, 30) | 0 | 0 | weights lost; plumbing run, not evidence |
| `qwen35_2b_m0_sft_v2corpus` (M0, corpus v2) | 4 (600–2400) | 4 | 0 | **intact on the Hub** |

The loss is asymmetric in a way that matters. In the M1 DPO run the discarded step 100 was the
*best* measured checkpoint of the three (`call_f1` 0.1715 against 0.0258 at step 300), and step
300 — the only survivor — was the worst. The published artefact is therefore the weakest state
of the run, while the card tabulates all three.

### Evidence that survives even where the weights do not

Deleting a checkpoint destroys the ability to *re-run inference*. It does not destroy the
ability to *check the arithmetic*. Every checkpoint's per-example predictions were written to
`runs/<experiment_id>/eval/checkpoint-N/predictions.jsonl` before the weights were removed, and
those files survived.

The recorded metrics were recomputed from those predictions with `opengrad.evaluation.routing`
and compared against the `metrics.json` values in the reports:

| Runs compared | Metrics compared | Max absolute difference |
|---|---:|---:|
| `qwen35_2b_m0_sft_full_v3` + `qwen35_2b_m0_sft_v2corpus` | 54 | 4.7e-07 |
| `qwen35_2b_m1_dpo_v1` + `qwen35_2b_m1_dpo_v1_restore` | 36 | 4.9e-07 |

Every difference is at the sixth decimal, from `metrics.json` rounding. So the published metrics
are exactly what the surviving predictions produce. They are not trusted on the strength of the
run that wrote them; they are recomputable from evidence that still exists.

Two limits on that mitigation:

* It verifies the **scoring**, not the **model**. It cannot tell you whether the model generalises
  elsewhere, and it cannot be used to sample new completions.
* Step 400 of `qwen35_2b_m0_sft_full_v3`'s evaluation artifacts are also missing — the directory
  does not exist on disk or in git, so the `call_f1` 0.0046 recorded for it in `curve.json` is the
  only trace. Its predictions were lost along with the weights. The other five points of that run
  recompute exactly; that one does not.

### The recovery attempt, and its failure

The M1 DPO weights were the ones worth recovering, so the run was repeated. The config is
`configs/experiments/qwen35_2b_m1_dpo_v1_restore.yaml`, identical to the original apart from the
experiment ID (the CLI refuses a second launch under an existing ID, which is what keeps a
finished run directory from being silently rewritten).

Everything that could be held fixed was held fixed, and this was checked rather than assumed:
same base checkpoint revision, same preference file with the same sha256
(`474a8bb1…`, 1,741 pairs), same seed, same hyperparameters, and the same software environment —
every package in the venv predates the original run, and neither `pyproject.toml` nor `uv.lock`
changed between the two.

It did not reproduce. `call_f1` by step:

| Step | Original | Re-run | Delta |
|---|---:|---:|---:|
| 100 | 0.1715 | 0.1036 | −0.0679 |
| 200 | 0.0419 | 0.1186 | +0.0767 |
| 300 | 0.0258 | 0.0153 | −0.0105 |

The divergence is qualitative, not just numeric. The original declined monotonically with step
100 best; the re-run peaked at step 200. `unsupported_accuracy` differs by as much as 0.36 at the
same step (0.632 vs 0.271 at step 200).

The regenerated weights were **not** published. They are a different model, and uploading them
under the original step labels would attach this run's measurements to the original's card. The
regenerated checkpoints and all their measurements remain in
`runs/qwen35_2b_m1_dpo_v1_restore/` and are committed, so the disagreement can be inspected
rather than taken on trust.

### What this does to published claims

**Unaffected.** The corpus finding — that the v1 canonical corpus carries tool-call supervision
for 9 of 55,719 trainable records, and that this explains the SFT collapse — is a property of the
data, measured at the training boundary, and does not depend on any checkpoint. M0 on corpus v2 is
unaffected: all four of its checkpoints are on the Hub.

**Weakened.** The M1 DPO result's *direction* appears in both the original and the re-run:
over-calling is eliminated (0.6425 → 0.0034 original, 0.0021 re-run), call recall collapses
(0.9722 → 0.0131 original, 0.0077 re-run), precision rises (0.4542 → 0.6800 original, 0.6667
re-run). That the failure is a genuine property of DPO on this preference set is supported.

**Withdrawn.** The claim that degradation is monotone from the first measured checkpoint, and
therefore that "step 100 is the best checkpoint". A single re-run with everything else pinned
produced a different trajectory that peaked at step 200. One run was never enough to support a
statement about the shape of the curve, and the weights that would have let anyone check it no
longer exist. Treat the per-checkpoint ranking as a single-run observation, not a finding.

### What changed

* `docs/CHECKPOINTS.md` §3 now makes the ordering explicit: upload **all** retained checkpoints,
  verify what landed, and only then delete locally. It states that a deleted checkpoint is gone
  rather than regenerable, and that disk pressure is not a reason to skip the upload — the
  retained set is bounded at ~3.8 GiB per checkpoint without optimizer state.
* The claim is deliberately worded as "not guaranteed reproducible" rather than "not
  reproducible", and this incident is the reason: the re-run was expected to reproduce and did
  not, but that was a measured outcome, not a known rule.
* Any statement about a training curve's *shape* now has to be treated as a single-run
  observation unless a repetition supports it.

### Open items

* `qwen35_2b_m0_sft_full_v3` has no Hub repository, so its surviving predictions and its negative
  result have no published location — and its predictions are still local-only and excluded from
  git by `.gitignore` (`runs/**/predictions.jsonl`). For a run whose weights are gone they are the
  only per-example evidence in existence, so this is the same exposure that caused this incident.
  Deciding whether to publish an eval-only repository for it is outstanding.
* The M0-on-corpus-v2 run's predictions are also local-only, but its weights are all on the Hub,
  so the exposure there is lower.

### Evidence preserved for this incident

The M1 DPO per-example predictions were uploaded to the model repository so the affected numbers
stay checkable, with digests recorded in
[`reports/incidents/INC-0001-prediction-digests.json`](../reports/incidents/INC-0001-prediction-digests.json):

| Checkpoint | Records | SHA-256 | Uploaded to |
|---|---:|---|---|
| step 100 | 3,650 | `fb944f7d…` | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO` `evaluation/checkpoint-100/` |
| step 200 | 3,650 | `f35e3fb4…` | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO` `evaluation/checkpoint-200/` |
| step 300 | 3,650 | `606c0d9b…` | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO` `evaluation/checkpoint-300/` |

