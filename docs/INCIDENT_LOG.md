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
| INC-0002 | 2026-10-03 | Gemini labeller searched the web and read the repository, including other models' answers | Closed 2026-10-04 — re-labelled under a tool gate; no detectable effect |

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
under the original step labels would attach this run's measurements to the original's card. What
is committed under `runs/qwen35_2b_m1_dpo_v1_restore/` is the run record, the training log, and the
per-checkpoint `metrics.json` / `curve.json`. The regenerated checkpoint weights and the
per-example predictions are **not** committed (and are not in the repository at all), so the
aggregate disagreement can be read but not recomputed from predictions.

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

* `qwen35_2b_m0_sft_full_v3` has no model weights anywhere and never will, so its negative result
  is published as an **evaluation record** rather than a model:
  [`arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV1-evaluation`](https://huggingface.co/datasets/arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV1-evaluation)
  (dataset repository, 27 files, digests in its `CHECKSUMS.json`). Its predictions were still
  local-only and gitignored, which was the same exposure that caused this incident; that is now
  closed for the M1 DPO run and this one.
* Step 400 of that run is the one row of the evaluation table with no predictions behind it. It
  cannot be reconstructed, and the table says so rather than dropping the row.
* The M0-on-corpus-v2 run's predictions are still local-only, but all four of its checkpoints are
  on the Hub, so the exposure there is lower.

### Evidence preserved for this incident

The M1 DPO per-example predictions were uploaded to the model repository so the affected numbers
stay checkable, with digests recorded in
[`reports/incidents/INC-0001-prediction-digests.json`](../reports/incidents/INC-0001-prediction-digests.json):

| Checkpoint | Records | SHA-256 | Uploaded to |
|---|---:|---|---|
| step 100 | 3,650 | `fb944f7d…` | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO` `evaluation/checkpoint-100/` |
| step 200 | 3,650 | `f35e3fb4…` | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO` `evaluation/checkpoint-200/` |
| step 300 | 3,650 | `606c0d9b…` | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO` `evaluation/checkpoint-300/` |

---

## INC-0002 — The Gemini labeller was not confined to its input

**Date found:** 2026-10-03 (the runs were on 2026-09-17 and 2026-09-18)
**Severity:** High for the P-DET-COVERAGE-v1 and v2 references, which assume three independent blind labellers
**Cause:** our runner relied on agy's own settings. On this machine agy runs every tool without asking
(`toolPermission: always-proceed`), and the `--sandbox` flag the runner passed restricts terminal commands only.
The procedure told the model to open nothing and search nothing; nothing enforced it
([`docs/UPSTREAM_ISSUES.md`](UPSTREAM_ISSUES.md) UP-0012).
**State:** closed 2026-10-04. The owner decided (2026-10-03) to re-label every Gemini batch of P-DET-COVERAGE-v1
(both layers) and P-DET-COVERAGE-v2 with tools locked down, and to rebuild the references as new versions. That is
done, and the comparison and the decisions that followed are at the end of this entry.

### What happened

agy's run records keep only its printed answer. Its local conversation store keeps every step, and reading it
(tool names, argument structure and counts only, no item text) shows:
- **Web searches.** 10 `search_web` calls in 5 Gemini attempts. Two of those attempts kept their labels, both in
  P-DET-COVERAGE-v2 (batch 10 attempt 2, batch 24).
- **Browsing.** In P-DET-COVERAGE-v1 and v2, Gemini listed, searched and read files across the home folder and this
  repository. In kept batches it read paths under `.annotation/`, the git-ignored working state that holds every
  model's batches, answers and stores: 7 batches in v1, 9 in v2. The routing layer (v1) browsed the home folder,
  not the repository.
- **Other models' answers.** In 6 kept batches Gemini read an answer file of gpt-5.6-sol or deepseek-v4.1-flash:
  it opened one with `view_file` in v1 batches 10, 11 and 18 and v2 batches 04 and 10, and searched one with
  `grep_search` (matches returned) in v2 batch 20. The three models ran at the same time, so those answers existed.
- Of the 287 Gemini attempts before the triage trial, 278 could be matched to their stored conversation (the
  conversation file created in the attempt's first minute); 9 cannot be checked.

### What was not affected

- deepseek-v4.1-flash (cline reports its own tool calls): 0 tool calls in every visible attempt of every task.
- gpt-5.6-sol (codex, read-only sandbox): its transcripts show no command, tool or web event in any task.
- Gemini's kept batches of `answer-strata-v1`, `punans-v1`, `punans-v2`, `punans-v2-trial` and
  `punans-v2-constructed`, and the flag-set triage trial: in every one that could be matched to its conversation,
  the only tool calls were reads of its own input. Three kept batches cannot be matched and so cannot be checked:
  `answer-strata-v1` batches 46 and 50 and `punans-v1` batch 04.

### What it does to published claims

Where Gemini read the repository or another model's answers, its vote was not independent and possibly not blind,
so the two-of-three consensus in those batches is weaker than the references state. Whether any label changed
because of it is unknown: answering would mean reading the content. Every result measured against these references
carries this caveat until the re-labelled references exist and are compared; [`reports/ERRATA.md`](../reports/ERRATA.md)
§34 lists them. The frozen references and tests stay as they are; the comparison will be new artifacts.

### What changed

- `scripts/agy_tool_gate.py`: a PreToolUse hook, written into every agy run directory, allows only reading the
  run's input file and denies everything else. The runner refuses an attempt in which the hook never allowed a
  read of the input, so a hook that is not in force cannot pass unnoticed.
- A canary run (agy 1.2.16, 2026-10-04) asked the model to search the web and list a directory. The gate log
  showed the search denied. In the conversation store, the search's result step is 116 bytes and carries a field
  none of the 10 executed searches of September had; those results were 3.9 to 12.3 KB. The model never tried to
  list the directory, so that half of the canary tested nothing. Its run directory and gate log were deleted;
  only the conversation file remains, in the local agy store.
- Not proven: that every kind of tool call (MCP tools, subagents) passes through the hook, and what agy does if
  the hook times out or fails. The archive therefore compares, per attempt, every tool call in the conversation
  store with the gate log; a mismatch means a call bypassed the gate.
- `scripts/archive_external_model_labels.py` records every attempt's web-tool calls in the archive manifest.
- `scripts/run_external_annotation.py --session` lets a re-label go into a new session, leaving the old intact.

### The re-label and the comparison (2026-10-04)

Gemini re-labelled every item of P-DET-COVERAGE-v1 (306 layer B, 30 routing) and P-DET-COVERAGE-v2 (420) in session
`model-gemini-r2`, under the tool gate: all 39 recorded attempts were matched to their conversation, the gate saw
every tool call in each, and none was a web call. In one v1 attempt Gemini tried `run_command`; the gate denied it. The labels, references (`*.reference.gemini-r2.*`) and archives
(`*.external-models.gemini-r2.*`) are new artifacts; nothing frozen was changed. The comparison followed the plan
committed before it ran ([`reports/incidents/INC-0002-relabel-comparison-plan.md`](../reports/incidents/INC-0002-relabel-comparison-plan.md));
its output is [`reports/incidents/INC-0002-relabel-comparison.json`](../reports/incidents/INC-0002-relabel-comparison.json).

| | Group A (read others' answers) | Group B (read the repository) | Group D (outside only) |
|---|---|---|---|
| v1: Gemini labels changed | 0 of 46 | 1 of 80 | 3 of 180 |
| v2: Gemini labels changed | 0 of 60 | 2 of 160 | 2 of 200 |
| v1: agreement with gpt and deepseek lost | 0 of 42 | 0 of 77 | 1 of 174 |
| v2: agreement with gpt and deepseek lost | 0 of 59 | 0 of 158 | 0 of 196 |

- **No detectable effect.** Every A minus D and B minus D interval includes zero. In the batches where Gemini read
  another model's answers, not one of its 106 labels changed on re-labelling (Wilson upper bound 7.7% in v1, 6.0%
  in v2), and it lost no agreement with the other two models. Copying would have shown as lost agreement.
- **The references.** v1: 3 of 306 reference labels change (DIRECT 44 to 43, UNKNOWN 23 to 24); routing: none; v2:
  4 of 420 (CLARIFY 99 to 96, UNSUPPORTED 138 to 141).
- **The classifier tests.** Re-scored from their saved predictions (re-scoring against the frozen references
  reproduces both frozen evaluations exactly), every qualification is unchanged. Classifier v1: UNSUPPORTED and
  CLARIFY qualify, C1 not authorised. Classifier v2: DIRECT (glaive only), UNSUPPORTED and CLARIFY qualify, CALL
  does not, and the rules authorise C1. Three metric values of v1 and six of v2 move slightly (for example v2's
  macro F1, 0.956 to 0.954); no status changes. One now sits exactly on its bar: v1's DIRECT precision on
  P-DET-COVERAGE-v1 is 40 of 50 = 0.800 against 0.80 (it passes, and DIRECT does not qualify in v1 anyway). v2's
  UNSUPPORTED challenge recall falls from 0.981 to 0.962, still above its bar.
- **Limits.** A model's labels vary between runs, and every old batch browsed somewhere, so group D is not a pure
  run-to-run baseline. "No detectable effect" is not "no effect": the A groups are 46 and 60 items. Three kept
  batches in other tasks cannot be checked (above). 46's 0.978 and 0.964 were not recomputed.
- **What it does not decide.** Which references count from now on, and whether 38's permissions stand, are the
  owner's decisions.

### Decisions (2026-10-04)

The owner decided, after the comparison:
- **The re-labelled references are current.** Future work uses `*.reference.gemini-r2.*` for P-DET-COVERAGE-v1,
  its routing layer and P-DET-COVERAGE-v2. The frozen references stay as the record, and the artifacts already
  built on them (the two one-shot classifier tests, the supply audits) are not re-run or edited.
- **38's permissions stand.** Re-scored against the re-labelled reference, classifier v2 keeps every
  qualification, so the balancing permission and the C1 authorisation of
  [38](research/study-002/38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md) stand (dated note in 38 §7).
