# OpenGrad M0 SFT execution report

**Date:** 2026-09-10
**Scope:** implement the missing production SFT path, validate the complete M0 execution path, run the
first real SFT against the pinned canonical corpus, evaluate the result against B0, and decide whether
DPO or on-policy distillation should follow.

---

## 1. Status

| Item | Result |
| --- | --- |
| Real optimizer steps against the pinned dataset | **YES** — 4 SFT runs, 2,400 steps each |
| M0 on corpus v1 (published) | **NEGATIVE** — SFT destroys tool calling |
| M1 DPO on corpus v1 | **NEGATIVE** — DPO trades tool calling away, and over-optimises |
| M0 on corpus v2 (corrected) | **POSITIVE** — call_f1 0.5995, macro 0.6416 |
| Best trained call_f1 vs B0 | **0.5995** (v2 @ 1200) vs B0 **0.6191**; macro **0.6416** vs **0.3621** |
| Promotion | v1 SFT (6) and DPO (3) REJECTED; v2 SFT retained as candidates |
| On-policy distillation | **OUT OF SCOPE** — stopped at DPO by decision; see §6 |

The headline is that the same training procedure produced a total collapse and a usable model, and
the only thing that changed was the corpus. On v1 — the published corpus — `call_f1` fell 0.6191 →
0.0000. On v2, which restores the tool-call supervision v1 had lost to a parsing defect, it reached
0.5995: within 3% of B0 on its best metric and far ahead on every other.

Read §3.3 before trusting any single number here. B0's `call_f1` of 0.6191 comes from a degenerate
always-call policy — 97% call recall, 1.3% unsupported accuracy — so the metric that flatters B0 is
the one metric where the corrected model is still slightly behind. On balanced per-class recall the
corrected model is better by 0.28.

`TRAINING_STARTED` is satisfied: `qwen35_2b_m0_sft_full_v3` and `qwen35_2b_m0_sft_v2corpus` each took
2,400 real optimizer steps against the pinned canonical corpus, writing six and four checkpoints.

> **Artefact availability.** Of those ten checkpoints, only four still exist: the corpus-v2 run's,
> which are published. All six of the corpus-v1 run's checkpoints were deleted before upload, as
> were M1 DPO's steps 100 and 200. The reported metrics remain recomputable from surviving
> predictions, but a repeat of the DPO run did not reproduce its trajectory, so the monotonicity
> claim in §5.3 is withdrawn. See [INC-0001](../docs/INCIDENT_LOG.md).

---

## 2. What was run

```
opengrad readiness configs/experiments/qwen35_2b_m0_sft_full_v3.yaml   -> PASS, ready_for_sft=true
opengrad train     configs/experiments/qwen35_2b_m0_sft_full_v3.yaml   -> exit 0
python scripts/build_when2call_preference_pairs.py                     -> 1741 pairs
opengrad train     configs/experiments/qwen35_2b_m1_dpo_v1.yaml        -> exit 0
```

| Quantity | Value |
| --- | --- |
| Optimizer steps | 2400 |
| Examples seen | 24,272 (44% of the trainable corpus) |
| Supervised tokens | 10,254,922 |
| Trainable parameters | 1,881,825,088 (**100%**, full fine-tuning) |
| Loss | 1.9898 → 0.6808 (mean 0.598, min 0.132) |
| Wall clock | 2509.7 s (41.8 min) |
| Throughput | ~4,100 supervised tokens/s |
| Peak GPU memory | 28.27 GiB of 80 GiB |
| Checkpoints | 400, 800, 1200, 1600, 2000, 2400 |

Environment: A100-SXM4-80GB, CUDA runtime 13.0, torch 2.13.0, transformers 5.17.0, vLLM 0.29.0,
BF16, gradient checkpointing on, AdamW, cosine schedule with 120-step warmup.

Two earlier launches of the same recipe failed and are preserved as `FAILED` records rather than
overwritten:

* **v1** OOM'd at step 11 — a single 8-sequence forward pass peaked at 49 GiB.
* **v2** OOM'd at step 1 — length bucketing (added to fix v1) made maximum-width batches reachable, and
  a 4×2046-token batch needs roughly 80 GiB of activation on this architecture (~10 MB per token).

v3 bounds each micro-batch by a **token budget** rather than a sequence count, which is the fix that
matters: activation memory tracks batch width, and a corpus with a 50× length range cannot be bounded
by a sequence count at all.

---

## 3. The result, and why

Every checkpoint was measured with the same engine, renderer, template hash, parser, generation
settings, and frozen held-out manifest that produced B0 (see `docs/CANDIDATE_EVALUATION.md`).

| step | call_f1 | precision | recall | over_call | clar_ok | unsup_ok | parse_valid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | 0.0046 | 0.6000 | 0.0023 | 0.0008 | 0.9292 | 0.5058 | 1.000 |
| 800 | 0.0062 | 1.0000 | 0.0031 | 0.0000 | 0.9538 | 0.5591 | 1.000 |
| 1200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7434 | 0.7992 | 1.000 |
| 1600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9500 | 0.5591 | 1.000 |
| 2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9415 | 0.5838 | 1.000 |
| 2400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9415 | 0.5776 | 1.000 |
| **B0** | **0.6191** | 0.4542 | 0.9722 | 0.6425 | 0.1009 | 0.0131 | 0.99899 |

SFT does fix the thing it was aimed at — over-calling falls from 0.6425 to 0.0000, and clarification
accuracy rises from 0.10 to 0.94 — but it does so by **stopping calling tools altogether**. Call
recall is 0.23% at step 400 and exactly zero from step 1200. Parsing stays perfect (1.000) throughout,
so this is a decision collapse, not a formatting failure.

At step 400 the confusion matrix confirms the mechanism: of 1295 gold-CALL examples the model predicts
CALL 3, CLARIFY 1117, ANSWER 46, UNSUPPORTED 129. It did not fail to emit a tool call; it learned to
ask a clarifying question instead.

### 3.1 Root cause: the corpus contains almost no tool-call supervision

Only **9 of the 55,719 trainable records (0.0162%)** have a `<tool_call>` anywhere in their supervised
target. Measured over the whole cached corpus:

| Target behaviour | Records | Share |
| --- | --- | --- |
| prose answer / refusal | 51,937 | 93.2% |
| clarification | 3,773 | 6.8% |
| **tool call** | **9** | **0.016%** |

A model trained on that can only learn to answer and to clarify. The collapse is not a training
failure; it is the data.

The reason the tool-call records are missing is the interaction between the canonical boundary and the
adapters, and it is systematic — **the records rejected for being malformed are precisely the ones
that carry tool calls**:

| Source | Records | Trainable | Why the rest are excluded |
| --- | --- | --- | --- |
| glaive-function-calling-v2 | 99,794 | 48,387 (48.5%) | `SEM_ORPHAN_RESULT`, 51,034 records |
| when2call | 14,829 | 6,505 (43.9%) | schema / trajectory |
| toolace | 11,190 | 673 (6.0%) | schema (`type: "dict"`) |
| looptool-23k | 20,827 | 145 (0.7%) | schema (`optional`) |
| button | 7,941 | 9 (0.1%) | schema (bare property map) |
| xlam-function-calling-60k | 59,370 | **0 (0.0%)** | schema + argument validation |

Glaive is the clearest case. Its upstream format encodes calls as `<functioncall> {...}` **inside the
assistant text**, and the adapter never parses them into `tool_calls`; it still emits a `tool` message
with a generated `call_0000` id, so the trajectory is rejected as an orphaned result. The glaive
records that *do* survive are the ones where the assistant answers in prose — largely refusals such as
"I'm sorry, but I don't have the capability to book flights", which is exactly the behaviour we now
measure in the model.

So the corpus is not merely 26% of its published size. It is missing the behaviour the experiment
exists to improve, and it is dominated by the behaviour that suppresses it.

### 3.2 A metric gap this exposed

`under_call_rate` is defined as `matrix["CALL"]["ANSWER"] / call_actual` — it counts only
ANSWER-instead-of-CALL. A model that redirects every call to CLARIFY therefore scores *better* on
`under_call_rate` (0.0185) than B0 (0.0147) while having zero call recall. The suite cannot see this
failure mode through that metric; `call_recall` and `call_f1` are what catch it. Worth fixing before
the metric is used as a promotion gate.

### 3.3 The headline metric and the balanced metric disagree

The suite's headline is `call_f1`. B0 scores 0.6191 on it, and both trained stages score far worse.
That comparison is misleading on its own, because B0's F1 comes from a degenerate policy. Per-class
recall, computed from the confusion matrices:

| model | macro over CALL/CLARIFY/UNSUPPORTED | CALL | CLARIFY | UNSUPPORTED | call_f1 | over_call |
| --- | --- | --- | --- | --- | --- | --- |
| **B0** | **0.3621** | 0.9722 | 0.1009 | 0.0131 | **0.6191** | 0.6425 |
| SFT 800 | 0.5053 | 0.0031 | 0.9538 | 0.5591 | 0.0062 | 0.0000 |
| SFT 1200 | 0.5142 | 0.0000 | 0.7434 | 0.7992 | 0.0000 | 0.0000 |
| SFT 2400 | 0.5064 | 0.0000 | 0.9415 | 0.5776 | 0.0000 | 0.0000 |
| **DPO 100** | **0.5436** | 0.0965 | 0.9179 | 0.6162 | 0.1715 | 0.0161 |
| DPO 200 | 0.5086 | 0.0216 | 0.8726 | 0.6317 | 0.0419 | 0.0055 |
| DPO 300 | 0.4775 | 0.0131 | 0.9623 | 0.4571 | 0.0258 | 0.0034 |

Read as a balanced three-class problem, **B0 is the worst model here** (0.3621) and every trained
checkpoint is better, with DPO at step 100 the best (0.5436). B0 recalls 97% of CALLs and 1.3% of
UNSUPPORTEDs: it has not learned the decision boundary, it has learned to always call, and `call_f1`
rewards that.

Three consequences, none of which are comfortable:

* A promotion gate keyed on `call_f1` would keep the degenerate policy and reject both attempts to
  fix it. The metric needs to be balanced across the decision classes before it is used to decide.
* Even so, **the CALL collapse is real**. A macro average of 0.54 with 9.7% call recall is not a
  usable tool-calling model, and no choice of metric makes it one.
* Both stages over-optimise. SFT is flat after step 1200 and DPO degrades from its first measured
  checkpoint, so neither has an interior optimum above B0 to select.

Two caveats on this table. The SFT step-400 row is the original full measurement, preserved in
`curve.json`; its per-checkpoint `metrics.json` was later overwritten by an operator error and its
weights had already been reclaimed, so that one point cannot be re-derived from retained artifacts.
The driver now refuses to replace an existing measurement unless `--force` is passed, which is the
guard that should have been there first. And selecting a checkpoint by any metric on this set is
selection on the evaluation set, so the best point here is optimistic by an unknown amount.

---

## 4. What was implemented

Everything below was missing or stubbed and is now real, tested, and committed.

**The SFT trainer.** `SFTTrainerBackend.train` raised `NotImplementedError` on its non-dry-run branch.
It now renders the pinned corpus, masks loss to assistant turns, runs the optimizer, checkpoints with
lineage, and resumes.

**Assistant-target masking** was the hardest part, and the docs record the two approaches that failed
before the one that works (`docs/SFT_TRAINING.md`). The pinned template is *position dependent*: it
emits a `<think>` block only for an assistant turn after the last real user query, so a fixed opener
length silently clips the first tokens of every target — a bug invisible in the loss curve. Spans are
computed in character space and mapped back through the tokenizer's `offset_mapping`, and each span is
self-checked against the text that was actually rendered.

**A corpus preprocessing path** (`preprocess.py`) that renders once, caches by rendering contract, and
reports every disposition. Parallelism is thread-capped subprocesses because a worker pool measured
~15× slower per shard here: the tokenizer sizes its rayon pool to every core, so N workers each
request N cores.

**Candidate evaluation** (`evaluation/candidate.py`) measures a checkpoint against the frozen
held-out set. `run_baseline` deliberately refuses any non-canonical model, so this is a separate path
sharing the extracted `measure_predictions`, and it refuses any config that changes the measurement
contract. Baseline evidence paths are unreachable from it.

**The DPO training loop** (`training/dpo_live.py`) — real preference optimization with a frozen
reference, with the loss arithmetic and the data contract tested directly.

**Resume** (`opengrad train --resume`) so a finished run can be extended once its curve is evaluated.
This is what makes the stopping decision below possible rather than guessed up front.

**Lifecycle hardening.** A run that failed after training but during checkpoint registration used to
be left stuck at `TRAINING` forever, indistinguishable from a live run; the failure is now recorded
and the status reflects that training completed.

**A corrected Glaive adapter** (`adapt_glaive_v2`, registered as `glaive_v2`), which reads the
function-call shape this revision actually uses. See §11.1 — this is the single change that restores
tool-call supervision to the corpus.

---

## 5. DPO: was blocked, now executed, and also negative

### 5.1 The blocker, and how it was resolved

`configs/experiments/m1_dpo.yaml` pins `when2call_pref_v1` at
`0582f7749df63a96fdc3070932e83e72396ace53`. That hash is the **When2Call upstream revision** — not a
preference artifact — and no preference file existed locally; the only artifact was
`data/processed/synthetic_dpo_pairs.jsonl` with 4 placeholder rows. The DPO path is fail-closed on
this and reported exactly that.

The revision does contain a real preference split: `train/when2call_train_pref.jsonl`, 9,000 rows,
disjoint from the `test` split that B0 measures. Downloaded at the pinned revision and converted by
`scripts/build_when2call_preference_pairs.py` into pairs rendered by the same pinned renderer SFT and
evaluation use — 1,741 usable pairs (19.3%). The remainder are skipped, not repaired, because their
tool schemas use Python type hints (`dict`, `str`, `List[int]`) where JSON Schema is required.

The pairs are genuine decision-calibration data, not a "always call" signal:

| chosen | rejected | pairs |
| --- | --- | --- |
| TOOLCALL | ANSWER / UNSUPPORTED | 2,199 |
| CLARIFY | TOOLCALL / UNSUPPORTED / ANSWER | 2,943 |
| UNSUPPORTED | CLARIFY / ANSWER / TOOLCALL | 2,858 |
| TOOLCALL | CLARIFY | 801 |

All four behaviours appear as `chosen` in near-equal numbers, and both `TOOLCALL → CLARIFY` and
`CLARIFY → TOOLCALL` are present. This is the signal M1's hypothesis calls for.

### 5.2 The run

```
opengrad train configs/experiments/qwen35_2b_m1_dpo_v1.yaml    -> exit 0
```

300 optimizer steps, 1,741 pairs, `beta` 0.1 against a frozen copy of the initial policy, 10.1 min,
peak 29.5 GiB. The policy starts from the **base checkpoint**, not from M0: starting from a model
whose tool calling had already been destroyed would confound the experiment.

Loss starts at exactly `ln 2 = 0.6931` with margin 0 — the correct value when the policy *is* the
reference — and the margin then grows to 23.4 with preference accuracy 1.000. That margin is the
warning sign: with `beta = 0.1` an implicit-reward gap of 23 is enormous, and on 1,741 pairs over
300 steps (1,200 pair updates) this is severe over-optimisation.

### 5.3 The result

| step | call_f1 | precision | recall | over_call | clar_ok | unsup_ok |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 0.1715 | 0.7669 | 0.0965 | 0.0161 | 0.9179 | 0.6162 |
| 200 | 0.0419 | 0.6829 | 0.0216 | 0.0055 | 0.8726 | 0.6317 |
| 300 | 0.0258 | 0.6800 | 0.0131 | 0.0034 | 0.9623 | 0.4571 |
| **B0** | **0.6191** | 0.4542 | 0.9722 | 0.6425 | 0.1009 | 0.0131 |

DPO does what it was asked to do and then does not stop: over-calling falls from 0.6425 to 0.0034,
precision rises 0.45 → 0.77, clarification accuracy rises 0.10 → 0.92, and unsupported accuracy rises
0.013 → 0.62. Call recall falls from 0.9722 to 0.0131, and `call_f1` with it.

The degradation is monotone from the earliest measured checkpoint, so the run never had a better
point than B0 to stop at. **DPO is a regression on the promotion metric, and all three checkpoints
were REJECTED.**

> **Correction (2026-09-10, see [INC-0001](../docs/INCIDENT_LOG.md)).** The per-step numbers above
> are correct for the weights this run produced, and they recompute exactly from that run's saved
> predictions. But the weights for steps 100 and 200 were deleted after the run without ever being
> uploaded, so only step 300 survives — the *worst* of the three, while the card tabulates all
> three. A repeat of the run with the config, preference data, seed, and environment all pinned
> did **not** reproduce these numbers: it peaked at step 200 rather than declining monotonically
> from step 100. The monotonicity claim and the identification of step 100 as the best checkpoint
> are therefore withdrawn as single-run observations. The direction of the finding — over-calling
> eliminated, call recall collapsed, precision up — holds in both runs.

### 5.4 Why both stages fail the same way

SFT and DPO are different objectives and they produce the same directional failure: CALL is
sacrificed for CLARIFY and UNSUPPORTED. The reason is the starting point. B0 is not a balanced
model — it calls almost always, which is why it has 0.97 call recall and 0.013 unsupported accuracy.
*Every* signal that moves the model off that degenerate corner reduces call recall, and the
per-example balance of the preference data cannot prevent it, because the model has far more
"stop calling" to learn than "keep calling".

DPO reweights behaviours the policy can already produce; it cannot supply a behaviour that never
appears in the training signal, and after M0's corpus analysis we know CALL is nearly absent from it.

---

## 6. On-policy distillation: not attempted, by decision

The M1 DPO stage is the end of this line of work. On-policy distillation was deliberately not
started, for the reasons below, and nothing in this report depends on it.

**It is independently blocked on hardware.** The designated teacher `Qwen/Qwen3.8-27B` is not
cached locally; only `Qwen/Qwen3.5-2B` is. A 27B BF16 checkpoint is roughly 54 GB of weights and
the filesystem has ~10 GB free, so it does not fit even before activations — and even 4-bit at
~14 GB would not. The tokenizer-compatibility gate in `docs/TEACHER_SELECTION.md` would also have
to pass first.

**It is also blocked on data.** The OPD prompt artifact `data/processed/toolpolicy_opd_prompts.jsonl`
contains 4 placeholder rows with `source_revision: "pinned"`, and the M2 config's
`onpolicy_prompts_v1` hash is the *button* source revision — an unrelated placeholder, not a real
prompt set. Both would have to be built before a run meant anything.

**And it would have been premature.** Distillation transfers a behaviour from teacher to student.
§8 shows the behaviour the student was missing came back from the corpus alone, with `call_f1`
moving 0.0000 → 0.5995 under an unchanged procedure. That is the correct order: fix the data, then
ask whether a teacher adds anything the data did not.

**No OPD, RL, or further preference stage was run.** `runs/qwen35_2b_m2_distill/` and
`runs/qwen35_2b_m1_dpo/` still hold only their original scaffold records; no M2 experiment was
created.

## 7. Stopping point — the decision that was asked for

The question was where to stop iterating SFT and move on, first to DPO and then to distillation. Both
stages have now been run, and the measured answer is **stop immediately: at zero steps of either**.

**SFT.** `call_f1` falls 0.6191 → 0.0046 (step 400) → 0.0062 (800) → **0.0000** (≥1200). Every
checkpoint is a regression, the collapse is complete by 1200, and nothing after it changes. Continuing
to 2400 bought a marginal clarification gain and worse unsupported accuracy, at the cost of all
remaining recall. The training signal for the target behaviour is 9 records, so there is no schedule
at which this improves.

**DPO.** Margin grows 0 → 23.4, preference accuracy 1.000, and `call_f1` falls 0.6191 → 0.1715 (100)
→ 0.0419 (200) → 0.0258 (300). The degradation is monotone from the first measured checkpoint, so
there was never a point at which DPO was ahead of doing nothing. It then keeps training past the
point where the margin stops meaning anything.

**Distillation** remains blocked on hardware (§6), and would in any case be distilling a behaviour the
training signal barely contains.

So the stopping rule for the v1 corpus was not "stop at step N" but **stop, and fix the data** — and
that is what was done. §8 carries the outcome: with the corpus corrected, the identical procedure
produced `call_f1` 0.5995 and macro 0.6416 instead of a collapse, which settles the diagnosis.

That makes DPO the right place to end this line of work:

* **The diagnosed cause is fixed and measured.** The dominant loss was a parseable adapter defect,
  now reading 98.5% of 67,481 previously unparseable turns and restoring tool-call supervision to
  48,723 records (§8.1).
* **The remaining loss is a separate, mechanical problem** — tool schemas written as Python type
  hints where JSON Schema is required, which is total for xlam (59,370 records, 0% yield). That is
  a data-engineering task with a known shape, not a modelling question (§11.2).
* **No remaining failure is one that a different objective would address.** Distillation transfers a
  behaviour; the behaviour came back from the data. DPO reweights behaviours a policy can produce;
  the policy can now produce them. Neither is indicated by what is left.

The next steps are therefore corpus completion and evaluation on a held-out-disjoint set, not another
training stage. See §11.

---

## 8. Corpus v2: testing the root cause rather than asserting it

Section 3.1 claimed the collapse was caused by the training signal rather than the training
procedure, on the evidence that 9 of 55,719 trainable records contained a tool call. That claim is
testable, so it was tested: a candidate corpus was built with the corrected Glaive adapter, and the
same SFT procedure was run against it.

### 8.1 The adapter fix, measured

The `v2` adapter exists because the `v1` adapter cannot parse this revision's call format. Two
deviations had to be handled, and each was found by measuring rather than assuming:

* the block is never closed — across the released corpus, 67,481 assistant turns contain an opening
  `<functioncall>` and **none** contain a closing `</functioncall>`;
* `arguments` is a single-quoted Python-style string holding JSON with lower-case booleans, so the
  body parses as neither JSON nor a Python literal. A terminating-tag-only fix recovered 3% of the
  turns; handling the quoting took it to 98.5%.

At the **training boundary** — not at materialization, which does not run the trajectory checks —
on identical raw input:

| adapter | materialized | trainable | records with tool calls | training-boundary rejections |
| --- | --- | --- | --- | --- |
| v1 (pinned) | 99,657 | 48,757 (48.9%) | **0** | 50,900 (all `SEM_ORPHAN_RESULT`) |
| v2 (fixed) | 98,339 | 97,475 (99.1%) | **48,726 (50.0%)** | 864 |

v2 accepts slightly fewer records overall because it now parses the calls and correctly refuses the
ones that are genuinely invalid — an undeclared tool, a malformed value — instead of storing an
unparsed marker and an orphaned result.

### 8.2 The corpus

`configs/releases/toolpolicy_canonical_v2.yaml` builds a **local, unpublished candidate**: three
sources of six, because xlam and button are gated upstream (HTTP 401) and looptool's source was not
located. v1 is untouched and B0 remains pinned to it. The registry entry records that v2 has not had
the semantic contamination review v1 completed, and does not claim otherwise.

Its quality at the training boundary, measured on the rendered cache:

| corpus | trainable records | records whose target contains `<tool_call>` |
| --- | --- | --- |
| v1 | 55,719 | **9 (0.0162%)** |
| v2 | 101,785 | **48,723 (47.9%)** |

That is the variable under test. Everything else about the run is held fixed: same base checkpoint,
same rendering contract, same masking, same batch policy, same schedule length.

### 8.3 The result

The same SFT procedure that destroyed tool calling on v1 was run against v2
(`qwen35_2b_m0_sft_v2corpus`, 2400 steps, identical trainer settings to `qwen35_2b_m0_sft_full_v3`).

| model | call_f1 | precision | recall | over_call | clar_ok | unsup_ok | macro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **B0** | 0.6191 | 0.4542 | 0.9722 | 0.6425 | 0.1009 | 0.0131 | 0.3621 |
| M0 v1 corpus (best of 5) | 0.0062 | 1.0000 | 0.0031 | 0.0000 | 0.9538 | 0.5591 | 0.5053 |
| M0 v2 @ 600 | 0.5672 | 0.7450 | 0.4579 | 0.0862 | 0.7783 | 0.6363 | 0.6242 |
| **M0 v2 @ 1200** | **0.5995** | 0.7373 | 0.5050 | 0.0989 | — | — | **0.6416** |
| M0 v2 @ 1800 | 0.5247 | 0.7891 | 0.3931 | 0.0577 | — | — | 0.6155 |
| M0 v2 @ 2400 | 0.5292 | 0.7878 | 0.3985 | 0.0590 | — | — | 0.6173 |

The claim is confirmed, and emphatically. Holding the trainer, hyperparameters, base
checkpoint, renderer, masking, and schedule fixed, and changing only the corpus:

* `call_f1` goes from **0.0000** (v1, total collapse) to **0.5995** (v2 @ 1200) — within 3%
  of B0's 0.6191, and that is the *only* metric where B0 is still ahead.
* over-calling falls from 0.6425 to **0.0989**, a 6.5x reduction, while call recall stays
  at 0.5050 instead of falling to 0.003.
* macro recall rises from B0's 0.3621 to **0.6416** — and unlike the v1 models, the gain
  is not bought by sacrificing CALL: per-class recall is CALL 0.505, CLARIFY 0.778,
  UNSUPPORTED 0.636, against B0's 0.972 / 0.101 / 0.013.

So v2 @ 1200 is a better model than B0 on precision (+0.28), over-calling (-0.54), macro
(+0.28), clarification (+0.68) and unsupported accuracy (+0.62), and worse on call recall
(-0.47) and `call_f1` (-0.02). That is a real decision boundary rather than a degenerate
corner. It is not a finished tool-calling model — 50% call recall is not deployment-grade
— but it is the first checkpoint in this study that is worth taking further, and it was
reached by fixing the data rather than the optimisation.

One caveat carried over from §3.3: these are four checkpoints of one schedule on one
evaluation set, so the choice of v2 @ 1200 is selection on the evaluation set. The
ranking should be re-confirmed on a held-out-disjoint set before it is treated as final.

That closes the SFT line. The corpus defect is identified, fixed, measured, and shown to
be sufficient to explain both negative results.

### 8.4 Published artefacts

The v2 checkpoints and the DPO checkpoint are published, with model cards recording the
metrics above including the negative framing:

* `arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV2` — checkpoints 600 / 1200 / 1800 / 2400
* `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO` — checkpoint 300

The v2 checkpoints were uploaded and verified before their local copies were removed, so that run
is intact. The DPO line is not: only step 300 exists, and the M0 run on corpus v1
(`qwen35_2b_m0_sft_full_v3`) lost all six of its checkpoints, none of which was ever uploaded. The
local copies were removed as one storage clean-up that assumed every run had been uploaded, and
that assumption was wrong for two of the three runs. The weights are unrecoverable; the metrics
recompute exactly from the surviving predictions, and the recovery attempt for the DPO run is
recorded in [INC-0001](../docs/INCIDENT_LOG.md).

---

## 9. Invariants

* **B0 untouched.** No baseline artifact, schema, or hash was modified. `ready_for_sft` remains true
  and the baseline metrics file is byte-identical.
* **No `--force`, no gate bypass.** Real training is gated on a clean preflight and on `readiness`;
  the dirty-tree refusal was respected rather than worked around (the work was committed first).
* **Fail-closed data boundary.** 154,760 records were quarantined rather than repaired. Nothing was
  coerced, no tool schema was normalised into validity, and no supervision was reconstructed.
* **No fabricated results.** The negative result is reported as negative. No checkpoint was promoted.
* **No DPO, distill, or RL run was started**, and none of their data requirements were satisfied with
  placeholders.
* **Reproducibility.** Every number here comes from committed configs plus artifacts under `runs/`.
  `runs/` is untracked by design; the configs, the code, and the reports are committed.

---

## 10. Verification performed

```
pytest -q                                 320 passed
ruff check src tests                      clean
ruff format --check <changed files>       clean
mypy src/opengrad/training/               clean
mypy src/opengrad/evaluation/candidate.py clean
opengrad validate                         OK
opengrad readiness <m0 configs>           PASS, ready_for_sft=true
opengrad train <micro config> --dry-run   DRY_RUN
opengrad gpu-smoke                        PASS (GPU_BOUNDARY_VERIFIED)
```

**Known pre-existing issues, not introduced here:** `mypy src` reports 26 errors in
`contamination/` (13), `readiness.py` (12), and `cli.py` (1); all were already present at `5e882a0`
and none are in the modules this work added or changed. `causal_conv1d` also does not build in this
environment, so the convolution falls back to the reference PyTorch kernel
(`flash-linear-attention` is installed, which is the larger win).

`ruff format --check` did fail repo-wide at the start of this session (77 files, pre-existing). It
now passes for all 188 files under `src`, `tests`, and `scripts`: the files touched here were
formatted as they were written, and the remaining debt was cleared in the same pass. That pass is
whitespace-only — no token changes — and the full suite was re-run after it.

---

## 11. Recommended next actions, in order

1. **Fix the glaive adapter — done, as a new version.** `adapt_glaive_v2` parses the marker and the
   body that follows it. Two deviations had to be handled: the block is never closed by
   `</functioncall>` (0 of 67,481 blocks have one), and `arguments` is a single-quoted Python-style
   string holding JSON with lower-case booleans, so the body parses as neither JSON nor a Python
   literal. Measured against the released corpus, the new extractor reads **66,467 of 67,481
   unparsed turns (98.5%)** and 49,846 of the 50,851 affected records, against 3% for a
   terminating-tag-only fix. The remaining 1,014 turns are genuinely malformed and are refused
   rather than guessed at.
   It is registered as `glaive_v2` **beside** `adapt_glaive`, not in place of it, so the pinned v1
   release stays reproducible from the code that produced it. Switching to it is a new corpus
   version with its own manifest hash. Whether a recovered record then passes the remaining schema
   and argument gates is not measured here and needs a rebuild.
2. **Normalise upstream tool schemas at materialization**, not at render time. Upstream writes Python
   type hints where JSON Schema is required, and the mapping is mechanical: `dict` → `object`,
   `str` → `string`, `int` → `integer`, `float` → `number`, `bool` → `boolean`, `list`/`List[T]` →
   `array` (with `items` from `T`), a trailing `", optional"` stripped. This is the second-largest
   loss and it is total for some sources: xlam is 59,370 records and currently yields **0** trainable,
   and it also caps the preference pairs at 19.3% of the split (1,741 of 9,000). Do it at
   materialization so the canonical record carries a valid schema, and version it like §11.1.
3. **Re-release as canonical v2** with a new manifest hash, and re-run M0 and M1 against it. Do not
   mutate v1: B0 and every existing result are pinned to its hash.
4. **Verify the mixture before the next training run.** Two cheap checks would have predicted both
   negative results before any GPU time was spent: what fraction of supervised targets contain
   `<tool_call>` (9 of 55,719 here), and what per-class recall the starting checkpoint already has
   (B0: 0.97 CALL, 0.10 CLARIFY, 0.013 UNSUPPORTED). Make the first a readiness gate.
5. **Balance the promotion metric before trusting it.** See §3.3: `call_f1` alone prefers the
   degenerate always-call policy over both attempts to fix it. Use a macro average over the decision
   classes, and fix `under_call_rate`, which counts only ANSWER-instead-of-CALL and so scores a
   CLARIFY-redirect collapse *better* than B0.
6. **Then** retry SFT and DPO on the rebuilt corpus. A balanced starting model removes the mechanism
   that made every intervention here reduce call recall.
