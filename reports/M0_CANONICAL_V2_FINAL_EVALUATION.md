# M0 on Canonical-v2 (final) — evaluation report

**Status:** evaluation complete. Selected checkpoint `checkpoint-1800`, `NOT PROMOTED`.
**Companion:** [`M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md`](M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md)

Three kinds of evidence appear below and are kept separate throughout. Conflating them is the
error this structure exists to prevent.

| Evidence | Set | Used for |
|---|---|---|
| **Checkpoint selection** | DEV, 2,373 examples, fingerprint `88a56821…` | Choosing the checkpoint |
| **Confirmatory** | 1,277 examples, fingerprint `d6d1e394…` | Measuring the chosen checkpoint once |
| **Historical** | the same partitions, rescored from frozen predictions | Comparison against the lineage |

The confirmatory partition is a **pre-registered internal** partition, not an untouched external
benchmark. The wider upstream population has already participated in earlier OpenGrad work,
including the partial-v2 checkpoint selection. It was frozen and committed before training, and
the selected checkpoint was scored on it exactly once.

---

## 1. Answer to the scientific question

> Does the finalized heterogeneous corpus recover tool-call recall and tool-selection behaviour
> while preserving the large partial-v2 gains in precision, unsupported handling, clarification,
> and reduced over-calling?

**Partly, and the answer has a cost that is reported rather than smoothed over.**

Call recall is recovered substantially: **0.5342 → 0.7594** (+0.225). `call_f1` improves
**0.6278 → 0.7470**. Precision, unsupported handling and clarification remain far above B0 but
are each **slightly below** partial-v2. Over-calling rises from partial-v2's very low 0.0922 to
0.1505 — still a quarter of B0's 0.6238, but a real movement in the wrong direction.

So: recall recovered, headline metric up, some calibration given back to get there.

## 2. Confirmatory results

Selected checkpoint `checkpoint-1800`, 1,277 examples, scored once:

| run | `call_f1` | precision | recall | over_call | clarify | unsupp | parse_ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.6264 | 0.4618 | 0.9735 | 0.6238 | 0.1186 | 0.0177 | 0.9992 |
| M0-v1 (corpus v1) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9299 | 0.5673 | 1.0000 |
| M1-DPO-v1 | 0.1729 | 0.7857 | 0.0971 | 0.0146 | 0.9057 | 0.6336 | 1.0000 |
| M0 partial-v2 @1200 | 0.6278 | 0.7610 | 0.5342 | 0.0922 | 0.7951 | 0.6225 | 0.9992 |
| **M0 final-v2 @1800** | **0.7470** | 0.7350 | **0.7594** | 0.1505 | 0.7682 | 0.5430 | 1.0000 |

### Deltas against the two comparators that matter

**vs B0.** `call_f1` +0.121, precision +0.273, over-call −0.473, clarification +0.650,
unsupported +0.525. Recall −0.214. Every behavioural failure B0 exhibits is substantially
improved, at the cost of the raw recall that B0 obtains by over-calling.

**vs M0 partial-v2.** `call_f1` **+0.119**, recall **+0.225**, precision **−0.026**,
over-call **+0.058**, clarification **−0.027**, unsupported **−0.080**.

## 3. DEV results (selection evidence)

| step | `call_f1` | precision | recall | over_call | clarify | unsupp | macro | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0 | 0.6153 | 0.4502 | 0.9715 | 0.6525 | 0.0914 | 0.0107 | — | — |
| 600 | 0.7191 | 0.5935 | 0.9121 | 0.3436 | 0.5152 | 0.5962 | 0.6745 | no |
| 1200 | **0.7382** | 0.6709 | 0.8207 | 0.2214 | 0.6705 | 0.6081 | **0.6998** | no |
| **1800** | 0.7092 | 0.6990 | 0.7197 | 0.1705 | 0.7576 | 0.5499 | 0.6757 | **yes** |
| 2400 | 0.7043 | 0.7056 | 0.7031 | 0.1613 | 0.7678 | 0.5523 | 0.6744 | yes |

### The selection is worth understanding, not just recording

The rule rejected the two highest-`call_f1` checkpoints. Both reach their headline number by
over-calling above the 0.20 ceiling — 0.3436 and 0.2214 — which is the same degenerate trade B0
makes at 0.6525. A rule keyed on `call_f1` would have selected 1200 and reported the best number
in the table.

The rule instead selected **1800** on a tie-break: 1800 and 2400 differ by 0.0013 in macro score,
inside the 0.01 tolerance, so the earlier checkpoint wins. Worth noting explicitly: **the
tie-break decided this selection**, and a reader should treat 1800 and 2400 as effectively tied
rather than as a ranking.

The trajectory is also visible across the checkpoints and does not favour the later ones:
precision rises monotonically (0.5935 → 0.7056) while recall falls (0.9121 → 0.7031), which is the
same calibration trade the promotion policy objects to at the B0 end.

## 4. Confirmatory agreement with DEV

| | DEV (2,373) | confirmatory (1,277) |
|---|---:|---:|
| `call_f1` | 0.7092 | 0.7470 |
| precision | 0.6990 | 0.7350 |
| recall | 0.7197 | 0.7594 |
| over_call | 0.1705 | 0.1505 |
| clarification | 0.7576 | 0.7682 |
| unsupported | 0.5499 | 0.5430 |

The confirmatory result is consistent with DEV and slightly more favourable on four of six
dimensions. No material disagreement, so no discrepancy to report — and the checkpoint was not
switched after seeing it.

## 5. Promotion

**NOT PROMOTED.** Fails `regression.call_recall`: 0.7197 on DEV against B0's 0.9715, a −0.252
delta against a 0.10 allowance.

This is a genuine tension, not a defect in the model or the gate. B0's recall of 0.9715 comes
from calling a tool on 65% of examples whose gold answer is not a call. The policy caps over-call
at 0.20 **and** forbids a recall drop beyond 0.10; a calibrated model cannot satisfy both. Every
checkpoint on this run, and the historical partial-v2 checkpoint, fail the same constraint.

The gate is unchanged. Moving it now would be fitting the experiment to its own output. It is
recorded as a design finding for the next experiment.

## 6. What was not measured, and one tempting misreading

**Not measured at all:** tool-selection accuracy, argument validity, schema validity. The
evaluator does not compute them and `behavioral-heldout-v2` does not label them. They are
therefore absent from every result above, including the comparison against B0 — a table cell that
does not appear is not a zero.

**`no_call_accuracy` is structurally unmeasurable.** The evaluation population contains **zero
`ANSWER` examples**: 1,295 CALL, 1,060 CLARIFY, 1,295 UNSUPPORTED. That dimension is 0.0 for
every model, and its 0.40 floor was removed from the policy before this run precisely because it
rejected every candidate unconditionally. The macro score is the mean over the three classes the
partition can measure.

**The misreading to avoid:** the corpus changed in **three** ways at once. xLAM was added
(56,090 trainable call-prediction records where the partial corpus contributed none), ToolACE
went from 697 to 11,051 accepted records after a validator defect was fixed, and When2Call from
4,000 to 6,505 after an interrupted materialization was corrected. The planned minus-xLAM arms
remove xLAM together with the entire `CALL_PREDICTION` channel because no other source currently
provides that contract. They estimate that joint removal, not an xLAM-content-only effect. The
separate `CALL_PREDICTION`-only versus `COMPLETE_TRAJECTORY`-only design studies supervision
composition, but remains source-confounded until another valid call-prediction source exists.

**The second misreading to avoid:** none of the 56,090 call-prediction records contains a tool
result. Finding a `call_recall` improvement is consistent with call-prediction supervision, but
this evaluation does not isolate tool-result interpretation, multi-turn recovery or multi-step
planning, so no claim is made about them in either direction. The config stated that expectation
in the hypothesis before the run.

## 7. Reproducing this analysis

```bash
python scripts/freeze_eval_partition.py                     # DEV / confirmatory partition
python scripts/evaluate_sft_checkpoints.py m0_sft_canonical_v2_final \
    --partition reports/evaluation/behavioral-heldout-v2-partition.json --partition-side dev
python scripts/select_checkpoint.py --run runs/m0_sft_canonical_v2_final --side dev
```

Machine-readable: `runs/m0_sft_canonical_v2_final/eval/dev/selection--dev.json`,
`runs/m0_sft_canonical_v2_final/eval/dev/curve.json`,
`runs/m0_sft_canonical_v2_final/eval/confirmatory/curve.json`.
