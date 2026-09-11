# M0-v2 joint xLAM + CALL_PREDICTION removal — evaluation report

**Status:** evaluation complete. Fixed-compute selected `checkpoint-1200`; matched-exposure selected
`checkpoint-1060`. Both `NOT PROMOTED`.
**Companion:** [`M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md`](M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md)

The treatment is **removing xLAM together with the corpus's `CALL_PREDICTION` supervision channel.**
Source identity and supervision type are perfectly aligned here, so this comparison estimates their
joint removal and supports **no xLAM-specific causal claim**.

## 1. Evidence kinds

| Evidence | Set | Used for |
|---|---|---|
| Checkpoint selection | DEV, 2,373 examples, fingerprint `88a56821…` | Choosing each arm's checkpoint |
| Confirmatory | 1,277 examples, fingerprint `d6d1e394…` | Measuring each chosen checkpoint once |
| Reference | the same partitions, from the completed M0-final run | The comparison |

The confirmatory partition is pre-registered **internal** evidence, not an untouched external
benchmark.

## 2. Confirmatory results — the headline

Each arm's selected checkpoint on the 1,277-example confirmatory partition, scored once:

| run | `call_f1` | precision | recall | over_call | clarify | unsupp | parse_ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.6191 | 0.4542 | 0.9722 | 0.6425 | 0.1009 | 0.0131 | — |
| **reference (full corpus) @1800** | **0.7470** | 0.7350 | **0.7594** | 0.1505 | 0.7682 | 0.5430 | 1.0000 |
| minus-xLAM **fixed-compute** @1200 | 0.6030 | 0.7893 | 0.4879 | 0.0716 | 0.8059 | 0.6026 | 1.0000 |
| minus-xLAM **matched-exposure** @1060 | 0.5557 | 0.8067 | 0.4238 | 0.0558 | 0.8194 | 0.6203 | 1.0000 |

### Deltas against the reference

| | `call_f1` | precision | recall | over_call | clarify | unsupp |
|---|---:|---:|---:|---:|---:|---:|
| fixed − reference | **−0.144** | +0.054 | **−0.272** | −0.079 | +0.038 | +0.060 |
| matched − reference | **−0.191** | +0.072 | **−0.336** | −0.105 | +0.051 | +0.077 |

Both arms move the same way: recall collapses, `call_f1` falls well below both the reference and
B0, precision rises, and over-calling falls to a small fraction of both.

## 3. This is the direction the missing channel predicts

None of the 56,090 `CALL_PREDICTION` records the reference trained on contains a tool result; the
supervised target is the call itself. Removing them removes the signal that teaches *whether to
call at all*, while the remaining `COMPLETE_TRAJECTORY` records still supervise full trajectories.
The measured outcome is exactly that: the models become conservative — high precision, low
over-call — but stop recalling the calls they should make.

This is the **expected and informative** result stated in advance in the reference config's
hypothesis, and it is consistent with the reference's own gain: the full corpus reaches
recall 0.7594 where the partial-v2 corpus, which also lacked the call-prediction channel, reached
0.5342.

**But it does not identify xLAM.** The channel and the source were removed together. Saying "xLAM's
call-prediction supervision caused the recall recovery" would require separating them, which this
corpus cannot do.

## 4. The two arms agree, and the exposure confound does not explain the result

| | fixed-compute | matched-exposure |
|---|---:|---:|
| Steps | 2,400 | 2,119 |
| Reference exposure over retained corpus | ~1.53× | ~1.0× |
| `call_f1` | 0.6030 | 0.5557 |
| recall | 0.4879 | 0.4238 |

The arm that trains **more** (2,400 steps, more exposure) is the arm that does better, which is the
opposite of what an exposure artifact would produce: giving the reduced corpus *more* passes does
not restore recall. The recall loss therefore tracks the missing supervision, not the training
budget. The two arms disagree in magnitude (the matched arm is worse) and that disagreement is
reported rather than resolved by preferring one.

## 5. DEV curves (selection evidence)

**Fixed-compute** (2,373 examples):

| step | `call_f1` | precision | recall | over_call | macro | eligible |
|---|---:|---:|---:|---:|---:|---|
| 600 | 0.5926 | 0.7199 | 0.5036 | 0.1078 | 0.5832 | yes |
| **1200** | **0.5635** | 0.7669 | 0.4454 | 0.0745 | **0.6270** | **selected** |
| 1800 | 0.4773 | 0.8161 | 0.3373 | 0.0418 | 0.6036 | yes |
| 2400 | 0.4845 | 0.8169 | 0.3444 | 0.0425 | 0.6061 | yes |

**Matched-exposure** (2,373 examples):

| step | `call_f1` | precision | recall | over_call | macro | eligible |
|---|---:|---:|---:|---:|---:|---|
| 530 | 0.3663 | 0.8285 | 0.2352 | 0.0268 | 0.5583 | yes |
| **1060** | 0.5186 | 0.7804 | 0.3884 | 0.0601 | **0.6213** | **selected (tie-break)** |
| 1590 | 0.5261 | 0.7905 | 0.3943 | 0.0575 | 0.6155 | yes |
| 2119 | 0.5161 | 0.7985 | 0.3812 | 0.0529 | 0.6140 | yes |

The frozen rule (`docs/evaluation/CHECKPOINT_SELECTION_RULE.md`) was applied unchanged: eligibility
on parse validity ≥ 0.99 and over-call ≤ 0.20, ranking by macro, 0.01 tie-break to the earlier
checkpoint. Every checkpoint in both arms was eligible — none over-called — and fixed-compute had
no tie (1200 won outright); matched-exposure's 1060 won a tie-break. The rule was not adjusted
after seeing these numbers.

Across steps both arms sharpen precision and suppress over-calling while recall keeps falling —
the same calibration trajectory, run further than the reference because the counter-pressure that
taught calling is gone.

## 6. Promotion

**NOT PROMOTED — all eight checkpoints, both arms.** The failing check is
`regression.call_recall` against B0's 0.9715, the same structural constraint every earlier
checkpoint fails. The gate is left **exactly as written**; it was not relaxed to accommodate a
result that makes the omission of the channel look more decisive.

The verdicts and their per-checkpoint reasons are recorded authoritatively in
`runs/checkpoint_registry.json` and `runs/central_ledger.jsonl`.

## 7. What was not measured

Tool-selection accuracy, argument validity, and schema validity are not computed by this evaluator,
so their absence is not a zero. `no_call_accuracy` remains structurally unmeasurable (the
evaluation population has no `ANSWER` examples). A single seed is not variance: a small delta
between arms is a finding to replicate, not a settled attribution.

## 8. Reproducing this analysis

```bash
python scripts/evaluate_sft_checkpoints.py m0_v2_final_minus_xlam_fixed_compute \
    --partition reports/evaluation/behavioral-heldout-v2-partition.json --partition-side dev
python scripts/select_checkpoint.py --run runs/m0_v2_final_minus_xlam_fixed_compute --side dev
python scripts/evaluate_sft_checkpoints.py m0_v2_final_minus_xlam_fixed_compute \
    --partition reports/evaluation/behavioral-heldout-v2-partition.json \
    --partition-side confirmatory --checkpoint-ids 1200
# and the same three for m0_v2_final_minus_xlam_matched_exposure (selected 1060)
```