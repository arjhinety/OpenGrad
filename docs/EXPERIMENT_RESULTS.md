# Experiment results

The complete intervention record for Study 001. The README carries only the primary progression;
this page is the full table, its caveats, and the published artifacts. The machine-checkable
per-record view (lifecycle status, `validity`, selected checkpoint, confirmatory score, report,
published artifact) is [`docs/EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md), regenerated from the
authoritative run artifacts.

> **Eight empirical results are recorded: the B0 baseline and seven post-training interventions.**
> Historical M0-v1 and M1-v1 were negative; partial-v2 recovered from a data defect; definitive
> M0-final-v2 restored balanced call behavior but did not pass its original promotion gate; both
> joint-removal ablations reduced recall; and M1-v2 moved within noise of M0 and was promoted.
> M1-v2 is promoted under a parent-relative gate (v4) introduced after M0 was
> evaluated; M0 also clears v4, and M1-v2 fails the v3 gate that rejected M0. The promotion
> reflects the gate change; the measured difference from M0 (+0.0078 call_f1, 7 of 453 calls,
> single seed) is within noise.

📊 **[Baseline-to-M0-final findings — charts and comparison](../reports/visual/index.html)** (also
[on the model card](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final/blob/main/findings.html)).
These charts cover the lineage through M0-final-v2; the complete record, including the later
ablations and M1-v2, is in the table below. Every figure is computed from the available per-example
predictions rather than copied from a report.

The table below lists interventions; the B0 baseline is recorded by the experiment store instead.

| Experiment | Model | Change | Capability Δ | Regression | Evidence status | Report |
| --- | --- | --- | --- | --- | --- | --- |
| [`qwen35_2b_m0_sft_full_v3`](../runs/qwen35_2b_m0_sft_full_v3/) | Qwen3.5-2B | M0 SFT on canonical corpus v1 | `call_f1` 0.6191 → **0.0000** | Collapsed: `call_recall` 0.9722 → 0.0000 | Negative; weights lost | [M0 report](../reports/M0_SFT_EXECUTION_REPORT.md) |
| [`qwen35_2b_m1_dpo_v1`](../runs/qwen35_2b_m1_dpo_v1/) | Qwen3.5-2B | DPO on When2Call preference pairs, applied directly to the base (no SFT parent) | `call_f1` 0.6191 → **0.1715** best | Over-calling fixed, tool calling destroyed | Negative; not reproducible | [M0 report §5](../reports/M0_SFT_EXECUTION_REPORT.md#5-dpo-was-blocked-now-executed-and-also-negative) |
| [`qwen35_2b_m0_sft_v2corpus`](../runs/qwen35_2b_m0_sft_v2corpus/) | Qwen3.5-2B | M0 SFT on partial corpus v2 | `call_f1` 0.6191 → **0.5995** best; macro recall 0.3621 → **0.6416** best (checkpoint 1200 of 4, chosen on this same full set; final checkpoint 2400: 0.5292 / 0.6173) | At checkpoint 1200: `call_recall` 0.9722 → 0.5050 (−0.467), `call_f1` −0.020 vs B0 | Partial recovery; not promoted | [M0 report §8](../reports/M0_SFT_EXECUTION_REPORT.md#8-corpus-v2-testing-the-root-cause-rather-than-asserting-it) |
| [`m0_sft_canonical_v2_final`](../runs/m0_sft_canonical_v2_final/) | Qwen3.5-2B | M0 SFT on **frozen Canonical-v2** | `call_f1` **0.7470**; recall 0.5342 → **0.7594** | Unsupported −0.080, clarification −0.027, precision −0.026, over-call +0.058 vs partial-v2 | Confirmatory; not promoted | [Execution](../reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md) · [Evaluation](../reports/M0_CANONICAL_V2_FINAL_EVALUATION.md) |
| [`m0_v2_final_minus_xlam_fixed_compute`](../runs/m0_v2_final_minus_xlam_fixed_compute/) | Qwen3.5-2B | **Joint xLAM + CALL_PREDICTION removal**, fixed compute (2,400 steps) | `call_f1` 0.7470 → **0.6030**; recall 0.7594 → **0.4879** | Precision +0.054, over-call −0.079 vs full corpus | Confirmatory; not promoted | [Execution](../reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md) · [Evaluation](../reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md) |
| [`m0_v2_final_minus_xlam_matched_exposure`](../runs/m0_v2_final_minus_xlam_matched_exposure/) | Qwen3.5-2B | **Joint xLAM + CALL_PREDICTION removal**, matched exposure (2,119 steps) | `call_f1` 0.7470 → **0.5557**; recall 0.7594 → **0.4238** | Precision +0.072, over-call −0.095 vs full corpus | Confirmatory; not promoted | [Execution](../reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md) · [Evaluation](../reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md) |
| [`m1_dpo_canonical_v2_final_v2`](../runs/m1_dpo_canonical_v2_final_v2/) | Qwen3.5-2B | M1 DPO calibration from selected M0-final-v2 | `call_f1` 0.7470 → **0.7548** (+0.0078, 351 vs 344 of 453 CALL examples); recall → **0.7748** | Over-call +0.0024; unsupported −0.0044; clarification −0.0027 vs M0 | **Confirmatory; promoted under v4** (fails v3 on all 4 DEV checkpoints; difference from M0 within noise) | [Execution](../reports/M1_DPO_EXECUTION_REPORT.md) · [Evaluation](../reports/M1_DPO_EVALUATION.md) |

## Caveats

**B0's `call_f1` comes from a degenerate policy.** It scores 0.6191 by calling a tool on 64.3% of
examples whose correct answer is not a call, recalling 97.2% of gold CALLs with 1.3%
unsupported-accuracy (full set, n=3,650; on the confirmatory partition B0 scores 0.6264, calling on
62.4% of no-call items). `call_f1` is the metric that flatters the baseline: partial-v2 was still
behind B0 on it (0.5995), while M0-final-v2 and M1-v2 are ahead on the confirmatory partition by
0.121 and 0.128. On balanced per-class (macro) recall the gap is far larger — B0 0.3699 against
M0-final-v2 0.6902 on the same partition. Do not read the leaderboard column as the finding.

**The table mixes evaluation sets.** B0, M0-v1, M1-v1 and partial-v2 are scored on the full
3,650-example held-out; M0-final-v2, both ablations and M1-v2 on the pre-registered 1,277-example
confirmatory partition. The partial-v2 recall of 0.5342 and the M0-final regression column compare
against partial-v2's confirmatory-partition row in the
[M0 evaluation report](../reports/M0_CANONICAL_V2_FINAL_EVALUATION.md); no per-partition artifact
for partial-v2 is committed, so those deltas cannot be recomputed from `runs/`. Every number here is
a single seed with no interval.

**The historical M1-v1 DPO result is not reproducible, and its best checkpoint no longer exists.** Its
steps 100 and 200 were deleted before upload, and a repeat run with config, data, seed, and
environment pinned did not reproduce the trajectory. The direction of that failure holds in both
runs; the claim that degradation is monotone from step 100 is withdrawn. M1-v1 is DPO applied
directly to the base (`parent_experiment_id: null`, reference = the initial policy), and its
published checkpoint 300 refuses 70.7% (933/1,319) of zero-shot GSM8K, so the general-capability
regression is not specific to SFT (see the [README](../README.md#the-tool-policy-gain-came-with-a-general-capability-regression)).
The parent-based M1-v2 is a separate identity, reported independently.

**The definitive M0 is not a promotion.** Under the repository's promotion policy at the time
(`tool_use_promotion_v3`) every checkpoint is `REJECT`, including the selected one. Checkpoints
1200, 1800 and 2400 fail `regression.call_recall`: they fall more than 0.10 below B0's DEV recall of
0.9715, which is itself a property of over-calling. Checkpoint 600 stays within that margin (−0.059)
and fails only the 0.20 over-call cap (0.3436). No checkpoint here met both constraints. The v3 gate
was left as written rather than adjusted after seeing the result. The later M1-v2 promotion used a
different, parent-relative gate (`tool_use_promotion_v4`, committed after M0's confirmatory results);
M0's own confirmatory metrics also clear v4, so the promotion reflects the gate change rather than a
measured improvement over M0.

**The minus-xLAM arms are a joint removal, not a pure xLAM ablation.** Canonical-v2 maps xLAM to
*every* `CALL_PREDICTION` record and the other three sources to `COMPLETE_TRAJECTORY`, so removing
xLAM also removes the corpus's entire call-prediction supervision channel. The two arms measure
**removing xLAM together with that channel**; they cannot separate source identity from supervision
type, so **no xLAM-specific causal claim** is made. Both arms lose far more recall than over-calling.
Exposure was not held fixed, so the result does not separate missing supervision from reduced
exposure: the "matched" arm logged 1.19× the reference's supervised tokens (6,743,788 vs 5,678,531),
and the selected checkpoints (1060 matched, 1200 fixed) had seen 3.36M / 3.80M supervised tokens
against 4.27M for the reference's selected checkpoint 1800. The separate `CALL_PREDICTION`-only vs `COMPLETE_TRAJECTORY`-only
design is prepared and unrun, and is source-confounded in the same way. See the
[ablation design](../reports/M0_V2_FINAL_ABLATION_DESIGN.md).

## Failed and invalid runs

Negative and failed work stays in the record. Beyond the table: the early CUDA-OOM attempts and the
scaffold run are annotated `SCAFFOLD_ONLY` or `FAILED`; the first M1 identity
(`m1_dpo_canonical_v2_final`) is preserved as a `FAILED` 119/120-step horizon-underflow run; and the
mock on-policy distillation attempt (`qwen35_2b_m2_distill`) is `INVALID` with `validity: MOCK_ONLY`.
Statuses, validity, and reasoning are in [`docs/EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md),
[`reports/M2_DECISION.md`](../reports/M2_DECISION.md), and
[`reports/M1_DPO_EXECUTION_REPORT.md`](../reports/M1_DPO_EXECUTION_REPORT.md).

## Published artifacts

| Artifact | Kind | Contents |
| --- | --- | --- |
| [`OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final`](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final) | model | the selected checkpoint (1800) + the findings page |
| [`OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV2`](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV2) | model | 4 checkpoints (600/1200/1800/2400) — intact |
| [`OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2`](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2) | model | promoted M1-v2; all 4 checkpoints (30/60/90/120), selected checkpoint 30 |
| [`OpenGrad-Qwen3.5-2B-M1-DPO`](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO) | historical M1-v1 artifact | checkpoint 300 only + the deleted checkpoints' predictions |
| [`OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-FixedCompute`](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-FixedCompute) | model | all 4 checkpoints (600/1200/1800/2400) of the joint-removal fixed-compute arm |
| [`OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-MatchedExposure`](https://huggingface.co/arjhinety/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-MatchedExposure) | model | all 4 checkpoints (530/1060/1590/2119) of the joint-removal matched-exposure arm |
| [`OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV1-evaluation`](https://huggingface.co/datasets/arjhinety/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV1-evaluation) | evaluation record | predictions and metrics for 5 of 6 checkpoints — **no weights exist** |

## The derived index

[`results/registry.jsonl`](../results/README.md) is a **derived index**, not a store: one summary row
per experiment, rebuilt from `runs/<experiment_id>/experiment.json`,
`runs/<experiment_id>/eval/`, and `runs/central_ledger.jsonl`. It can be deleted at any time —
`opengrad results rebuild-registry` regenerates it byte-for-byte — so the authoritative values stay
in the run artifacts and the index only makes them discoverable.
`opengrad results validate-registry` reports any divergence.

Do not confuse passing CPU tests with ML evidence: they validate infrastructure and fixtures, not
model quality. Conversely, the B0 numbers are a real measurement of a real model, but of a
*baseline* — and the interventions trained against it have now been measured in both directions.
