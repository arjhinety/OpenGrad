# Experiment status

Generated view over the authoritative artifacts — `runs/<id>/experiment.json`, the per-run
and central ledgers, `runs/checkpoint_registry.json`, the per-checkpoint evaluation metrics,
and `reports/releases/*.json`. It is a view, not a source of truth: no field is invented, an
absent value renders as an em dash, and the file is regenerated deterministically by

```text
python scripts/reporting/generate_experiment_status.py
```

A byte-identical regeneration is required by `tests/results/test_state_consistency.py`, so a
stale table fails the suite instead of drifting quietly. Prose lives in the
[README](../README.md#results); the numeric claims there and the counting convention below
are checked against this table and the registry.

## Counting convention

- **Executed post-training interventions with real model evidence: 7.** The B0 baseline is recorded separately and is not an intervention. An arm that only failed to launch (resource, registration, or horizon failure) is execution history, not an intervention.
- **Promoted models: 1.** A promotion only counts when the record is valid, so
  the scaffold-era promotion is excluded by construction.
- **Invalid/mock records: 1** — preserved as evidence, never counted as trained.
- **Scaffold-era records: 2** — identity and lifecycle history only; no real
  model evidence.
- Negative and rejected results stay in the table: the record documents failures as well as
  successes.

`validity` is an annotation on the authoritative experiment record (`MOCK_ONLY`,
`SCAFFOLD_ONLY`, `NON_REPRODUCIBLE_REPEAT`), appended to the ledger rather than overwriting
history. `status` is the lifecycle state, including when that state is historical.

## Records

| Experiment | Method | Datasets | Status | Validity | Promotion | Confirmatory `call_f1` / `call_recall` | Reports | Published |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`qwen35_2b_m0_sft`](../runs/qwen35_2b_m0_sft/) | sft | canonical_v1 | PROMOTED | SCAFFOLD_ONLY | PROMOTED | — | — | — |
| [`qwen35_2b_m1_dpo`](../runs/qwen35_2b_m1_dpo/) | dpo | when2call_pref_v1 | TRAINED | SCAFFOLD_ONLY | — | — | — | — |
| [`qwen35_2b_m2_distill`](../runs/qwen35_2b_m2_distill/) | on_policy_distillation | onpolicy_prompts_v1 | INVALID | MOCK_ONLY | — | — | [`reports/M2_DECISION.md`](../reports/M2_DECISION.md) | — |
| [`tool_calling/qwen35_2b/baseline`](../runs/tool_calling/qwen35_2b/baseline/) | evaluation | reports/evaluation/behavioral-heldout-v2.manifest.json | EVALUATED | — | — | 0.6191 / 0.9722 | [`reports/baselines/qwen35_2b_baseline/RESULT.md`](../reports/baselines/qwen35_2b_baseline/RESULT.md) | — |
| [`qwen35_2b_m0_sft_full`](../runs/qwen35_2b_m0_sft_full/) | sft | canonical_v1 | FAILED | — | — | — | — | — |
| [`qwen35_2b_m0_sft_full_v2`](../runs/qwen35_2b_m0_sft_full_v2/) | sft | canonical_v1 | FAILED | — | — | — | — | — |
| [`qwen35_2b_m0_sft_micro`](../runs/qwen35_2b_m0_sft_micro/) | sft | canonical_v1 | FAILED | — | — | — | — | — |
| [`qwen35_2b_m0_sft_full_v3`](../runs/qwen35_2b_m0_sft_full_v3/) | sft | canonical_v1 | REJECTED | — | — | — | [`reports/M0_SFT_EXECUTION_REPORT.md`](../reports/M0_SFT_EXECUTION_REPORT.md) | [`arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV1-evaluation`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV1-evaluation) |
| [`qwen35_2b_m1_dpo_v1`](../runs/qwen35_2b_m1_dpo_v1/) | dpo | when2call_pref_v1 | REJECTED | — | — | — | [`reports/M0_SFT_EXECUTION_REPORT.md`](../reports/M0_SFT_EXECUTION_REPORT.md) | [`arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO) |
| [`qwen35_2b_m0_sft_v2corpus`](../runs/qwen35_2b_m0_sft_v2corpus/) | sft | canonical_v2 | TRAINED | — | — | — | [`reports/M0_SFT_EXECUTION_REPORT.md`](../reports/M0_SFT_EXECUTION_REPORT.md) | — |
| [`qwen35_2b_m1_dpo_v1_restore`](../runs/qwen35_2b_m1_dpo_v1_restore/) | dpo | when2call_pref_v1 | TRAINED | NON_REPRODUCIBLE_REPEAT | — | — | — | — |
| [`m0_sft_canonical_v2_final`](../runs/m0_sft_canonical_v2_final/) | sft | canonical_v2_final | REJECTED | — | — | 0.7470 / 0.7594 | [`reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md`](../reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md) · [`reports/M0_CANONICAL_V2_FINAL_EVALUATION.md`](../reports/M0_CANONICAL_V2_FINAL_EVALUATION.md) | [`arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final) |
| [`m0_v2_final_minus_xlam_fixed_compute`](../runs/m0_v2_final_minus_xlam_fixed_compute/) | sft | canonical_v2_final | REJECTED | — | — | 0.6030 / 0.4879 | [`reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md`](../reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md) · [`reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md`](../reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md) | [`arrochi112/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-FixedCompute`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-FixedCompute) |
| [`m0_v2_final_minus_xlam_matched_exposure`](../runs/m0_v2_final_minus_xlam_matched_exposure/) | sft | canonical_v2_final | REJECTED | — | — | 0.5557 / 0.4238 | [`reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md`](../reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md) · [`reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md`](../reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md) | [`arrochi112/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-MatchedExposure`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-MatchedExposure) |
| [`m1_dpo_canonical_v2_final`](../runs/m1_dpo_canonical_v2_final/) | dpo | m1_calibration_preference_pairs_v1 | FAILED | — | — | — | [`reports/M1_DPO_EXECUTION_REPORT.md`](../reports/M1_DPO_EXECUTION_REPORT.md) | — |
| [`m1_dpo_canonical_v2_final_v2`](../runs/m1_dpo_canonical_v2_final_v2/) | dpo | m1_calibration_preference_pairs_v1 | PROMOTED | — | PROMOTED | 0.7548 / 0.7748 | [`reports/M1_DPO_EXECUTION_REPORT.md`](../reports/M1_DPO_EXECUTION_REPORT.md) · [`reports/M1_DPO_EVALUATION.md`](../reports/M1_DPO_EVALUATION.md) | [`arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2) |

Confirmatory scores are the pre-registered partition values quoted in the reports and
the model cards; the baseline row shows its full-set score, and a row with no
confirmatory artifact shows an em dash rather than a convenience number. Rows are
ordered by launch timestamp so the chronology is stable.
