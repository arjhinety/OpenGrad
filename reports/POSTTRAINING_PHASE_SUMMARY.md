# Post-training phase summary

**Status:** M0 closed; M1 completed and promoted; M2 not run because its current path is mock-only and not justified by the M1 evidence.

M1-v2 is promoted under a parent-relative gate (v4) introduced after M0 was evaluated; M0 also
clears v4, and M1-v2 fails the v3 gate that rejected M0. The promotion reflects the gate change;
the measured difference from M0 (+0.0078 call_f1, 7 of 453 calls, single seed) is within noise.

## Lineage

```text
B0
  ↓
M0 SFT on Canonical-v2 final — selected 1800, not promoted under v3 (clears v4)
  ↓
M1 DPO on selected M0-final-v2 — selected 30, promoted under tool_use_promotion_v4 (REJECT under v3)
  ↓
M2 on-policy distillation — not run / not justified
```

## Confirmatory metrics

| Model | `call_f1` | Precision | Recall | Over-call | Clarification | Unsupported |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.6264 | 0.4618 | 0.9735 | 0.6238 | 0.1186 | 0.0177 |
| M0 final-v2 @1800 | 0.7470 | 0.7350 | 0.7594 | 0.1505 | 0.7682 | 0.5430 |
| M1 DPO-v2 @30 | **0.7548** | 0.7358 | **0.7748** | 0.1529 | 0.7655 | 0.5386 |
| M0 minus-xLAM fixed @1200 | 0.6030 | 0.7893 | 0.4879 | 0.0716 | 0.8059 | 0.6026 |
| M0 minus-xLAM matched @1060 | 0.5557 | 0.8067 | 0.4238 | 0.0558 | 0.8194 | 0.6203 |

All confirmatory measurements are pre-registered internal evidence, not untouched external tests.
The B0 row is recomputed from B0's predictions on the 1,277-example confirmatory partition; on the
full 3,650 set B0 is 0.6191 / 0.4542 / 0.9722 / 0.6425 / 0.1009 / 0.0131.

## Conclusions

- M0 final-v2 strongly recovered tool-call recall relative to partial-v2.
- Removing xLAM also removes Canonical-v2's only current `CALL_PREDICTION` channel; both ablation
  arms substantially lose recall.
- The paired fixed-compute and matched-exposure results support the importance of that joint
  xLAM/call-prediction channel, but cannot identify source identity versus supervision type.
- M1 DPO starting from the selected M0 checkpoint preserved the calibrated frontier without
  collapsing into the conservative minus-xLAM policy. Its call F1/recall gain over M0 (+0.0078,
  7 of 453 calls, single seed) is within noise, and over-call rose slightly (0.1505 → 0.1529).
- M0 and M1-v2 both carry a general-capability regression against Base (GSM8K 0-shot 67.4% → 0.0%,
  IFEval prompt-strict 67.8% → 45.1% / 45.8%, MMLU-Pro 49.0% → 37.0%;
  `results/final_campaign_verdict.json`). It is associated with tool-policy post-training on
  When2Call-derived data and also appears after DPO applied directly to the base (M1-v1), so it is
  not specific to SFT. Causation is not established: one lineage, one seed, no replicate.
- Tool selection, argument validity, schema validity and `no_call_accuracy` remain unmeasured/NA;
  they are not silently treated as satisfied.
- M2 was not run: the current implementation is mock infrastructure without a live teacher/student
  update path or evidence-backed teacher/data contract. A future M2 requires a separate design
  freeze and a real implementation.

## Evidence

- M0 closure: `reports/M0_PHASE_CLOSURE.md`
- M1 execution: `reports/M1_DPO_EXECUTION_REPORT.md`
- M1 evaluation: `reports/M1_DPO_EVALUATION.md`
- M2 decision: `reports/M2_DECISION.md`
- M1 readiness: `reports/data/m1-dpo-canonical-v2-final-v2-readiness.json`
- M1 promotion: `reports/data/m1-dpo-canonical-v2-final-v2-promotion.json`
