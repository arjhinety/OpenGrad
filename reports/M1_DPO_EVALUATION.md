# M1 DPO evaluation report

**Status:** `PROMOTED` under prospective `tool_use_promotion_v4`
**Experiment:** `m1_dpo_canonical_v2_final_v2`
**Selected checkpoint:** `dpo-checkpoint-30`
**Parent:** M0-final-v2 selected checkpoint 1800

M1-v2 is promoted under a parent-relative gate (v4) introduced after M0 was evaluated; M0 also
clears v4, and M1-v2 fails the v3 gate that rejected M0. The promotion reflects the gate change;
the measured difference from M0 (+0.0078 call_f1, 7 of 453 calls, single seed) is within noise.

## Evidence separation

- DEV: 2,373 examples, fingerprint `88a56821…`, all four checkpoints scored for selection.
- Confirmatory: 1,277 examples, fingerprint `d6d1e394…`, selected checkpoint 30 scored once.
- Confirmatory evidence is pre-registered internal evidence, not an untouched external benchmark.

The frozen selector chose checkpoint 30. All four checkpoints were within the 0.01 macro tolerance
(0.6736–0.6768), so the earlier-checkpoint tie-break was applied, but checkpoint 30 was also the
outright macro maximum (0.6768), so the tie-break did not change the choice. Confirmatory evidence
was not consulted for that choice.

## Results

| run | `call_f1` | precision | recall | over_call | clarify | unsupp | parse_ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.6264 | 0.4618 | 0.9735 | 0.6238 | 0.1186 | 0.0177 | 0.9992 |
| M0-final @1800 | 0.7470 | 0.7350 | 0.7594 | 0.1505 | 0.7682 | 0.5430 | 1.0000 |
| **M1-v2 @30** | **0.7548** | 0.7358 | **0.7748** | 0.1529 | 0.7655 | 0.5386 | 1.0000 |

All rows are the 1,277-example confirmatory partition; the B0 row is recomputed from B0's committed
predictions on that partition (B0 on the full 3,650 set is 0.6191 / 0.4542 / 0.9722 / 0.6425 /
0.1009 / 0.0131).

Deltas versus the selected M0 parent:

- `call_f1`: +0.0078 (351 vs 344 correct of 453 CALL examples)
- precision: +0.0008
- recall: +0.0155
- over-call: +0.0024
- clarification: −0.0027
- unsupported: −0.0044

M1 preserves the M0 calibrated frontier. The call F1 and recall gains are 7 extra correct calls out
of 453 on a single seed with no interval, which is within noise. Over-calling is slightly worse, not
better, so the result is calibration retention, not a frontier movement.

## Prospective promotion policy

`tool_use_promotion_v4` was frozen before M1 checkpoint metrics were inspected, but after M0's
confirmatory results were known: it was committed on 2026-09-11 at 17:10 UTC (`92ca2b9`), after the
M0 confirmatory results at 08:58 UTC (`26234c2`). It does not compare recall to B0's degenerate
always-call recall. It requires a balanced frontier against the selected M0 parent:

- precision ≥ 0.65;
- recall ≥ 0.60;
- measurable macro recall ≥ 0.60;
- over-call ≤ 0.20;
- clarification ≥ 0.60;
- unsupported ≥ 0.40;
- parse-valid ≥ 0.99;
- no excessive regression against M0 on measured precision, recall, clarification or unsupported.

M1-v2 passes every measurable check and is authoritatively **PROMOTED**. `no_call_accuracy` is NA
because the partition contains zero ANSWER examples. Tool-selection accuracy, argument validity and
schema validity are not computed by the evaluator and are not treated as satisfied.

The gate change matters to that verdict. M0 checkpoint 1800's confirmatory metrics also clear every
v4 floor, and under `tool_use_promotion_v3`, the B0-relative gate that rejected M0, all four M1-v2
DEV checkpoints are REJECT on `regression.call_recall`
(`runs/m1_dpo_canonical_v2_final_v2/eval/dev/selection--dev.json`).

## M1 conclusion

M1 DPO did not recreate the conservative minus-xLAM behavior. It preserved the M0 tool-routing
behavior and made a small movement in recall/F1 while retaining precision and clarification. The
small magnitude, single seed and unmeasured tool/argument dimensions limit the strength of the claim.
It also inherits M0's general-capability regression (results/final_campaign_verdict.json): GSM8K
0-shot 67.4% → 0.0% (100% refusal), 8-shot 70.4% → 55.5%, IFEval prompt-strict 67.8% → 45.8%,
MMLU-Pro 49.0% → 37.0%.

M2 is not automatically justified by a generic desire to use GPU time. The current M2 implementation
is mock infrastructure, lacks a live teacher/student update path, and has no authoritative teacher
identity or valid prompt-state dataset. It should only proceed after a real teacher-guided design
and readiness contract answers a distinct scientific question.

Machine-readable evidence:

- `runs/m1_dpo_canonical_v2_final_v2/eval/dev/curve.json`
- `runs/m1_dpo_canonical_v2_final_v2/eval/dev/selection--dev.json`
- `runs/m1_dpo_canonical_v2_final_v2/eval/confirmatory/curve.json`
- `reports/data/m1-dpo-canonical-v2-final-v2-promotion.json`
