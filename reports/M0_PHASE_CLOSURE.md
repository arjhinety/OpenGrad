# M0 / SFT phase closure

**Status:** `CLOSED / FROZEN`
**Date:** 2026-09-11

This document closes the M0 lineage before M1 begins. The historical experiment identities, corpus
fingerprints, evaluation partitions, policy versions, decisions, and retained checkpoint evidence
are not reopened by later DPO or distillation work.

## Frozen lineage

| Identity | Dataset / treatment | Status | Evidence |
|---|---|---|---|
| B0 | immutable Qwen3.5-2B baseline | `EVALUATED` | `reports/baselines/qwen35_2b_baseline/` |
| M0-v1 | Canonical-v1 SFT | historical negative; weights lost | `reports/M0_SFT_EXECUTION_REPORT.md` |
| M1-v1 historical DPO | When2Call preference DPO | historical negative; intermediate weights lost | `docs/INCIDENT_LOG.md` (`INC-0001`) |
| M0 partial-v2 | partial Canonical-v2 SFT | historical partial result | `runs/qwen35_2b_m0_sft_v2corpus/` |
| M0 final-v2 | full frozen Canonical-v2 SFT | selected checkpoint 1800; `REJECTED` on promotion | `reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md` |
| minus-xLAM fixed | remove xLAM + `CALL_PREDICTION`, 2,400 steps | `REJECTED` | `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md` |
| minus-xLAM matched | remove xLAM + `CALL_PREDICTION`, 2,119 steps | `REJECTED` | same paired report |

All later phases must reference these identities, not create replacement interpretations of them.

## M0 final finding

The selected full-corpus checkpoint is `m0_sft_canonical_v2_final/checkpoint-1800`.
On the pre-registered internal confirmatory partition (1,277 examples, scored once), it reached:

- `call_f1`: **0.7470**
- call precision: **0.7350**
- call recall: **0.7594**
- over-call rate: **0.1505**
- clarification accuracy: **0.7682**
- unsupported accuracy: **0.5430**

The full final-v2 corpus strongly recovered tool-call recall relative to partial-v2. It was not
promoted because the frozen promotion policy rejected its call-recall regression against B0. No
M0 threshold or policy was changed after seeing this result. A parent-relative gate
(`tool_use_promotion_v4`) was introduced afterwards for M1; M0 checkpoint 1800's confirmatory
metrics also clear every v4 floor, and M1-v2 fails the v3 gate that rejected M0
(see [`M1_DPO_EVALUATION.md`](M1_DPO_EVALUATION.md)).

## Paired minus-xLAM finding

Canonical-v2 has one current `CALL_PREDICTION` source: xLAM. Removing xLAM therefore also removes
the entire call-prediction supervision channel. The minus-xLAM arms are consequently an experiment
on **removing xLAM together with the corpus's `CALL_PREDICTION` channel**, not a pure xLAM-content
ablation.

| Arm | Selected | `call_f1` | Precision | Recall | Over-call |
|---|---:|---:|---:|---:|---:|
| Full final-v2 reference | 1800 | 0.7470 | 0.7350 | 0.7594 | 0.1505 |
| Minus xLAM, fixed compute | 1200 | 0.6030 | 0.7893 | 0.4879 | 0.0716 |
| Minus xLAM, matched exposure | 1060 | 0.5557 | 0.8067 | 0.4238 | 0.0558 |

Both ablations substantially lose recall and become conservative. The fixed-compute and
matched-exposure arms support the importance of the joint xLAM/call-prediction channel. They do
**not** independently identify xLAM source identity versus supervision type. A valid xLAM-specific
causal attribution requires another evidence-backed `CALL_PREDICTION` source.

The separate `CALL_PREDICTION`-only versus `COMPLETE_TRAJECTORY`-only supervision-composition
design remains prepared but unlaunched. With the present source mapping it is also source-confounded
and must not be reported as a clean supervision-type causal estimate.

## Frozen data and evaluation identities

- Full Canonical-v2 final fingerprint: `8ced403b…`, 173,237 canonical / 161,966 trainable records.
- Minus-xLAM filtered trainable set: 105,876 `COMPLETE_TRAJECTORY` records and 0
  `CALL_PREDICTION` records.
- Matched-exposure calculation: `2400 × (29,630,369 / 33,565,721) = 2118.6 → 2119`. This matches
  corpus-level supervised tokens, not logged exposure: the training logs record 6,743,788
  supervised tokens for the matched arm against 5,678,531 for the reference (1.19×), and the
  selected checkpoints saw 3.36M (matched @1060) and 3.80M (fixed @1200) against the reference's
  4.27M (@1800). Exposure is therefore not held fixed between the selected checkpoints.
- DEV partition: 2,373 examples, fingerprint `88a56821…`.
- Confirmatory partition: 1,277 examples, fingerprint `d6d1e394…`.
- Confirmatory evidence is **pre-registered internal evaluation**, not an untouched external test.
- The selection policy and promotion policy used by M0 are frozen and remain historical.

The paired freeze manifest is `reports/data/m0-v2-minus-xlam-freeze.json`; the M0-final freeze is
`reports/data/m0-final-freeze.json`. Both verify their recorded hashes and all eight minus-xLAM
weight files remain locally retained and were uploaded before any scratch staging data was removed.

## Registry closure

`results/registry.jsonl` remains derived. The known B0 legacy layout is now explicit: its four
load-bearing artifacts live under `reports/baselines/qwen35_2b_baseline/`, so the registry does not
invent per-checkpoint metrics and does not report that known layout as a missing-evaluation error.
The remaining integrity findings are genuine historical gaps: the lost M0-v1 step-400 metrics and
the early `qwen35_2b_m0_sft` record with no per-checkpoint metrics. They are preserved, not repaired
with synthetic values.

## Closure checks

- `ruff check src/ tests/ scripts/`: PASS
- `mypy src/`: PASS, 143 source files
- `pytest tests/`: PASS
- M0-final freeze verification: PASS
- paired minus-xLAM freeze verification: PASS
- all eight checkpoint weights: retained locally and verified remotely
- repository: clean and synchronized before M1 design begins

M1 may reference the selected M0 final-v2 checkpoint, but it may not alter any item above.
