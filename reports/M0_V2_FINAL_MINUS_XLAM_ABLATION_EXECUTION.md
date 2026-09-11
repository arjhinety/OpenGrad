# M0-v2 joint xLAM + CALL_PREDICTION removal — execution report

**Status:** both arms `TRAINED / NOT PROMOTED`
**Date:** 2026-09-11
**Reference:** [`M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md`](M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md)
**Design (frozen pre-run):** [`M0_V2_FINAL_ABLATION_DESIGN.md`](M0_V2_FINAL_ABLATION_DESIGN.md)
**Evaluation:** [`M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md`](M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md)

## 1. What this experiment is, and is not

Canonical-v2 maps `xlam-function-calling-60k` to every `CALL_PREDICTION` record and maps the other
three sources to `COMPLETE_TRAJECTORY`. Removing xLAM therefore removes a source and the corpus's
entire call-prediction supervision channel **at once**. The treatment under test is:

> **removing xLAM together with the corpus's `CALL_PREDICTION` supervision channel.**

It is **not** a pure xLAM-content ablation, and no xLAM-specific causal claim can be based on it.
Source identity and supervision type are perfectly aligned in this corpus, so nothing here
separates them. A clean source-specific attribution needs another evidence-backed
`CALL_PREDICTION` source; a clean supervision-type attribution needs multiple source identities
within each contract. The separate `CALL_PREDICTION`-only vs `COMPLETE_TRAJECTORY`-only design
(`m0_v2_final_supervision_*`) is prepared and **not run**; it is source-confounded in the same way.

## 2. Arms and what each holds fixed

| Arm | Steps | Held fixed | Residual confound |
|---|---:|---|---|
| `m0_v2_final_minus_xlam_fixed_compute` | 2,400 | optimizer-step/GPU budget | reduced corpus sees ~1.53× reference exposure |
| `m0_v2_final_minus_xlam_matched_exposure` | 2,119 | measured supervised-token exposure | 11.7% fewer optimizer steps/FLOPs |

Both arms filter `xlam-function-calling-60k` from the identical frozen corpus and declare the
complete post-filter contract set explicitly (`COMPLETE_TRAJECTORY`, a non-empty selection).
Everything else — base model and revision, seed, optimizer, learning rate, scheduler, batch
geometry, sequence length, precision, renderer, loss masking, promotion policy, evaluation
partitions — is identical to the reference and to each other.

The matched horizon is computed, not chosen:

```text
matched_steps = 2400 x (29,630,369 / 33,565,721) = 2118.6 -> 2119
```

re-measured before launch and confirmed unchanged (`reports/data/canonical-v2-ablation-mass.json`).

## 3. Identity

| Field | Value |
|---|---|
| Parent corpus fingerprint | `8ced403b996e563d6e279aee7fdb346fc829fe5ff6af9daf8ef47c0a4007e161` |
| Filtered trainable records | 105,876 (Glaive 97,112 · ToolACE 2,259 · When2Call 6,505) |
| Filtered supervised tokens | 29,630,369 |
| Post-filter supervision | `COMPLETE_TRAJECTORY` 105,876 · `CALL_PREDICTION` 0 |
| Base model | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Filtered render cache | `…-toolpolicy-canonical-v2-final-filter-d9f0ecbf9b64` (identity verified: both filters present) |
| Launch commits | fixed `95a8225`, matched `0807fa3` (both arms from clean trees) |
| Readiness receipts | `reports/data/m0_v2_final_minus_xlam_fixed_compute-readiness.json`, `…matched_exposure-readiness.json` |

## 4. Pre-run readiness

Each arm was gated before launch and the receipt saved. Both: **PASS**, `ready_for_sft: true`,
`blocking_gates: []`, 23/23 gates passing.

The `supervision_composition` gate reads the arm's own filtered yield report and measures
`COMPLETE_TRAJECTORY` 105,876 with no `CALL_PREDICTION`, matching the declared selection. The
negative control — the same config mutated to select `CALL_PREDICTION` after xLAM is removed —
returns the blocking **`SUPERVISION_SELECTION_MISMATCH`** rather than training on a channel the
corpus no longer has. ToolACE remains a visible, non-blocking `ANOMALY` (yield 0.2044 against an
independent 0.3 floor); the floor was not fitted to the measurement.

A separate pre-run defect was found and fixed: a filtered run resolved to the reference's render
cache and its rebuild would have evicted the 161,966-row cache the reference evidence is measured
from. The filtered cache is now isolated by a filter key, and both arms correctly shared one
render.

## 5. Training

| | fixed-compute | matched-exposure |
|---|---:|---:|
| Steps | **2,400 / 2,400** | **2,119 / 2,119** |
| Loss | 1.3181 → 0.6940 (min 0.0185, mean 0.4617) | 1.3181 → 0.4198 |
| Supervised tokens seen | 7,683,301 | 6,743,788 |
| Examples seen | 27,591 | 24,418 |
| Wall clock | ~42 min | ~37 min |
| Throughput | ~3,045 sup tok/s | ~3,031 sup tok/s |
| Peak VRAM | 28.21 GiB | 28.21 GiB |
| Checkpoints | 600, 1200, 1800, 2400 | 530, 1060, 1590, 2119 |
| Interruptions / NaN / OOM | none | none |

Both arms trained stably for their full horizon from the same filtered cache; the only difference
is the horizon, which is the point.

## 6. Where the numbers are

- DEV curves and selections: `runs/<arm>/eval/dev/{curve.json,selection--dev.json}`
- Confirmatory (scored once each): `runs/<arm>/eval/confirmatory/curve.json`
- Machine-readable freeze: `reports/data/m0-v2-minus-xlam-freeze.json`

The evaluation and its interpretation are in the companion evaluation report.
