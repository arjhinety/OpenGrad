# Canonical-v2 ablation design

**Status:** `FROZEN_PRE_RUN`

## Executed next: joint xLAM-plus-CALL_PREDICTION removal

Canonical-v2 maps xLAM to every `CALL_PREDICTION` record and maps every other retained source to
`COMPLETE_TRAJECTORY`. Removing xLAM therefore removes both a source and the corpus's entire
call-prediction supervision channel. The comparison against full-v2 estimates that **joint
removal**; it is not a pure xLAM-content ablation and cannot support xLAM-specific causal
attribution.

Two arms expose the compute/exposure sensitivity and must be reported together:

| Arm | Steps | What is fixed | Additional limitation |
|---|---:|---|---|
| `m0_v2_final_minus_xlam_fixed_compute` | 2,400 | optimizer-step/GPU budget | retained corpus receives about 1.53× reference exposure |
| `m0_v2_final_minus_xlam_matched_exposure` | 2,119 | measured supervised-token exposure | spends 11.7% fewer optimizer steps/FLOPs |

Both filter `xlam-function-calling-60k` and explicitly select the complete post-filter contract
set: `COMPLETE_TRAJECTORY` only, 105,876 trainable records. `CALL_PREDICTION` must not remain as an
empty declaration. Readiness is required to reject that invalid combination with the blocking
error `SUPERVISION_SELECTION_MISMATCH`.

A single seed is a finding to record and replicate, not settled attribution. If the two arms
disagree, that disagreement is reported rather than resolved by preferring one.

## Separate, not executed: supervision-composition pair

The configs below remain a distinct future design:

- `m0_v2_final_supervision_call_prediction_only`
- `m0_v2_final_supervision_complete_trajectory_only`

They select one supervision contract through `supervision.include` and do not use the minus-xLAM
source filter. They study supervision composition operationally, but are still source-confounded in
the present corpus: call-prediction-only is xLAM-only at the trainable boundary, and
complete-trajectory-only contains only the other sources. The complete-only trainable sample set is
therefore equivalent to the fixed-compute joint-removal arm today. Neither config is authorized for
launch in this work.

A clean source-specific xLAM attribution requires at least one additional evidence-backed
`CALL_PREDICTION` source so source identity can vary while supervision type is held fixed. A clean
supervision-type attribution likewise requires multiple source identities within each compared
contract. Until then, reports must use joint-removal or source-confounded language.
