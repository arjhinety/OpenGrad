# 40 — `study_002_prereg_v8` (ADOPTED 2026-09-24)

> **Status: ADOPTED 2026-09-24, items A–D as drafted.** It was written at the study owner's request after
> the 2026-09-24 repository review found two rules the preregistration requires but never quantifies, and
> the owner adopted it the same day. It is recorded in [03](03-PREREGISTRATION.md) (at the end) and
> `reports/ERRATA.md` §24. `ADOPTED_PARAMETERS` in `src/opengrad/verification/study_002_gate.py` carries
> A and B, and the gate wraps `tool_use_promotion_v6` at contract 3. The text below is kept as drafted.

## Why an amendment, and why now

[03](03-PREREGISTRATION.md) permits amendments for defects — "a gate that cannot fail … an unmeasurable
dimension" — and forbids moving a threshold after a result is visible. Both conditions are met here:

- **Two rules have no value.** [11](11-THRESHOLDS.md) check 12 requires truncation "imbalance within the
  declared factor", and no document in this study declares a factor. [11](11-THRESHOLDS.md)'s register makes
  `min_refusal_correctness` adjudicable "if `P-UNANS` is large enough", and no document states how large.
  A gate cannot implement either, so as written check 12 and check 4 could only pass vacuously or block.
- **Nothing has been scored.** No arm has launched, no four-mode evaluation bundle exists, and `P-UNANS` is
  not built. No number these rules govern has been seen.

## Proposed items

### A. The truncation imbalance factor (check 12)

- **Unit.** A truncation rate is reported per **stage** (the prompt mode a sentinel runs in: `0-shot`,
  `8-shot`, `elicit`, `5-shot`, [08](08-SENTINEL-SPEC.md)) per **arm**, as [08](08-SENTINEL-SPEC.md) rule 2
  requires.
- **Rule.** Within a stage, two arms are **imbalanced** when their truncation rates differ by more than
  **2 percentage points** *and* the larger is more than **2.0×** the smaller (a zero against a non-zero rate
  counts as exceeding the ratio). An imbalanced stage returns `INVALID_COMPARISON`, and its comparisons are
  reported only as a truncation-adversarial interval (08 rule 2), never as a point.
- **Basis.** The one measured case: MMLU-Pro at 2,048 tokens truncated Base on 21.8% and M0 on 6.3% of
  12,032 items — a **3.46×** ratio and a **15.5pp** gap, which [08](08-SENTINEL-SPEC.md) already treats as
  invalidating a point comparison. 2.0× is a judgement, not a derivation: it flags half that ratio. The 2pp
  absolute gap exists so that two near-zero rates (0.0% against 0.4%) are not called imbalanced by the
  ratio alone. **Alternatives the owner may prefer:** 1.5× (stricter) or 3.0× (flags only cases as bad as
  the MMLU-Pro one).
- **Code.** `PreregParameters(truncation_max_ratio=2.0, truncation_min_gap=0.02)`.

### B. The minimum `P-UNANS` size (check 4)

- **Rule.** `P-UNANS` has **n ≥ 385**. Below it, `refusal_correctness` is `UNDER_POWERED`, check 4 fails, and
  H6 is not adjudicated.
- **Basis.** The study's own resolvability rule ([06](06-SPLIT-SPEC.md) §C2): a row resolves the
  differences its worst-case resolvable margin allows, and the study makes no claim on a row that cannot
  resolve 10pp. The smallest n with a resolvable margin at or under 10pp is **385** (9.989pp);
  n = 384 gives 10.002pp. (06 writes "≈384"; the exact figure is 385.)
- **Code.** `PreregParameters(p_unans_min_n=385)`.

### C. The gate wraps `tool_use_promotion_v6`

- **Change.** `study_002_gate_v1` reads its behavioural checks and its decision from `tool_use_promotion_v6`
  instead of v5.
- **Why.** v5 has three ways to promote without measuring (`reports/ERRATA.md` §19): with no confusion
  matrix it treats every mode as measured; with `refusal_correctness` or the baseline `answer_rate` absent
  it skips the safety floor and the drop bound; and at an exact threshold it decides by float error
  (`0.90 − 0.60` fails `≤ 0.30`). v6 closes all three and changes **no threshold**. The gate's contract 2
  already blocks the first two on its own; the third needs v6.
- **Code.** `PromotionPolicyV6` exists and is tested (`tests/evaluation/test_tool_use_promotion_v6.py`). The
  switch is a gate contract bump (2 → 3).

### D. Wording corrections that change no rule

These are recorded here so the adopted text is internally consistent; each is also in `reports/ERRATA.md` §19
as a correction of the current text.

1. [11](11-THRESHOLDS.md) "`v5` is v4 **plus**": v5 is v3 plus. The "v3/v4 value" column lists the v3 values
   only. v4 (`M1CalibrationPolicy`) sets `min_call_precision` 0.65, `min_call_recall` 0.60,
   `min_macro_recall` 0.60, `min_clarification_accuracy` 0.60, `min_unsupported_accuracy` 0.40 and
   per-dimension regressions 0.05 / 0.10 / 0.10 / 0.10. v5's own values are unchanged.
2. The resolvability rule, stated once: **a threshold is adjudicable on a row whose worst-case resolvable
   margin is at most 10pp.** A *smaller* resolvable margin is finer resolution, not a deficiency. So
   `max_over_call_rate` on the 824 non-`CALL` items (6.8pp, not 6.9pp) is adjudicable, and
   `min_clarification_accuracy` on `CLARIFY` at n = 371 (10.2pp) is **not** — [11](11-THRESHOLDS.md)'s
   register marks it "yes". It needs n ≥ 385, or it is reported `WITHIN_NOISE`.

## What stays the same

Every threshold value in [11](11-THRESHOLDS.md), the arms, the seeds, the partition, the `n ≥ 200` mode floor,
the `ANSWER` strata sizing (8pp, n ≈ 601), the classifier and P-DET decisions of v2–v7, and every population.

## Candidates already scored / arms launched

None / none.

## If adopted

1. Record `study_002_prereg_v8` in [03](03-PREREGISTRATION.md) under **Amendments**, dated, citing this file.
2. Record it in `reports/ERRATA.md` (G15).
3. Set `ADOPTED_PARAMETERS` to the values above, switch the gate to `PromotionPolicyV6`, and bump
   `STUDY_002_GATE_CONTRACT` to 3, in one commit with the tests that assert the new behaviour.
4. Change this file's status line to "ADOPTED `<date>`"; keep the text.

An owner who adopts only part of it adopts the listed items by letter; the rest stay draft.
