# Promotion Policy & Regression Gates

**Building in Public.** Promotion of model checkpoints to validated production or release status is governed by deterministic rules.

---

## 1. Rule Types

A promotion policy (`src/opengrad/promotion/policy.py`) evaluates three classes of criteria:

1. **`must_pass` (Absolute Quality Floors)**:
   - Specific benchmark scores that any production candidate must achieve.
   - Example: `bfcl-v4 >= 65.0%`.
2. **`max_regression` (Non-Regression Ceiling)**:
   - Maximum allowable performance drop relative to the designated baseline.
   - Example: `ifeval >= baseline - 1.0%`, `mmlu-pro >= baseline - 0.5%`.
3. **`minimum_improvement` (Target Capability Gain)**:
   - Mandatory capability improvement on the targeted hypothesis.
   - Example: `bfcl-v4 >= baseline + 2.0%`.

---

## 2. Decision Outcomes

- **`PROMOTE`**: All `must_pass`, `max_regression`, and `minimum_improvement` rules passed.
- **`REJECT`**: One or more rules failed. Checkpoint cannot become the new baseline.
- **`REVIEW`**: No regressions detected, but target improvement did not meet threshold. Requires researcher review.

---

## 3. Automated Reporting

Every promotion evaluation generates:
- `runs/<experiment-id>/promotion/verdict.json`: Machine-readable audit payload.
- Formatted markdown tables showing observed values vs. threshold limits.
