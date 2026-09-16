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

## 2.5 Required response-mode coverage — added after a gate failure

**A promotion gate is only as broad as the held-out set behind it.** `tool_use_promotion_v4`
promoted M1-v2 on a frozen partition containing **only** `tool_call`, `request_for_info` and
`cannot_answer` examples — and **no ANSWER examples at all**. The checkpoint it promoted refuses
100% of bare arithmetic questions and scores 22.0pp below Base on IFEval prompt-strict (67.8% →
45.8%). Nothing in the gate could have detected that, because the gate never asked the model to
simply answer something.

The promotion itself is also weaker than the word suggests. M1-v2 is promoted under a
parent-relative gate (v4) introduced after M0 was evaluated; M0 also clears v4, and M1-v2 fails the
v3 gate that rejected M0. The promotion reflects the gate change; the measured difference from M0
(+0.0078 call_f1, 7 of 453 calls, single seed) is within noise.

Any future promotion gate **must** evaluate held-out examples covering all four response modes:

| mode | what it tests |
|---|---|
| `ANSWER` | the model answers directly when it can, without a tool and without declining |
| `TOOL_CALL` | the model calls the right tool with the right arguments |
| `REFUSE` | the model declines when the request is genuinely outside its reach |
| `CLARIFY` | the model asks when the request is underspecified |

and **must** report, per mode:

- `answer_rate` and `refusal_rate`
- `accuracy_given_answer` wherever a correctness notion exists — reported **separately** from
  aggregate accuracy, never collapsed into it
- tool-call precision / recall / F1 and `over_call_rate`
- ordinary instruction-following capability

### Mandatory anti-pattern guard

A candidate **must be rejected** when tool-policy metrics improve while direct-answer behaviour
collapses. Concretely, a material rise in `refusal_rate` or fall in `answer_rate` on `ANSWER`-mode
held-out examples is a **blocking regression**, regardless of how far `call_f1` rose. That exact
combination is what happened here and what the gate let through.

### Response-mode confusion matrix

Gates should report a confusion matrix over `{ANSWER, TOOL_CALL, REFUSE, CLARIFY}` — expected mode
against observed mode. A scalar score cannot distinguish "answered correctly" from "declined a task
it could do"; the matrix makes that substitution visible. Note that an all-zero row for any mode
means that mode is **untested**, not that it passed.

### Keep the gate generic

Do **not** hard-code GSM8K, IFEval or any single benchmark as the protection. Those are the
instruments that happened to expose this failure; a gate overfitted to them would simply move the
blind spot. The requirement is *coverage of response modes*, satisfiable by any held-out set that
genuinely spans them.

Evidence: [`reports/FINAL_CAMPAIGN_AUDIT.md`](../reports/FINAL_CAMPAIGN_AUDIT.md) ·
[`reports/GENERAL_CAPABILITY_REGRESSION.md`](../reports/GENERAL_CAPABILITY_REGRESSION.md)

---

## 2.6 Pre-training gate: model components (from 2026-09-16)

Before a checkpoint can be a candidate, the path that trains it has to be validated. From
`full-model-components-v1`, trainers carry and train the vision encoder and the native MTP layer. That path has
only been tested on a tiny CPU model, so readiness gate `model_components_validation` blocks every real SFT or
DPO run that carries either component until two checks have committed evidence:
- **`gpu_smoke_test`:** memory, throughput and checkpoint size on the real model; vLLM loading with MTP
  drafting; the measurement of the untuned MTP loss weights.
- **`gguf_export`:** keeps MTP and produces a working vision projector.

Both are tracked, with their exact requirements, in `reports/training/model-components-validation.json`.
Text-only runs, including every Study 002 arm, are exempt. See
[`MODEL_COMPONENT_POLICY.md`](MODEL_COMPONENT_POLICY.md) §10.

---

## 3. Automated Reporting

Every promotion evaluation generates:
- `runs/<experiment-id>/promotion/verdict.json`: Machine-readable audit payload.
- Formatted markdown tables showing observed values vs. threshold limits.
