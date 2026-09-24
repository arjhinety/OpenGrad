---
name: opengrad-promotion-gates
description: How to write, change and test gates in OpenGrad so they cannot pass vacuously — promotion policies (must_pass / max_regression / minimum_improvement, tool_use_promotion_v3/v4/v5/v6, quantization_preservation_v1), the Study 002 gate study_002_gate_v1, mandatory response-mode coverage and the anti-pattern guard, promote/reject and the checkpoint lifecycle, execution accounting for verification gates, readiness and pre-training gates with evidence files, and the testing discipline every gate needs. This skill should be used whenever a gate, threshold, promotion policy or verdict is added, changed, evaluated or reported, and before calling any checkpoint promoted.
---

# OpenGrad promotion and gates

A gate in OpenGrad is a decision rule. Its failure modes are known from Study 001:
- M1-v2 was promoted by a gate whose held-out set contained no ANSWER examples;
- a freeze gate skipped every corpus;
- a registry validator ran nothing and exited 0.

Every gate is written so each of those is impossible.

## Principles (non-negotiable)

1. **Fixed before results (G1).** A policy version is pinned in the experiment config *before* evaluation
   (`promotion.policy_version`; readiness requires M1 to pin `tool_use_promotion_v4`). A new rule is a new policy
   version. Never edit a version in place, and never relax a threshold to pass a run.
2. **Coverage before scores (G2).** A promotion held-out set must cover ANSWER, TOOL_CALL, REFUSE and CLARIFY,
   report `answer_rate`, `refusal_rate` and `accuracy_given_answer` per mode, and show the response-mode
   confusion matrix. **Anti-pattern guard:** reject when tool metrics rise while direct answering collapses.
   An all-zero row means the mode is *untested*, not passed (`docs/PROMOTION_POLICY.md` §2.5).
3. **Margins against noise (G3).** Compare a gate margin with rerun noise before calling it a difference, and
   state the counts ("7 more of 453 gold calls recalled", not "+0.0078"). Keep the count and the delta on the
   same metric: those 7 calls are the *recall* difference, while +0.0078 is the `call_f1` difference
   (`reports/ERRATA.md` §19).
4. **Prove it ran.** A verification gate proves both that its assertions passed and that the expected
   assertions executed (`src/opengrad/verification/accounting.py`). Report an execution census. BLOCKED
   (could not check) is never success.
5. **Evidence, not intent.** A gate reading a status file accepts `PASS` only with an evidence path that exists
   and the current policy version. `model_components_validation` (`src/opengrad/readiness_states.py`) is the pattern.
6. **A wrapping gate reads the wrapped decision.** `study_002_gate_v1` contract 1 read seven of v5's dimensions
   and never its `decision`, so it passed candidates v5 rejected (`reports/ERRATA.md` §19). A gate built on a
   policy fails whenever that policy does not promote, and a test asserts it over mutated inputs.
7. **The input may not declare its own requirement.** Required sentinel lists, provenance fields and
   population sizes are constants in code, pinned to the spec; a bundle that lists `required: ["x"]` and
   `present: ["x"]` must not pass. A value the preregistration requires but never quantifies is `None` and
   blocks (`PreregParameters`), never a default.
8. **Thresholds are compared, not float-subtracted.** `0.9 - 0.6` is `0.30000000000000004`; compare through
   `at_least` / `at_most` in `src/opengrad/verification/resolvability.py` and test every threshold exactly at
   its boundary.

## Where the gates live

| Gate | Code | Config and records |
|---|---|---|
| Generic promotion rules | `src/opengrad/promotion/policy.py` (`PromotionPolicy`) | experiment `promotion:` block |
| Tool-use promotion v3 / v4 (parent-relative) | `src/opengrad/promotion/tool_use_policy.py`, `src/opengrad/promotion/m1_calibration.py` | `docs/evaluation/CHECKPOINT_SELECTION_RULE.md` |
| Tool-use promotion v5 (Study 002: ANSWER floors, `NOT_EVALUABLE`) | `PromotionPolicyV5` in `src/opengrad/promotion/tool_use_policy.py` (subclasses the v3 class, not v4) | `docs/research/study-002/11-THRESHOLDS.md` |
| Study 002 gate `study_002_gate_v1`, contract 3 (wraps v6; `study_002_prereg_v8`) | `src/opengrad/verification/study_002_gate.py` over `population_validators.py` and `resolvability.py` | `python -m opengrad.verification.study_002_gate --self-test`; undeclared prereg values in `PreregParameters` |
| Regression detection | `src/opengrad/promotion/regression.py` | `opengrad compare` output |
| Quantization preservation | `src/opengrad/promotion/quantization.py` (`quantization_preservation_v1`) | `release/gguf/quantization_preservation_v1.json` |
| Promotion artifacts | `src/opengrad/promotion/artifacts.py` | `runs/<id>/promotion/` |
| Readiness (baseline, SFT, DPO) | `src/opengrad/readiness.py` | `opengrad readiness <config> --json` |
| Pre-training component gate | `_model_components_state` in `src/opengrad/readiness_states.py` | `reports/training/model-components-validation.json` |
| Stage authorization | `src/opengrad/experiments/gates.py` | — |

## Promote and reject

```bash
opengrad checkpoint inspect <checkpoint_id>
opengrad promote <checkpoint_id> --reason "<which gate, which version, which evidence>"
opengrad reject  <checkpoint_id> --reason "<which rule failed>"
```

- Checkpoints start as `CANDIDATE`. The newest is never "best" automatically, and `PROMOTED` requires passing the
  pinned policy (`docs/CHECKPOINTS.md` §1).
- Reports state which policy version promoted a checkpoint, and whether an older version would have rejected it.
  M1-v2 passes v4 and fails v3, and that fact travels with the claim.

## Testing a gate (required for every new or changed gate)

- **Each rule separately:** a pass case, and a fail case per rule, with literal numbers at the boundary. The
  test asserts the exact computed value, e.g. "this input must yield 4/5 = 0.8".
- **Fail closed:** missing, unreadable or stale inputs fail with a specific code. A dormant gate (no declared
  input) is tested as dormant, and then as failing once declared (`tests/config/test_readiness.py`,
  `tests/data/test_yield_gate.py`).
- **Vacuous pass is a failure:** an empty population, a zero-row mode, or zero executed checks must not pass.
  Assert the execution count.
- **Units:** pin the counting unit of the numerator and the denominator separately (paths versus leaf
  elements, items versus tokens).
- **The committed state:** test the real committed record, e.g. the pending component gate blocks a run
  carrying vision/MTP today. A status flip then has to be a deliberate test change.
- **Where tests live:** `tests/evaluation/test_tool_use_promotion_policy.py`,
  `tests/evaluation/test_tool_use_promotion_v5.py`, `tests/verification/test_study_002_gate.py`,
  `tests/verification/test_population_validators.py`, `tests/verification/test_resolvability.py`,
  `tests/experiments/test_promotion_and_regression.py`, `tests/experiments/test_gates.py`,
  `tests/config/test_readiness.py`.

## Sources of truth

`docs/PROMOTION_POLICY.md` · `docs/CHECKPOINTS.md` · `docs/research/GUARDRAILS.md` (G1–G3, G8) ·
`docs/MODEL_COMPONENT_POLICY.md` §10 · `reports/FINAL_CAMPAIGN_AUDIT.md`

## Keeping this skill current

Add a row to the gate table in the same commit that adds a gate or a policy version. Record every gate failure
that changed a rule, as a short "why" line in the principles.
