# Glossary

Short identifiers and status words used across OpenGrad, each with where it is defined. When a document and
this page disagree, the linked definition wins and this page is the bug.

## Models and stages

| Term | Meaning | Defined in |
|---|---|---|
| **B0** | The base model, `Qwen/Qwen3.5-2B` at revision `15852e8c…`, evaluated unchanged. The baseline every stage is compared with. | [`README.md`](../README.md#latest-results), [`registry/models.yaml`](../registry/models.yaml) |
| **M0** | Study 001's supervised fine-tune (SFT) of B0 on Canonical-v2 final (checkpoint 1800). Not promoted. | [`docs/EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) |
| **M1-v1** | DPO applied directly to B0, with no SFT parent. Its best checkpoints no longer exist. | [`docs/EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) |
| **M1-v2** | DPO calibration started from M0 (checkpoint 30). The model promoted under `tool_use_promotion_v4`. | [`reports/M1_DPO_EVALUATION.md`](../reports/M1_DPO_EVALUATION.md) |
| **M2** | On-policy distillation. Never executed (mock-only scaffold); deferred to Study 003. | [`docs/ON_POLICY_DISTILLATION.md`](ON_POLICY_DISTILLATION.md) |
| **PTQ** | Post-training quantization (the GGUF ladder, closed). | [`reports/PTQ_PHASE_CLOSURE.md`](../reports/PTQ_PHASE_CLOSURE.md) |
| **QAD** | Quantization-aware distillation; decided not required for release. | [`reports/QAD_DECISION.md`](../reports/QAD_DECISION.md) |
| **MTP** | The base model's native multi-token-prediction head, carried or excluded by trainers per policy. | [`docs/MODEL_COMPONENT_POLICY.md`](MODEL_COMPONENT_POLICY.md) |

## Response modes

A tool-use request has one correct behaviour, its **mode** (also "decision"): **CALL** (call a tool),
**ANSWER** / **DIRECT** (answer directly, without a tool; `DIRECT` is the classifier's name for it),
**CLARIFY** (ask for missing information) and **UNSUPPORTED** (decline: no tool or knowledge can serve it).
Study 001's held-out set had no ANSWER items, which is why its regression went unseen.

## Study 002 identifiers

"C1" and "C2" each have **three** meanings in Study 002. Always qualify them.

| Term | Meaning | Defined in |
|---|---|---|
| **arm C0, C1, C2, R1, …** | Training arms of the experiment matrix. `C0` reproduces Canonical-v2; arm `C1` removes the synthetic subset. | [04-ARM-MATRIX](research/study-002/04-ARM-MATRIX.md) |
| **criterion C1 (coverage)** | Every mode in `P-CONF` has n > 0. | [06-SPLIT-SPEC](research/study-002/06-SPLIT-SPEC.md) §C1 |
| **criterion C2 (resolvability)** | Every row prints n and its resolvable margin; no claim on a row that cannot resolve 10pp. | [06-SPLIT-SPEC](research/study-002/06-SPLIT-SPEC.md) §C2 |
| **workstream C1** | Building canonical-v3 (provenance repair, schema normalization, classifier labels, decision balance). "C1 is authorised" means this. | [21-C1-IMPLEMENTATION-STATUS](research/study-002/21-C1-IMPLEMENTATION-STATUS.md) |
| **L1 … L12** | Lessons from Study 001; **L1** is "the held-out set could not see the failure" (no ANSWER items). | [01-LESSONS](research/study-002/01-LESSONS-FROM-STUDY-001.md) |
| **H1 … H6** | Study 002's hypotheses; H1: refusal-`ANSWER` label disagreement is enough to cause the collapse. | [02-RESEARCH-QUESTIONS](research/study-002/02-RESEARCH-QUESTIONS.md) |
| **P-DEV, P-CONF, P-UNANS, P-SEALED** | Populations: development (checkpoint selection), confirmatory (scored once per arm; the gate reads it), genuinely unanswerable (refusal correctness, H6), never-read reserve. | [06-SPLIT-SPEC](research/study-002/06-SPLIT-SPEC.md) §Populations |
| **P-DET, P-DET-COVERAGE** | Labelled samples for validating the refusal/decision classifier. P-DET-v1 is frozen (581 items); P-DET-COVERAGE-v1/v2 add DIRECT and CALL coverage. | [22-PDET-PROTOCOL](research/study-002/22-PDET-PROTOCOL.md), [30](research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md), [36](research/study-002/36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md) |
| **REP-A** | A re-run of arm C0 at a fixed seed: the nondeterminism floor every threshold margin is compared with. | [05-SEED-AND-REPRODUCIBILITY-POLICY](research/study-002/05-SEED-AND-REPRODUCIBILITY-POLICY.md) |
| **V1 … V12** | The provenance validators (mode coverage, metric denominators, …, resolvable margin). | [15-PROVENANCE-VALIDATORS](research/study-002/15-PROVENANCE-VALIDATORS.md) |
| **`study_002_prereg_vN`** | A preregistration version; v1 is the original contract, v2–v8 are amendments (v8 adopted 2026-09-24). | [03-PREREGISTRATION](research/study-002/03-PREREGISTRATION.md) |
| **`study_002_gate_v1`** | The Study 002 evaluation gate (17 checks at contract 3, over `tool_use_promotion_v6`). | [11-THRESHOLDS](research/study-002/11-THRESHOLDS.md) |
| **G1 … G17** | Guardrails derived from the claims Study 001 got wrong. | [GUARDRAILS](research/GUARDRAILS.md) |

## Status words

| Word | Meaning |
|---|---|
| **frozen** | The bytes are fixed and recorded by hash; the artifact is never edited. A number later found wrong is corrected in [`reports/ERRATA.md`](../reports/ERRATA.md). A frozen *study* is fixed at a git tag. |
| **pinned** | A file whose sha256 is recorded somewhere that a check reads (a `.sha256` sidecar, a manifest, a code constant). Pinned files are frozen. |
| **verified** | A claim whose evidence was checked against an immutable anchor: a pinned file's hash, or an external source at an exact revision. `verified: true` without the anchor is a defect. |
| **provisional** | Accepted for now, with stated conditions that withdraw it (e.g. a qualification resting on a model-made reference). |
| **gold** | A reference label frozen as the answer key. Model-made labels are a *model reference*, never gold; no Study 002 label is gold yet. |
| **WIP export** | A working annotation snapshot (`annotation/wip/`), not a gold freeze. |
| **canonical** | The normalized, source-independent record format (the "canonical IR"), and the corpora built in it (Canonical-v1, Canonical-v2 final, canonical-v3). |
| **promoted** | A checkpoint that passed the promotion policy version pinned before its evaluation. It says which policy, not that the model is better overall. |
| **PASS / FAIL / BLOCKED_*** | Gate outcomes. `BLOCKED_*` means a required input was absent and the gate could not check; it is never a pass. |
| **NOT_EVALUABLE / UNDER_POWERED / WITHIN_NOISE** | The population cannot measure a dimension / a mode is below n = 200 / a difference is smaller than the row can resolve. None supports a claim. |
