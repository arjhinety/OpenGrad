# Tool-use mixture methodology

Status: PRE_GPU_COMPLETE / BASELINE_INFERENCE_READY. Accessible corpora are materialized and audited; no model has been trained or evaluated.

*Status update (2026-09-13):* the "no model has been trained or evaluated" line above is historical. B0 has been scored, the M0 SFT line (corpus v1, corpus v2, Canonical-v2 final, two minus-xLAM ablations) and DPO (M1-v1 directly on the base, M1-v2 on M0) have been trained and evaluated, and M2 was not run. See [`docs/EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md).

## Research framing

Tool-call syntax, tool-use policy, and agent task completion are different claims. OpenGrad studies reliable tool-use policy: whether a model should CALL, ANSWER, CLARIFY, or report UNSUPPORTED; which tool and arguments to use; how to handle observations, state, dependencies, and failure; and when to stop. A valid serialized call is protocol evidence, not evidence of a reliable policy.

## Two orthogonal axes

Every normalized example retains source provenance (`dataset_id`, revision, original split, and contamination status) and behavioral semantics. Sources are inputs, not capabilities: one source may populate many capabilities, and one capability may be supplied by many sources. Labels use `known`, `derived`, `heuristic`, or `unknown`; heuristic labels are never silently presented as ground truth. The canonical taxonomy is `registry/tool_behaviors.yaml`.

Canonical metadata may include `behavior`, `tool_context`, `interaction`, `training`, and `counterfactual` objects in addition to the backwards-compatible conversation fields. Catalogue metadata records zero, one, small, medium, or large tool sets, relevant tools, distractors, near duplicates, and measurable catalogue token count.

## Examples and coverage

The data audit reports measured decision and capability distributions, source attribution, turn shape, catalogue size, and distractors. Run `uv run opengrad-data-audit --records <canonical.jsonl> --config <mixture.yaml> --json`. It does not invent percentages for unmaterialized sources.

Hard no-call examples contain tempting tools or overlapping descriptions; easy no-call examples merely lack a relevant tool. Direct-answer/ordinary-instruction retention is a first-class `ANSWER` capability and is tracked separately from tool-policy data. Counterfactual groups preserve a group ID, variant, and one changed factor so decision-boundary consistency can be evaluated without creating a synthetic corpus here.

## M0, M1, and M2

* M0 (`source_baseline_v1.yaml`, with the historical `mixture_v1.yaml` alias) preserves the source-oriented hypothesis as a control.
* M1 (`balanced_policy_v1.yaml`) is a behaviorally balanced hypothesis. Its proposed weights are not findings.
* M2 (`residual_policy_v1.yaml`) is schema-ready only. It requires a fixed baseline, a residual profile with a minimum evidence threshold, and a versioned deterministic residual-to-mixture algorithm. No weights are generated before baseline results.

Selection is intended to be: desired behavior distribution -> eligible canonical examples -> source/provenance and contamination constraints -> model renderer -> frozen split. Evaluation-only and preference-only data remain excluded from ordinary SFT; APIGen-MT/τ-bench overlap safeguards remain in force.

## Baseline diagnosis and residuals

A baseline must emit overall metrics, the CALL/ANSWER/CLARIFY/UNSUPPORTED confusion matrix, directional under-/over-calling, wrong-tool, argument, clarification, unsupported, observation, multi-turn, and catalogue-size sensitivity. Each residual records the model, baseline experiment, revision, source evaluation, confidence, and sample count. Small samples require caution; M2 requires a documented minimum count. `opengrad.data.residuals.residual_to_weights` maps residual IDs deterministically with configurable floors/caps and records its algorithm version.

## Evaluation layers

1. Deterministic behavioral evaluation covers decision classes, routing, selection, arguments, state, observation, and recovery, with call precision/recall/F1, must-call/no-call, directional rates, clarification and unsupported metrics, schema validity, groundedness, dependency/parallelism, and multi-turn success.
2. External benchmarks (where configured) provide transfer measurements such as BFCL, When2Call, τ-bench/τ², ToolSandbox, MCPMark, and Toolathlon.
3. Deployment-derived routing evaluation is a planned held-out family motivated by phenomena observed in OpenWeights. It will use independently generated prompts/entities and will never copy training prompts or imply an official OpenWeights benchmark. Layer 3 is not implemented.

OpenWeights observation != OpenGrad baseline != OpenGrad result. External observations motivate hypotheses; only an executed OpenGrad evaluation can be called reproduced or a result.

## Planned ablations and later work

The first controlled family is B0 base instruct, M0 generic/source-oriented SFT, M1 balanced SFT, and M2 residual-driven SFT, with matched checkpoint, budget, optimizer, evaluation, and sampling where feasible. Preference optimization remains conditional on diagnosis; on-policy distillation samples remaining student residuals and records why each example entered a batch. Capability data is separate from MTP/speculative continuation data, although future workload profiles should include prose, reasoning, tool decisions, structured calls/JSON, observations, short/long responses, code, and multi-turn agents.

These are hypotheses, not findings: H1 behavioral balancing, H2 residual targeting, H3 retention, H4 counterfactual supervision, and H5 catalogue robustness.
