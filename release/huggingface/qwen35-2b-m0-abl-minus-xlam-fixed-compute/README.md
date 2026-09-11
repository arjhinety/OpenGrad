---
base_model: Qwen/Qwen3.5-2B
base_model_relation: finetune
library_name: transformers
license: other
license_name: composite-per-source
tags:
  - tool-calling
  - function-calling
  - supervised-fine-tuning
  - qwen3.5
  - opengrad
  - ablation
datasets:
  - arrochi112/OpenGrad-ToolPolicy-Canonical-v2-minus-xlam
---

# OpenGrad — Qwen3.5-2B, joint xLAM + CALL_PREDICTION removal (fixed-compute)

Full-parameter SFT of `Qwen/Qwen3.5-2B` on Canonical-v2 with xLAM removed, training for the
reference's 2,400-step budget. Produced by [OpenGrad](https://github.com/arjhinety/OpenGrad).

**Research artifact, not a production model.** Published because the experiment's whole point is a
negative result that is only checkable if the weights behind it are available.

## What this experiment actually removed

Canonical-v2 maps xLAM to **every** `CALL_PREDICTION` record and the other three sources to
`COMPLETE_TRAJECTORY`. Removing xLAM therefore removed a source **and** the corpus's entire
call-prediction supervision channel at the same time:

> this model measures the effect of **removing xLAM together with the `CALL_PREDICTION` channel**.

It is **not a pure xLAM-content ablation**, and **no xLAM-specific causal claim** can be based on
it. Source identity and supervision type are perfectly aligned in this corpus. Separating them
would need another evidence-backed `CALL_PREDICTION` source.

## Results (confirmatory partition, 1,277 examples, scored once)

| run | `call_f1` | precision | recall | over_call | clarify | unsupp |
|---|---:|---:|---:|---:|---:|---:|
| B0 (untrained) | 0.6191 | 0.4542 | 0.9722 | 0.6425 | 0.1009 | 0.0131 |
| reference (full corpus) @1800 | **0.7470** | 0.7350 | **0.7594** | 0.1505 | 0.7682 | 0.5430 |
| **this arm** (fixed-compute) @1200 | 0.6030 | 0.7893 | 0.4879 | 0.0716 | 0.8059 | 0.6026 |
| matched-exposure arm @1060 | 0.5557 | 0.8067 | 0.4238 | 0.0558 | 0.8194 | 0.6203 |

Removing the joint source/channel drops recall far below both the reference and B0 while
precision rises and over-calling falls to a small fraction. The models become conservative but
stop recalling the calls they should make — the direction the missing supervision predicts.

The arm that trains **more** (this one: 2,400 steps, ~1.53× exposure over the retained corpus)
does **better** than the matched-exposure arm, so the recall loss tracks the missing supervision
rather than the reduced training budget. The two arms are one experiment and must be read
together.

## Honest limits

**Not a promotion.** Every checkpoint of both arms is `REJECTED` on `regression.call_recall`
against B0, as is every checkpoint of the reference run. The gate was left exactly as written.

**Single seed.** A small between-arm difference is a finding to replicate, not a settled result.

**Some behaviours are unmeasured.** Tool-selection accuracy, argument validity and schema
validity are not computed; their absence is not a zero. The evaluation population has no
direct-answer examples.

## Checkpoints

All four retained checkpoints are published under their step: `checkpoint-600`, `checkpoint-1200`
(the DEV-selected one), `checkpoint-1800`, `checkpoint-2400`. Selection used the 2,373-example DEV
partition and the rule frozen before the run; the confirmatory partition was scored once, on
`checkpoint-1200` only.

## Intended use

Research artifact. Not safety-tuned, not aligned, not for autonomous tool use. It inherits the
limitations of its public training sources, including synthetic data and unverified tool
invocations, and it will behave poorly outside the decision-boundary behaviour it was trained for.