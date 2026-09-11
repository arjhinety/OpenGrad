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

# OpenGrad — Qwen3.5-2B, joint xLAM + CALL_PREDICTION removal (matched-exposure)

Full-parameter SFT of `Qwen/Qwen3.5-2B` on Canonical-v2 with xLAM removed, training for a horizon
that matches the reference's **supervised-token exposure** (2,119 steps). Produced by
[OpenGrad](https://github.com/arjhinety/OpenGrad).

**Research artifact, not a production model.** Published because the experiment's whole point is a
negative result that is only checkable if the weights behind it are available.

## What this experiment actually removed

Canonical-v2 maps xLAM to **every** `CALL_PREDICTION` record and the other three sources to
`COMPLETE_TRAJECTORY`. Removing xLAM therefore removed a source **and** the corpus's entire
call-prediction supervision channel at the same time:

> this model measures the effect of **removing xLAM together with the `CALL_PREDICTION` channel**.

It is **not a pure xLAM-content ablation**, and **no xLAM-specific causal claim** can be based on
it. Source identity and supervision type are perfectly aligned in this corpus.

## The step count is computed, not chosen

```text
matched_steps = 2400 x (29,630,369 / 33,565,721) = 2118.6 -> 2119
```

Matching on supervised (loss-bearing) tokens rather than records or rendered tokens, because that
is the mass the loss acts on. xLAM's records are short, so the measures disagree sharply
(records → 1,569, rendered tokens → 1,544, supervised tokens → 2,119). This arm additionally
spends **11.7% fewer optimizer steps/FLOPs** than the reference.

## Results (confirmatory partition, 1,277 examples, scored once)

| run | `call_f1` | precision | recall | over_call | clarify | unsupp |
|---|---:|---:|---:|---:|---:|---:|
| B0 (untrained) | 0.6191 | 0.4542 | 0.9722 | 0.6425 | 0.1009 | 0.0131 |
| reference (full corpus) @1800 | **0.7470** | 0.7350 | **0.7594** | 0.1505 | 0.7682 | 0.5430 |
| fixed-compute arm @1200 | 0.6030 | 0.7893 | 0.4879 | 0.0716 | 0.8059 | 0.6026 |
| **this arm** (matched-exposure) @1060 | 0.5557 | 0.8067 | 0.4238 | 0.0558 | 0.8194 | 0.6203 |

Recall collapses furthest here while precision is highest and over-calling is lowest — the same
direction as the fixed-compute arm, run further. Because the arm that trains **more** does
**better**, the recall loss is not explained by the reduced budget; it tracks the missing
supervision. The two arms are one experiment and must be read together.

## Honest limits

**Not a promotion.** Every checkpoint of both arms is `REJECTED` on `regression.call_recall`
against B0. The gate was left exactly as written.

**Single seed.** A small between-arm difference is a finding to replicate, not a settled result.

**Some behaviours are unmeasured.** Tool-selection accuracy, argument validity and schema
validity are not computed; their absence is not a zero.

## Checkpoints

All four retained checkpoints are published under their step: `checkpoint-530`, `checkpoint-1060`
(the DEV-selected one), `checkpoint-1590`, `checkpoint-2119`. Selection used the 2,373-example DEV
partition and the rule frozen before the run; the confirmatory partition was scored once, on
`checkpoint-1060` only.

## Intended use

Research artifact. Not safety-tuned, not aligned, not for autonomous tool use. It inherits the
limitations of its public training sources, including synthetic data and unverified tool
invocations, and it will behave poorly outside the decision-boundary behaviour it was trained for.