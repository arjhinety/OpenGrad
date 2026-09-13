---
pretty_name: OpenGrad Qwen3.5-2B M0 SFT CorpusV1 — evaluation record
language:
- en
license: apache-2.0
task_categories:
- text-generation
tags:
- tool-calling
- function-calling
- evaluation
- negative-result
- opengrad
- artifact-loss
configs:
- config_name: default
  data_files:
    - split: predictions
      path: "evaluation/checkpoint-*/predictions.jsonl"
---

# OpenGrad — Qwen3.5-2B, M0 SFT on corpus v1: evaluation record

This repository holds the **evaluation evidence** for one OpenGrad experiment,
`qwen35_2b_m0_sft_full_v3`: a full-parameter supervised fine-tuning run of
`Qwen/Qwen3.5-2B` on the published `OpenGrad ToolPolicy Canonical v1` corpus.

**There are no model weights here, and none exist.** Every checkpoint this run produced was
deleted from local storage before it was uploaded, and none of them can be recovered. This
repository is what survives: the per-example predictions and metrics from five of the six
checkpoints that were evaluated. It is published because the run is a negative result whose
numbers should be checkable even though the models are gone.

The full account of the loss is
[INC-0001 in the OpenGrad incident log](https://github.com/arjhinety/OpenGrad/blob/master/docs/INCIDENT_LOG.md).
It is recorded rather than omitted: the surviving artefact is not the intended one, and a
reader should not have to infer that from a missing directory.

## What the run measured

Scored on the frozen When2Call held-out split (3,650 examples), identical engine, renderer,
template, parser and generation settings as the B0 baseline. `call_f1` is the primary metric:

| checkpoint | call_f1 | precision | recall | over-call | under-call | clar_ok | unsup_ok | predictions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | 0.0046 | 0.6000 | 0.0023 | 0.0008 | 0.0347 | 0.9292 | 0.5058 | **lost** |
| 800 | 0.0062 | 1.0000 | 0.0031 | 0.0000 | 0.0193 | 0.9538 | 0.5591 | in `evaluation/checkpoint-800/` |
| 1200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0263 | 0.7434 | 0.7992 | in `evaluation/checkpoint-1200/` |
| 1600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0170 | 0.9500 | 0.5591 | in `evaluation/checkpoint-1600/` |
| 2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0193 | 0.9415 | 0.5838 | in `evaluation/checkpoint-2000/` |
| 2400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0185 | 0.9415 | 0.5776 | in `evaluation/checkpoint-2400/` |
| **B0** | **0.6191** | 0.4542 | 0.9722 | 0.6425 | 0.0147 | 0.1009 | 0.0131 | baseline, separate |

One gap is worth stating separately: checkpoint 400 was evaluated — its row in the table comes
from `evaluation/curve.json` — but its artifacts are missing too, so there are no predictions
behind that row. The other five rows are fully checkable.

## The finding

Fine-tuning on corpus v1 **destroyed** tool calling rather than improving it. By step 1200 the
model had stopped emitting tool calls entirely: `call_recall` 0.9722 → 0.0000, `call_f1`
0.6191 → 0.0000. `over_call_rate` fell to 0.0000, which is not a calibration win but the same
collapse seen from the other side — the baseline's 0.6425 over-call rate came from an
always-call policy, and removing all calls removes all over-calls with it.

The evidence points to the data rather than the training procedure. At the training boundary,
corpus v1 retained 55,719 records of which **9 (0.0162%)** contained a tool call in their
supervised target. The records that carried tool calls were the ones being discarded: of the
154,760 that failed to render, 51,034 were Glaive records rejected as orphaned tool results
because the Glaive adapter could not parse that revision's call format; xLAM's records all end
on a call that no tool result answers; and most ToolACE, LoopTool and BUTTON records failed the
canonical schema contract. There was almost nothing in the signal to learn tool calling from.

The test of that explanation is the corpus-v2 run — same base checkpoint, same procedure, same
hyperparameters, same evaluation — which reached `call_f1` 0.5995 and `call_recall` 0.5050 at
checkpoint 1200, the best of its four on this same set. See
[`arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV2`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV2).
The corpus was the only thing changed, but it changed in more than one way: v2 also dropped
three of v1's six sources (xLAM, BUTTON, LoopTool), used a smaller When2Call slice (4,000
records against 14,829), and has 101,785 trainable records against 55,719. The pair points to
the corpus rather than the procedure; it does not isolate tool-call supervision as the cause.
That comparison is what makes this run evidence rather than just a failed attempt, and it is
also why this repository may be useful: it is the losing half of a controlled pair, kept
because the loss is the measurement.

## What is lost, precisely

* All six checkpoints (steps 400–2400). No weights, no optimizer state, no ability to run new
  inference or sample completions.
* The predictions and metrics for step 400.
* The ability to test whether any of these models generalises anywhere other than this one
  evaluation set.

What survives, and what it supports:

* The recorded metrics are not taken on trust. They are re-derivable from the predictions in
  this repository with `opengrad.evaluation.routing.routing_metrics`, and were verified to match
  to within `metrics.json` rounding before publication.
* The direction and magnitude of the collapse are fully supported across five checkpoints.
* Any claim about *why* it happened rests on the corpus analysis, not on these checkpoints.

## Reproducing the numbers

```python
import json
from opengrad.evaluation.routing import routing_metrics

rows = [json.loads(line) for line in open("evaluation/checkpoint-1200/predictions.jsonl")]
metrics = routing_metrics(
    [row["expected_decision"] for row in rows],
    [row["prediction"]["decision"] for row in rows],
)
```

Each `metrics.json` carries the same values under `baseline_comparison.metrics[...]["candidate"]`,
alongside the frozen B0 values it was compared against.

## Training configuration

| | |
| --- | --- |
| Base model | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Corpus | `canonical_v1`, release manifest sha256 `181b3fba…` |
| Tuning method | full parameter (1.88B trainable, no LoRA, no quantisation) |
| Precision | bfloat16 |
| Steps | 2,400 (max), cosine schedule, 120-step warmup |
| Batch | 8 × 2 accumulation, 4,096-token micro-batch budget |
| Window | 2,048 tokens |
| LR | 1e-5, AdamW, grad clip 1.0 |
| Seed | 42 |
| Hardware | 1× A100-SXM4-80GB, 41.6 min, 28.3 GiB peak |
| Trained | 24,272 examples · 10,254,922 supervised tokens |
| Loss | 1.9898 → 0.6808 (min 0.1316, mean 0.5976) |

Note that the training loss looks healthy throughout. A low loss on a corpus containing almost
no tool-call supervision is exactly what this run demonstrates: the loss fell while the
capability being measured disappeared. It is a caution against reading a converging loss curve
as evidence that training worked.

## Files

```
evaluation/curve.json                       all six points, including the lost one
evaluation/checkpoint-*/predictions.jsonl   one line per held-out example: expected vs predicted decision
evaluation/checkpoint-*/metrics.json        scored metrics and the B0 comparison
evaluation/checkpoint-*/environment.json    engine, renderer, template hash, revisions at run time
evaluation/checkpoint-*/residual-profile.json
run/resolved_config.yaml                    the exact contract the run executed
run/experiment.json, run/events.jsonl       experiment record and per-step training events
run/train_log.jsonl                         the loss curve the run produced
run/dataset_manifest.json                   corpus lineage and pinned hash
```

## Intended use and limits

Research artefact and record of a failure. Not a model: nothing here can be loaded for
inference. Not a benchmark: the 3,650 examples are the frozen held-out split, and the
predictions are one model's outputs on it.

Use it to check the claim that corpus v1 could not teach tool calling, or to compare against a
reproduction. Do not use it to draw conclusions about `Qwen/Qwen3.5-2B` itself, and do not
treat the five surviving checkpoints as representative of the sixth.
