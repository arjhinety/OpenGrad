---
base_model: Qwen/Qwen3.5-2B
base_model_relation: finetune
library_name: transformers
license: other
license_name: composite-per-source
license_link: https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE
tags:
  - tool-calling
  - function-calling
  - supervised-fine-tuning
  - qwen3.5
  - opengrad
datasets:
  - arrochi112/OpenGrad-ToolPolicy-Canonical-v2
---

# OpenGrad — Qwen3.5-2B, M0 SFT on Canonical-v2 (final)

Full-parameter supervised fine-tuning of `Qwen/Qwen3.5-2B` for tool-calling decision
boundaries, produced by [OpenGrad](https://github.com/arjhinety/OpenGrad).

**Research artifact, not a production model.** It is published because the experiment behind it
ran the *same* training recipe twice on two corpora, and the two results differ enough to be
worth reading. The recipe is unchanged from the earlier run: same base checkpoint, same
learning rate, schedule, batch geometry, sequence length, seed and 2,400-step horizon. The
corpus is the only variable, though it changed in three ways at once (see below).

**📊 [Baseline findings — charts and full comparison](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final/blob/main/findings.html)**

## Known limitation: general-capability regression

**Do not use this checkpoint as a general-purpose assistant.** It is a research artifact for
tool-call routing: deciding whether to call a tool, ask for a missing detail, or decline.

It shows a general-capability regression associated with tool-policy post-training on
When2Call-derived data. It appears after SFT (this model, M0) and also after DPO applied directly
to the base (M1-v1), so it is not specific to SFT. Causation is not established: one lineage, one
seed, no replicate.

| benchmark | Qwen3.5-2B base | **this model @1800** |
|---|---:|---:|
| GSM8K 0-shot accuracy | 67.4% | **0.0%** |
| GSM8K 0-shot refusals | 0 of 1,319 | **1,319 of 1,319** |
| GSM8K 8-shot accuracy | 70.4% | **56.3%** |
| IFEval prompt-level strict | 67.8% | **45.1%** |
| MMLU-Pro | 49.0% | **37.0%** |

Asked a GSM8K question with no exemplars, this model declines every time; with eight exemplars it
answers every question and gets 56.3% right. M1-v2, the DPO model trained from this checkpoint,
inherits the same profile (0-shot refusals 1,319 of 1,319, 8-shot 55.5%, IFEval 45.8%, MMLU-Pro
37.0%). M1-v1 ([`OpenGrad-Qwen3.5-2B-M1-DPO`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO),
DPO on the base model with no SFT) refuses 70.7% (933 of 1,319) of the same zero-shot questions;
its MMLU-Pro at the corrected budget is unmeasured. The corpus card documents refusal-shaped
targets labelled ANSWER in this model's training data (18,114 of 173,237 records); that is a
descriptive count, not a causal finding. Source: `results/final_campaign_verdict.json` in the
repository.

## What it does

| | this model | B0 (untrained) | previous run† |
|---|---:|---:|---:|
| `call_f1` | **0.7470** | 0.6264 | 0.6278 |
| call precision | 0.7350 | 0.4618 | 0.7610 |
| call recall | 0.7594 | 0.9735 | 0.5342 |
| over-call rate | 0.1505 | 0.6238 | 0.0922 |
| clarification accuracy | 0.7682 | 0.1186 | 0.7951 |
| unsupported accuracy | 0.5430 | 0.0177 | 0.6225 |

† The previous-run column (partial-v2, checkpoint 1200) comes from the M0 evaluation report; no
per-partition artifact for it is committed, so it can't be recomputed from the repository.

The untrained baseline scores well on `call_f1` for a reason worth understanding before reading
that column: it calls a tool on **62%** of requests whose correct answer is *not* a call. It
recalls 97% of the calls that should be made and gets 2% of unsupported requests right. That is
one behaviour, not a policy.

This model calls far less and decides more: over-calling drops to 0.1505 while recall stays at
0.7594, so `call_f1` rises to 0.7470.

## The honest part

**This is not a promotion.** By the repository's promotion policy every checkpoint of this run is
`REJECT`, including the one published here, because the policy caps over-call at 0.20 *and*
forbids a recall drop greater than 0.10 against B0 — and B0's recall of 0.9735 is itself a
property of over-calling. No checkpoint here met both: 600 fails on over-call alone, 1200 on
both, 1800 and 2400 on the recall drop. The gate was left exactly as written rather than
adjusted after seeing the result.

**It is not strictly better than the previous run.** Against `OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV2`
it gains 0.119 `call_f1` and 0.225 recall, and gives back 0.026 precision, 0.058 over-call,
0.027 clarification accuracy and 0.080 unsupported accuracy. Whether that trade is desirable
depends on the deployment, and the numbers are here so you can decide rather than being told.

**The improvement is not attributable to one data source.** The corpus changed in three ways at
once — a new source (xLAM/APIGen) was added, a validator defect was fixed that took ToolACE from
697 to 11,051 accepted records, and an interrupted materialization was corrected that took
When2Call from 4,000 to 6,505. The two minus-xLAM ablation arms have since run
([fixed-compute](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-FixedCompute),
[matched-exposure](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-MatchedExposure)).
xLAM is this corpus's only `CALL_PREDICTION` source, so they remove xLAM together with that whole
channel: they measure the joint removal and cannot separate xLAM content from supervision type.
Every checkpoint of both arms is `REJECT`. On the confirmatory partition recall falls to 0.4879
(fixed-compute @1200) and 0.4238 (matched-exposure @1060) against this model's 0.7594, and
`call_f1` to 0.6030 and 0.5557. The arm that trained more did better, but exposure is not ruled
out: the matched arm logged 6,743,788 supervised tokens against this run's 5,678,531 (1.19×, not
~1.0×), and the arms' selected checkpoints saw 3.80M and 3.36M against 4.27M for this model at
1800. Single seed. Attributing the gain to one of the three changes still needs another valid
call-prediction source.

**Some behaviours are not measured at all.** Tool-selection accuracy, argument validity and
schema validity are not computed by this evaluator, so their absence from the table above is not
a zero. The evaluation population also contains no direct-answer examples, so the
don't-call-when-you-should-answer behaviour is untested.

## Training

| | |
| --- | --- |
| Base model | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Corpus | [Canonical-v2 (final)](https://huggingface.co/datasets/arrochi112/OpenGrad-ToolPolicy-Canonical-v2), 161,966 trainable records |
| Supervision | 56,090 `CALL_PREDICTION` · 105,876 `COMPLETE_TRAJECTORY` |
| Method | full-parameter SFT, bfloat16, AdamW (fp32 state) |
| Steps | 2,400 (cosine, 120-step warmup) |
| Batch | 8 × 2 accumulation, 4,096-token micro-batch budget |
| Window | 2,048 tokens |
| Seed | 42 |
| Hardware | 1× A100-SXM4-80GB, 42 min, 28.0 GiB peak |
| Loss | 1.5725 → 0.3172 |

Note that 2,400 steps covered **0.169 epochs** of this corpus (27,372 examples seen) against the
previous run's 0.272 (27,672 of 101,785), because the corpus grew. The nominal 0.237 and 0.377
assume 16 examples per step; batches are token-budgeted, so fewer were seen. The step count was
held fixed on purpose: this is a causal corpus experiment, not a hyperparameter search.

## Which checkpoint this is, and why

`checkpoint-1800` of 600/1200/1800/2400. The selection rule was written and committed **before**
any checkpoint was evaluated, and it is the reason the two highest-scoring checkpoints are not
here:

| step | `call_f1` | over-call | eligible |
|---|---:|---:|---|
| 600 | 0.7191 | 0.3436 | no — over-calls |
| 1200 | **0.7382** | 0.2214 | no — over-calls |
| **1800** | 0.7092 | 0.1705 | **selected** |
| 2400 | 0.7043 | 0.1613 | yes |

Both disqualified checkpoints buy their `call_f1` the same way the baseline does. Among the
survivors, 1800 and 2400 differ by 0.0013 in balanced score, inside the pre-registered 0.01
tolerance. 1800 also has the higher macro score (0.6757 against 0.6744), so the tie-break was
applied but did not change the selection. **Treat 1800 and 2400 as tied, not ranked.**

## Evaluation, and what it is

Checkpoint selection used the 2,373-example DEV partition. The published result is the
**1,277-example pre-registered internal confirmatory partition**, scored **once**, on the
selected checkpoint only.

This is **not an untouched external benchmark.** The wider upstream evaluation population has
already influenced earlier work in this project, which is exactly why the partition was frozen in
advance and why the wording here is precise. Treat these as pre-registered internal results.

## Intended use and limits

Research artifact. Not safety-tuned, not aligned, and not intended for autonomous tool use. It was
trained on public tool-calling datasets whose licences are documented per source on the corpus
card, and it inherits their limitations, including synthetic data and unverified tool
invocations. It should not be deployed without task-specific evaluation, and it will behave poorly
on anything outside the decision-boundary behaviour it was trained for.

## License and attribution

These weights are derived from [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)
(revision `15852e8c`), released under the **Apache License 2.0** ([license text at that revision](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE)).
That license applies to this derivative: keep the license and its notices, and note that these weights
are **modified** from the original by the post-training described above.

The training data are modified derivatives of upstream datasets with their own terms (CC-BY-4.0 and
Apache-2.0, attribution required), listed per source in the dataset card's `source-licenses.md`.
`license: other` / `composite-per-source` records that no single license covers every component; it
does not relicense any of them.
