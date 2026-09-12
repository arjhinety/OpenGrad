# QAD experiment plan — M1-v2 Q4-class recovery

**Status:** specified and ready; substantive runs **not started**. See
[`QAD_DECISION.md`](QAD_DECISION.md) for why the GPU spend is held.

Written before any QAD training so no threshold, budget, or stop condition can be chosen after
seeing a result.

## Hypothesis

> Can a student learn to preserve the BF16 tool-use policy while trained under the numerical
> distortions it will actually experience at Q4-class inference?

Not "can it score higher on `call_f1`" — PTQ Q4 already does that while being a worse policy.

## Experiment namespace

QAD is a separate family. **No existing PTQ artifact is modified.**

```
results/quantization/
  ptq/                     # existing, CLOSED, immutable
  qad/
    configs/               # frozen per-run config, one file per run
    runs/                  # training logs, loss curves, checkpoints metadata
    artifacts/             # exported quantized artifacts + hashes
    reports/               # per-run evaluation
```

## Three representations, never conflated

| # | representation | what it is |
|---|---|---|
| 1 | normal post-training model | the BF16 parent, `dpo-checkpoint-30` — the teacher |
| 2 | fake-quantized / QAD training representation | student with quantization simulation active; **still BF16/FP tensors**, not a deployable artifact |
| 3 | final deployed quantized representation | the exported Q4-class GGUF; the only thing that may carry a behavioural verdict |

A metric measured on (2) is **not** a claim about (3). Only (3) is scored against the gate.

## Distillation semantics

The BF16 parent is the behavioural **teacher**; no other teacher without a documented reason.
Terminology follows upstream NVIDIA/ModelOpt usage of *Quantization-Aware Distillation*.

Token cross-entropy against the original dataset is **not** distillation and does not satisfy this
plan. The objective must transfer teacher behaviour (logit/KL-based) while fake quantization is
active in the student. The teacher runs under `torch.no_grad()`, holds no `requires_grad=True`
parameter, and is absent from the optimizer — invariants already covered by the planned tests.

## Per-run record (mandatory)

Every run records: parent checkpoint · teacher checkpoint · quantization scheme · quantizer config ·
fake-quant config · calibration data hash · training data hash · seed · optimizer · LR schedule ·
batch size · sequence length · training tokens and examples · steps · precision · GPU · software
revisions · checkpoint hash · final artifact hash.

## Stage 1 — bounded diagnostic (the only stage authorised without a further decision)

**Budget: 300 optimizer steps, one A100-80GB, single run.** Not a hyperparameter search.

Purpose is to answer five questions, none of which require the held-out set:

1. Does training remain numerically stable (no NaN/inf, no loss divergence)?
2. Does the quantization-aware loss actually decrease?
3. Is the exported Q4 artifact **materially different** from PTQ Q4 — different weights, different
   hash, different generations?
4. Does policy behaviour move **toward** BF16 on development data?
5. Are gains visible on DEV without touching the confirmatory partition?

Checkpoints at steps 25/50/100/150/200/300. Selection on the **QAD validation split only**.

**The confirmatory partition is not touched during Stage 1.** It is the final verdict, not a
hyperparameter oracle. Repeatedly evaluating it while tuning would silently convert it into
training signal.

## Stage 2 — escalation (requires an explicit go/no-go on Stage 1 evidence)

Only if Stage 1 shows genuine movement of the behavioural vector toward BF16. Budget to be fixed
before starting, in this file, as an amendment.

## Promotion criteria

A QAD candidate is compared against **both**:

* **A. BF16 reference** — the behavioural target
* **B. PTQ-Q4_K_M** — the equivalent-bit baseline it must beat to justify existing

Reported for every candidate, as absolute and relative deltas against **both** A and B:

`call_f1` · `call_precision` · `call_recall` · `over_call_rate` · `clarification_accuracy` ·
`unsupported_accuracy` · valid-output / tool-call validity rate · per-example decision agreement ·
artifact size · compression ratio · prompt-processing throughput · decode throughput ·
peak RAM/VRAM where measurable.

Acceptance uses the **existing frozen gates, unchanged**:

* primary: `quantization_preservation_v1` — the same gate Q6_K passed
* secondary: `gguf_release_selection_v1` — the bar no PTQ rung met

**No QAD-specific threshold may be invented after observing results.** If a new acceptance rule is
ever needed it must be frozen as a new policy version, before the candidate is scored.

### Outcome grading, fixed in advance

| outcome | meaning |
|---|---|
| **major success** | passes the same primary gate as Q6_K while remaining Q4-class size |
| **strong success** | additionally passes the secondary release bar |
| **meaningful positive** | materially recovers `over_call_rate`, `clarification_accuracy`, `call_precision` and decision agreement at Q4 size, even if the primary gate still fails |
| **negative result** | none of the above — a valid, publishable conclusion |

## Stop conditions

Stop QAD work when any of these holds. These exist to reach a negative result quickly rather than
to keep searching until something looks positive.

- successive runs do not materially improve behavioural fidelity
- improvements are within run-to-run noise
- gains appear **only** in `call_f1` — the known-misleading metric
- `over_call_rate` remains badly degraded
- `clarification_accuracy` remains collapsed
- the resulting artifact is no smaller than an already-accepted rung
- training or export complexity outweighs the 18.1% deployment ceiling

**"QAD failed to recover Q4 policy fidelity under the tested budget" is a complete result.**
Training will not be extended to manufacture a positive one.

## Explicitly out of scope

MediaTek port work is a **separate track** and must not be mixed into this experiment
(see [`MEDIATEK_PORT_STATUS.md`](MEDIATEK_PORT_STATUS.md)). No new dataset, no DPO, no policy
rewrite, no template or tokenizer change.
