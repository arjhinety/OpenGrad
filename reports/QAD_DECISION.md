# QAD decision record — M1-v2

**Decision date basis:** the closed PTQ phase (`manifests/quantization/ptq_phase_closure_v1.json`).
This record is written *before* any QAD training so the justification cannot be reverse-engineered
from a result.

```
QAD_REQUIRED_FOR_RELEASE            = false
QAD_JUSTIFIED_AS_COMPRESSION_RESEARCH = true
QAD_RECOMMENDED_TO_RUN_NOW          = deferred — see "The prize is smaller than assumed"
```

## Why QAD is not required for release

| requirement that would force QAD | status |
|---|---|
| a Q4-class artifact for practical device deployment | **not established** — no frozen deployment requirement states Q6_K is too large |
| substantially lower memory than Q6_K | **not achievable by QAD** — see below, the gap is 0.263 GiB |
| recovery of the severe Q4 policy degradation | real, but this is a *research* goal, not a release blocker |
| a deployable representation meeting the same contract at lower precision | Q6_K already meets the primary contract |

`Q6_K` passes the frozen `quantization_preservation_v1` gate at 1.450 GiB (2.43× compression), and
`Q8_0` provides a more conservative option at 1.874 GiB. A shippable, gate-passing GGUF exists
today. Nothing about the release depends on QAD.

## The prize is smaller than assumed

This is the finding that most affects the decision, and it was not visible before the ladder was
built.

| rung | GiB | vs Q6_K |
|---|---:|---:|
| Q2_K | 0.902 | −37.8% |
| Q4_K_M | 1.187 | **−18.1%** |
| **Q6_K (accepted)** | **1.450** | — |
| Q8_0 | 1.874 | +29.2% |

**The whole 2-bit-to-8-bit ladder spans only 2.08× in artifact size, while nominal bits per weight
span 4×.** The cause is architectural and was measured directly in the ExecuTorch audit: the tied
embedding matrix is 248,320 × 2,048 ≈ 508.6M parameters — roughly 27% of the model — and k-quants
keep `token_embd` at higher precision than the transformer blocks. Compression therefore saturates.

So the *entire* achievable benefit of a successful QAD-Q4 programme, assuming it fully recovers
BF16 policy behaviour, is **0.263 GiB against an artifact that already passes the gate**. Even the
absolute floor (Q2_K, which fails catastrophically) is only 0.547 GiB below Q6_K.

That does not make QAD scientifically uninteresting. It does mean the deployment case for it is
weak, and that the honest framing is compression research rather than release engineering.

## Why QAD remains justified as research

The Q4 failure is not a mild degradation, and its *shape* is the interesting part:

| metric | BF16 | PTQ Q4_K_M | direction |
|---|---:|---:|---|
| `call_f1` | 0.7572 | **0.7594** | *better* |
| `call_precision` | 0.7344 | 0.6612 | −10.0% |
| `over_call_rate` | 0.1553 | 0.2512 | +61.7% worse |
| `clarification_accuracy` | 0.7682 | 0.6388 | −16.8% |
| decision agreement vs BF16 | — | 0.872 | 164 flips, 76 broke |

PTQ Q4 **improves the headline metric while the policy degrades underneath it.** The model calls
tools far more often and clarifies far less; `call_f1` rises because recall rises, and recall rises
because it over-calls. This is a clean demonstration that **`call_f1` alone must never be used as
the quantization acceptance criterion**, and it is a well-posed research question: can training
under simulated low-bit distortion recover the *policy vector*, not the headline number?

## The objective, stated so it cannot be gamed

The QAD objective is **not** "beat PTQ-Q4_K_M on `call_f1`". PTQ Q4 already beats BF16 on `call_f1`
while being a worse policy, so that target is actively misleading.

The objective is to recover the behavioural vector — `call_precision`, `call_recall`,
`over_call_rate`, `clarification_accuracy`, `unsupported_accuracy`, valid-output rate, and
per-example decision agreement — while retaining Q4-class size.

## Why QAD rather than plain QAT

QAT and QAD share all the expensive machinery — fake quantization, the straight-through estimator,
the export path. **Only the loss differs**, so this is a loss-function choice, not an architectural
one, and switching later is cheap.

QAD is chosen here for a reason specific to this checkpoint:

**The policy being preserved came from DPO, not SFT.** M1-v2 is SFT → `dpo-checkpoint-30`, and the
DPO stage is what produced the low `over_call_rate` of 0.1553. The measured Q4 failure mode is
exactly that number reverting to 0.2512. Plain QAT on the SFT corpus with token cross-entropy pulls
the student back toward the **pre-DPO distribution — the one that over-called** — so it would
plausibly reinforce the failure the experiment exists to fix. A QAT variant that preserved the
policy correctly would need the DPO objective running under fake quantization, which reintroduces
preference data and is explicitly excluded above as a confound.

QAD avoids the proxy entirely: the BF16 teacher **is** the post-DPO policy, immutable and hashed
(`903f9b11…`). Matching its outputs targets the quantity of interest directly.

Two secondary advantages:

* **Signal density.** KL against the teacher's full next-token distribution carries far more
  information per token than a hard label, which matters under the 300-step diagnostic budget.
* **Teacher cost.** Zero — it already exists and is frozen.

**QAT would be preferable if** no teacher existed, or if the goal were to *exceed* BF16 behaviour at
low bits rather than preserve it. Neither applies. A hybrid (KL + a small CE term) is the obvious
variant, but any CE weight reintroduces the pre-DPO pull described above, so it is not part of
Stage 1.

## Scope discipline

QAD is a **new experiment family**. It does not overwrite, edit, or supersede any PTQ artifact or
verdict. The frozen PTQ rows stay visible even if a QAD artifact later outperforms them.

Explicitly excluded from this experiment, because each would confound attribution:
no new tool-use dataset, no DPO, no policy rewrite, no chat-template change, no tokenizer change,
no synthetic-data expansion.

## Recommendation

Record the decision as above and hold the substantive GPU spend. The bounded diagnostic protocol is
specified in [`QAD_EXPERIMENT_PLAN.md`](QAD_EXPERIMENT_PLAN.md) and is ready to run, but committing
to it should be a deliberate choice given an 18.1% ceiling on the deployment benefit. A negative
result — "QAD did not recover Q4 policy fidelity under the tested budget" — is an acceptable and
publishable outcome, and the stop conditions are written to reach it quickly rather than to
manufacture a positive one.

## What would change this decision

- A frozen deployment requirement showing 1.450 GiB is too large for a target device.
- Evidence that Q4-class decode throughput or memory on a real device differs enough from Q6_K to
  matter — currently unknown, because all throughput figures are A100 reference numbers and
  OpenWeights device validation has not run.
- A quantization scheme that also compresses `token_embd` without destroying behaviour, which would
  raise the ceiling above 18.1% and change the arithmetic entirely.
