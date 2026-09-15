# 02 — Research questions and hypotheses

Study 002 answers a question the programme already carries as **RQ3 — Regression**
(`docs/research/research-program.md:14`): *"What ordinary instruction-following, reasoning,
calibration, latency, or robustness capabilities regress as tool reliability improves?"* Study 001
answered the observation half of RQ3 — a capability regression the promotion gate could not see — but
admitted it could not answer the mechanism half: *"It is a general-capability regression associated
with tool-policy post-training on When2Call-derived data; causation is not established (one lineage,
one seed, no replicate)"* (`ROADMAP.md:102-104`).

Study 002 is the mechanism half. It also feeds back into **RQ1 — Reliable tool policy**
(`docs/research/research-program.md:12`): an intervention that repairs direct answering while
degrading the tool policy is not a fix (`ROADMAP.md:139-142`).

This document fixes the questions and hypotheses. Thresholds are in [11](11-THRESHOLDS.md); the arms
that test them are in [04](04-ARM-MATRIX.md).

## Questions

| RQ | Question | Type | Primary artifact |
|---|---|---|---|
| **RQ2.1 — Mechanism** | Does refusal-targeted supervision whose decision label is `ANSWER` and whose target text is a refusal *causally* produce the direct-answer collapse? | causal, confirmatory | [04](04-ARM-MATRIX.md), [11](11-THRESHOLDS.md) |
| **RQ2.2 — Boundary** | Is the collapse a property of the model's *decision* under a given prompt, or of its *ability*? Why does 8-shot prompting restore 55.5% accuracy where 0-shot yields 0%? | mechanistic, characterising | [08](08-SENTINEL-SPEC.md), [09](09-BENCHMARK-PLAN.md) |
| **RQ2.3 — Correction** | Does changing only the label and target contract of those records — same records, same compute, same exposure — restore direct answering? | causal, confirmatory | [04](04-ARM-MATRIX.md) |
| **RQ2.4 — Dose** | Does collapse magnitude scale with the *share* of refusal-`ANSWER` records in the supervised mixture? | dose-response, exploratory | [04](04-ARM-MATRIX.md) |
| **RQ2.5 — Side effects** | What does the intervention do to *correct* refusal on genuinely unanswerable prompts, to clarification, and to over-call? | safety, confirmatory | [07](07-METRIC-SPEC.md), [11](11-THRESHOLDS.md) |
| **RQ2.6 — Replication** | Does the mechanism hold at a second model scale or a second corpus construction? | replication, exploratory | [14](14-HETEROGENEITY-POLICY.md) |

RQ2.1, RQ2.3 and RQ2.5 are the **confirmatory** questions. RQ2.2, RQ2.4 and RQ2.6 are declared
**exploratory**: reported with intervals, but carrying no study-level verdict, because a verdict
requires a design that varies one factor alone and most of them do not.

## Hypotheses

Each hypothesis names what would falsify it. A hypothesis no arm in [04](04-ARM-MATRIX.md) can falsify
is not listed.

### H1 — The refusal-`ANSWER` disagreement is sufficient to induce the collapse
Training on records whose label says `ANSWER` and whose target is a refusal teaches the boundary
"answering looks like declining". Under matched compute, exposure and mixture, a corpus in which only
those labels and targets are corrected will not collapse.

**Falsified if:** the corrected corpus (arm `R1`) still collapses zero-shot direct answering, measured
by the sentinel endpoint of [11](11-THRESHOLDS.md); or if the collapsed reference corpus (`C0`)
reproduces the collapse without any such records.

**Competing explanation.** *Any* narrow tool-policy post-training on this mixture might degrade general
capability. Arms `C1` (mixture without synthetic/back-translated records) and `C2` (mixture away from
When2Call) exist to separate these explanations rather than assume H1.

### H2 — The collapse is a decision effect, not an ability loss
The 8-shot/0-shot asymmetry (0% vs 55.5% on the same 1,319 items) is a prompt-conditional *decision*
failure: the capability is present and a context demonstrating an answering mode recovers it.

**Falsified if:** 8-shot accuracy also falls to near zero, or the explicit answer-elicitation prompt in
[08](08-SENTINEL-SPEC.md) fails to recover a substantial fraction. Falsification here is a materially
worse finding than H1 — an ability loss rather than a policy loss — and must be reported as such,
including in any model card.

### H3 — Collapse is monotone in exposure to the flagged mixture
**Falsified if:** the dose arms are not ordered, or the ordering is inside the seed interval
([10](10-STATISTICS-PLAN.md)).

### H4 — Relabelling alone restores direct answering; correcting the target text is not required
If H1 holds, the defect is the *label*. A corpus whose flagged labels are corrected but whose target
text is untouched should recover most of the effect.

**Falsified if:** label-only correction (`R2`) recovers substantially less than label+target correction
(`R1`). The difference is then itself a finding about which part of the training contract carries the
behaviour — and it would change the recommended correction for the published corpus.

### H5 — The correction does not break the tool policy
**Falsified if:** `call_f1`, `call_precision`, `over_call_rate` or `clarification_accuracy` degrade
beyond the frozen thresholds, or any of them is unmeasurable rather than measured
([07](07-METRIC-SPEC.md)).

### H6 — The correction does not increase hallucination on genuinely unanswerable prompts
The intervention must not convert correct refusals into confident fabrication. This is a *safety*
hypothesis, evaluated with a metric Study 001 never had: **refusal correctness**
([07](07-METRIC-SPEC.md)).

**Falsified if:** refusal correctness on the curated unanswerable set drops beyond threshold, or
`CANNOT_ANSWER` classification accuracy degrades. A falsification here blocks promotion regardless of
what the direct-answer endpoint does — the failure mode it protects against is harder to detect than
over-refusal and worse in consequence (`ROADMAP.md:120-124`).

### H7 — The mechanism replicates at a second scale or corpus construction
Declared exploratory. Failure to replicate is reported as a scope limit on H1, not as a defect. The
Study 001 lesson is that a single lineage cannot support a general claim (G13).

## Claims this study will not make

- It will not claim a general law about post-training. The result is local: this corpus construction,
  this model family, these scales.
- It will not claim that refusal supervision is *the only* cause. If `C1` or `C2` also collapse, the
  claim narrows to "refusal-`ANSWER` disagreement is one sufficient cause", and the alternative is
  reported as a live competing explanation with its own arm and its own number.
- It will not claim the intervention is a net win unless H5 and H6 both hold. A
  `TOOL_POLICY_REGRESSION` or `SAFETY_REGRESSION` verdict is a publishable outcome —
  `docs/research/STUDIES.md:72-73` already commits to this: *"A null result is publishable."*
- It will not describe any result as an ability change on the strength of the refusal endpoint alone.
  Ability, decision and format are separated by the metric set in [07](07-METRIC-SPEC.md).

## Verdict vocabulary

The study-level verdict is one of, and is computed rather than written
([11](11-THRESHOLDS.md), [15](15-PROVENANCE-VALIDATORS.md)):

| Verdict | Meaning |
|---|---|
| `MECHANISM_SUPPORTED_RELABEL_RECOMMENDED` | H1 and H4 hold; the label is the defect; corpus correction is the recommended action |
| `MECHANISM_SUPPORTED_CORRECTION_INSUFFICIENT` | H1 holds but neither `R1` nor `R2` restores direct answering; the label is implicated but not sufficient |
| `MECHANISM_NOT_SUPPORTED` | `C0` does not collapse, or a control corpus collapses without the flagged records |
| `MECHANISM_INCONCLUSIVE` | H1 cannot be separated from the competing explanations within the seed interval |
| `TOOL_POLICY_REGRESSION` | a candidate fails H5 under the frozen `tool_use_promotion_v5` gate |
| `SAFETY_REGRESSION` | a candidate fails H6 |
| `NOT_EVALUABLE` | a required mode or endpoint is unmeasured (`UNMEASURED`), or a gate is vacuous |

`NOT_EVALUABLE` is not a null result and may not be reported as one. It is the verdict Study 001 would
have produced had the ANSWER mode been a required, non-vacuous dimension
(`src/opengrad/promotion/tool_use_policy.py:33-37`).