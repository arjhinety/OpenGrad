# 20 — Closure report

**Status: `DESIGN_CLOSED` / `RUN_NOT_STARTED`.** This report closes the design phase and states explicitly that
no training or evaluation has been performed for Study 002. Two closures exist and must not be confused:

| Closure | Meaning | Status |
|---|---|---|
| `DESIGN_CLOSED` | the design set is complete, internally consistent and pre-registered | **reached** |
| `RUN_CLOSED` | the arms ran, the verdict is computed, the artifacts are retained | **not reached** |

No number in this study has been produced yet. A reader looking for a result should look at the verdict
vocabulary in [02](02-RESEARCH-QUESTIONS.md) and the gate in [11](11-THRESHOLDS.md), not for one.

## Adversarial self-review

Five perspectives, each asked to attack the design, with the criticisms that survived and what changed because
of them. Criticisms answered rather than acted on are listed with the answer, so a reader can judge whether
the answer was adequate.

### P1 — The statistician

| Criticism | Response | Change |
|---|---|---|
| `k = 3` cannot estimate seed variance, yet the plan reports a mean over seeds. | Correct, and the plan says so: no standard deviation is computed, seed uncertainty is the range, and sign agreement is required. The *mean over three seeds* is still a summary with no interval. | [10](10-STATISTICS-PLAN.md) prints per-seed values prominently and marks the mean as a summary, not an estimate |
| Margins were compared against a noise band that does not exist yet. | True: `REP-A` has not run. The comparison is defined and is a pass/fail precondition, not a claim. | [05](05-SEED-AND-REPRODUCIBILITY-POLICY.md) makes `REP-A` the rejection test for any margin smaller than the residual |
| The `over_call_rate` row resolves only 6.9pp on the existing partition, below the 10pp floor. | A real finding, produced by doing the arithmetic rather than by asserting adequacy. | [11](11-THRESHOLDS.md) marks that row `WITHIN_NOISE` below 6.9pp unless the partition is enlarged |
| Two families with Holm correction is a weak control for ~10 comparisons. | Accepted as a limitation; the primary claim is a single pre-registered comparison with an interval, and the corrected family guards only secondary claims. | [10](10-STATISTICS-PLAN.md) names both families and bars exploratory results from the abstract |

### P2 — The safety reviewer

| Criticism | Response | Change |
|---|---|---|
| The intervention could convert correct refusals into confident fabrication, while the headline is the direct-answer restoration. | Accepted. This is why H6 exists and why `refusal_correctness` is a new metric rather than a caveat. | [07](07-METRIC-SPEC.md) defines `refusal_correctness`; [11](11-THRESHOLDS.md) makes `SAFETY_REGRESSION` a blocking verdict |
| The refusal detector is precision-biased, so a rise in `answer_rate` could be partly an artefact. | Accepted and made directional: false negatives inflate `answer_rate`, which is the direction that favours H1. | [07](07-METRIC-SPEC.md) states H1 is the optimistically biased direction, requires measured recall, and H1's criterion includes `delta_no_call_acc` |
| `P-UNANS` does not exist, so the safety metric's population is unverified. | Accepted as the study's largest unbuilt dependency. | [03](03-PREREGISTRATION.md) stop rule 2 halts the study if inter-labeller agreement is not reached; [06](06-SPLIT-SPEC.md) requires the agreement record |

### P3 — The systems and reproducibility reviewer

| Criticism | Response | Change |
|---|---|---|
| Six of ten scaffold components are `NOT_IMPLEMENTED`, so this is a plan rather than a pre-registration. | Correct. That is why the deliverable is a design set with a pre-GPU gate, and why every `NOT_IMPLEMENTED` row is a blocking check rather than an assumption. | [04](04-ARM-MATRIX.md) scaffold table; [16](16-GPU-READINESS-GATE.md) checks 6, 11, 12 |
| "Matched exposure" is repeated from Study 001, which measured 1.19× for a nominally matched arm. | Accepted, with the mechanical correction: batches are token-budgeted, so **steps cannot hold tokens fixed**; only measured supervised tokens can. | [04](04-ARM-MATRIX.md) states this and downgrades any unmatched comparison to descriptive |
| The new gate could pass vacuously — the very defect being fixed. | Accepted, and it is why check 13 is a *self-test*: the gate must be observed failing. | [16](16-GPU-READINESS-GATE.md) check 13; [15](15-PROVENANCE-VALIDATORS.md) negative tests |
| Hardware-agnostic execution is claimed but only one provider runs the primary arms. | True. The claim is *runnable* on any provider plus a two-provider heterogeneity check on `C0`/`R1`; cross-provider claims require `HOMOGENEOUS`. | [13](13-HARDWARE-AGNOSTIC-EXECUTION.md), [14](14-HETEROGENEITY-POLICY.md) |
### P4 — The epistemics and audit reviewer

| Criticism | Response | Change |
|---|---|---|
| The study cites its own prior work 90+ times. Does volume substitute for argument? | Partly a fair hit. Each citation is a defect with a `file:line` and a mechanism it motivates; the defect → rule → mechanism column in [01](01-LESSONS-FROM-STUDY-001.md) exists so a reader can check the inference instead of accepting the citation. | [01](01-LESSONS-FROM-STUDY-001.md) mechanism table |
| The scope split contradicts a published scope statement. | Accepted and recorded rather than silent: `STUDIES.md:59-83` assigned three subjects to Study 002; this design set splits them. | [README](README.md) records the decision; `STUDIES.md` is updated in the same commit |
| The competing explanations are the authors' own choices, so the study can still miss the true cause. | Accepted as a permanent limit. Mitigations: `C1`, `C2` and `X1` run regardless of the Tier A outcome, and a non-collapsing control narrows the claim instead of being ignored. | [04](04-ARM-MATRIX.md) Tier B runs unconditionally; [02](02-RESEARCH-QUESTIONS.md) states the narrowing |
| A null relabelling result is pre-declared publishable, which could launder a failed intervention. | The pre-declaration is what makes it honest. The failure to fear is an unpublishable null being renamed a partial success, and the verdict vocabulary has no slot for that. | [02](02-RESEARCH-QUESTIONS.md) verdict vocabulary; [03](03-PREREGISTRATION.md) stop rules |

### P5 — The reader and reviewer

| Criticism | Response | Change |
|---|---|---|
| Eleven arms and 33 runs with no cost estimate. | Partly addressed. An anchor now exists (A100 80GB / Verda / $1.79 per GPU-hour / ≈1.5 days for the whole M0 → M1 lineage / ≈$64.44), and [16](16-GPU-READINESS-GATE.md) derives a planning envelope of ≈$800–1,700 for 35 runs. The range is wide because the SFT/DPO phase split was never recorded, and that is named as the missing input rather than smoothed over. | [16](16-GPU-READINESS-GATE.md) cost basis; open item 1 |
| The mode vocabulary is already inconsistent across documents in the repository. | Accepted, and it is now a validator: `V4 vocabulary_lock` with a published alias table. | [06](06-SPLIT-SPEC.md), [15](15-PROVENANCE-VALIDATORS.md) |
| A reader cannot tell which numbers are new and which are Study 001's. | Accepted, and it is `V11`: mixing a Study 001 frozen number with a Study 002 number in one table fails the build. | [15](15-PROVENANCE-VALIDATORS.md) |
| The design set is long. | It is, and the length sits where Study 001 was thin: populations, denominators, thresholds with margins. [README](README.md) is the entry point; `01`–`02` carry the argument. | — |

## Open items that closure does not resolve

Recorded as unresolved rather than presented as complete:

1. **Cost estimate: a basis now exists, precision does not.** The anchor is recorded — A100 80GB on Verda,
   on-demand at **$1.79/GPU-hour**, the whole M0 → M1 training in **≈1.5 days**, so **≈$64.44** — and
   [16](16-GPU-READINESS-GATE.md) derives a planning envelope of **≈$800–1,700 for the 35-run arm set**
   across four stated assumptions. What is still missing is (a) the **SFT/DPO phase split** inside those ~36
   hours, which is the one input that turns the range into a per-arm figure, and (b) the **actual available
   credit**, which this repository's own ledger rules forbid replacing with an envelope
   (`envelope_is_not_a_balance: true`; *"Do not size a run from this field."*). Both must be supplied before
   launch. The estimate is a basis and is labelled approximate; the recording rules that make the next
   estimate exact are in [16](16-GPU-READINESS-GATE.md).
2. **The `ANSWER` strata source is not chosen.** [06](06-SPLIT-SPEC.md) specifies the properties the source
   must have and points at materialized corpora containing `ANSWER` gold, but names no source. Until one is
   chosen, H1 cannot be adjudicated and the study's central endpoint is unbuilt.
3. **`P-UNANS` does not exist.** H6's population is described but not curated.
4. **`HEURISTIC_REGEX_v2` does not exist**, and neither version's precision has been measured — v1's
   precision-biased design is an intention (`capability.py:28-31`), not a measurement.
5. **No engine-pinning decision.** The study fixes the engine, but which engine is fixed has not been
   recorded.
6. **`REP-A` and `REP-B` are planned, not run**, so every margin-versus-noise comparison is currently a
   precondition rather than a result.
7. **Studies 003 and 004 have no pre-registration.** [18](18-STUDY-003-ROADMAP.md) and
   [19](19-STUDY-004-ROADMAP.md) are roadmaps with draft questions.

An item on this list is not a defect in the design; presenting one as closed would be.

## What closure requires from the run phase

1. Every check in [16](16-GPU-READINESS-GATE.md) passes, with artifacts.
2. The arm set completes in tier order, with Tier B run regardless of Tier A's outcome.
3. The verdict is **computed** by the [11](11-THRESHOLDS.md) function, including `NOT_EVALUABLE` wherever a
   required mode or endpoint is unmeasured.
4. Every claim in the paper passes `V10` (evidence reachable) and `V11` (no cross-study mixing).
5. The unpublishable results are published: negative arms, discarded passes, and findings against this
   study's own design — including, if it happens, that the mechanism was not supported.
6. `reports/ERRATA.md` carries every amendment with its re-run evidence (G15).

## Residual risk, stated plainly

- **The intervention may not work.** The corpus may carry the refusal contract in a way relabelling does not
  remove, or in a way its removal damages capability. Both are pre-registered outcomes with verdict names,
  and neither is a failure of the study.
- **The population may not be fixable at acceptable cost.** If the `ANSWER` strata set cannot be made
  disjoint, balanced and large enough, H1 becomes `NOT_EVALUABLE` — and reporting that is better than
  adjudicating it on a population that cannot resolve it.
- **The result may not generalise.** One lineage, one base model, one corpus construction, two scales at most,
  and three external instruments. G13 is not a formality, and the frozen campaign verdict already says it in
  the author's own words: *"Single lineage, single base model, no replicate; no cross-family generalization
  claimed."* (`results/final_campaign_verdict.json:589-593`)

## The one sentence this study exists to earn

*A checkpoint that refuses to answer questions it can answer was promoted because its evaluation could not
represent the possibility that answering was the right behaviour — and the same class of error, an artifact
that cannot measure what it asserts, is what the 100 unsupported claims in Study 001 had in common.*