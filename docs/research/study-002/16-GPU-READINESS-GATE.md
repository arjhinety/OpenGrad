# 16 — GPU readiness gate

This is the gate that must return `READY` before any training run is launched. It restates
`docs/research/GUARDRAILS.md:134-141` (pre-GPU) and `:143-158` (pre-freeze) as blocking, machine-checkable
conditions, and it is deliberately able to say **no**.

The governing precedent is `reports/PRE_SFT_READINESS_REPORT.md:8-11`: *"This report is left as it stood at
launch, because a readiness record that is edited after the run stops being evidence that readiness was
established **before** it."* Study 002's readiness record is written once, before launch, and never edited
afterwards; a correction is a new record that supersedes the old one by reference.

## Blocking checks

| # | Check | Evidence artifact | Pass condition | Maps to |
|---|---|---|---|---|
| 1 | preflight `READY` | `gpu_preflight` record | `status: READY`; `compatibility.result` = `COMPATIBLE` for a primary arm | [13](13-HARDWARE-AGNOSTIC-EXECUTION.md) |
| 2 | gate version frozen | [11](11-THRESHOLDS.md) at the prereg commit | `study_002_gate_v1` / `tool_use_promotion_v5` present, with no post-result amendment | G1 |
| 3 | all four modes covered | `P-CONF` gold-count table | every mode `n > 0`; worst-case resolvable margin ≤ the margin it is used to test | G2, [06](06-SPLIT-SPEC.md) |
| 4 | `ANSWER` strata materialized and disjoint | strata manifest + contamination check | fingerprint read from artifact; no collision with a training corpus; no benchmark item (`prohibit_training: true`) | [06](06-SPLIT-SPEC.md), [09](09-BENCHMARK-PLAN.md) |
| 5 | `P-UNANS` curated with agreement | labeller-agreement record | agreement floor met; disagreements removed and counted | [03](03-PREREGISTRATION.md) stop rule 2 |
| 6 | detector measured | `P-DET` labels + precision/recall | precision ≥ floor in [11](11-THRESHOLDS.md); sample ≥ 300, stratified, ≥ 3 arms | [07](07-METRIC-SPEC.md) |
| 7 | corpus fingerprint recorded | corrected corpus artifact | `content_hash` produced by repository code; per-source counts; disposition map for all 18,114 flagged records | G4, [12](12-ARTIFACT-RETENTION.md) |
| 8 | exposures matched and measurable | batch-composition logs | supervised tokens measured per arm; the token-budgeted geometry that makes *steps* a non-control is identified | G5, `#6` |
| 9 | seeds fixed | [05](05-SEED-AND-REPRODUCIBILITY-POLICY.md) | `k = 3`, identical values across arms; `REP-A` / `REP-B` planned | G11 |
| 10 | sentinels registered | sentinel registration | every sentinel in [08](08-SENTINEL-SPEC.md) declared with its mode and blocking status | G2 |
| 11 | validators exist **and have failing tests** | [15](15-PROVENANCE-VALIDATORS.md) negative tests | every validator fails on its fixture with the expected code | [15](15-PROVENANCE-VALIDATORS.md) |
| 12 | retention paths exercised | `provider: cpu` smoke run | every required artifact in [12](12-ARTIFACT-RETENTION.md) written by a real run | [12](12-ARTIFACT-RETENTION.md) |
| 13 | **the gate can fail** | vacuity self-test | the gate returns FAIL on an empty or synthetic population | [03](03-PREREGISTRATION.md) stop rule 4 |
| 14 | cost ceiling and accounting | cost ledger | budget stated; discarded-work accounting in place | [09](09-BENCHMARK-PLAN.md) |

Fourteen checks, every one blocking, every one with an artifact rather than a statement. A check that cannot
be evaluated is `BLOCKED`, which is not a pass.

> **Status, 2026-09-20.** Checks 2 and 13 now have code. `study_002_gate_v1` and
> `tool_use_promotion_v5` are implemented (`src/opengrad/verification/study_002_gate.py`,
> `src/opengrad/promotion/tool_use_policy.py`), and the gate's vacuity self-test runs
> (`python -m opengrad.verification.study_002_gate --self-test`), failing on the three fixtures named
> below. The gate reports `BLOCKED_INPUT_MISSING` until a four-mode evaluation bundle exists, so this is
> not yet a `READY` record: checks 3–6 and 8–14 still have no artifact. No arm has launched.

## The self-test (check 13) is the most important one

`study_002_gate_v1` must be shown to return **FAIL** on: a population with an empty required mode; a metrics
record carrying `no_call_accuracy: 0.0` beside a zero `ANSWER` row; and a missing sentinel. A gate that has
never been observed failing is a gate whose failure path is untested.

Study 001's promotion gate is the demonstration of both directions. `tool_use_promotion_v2` was not vacuous in
spirit — it asserted a floor it meant to enforce — but it could not distinguish an empty class from a failed
one (`tool_use_policy.py:33-37`), so it rejected every candidate unconditionally and then had to be relaxed in
v3. A gate that can only fail is as useless as one that cannot fail, and both are tested here.
## Pre-freeze checklist

`docs/research/GUARDRAILS.md:143-158` governs the freeze that ends the study. Its conditions are met by:

| Requirement | Where satisfied |
|---|---|
| verdict computed from artifacts, not written | [11](11-THRESHOLDS.md) verdict function |
| every claim attached to a number and an artifact | [15](15-PROVENANCE-VALIDATORS.md) `V10` |
| unpublishable claims published, not deleted | [01](01-LESSONS-FROM-STUDY-001.md) L13, [20](20-CLOSURE-REPORT.md) |
| errata applied everywhere | `reports/ERRATA.md`, [03](03-PREREGISTRATION.md) amendment procedure |
| limitations stated in the artifact, not only in prose | [13](13-HARDWARE-AGNOSTIC-EXECUTION.md) `limitations[]`, [14](14-HETEROGENEITY-POLICY.md) |
| the published corpus is not silently mutated | [12](12-ARTIFACT-RETENTION.md) |

## What blocks what

| Gate status | Consequence |
|---|---|
| `READY` | the arm set may launch, in tier order ([04](04-ARM-MATRIX.md)) |
| `BLOCKED` | launch refused; the blocking check is named with its evidence gap |
| `INCOMPLETE` | launch refused; checks that have not run are listed as not-run rather than assumed |
| `NOT_RUN` | launch refused; a study launched without a readiness record has no readiness evidence |

No partial credit. A study that launches with eleven of fourteen checks passing has eleven checks of evidence
and no readiness.

## Cost basis, estimate and recording

### The anchor

| Item | Value | Confidence |
|---|---|---|
| Accelerator | A100 80GB | as used |
| Provider / billing | Verda, on-demand | as used |
| Hourly rate | **$1.79 / GPU-hour** | rate at the time; the rate is a recorded field, not an assumption |
| Work covered | the whole M0 → M1 training | SFT + DPO, as one lineage |
| Runtime | **≈ 1.5 days ≈ 36 GPU-hours** | **approximate — the author's recollection, not a measured ledger figure** |
| Derived cost | **≈ $64.44** (36 h × $1.79) | arithmetic on the two approximate inputs above |

This is a **basis, not a measurement**. It is recorded here because an order-of-magnitude figure that is
labelled approximate is worth more than no figure, and because the previous study's readiness record had
none. Every number derived from it below inherits the same caveat and is reported as an envelope.

### Why the rate must be recorded per run, not assumed

The Study 001 diagnosis campaign is in the ledger at a **different** rate:
`results/benchmarks/h200/capability_v1/cost_ledger.json` records `gpu: NVIDIA H200`,
`gpu_hourly_usd: 4.54`, `total_container_seconds: 7055.4`, `continuation_total_usd: 8.898`. The implied rate
reconciles exactly (8.898 ÷ 1.9598 h = $4.54/h), so the ledger is internally consistent — and the same work
priced at the two providers differs by **2.54×**:

| Work | H200 @ $4.54/h | A100 @ $1.79/h |
|---|---|---|
| diagnosis continuation (7,055.4 container-seconds) | $8.898 (measured) | ≈ **$3.51** (derived) |
| one MMLU-Pro @2048 stage run (732.53 container-seconds) | $0.9238 (measured) | ≈ **$0.36** (derived) |

So "the cost of a run" is undefined without the rate. Every Study 002 run record carries
`gpu_hourly_usd` and the provider, and a cost comparison across two rates must state both.

### Estimate for the arm set

The arm set is 33 training runs plus 2 repeats = **35 runs** ([04](04-ARM-MATRIX.md)). The anchor prices one
**whole M0 → M1 lineage**; a Study 002 arm is an SFT retrain, so the per-arm cost is *below* $64.44 — but by
how much is unknown, because the SFT/DPO split inside the 36 hours was never recorded. The split is the
single input that would tighten this estimate, and it is the first thing to measure on the first real run.

| Assumption for one arm | Per arm | 35 runs | Status |
|---|---|---|---|
| SFT ≈ ⅓ of the anchor (12 h) | $21.48 | **$751.80** | unverified split |
| SFT ≈ ½ (18 h) | $32.22 | **$1,127.70** | unverified split |
| SFT ≈ ⅔ (24 h) | $42.96 | **$1,503.60** | unverified split |
| arm = whole lineage (36 h) | $64.44 | **$2,255.40** | absolute upper bound, not a prediction |

Evaluation is additional and small but not zero: the one measured data point is a single MMLU-Pro @2048 stage
run at 732.53 container-seconds, so a per-arm evaluation pass of `ifeval` + `gsm8k` × 3 modes + `mmlu-pro` +
`when2call-eval` is **≈ $1–3 per arm**, i.e. **≈ $35–105** across 35 runs at the Verda rate. That is an
estimate from one measured input, and it is labelled as one.

**Working planning envelope: ≈ $800–1,600 excluding evaluation, ≈ $900–1,700 including it.** It is stated as a
range because the phase split is unknown, and the range is wide enough to be honest about that.

### The envelope is not a balance

The repository already learned this, in its own words, and Study 002 inherits the field and the rule:

- `envelope_is_not_a_balance: true`, with the note: *"NOT an account balance and NOT available credit. The two
  diverged badly in this run: planning continued against the envelope while actual remaining credit was far
  lower. **Do not size a run from this field.**"*
- `REMAINING_CREDIT_BALANCE: NOT_QUERYABLE`, with *"`modal billing` exposes metered consumption and credits
  applied, not a remaining balance. No balance is asserted."*

Consequently check 14 is satisfied by a **quantity available**, not by the envelope: before the first launch,
the actual available credit or spend authorisation is confirmed and recorded, and the planning envelope above
is recorded beside it as a separate field. `ROADMAP.md` step 16 item 6 says the same thing: *"Confirm actual
available credit — not a planning envelope — before committing."*

### What is recorded, per run

The existing ledger schema is inherited unchanged — `schema_version`, `gpu`, `gpu_hourly_usd`,
`total_container_seconds`, `campaign_total_usd`, `per_stage`, `retained_runs_usd`, `superseded_runs_usd`,
`superseded_runs_seconds`, `infrastructure_waste_usd`, `failed_launches`, `envelope_*` — with its
distinctions kept intact, because each one was bought with a real mistake:

- **superseded ≠ waste.** *"A superseded run is NOT infrastructure waste when its results are retained."*
  The 768-token MMLU-Pro pass is superseded and counted in full, because hiding it would make the next
  study's budget look cheaper than this one's history.
- **waste is separated and asserted, not assumed.** *"Every failed launch in this continuation died
  client-side or at module import, before a GPU was allocated. Cost $0.00 — asserted from the run log, not
  assumed."*
- **container overhead is not loaded onto a benchmark.** *"Model download, weight verification, engine start
  and teardown... reported separately rather than loaded onto a benchmark's cost-per-result."*
- **CPU work is $0 GPU cost** and is reported as such, not folded in.

Study 002 adds four fields per entry: `arm`, `seed`, `tier`, and `disposition`
(`RETAINED` / `SUPERSEDED` / `REPLACEMENT` / `DISCARDED`), so that a reader can total the cost of the
confirmatory claim separately from the cost of the exploratory tiers, and can see the price of the work that
produced nothing.

## The gate is not a formality

Three of its fourteen checks exist because the corresponding claim was false in Study 001 at the time it was
published:

- check 3 (mode coverage) exists because the population could not measure the mode a gate asserted a floor
  on, and the promotion went through anyway;
- check 11 (validators with failing tests) exists because a gate in this repository *"could report PASS
  without running"*;
- check 13 (the gate can fail) exists because a gate that cannot fail is indistinguishable from one that did
  the work and found nothing wrong.

Each of those is a documented defect rather than a hypothetical, and each is cited above at its source.