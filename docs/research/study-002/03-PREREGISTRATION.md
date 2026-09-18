# 03 — Pre-registration

**`study_002_prereg_v1`.** This document is the frozen decision contract for Study 002. It is
committed **before the first Study 002 training run is launched and before any Study 002 evaluation
result is read**. It exists because Study 001's central methodological error was a decision rule that
arrived after the results it judged: M1-v2 was *"promoted under a parent-relative gate (v4) introduced
after M0 was evaluated"*, and it *"fails the v3 gate that rejected M0"*
(`docs/EXPERIMENT_RESULTS.md:13-16`, finding `#2`). G1 states the rule this document enforces:
*"the gate version and decision rule are committed before the first candidate is scored. A gate
revised afterwards is re-run on every earlier candidate under both versions, and the write-up names
both."* (`docs/research/GUARDRAILS.md:20-22`)

## What this document freezes

| Item | Frozen value | Where |
|---|---|---|
| Research questions and hypotheses | RQ2.1–RQ2.6, H1–H7 | [02](02-RESEARCH-QUESTIONS.md) |
| Arm set and each arm's single varied factor | tiered arm matrix | [04](04-ARM-MATRIX.md) |
| Seed count, seed values, repeat policy | `k = 3`, 2 repeats | [05](05-SEED-AND-REPRODUCIBILITY-POLICY.md) |
| Partition construction and mode coverage | DEV / CONFIRMATORY / SEALED + sentinel splits | [06](06-SPLIT-SPEC.md) |
| Metric definitions, populations, caveat fields | full metric table | [07](07-METRIC-SPEC.md) |
| Sentinel endpoints, prompts and schedule | sentinels, 3 prompt modes | [08](08-SENTINEL-SPEC.md) |
| Benchmark suites and their tiers | `regression_core`, `full_post_training` | [09](09-BENCHMARK-PLAN.md) |
| Statistical analysis | paired clustered bootstrap; Holm–Bonferroni | [10](10-STATISTICS-PLAN.md) |
| Every threshold and the verdict function | `tool_use_promotion_v5` + `study_002_gate_v1` | [11](11-THRESHOLDS.md) |
| Decision rules | verdict vocabulary, computed not written | [02](02-RESEARCH-QUESTIONS.md), [11](11-THRESHOLDS.md) |
| Provenance requirements | validators and error codes | [15](15-PROVENANCE-VALIDATORS.md) |
| Cost basis and planning envelope | $1.79 per GPU-hour; ≈1.5 days ≈ $64.44 for one M0 → M1 lineage; ≈$800–1,700 for the 35-run arm set — **approximate, basis only** | [16](16-GPU-READINESS-GATE.md) |
| Pre-GPU gate | blocking checks | [16](16-GPU-READINESS-GATE.md) |

The cost basis is frozen here, at the preregistration rather than in a narrative, for two reasons: a later
change in provider price must not be presentable as a saving, and the estimate's provenance must be as
checkable as any other input. It is labelled approximate because its two inputs (a runtime recalled as "about
a day and a half", and a rate recorded at the time) are not ledger measurements. The rules that will replace
it with measured container-seconds plus a recorded `gpu_hourly_usd` per run — and the rule that a planning
envelope is **not** available credit — are in [16](16-GPU-READINESS-GATE.md).

## Prior knowledge, declared

Pre-registration is only meaningful if what was already known is stated. The following were known to
the study author before this document was written and are treated as prior, not as results:

1. Study 001's frozen numbers, including the collapse itself — 0-shot GSM8K refusal, 8-shot 55.5%
   accuracy, 5-shot MMLU-Pro no refusal, IFEval strict 67.8 → 45.8
   (`reports/GENERAL_CAPABILITY_REGRESSION.md`).
2. M1-v1's 70.7% refusal rate with **no SFT parent** — the collapse is not specific to SFT
   (`ROADMAP.md:100-102`). H1 is therefore stated as *refusal-`ANSWER` disagreement in the supervised
   data*, not *SFT*, and arms `C1`/`C2` exist because of this prior.
3. The heuristic detector's output on Canonical-v2: 18,114 of 173,237 (10.5%), When2Call 4,038 of
   6,505 (62.1%) — including the fact that a wrong earlier version of this audit exists and is still
   cited in places (`#80`).

**Not known, and therefore not assumed:** the detector's precision, the answerability partition of the
flagged records, whether relabelling works, and whether the 55.5% 8-shot figure survives re-measurement
under the Study 002 evaluator. No threshold in [11](11-THRESHOLDS.md) is set from a Study 001 number
that Study 002 re-measures; where a Study 001 number is the basis, it is marked, and the re-measurement
is reported beside it.

## Guardrails pre-registration checklist

`docs/research/GUARDRAILS.md:134-141` defines the checklist that must be satisfied before GPU time is
spent. Satisfying it is this document's primary obligation.

- [x] **Decision rule and gate version committed (G1)** — `study_002_gate_v1` /
      `tool_use_promotion_v5` in [11](11-THRESHOLDS.md), plus the amendment rule below.
- [x] **Promotion partition covers every deployed response mode; capability sentinels scheduled (G2)**
      — [06](06-SPLIT-SPEC.md) requires a minimum `n` per mode including `ANSWER`; the pre-GPU gate
      fails on any mode with `n = 0`; sentinels run **before** promotion, not after.
- [x] **Reference rerun planned so gate margins can be compared with noise (G3)** — the planned
      repeats in [05](05-SEED-AND-REPRODUCIBILITY-POLICY.md); every threshold in [11](11-THRESHOLDS.md)
      declares a margin and a noise-band comparison.
- [x] **Training data fingerprint recorded, and any audit reads that fingerprint (G4)** —
      [15](15-PROVENANCE-VALIDATORS.md).
- [x] **Every factor that differs between arms listed; exposure logging on (G5)** — the single-factor
      table in [04](04-ARM-MATRIX.md), plus supervised-token accounting per arm.
- [x] **Seeds and intervals planned, or the single-seed limit stated in advance (G11)** — `k = 3` and
      the paired clustered bootstrap in [05](05-SEED-AND-REPRODUCIBILITY-POLICY.md),
      [10](10-STATISTICS-PLAN.md).

## Stop rules

Declared now, so that stopping is not a post-hoc decision either:

1. **Detector precision stop.** If `HEURISTIC_REGEX_v2`'s measured precision is below the floor in
   [11](11-THRESHOLDS.md), the corpus intervention does not proceed: the study reports the detector's
   measured performance and stops. At 80% precision roughly 3,600 records would be relabelled wrongly
   (`ROADMAP.md:128-131`), which makes the intervention's own target ambiguous.
2. **Answerability stop.** If the curated answerability partition — which does not exist yet
   (`ROADMAP.md:132-135`) — cannot reach the inter-labeller agreement floor, the study does not proceed
   to training. An intervention with an undefined target cannot be pre-registered.
3. **Non-vacuity stop.** Any required mode or endpoint with `n = 0`, or an unmeasurable denominator,
   stops the study at `NOT_EVALUABLE`. It is never resolved by widening an interval or dropping a mode.
4. **Vacuous-gate stop.** If `study_002_gate_v1` returns PASS on an empty or synthetic population
   during [16](16-GPU-READINESS-GATE.md), the gate is defective and the study halts until fixed. This
   is the L1 defect re-tested against the new gate rather than assumed fixed.

## Amendments

An amendment is any change to a threshold, a partition, an arm, a seed count or a decision rule after
this commit. The procedure:

1. Record it here with a date, the reason and the item changed, as `study_002_prereg_v2`.
2. Record it in [`reports/ERRATA.md`](../../../reports/ERRATA.md) (G15: applied to every copy).
3. Re-run the revised rule against **every** candidate already scored under the previous version and
   report both versions side by side (G1). Study 001's v3 → v4 change failed exactly here.
4. Any arm launched under the previous version is reported under both versions or not at all.

Amendments are permitted for defects — a gate that cannot fail, a wrong pinned hash, an unmeasurable
dimension. They are not permitted for thresholds after a result is visible: a threshold moved after
seeing data is a new study, and it goes into the next study
(`docs/research/STUDIES.md:22-25`).

### `study_002_prereg_v2` — 2026-09-15

- **Item changed:** the source of the P-DET reference labels used by the classifier acceptance rule (22 §6,
  22 §7). Before: human gold. After: a composite. Human labels are used where they exist (Pass A, items
  #1–#11). Labels from the declared model annotator `model.claude-opus-5` are used everywhere else, and are
  recorded and exported as model judgments. Qualifications measured against it are `MODEL_REFERENCE` and
  provisional.
- **Reason:** human annotation of all 581 items is not feasible with current resources. It is deferred
  until it can be funded.
- **Not changed:** every threshold, the partition, the arms, the seed count, the population and the rubric.
- **Candidates already scored:** none. No classifier has been implemented or scored on P-DET.
- **Nature:** not a defect repair. It is a resource-driven change made before any result exists.
- **Full record:** [28-PDET-MODEL-LABEL-AMENDMENT.md](28-PDET-MODEL-LABEL-AMENDMENT.md).

### `study_002_prereg_v3` — 2026-09-15

- **Item changed:** the P-DET annotation field `annotator_rationale` (23 §2). Before: required for every
  label. After: optional. UNKNOWN still needs an ambiguity status other than `NONE`.
- **Reason:** writing a sentence per item made continued human review too slow. This is an ergonomics
  change, not a taxonomy change.
- **Not changed:** the labels, their definitions, every other field and value, both constraints, the
  population, every threshold, the partition, the arms and the seed count. Every existing rationale is
  kept: 11 human and 570 model. A label without a rationale is not lower-confidence for that reason. The
  composite reference, `model-a`'s provisional status and the precedence of human labels are all as in
  `study_002_prereg_v2`.
- **Candidates already scored:** none. No classifier has been implemented or scored on P-DET.
- **Nature:** not a defect repair. It is a resource-driven change made before any result exists.
- **Full record:** [29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md](29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md).

### `study_002_prereg_v4` — 2026-09-16

- **Item changed:** classifier validation gains a second population, P-DET-COVERAGE-v1, adopted as
  preregistered in [30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md). It covers the modes P-DET-v1 cannot:
  P-DET-v1 holds no CALL and no DIRECT item, so it cannot authorise C1 (`reports/ERRATA.md` §14). Fixed by
  this amendment:
  - the population: seed, strata and quotas (§7), the dedup and exclusion rules (§8–§9) and blinding
    (§10), drawn from normalization-v3 fingerprint `60d3123e…`;
  - the acceptance rules and their minimum sizes, 50 / 50 / 30 / 20 (§11). These are executable in
    `src/opengrad/verification/pdet_coverage_metrics.py`;
  - the structural CALL evidence basis (§4).
- **Reason:** a defect in coverage. Without a population containing CALL and DIRECT, no classifier can be
  qualified for those modes, and the C1 authorisation rule (22 §6) cannot be met.
- **Not changed:**
  - P-DET-v1, its population, labels and freeze state, and 22 and 23;
  - every threshold of the study, the partition, the arms, the seed count and the corpus of every arm.
  30 adds a population; it modifies nothing.
- **Candidates already scored:** none. No classifier has been implemented or scored on either population.
- **Arms launched under an earlier version:** none. Nothing has been trained.
- **Nature:** a defect repair (missing coverage), made before the population was drawn and before any label
  or result on it existed.
- **Full record:** [30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md).
  Its file name keeps `-DRAFT` because committed reports cite that path; its status line says ADOPTED.

### `study_002_prereg_v5` — 2026-09-17

- **Item changed:** who produces P-DET-COVERAGE-v1's reference labels (30 §10). Before: human gold only.
  After: three declared non-Claude model annotators (Gemini 3.8 Flash High, gpt-5.6-sol,
  deepseek-v4.1-flash) label every item independently and blind, and an item's reference label is the one at
  least two give; a three-way split is `NO_CONSENSUS` and is excluded from metrics.
- **Reason:** the study owner will not annotate and no human annotator is available. Claude is excluded
  because it builds the classifier under test (28 item 3).
- **Not changed:** the population, the tasks, the rubric, the blinding, the acceptance rules and minimum
  sizes (now counted on consensus items), the arms, the seed count, and P-DET-v1.
- **Candidates already scored:** none. No label and no classifier exist.
- **Nature:** resource-driven, before any label or result exists. Qualifications against this reference
  are `MODEL_REFERENCE` and provisional; whether they may grant balancing permission is decided separately.
- **Full record:** [34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md](34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md).

### `study_002_prereg_v6` — 2026-09-17

- **Item changed:** what the prose decision classifier may be validated on, for a new classifier version.
  - The classifier input contract gains `prose-decision-input-v2`, whose unit is the first assistant reply of
    any record, whatever follows it (36 §2). v1 stays in force.
  - A third validation population, P-DET-COVERAGE-v2: 420 first replies from Glaive and ToolACE, drawn under
    30 §7–§9 with a new seed and new quotas, labelled like P-DET-COVERAGE-v1 under 34 (36 §3).
  - `prose-decision-classifier-v2` is developed on first-reply data and tested once on P-DET-COVERAGE-v2; P-DET-v1
    and P-DET-COVERAGE-v1 are development-exposed for it (36 §4).
- **Reason:** a defect in coverage found after `prose-decision-classifier-v1`'s one-shot test. DIRECT, which C1
  needs, could not qualify: the unused single-exchange pool holds about 22 DIRECT items, while Study 001's
  corpus holds thousands of direct answers as first replies of longer conversations, which contract v1 excluded
  (35 §4–§6).
- **Not changed:** every threshold and minimum size (22 §6, 30 §11, `pdet-coverage-metrics-v1`), the C1 rule
  (DIRECT and UNSUPPORTED), the arms, the seed count, the partition, P-DET-v1, P-DET-COVERAGE-v1, contract v1 and
  classifier v1 with its result.
- **Candidates already scored:** `prose-decision-classifier-v1`, on P-DET-v1 and P-DET-COVERAGE-v1 (33 §8). Its
  result stands as recorded; no rule it was scored under changes, so there is nothing to re-run.
- **Arms launched under an earlier version:** none. Nothing has been trained.
- **Nature:** a defect repair (the validation unit excluded where DIRECT lives), made **after** v1's result was
  visible, which is disclosed. The new population is drawn after adoption and scored only for v2.
- **Full record:** [36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md](36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md).

## Registration of the unit of analysis

- The **unit of inference** is the seed, clustered by item ([10](10-STATISTICS-PLAN.md)).
- The **unit of comparison** is the arm at a step count decided by the frozen selection rule on DEV
  only — never by a metric read on CONFIRMATORY.
- The **unit of reporting** is `(arm, partition, protocol, device_class)`. A row missing any of the
  four is not renderable ([07](07-METRIC-SPEC.md), [15](15-PROVENANCE-VALIDATORS.md)).

### `study_002_prereg_v7` — 2026-09-18

- **Item changed:** the two permissions 22 §6 and 35 §1 left to the study owner, now taken under the owner's
  explicit delegation of authority (38).
  - Balancing permission is granted, provisionally: UNSUPPORTED and CLARIFY on both layer B sources, DIRECT on
    **glaive only**, and **not** CALL, which was `NOT_EVALUABLE` in the test (37 §8).
  - **C1 is authorised** to proceed past its classifier gate: 21's phases 3-5 (classifier wired into the mixture
    machinery, materialized balance, new canonical-v3 artifacts), under 38 §3's conditions.
- **Reason:** `prose-decision-classifier-v2`, frozen before the reference existed, passed every gating row on
  P-DET-COVERAGE-v2 against a three-model consensus with no `NO_CONSENSUS` item: DIRECT recall 0.985 and
  precision 0.929, UNSUPPORTED recall 0.964, CLARIFY f1 0.942, macro F1 0.956, abstention 0.003 (37 §8). DIRECT,
  the mode C1 exists to restore and the one v1 could not measure, qualifies.
- **Not changed:** every threshold and minimum (22 §6, 30 §11, `pdet-coverage-metrics-v1`), the C1 rule itself,
  the arms, the seeds, the partition, both contracts, and every population. No training run is authorised: that
  stays a separate decision (38 §4). No gold is frozen; P-DET-v1 remains unfrozen. The qualifications stay
  `MODEL_REFERENCE` and provisional, and 38 §5 names what withdraws them.

## Why there is no "best arm" selection

Study 002 does not select a best arm. It tests one mechanism and one correction: `C0` versus `R1` under
matched compute and exposure, with `C1`, `C2`, `R2` and `R3` as competing explanations and
alternatives. A "best of the arms" ranking would reintroduce the `#62` defect — a maximum selected on
the set it is reported on — because these arms differ in kind, not in degree.

## What would make this pre-registration worthless

Stated explicitly, as a tripwire:

1. Running any arm before [16](16-GPU-READINESS-GATE.md) has produced a `READY` record.
2. Reading the CONFIRMATORY or SEALED partition before the arm set is complete and the DEV selection
   is recorded ([06](06-SPLIT-SPEC.md)).
3. Changing a threshold after a score is visible without the amendment procedure above.
4. Reporting a Study 002 number in the same table as a frozen Study 001 number without the evaluator
   version and partition on both (G9).
5. Publishing a claim about ability, decision or format that the metric in [07](07-METRIC-SPEC.md)
   cannot support.