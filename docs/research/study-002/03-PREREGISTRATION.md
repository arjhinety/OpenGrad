# 03 — Pre-registration

> **Current version: `study_002_prereg_v12` (2026-09-25).** The text below is v1, the original contract;
> amendments v2–v6 follow under "Amendments", v7 after "Registration of the unit of analysis", and v8–v12 at
> the end of this document ([40](40-PREREG-V8-DRAFT.md), [41](41-ANSWER-STRATA-AMENDMENT.md),
> [42](42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md), [43](43-PUNANS-AMENDMENT-DRAFT.md),
> [44](44-PUNANS-V2-AMENDMENT-DRAFT.md)). Entries are appended so cited line numbers hold.

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

### `study_002_prereg_v7` (recorded below)

Recorded below, after "Registration of the unit of analysis", where it was first written; its ERRATA entry
is `reports/ERRATA.md` §18 (added 2026-09-24).

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

## Registration of the C1 corpus (canonical-v3) — 2026-09-20

This is a **registration, not an amendment**: no threshold, arm, seed count, partition or decision rule
changes, so nothing already scored needs re-running and no amendment numbering applies. It records the
artifact the C1 workstream ([21](21-C1-IMPLEMENTATION-STATUS.md)) produced and, more importantly, what it
is not.

- **The artifact exists.** canonical-v3: 88,056 records, 22,014 per decision across
  `CALL`/`ANSWER`/`CLARIFY`/`UNSUPPORTED`, selected under the supply-limited equal-shares rule fixed in
  [39](39-CANONICAL-V3-DECISION-BALANCE-SPEC.md) before it was computed, from normalization-v3 fingerprint
  `60d3123e…`, with behaviour labels from the frozen `prose-decision-classifier-v2` (`labels.classifier`).
  Its `versions` block names that classifier.
- **Provenance is checked and passes.** The pre-GPU provenance gate
  (`src/opengrad/data/provenance_gate.py`, 21 phase 6) returns `PASS` on the committed artifacts, recorded
  as [`reports/canonical-v3/provenance-gate-v2.json`](../../../reports/canonical-v3/provenance-gate-v2.json);
  [`…-v1.json`](../../../reports/canonical-v3/provenance-gate-v1.json) holds the classifier-version
  finding it caught, since fixed (21 phase 6).
- **It is not an arm's corpus.** No arm of [04](04-ARM-MATRIX.md) moves to canonical-v3; no arm's corpus,
  supervised tokens or steps change. canonical-v3 is not an operative corpus of any arm
  ([31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) §9.6). The balanced corpus drops ≈70,800
  structural-call records relative to its supply; any later arm that wants more call data must declare it as
  its own factor (31 §9.6, [39](39-CANONICAL-V3-DECISION-BALANCE-SPEC.md) §2).
- **Entering a study remains open, and remains the study owner's decision.** If canonical-v3 later enters
  Study 002 or 003 it enters as its own numbered amendment and as its own measurable factor, never silently
  inside another arm (31 §9.6, 39 §4). This registration does not decide it.
- **No training is authorised.** [16](16-GPU-READINESS-GATE.md) has produced no `READY` record, and the arm
  matrix's `NOT_IMPLEMENTED` components are unchanged.

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

### `study_002_prereg_v8` — 2026-09-24

Appended here rather than under "Amendments" so that line numbers other documents cite do not move.

- **Items changed** (all four items of [40](40-PREREG-V8-DRAFT.md), adopted as drafted by the study owner):
  - **A.** Check 12's truncation "declared factor": within a stage, two arms are imbalanced when their
    truncation rates differ by more than 2 percentage points **and** the larger is more than 2.0× the smaller.
  - **B.** `P-UNANS` has n ≥ 385, the smallest n whose worst-case resolvable margin is at most 10pp. Below it,
    `refusal_correctness` is `UNDER_POWERED` and check 4 fails.
  - **C.** `study_002_gate_v1` wraps `tool_use_promotion_v6` instead of v5, at gate contract 3. v6 changes no
    threshold; it stops v5 promoting without measuring.
  - **D.** Wording corrections to [11](11-THRESHOLDS.md): v5 is v3 plus, not v4 plus, and the 10pp
    resolvability rule stated once, under which `CLARIFY` at n = 371 is not adjudicable.
- **Reason:** a defect. The preregistration required both A and B and quantified neither, so the gate could
  only block on them; v5 could promote on a bundle it never measured (`reports/ERRATA.md` §19).
- **Not changed:** every threshold value in [11](11-THRESHOLDS.md), the arms, the seeds, the partition, the
  `n ≥ 200` mode floor, the `ANSWER` strata sizing, every classifier and P-DET decision of v2–v7, and every
  population.
- **Candidates already scored:** none under this gate, so there is nothing to re-run.
- **Arms launched under an earlier version:** none. No evaluation bundle exists, and no number these rules
  govern has been seen.
- **Code:** `ADOPTED_PARAMETERS` and `STUDY_002_GATE_CONTRACT = 3` in
  `src/opengrad/verification/study_002_gate.py`; the ERRATA entry is `reports/ERRATA.md` §24.

### `study_002_prereg_v9` — 2026-09-24

- **Item changed:** [06](06-SPLIT-SPEC.md) "Building `P-CONF` for four modes", item 2.
  - **The `ANSWER` strata set's source is named.** It has two pools:
    - pool N, BFCL's two no-call files at the gorilla revision the benchmark registry pins, relabelled;
    - pool K, 850 Natural Questions questions (`nq_open` validation) paired with BFCL tool schemas that cannot
      serve them.
  - **How the pools are screened and labelled.** Both pools are screened against every training corpus and the
    When2Call evaluation splits, and labelled blind by the three non-Claude annotators of 34, with a
    two-of-three reference.
  - **Two strata, reported separately:** `ANSWER-natural` and `ANSWER-constructed`, each judged against 06 on
    its own n.
  - **Balance is measured, not enforced.** 06's balance requirement is replaced by a measured report of length
    and tool count: source balance is impossible, and length balance by subsampling would cut the strata below
    their sizing.
- **Reason:** the gap 06 §C2 records. No repository pool reaches the `n ≥ 200` floor. The screening of 27
  public candidates found one usable natural source, estimated at about 389 `ANSWER` items: enough for the
  floor, not for 8 points (`registry/source_screening.yaml`, `study-002-answer-heldout`).
- **Owner decision:** 2026-09-24, *"Proceed with your recommendation on the ANSWER source."*
- **Not changed:**
  - every threshold and the `n ≥ 200` floor;
  - C1, C2, the one-shot discipline and the confirmatory partition;
  - `P-DEV`, `P-UNANS` and `P-SEALED`;
  - every classifier, P-DET and C1 decision of v2–v8.
- **Candidates already scored:** none. The candidate population was drawn by
  `src/opengrad/verification/answer_strata.py`:
  - 1,767 items (917 in pool N, 850 in pool K);
  - sha256 `25ef4c2579affb3591aa542f3513eff6af82b4e5bebdd819457e7ae0b52bbfa8`.

  It was committed with this entry, before any label existed.
- **Arms launched under an earlier version:** none. Nothing has been trained, and no `P-CONF` score exists.
- **Full record:** [41-ANSWER-STRATA-AMENDMENT.md](41-ANSWER-STRATA-AMENDMENT.md); the ERRATA entry is
  `reports/ERRATA.md` §26.

### `study_002_prereg_v10` — 2026-09-24

- **Items changed:** [41](41-ANSWER-STRATA-AMENDMENT.md) §7 and §9.
  - The `ANSWER` strata set is labelled by two non-Claude models: Gemini 3.8 Flash (High) and
    deepseek-v4.1-flash.
  - Its reference label is the label both give. A disagreement is `NO_CONSENSUS` and is excluded.
- **Reason:** 41's third annotator cannot run. Codex refuses gpt-5.6-sol for this account ("not supported
  when using Codex with a ChatGPT account"), including for a test prompt with no item in it.
- **Owner decision:** 2026-09-24, *"Two-model rule (v10)"*, chosen over restoring access and over
  substituting another model.
- **Not changed:**
  - the population (sha256 `25ef4c25…`), the screen, the rubric and the procedure file;
  - 41 §10–§13;
  - P-DET-COVERAGE and 34's three-model rule.
- **Candidates already scored:** none.
- **Labels already recorded:** deepseek-v4.1-flash had labelled 160 of 1,767 items; the other two models
  none. Only completion counts were read, never a label.
- **Stated consequence:** both models must agree, so the strata are smaller than a two-of-three rule would
  give. Each is judged on that n (41 §10).
- **Full record:** [42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md](42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md); the
  ERRATA entry is `reports/ERRATA.md` §27.

### `study_002_prereg_v11` — 2026-09-25

- **Items changed:** [06](06-SPLIT-SPEC.md) "Disjointness and contamination", the `P-UNANS` bullet;
  [07](07-METRIC-SPEC.md) `refusal_correctness`; [03](03-PREREGISTRATION.md) stop rule 2.
  - **`P-UNANS` is defined, sourced and constructed:**
    - **Sources:** KUQ and SelfAware, pinned. Each question is paired with BFCL tools that cannot serve it.
    - **Two strata:** `P-UNANS-unknowable` and `P-UNANS-false-premise`.
    - **Labelling:** blind, by the two annotators of 42, both of whom must agree.
  - **`refusal_correctness`** is computed on the unknowable stratum only. The false-premise stratum has its own
    measure, `premise_rejection`, which is `NOT_EVALUABLE` until a judge is specified and validated.
  - **Stop rule 2's agreement floor** is raw pairwise agreement ≥ 0.80, with Cohen's κ reported.
- **Reason:** `P-UNANS` did not exist, and neither did its agreement floor (06, 07, 11 check 4, 40 item B).
- **Owner decisions, 2026-09-25:**
  - after the source screening `study-002-punans`: sources KUQ + SelfAware; false premises included, as a
    separate stratum; the agreement floor; non-serving tools;
  - adoption as drafted, with the draw authorised: *"yes"*.
- **Not changed:**
  - every threshold, including `min_refusal_correctness` 0.70 and n ≥ 385;
  - `P-CONF-v1` and the `ANSWER` strata;
  - the arms and seeds.
- **Candidates already scored:** none. No P-UNANS item exists before this entry. The candidate population is
  drawn and committed with it, before any label.
- **Arms launched under an earlier version:** none.
- **Full record:** [43-PUNANS-AMENDMENT-DRAFT.md](43-PUNANS-AMENDMENT-DRAFT.md); the ERRATA entry is
  `reports/ERRATA.md` §28.

### `study_002_prereg_v12` — 2026-09-25

- **Context.** Under v11, stop rule 2 fired: the two labelling models agreed on 0.770 of `P-UNANS-v1`'s items,
  below the 0.80 floor, and Study 002 stopped before training
  ([negative result](../../../reports/study-002/punans-v1/NEGATIVE-RESULT.md)). That result stands, and none of
  its items or labels is reused.
- **Items changed:** [43](43-PUNANS-AMENDMENT-DRAFT.md) §3–§5 (sources, definition, pool), §8 (labels) and §9
  (the floor's measure); a trial set is added. In full:
  - **Sources:** unused KUQ questions (`unknowns_all.jsonl`), KUQP's future questions and BIG-bench Known
    Unknowns, pinned; SelfAware is dropped.
  - **Definition:** unknowable questions only. The false-premise stratum and `premise_rejection` are dropped
    from Study 002.
  - **Labels:** membership only: `UNKNOWABLE`, `NOT_UNKNOWABLE`, `UNKNOWN`.
  - **Trial set:** 100 questions, drawn with the 800-question main set before any label, labelled first, never
    part of `P-UNANS`. The procedure may be revised once after it; the floor may not.
  - **Floor:** raw agreement ≥ 0.80 on the main set, unchanged in value, with κ reported. 44 §10 records that
    0.80 is easier to reach over three labels than over six, and that this must be said wherever a pass is
    reported.
- **Reason:** the owner reopened the study for a second attempt. The write-up of the first expected the same
  six-label instrument to fail again on a fresh draw, and the second source screening (`study-002-punans-v2`)
  found only 61 fresh questions beyond KUQ.
- **Owner decisions, 2026-09-25:**
  - reopen Study 002, reversing the acceptance of the stop as final earlier the same day;
  - after the screening: membership-only labels, a trial set, no false-premise stratum, unused KUQ plus
    BIG-bench;
  - adoption as drafted, with KUQP (which the owner had first been told, wrongly, names past years) and the draw
    authorised, with no labelling runs yet: *"proceed, but dont do the runs yet"*.
- **Not changed:**
  - every threshold, including `min_refusal_correctness` 0.70 and n ≥ 385;
  - stop rule 2 and its 0.80 value;
  - `P-CONF-v1`, the `ANSWER` strata and `P-UNANS-v1`;
  - the arms and seeds.
- **Candidates already scored:** none. The trial and main sets are drawn and committed with this entry, before
  any label.
- **Arms launched under an earlier version:** none.
- **Full record:** [44-PUNANS-V2-AMENDMENT-DRAFT.md](44-PUNANS-V2-AMENDMENT-DRAFT.md); the ERRATA entry is
  `reports/ERRATA.md` §29.
