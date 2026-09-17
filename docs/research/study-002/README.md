# Study 002 — design set

**Status: DESIGN / PRE-REGISTRATION. No GPU time has been spent. No study result exists.**

Study 002 must *explain* the Study 001 refusal/direct-answer regression before it tries to *correct*
it. Study 001 ended with a measured regression and an unestablished cause: post-training checkpoints
leave 100% of 1,319 bare zero-shot GSM8K questions unanswered while the same questions are answered at
55.5% accuracy with eight exemplars, and M1-v1 — DPO applied directly to the base with no SFT parent —
still refuses 70.7% (`ROADMAP.md:96-104`). One lineage, one seed, no replicate: chronology is all
Study 001 has, and it says so.

This is also the first study designed hardware-agnostically from the start: the execution contract
names a *device class* and a *compatibility result*, never a model of accelerator, so evidence from an
H200, an A100, an MI300 or a future device is comparable rather than confounded
([13](13-HARDWARE-AGNOSTIC-EXECUTION.md)).

## Current state

| | Status |
|---|---|
| P-DET-v1 population | **Frozen.** 581 items (400 prevalence, 181 challenge), `population_sha256` `6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b` ([24](24-PHASE-3-REPORT.md)) |
| Annotation tooling | **Ready.** `opengrad-annotate`, task `pdet-v1` ([tool guide](../../ANNOTATION_TOOL.md), [25](25-PDET-IMPLEMENTATION-ADDENDUM.md), [26](26-PDET-ANNOTATOR-CHECKLIST.md)) |
| Reference labels | **Every item has a human label; not frozen.** Under [28](28-PDET-MODEL-LABEL-AMENDMENT.md) (`study_002_prereg_v2`) the reference is a composite in which a human `pass-a` label takes precedence over a model judgment. Pass A (`arjhinety`) now labels all 581 items, so the composite takes every label from the human pass. The 570 model judgments (`model.claude-opus-5`, 12 blind batches, each audited) are kept, superseded, as a comparison. The distribution is below. |
| Human labels | **581 of 581, one annotator, not frozen, not yet gold.** Items #1–#11 were labeled first, with rationales. The other 570 were saved in one session on 2026-09-15, 17:21–17:35 local time: a median of 0.83 s between saves, with no rationale ([29](29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md) makes it optional), no note, no flag and no boundary rule cited. There were 13 relabels. Human–model agreement on the 570 items both labeled: **559 (98.1%)**. That is agreement between one human and one model, not inter-annotator agreement (28, item 4). The single-annotator re-read of 22 §4 (both UNKNOWN items) has not been done. |
| Human review | **Pass complete** in the existing tool. Since [29](29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md) (`study_002_prereg_v3`) the rationale is optional. That is an ergonomics change: labels, population and definitions are unchanged, and existing rationales are kept. All 117 items of the pinned `priority-review` queue are labeled. The model's judgments were never shown to the human pass. |
| Model-label provenance | **Archived** in `reports/pdet/provenance/model-a/`: batches, raw answers, audits, the procedure, the ingest log and the audit scripts, byte for byte, with a manifest of hashes. The verbatim transcripts are kept locally (they hold an e-mail address), and their hashes are tracked. |
| Metric-eligible items | 578 of 581. Three items are exposed worked examples, annotated but excluded from metrics ([27](27-PDET-EXPOSED-WORKED-EXAMPLES.md)) |
| P-DET classifier | **Not implemented**, so there is no classifier validation ([24](24-PHASE-3-REPORT.md)). P-DET-v1 contains no CALL and no DIRECT item (below), so it **cannot** qualify a classifier for those modes, and under 22 §6 it cannot authorise C1. |
| canonical-v3 | **Sources and adapters fixed; pre-classifier `normalization-v3` built** (fingerprint `60d3123e…`, adapter version `2.2.0`: the ToolACE repair, then ToolACE call-final records of one measured shape read as `CALL_PREDICTION`, prospectively and as OpenGrad's inference, [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) §9). Canonical-v2 and every arm are unchanged. No classifier and no behaviour label; not the training corpus. The classifier's input contract is `prose-decision-input-v1` ([32](32-CLASSIFIER-INPUT-CONTRACT.md)). |
| Training components | **Every arm trains text-only** (decided 2026-09-16). From `full-model-components-v1` ([policy](../../MODEL_COMPONENT_POLICY.md)) trainers carry vision and MTP by default. Study 002's arms declare `trainer.model_components: {vision: exclude, mtp: exclude}` instead, so they share Study 001's trainer and `C0` stays a true reproduction. The preregistration is unchanged. A training config for an arm that omits the exclusion carries the components and is blocked by the `model_components_validation` readiness gate. |
| P-DET-COVERAGE-v1 | **Adopted, drawn, not annotated.** Adopted as `study_002_prereg_v4` ([30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md)) to cover CALL and DIRECT, which P-DET-v1 cannot. 336 items (sha `755bc16e…`), split into the tasks `pdet-coverage-v1` (306 prose items) and `pdet-coverage-v1-routing` (30 structured-call items). Its reference labels come from a three-model non-Claude consensus (`study_002_prereg_v5`, [34](34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md)); no human annotates. |
| Label-timing finding (P-DET-v1) | **Recorded 2026-09-17, not resolved.** In the tracked `pass-a` change log, 595 saves have a median gap of 0.83 s; 61% came within 1 s of the previous save and 90% within 3 s. Agreement with the blind model pass is 98.1%, and many items are near-repeats, so fast labels are not shown to be wrong. How many would change on a careful read is unmeasured. The 22 §4 re-read covers only the 2 UNKNOWN items here, because nothing was flagged or cited a boundary rule. |
| **Next step** | Decided by the study owner on 2026-09-17: **P-DET-v1 stays unfrozen for now** (nothing waits on it, and the timing finding is unmeasured); a sample re-label to measure consistency stays available before any freeze. **The prose decision classifier is to be built now**, before any P-DET-COVERAGE label exists, with development data that excludes every P-DET-v1 and P-DET-COVERAGE-v1 item (22 §7, 30 §9). From 2026-09-17 the study owner annotates nothing: P-DET-COVERAGE-v1 is labelled by three non-Claude models (`study_002_prereg_v5`). Nothing is frozen. |

### The current reference distribution: a finding, not a target

Recounted from the stored annotations by `opengrad-annotate reference pdet-v1 --sessions pass-a model-a`,
on 2026-09-15, after Pass A labeled every item. The model column is the superseded `model-a` pass.

| Label | Reference (all from `pass-a`) | Metric-eligible | `model-a` (superseded) |
|---|---:|---:|---:|
| CALL | 0 | 0 | 0 |
| DIRECT | 0 | 0 | 4 |
| CLARIFY | 281 | 281 | 271 |
| UNSUPPORTED | 298 | 297 | 294 |
| UNKNOWN | 2 | 0 | 1 |
| **Total** | **581** | **578** | **570** |

Both UNKNOWN items are exposed worked examples ([27](27-PDET-EXPOSED-WORKED-EXAMPLES.md)), so no
metric-eligible item is UNKNOWN. The human and model labels differ on 11 items: #62, #79, #164, #263,
#285, #317, #319, #320, #358, #411 and #513. The model's 4 DIRECT labels became UNKNOWN (#62), CLARIFY
(#164) and UNSUPPORTED (#317, #513).

- **What this is.** A finding about this population. It is not the proportions of these modes in general,
  and not a target. Nothing may be balanced toward or away from it.
- **What it is not.** It is one annotator's labels, not frozen, and not gold. It does not permit C1
  balancing ([22](22-PDET-PROTOCOL.md) §6–§7).

### Finding: P-DET-v1 contains no CALL and no DIRECT item, by construction

The absence is in the source, not in the sampling or the annotation.

- **One source.** The P-DET-v1 candidate population is every valid record of
  `data/processed/normalization-v1/when2call-sft`: When2Call, split `train_sft`, 14,829 records (22 §5).
  The frozen loader filters nothing by behaviour.
- **No tool calls.** In those 14,829 records, **0** carry a tool call; each is one user message and one
  prose assistant reply. The When2Call adapter records the same measurement (`src/opengrad/data/adapters.py`,
  `adapt_when2call`). So the pool contains no machine-readable call payload, and CALL is impossible.
- **No direct answers either.** The replies are requests for information or declines. By first word:
  - "To…": 7,068 (47.7%);
  - "Apologies…": 4,638 (31.3%);
  - "I'm…": 2,306 (15.6%);
  - "I…": 599 (4.0%);
  - everything else: 218.

  This matches When2Call's own design. Its test set (`data/raw/when2call/test/when2call_test_mcq.jsonl`,
  3,652 questions) offers four answers per question: `direct`, `tool_call`, `request_for_info` and
  `cannot_answer`. The correct answer is always `tool_call` (1,295), `cannot_answer` (1,295) or
  `request_for_info` (1,062), never `direct`. In When2Call's own scoring, answering directly is never
  the right behaviour.
- **The challenge component confirms it.** The frozen builder's own call-payload predicate
  (`serialized_call_shape`, `src/opengrad/verification/pdet.py`) matches 0 of the 14,829 responses, so
  that family could not be filled ([24](24-PHASE-3-REPORT.md) §5 lists only the families that fired).

What follows under the frozen rules:

- **Coverage.** 22 §5 wants 50 gold examples per mode. With 0 CALL and 0 DIRECT this is reported as a
  finding, and no classifier may be granted balancing permission for either mode on P-DET-v1.
- **Thresholds.** 22 §6 requires DIRECT recall and precision ≥ 0.80 and CALL precision ≥ 0.95. None of
  these can be measured here, so a classifier cannot pass them. "DIRECT … failing → C1 is not authorised"
  therefore applies: **P-DET-v1 cannot authorise C1.**
- **Study 001.** When2Call-derived records are part of the supervision M0 trained on. The Canonical-v2
  final corpus holds 6,505 of them, and 4,038 of those have a refusal-shaped target labelled `ANSWER`
  ([`ROADMAP.md`](../../../ROADMAP.md)). A source that teaches only asking and declining fits the
  regression Study 002 investigates, but that is an association, not a cause. The causal question stays
  with the arm design ([04](04-ARM-MATRIX.md)).

Covering CALL and DIRECT needs a validation population drawn from sources that contain them, for example
records with structured tool calls and records with direct answers. That would be a new, separately
frozen population. It is a design decision for the study owner, and none has been made.

## Outputs in this directory

| # | Output | File |
|---|---|---|
| 1 | Lessons from Study 001 — adversarial methodology audit | [01-LESSONS-FROM-STUDY-001.md](01-LESSONS-FROM-STUDY-001.md) |
| 2 | Research questions and hypotheses | [02-RESEARCH-QUESTIONS.md](02-RESEARCH-QUESTIONS.md) |
| 3 | Pre-registration | [03-PREREGISTRATION.md](03-PREREGISTRATION.md) |
| 4 | Arm matrix | [04-ARM-MATRIX.md](04-ARM-MATRIX.md) |
| 5 | Seed and reproducibility policy | [05-SEED-AND-REPRODUCIBILITY-POLICY.md](05-SEED-AND-REPRODUCIBILITY-POLICY.md) |
| 6 | Split specification | [06-SPLIT-SPEC.md](06-SPLIT-SPEC.md) |
| 7 | Metric specification | [07-METRIC-SPEC.md](07-METRIC-SPEC.md) |
| 8 | Sentinel specification | [08-SENTINEL-SPEC.md](08-SENTINEL-SPEC.md) |
| 9 | Benchmark plan | [09-BENCHMARK-PLAN.md](09-BENCHMARK-PLAN.md) |
| 10 | Statistics plan | [10-STATISTICS-PLAN.md](10-STATISTICS-PLAN.md) |
| 11 | Pre-registered thresholds | [11-THRESHOLDS.md](11-THRESHOLDS.md) |
| 12 | Artifact retention | [12-ARTIFACT-RETENTION.md](12-ARTIFACT-RETENTION.md) |
| 13 | Hardware-agnostic execution spec | [13-HARDWARE-AGNOSTIC-EXECUTION.md](13-HARDWARE-AGNOSTIC-EXECUTION.md) |
| 14 | Heterogeneity policy | [14-HETEROGENEITY-POLICY.md](14-HETEROGENEITY-POLICY.md) |
| 15 | Provenance validators | [15-PROVENANCE-VALIDATORS.md](15-PROVENANCE-VALIDATORS.md) |
| 16 | GPU readiness gate | [16-GPU-READINESS-GATE.md](16-GPU-READINESS-GATE.md) |
| 17 | Paper outline | [17-PAPER-OUTLINE.md](17-PAPER-OUTLINE.md) |
| 18 | Study 003 roadmap | [18-STUDY-003-ROADMAP.md](18-STUDY-003-ROADMAP.md) |
| 19 | Study 004 roadmap | [19-STUDY-004-ROADMAP.md](19-STUDY-004-ROADMAP.md) |
| 20 | README / ROADMAP / STUDIES updates | this file, [`ROADMAP.md`](../../../ROADMAP.md), [`STUDIES.md`](../STUDIES.md) |
| 21 | Closure report — five-perspective adversarial self-review | [20-CLOSURE-REPORT.md](20-CLOSURE-REPORT.md) |
| 22 | C1 implementation status — provenance repair and schema normalization | [21-C1-IMPLEMENTATION-STATUS.md](21-C1-IMPLEMENTATION-STATUS.md) |
| 23 | P-DET protocol — frozen annotation contract, decision tree, sampling, acceptance | [22-PDET-PROTOCOL.md](22-PDET-PROTOCOL.md) |
| 24 | P-DET annotation instrument — for the human annotator | [23-PDET-ANNOTATION-INSTRUMENT.md](23-PDET-ANNOTATION-INSTRUMENT.md) |
| 25 | Phase 3 report — P-DET frozen | [24-PHASE-3-REPORT.md](24-PHASE-3-REPORT.md) |
| 26 | P-DET implementation addendum — how the annotation tool implements the instrument; naming-only deviation | [25-PDET-IMPLEMENTATION-ADDENDUM.md](25-PDET-IMPLEMENTATION-ADDENDUM.md) |
| 27 | P-DET annotator checklist — read before each pass | [26-PDET-ANNOTATOR-CHECKLIST.md](26-PDET-ANNOTATOR-CHECKLIST.md) |
| 28 | P-DET addendum — three exposed worked examples, excluded from metrics | [27-PDET-EXPOSED-WORKED-EXAMPLES.md](27-PDET-EXPOSED-WORKED-EXAMPLES.md) |
| 29 | P-DET amendment `study_002_prereg_v2` — model reference labels, human annotation deferred | [28-PDET-MODEL-LABEL-AMENDMENT.md](28-PDET-MODEL-LABEL-AMENDMENT.md) |
| 30 | P-DET amendment `study_002_prereg_v3` — the rationale becomes optional (ergonomics, not taxonomy); review order | [29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md](29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md) |
| 31 | P-DET-COVERAGE-v1 preregistration — DIRECT and tool-boundary coverage; **adopted as `study_002_prereg_v4`** (2026-09-16) after an engineering review and a counts-only dry run (30 §13); **drawn** the same day (336 records, sha `755bc16e…`); annotation tasks `pdet-coverage-v1` and `pdet-coverage-v1-routing` set up, not annotated | [30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md) |
| 32 | Canonical-v3 sources and adapters; the `normalization-v3` pre-classifier artifact and its structural audit | [31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) |
| 33 | Classifier input contract `prose-decision-input-v1` — what the prose classifier may read; eligibility | [32-CLASSIFIER-INPUT-CONTRACT.md](32-CLASSIFIER-INPUT-CONTRACT.md) |
| 34 | Prose decision classifier development plan — a model-labelled development set, used only for development; freeze before any test (plan, before implementation) | [33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md](33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md) |
| 35 | P-DET-COVERAGE-v1 amendment `study_002_prereg_v5` — the reference is a three-model non-Claude consensus (two of three); no human annotator | [34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md](34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md) |

Frozen P-DET artifacts live outside this directory, beside the other evaluation artifacts:
`reports/pdet/pdet-v1.population.jsonl` and `reports/pdet/pdet-v1.manifest.json`, verified by
`python -m opengrad.verification.pdet --verify`. Annotation packages will be written as new artifacts under
`reports/pdet/annotation/`, which does not exist until the first export. The pinned review queue is in
`reports/pdet/review/`, and the model-label audit trail in `reports/pdet/provenance/model-a/`. Both name
model labels, so an annotator does not open them.

## Reused, not reinvented

Every extension point below is a verified path in this repository.

| Existing machinery | Path | Study 002 use |
|---|---|---|
| Hardware-agnostic GPU preflight record | `registry/gpu_preflight.schema.json` | already `provider: [nvidia, amd, cpu, unknown]` with `cuda_version`/`rocm_version` both optional — the basis of the hardware-agnostic contract (13) |
| Provenance record | `registry/provenance.schema.json` | `run_id`, `status`, `compute_provider`, `hardware` — extended for device class and driver stack (13, 15) |
| Gate execution accounting | `src/opengrad/verification/accounting.py` | `NONVACUOUS_PREFIX`, `ACCOUNTING_PREFIX`, `REQUIRED_NONEMPTY` — every new gate declares a population policy (15, 16) |
| Repository readiness gates | `src/opengrad/readiness.py:1185-1610` | gate names and `PASS/WARN/FAIL` + `error_code` convention extended (16) |
| Experiment preflight | `src/opengrad/experiments/preflight.py:55-202` | new checks as `PreflightCheckItem`s (15) |
| Promotion policy versioning | `src/opengrad/promotion/tool_use_policy.py` | `tool_use_promotion_v5` adds a refusal sentinel and an ANSWER-mode floor (07, 11) |
| Regression sentinels | `scripts/score_{gsm8k,ifeval,mmlu_pro,sentinel}.py` | reused unchanged; only the schedule and the statistics change (08, 09) |
| Benchmark tiers and suites | `configs/benchmarks/*.yaml`, `configs/benchmark_suites/*.yaml` | `regression_core` and `full_post_training` drive the plan (09) |
| Guardrails G1–G17 | `docs/research/GUARDRAILS.md` | each rule gets an owning output, below |
## Guardrail traceability

A rule without an enforcement is the defect Study 001 kept repeating, so every guardrail is assigned
an owning output here.

| Rule | Subject | Owning output |
|---|---|---|
| G1 | Decision rules fixed before results | 03, 11 |
| G2 | Held-out covers every response mode | 06, 08 |
| G3 | Gate margin vs rerun noise | 10, 11 |
| G4 | Audit the exact data trained on | 15 |
| G5 | "Matched"/"isolates" measured | 04 |
| G6 | Lineage and launch facts from the run record | 15 |
| G7 | Repeat for "reproducible", file for "preserved" | 05, 12 |
| G8 | Scaffolds described as scaffolds | 04, 16 |
| G9 | One population per comparison | 06, 07 |
| G10 | Report the whole result | 04, 07 |
| G11 | State uncertainty | 10 |
| G12 | Caveats travel with the number | 07, 12 |
| G13 | Causal language needs a causal design | 02, 04 |
| G14 | Numbers generated or tested, never retyped | 15 |
| G15 | A correction is applied to every copy | 12 |
| G16 | Status words change in the same commit | 16 |
| G17 | Public surfaces change with the repository | 12 |

## Ordering

[03-PREREGISTRATION.md](03-PREREGISTRATION.md) is the entry point. Any change to a threshold, split,
arm or decision rule after the pre-registration commit is an amendment recorded in
[`reports/ERRATA.md`](../../../reports/ERRATA.md) **and** in the pre-registration file itself, with
both versions reported. The Study 001 failure mode was a gate introduced after the results existed
(`docs/research/GUARDRAILS.md:16-22`).

## Scope decision (a documented change)

`docs/research/STUDIES.md:59-83` currently scopes Study 002 as three subjects: dataset corrections
(including refusal relabelling), on-policy distillation (RQ4) and speculative decoding (RQ5). This
design set **splits that scope**, and the split is recorded here rather than made silently:

- **Study 002 becomes the mechanism study** — the refusal/direct-answer regression: its cause, its
  boundary conditions, and its correction. One phenomenon, one intervention family, one confirmatory
  claim.
- **On-policy distillation moves to [Study 003](18-STUDY-003-ROADMAP.md).**
- **Speculative decoding moves to [Study 004](19-STUDY-004-ROADMAP.md).**

The reason is not capacity, it is validity. Two of the three subjects have no live implementation
(`#66`, `#67`; `reports/M2_DECISION.md`), so pre-registering thresholds for them now would be the
`#24`/`#26` defect at study scale — a decision rule written before the thing it decides about exists.
More importantly, all three share one scarce input: the frozen Study 001 checkpoints. Running three
unrelated subjects inside one study means one freeze, one claim audit and one errata surface covering
three separate causal claims, which is precisely how Study 001 accumulated 100 unsupported claims
across a single freeze. And the regression question gates the other two: if the base model's
direct-answering capability and the tool policy are entangled with refusal supervision, then any
efficiency or distillation result measured against these checkpoints inherits the entanglement.

`STUDIES.md` is updated accordingly in the same commit as this directory
([20](20-CLOSURE-REPORT.md) records the change). No Study 001 number, checkpoint or artifact changes.

## Non-goals

- Study 002 does **not** re-open, re-score or re-interpret any frozen Study 001 number. Frozen
  checkpoints are inputs.
- Study 002 does **not** claim causation from one lineage. Establishing that refusal-targeted
  supervision causes the collapse requires varying that factor alone under matched compute and
  exposure ([04](04-ARM-MATRIX.md)); anything less is reported as an association, with the confounds
  named (G13).
- Study 002 does **not** treat the 18,114 figure as established. It is a heuristic detector's output
  on a corpus whose identity Study 001 got wrong once already (`#80`); its precision and recall are
  measured before any GPU time ([15](15-PROVENANCE-VALIDATORS.md), [16](16-GPU-READINESS-GATE.md)).
- Study 002 does **not** delete refusals from the corpus. The defect under investigation is the
  **label**, not the refusal: a model that cannot report today's trending topics is behaving
  correctly, and removing those targets would trade over-refusal for undetectable hallucination
  (`ROADMAP.md:120-124`).