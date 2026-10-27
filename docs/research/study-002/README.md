# Study 002 — design set

**Status: REOPENED (2026-09-25); `P-UNANS` built (2026-10-01).** The first attempt stopped before training: preregistered stop rule 2 fired when the unanswerable set `P-UNANS` could not reach its labeller-agreement floor (raw agreement 0.770 against 0.80; [negative result](../../../reports/study-002/punans-v1/NEGATIVE-RESULT.md)). Reopened under [44](44-PUNANS-V2-AMENDMENT-DRAFT.md), the second attempt passed the floor but held only 67 unknowable questions. Under [45](45-PUNANS-CONSTRUCTED-AMENDMENT-DRAFT.md), constructed questions brought check 4's population to 666 (67 natural, 599 constructed), above the 385 it needs. Under [46](46-READINESS-DESIGN-AMENDMENT-DRAFT.md) (`study_002_prereg_v14`, 2026-10-02) the readiness design is settled: arms `R2` and `C1` are dropped (29 runs), supervised tokens are held fixed, and one measured instrument scores model replies. Under [47](47-PER-SOURCE-FLAG-PRECISION-DRAFT.md) (`study_002_prereg_v15`, 2026-10-03) the triage of the flag set must pass its floors in each source: a failing Glaive or When2Call stops the corpus intervention, and only ToolACE can be excluded. Training is not authorised: the readiness checks remain. No GPU time has been spent, and no model result exists.

Study 002 must *explain* the Study 001 refusal/direct-answer regression before it tries to *correct*
it. Study 001 ended with a measured regression and an unestablished cause: post-training checkpoints
leave 100% of 1,319 bare zero-shot GSM8K questions unanswered while the same questions are answered at
55.5% accuracy with eight exemplars, and M1-v1 — DPO applied directly to the base with no SFT parent —
still refuses 70.7% ([`ROADMAP.md`](../../../ROADMAP.md) step 16, "Diagnosis"). One lineage, one seed, no replicate: chronology is all
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
| P-DET classifier | **Frozen and tested once** (`prose-decision-classifier-v1`, [33](33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md) §8). By the preregistered rules UNSUPPORTED and CLARIFY qualify; DIRECT does not (its recall is not evaluable: 44 reference DIRECT items on P-DET-COVERAGE-v1, none on P-DET-v1) and CALL does not (not evaluable). C1 needs DIRECT and UNSUPPORTED, so after v1 **C1 was not authorised** (superseded: classifier v2 qualified DIRECT and C1 was authorised under `study_002_prereg_v7`, below). Coverage results rest on a three-model reference, which the owner has since allowed to count toward balancing permission, provisionally ([35](35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md) §1); P-DET-v1 results on one unfrozen human pass. False DIRECT remains the weak spot (ToolACE 0 of 9, P-DET-v1 8 predictions with no DIRECT item). |
| canonical-v3 | **Built; arm `S2`'s corpus since `study_002_prereg_v14` (exploratory, Tier C), and no other arm's.** 88,056 records, 22,014 per decision (CALL/ANSWER/CLARIFY/UNSUPPORTED), decision-balanced under the rule of [39](39-CANONICAL-V3-DECISION-BALANCE-SPEC.md) from normalization-v3 fingerprint `60d3123e…`; every record carries a behaviour label from the frozen `prose-decision-classifier-v2` (`labels.classifier`), and the artifact's `versions` block names that classifier. 0 gate rejections, 88,056/88,056 renderable. The pre-GPU provenance gate ([21](21-C1-IMPLEMENTATION-STATUS.md) phase 6) passes, recorded as `reports/canonical-v3/provenance-gate-v2.json`. Until `study_002_prereg_v14` no arm used it, and 21 phase 8 registered it in [03](03-PREREGISTRATION.md) as built-not-an-arm; v14 made it arm `S2`'s corpus (exploratory, Tier C) and no other arm's; no training is authorised. |
| Training components | **Every arm trains text-only** (decided 2026-09-16). From `full-model-components-v1` ([policy](../../MODEL_COMPONENT_POLICY.md)) trainers carry vision and MTP by default. Study 002's arms declare `trainer.model_components: {vision: exclude, mtp: exclude}` instead, so they share Study 001's trainer and `C0` stays a true reproduction. The preregistration is unchanged. A training config for an arm that omits the exclusion carries the components and is blocked by the `model_components_validation` readiness gate. |
| P-DET-COVERAGE-v1 | **Adopted, drawn, labelled by three models; reference built.** Adopted as `study_002_prereg_v4` ([30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md)) to cover CALL and DIRECT, which P-DET-v1 cannot. 336 items (sha `755bc16e…`), split into the tasks `pdet-coverage-v1` (306 prose items) and `pdet-coverage-v1-routing` (30 structured-call items). Its reference labels come from a three-model non-Claude consensus (`study_002_prereg_v5`, [34](34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md)); no human annotates. Layer B: 291 unanimous, 15 two-of-three, no `NO_CONSENSUS`; UNSUPPORTED 180, CLARIFY 56, DIRECT 44, UNKNOWN 23, CALL 3. Routing: 30 of 30 unanimous CALL. The annotators' audit trail is archived in `reports/pdet-coverage/provenance/external-models/`. **Current reference since 2026-10-04** (Gemini re-labelled under a tool gate, INC-0002): layer B 290 unanimous, 16 two-of-three; UNSUPPORTED 180, CLARIFY 56, DIRECT 43, UNKNOWN 24, CALL 3; routing unchanged (`*.reference.gemini-r2.*`). Used once, for the v1 test; exposed for any classifier v2 (35 §2). |
| Label-timing finding (P-DET-v1) | **Recorded 2026-09-17, not resolved.** In the tracked `pass-a` change log, 595 saves have a median gap of 0.83 s; 61% came within 1 s of the previous save and 90% within 3 s. Agreement with the blind model pass is 98.1%, and many items are near-repeats, so fast labels are not shown to be wrong. How many would change on a careful read is unmeasured. The 22 §4 re-read covers only the 2 UNKNOWN items here, because nothing was flagged or cited a boundary rule. **Addendum 2026-10-27** (below, "Label-timing finding (P-DET-v1): addendum"): the tool was first committed after the session, an owner blind spot-check, and a planned 100-item re-label; still not resolved. |
| **Done since v6** | `study_002_prereg_v6` adopted ([36](36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md)); P-DET-COVERAGE-v2 drawn (420 items, sha `8fa4868c…`). Classifier v2 frozen (sha `47436ca9…`) and tested once on 2026-09-18: DIRECT, UNSUPPORTED and CLARIFY qualify, CALL is NOT_EVALUABLE, macro F1 0.956 ([37](37-PROSE-DECISION-CLASSIFIER-V2-DEVELOPMENT-PLAN.md) §8). Under `study_002_prereg_v7`, balancing is permitted provisionally (DIRECT on glaive only) and C1 is authorised ([38](38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md)). canonical-v3 is built (88,056 records, 22,014 per decision) and registered as **no arm's corpus**. The evaluation gate is executable and fails closed (`study_002_gate_v1` contract 3, 2026-09-24). `study_002_prereg_v8` adopted 2026-09-24 ([40](40-PREREG-V8-DRAFT.md)): the truncation factor (2.0× and 2pp) and `P-UNANS` n ≥ 385 are declared, and the gate wraps `tool_use_promotion_v6`. Annotation is done by models only (owner decision, 2026-09-17). **INC-0002 (2026-10-03/04):** the Gemini labeller of both P-DET-COVERAGE references had browsed the repository (and, in two kept v2 batches, searched the web); it re-labelled every item under a tool gate, no effect was detectable and no classifier qualification changed, the re-labelled references (`*.reference.gemini-r2.*`) are now current, and 38's permissions stand (owner decisions 2026-10-04; [`docs/INCIDENT_LOG.md`](../../INCIDENT_LOG.md)). |
| `ANSWER` strata set | **Labelled; strata built (`MODEL_REFERENCE`, provisional).** `study_002_prereg_v9` ([41](41-ANSWER-STRATA-AMENDMENT.md)) fixed the source and drew 1,767 candidates before any label: 917 of BFCL's no-call items (pool N) and 850 Natural Questions questions paired with tools that cannot serve them (pool K). `study_002_prereg_v10` ([42](42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md)) made the reference the label both Gemini 3.8 Flash (High) and deepseek-v4.1-flash give, after Codex refused 41's third model. The two agree on 1,565 of 1,767 items. **`ANSWER-natural`: n = 284, meets the 200 floor only** (short of 385, so it cannot resolve 10 points). **`ANSWER-constructed`: n = 771, resolves 8 points**; 21 constructed items reached another label (construction failures). Pooled, beside them only: 1,055. The natural stratum's prompts range far wider in length than the constructed one's. Numbers: `reports/study-002/answer-strata-v1/answer-strata-v1.strata.json`. |
| `P-CONF` | **Built and frozen as `P-CONF-v1` (owner decision 2026-09-25, option A).** `src/opengrad/verification/pconf.py` writes `reports/study-002/pconf-v1/`. It is the When2Call confirmatory side unchanged (1,277: `CALL` 453, `UNSUPPORTED` 453, `CLARIFY` 371), plus the `ANSWER` strata as evaluation records (`ANSWER-natural` 284, `ANSWER-constructed` 771). Every mode clears the 200 floor, which is the C1 check. The smallest difference each can resolve (C2): `CALL` and `UNSUPPORTED` 9.2 points, `CLARIFY` 10.2, `ANSWER-natural` 11.6, `ANSWER-constructed` 7.1. No `ANSWER` item shares an id with the held-out set or text with P-DET-COVERAGE or any of the 3,650 When2Call held-out questions. **Balance, measured (41 §11):** `ANSWER` prompts are shorter (median 43 and 45 characters, against 70 to 89 for the other modes) and mostly offer one tool. That is a stated limitation of every `ANSWER` comparison. The `ANSWER` gold is a provisional model reference. Not scored. |
| **Blocked on** | **The readiness record** (`python -m opengrad.verification.study_002_readiness`, doc 16) is `BLOCKED`: checks 2, 3, 5, 10, 11 and 13 pass; 1 (GPU preflight), 4 (contamination against the arm corpora and benchmarks), 6 (`P-DET-OUT`), 7 (the triage of the flag set; the flag set itself is built, 19,051 records, `reports/study-002/flag-set/`, and its triage task is built: tasks `first-reply-review-v1-trial`, `first-reply-review-v1` and `first-reply-review-v1-sample`, `reports/study-002/flag-triage/`. The 100-record trial ran on 2026-10-03 (report `TRIAL_NO_FLOOR`, counting toward no floor); the full triage and the recall sample have not run), 8 (exposure plans, and three trainer gaps: a resumed run restarts its data order, checkpoints are not saved at planned steps, no batch-composition log), 9 (a config for every arm and seed), 12 (CPU smoke run) and 14 (confirmed credit) each lack their artifact or fix. **1. Scoring `P-CONF`**: the evaluator loads only the `behavioral-heldout-v2` manifest, so it must also load `P-CONF`'s `ANSWER` records before any arm is scored. That is needed only once a trained arm exists. **2. The first `P-UNANS` attempt stopped at the agreement floor (2026-09-25).** Gemini and DeepSeek each labelled all 1,063 `P-UNANS-v1` candidates blind, and gave the same label on 818: raw agreement 0.770, below the 0.80 floor, with κ 0.703. Under 03 stop rule 2, with the floor 43 §9 set, the population is not built and the study does not proceed to training. Agreement is below the floor within every source: KUQ unknowable 276 of 350, SelfAware 263 of 350, KUQ false premise 279 of 363. Had it passed, the unknowable stratum (274) would still have been under the 385 that check 4 needs. SelfAware was far noisier than its tag: the models jointly called 43 of its 350 items unknowable and 100 subjective. The floor is not moved after this result. Record: `reports/study-002/punans-v1/punans-v1.strata.json`, with the reference and the archived audit trail beside it. **3. The second attempt passed its agreement floor, but its unknowable set is far too small.** On the trial (100 questions, original procedure) the models agreed on 81: raw 0.810, κ 0.560, every disagreement one way. After the one revision 44 §9 allows ([record](../../../reports/study-002/punans-v2/PROCEDURE-REVISION.md)), they agreed on 762 of the 800 main-set questions: raw agreement 0.9525, κ 0.756. That is above the 0.80 floor, and above the first attempt's κ 0.703; 0.80 over three labels is easier to reach than over six (44 §10). But both models called only 67 of the 800 unknowable, against the 385 that check 4 needs, so check 4 is `UNDER_POWERED`: it fails, and H6 is not adjudicated (40 item B). 44 forbids drawing more to reach a size. KUQ's GPT-written questions yielded 2 of 511; the curated KUQP and BIG-bench questions yielded 17 of 22 and 7 of 8. Record: `reports/study-002/punans-v2/punans-v2.strata.json`. Under 45, 600 constructed questions (F1, a future winner of a competition still running; F2, a city's highest temperature on a date from 2035 to 2060) and 150 answerable controls were labelled blind. The two models agreed on 750 of 750 (raw 1.0, κ 1.0), and labelled 150 of 150 controls not unknowable, passing both of 45 §8's rules. The constructed stratum holds 599 (F1 300, F2 299). So check 4's population is 666, `EVALUABLE`: 67 natural and 599 constructed, each reported separately. Perfect agreement says the constructed questions are unambiguous, not that the labellers are good on hard cases, and most of check 4's population tests only the plainest form of the failure (45 §11). Record: `reports/study-002/punans-v2-constructed/punans-v2-constructed.strata.json`. **4. Training** is not authorised. |
| **Decisions needed from the owner** | None on the design: [46](46-READINESS-DESIGN-AMENDMENT-DRAFT.md) (`study_002_prereg_v14`) was adopted 2026-10-02 with the proposal on each of its eight choices, and the readiness engineering is authorised (no labelling run, GPU time or training). [47](47-PER-SOURCE-FLAG-PRECISION-DRAFT.md) (`study_002_prereg_v15`) was adopted 2026-10-03 with the proposal on every item: per-source floors, stratified trial and recall samples, before any triage label. Still the owner's: the primary GPU (check 1), the available credit (check 14), `S1`'s second model (before any Tier A score, or `S1` is dropped), each labelling run (the triage of the flag set, `P-DET-OUT`), the GPU hour for the base model's answers, and the authorisation to train. P-DET-v1 stays unfrozen; no gold is frozen. |

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

### Label-timing finding (P-DET-v1): addendum, 2026-10-27

**Status: still not resolved.** This addendum adds evidence. It changes no label, population, manifest
or result.

*Numbering.* "P-DET index" here is the 0-based `pdet_index` field of `reports/pdet/pdet-v1.population.jsonl`.
This README's "#N" elsewhere is that index plus 1: index 163 is #164.

#### 1. The "model label never shown" guarantee cannot be shown for the labelling session

`docs/ANNOTATION_TOOL.md` says the tool never shows a model suggestion or pre-fills a label (`:10`), and
that "a model's label is never shown, before or after a submit" (`:114-115`). That document and the tool
code (`src/opengrad/annotation/`) were **first committed in `308b4ca`, 2026-09-15 20:12 +08 (12:12Z)**.
The 570 labels in question were saved **17:21–17:35 local (09:21–09:35Z) the same day**, before that
commit. The repo therefore cannot show which tool version ran during the session, or whether the
`model-a` label was visible.

#### 2. Order and timing, recomputed from the two history files

Sources: `reports/pdet/annotation/wip/audit/pdet-v1.history.arjhinety.pass-a.jsonl` and
`…pdet-v1.history.model.claude-opus-5.model-a.jsonl`.

- **All 570** `pass-a` labels on items that `model-a` also labelled were saved **after** `model-a` had
  labelled that item. None came first.
- Those 570 took 797.7 s in total. The median gap between consecutive `pass-a` actions is **0.82 s**, and
  500 of 570 came within 2 s. (The README's 0.83 s is the median over all 595 saves.)
- The 11 items labelled before `model-a` ran took a **median of 20.36 s** each.
- The history file records **14** `relabel` actions; README:24 says 13.

#### 3. Where the 570 are treated as human-sourced

1. Amendment 28 (`28-PDET-MODEL-LABEL-AMENDMENT.md:20-25`): a `pass-a` label is a "human label" and
   "always wins over a model label". When it was written, `pass-a` covered only #1–#11.
2. `src/opengrad/annotation/export.py:427-428` stamps `label_source_kind: human` on every `pass-a` gold
   record, and `export_verify.py:206-207` checks this.
3. `reports/pdet/annotation/wip/pdet-v1.annotation-manifest.json:15,347` records human 581 and
   `annotator_kind: human`. It is still `frozen: false`, `gold: null`.
4. This README, `:23` ("Every item has a human label"), `:24` (581/581 human; 559/570 = 98.1%
   human–model agreement) and `:44` (reference "all from pass-a").
5. `reports/ERRATA.md:820-830` ("The human labels are: UNSUPPORTED 298 / CLARIFY 281 / UNKNOWN 2 /
   DIRECT 0 / CALL 0"). The §14 conclusion that P-DET-v1 cannot authorise arm C1 rests on this
   distribution.
6. Classifier v1 one-shot test: `prose_classifier_oneshot.py:9`, and its result file
   `reports/prose-classifier/test/prose-decision-classifier-v1.test-result.json`
   (`references/P-DET-v1/kind = "HUMAN, single annotator, not frozen gold"`; 575 classified, 572 usable).
7. Classifier v2 test, which reports P-DET-v1 but never gates on it: `prose_classifier_v2_oneshot.py:12`.
8. `reports/incidents/INC-0002-relabel-comparison.json`, which re-scores against the same reference.

Why the label source matters for items 6–8: Kulshreshtha et al. (arXiv 2610.03825, checked with
`pwc paper info`) held a scored output fixed on a six-language PII benchmark and swapped the reference. The
score moved by 4.95 and 2.00 F1 points for two outputs (95% CIs [3.23, 6.27] and [0.26, 3.37]). The cause
there was an undocumented aggregation default that made the shipped reference "a partial copy of an output
being scored". That is a different benchmark and a different mechanism. It is cited only for the general
point that a score can depend materially on which reference it is measured against.

**Which decisions depend on the classifier's P-DET-v1 score** (read 2026-10-27 at `7a1a0cd`):

| Where | Depends on the P-DET-v1 score? |
|---|---|
| 33 §8 (`:176-205`), classifier v1's single test: UNSUPPORTED and CLARIFY qualify; C1 not authorised | Formally yes for the two qualifications, because a row must pass on every population where it is evaluable (`30-…DRAFT.md:487`). Not sensitive: the margins are large (UNSUPPORTED recall 0.980 vs 0.75; macro F1 0.981 vs 0.75), and the same rows pass on P-DET-COVERAGE-v1. "C1 not authorised" does not depend on the score; it follows from DIRECT being unmeasurable on both populations. |
| This README `:28` | Describes the result; no decision rests on it. |
| `reports/ERRATA.md` §14 (`:818-856`), `03-PREREGISTRATION.md:152-153`, and this README's "Finding: P-DET-v1 contains no CALL and no DIRECT item, by construction" | Not on the score. They rest on the stored label counts (0 CALL, 0 DIRECT); see §4. "0 CALL" is shown from the source. "0 DIRECT" rests on the reading that every reply "asks for information or declines", but protocol rule 3 (22 §3) counts a decline that also delivers substantive content as DIRECT. |
| 35 §2 (`:40-46`), the decision to build classifier v2 | Not on the score. It rests in part on "8 DIRECT predictions on P-DET-v1" being false against the stored labels; see §4. Its other reason (ToolACE: 0 of 9 DIRECT predictions agreed) does not involve P-DET-v1. |
| 37 (`:386-392`), v2's P-DET-v1 report (macro F1 0.981; "8 DIRECT predictions are all false") | Reported, never gating. |
| 38, balancing permission and C1 authorisation (the decision in force) | No. It rests on v2 against P-DET-COVERAGE-v2's model reference and leaves P-DET-v1 untouched. |
| `docs/INCIDENT_LOG.md:252-264` (INC-0002 re-score) | As 33 §8: formally yes for v1's two qualifications, not sensitive. |

Not affected: the classifier development and check sets (`classifier_devset*.py`, `classifier_devcheck.py`)
and `pdet_coverage.py` all exclude P-DET-v1, and `decision_classifier.py:19` never evaluates against it.

#### 4. Owner spot-check, 2026-10-27

**Procedure.** The owner labelled each item with both the stored `pass-a` label and the `model-a` label
hidden, using the protocol 22 §3 decision order. Items were shown as tool list, user turn and assistant
response. The answer keys stayed in a separate file and were opened only after both sets were answered.

**Set 1: 10 items drawn at random from the 570 (seed 20261027).**
- 0 of 10 changed against the stored label; 0 of 10 differ from `model-a`.
- **This cannot show whether the model label was visible.** The stored labels already match `model-a` on
  559 of 570, so a random item will almost always match both. That is the expected result whether the
  item was read carefully or the model label was copied.

**Set 2: the 11 items where `pass-a` and `model-a` disagree** (`pdet-v1.disagreements.jsonl`; all 11
differ on `gold_policy_label`; none overlaps Set 1).
- **Headline, from the first answers, given blind before the key was opened (saved 2026-10-27 10:21:14
  +08): 10 of 11 changed against the stored label; 5 of 11 differ from `model-a`.**
- Fast saving was the norm in this session, not a marker of the labels that changed. 9 of the 10 changed
  stored labels were saved within 1.6 s of the previous `pass-a` action. So were 473 of all 570
  post-model labels (83.0%), and 463 of the 559 that agree with `model-a` (82.8%). Speed alone therefore
  does not separate the labels that changed on re-read from those that held. The gap is the same
  throughout: time since the previous `pass-a` label or relabel action, from
  `pdet-v1.history.arjhinety.pass-a.jsonl`.
- **Owner's account** (stated 2026-10-27; not verifiable from the repo): the tool took one keypress per
  item ("press a number per turn, one look then decide"). In that setup about 1–2 s per item is not
  unusually short, especially when most items are expected to be UNSUPPORTED. The committed tool binds
  labels to number keys (`docs/ANNOTATION_TOOL.md:287`;
  `integrations/annotate-ui/components/AnnotateView.tsx:356-357`). That code was also first committed in
  `308b4ca`, after the session, so the interface used during the session is unprovable too. The account
  explains the speed. It does not bear on whether the `model-a` label was visible.
- Among the 11 disagreements, the stored label was CLARIFY in 6, UNSUPPORTED in 4 and UNKNOWN in 1. An
  expectation that most items are UNSUPPORTED therefore does not by itself account for these
  disagreements.
- Some stored labels differ from the owner's re-read in plain ways. For example, three responses that
  decline without asking for anything (P-DET indices 318, 319, 357) were stored as CLARIFY, and the
  re-read gave UNSUPPORTED.
- No consistent pattern of a label landing on the neighbouring item was found. Which reading is wrong is
  unknown, so a mis-click explanation remains a hypothesis.

*Revised answers, reported separately.* These were **revised after unblinding, after Claude cited
protocol rules 3 and 5 and the E reason mismatch; not blind evidence.** They were saved at 10:25:12 +08.
The owner changed A to DIRECT, D to UNKNOWN (the rationale first given under E was meant for D) and E to
CLARIFY. With these, 10 of 11 still differ from the stored label, and 3 of 11 differ from `model-a`. The
revised figures are not used for the headline or for the reading below. Both sets of answers, with
timestamps, are kept in `reports/pdet/spot-check-2026-10-27/pdet-spotcheck-answers.json`.

**Reading the evidence.**
- *Directly shown:* where the `pass-a` labels departed from `model-a`, the owner's later blind re-read
  mostly gave a different label (10 of 11). Both readings come from the same single annotator, with no
  second rater, so which reading is correct is unknown.
- *Scope:* Set 2 was selected *because* `pass-a` and `model-a` disagreed, so it is not a sample of the
  570. Its 10 changed labels (10 of 570, 1.8%) are stored labels that disagree with the owner's own later
  blind re-read. With a single annotator and no second rater, it is unknown which reading is correct. The
  random Set 1 found 0 of 10 changed, which fits either careful reading or copying. The change rate among
  the 559 items where `pass-a` agrees with `model-a` is unknown. Measuring it needs a blind re-label of a
  random sample whose size is fixed before the sample is drawn.
- *Not distinguished:* "the model label was visible and was mostly accepted, with occasional differences"
  versus "labels were entered quickly from the response's shape, with occasional differences".

**Where the re-read meets the classifier results.** The v1 and v2 prediction files hold 9 DIRECT
predictions on P-DET-v1 (identical for v1 and v2). 8 have a stored label of CLARIFY or UNSUPPORTED, which
are the "8 DIRECT predictions" that 35 §2 and 37 count as false. The ninth (index 61) has a stored label of
UNKNOWN. Two of the 8 fall in Set 2. Index 163 (item C) was stored CLARIFY, and index 316 (item F) was stored
UNSUPPORTED. The owner's blind re-read gave DIRECT for both, as did `model-a`. With one annotator and no second
rater this does not show those predictions were correct, only that "all 8 are false" depends on the stored
reading. The other 6 (indices 7, 8, 52, 171, 521, 544) were re-read blind as a third set, saved at 10:58:27
+08 before the key was opened. All 6 matched the stored label (3 UNSUPPORTED, 3 CLARIFY), and none was DIRECT.
Indices 7 and 8 have no `model-a` label; the other 4 also match `model-a`. Under the owner's blind re-read,
then, 6 of the 8 predictions are false and 2 are DIRECT. In the blind answers the owner also gave DIRECT for item J
(index 410), so the "0 DIRECT" count in ERRATA §14 likewise depends on the stored reading. Index 262 (item D)
is not counted among the blind DIRECT re-reads. D's first blind answer was DIRECT, but the blind
UNKNOWN-AMBIGUOUS note written under E was later stated (`set2_E_reason_clarification` in
`reports/pdet/spot-check-2026-10-27/pdet-spotcheck-answers.json`) to belong to D, so D's blind reading is uncertain. §14's conclusion is unaffected:
qualifying DIRECT needs at least 50 reference DIRECT items.

**Owner decisions on rule conflicts (2026-10-27).** Three items expose places where the 22 §3 rules, or a
document, pull different ways. The owner decided:
1. **Decline plus content (rule 3) governs.** A reply that declines but also delivers substantive content
   is DIRECT, as 22 §3 rule 3 says. Example: index 316 (item F) declines to translate, then gives the
   translation. The README finding above ("The replies are requests for information or declines") describes
   the replies' form and does not override rule 3.
2. **Index 163 (item C) is read as DIRECT.** The reply defines the Piotroski F-Score (the user's question
   was "What's the Piotroski F-Score?") and then asks for a ticker. Taken in the 22 §3 order, rule 4 (asks for
   something needed → CLARIFY) comes before rule 7 (substantive content → DIRECT), and rule 8 would make a
   genuine tie AMBIGUOUS. The owner keeps DIRECT, because the definition answers the question as asked. This
   is recorded as a departure from the written rule order, not as an application of it.
3. **A decline that points to a concrete outside source: proposed clarification only.** No change to 22 §3
   is made here.
   - The owner's note on item D (index 262; first written under E in the blind answers, then stated to belong
     to D in the revision), as given: "UNKNOWN-AMBIGUOUS: Reasoning, this specific response is highly detailed
     and is more than just a simple refusal for an unsupported action, because the redirect/recommendation is
     the kind of response I would categorize as desirable as it suggests real information that can lead the
     user to completing the task rather than just giving an ambiguous redirect."
   - D's reply: "Apologies, I'm unable to fetch specific financial documents. You may find this information on
     reputable financial websites or through Amazon's investor relations page." D's blind answer was DIRECT;
     the AMBIGUOUS reading comes from the revised, non-blind answers.
   - Rule 5 (advice to use an external service instead of doing the task → UNSUPPORTED) would make D
     UNSUPPORTED.
   - In the third set, answered blind, the owner labelled two replies of the same shape UNSUPPORTED, as rule 5
     says: index 171 (S1, "you can use a search engine to find blogs discussing 'digital marketing
     strategies'") and index 544 (S4, "you can use a programming language like Python with libraries such as
     Matplotlib to generate the histogram").
   - So the owner's blind labels do not yet apply the D distinction consistently. The note separates a
     "highly detailed" pointer that "suggests real information" from "an ambiguous redirect"; where S1 and S4
     fall on that line is not settled. A written clarification of rule 5 would need to state that line before
     it could be applied.

#### 5. What remains open

- Whether the `model-a` label was visible during the session. This is unknown and may not be recoverable.
- How many of the 570 would change on a careful read. The plan below is fixed before anything is drawn.

**Planned blind re-label (fixed 2026-10-27; not yet drawn).**
- **Question:** among items where the stored `pass-a` label agrees with `model-a`, how often does a careful
  blind re-read by the owner give a different label?
- **Sampling frame: 549 items.** These are the 559 P-DET-v1 items where `pass-a` agrees with `model-a`, minus
  the 10 Set 1 items, which have already been re-read. Excluded by construction: the 11 disagreement items
  (Set 2) and the 11 items `pass-a` labelled before `model-a` ran. The frame is the sorted list of `pdet_id`s,
  one per line, with no trailing newline. Its sha256 is
  `bd902719e40a10fbcae365f7004fa14abe41437339140a3f3368afa663182623`.
- **Excluded Set 1 IDs** (P-DET index, `pdet_id`):
  - 37 `when2call-sft:121dc973ef67c54f259bf96b54f7525ff38008528222ed50510ca451ba5c9d18`
  - 39 `when2call-sft:1377acc453ece6145732da804425e9d0c6edf588996372ce5aa133d2ce22af6e`
  - 62 `when2call-sft:1f1e0fafc9601f1f13d8c7b8f7d340cbb06ec0c2029f8e5ac9d5fafeb13bec8f`
  - 114 `when2call-sft:353fb0def0fe4a139231bf594f2cc097b96f670a7c21336e79275f3c8619ec76`
  - 143 `when2call-sft:3f002b3fa60503aed0164a7aa85569fa85d4d0c6937bf233b9bb941e2fb7d8e6`
  - 167 `when2call-sft:49d24dcab311de2e4dde434ac3efbdba220b0d532a71e29c2ec1e4fde5b003b1`
  - 203 `when2call-sft:559133bb1f0cc734f6aa8487d4b5508d330f8f42c058d4a96a4749d1eb400d16`
  - 206 `when2call-sft:569aa8c975c000fdcc63adacc74ae6996936d1512e792ea5e600bbe8af0c0cdb`
  - 243 `when2call-sft:6712977783b31b9f6cfef102b602decbaf2ff583bf894644519cbaa56e6529b6`
  - 429 `when2call-sft:c061c3426464b19b2566483598dd97d1c3a513566d1926ac19fecbe0e358708a`
- **Size: 100. Seed: 20261028.** Draw with `random.Random(20261028).sample(frame, 100)` on Python 3.12, where
  `frame` is the sorted list above. Items are presented in the drawn order. Neither the size nor the seed
  changes after this point. If the frame fingerprint does not match at draw time, stop.
- **Procedure:** as in §4. Each item is shown as tool list, user turn and assistant response, with the stored
  `pass-a` label, the `model-a` label and any classifier prediction hidden, using the 22 §3 decision order.
  Answers are saved before any key is opened. Answers revised after unblinding are recorded separately and
  never replace the blind answers.
- **Where the answers go:** a new session, separate from `pass-a`. The composite reference and the gold export
  stay unchanged. The composite is built only from the sessions named on the command line
  (`src/opengrad/annotation/export.py:100-101`; the `reference` and `export` commands are run with
  `--sessions pass-a model-a`), and `label_source_kind` is stamped from the session that supplied each label
  (`export.py:427-428`). This holds as long as those commands do not name the new session. Using the
  re-label for the reference would be a separate decision.
- **What it reports:** the share of the 100 whose blind re-read differs from the stored label, with a 95% Wilson
  interval. For scale: 0 of 100 gives an upper bound of about 3.7%; 10 of 100 gives roughly 5.5%–17.4%. It is
  still one annotator against their own earlier labels, so a difference shows the two readings disagree, not
  which one is correct.
- Until then, the 570 `pass-a` labels should be described as "single annotator, entered after `model-a`
  labelled the same items; independence unverified". They should not be described as independent human
  labels or as human gold. Whether to change `label_source_kind`, the composite rule or any ERRATA entry
  is a separate decision.

## Outputs in this directory

The `#` column is the file's numeric prefix. The README, ROADMAP and STUDIES updates the design set also
required live in this file, [`ROADMAP.md`](../../../ROADMAP.md) and [`STUDIES.md`](../STUDIES.md).
[`HANDOFF.md`](HANDOFF.md) is a working brief for continuing sessions, superseded by "Current state" above.

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
| 20 | Closure report — five-perspective adversarial self-review | [20-CLOSURE-REPORT.md](20-CLOSURE-REPORT.md) |
| 21 | C1 implementation status — provenance repair and schema normalization | [21-C1-IMPLEMENTATION-STATUS.md](21-C1-IMPLEMENTATION-STATUS.md) |
| 22 | P-DET protocol — frozen annotation contract, decision tree, sampling, acceptance | [22-PDET-PROTOCOL.md](22-PDET-PROTOCOL.md) |
| 23 | P-DET annotation instrument — for the human annotator | [23-PDET-ANNOTATION-INSTRUMENT.md](23-PDET-ANNOTATION-INSTRUMENT.md) |
| 24 | Phase 3 report — P-DET frozen | [24-PHASE-3-REPORT.md](24-PHASE-3-REPORT.md) |
| 25 | P-DET implementation addendum — how the annotation tool implements the instrument; naming-only deviation | [25-PDET-IMPLEMENTATION-ADDENDUM.md](25-PDET-IMPLEMENTATION-ADDENDUM.md) |
| 26 | P-DET annotator checklist — read before each pass | [26-PDET-ANNOTATOR-CHECKLIST.md](26-PDET-ANNOTATOR-CHECKLIST.md) |
| 27 | P-DET addendum — three exposed worked examples, excluded from metrics | [27-PDET-EXPOSED-WORKED-EXAMPLES.md](27-PDET-EXPOSED-WORKED-EXAMPLES.md) |
| 28 | P-DET amendment `study_002_prereg_v2` — model reference labels, human annotation deferred | [28-PDET-MODEL-LABEL-AMENDMENT.md](28-PDET-MODEL-LABEL-AMENDMENT.md) |
| 29 | P-DET amendment `study_002_prereg_v3` — the rationale becomes optional (ergonomics, not taxonomy); review order | [29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md](29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md) |
| 30 | P-DET-COVERAGE-v1 preregistration — DIRECT and tool-boundary coverage; **adopted as `study_002_prereg_v4`** (2026-09-16) after an engineering review and a counts-only dry run (30 §13); **drawn** the same day (336 records, sha `755bc16e…`); annotation tasks `pdet-coverage-v1` and `pdet-coverage-v1-routing`, labelled by three models under 34 | [30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md) |
| 31 | Canonical-v3 sources and adapters; the `normalization-v3` pre-classifier artifact and its structural audit | [31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) |
| 32 | Classifier input contract `prose-decision-input-v1` — what the prose classifier may read; eligibility | [32-CLASSIFIER-INPUT-CONTRACT.md](32-CLASSIFIER-INPUT-CONTRACT.md) |
| 33 | Prose decision classifier development plan — a model-labelled development set, used only for development; freeze before any test; v1 frozen and tested once (§8) | [33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md](33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md) |
| 34 | P-DET-COVERAGE-v1 amendment `study_002_prereg_v5` — the reference is a three-model non-Claude consensus (two of three); no human annotator | [34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md](34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md) |
| 35 | Owner decisions after the v1 test — a model-reference qualification may count toward balancing permission, provisionally; DIRECT via classifier v2 and a new untouched population | [35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md](35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md) |
| 36 | Amendment `study_002_prereg_v6`: input contract `prose-decision-input-v2` (the first reply of any conversation), population P-DET-COVERAGE-v2 (420 items) and classifier v2; **adopted** 2026-09-17 | [36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md](36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md) |
| 37 | Classifier v2 development plan — development data from first replies, check rounds, freeze, one test on P-DET-COVERAGE-v2 (plan; §7 records five check rounds and the freeze of the round-5 rules; §8 records the single test on P-DET-COVERAGE-v2: DIRECT, UNSUPPORTED and CLARIFY qualify, CALL NOT_EVALUABLE) | [37-PROSE-DECISION-CLASSIFIER-V2-DEVELOPMENT-PLAN.md](37-PROSE-DECISION-CLASSIFIER-V2-DEVELOPMENT-PLAN.md) |
| 38 | Balancing permission and C1 authorisation, taken under delegated authority on the v2 test result: DIRECT (glaive only), UNSUPPORTED and CLARIFY provisionally permitted, CALL not; C1 cleared for 21's phases 3-5, no training authorised | [38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md](38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md) |
| 39 | canonical-v3 decision balance: the selection rule, fixed before it was computed, and what a flat decision prior costs in call supervision | [39-CANONICAL-V3-DECISION-BALANCE-SPEC.md](39-CANONICAL-V3-DECISION-BALANCE-SPEC.md) |
| 40 | **Adopted 2026-09-24:** `study_002_prereg_v8` — the truncation imbalance factor and the `P-UNANS` size the gate needs, the gate on `tool_use_promotion_v6`, and wording corrections | [40-PREREG-V8-DRAFT.md](40-PREREG-V8-DRAFT.md) |
| 41 | **Adopted 2026-09-24:** `study_002_prereg_v9`: the `ANSWER` strata's sources (BFCL no-call items and Natural Questions with tools that cannot serve them), construction and labelling | [41-ANSWER-STRATA-AMENDMENT.md](41-ANSWER-STRATA-AMENDMENT.md) |
| 42 | **Adopted 2026-09-24:** `study_002_prereg_v10`: the `ANSWER` strata reference uses two models that must both agree | [42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md](42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md) |
| 43 | **Adopted 2026-09-25:** `study_002_prereg_v11`: `P-UNANS` sources (KUQ, SelfAware), definition, strata, agreement floor, tools and scoring | [43-PUNANS-AMENDMENT-DRAFT.md](43-PUNANS-AMENDMENT-DRAFT.md) |
| 44 | **Adopted 2026-09-25:** `study_002_prereg_v12`: the second `P-UNANS` attempt: membership-only labels, a trial set, no false-premise stratum, fresh KUQ, KUQP and BIG-bench questions | [44-PUNANS-V2-AMENDMENT-DRAFT.md](44-PUNANS-V2-AMENDMENT-DRAFT.md) |
| 45 | **Adopted 2026-10-01:** `study_002_prereg_v13`: constructed `P-UNANS` questions (future winners, future dated measurements, with answerable controls) to reach check 4's 385 | [45-PUNANS-CONSTRUCTED-AMENDMENT-DRAFT.md](45-PUNANS-CONSTRUCTED-AMENDMENT-DRAFT.md) |
| 46 | **Adopted 2026-10-02:** `study_002_prereg_v14`: the readiness gate's design: `R2` and `C1` dropped, supervised tokens held fixed, the flag set and its triage, corrected text, `S2`, one measured instrument for model replies, `S-ANS-E` | [46-READINESS-DESIGN-AMENDMENT-DRAFT.md](46-READINESS-DESIGN-AMENDMENT-DRAFT.md) |
| 47 | **Adopted 2026-10-03:** `study_002_prereg_v15`: per-source floors for the flag-set triage (a failing Glaive or When2Call stops, ToolACE can be excluded, the pooled floors stay over all of `F`), the stratified trial and recall samples, recall per predicted label | [47-PER-SOURCE-FLAG-PRECISION-DRAFT.md](47-PER-SOURCE-FLAG-PRECISION-DRAFT.md) |
| 48 | **Plan, 2026-10-13 (not an amendment):** gate contract 5. Rules 15 and 46 §10 already state, now enforced (numbers bound to artifacts, descriptive populations), the order of the rest, and the items for the next amendment | [48-GATE-CONTRACT-5-DESIGN.md](48-GATE-CONTRACT-5-DESIGN.md) |

Frozen P-DET artifacts live outside this directory, beside the other evaluation artifacts:
`reports/pdet/pdet-v1.population.jsonl` and `reports/pdet/pdet-v1.manifest.json`, verified by
`python -m opengrad.verification.pdet --verify`. Annotation packages will be written as new artifacts under
`reports/pdet/annotation/` (the tool's working exports are under `annotation/wip/`; a gold freeze is a separate artifact). The pinned review queue is in
`reports/pdet/review/`, and the model-label audit trail in `reports/pdet/provenance/model-a/`. Both name
model labels, so an annotator does not open them.

## Reused, not reinvented

Every extension point below is a verified path in this repository.

| Existing machinery | Path | Study 002 use |
|---|---|---|
| Hardware-agnostic GPU preflight record | `registry/gpu_preflight.schema.json` | already `provider: [nvidia, amd, cpu, unknown]` with `cuda_version`/`rocm_version` both optional — the basis of the hardware-agnostic contract (13) |
| Provenance record | `registry/provenance.schema.json` | `run_id`, `status`, `compute_provider`, `hardware` — extended for device class and driver stack (13, 15) |
| Gate execution accounting | `src/opengrad/verification/accounting.py` | `NONVACUOUS_PREFIX`, `ACCOUNTING_PREFIX`, `REQUIRED_NONEMPTY` — every new gate declares a population policy (15, 16) |
| Repository readiness gates | `readiness()` in `src/opengrad/readiness.py`, states in `readiness_states.py` | gate names and `PASS/WARN/FAIL` + `error_code` convention extended (16) |
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
- Study 002 does **not** recommend deleting refusals from the corpus. A model that cannot report today's
  trending topics is behaving correctly, and removing those targets would trade over-refusal for undetectable
  hallucination. Arm `R3` deletes them only to *measure* that trade, and may never be promoted. Under
  `study_002_prereg_v14` the defect studied is the reply text, not the label: the trainer never renders the
  label ([46](46-READINESS-DESIGN-AMENDMENT-DRAFT.md) §3), so the correction is to answer questions that were
  wrongly declined.