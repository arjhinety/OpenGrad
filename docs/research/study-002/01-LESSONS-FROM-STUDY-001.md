# 01 — Lessons from Study 001

**Posture: adversarial.** This document exists to make Study 002 harder to fool, not to re-litigate
Study 001. Study 001 is frozen at tag `study-001` (`docs/research/STUDIES.md:32`) and nothing here
changes a frozen number. Where a lesson is a defect, the defect is named, evidenced, and paired with
the Study 002 mechanism that would catch it.

Finding ids (`#27`, `#80`) are `id` values in
[`reports/audits/study-001-claim-audit/findings.json`](../../../reports/audits/study-001-claim-audit/findings.json).
That ledger holds **93 findings — 13 HIGH, 42 MED, 38 LOW** — and the pre-freeze site recheck added 7
(`R1`–`R7`), for **100 claims that the committed artifacts did not support**
(`docs/research/STUDIES.md:26-28,47-49`). Study 001's own guardrail document was written for this
study: *"Rules for Study 002 onwards, each derived from a mistake Study 001 actually made"*
(`docs/research/GUARDRAILS.md:3`), and it ends with the two checklists Study 002 must satisfy —
before GPU time (`:134-141`) and before publication or freeze (`:143-158`).

The 93 ledger findings distribute over 10 primary categories plus 12 compound labels:

| Category | n | | Category | n |
|---|---|---|---|---|
| `WRONG-NUMBER` | 19 | | `CHERRY-PICK` | 4 |
| `UNSUPPORTED` | 13 | | `MISSING-CAVEAT` | 3 |
| `MISLEADING` | 12 | | `GATE-MISAPPLIED` | 2 |
| `INTERNAL-CONTRADICTION` | 11 | | `REPRODUCIBILITY` | 1 |
| `STALE` | 10 | | compound labels | 12 |
| `WRONG` | 6 | | **total** | **93** |

**The distribution is the most important lesson in this document.** Zero findings are invented
measurements. The dominant failure is a number that drifted away from its artifact: 19 transcription
errors and 10 stale restatements against 0 fabrications. Study 001 did not lie; it failed to keep its
prose attached to its evidence. Every mechanism in this design set is therefore aimed at
**attachment** — generated numbers, declared populations, machine-checked claims — rather than at
sincerity.

---

## L1. The held-out set could not see the failure (G2)

**Defect.** The promotion partition contained tool-call, clarification and cannot-answer examples and
**no plain ANSWER examples**. A model that refuses every bare arithmetic question therefore passed
every gate applied to it. The regression was surfaced by a 7-case external sentinel, not by the
primary evaluation.

**Evidence.**
- `src/opengrad/promotion/tool_use_policy.py:33-37`: the v3 bump documents *"The frozen behaviour set
  contains zero ANSWER examples, so `no_call_accuracy` was 0.0 for every model including B0, and its
  0.40 floor rejected every candidate unconditionally."*
- `src/opengrad/promotion/tool_use_policy.py:64-72`: `measurable_dimensions()` exists because a
  per-class dimension over an empty class is **unmeasurable**, not failed.
- `docs/evaluation/CHECKPOINT_SELECTION_RULE.md:34-39`: the ranking *excludes* `no_call_accuracy`,
  because including it "would drag every score down by the same constant".
- `reports/FINAL_CAMPAIGN_AUDIT.md:26-28`: *"The regression was invisible to the promotion gate
  because the frozen confirmatory partition contains **no ANSWER examples**."*
- `reports/GENERAL_CAPABILITY_REGRESSION.md:4-6`: the regression was raised by the 7-case OpenWeights
  sentinel.
- Rule derived: G2, `docs/research/GUARDRAILS.md:24-29`.

**Root cause is a vacuity bug, not a statistics bug.** An empty class and a satisfied class were both
encoded as the float `0.0`. The verification layer had already been hardened against exactly this
(`src/opengrad/verification/accounting.py:1-19`) but the evaluation metrics never were.

**Study 002 mechanism.** Every metric declares a population policy and a denominator; a zero
denominator is reported `UNMEASURED`, never `0.0`; a gate FAILs when a declared required mode is
unmeasured ([06](06-SPLIT-SPEC.md), [07](07-METRIC-SPEC.md), [15](15-PROVENANCE-VALIDATORS.md)). The
new partition is *constructed* to be non-vacuous per mode, and that construction is asserted in CI.

---

## L2. Decision rules changed after the results existed (G1, G3)

**Defect.** The promoted artifact's advantage over its own parent is 7 of 453 calls on one seed, and
it was promoted under a gate introduced *after* its parent had been evaluated. The promotion reflects
a change of decision rule, not a measured improvement.

**Evidence.**
- `#2` (`CHERRY-PICK`, HIGH; `POSTTRAINING_PHASE_SUMMARY.md:3,12`, `M1_DPO_EVALUATION.md:3,53`,
  `M2_DECISION.md:10`, `docs/EXPERIMENT_RESULTS.md:12,31`). Claim: *"M0 not promoted → M1 promoted"*.
---

## L3. Prose facts were written from the plan, not the run record (G6)

**Defect.** The launch commit and the working-tree cleanliness of the largest ablation were recorded
wrong, in the direction of looking cleaner than they were.

**Evidence.**
- `#1` (`WRONG-NUMBER`, HIGH; `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md:55`). Claim:
  *"launch commits: matched 0807fa3 (both arms from clean trees)"*. Actual: *"Launched at 12:49 from
  500cb4e with git_dirty: True (experiment.json, events.jsonl). 0807fa3 is the 13:26 commit that
  recorded the finished run."*
- `#81` (`INTERNAL-CONTRADICTION`, HIGH; `results/benchmarks/checkpoint_ladder.json:217`,
  `final_campaign_verdict.json:599`, `reports/GENERAL_CAPABILITY_REGRESSION.md:39`,
  `reports/FINAL_CAMPAIGN_AUDIT.md:50`). Claim: the M1 historical identity sits on a different SFT
  parent (`CorpusV2`), so SFT introduced the regression and preference training did nothing. Actual:
  *"experiment.json and the Hub checkpoint-300 metadata show parent null and reference
  initial_policy: DPO directly on the base. It still refuses 70.7% of 0-shot GSM8K."* This single
  error would have changed the study's conclusion.
- Rule: G6, `docs/research/GUARDRAILS.md:50-57` — *"every lineage, reference, launch commit and dirty
  flag in prose is copied from `experiment.json` / `events.jsonl`, never from memory or the plan."*

**Study 002 mechanism.** Prose lineage claims are *rendered* from the run record by a generator, and a
CI check fails when a document states a commit, parent, dirty flag or seed that disagrees with the
run's `experiment.json` ([15](15-PROVENANCE-VALIDATORS.md)). Note that
`src/opengrad/experiments/preflight.py:166-184` already fixed a related bug — the git check was
reading a nested key that never existed and always printed "unknown". Study 002 treats that class of
silently-wrong output as a gate failure, not a bug report.

---

## L5. Populations were mixed inside a single table (G9)

**Defect.** Numbers computed on the full 3,650-example set appeared in rows labelled with the
1,277-example confirmatory partition. Study 001's own guardrail names four separate findings.

**Evidence.**
- `#9` (`INTERNAL-CONTRADICTION`, MED). Actual: *"These are the full 3,650-example numbers. B0 on the
  confirmatory partition is 0.6264 / 0.4618 / 0.9735 / 0.6238 / 0.1186 / 0.0177."*
- `#83` (`WRONG-NUMBER`, MED; public site). Actual: *"0.6191 and 64.3% are the full 3,650 set.
  Confirmatory B0: 0.6264 / P 0.4618 / R 0.9735 / over-call 0.6238."*
- `#84` (`WRONG-NUMBER`, MED): two public model cards carry the same wrong B0 row.
- `#53` and `#91` (`MISLEADING` / `INTERNAL-CONTRADICTION`): full-set rows under a partition heading,
  and one row mixing units — *"like-for-like is 48,723 of 101,785 trainable (47.9%)"*.
- `#42` (`MISLEADING`, LOW): cost figures paired across populations, including *"$0.757 for a run that
  returned no results. Non-reporting spend is $4.64 (52%)."*
- Rule: G9, `docs/research/GUARDRAILS.md:73-78` — each table and figure states its partition and n,
  every row is recomputed on that partition, and numerator and denominator share a counting unit.

**Study 002 mechanism.** Comparison tables are generated from result JSON, not typed; every row
carries `partition_id` and `n`; the renderer refuses to place rows with different `partition_id` under
one heading ([07](07-METRIC-SPEC.md), [15](15-PROVENANCE-VALIDATORS.md)).

---

## L6. Best-of-N selected on the evaluation set, then unlabelled (G10)

**Defect.** A checkpoint was selected by taking the best of four checkpoints **on the same set it was
then reported on**, and the row was not labelled as a selected maximum — while a neighbouring row for
another arm *was* labelled.

**Evidence.**
- `#62` (`CHERRY-PICK`, LOW; `docs/EXPERIMENT_RESULTS.md:27`). Actual: *"Best of 4 checkpoints, chosen
  on the same set; the final checkpoint scores 0.5292. Unlike the M1-v1 row, it isn't labelled
  best."*
- `#20` (`CHERRY-PICK`, LOW). Actual: the regression column *"leaves out unsupported −0.080, the
  largest regression, and clarification −0.027."*
- Rule: G10, `docs/research/GUARDRAILS.md:80-84`.

**Study 002 mechanism.** Selection uses DEV only, via the frozen rule
(`docs/evaluation/CHECKPOINT_SELECTION_RULE.md`); the confirmatory partition scores exactly the
protocol-mandated checkpoint per arm; any selected maximum is labelled `best-of-K` with K and the
selection set named; regression columns are *generated* from the full metric vector, so a metric
cannot be silently omitted ([04](04-ARM-MATRIX.md), [06](06-SPLIT-SPEC.md), [07](07-METRIC-SPEC.md)).

---

## L7. "Reproducible" and "preserved" were asserted, not demonstrated (G7)

**Defect.** "Reproducible" was claimed for a lineage whose only attempt at a repeat failed, and
"preserved" was claimed for artifacts that are not in git. Freeze pins could not be verified off the
author's machine at all.

**Evidence.**
- `#49` (`UNSUPPORTED`, HIGH; `README.md:247-249`, public). Actual: *"The only repeat ever attempted
  (m1_dpo_v1_restore) is marked NON_REPRODUCIBLE_REPEAT and its curves diverge. No other run has a
  repeat."*
- `#68` (`UNSUPPORTED`, MED; `docs/…/INCIDENT_LOG.md:110-113`). Actual: *"Only metrics.json/curve.json
  are in git; there are no checkpoints or predictions."*
- `#41` (`UNSUPPORTED`, LOW; `FINAL_CAMPAIGN_AUDIT.md:126-130`). Actual: *"M1_DPO_HISTORICAL/
  superseded_mmlu_768/ is empty; the 49.7% survives only in the ledger."*
- `#27` (`REPRODUCIBILITY`, MED; `results/benchmarks/h200/PRESERVED_STATE_v1.json`). Actual: *"The
  pins were computed over Windows CRLF working-tree bytes, so they never match the LF bytes a clone
  checks out. The state can't be verified off the author's machine."*
- `#12` (`UNSUPPORTED`, MED): confirmatory rows reported for checkpoints *"aren't named"*, with *"no
  per-partition artifact committed, the predictions untracked"*.
- Rule: G7, `docs/research/GUARDRAILS.md:59-64`.

**Study 002 mechanism.** "Reproducible" may only be claimed for an arm with a matching seed-repeat
whose artifacts agree within the tolerance in [05](05-SEED-AND-REPRODUCIBILITY-POLICY.md);
"preserved" only for paths that exist in the committed tree, verified by a CI job that walks the
declared manifest ([12](12-ARTIFACT-RETENTION.md)); all pins are computed over LF-normalised git blobs
and re-verified on a Linux clone.

---

## L8. Caveats did not travel with the numbers (G12)

**Defect.** Budget and truncation caveats were known internally and attached to no row. The most
consequential capability caveat was absent from every public surface.

**Evidence.**
- `#30` (`UNSUPPORTED`, MED; `reports/FINAL_CAMPAIGN_AUDIT.md:151,349-351`). Claim: raising the IFEval
  cap to 2,560 tokens removed the truncation confound. Actual: *"At 2560 tokens, 63 (M1-v2), 67 (M0)
  and 47 (Base) of 541 still hit the cap. No report discloses IFEval truncation."*
- `#51` (`MISLEADING`, MED; public). Actual: *"Base still truncates on 21.8% of items vs M0's 6.3%,
  and the adversarial interval is 6.4–33.1pp. EVALUATION.md breaks its own no-point-estimate rule
  here."*
- `#52` (`MISLEADING`, MED; `README.md:130-131`, public). Claim: *"answers 55.5% of the same 1,319
  questions with 8 exemplars"*. Actual: *"The answer rate is 100%; 55.5% is the accuracy."* — the same
  rate-versus-accuracy conflation as the headline.
- `#78` (`MISSING-CAVEAT`, HIGH; public model card + `release/huggingface/…-m1-dpo-canonicalv2-final-v2/README.md`).
  Actual: *"GSM8K 0-shot 67.4% → 0% (100% refusal), 8-shot 70.4 → 55.5, MMLU-Pro 49.0 → 37.0, IFEval
  strict 67.8 → 45.8. The site links here as the model."*
- `#79` (`MISSING-CAVEAT`, HIGH): the M0 SFT card is *"presented as the healthy M0, with no capability
  caveat"* — *"This is the stage where the regression first appears: GSM8K 0-shot 0%, IFEval 45.1%,
  MMLU-Pro 37.0%."*
- `#90` (`MISSING-CAVEAT`, MED): a normalization report saying *"FULL_DATA_VALIDATED; no blocker
  remains"* while *"silent on the Glaive adapter defect that collapsed v1 to 9 tool-call targets"*.
- Rule: G12, `docs/research/GUARDRAILS.md:91-96` — a budget, truncation or refusal caveat is attached
  to every row it affects, including model and dataset cards.

**Study 002 mechanism.** Caveats are fields on the metric record, not prose: `truncated_n`,
`budget_hit_rate`, `refusal_rate`, `answered_n`. The renderer emits the caveat row automatically
wherever the metric appears, and a card cannot be published while a `MISSING-CAVEAT` flag is set
([07](07-METRIC-SPEC.md), [12](12-ARTIFACT-RETENTION.md), [15](15-PROVENANCE-VALIDATORS.md)).

---

## L9. Causal language on a design that varied several things (G13)

**Defect.** The study's most-quoted sentences are causal ("SFT introduced…", "the DPO stage produced…",
"the engine change alone…") while the designs behind them varied parent, architecture, tokenizer and
hardware.

**Evidence.**
- `#31` (`OVERSTATED-CAUSATION`, MED). Actual: *"The verdict says ordering is not causation, and the
  Base edge also changes architecture and tokenizer. M1-v1 (DPO on base, no SFT) refuses 70.7%."*
- `#34` (`OVERSTATED`, MED; `reports/H200_BENCHMARK_RUN.md:117-120`). Actual: *"Hardware also differs
  (H200 vs A100), and 3 of the 21 flips are tokenizer-caused."*
- `#43` (`INTERNAL-CONTRADICTION`, LOW): *"The verdict supports runtime choice: hardware is
  confounded."*
- `#25` (`WRONG-NUMBER / OVERSTATED-CAUSATION`, HIGH; `reports/QAD_DECISION.md:85-91`). Actual: *"M0
  0.1505 < M1-v2 0.1529, so DPO raised over-call slightly. 0.1553 is the llama.cpp BF16 figure."*
- `#33` (`MISLEADING`, MED): a capability drop reported as *"the ability itself is worse"* while
  excluding 2,136 truncated Base items — *"On items both stages attempted the drop is 17.7pp, and no
  interval is given."*
- Rule: G13, `docs/research/GUARDRAILS.md:98-103`.

**Study 002 mechanism.** Each arm in [04](04-ARM-MATRIX.md) declares the single factor it varies and
the factors it holds fixed; a claim is written causally only if the factor is varied alone **and** the
interval excludes zero. Hardware, engine and dtype are recorded as provenance, and any cross-device
comparison is a pre-declared heterogeneity analysis, not a result
([13](13-HARDWARE-AGNOSTIC-EXECUTION.md), [14](14-HETEROGENEITY-POLICY.md)).

---

## L10. A gate that was applied to the wrong rung, and one hard-coded after scoring (G1, G10)

**Defect.** A pre-registered selection rule ("smallest passing format wins") was overridden in prose,
and the roles were assigned in code after the scores were known.

**Evidence.**
- `#24` (`GATE-MISAPPLIED`, HIGH; `reports/PTQ_PHASE_CLOSURE.md:22,29`,
  `QUANTIZATION_REPRODUCTION.md:132`). Claim: *"Q8_0 = RECOMMENDED_RELEASE, Q6_K =
  MEMORY_OPTIMIZED_RELEASE"*. Actual: *"The pre-registered rule is smallest passing format wins, and
  the closure manifest records recommended_release_rung: Q6_K. The roles were hard-coded after scoring
  (close_ptq_phase.py:48)."*
- `#26` (`CHERRY-PICK / UNSUPPORTED`, HIGH), quoted in L2.

**Study 002 mechanism.** Selection rules are executable: the release/decision function reads the
pre-registered rule and the result JSON and *computes* the designation, so a prose override is a
validator failure ([11](11-THRESHOLDS.md), [15](15-PROVENANCE-VALIDATORS.md)). No role string may be
assigned by a literal in a closure script.

---

## L11. Scaffolds described as working (G8)

**Defect.** Two documented capabilities did not exist: a tokenizer gate documented as fail-closed that
was hard-coded to pass, and an on-policy distillation path documented as supporting three modes whose
live path raises.

**Evidence.**
- `#66` (`UNSUPPORTED`, MED). Actual: *"agent_cli.py:711 hardcodes mock_compatible=True, so the gate
  always passes. The code's own mock gives Qwen2.5 a vocabulary of 151,936."*
- `#67` (`STALE`, MED). Actual: *"The live path raises NotImplementedError; M2 was closed as NOT RUN."*
- Rule: G8, `docs/research/GUARDRAILS.md:66-69` — *"a feature is described as working only if a test
  exercises its live path."*

**Study 002 mechanism.** Every scaffold Study 002 depends on is listed in
[16-GPU-READINESS-GATE.md](16-GPU-READINESS-GATE.md) with a status of `LIVE`, `INTERFACE_ONLY` or
`NOT_IMPLEMENTED`, and the status is asserted by a test that exercises the live path or asserts the
raise. Study 002 does not depend on the distillation scaffold at all (see [18](18-STUDY-003-ROADMAP.md)).

---

## L12. Public surfaces drifted from the repository (G17, G15)

**Defect.** Corrections landed in the repository and never reached the places people actually read —
model cards on the Hub, the website, and a released dataset card. Two HIGH findings are purely this.

**Evidence.**
- `#82` (`STALE + MISSING-CAVEAT`, HIGH; Hub card `arrochi112/OpenGrad-ToolPolicy-Canonical-v1`, rev
  `bb295d8a`). Actual: *"The repo template was updated but the Hub card wasn't. The card also never
  says v1 collapses to 9 tool-call targets out of 55,719 trainable rows (unparsed Glaive), so anyone
  training on it would reproduce the collapse."*
- `#78`, `#79` (both `MISSING-CAVEAT`, HIGH): the promoted model card and the M0 card omit the
  capability regression.
- `#83`, `#84`, `#85`, `#86` (`WRONG-NUMBER`, MED): the website and four cards carry full-set numbers
  under confirmatory headings, a quarantine table that *"sums to 12,502, but 11,271 were dropped"*, and
  an evaluation-example count of *"3,950 distinct"* where the truth is 3,652 (the 300 judge rows
  duplicate MCQ rows).
- `#60` (`WRONG-NUMBER`, LOW; `ROADMAP.md:127`): a cost stated as *"$8.90"* where *"the campaign total
  is $11.39; $8.90 is the continuation only."*
- Rules: G15 `docs/research/GUARDRAILS.md:114-118` (a correction is applied to every copy — four of
  the seven recheck findings were errors already fixed in the repository and never fixed on the site);
  G17 `:127-130`.

**Study 002 mechanism.** Public surfaces are **rendered from the same source as the report**; a release
check fails when a card, page or dataset card is byte-stale relative to its source, and a correction
carries a propagation checklist ([12](12-ARTIFACT-RETENTION.md)).

---

## L13. Three vacuity checks that Study 002 must not repeat at a larger scale

Study 001's verification layer was already the project's best feature, and two of its own checks are
evidence that vacuity is a systemic risk here, not a one-off:

- The **non-vacuity** policy in `src/opengrad/verification/__init__.py` distinguishes
  `REQUIRED_NONEMPTY` from `CONDITIONALLY_REQUIRED` and `OPTIONAL`, with `NONVACUOUS_PREFIX` and
  `ACCOUNTING_PREFIX` failure codes in `src/opengrad/verification/accounting.py:1-19`; the registry
  test `tests/registry/test_non_vacuity.py` exists specifically to catch a gate that would pass on an
  empty population.
- And yet the **evaluation metric layer had no such policy**, which is L1. The lesson generalises:
  verification hardening does not propagate to new measurement surfaces by itself. Study 002 therefore
  requires a non-vacuity declaration for every new metric, gate and partition *as a merge condition*,
  not as a later cleanup ([15](15-PROVENANCE-VALIDATORS.md), [16](16-GPU-READINESS-GATE.md)).

---

## Summary: defect → rule → Study 002 mechanism

| # | Defect (evidence) | Rule | Mechanism |
|---|---|---|---|
| L1 | Held-out set with no ANSWER examples (`tool_use_policy.py:33-37`) | G2 | Per-mode population policy; `UNMEASURED` ≠ `0.0`; ANSWER floor in v5 (06, 07, 11) |
| L2 | Gate introduced after results; promotion inside noise (`#2`, `#50`, `#76`) | G1, G3, G11 | Committed thresholds; re-score all candidates under both rules; noise band (03, 10, 11) |
| L3 | Launch commit and parent wrong (`#1`, `#81`) | G6 | Lineage rendered from `experiment.json`; CI cross-check (15) |
| L4 | Audit ran on the wrong corpus (`#80`) | G4 | Audit declares and is checked against the training-corpus hash (15) |
| L5 | Populations mixed in one table (`#9`, `#53`, `#83`, `#84`, `#91`) | G9 | Tables generated; `partition_id` + `n` per row; renderer refuses mixing (07, 15) |
| L6 | Best-of-4 on the eval set, unlabelled (`#62`, `#3`, `#20`) | G10 | DEV-only selection; `best-of-K` label; regression column generated (04, 06, 07) |
| L7 | "Reproducible"/"preserved" unproven (`#49`, `#68`, `#41`, `#27`) | G7 | Repeat required for the word; manifest walked in CI; LF pins (05, 12) |
| L8 | Caveats never travelled (`#30`, `#51`, `#52`, `#78`, `#79`, `#90`) | G12 | Caveats as metric fields, auto-rendered everywhere (07, 12) |
| L9 | Causal language on confounded designs (`#31`, `#34`, `#43`, `#25`, `#33`) | G13 | One factor per arm; causal wording requires a single varied factor + interval (02, 04, 13, 14) |
| L10 | Rule applied to the wrong rung, roles hard-coded (`#24`, `#26`) | G1, G10 | Designation computed from the rule + result JSON (11, 15) |
| L11 | Scaffolds described as working (`#66`, `#67`) | G8 | `LIVE`/`INTERFACE_ONLY`/`NOT_IMPLEMENTED` asserted by test (16) |
| L12 | Public surfaces drifted, corrections unpropagated (`#82`, `#78`, `#79`, `#83`–`#86`, `#60`) | G15, G17 | Surfaces rendered from the report source; staleness check (12) |
| L13 | Vacuity existed at the gate layer but not the metric layer | G2 | Non-vacuity declaration is a merge condition for every new surface (15, 16) |

**The one-sentence lesson.** Study 001's errors were almost all *detachment* errors — a true number
retyped in the wrong place, or a rule revised after the number existed — so Study 002's defences are
aimed at keeping numbers, populations, rules and prose in one generated chain rather than at writing
with more care.

---

## L4. The audited corpus was not the trained corpus (G4)

**Defect.** The refusal-supervision audit that motivates the entire Study 002 intervention was run
over `normalization-v1` sources, not the published Canonical-v2 corpus M0 actually trained on. The
headline count was wrong by ~3,600 records, and one per-source count exceeded the number of records
M0 ever saw.

**Evidence.**
- `#80` (`WRONG-NUMBER`, HIGH; `results/…/sft_refusal_supervision_audit.json`,
  `scripts/audit_sft_refusal_supervision.py:41-46`, `reports/FINAL_CAMPAIGN_AUDIT.md:179-181`,
  `ROADMAP.md:91,108`). Claim: *"the training corpus: 21,749 of 217,903 (10.0%); when2call-sft 7,490 /
  14,829 (50.5%)"*. Actual: *"The audit read normalization-v1: held-out rows, BUTTON, LoopTool and the
  v1 Glaive adapter. The same detector on published Canonical-v2 gives 18,114 of 173,237 (10.5%);
  When2Call 4,038 / 6,505 (62.1%). The 7,490 exceeds the 6,505 When2Call records M0 trained on."*
- Rule: G4, `docs/research/GUARDRAILS.md:37-41` — re-derive the count on the artifact actually trained
  on before proceeding.

**Study 002 mechanism.** Every audit record declares the dataset hash it was computed over, and the
pre-registration refuses to accept an audit whose hash ≠ the corpus hash in the training manifest
([15](15-PROVENANCE-VALIDATORS.md)). The generator writes the hash and the count into the same JSON.
  Actual: *"M1-v2 fails v3, the gate M0 failed, on all 4 DEV checkpoints. It was promoted under v4,
  committed after M0's confirmatory results, and M0@1800 also passes v4. None of this is disclosed."*
- `#50` (`CHERRY-PICK`, HIGH; `README.md:100-101`). Actual: *"v4 compares against the M0 parent, not
  B0. Under v3 every M1-v2 checkpoint is REJECT; DEV recall 0.7257 vs B0 0.9715."*
- `#76` (`UNSUPPORTED`, MED; `docs/research/methodology.md:3`). Actual: *"The promoted M1-v2 is
  single-seed with no interval; its +0.0078 call_f1 over M0 is 7 more correct calls out of 453."*
- `#3` (`WRONG`, HIGH; `docs/EXPERIMENT_RESULTS.md:27`). A cell reading *"Regression: None
  measured"* hid recall `0.9722 → 0.505 (−0.467)`.
- Rules: G1 `docs/research/GUARDRAILS.md:16-22`; G3 `:31-35`; G10 `:80-84`; G11 `:86-89`.

**Study 002 mechanism.** The gate version, all thresholds and the decision rule are committed before
the first candidate is scored, in [03](03-PREREGISTRATION.md) and [11](11-THRESHOLDS.md). A revised
rule must be re-scored against **every** earlier candidate and both versions reported. Gate margins
must exceed the measured rerun noise band ([10](10-STATISTICS-PLAN.md)), which Study 001 knew it had
not done: `#26` records that *"the H200 vLLM BF16 rerun itself fails the gate: recall 0.7638 <
0.7671. Q6_K clears precision by one example (357/490 vs 356.96). Gate pass/fail is inside rerun
noise, and no document says so."*