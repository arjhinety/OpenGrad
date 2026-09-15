# 24 — Phase 3 report: P-DET frozen

**Protocol:** `pdet-002-v1` · [22-PDET-PROTOCOL.md](22-PDET-PROTOCOL.md) · **Frozen sample:**
`reports/pdet/pdet-v1.population.jsonl` ·
`population_sha256 = 6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b`

**Outcome: the protocol and the validation population are `FROZEN`. The gold labels are `BLOCKED` on human
annotation.** Phase 3 is therefore **not** fully closed — a frozen *population without gold labels* cannot
yet evaluate a classifier — and this report says so rather than implying otherwise.

## 1. Research purpose

P-DET answers one question before the classifier is allowed to touch training data: *can a deterministic
prose classifier distinguish the Study 002 policy modes reliably enough that its inferred labels may drive
corpus construction and behaviour-balanced sampling?* It validates `HEURISTIC_POLICY_v1` — an **ETL
component, not a model** — and its results are **not** evidence that Study 002's policy improved.

## 2. Frozen definitions

`CALL` (machine-readable payload present) · `DIRECT`/`ANSWER` (substantive content delivered, no tool, no
required request, no decline) · `CLARIFY` (request *necessary* to proceed correctly) · `UNSUPPORTED`
(cannot act with available capability; decline or explain) · `UNKNOWN`/`AMBIGUOUS` (**an annotation status,
not a fifth mode**: the rubric cannot resolve the item to one mode). Full positive criteria, exclusion
criteria and the eight boundary discriminations are in [22 §1–§2](22-PDET-PROTOCOL.md); the semantic decision
tree is [22 §3](22-PDET-PROTOCOL.md). The rubric classifies *intended behaviour in context* — a trailing `?`
is not automatically `CLARIFY`, and refusal-shaped wording is not automatically `UNSUPPORTED`.

## 3. Sampling design and actual frozen counts

| Quantity | Target | **Realized** |
|---|---|---|
| Prevalence component | 400 | **400** |
| Challenge component | 200 | **181** (shortfall 19, **not** backfilled) |
| **Total** | 600 | **581** |

Seed `opengrad-pdet-002-v1`; ranking `sha256(seed ‖ pdet_id)` descending; no RNG state. Candidate population
14,829 → **14,828** eligible (1 dropped) → **12,402** after dedup → 581 selected. All 581 `pdet_id`s,
`raw_record_hash`es, `upstream_id`s and responses are pairwise distinct.

## 4. Source distribution

| Field | Value |
|---|---|
| `source_dataset` | `when2call` ×581 |
| `source_split` | `train_sft` ×581 |
| `source_revision` | **`null` ×581** — a provenance gap: the upstream revision is recorded in the manifest (`source_sha256 f992975f…`) and the release config (`0582f774…`), **not** per record |
| shards represented | 15 of 15 |
| dedup effect | 2,426 responses dropped as duplicate text, across 1,160 groups |

## 5. Challenge-set construction

Ten families fired, selected from frozen predicates over observable text only:

| family | items |
|---|---|
| `refusal_plain` | 93 |
| `question_mark` | 56 |
| `caveat_then_content` | 51 |
| `no_tools_offered` | 38 |
| `question_without_mark` | 32 |
| `tool_mentioned_no_payload` | 25 |
| `advice_external_service` | 21 |
| `short_plain` | 20 |
| `polite_followup` | 18 |
| `refusal_with_question` | **3** |

Membership counts overlap (an item may satisfy several families); the manifest records the assignment rule.

**Two family shortfalls, reported not repaired:** `refusal_with_question` had only **3** eligible items
against a quota of 20 (shortfall 17) and `polite_followup` 18 of 20 (shortfall 2). This is a **finding about
the data**, and an important one: in this population, refusal-shaped prose that *also* asks a question is
nearly absent — precisely the decline-versus-clarify boundary the intervention most needs to separate.

## 6. Annotation process and agreement

**Not executed.** No human annotation has occurred; `gold_policy_label`, `ambiguity_status`, `annotator_id`,
`annotator_rationale`, `boundary_rule_cited` and `annotation_version` are `null` for all 581 items. Therefore
**no inter-annotator agreement exists to report**, and under [22 §4](22-PDET-PROTOCOL.md) the single-annotator
limitation applies and must be stated wherever these labels are used. The annotation instrument is the frozen
population itself: each record already carries the context shown, the response, the tool schema, and every
provenance field.

## 7. Gold label distribution

**`BLOCKED`.** No gold labels exist. The distribution, the per-class `n`, and whether the 50-per-mode coverage
requirement is met are all **UNKNOWN** until annotation runs. By the coverage rule, any mode that cannot
reach 50 gold examples will be reported as a finding and will **not** qualify for balancing.

## 8. Frozen classifier metrics and acceptance thresholds (for Phase 4)

Metrics: overall accuracy · macro F1 · per-class precision/recall/F1 · full confusion matrix · abstention
rate · **each reported separately on the prevalence and challenge subsets**.

| Criterion | Threshold |
|---|---|
| `DIRECT` recall | ≥ 0.80 |
| `DIRECT` precision | ≥ 0.80 |
| `UNSUPPORTED` recall | ≥ 0.75 |
| `CLARIFY` F1 | ≥ 0.70 |
| `CALL` precision | ≥ 0.95 |
| macro F1 | ≥ 0.75 |
| abstention rate | ≤ 0.15 |
| each mode, challenge subset | recall ≥ 0.60 |

These are **Study 002 design choices**, not empirically established values, and are **not** derived from the
audit-time MCQ probe — that probe is prior exploratory evidence only and may not seed or justify thresholds
after the fact. Failure behaviour: a mode below threshold is denied balancing permission; if `DIRECT` or
`UNSUPPORTED` fails, **C1 is not authorised**; and **iterating regexes against P-DET until it passes is
prohibited** — a post-P-DET classifier change increments the version, marks P-DET `DEVELOPMENT_EXPOSED`, and
requires a **second untouched validation set**.

## 9. Contamination audit

Checked against 2,295 held-out unique questions, 3,650 evaluation example ids (1,277 confirmatory + 2,373
DEV) and 2 quarantined ids:

| Check | Result |
|---|---|
| frozen item whose prompt equals a held-out question | **0** |
| frozen item carrying a confirmatory or DEV example id | **0** |
| frozen item carrying a quarantined id | **0** |
| frozen item overlapping any evaluation-only namespace | **0** |

One candidate was dropped for held-out prompt overlap (the record documented in
`behavioral-heldout-v2-quarantine.json`), which is why eligibility is 14,828 rather than 14,829. Excluding
both the confirmatory **and** DEV partitions is deliberate: DEV is used repeatedly for checkpoint selection,
so it is model-evaluation material and cannot serve as development data.

## 10. Artifact hashes and manifest locations

| Artifact | Path | Size |
|---|---|---|
| Frozen population | `reports/pdet/pdet-v1.population.jsonl` | 1,247,612 B |
| Manifest | `reports/pdet/pdet-v1.manifest.json` | 4,459 B |
| Manifest digest sidecar | `reports/pdet/pdet-v1.manifest.json.sha256` | 66 B |

`population_sha256 = 6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b`. The manifest records
the protocol version, seed, ranking rule, candidate-population funnel, targets versus realized sizes, exclusion
and dedup statistics, strata quotas versus realized, challenge family counts and shortfalls,
`coverage_requirement_per_mode: 50`, the excluded populations, `classifier_status_at_selection:
NOT_IMPLEMENTED`, `selection_used_classifier: false`, `gold_labels_present: false`, the blocked-on note, and
the statement that examples were chosen from the frozen specification and observable dataset characteristics
only. The verifier is `src/opengrad/verification/pdet.py` (`--build` / `--verify`).

**Reproducibility was proved, not asserted:** a rebuild produced byte-identical output and `--verify` returns
`PASS` with zero errors. Four tamper tests prove the verifier *fails* on an edited population, an edited
manifest, a manifest that contradicts the labels, and on missing artifacts. 29 tests pass in
`tests/verification/test_pdet.py`.

## 11. Remaining UNKNOWN / BLOCKED

1. **`BLOCKED` — gold labels.** Require human annotation. Nothing downstream (classifier qualification,
   balancing, C1 authorisation) can proceed without them.
2. **`UNKNOWN` — per-mode gold coverage**, and therefore whether the 50-per-mode rule can be met.
3. **`UNKNOWN` — inter-annotator agreement** (no annotators yet).
4. **`UNKNOWN` — `source_revision` per record**; it is `null` in the frozen records and available only at
   manifest/config level. Fixing this belongs to the v3 adapter, not to P-DET.
5. **Known limitation — challenge-family oversubscription:** items satisfying several boundaries are assigned
   to the alphabetically first family with quota, so family counts understate total boundary membership.
6. **Known limitation — predicate-based families** are observable proxies (cue lists), deliberately simple.
   They select stress cases; they do **not** label anything, and they are not the classifier's rules.
7. **Carried forward from Phase 2:** `decision_classifier.py` is unimplemented, the adapter is unwired,
   canonical-v3 does not exist, no mixture materializer exists, and the pre-GPU gate module is partial.

## 12. Confirmation of what Phase 3 did not touch

- `decision_classifier.py` — **not created, not implemented, not tuned.**
- Adapters — **not modified**; no classifier integrated; `adapters.py` unchanged.
- Canonical-v3 — **not materialized**; no new normalization or release artifact.
- Renderer — **not modified**; no decision token, tag, hidden control marker or new decision field
  introduced into any prompt or target. Any renderer-visible C2 intervention remains a separately
  preregistered future arm.
- Mixture logic — **not changed**; `balanced_policy_v1.yaml` remains `HYPOTHESIS_ONLY`.
- GPU — **no GPU work**; every step was CPU/local.
- Study 001 and canonical-v1/v2 — **untouched**: `git diff --stat` shows only the two documentation files
  modified during the earlier design-set work; every Phase 2 and Phase 3 file is new and untracked.
- The audit-time MCQ feasibility probe was **not** used as classifier validation and is documented only as
  prior exploratory evidence.

## Verdict

**Is the protocol sufficient to begin classifier implementation?**

The *validation protocol* is ready: rubric, decision tree, sampling rule, hash-pinned frozen population,
preregistered metrics and thresholds, the anti-overfitting rule, and a working freeze verifier all exist, and
the freeze is reproducible to the byte.

But **Phase 4 must not measure a classifier against gold labels that do not exist.** The immediate next step
is therefore **human annotation of the 581 frozen items under [22 §3–§4](22-PDET-PROTOCOL.md)** — not
classifier code. If that annotation is not feasible, the honest alternative is to record that heuristic labels
**cannot** be validated and therefore **may not** drive behaviour balancing, which would leave C1
unauthorised. Phase 4 has **not** been started.