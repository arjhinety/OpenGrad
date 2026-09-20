# 21 — C1 implementation status (provenance + schema normalization)

**Status: `PARTIAL — 7 of 9 phases landed` (updated 2026-09-20). canonical-v3 exists: 88,056 records, decision-balanced, every gate recorded. The renderer is proven unchanged across the C1 intervention (phase 7), and the pre-GPU provenance gate passes on the committed artifacts (phase 6). It is no arm's corpus, and no training was run.**

> **Status note, 2026-09-15.** Sources, adapters and the pre-classifier `normalization-v3` artifact now exist
> ([31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md)), as does the classifier input contract
> ([32](32-CLASSIFIER-INPUT-CONTRACT.md)). Schema translation is applied in the normalization-v3 builder,
> not wired into `adapt_when2call`, which is unchanged. Phases 3–4 (the classifier and the mixture) are
> still not started. So no canonical-v3 *training* artifact exists, and nothing below is otherwise revised.
> *(Superseded by the 2026-09-18 notes below: phases 3-5 have since landed. Kept as written, dated.)*

> **Status note, 2026-09-18 (later).** Phases 3, 4 and 5 landed: every normalization-v3 record carries a
> behaviour label, a decision-balanced selection plan exists (39), and **canonical-v3 is built** -- 88,056
> records, 22,014 per decision, every gate run and recorded (contamination, supervision, semantic trajectory,
> and a renderability pass over all 88,056 under the pinned Qwen3.5-2B renderer, 0 failures). It is not any
> arm's corpus (phase 8 is untouched) and **nothing is trained** (38 §4).
>
> **Status note, 2026-09-18.** The classifier gate is passed: `prose-decision-classifier-v2` is frozen and
> tested once on P-DET-COVERAGE-v2 (37 §7-§8), and **C1 is authorised** to proceed to phases 3-5 under the
> conditions of [38](38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md) §3, with balancing permission granted
> provisionally (DIRECT on glaive only, no CALL). At the time of this note phases 3-5 had not started; they
> landed the same day, as the note above records. No training run is authorised.

> **Status note, 2026-09-20.** Phase 7 is proven, and only phase 7 changed. The renderer is unchanged across
> the C1 intervention: the identity canonical-v3 recorded while rendering all 88,056 records equals the pinned
> Study 001 contract in `reports/evaluation/behavioral-heldout-v1.manifest.json`, the committed renderer
> snapshot `tests/fixtures/rendered/qwen35_2b_metadata.json`, and the renderer class constant
> (`Qwen35_2BRenderer.renderer_version`, `…model_revision`). The equality is required, not asserted, by
> `tests/data/test_canonical_v3.py::test_the_renderer_identity_is_unchanged_across_the_c1_intervention`, which
> reads only committed artifacts and so runs on a clean checkout. No artifact, hash, threshold, population,
> contract or metric was touched; canonical-v3's fingerprint is unchanged. This supersedes the `PENDING` status
> in the 2026-09-18 correction below, which is kept as written, dated. Phases 6, 8 and 9 remain open.

> **Status note, 2026-09-20 (later).** Phase 6 also landed: `src/opengrad/data/provenance_gate.py`, the
> pre-GPU provenance gate, six checks over the C1 artifacts on the shared accounting contract. It runs on a
> clean checkout, and it **fails**: canonical-v3's `versions.decision_classifier_version` names
> `prose-decision-classifier-v1` while its behaviour labels came from the frozen
> `prose-decision-classifier-v2`, so the gate returns `FAIL_CLASSIFIER_VERSION`. Every other check passes. By
> 38 §3 condition 5 a failing pre-GPU gate means **no training run**; the recorded run is
> `reports/canonical-v3/provenance-gate-v1.json`. The remediating rebuild (record the applied classifier in
> canonical-v3's version block and rebuild it, a new immutable artifact) is a separate owner decision and is
> **not** done here. Phases 8 and 9 remain open.

> **Status note, 2026-09-20 (latest).** The phase-6 finding is fixed. `canonical_v3.py` now records the
> classifier that actually labelled the records (`labels.classifier.version`) in the artifact's `versions`
> block, and canonical-v3 was rebuilt from the same inputs: the shards are **byte-identical**, `content_hash`
> is unchanged (`663c701e…`), and only `versions.decision_classifier_version` moved (`…-v1` → `…-v2`), so no
> other copy needed re-anchoring (G15). The gate now returns **`PASS`**, recorded as
> `reports/canonical-v3/provenance-gate-v2.json`; `…-v1.json` is kept as the finding's evidence, and a
> recorded run is never edited. `versions.py` was deliberately not touched — it is in normalization-v3's
> `CODE_MODULES`, so changing it would invalidate that frozen artifact — so the override lives in
> `canonical_v3.py`. Phases 8 and 9 remain open, and training is still not authorised.

This is a **new Study 002 intervention**, not a repair of Study 001. Nothing historical was touched: no v1/v2
artifact, manifest, hash, report or commit was modified or rewritten, and no Study 001 conclusion was
revised. Only three files were **added** in this phase; **zero existing files were changed**, so the
canonical-v2 code path is byte-identical to before and its outputs cannot drift.

| Phase | Requirement | Status |
|---|---|---|
| 1 | Provenance repair: one authoritative version source + disagreement invariant | **DONE** |
| 2 | Source-scoped schema normalization (shared validator untouched) | **DONE** |
| 3 | Deterministic versioned behaviour classifier | **DONE** (frozen `prose-decision-classifier-v2`, applied to all 181,433 records by `behaviour_labels.py`) |
| 4 | Wire decisions/capabilities into mixture machinery + materialized balance | **DONE for decisions** (`decision_balanced` mixture class + `canonical_v3_balance.py`, spec 39); capabilities remain unlabelled, so `balanced_policy_v1.yaml` stays HYPOTHESIS_ONLY |
| 5 | New immutable canonical-v3 artifacts | **DONE** (88,056 records, 9 shards, every gate run: 0 rejected, 88,056/88,056 rendered) |
| 6 | Pre-GPU validation gates | **DONE** — `src/opengrad/data/provenance_gate.py`, six checks on the accounting contract; committed-artifact verdict **PASS** after the classifier-version fix |
| 7 | Renderer unchanged, with equality proof | **DONE** — equality required by `tests/data/test_canonical_v3.py::test_the_renderer_identity_is_unchanged_across_the_c1_intervention` |
| 8 | Study 002 preregistration update | **NOT STARTED** |
| 9 | Full audit package | **THIS DOCUMENT** (partial by construction) |

## Phase 1 — provenance repair (done)

**Defect, verified:** `adapters.py:24` holds `ADAPTER_VERSION = "1.0.0"` and writes it into every record's
`metadata.adapter_version` (measured: `'1.0.0'` for all 1,000 records of a shard), while
`materialize.py:243` hard-codes `"adapter_version": "1.0.2"` into the manifest config. One artifact, two
adapter versions, so the corpus cannot be traced to the code that produced it.

**Fix:** `src/opengrad/data/versions.py` declares each versioned concern once (`ADAPTER_VERSION = "2.0.0"`,
`SCHEMA_NORMALIZATION_VERSION`, `DECISION_CLASSIFIER_VERSION`, plus taxonomy/schema/supervision versions)
and exposes two invariants:

- `check_version_agreement(record_metadata, manifest_config)` — fails with `PROV_VERSION_MISMATCH` when a
  record and its manifest disagree. Deliberately tolerant of *historical* versions: two layers that agree on
  an old version are internally consistent, and the published corpora are not edited to conform.
- `check_artifact_matches_authoritative_versions(manifest_config)` — fails when a **newly built** artifact
  claims a version other than the one this code produces.

The major bump is substantive, not cosmetic: decision derivation changes from message-shape inference to a
versioned classifier, so pre- and post-change records must not share a version string.

## Phase 2 — source-scoped schema normalization (done)

**Defect, verified and reproduced exactly:** When2Call's tool schemas are Python/typing annotations
(`parameters.type = "dict"` on all 34,287 tools; property types `"str, optional"` 18,607,
`"int, optional"` 8,010, `"bool, optional"` 1,936, `"List[int]"` 1,079, `"List[float]"` 956,
`"float, optional"` 495, `"List"` 299, `"Dict"` 172, `"List[List[int]]"` 170, `"set"` 152,
`"Tuple[float, float]"` 106). The canonical vocabulary is JSON Schema names, so 8,445 of 15,000 rows were
quarantined as `SCH_UNSUPPORTED_TYPE` — matching the release manifest's
`rejected_SCH_UNSUPPORTED_TYPE: 8445` to the row.

**Fix:** `src/opengrad/data/source_schema.py` translates at the **adapter boundary**; `opengrad.data.schema`
is untouched and still validates strictly *after* translation. Representable forms translate faithfully
(`str, optional` → `string` + `nullable`; `List[T]` → `array` with `items`; `Dict[K,V]` → `object`;
`Set[T]` → `array` + `uniqueness_not_enforced`; homogeneous `Tuple[T,...]` → `array` + `arity_not_enforced`;
`Optional[X]` / `Union[X,None]` → `X` + `nullable`). Forms whose semantics cannot be represented are
**quarantined with an explicit `SRC_*` reason**, never coerced: heterogeneous tuples
(`SRC_SCHEMA_HETEROGENEOUS_TUPLE`), multi-type unions (`SRC_SCHEMA_MULTI_TYPE_UNION`), unknown
parameterisations (`SRC_SCHEMA_UNREPRESENTABLE_TYPE`).

**Measured recovery** (read-only, over the cached upstream sources):

| source | rows | OK before | OK after | recovered | newly rejected |
|---|---|---|---|---|---|
| When2Call `train_sft` | 15,000 | 6,555 | **14,872** | **8,317** | **0** |
| ToolACE | 11,300 | 11,160 | 11,160 | 0 | 0 |
| Glaive | 112,960 | 112,827 | 112,827 | 0 | 0 |

Reconciliation: `6,555 − 12 (duplicate tool name) − 38 (duplicates) = 6,505` — the published accepted count,
so the reconstruction reproduces the released pipeline exactly. The 128 remaining When2Call quarantines are
`SRC_SCHEMA_MULTI_TYPE_UNION` 73 and `SRC_SCHEMA_UNREPRESENTABLE_TYPE` 55.

**ToolACE and Glaive recovery is genuinely 0**, and that is a finding rather than a shortfall: their losses
are a *different* class — `SCH_UNSUPPORTED_KEYWORD` (113 / 62), `SCH_PROPERTIES_NOT_OBJECT` 16,
`SCH_ADDITIONAL_PROPERTIES_NOT_BOOLEAN` (10 / 47), `SCH_REQUIRED_NOT_STRING_LIST` (0 / 20), and Glaive's
malformed-call syntax (1,094). Type translation cannot and should not fix those.

### The regression the measurements caught

Three annotation shapes made translation **stricter than the canonical layer**, so it *lost* rows instead of
recovering them: a nested schema under `type` (canonical recovers it, `schema.py:151-154`), a JSON-Schema
type *list* (`["string","null"]`), and an unparseable type value. Measured cost before the fix: 68 ToolACE
rows and 1 Glaive row. The contract is now explicit — **translation is never stricter than the canonical
layer on a type annotation** — and a multi-type list is dropped *with a recorded note*, exactly as canonical
drops it, so the coercion is visible rather than silent. A parametrized test guards all three shapes.

## Phase 6 — pre-GPU provenance gate (done 2026-09-20)

`src/opengrad/data/provenance_gate.py` is the module [`versions.py`](../../../src/opengrad/data/versions.py)
names as a consumer of its version fields but that never existed. It is the machine-checkable
provenance/identity gate over the C1 artifacts, built on the shared accounting contract
(`opengrad.verification.accounting`, doc [15](15-PROVENANCE-VALIDATORS.md)) and carrying its own
`PROVENANCE_GATE_CONTRACT = 1` — it does not bump the shared `VERIFIER_CONTRACT`, which governs
publication verification. It reads committed artifacts, so `python -m opengrad.data.provenance_gate
--verify` runs on a clean checkout; a git-ignored local build is reported `BLOCKED_INPUT_MISSING`, never a
pass, and a real disagreement is `FAIL`. It authorises no training.

Six checks, each a `ValidationResult`; the verdict on the committed artifacts:

| Check | Policy | Verdict |
|---|---|---|
| `versions_authoritative` — each C1 manifest's version block equals the authoritative constants | `REQUIRED_NONEMPTY` | `PASS` |
| `classifier_identity` — the applied classifier is the frozen `prose-decision-classifier-v2`, through a known contract, and agrees with any declared version | `REQUIRED_NONEMPTY` | `PASS` |
| `renderer_identity` — canonical-v3's recorded renderer equals the pinned Study 001 contract (phase 7) | `REQUIRED_NONEMPTY` | `PASS` |
| `authorisation_recorded` — each artifact names the 38 authorisation | `REQUIRED_NONEMPTY` | `PASS` |
| `record_version_agreement` — per record, metadata and manifest versions agree (the phase-1 invariant) | `CONDITIONALLY_REQUIRED` | `PASS` over 88,056 records; `BLOCKED_INPUT_MISSING` when the git-ignored shards are absent |
| `selection_plan_agreement` — canonical-v3 selects exactly the balance plan's ids | `CONDITIONALLY_REQUIRED` | `PASS` |

Recorded, never edited: [`provenance-gate-v1.json`](../../../reports/canonical-v3/provenance-gate-v1.json) holds
the original finding (overall **`FAIL`**); [`provenance-gate-v2.json`](../../../reports/canonical-v3/provenance-gate-v2.json)
holds the resolved state (overall **`PASS`**). A correction is a new record, not an edit. Tests:
`tests/data/test_provenance_gate.py` (every check has a fixture that makes it fail; the committed-artifact test
asserts the resolved state and would still catch a regression).

### The finding: `FAIL_CLASSIFIER_VERSION` (found, then fixed)

canonical-v3's manifest `versions.decision_classifier_version` **was** `prose-decision-classifier-v1` (from
`versions.provenance_versions()` / `DECISION_CLASSIFIER_VERSION`) while the artifact's behaviour labels came
from the frozen `prose-decision-classifier-v2`, as its own `labels.classifier` recorded. The version block had
never been checked before this gate.

**Fixed.** `canonical_v3.py` now records `labels.classifier.version` in the artifact's `versions` block, and
canonical-v3 was rebuilt from the same inputs: the shards are byte-identical and only that one field moved, so
`content_hash` (`663c701e…`) and every downstream reference are unchanged (no G15 re-anchoring was needed). The
override lives in `canonical_v3.py` rather than `versions.py` because the latter is in normalization-v3's
`CODE_MODULES`, where a byte change would invalidate that frozen artifact and force its own rebuild. No
threshold, population, contract, metric or other artifact was touched.

## Phase 7 — renderer identity (proven 2026-09-20)

Nothing in this phase touches the renderer: `renderers.py` was not modified, and neither was any module on
the render path. The proof the C1 specification requires — record `renderer_version` (`qwen3_5_2b_v1`) and
`template_hash` (`273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`) before and after, and
require equality — is now **executed and enforced** by
`tests/data/test_canonical_v3.py::test_the_renderer_identity_is_unchanged_across_the_c1_intervention`. It
requires equality across three committed records: the frozen Study 001 contract
(`reports/evaluation/behavioral-heldout-v1.manifest.json`, `model_renderer_contract`), the committed renderer
snapshot (`tests/fixtures/rendered/qwen35_2b_metadata.json`), and the identity canonical-v3 recorded while
rendering all 88,056 records (`reports/canonical-v3/canonical-v3.manifest.json`,
`gates.renderability.identity`). All three records agree, and the renderer class constant
(`Qwen35_2BRenderer`) is the fourth witness the test checks them against.

> **Correction, 2026-09-18.** This paragraph previously gave the template hash with three characters dropped
> (`…fb17e08d5f2a5ddfb53…`, 61 hex digits instead of 64). The authoritative value in
> `src/opengrad/evaluation/runner.py:33` was and is correct, and it is what the renderer produced while building
> canonical-v3 (39 §3, gate 5), so only this document was wrong. Phase 7's before-and-after equality proof is
> still `PENDING`: canonical-v3 records the renderer identity it observed, which is not the same as proving the
> renderer unchanged across the intervention.

> **Resolved, 2026-09-20.** The `PENDING` sentence above is superseded: the equality proof now exists as the
> test named in the paragraph before it, and the correction block is kept as written, dated. The limitation the
> block names is what the test closes — it asserts the equality across the recorded before-and-after identity
> rather than merely noting the value canonical-v3 observed.

## UNKNOWN / BLOCKED register

1. **`UNKNOWN` — whether the When2Call `train_sft` population contains any `direct`-gold items.** The split
   exposes only `tools` + `messages`; there is no label column to join on.
2. **`UNKNOWN` — the per-behaviour composition of the recovered When2Call rows** until the classifier exists
   and is independently validated.
3. **`BLOCKED` — classifier validation.** `P-DET` does not exist. The classifier may **not** be tuned or
   validated against Study 002's held-out partition, and the audit-time feasibility probe over MCQ response
   alternatives is explicitly **not** validation and may not be used as a threshold basis.
4. **`BLOCKED` — canonical-v3** on phases 3–4.
5. **Out of scope, documented not fixed:** the keyword-class rejections above, and `xlam_types.py`'s second
   annotation parser, which has its own reason codes and explicitly rejects `union` / `optional` /
   `callable` / `set` (`:98-103`) at a cost of 2,028 xLAM rows. Changing it would alter another source's
   semantics, so it is not folded into C1.

## Causation language (unchanged)

The training-side label collapse is a **plausible causal mechanism**, and the missing direct-answer
evaluation stratum explains why the Study 001 gate **could not see** the regression. **Causation is not
established**, and no report from this work may claim that the canonical-label collapse caused the GSM8K
regression. Only a controlled ablation can establish that. GSM8K stays a capability/direct-response
regression sentinel; Study 002 is not a math-answerability study, and at least one non-math direct-answer
probe is required.

## Files added in this phase

| File | Purpose |
|---|---|
| `src/opengrad/data/versions.py` | one authoritative version source + the two provenance invariants |
| `src/opengrad/data/source_schema.py` | source-scoped type-vocabulary translation + quarantine reasons |
| `tests/data/test_source_schema.py` | 64 tests: translation table, quarantine codes, idempotence, the never-stricter-than-canonical contract, tool-list shapes, both provenance invariants |
| `docs/research/study-002/21-C1-IMPLEMENTATION-STATUS.md` | this record |

No file was modified. Verified: `python -m pytest tests/data/test_source_schema.py` → 64 passed.

## Next, in order

1. `decision_classifier.py` — deterministic, versioned, emits the existing decisions/capabilities with
   `confidence="heuristic"` and classifier provenance; `known` reserved for genuine upstream labels.
2. `P-DET` sampling harness (deterministic, stratified, hashed) plus annotation protocol; validation stays
   `BLOCKED` until human labels exist.
3. Wire translation + classifier into `_base` / `adapt_when2call`, with every version derived from
   `versions.py`; then `normalization-v3` and the canonical-v3 release with new fingerprints.
4. Deterministic mixture materializer — none exists today (`mixture.py` only *validates* weights) — with
   frozen weights, caps, dedup, replacement policy and seed; realized counts read from the artifact.
5. Pre-GPU gate module covering every listed failure condition, the renderer equality proof, the
   preregistration update, and only then any consideration of training.