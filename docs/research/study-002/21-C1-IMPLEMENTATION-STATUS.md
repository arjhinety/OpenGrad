# 21 — C1 implementation status (provenance + schema normalization)

**Status: `COMPLETE — 9 of 9 phases landed` (updated 2026-09-20). C1 is built: canonical-v3 exists (88,056 records, decision-balanced, every gate recorded), the renderer is proven unchanged (phase 7), the pre-GPU provenance gate passes (phase 6), and the corpus is registered in the preregistration as built but not an arm (phase 8). It is no arm's corpus, and no training was run.**

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

> **Status note, 2026-09-20 (phase 8).** The preregistration records the C1 corpus.
> [03](03-PREREGISTRATION.md) gains a **registration, not an amendment**: canonical-v3 exists, the
> provenance gate passes, and **no arm of [04](04-ARM-MATRIX.md) moves to it**. No threshold, arm, seed,
> partition or decision rule changed. Entering a study stays a separate owner decision (31 §9.6). Phase 9
> (the audit package) is this document.

> **Status note, 2026-09-20 (phase 9).** The audit package is complete: phases 3–5 and 8 are recorded, an
> evidence index maps every C1 claim to its artifact, and the register below is resolved. C1's
> implementation is complete. Training is still **not** authorised: [16](16-GPU-READINESS-GATE.md) has
> produced no `READY` record, and the two remaining decisions are the study owner's (31 §9.6, [35](35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md)).

This is a **new Study 002 intervention**, not a repair of Study 001. Nothing historical was touched: no
Study 001 artifact, canonical-v1/v2 manifest or hash, P-DET-v1 population, or frozen report was modified or
rewritten, and no Study 001 conclusion was revised. Phases 1–2 added three files and changed none, so the
canonical-v2 code path is byte-identical; later phases added their own modules, and the only change to an
existing module was the phase-6 `canonical_v3.py` fix, which moved one field of canonical-v3's own manifest
and left every shard byte-identical.

| Phase | Requirement | Status |
|---|---|---|
| 1 | Provenance repair: one authoritative version source + disagreement invariant | **DONE** |
| 2 | Source-scoped schema normalization (shared validator untouched) | **DONE** |
| 3 | Deterministic versioned behaviour classifier | **DONE** (frozen `prose-decision-classifier-v2`, applied to all 181,433 records by `behaviour_labels.py`) |
| 4 | Wire decisions/capabilities into mixture machinery + materialized balance | **DONE for decisions** (`decision_balanced` mixture class + `canonical_v3_balance.py`, spec 39); capabilities remain unlabelled, so `balanced_policy_v1.yaml` stays HYPOTHESIS_ONLY |
| 5 | New immutable canonical-v3 artifacts | **DONE** (88,056 records, 9 shards, every gate run: 0 rejected, 88,056/88,056 rendered) |
| 6 | Pre-GPU validation gates | **DONE** — `src/opengrad/data/provenance_gate.py`, six checks on the accounting contract; committed-artifact verdict **PASS** after the classifier-version fix |
| 7 | Renderer unchanged, with equality proof | **DONE** — equality required by `tests/data/test_canonical_v3.py::test_the_renderer_identity_is_unchanged_across_the_c1_intervention` |
| 8 | Study 002 preregistration update | **DONE** — a registration, not an amendment: canonical-v3 recorded in [03](03-PREREGISTRATION.md) as built and not an arm's corpus; entering a study stays a separate owner decision |
| 9 | Full audit package | **DONE** — this document, completed 2026-09-20: phases 3–5 and 8 recorded, an evidence index added, the register resolved |

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

## Phase 3 — the behaviour classifier applied (done 2026-09-18)

`src/opengrad/data/behaviour_labels.py` (21 phase 3) runs the frozen `prose-decision-classifier-v2` over
every normalization-v3 record's first assistant reply, through contract `prose-decision-input-v2`, and
writes one label per record to `data/processed/behaviour-labels-v1/`, with counts in
`reports/canonical-v3/behaviour-labels-v1.counts.json`.

- **181,433 records labelled:** `CALL_BY_STRUCTURE` 92,851, `DIRECT` 34,190, `CLARIFY` 22,014,
  `UNSUPPORTED` 22,419, `ABSTAIN` 7,791, `UNLABELLED` 2,156, `CALL` 12.
- **`weight_permitted` follows 38 §2 exactly:** UNSUPPORTED and CLARIFY on both layer B sources, DIRECT on
  **glaive only** (33,927 of 34,190), never `CALL`/`CALL_BY_STRUCTURE`/`ABSTAIN`/`UNLABELLED`.
- The pass refuses to run unless the classifier source is byte-for-byte the frozen one (tag
  `prose-decision-classifier-v2`, source sha256 LF `47436ca9…`); `tests/data/test_behaviour_labels.py`
  asserts the pin equals the one-shot runner's constant.
- It writes labels only: no mixture, no sampling weight, no training corpus.

## Phase 4 — the decision balance (done 2026-09-18)

`src/opengrad/data/canonical_v3_balance.py` (21 phase 4) implements the rule of
[39](39-CANONICAL-V3-DECISION-BALANCE-SPEC.md), fixed **before** it was computed: equal shares over the four
decisions, supply-limited, ranked deterministically by `sha256(seed | record id)` with seed
`opengrad-canonical-v3-decision-balance-v1`. It is a **plan, not a corpus** — no shard, no arm, no training.

- Supply: `CALL` 92,851 (structural), `ANSWER` 33,927 (glaive DIRECT), `CLARIFY` 22,014, `UNSUPPORTED` 22,419.
- The binding stratum is `CLARIFY` at 22,014, so the plan selects **88,056** records and leaves **70,837**
  structural-call records out. That loss is the price of a flat decision prior and is stated in 39 §2.
- Recorded at `reports/canonical-v3/decision-balance-v1.json`; `--verify` recomputes it byte for byte.
  `tests/data/test_canonical_v3_balance.py` covers it. Config
  `configs/data/tool_calling/decision_balance_v1.yaml` (`mixture_class: decision_balanced`).
- `balanced_policy_v1.yaml` (capabilities) stays `HYPOTHESIS_ONLY`: no capability labeller exists (39 §1).

## Phase 5 — the canonical-v3 artifact (done 2026-09-18)

`src/opengrad/data/canonical_v3.py` (21 phase 5) materialises the plan into an immutable artifact at
`data/processed/canonical-v3/` (git-ignored), tracked by `reports/canonical-v3/canonical-v3.manifest.json`.
Each record gains one `metadata.behavior` block in the *mixture* vocabulary — the classifier's `DIRECT`
becomes `ANSWER`; `confidence` is `known` for structural `CALL` and `heuristic` otherwise; `capabilities` is
always empty, because no capability labeller exists.

- **88,056 records, 22,014 per decision, 9 shards;** `content_hash` `663c701e…`.
- **Five gates run before any shard is written, each recorded with its counts:** membership, contamination,
  supervision kind, semantic trajectory, renderability. **0 rejected**; 88,056/88,056 rendered under the
  pinned Qwen3.5-2B renderer (identity `qwen3_5_2b_v1`, template `273d8e0e…`).
- `--build` refuses to overwrite; `--verify` re-hashes every shard and re-checks the decisions. It never
  writes into normalization-v3, whose `verify` still refuses a behaviour label on a record.
- No arm uses it and no training is authorised (38 §4). Its `versions` block names the classifier that
  labelled it — the phase-6 fix below.

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
> `PINNED_TEMPLATE_HASH` (`runner.py:33` then, `evaluation/backends.py` from 2026-09-24) was and is correct, and it is what the renderer produced while building
> canonical-v3 (39 §3, gate 5), so only this document was wrong. Phase 7's before-and-after equality proof is
> still `PENDING`: canonical-v3 records the renderer identity it observed, which is not the same as proving the
> renderer unchanged across the intervention.

> **Resolved, 2026-09-20.** The `PENDING` sentence above is superseded: the equality proof now exists as the
> test named in the paragraph before it, and the correction block is kept as written, dated. The limitation the
> block names is what the test closes — it asserts the equality across the recorded before-and-after identity
> rather than merely noting the value canonical-v3 observed.

## Phase 8 — preregistration registration (done 2026-09-20)

Recorded in [03](03-PREREGISTRATION.md) as a **registration, not an amendment**: canonical-v3 exists, the
pre-GPU provenance gate passes, and **no arm of [04](04-ARM-MATRIX.md) moves to it**. No threshold, arm,
seed, partition or decision rule changed, so nothing already scored is re-run. Entering a study stays a
separate owner decision ([31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) §9.6), and no training is
authorised.

## Phase 9 — the audit package (this document)

This document is the C1 audit package. It records every phase, names the artifact behind every claim, and
resolves the register below.

| Claim | Artifact | Checked by |
|---|---|---|
| One authoritative version source; record and manifest versions agree | `src/opengrad/data/versions.py` | `tests/data/test_source_schema.py`; the gate's `record_version_agreement` |
| Source-scoped schema translation is never stricter than the canonical layer | `src/opengrad/data/source_schema.py` | `tests/data/test_source_schema.py` (64 tests) |
| 181,433 records labelled by the frozen classifier, with 38 §2 weights | `reports/canonical-v3/behaviour-labels-v1.counts.json` | `tests/data/test_behaviour_labels.py` |
| The plan is 88,056 records, equal per decision, reproducible | `reports/canonical-v3/decision-balance-v1.json` | `python -m opengrad.data.canonical_v3_balance --verify`; `tests/data/test_canonical_v3_balance.py` |
| canonical-v3 is 88,056 records, 0 gate rejections, all renderable | `reports/canonical-v3/canonical-v3.manifest.json` | `python -m opengrad.data.canonical_v3 --verify`; `tests/data/test_canonical_v3.py` |
| The renderer is unchanged across the intervention | the manifest's `gates.renderability.identity`, against the frozen Study 001 contract | `tests/data/test_canonical_v3.py::test_the_renderer_identity_is_unchanged_across_the_c1_intervention` |
| Provenance passes on the committed artifacts | `reports/canonical-v3/provenance-gate-v2.json` | `python -m opengrad.data.provenance_gate --verify`; `tests/data/test_provenance_gate.py` |
| canonical-v3 is not an arm's corpus; no training is authorised | [03](03-PREREGISTRATION.md)'s registration; [16](16-GPU-READINESS-GATE.md) has no `READY` record | this document; [04](04-ARM-MATRIX.md) |

Every number above is read from the named artifact, not retyped (G14). No frozen Study 001 artifact,
canonical-v1/v2, P-DET-v1 or P-DET-COVERAGE population was modified by any phase.

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

> **Update, 2026-09-20.** Items 2–4 are resolved by the phases above and are kept as written, dated: the
> classifier exists and is frozen (`prose-decision-classifier-v2`) and validated on P-DET-COVERAGE-v2
> (37 §8); P-DET-v1 exists (581 items, frozen population); canonical-v3 is built. Item 1 — whether the
> When2Call `train_sft` split holds any `direct`-gold item — stays `UNKNOWN`. Item 5 is unchanged.

## Causation language (unchanged)

The training-side label collapse is a **plausible causal mechanism**, and the missing direct-answer
evaluation stratum explains why the Study 001 gate **could not see** the regression. **Causation is not
established**, and no report from this work may claim that the canonical-label collapse caused the GSM8K
regression. Only a controlled ablation can establish that. GSM8K stays a capability/direct-response
regression sentinel; Study 002 is not a math-answerability study, and at least one non-math direct-answer
probe is required.

## Files added in phases 1–2

| File | Purpose |
|---|---|
| `src/opengrad/data/versions.py` | one authoritative version source + the two provenance invariants |
| `src/opengrad/data/source_schema.py` | source-scoped type-vocabulary translation + quarantine reasons |
| `tests/data/test_source_schema.py` | 64 tests: translation table, quarantine codes, idempotence, the never-stricter-than-canonical contract, tool-list shapes, both provenance invariants |
| `docs/research/study-002/21-C1-IMPLEMENTATION-STATUS.md` | this record |

No file was modified in phases 1–2. Verified: `python -m pytest tests/data/test_source_schema.py` → 64 passed.

## Next, in order

Phases 1–9 have landed. What remains is not engineering:

1. **The study owner's decisions**, which no phase above took:
   - whether canonical-v3 enters Study 002 or 003, and as which arm or factor ([31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) §9.6,
     [03](03-PREREGISTRATION.md)'s registration);
   - whether to re-read the 2 UNKNOWN P-DET-v1 items and freeze its gold ([35](35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md)).
2. **[16](16-GPU-READINESS-GATE.md), the pre-training readiness gate** — the fourteen blocking checks
   (`study_002_gate_v1`), most of which need artifacts that do not exist yet (P-CONF, sentinels, the
   answerability partition). It has produced no `READY` record, so **no training run is authorised**.
3. **Only then**, any consideration of training.

The list this section carried earlier — the classifier, the P-DET harness, wiring the classifier into the
artifacts, a deterministic mixture materializer, and a pre-GPU gate module — is complete:
`decision_classifier_v2.py`, `canonical_v3_balance.py`, `canonical_v3.py` and `provenance_gate.py` exist, and
the renderer proof and this preregistration update landed. The earlier text is in git history.