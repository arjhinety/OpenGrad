# 32 — Classifier input contract `prose-decision-input-v1`

**Status: ADOPTED as the input contract for the future Study 002 prose decision classifier, 2026-09-15.**
It resolves blocker **B-2** of [30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md). It defines what the
classifier may read and which records it may receive. It contains **no classification logic**: no
classifier exists, no behaviour label is produced, and nothing here decides DIRECT, CLARIFY or UNSUPPORTED.

| | |
|---|---|
| Contract version | `prose-decision-input-v1` (`opengrad.data.versions.CLASSIFIER_INPUT_CONTRACT_VERSION`) |
| Code | `src/opengrad/data/classifier_input.py` |
| Tests | `tests/data/test_classifier_input.py` |
| Input representation | `normalization-v3` rows only ([31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md)) |

## 1. Scope

The contract covers one stage: the **pre-action decision in a tool-free, single-exchange record**. One user
turn, one assistant turn (a leading system message is allowed), no structured call and no tool result
anywhere. That is the unit of P-DET-v1 and of layer B of P-DET-COVERAGE-v1 (30 §4, §7.1).

Any other stage needs its own, separately preregistered contract. This one grants nothing outside its unit.

## 2. What the classifier may read

Exactly four fields, and nothing else (`FEATURE_FIELDS`):

| Field | Content |
|---|---|
| `user_message` | the content of the single user turn, byte for byte as in normalization-v3 |
| `assistant_response` | the content of the final (and only) assistant turn, byte for byte |
| `tools` | the tool definitions presented with the record, as normalized in normalization-v3 (`name`, `description`, `parameters`); empty when none are presented |
| `structured_call_present` | whether a structured tool call exists. Always `false` for an eligible record (§4); carried so the invariant is visible in the input itself |

**The system message is not a feature.** Two reasons:
- The tools it carries are already presented, normalized, in `tools`.
- The remaining system text is a per-source template (Glaive's "You are a helpful assistant with access
  to the following functions…", ToolACE's "You are an expert in composing functions…"). As a feature it
  would identify the source, which §3 forbids.

Its presence is recorded as a structural fact (`metadata.structure.leading_system_message`,
`metadata.system`). It is never shown to the classifier.

## 3. What the classifier may never read

- the source dataset name, or anything that encodes it (the id, `raw_record_hash`, the adapter name);
- the hidden sampling stratum of any validation population;
- P-DET labels (P-DET-v1 or P-DET-COVERAGE-v1), model reference labels, held-out labels;
- future evaluation outcomes.

**Identity is provenance.** `ClassifierProvenance` holds the record key (`<source>:<raw_record_hash>`), the
id, the source name and dataset id, both hashes, and the normalization, adapter and schema versions. It is
serialized separately (`provenance_json`) and never enters the feature serialization or its hash. A test
builds two records that differ only in source identity. Their feature hashes are equal and their provenance
differs (`test_source_identity_is_provenance_not_a_feature`).

**Known limit.** Tool-definition style can still correlate with source after normalization, for example
When2Call's translated `nullable` flags. That is presented content, and removing it would change what the
assistant was shown. It is recorded here, not engineered away.

## 4. Eligibility

`eligibility(record, heldout)` answers one question: *is this record structurally eligible to be
classified by the prose decision classifier?* It is deterministic and reads only structure:
- roles;
- whether structured calls and tool results are present;
- emptiness;
- trajectory validity;
- held-out membership.

It never reads what a response says.

Every applicable reason is returned, in this precedence order. The first is the primary reason.

| Reason | Applies when |
|---|---|
| `EVALUATION_ONLY_OR_HELDOUT` | `metadata.eligibility == evaluation_only`, or an evaluation split name, or the id is a held-out id, or a user turn equals a held-out prompt (normalized: case-folded, whitespace collapsed; the P-DET-v1 rule) |
| `MALFORMED_OR_UNRENDERABLE` | parse status not `VALID`; no messages; non-text user or system content; no user turn; final message not an assistant turn; final assistant turn with neither a call nor non-empty text; or any trajectory issue from `validate_training_trajectory`, the gate `Qwen35_2BRenderer.render_sft` applies before rendering |
| `STRUCTURAL_CALL` | any assistant turn carries structured `tool_calls`. Such records belong to **layer A**, structural CALL routing, and never to the prose classifier |
| `POST_TOOL_RESULT_RESPONSE` | any `tool` message is present. The response is downstream of a tool result |
| `MULTI_TURN_UNSUPPORTED` | more than one user turn, or a tool-free conversation that is not exactly one user turn followed by one assistant turn |

A record that is not a normalization-v3 row raises `ContractViolation` instead of returning a reason:
- no `normalization_version: normalization-v3`;
- an adapter version other than the authoritative `2.0.0`;
- or any `metadata.behavior`.

So a normalization-v1 row, or a row that has already been classified, can never be fed to the classifier
through this contract.

### Held-out inputs (pinned by SHA-256 in the code)

| Input | Contributes |
|---|---|
| `.cache/normalization/raw/when2call_test_mcq.parquet` | `question`: 3,652 rows, 2,295 distinct normalized prompts |
| `.cache/normalization/raw/when2call_test_llm_judge.parquet` | `question`: all 286 distinct prompts are among the MCQ prompts |
| `reports/evaluation/behavioral-heldout-v2-partition.json` | 3,650 ids (P-DEV 2,373, P-CONF 1,277) |
| both `behavioral-heldout-v2-quarantine*.json` files | 2 ids (the same two in each) |

The P-DET-COVERAGE-v1 draw-time exclusions of 30 §9 are population-construction rules. They are not
classifier eligibility, and they stay in 30:
- the QAD recovery set;
- P-DET-v1 overlap;
- sentinel prompts;
- later evaluation sets.

## 5. Post-tool responses

A response written after a tool result is **excluded**, with reason `POST_TOOL_RESULT_RESPONSE`. It is not
DIRECT, it is not forced into CALL, and the prose classifier never sees it. A future classifier for that
stage needs its own, separately preregistered contract and validation population.

## 6. Multi-turn records

Multi-turn records are **ineligible**, with reason `MULTI_TURN_UNSUPPORTED`. No frozen semantic contract
exists for them.

**No multi-turn conversation is reduced to its final message.** Earlier turns change the decision context:
a question answered two turns ago, a clarification already given. Dropping them would present the
classifier with a different decision than the one the assistant made. `build_classifier_input` raises
`IneligibleRecord` for such a record and never truncates (`test_multi_turn_is_excluded_and_never_flattened_to_its_final_message`).

A single user turn followed by a tool loop is excluded by its tool use and is not also counted as
multi-turn.

## 7. The input object and its serialization

`build_classifier_input(record, heldout) -> ClassifierInput`, for eligible records only.

- `ClassifierInput.features` (`ClassifierFeatures`) and `ClassifierInput.provenance`
  (`ClassifierProvenance`) are frozen dataclasses.
- `features_json()` is `stable_json({"contract_version": …, **features})`: keys sorted at every depth, no
  insignificant whitespace, UTF-8 text unescaped. It is independent of input key order
  (`test_serialization_is_deterministic_and_key_order_free`).
- `features_sha256()` is the SHA-256 of that JSON. A literal value is pinned in
  `test_features_hash_is_pinned`, so any change to field names, ordering or encoding breaks the test and
  has to come with a new contract version.

## 8. What changes require a new version

- adding, removing or renaming a feature;
- showing the system message;
- admitting multi-turn or post-tool records;
- changing an exclusion rule or the held-out inputs;
- changing the serialization.

Any of these is `prose-decision-input-v2`, recorded before any population is drawn against it.
