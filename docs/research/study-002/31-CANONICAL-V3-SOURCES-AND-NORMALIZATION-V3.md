# 31 — Canonical-v3 sources and the `normalization-v3` pre-classifier artifact

**Status: BUILT, 2026-09-15; rebuilt 2026-09-16 under adapter version `2.1.0` (ToolACE repair, §5), and
again the same day under `2.2.0` (ToolACE call-final records read as `CALL_PREDICTION`, §9).** This resolves blocker **B-1** of
[30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md). It fixes which sources canonical-v3 reads and how each is
adapted, and it builds the representation the future prose decision classifier will receive.

It is **not** the canonical-v3 training corpus:
- no decision classifier exists or ran;
- no behaviour label is present;
- nothing was balanced, mixed or sampled;
- the renderer was not touched;
- nothing was trained.

| | |
|---|---|
| Source-and-adapter manifest | `configs/releases/toolpolicy_canonical_v3_sources.yaml`, sha256 (LF) `5bb4961cd100b4d5975a9596bee345e5234ce527029a4d6183f0b9a662bf2f90` |
| Builder | `src/opengrad/data/normalization_v3.py` (`python -m opengrad.data.normalization_v3 --build` / `--verify`) |
| Artifact | `data/processed/normalization-v3/` (git-ignored data, like v1 and v2) |
| **Fingerprint** | **`60d3123e1c6cef67f904a81f29669dcff44ecc5aeb0e71178a3b5d4e94bfa75a`** (adapter version `2.2.0`; supersedes `56e8abf2…`, built under `2.1.0` before the §9 decision, which superseded `2bd38492…`, built under `2.0.0` before the ToolACE repair) |
| Top manifest sha256 | `88b22ccc8ead4d0ba95156ab4a6c6a4232011d2d0ce490916982aca37752ee9e` |
| Tracked anchors | `reports/normalization-v3/manifests/` (copies of all five manifests), `normalization-v3.determinism.json`, `normalization-v3.structural-audit.json`, `pdet-v1-representation-audit.json` |
| Tests | `tests/data/test_normalization_v3.py` |

## 1. Sources and adapters

The four sources of canonical-v2-final. Every Study 002 arm is Canonical-v2-derived
([04](04-ARM-MATRIX.md)), so canonical-v3 adds no source and drops none.

| Source | Upstream revision | Raw artifact sha256 | Adapter (key → function) | Schema translation |
|---|---|---|---|---|
| xLAM | `26d14ebf…97866` | `bec51a69…c527b` (60,000 rows) | `xlam` → `adapt_xlam` | no (xLAM's own `xlam_types` parser) |
| Glaive | `e7f4b645…23221ac` | `7e7e32f0…6b8` (112,960) | **`glaive_v2` → `adapt_glaive_v2`** | no (JSON Schema already) |
| ToolACE | `6bda777c…4f734` | `7a7a6a2c…ab15f` (11,300) | **`toolace_v3` → `adapt_toolace_v3`** (§5, §9) | no (JSON Schema already) |
| When2Call | `0582f774…ace53` | `f992975f…07ab3` (15,000) | `when2call` → `adapt_when2call` | **yes** (`translate_source_tools`, before the adapter) |

**Excluded:**
- **LoopTool.** It is not in canonical-v2-final, whose build could not locate its upstream.
- **BUTTON.** It is not in canonical-v2-final (gated upstream, HTTP 401), and every one of its records
  ends in prose after a tool result.

Per source, the manifest also records the remaining provenance fields:
- the raw file path;
- whether structured calls are produced;
- how textual call syntax is handled;
- whether tool results are retained;
- special-token cleaning;
- multi-turn handling;
- where the system message goes;
- eligibility for C1 training and for P-DET-COVERAGE-v1 sampling.

Every row carries its provenance:
- `adapter`, `adapter_key` and `adapter_function`;
- `adapter_version` `2.2.0`, including inside the supervision block;
- `schema_normalization_version`;
- `schema_translation`, with every change it made;
- `normalization_version`;
- upstream revision, raw artifact sha256, raw row index and `raw_record_hash`.

**The Glaive choice is made, not inherited.** `adapt_glaive` (v1) needs a `</functioncall>` terminator
that this revision never has. So it leaves every call as text and orphans its tool result. `adapt_glaive_v2`
parses the unterminated blocks. Nothing was inferred from normalization-v1.

The choice also reproduces canonical-v2-final exactly:

- **Glaive:** 98,339 records. Every rejection reason matches v2-final's release manifest count for count,
  including 866 `malformed … at offset 2`, a message only `extract_glaive_calls` produces.
- **ToolACE:** 11,051 records, every reason matching.
- **xLAM:** 57,342 records. v3 reads the 60,000 upstream rows, where v2-final re-normalized the
  59,370-record v1 derivative.

The measurement settles what v2-final's manifests could not: v2-final used `glaive_v2`, even though they
record `adapter_version: 1.0.2`.

## 2. How it is built

1. The raw file's sha256 must equal the manifest's.
2. The registered adapter runs unchanged on the untouched raw row. For When2Call, the raw `tools` are
   translated first.
3. `raw_record_hash` and the id are computed on the untouched raw row, with `_base`'s rule, so
   translation cannot change a record's identity. For untranslated rows the builder checks that its rule
   reproduces the adapter's own identity.
4. Metadata is rebuilt:
   - the `behavior` block is dropped. `_base` stamps `ANSWER` whenever no call is present, and that is
     behavioural inference;
   - every version comes from `versions.py`;
   - a `metadata.structure` block of structural facts is added: roles, exchange shape, turn and call
     counts, final-turn position relative to tool use, and trajectory issue codes.
5. Dedup is by `canonical_hash`, first occurrence wins, as in `materialize`. Every row that is not
   accepted is written, with its reason, to `dispositions.jsonl`.

**Determinism.** Two independent full builds, into different directories, gave the same fingerprint, and
**all 29 files are byte-identical** (`normalization-v3.determinism.json`). `--verify` re-hashes every shard
and row and re-checks every row's provenance. It passes on both builds. The check was repeated in full
for the `2.1.0` rebuild and again for the `2.2.0` rebuild.

## 3. Structural audit (per source)

From `reports/normalization-v3/normalization-v3.structural-audit.json`. It reads structure and syntax only.
No refusal, question or direct-answer cue ran. Counts are records unless stated otherwise.

| | xLAM | Glaive | ToolACE | When2Call |
|---|---:|---:|---:|---:|
| Raw rows | 60,000 | 112,960 | 11,300 | 15,000 |
| **Accepted** | **57,342** | **98,339** | **11,051** | **14,701** |
| Rejected | 2,308 | 1,269 | 246 | 175 |
| Duplicates | 350 | 13,352 | 3 | 124 |
| Single exchange / multi-turn | 57,342 / 0 | 14,303 / 84,036 | 10,263 / 788 | 14,701 / 0 |
| Final assistant turn has a structured call | 57,342 | 0 | 8,442 | 0 |
| Final prose after a tool result | 0 | 47,867 | 636 | 0 |
| Final prose, no prior tool use | 0 | 48,752 | 1,973 | 14,701 |
| Final turn is a user or tool message | 0 | 1,424 user, 295 tool | 0 | 0 |
| Tools available / none | 57,342 / 0 | 63,741 / 34,598 | 10,354 / 697 | 12,484 / 2,217 |
| Declared supervision | `CALL_PREDICTION` 57,342 (`upstream_declared`) | `COMPLETE_TRAJECTORY` 98,339 | `CALL_PREDICTION` 8,425, `COMPLETE_TRAJECTORY` 2,626 (both `source_adapter`, §9) | `COMPLETE_TRAJECTORY` 14,701 |
| Records with a trajectory issue | 1,231 (`ARG_TYPE`) | 864 | 52 (8,476 under `2.1.0`, §9) | 0 |

Glaive also has 1 record whose final assistant turn is empty.

**Rejections by reason:**
- **xLAM:**
  - `XLAM_TYPE_UNSUPPORTED_UNION` 1,225;
  - `…_CALLABLE` 538;
  - `…_SET` 286;
  - `ADAPTER_DUPLICATE_TOOL_NAME` 259.
- **Glaive:**
  - `ADAPTER_GLAIVE_MALFORMED_CALL` 1,094;
  - `SCH_UNSUPPORTED_KEYWORD` 62;
  - `SCH_ADDITIONAL_PROPERTIES_NOT_BOOLEAN` 47;
  - `ADAPTER_UNDECLARED_TOOL` 25;
  - `ADAPTER_GLAIVE_ARGUMENTS_NOT_OBJECT` 18;
  - `SCH_REQUIRED_NOT_STRING_LIST` 18;
  - `SCH_DUPLICATE_REQUIRED` 2;
  - `SCH_SCHEMA_NOT_OBJECT` 2;
  - `INVALID_ARGUMENTS_JSON` 1.
- **ToolACE:**
  - `SCH_UNSUPPORTED_KEYWORD` 112;
  - `INVALID_ARGUMENT_SYNTAX` 89;
  - `ADAPTER_UNDECLARED_TOOL` 18;
  - `SCH_PROPERTIES_NOT_OBJECT` 16;
  - `SCH_ADDITIONAL_PROPERTIES_NOT_BOOLEAN` 10;
  - `SCH_ENUM_INVALID` 1.
- **When2Call:**
  - `SRC_SCHEMA_MULTI_TYPE_UNION` 73;
  - `SRC_SCHEMA_UNREPRESENTABLE_TYPE` 55;
  - `ADAPTER_DUPLICATE_TOOL_NAME` 47.

**Residual call syntax in prose turns** (assistant turns without `tool_calls`; unit: turns). The special
tokens checked are end-of-text, `im_start`, `im_end`, `eot_id`, `</s>` and `[INST]`.

| Source | Residual markers | Special tokens or leading `:` |
|---|---|---|
| xLAM | none | none |
| Glaive | **0 `<functioncall>`**. 368 turns contain a JSON object with a `"name"` key (340 parse, 28 do not); 44 of them are final turns, all in records that are not eligible | none |
| ToolACE | 6 turns with a parseable JSON `"name"` object, all final | none |
| When2Call | **0 `<TOOLCALL>`** | none |

Malformed textual calls do not survive as text; they cause rejection:
- Glaive: 1,094 malformed calls, 18 argument payloads that are not objects, 1 unparseable argument JSON;
- ToolACE: 89 records with invalid call syntax.

**Call conservation, per accepted row.** Calls written in the source's own syntax in the raw row equal the
structured calls in the v3 row for every accepted row:

| Source | Rows | Unit compared |
|---|---|---|
| Glaive | 98,339 / 98,339 | `<functioncall>` markers vs `tool_calls` entries |
| xLAM | 57,342 / 57,342 | named `answers` entries vs `tool_calls` entries |
| ToolACE | 11,051 / 11,051 | call-bearing turns, since one bracket turn can hold several calls |
| When2Call | 14,701 / 14,701 | no raw calls exist |

**Invariants, every row:**
- no behaviour label;
- adapter version authoritative, in metadata and in the supervision block;
- adapter key as in the manifest;
- the stored `structure` block equals a recomputation from the messages.

All four hold with 0 violations.

## 4. Glaive: textual calls become structured

- **Every valid `<functioncall>` becomes a structured `tool_calls` entry.** Conservation holds on all
  98,339 accepted rows, 63,218 of the raw rows carry markers, and 0 markers remain in any assistant
  content.
- **No invalid call survives as text.** 1,242 rejected raw rows contain markers:
  - 1,113 because a call did not parse (1,094 malformed, 18 non-object arguments, 1 unparseable argument
    JSON);
  - 129 for schema or undeclared-tool reasons.

  The unit tests `test_glaive_functioncall_becomes_a_structured_call_and_leaves_no_text` and
  `test_glaive_malformed_call_is_rejected_not_kept_as_prose` pin one case of each kind.

**Consequence for P-DET-COVERAGE-v1:** Glaive supplies no textual-call (X) candidates, as 30 §5 predicted.

## 5. ToolACE: call text duplicated beside the structured call — repaired in `2.1.0`

**The defect (found in the `2.0.0` build, fingerprint `2bd38492…`).** `adapt_toolace` sets `tool_calls` from
bracket syntax, but removes the bracket text from `content` only when the text contains `[Function`. So
**9,785 of 9,786** ToolACE call turns held their call twice: once structured, and once as the original
text in `content`. The prose classifier was never affected, since no eligible record has a structured call
([32](32-CLASSIFIER-INPUT-CONTRACT.md) §4). Layer-A display and any training target rendered from those
turns were affected; canonical-v2-final used the same adapter.

**The repair.** A new adapter, `adapt_toolace_v2` (key `toolace_v2`), with `ADAPTER_VERSION` bumped to
`2.1.0` in `src/opengrad/data/versions.py`. `adapt_toolace` is kept unchanged so the v1 and v2 corpora stay
reproducible, following the `adapt_glaive_v2` precedent. The rule:
- remove the span that parsed into calls, and only that span (`TOOLACE_CALL_BLOCK`, shared with the parser
  so the two can never cover different text);
- keep any prose around it;
- give a turn that was nothing but the call `content: null`;
- a block that does not parse still raises, so the record is quarantined, never silently stripped.

**Measured on the rebuilt artifact** (`normalization-v3.structural-audit.json`):

| ToolACE | `2.0.0` (`2bd38492…`) | `2.1.0` (`56e8abf2…`) |
|---|---:|---:|
| Accepted / rejected / duplicates | 11,051 / 246 / 3 | 11,051 / 246 / 3 |
| Structured call turns | 9,786 | 9,786 |
| Call turns whose content still holds call text | 9,785 | **0** |
| Call turns with any non-empty content | 9,785 | 0 |
| Call conservation (call-bearing turns, raw vs structured) | 11,051 / 11,051 | 11,051 / 11,051 |

Every ToolACE call turn in this revision turned out to be the call alone, so no surrounding prose needed
keeping; the rule is tested on a synthetic turn that has some. The calls, tools, rejections and membership
of every source are unchanged. Every content hash still moves, because each row now records `2.1.0`. The
P-DET-v1 representation audit (§7) and the P-DET-COVERAGE supply (30 §7.3) are unchanged, including X at 6:
those six are JSON-shaped prose in tool-free ToolACE records, not bracket calls.

## 6. When2Call membership differs from canonical-v2-final

Schema translation raises When2Call from v2-final's 6,505 records to **14,701**. This is the recovery
measured in [21](21-C1-IMPLEMENTATION-STATUS.md), less 47 duplicate-tool-name rejections and 124
duplicates. canonical-v3 therefore does not have v2-final's When2Call membership.

Whether C1's training membership keeps v2-final's 6,505 or takes the 14,701 is a mixture-phase decision
([04](04-ARM-MATRIX.md) requires identical membership for R1 vs C0). It is not made here.

## 7. P-DET-v1 against normalization-v3 (audit, not migration)

`reports/normalization-v3/pdet-v1-representation-audit.json`. P-DET-v1 is unchanged. Each item is matched
by `raw_record_hash`.

- **575 of 581 items are structurally equivalent.** All of the following hold for them:
  - the record is present;
  - the user message and the response are byte-identical;
  - the upstream id is identical;
  - the tools are identical (101) or differ only by source-schema-normalization-v1 (474);
  - the record is eligible under the contract.
- **6 items are not in normalization-v3.** Their When2Call records are quarantined by schema translation:
  - `SRC_SCHEMA_MULTI_TYPE_UNION`: 5;
  - `SRC_SCHEMA_UNREPRESENTABLE_TYPE`: 1.

  They are tool items #13, #39, #52 and #349 (challenge) and #347 and #371 (prevalence), none of them an
  exposed worked example. The classifier can never receive these records.

**Limitation (recorded, P-DET-v1 unchanged).** 6 of 581 P-DET-v1 items (1.0%) fall outside the
classifier's input representation, so the classifier cannot be scored on them as it would be deployed.

- **Recommendation.** When classifier metrics are computed, report them on all items as 22 specifies, and
  also on the 575 representable items, listing the 6 as `NOT_IN_CLASSIFIER_REPRESENTATION`.
- **Who decides.** Adopting that is the study owner's decision, and it would be an amendment. It is not
  made here.
- **Size.** At 1.0% the limitation does not materially limit P-DET-v1's validity for the modes it measures.
  All 575 other items are byte-identical in what the classifier sees.

Two metadata differences are invisible to the classifier:
- the split is `train` in v3 and `train_sft` in v1;
- the behaviour and version fields differ.

## 8. What this does not do

- No decision classifier, no heuristic label, no behaviour-balanced mixture.
- No renderer change: `renderers.py` is byte-identical.
- No P-DET-COVERAGE-v1 draw.
- No training.
- No change to any v1/v2 artifact, to P-DET-v1 or to `reports/pdet/`. Checked by a before-and-after hash
  diff: only the new files and `versions.py` differ.

## 9. ToolACE call-final records: `CALL_PREDICTION` from adapter version `2.2.0` (decision, 2026-09-16)

### 9.1 The question, as it was recorded

The P-DET-COVERAGE-v1 review (30 §12 U-8) recorded this as an open item. Under `2.1.0`, 8,472 of the
11,051 accepted ToolACE records failed `MISSING_TOOL_RESULT`, and 8,442 of them ended on a structured call.
The trajectory gate runs before `render_sft`, so none of them could be rendered for training.

This was the documented status quo, not a new loss. ToolACE was `COMPLETE_TRAJECTORY` because its call-final
rows were "mixed, unresolved" (`reports/SUPERVISION_CONTRACT_REPORT.md` §3–§4), and canonical-v2-final
quarantines the same records (`configs/data/yield_expectations.yaml`, measured yield 0.204). The first
wording of this item said a canonical-v3 mixture "would silently lose" them; that overstated it. The
supervision report's §4 names the evidence that would resolve it: a documented format, a maintainer
statement, or a construction pattern.

### 9.2 Evidence (reproduced from the pinned raw rows)

**Upstream says nothing about how the rows were cut.** Checked at the pinned revision `6bda777c…`:
- the dataset repository holds only `.gitattributes`, `README.md` and `data.json`, with no loading script;
- the dataset card (sha256 `4e82e477…`) has no format statement;
- the ToolACE-8B model card (revision `e9ee4e01…`) shows a system prompt and a user message followed by a generated call, which
  is inference usage, not a description of the data;
- the paper (arXiv 2409.00920v2) describes a tool agent that simulates results during dialog generation
  (§2.2.1) and releases "a subset of the data" (abstract). Its few-shot experiment uses training samples as
  retrieved examples, and Figure 16 renders them as a user turn followed by the call. None of the three
  shown there is among the released rows (searched by user text: 0 of 3), so this supports the form, not
  these rows.

**The construction pattern** (`scripts/audit_toolace_call_final_shape.py` →
`reports/normalization-v3/toolace-call-final-shape.json`; counts only). It was computed twice, the second
time with independent code: pyarrow instead of the builder's reader, and the adapter-independent call
readers. Both agree on every count below. Of the 11,300 raw rows:
- every row ends on an assistant turn: 8,650 on a call and 2,650 on prose, 0 on a tool result or a user
  turn. If results had been lost, some rows would be expected to end elsewhere;
- no row is a proper prefix of another, so the rows are not overlapping cuts of one dialog;
- no call-final row has an earlier call turn without a result;
- 8,498 of the 8,650 call-final rows are a single user turn followed by the call;
- 647 prose-final rows contain calls with their results, so full trajectories are released as such.

**Subgroups that do not fit, found before any rule was written:**
- 9 call-final rows whose final call follows a tool result instead of a user turn: a mid-execution step,
  where a lost result is most plausible;
- 5 whose final turn is not only the call block;
- 10 earlier call turns with fewer tool-result turns than calls;
- 6 rows that exactly duplicate another row, which dedup handles.

None contradicts the reading, but the rule does not admit the first three.

### 9.3 Decision

**Study owner, 2026-09-16.** Call-final ToolACE records that satisfy the rule below are read as
`CALL_PREDICTION`. This applies from adapter version `2.2.0` on and is not retroactive.

Three claims have to be kept apart:
- **Supported:** the structure strongly supports reading these rows as call-prediction examples.
- **Implemented:** OpenGrad now classifies qualifying rows as `CALL_PREDICTION`.
- **Not claimed:** that ToolACE upstream defines them this way. No upstream source says so.

Each record carries this in its existing supervision block: `kind: CALL_PREDICTION`, `assignment:
source_adapter` (the supervision report's term for an adapter-decided kind, never `upstream_declared`), and
a `note` that states it is "OpenGrad corpus-level structural inference, not an upstream ToolACE
annotation". The note also names the evidence file. The source manifest states `upstream_declared: false`.

### 9.4 The rule

`adapters.toolace_call_prediction_shape`, applied by `adapt_toolace_v3` to the messages `adapt_toolace_v2`
produces. It is deterministic, and all three conditions must hold:
1. the final message is an assistant turn with structured calls and no other content;
2. the message before it is a user turn;
3. every earlier assistant call turn is followed by exactly as many tool results as it has calls.

Any other record keeps `COMPLETE_TRAJECTORY` and is quarantined exactly as under `2.1.0`. The trajectory
gate is not changed. Under `CALL_PREDICTION` it still rejects:
- an undeclared tool;
- invalid arguments;
- an orphan or out-of-order result;
- an empty assistant turn.

### 9.5 Counts

`scripts/report_toolace_call_prediction.py` → `reports/normalization-v3/toolace-call-prediction.json`.
"Before" is computed, not remembered: each accepted row is validated again under `COMPLETE_TRAJECTORY`,
which is what `adapt_toolace_v2` declared for every record. It equals the superseded `2.1.0` structural
audit.

| ToolACE | `2.1.0` (`56e8abf2…`) | `2.2.0` (`60d3123e…`) |
|---|---:|---:|
| Raw rows / rejected by the adapter / duplicates | 11,300 / 246 / 3 | 11,300 / 246 / 3 |
| Accepted | 11,051 | 11,051 |
| Declared `CALL_PREDICTION` / `COMPLETE_TRAJECTORY` | 0 / 11,051 | 8,425 / 2,626 |
| Pass the trajectory gate | 2,575 | **10,999** (8,424 + 2,575) |
| Quarantined | 8,476 (8,442 call-final) | **52** |
| Recovered as `CALL_PREDICTION` / newly quarantined | — | 8,424 / 0 |

The adapter's rejections (§3) and the 2,575 records that already passed are unchanged. So are every
record's messages, tools and identity: `test_v3_changes_only_the_declaration_never_the_messages_tools_or_identity`
checks this.

**What remains quarantined** (one category per record; codes are the trajectory gate's):

| Declared | Category | Records | Issue codes |
|---|---|---:|---|
| `COMPLETE_TRAJECTORY` | prose-final, a call turn has fewer results than calls | 30 | `MISSING_TOOL_RESULT` 30, `INVALID_MESSAGE_SEQUENCE` 30 |
| `COMPLETE_TRAJECTORY` | call-final, an earlier call turn has fewer results than calls (rule 3) | 9 | `MISSING_TOOL_RESULT` 9, `INVALID_MESSAGE_SEQUENCE` 9 |
| `COMPLETE_TRAJECTORY` | call-final, the final call follows a tool result (rule 2) | 8 | `MISSING_TOOL_RESULT` 8, `INVALID_MESSAGE_SEQUENCE` 1 |
| `COMPLETE_TRAJECTORY` | an assistant turn with neither content nor calls | 4 | `EMPTY_ASSISTANT_OUTPUT` 4 |
| `CALL_PREDICTION` | an assistant turn with neither content nor calls | 1 | `EMPTY_ASSISTANT_OUTPUT` 1 |

The raw subgroups of §9.2 are larger than these categories for two reasons, measured on the rows:
- of the 9 raw rows whose final call follows a tool result, 1 is not accepted by the adapter;
- the 10 short call turns sit in 10 accepted call-final rows, but each quarantined record is counted once,
  in the first category that applies. The order is: empty turn, final call after a tool result, then too few
  results.

### 9.6 What did not change

- **Canonical-v2.** It is not rebuilt, and its manifests, hashes, reports and quarantine record are untouched.
  - `adapt_toolace` and `adapt_toolace_v2` are byte-for-byte unchanged, and both still declare
    `COMPLETE_TRAJECTORY`.
  - `toolpolicy_canonical_v2.yaml` and `toolpolicy_canonical_v2_final.yaml` are unchanged.
  - Tests pin all of this (`tests/data/test_toolace_call_prediction.py`).
  - The earlier quarantine was the right reading of what was known then. No erratum is filed: no published
    claim became false.
- **Study 001 and P-DET-v1.** Their artifacts, annotation state and hashes are unchanged.
  - The P-DET-v1 representation audit (§7) is identical apart from the fingerprint.
  - A before-and-after hash snapshot covered every git-ignored data, cache, release, annotation and results
    file.
- **Study 002's arms.** This is case A: Study 002 is preregistered (`study_002_prereg_v1`–`v3`), and every
  arm in [04](04-ARM-MATRIX.md) is Canonical-v2-derived, with `C0` "exactly as published".
  - No arm is moved to canonical-v3, and no arm's corpus changes.
  - canonical-v3 is not an operative corpus of any arm. 21 phase 8 (2026-09-20) registered it in
    [03](03-PREREGISTRATION.md) as built but not an arm; entering a study remains a separate owner decision.
- **P-DET-COVERAGE-v1 (a draft).** Its pins and counts-only dry run were re-recorded on the new artifact
  (30 §13).
  - Layer B is identical in every count.
  - Only layer A's ToolACE split across gate status moved: 41 gate-rejected and 8,885 valid, against 8,374
    and 552 before.
- **Nothing was trained.** No classifier, mixture, sampling weight or behaviour label was introduced.

**Proposed, not added.** If canonical-v3 is later preregistered for Study 002 or 003, the recovered records
should enter as their own measurable factor, never silently inside another arm. That means an arm or
ablation that differs from its reference only by including the 8,424 recovered `CALL_PREDICTION` records.
Adding it requires a numbered amendment decided by the study owner. **Deferred by the study owner
(2026-09-16)** until canonical-v3 enters a study, because the classifier and the mixture it would depend on do
not exist yet.

### 9.7 Uncertainty

- The reading is inferred, not upstream-stated. A maintainer statement that these rows are truncated
  trajectories would reverse it, and the reversal would be a new adapter version.
- The rule admits records whose shape matched the evidence. It does not prove every admitted row was
  generated as a next-call sample, and a truncated trajectory that happens to end right after a user turn
  would look identical.
- The prefix and ending checks are corpus-level: they speak for the release as a whole, not for any one
  row.

### 9.8 Files responsible

- `src/opengrad/data/adapters.py`: `toolace_call_prediction_shape`, `adapt_toolace_v3`,
  `TOOLACE_CALL_PREDICTION_NOTE`;
- `src/opengrad/data/versions.py`: `ADAPTER_VERSION` `2.2.0`, with the reason;
- `configs/releases/toolpolicy_canonical_v3_sources.yaml`: the ToolACE adapter, its `supervision` block, and
  `adapt_toolace_v2` under `rejected_alternative`;
- the evidence and counts: `scripts/audit_toolace_call_final_shape.py`,
  `scripts/report_toolace_call_prediction.py`, and `reports/normalization-v3/toolace-call-final-shape.json`,
  `toolace-call-prediction.json`, `normalization-v3.structural-audit.json` and
  `normalization-v3.determinism.json`;
- `tests/data/test_toolace_call_prediction.py`.
