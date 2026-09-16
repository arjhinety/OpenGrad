# 31 — Canonical-v3 sources and the `normalization-v3` pre-classifier artifact

**Status: BUILT, 2026-09-15; rebuilt 2026-09-16 under adapter version `2.1.0` (ToolACE repair, §5).** This resolves blocker **B-1** of
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
| Source-and-adapter manifest | `configs/releases/toolpolicy_canonical_v3_sources.yaml`, sha256 (LF) `cc40f64eaa1d6dbff618e64d2ad3f4b7fe46b053d3db31e67bcdf1d272ef0b6b` |
| Builder | `src/opengrad/data/normalization_v3.py` (`python -m opengrad.data.normalization_v3 --build` / `--verify`) |
| Artifact | `data/processed/normalization-v3/` (git-ignored data, like v1 and v2) |
| **Fingerprint** | **`56e8abf2f952907c0e936ac9397bde5b0a0c4a14eda3a3ccbbd47960bff2fda3`** (adapter version `2.1.0`; supersedes `2bd38492…`, built under `2.0.0` before the ToolACE repair) |
| Top manifest sha256 | `9ff1f9d658586c2818283b127fc46c92464cd186ec200626c23001049d74e88f` |
| Tracked anchors | `reports/normalization-v3/manifests/` (copies of all five manifests), `normalization-v3.determinism.json`, `normalization-v3.structural-audit.json`, `pdet-v1-representation-audit.json` |
| Tests | `tests/data/test_normalization_v3.py` |

## 1. Sources and adapters

The four sources of canonical-v2-final. Every Study 002 arm is Canonical-v2-derived
([04](04-ARM-MATRIX.md)), so canonical-v3 adds no source and drops none.

| Source | Upstream revision | Raw artifact sha256 | Adapter (key → function) | Schema translation |
|---|---|---|---|---|
| xLAM | `26d14ebf…97866` | `bec51a69…c527b` (60,000 rows) | `xlam` → `adapt_xlam` | no (xLAM's own `xlam_types` parser) |
| Glaive | `e7f4b645…23221ac` | `7e7e32f0…6b8` (112,960) | **`glaive_v2` → `adapt_glaive_v2`** | no (JSON Schema already) |
| ToolACE | `6bda777c…4f734` | `7a7a6a2c…ab15f` (11,300) | **`toolace_v2` → `adapt_toolace_v2`** | no (JSON Schema already) |
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
- `adapter_version` `2.1.0`, including inside the supervision block;
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
for the `2.1.0` rebuild.

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
| Records with a trajectory issue | 1,231 (`ARG_TYPE`) | 864 | 8,476 (8,472 `MISSING_TOOL_RESULT`) | 0 |

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
