# 30 — P-DET-COVERAGE-v1: preregistration draft

**Status: DRAFT, 2026-09-15. Not adopted. No example has been drawn.** Both sampling blockers of §12 are
resolved:
- **B-1**, the canonical-v3 representation: [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md);
- **B-2**, the classifier input contract: [32](32-CLASSIFIER-INPUT-CONTRACT.md).

Sampling is therefore no longer blocked on a missing artifact. It still waits for this draft's adoption,
for the recorded hashes (§12), and for the builder, which is not yet written.

This document specifies a second classifier-validation population. It **complements** frozen P-DET-v1 and
does not replace it.

It modifies nothing:
- not P-DET-v1, `src/opengrad/verification/pdet.py`, `reports/pdet/`, 22 or 23;
- not canonical-v2 or any other frozen artifact.

No classifier is implemented, no canonical-v3 artifact is built, and nothing is trained. When the study owner
adopts it, it is recorded as amendment `study_002_prereg_v4` in [03](03-PREREGISTRATION.md) and in
[`reports/ERRATA.md`](../../../reports/ERRATA.md). Until then it binds nothing.

| | |
|---|---|
| Population | `P-DET-COVERAGE-v1` |
| Protocol id | `pdet-coverage-002-v1` |
| Seed | `opengrad-pdet-coverage-002-v1` |
| Output (at draw time) | `reports/pdet-coverage/` (never `reports/pdet/`) |
| Feasibility evidence | `scripts/audit_pdet_coverage_supply.py` → `reports/pdet-coverage/pdet-coverage-v1.supply-proxy.json` (counts only, over a **proxy**; §5) |

## 1. Purpose

P-DET-v1 is 581 naturally sampled When2Call prose responses. Its human labels are 298 UNSUPPORTED,
281 CLARIFY, 2 UNKNOWN, **0 DIRECT and 0 CALL**. That is by construction: the source split contains no tool
call and no direct answer ([README](README.md), [`reports/ERRATA.md`](../../../reports/ERRATA.md) §14). Under
22 §5–§6 it therefore cannot validate a classifier on DIRECT or CALL, and so it cannot authorise C1.

P-DET-COVERAGE-v1 exists to **cover the missing classifier boundaries**:
- DIRECT;
- DIRECT against UNSUPPORTED;
- DIRECT against CLARIFY;
- prose that mentions a tool without invoking it;
- textual tool calls, where they survive into the classifier's input.

It is a constructed coverage set. **It estimates no natural prevalence**, and its class mix is not a
property of any corpus. The two populations are reported separately, always:

| Population | What it is | Main use |
|---|---|---|
| `P-DET-v1` | naturally sampled When2Call prose behaviour | CLARIFY, UNSUPPORTED, false DIRECT/CALL on natural data |
| `P-DET-COVERAGE-v1` | deliberately constructed boundary coverage | DIRECT, the DIRECT boundaries, tool-related prose |

## 2. DIRECT, for this study

> **`DIRECT` means the assistant answers the user's request normally, without requiring or using a tool
> for that response.**

The classifier's **decision point** is the moment *before any tool has been used*. There, the assistant
either:
- calls a tool (**CALL**);
- answers (**DIRECT**);
- asks for something it needs (**CLARIFY**);
- declines or explains that it cannot act (**UNSUPPORTED**).

**UNKNOWN / AMBIGUOUS** stays an annotation status, not a fifth mode. Every other part of the rubric is
22 §1–§3 unchanged: positive criteria, exclusions, the eight boundary discriminations and the decision tree.
This definition narrows 22 §1.2 in one respect only, stated in §3. 22 is not edited.

## 3. Post-tool responses are excluded

A prose response written **after a tool result** (user → tool call → tool result → assistant prose) is a
*downstream tool-result response*. It is outside the four-way pre-action decision. It is **not DIRECT**,
is not forced into CALL, and is not annotated in this population.

**Rule.** A record in which any tool has been used is out of layer B (§4). A tool has been used when any
of these appears: an assistant turn with structured `tool_calls`, a `tool`-role message, or (proxy only)
a Glaive `<functioncall>` marker. The builder records every such record under reason
`DOWNSTREAM_TOOL_RESULT_RESPONSE` or `STRUCTURAL_CALL`, with counts per source in the manifest. Records
whose first tool use is a structured call are eligible only for layer A.

Proxy scale, from `pdet-coverage-v1.supply-proxy.json`. The final turn is prose after tool use in:

| Source | Records |
|---|---:|
| Glaive | 49,287 |
| BUTTON | 7,941 (all of them) |
| ToolACE | 644 |

**Multi-turn tool-free records** are also out of the unit, under reason `MULTI_TURN_UNIT_UNDEFINED`. How
the classifier will label a conversation with several assistant turns is not specified (§12, B-2), so no
population can yet be built to match it. The proxy has 34,450 such Glaive records and 311 LoopTool
records. **No classifier permission on multi-turn records follows from P-DET evidence.** Covering them
takes a separately preregistered component, added as an amendment before any draw.

## 4. Two layers: structural CALL routing and prose decision classification

**Layer A — structural CALL routing.** A record with an authoritative structured call is CALL **by its
structure**. No prose heuristic guesses it. This is today's rule in `_base`
(`src/opengrad/data/adapters.py`): any structured `tool_calls` or `tool` message gives `decision: CALL`,
`confidence: derived`. Layer A is validated two ways:

1. **Exhaustively and deterministically**, over the whole canonical-v3 pre-classifier artifact (§5):
   - every record with a structured call is routed CALL;
   - no record without one is routed CALL by structure;
   - the canonical validator's trajectory rejections (for example `SEM_ORPHAN_RESULT`) are counted per
     source.

   This is a code check, not a sample.
2. **A 30-item human spot-check** in a separate annotation task (`pdet-coverage-v1-routing`). It confirms
   that each structured payload is a real invocation of an offered tool, consistent with the request. It
   guards against adapter artefacts, such as a wrong extraction in `adapt_glaive_v2`. Allocation:
   - Glaive 15 (reconstructed from malformed upstream text: the highest risk);
   - ToolACE 10 (parsed from bracket syntax);
   - xLAM 5 (natively structured);
   - LoopTool +5, only if canonical-v3 includes it.

   With 0 errors in 30, the 95% upper bound on the error rate is 10% by the rule of three (Wilson 11.4%).

**Layer B — prose decision classification.** The semantic classifier is tested on tool-free,
single-exchange records (§7). There it must decide DIRECT, CLARIFY or UNSUPPORTED. It may output CALL only
for a textual call that survives into its input (stratum X).

> **Success on structured CALL records is never reported as evidence that the prose classifier recognises
> CALL.** Layer A results are reported only in layer A.

## 5. The future classifier's input representation

**Rule (preregistered):** *sample from the post-adapter representation used by the future canonical-v3
pipeline.* No other artifact may stand in for it. In particular, `normalization-v1` is **not** a proxy
for canonical-v3. Study 001's `#80` was exactly that substitution: an audit read normalization-v1 rather
than the trained corpus ([01](01-LESSONS-FROM-STUDY-001.md) L4).

What the repository establishes:

| Representation | Glaive adapter | Tool calls | Evidence |
|---|---|---|---|
| `normalization-v1` (the local artifact) | `adapt_glaive` (v1) | as **text**: `: <functioncall> {…} <|endoftext|>` in assistant content, with `tool_calls: []`. **0 of 99,794** records carry a structured call. | measured; [`reports/data-normalization-v1.md`](../../../reports/data-normalization-v1.md), Glaive caveat |
| canonical-v2 and canonical-v2-final (the corpus M0 trained on) | `adapt_glaive_v2` (`glaive_v2`) | **structured**; markers, the leading `:` and end tokens stripped | `configs/releases/toolpolicy_canonical_v2.yaml` header: on identical input, the v1 adapter gave 0 records with tool calls and 50,900 `SEM_ORPHAN_RESULT` rejections, the v2 adapter 48,726 records with tool calls. [`M0_SFT_EXECUTION_REPORT.md`](../../../reports/M0_SFT_EXECUTION_REPORT.md) §11.1: 66,467 of 67,481 call turns read, 1,014 malformed ones refused. |
| canonical-v3 (`normalization-v3`, [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md)) | **`adapt_glaive_v2`** (`glaive_v2`), chosen and recorded in `configs/releases/toolpolicy_canonical_v3_sources.yaml` | **structured**. On all 98,339 accepted records, raw `<functioncall>` markers equal the structured calls; 0 markers remain as text; unparseable calls are rejected | built and measured (fingerprint `56e8abf2…`, adapter version `2.1.0`); reproduces v2-final's Glaive counts reason for reason |

Two consequences:
- **The manifests can't settle it.** Every normalization and release manifest records `adapter_version:
  1.0.2`, whichever Glaive adapter ran (the phase-1 provenance defect in 21). The adapter can't be read
  from a v2 manifest, and the v3 manifest must record the adapter itself.
- **Glaive textual calls do not survive `glaive_v2`.** A malformed `<functioncall>` raises, and the record
  is rejected rather than kept as text (`extract_glaive_calls`). So if canonical-v3 uses `glaive_v2`,
  Glaive supplies no textual-call candidates at all.

**Therefore sampling was `BLOCKED`** until both of these existed. Both now do ([31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md)):
- a canonical-v3 source and adapter manifest. Per source it names the artifact, the upstream revision, the
  registered adapter and its version, and the schema-normalization version (`versions.py`);
- a pre-classifier `normalization-v3` artifact, materialized by repository code from that manifest, with
  its manifest hashes recorded.

That is not circular. The classifier writes only `metadata.behavior`, and the message representation is
fixed by the adapters and the schema translation. So the artifact P-DET-COVERAGE-v1 needs can be built
before the classifier exists. The builder will refuse any input path that is not the recorded canonical-v3
pre-classifier artifact, and will refuse every `normalization-v1` path by name.

The same fidelity question applies to P-DET-v1 and is recorded here without changing it. `pdet.py` drew
from `normalization-v1/when2call-sft` (14,829 records). canonical-v2-final kept 6,505 When2Call records;
the schema translation in 21 recovers 14,872. The When2Call adapter did not change. Once canonical-v3
exists, check that every P-DET-v1 item is present in it with the same text, and report any that is not.

**Checked** (31 §7, `reports/normalization-v3/pdet-v1-representation-audit.json`):
- 575 of 581 items are structurally equivalent: the same user text and response, byte for byte, and
  tools identical or differing only by the schema translation;
- 6 items (tool items #13, #39, #52, #347, #349, #371) are quarantined in v3 by schema translation, so
  the classifier can never receive them. This is recorded as a limitation. P-DET-v1 is unchanged.

## 6. Candidate sources

The proxy counts come from `pdet-coverage-v1.supply-proxy.json`, over normalization-v1. The v2-final
figures are from `.release/hf/toolpolicy-canonical-v2-final/release-manifest.json`.

| | Glaive | ToolACE | xLAM | LoopTool |
|---|---|---|---|---|
| **Upstream revision** | `7e7e32f0…` (source sha256) | `7a7a6a2c3b1003c7…` | v2-final `26d14ebf…` (rebuilt from the v1 derivative); proxy `bec51a69…` | `189328fc…` |
| **Artifact / adapter** | proxy: `adapt_glaive` (v1); v2-final: `adapt_glaive_v2`; v3: **`adapt_glaive_v2`** (31) | `adapt_toolace` (one version); v3: **`adapt_toolace`** (31) | `adapt_xlam` + `xlam_types.py`; v3: **`adapt_xlam`** (31) | `adapt_looptool`; not in canonical-v2-final ("upstream source was not located"); **not in v3** (31) |
| **What the classifier would see** (if v3 keeps the v2-final adapters) | tool-free prose, prefix and end tokens stripped; calls structured, no textual calls | tool-free prose; calls structured, with 6 textual-call-shaped responses in the proxy | structured calls only | structured calls; a few tool-free single exchanges |
| **Records** | raw 112,960 · v2-final 98,339 · proxy 99,794 | raw 11,300 · v2-final 11,051 · proxy 11,190 | raw 60,000 · v2-final 57,342 · proxy 59,370 | raw 23,040 · proxy 20,827 |
| **Tool-free single exchanges (proxy)** | 14,329 (14,176 with tools offered) | 2,002 (1,305 with tools offered) | 0 | 324 (242 with tools offered) |
| **Structured-call records (proxy)** | 0 (text only in v1); v2 build: part of the 48,726 | 9,187 | 59,370 | 20,192 |
| **Proposed role** | layer B: R, Q, P1, P2, M; layer A: 15 | layer B: all strata including X; layer A: 10 | layer A only: 5 | layer B and layer A only if canonical-v3 includes it |
| **Label authority** | layer B: human; layer A: structural | same | structural | same as Glaive |
| **Overlap constraints** | §9 for all sources. Glaive also feeds the QAD recovery set (`m1_v2_qad_recovery_v1.jsonl`: 80 Glaive `ANSWER` and 13 Glaive `CALL` items), and those items are excluded. | §9; QAD set: 54 ToolACE CALL items, excluded | §9; QAD set: 74 xLAM CALL items, excluded | §9 |

- **BUTTON is not a candidate.** All 7,941 of its records end in prose after tool use (§3), and it is
  access-gated upstream.
- **When2Call is not a candidate either.** P-DET-v1 covers it, and keeping it out keeps the two
  populations separate.

## 7. Strata and quotas

### 7.1 Unit

**Layer B unit:** one tool-free, single-exchange record. That means exactly one user turn and one
assistant turn (a system message is allowed), and no tool use anywhere. This is the same unit as P-DET-v1
and as the refusal-supervision audit ([`ROADMAP.md`](../../../ROADMAP.md)). The classifier input is that
user message, the offered tools and that assistant response, subject to §12 B-2.

### 7.2 Stratum predicates (frozen; first match in this order; sampling strata, never labels)

| Stratum | Observable predicate on the assistant response | Boundary it covers |
|---|---|---|
| **X** textual or near-call form | contains `{"name":`, `"arguments":`, `<functioncall>`, `<tool_call>` or `<TOOLCALL>` | a real textual call vs tool-related prose (22 §1.1 counts a serialized call as CALL) |
| **M** tool mentioned, not invoked | names an offered tool (case-insensitive, name length ≥ 4), or matches the invocation-talk pattern in the audit script | prose that mentions or proposes a tool: CALL vs AMBIGUOUS, DIRECT, CLARIFY (22 §1.1 exclusions) |
| **R** refusal or hedge | `detect_refusal` (`opengrad.evaluation.capability`, as in P-DET-v1) or a P-DET-v1 hedge cue | DIRECT vs UNSUPPORTED |
| **Q** question | contains `?` or a P-DET-v1 question cue | DIRECT vs CLARIFY |
| **P1** plain prose, tools offered | none of the above, tools offered | DIRECT when a tool is available but not needed (the DIRECT/CALL decision) |
| **P2** plain prose, no tools | none of the above, no tools offered | DIRECT, baseline |

The cue lists and patterns are the ones in `scripts/audit_pdet_coverage_supply.py`, and the builder must
use them byte for byte.

Generic words such as "function", "tool" or "API" are **not** an M cue, and that is measured. In the
proxy, 11,002 of 14,329 Glaive tool-free single-exchange responses contain one, mostly inside the
refusal template "my current function allows me to …". Those words would put Glaive's refusals into M.

### 7.3 Quotas and how they follow from the strata

The requirement is **at least 50 usable gold examples per required mode or boundary** (22 §5), where
usable means gold that is not UNKNOWN and not excluded.

- **Boundary strata (M, R, Q): 60 each.** That allows up to 1 in 6 items to end up unusable
  (60 × 5/6 = 50). The allowance is conservative: P-DET-v1 had 2 UNKNOWN in 581, but these strata sit on
  the ambiguous boundaries on purpose.
- **DIRECT: P1 60 and P2 30 (90 candidates).** Together they give 50 gold DIRECT if at least 56% of plain
  responses are gold DIRECT. Any DIRECT gold in R, Q and M adds to that. P1 is weighted over P2 because
  the DIRECT/CALL decision exists only when a tool is offered.
- **X: every eligible item, up to 60.** This stratum can't be manufactured.
- **Layer A: 30 (35 with LoopTool).** This is an audit, not a classifier boundary; see §4.

| Stratum | Quota | Proxy supply (distinct skeletons), Glaive / ToolACE / LoopTool |
|---|---:|---|
| X | ≤ 60 | 0 / 6 / 0 → **shortage 54 expected** |
| M | 60 | 19 / 722 / 15 |
| R | 60 | 10,694 / 228 / 145 |
| Q | 60 | 102 / 169 / 152 |
| P1 | 60 | 58 / 375 / 3 |
| P2 | 30 | 65 / 491 / 0 |

**Source allocation inside a stratum:**
1. Split the quota equally across the included sources that have supply.
2. When a source runs short, give its remainder to the others, largest remaining supply first.
3. Break ties by source name.

This is allocation, not backfill: it never moves quota *between* strata.

On proxy supply this gives:

| Stratum | With LoopTool (Glaive / ToolACE / LoopTool) | Without LoopTool (Glaive / ToolACE) |
|---|---|---|
| X | 0 / 6 / 0 | 0 / 6 |
| M | 19 / 26 / 15 | 19 / 41 |
| R | 20 / 20 / 20 | 30 / 30 |
| Q | 20 / 20 / 20 | 30 / 30 |
| P1 | 28 / 29 / 3 | 30 / 30 |
| P2 | 15 / 15 / 0 | 15 / 15 |

Either way layer B is **276** (at most 330 if X is fully supplied), and the population is 306–311 with
layer A. The real numbers are computed at draw time from the canonical-v3 artifact and printed in the
manifest. These proxy figures only show that the design can be supplied.

**Measured on the real input** (`reports/pdet-coverage/pdet-coverage-v1.supply-v3.json`). This is supply
analysis, not gold and not a draw. The pool is every normalization-v3 record of Glaive or ToolACE that is
eligible under `prose-decision-input-v1`: Glaive 14,303 and ToolACE 1,972. The stratum predicates are the
ones above, byte for byte. §9's draw-time exclusions are not applied.

| Stratum | Glaive: records / distinct prompts / skeletons | ToolACE: records / distinct prompts / skeletons | Allocation implied (Glaive / ToolACE) |
|---|---|---|---|
| X | 0 / 0 / 0 | 6 / 6 / 6 | 0 / 6, **shortage 54** |
| M | 19 / 14 / 19 | 706 / 706 / 706 | 14 / 46 |
| R | 14,050 / 210 / 10,675 | 228 / 227 / 227 | 30 / 30 |
| Q | 105 / 93 / 102 | 168 / 166 / 166 | 30 / 30 |
| P1 | 64 / 14 / 58 | 371 / 371 / 365 | 14 / 46 |
| P2 | 65 / 65 / 65 | 493 / 491 / 491 | 15 / 15 |

- **How the allocation is computed.** It takes each source's supply as the smaller of distinct prompts and
  distinct skeletons. That is an upper bound on what §9's dedup leaves; the exact figure is computed at
  draw time.
- **The binding constraint is prompts, not responses.** Glaive's 14,050 R records come from only 210
  distinct user prompts, and its P1 records from 14.
- **Layer B stays at 276**, with X the only shortage, as predicted.

**No backfill.** An undersupplied stratum takes everything it has. Its shortage is reported *before*
annotation, and no other stratum grows to make up for it. If fewer than 50 usable gold items result, that
is a coverage finding under 22 §5, and the matching mode or boundary gets no balancing permission. The
proxy already predicts that X falls short (6 of 60).

## 8. Deterministic sampling plan

1. **Input.** Only the recorded canonical-v3 pre-classifier artifact (§5), for the sources its manifest
   includes out of Glaive, ToolACE, xLAM and LoopTool. The builder checks the artifact's manifest hashes
   against the values recorded in this document at adoption.
2. **Validity.** Keep a record only if its parse status is `VALID` and it passes the canonical validator.
3. **Exclusions** as in §9, each counted by reason code.
4. **Unit and layer.** Layer B: tool-free single exchanges (§7.1). Layer A: records whose first assistant
   turn carries a structured call. Everything else is counted under its §3 reason code.
5. **Neutral ids.** Each record gets the id `pdetcov:` + `sha256(source_dataset + ":" + raw_record_hash)`.
   The id reveals no source; the source is kept in the record, blinded (§10).
6. **Stratum**, by first match (§7.2).
7. **Dedup** as in §9. In rank order, the highest-ranked member survives.
8. **Ranking.** Within each stratum and source, sort by `sha256(seed + "|" + stratum + "|" + id)`,
   descending. This is the P-DET-v1 idiom (`sha256(seed ‖ pdet_id)` descending), with the stratum added so
   strata are independent. There is no RNG state.
9. **Take** the quota per stratum and source (§7.3), with the skeleton cap.
10. **Write** `reports/pdet-coverage/pdet-coverage-v1.population.jsonl`,
    `pdet-coverage-v1.manifest.json` and a `.sha256` sidecar. The manifest holds:
    - the input manifest hashes;
    - every exclusion and dedup count;
    - supply, quota, realized count and shortage for each stratum and source;
    - the code version.

    The draw is byte-reproducible.
11. **Presentation order** for annotation: `sha256(seed + ":order:" + id)`, so strata, layers and sources
    are interleaved.

New, versioned code: `src/opengrad/verification/pdet_coverage.py` with `--build` and `--verify`. It reads
constants from its own module and imports nothing from `pdet.py`. `pdet.py` and `reports/pdet/` are not
touched.

## 9. Contamination and dedup

**Excluded.** Each exclusion is counted by reason:
- **held-out ids:** every id in `reports/evaluation/behavioral-heldout-v2-partition.json` (P-DEV 2,373,
  P-CONF 1,277), in `behavioral-heldout-v2-quarantine.json` and in
  `behavioral-heldout-v2-quarantine--toolpolicy-canonical-v2-final.json`;
- **held-out prompts:** any record whose normalized user prompt equals a question in:
  - the When2Call MCQ set (the P-DET-v1 rule);
  - the When2Call LLM-judge set (whose rows duplicate MCQ rows; added as a guard);
  - the frozen sentinel files: S-ANS-0 and S-ANS-8 GSM8K, S-IF IFEval, S-MMLU MMLU-Pro and S-OW7
    ([08](08-SENTINEL-SPEC.md));
- **the QAD recovery set:** any record in `manifests/quantization/m1_v2_qad_recovery_v1.jsonl`, matched by
  canonical id and normalized prompt. 06 names it as a candidate source for P-CONF's `ANSWER` set, and
  excluding it keeps that option clean;
- **P-DET-v1:** any record whose id or `raw_record_hash` is one of P-DET-v1's items, or whose normalized
  prompt or normalized response equals one of theirs;
- **later evaluation sets:** P-UNANS, P-SEALED and P-CONF's `ANSWER` set, once each is frozen. Between
  this population and P-CONF's `ANSWER` set, **whichever freezes second excludes the other**;
- **quarantined records:** anything the canonical-v3 pipeline quarantines or rejects.

**Classifier development** must exclude every P-DET-COVERAGE-v1 id, as it must exclude P-DET-v1. The 22 §6
prohibition applies unchanged: no iterating the classifier against either population. The stratum
predicates are published here *before* the classifier exists, and results are reported per stratum, so a
classifier that leans on the same cues shows up.

**Dedup, in order:**
1. by `raw_record_hash`;
2. by normalized response text (case-folded, whitespace collapsed);
3. by normalized user prompt, one per prompt;
4. at most one item per response **skeleton** per stratum. The skeleton is the case-folded response with
   offered tool names, quoted spans and numbers masked, as in the audit script. Glaive's templated
   refusals are the reason: the proxy has 14,076 R-stratum records but 10,694 distinct skeletons.

**Exposed items.** Neither this document nor any annotator-facing text contains text from the eligible
pools. If exposure is found after the freeze, the item stays in the population and gets the
`EXPOSED_WORKED_EXAMPLE` metric exclusion, as in [27](27-PDET-EXPOSED-WORKED-EXAMPLES.md).

## 10. Annotation plan

- **Tasks.** In `opengrad-annotate`, two new tasks with their own state files: `pdet-coverage-v1`
  (layer B) and `pdet-coverage-v1-routing` (layer A). Neither touches the `pdet-v1` task or its store.
- **Labels.** CALL, DIRECT, CLARIFY, UNSUPPORTED and UNKNOWN, with 22 §1–§3 semantics plus §2–§3 of this
  document.
  - UNKNOWN needs an ambiguity status.
  - The rationale is optional from the start, as under [29](29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md); a
    new task needs no amendment for that.
- **On screen:**
  - the user message;
  - the offered tools;
  - the assistant response being annotated;
  - for layer A, its structured call payload;
  - no system message: the classifier input contract does not include it (§12 B-2, [32](32-CLASSIFIER-INPUT-CONTRACT.md) §2).
- **Blind to:**
  - the source dataset (hidden, because it would cue the label: xLAM always calls);
  - the stratum, the layer and the cue flags;
  - the adapter;
  - the expected behaviour;
  - any classifier prediction or model suggestion;
  - any P-DET-v1 label.

  Ids are neutral (§8), and the order interleaves everything (§8.11).
- **Rubric panel.** The same excerpts as P-DET-v1 plus §2–§3 of this document. Never §6–§9, since the
  strata and cues are sampling cues. Their names and cue strings go into `instruction_forbidden_terms`.
- **Minimal UI change, before annotation.** A readable render of the annotated assistant turn's structured
  tool calls (name and arguments), shown as part of the response being annotated and hidden when there is
  no call. Nothing else changes. The config already allows a `json` display field, so this is a small
  addition.
- **Annotators.**
  - The study owner does a single pass, then the 22 §4 single-annotator re-read (flagged, UNKNOWN and
    boundary-cited items) before any freeze.
  - With a second independent annotator it becomes two passes plus adjudication.
  - **Gold is human only.** No model-generated labels and no composite; amendment 28 does not extend to
    this population.

## 11. Planned metrics

Everything is reported **per population, per class and per stratum**, with the 95% Wilson interval and the
resolvable margin ([06](06-SPLIT-SPEC.md) C2).

Worst-case half-widths:

| n | Half-width |
|---:|---:|
| 30 | 16.8pp |
| 50 | 13.4pp |
| 60 | 12.3pp |
| 90 | 10.1pp |

For example, 40 of 50 is 0.80, with interval [0.670, 0.888].

**Layer B (prose classifier):**
- **DIRECT:** recall and precision on P-DET-COVERAGE-v1. The false-DIRECT count and rate on P-DET-v1,
  where gold DIRECT is 0.
- **UNSUPPORTED recall and CLARIFY F1:** P-DET-v1 is primary (natural data); strata R and Q are
  secondary.
- **Boundary accuracy per stratum:**
  - R: DIRECT vs UNSUPPORTED;
  - Q: DIRECT vs CLARIFY;
  - M: tool mention;
  - P1: DIRECT with a tool offered.
- **Prose-layer false CALL:** the count and rate of CALL predictions on items whose gold is not CALL, in
  both populations. This is the operational form of the 22 §6 CALL-precision risk ("a false CALL injects a
  wrong tool-call target").
- **Textual-CALL recall and precision:** only if X gives at least 50 usable gold. Otherwise
  `NOT_EVALUABLE`, with its n.
- **Also:** the confusion matrix, macro F1 per population, and the abstention rate.
- **Challenge-subset recall.** P-DET-COVERAGE-v1 is a challenge set, so the 22 §6 "each mode, challenge
  subset, recall ≥ 0.60" row is evaluated on it, for every mode with n ≥ 50.

**Layer A (structural routing):**
- exhaustive mismatch counts, which must be 0;
- validator rejections per source;
- human spot-check agreement, reported only here.

**How the 22 §6 thresholds apply across two populations.** Each threshold is evaluated on every population
where it is defined (enough gold for recall, enough predictions for precision), and it **must pass on each
of them**. A threshold defined on no population is `NOT_EVALUABLE`, and that mode does not qualify.

P-DET-COVERAGE-v1 precision depends on its constructed mix. It is labelled that way and never read as a
prevalence estimate.

A **combined** score over both populations may appear only as a secondary summary. It never feeds a gate,
and the populations are never merged into one prevalence statistic.

The thresholds themselves are the frozen 22 §6 values. This draft changes none.

## 12. Blockers and unknowns

| Id | Item | Effect | What resolves it |
|---|---|---|---|
| **B-1** | canonical-v3 post-adapter representation: sources, the adapter for each (Glaive above all), versions, a pre-classifier artifact | **RESOLVED** ([31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md)) | source manifest `configs/releases/toolpolicy_canonical_v3_sources.yaml` (sha256 LF `cc40f64eaa1d6dbff618e64d2ad3f4b7fe46b053d3db31e67bcdf1d272ef0b6b`); `normalization-v3` fingerprint `56e8abf2f952907c0e936ac9397bde5b0a0c4a14eda3a3ccbbd47960bff2fda3`, top manifest sha256 `da651a46dc3848cb7cfe9755e113a815f18a078a095978a65b36f8b739a00c29` (adapter version `2.1.0`, after the ToolACE repair; supersedes `2bd38492…`) |
| **B-2** | classifier input contract: which fields it reads (system message? tools? which assistant turn?), and how it labels multi-turn records | **RESOLVED** ([32](32-CLASSIFIER-INPUT-CONTRACT.md)) | `prose-decision-input-v1`: user message, final response, presented tools, structural-call flag. The system message is **not** shown, so §10 shows no system message. Multi-turn and post-tool records are ineligible |
| U-1 | whether LoopTool and BUTTON are in canonical-v3 | source allocation (§7.3 shows both cases) | **resolved: neither is** (31 §1). The "without LoopTool" column of §7.3 applies, and layer A is 30 |
| U-2 | ToolACE's representation in v3 | X supply | **resolved and repaired** (31 §5): structured calls. Under `2.0.0`, 9,785 of 9,786 call turns also kept the call text in `content`; `adapt_toolace_v2` (`2.1.0`) removes it (0 remain). Layer B and X supply are unchanged |
| U-3 | X supply in v3 | textual CALL `NOT_EVALUABLE` expected | **measured: 6** (all ToolACE; shortage 54; `pdet-coverage-v1.supply-v3.json`), to be re-reported at draw time |
| U-4 | gold yield per stratum | whether 50 usable per boundary is reached | annotation; any shortfall is a 22 §5 finding, not a top-up |
| U-5 | multi-turn tool-free records (proxy: Glaive 34,450, LoopTool 311) | not covered; no classifier permission on them from P-DET evidence | a separate, preregistered multi-turn component, once B-2 defines the unit |
| U-6 | whether P-DET-v1's items appear in canonical-v3 When2Call with the same text | P-DET-v1's representation fidelity | **checked**: 575 of 581 equivalent, 6 quarantined in v3 (§5; 31 §7) |
| U-7 | the source of P-CONF's `ANSWER` set | cross-exclusion | handled by the "whichever freezes second" rule (§9) |

This draft contains no unresolved design choice of its own. The DIRECT definition, the post-tool
exclusion, the two-layer split and the sampling rule are fixed by the study owner's instructions. The
rest follows from them and from the repository evidence cited above. **B-1 and B-2 are resolved**
([31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md), [32](32-CLASSIFIER-INPUT-CONTRACT.md)), and their
hashes are recorded in the table above. Next:
1. `pdet_coverage.py` is written against the normalization-v3 fingerprint and `prose-decision-input-v1`;
2. the draft goes to the study owner for adoption;
3. only then is anything drawn.
