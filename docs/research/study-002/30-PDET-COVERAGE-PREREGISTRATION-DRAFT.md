# 30 — P-DET-COVERAGE-v1: preregistration draft

**Status: DRAFT, 2026-09-15; revised 2026-09-16 after an engineering review and a counts-only dry run
(§13), and re-anchored the same day to normalization-v3 under adapter version `2.2.0` (U-8). Not adopted. No population has been drawn or written; no item has been annotated.** Both sampling
blockers of §12 are resolved:
- **B-1**, the canonical-v3 representation: [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md);
- **B-2**, the classifier input contract: [32](32-CLASSIFIER-INPUT-CONTRACT.md).

The builder exists (`src/opengrad/verification/pdet_coverage.py`). It refuses to write a population to
`reports/pdet-coverage/` while this document is a draft. Sampling waits only for adoption.

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
| Builder | `python -m opengrad.verification.pdet_coverage --build \| --verify \| --dry-run --output-dir DIR` |
| Structural call evidence | `src/opengrad/verification/call_fidelity.py` (§4) |
| Acceptance rules, as code | `src/opengrad/verification/pdet_coverage_metrics.py` (§11) |
| Pre-adoption dry run | `reports/pdet-coverage/pdet-coverage-v1.dry-run.json` (counts and hashes only; §13) |

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
a Glaive `<functioncall>` marker. The builder records every such record under the input contract's reason
(`STRUCTURAL_CALL` or `POST_TOOL_RESULT_RESPONSE`, 32 §4), with counts per source in the manifest. Records
whose first assistant turn is a structured call are eligible only for layer A.

Proxy scale, from `pdet-coverage-v1.supply-proxy.json`. The final turn is prose after tool use in:

| Source | Records |
|---|---:|
| Glaive | 49,287 |
| BUTTON | 7,941 (all of them) |
| ToolACE | 644 |

**Multi-turn tool-free records** are also out of the unit, under reason `MULTI_TURN_UNSUPPORTED` (32 §6). How
the classifier will label a conversation with several assistant turns is not specified (§12, B-2), so no
population can yet be built to match it. The proxy has 34,450 such Glaive records and 311 LoopTool
records. **No classifier permission on multi-turn records follows from P-DET evidence.** Covering them
takes a separately preregistered component, added as an amendment before any draw.

## 4. Two layers: structural CALL routing and prose decision classification

**Layer A — structural CALL routing.** A record with an authoritative structured call is CALL **by its
structure**. No prose heuristic guesses it. This is today's rule in `_base`
(`src/opengrad/data/adapters.py`): any structured `tool_calls` or `tool` message gives `decision: CALL`,
`confidence: derived`. Layer A is validated two ways:

1. **Exhaustively and deterministically**, over every accepted normalization-v3 row of Glaive, ToolACE
   and xLAM (`call_fidelity.py`, recorded as `structural_call_evidence` in the manifest). "Every structured
   call is routed CALL" cannot fail, because routing *is* that structure, so the check is one that can:
   - every call the **raw upstream row** expresses in its own syntax is re-read by a parser that shares no
     code with the adapters, and must equal the structured call: same count, same name, same arguments,
     in order. For ToolACE the first reader is Python's own parser (names and argument names masked). A
     turn it cannot read goes to a second reader written from ToolACE's documented format (names may hold
     spaces, commas or parentheses; values are Python or JSON literals). That reader shares the adapter's
     reading of the format, so rows it reads are weaker evidence and are counted separately
     (`rows_read_by_format_grammar`);
   - it runs on rows whatever their trajectory-gate status, split by status, so records the input
     contract rejects are checked too (under adapter version `2.1.0` that was ToolACE's 8,472
     `MISSING_TOOL_RESULT` rows; from `2.2.0` most of them declare `CALL_PREDICTION` and pass the gate, 31 §9);
   - raw rows normalization-v3 did not accept are accounted for from the disposition ledger (raw calls
     held, by disposition and reason);
   - status: `FAIL` on any mismatch; `INCOMPLETE` when some rows cannot be read independently (they are
     unverified, never counted as passing); `PASS` only when every call row was compared and matched.

   This is a code check, not a sample. Its dry-run result is in §13.
2. **A 30-item human spot-check** in a separate annotation task (`pdet-coverage-v1-routing`). It confirms
   that each structured payload is a plausible invocation of an offered tool, consistent with the request.
   It cannot detect a plausible extraction that changed an argument; item 1 does that. Allocation:
   - Glaive 15, ToolACE 10, xLAM 5 (LoopTool is not in canonical-v3);
   - within a source, split across trajectory-gate status in proportion to supply, with at least one
     item from each status that has supply. Gate-rejected records are eligible here (never for layer B),
     so the audit also covers the records the renderer rejects.

   With 0 errors in n items from one source, the 95% upper bound on that source's error rate is 3/n (rule
   of three). Bounds are reported **per source, never pooled**: the fixed 15/10/5 split does not represent
   any corpus.

**What CALL balancing permission rests on.** Layer B textual CALL is `NOT_EVALUABLE` unless stratum X
yields enough usable gold (§11). Structural CALL rests on item 1, not on the spot check.

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
| canonical-v3 (`normalization-v3`, [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md)) | **`adapt_glaive_v2`** (`glaive_v2`), chosen and recorded in `configs/releases/toolpolicy_canonical_v3_sources.yaml` | **structured**. On all 98,339 accepted records, raw `<functioncall>` markers equal the structured calls; 0 markers remain as text; unparseable calls are rejected | built and measured (fingerprint `60d3123e…`, adapter version `2.2.0`; Glaive unchanged since `2.1.0`); reproduces v2-final's Glaive counts reason for reason |

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
| **Artifact / adapter** | proxy: `adapt_glaive` (v1); v2-final: `adapt_glaive_v2`; v3: **`adapt_glaive_v2`** (31) | proxy and v2-final: `adapt_toolace`; v3: **`adapt_toolace_v3`** (31 §5, §9) | `adapt_xlam` + `xlam_types.py`; v3: **`adapt_xlam`** (31) | `adapt_looptool`; not in canonical-v2-final ("upstream source was not located"); **not in v3** (31) |
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
proxy, 11,002 of 14,329 Glaive tool-free single-exchange responses contain one, mostly inside a single
templated refusal. Those words would put Glaive's refusals into M. (No pool text is quoted here: this
document is annotator-facing, §9.)

**M match kind (reporting only).** An offered tool name can be an ordinary word, so an M item may not
really mention a tool. For each realized M item the builder records, **in the manifest only**, why it
matched: `invocation_talk`, `identifier_shaped_tool_name` (underscore, dot, hyphen, digit or camelCase)
or `word_tool_name`. It never changes the stratum or the draw and is never shown on an annotated item.
M metrics are reported on all of M and on the genuine-mention subset (the first two kinds).

### 7.3 Quotas and how they follow from the strata

The requirement is **at least 50 usable gold examples per required mode or boundary** (22 §5), where
usable means gold that is not UNKNOWN and not excluded.

- **Boundary strata (M, R, Q): 60 each.** That allows up to 1 in 6 items to end up unusable
  (60 × 5/6 = 50). The allowance is conservative: P-DET-v1 had 2 UNKNOWN in 581, but these strata sit on
  the ambiguous boundaries on purpose.
- **DIRECT: P1 80 and P2 40 (120 candidates).** Together they give 50 gold DIRECT if at least 42% of plain
  responses are gold DIRECT (50/120). Any DIRECT gold in R, Q and M adds to that. P1 is weighted over P2
  because the DIRECT/CALL decision exists only when a tool is offered. *(Raised from P1 60 and P2 30, which
  needed a 56% yield, before any label existed; §13.)* The hard DIRECT cases sit in R, Q and M, and §11
  gates them separately, so P1 and P2 cannot carry DIRECT alone.
- **X: every eligible item, up to 60.** This stratum can't be manufactured.
- **Layer A: 30 (35 with LoopTool).** This is an audit, not a classifier boundary; see §4.

| Stratum | Quota | Proxy supply (distinct skeletons), Glaive / ToolACE / LoopTool |
|---|---:|---|
| X | ≤ 60 | 0 / 6 / 0 → **shortage 54 expected** |
| M | 60 | 19 / 722 / 15 |
| R | 60 | 10,694 / 228 / 145 |
| Q | 60 | 102 / 169 / 152 |
| P1 | 80 | 58 / 375 / 3 |
| P2 | 40 | 65 / 491 / 0 |

**Source allocation inside a stratum** (`allocate`):
1. Split the quota equally across the included sources that have supply.
2. When a source runs short, split its remainder equally among the others; units too few to split go one
   each to the sources with the largest remaining supply.
3. Break ties by source name.

This is allocation, not backfill: it never moves quota *between* strata.

On proxy supply, at the original quotas (P1 60, P2 30), this gives:

| Stratum | With LoopTool (Glaive / ToolACE / LoopTool) | Without LoopTool (Glaive / ToolACE) |
|---|---|---|
| X | 0 / 6 / 0 | 0 / 6 |
| M | 19 / 26 / 15 | 19 / 41 |
| R | 20 / 20 / 20 | 30 / 30 |
| Q | 20 / 20 / 20 | 30 / 30 |
| P1 | 28 / 29 / 3 | 30 / 30 |
| P2 | 15 / 15 / 0 | 15 / 15 |

At those original quotas layer B was **276** either way. These proxy figures only show that the design can
be supplied. The real numbers, at the current quotas, are computed from the canonical-v3 artifact: the
pre-adoption dry run's are in §13, and the draw prints its own in the manifest.

**Measured on the real input** (`reports/pdet-coverage/pdet-coverage-v1.supply-v3.json`), before §9's
exclusions and dedup. This is supply analysis, not gold and not a draw. Its "allocation implied" column is
an upper bound at the original quotas and is **superseded by the dry run in §13**: after §9, Glaive keeps far
fewer distinct prompts in M and P1 than this table suggests. The pool is every normalization-v3 record of Glaive or ToolACE that is
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
- At the original quotas layer B stayed at 276, with X the only shortage. At the current quotas the dry
  run realizes more (§13), still with X the only shortage.

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
7. **Dedup** as in §9. Layer B strata are processed **scarcest first** (fewest candidates after the §9
   exclusions; ties in stratum order X, M, R, Q, P1, P2), each in rank order, and the first member of a
   duplicate group survives. A prompt shared across strata therefore stays in the stratum with least to
   spare. Layer A is processed in rank order.
8. **Ranking.** Within each stratum and source, sort by `sha256(seed + "|" + stratum + "|" + id)`,
   descending. This is the P-DET-v1 idiom (`sha256(seed ‖ pdet_id)` descending), with the stratum added so
   strata are independent. There is no RNG state.
9. **Take** the quota per stratum and source (§7.3), with the skeleton cap. Layer A takes each source's
   quota split across trajectory-gate status (§4).
10. **Write** `reports/pdet-coverage/pdet-coverage-v1.population.jsonl`,
    `pdet-coverage-v1.manifest.json` and a `.sha256` sidecar. The manifest holds:
    - the input manifest hashes;
    - every exclusion and dedup count;
    - supply, quota, realized count and shortage for each stratum and source;
    - the code version.

    The draw is byte-reproducible.
11. **Presentation order** for annotation: `sha256(seed + ":order:" + id)`, so strata, layers and sources
    are interleaved.

Code: `src/opengrad/verification/pdet_coverage.py` with `--build`, `--verify` and `--dry-run`. It reads
constants from its own module and imports nothing from `pdet.py`. `pdet.py` and `reports/pdet/` are not
touched. The builder:
- refuses any input whose top manifest, fingerprint, per-source manifests, shards or canonical-v3 source
  manifest (`5bb4961c…`) differ from the recorded ones, and every exclusion input whose bytes changed;
- refuses `reports/pdet/` always, and a population in `reports/pdet-coverage/` until adoption;
- in `--dry-run`, writes counts and hashes only: no item, no item id, no per-item source or stratum. The
  draw is byte-reproducible, so a written population *is* the future blind sample.

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
3. by normalized user prompt, **one per prompt across all of layer B** (not per stratum: items sharing a
   prompt are not independent, and the Wilson intervals of §11 assume independence). Which stratum keeps a
   shared prompt is fixed by the processing order of §8.7;
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
  - the trajectory-gate status (layer A) and the M match kind, which exist in the manifest only;
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

**The rules are code.** `src/opengrad/verification/pdet_coverage_metrics.py` (`pdet-coverage-metrics-v1`)
computes every row below from gold labels and predictions, and its tests pin worked numbers
(`tests/verification/test_pdet_coverage_metrics.py`). It was written before any label or classifier existed,
so no choice here can follow the results. The thresholds are the frozen 22 §6 values; what 22 left open is
fixed as follows. The study owner accepted the minimum sizes on 2026-09-16; they bind on adoption.

| Rule | Fixed as |
|---|---|
| Usable gold | gold in {CALL, DIRECT, CLARIFY, UNSUPPORTED}, not excluded (e.g. `EXPOSED_WORKED_EXAMPLE`) |
| Recall row evaluable | at least **50** usable gold of the mode on that population (22 §5) |
| Precision row evaluable | at least **50** predictions of the mode on that population. So DIRECT precision on P-DET-v1 (0 gold) gates only once there are 50 DIRECT predictions there, and then fails |
| Across populations | a row must pass on **every** population where it is evaluable; "primary" and "secondary" are reporting labels only. A row evaluable nowhere is `NOT_EVALUABLE`, and its mode does not qualify |
| Challenge rows | P-DET-COVERAGE-v1: strata **R, Q, M, X only**, never P1/P2; P-DET-v1: its frozen challenge component. At least **30** usable gold of the mode; recall ≥ 0.60 |
| False CALL | a CALL prediction on UNKNOWN (ambiguous) gold counts in the CALL precision denominator. In stratum M, CALL predictions on UNKNOWN gold are a gated count: **0** allowed |
| Abstention | a miss for recall, excluded from precision denominators; rate ≤ 0.15 over usable items |
| Macro F1 | over the modes with at least 50 usable gold in that population |
| DIRECT precision on the pool | per source, each coverage stratum's DIRECT precision weighted by its share of that source's eligible pool (`pool_strata` in the manifest, before any sampling), with a seeded stratified bootstrap interval. Evaluable with at least **20** DIRECT predictions from the source and a sample in every pool stratum; ≥ 0.80 to pass |
| Qualification | a mode qualifies when all its rows pass (DIRECT: recall, precision, challenge recall; UNSUPPORTED: recall, challenge; CLARIFY: F1, challenge; CALL: precision, challenge, CALL-on-ambiguous in M) and macro F1 and abstention pass. DIRECT balancing is permitted **only for sources** whose pool-weighted precision passes. C1 needs DIRECT and UNSUPPORTED (22 §6) |

Why the challenge rule matters: P1 and P2 are cue-free by construction, so they hold the easy DIRECT cases.
With 80 easy DIRECT right and 15 hard DIRECT wrong, pooled recall is 80/95 = 0.84 and passes, but the
challenge row sees only the 15 hard items and is not evaluable, so DIRECT does not qualify. Why the pool
weighting matters: Glaive's eligible pool is almost entirely stratum R, so a small false-DIRECT rate on R
dominates the DIRECT labels the classifier would write there, however clean P1 looks.

**Also reported, never gated:** boundary accuracy per stratum (R: DIRECT vs UNSUPPORTED; Q: DIRECT vs
CLARIFY; M: tool mention, on all of M and on its genuine-mention subset; P1: DIRECT with a tool offered),
the confusion matrix, both sides' gold counts for each boundary, and textual-CALL recall and precision on X
(`NOT_EVALUABLE` below 50 usable gold, with its n).

**Layer A (structural routing):** the structural call evidence of §4 (mismatches must be 0, unverified
rows reported), validator rejections per source, and human spot-check agreement with per-source bounds,
reported only here. Success on layer A is never evidence that the prose classifier recognises CALL.

P-DET-COVERAGE-v1 precision depends on its constructed mix. It is labelled that way and never read as a
prevalence estimate.

A **combined** score over both populations may appear only as a secondary summary. It never feeds a gate,
and the populations are never merged into one prevalence statistic.

The thresholds themselves are the frozen 22 §6 values. This draft changes none of them.

## 12. Blockers and unknowns

| Id | Item | Effect | What resolves it |
|---|---|---|---|
| **B-1** | canonical-v3 post-adapter representation: sources, the adapter for each (Glaive above all), versions, a pre-classifier artifact | **RESOLVED** ([31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md)) | source manifest `configs/releases/toolpolicy_canonical_v3_sources.yaml` (sha256 LF `5bb4961cd100b4d5975a9596bee345e5234ce527029a4d6183f0b9a662bf2f90`); `normalization-v3` fingerprint `60d3123e1c6cef67f904a81f29669dcff44ecc5aeb0e71178a3b5d4e94bfa75a`, top manifest sha256 `88b22ccc8ead4d0ba95156ab4a6c6a4232011d2d0ce490916982aca37752ee9e` (adapter version `2.2.0`, after the U-8 decision; supersedes `56e8abf2…` under `2.1.0` and `2bd38492…` under `2.0.0`) |
| **B-2** | classifier input contract: which fields it reads (system message? tools? which assistant turn?), and how it labels multi-turn records | **RESOLVED** ([32](32-CLASSIFIER-INPUT-CONTRACT.md)) | `prose-decision-input-v1`: user message, final response, presented tools, structural-call flag. The system message is **not** shown, so §10 shows no system message. Multi-turn and post-tool records are ineligible |
| U-1 | whether LoopTool and BUTTON are in canonical-v3 | source allocation (§7.3 shows both cases) | **resolved: neither is** (31 §1). The "without LoopTool" column of §7.3 applies, and layer A is 30 |
| U-2 | ToolACE's representation in v3 | X supply | **resolved and repaired** (31 §5): structured calls. Under `2.0.0`, 9,785 of 9,786 call turns also kept the call text in `content`; `adapt_toolace_v2` (`2.1.0`) removes it (0 remain). Layer B and X supply are unchanged |
| U-3 | X supply in v3 | textual CALL `NOT_EVALUABLE` expected | **measured: 6** (all ToolACE; shortage 54; `pdet-coverage-v1.supply-v3.json`), to be re-reported at draw time |
| U-4 | gold yield per stratum | whether 50 usable per boundary is reached | annotation; any shortfall is a 22 §5 finding, not a top-up |
| U-5 | multi-turn tool-free records (proxy: Glaive 34,450, LoopTool 311) | not covered; no classifier permission on them from P-DET evidence | a separate, preregistered multi-turn component, once B-2 defines the unit |
| U-6 | whether P-DET-v1's items appear in canonical-v3 When2Call with the same text | P-DET-v1's representation fidelity | **checked**: 575 of 581 equivalent, 6 quarantined in v3 (§5; 31 §7) |
| U-7 | the source of P-CONF's `ANSWER` set | cross-exclusion | handled by the "whichever freezes second" rule (§9) |
| U-8 | why 8,472 ToolACE records fail `MISSING_TOOL_RESULT` | none on layer B; layer A's ToolACE gate split | **decided** (study owner, 2026-09-16; [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) §9): from adapter version `2.2.0`, call-final records of one measured shape declare `CALL_PREDICTION`, as OpenGrad's structural inference, not ToolACE's statement. The calls were already extracted exactly (§4). Canonical-v2 and every Study 002 arm are unchanged |
| U-9 | ToolACE call rows the first independent reader could not parse | structural call evidence was `INCOMPLETE` | **resolved** (study owner, 2026-09-16): a second, format-grammar reader reads them; every call row is now compared, and those rows are counted as weaker evidence (§4, §13) |

The DIRECT definition, the post-tool exclusion, the two-layer split and the sampling rule are fixed by the
study owner's instructions; each revision in §13 was decided by the study owner. **B-1 and B-2 are
resolved** ([31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md), [32](32-CLASSIFIER-INPUT-CONTRACT.md)),
and their hashes are recorded in the table above. Next:
1. done: `pdet_coverage.py`, its structural call evidence and the metric code are written and tested;
2. the study owner adopts this draft (amendment `study_002_prereg_v4`), which makes the minimum sizes of
   §11 binding;
3. only then is anything drawn.

## 13. Pre-adoption engineering review and dry run (2026-09-16)

**What happened, in order.** No label was involved at any step.
1. The builder was written and run twice into a scratch directory. Both draws were byte-identical. Their
   population files were deleted unread: no item text was printed, and the study owner was shown counts
   only.
2. An engineering review (gstack `/plan-eng-review`) examined this draft, the builder and those counts,
   with two independent outside reviews (Codex, and a separate Claude Opus 5 session) that read the plan
   and counts, not the pools.
3. The study owner decided each finding. The design changes below were made because of what the counts
   showed about supply, never because of any outcome.
4. The revised builder was dry-run in counts-only mode twice, byte-identically, and the record was written
   to `reports/pdet-coverage/pdet-coverage-v1.dry-run.json`. It holds counts and hashes, no item and no
   item id. `test_the_real_draw_reproduces_the_recorded_dry_run` re-draws and compares where the inputs
   exist.
5. After the U-8 decision, normalization-v3 was rebuilt under adapter version `2.2.0`, the builder's pins
   were moved to it, and the dry run was re-recorded twice, byte-identically. Layer B is unchanged in every
   count. Only layer A's ToolACE split across gate status moved, because most call-final ToolACE records now
   pass the gate. The undrawn population's sha256 therefore moved from `4c7ca7b5…` to the value below. No
   population was written at any step.

**Exposure disclosure.** Two pieces of pool-adjacent text reached the study owner's session during the
review, before annotation: §7.2's quoted refusal template (now removed from this document), and, in one
tool output, Glaive refusal records from the tracked quantization calibration manifests. Neither names an
item of this population, but templated Glaive refusals may resemble items in R. If the study owner judges
any item exposed at annotation time, it gets the `EXPOSED_WORKED_EXAMPLE` exclusion (§9, 27).

**Decisions (study owner, 2026-09-16).**

| # | Finding | Decision |
|---|---|---|
| D1, D7 | one-per-prompt dedup kept a shared prompt in whichever stratum ranked it higher; Glaive fell to 2 of 60 in M and 1 of 60 in P1 | keep one per prompt across layer B; process strata scarcest first (§8.7, §9) |
| D2 | 50 usable DIRECT needed a 56% yield from 90 candidates, with no prior estimate | P1 80, P2 40: a 42% yield (§7.3) |
| D3, D8 | the promised exhaustive layer A check had no code, and routing-vs-structure cannot fail | independent re-read of every raw call, all rows including gate-rejected ones; spot check split by gate status (§4) |
| D4 | a written dry-run population is the future blind sample | populations deleted; `--dry-run` writes counts only (§8) |
| D5 | the design changed after a (label-free) dry run | this section |
| D6 | untested refusal and failure branches | tests for every branch, plus a real-artifact reproduction test that skips, never passes, without inputs |
| D9 | pooled DIRECT recall can pass on the easy P1/P2 items alone | challenge rows on R, Q, M, X only, minimum 30 (§11) |
| D10 | nothing checked DIRECT precision on the pool the classifier will label | pool-weighted precision per source; DIRECT permission per source (§11) |
| D11 | CALL on ambiguous gold escaped every metric | counted as false CALL; 0 allowed in M (§11) |
| D12 | "defined" had no numbers | numeric minimums and the rules as tested code (§11) |
| D13 | nothing named would justify CALL permission | structural call evidence is the basis; spot-check bounds per source (§4) |
| D14 | §7.2 quoted pool text | quote removed; a test keeps quoted pool text out of this document |
| D15 | the source manifest's hash was recorded but not enforced | the builder refuses any other source manifest (§8) |
| D16 | M may catch ordinary-word tool names | manifest-only match kind; M reported on its genuine-mention subset (§7.2, §11) |
| D17 | 8,472 ToolACE records fail `MISSING_TOOL_RESULT` | open item in 31 (U-8) |
| U-8 | the construction pattern of ToolACE's call-final rows (31 §9) | read qualifying records as `CALL_PREDICTION` prospectively (adapter version `2.2.0`); the pins and the dry run below were re-recorded on the rebuilt artifact |
| U-9 | 120 ToolACE call rows were unreadable by the first independent reader, so the evidence was `INCOMPLETE` | extend the reader: a second, format-grammar reader, its rows counted as weaker evidence (§4) |
| §11 minimums | 50 gold (recall), 50 predictions (precision), 30 hard gold (challenge), 20 DIRECT predictions per source | accepted as proposed |

**The revised dry run** (generated from the record; `test_dry_run_table_in_the_preregistration_is_generated`
keeps this block equal to it):

<!-- dry-run-table:start (generated by pdet_coverage.render_dry_run_table) -->
| Stratum | Quota | Supply after dedup | Realized | Shortage | glaive | toolace |
|---|---:|---:|---:|---:|---:|---:|
| X | 60 | 6 | 6 | 54 | 0 | 6 |
| M | 60 | 712 | 60 | 0 | 11 | 49 |
| R | 60 | 410 | 60 | 0 | 30 | 30 |
| Q | 60 | 254 | 60 | 0 | 30 | 30 |
| P1 | 80 | 370 | 80 | 0 | 6 | 74 |
| P2 | 40 | 551 | 40 | 0 | 20 | 20 |

| Layer A source | Quota | Realized | Gate-rejected realized / supply | Valid realized / supply |
|---|---:|---:|---:|---:|
| glaive | 15 | 15 | 1 / 107 | 14 / 5943 |
| toolace | 10 | 10 | 1 / 41 | 9 / 8885 |
| xlam | 5 | 5 | 1 / 1211 | 4 / 54356 |

- Realized: layer B 306, layer A 30, total 336.
- Population sha256 (not written): `755bc16e79ceb1cb9e461c0fe8b9125628c16f263ed24f9e008f55b613a7158c`.
- Stratum M match kinds: identifier_shaped_tool_name 44, invocation_talk 2, word_tool_name 14.
- Structural call evidence: **PASS**; mismatched rows glaive 0, toolace 0, xlam 0; rows the independent readers could not parse (unverified) glaive 0, toolace 0, xlam 0; rows read only by the format-grammar fallback (weaker evidence) glaive 0, toolace 120, xlam 0.
<!-- dry-run-table:end -->
