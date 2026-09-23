# Study 002 — handoff (2026-09-15)

> **SUPERSEDED for current state (2026-09-24).** This brief describes the repository as of 2026-09-15 and
> was committed on 2026-09-18; much of §2 and §20 is out of date (a classifier now exists and is frozen,
> P-DET-COVERAGE-v1 and -v2 are drawn and labelled, canonical-v3 is built, the normalization-v3
> fingerprint is `60d3123e…`). The current state, blockers and open decisions are in
> [README.md — Current state](README.md#current-state). Keep this file for its history and its
> working procedures, not for status.

This file lets a new engineer or researcher continue Study 002 without the chat history that produced it.
Every number below was checked against the repository or the local working state on 2026-09-15, and the
file or command it comes from is named. Where the brief for this handoff and the repository disagreed, the
repository wins and the difference is stated (§7).

Status words used throughout:
- **ESTABLISHED:** measured, with the artifact named.
- **PROVISIONAL:** exists but is not final, for example not frozen.
- **HYPOTHESIS:** plausible, not tested.
- **BLOCKED:** cannot proceed until something else exists.
- **NOT STARTED:** no work exists.

---

## 1. What Study 002 is trying to solve

Study 001 improved tool-calling metrics. A later evaluation then exposed a severe regression in ordinary
direct answering: after supervised fine-tuning, the model refused 100% of bare GSM8K questions (1,319 of
1,319). The base model answered them, with 67.4% zero-shot accuracy
(`reports/GENERAL_CAPABILITY_REGRESSION.md`, `reports/ERRATA.md`, `ROADMAP.md` item 16).

The follow-up audit found two things:

1. **The training pipeline collapsed several different non-tool behaviours into one label, `ANSWER`.** A
   response that answers, one that asks a clarifying question and one that declines all became `ANSWER`
   whenever no tool call was present.
2. **The original evaluation had no real DIRECT gold population.** The held-out evaluation was When2Call's
   MCQ test set, where "answer directly" is never the correct answer. So a model that stopped answering
   directly could still pass the gate.

**Study 002 tests one idea.** If CALL, DIRECT, CLARIFY and UNSUPPORTED are kept separate while the
training data is built and balanced, can the tool-calling gains be kept without the direct-response
collapse?

**Causation has not been established.**
- The training-label collapse is a *plausible mechanism*.
- The missing DIRECT evaluation population *explains why the regression could go undetected*.
- What actually *caused* the regression still needs a controlled ablation. That ablation is Study 002's
  arm design, [04](04-ARM-MATRIX.md).

Nothing in this file, and nothing done so far, establishes a cause.

---

## 2. Current overall status

**Phase:** validation-population construction. The data representation the future classifier will read is
established (B-1, B-2). The next engineering task is the ToolACE representation repair (§12). No classifier
exists, and nothing has been trained.

| Item | Status | Evidence |
|---|---|---|
| Study 001 | frozen, untouched | tag `study-001`; `docs/research/STUDIES.md` |
| canonical-v1 / canonical-v2 / canonical-v2-final | historical, never rewritten | `configs/releases/toolpolicy_canonical_v2*.yaml`, `.release/hf/…/release-manifest.json` |
| P-DET-v1 population | **frozen** (581 rows) | `reports/pdet/pdet-v1.population.jsonl`, `python -m opengrad.verification.pdet --verify` → `PASS` |
| P-DET-v1 annotation | human pass complete; model pass kept as superseded comparison; **gold NOT frozen** | `.annotation/pdet-v1.sqlite3` (§7) |
| P-DET-COVERAGE-v1 | preregistration **draft**, not adopted; **no population drawn** | [30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md) |
| B-1: canonical-v3 source/adapter representation | **complete** | [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) |
| B-2: classifier input contract | **complete** | [32](32-CLASSIFIER-INPUT-CONTRACT.md) |
| canonical-v3 pre-classifier representation (`normalization-v3`) | built, reproducible, local only | fingerprint `2bd384929a5b870bc3836729c808362c60d0d98717f086587e12af6608104aaa` |
| Decision classifier | **does not exist** | — |
| Behaviour-balanced mixture materialization | **does not exist** | `src/opengrad/data/mixture.py` only validates weights ([21](21-C1-IMPLEMENTATION-STATUS.md)) |
| Study 002 training | **none has occurred** | — |

**Git (verified with `git branch --show-current` and `git log`):**
- branch `study-002/canonical-v3-b1-b2`;
- head commit `8e78837`, whose parent is `62a5721`, which is `master` and `origin/master`;
- the branch is **local only**: no upstream, not pushed, not merged.

§19 gives the full working-tree state.

---

## 3. The behavioural taxonomy

Four policy modes, describing what the assistant does at the **decision point**, before any tool has been
used ([22](22-PDET-PROTOCOL.md) §1–§3; the DIRECT wording from [30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md) §2):

| Mode | Plain meaning |
|---|---|
| **CALL** | The assistant invokes a tool. |
| **DIRECT** | **The assistant answers the user's request normally, without requiring or using a tool.** |
| **CLARIFY** | The assistant asks the user for something it needs before it can act (a missing parameter, an ambiguity). |
| **UNSUPPORTED** | The assistant declines, or explains that it cannot do what was asked with what it has. |

- **UNKNOWN / AMBIGUOUS is not a fifth behaviour.** It is an annotation or data status: "this item cannot
  be assigned a mode". In the annotation tool, UNKNOWN requires an `ambiguity_status`.
- **A prose response after a tool result is not DIRECT.** The sequence user → tool call → tool result →
  assistant prose is a downstream tool-result response, outside the four-way pre-action decision. The
  current classifier contract excludes these records ([32](32-CLASSIFIER-INPUT-CONTRACT.md) §5).
- **Naming.** The code's taxonomy (`src/opengrad/data/behavior.py`, `DECISIONS`) calls DIRECT `ANSWER`:
  `{CALL, ANSWER, CLARIFY, UNSUPPORTED}`. The annotation label `DIRECT` corresponds to the code's `ANSWER`.

---

## 4. Study 001 audit findings

| Finding | Evidence |
|---|---|
| The canonical taxonomy already supported four decisions: CALL / ANSWER / CLARIFY / UNSUPPORTED. | `src/opengrad/data/behavior.py` (`DECISIONS`) |
| Adapter behaviour derivation is message-shape only. Any `tool` message or structured `tool_calls` gives `CALL`; **everything else gives `ANSWER`** (`confidence: derived`). Prose records therefore became `ANSWER` unless they held a structured call. | `src/opengrad/data/adapters.py`, `_base` (the `metadata["behavior"]` block) |
| The corpus M0 trained on (canonical-v2-final, 173,237 records) carries **only `CALL` (116,006) and `ANSWER` (57,231)** decision labels. Per source: xLAM CALL 57,342; Glaive CALL 49,586 / ANSWER 48,753; ToolACE CALL 9,078 / ANSWER 1,973; When2Call ANSWER 6,505. | `results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json` (`per_source.*.decision_distribution`) |
| CLARIFY and UNSUPPORTED are absent as real training decisions. No record is labelled either, and zero records are labelled `CANNOT_ANSWER`. | same file; `ROADMAP.md` item 16 |
| Refusal-shaped supervised targets labelled `ANSWER`: 18,114 of 173,237 (10.5%). By source: When2Call 4,038 of 6,505 (62.1%), Glaive 14,066 of 98,339 (14.3%), ToolACE 10, xLAM 0. These counts come from a **heuristic detector whose precision has not been measured**. | same file; `ROADMAP.md` item 16; [03](03-PREREGISTRATION.md) "Prior knowledge" item 3 |
| Study 001's held-out evaluation had no DIRECT gold. When2Call's MCQ test (3,652 questions) marks the correct answer as `tool_call` (1,295), `cannot_answer` (1,295) or `request_for_info` (1,062), **never `direct`**. | [README](README.md), finding "P-DET-v1 contains no CALL and no DIRECT item" |
| Regression signals first appear on Base → M0, the SFT stage. M1-v1 (DPO directly on the base) also refuses 70.7% (933 of 1,319) of zero-shot GSM8K, so the effect is not specific to SFT. | `reports/FINAL_CAMPAIGN_AUDIT.md`; `reports/ERRATA.md` |

What these findings do and do not show:

- **They explain why a direct-response collapse could pass the gate.**
- **They do not establish what caused it.** Study 001 has one lineage, one seed and no replicate.
- **Beware a superseded audit.** An earlier refusal audit (`…/sft_refusal_supervision_audit.json`: 21,749 of
  217,903, and When2Call 7,490 of 14,829) measured the normalization-v1 sources, not M0's corpus. Do not
  quote it for M0 (`reports/ERRATA.md`; [01](01-LESSONS-FROM-STUDY-001.md) L4, "#80").

---

## 5. Provenance and version work (done)

**The defect.** One artifact recorded two adapter versions:
- every canonical-v2 record's `metadata.adapter_version` is `"1.0.0"`, the module constant in
  `src/opengrad/data/adapters.py`;
- the materialization manifest's `config.adapter_version` is the literal `"1.0.2"`, hard-coded in
  `src/opengrad/data/materialize.py`.

So the adapter that produced a v2 record cannot be read from the v2 artifacts
([21](21-C1-IMPLEMENTATION-STATUS.md), phase 1).

**The fix.** `src/opengrad/data/versions.py` is the single authoritative version source. Current values:

| Constant | Value |
|---|---|
| `ADAPTER_VERSION` | `2.0.0` |
| `SCHEMA_NORMALIZATION_VERSION` | `source-schema-normalization-v1` |
| `DECISION_CLASSIFIER_VERSION` | `prose-decision-classifier-v1` (named, not implemented) |
| `BEHAVIOR_TAXONOMY_VERSION` | `tool-use-behavior-taxonomy-v1` |
| `CANONICAL_SCHEMA_VERSION` | `tool_use_ir_v1` |
| `SUPERVISION_CONTRACT_VERSION` | `supervision_contract_v1` |
| `NORMALIZATION_VERSION` | `normalization-v3` |
| `CLASSIFIER_INPUT_CONTRACT_VERSION` | `prose-decision-input-v1` |

Two separate checks:
- `check_version_agreement(record_metadata, manifest_config)` checks **internal agreement**: a record and
  its manifest must not disagree. It deliberately tolerates historical versions, so v1/v2 stay readable.
- `check_artifact_matches_authoritative_versions(manifest_config)` checks **agreement with the current
  code**. It applies to newly built artifacts only.

Historical artifacts were **intentionally not rewritten**. The v1/v2 manifests keep `1.0.2`, and their
records keep `1.0.0`.

Two scope notes:
- `adapters.ADAPTER_VERSION` is still `"1.0.0"` in `adapters.py`, which was left unchanged so v1/v2 stay
  reproducible.
- The normalization-v3 builder overwrites the version in every row, including the supervision block, with
  `versions.ADAPTER_VERSION`.

---

## 6. Schema normalization (done)

- **Why When2Call lost rows.** When2Call's tool schemas use Python/typing annotations (`"dict"`,
  `"str, optional"`, `"List[int]"`, …), not JSON Schema type names. The strict canonical validator
  therefore quarantined **8,445 of 15,000** When2Call rows as `SCH_UNSUPPORTED_TYPE`. That matches the v2
  release manifest's count exactly.
- **Where translation happens.** `src/opengrad/data/source_schema.py` (`translate_source_tools`,
  `translate_source_schema`) translates at the source boundary, **before** the strict validator.
- **The shared validator (`src/opengrad/data/schema.py`) was not weakened.**
- **Unrepresentable forms are quarantined with explicit `SRC_SCHEMA_*` reasons**, never coerced.
- **Version:** `source-schema-normalization-v1`.

Measured recovery ([21](21-C1-IMPLEMENTATION-STATUS.md), phase 2):

| Source | Rows | OK before | OK after | Recovered | Newly rejected |
|---|---:|---:|---:|---:|---:|
| When2Call `train_sft` | 15,000 | 6,555 | 14,872 | 8,317 | 0 |
| ToolACE | 11,300 | 11,160 | 11,160 | 0 | 0 |
| Glaive | 112,960 | 112,827 | 112,827 | 0 | 0 |

- **Not everything was recovered.** 128 When2Call rows stay quarantined: `SRC_SCHEMA_MULTI_TYPE_UNION` 73
  and `SRC_SCHEMA_UNREPRESENTABLE_TYPE` 55.
- **ToolACE and Glaive lose rows to different error classes**, which translation does not and should not
  fix: `SCH_UNSUPPORTED_KEYWORD`, `SCH_PROPERTIES_NOT_OBJECT`, `SCH_ADDITIONAL_PROPERTIES_NOT_BOOLEAN`,
  `SCH_REQUIRED_NOT_STRING_LIST`, and Glaive's malformed call syntax.
- **Where translation applies.** In normalization-v3 it runs **only for When2Call**, per the source
  manifest. xLAM keeps its own `xlam_types.py` parser; 21 leaves it out of scope.
- **Tests:** `tests/data/test_source_schema.py`.

---

## 7. P-DET-v1

**Purpose.** A frozen, independently annotated population for validating the *future* deterministic
decision classifier before it may be used for C1 balancing ([22](22-PDET-PROTOCOL.md)).

| | |
|---|---|
| Population | `reports/pdet/pdet-v1.population.jsonl`, **581 rows** |
| Split | 400 prevalence + 181 challenge (targets 400 + 200; challenge shortfall 19) |
| `population_sha256` | `6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b` |
| Manifest | `reports/pdet/pdet-v1.manifest.json` (sidecar digest `a10bcb7866012aed35a04821f164a8165de2c625de8f351d6c68c5f836eae15c`) |
| Protocol | `pdet-002-v1`, seed `opengrad-pdet-002-v1`: [22-PDET-PROTOCOL.md](22-PDET-PROTOCOL.md) |
| Candidate pool | `data/processed/normalization-v1/when2call-sft` (14,829 records) |
| Annotation instrument | [23-PDET-ANNOTATION-INSTRUMENT.md](23-PDET-ANNOTATION-INSTRUMENT.md) |
| Annotator checklist | [26-PDET-ANNOTATOR-CHECKLIST.md](26-PDET-ANNOTATOR-CHECKLIST.md) |
| Implementation addenda | [25-PDET-IMPLEMENTATION-ADDENDUM.md](25-PDET-IMPLEMENTATION-ADDENDUM.md), [27-PDET-EXPOSED-WORKED-EXAMPLES.md](27-PDET-EXPOSED-WORKED-EXAMPLES.md) |
| Amendments | [28](28-PDET-MODEL-LABEL-AMENDMENT.md) `study_002_prereg_v2` (model reference labels); [29](29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md) `study_002_prereg_v3` (rationale optional) |
| Builder / verifier (frozen code) | `src/opengrad/verification/pdet.py` |
| Verify | `python -m opengrad.verification.pdet --verify` (read-only; `PASS` on 2026-09-15) |

### Current annotation state

Read from `.annotation/pdet-v1.sqlite3` through a **read-only** copy on 2026-09-15. The last annotation
update was at `2026-09-15T09:35:12Z`.

| Session | Annotator | Kind | Items labeled | Label distribution |
|---|---|---|---:|---|
| `pass-a` | `arjhinety` | human | **581** | UNSUPPORTED 298, CLARIFY 281, UNKNOWN 2, DIRECT 0, CALL 0 |
| `model-a` | `model.claude-opus-5` | model (Claude Opus) | 570 | UNSUPPORTED 294, CLARIFY 271, DIRECT 4, UNKNOWN 1 |

- **The brief for this handoff said "11 human Pass A labels, 570 model labels". That is out of date.** It
  matches the work-in-progress export `reports/pdet/annotation/wip/` (exported `2026-09-15T08:09:58Z`:
  pass-a 11, model-a 570). The store has since recorded human labels for all 581 items. The WIP export is
  **stale**.
- **Rationales.** Only 11 of the 581 human labels carry a rationale. Amendment 29 made the rationale
  optional, and under it an empty rationale does not make a label lower-confidence.
- **Flags.** The human pass has 0 flagged items; the model pass has 64.
- **Composite reference** (first listed session wins, `--sessions pass-a model-a`): every item now takes the
  human label, so the composite equals the human distribution above.
- **Human–model agreement** on the 570 items both labeled: **559**. That is one human against one model,
  **not** inter-annotator agreement.
- **Gold is NOT frozen.** The `freezes` table is empty. The 22 §4 single-annotator re-read, which covers
  both UNKNOWN items, has not been done.
- **The definition amendment is recorded in the store.** `definition_history`: new definition SHA-256
  `c00eab8d903cfbf5b3f033763306d063183f5e8c8a22f6ed81e52489695c958e`, recorded `2026-09-15T08:44:48Z`, citing
  [29](29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md).

Rules that stay in force (28, 29):
- Where both exist, human labels take precedence.
- Model labels are model judgments. **They must never be called human-validated.**
- Any classifier result measured against a composite that contains model-sourced items is provisional.
- Only the study owner decides whether to re-read the 2 UNKNOWN items and freeze.

### Exposed worked examples (amendment present)

Three items are shown with illustrative labels in 23 §4, so they are status `EXPOSED_WORKED_EXAMPLE`
([27](27-PDET-EXPOSED-WORKED-EXAMPLES.md)): tool items **#3, #4, #62** (`pdet_index` 2, 3, 61, all challenge).
They remain in the population and are annotated normally, but they are excluded from metrics, agreement
statistics and "untouched gold" claims. That leaves **578 of 581 items metric-eligible**. Both human
UNKNOWN items are among the three.

### Why P-DET-v1 cannot stand alone

- **CALL: 0 items, by construction.** The source split contains no tool call.
- **DIRECT: 0 human labels.** When2Call's SFT responses ask or decline. The model pass labelled 4 DIRECT,
  and the human pass labelled none of those DIRECT.
- **So several frozen 22 §6 thresholds cannot be evaluated on P-DET-v1:**
  - DIRECT recall ≥ 0.80;
  - DIRECT precision ≥ 0.80;
  - CALL precision ≥ 0.95;
  - challenge recall ≥ 0.60 for those modes.

  22 §5 also asks for ≥ 50 gold examples per mode.
- **Consequence: P-DET-v1 alone cannot authorise C1** ([README](README.md), finding; `reports/ERRATA.md`
  §14).

---

## 8. The annotation tool (`opengrad-annotate`)

Full guide: [`docs/ANNOTATION_TOOL.md`](../../ANNOTATION_TOOL.md). Code: `src/opengrad/annotation/`. UI:
`integrations/annotate-ui/`. Task config: `configs/annotation/pdet-v1.yaml`.

```bash
# install (once)
uv pip install -e ".[annotation]"          # or: pip install -e ".[annotation]"
cd integrations/annotate-ui && npm install && npm run build && cd ../..

# preflight and verification
python -m opengrad.verification.pdet --verify     # frozen population must still verify
opengrad-annotate check pdet-v1                   # read-only preflight

# start a pass; run the same command again to RESUME where you stopped (aliases: serve, resume)
opengrad-annotate start pdet-v1 --annotator <your-id> --session pass-a

# progress; label counts only for the named session
opengrad-annotate status pdet-v1 --session pass-a

# re-verify every change-log hash chain
opengrad-annotate audit pdet-v1

# composite reference counts
opengrad-annotate reference pdet-v1 --sessions pass-a model-a

# verify an exported package
opengrad-annotate verify <manifest> [--require-source]
```

| | |
|---|---|
| Web UI | <http://127.0.0.1:8765/> (loopback only) |
| SQLite working state (git-ignored) | `.annotation/pdet-v1.sqlite3` |
| WIP export (never gold) | `reports/pdet/annotation/wip/` (`opengrad-annotate export …`) |
| Eventual gold export | `reports/pdet/annotation/`, manifest `pdet-v1.annotation-manifest.json` (`freeze-gold`, which **locks** the sessions) |
| Pinned review queue | `reports/pdet/review/pdet-v1.priority-review.json` (117 items; its file names reference labels, so an annotator does not open it) |
| Model-label audit trail | `reports/pdet/provenance/model-a/` (verbatim transcripts in `local/`, git-ignored because they contain an e-mail address) |

Behaviour:
- **Rationale is optional** for ordinary labels (amendment 29). CALL, DIRECT, CLARIFY and UNSUPPORTED save
  at once.
- **UNKNOWN still requires an `ambiguity_status`.** It stays configured that way in
  `configs/annotation/pdet-v1.yaml`.
- **Model judgments are hidden from human passes, before and after a label is submitted.** A pass sees
  counts only ("provisional, model only"), never a model label.
- **Challenge families and other blinded fields are hidden** (`blind_fields` in the config), as are the
  rubric sections carrying sampling cues.
- **Sessions are isolated.** One server serves one session. There is no route that returns another
  session's labels, and a session refuses to open under a different annotator id.
- **Every change is saved at once with a hash-chained history.** Restarting `start` resumes.
- **Passes are never silently overwritten.**
  - A relabel is an explicit, recorded change; undo appends rather than deletes.
  - Adjudication writes its own records and never edits the passes.
  - `freeze-gold` locks the sessions it names.
  - A frozen gold package directory is never overwritten.

**Limitations** (the guide's "Integrity model and limitations"):
- no application authentication; the server binds to loopback only;
- annotator ids are declared, not authenticated;
- hashes and `.sha256` sidecars are integrity checks, **not signatures**;
- agreement statistics (raw agreement, Cohen's κ) are not computed by the tool;
- span annotation is not implemented.

**Operational notes (2026-09-15):**
- An annotation server was still listening on `127.0.0.1:8765` when this file was written (Pass A, PID
  18948).
- While it runs, `uv run …` can fail to reinstall `opengrad-annotate.exe` ("Access is denied"). Call
  `.venv/Scripts/python.exe -m …` directly instead.
- Never point tests or ad-hoc scripts at `.annotation/`. Some CLI paths write to the store they open.
  Inspect a read-only copy (`sqlite3.connect("file:…?mode=ro", uri=True)` plus `backup`).

---

## 9. P-DET-COVERAGE-v1 (draft, not drawn)

**Why a second population instead of changing P-DET-v1.**
- P-DET-v1 is frozen and may not be edited or resampled.
- It covers CLARIFY- and UNSUPPORTED-like prose from When2Call.
- P-DET-COVERAGE-v1 is meant to cover the **missing DIRECT and tool-boundary cases**, drawn from the
  representation the classifier will actually receive (canonical-v3 / normalization-v3).
- The two populations are always reported separately.
- **P-DET-COVERAGE-v1 is a constructed coverage set, not a natural-prevalence benchmark.** Its class mix is
  a property of the design, not of any corpus.

**Draft:** [`docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md`](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md).
Status DRAFT, not adopted. Protocol id `pdet-coverage-002-v1`, seed `opengrad-pdet-coverage-002-v1`, output
directory (at draw time) `reports/pdet-coverage/`.

Two layers:
- **Layer A:** structural CALL routing. Checked exhaustively by code, plus a 30-item human spot-check:
  Glaive 15, ToolACE 10, xLAM 5.
- **Layer B:** prose classification, on tool-free single exchanges.

**Layer-B strata** (sampling strata, never labels; first match in this order; predicates frozen in
`scripts/audit_pdet_coverage_supply.py`):

| Stratum | Observable predicate on the assistant response | Quota |
|---|---|---:|
| **X** | textual call syntax in the text (`{"name":`, `"arguments":`, `<functioncall>`, `<tool_call>`, `<TOOLCALL>`) | ≤ 60 |
| **M** | mentions an offered tool by name (length ≥ 4) or matches invocation talk, without calling it | 60 |
| **R** | refusal or hedge (`detect_refusal` or a P-DET-v1 hedge cue) | 60 |
| **Q** | question (`?` or a P-DET-v1 question cue) | 60 |
| **P1** | plain prose, tools offered | 60 |
| **P2** | plain prose, no tools offered | 30 |

- No backfill between strata. An undersupplied stratum takes what it has, and its shortage is reported
  before annotation.
- Measured supply is in §14.

---

## 10. B-1 and B-2 (both resolved)

### B-1: the canonical-v3 source-and-adapter representation

**Source manifest:** `configs/releases/toolpolicy_canonical_v3_sources.yaml` (sha256 over LF bytes
`a38f95d71d96967ffc7bc91810b6a3ab09f0223a958fb41bde98f3a53861e215`).

| Source | Adapter key → function | Schema translation | Accepted in normalization-v3 |
|---|---|---|---:|
| xLAM | `xlam` → `adapt_xlam` | no (own `xlam_types` parser) | 57,342 |
| Glaive | `glaive_v2` → `adapt_glaive_v2` | no | 98,339 |
| ToolACE | `toolace` → `adapt_toolace` | no | 11,051 |
| When2Call | `when2call` → `adapt_when2call` | **yes** | 14,701 |

- **Excluded:**
  - **LoopTool:** not in canonical-v2-final.
  - **BUTTON:** not in canonical-v2-final (gated upstream), and all of its records end in post-tool prose.
- **Artifact:** `data/processed/normalization-v3/`, **git-ignored and local only**.
- **Fingerprint:** `2bd384929a5b870bc3836729c808362c60d0d98717f086587e12af6608104aaa`, verified 2026-09-15.
  `--verify` returns `PASS`, and the tracked copy
  `reports/normalization-v3/manifests/normalization-v3.manifest.json` has the same fingerprint.
- **Determinism:** two independent full builds produced 29 byte-identical files
  (`reports/normalization-v3/normalization-v3.determinism.json`).
- **normalization-v3 is pre-classifier.** It holds:
  - normalized messages and tools;
  - structured calls;
  - a `metadata.structure` block of structural facts;
  - full provenance.

  It holds **no behaviour label**. The adapter's message-shape `ANSWER` default is dropped.

**Rebuild and verify.** About 11 minutes with 12 worker processes. The builder refuses a non-empty output
directory, so rebuild into a new one:

```bash
.venv/Scripts/python.exe -m opengrad.data.normalization_v3 --build --output <new-empty-dir>
.venv/Scripts/python.exe -m opengrad.data.normalization_v3 --verify            # checks data/processed/normalization-v3
.venv/Scripts/python.exe -m opengrad.data.normalization_v3 --verify --output <dir>
```

The raw inputs are in `.cache/normalization/raw/` (git-ignored) and are pinned by SHA-256 in the manifest.
`--verify` also fails if the source manifest, or any module listed in `CODE_MODULES` in
`src/opengrad/data/normalization_v3.py`, has changed since the build. **Any adapter or version change
therefore requires a rebuild and a new fingerprint.**

Other artifacts:
- **Audit scripts** (read-only; counts only):
  - `scripts/audit_normalization_v3_structure.py`;
  - `scripts/audit_pdet_v1_representation.py`;
  - `scripts/audit_pdet_coverage_supply_v3.py`.
- **Tests:** `tests/data/test_normalization_v3.py`.

### B-2: the classifier input contract `prose-decision-input-v1`

Specification: [32](32-CLASSIFIER-INPUT-CONTRACT.md). Code: `src/opengrad/data/classifier_input.py`.
Tests: `tests/data/test_classifier_input.py`.

**What the prose classifier sees (exactly four fields):**
- the user message;
- the final assistant response;
- the available (presented) tools, as normalized;
- `structured_call_present`.

**Not a feature:** the system message. Its text is a per-source template and would reveal the source.

**Must never be used as features:**
- source dataset identity, including the id, the `raw_record_hash` and the adapter;
- the hidden sampling stratum;
- P-DET labels;
- model-generated reference labels;
- held-out or evaluation labels;
- future evaluation outcomes.

Identity lives in `ClassifierProvenance`, which is serialized separately. The feature serialization is
deterministic, and a test pins its hash.

**Eligibility exclusions**, in precedence order. `eligibility(record, heldout)` is structural and never
reads what a response says.

| Reason | Meaning |
|---|---|
| `EVALUATION_ONLY_OR_HELDOUT` | the record is evaluation-only, or matches a held-out id or a held-out prompt (When2Call MCQ and LLM-judge questions, the behavioral-heldout-v2 partition and quarantine; pinned by SHA-256) |
| `MALFORMED_OR_UNRENDERABLE` | parse failure, empty or missing turns, or any trajectory issue from the gate the renderer applies |
| `STRUCTURAL_CALL` | a structured call is present, so the record is routed to **Layer A**, never to the prose classifier |
| `POST_TOOL_RESULT_RESPONSE` | a tool result is present |
| `MULTI_TURN_UNSUPPORTED` | more than one user turn, or a tool-free record that is not exactly user → assistant. **Never flattened to its final message.** |

A record that is not a normalization-v3 row, or that already carries a behaviour label, is refused outright
with `ContractViolation`.

---

## 11. Glaive findings

- **normalization-v1 left Glaive calls as text.** The local, historical normalization-v1 was built with
  `adapt_glaive` (v1), which needs a `</functioncall>` terminator that this upstream revision never has.
  So it left every call as textual `<functioncall> {…}` inside assistant content, and 0 of 99,794 records
  carry a structured call.
- **`adapt_glaive_v2` converts valid calls into structured `tool_calls` and removes the marker.**
  - On all 98,339 accepted normalization-v3 rows, the raw `<functioncall>` count equals the structured
    call count.
  - No `<functioncall>` marker remains in any assistant content.
  - A call that cannot be parsed makes the record **rejected**: 1,094 malformed calls, 18 argument payloads
    that are not objects, 1 unparseable argument JSON. It is never passed through as prose.
- **canonical-v3 intentionally uses the v2 adapter.**
- **The reproduction settles which adapter Study 001 used.** Rebuilding Glaive with `glaive_v2` reproduces
  canonical-v2-final's Glaive count (98,339) and every rejection reason, count for count. That includes 866
  `malformed … at offset 2`, a message only the v2 parser emits. So canonical-v2-final, and Study 001's M0,
  effectively used the v2 representation, even though the historical metadata says `adapter_version 1.0.2`.
- **Consequence for P-DET-COVERAGE:** Glaive supplies no textual-call (X) candidates.

---

## 12. ⚠ ToolACE defect — THE NEXT ISSUE

**Finding (ESTABLISHED; `reports/normalization-v3/normalization-v3.structural-audit.json`,
`structured_call_turns`):**

`adapt_toolace` (`src/opengrad/data/adapters.py`) parses bracket call syntax (`[Name(arg=…)]`) into
structured `tool_calls`. However, it removes the text from `content` only when the text contains
`[Function`. As a result, **9,785 of 9,786 ToolACE call turns keep the call twice**:
- once as a structured call;
- once as the original bracket text in `content`.

The same audit's marker-regex count is 9,612. That regex needs an identifier character right after `[`, so
it undercounts; 9,785 is the count of call turns whose content is non-empty and starts with `[`. The source
manifest records the defect. It was **not repaired**.

Why it matters:
- **Not affected: the current prose classifier input.** Records with a structured call are routed to Layer
  A and never reach it.
- **Affected:**
  - the **Layer-A display**: annotators of ToolACE spot-check items would see both forms;
  - **future model-visible training targets**: a rendered target would contain the call twice.
    canonical-v2-final used the same adapter, so this is also a property of Study 001's training data. What
    it does to the model is **not measured**.
- **So it should be repaired before P-DET-COVERAGE-v1 is sampled, and before canonical-v3 training data is
  finalized.**

**Intended next engineering action (NOT DONE):**
1. Repair ToolACE **at the adapter boundary**, as a new adapter (for example `adapt_toolace_v2` under a new
   `ADAPTERS` key). Keep `adapt_toolace` untouched so v1/v2 stay reproducible, following the
   `adapt_glaive_v2` precedent.
2. Strip the redundant textual call **only after it has parsed successfully** into structured calls.
3. **Preserve any legitimate surrounding prose.**
4. **Quarantine malformed cases** with an explicit reason instead of silently stripping them.
5. Bump the adapter version **through `src/opengrad/data/versions.py`**, and update the source manifest's
   ToolACE entry (adapter key, function and row label).
6. Rebuild normalization-v3 into a new directory and record the **new fingerprint**.
7. Rerun the structural audit, the P-DET-v1 representation audit and the supply audit.

---

## 13. P-DET-v1 against the v3 representation (audit, not migration)

From `reports/normalization-v3/pdet-v1-representation-audit.json`, with each item matched by
`raw_record_hash`:
- **575 of 581 items remain structurally compatible.** Each is present, has a byte-identical user message
  and response and an identical upstream id, and is eligible under the contract. Tools are identical for
  101 items and differ only by the schema translation for 474.
- **6 items are not in normalization-v3.** Schema translation quarantines their When2Call records:

| Tool item # | Component | Reason | `pdet_id` |
|---:|---|---|---|
| 13 | challenge | `SRC_SCHEMA_MULTI_TYPE_UNION` | `when2call-sft:088d349e8d3507c0780d5ac559e03eea0d5c6bc88d2c2b30d89d3af94839cf9d` |
| 39 | challenge | `SRC_SCHEMA_MULTI_TYPE_UNION` | `when2call-sft:12889fc214119593c4b33d8027ea43c09a199567a077c60344fcc6e99544ab37` |
| 52 | challenge | `SRC_SCHEMA_MULTI_TYPE_UNION` | `when2call-sft:18ecd285a84a73cdb3771c770c9fbe7a3cbc88c9d1a8972dd45e29b42b865b59` |
| 347 | prevalence | `SRC_SCHEMA_MULTI_TYPE_UNION` | `when2call-sft:990289e0e0c6bb19229fa5aa168d46a44bf7c51ef51e30e402900dd7909a4cad` |
| 349 | challenge | `SRC_SCHEMA_UNREPRESENTABLE_TYPE` | `when2call-sft:99319dc22bfe7b62a5002e75a7537a6922e75610d96753f01e31a3eb7c93a30c` |
| 371 | prevalence | `SRC_SCHEMA_MULTI_TYPE_UNION` | `when2call-sft:a1a6fdfc0e2e4670363cb6951418c9c1f79630d373cf77c0890182eac77d4cc8` |

None of the six is an exposed worked example.

**P-DET-v1 stays frozen. It is not migrated or rewritten.** This is a limitation to report, 6 of 581
(1.0%), not a reason to alter the old population. [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) §7
recommends a reporting rule: report on all items as 22 specifies, plus on the 575, with the 6 marked
`NOT_IN_CLASSIFIER_REPRESENTATION`. Adopting that rule is an amendment for the study owner and **has not
been adopted**.

---

## 14. P-DET-COVERAGE supply (latest measurement)

From `reports/pdet-coverage/pdet-coverage-v1.supply-v3.json`, over normalization-v3 fingerprint
`2bd38492…`. The pool is every record eligible under `prose-decision-input-v1`: Glaive 14,303 and
ToolACE 1,972.

**These are candidate supply counts, not gold labels, and nothing was sampled.** The draft's §9 draw-time
exclusions (QAD recovery set, P-DET-v1 overlap, sentinel prompts) are not applied yet.

| Stratum | Glaive records | Glaive distinct prompts | Glaive distinct skeletons | ToolACE records | ToolACE distinct prompts | ToolACE distinct skeletons | Quota |
|---|---:|---:|---:|---:|---:|---:|---:|
| X | 0 | 0 | 0 | 6 | 6 | 6 | ≤ 60 |
| M | 19 | 14 | 19 | 706 | 706 | 706 | 60 |
| R | 14,050 | 210 | 10,675 | 228 | 227 | 227 | 60 |
| Q | 105 | 93 | 102 | 168 | 166 | 166 | 60 |
| P1 | 64 | 14 | 58 | 371 | 371 | 365 | 60 |
| P2 | 65 | 65 | 65 | 493 | 491 | 491 | 30 |

- **X (textual call) is under-supplied**: 6 against 60, a shortage of 54. Textual-CALL metrics are expected
  to be `NOT_EVALUABLE`.
- **If the ToolACE repair later makes X disappear entirely, that is acceptable.** Valid calls should be
  routed structurally, not appear as text.
- **Distinct prompts, not records, are the binding supply.** Glaive's R records come from only 210 distinct
  user prompts, and its P1 records from 14.
- The earlier proxy numbers in `reports/pdet-coverage/pdet-coverage-v1.supply-proxy.json` came from
  normalization-v1 and are superseded for sampling purposes.

---

## 15. What must happen next (in order)

1. Read this file, then [README](README.md), [21](21-C1-IMPLEMENTATION-STATUS.md), [22](22-PDET-PROTOCOL.md),
   [28](28-PDET-MODEL-LABEL-AMENDMENT.md), [29](29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md),
   [30](30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md), [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) and
   [32](32-CLASSIFIER-INPUT-CONTRACT.md).
2. Verify the repository state:
   - the branch is `study-002/canonical-v3-b1-b2` at `8e78837`;
   - `pdet --verify` passes;
   - `normalization_v3 --verify` passes.
3. Do not modify frozen v1/v2 artifacts or P-DET-v1.
4. Repair the ToolACE duplicate textual-call representation at the adapter boundary (§12).
5. Bump the adapter version through `src/opengrad/data/versions.py`.
6. Rebuild normalization-v3 into a new directory and record the new fingerprint, in 31 and in 30 §12.
7. Rerun the structural audits (`scripts/audit_normalization_v3_structure.py`,
   `scripts/audit_pdet_v1_representation.py`).
8. Rerun the P-DET-COVERAGE supply analysis (`scripts/audit_pdet_coverage_supply_v3.py`).
9. Confirm that `prose-decision-input-v1` still applies unchanged. Any change to its fields, exclusions or
   serialization means a new contract version.
10. Only then implement and finalize the deterministic P-DET-COVERAGE draw code (planned as
    `src/opengrad/verification/pdet_coverage.py`, 30 §8), and bring the draft to the study owner for
    adoption.
11. Draw and freeze P-DET-COVERAGE-v1 exactly as preregistered.
12. Annotate that population with humans, using the existing annotation app, in new tasks
    `pdet-coverage-v1` and `pdet-coverage-v1-routing` (30 §10). Gold is human only.
13. Only after the validation populations are ready, implement and test the behaviour classifier.
14. Do not authorize C1 balancing until the classifier meets the preregistered validation requirements (22
    §6, applied per population as in 30 §11).
15. No GPU training until the Study 002 preregistration and the pre-GPU gates pass
    ([16](16-GPU-READINESS-GATE.md)).

Separate study-owner decisions, not engineering steps:
- whether to re-read the 2 UNKNOWN items and freeze P-DET-v1 gold;
- whether C1's When2Call membership follows v2-final (6,505 records) or v3 (14,701).

---

## 16. Do not

- rewrite Study 001, its results or its conclusions;
- modify canonical-v1 or canonical-v2 (manifests, shards, hashes, release files);
- modify, resample or migrate the P-DET-v1 population (`reports/pdet/pdet-v1.population.jsonl`,
  `pdet.py`, 22, 23);
- freeze any P-DET composite as human gold without the study owner's decision and the 22 §4 re-read, and
  never call a composite containing model-sourced labels human gold. Until 08:09 UTC on 2026-09-15 it was
  model-heavy (570 model labels, 11 human); check which session each label comes from;
- call `model-a` labels human validation, or report human–model agreement as inter-annotator agreement;
- use normalization-v1 as a stand-in for canonical-v3;
- implement classifier behaviour before the validation populations are constructed;
- silently flatten a multi-turn conversation to its final message;
- treat a post-tool-result summary as DIRECT;
- treat structured calls as a prose-classification problem;
- sample P-DET-COVERAGE before the ToolACE representation is resolved;
- train anything yet;
- point tests at `.annotation/` or the running annotation server;
- describe hashes as signatures, or annotator ids as authenticated.

---

## 17. Repository map

| Path | Purpose |
|---|---|
| `docs/research/study-002/README.md` | Study 002 overview and current state |
| `docs/research/study-002/03-PREREGISTRATION.md` | preregistration and amendment log |
| `docs/research/study-002/04-ARM-MATRIX.md` | arms (C0, R1, …, C1) and what each holds fixed |
| `docs/research/study-002/16-GPU-READINESS-GATE.md` | pre-GPU gate |
| `docs/research/study-002/21-C1-IMPLEMENTATION-STATUS.md` | provenance repair and schema normalization, with a status note |
| `docs/research/study-002/22-PDET-PROTOCOL.md` | P-DET protocol and classifier acceptance thresholds (§6) |
| `docs/research/study-002/23-PDET-ANNOTATION-INSTRUMENT.md` | annotation instrument |
| `docs/research/study-002/25-PDET-IMPLEMENTATION-ADDENDUM.md` | how the tool implements the instrument |
| `docs/research/study-002/26-PDET-ANNOTATOR-CHECKLIST.md` | annotator checklist |
| `docs/research/study-002/27-PDET-EXPOSED-WORKED-EXAMPLES.md` | the three metric-excluded items |
| `docs/research/study-002/28-PDET-MODEL-LABEL-AMENDMENT.md` | model reference labels (`study_002_prereg_v2`) |
| `docs/research/study-002/29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md` | rationale optional (`study_002_prereg_v3`) |
| `docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md` | P-DET-COVERAGE-v1 draft |
| `docs/research/study-002/31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md` | B-1 results and structural audit |
| `docs/research/study-002/32-CLASSIFIER-INPUT-CONTRACT.md` | B-2 contract |
| `configs/releases/toolpolicy_canonical_v3_sources.yaml` | canonical-v3 source-and-adapter manifest |
| `src/opengrad/data/versions.py` | authoritative versions and the two provenance checks |
| `src/opengrad/data/source_schema.py` | source-scoped schema translation |
| `src/opengrad/data/adapters.py` | source adapters (`ADAPTERS`), including `_base` |
| `src/opengrad/data/normalization_v3.py` | normalization-v3 builder and verifier |
| `src/opengrad/data/classifier_input.py` | `prose-decision-input-v1`: eligibility and input object |
| `src/opengrad/verification/pdet.py` | P-DET-v1 builder and verifier (frozen) |
| `reports/pdet/` | P-DET-v1 population, manifest, review queue, model-label provenance, WIP export |
| `reports/normalization-v3/` | v3 manifest copies, determinism record, structural audit, P-DET-v1 representation audit |
| `reports/pdet-coverage/` | supply reports (`…supply-v3.json` current; `…supply-proxy.json` superseded) |
| `docs/ANNOTATION_TOOL.md` | annotation app guide |
| `configs/annotation/pdet-v1.yaml` | P-DET-v1 annotation task |
| `src/opengrad/annotation/`, `integrations/annotate-ui/` | annotation app code and UI |
| `.annotation/pdet-v1.sqlite3` | annotation working state (git-ignored, local) |
| `data/processed/normalization-v3/` | the artifact (git-ignored, local) |
| `.cache/normalization/raw/` | pinned raw source files (git-ignored, local) |
| `reports/ERRATA.md` | errata, including §13–§14 on P-DET |

**Rebuild normalization-v3:** `.venv/Scripts/python.exe -m opengrad.data.normalization_v3 --build --output <new-empty-dir>`.

---

## 18. Evidence status

| Claim | Status | Evidence |
|---|---|---|
| Study 001's gate lacked a DIRECT gold population | **ESTABLISHED** | When2Call MCQ answer distribution ([README](README.md)) |
| M0's training corpus had only CALL and ANSWER decision labels | **ESTABLISHED** | `sft_refusal_supervision_audit_canonical_v2.json` |
| 18,114 refusal-shaped targets are labelled ANSWER | **ESTABLISHED as a detector output**; detector precision NOT MEASURED | same file; [03](03-PREREGISTRATION.md) |
| The canonical training-label collapse caused the regression | **HYPOTHESIS** | [03](03-PREREGISTRATION.md), [04](04-ARM-MATRIX.md) |
| The adapter-version inconsistency existed (record 1.0.0 vs manifest 1.0.2) | **ESTABLISHED** | [21](21-C1-IMPLEMENTATION-STATUS.md) phase 1 |
| canonical-v2-final used `glaive_v2` | **ESTABLISHED (by reproduction)** | [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) §1 |
| The normalization-v3 representation is reproducible | **ESTABLISHED** | `normalization-v3.determinism.json`; `--verify` PASS |
| ToolACE duplicate textual-call defect | **ESTABLISHED** | structural audit, `structured_call_turns` |
| Fixing ToolACE will improve model behaviour | **UNKNOWN** | not tested |
| Four-way behaviour balancing will prevent the GSM8K collapse | **UNTESTED** | Study 002 arms not run |
| Classifier accuracy | **NOT YET MEASURED** | no classifier exists |
| P-DET-v1 human labels | **PROVISIONAL** (complete, single annotator, not frozen) | `.annotation/pdet-v1.sqlite3` |
| P-DET-v1 alone can authorise C1 | **NO** (no CALL or DIRECT gold) | 22 §5–§6; `reports/ERRATA.md` §14 |
| 575 of 581 P-DET-v1 items are representable in v3 | **ESTABLISHED** | `pdet-v1-representation-audit.json` |
| P-DET-COVERAGE strata can be supplied (except X) | **ESTABLISHED as supply**; gold yield UNKNOWN | `pdet-coverage-v1.supply-v3.json` |

---

## 19. Working tree and Git state (2026-09-15, at the time of writing)

**Branch and commit:**
- branch `study-002/canonical-v3-b1-b2` at commit `8e78837`;
- parent commit `62a5721` = `master` = `origin/master`;
- the branch is **local, not pushed**.

**Tracked files modified and not committed.** These pre-date B-1/B-2 and come from the earlier P-DET and
annotation work; they were left as they were:
- `.gitattributes`;
- `.gitignore`;
- `ROADMAP.md`;
- `docs/research/STUDIES.md`;
- `pyproject.toml`;
- `reports/ERRATA.md`;
- `uv.lock`.

**Untracked, relevant to this work, not committed:**
- `docs/research/study-002/`: every document except 21, 30, 31 and 32, which are committed. That includes
  `README.md` and 01–20, 22–29, plus **this `HANDOFF.md`**.
- `docs/ANNOTATION_TOOL.md`, `configs/annotation/`, `src/opengrad/annotation/`, `integrations/annotate-ui/`,
  `tests/annotation/`.
- `src/opengrad/verification/pdet.py`, `tests/verification/`.
- `reports/pdet/`: the frozen P-DET-v1 population and manifest, the review queue, the model-label
  provenance, the stale WIP export.
- `scripts/archive_pdet_model_batches.py`.

**Committed in `8e78837`:**
- the source manifest;
- `versions.py`, `source_schema.py`, `normalization_v3.py` and `classifier_input.py`;
- their tests;
- the four audit scripts;
- `reports/normalization-v3/` and `reports/pdet-coverage/`;
- docs 21, 30, 31 and 32.

**Local only, intentionally excluded from Git (`.gitignore`):**
- `data/processed/normalization-v3/`: the artifact. Rebuild per §10.
- `.annotation/`: annotation working state, **not committed**. It holds 581 human and 570 model labels and
  is the only copy of the post-08:09 labels. Back it up before any risky operation.
- `.cache/normalization/raw/`: the pinned raw sources.
- `reports/pdet/provenance/*/local/`: transcripts that contain an e-mail address.

Whether to commit the untracked Study 002 documents and `reports/pdet/` is **for the study owner to
decide**. Nothing here commits, stashes or cleans them.

---

## 20. NEXT SESSION START HERE

**CURRENT NEXT ENGINEERING TASK: ToolACE structured-call/text duplication repair before P-DET-COVERAGE-v1 sampling.**

Checklist:

1. `git status` and `git log --oneline -3`. Expect branch `study-002/canonical-v3-b1-b2` at `8e78837`, with
   the dirty tree listed in §19.
2. `.venv/Scripts/python.exe -m opengrad.verification.pdet --verify`. Expect `PASS` and population
   `6ab92087…e781b`.
3. `.venv/Scripts/python.exe -m opengrad.data.normalization_v3 --verify`. Expect `PASS` and fingerprint
   `2bd38492…aaa`.
4. Read §12 of this file, [31](31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md) §5, and `adapt_toolace` in
   `src/opengrad/data/adapters.py`.
5. Read [32](32-CLASSIFIER-INPUT-CONTRACT.md) before touching anything that feeds the classifier input.
6. Leave `.annotation/` and any running annotation server alone.
7. Start the ToolACE repair as a new, versioned adapter (§12, steps 1–7). Do not draw, classify or train.
