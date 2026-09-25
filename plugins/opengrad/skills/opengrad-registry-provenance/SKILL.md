---
name: opengrad-registry-provenance
description: How OpenGrad's registries, provenance claims and written records stay trustworthy — registry YAML and JSON schemas validated by opengrad-validate, the immutable-anchor rule for every verified claim, authoritative versions, generated documentation views, reports and hash-pinned artifacts, errata and the incident log, and repository-level checks (opengrad-preflight, status, doctor). This skill should be used for any change under registry/, src/opengrad/registry/, src/opengrad/reporting/, reports/ or docs that state repository status, and whenever a claim is marked verified or a number is written into a report.
---

# OpenGrad registries and provenance

## Registries

`registry/` holds the declarative identity and version contracts:
- models, datasets, benchmarks, runtimes, runtime components, hardware;
- tool behaviours, the error taxonomy, training recipes, workload profiles;
- JSON schemas for experiments, evaluation manifests, residual profiles, provenance, interop, HF release
  manifests and GPU preflight.

```bash
.venv/Scripts/opengrad-validate.exe      # same checks as `opengrad validate --json`; CI runs it
```

- **Validation** lives in `src/opengrad/registry/validate.py` (entry point, checks, `result_from`), with the
  `validate_*` validators in `validators.py`. That covers schema validity, revision pins,
  `derived_from` citations and the publication revision chain (exactly one current Hub revision per
  repository).
- **The validator must prove it ran.** It once exited 0 having checked nothing. A new check registers itself in
  the execution census (`src/opengrad/verification/accounting.py`).
- **Changing a registry entry** means updating its revision and citations together. Never leave a revision
  string such as `TO_BE_RECORDED` in a record that claims verification.
- `opengrad-preflight` (`src/opengrad/registry/preflight.py` via `opengrad.cli:preflight_cli`) is a repository-level
  check: git, registries, required layout.

## Dataset records (`registry/datasets.yaml`, schema version 2)

- **The schema is the definition.** Every record conforms to `registry/dataset_record.schema.json`
  (JSON Schema 2020-12). `validate_structure` enforces it for any file declaring `schema_version: 2`, and a
  version-2 file that drops `record_schema` is an error, not a skip. `tests/registry/test_dataset_record_schema.py`
  pins the committed file at version 2 and proves each rule rejects a record that breaks it.
- **The procedure is `docs/datasets/ADDING_A_SOURCE.md`.** Only *adopted* sources and OpenGrad's derived corpora
  belong in this file: the training firewall (`materialize._training_split_allowlist`, also called by the frozen
  `normalization_v3` builder) reads `id`, `intended_stages` and `allowed_splits` from it. Keep those three names;
  the same test pins the allowlist literal.
- **One field per fact.** Upstream revision is `source_revision`; OpenGrad's processed identity is
  `processed_dataset_hash.value` (a digest, or null with the git-ignored `manifest` path); upstream file digests are
  `distribution[].sha256`, read from the Hub's LFS record at the pinned revision or hashed from bytes proven
  identical to it. `checksum`, `exact_revision`, `original_sample_count` and `retained_sample_count` were retired
  on 2026-09-24 and are rejected.
- **Enums are closed.** Redistribution, overlap and contamination fields take only the schema's values; an overlap
  audit is named by its artifact in `overlap_evidence`, never by a sentence in the enum. `validate_references`
  checks `papers` ids, evidence paths and that every distribution URL carries the pinned revision.
- **Croissant.** Each schema property names its Croissant 1.1 / Croissant RAI 1.0 counterpart in `x-croissant`;
  `responsible_use` holds the RAI fields and needs `evidence`.
  - `opengrad croissant [<id>] [--out DIR] [--strict]` (`src/opengrad/registry/croissant.py`) exports records as
    Croissant JSON-LD, reading that mapping from the schema. It uses the standard context and never invents a
    required property: a missing one is reported, and `--strict` exits 1.
  - `tests/registry/test_croissant.py` pins which records are incomplete: the derived corpora have no
    `distribution`.
  - mlcroissant 1.1.0 reports 0 errors on all records. Its only warnings are the commit SHA as `version` and a
    missing `citeAs` where no paper exists.
- **Screen before adopting.** Every candidate considered for a purpose, including the rejected ones, goes in
  `registry/source_screening.yaml` (`registry/source_screening.schema.json`). The `screening` gate of
  `opengrad-validate` checks it:
  - every candidate has a verdict per criterion;
  - an EXCLUDED candidate names FAIL criteria;
  - a SHORTLISTED candidate fails none;
  - evidence resolves;
  - an adopted candidate is also registered in `datasets.yaml`.

  `owner_decision` stays PENDING until the study owner decides. Adoption names the registered records and the
  amendment that records it (`study-002-answer-heldout` → 41, `study_002_prereg_v9`), and
  `tests/registry/test_source_screening.py` pins it. A second screening, `study-002-punans` (Study 002's unanswerable set),
  has its counts written by `scripts/audit_punans_supply.py`, which downloads each read source at a pinned revision
  and refuses a file whose sha256 differs. An evaluation-only record has `intended_stages: [evaluation]`
  and `allowed_splits: []`, so the training firewall never admits it.
- **Views are generated.** `scripts/reporting/generate_source_views.py` writes `docs/datasets/SOURCE_REGISTRY.md`
  and `docs/datasets/SOURCE_SCREENING.md`, including the screening flow counts; the test requires them current.
  Two hand-written documents still carry revision tables: the dated redistribution audit and
  `docs/data/normalization-sources.md`. `tests/registry/test_source_docs_agree.py` fails when either disagrees
  with the registry.
  Numbers in a screening report come from its artifacts. The BFCL figures come from
  `scripts/audit_bfcl_answer_supply.py`, and the same test checks the report against them.

## Provenance: every verified claim needs an immutable anchor

From `docs/PROVENANCE.md`, implemented in `src/opengrad/registry/provenance.py`:
- **A mutable or local source** (a repository path, a local file) needs a content digest: `source_sha256`. A path
  alone pins nothing.
- **An externally immutable source** (a Hub revision, a git commit) needs the immutable revision.
- `verified: true` without the matching anchor is a defect, not a formality.

Related rules:
- **Versions** of canonical artifacts come only from `src/opengrad/data/versions.py`. Record-level and
  manifest-level versions must agree.
- **Hash-pinned files** (manifests with `.sha256` sidecars, frozen reports) are corrected by a new artifact or an
  erratum, never edited in place.
- **Line endings matter to hashes.** Hash LF-normalized text where the code says so (`lf_sha256`), and never
  rewrite a pinned file's bytes. A new pin hashes the **committed blob**
  (`read_evidence_bytes` in `src/opengrad/registry/provenance.py`), never a Windows working tree.
- **Pins are verified in CI, not asserted.** `tests/results/test_preserved_state.py` runs
  `scripts/preserve_h200_state.py --verify` against `results/benchmarks/h200/PRESERVED_STATE_v2.json` and checks
  every tracked `.sha256` sidecar against its committed file. A new pinned set gets the same kind of test in the
  commit that pins it.
- **One sha256 helper.** Plain hashing (raw bytes, UTF-8 text, a file's exact bytes) uses `sha256_bytes`,
  `sha256_text` or `sha256_file` from `src/opengrad/hashing.py`; `tests/repo/test_hash_helpers.py` fails on a new
  local wrapper. The wrappers it allows are standalone and Modal scripts, and files whose own source hash a frozen
  artifact records (editing those changes the recorded hash). A hash of something computed (canonical JSON,
  normalised text, a seeded key) keeps its own named function, built on `sha256_text`.

## Written records

- **Generated views are regenerated, not edited:**
  - `docs/EXPERIMENT_STATUS.md` from `scripts/reporting/generate_experiment_status.py` (`--check` in tests);
  - `results/registry.jsonl` from `opengrad results rebuild-registry` — but the committed file is also pinned by
    `reports/data/m0-final-freeze.json`, so a rebuild that changes its bytes needs an erratum (`reports/ERRATA.md` §21);
  - `results/final_campaign_verdict.json` from `scripts/build_final_verdict.py`.
- **`src/opengrad/reporting/generate.py`** writes synthetic test reports only. Its output is labelled as not a
  research result, and must never be presented as one.
- **`reports/`** holds written analyses. Numbers in them come from the artifacts that produced them (G14), and a
  change of status updates every surface in the same commit (G15–G17). `docs/DOCUMENTATION_STATE_AUDIT.md`
  records the last documentation/state reconciliation.
- **Corrections:**
  - a wrong number in a frozen study goes to `reports/ERRATA.md`;
  - an operational mistake that changes what can be claimed goes to `docs/INCIDENT_LOG.md`, with evidence under
    `reports/incidents/`.

  Entries are appended and superseded, never rewritten.

## Repository state commands

- `opengrad status --json`: authoritative repository and experiment state.
- `opengrad doctor --json`: environment and tooling diagnosis.
- `opengrad env capture`: the environment record embedded in runs (`src/opengrad/env_capture.py`).

## Keeping this skill current

Update this skill in the same commit when a registry file or schema is added, a generated view gains a
generator, or the provenance rule changes.
