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

- **Validation** lives in `src/opengrad/registry/validate.py`. That covers schema validity, revision pins,
  `derived_from` citations and the publication revision chain (exactly one current Hub revision per
  repository).
- **The validator must prove it ran.** It once exited 0 having checked nothing. A new check registers itself in
  the execution census (`src/opengrad/verification/accounting.py`).
- **Changing a registry entry** means updating its revision and citations together. Never leave a revision
  string such as `TO_BE_RECORDED` in a record that claims verification.
- `opengrad-preflight` (`src/opengrad/registry/preflight.py` via `opengrad.cli:preflight_cli`) is a repository-level
  check: git, registries, required layout.

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
