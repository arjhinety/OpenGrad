---
name: opengrad-data-pipeline
description: How to change OpenGrad's data pipeline correctly — source adapters and adapter versioning, the canonical IR and supervision contracts, the normalization-v3 builder (build, verify, determinism, audits, re-anchoring docs), canonical releases and mixtures, rendering and loss masks, and the opengrad-data / opengrad-data-audit tooling. This skill should be used for any change under src/opengrad/data/, configs/releases/, configs/data/ or reports/normalization-v3/, and before rebuilding or citing a corpus fingerprint.
---

# OpenGrad data pipeline

The flow is upstream dataset → source adapter → canonical IR → validation and quarantine → dedup →
contamination boundary → exact-model renderer and loss mask → training or evaluation artifact
(`docs/architecture/repository.md`). Canonical records stay model-independent. Rendering is a separate,
exact-checkpoint step (`src/opengrad/data/renderers.py`, `docs/models/renderer-matrix.md`).

## Invariants

- **Supervision is declared by the adapter, never inferred by a validator or at training time.** Every
  record carries a supervision contract, such as `COMPLETE_TRAJECTORY` or `CALL_PREDICTION`. A new shape
  gets a new contract. An existing contract is never widened, and no source-name conditional is added to a
  validator.
- **An adapter's structural reading is never presented as upstream's.** `upstream_declared` is only for
  what upstream documents (xLAM's card). A per-record rule the adapter applies from measured corpus
  structure uses `source_adapter`, with a note naming the evidence artifact (`toolace_v3`:
  `toolace_call_prediction_shape`, evidence `reports/normalization-v3/toolace-call-final-shape.json`).
  Admit only the validated shape; subgroups outside it keep the stricter contract.
- **Malformed input is quarantined with a reason code, never repaired into plausibility.** Nothing may
  fabricate a tool result.
- **Versions come from one place:** `src/opengrad/data/versions.py`. Never write a version string literal
  anywhere else. `check_artifact_matches_authoritative_versions` enforces this for new artifacts.
- **Frozen corpora (canonical-v1/v2 releases) are never rebuilt in place.**

## Changing an adapter's behaviour

Follow the `glaive_v2` / `toolace_v2` / `toolace_v3` precedent in `src/opengrad/data/adapters.py`
(`toolace_v3` wraps v2 and changes only the declared supervision):
1. Keep the old adapter byte-for-byte, so older corpora stay reproducible.
2. Add `adapt_<source>_v2` and register it in `ADAPTERS` under a new key, with a new `row_label` and
   `source_format`.
3. Bump `ADAPTER_VERSION` in `versions.py` with a comment that states the measured defect and why records
   before and after are not interchangeable.
4. Point the source manifest (`configs/releases/toolpolicy_canonical_v3_sources.yaml`) at the new key,
   function and row label, and record the old one under `rejected_alternative` with the measured reason.
5. Add tests for the repaired case, the preserved case (e.g. surrounding prose kept), and the
   malformed-input case (still quarantined), plus a test that the manifest agrees with the code.

## normalization-v3 (pre-classifier artifact)

```bash
.venv/Scripts/python.exe -m opengrad.data.normalization_v3 --build --output <new empty dir> --workers 8
.venv/Scripts/python.exe -m opengrad.data.normalization_v3 --verify
```

- The build refuses a non-empty output directory. Build into a new directory, then swap it into
  `data/processed/normalization-v3/` (git-ignored). A full build takes about 15 minutes.
- **The fingerprint** covers content hashes, versions and the source manifest's LF sha256. The top manifest
  also records `code_sha256_lf` of the code modules, so **any byte change** to `normalization_v3.py`,
  `versions.py`, the adapters or the source YAML makes `--verify` fail until rebuilt, even when the
  fingerprint would not change.
- **Determinism** is proven by two independent full builds hashed file by file (29 files). Record the result
  in `reports/normalization-v3/normalization-v3.determinism.json`, including what the new build supersedes.
- **After a rebuild, rerun every audit** and diff the output. Unchanged numbers are a result worth stating:
  - `scripts/audit_normalization_v3_structure.py` → `normalization-v3.structural-audit.json`
  - `scripts/audit_pdet_v1_representation.py` → `pdet-v1-representation-audit.json`
  - `scripts/audit_pdet_coverage_supply_v3.py` → `reports/pdet-coverage/pdet-coverage-v1.supply-v3.json`
  - `scripts/audit_toolace_call_final_shape.py` → `toolace-call-final-shape.json` (raw rows only, so it
    changes only with the raw artifact; `tests/data/test_toolace_call_prediction.py` regenerates it)
- **Re-anchor every copy of the fingerprint and top-manifest sha** (G15):
  - `reports/normalization-v3/manifests/`;
  - `docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md`,
    `31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md` and the study-002 `README.md`;
  - the pinned constants in `src/opengrad/verification/pdet_coverage.py` (`INPUT_FINGERPRINT`,
    `INPUT_TOP_MANIFEST_SHA256`, `SOURCE_MANIFEST_SHA256`), which refuse any other input. Then re-record the
    counts-only dry run (`opengrad-annotation` covers it) and regenerate 30 §13's table.

  Grep for the old hash afterwards. Only deliberate "supersedes" history may remain.
- The classifier may read only what `prose-decision-input-v1` allows (`src/opengrad/data/classifier_input.py`,
  doc `32-CLASSIFIER-INPUT-CONTRACT.md`). Changing a feature, an exclusion or the serialization requires a new
  contract version.
- The classifier is `src/opengrad/data/decision_classifier.py` (`prose-decision-classifier-v1`, plan
  `33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md`). It takes `ClassifierFeatures` only and returns a
  `Decision` naming the 22 §3 tree step and the matched evidence. Rules:
  - develop only against `reports/prose-classifier/dev/` (`python scripts/evaluate_prose_classifier_dev.py
    --show N`), and report the numbers as agreement with model labels, never accuracy;
  - never open P-DET-v1 or P-DET-COVERAGE-v1 files while developing
    (`tests/data/test_decision_classifier.py` checks that the module reads no file);
  - the held-out check set (`--set devcheck`, `src/opengrad/verification/classifier_devcheck.py`) is scored,
    not read, and never tuned against;
  - once frozen, any rule change is a new classifier version (33 §5, 22 §6).

## Releases, mixtures and yield

- **Release definitions:** `configs/releases/` (the final v2 is `toolpolicy_canonical_v2_final.yaml`). Build and
  validate with `opengrad-data build-hf-release` and `opengrad-data validate-release`. Publishing is covered in
  `opengrad-openweights-release`.
- **Canonical is not trainable:** a canonically valid source can yield zero trainable records (xLAM once did).
  Measure with `opengrad-data yield-report` against `configs/data/yield_expectations.yaml`. The readiness
  `renderability_yield` gate blocks a collapse.
- **Mixture method:** `src/opengrad/data/mixture.py` with `docs/data/tool-use-mixture-methodology.md`. Mixture
  and C1 membership decisions belong to the study owner.

## Tooling map

- **`opengrad-data`** (`src/opengrad/data/cli.py`):
  - sources and materialization: `opengrad-data normalize`, `opengrad-data materialize-parquet`;
  - inspection and checks: `opengrad-data audit`, `opengrad-data inspect`, `opengrad-data validate`;
  - rendering: `opengrad-data render`, `opengrad-data render-materialized`, `opengrad-data token-stats`;
  - releases: `opengrad-data build-hf-release`, `opengrad-data validate-release`;
  - trainability and bounded corpus analysis: `opengrad-data yield-report`, `opengrad-data real-audit`.
- **`opengrad-data-audit`:** data audit entry point.
- **`opengrad` checks on records:**
  - `opengrad data-audit` and `opengrad audit-corpus`: strict semantic audit of canonical training records;
  - `opengrad validate-data <records> --mode sft`: trajectory and schema validation;
  - `opengrad inspect-template`: rendering and loss masks.
- **Tool-call text parsing:** `src/opengrad/formatting/` (parser and formatting adapters).

## Sources of truth

`docs/data/normalization-architecture.md` · `docs/data/CANONICAL_TOOL_SCHEMA.md` · `docs/DATASET_MANIFESTS.md` ·
`reports/SUPERVISION_CONTRACT_REPORT.md` · `docs/research/study-002/31-CANONICAL-V3-SOURCES-AND-NORMALIZATION-V3.md`

## Keeping this skill current

Update this skill in the same commit when a normalization version, contract version, audit script or anchor
location changes.
