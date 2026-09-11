# OpenGrad Phase 0 Bootstrap Record

Historical phase report. This record describes the initial bootstrap checkpoint; current pre-GPU completion is recorded in `ROADMAP.md` and `reports/data-normalization-v1.md`.

Phase 0 established the repository and research infrastructure. Phase 0.5 added CPU-only pre-experiment validation; no model inference, training, checkpoint modification, or full dataset materialization has occurred.

## Deliverables

- Model-, dataset-, benchmark-, runtime-, and hardware registries
- JSON Schema contracts for experiments, provenance, and interoperability
- Canonical tool-conversation representation and source/model adapter boundaries
- Contamination and error-taxonomy infrastructure
- OpenPapers and OpenWeights integration boundaries
- Reproducibility, contribution, reporting, and Hugging Face documentation
- CPU-safe validation, smoke harnesses, stage gates, lineage checks, and environment capture

## Research record

The repository contains plans, schemas, fixtures, and metadata only. It contains no OpenGrad benchmark scores, trained checkpoints, or claims of completed research. Historical mixture proportions are hypotheses. `Salesforce/APIGen-MT-5k` remains excluded from the clean default because of benchmark-overlap risk.

## Verification

The current checkout is verified by Ruff, formatting checks, pytest, mypy, registry validation, CLI preflight, benchmark mock smoke tests, and whitespace checks. See `PRE_EXPERIMENT_REPORT.md` for the current Phase 0.5 record.

## Source and licensing notes

Scholarly metadata records retain source URLs, revisions, verification status, and unresolved fields. OpenPapers is optional research infrastructure; it is not a runtime dependency. OpenWeights is a separate downstream project. OpenGrad source code is licensed under Apache-2.0, while third-party datasets, models, benchmark assets, papers, and imported code retain their own terms.
