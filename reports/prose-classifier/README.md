# Prose decision classifier — development and test evidence (Study 002)

This directory holds the populations, model labels, check rounds and one-shot tests behind the two frozen
decision classifiers (`prose-decision-classifier-v1` and `-v2`). The plans are
[33](../../docs/research/study-002/33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md) (v1) and
[37](../../docs/research/study-002/37-PROSE-DECISION-CLASSIFIER-V2-DEVELOPMENT-PLAN.md) (v2). Every file
here is evidence: do not edit it; correct it in [`reports/ERRATA.md`](../ERRATA.md).

## Which set is which

Three names look alike and mean different things. The **set name** is what matters; the directory is only
where its label archive landed.

| Set name | For classifier | Role | Population and manifest | Labels and audit trail |
|---|---|---|---|---|
| `prose-classifier-dev-v1` | v1 | development set | `dev/` | `dev/annotation/`, `dev/provenance/` |
| `prose-classifier-devcheck-v1` | v1 | first check set (the developer then read its disagreements, so it became development data) | `dev/` | `devcheck/` |
| `prose-classifier-devcheck-v2` | v1 | second, fresh check set for v1 — **"v2" is the check number, not the classifier** | `dev/` | `devcheck-v2/` |
| `prose-classifier-dev-v2` | v2 | development set from first replies | `dev-v2/` | `dev-v2/annotation/`, `dev-v2/provenance/` |
| `prose-classifier-v2-devcheck-1` … `-5` | v2 | the five check rounds' fresh sets | `dev-v2/` | `v2-devcheck-N/` |

- **Agreement files** in `dev-v2/` are named `prose-decision-classifier-v2.roundR.<set>-agreement.json`, where
  `<set>` abbreviates the set: `dev-v1`, `dev-v2`, `devcheck-v1`, `devcheck-v2`, and `v2-check-N` for
  `prose-classifier-v2-devcheck-N`. Round R's rules were measured on every set drawn so far.
- **One-shot tests** of the frozen classifiers are `test/` (v1, on P-DET-COVERAGE-v1 and P-DET-v1) and
  `test-v2/` (v2, on P-DET-COVERAGE-v2). Each was run once.

## The `annotation/wip/` name

`annotation/wip/` is the export folder name `opengrad-annotate export` always uses. It does not mean the
labels are unfinished: each set's labels were exported once and pinned by the `.sha256` sidecars beside them.
No label here is gold; all are model labels (see the [glossary](../../docs/GLOSSARY.md)).

## Local paths in the audit trails

The `provenance/*/…audit-trail.manifest.json` files record the author's absolute file paths. They are pinned,
so they are kept; the writer now records repo-relative paths ([`reports/ERRATA.md`](../ERRATA.md) §22).
