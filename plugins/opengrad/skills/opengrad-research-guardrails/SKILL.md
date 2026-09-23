---
name: opengrad-research-guardrails
description: Research-integrity rules for OpenGrad — the G1–G17 guardrails, study freezing and errata, preregistration and amendments, frozen artifacts that must never change, phase gating by the study owner, and how to report numbers, negative results and incidents. This skill should be used before changing anything under docs/research/, reports/ or a frozen artifact, before writing any claim or number into prose, and whenever a task could start a new research phase.
---

# OpenGrad research guardrails

OpenGrad publishes numbered studies. Study 001 is frozen and Study 002 is in progress. Its lessons are
codified in `docs/research/GUARDRAILS.md` (G1–G17). Read that file before any research-facing change. What
follows is the operational core.

## Never modify

- **Frozen studies and their artifacts:** Study 001 checkpoints, corpora, evaluations and reports. A wrong
  number in a frozen study is corrected in `reports/ERRATA.md`, never in place (`docs/research/STUDIES.md`).
- **Frozen populations and corpora:** P-DET-v1 (`population_sha256` `6ab92087…`), canonical-v1/v2 releases, and
  hash-pinned reports. Source datasets are read-only.
- **A committed preregistration:** `docs/research/study-002/03-PREREGISTRATION.md` and the numbered design docs
  it fixes, such as `04-ARM-MATRIX.md`. Change one only through a new numbered amendment that states its
  version (the precedents are `28-PDET-MODEL-LABEL-AMENDMENT.md` → `study_002_prereg_v2`, and
  `29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md` → `study_002_prereg_v3`, and
  `30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md` → `study_002_prereg_v4`, adopted in place with its status line
  changed, and `34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md` → `study_002_prereg_v5`), written *before* results it could
  influence (G1).
- **A gate or threshold, to make a run pass.** A threshold that blocks a correct result is a finding about the
  threshold and is recorded, not edited around (`docs/contributing/README.md`).

New results go into new artifacts. Rebuilding an artifact means a new version or fingerprint that records what
it supersedes, never a silent overwrite.

## Phase gating

The study owner authorises each phase explicitly: freezing gold labels, drawing a population
(P-DET-COVERAGE), implementing or validating the classifier, authorising C1 balancing, changing the mixture,
training. Do not start a phase because the previous one finished.
- Finish the authorised work.
- Report it.
- Ask with concrete options: the recommended one first, with trade-offs.

A decision the owner makes is recorded where the work lives (a doc row, config field or gate record), in the
same commit.

## Writing claims and numbers

- **G14:** numbers are generated or tested, never retyped. Copy them from the artifact or JSON that produced
  them, and prefer a script that writes the table.
- **G5 / G9:** "matched" and "isolates" must be measured, and every row of a comparison uses the same
  population.
- **G10 / G11 / G12:** report the whole result, state uncertainty (counts behind percentages, seeds, noise), and
  keep caveats attached to the number on every surface.
- **G13:** causal language needs a causal design.
- **G16 / G17:** status words, and public surfaces (Hugging Face cards, the site), change in the same commit as
  the status.
- Never describe the system as providing cryptographic identity or authenticity it does not provide.
- Never report an unrun suite as passing. A skipped or blocked check is not a pass.

## Labels, judgments and human passes

- Model judgments stay hidden from human annotation passes.
- Labels are changed only by a human through the annotation UI. Never by script, and never by editing the
  store. See `opengrad-annotation`.
- Only the owner decides whether a label is revised. When asked to cross-check a label, report the evidence
  and leave the label unchanged unless told otherwise.

## Negative results and incidents

- A null or negative result is written up (`docs/contributing/negative-result.md`) with the same rigor as a
  positive one.
- An operational mistake that changes what can be claimed gets an entry in `docs/INCIDENT_LOG.md`. Entries are
  never edited to look better; a later entry supersedes an earlier one.
- **Amendments are recorded twice**, in `03-PREREGISTRATION.md` and in `reports/ERRATA.md`, in the same commit;
  `study_002_prereg_v7` was recorded only once and needed a late entry (ERRATA §18). A draft amendment
  (`40-PREREG-V8-DRAFT.md`) is recorded in neither until the owner adopts it.
- **A frozen study's files are not edited in place, even to add a recovered number** (ERRATA §20); the number
  goes into ERRATA.
- **Where a new study's evidence goes:** a `reports/study-<number>/` directory, kebab-case names with the version last. Existing
  Study 002 evidence stays where it is (its paths are pinned); the generated `reports/README.md` records its study.
- Freezing a study requires the claim audit (`reports/audits/study-001-claim-audit/README.md` is the precedent)
  and matching git tags, as `docs/research/STUDIES.md` describes.

## Sources of truth

`docs/research/GUARDRAILS.md` · `docs/research/STUDIES.md` · `docs/research/study-002/README.md` (current state
table) · `docs/research/methodology.md` · `docs/research/reproducibility.md` · `reports/ERRATA.md` ·
`docs/INCIDENT_LOG.md`

## Keeping this skill current

When a study freezes, a new guardrail is added, or a preregistration version changes, update "Never modify"
and the amendment precedents in the same commit.
