# README information-architecture audit — 2026-09-12

This audit precedes and justifies the README refactor. It records what the README currently
contains, what happens to each part, where moved material goes, and which claims were stale or
ambiguous. Numbers are resolved from authoritative artifacts, not from the previous README prose.

## Problem statement

`README.md` (548 lines, ~6,030 words, 40 top-level sections) is simultaneously acting as landing
page, research paper, experiment log, dataset card, benchmark registry, architecture spec,
methodology doc, provenance record, and contributor guide. A cold visitor cannot answer the eight
landing-page questions (what / why / active track / latest promoted result / current dataset /
executed vs planned / how to run / where the evidence lives) within two screenfuls: the promoted
result sits at line ~399, Quick Start at line ~475, and the only navigation table at line ~492.

## Section disposition

Legend: **KEEP** (stays, tightened) · **COMPRESS** (stays, reduced to a summary + link) ·
**MOVE** (detail moves to docs; README keeps a pointer) · **REMOVE** (folded away entirely).

| Current section (README line) | Action | Destination / notes |
|---|---|---|
| Hero, tagline, 4-line philosophy trio (7–20) | COMPRESS | 2–3 sentence overview; philosophy to Research Principles |
| Blockquote capability list (22) | COMPRESS | one factual sentence; drop "complete … operating system" |
| Badge row (11–14) | KEEP | add HF models + datasets badges, 6 max |
| `## Primary Research Highlights & Architecture` (26) | REMOVE | its three subsections are dataset/benchmark/trainer detail that moved below and to docs |
| `### 1. Benchmarks Tiers A–E` (28–35) | MOVE | `docs/benchmarks/README.md` (inventory) + existing `docs/evaluation/BENCHMARK_STRATEGY.md` |
| `### 2. OpenWeights device testing` (37–43) | COMPRESS | one paragraph in *How OpenGrad Works*; detail already in `docs/research/motivation.md`, `docs/openweights/README.md`, `docs/evaluation/OPENWEIGHTS_ON_DEVICE_TESTING.md` |
| `### 3. Experiment OS & future RL` (45–49) | COMPRESS | Quick Start + `docs/EXPERIMENT_FOUNDATION.md`, `docs/FUTURE_RL_INTEGRATION.md` |
| `## At a glance` (54–62) | MOVE | superseded by *Current Research Status*; the Q&A is duplicated by the status table |
| `## Why OpenGrad?` (65–75) | COMPRESS | one short paragraph; tradeoff block to `docs/research/motivation.md` |
| `## From deployment problems to research questions` (78–97) | MOVE | already complete in `docs/research/motivation.md` (+ its mermaid); README keeps one paragraph |
| `## Research questions` RQ1–RQ6 (100–126) | MOVE | new `docs/research/research-program.md`, collapsed to one table |
| `## Study 001` (128–139) | MOVE | new `docs/research/research-program.md` |
| `## Experimental decision pipeline` (142–159) | MOVE | new `docs/research/research-program.md` (mermaid) |
| `## Current research status` (162–186) | KEEP | promoted to position 1; keeps anchor `#current-research-status` |
| `## Dataset releases` + 3 subsections (187–215) | MOVE | refresh stale `docs/publishing/huggingface-datasets.md` |
| Dataset source table (217–235) | MOVE | `docs/publishing/huggingface-datasets.md`; source identity already in `docs/data/normalization-sources.md`, numbers in `reports/CANONICAL_V2_COMPLETION_REPORT.md`, contracts in `reports/SUPERVISION_CONTRACT_REPORT.md` |
| Mixture M0/M1/M2 paragraph (233) | MOVE | already complete in `docs/data/tool-use-mixture-methodology.md` (incl. the training-phase disambiguation) |
| `## Benchmark program` (240–266) | MOVE | `docs/benchmarks/README.md`; philosophy/axes already in `docs/evaluation/BENCHMARK_STRATEGY.md` |
| `### Benchmark inventory and counting convention` (255–265) | MOVE | `docs/benchmarks/README.md` (new anchor); updates `docs/foundation/README.md` |
| `### Baseline execution gate` (267–281) | MOVE | `docs/EVALUATION.md` |
| `## Measurement, not leaderboard chasing` (283–312) | MOVE | capability taxonomy already in `docs/evaluation/BENCHMARK_STRATEGY.md` §1/§3; efficiency wishlist to `docs/inference/efficiency.md` |
| `## Capability × efficiency` (314–329) | MOVE | framing to `docs/research/research-program.md`; interface detail already in `docs/optimization/README.md` |
| `## Research architecture` (331–358) | MOVE | mermaid stays in README (one diagram); directory-role list → `docs/architecture/repository.md` |
| `## Reproducibility and provenance` (360–373) | MOVE | already a subset of `docs/research/reproducibility.md`; README links it |
| `## Negative results are results` (375–379) | MOVE | expand `docs/contributing/negative-result.md`; INC-0001 already in `docs/INCIDENT_LOG.md` |
| `## Provenance across projects` (381–394) | MOVE | label vocabulary already in `docs/research/methodology.md` |
| `## Results` full table + 4 caveats + artifacts + index note (396–474) | MOVE | new `docs/EXPERIMENT_RESULTS.md`; README keeps the 4-row progression table |
| `### Illustrative future record` YAML (442–473) | MOVE | `docs/EXPERIMENT_FOUNDATION.md` |
| `## Development setup` (475–487) | MOVE | becomes *Quick Start* higher up |
| `## Reproducing experiments` (488–491) | COMPRESS | one paragraph + links |
| `## Navigation` (492–509) | MOVE | becomes the *Start Here* table at the top |
| `## Multi-model design` (510–517) | MOVE | `docs/models/README.md` (the "≠" code block) |
| `## How to challenge a result` (518–529) | COMPRESS | folds into *Contributing* |
| `## Related projects` (530–537) | COMPRESS | folds into *Collaboration / related projects* |
| `## Roadmap` (538–541) | COMPRESS | one pointer to `ROADMAP.md` |
| `## Citation and license` (542–545) | KEEP | unchanged in substance |
| `## Acknowledgements` (546–548) | KEEP | one line |

## Duplicated concepts found

Each of these appears two to four times in the current README; the refactor states each once:

- **"Infrastructure validation is not an ML result"** — hero, Benchmark program, Results, Development setup.
- **"Negative results are evidence"** — hero, At a glance, Results caveats, Negative results section, Contributing.
- **"Planned ≠ implemented"** — blockquote, status table, capability×efficiency, related projects.
- **Provenance/ownership boundary with OpenWeights** — "From deployment problems", "Provenance across projects", Related projects, Acknowledgements.
- **Measurement philosophy** — "Why OpenGrad?", "Measurement, not leaderboard chasing", Results.
- **Benchmark inventory** — Primary highlights Tier list *and* Benchmark program table *and* counting convention.

## Stale or ambiguous claims encountered

| Claim | Status | Resolution |
|---|---|---|
| "complete post-training experiment operating system" | Exaggerated | factual phrasing; OPD is scaffold-only |
| "16-benchmark evaluation system" | **Already wrong** (17 tiered) | counting convention retained, moved to `docs/benchmarks/README.md` |
| On-policy distillation as an active trainer backend | Corrected in the prior audit; must not regress | README says scaffold only; `docs/EXPERIMENT_STATUS.md` shows M2 `INVALID` |
| "no external benchmark score exists" | **Verified true** | `reports/benchmarks/*/environment.json` reports `backend: mock` for all five families |
| M2 status | `INVALID` / `MOCK_ONLY` | never presented as a training result |
| Dataset counts | Canonical-v2 final is current (173,237 / 161,966 / 4 sources) | v1 (213,951) and partial-v2 (103,036) labelled historical |
| "opengrad readiness reports PASS" | False on a fresh checkout | replaced with the observed blocking gates (prior audit) |
| README line 245 "B0 and 12 candidate checkpoints" on When2Call | Ambiguous ("candidate" scope unstated) | restated as the executed behavioral held-out with the report as source |

## Proposed new README hierarchy

1. Hero + 2–3 sentence overview
2. Badge row (CI, Python, License, Research status, HF models, HF datasets)
3. **Current Research Status** (compact table; anchor preserved)
4. **Start Here** navigation table + compact 10-entry TOC
5. **Latest Results** (4-row progression table → `docs/EXPERIMENT_RESULTS.md`)
6. **Quick Start** (real commands only)
7. **What OpenGrad Is** (problem + why, one paragraph)
8. **How OpenGrad Works** (one mermaid pipeline + OpenWeights paragraph)
9. **Research Program** (RQ table → `docs/research/research-program.md`)
10. **Datasets & Evaluation** (two compact tables → docs)
11. **Current Limitations**
12. **Reproducing Experiments**
13. **Repository Structure** (compact code block)
14. **Contributing** (incl. challenging a result)
15. **Citation and License**
16. **Collaboration / related projects** + acknowledgements

Target: ~230–280 lines, ~2,400–3,000 words (≈50–55% reduction), with detail preserved in docs.

## Acceptance test

A cold visitor should be able to answer, within two screenfuls: what OpenGrad is; which model and
dataset are current; that M1-v2 DPO is the promoted result and M2 is not executed; what is executed
versus planned; how to install and validate; and where the full evidence lives.

## Outcome

| Measure | Before | After |
|---|---:|---:|
| Lines | 548 | 248 (−55%) |
| Words | 6,031 | 1,744 (−71%) |
| `##`/`###` sections | 40 | 14 `##`, no `###` |
| Quick Start position | line ~475 | line 54 (~22%) |
| Latest results position | line ~399 | line 86 (~35%) |
| Navigation | one table at line ~492 | Start Here table + TOC at lines 39–53 |

New pages: [`docs/EXPERIMENT_RESULTS.md`](../docs/EXPERIMENT_RESULTS.md) (full results table,
caveats, published artifacts), [`docs/research/research-program.md`](../docs/research/research-program.md)
(RQ1–6, Study 001, decision pipeline, capability × efficiency), and this audit.

Refreshed existing homes (no duplicate docs created): `docs/publishing/huggingface-datasets.md`
(was v1-only and stale), `docs/benchmarks/README.md` (was three lines; now the inventory, tier
list, counting convention, harness status), `docs/EVALUATION.md` (baseline execution gate),
`docs/inference/efficiency.md` (planned measurements, target-attached terminology),
`docs/models/README.md` (multi-model boundary), `docs/contributing/negative-result.md` (retention
and correction rules), `docs/EXPERIMENT_FOUNDATION.md` (illustrative record, `INVALID` state),
`docs/architecture/repository.md` (repository layout; corrected a stale intervention count).

Cross-references fixed: `ROADMAP.md` (`#results` → `#latest-results`), `docs/foundation/README.md`
(benchmark anchor now points at `docs/benchmarks/README.md`), and the status-document generator
(link updated, `docs/EXPERIMENT_STATUS.md` regenerated).

## Validation

- **Links and anchors:** all 133 tracked Markdown files checked — 0 missing targets, 0 broken
  anchors. Now enforced by `test_documentation_links_and_anchors_resolve`.
- **Tests:** `tests/results` includes the new link test; the full suite shows the same 11
  pre-existing environment failures as unchanged `HEAD` (see `DOCUMENTATION_STATE_AUDIT.md`).
- **Registry:** `opengrad-validate` → `registry validation: OK`; generated status document
  `--check` → `CURRENT`.
- **Lint:** `ruff check src scripts tests integrations` → clean.

No unresolved information-architecture issues. The one caveat is that `opengrad readiness` behaviour
is environment-dependent, which the Quick Start states explicitly rather than promising a pass.

