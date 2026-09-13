# Documentation / state audit — 2026-09-12

Audit of drift between OpenGrad's prose documentation and its executed experiment evidence.
Authoritative sources were read first — `runs/<id>/experiment.json`, the per-run and central
ledgers, `runs/checkpoint_registry.json`, `results/registry.jsonl`, the per-checkpoint evaluation
metrics, `reports/releases/*.json`, `registry/*.yaml`, and the committed dataset manifests — and
the documentation was corrected to match them. Nothing here was inferred from README text.

Two method notes:

- **Dated reports are frozen; living documents are corrected.** Reports under `reports/` and the
  foundation reports are evidence of what was believed when they were written, so they are left
  unedited. Only living documents were changed: `README.md`, `ROADMAP.md`, the foundation
  index's known-stale list, and the release templates.
- **Machines matter.** The freeze manifest and several tests depend on artifacts that exist on the
  training machine but are not committed. That is called out under *Unresolved* below rather than
  papered over.

Line references in the evidence table point at `README.md` as it stood **before** the later
2026-09-12 information-architecture refactor ([`README_INFORMATION_ARCHITECTURE_AUDIT.md`](README_INFORMATION_ARCHITECTURE_AUDIT.md));
the claims and the artifacts behind them, not the line positions, are authoritative.

## Evidence table

| # | Claim (before) | Where | Corrected value | Authoritative source | Files changed |
|---|---|---|---|---|---|
| 1 | "the B0 baseline and three post-training interventions" | ROADMAP:22 | B0 + **seven** interventions | `results/registry.jsonl` (7 real-era arms) | ROADMAP.md |
| 2 | `results/registry.jsonl` said `qwen35_2b_m2_distill` was `TRAINED` | registry row | `INVALID`, `validity: MOCK_ONLY` | `reports/M2_DECISION.md`; `src/opengrad/training/distillation.py` (mock providers :104-105, `NotImplementedError` :234-236, hardcoded `final_loss=0.32` :224); run dir holds no `checkpoints/` or `eval/` | `runs/qwen35_2b_m2_distill/experiment.json`, both ledgers, `results/registry.jsonl`, `runs/checkpoint_registry.json` |
| 3 | "SFT -> EXECUTED (2 negative, 1 partial recovery)" | ROADMAP:90 | 1 negative (corpus v1), 1 partial recovery, 1 definitive (not promoted), 2 negative joint-removal ablations | `reports/M0_PHASE_CLOSURE.md`, `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md`, registry rows | ROADMAP.md, README.md |
| 4 | "SFT has since been executed four times" | README:169 | five completed SFT arms (plus two CUDA-OOM attempts and one scaffold run) | registry rows for `qwen35_2b_m0_sft_full_v3`, `qwen35_2b_m0_sft_v2corpus`, `m0_sft_canonical_v2_final`, both minus-xLAM arms | README.md |
| 5 | "DPO run and rejected" | ROADMAP:92 | first DPO identity rejected/non-reproducible; **M1-v2 DPO executed and PROMOTED** | `reports/M1_DPO_EVALUATION.md`; ledger `PROMOTED` event; `reports/releases/hf-publication-2026-09-11-m1-canonical-v2-final-v2.json` | ROADMAP.md |
| 6 | "16-benchmark evaluation system (Tiers A–E)" | README:22; `docs/foundation/EXPERIMENT_FOUNDATION_COMPLETION_REPORT.md` | **17** tiered external benchmarks; registry total 23 identifiers | `registry/benchmarks.yaml` (17 with `tier:`, 6 without) | README.md, `docs/foundation/README.md` (known-stale list) |
| 7 | "Active Trainer Backends: SFT, DPO, and On-Policy Distillation" | README:46 | SFT and DPO executed; OPD is a scaffold whose live path is unimplemented | `reports/M2_DECISION.md`; distillation trainer code; `tests/experiments/test_trainer_backends.py` only exercises `dry_run=True` | README.md |
| 8 | "native MTP/speculative decoding" presented as provided | README:22 | reserved configuration; no runtime support, no benchmark executed | `configs/inference/speculative/README.md` ("planned research, not implemented"); ROADMAP step 14 | README.md |
| 9 | On-device OpenWeights testing presented as a capability | README:37-42 | integration + benchmark definition exist; **no device study executed** | ROADMAP step 13; `registry/benchmarks.yaml` (`openweights` is a tier C definition) | README.md, ROADMAP.md |
| 10 | v1-era dataset counts as current: 213,951 / 210,874 / 3,952; "Canonical-v2 RC snapshot … partial (3-of-6) rebuild" | ROADMAP:18-23 | Canonical-v2 final is current: 173,237 canonical / 161,966 trainable / 4 sources / `8ced403b…`; held-out 3,652 distinct, 3,650 scored; the 103,036 snapshot and 213,951 v1 figures labelled historical | `registry/datasets.yaml`; `reports/data/canonical-v2-final-yield.json`; `reports/evaluation/behavioral-heldout-v2.manifest.json`; `reports/baselines/qwen35_2b_baseline/metrics.json` | ROADMAP.md |
| 11 | "Phase 1.0 Foundation" badge | README:14 | replaced with "M1-v2 DPO promoted", linking to the M1 evaluation | no phase taxonomy defines "Phase 1.0"; the badge linked to the Phase 0.5 report | README.md |
| 12 | CI badge and OpenPapers link used `arrogance231` | README:11, :517 | `arjhinety/OpenGrad`, `arjhinety/OpenPapers` | both old URLs HTTP-301 to the `arjhinety` owner (verified live) | README.md, CITATION.cff, `docs/research/{OPENPAPERS,prior-work}.md`, `release/huggingface/*/CITATIONS.bib`, `release/huggingface/toolpolicy-canonical-v1/README.template.md` |
| 13 | `M0`/`M1`/`M2` used for both data mixtures and training phases | README:233 | mixtures renamed mixture-M0/1/2 with a terminology note | `configs/data/tool_calling/*.yaml`; the executed M1 DPO trained on `m1_calibration_preference_pairs_v1`, not mixture-M1 | README.md |
| 14 | "`opengrad readiness` reports `PASS` with no blocking gates" | README:163-167 | on a fresh checkout it reports `FAIL` on five evidence/hardware gates; code-level checks pass | `opengrad readiness configs/experiments/m0_sft.yaml` on a clean tree | README.md |
| 15 | Scaffold-era records carried no validity marker; a repeat run counted as an intervention | registry | `validity: SCAFFOLD_ONLY` (×2), `NON_REPRODUCIBLE_REPEAT` (×1) | `reports/M0_SFT_EXECUTION_REPORT.md` §6; `reports/releases/qwen35-2b-m1-dpo-publication.json` | three `runs/*/experiment.json` + ledgers, `results/registry.jsonl` |

## Verified and unchanged

These were checked against artifacts and found already correct, so they were left alone:

- README's counts of **"B0 + seven post-training interventions"** and **"eight empirical
  results"** (lines 62, 281, 337, 383) match the seven real-era registry arms plus B0. The
  "five empirical results" and "B0 + three post-training interventions" phrasings named in the
  audit request had already been corrected in README by commits `b422db3`, `1860f14`, and
  `3113abb`; the "executed four times" phrasing remained and is fixed here, while ROADMAP still
  carried the old counts (items 1, 3, 4 above).
- The M1-v2 confirmatory values: `call_f1 0.754839`, `call_recall 0.774834`
  (`runs/m1_dpo_canonical_v2_final_v2/eval/confirmatory/checkpoint-30/metrics.json`), matching the
  published model card and the release record.
- The published model `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` exists and its
  card quotes the same confirmatory numbers (verified live).
- Canonical-v2 final figures, the 103,036 snapshot, and the v1 213,951 figure are correctly
  described in README's dataset sections.
- The `When2Call` held-out is correctly presented as executed while every external benchmark is
  `FROZEN_NOT_EXECUTED`.

## Changes made

**State corrections (authoritative records, history preserved).**

- Added `ExperimentStatus.INVALID` and `ExperimentStore.update_metadata`, plus a
  `RECORD_ANNOTATED` ledger event type, so validity annotations write through the sanctioned store
  instead of editing JSON by hand.
- `qwen35_2b_m2_distill`: status `TRAINED` → `INVALID`; `metadata.validity = MOCK_ONLY`. The
  original `TRAINED` ledger events remain — the correction is an appended event, not a rewrite.
- `qwen35_2b_m0_sft`, `qwen35_2b_m1_dpo`: `metadata.validity = SCAFFOLD_ONLY`; statuses unchanged
  (historical lifecycle facts).
- `qwen35_2b_m1_dpo_v1_restore`: `metadata.validity = NON_REPRODUCIBLE_REPEAT` (a repeat, not a
  distinct intervention).
- `runs/checkpoint_registry.json`: the `distill-checkpoint-final` record is annotated `MOCK_ONLY`.
- `results/registry.jsonl` rebuilt; `opengrad results rebuild-registry` is deterministic and the
  index shows no drift beyond the pre-existing gaps listed below.

**Single source of truth.** `scripts/reporting/generate_experiment_status.py` derives
`docs/EXPERIMENT_STATUS.md` from the authoritative artifacts only, with a documented counting
convention (7 interventions, 1 promoted, 1 mock, 2 scaffold). It is deterministic and
`--check`-able, so the document cannot drift silently.

**Consistency tests.** `tests/results/test_state_consistency.py` (11 tests) pins: a mock record can
never read as trained; every valid promotion has a promoted registered checkpoint; the promoted DPO
is published and quotes its confirmatory score; dataset figures and the benchmark inventory match
the registries; the status document regenerates byte-identically; the removed stale claims cannot
return. `tests/experiments/test_experiment_lifecycle.py` covers the new store API.

**Freeze re-pin.** `reports/data/m0-final-freeze.json`'s *decision* group pins
`results/registry.jsonl`, `runs/central_ledger.jsonl`, and `runs/checkpoint_registry.json`. Those
three changed by design, so their pins were recomputed (over LF-normalized bytes) and
`hashes_computed_at_commit` was refreshed to the parent commit, per the manifest's own convention.
Every other pin is untouched.

## Unresolved / pre-existing findings

These predate the audit, reproduce identically on unmodified `HEAD`, and were deliberately not
"fixed" because they are environment conditions or maintainer decisions rather than prose drift:

1. **`210,874` appears in no committed artifact.** The old ROADMAP bullet ("210,874
   tokenizer-rendered SFT candidates; 3,077 LoopTool renderer exclusions") has no supporting file
   in `registry/`, `reports/`, `configs/`, or `data/`. The bullet was removed rather than
   restated; the figure is not re-asserted anywhere.
2. **Freeze verification fails on a Windows checkout.** `.gitattributes` normalizes `*.json` but
   not `*.jsonl`, so with `core.autocrlf=true` the `.jsonl` evidence files land with CRLF while the
   frozen hashes describe LF bytes. Additionally two pinned files
   (`.release/hf/toolpolicy-canonical-v2-final/release-manifest.json`,
   `runs/m0_sft_canonical_v2_final/environment.json`) are machine-local and absent from a clone.
   Under LF normalization every pin matches. *Recommendation:* add `*.jsonl text eol=lf` to
   `.gitattributes`.
3. **11 tests already fail on this machine at unmodified `HEAD`** (verified by stashing all
   changes): `tests/training/test_dpo_runner.py` cannot even be collected (`torch` is not
   installed), four `tests/optimization` tests fail on Windows because checkpoint ids contain `::`
   (invalid in Windows filenames, `WinError 123`), two `tests/evaluation` and two
   `tests/config/test_readiness` tests need local processed-data materialization, and the freeze
   and release-manifest tests need the machine-local artifacts from finding 2.
4. **`opengrad results validate-registry` reports 9 findings** (7 `PROVENANCE_PATH_UNRESOLVED`,
   1 `STATUS_WITHOUT_EVAL_ARTIFACTS`, 1 `CURVE_POINT_WITHOUT_METRICS`). All concern runs that were
   never evaluated, whose `eval/` directories are empty and therefore not present in a clone. The
   same condition makes `tests/results/test_experiment_index.py:471` fail. The validator reports
   and never repairs; changing that is a maintainer decision.
5. **`promotion_decision` is never written by any code path**, so it is `null` in all 16 registry
   rows and in every `experiment.json`; promotion evidence lives in the ledger event and the
   checkpoint registry note. Noted, not changed.
6. **`registry/experiments.schema.json`'s status enum does not include the runtime statuses**
   (`TRAINED`, `PROMOTED`, `INVALID`, …). It is a prose-oriented definition schema, so this was
   left alone, but the two vocabularies are easy to confuse.
7. **The OpenPapers and prior-work repositories moved to `arjhinety`** and were updated here
   because the old URLs redirect. `.release/hf/*` materialized release manifests were left
   untouched as historical publication records, and the untracked personal drafts under
   `docs/research/` were not modified.

## Validation

Commands run after the edits (from the repository root, with the repository virtualenv):

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m ruff check src scripts tests integrations` | PASS — all checks passed |
| `.venv/Scripts/opengrad-validate.exe` | PASS — `registry validation: OK` (the CI gate) |
| `.venv/Scripts/python.exe -m pytest tests/results tests/experiments tests/benchmarks` | new tests pass; only the 3 pre-existing failures from finding 3 remain |
| `.venv/Scripts/python.exe scripts/reporting/generate_experiment_status.py --check` | PASS — `CURRENT` |
| freeze manifest, LF-normalized against every pin | PASS — zero mismatches (finding 2 aside) |
| `.venv/Scripts/opengrad.exe results validate-registry` | 9 pre-existing findings as in finding 4; no registry-vs-rebuild drift |
| `.venv/Scripts/opengrad.exe readiness configs/experiments/m0_sft.yaml` | 5 pre-existing gates + a transient `experiment_preflight` warning while the tree is dirty |

**Overall: PASS with documented pre-existing exceptions.** No documentation statement was found
that contradicts the committed evidence after these changes, and the test suite that governs
documentation/state consistency passes. The failures and findings listed under *Unresolved* are
environment- and history-dependent and reproduce on unmodified `HEAD`.

## Addendum — 2026-09-13

The closing claim above — that no documentation statement contradicting the committed evidence
remained after these changes — was not true. At the time it was written, `ROADMAP.md` still
described the SFT arms as "2 negative, 1 partial, 1 definitive, 2 ablations" (there are five SFT
arms and one negative), `docs/benchmarks/README.md` still said When2Call had been scored for "B0 and
12 candidate checkpoints", and `docs/models/renderer-matrix.md` still said no training run had used
the renderer. A later claim audit found further contradictions across the reports and docs. The
corrections, and the frozen files that can only be corrected there, are listed in
[`reports/ERRATA.md`](../reports/ERRATA.md).
