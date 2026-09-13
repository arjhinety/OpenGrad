# Guardrails

Rules for Study 002 onwards, each derived from a mistake Study 001 actually made. The Study 001
claim audit found 93 claims across this repository, the Hugging Face cards and the project site
that the committed artifacts did not support (13 high, 42 medium, 38 low), and a recheck of the
site before the freeze found 7 more. Every one is logged, with its resolution, in
[`reports/audits/study-001-claim-audit/`](../../reports/audits/study-001-claim-audit/README.md).
Numbers like #39 below are finding ids from that ledger; R1–R7 are the recheck findings.

A rule here is only useful if it is applied before the mistake, so the two checklists at the end
are part of the study process in [STUDIES.md](STUDIES.md): one before any GPU time is spent, one
before anything is published or frozen.

## Experiment design

**G1. Decision rules are fixed before results, and applied as written.** M1-v2 was promoted under a
gate (v4) introduced after M0's results were in; it fails the v3 gate that rejected M0, and M0
also clears v4 (#2, #50). The pre-registered "smallest passing format wins" rule selected Q6_K, yet
the report named Q8_0 (#24). A gate check was misreported as the one that failed (#4, #18).
*Check:* the gate version and decision rule are committed before the first candidate is scored. A
gate revised afterwards is re-run on every earlier candidate under both versions, and the write-up
names both.

**G2. The held-out set covers every response mode the model is used for.** The promotion partition
had tool-call, clarification and cannot-answer examples and no plain ANSWER examples, so a model
that refuses every bare arithmetic question passed. This is the finding of Study 001.
*Check:* list the response modes the model is deployed for and confirm each has examples in the
promotion partition. General-capability sentinels (IFEval, GSM8K zero-shot, MMLU-Pro) run before
promotion, not after.

**G3. A gate margin is compared with rerun noise.** The unquantized BF16 reference fails the
quantization gate on an H200 rerun (recall 0.7638 < 0.7671), and Q6_K clears precision by one
example (#26, #36).
*Check:* rerun the reference once and report the gate margin next to the rerun delta. A pass
inside that delta is reported as within noise, not as a pass.

**G4. Audit the exact data the model trained on.** The refusal-label audit read the
normalization-v1 sources (217,903 records, including held-out rows and sources M0 never saw)
instead of the Canonical-v2 corpus M0 trained on (#58, #80).
*Check:* an audit records the fingerprint of the corpus it read, and it must equal the fingerprint
in the trained run's `experiment.json`.

**G5. "Matched" and "isolates" are measured, not intended.** The matched-exposure ablation arm saw
1.19× the reference's supervised tokens (#6); "the two differ in exactly one respect" hid a
change from six sources to three (#89); a nominal 38,400-example estimate was reported as examples
seen (#11, #92).
*Check:* report the logged exposure (supervised tokens, examples seen) for every arm, and list
every factor that differs between arms.

## Provenance

**G6. Lineage and launch facts come from the run record.** M1-v1 was described as DPO on an SFT
parent; its `experiment.json` says `parent_experiment_id: null`, `reference: initial_policy` (#81,
#65, #75). An ablation was said to launch from a clean `0807fa3`; it launched from `500cb4e` with
`git_dirty: true` (#1). A registry claimed parent ids it did not record (#15).
*Check:* every lineage, reference, launch commit and dirty flag in prose is copied from
`experiment.json` / `events.jsonl`, never from memory or the plan.

**G7. "Reproducible" requires a repeat; "preserved" requires the file.** The only repeat ever run
did not reproduce (#49); regenerated checkpoints and superseded results said to be preserved were
not in git (#41, #68, #12). Hash pins computed over Windows CRLF bytes never verified on a clone
(#27).
*Check:* claim reproducibility only for runs with a matching repeat. Claim preservation only for
paths that exist in git. Freeze pins are verified in CI on a Linux clone.

**G8. Scaffolds are described as scaffolds.** A tokenizer gate hardcoded to pass was called
fail-closed (#66); the distillation path raises `NotImplementedError` but was described as
supporting three modes (#67); a mock M2 run was said not to exist (#21).
*Check:* a feature is described as working only if a test exercises its live path.

## Reporting

**G9. Every row of a comparison uses the same population.** B0's full-set numbers (n=3,650) sat
under "confirmatory" headings beside M0 and M1 on the 1,277-example partition (#9, #53, #83, #84).
Cost and run counts from different populations were paired (#42, #60, R3). Counts mixed trainable
and canonical records (#91, #85, #86).
*Check:* each table and figure states its partition and n. Every row is recomputed on that
partition. Numerator and denominator use the same counting unit.

**G10. Report the whole result.** A regression column listed precision and over-call but left out
the largest regression (#20); a best-of-four checkpoint chosen on the evaluation set was not
labelled as such (#62); a "Regression: None measured" cell hid a −0.467 recall drop (#3).
*Check:* regressions are reported next to gains, the final checkpoint next to any selected one, and
selection on the evaluation set is labelled.

**G11. State uncertainty.** The promoted model's +0.0078 call F1 over M0 is 7 of 453 calls from
one seed (#76). Point estimates were given where the evaluation's own rules forbid them (#51, #33).
*Check:* every comparison states n, seed count, and an interval or "no interval". A difference of
a few examples on one seed is reported as within noise.

**G12. Caveats travel with the number, on every surface.** The 2,560-token IFEval cap was hit by
up to 67 of 541 responses per model and disclosed nowhere (#30, R6). "55.5%" was called an answer rate; it is
the accuracy, and the answer rate is 100% (#52). The promoted model's card omitted the regression
entirely (#78, #79, #82, #90).
*Check:* a budget, truncation or refusal caveat is attached to every row it affects, including in
the model and dataset cards.

**G13. Causal language needs a causal design.** "SFT introduced two separable regressions" was
written when DPO on the base, with no SFT, also refuses 70.7% (#31); an engine effect was claimed
while hardware also changed (#34, #43); DPO was credited with an over-call drop it did not produce
(#25).
*Check:* write "associated with" unless an experiment varies only that factor, and name the
confounds.

## Publication

**G14. Numbers are generated or tested, never retyped.** Most wrong numbers were transcription
slips: 22.7pp for M1-v2 where 22.7 is M0's (#37), 71.0% beside 70.7% for one quantity (#40),
0.1553 from a different engine (#25), rounded figures that drifted (#16, #17, #19, #39, #46, #70,
#72, #88).
*Check:* figures in reports come from a generator script or are asserted by a test against the
JSON. One quantity has one value everywhere it appears.

**G15. A correction is applied to every copy.** Four of the seven recheck findings were errors
already corrected in the repository and never corrected on the site: 99.6% (#39 → R1), the
sentinel items (#32 → R2), the IFEval cap (#30 → R6) and the cost pairing (#60 → R3).
*Check:* before closing a correction, search the repository, the site source, the card templates
and the append-only ledgers for the same figure or phrase, and fix or erratum every hit.

**G16. Status words change in the same commit as the status.** Docs kept saying PLANNED,
`FROZEN_NOT_EXECUTED` or "no result exists" after the stage had run (#28, #35, #45, #57, #69, #71,
#74, #93), and generated files were hand-edited instead of fixed at the generator (#63).
*Check:* the commit that records a stage's result also updates every status line that mentions the
stage (search for the stage name and its status words), and generated files change only through
their generator.

**G17. Public surfaces change with the repository.** The Hugging Face cards and the site lagged the
repository by several corrections (#78, #79, #82, #83, #84, #64, #87).
*Check:* a result or correction that affects a card or the site is not done until the card or site
is updated and re-fetched to confirm it matches.

## Checklists

### Before any GPU time is spent (pre-registration)

- [ ] Decision rule and gate version committed (G1)
- [ ] Promotion partition covers every deployed response mode; capability sentinels scheduled (G2)
- [ ] Reference rerun planned so gate margins can be compared with noise (G3)
- [ ] Training data fingerprint recorded, and any audit reads that fingerprint (G4)
- [ ] Every factor that differs between arms listed; exposure logging on (G5)
- [ ] Seeds and intervals planned, or the single-seed limit stated in advance (G11)

### Before publishing, or freezing a study

- [ ] Every number recomputed from the committed JSON, by a generator or test (G14)
- [ ] Every table states its partition and n, and all rows use it (G9)
- [ ] Regressions, final checkpoints and selection effects reported (G10)
- [ ] n, seeds and an interval or "no interval" on every comparison (G11)
- [ ] Caveats on every affected row, on every surface (G12)
- [ ] Causal wording checked against the design (G13)
- [ ] Lineage and launch facts copied from the run record (G6)
- [ ] Reproducibility and preservation claims backed by a repeat and by files in git (G7)
- [ ] Scaffolds described as scaffolds (G8)
- [ ] Status lines updated in the same change; generated files regenerated (G16)
- [ ] Each correction applied to every copy: repo, site, cards, ledgers (G15)
- [ ] Cards and site updated and re-fetched (G17)
- [ ] For a freeze: an independent claim audit of the repository, cards and site, with every
      finding resolved, logged under `reports/audits/`
