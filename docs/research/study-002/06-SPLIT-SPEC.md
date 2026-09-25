# 06 — Split specification

Study 001's evaluation population could not see the thing it was used to judge. `docs/EVALUATION.md:12-17`
states it first: *"The frozen tool-policy partition held `tool_call`, `request_for_info` and
`cannot_answer` examples and **no ANSWER examples**. It therefore could not detect a checkpoint that
stopped answering ordinary questions — and did not."* This document specifies the populations Study 002
scores on, and makes their coverage a checkable property rather than a hope.

## The measured defect, stated numerically

Counted from `results/benchmarks/h200/predictions_vllm-bf16_confirmatory.json` (n = 1,277), the **gold**
decision distribution of the frozen confirmatory partition is:

| Gold class | n | share |
|---|---|---|
| `CALL` (tool call expected) | 453 | 35.5% |
| `UNSUPPORTED` (cannot answer) | 453 | 35.5% |
| `CLARIFY` (asking expected) | 371 | 29.1% |
| **`ANSWER` (direct answer expected)** | **0** | **0.0%** |

The same file records the **predicted** decision counts: `ANSWER` 28, `CALL` 469, `CLARIFY` 510,
`UNSUPPORTED` 270. The model answered directly 28 times (2.2%) on a population where a direct answer is
never the correct decision. Every one of those is a scoring failure, and a gate over this population
implicitly rewards never answering.

The consequence is already written down in code at `src/opengrad/promotion/tool_use_policy.py:33-37`:
*"The frozen behaviour set contains zero ANSWER examples, so `no_call_accuracy` was 0.0 for every model
including B0, and its 0.40 floor rejected every candidate unconditionally."*

Three vocabularies are in use for the same four classes, and Study 002 treats that as part of the defect:

| Source | Names used |
|---|---|
| `docs/PROMOTION_POLICY.md:45-52` | `ANSWER`, `TOOL_CALL`, `REFUSE`, `CLARIFY` |
| `src/opengrad/promotion/tool_use_policy.py:52-57` | `CALL`, `ANSWER`, `CLARIFY`, `UNSUPPORTED` |
| `docs/EVALUATION.md:14` (prose) | `tool_call`, `request_for_info`, `cannot_answer`, `ANSWER` |

**Canonical enum for Study 002:** `{ANSWER, CALL, CLARIFY, UNSUPPORTED}`, with a required alias table in
[15](15-PROVENANCE-VALIDATORS.md) mapping `TOOL_CALL→CALL`, `REFUSE→UNSUPPORTED`,
`request_for_info→CLARIFY`, `cannot_answer→UNSUPPORTED`, `tool_call→CALL`. A validator rejects any
artifact using an unmapped mode name, because a metric keyed on `no_call_accuracy` that silently
measures nothing is exactly how this defect survived to promotion.

## Populations

| id | Population | Role | Scoring discipline |
|---|---|---|---|
| `P-DEV` | development split | checkpoint selection, per seed | may be scored repeatedly |
| `P-CONF` | confirmatory split, all four modes | confirmatory claims | **scored once per arm** |
| `P-UNANS` | curated genuinely-unanswerable set | refusal correctness (H6) | scored once, sealed |
| `P-SEALED` | never-read reserve | Study 003/004 or a future replication | not read in Study 002 |
| `P-DET` | hand-labelled detector sample | `HEURISTIC_REGEX_v2` precision/recall | labeller agreement reported |
| `S-*` | frozen external benchmark files | sentinels ([08](08-SENTINEL-SPEC.md)) | unchanged, frozen upstream |

`P-CONF` is the population the gate reads. It **must** contain all four gold modes, and that requirement
is asserted rather than described.

## Reusing the existing partition

`reports/evaluation/behavioral-heldout-v2-partition.json` is a real artifact and Study 002 reuses it
rather than reinventing it:

- algorithm `stratified_by_expected_decision__sha256(seed||example_id)_descending`, confirmatory
  fraction 0.35;
- `dev` 2,373 examples, fingerprint `88a56821edfc8614285846e8bef60cf8aced124ce3b5a2d4054e8344618c9980`;
- `confirmatory` 1,277 examples, fingerprint `d6d1e394a89b5ec8b9ed41ef233d752ff50f69abe41a6e06e34878c1088f32ba`;
- parent manifest `reports/evaluation/behavioral-heldout-v2.manifest.json`, sha256
  `8bb6ad2e37613c996476bb734b93ef792eae807ec2cccf6b200d96a47332c640`;
- population 3,650 (3,652 distinct before two quarantined items were removed,
  `docs/evaluation/CHECKPOINT_SELECTION.md:25`).

What it **cannot** be reused for is a four-mode claim, because its population contains no `ANSWER` gold.
`behavioral-heldout-v2.manifest.json` describes its `when2call-mcq` split as *"CALL/ANSWER/CLARIFY/
UNSUPPORTED behavioral decisions"*, while the materialized gold labels carry three of those four classes.
Study 002 does not resolve that discrepancy by assertion: it **counts** the gold labels of every
population it scores and prints the table above, and a population whose stated coverage and measured
coverage disagree is a `FAIL_NONVACUOUS` at [16](16-GPU-READINESS-GATE.md).

## Building `P-CONF` for four modes

`P-CONF` is reconstructed as the union of:

1. **All three covered modes from the existing confirmatory side** (1,277 examples: `CALL` 453,
   `UNSUPPORTED` 453, `CLARIFY` 371) — reused so Study 002 keeps continuity with Study 001's frozen
   numbers instead of replacing the evidence base it is explaining.
2. **An `ANSWER`-gold strata set**, drawn from a source that is (a) disjoint from every Study 002 training
   corpus by construction, (b) never used for checkpoint selection or hyperparameter choice, and (c)
   materialized through `opengrad.data.materialize` so its `content_hash` is produced by repository code
   rather than hand-written. `ANSWER`-labelled records already exist in the repository's own materialized
   corpora — `manifests/quantization/m1_v2_qad_recovery_v1.jsonl` carries `CALL` 141, `ANSWER` 140,
   `CLARIFY` 100, `UNSUPPORTED` 100 — so this is a **hold-out selection problem, not a data-availability
   problem**. The chosen source is named in the manifest and its provenance chain recorded (G4).

The `ANSWER` strata set is built to be balanced against the other three modes in prompt length,
`context_bucket` and source, because a four-mode comparison whose `ANSWER` rows are systematically more
or less difficult would confound the very endpoint it exists to measure.

## Coverage and resolvability rules

These two rules replace "the set is small, so be careful" with something a validator can check.

**C1 — Coverage.** Every mode in `P-CONF` has `n > 0`. Any mode with `n = 0` fails with
`FAIL_NONVACUOUS` and yields `NOT_EVALUABLE` for every dimension keyed on that truth class. This
re-tests L1 directly: the v3 fix made an unmeasurable dimension *skip* (`tool_use_policy.py:171-172`),
which restored satisfiability but left the blind spot in place. Study 002 requires the population to be
fixed **and** keeps the skip-path so that a future regression is visible rather than silent.

**C2 — Resolvability.** Every reported row prints `n`, the 95% Wilson interval and the **resolvable
margin** = 2 × the worst-case half-width (the normal-approximation half-width `z·sqrt(0.25/n)`, slightly
larger than Wilson's and so conservative; corrected 2026-09-24, `reports/ERRATA.md` §19). The margin is quoted at the worst case `p = 0.5`, so it is a function
of `n` alone and cannot be flattered by an observed rate near 0 or 1; the observed-rate margin is printed
alongside as a second, non-binding figure. A comparison whose observed margin is smaller than its
resolvable margin is reported `WITHIN_NOISE`, may not support a claim, and may not enter a gate decision.
Worked figures on the populations actually in use, so the arithmetic is checkable:

| Population | n | worst-case half-width | resolvable margin |
|---|---|---|---|
| full frozen set (one mode) | 1,277 | 2.74pp | **5.5pp** |
| `CALL` in `P-CONF` | 453 | 4.60pp | **9.2pp** |
| `UNSUPPORTED` in `P-CONF` | 453 | 4.60pp | **9.2pp** |
| `CLARIFY` in `P-CONF` | 371 | 5.09pp | **10.2pp** |
| any mode at the hard floor | 200 | 6.93pp | **13.9pp** |
| a mode built to resolve 8pp | 601 | 4.00pp | **8.0pp** |

A mode also carries a hard floor of **n ≥ 200**: below it the mode is `UNDER_POWERED`, its dimensions are
`NOT_EVALUABLE`, and the study claims nothing on that mode. `ANSWER` at n = 0 is therefore not a weak
measurement; it is an absent one, and is reported as absent.

The `ANSWER` strata set must supply enough examples for the smallest margin it is used to test.
[11](11-THRESHOLDS.md) fixes that margin; the strata size is derived from it by C2 and printed in the
manifest, so a reader can verify that the partition can resolve what the study claims on it. Because the
existing three-mode rows already sit between 9.2pp and 10.2pp, **no claim in this study uses a margin
below 10pp**, and the `ANSWER` strata set is sized to resolve 8pp or better.

> **Status, 2026-09-20.** C1 and C2 are now machine-checkable: `V1 mode_coverage` and
> `V12 resolvable_margin` (`src/opengrad/verification/population_validators.py`) enforce them, and the C2
> arithmetic (`src/opengrad/verification/resolvability.py`) reproduces this section's worked figures. The
> `ANSWER` strata set itself is **not built**: its source is the study owner's decision (open item 2 of
> [20](20-CLOSURE-REPORT.md)), and it must satisfy both the `n ≥ 200` floor and the 8pp sizing above — the
> one source this section cites carries 140 `ANSWER` items, below the hard floor.
>
> A survey of the repository's labelled pools finds **no disjoint pool at or above the `n ≥ 200` floor**.
> The frozen held-out (`results/quantization/frozen_behavioral_eval_v1.jsonl`, 3,650) and the frozen confirmatory prompts
> (`frozen_prompts_confirmatory_v1.jsonl`, 1,277) carry **`ANSWER` 0**. The largest non-training
> `ANSWER`-gold pools found are the QAD recovery corpus (140), its local calibration sibling (100) and
> `m1_v2_imatrix_calibration_v2` (300), all **training-derived** and drawn from the same four sources, so no
> non-training pool reaches the floor; P-DET-COVERAGE-v1 and -v2 hold 44 and 132
> model-consensus `DIRECT` items, which are classifier-validation and provisional, not model gold. The
> only large `DIRECT` pools — about 43,300 first answers in canonical-v2-final that frozen classifier v1
> *predicts* `DIRECT` (33,831 of 37,512 with no tools offered, 9,455 with tools; predictions, not gold) — are the
> **training corpora** themselves and are barred by the disjointness rule. Against C2 this section asks
> for `n ≈ 601` to resolve 8pp (385 for the 10pp floor this study uses), so the `ANSWER` stratum needs a
> **new held-out source**, or a documented decision to accept `UNDER_POWERED`; selection from what already
> exists does not clear the floor.

## One-shot discipline on `P-CONF`

`P-CONF` is scored **once per arm**, on a checkpoint chosen in advance, exactly as
`docs/evaluation/CHECKPOINT_SELECTION.md:35-37` requires: *"Evaluating many checkpoints on it and
reporting the best repeats the selection problem on the set that exists to avoid it."* Enforcement:

- the prediction artifact records `partition` and the partition fingerprint; a second scoring pass by the
  same arm on the same partition is a `FAIL_ONE_SHOT` error;
- a re-scoring pass forced by a defect is a **replacement**, recorded with both run ids retained, and it
  is noted in the write-up (`#62` is the defect this prevents);
- selection uses `P-DEV` only, and the selected step per seed is recorded.

## Disjointness and contamination

- `P-DEV`, `P-CONF`, `P-UNANS`, `P-SEALED` are pairwise disjoint by construction, and the partition
  artifact carries the example ids so the assertion is mechanical.
- Every population is disjoint from every training corpus **by construction**: the training releases
  exclude the held-out manifest (`CHECKPOINT_SELECTION.md:22`).
- The `ANSWER` strata set is checked against the training corpus by exact-text and near-duplicate
  fingerprint before use; a collision removes the item and the removal is counted and reported.
- `P-UNANS` is curated for genuine unanswerability (a question with no determinable answer, or one
  requiring information absent from the prompt), and its entries carry a labeller-agreement record
  ([07](07-METRIC-SPEC.md)). An "unanswerable" set whose items are merely open-ended would make
  refusal correctness uninterpretable, which is why agreement is a stop rule in
  [03](03-PREREGISTRATION.md).

## What this spec does not do

It does not re-partition the frozen 1,277 or re-score any Study 001 claim. The frozen partition stays as
it is, with its fingerprint, so that Study 001's record remains readable and its three-mode numbers
remain comparable. Study 002 adds a mode to the confirmatory population and freezes the result as a new
partition artifact with its own fingerprint and its own gold-count table.

> **Status, 2026-09-25: `P-CONF-v1` is built** (owner decision, option A). `src/opengrad/verification/pconf.py`
> writes `reports/study-002/pconf-v1/`: the new partition artifact this section asks for, with its own
> fingerprint and gold-count table (`CALL` 453, `ANSWER` 1,055, `CLARIFY` 371, `UNSUPPORTED` 453), and the `ANSWER`
> items as evaluation records. The frozen 1,277 are reused unchanged.
> - **The `ANSWER` source:** [41](41-ANSWER-STRATA-AMENDMENT.md), as amended by
>   [42](42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md).
> - **The strata stay apart:** `ANSWER-natural` 284 meets the floor only, and `ANSWER-constructed` 771 resolves 7.1 points.
> - **Balance** is measured in the artifact, not enforced (41 §2).
> - **Collisions:** none with the development side, P-DET-COVERAGE, or the When2Call held-out text.
> - **Not scored.** The evaluator must learn to load it first.

