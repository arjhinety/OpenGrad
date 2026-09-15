# 05 — Seed and reproducibility policy

G11: *"Seeds and intervals are planned, or the single-seed limit is stated in advance"*
(`docs/research/GUARDRAILS.md`). Study 001 reported point comparisons between checkpoints with no noise
band, and its promotion gate had no margin-versus-noise test (`#2`: M1-v2 *"fails v3, the gate M0
failed"*, and was promoted under a v4 committed after M0's confirmatory results). This document fixes
the seed policy that makes a margin-versus-noise test possible.

## Seed count

**`k = 3` seeds per arm, identical values across all arms.**

| Seed slot | Value | Role |
|---|---|---|
| `s0`, `s1`, `s2` | seeded `0`, `1`, `2` at the trainer's data-order/init-state entry point | every arm and both repeated controls |

Every arm uses the **same three values**. Arms are compared seed-wise (paired), never by comparing two
arm-level means averaged over different seed sets — a mistake that inflates apparent separation and is
invisible in a mean.

`k = 3` is a declared floor, not a sufficient sample for fine effect sizes. It buys a seed interval for
the primary endpoint ([10](10-STATISTICS-PLAN.md)) and nothing more. Any claim requiring two arms to be
separated by less than the observed seed spread is **not made**: it is reported as
`MECHANISM_INCONCLUSIVE` ([02](02-RESEARCH-QUESTIONS.md)).

## What a seed controls

| Controlled by the seed | Not controlled by the seed |
|---|---|
| Example order and batch composition | Corpus contents |
| Dropout and any stochastic layer state | Hyperparameters, schedule, step budget |
| Data-loader shuffling, any mixture sampling | Evaluator version and decoder settings |
| Any stochastic tie-break among tied DEV scores | Prompts and tokenizer |

If any trainer component consumes randomness outside this list, it is enumerated in the run record and
its seed source is named. An unnamed stochastic source is a blocking check in
[16](16-GPU-READINESS-GATE.md).

## Decoding determinism at evaluation

Evaluation runs at **greedy / temperature 0** so that no seed enters the score. Where a benchmark
requires sampling, the sampling parameters are pinned in the manifest and disclosed in the table's
caveat field ([07](07-METRIC-SPEC.md)). IFEval strict is a format-and-decoding-scale endpoint, so
leaving decoding unpinned during evaluation would confound exactly what this study must separate
([09](09-BENCHMARK-PLAN.md)).

## What the two repeats measure

| Repeat | Configuration | Measures |
|---|---|---|
| `REP-A` | `C0`, seed `s0`, re-run | **nondeterminism floor**: kernel/library nondeterminism with the seed fixed. Noise no seed interval can explain away. |
| `REP-B` | `A_M0` re-scored under the Study 002 evaluator | **evaluator reproducibility**: whether re-scoring the frozen checkpoint reproduces the Study 002 anchor |

`REP-A` is the number against which every threshold margin is compared ([11](11-THRESHOLDS.md)). If
`REP-A`'s residual is larger than a proposed margin, that margin is smaller than noise and the
threshold is rejected at pre-GPU review.

## Reporting rules

1. Every arm reports **all three seed values**, not just a mean. The mean appears *alongside* the three,
   plus the seed interval.
2. A single-seed result may be reported only with the literal label `single-seed (k=1) — no interval`.
   It may not enter a gate decision and may not appear in an abstract.
3. The number of seeds is reported per row, including when it differs. A table whose rows have
   different `k` prints `k` per row rather than in a footnote.
4. Seed-level records are retained as artifacts ([12](12-ARTIFACT-RETENTION.md)) so a reader can
   recompute an interval from the release rather than from the prose.

## Reproducibility fingerprint

Every run record carries, at minimum: study id, arm id, seed, corpus version and **corpus fingerprint
read from the artifact** (G4), tokenizer hash, chat-template hash, base-model revision, code revision,
trainer-config hash, evaluator version, generation config, supervised-token tally, example count,
provider and `device_class` ([13](13-HARDWARE-AGNOSTIC-EXECUTION.md)), run status and timestamps.
`<!-- -->provenance.schema.json` already carries `run_id`, `status`, `compute_provider` and `hardware`;
the remaining fields are required by [15](15-PROVENANCE-VALIDATORS.md).

The fingerprint is read from the artifact that was actually trained on, never retyped. Study 001's
`#80` is the sharpest instance of the opposite: an audit figure of 21,749 of 217,903 (10.0%) with
When2Call at 7,490 of 14,829 (50.5%) against a corpus that was not the published one — *"The audit read
normalization-v1: held-out rows, BUTTON, LoopTool and the v1 Glaive adapter"* — while the same detector
on **published** Canonical-v2 gives 18,114 of 173,237 (10.5%) and When2Call 4,038 of 6,505 (62.1%).

## Prohibited practices

Declared here so that a violation is a violation rather than a judgement call:

- Dropping a seed because its result is inconvenient, or reporting a `k` that varies per arm without
  saying so.
- Reporting the best seed, the best checkpoint, or a max over checkpoints as an arm's result. Study 001
  did this: `#62` — *"Best of 4 checkpoints, chosen on the same set; the final checkpoint scores 0.5292.
  Unlike the M1-v1 row, it isn't labelled best."*
- Changing `k` after results are visible.
- Re-running until a threshold is crossed. Each arm's seed set is fixed; a re-run forced by a defect is
  recorded as a replacement run, with both run records retained.
- Comparing a Study 002 mean to a Study 001 frozen number as though the two shared a `k` and an
  evaluator (G9; `#9`, `#53`, `#83`, `#84`).

## Checkpoint selection

Selection among checkpoints within an arm is made on **DEV only**, by the frozen rule in
[06](06-SPLIT-SPEC.md), per seed, and the selected step for each seed is recorded in the run record. A
tie is broken toward the lower step index, deterministically — never by a CONFIRMATORY read.

## Seed policy across hardware

The same three seed values are used on every provider (`nvidia`, `amd`, `cpu`). Bit-exact reproduction
across providers is **not** claimed and is not required for this study's claims, which are about
corpus-level mechanism. Where a claim would depend on cross-provider identity, it is out of scope and
says so ([13](13-HARDWARE-AGNOSTIC-EXECUTION.md), [14](14-HETEROGENEITY-POLICY.md)).