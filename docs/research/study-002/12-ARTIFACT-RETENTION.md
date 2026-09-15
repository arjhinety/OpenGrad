# 12 — Artifact retention

A claim is only as durable as the artifact behind it, and Study 001's audit shows what happens when the
artifact is missing or superseded: a claim keeps its number while its basis changes underneath
(`#67` — *"The live path raises NotImplementedError; M2 was closed as NOT RUN"*; `#80` — an audit figure
computed from a corpus that was not the published one).

## Required artifacts per arm

| Artifact | Contents | Retention |
|---|---|---|
| `run.json` | study id, arm, seed, corpus version + **fingerprint read from the artifact**, tokenizer and chat-template hashes, base revision, code revision, trainer-config hash, supervised-token tally, example count, provider, `device_class`, status, timestamps | permanent, committed |
| `ledger.jsonl` | append-only event log (`TRAINING`, `TRAINED`, `INVALID`, `RECORD_ANNOTATED`, …), never edited | permanent, committed |
| `selection--dev.json` | per-seed chosen step, the DEV rule version, the number of checkpoints compared | committed |
| `metrics--<partition>.json` | every metric with `n`, denominator, population policy, metric version | committed |
| `predictions--<partition>.jsonl` | per-item: `example_id`, gold, observed, parser status, prompt hash, `raw_output`, `finish_reason` | committed or released with a manifest hash |
| `confusion--<partition>.json` | the four-mode matrix, `{ANSWER, CALL, CLARIFY, UNSUPPORTED}²` | committed |
| `sentinel--<id>.json` | per-item outcome + failure class, plus the truncated-rate tally | committed |
| `verdict.json` | policy version, decision, failed dimensions, `checks[]`, `unmeasured_dimensions[]`, `unmeasurable_dimensions[]`, `measurable_dimensions[]` — the existing shape from `tool_use_policy.py:215-232` | committed |
| `census.json` | `discovered`, `checked`, `passed`, `failed`, `blocked`, `skipped[]` with reasons | committed |
| `cost.json` | `gpu`, `gpu_hourly_usd`, `container_seconds`, `usd`, `arm`, `seed`, `tier`, `disposition` (`RETAINED`/`SUPERSEDED`/`REPLACEMENT`/`DISCARDED`), plus the `envelope_*` fields kept separate from any credit balance | committed, one entry per run |
| `cost_ledger.json` | the rolled-up ledger: per-stage totals, `retained_runs_usd`, `superseded_runs_usd`, `infrastructure_waste_usd`, `failed_launches[]` | committed, append-only |

Cost is an artifact, not a footnote. Three rules make the recording a mechanism rather than an intention:

1. **Every run records its own rate.** `gpu_hourly_usd` and the provider travel with each entry, because the
   same work priced on an A100 80GB at $1.79/GPU-hour and on an H200 at $4.54/GPU-hour differs by 2.54× — and
   a cost comparison that omits the rate is meaningless ([16](16-GPU-READINESS-GATE.md)).
2. **Discarded work is retained with its cost.** `SUPERSEDED` and `DISCARDED` entries keep their
   container-seconds and their `usd`. A ledger that omits them makes the next study look cheaper than this
   one's history was — the 768-token MMLU-Pro pass being the existing case, counted in full at $3.8864 and
   3,081.7 container-seconds in `results/benchmarks/h200/capability_v1/cost_ledger.json`.
3. **An envelope is labelled as an envelope.** `envelope_is_not_a_balance: true` and
   `REMAINING_CREDIT_BALANCE: NOT_QUERYABLE` are preserved in any Study 002 cost artifact, so planning
   arithmetic can never be mistaken for available credit. This repository already learned that lesson: *"The
   two diverged badly in this run: planning continued against the envelope while actual remaining credit was
   far lower. Do not size a run from this field."*
| `torch_seed`, `env_freeze.txt` | library versions and CUDA/ROCm runtime string | committed |

`predictions--*.jsonl` is the artifact that makes every other row checkable, and its absence is a hard
failure rather than a caveat. `manifests/quantization/ptq_phase_closure_v1.json` is the precedent for how to
handle its absence honestly: *"aggregate metrics only; per-example vLLM predictions were never preserved,
so no per-example vLLM-to-llama.cpp agreement is claimed anywhere in this phase."* The rule Study 002
adopts is the same one applied *before* any run rather than after: **if predictions are not retained, no
per-item claim is made.**

## Corpus artifacts

The training corpus is retained as a versioned artifact with: the source list and pinned revisions, the
materialization command, the `content_hash` **produced by repository code**, the record count, the
per-source counts, the detector version and its flagged counts, and the answerability labels with their
labeller-agreement record.

Study 001's Canonical-v2 stays published and unchanged; the corrected corpus is a **new version** with its
own fingerprint, exactly as `docs/research/STUDIES.md:70-71` requires:
*"The published Canonical-v2 corpus is not redistributed. Study 002 produces a Study 002 corpus version, with
its own fingerprint, and a mapping from the 18,114 flagged records to their dispositions."*

That mapping is itself a required artifact: for each of the 18,114 records flagged in Canonical-v2, its
disposition — relabelled, target-corrected, left unchanged with a reason, or removed — and the arm in which
each disposition appears. It is the artifact that lets a reader reconstruct `R1`, `R2` and `R3` from the
published corpus plus one file.

## Flags and records that must survive

- **Invalid and superseded runs are retained, not deleted.** `runs/central_ledger.jsonl` shows the pattern:
  a `TRAINED` event, then an `INVALID` event *"supersedes: TRAINED"* with its evidence and reason, with the
  original retained because the ledger is append-only. Study 002 keeps that invariant: a run that was
  invalidated keeps its record and gains an annotation.
- **Discarded passes are retained with their reason and their cost**, including the 768-token MMLU-Pro
  pass, whose whole point is that it was thrown away.
- **Replacement runs keep both records**, so a reader can see what changed and why.

## Retention horizon

- Everything listed above is committed to the repository or released with a manifest carrying its sha256.
- Per-example predictions are retained indefinitely; they are the only way the resolvable margins in
  [11](11-THRESHOLDS.md) can be recomputed independently.
- Large binaries (checkpoints, quantized artifacts) follow the existing release convention: a released
  artifact plus a manifest hash in-repo, so the repository remains cloneable.
- Nothing is pruned for space without a recorded decision naming the artifact and the reason. Study 001's
  cost ledger already tracks which results were retained per run (`results_retained`), and Study 002 uses
  the same field so that "not retained" is visible in the ledger rather than discovered by a reader.

## Verification

`[15](15-PROVENANCE-VALIDATORS.md)` defines the validators that read these artifacts, and
`[16](16-GPU-READINESS-GATE.md)` refuses to spend GPU time until the retention paths exist and are
exercised on a smoke run. A retention policy that has never been executed is a claim about a scaffold,
which is `#66`/`#67` again.