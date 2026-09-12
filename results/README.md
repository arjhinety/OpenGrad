# The results namespace

This directory holds the **derived experiment index**. It is not a store.

```text
results/
  registry.jsonl     derived: a rebuildable projection of authoritative run artifacts
```

## The hierarchy

```text
Authoritative — owns the value, never regenerated from anything else
    runs/<experiment_id>/experiment.json     identity, configuration, dataset hashes,
                                             model/tokenizer revisions, environment, status
                                             (written only by ExperimentStore)
    runs/<experiment_id>/eval/               per-checkpoint metrics, predictions, benchmark
                                             artifacts, curve.json
    runs/central_ledger.jsonl                append-only lifecycle transitions

Derived — a projection, safe to delete
    results/registry.jsonl                   one discovery/summary row per experiment
    results/final_campaign_verdict.json      capability-campaign verdict; regenerate, never edit
```

## Benchmark campaigns are a separate namespace

`registry.jsonl` projects the **training** experiment store (`runs/<experiment_id>/`). Benchmark
campaigns are not training experiments and have no `runs/` identity, so they are **not** projected
into it — adding rows there would make it a projection of two different things.

The capability-regression campaign lives under `results/benchmarks/` with its own hierarchy:

```text
Authoritative — owns the value
    results/benchmarks/h200/capability_v1/evidence/*.jsonl   per-example: input, raw output,
                                                             parsed output, expected answer, score,
                                                             failure class, refusal flag, tokens
    results/benchmarks/h200/capability_v1/<STAGE>/*_scores.json   per-benchmark aggregates
    results/benchmarks/h200/capability_v1/gpu_runs.jsonl     append-only, one row per GPU run
    results/benchmarks/checkpoint_ladder.json                stage identities and hashes
    results/benchmarks/h200/PRESERVED_STATE_v1.json          immutability manifest

Derived — regenerate, never hand-edit
    results/benchmarks/capability_findings.jsonl             one row per benchmark × checkpoint
    results/benchmarks/h200/capability_v1/final_campaign_audit.json   full recomputation
    results/final_campaign_verdict.json                      campaign verdict + pointers
```

Regenerate the derived artifacts with:

```bash
python scripts/audit_campaign_final.py        # recompute every metric from per-example rows
python scripts/build_capability_evidence.py   # evidence files + findings ledger
python scripts/build_regression_analysis.py   # transition matrix, deltas, intervals
python scripts/build_final_verdict.py         # verdict (use --verify to detect drift)
```

`build_final_verdict.py --verify` fails if the committed verdict has drifted from the artifacts,
which is what keeps it a projection rather than a second source of truth.

`results/registry.jsonl` exists so that discovering experiments, summarising them, comparing them
and rendering reports do not require walking the filesystem and re-parsing every artifact. It is
regenerated from the three authoritative sources above and never written from anywhere else.

**Delete it and nothing is lost.** Authoritative state is untouched by its absence, and:

```bash
opengrad results rebuild-registry
```

recreates it byte-for-byte. That property is tested, not assumed.

## Commands

```bash
opengrad results rebuild-registry     # regenerate from authoritative artifacts
opengrad results validate-registry    # report differences; repairs nothing
opengrad results show                 # print the materialized summary
```

## What a row contains

One row per experiment, ordered by `experiment_id`, with sorted keys: status, model and
tokenizer revisions, dataset fingerprints, a training-config fingerprint, timestamps, the
evaluated checkpoints with their authoritative metrics, the promotion and rejection state read
from the ledger, and `provenance` paths back to the artifacts every value came from.

Two things it deliberately does **not** contain:

* **No "best checkpoint".** `docs/CHECKPOINTS.md` states that the newest checkpoint is never
  automatically "best" and that "best" requires a formal promotion policy. Selecting a maximum
  over `call_f1` in the index would reintroduce the metric-only gate the promotion-policy work
  removed. The index records the **promoted** checkpoint when the ledger says one was promoted,
  otherwise the **latest evaluated** one, and names which rule applied in
  `headline_checkpoint_selection`.
* **No invented suite names.** Evaluation artifacts record the held-out *manifest* and a `kind`,
  not a suite label, so the index reports those identifiers verbatim in `eval_manifests` and
  `eval_kinds` rather than mapping them onto a name they never carried.

## Synchronisation

`ExperimentStore` refreshes the index after an authoritative transition commits — experiment
creation, status change, or record registration. The ordering is one-directional by design:

```text
write experiment.json  ->  append ledger  ->  refresh derived registry
```

Never the reverse. A registry failure cannot fail an authoritative write: `refresh_registry()`
returns `False` and the record and ledger remain correct, because the index is recoverable and
they are not. Writes are atomic (`registry.jsonl.tmp` -> `fsync` -> `os.replace`), so an
interrupted build cannot leave a partially written projection.

## Validation

`validate-registry` compares the materialized file against a fresh projection and reports two
distinct kinds of finding rather than returning a boolean:

```text
DRIFT      the registry disagrees with authoritative state -> rebuild it
INTEGRITY  authoritative state itself has a gap -> the registry mirrors it faithfully,
           and the gap is not the registry's to fix
```

Drift codes include `REGISTRY_MISSING`, `REGISTRY_NOT_BUILT` (a zero-byte placeholder while
experiments exist), `EXPERIMENT_MISSING_FROM_REGISTRY`, `REGISTRY_ROW_WITHOUT_EXPERIMENT`,
`DUPLICATE_EXPERIMENT_ID`, `MALFORMED_REGISTRY_ROW`, `FIELD_MISMATCH` and
`PROVENANCE_PATH_UNRESOLVED`.

Validation never repairs anything. It exits non-zero on drift, and `INTEGRITY` findings alone do
not fail it — a faithfully mirrored gap in the underlying evidence is not a registry defect.
