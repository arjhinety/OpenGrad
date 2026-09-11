# Experiments

Runs receive stable IDs and append-only records. Several real runs now exist — the B0 baseline and the M0/M1 post-training interventions — recorded under `runs/`; the Phase-0 "no executable run" framing is superseded.

Each experiment's authoritative record is `runs/<experiment_id>/experiment.json` (identity, configuration, dataset hashes, revisions, environment, lifecycle status), with its evaluation artifacts under `runs/<experiment_id>/eval/` and its lifecycle transitions appended to `runs/central_ledger.jsonl`. These are written only by `ExperimentStore`.

[`results/registry.jsonl`](../results/README.md) is a **derived index** over those artifacts — one discovery row per experiment, rebuildable byte-for-byte with `opengrad results rebuild-registry` and never a place that owns experiment state. Use `opengrad results validate-registry` to check the projection against the artifacts. See [the results namespace](../results/README.md).
