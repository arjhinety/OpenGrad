# Contributing

How to add or challenge a result. Showing that something fails to reproduce is valuable research.

| Document | Covers |
|---|---|
| [experiment-pr.md](experiment-pr.md) | Proposing a new experiment |
| [reproduction-pr.md](reproduction-pr.md) | Reproducing an existing result, or reporting that you could not |
| [negative-result.md](negative-result.md) | Writing up a negative or null result |

Two project conventions are worth reading before your first pull request. Results are recorded in
[`runs/<experiment_id>/`](../../runs/) and indexed by a derived
[`results/registry.jsonl`](../../results/README.md) that can be rebuilt from them, so the run
artifacts are authoritative and the index is not. And a gate is never relaxed to make a run pass:
if a threshold blocks a correct result, that is a finding about the threshold, and it is recorded
rather than edited around.
