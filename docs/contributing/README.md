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

## Development skills

The OpenGrad development skills ship with the code as the `opengrad` Claude Code plugin
([`plugins/opengrad/`](../../plugins/opengrad/README.md)). They are task instructions for coding agents, and a
checklist for people. Each covers one area of the repository: data, annotation, experiments and readiness,
training, evaluation, promotion gates, releases, registries and provenance, research guardrails.
`opengrad-development` is the entry point.

Install them with `/plugin marketplace add arjhinety/OpenGrad` and `/plugin install opengrad@opengrad`. Inside
this repository, the marketplace is registered for you once you trust the folder.

`tests/skills/test_skills.py` keeps them current. CI fails when:
- a package, console script or CLI subcommand has no skill covering it;
- a skill cites a path that no longer exists;
- the plugin manifests disagree.

A change that alters a workflow updates its skill in the same commit; see `opengrad-skills-maintenance`.
