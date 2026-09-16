---
name: opengrad-experiments-readiness
description: How OpenGrad experiments are configured, gated and recorded — experiment configs, opengrad preflight, opengrad readiness and its gate sets (ready_for_baseline / ready_for_sft / ready_for_dpo), opengrad gpu-smoke, the authoritative run store (runs/, ExperimentStore, ledgers) versus the derived results index, status/doctor/env capture, and the generated experiment status view. This skill should be used before launching any real run, when adding or editing an experiment config, when adding a readiness gate, and when reporting experiment state.
---

# OpenGrad experiments and readiness

## State: authoritative versus derived

- **Authoritative:**
  - `runs/<experiment_id>/experiment.json`, written only by `ExperimentStore` (`src/opengrad/experiments/store.py`);
  - `runs/<experiment_id>/eval/`;
  - the append-only `runs/central_ledger.jsonl`;
  - `runs/checkpoint_registry.json`.
- **Derived:** `results/registry.jsonl`, rebuilt with `opengrad results rebuild-registry`. Check drift with
  `opengrad results validate-registry`, and read it with `opengrad results show`.
- **Generated:** `docs/EXPERIMENT_STATUS.md`, written by `python scripts/reporting/generate_experiment_status.py`.
  `--check` fails when it is stale, and `tests/results/test_state_consistency.py` requires a byte-identical
  regeneration. Never hand-edit it; regenerate after any state change.

Never hand-write run artifacts or ledgers to "fix" state. Record the real transition through the store.

## Before any real run

```bash
opengrad preflight <config> --json        # config-level readiness
opengrad readiness <config> --json        # every gate, blocking list, ready_for_* flags
opengrad gpu-smoke [config] --json        # bounded real-model GPU boundary receipt
```

`opengrad train <config>` calls `readiness` and refuses a real SFT or DPO run unless `status == PASS` and
`ready_for_sft` / `ready_for_dpo` is true (`src/opengrad/agent_cli.py`). `--dry-run` is a CPU plumbing
mock: never evidence, never an experiment record.

**The gate sets** live at the end of `readiness()` in `src/opengrad/readiness.py`:
- `ready_for_baseline`: repository, config, revisions, template, evaluation manifest and materialization,
  contamination, leakage, disk, storage, parser, GPU probe;
- `ready_for_sft` adds: GPU boundary, real B0, baseline artifacts, training-data policy, dataset revision and
  snapshot, experiment preflight, renderability yield, supervision composition, and
  `model_components_validation`;
- `ready_for_dpo` adds: GPU boundary, real B0, baseline artifacts, experiment preflight, the DPO contract, and
  `model_components_validation`.

A gate that blocks a correct run is a finding about the gate. Record it; never relax the gate to pass.

## Experiment configs

- Live in `configs/experiments/`, validated by `ExperimentConfig` (`src/opengrad/experiments/schema.py`) and
  `registry/experiments.schema.json`.
- **Required pins:** `model.model_id` / `model_revision` / `tokenizer_revision`, `datasets.manifest_ids` and
  `datasets.hashes`, `reproducibility.seed` and `precision`, and explicit `trainer.tuning_method` (SFT) or
  `trainer.reference` (DPO). Nothing is inferred.
- **Existing configs describe runs that happened.** Do not edit them to mean something new; add a new config
  with a new experiment id and record the parent.
- **Study 002 arm configs** must declare `trainer.model_components: {vision: exclude, mtp: exclude}` (see
  `opengrad-training`).
- **Comparisons:** `opengrad experiment list`, `opengrad experiment show <id>`, and `opengrad experiment diff <a> <b>`
  to diff two experiments' configurations.

## Adding a readiness gate

1. Write `_<name>_state(root, raw) -> tuple[bool, str, str | None]` in `src/opengrad/readiness.py`. It returns
   ok, a human detail and an error code, and is pure given the root, so tests use `tmp_path`.
2. Call `add(name, "PASS"|"FAIL", detail, code)` inside `readiness()`. For configs the gate does not apply to,
   add a PASS that says so (the gate list stays stable).
3. Add the name to every `ready_for_*` set it must block.
4. Decide dormancy explicitly. A gate may be dormant until a config declares its input
   (`_renderability_state` is the pattern), but once declared, a missing input fails closed.
5. Test the pass case, each fail case, and fail-closed on a missing or stale input in `tests/config/test_readiness.py`.
   More gate-testing discipline is in `opengrad-promotion-gates`.

## Environment and diagnostics

- `opengrad status --json`: authoritative repository and experiment state.
- `opengrad doctor --json`: environment, tooling and on-device testing surfaces.
- `opengrad env capture`: environment capture (`src/opengrad/env_capture.py`) recorded into runs.
- `opengrad validate --json`: registries, with the same checks as `opengrad-validate`.
- `src/opengrad/hardware/probe.py`: the accelerator probe. `src/opengrad/config/validate.py` validates config
  files, and `src/opengrad/experiments/cost.py` estimates run cost.
- Integrations (the `integrations/opengrad-mcp/` MCP server) call `opengrad … --json` and never bypass these
  gates (`docs/AGENT_INTEGRATION.md`).

## Sources of truth

`docs/EXPERIMENT_FOUNDATION.md` · `docs/TRAINING_LIFECYCLE.md` · `results/README.md` · `docs/architecture/repository.md` ·
`docs/research/study-002/16-GPU-READINESS-GATE.md`

## Keeping this skill current

When a gate is added to a `ready_for_*` set, or a new state artifact appears, update the gate-set list and the
state table in the same commit.
