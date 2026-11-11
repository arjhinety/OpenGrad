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
  snapshot, experiment preflight, renderability yield, supervision composition,
  `model_components_validation` and `determinism_declared`;
- `ready_for_dpo` adds: GPU boundary, real B0, baseline artifacts, experiment preflight, the DPO contract,
  `model_components_validation` and `determinism_declared`.

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

1. Write `_<name>_state(root, raw) -> tuple[bool, str, str | None]` in `src/opengrad/readiness_states.py`
   (constants and file/config helpers are in `readiness_contracts.py`; `readiness.py` re-exports both). It returns
   ok, a human detail and an error code, and is pure given the root, so tests use `tmp_path`.
2. Call `add(name, "PASS"|"FAIL", detail, code)` in the phase function of `readiness()` it belongs to
   (`_training_gates` for SFT/DPO gates; the phases run in gate order, which the JSON output preserves). For configs the gate does not apply to,
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

## Run records

Runs are recorded by `src/opengrad/runrecords/` (owner's rule, 2026-11-11), beside the run store and never inside
it:
- `opengrad train` and `opengrad evaluate` record themselves, in process. The log captures what is written through
  `sys.stdout`, `sys.stderr` and their `.buffer`. It does not capture output that C extensions (CUDA, NCCL) or child
  processes write straight to the file descriptors.
- Launch GPU training and evaluation through the launcher, `python -m opengrad.runrecords run --kind training --
  opengrad train …`, so CUDA, NCCL and worker-process (vLLM) output is in the log too. The launcher tells the
  child it is recorded (`OPENGRAD_RUN_RECORD_PARENT`), so the in-process record steps aside and each run has one
  record.
- Record any other command, with all of its output, as a child process:
  `python -m opengrad.runrecords run --kind <training|evaluation|freeze|script|annotation|ci> -- <command>`.
- Each run gets a git-ignored folder, `.run-records/<run_id>/` (`OPENGRAD_RUN_RECORDS_DIR` overrides it; the test
  suite uses that). It holds the full output, a start record and a summary. A start with no summary reads as
  `INCOMPLETE`.
- The summary holds:
  - the command and any error text, redacted: secrets, credential-named flags and fields (a name part such as
    `token`, `key`, `password` or `auth`, split at `-`, `_`, `.` and camelCase, or an all-caps name ending in one;
    `--no-…` and `--use-…` switches keep the next argument), URL credentials, bearer tokens, Google keys and absolute paths;
  - the commit and the tracked-diff hash;
  - every installed distribution and the `uv.lock` hash;
  - the GPU, as reported by `nvidia-smi` or `rocm-smi`;
  - allowlisted environment variables;
  - the exit status, the duration and the log's sha256.

  Some forms are not redacted, such as a password attached to a short flag (`-pSecret`), so don't pass
  credentials on a command line.
- If a record cannot be started or finished, the run still goes ahead and a warning is printed on stderr.
- `python -m opengrad.runrecords pending` lists runs that failed, were interrupted or were killed and have no
  disposition. Dispose of each with `dispose RUN_ID --upstream UP-NNNN`, naming its entry in `docs/upstream/`, or
  with `--reason`. `export` then copies valid summaries into `run-records/` for committing.
  `tests/runrecords/test_committed.py` fails CI on a committed run that did not pass and has no disposition, or a
  summary with a secret, an email address, a home path or an absolute path.
- `upload --repo OWNER/NAME`, on the owner's word only, sends a durable copy of each log to an existing private
  dataset repository. It refuses any log holding a secret, an email address, a home path, a Google API key, a
  bearer token or URL credentials. Annotation logs stay local unless named with `--kind annotation`.
- Tests record only dummy child commands and stub `huggingface_hub` (UP-0030). Recording on a remote GPU machine
  is not yet verified.

## Sources of truth

`docs/EXPERIMENT_FOUNDATION.md` · `docs/TRAINING_LIFECYCLE.md` · `results/README.md` · `docs/architecture/repository.md` ·
`docs/research/study-002/16-GPU-READINESS-GATE.md`

## Keeping this skill current

When a gate is added to a `ready_for_*` set, or a new state artifact appears, update the gate-set list and the
state table in the same commit.
