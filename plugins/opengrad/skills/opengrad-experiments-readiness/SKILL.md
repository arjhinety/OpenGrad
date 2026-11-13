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
- Every script and module entry point records itself (run-record stage 3, owner's rule). Its
  `if __name__ == "__main__":` block calls `run_script(__file__, main)` from `src/opengrad/runrecords/scripts.py`,
  never `main()` itself, so a test calling `main()` records nothing. A console command points at a
  `recorded_main` wrapper.
  - That module holds the policy table: each entry point's kind, and the non-zero exit codes it returns by design,
    each with a reason (a `--check` reporting stale output, a scan reporting findings). Such a run records as
    `PASS` with its code kept, so it needs no disposition (owner, 2026-11-13).
  - An entry point that reaches held-out data is listed in `HELD_OUT_ENTRY_POINTS`: its own code, or a module it
    imports, names a marker (`HELD_OUT_MARKERS`: `frozen_prompts`, `when2call_test`, `pconf-v1`, `answer-records`,
    `behavioral-heldout-v2-partition`, `confirmatory`, `punans`). The scan reads code only, not comments or
    docstrings. Its log is recorded but never uploaded; upload lists it as kept local by policy (owner,
    2026-11-13).
  - Upload is an allowlist (owner, 2026-11-13): a log uploads only when its entry point was reviewed and
    approved: `opengrad train` and `opengrad evaluate` (`APPROVED_SUBCOMMANDS`), the labelling runner and the
    fifteen pinned scripts (`PINNED_CONSOLE`). Any other log stays local and is listed as not approved, counted
    for exit 3, until its entry point's console output is reviewed and it is added. The checks below still apply
    to an approved entry point; they were the whole rule before, and three audits each found a new way past them.
  - Upload decides again from each record's command (`upload_decision`, whose docstring gives the order), so it
    also covers a script launched through `python -m opengrad.runrecords run`, one not converted yet, and
    in-process `opengrad train`/`evaluate`. A held-out entry point named anywhere in the command keeps the log
    local: a script path in any spelling (`./`, `//`, `..`, any case, absolute inside the checkout), one inside a
    shell string (`bash -c "python scripts/x.py"`), a module after `-m`. The run-record command wrapping
    another run cannot be traced. `opengrad baseline`, which runs a model on the held-out set with a default
    config its command does not show, is never uploaded (`HELD_OUT_SUBCOMMANDS`).
  - When a run starts, two checks are recorded before redaction. `held_out_inputs`: a marker in any argument (a
    path outside the repository included), or a `configs/` file (relative or absolute) that names one.
    `held_out_reach`: the marker-naming files the entry point loads, scanned on disk, so a script not tracked
    yet, or with local edits, that reaches held-out data stays local. A config counts by its keys, single-word
    values, and the path-like pieces of spaced values, not prose: a description saying "confirmatory IDs are
    excluded" reaches nothing. A config that does not parse, or is not YAML/JSON/TOML, is read whole. Upload
    checks the recorded arguments again. `opengrad evaluate` reaches data only through suite and benchmark
    configs, which a test keeps free of held-out names.
  - Fail closed (owner, 2026-11-13): a command it cannot trace to a repository entry point (a shell, pytest, an
    outside tool, `-c`, a `uv run` option, an interpreter option it does not know, `PYTHONSTARTUP`), or one naming
    a `configs/` file that is gone, stays local and is listed, counted for exit 3.
  - `export`, `dispose` and `upload` all write the committed summary through `_exportable`: a run whose log
    stays local, or whose entry point is pinned (below), keeps only the exception type of its error, since
    redaction does not remove test content.
  - `PINNED_CONSOLE` lists the entry points that upload although the scan reaches held-out data, each with the
    test that pins its console to status lines (ids, counts, codes, fixed text). Only one reaching held-out data
    through `classifier_input.py` alone (the When2Call questions loaded as an exclusion set, never printed) may be
    added, after its printed objects are checked (owner, 2026-11-13). `tests/runrecords/test_console_pins.py`
    snapshots the console calls of each such entry point and every file it loads; it does not see a change to
    what a pinned call prints (`summary()` returning new fields), so whoever changes such a function reviews the
    printed object again before regenerating the snapshot; exception text is not pinned either. These fifteen
    still pass the argument checks: given a held-out config, they stay local. A pinned entry point whose scan at
    start finds more than its review allowed stays local. The labelling runner
    (`scripts/run_external_annotation.py`) is pinned by `tests/annotation/test_run_external_annotation.py`: its
    logs go up as redacted annotation copies even on a P-UNANS session or config (owner, 2026-11-13); model
    transcripts stay in the batch folder.
  - `tests/runrecords/test_scripts_policy.py` checks every `__main__` block and console command. It keeps the
    held-out list equal to the scan (`held_out_reach`: imports, relative ones included, and `.py` files loaded by
    path; the policy files themselves are not followed) minus `PINNED_CONSOLE`. The allowlists of entry points not
    yet converted have ceilings; lower them as entry points are converted, and raise them only with the owner's
    word. Open gaps for the next slice: `docs/upstream/UP-0038.md`.
  - Converted so far: the annotation entry points (`opengrad-annotate`, `scripts/run_external_annotation.py`).
- Each run gets a git-ignored folder, `.run-records/<run_id>/` (`OPENGRAD_RUN_RECORDS_DIR` overrides it; the test
  suite uses that). It holds the full output, a start record and a summary. A start with no summary reads as
  `INCOMPLETE`.
- The summary holds:
  - the command and any error text, redacted: secrets, credential-named flags and fields (a name part such as
    `token`, `key`, `password` or `auth`, split at `-`, `_`, `.` and camelCase, or an all-caps name ending in one;
    `--no-…` and `--use-…` switches keep the next argument; tokenizer settings such as `pad_token` and
    `eos_token_id` are not credential names, see below), URL credentials, bearer tokens, Google keys and absolute
    paths;
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
- `upload --repo OWNER/NAME`, on the owner's word only, sends a durable copy of every log (each kind) to a
  dataset repository. It refuses, sending nothing, unless the Hub reports the repository as existing and private.
  That check is the real safety boundary: redaction is best effort (owner's choice, 2026-11-11), and promises only
  the forms its tests cover. The repository is `arjhinety/opengrad-run-logs`, created private on 2026-11-13. The
  check runs only at upload time, so never make that repository public and never include it in a release, card
  publication or other publish step: whatever redaction missed in the logs already there would be published.
  The publish scripts refuse it (`src/opengrad/publication/private_repos.py`), but changing its visibility on the
  Hub website is not guarded.
- Detection ignores terminal colour and cursor codes (CSI sequences, character-set selections, two-character
  cursor codes); escape strings that carry text (titles, hyperlinks) are checked as text. It covers, and a
  redacted copy removes:
  - PEM and PGP private-key blocks;
  - known token formats (`sk-`, `sk-proj-`, `hf_`, `ghp_`, AWS, Slack, Google keys);
  - credential-named fields and credential flags with a value, quoted values included. An unquoted field value
    is hidden to the end of its line or to the first quote, `}` or `]`, spaces and commas included, so the text
    up to that point is lost too;
  - bearer and basic auth on one line;
  - URL credentials, an empty user name included (`redis://:pw@host`);
  - emails, plain or percent-encoded;
  - home folders: any drive or case, names with spaces or apostrophes, UNC shares, percent-encoded.
  The redacted copy also removes other absolute paths.
- A log with any finding (a line over 100,000 characters counts as one), and every annotation log, is sent only as
  the redacted copy (`output.redacted.log`), which is checked again; if anything is still found, the log stays
  local. Any other log is sent byte for byte, ordinary absolute paths included (owner's choice C). A log that is
  not clean UTF-8 text (UTF-16, NUL bytes) stays local. Whenever a log stays local, upload lists it and exits 3,
  so a run with logs left unbacked never reports success. A log that changed since its run stops the upload
  (exit 1), and the logs kept local so far are still listed. Each upload is downloaded again to check the bytes. The summary's `log.durable` records which copy was sent, why, and its
  sha256 beside the original's.
- Any name with a credential word as one of its parts counts as a credential field, in logs and in summaries,
  except a closed list of tokenizer settings: `pad`, `eos`, `bos`, `unk`, `sep`, `cls`, `mask`, `sos`, `eot`,
  `boi`, `eoi`, `boa` or `eoa` followed by `_token`, `_token_id` or `_token_ids`, and `decoder_start_token_id`
  and `token_type_ids`, in any case or separator. Their lines, and the numbers after them, go up unchanged.
  Only the bare name is exempt: a dotted path such as `model.config.pad_token_id` still counts, and so do other
  tokenizer names (`image_token`), the safe direction.
- Known misses, sent as is: a bare user name (`USERNAME=arro`), header schemes other than bearer and basic
  (`Authorization: Token ...`, cookies), a flag value starting with `-`, a token glued to a preceding letter
  (an escaped `\n` in a printed repr, then `hf_...`), a field inside escaped JSON (`\"api_key\": ...`), an
  unquoted credential value holding a quote or bracket (the part after it), an unterminated quoted value, a
  credential named exactly like a listed tokenizer setting, credentials split across lines, and any other form the patterns don't know. A credential value on the line
  after its YAML name, or a mapping under a credential-named key, is detected but not hidden, so that log stays
  local. Don't print credentials in a run's output.
- Removed though harmless (the safe direction): "basic" followed by a long word, any path segment named
  `users` or `home` (URLs included), and the rest of a line after a home folder with no separator after the name.
- Tests record only dummy child commands and stub `huggingface_hub` (UP-0030). Recording on a remote GPU machine
  is not yet verified.

## Sources of truth

`docs/EXPERIMENT_FOUNDATION.md` · `docs/TRAINING_LIFECYCLE.md` · `results/README.md` · `docs/architecture/repository.md` ·
`docs/research/study-002/16-GPU-READINESS-GATE.md`

## Keeping this skill current

When a gate is added to a `ready_for_*` set, or a new state artifact appears, update the gate-set list and the
state table in the same commit.
