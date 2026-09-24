---
name: opengrad-development
description: Entry point for any development work in the OpenGrad repository — repository map, environment (Windows, Git Bash, .venv, uv), how to verify a change the way CI does, known pre-existing failures, commit and push rules, and which opengrad-* skill owns each area. This skill should be used at the start of any OpenGrad task, before editing code, and before committing or pushing.
---

# OpenGrad development

OpenGrad is a provenance-first post-training research repository: `Qwen/Qwen3.5-2B` pinned at revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, tool-use SFT/DPO, frozen evaluations, and published studies. Every
number must trace to an artifact. Correctness here means *provenance and gates hold*, not only *tests pass*.

## Route to the owning skill

| Work | Skill |
|---|---|
| Studies, preregistration, frozen artifacts, research claims | `opengrad-research-guardrails` |
| Adapters, normalization, canonical corpora, releases, mixtures, yield | `opengrad-data-pipeline` |
| Human/model labels, P-DET, `opengrad-annotate` | `opengrad-annotation` |
| Experiment configs, preflight, readiness gates, run store, results index | `opengrad-experiments-readiness` |
| SFT, DPO, distillation, preferences, checkpoints, vision/MTP components | `opengrad-training` |
| Baseline and candidate evaluation, benchmarks, contamination, failures | `opengrad-evaluation` |
| Promotion policy, promote/reject, testing any gate | `opengrad-promotion-gates` |
| Hugging Face, GGUF, ExecuTorch, quantization/optimization, publication checks | `opengrad-openweights-release` |
| Registries, provenance claims, reports, incidents, errata, status docs | `opengrad-registry-provenance` |
| Adding or updating these skills | `opengrad-skills-maintenance` |

## Repository map

`docs/architecture/repository.md` is the authoritative layout, and
`tests/results/test_generated_indexes.py` fails when a tracked top-level directory is missing from it. The generated indexes `reports/README.md`, `scripts/README.md` and `docs/README.md` come from
`scripts/reporting/generate_indexes.py` (run it after adding a file; `--check` in tests). In short:
- `src/opengrad/` holds all library code. `src/opengrad/cli.py` is `opengrad`, and `src/opengrad/agent_cli.py`
  holds the train/readiness boundary that integrations share.
- `runs/` is authoritative experiment state (written only by `ExperimentStore`). `results/` is a derived index.
- `configs/`, `registry/`, `reports/`, `docs/`, `scripts/` (one-off and campaign tooling) and `tests/` (CPU-safe).
- Large data, checkpoints and caches are git-ignored (`data/`, `.cache/`) and referenced by hash or revision.
  Two small files under `data/processed/` are tracked on purpose (the M1 preference pairs, `registry/datasets.yaml`).

## Environment

- Windows with Git Bash. Use POSIX syntax and forward slashes. The interpreter is `.venv/Scripts/python.exe`,
  and console scripts live in `.venv/Scripts/` (`opengrad.exe`, `opengrad-data.exe`, …). `uv run <cmd>` also works.
- The dev venv has **no torch**. Torch-dependent tests skip there. To run them, build a scratch CPU env
  (recipe in `opengrad-training`).
- `core.autocrlf=true`. Python writes files with CRLF on Windows unless bytes are written explicitly. Many
  source files are CRLF. When editing by script, read and write bytes and preserve the file's existing
  endings. **A changed byte in a hashed file changes its hash**: a trailing newline in `src/opengrad/data/versions.py`
  once invalidated a built artifact.
- Treat frozen artifacts as byte-exact. Never "normalize" line endings of tracked data files. A hash-pinned file
  type needs a `.gitattributes` rule (`-text`) so every platform checks it out byte for byte.
- **CI runs on Linux and Windows; development happens on Windows.** Code must be portable in both directions:
  - compare repository paths as `relative_to(root).as_posix()`, never `str(path)`;
  - never build a filename from an id containing `:` or other Windows-forbidden characters;
  - give every script with a shebang git mode `100755` (`git update-index --chmod=+x`), or ruff `EXE001`
    fails CI; `tests/repo/test_executable_bits.py` catches it on Windows too.
  - GitHub Actions results: `gh run list` and `gh run view <id> --log-failed`.

## Verify a change the way CI does

CI (`.github/workflows/ci.yml`) has three jobs. `test` (Linux) runs these. Locally, run the lint, types and
checks, but run only the tests that cover the files you changed (their own test files, tests that import the
changed modules, and `tests/skills` when a skill changed); CI runs the whole suite:

```bash
uv sync --locked --extra dev           # CI fails on a stale uv.lock
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m mypy src                                    # strict; see [tool.mypy] overrides
.venv/Scripts/python.exe -m pytest -p no:cacheprovider -q <the tests covering your change>
.venv/Scripts/opengrad-validate.exe
.venv/Scripts/python.exe scripts/preserve_h200_state.py --verify      # frozen H200 pins, on committed blobs
.venv/Scripts/python.exe scripts/repo/check_publication_hygiene.py    # secrets and local paths
```

`test-windows` runs pytest, `opengrad-validate`, the preserved-state verify and the hygiene scan on
`windows-latest` with `core.autocrlf=true`, where line endings make the byte-pinned checks differ from Linux.
Strict mypy needs a venv without the ML extras to match CI; `transformers`, `safetensors`, `huggingface_hub` and
`torch` are `ignore_missing_imports` in `pyproject.toml` because CI does not install them.
The `mcp` job runs `npm run check` and `npm test` in `integrations/opengrad-mcp`. Actions are pinned to
commit SHAs and the workflow has `contents: read` only; bump a pin by resolving the tag
(`git ls-remote https://github.com/<owner>/<action> refs/tags/<tag>`), never by writing a tag name back.
Python is 3.11 (`.python-version`); the pre-commit ruff `rev` equals ruff in `uv.lock`.

- `pytest` deselects `network` tests by default (they reach external hosts). Run them explicitly with
  `pytest -m network` when a change touches publication resolution.
- Git-ignored data (`data/`, `.cache/`) does not exist in CI. A check that needs it must report a `BLOCKED_*`
  status there, never PASS and never a content FAIL (`src/opengrad/verification/accounting.py`).
- Also run `ruff format --check` **on the files you changed**. Many existing files are not formatted, so
  never reformat untouched files; that buries the real diff.
- Report failures faithfully. Separate pre-existing failures from new ones by running the same tests on a
  clean worktree (`git worktree add --detach <scratch> HEAD`), then remove it.
- Pre-existing failures are tracked in `references/known-failures.md`. Update that file in the commit that
  fixes or introduces one.

## Commit and push rules

- Commit only when the user asks, or when a task the user started naturally ends in a commit. Push only when
  asked.
- Never commit:
  - credentials;
  - transcripts containing personal e-mail;
  - bulk data or checkpoints.
- Commit messages explain *why* and state measured facts (counts, hashes), the way `git log` already does.
- GitHub push protection can block a push that contains a secret-like string. Never bypass it for the user.
  Verify whether the token is real, report it, and let the user decide or unblock it.

## Standing working rules

- The user gates research phases explicitly: drawing populations, training a classifier, freezing gold,
  changing the mixture, training. Ask before starting a new phase, and offer choices.
- Status words (`PENDING`, `FROZEN`, `PASS`) change in the same commit as the status (G16). A number in prose
  is generated or tested, never retyped (G14). See `opengrad-research-guardrails`.

## Keeping this skill current

Update the routing table when a skill is added, and `references/known-failures.md` whenever the failing set
changes. `tests/skills/test_skills.py` fails CI when a package, console script or `opengrad` subcommand has
no owning skill, or when a skill cites a path that no longer exists.
