---
name: opengrad-annotation
description: How to run and change OpenGrad's annotation system safely — the opengrad-annotate CLI (check, start, status, audit, export, freeze-gold, model-batch, model-ingest, reference, review-queue, adjudicate, verify), the P-DET-v1 task, blinding of model judgments, the hash-chained SQLite store that only the UI may write, WIP snapshots versus gold freezes, and the frozen P-DET population check. This skill should be used for any work touching src/opengrad/annotation/, integrations/annotate-ui/, configs/annotation/, reports/pdet/, src/opengrad/verification/ or human/model labels.
---

# OpenGrad annotation

The full guide is `docs/ANNOTATION_TOOL.md`. The P-DET task is `configs/annotation/pdet-v1.yaml`, with the model
procedure in `configs/annotation/pdet-v1.model-procedure.md`. The Python side is `src/opengrad/annotation/`
(stdlib server, SQLite store, export). The UI is `integrations/annotate-ui/` (Next.js static export that only
renders).

## Hard rules

- **Never write the store directly.** Working state is `.annotation/<task>.sqlite3` (git-ignored). Every label
  change is a hash-chained record written by the service when a human acts in the UI. Scripts, SQL and tests
  must not modify it.
- **Tests never touch the real `.annotation/`.** Use a temporary root.
- **Model judgments stay hidden from human passes.** Blinded fields are never displayed, filtered on or sent to
  the browser. Do not show a human a model label "to help".
- **Labels are the owner's.** Cross-check and report evidence on request, but leave labels alone unless the
  human changes them in the UI.
- **`export` is a WIP snapshot, never gold** (`frozen: false`, `gold: null`, written under `…/wip/`), and it
  locks nothing.
  - **`freeze-gold` locks the sessions it cites** and refuses unless every freeze condition holds. Run it only on
    the owner's explicit instruction.
  - A directory holding a gold freeze is never overwritten.
- **The P-DET-v1 population is frozen.** Verify it before and after annotation work. `--verify` must print
  `"status": "PASS"`.

```bash
.venv/Scripts/python.exe -m opengrad.verification.pdet --verify
```

- **P-DET-COVERAGE-v1 is not drawn, and the study owner is its blind annotator.** Its builder is
  `src/opengrad/verification/pdet_coverage.py` (preregistration draft
  `docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md`).
  - The draw is byte-reproducible, so any written population *is* the future blind sample. Before adoption,
    use only `--dry-run`, which writes counts and hashes, never items. Never print item text, ids, sources
    or strata, and never quote pool text in an annotator-facing document (a test scans 30 for it).
  - `--build` refuses `reports/pdet-coverage/` until 30 is adopted, and `reports/pdet/` always.
  - Structural CALL evidence is `src/opengrad/verification/call_fidelity.py`; the acceptance rules are code
    in `src/opengrad/verification/pdet_coverage_metrics.py`. Change either only through 30.

```bash
.venv/Scripts/python.exe -m opengrad.verification.pdet_coverage --dry-run --output-dir reports/pdet-coverage
.venv/Scripts/python.exe -m opengrad.verification.call_fidelity   # counts only
```

## Common workflows

```bash
opengrad-annotate check pdet-v1                           # read-only preflight; creates nothing
opengrad-annotate start pdet-v1 --annotator <id> --session pass-a --dry-run
opengrad-annotate start pdet-v1 --annotator <id> --session pass-a   # start or resume (serves the UI)
opengrad-annotate status pdet-v1 --session pass-a
opengrad-annotate audit pdet-v1                           # re-verify every change-log hash chain
opengrad-annotate reference pdet-v1 --sessions pass-a model-a       # composite reference counts
opengrad-annotate export pdet-v1 --sessions pass-a model-a --composite   # WIP checkpoint of the dataset
opengrad-annotate verify <manifest>                       # re-hash an exported package
```

- **Checkpointing work without freezing:** run `audit`, then `export` (with `--composite` when the task allows
  it), then `verify` the written manifest. Commit the refreshed `reports/pdet/annotation/wip/` snapshot, and
  state in the commit message that it is not gold.
- **Model annotators:** `model-batch` renders the next unlabeled items for a declared model annotator and
  `model-ingest` validates and records its answers. Keep every batch, raw answer and audit.
  `reports/pdet/provenance/model-a/` is the precedent, archived byte for byte with a hash manifest. Transcripts
  containing a personal e-mail address stay untracked, and only their hashes are committed.
- **Review:** `review-queue` writes a pinned queue from the composite reference. `adjudicate` compares two
  finished passes, or reviews one.
- **Starting the UI** needs a build (`npm install && npm run build` in `integrations/annotate-ui`). For UI
  development, run `npm run dev` beside `opengrad-annotate start`.

## Taxonomy facts that trip people up

- The P-DET modes are CALL, DIRECT (code `ANSWER`), CLARIFY and UNSUPPORTED. **UNKNOWN is an ambiguity status,
  not a mode.** The config forbids a mode label with a non-`NONE` status, and UNKNOWN requires a non-`NONE`
  status.
- Exposed worked examples are annotated but excluded from metrics
  (`docs/research/study-002/27-PDET-EXPOSED-WORKED-EXAMPLES.md`).
- Protocol and definitions: `docs/research/study-002/22-PDET-PROTOCOL.md`, the instrument
  `docs/research/study-002/23-PDET-ANNOTATION-INSTRUMENT.md`, and the annotator checklist
  `docs/research/study-002/26-PDET-ANNOTATOR-CHECKLIST.md`.

## Sources of truth

`docs/ANNOTATION_TOOL.md` · `configs/annotation/pdet-v1.yaml` · `reports/pdet/pdet-v1.manifest.json` ·
`docs/research/study-002/README.md` (current label state)

## Keeping this skill current

Update the command list when `src/opengrad/annotation/cli.py` gains or renames a subcommand, and the hard rules
when a freeze condition or a blinding rule changes.
