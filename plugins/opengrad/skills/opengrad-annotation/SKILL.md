---
name: opengrad-annotation
description: How to run and change OpenGrad's annotation system safely — the opengrad-annotate CLI (check, start, status, audit, export, freeze-gold, model-batch, model-ingest, reference, review-queue, adjudicate, verify), the P-DET-v1 task, blinding of model judgments, the hash-chained SQLite store that only the UI may write, WIP snapshots versus gold freezes, and the frozen P-DET population check. This skill should be used for any work touching src/opengrad/annotation/, integrations/annotate-ui/, configs/annotation/, reports/pdet/, src/opengrad/verification/ or human/model labels.
---

# OpenGrad annotation

The full guide is `docs/ANNOTATION_TOOL.md`. The P-DET task is `configs/annotation/pdet-v1.yaml`, with the model
procedure in `configs/annotation/pdet-v1.model-procedure.md`. The Python side is `src/opengrad/annotation/`
(stdlib server, SQLite store, export; `config.py`/`config_types.py`, `service.py`/`service_support.py` and
`export.py`/`export_common.py`/`export_verify.py` are split pairs, the first of each re-exporting the rest). The UI is `integrations/annotate-ui/` (Next.js static export that only
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

- **P-DET-COVERAGE-v1 is drawn (`reports/pdet-coverage/pdet-coverage-v1.population.jsonl`, 336 records),
  and labelled by three non-Claude models (`study_002_prereg_v5`, doc 34); the study owner annotates nothing.** Its builder is
  `src/opengrad/verification/pdet_coverage.py` (preregistration
  `docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md`, adopted as `study_002_prereg_v4`).
  - The draw is byte-reproducible, so the written population *is* the blind sample. Never open, print or
    summarise the population file; `--verify` and `--dry-run` report counts and hashes only. Never print item text, ids, sources
    or strata, and never quote pool text in an annotator-facing document (a test scans 30 for it).
  - `--build` may write `reports/pdet-coverage/` only since adoption, never overwrites a written
    population, and always refuses `reports/pdet/`.
  - Structural CALL evidence is `src/opengrad/verification/call_fidelity.py`; the acceptance rules are code
    in `src/opengrad/verification/pdet_coverage_metrics.py`. Change either only through 30.

```bash
.venv/Scripts/python.exe -m opengrad.verification.pdet_coverage --verify --output-dir reports/pdet-coverage
.venv/Scripts/python.exe -m opengrad.verification.call_fidelity   # counts only
opengrad-annotate check pdet-coverage-v1           # layer B, 306 items (configs/annotation/pdet-coverage-v1.yaml)
opengrad-annotate check pdet-coverage-v1-routing   # layer A, 30 items (…-routing.yaml)
# the three declared external annotators (amendment study_002_prereg_v5); run each until done:
.venv/Scripts/python.exe scripts/run_external_annotation.py pdet-coverage-v1 --annotator model.gpt-5.6-sol --size 20
```

  - `scripts/run_external_annotation.py` runs `agy` (Gemini, the input as `input.md` in an otherwise empty
    temporary directory, because agy ignores standard input started from Python), `codex exec` (standard input,
    read-only sandbox) and `cline --json` (standard input; its text output interleaves colour codes) outside the
    repository, then records answers through `model-ingest`'s validator. Network drops ("no such host") are
    common: it waits and retries. It never prints item text, labels or rationales. It resumes safely after an
    interruption: an unrecorded batch keeps its items unlabelled and they are re-batched.
  - Once all three finish: `opengrad-annotate export <task> --sessions model-gemini model-gpt model-deepseek`,
    `verify --require-source`, then `python -m opengrad.verification.pdet_coverage_reference --task <task>
    --package <manifest>`. The export says INCOMPLETE only because gold freezes compare at most two passes.
    Archive the trail with `scripts/archive_external_model_labels.py --task <task>` (raw CLI streams stay under
    `local/`, untracked).
  - **Supply for a next population** (35 §2, §4): `scripts/audit_pdet_coverage_v2_supply.py` counts the unused
    layer B pool (after P-DET-COVERAGE-v1 and every development and check set) and projects DIRECT from existing
    label yields. Counts only. Its 2026-09-17 run found about 22 expected DIRECT among single exchanges, short of 50.
    `scripts/audit_corpus_direct_prevalence.py` does the same for every single exchange of normalization-v3 (35 §5),
    and `scripts/audit_canonical_v2_final_direct.py` counts Study 001's corpus turn by turn, multi-turn included
    (35 §6). Single-exchange counts miss most of Glaive's prose, which sits in multi-turn conversations.
    `scripts/audit_first_reply_supply.py` dry-runs draft 36's first-reply unit (counts only; v1 predictions for sizing).
- **P-DET-COVERAGE-v2** (36, `study_002_prereg_v6`) is built by `src/opengrad/verification/pdet_coverage_v2.py`:
  420 first replies (contract `prose-decision-input-v2`) from Glaive and ToolACE, excluding P-DET-COVERAGE-v1 and
  every development and check set, with its own seed and quotas; `unit_kind` is blinded like source and stratum.
  The same counts-only rules apply: never open, print or summarise its population file.
  `python -m opengrad.verification.pdet_coverage_v2 --dry-run | --build | --verify` (default output
  `reports/pdet-coverage-v2/`; refuses `reports/pdet/` and `reports/pdet-coverage/`, never overwrites).
  Its annotation task is `pdet-coverage-v2` (`configs/annotation/pdet-coverage-v2.yaml`, procedure
  `pdet-coverage-v2.model-procedure.md`: v1's plus a first-reply instruction), run with the same three external
  annotators and the same finish-up steps.
- **Classifier v2 development sets** (37 §3): `src/opengrad/verification/classifier_devset_v2.py --set dev --build |
  --verify` draws first replies into `reports/prose-classifier/dev-v2/`, excluding P-DET-COVERAGE-v2 and everything
  it excludes. Task `prose-classifier-dev-v2` (`configs/annotation/prose-classifier-dev-v2.yaml`) is labelled by
  Claude subagents under `prose-classifier-dev-v2.model-procedure.md` (v1's plus the first-reply instruction),
  session `model-dev-v2`. Development labels only, never gold.
- **ANSWER strata candidates** (41, `study_002_prereg_v9`) come from `src/opengrad/verification/answer_strata.py`:
  1,767 items, pool N (BFCL no-call items) plus pool K (Natural Questions questions with tools that cannot serve them),
  written to `reports/study-002/answer-strata-v1/`. The same counts-only rules apply.
  `python -m opengrad.verification.answer_strata --dry-run | --build | --verify`: the inputs are cached in
  `.cache/answer-strata/`, and the screen needs `data/processed/normalization-v1`. Without them, `--verify` reports
  `BLOCKED_INPUT_MISSING` rather than re-deriving. Task `answer-strata-v1`
  (`configs/annotation/answer-strata-v1.yaml`, procedure `answer-strata-v1.model-procedure.md`) serves only 41 §8
  and blinds pool, source and reference answers. Since `study_002_prereg_v10` (42) it is labelled by two external
  annotators, Gemini and DeepSeek, because Codex refuses gpt-5.6-sol for this account. Export with
  `--sessions model-gemini model-deepseek`. The strata are the items both label `ANSWER`, per pool, reported
  separately (41 §9, 42). `pdet_coverage_reference.TASK_ANNOTATORS` holds that per-task annotator set; the
  P-DET-COVERAGE tasks keep three.
  - **After the reference:** `pdet_coverage_reference --task answer-strata-v1` writes a `reference/`
    directory beside the population. Then `python -m opengrad.verification.answer_strata_report`
    (`src/opengrad/verification/answer_strata_report.py`) writes the per-stratum n, the §10 sizing status and the
    §11 balance, plus a membership file of ids and strata. The comparison with the other modes stays
    `BLOCKED_PCONF_NOT_BUILT` until P-CONF exists.

  - Both tasks read the one hash-pinned population and pick their layer with `source.select`. Source, stratum,
    layer, gate and every provenance field are blinded, and neither task declares metadata chips or filters
    (`tests/annotation/test_annotation_pdet_coverage_tasks.py`). Since `study_002_prereg_v5` the reference is a
    two-of-three consensus of Gemini 3.8 Flash (High), gpt-5.6-sol and deepseek-v4.1-flash; no Claude model labels it,
    because Claude builds the classifier under test.
  - When inspecting the population for engineering, print counts only. Never dump a record's keys or values:
    tool parameter names are item content too.

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
- **Classifier development labels:** `prose-classifier-dev-v1` (`configs/annotation/prose-classifier-dev-v1.yaml`,
  250 items, 33 §3) was labelled by session `model-dev` (Claude): one batch split across five parallel
  subagents, answers merged in batch order and ingested once. `scripts/archive_devset_model_labels.py` audits
  each subagent's transcript into `reports/prose-classifier/dev/provenance/model-dev/`. These labels only
  develop the classifier; they are never gold or evidence of accuracy. The held-out check set
  `prose-classifier-devcheck-v1` (`configs/annotation/prose-classifier-devcheck-v1.yaml`, 125 items, 33 §5a) is
  labelled the same way under session `model-devcheck`, with the same procedure file, and archived with
  `--task prose-classifier-devcheck-v1`. The second check set `prose-classifier-devcheck-v2`
  (`configs/annotation/prose-classifier-devcheck-v2.yaml`, session `model-devcheck-v2`) follows the same route.
  Its `check` reports one item because an offered tool has a parameter named `source_name`, which is a key
  collision, not a leak of the item's source.
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
