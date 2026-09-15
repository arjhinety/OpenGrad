# P-DET `model-a`: audit trail of the model reference labels

**Do not open these files while annotating P-DET.** They contain the model's label for every item it
labeled (26, item 9).

The 570 labels in session `model-a` are **provisional model judgments** by the declared model annotator
`model.claude-opus-5` (Claude Opus 5), under amendment
[28](../../../../docs/research/study-002/28-PDET-MODEL-LABEL-AMENDMENT.md). They are not human labels,
not human gold, and not frozen. A human label in `pass-a` always takes precedence over them. Since
2026-09-15 every item has a `pass-a` label, so these judgments are a superseded comparison.

The trail was produced in the git-ignored working directory `.annotation/pdet-v1.model-batches/model-a/`.
It was packaged here by `scripts/archive_pdet_model_batches.py`, byte for byte. No file was rewritten.

| File | Tracked | Contents |
|---|---|---|
| `pdet-v1.model-a.audit-trail.tar.gz` | yes | See the list below. |
| `pdet-v1.model-a.audit-trail.manifest.json` | yes | The SHA-256 and size of every archive member, a per-batch table, the ingest summary, and the hashes of the transcripts. |
| `*.sha256` | yes | Sidecar hashes of the archive and the manifest. |
| `local/pdet-v1.model-a.transcripts.tar.gz` | **no** | The 12 subagent transcripts and their `.meta.json` files, verbatim. |

The tracked archive holds:

- the 12 batches (`batch-NN.json`, and the `batch-NN.md` each model read), numbered 01–12 with no gaps;
- the raw answers (`batch-NN.answers.json`, the subagent's final message as returned);
- the audits (`batch-NN.audit.json`);
- the pinned procedure file;
- the ingest log (every `model-a` change-log entry from the store, and the session record);
- the audit scripts that were used.

## What each batch row in the manifest proves

- **The batch has not changed since it was prepared.** `content_sha256` is re-derived from the archived
  batch content.
- **The procedure is the pinned one.** `procedure_sha256` equals the pin in
  `configs/annotation/pdet-v1.yaml`.
- **The answers are the audited answers.** The answers file, with CRLF turned back into LF, hashes to the
  audit's `answers_sha256`.
- **The transcript is the one that was audited.** The audit checked, from each transcript, that:
  - the prompt was exactly the procedure plus one batch line;
  - every tool call was a Read of that batch file;
  - the model was `claude-opus-5`.
  Each transcript still hashes to the value its audit recorded.

## Line endings

The batch, answers and audit files were written on Windows through text-mode writes, so they carry CRLF
line endings, and they are archived with those bytes. The audit hashed each answer text before that
translation. `answers_lf_sha256` in the manifest is the comparable hash. The tool now writes batch files
as bytes, so future batches are identical on every platform.

## Why the transcripts are not tracked

The transcripts contain the session's injected context, including the operator's e-mail address and local
file paths. They are evidence, so they are kept verbatim and never edited. For that reason the archive
holding them stays out of version control. The tracked manifest records its SHA-256 and each transcript's,
so a retained copy can be verified. If the local copy is lost, those four checks still stand on the
tracked archive. What is lost is the ability to re-run the transcript checks.

## Verify

```
python -m pytest tests/annotation/test_annotation_model_a_archive.py
```
