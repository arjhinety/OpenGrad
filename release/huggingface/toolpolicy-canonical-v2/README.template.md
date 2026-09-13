---
pretty_name: OpenGrad ToolPolicy Canonical v2 (partial, M0 snapshot)
language:
- en
license: other
task_categories:
- text-generation
configs:
- config_name: canonical
  data_files:
  - split: train
    path: "*.parquet"
---

> This is a provenance-preserving canonical candidate corpus. It is a pre-training canonical release, not an empirically selected or recommended training mixture.

## What this release is

OpenGrad ToolPolicy Canonical v2 is a provenance-preserving, model-independent normalization of public tool-use and function-calling datasets. It was built to test one hypothesis with a measurement attached: that the tool-call collapse observed in the M0 SFT experiments on v1 was caused by the training corpus containing almost no tool-call supervision, rather than by the training procedure.

It is a **partial** build. Three of the six sources that appear in v1 could not be fetched for this snapshot, so the composition differs from v1 and this release is **not a drop-in replacement for v1**. Read it as an ablation on the presence of tool-call supervision.

v1 is left untouched and remains pinned by the M0 SFT runs that trained on it. (The B0 baseline pins only the held-out evaluation manifest, not a training corpus.)

## Why this snapshot exists

Under v1, the training boundary retained 55,719 records of which exactly **9** contained a tool call (0.016%). Under this release, roughly **half** the retained records do. Training the identical procedure on each corpus is what distinguishes a data-coverage failure from a training failure, and it is the only reason this partial corpus is published before completion.

The measured state of this snapshot, on the payload as published:

| Quantity | v1 | this release |
|---|---:|---:|
| Published canonical records | 213,951 | {{RECORD_COUNT}} |
| Trainable records whose target contains a tool call | 9 of 55,719 (0.016%) | 48,723 of 101,785 (47.9%) |
| Sources | 6 | 3 |

Both columns of the tool-call row are counted at the training boundary. Before rendering, 49,423 of this release's 103,036 canonical records (48.0%) contain a tool call; that figure is not comparable with v1's 9, which counts trainable records.

## Findings this corpus produced

This corpus exists to answer one question, and it answered it. One experiment was run on it — M0 SFT, experiment ID `qwen35_2b_m0_sft_v2corpus` — using the **same** procedure, hyperparameters, base checkpoint (`Qwen/Qwen3.5-2B`), seed, and held-out evaluation as the M0 SFT runs on v1. The corpus was the only variable, but it changed in more than one way (see below). Full configuration: [`configs/experiments/qwen35_2b_m0_sft_v2corpus.yaml`](https://github.com/arjhinety/OpenGrad/blob/master/configs/experiments/qwen35_2b_m0_sft_v2corpus.yaml).

Scored on the frozen When2Call held-out split (3,650 distinct examples), best over checkpoints:

| Metric | B0 baseline (untrained) | M0 SFT on v1 | M0 SFT on this corpus |
|---|---:|---:|---:|
| `call_f1` | 0.6191 | 0.0000 – 0.0062 | **0.5247 – 0.5995** |
| `call_precision` | 0.4542 | 0.0000 – 1.0000 | **0.7373 – 0.7891** |
| `call_recall` | 0.9722 | 0.0000 – 0.0031 | **0.3931 – 0.5050** |
| `over_call_rate` (lower is better) | 0.6425 | 0.0000 – 0.0008 | **0.0577 – 0.0989** |
| `unsupported_accuracy` | 0.0131 | 0.5058 – 0.7992 | **0.5923 – 0.6363** |
| `clarification_accuracy` | 0.1009 | 0.7434 – 0.9538 | **0.7783 – 0.8613** |

**The finding.** On v1, SFT destroyed tool calling: `call_recall` fell to 0.0031 and `call_f1` effectively to zero, while `over_call_rate` collapsed to 0.0000 — the model stopped emitting tool calls almost entirely, monotonically, across every checkpoint. On this corpus, the same procedure preserved the capability: `call_recall` rose roughly 160× (0.0031 → 0.5050) and `over_call_rate` fell from the baseline's 0.6425 to 0.0577, with `call_precision` improving from 0.4542 to 0.7891.

This points to a **data-coverage failure rather than a training-procedure failure**: v1's training signal contained 9 tool-call targets out of 55,719 retained records, so there was nothing to learn the behaviour from. It does not isolate tool-call coverage as the cause, because other things changed at the same time: three of v1's six sources were dropped (xLAM, BUTTON, LoopTool), the When2Call slice went from 14,829 to 4,000 records, and the trainable set grew from 55,719 to 101,785 records. Supplying tool-call supervision, together with those changes, produced a model that calls tools without calling them indiscriminately. A further signal that the training procedure behaves normally here: this corpus yields an interior optimum (best `call_f1` at step 1200, with steps 1800 and 2400 declining), whereas on v1 the metric decayed to zero and stayed there.

**What this finding does not claim.** The best `call_f1` on this corpus (0.5995) remains marginally below the B0 baseline's 0.6191, so there is no claim that SFT on this corpus beats the untrained baseline on the headline metric. B0 reaches 0.6191 with a degenerate near-always-call policy — `call_recall` 0.9722 against `unsupported_accuracy` 0.0131 and `over_call_rate` 0.6425 — so its score reflects calling on almost every example rather than deciding when to call. This corpus's contribution is a non-degenerate policy, not a higher headline number. No checkpoint from this run was promoted, and none is distributed with this release.

Training statistics for the run: 2,400 optimizer steps, 27,672 examples, 8,011,435 supervised tokens, 41.9 minutes on one A100-SXM4-80GB, training loss 1.5332 → 0.5582 (min 0.0220, mean 0.4618). Full write-up: [`reports/M0_SFT_EXECUTION_REPORT.md`](https://github.com/arjhinety/OpenGrad/blob/master/reports/M0_SFT_EXECUTION_REPORT.md).

## Scope limits of this snapshot

This release is incomplete by construction, and the omissions are material:

* `Salesforce/xlam-function-calling-60k` (59,370 records in v1) is **absent** — the upstream repository is access-gated.
* `BUTTON` (7,941 records in v1) is **absent** — the upstream repository is access-gated.
* `LoopTool-23k` (20,827 records in v1) is **absent** — the upstream source was not located.

No xLAM, BUTTON, or LoopTool-derived records are present in this payload, and no attribution is claimed for them. A completed v2 release, if built, would carry its own version and manifest hash.

This snapshot also differs from v1 in its Glaive adapter. v1 consumed the earlier adapter, which left the source's unterminated `<functioncall>` blocks inside the assistant message unparsed; this release uses the later adapter, which parses them. That change is a modification relative to v1 and is disclosed in `source-licenses.md`.

## What this release is not

It is not a final recommended training mixture, a Qwen3.5 training dataset, or a post-training result. It is the corpus used for one M0 experiment; no model checkpoint is distributed with it. No claim is made that training on all records or their natural proportions is optimal, and no claim is made that the upstream access gates have been removed or worked around.

## Configurations

The payload is a unified Parquet table. Filter by `source_dataset` and `source_split` for source-level views. Preference and evaluation artifacts are excluded from this release.

## Source manifest

| Source | Role | Upstream | Pinned revision or input-file SHA-256 † | Raw count | Canonical retained | Published count | License/terms | Adapter version |
|---|---|---|---|---:|---:|---:|---|---|
{{SOURCE_TABLE}}

† For When2Call this is a Hub commit. For Glaive and ToolACE it is **not** a Hub revision: it is the SHA-256 of the local input file the adapter read (for ToolACE, the first 16 hex characters of `7a7a6a2c3b1003c789bb…`), and it does not match the Hub file's LFS hash either. The Hub revisions are `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` (Glaive) and `6bda777c88d21e5a204703c1ee45597a8fa4f734` (ToolACE): Canonical-v1 records the same input files, by SHA-256, against those revisions, and the Hub history shows neither upstream data file has changed since it was uploaded.

"Adapter version" above is the version each source's artifact manifest records for the materializer that ran. It is not the per-row value: every row also carries an `adapter` name and an `adapter_version` field, and for the Glaive records that per-row version reads `1.0.0` in both v1 and this release. The two Glaive builds are distinguished per row by the **`adapter` name**, `glaive_function_calling_v2_v1` (v1) versus `glaive_function_calling_v2_v2` (this release), not by that version field.

## Record count

This build contains `{{RECORD_COUNT}}` published canonical SFT records. Excluded sources are recorded in the release manifest: `{{EXCLUDED_SOURCES}}`.

## Canonical schema

Each row includes `opengrad_id`, `source_dataset`, `source_repo`, `source_record_id`, `source_split`, `source_revision`, `source_license`, `redistribution_status`, `modification_status`, `adapter`, `adapter_version`, `canonical_schema_version`, `canonical_hash`, `quality_status`, `contamination_status`, behavior decision/confidence/capabilities, and JSON-serialized canonical `tools`, `messages`, and `metadata` fields.

## Normalization and quality

Source adapters convert native formats into a shared semantic intermediate representation. Invalid or ambiguous source records are quarantined rather than silently repaired. The release contains only records allowed by the release policy; upstream attrition and quarantine counts remain documented in the GitHub reports.

Quarantine is deliberate. Where a source record cannot be converted without guessing — an unresolvable tool result, a tool schema that is not valid JSON Schema, a missing user turn — it is dropped and counted rather than coerced into training text, because training on a repaired guess produces supervision that no measurement can interpret.

Notable limitations specific to this build: malformed ToolACE records are quarantined rather than repaired, and Glaive canonical duplicates are removed before release.

## Provenance and versioning

The release manifest records the OpenGrad commit, source manifest hashes, source revisions, adapter versions, output shard hashes, release filters, and generation parameters. Future experiments must pin this release by exact Hub revision and record the experiment-specific mixture separately.

## Evaluation boundary

The frozen When2Call evaluation namespace (MCQ, LLM-judge, and preference artifacts) is excluded from this payload and remains separate in the OpenGrad GitHub repository. It is verified absent, not merely intended to be absent. No held-out example appears in this release.

## Licensing and citations

OpenGrad source code is Apache-2.0. Upstream dataset terms remain source-specific and are documented in `source-licenses.md`; this release does not relicense upstream data. See `CITATIONS.bib` for source references.

## Reproduction

From the OpenGrad repository, run:

`uv run opengrad-data build-hf-release --release-config configs/releases/toolpolicy_canonical_v2.yaml --output .release/hf/toolpolicy-canonical-v2`

Then validate:

`uv run opengrad-data validate-release --input .release/hf/toolpolicy-canonical-v2`

The commands build local staging only. They do not upload to Hugging Face.

## Responsible use

Use the data in accordance with each upstream source's terms, attribution requirements, and restrictions. Do not infer that a valid tool call demonstrates reliable tool-use policy or task completion. Records containing tool calls were normalized from upstream annotations and have not been verified against live tool execution.
