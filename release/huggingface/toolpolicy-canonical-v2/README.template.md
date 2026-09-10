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

OpenGrad ToolPolicy Canonical v2 is a provenance-preserving, model-independent normalization of public tool-use and function-calling datasets. It was built to test one hypothesis with a measurement attached: that the tool-call collapse observed in the M0 and M1 experiments on v1 was caused by the training corpus containing almost no tool-call supervision, rather than by the training procedure.

It is a **partial** build. Three of the six sources that appear in v1 could not be fetched for this snapshot, so the composition differs from v1 and this release is **not a drop-in replacement for v1**. Read it as an ablation on the presence of tool-call supervision.

v1 is left untouched and remains pinned by the B0 baseline and every result that depends on it.

## Why this snapshot exists

Under v1, the training boundary retained 55,719 records of which exactly **9** contained a tool call (0.016%). Under this release, roughly **half** the retained records do. Training the identical procedure on each corpus is what distinguishes a data-coverage failure from a training failure, and it is the only reason this partial corpus is published before completion.

The measured state of this snapshot, on the payload as published:

| Quantity | v1 | this release |
|---|---:|---:|
| Published canonical records | 213,951 | {{RECORD_COUNT}} |
| Records containing at least one tool call | 9 of 55,719 trainable (0.016%) | 49,423 of 103,036 (48.0%) |
| Sources | 6 | 3 |

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

| Source | Role | Upstream | Pinned revision | Raw count | Canonical retained | Published count | License/terms | Adapter version |
|---|---|---|---|---|---:|---:|---:|---|---|
{{SOURCE_TABLE}}

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
