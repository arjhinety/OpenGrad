---
pretty_name: OpenGrad ToolPolicy Canonical v1
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

OpenGrad ToolPolicy Canonical v1 is a provenance-preserving, model-independent normalization of several public tool-use and function-calling datasets. It is released as a pre-training candidate corpus for controlled research into tool-use policy in small open-weight language models. See [OpenGrad](https://github.com/arjhinety/OpenGrad) for the production methodology and reproducibility artifacts.

## Known defect: 9 tool-call targets at the training boundary

**Do not use v1 as a tool-calling SFT corpus as-is.** Rendered for Qwen3.5-2B under OpenGrad's trajectory contract, only 55,719 of the 213,951 records are trainable, and only **9** of those (0.016%) have a tool call in the supervised target. Training the M0 SFT recipe on it reproduced a tool-call collapse: call recall 0.0031 at step 800 and 0.0000 from step 1200 onward, on the 3,650-example held-out set.

The main cause is the v1 Glaive adapter (`glaive_function_calling_v2_v1`), which left the source's unterminated `<functioncall>` blocks unparsed. 50,851 of the 99,794 Glaive records carry their call as plain assistant text with an empty `tool_calls` list, so their function-response turns are orphaned tool results that fail the trajectory contract, and the Glaive records that remain contain no calls. xLAM's 59,370 records all end on a call that no tool result answers, which that contract also rejects, and most ToolACE, LoopTool and BUTTON records fail the schema layer.

[Canonical-v2](https://huggingface.co/datasets/arrochi112/OpenGrad-ToolPolicy-Canonical-v2) addresses both: its Glaive adapter parses the calls, and its `CALL_PREDICTION` contract admits call-only records such as xLAM's. If you use v1's Glaive records in another pipeline, their calls are unparsed text, not structured `tool_calls`.

## xLAM / APIGen provenance

This release includes 59,370 normalized records derived from `Salesforce/xlam-function-calling-60k` at revision `26d14ebfe18b1f7b524bd39b404b50af5dc97866`. The upstream dataset declares CC BY 4.0. Redistribution of these normalized xLAM-derived records is permitted under CC BY 4.0, subject to attribution and the applicable license terms. OpenGrad modifies the records through canonical schema conversion, tool-definition and message normalization, structural validation, invalid-record filtering, deduplication, and metadata augmentation where represented by the canonical artifact. These are modified derivatives; the original Salesforce/APIGen authors retain attribution, and users should cite APIGen. OpenGrad is not affiliated with or endorsed by Salesforce or the APIGen authors. The upstream repository uses a Hugging Face access gate; that upstream access mode is distinct from downstream redistribution permission, and this public OpenGrad dataset is not gated solely for that reason. Any upstream ethical-use statements remain source context and do not replace the applicable license terms.

## What this release is not

It is not a final recommended training mixture, a Qwen3.5 training dataset, M0, M1, M2, or a post-training result. No claim is made that training on all records or their natural proportions is optimal. Baseline evaluation and post-training have since run: the early M0 SFT runs trained on this release and collapsed (see above), and the definitive M0 trained on Canonical-v2.

## Configurations

The payload is a unified Parquet table. Filter by `source_dataset` and `source_split` for source-level views. Preference and evaluation artifacts are excluded from this release.

## Source manifest

| Source | Role | Upstream | Pinned revision | Raw count | Canonical retained | Published count | License/terms | Adapter version |
|---|---|---|---|---:|---:|---:|---|---|
{{SOURCE_TABLE}}

## Record count

This build contains `{{RECORD_COUNT}}` published canonical SFT records. Excluded sources are recorded in the release manifest: `{{EXCLUDED_SOURCES}}`.

## Canonical schema

Each row includes `opengrad_id`, `source_dataset`, `source_repo`, `source_record_id`, `source_split`, `source_revision`, `adapter`, `adapter_version`, `canonical_schema_version`, `canonical_hash`, `quality_status`, `contamination_status`, behavior decision/confidence/capabilities, and JSON-serialized canonical `tools`, `messages`, and `metadata` fields.

## Normalization and quality

Source adapters convert native formats into a shared semantic intermediate representation. Invalid or ambiguous source records are quarantined rather than silently repaired. The release contains only records allowed by the release policy; upstream attrition and quarantine counts remain documented in the GitHub reports.

Notable limitations include 59 BUTTON duplicate-tool-definition failures, quarantined malformed ToolACE records, Glaive canonical duplicates removed before release, and 3,077 canonical-valid LoopTool records that are incompatible with the pinned Qwen renderer because they contain no user query. Those LoopTool records are not removed from this model-independent canonical release solely because of Qwen renderability.

## Provenance and versioning

The release manifest records the OpenGrad commit, source manifest hashes, source revisions, adapter versions, output shard hashes, release filters, and generation parameters. Future experiments must pin this release by exact Hub revision and record the experiment-specific mixture separately.

## Evaluation boundary

The frozen 3,952-record When2Call evaluation namespace is not included. It remains separate in the OpenGrad GitHub repository. This is a model-independent training corpus; the B0 baseline and post-training results now exist and are recorded in the OpenGrad repository, not in this release.

## Licensing and citations

OpenGrad source code is Apache-2.0. Upstream dataset terms remain source-specific and are documented in `source-licenses.md`; this release does not relicense upstream data. See `CITATIONS.bib` for source references.

## Reproduction

From the OpenGrad repository, run:

`uv run opengrad-data build-hf-release --release-config configs/releases/toolpolicy_canonical_v1.yaml --output .release/hf/toolpolicy-canonical-v1`

Then validate:

`uv run opengrad-data validate-release --input .release/hf/toolpolicy-canonical-v1`

The commands build local staging only. They do not upload to Hugging Face.

## Responsible use

Use the data in accordance with each upstream source's terms, attribution requirements, and restrictions. Do not infer that a valid tool call demonstrates reliable tool-use policy or task completion.
