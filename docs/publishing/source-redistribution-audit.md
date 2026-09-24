# Source redistribution audit for OpenGrad ToolPolicy Canonical v1

Audit scope: publication preparation only. This is a provenance and metadata audit, not legal advice. The release builder fails closed for sources whose upstream distribution boundary is not clearly cleared.

| Source | Upstream source | Pinned revision | Observed license metadata | Distribution status | Publication treatment |
|---|---|---|---|---|---|
| xLAM | https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k | `26d14ebfe18b1f7b524bd39b404b50af5dc97866` | HF card declares CC-BY-4.0; upstream access is gated | PERMITTED_WITH_ATTRIBUTION | Included as a modified derivative with attribution, APIGen citation, and modification disclosure; downstream release remains public |
| BUTTON | https://github.com/PKU-Baichuan-MLSystemLab/BUTTON | `47cb720ed223b249a2f1d0a3faf1cb1eb7175622` | GitHub repository metadata reports CC-BY-4.0 | REDISTRIBUTION_WITH_ATTRIBUTION | Eligible with attribution; retain upstream notices and dataset citation |
| ToolACE | https://huggingface.co/datasets/Team-ACE/ToolACE | `6bda777c88d21e5a204703c1ee45597a8fa4f734` | HF dataset metadata reports Apache-2.0 | REDISTRIBUTION_WITH_ATTRIBUTION | Eligible with attribution; verify card terms again immediately before upload |
| LoopTool-23k | https://huggingface.co/datasets/zhangkangning/LoopTool-23k | `b6c572d442ed4f2177f23645d8e9a77522e712c3` | HF dataset metadata reports Apache-2.0 | REDISTRIBUTION_WITH_ATTRIBUTION | Eligible with attribution; preserve possible derivation warning |
| Glaive Function Calling v2 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` | HF dataset metadata reports Apache-2.0 | REDISTRIBUTION_WITH_ATTRIBUTION | Eligible with attribution; verify card terms again immediately before upload |
| When2Call | https://huggingface.co/datasets/nvidia/When2Call and https://github.com/NVIDIA/When2Call | `0582f7749df63a96fdc3070932e83e72396ace53` | HF dataset metadata reports CC-BY-4.0 | REDISTRIBUTION_WITH_ATTRIBUTION | Eligible with attribution; keep SFT, preference, and evaluation configurations separate |

## xLAM decision

Previous state: redistribution was conservatively unresolved because the upstream Hugging Face repository uses gated access.

Reviewed evidence:

- official Salesforce xLAM dataset repository and card
- declared CC BY 4.0 license
- current gate conditions requiring acknowledgment and APIGen citation
- pinned OpenGrad source revision and local normalization manifest

Decision: normalized xLAM-derived records may be included in OpenGrad's public dataset release under CC BY 4.0 with attribution, modification disclosure, upstream provenance, and APIGen citation. The upstream access gate is modeled independently and is not reproduced downstream. This is a project provenance decision, not formal legal advice.

Decision date: 2026-09-04
Implementation revision: recorded in the OpenGrad commit that updates this policy.

## Mixed-license policy

The Apache-2.0 license in the OpenGrad repository applies to OpenGrad source code, not automatically to upstream datasets. The Hugging Face dataset card must retain this per-source table, upstream links, revisions, citations, and required attribution. A single top-level license field must not be used to imply relicensing of every upstream component.

## Recheck requirement

Upstream cards, repository terms, gated conditions, and redistribution status must be rechecked immediately before any future update. The v1.0.0 release includes xLAM under the documented CC BY 4.0 attribution and modification terms; its upstream access gate is not reproduced downstream.

## Later changes (appended 2026-09-24)

The table above is the dated record of the 2026-09-04 decision and is not rewritten.
`registry/datasets.yaml` moved to schema version 2 on 2026-09-24
(`registry/dataset_record.schema.json`), with these effects on this audit:
- **One spelling per outcome.** xLAM's `PERMITTED_WITH_ATTRIBUTION` is now spelled
  `REDISTRIBUTION_WITH_ATTRIBUTION`, the same outcome the other sources have. The upstream gate the distinction
  recorded is `upstream_access_mode: gated`.
- **Each record names this audit** as its `redistribution_basis`.
- **Current per-source view.** The generated [`docs/datasets/SOURCE_REGISTRY.md`](../datasets/SOURCE_REGISTRY.md)
  shows the current per-source status.
- **Drift test.** `tests/registry/test_source_docs_agree.py` fails if this table's revisions stop matching the
  registry.
