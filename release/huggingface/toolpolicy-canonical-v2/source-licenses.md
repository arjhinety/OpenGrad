# Source licensing and attribution

This file records observed upstream metadata for publication preparation. It is not legal advice. Verify current upstream terms immediately before upload.

This release is a **partial** v2 candidate and contains records from three sources only. Sources that appear in v1 but not here are listed under "Sources not in this release" and no attribution is claimed for them.

| Source | Upstream | Observed license/terms | Status |
|---|---|---|---|
| Glaive Function Calling v2 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | Apache-2.0 dataset metadata | REDISTRIBUTION_WITH_ATTRIBUTION; normalized derivative |
| ToolACE | https://huggingface.co/datasets/Team-ACE/ToolACE | Apache-2.0 dataset metadata | REDISTRIBUTION_WITH_ATTRIBUTION; normalized derivative |
| When2Call | https://huggingface.co/datasets/nvidia/When2Call | CC-BY-4.0 dataset metadata | REDISTRIBUTION_WITH_ATTRIBUTION; split boundaries preserved |

## Sources not in this release

The following sources are part of the v1 release and of the intended v2 composition, but are absent from this partial snapshot. Their records and attribution do not appear in this payload:

| Source | Reason absent |
|---|---|
| `Salesforce/xlam-function-calling-60k` | Upstream repository is access-gated; not fetched. |
| `BUTTON` | Upstream repository is access-gated; not fetched. |
| `LoopTool-23k` | Upstream source was not located. |

## Modifications

Records in this release are modified derivatives of the sources above. OpenGrad modifies them through canonical schema conversion, tool-definition and message normalization, structural validation, invalid-record filtering, deduplication, and metadata augmentation.

The Glaive records in this release additionally use a later adapter than v1 did. The v1 adapter left the source's unterminated `<functioncall>` blocks inside the assistant message as text, producing a tool result with no matching call; the later adapter parses those blocks into structured calls. This is a material modification relative to v1 and is why the two releases are not interchangeable. The canonical record carries the adapter version that produced it, so the two can be told apart per row.

## Attribution

The OpenGrad Apache-2.0 source license does not relicense upstream datasets. Attribution, notices, source links, revisions, and citations must remain visible in any Hub release. Cite the specific upstream sources actually present in the payload; see `CITATIONS.bib`.

The upstream access gate on xLAM is an access mechanism and is distinct from downstream redistribution permission. Because no xLAM-derived records are present in this snapshot, no downstream redistribution claim is made for them here.
