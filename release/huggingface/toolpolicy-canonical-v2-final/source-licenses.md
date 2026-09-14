# Source licensing and attribution

This file records observed upstream metadata for publication preparation. It is not legal advice. Verify current upstream terms immediately before upload.

This is the **completed** v2 release: four sources, 173,237 canonical records, 176 shards. Every source present in the payload is listed in the table below. Sources that were considered and left out are listed under "Sources not in this release", and no attribution is claimed for them.

| Source | Upstream | Observed license/terms | Status |
|---|---|---|---|
| xLAM | https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k | CC-BY-4.0; upstream access gate | PERMITTED_WITH_ATTRIBUTION; included as normalized derivative |
| Glaive Function Calling v2 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | Apache-2.0 dataset metadata | REDISTRIBUTION_WITH_ATTRIBUTION; normalized derivative |
| ToolACE | https://huggingface.co/datasets/Team-ACE/ToolACE | Apache-2.0 dataset metadata | REDISTRIBUTION_WITH_ATTRIBUTION; normalized derivative |
| When2Call | https://huggingface.co/datasets/nvidia/When2Call | CC-BY-4.0 dataset metadata | REDISTRIBUTION_WITH_ATTRIBUTION; split boundaries preserved |

Contributions, which sum to the release totals:

| Source | Canonical | Trainable |
|---|---:|---:|
| xLAM | 57,342 | 56,090 |
| Glaive Function Calling v2 | 98,339 | 97,112 |
| ToolACE | 11,051 | 2,259 |
| When2Call | 6,505 | 6,505 |
| **total** | **173,237** | **161,966** |

This differs from the historical partial-v2 snapshot, which was a three-source build. That snapshot
is unchanged and still published; its own `source-licenses.md` describes its own payload and is not
this file.

## xLAM: how these records were obtained

The xLAM upstream repository is access-gated. Its records were **not** fetched from it. They are
reconstructed from the published Canonical-v1 release, which retains the source's 59,370 records
with `parameters` unmodified, and the reconstruction is verified record by record: `tools` is
compared against that derivative verbatim and `query` is checked non-empty, for all 59,370 records,
with 0 mismatches. The derivation is recorded in `data/raw/xlam/derivation.json`. This is a
representational recovery, not a semantic inference; it invents no field and repairs nothing. See
`reports/CANONICAL_V2_COMPLETION_REPORT.md`.

The access gate is an access mechanism and is distinct from downstream redistribution permission.
The manifest records xLAM as `PERMITTED_WITH_ATTRIBUTION` and `public_allowed` downstream, and the
xLAM-derived records in this release are modified normalized derivatives distributed under the
applicable CC BY 4.0 terms. Preserve Salesforce/APIGen attribution and cite APIGen: xLAM's manifest
entry requires citation of the **APIGen** paper rather than of the source repository, and it is the
only source in this release whose `citation_target` is not `source`. See `CITATIONS.bib`.

## Sources not in this release

The following sources are part of the v1 release and of the intended v2 composition, but are absent from this payload. Their records and attribution do not appear here:

| Source | Reason absent |
|---|---|
| `BUTTON` | Upstream repository is access-gated (HTTP 401); not fetched, not approximated. |
| `LoopTool-23k` | Upstream source was not located; not fetched, not approximated. |

## Modifications

Records in this release are modified derivatives of the sources above. OpenGrad modifies them through canonical schema conversion, tool-definition and message normalization, structural validation, invalid-record filtering, deduplication, and metadata augmentation.

The Glaive records in this release additionally use a later adapter than v1 did. The v1 adapter left the source's unterminated `<functioncall>` blocks inside the assistant message as text, producing a tool result with no matching call; the later adapter parses those blocks into structured calls. This is a material modification relative to v1 and is why the two releases are not interchangeable.

The two builds can be told apart per row by the `adapter` field, which reads `glaive_function_calling_v2_v1` in v1 and `glaive_function_calling_v2_v2` here. Note that the per-row `adapter_version` field does **not** distinguish them: it reads `1.0.0` in both. The "Adapter version" column in the card is a separate value — the materializer version each source's artifact manifest records — and is not the per-row field of the same name.

## Attribution

The OpenGrad Apache-2.0 source license does not relicense upstream datasets. Attribution, notices, source links, revisions, and citations must remain visible in any Hub release. Cite the specific upstream sources actually present in the payload; see `CITATIONS.bib`.

Every record carries `source_dataset`, `source_repo`, `source_split`, `source_license`, `redistribution_status` and `modification_status`, so each source's terms travel with the row rather than stopping at this file.

A derived view that removes a source — `OpenGrad-ToolPolicy-Canonical-v2-minus-xlam` — inherits this file verbatim and therefore continues to list xLAM alongside the three sources it retains. That is deliberate: an attribution file should over-include rather than risk dropping a licence obligation, and the parent release is where xLAM's terms continue to apply.
