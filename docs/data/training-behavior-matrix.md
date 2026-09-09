# Training behavior matrix

This matrix is generated from the current canonical audit, not a claim about
unmaterialized corpora. Counts and overlap caveats are in
`reports/data-normalization-v1.md`; `FULLY_MEASURED` means the source was
materialized and audited, not that every behavior is equally represented.

| Source | Must-call | No-call / retention | Clarify | Selection | Arguments | Multi-turn | Parallel | Recovery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xLAM | FULLY_MEASURED | FULLY_MEASURED | AUDITED | FULLY_MEASURED | FULLY_MEASURED | AUDITED | AUDITED | AUDITED |
| When2Call | FULLY_MEASURED | FULLY_MEASURED | AUDITED | AUDITED | AUDITED | AUDITED | AUDITED | AUDITED |
| ToolACE | FULLY_MEASURED | FULLY_MEASURED | AUDITED | FULLY_MEASURED | FULLY_MEASURED | AUDITED | AUDITED | AUDITED |
| BUTTON | FULLY_MEASURED | FULLY_MEASURED | AUDITED | FULLY_MEASURED | FULLY_MEASURED | FULLY_MEASURED | AUDITED | AUDITED |
| LoopTool | FULLY_MEASURED | FULLY_MEASURED | AUDITED | FULLY_MEASURED | FULLY_MEASURED | FULLY_MEASURED | AUDITED | AUDITED |
| Glaive | FULLY_MEASURED | FULLY_MEASURED | AUDITED | FULLY_MEASURED | FULLY_MEASURED | AUDITED | AUDITED | AUDITED |

`AUDITED` marks a capability whose presence/quality was checked but for which
the current report does not publish a source-level count. A source can
contribute to multiple columns and a column can contain examples from multiple
sources.
