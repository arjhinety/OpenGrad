# Models

Model-family boundaries: which exact checkpoints are supported, what renders them, and what is
verified about each.

| Document | Covers |
|---|---|
| [renderer-matrix.md](renderer-matrix.md) | Which model families have a renderer, and how each was verified |

Qwen3.5-2B is the study model and the only family with an executed renderer. The declarative
identity lives in [`registry/models.yaml`](../../registry/models.yaml); other families are
reserved namespaces with no implemented adapter.

A renderer is not validated by existing — the pinned chat template has golden fixtures, and a run
records the template hash it actually used, so a silent template change cannot pass unnoticed.
