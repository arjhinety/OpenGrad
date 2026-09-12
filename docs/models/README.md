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

## Scope is not a permanent commitment

OpenGrad is not a Qwen repository. Qwen3.5-2B is the first planned target, not the permanent scope.
Configuration namespaces reserve future work for Qwen, Gemma, Llama, Phi, SmolLM, and LFM, but those
namespaces currently contain **no implemented model adapters and no results**. The distinction
matters:

```text
implemented infrastructure ≠ planned model ≠ working adapter ≠ replicated result
```

Cross-model replication is a later roadmap stage and has not been executed.
