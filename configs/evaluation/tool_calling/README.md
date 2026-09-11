# Tool-calling evaluation configurations

**In use.** This namespace holds the frozen measurement contracts, not reserved space.

| File | What it is |
|---|---|
| `qwen35_2b_baseline.yaml` | The B0 baseline contract: pinned model and tokenizer revision, renderer, template hash, engine version, generation settings and the held-out manifest it evaluates against |

Every measurement in the project is produced through a config derived from this one. A candidate
config may change the model and nothing else — the renderer, template hash, seed, generation
settings, runtime and evaluation manifest are invariant, so a delta is attributable to the
checkpoint rather than to the measurement. See
[`docs/CANDIDATE_EVALUATION.md`](../../../docs/CANDIDATE_EVALUATION.md).
