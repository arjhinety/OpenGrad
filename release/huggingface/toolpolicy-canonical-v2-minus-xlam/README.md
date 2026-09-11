---
pretty_name: OpenGrad ToolPolicy Canonical v2 — minus xLAM
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

> This is a **source-ablation view** of a frozen research corpus. It is the training input of one
> experiment, published so that experiment's data can be inspected independently. It carries no
> result, and it is not a recommended training mixture.

## What this is

[`OpenGrad-ToolPolicy-Canonical-v2`](https://huggingface.co/datasets/arrochi112/OpenGrad-ToolPolicy-Canonical-v2)
with **one source removed**: xLAM/APIGen. Three sources remain, 115,895 canonical records, 118
shards.

It exists because OpenGrad ran a source ablation against the frozen corpus, and an ablation is only
checkable if its input is available. This view is the exact input of that experiment.

## It is a selection, not a rebuild

The parent corpus writes its shards per source, so removing a source is the removal of that
source's shards and nothing else. Every retained shard here is a **byte-for-byte copy** of the
parent's: no record was parsed, rewritten, re-encoded, or re-sharded.

That is verifiable rather than asserted. Every shard's SHA-256 is recorded in
`release-manifest.json` under `output_shards`, and each one equals the parent's hash for the same
file. The manifest also records, under `derived_view`, the parent's fingerprint, its Hub revision,
and the names and hashes of the 58 shards that were excluded — so the exclusion can be checked
against the parent dataset as well as against this one.

```text
parent fingerprint : 8ced403b996e563d6e279aee7fdb346fc829fe5ff6af9daf8ef47c0a4007e161
derived fingerprint: f8ba687e16d8ab740b78a503adb41f4be28bd775f925291d6da7ec31be1ac5ac
```

Consequences worth stating, because they are properties an ablation depends on: supervision
contracts, renderer behaviour, loss masks, contamination decisions, source provenance, and
trainability decisions are **identical to the parent** for every retained record. The only
difference in the data is that xLAM is absent.

## What was removed, and what it costs

| Source | Canonical records | Trainable | Shards |
|---|---:|---:|---:|
| glaive-function-calling-v2 | 98,339 | 97,112 | 99 |
| toolace | 11,051 | 2,259 | 12 |
| when2call | 6,505 | 6,505 | 7 |
| **retained total** | **115,895** | **105,876** | **118** |
| *xLAM/APIGen (removed)* | *57,342* | *56,090* | *58* |

xLAM is **34.6% of the parent's trainable records but only 11.7% of its loss-bearing tokens.**
Those two numbers differ because xLAM records are short: under its `CALL_PREDICTION` contract the
supervised target is the tool call itself and no tool result or final response follows it, so a
record ends where a `COMPLETE_TRAJECTORY` record would continue. Measured across the parent's
rendered samples:

| | full corpus | minus xLAM | ratio |
|---|---:|---:|---:|
| trainable records | 161,966 | 105,876 | 0.654 |
| rendered tokens | 99,717,548 | 64,146,939 | 0.643 |
| supervised (loss-bearing) tokens | 33,565,721 | 29,630,369 | 0.883 |

That asymmetry is the reason the ablation needed two arms and two different step counts rather
than one. It is also the most transferable thing this view demonstrates: in a heterogeneous corpus,
a source's share of records is a poor proxy for its share of training signal, and an ablation that
matched on record counts would have confounded two variables.

## Supervision contracts

xLAM was the only source carrying `CALL_PREDICTION`, so this view is entirely `COMPLETE_TRAJECTORY`
— 105,876 trainable records. That is a consequence of excluding the source, not a second filter:
the ablation removes a source, and no contract was targeted.

| Contract | Trainable here | In the parent |
|---|---:|---:|
| `COMPLETE_TRAJECTORY` | 105,876 | 105,876 |
| `CALL_PREDICTION` | 0 | 56,090 |

## Intended use

As the input of the xLAM source ablation, and as a comparison artifact for anyone examining what
the parent corpus contains. Training on it is what the ablation did, not a recommendation that
anyone else should: the three remaining sources are the parent's, unbalanced and unmodified, and
the parent was never a recommended mixture either.

Not suitable as a benchmark. A subset of a training corpus is still training data.

## Attribution

The parent's `CITATIONS.bib` and `source-licenses.md` are inherited verbatim, so they still list
xLAM alongside the three retained sources. That is deliberate: attribution files should
over-include rather than risk dropping a licence obligation. xLAM's entry documents the source
this view removes, and the parent dataset is where those terms continue to apply.
